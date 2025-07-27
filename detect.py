import argparse
import os
import platform
import sys
from pathlib import Path
import yaml
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
with open('data.yaml', 'r') as f:
    class_names = yaml.safe_load(f)['names']
    
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    print(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        print("detection ",pred)

        img_width, img_height = 1920, 1080

        def convert_to_pixel_coordinates(pred, img_width, img_height):
            pixel_coords = pred[0].clone()
            pixel_coords[:, 0] = (pixel_coords[:, 0] * img_width / 640).int()
            pixel_coords[:, 1] = (pixel_coords[:, 1] * img_height / 384).int()
            pixel_coords[:, 2] = (pixel_coords[:, 2] * img_width / 640).int()
            pixel_coords[:, 3] = (pixel_coords[:, 3] * img_height / 384).int()
            return pixel_coords

        pixel_coordinates = convert_to_pixel_coordinates(pred, img_width, img_height)

        def format_pixel_coordinates(pred):
            formatted_pred = []
            for row in pred:
                x_min, y_min, x_max, y_max, confidence, cls = row
                formatted_row = [
                    int(x_min),
                    int(y_min),
                    int(x_max),
                    int(y_max),
                    float(confidence),
                    int(cls)
                ]
                formatted_pred.append(formatted_row)
            return formatted_pred

        detection_data = format_pixel_coordinates(pixel_coordinates)

        

        original_x, original_y = 1464, 705
        new_x, new_y = 485, 250

        scale_x = new_x / original_x
        scale_y = new_y / original_y

        scaled_boxes = []
        for box in detection_data:
            x1, y1, x2, y2, confidence, class_id = box
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            scaled_box = [x1_scaled, y1_scaled, x2_scaled, y2_scaled, confidence, class_id]
            scaled_boxes.append(scaled_box)

        def boxes_overlap(box1, box2):
            x1, y1, x2, y2 = box1[:4]
            x1_p, y1_p, x2_p, y2_p = box2[:4]
            overlap_x = not (x2 < x1_p or x2_p < x1)
            overlap_y = not (y2 < y1_p or y2_p < y1)
            return overlap_x and overlap_y

        def box_within(box1, box2):
            x1, y1, x2, y2 = box1[:4]
            x1_p, y1_p, x2_p, y2_p = box2[:4]
            return x1 <= x1_p and y1 <= y1_p and x2 >= x2_p and y2 >= y2_p

        def filter_boxes(boxes):
            filtered_boxes = []
            main_box = boxes[0]
            for box in boxes:
                if box[5] == 31:
                    continue
                if box_within(main_box, box):
                    keep = True
                    for i, other in enumerate(filtered_boxes):
                        if boxes_overlap(box, other):
                            if box[4] > other[4]:
                                filtered_boxes[i] = box
                            keep = False
                            break
                    if keep:
                        filtered_boxes.append(box)
               
            return filtered_boxes

        filtered_boxes1 = filter_boxes(scaled_boxes)
        filtered_boxes1.insert(0, scaled_boxes[0])

        def group_by_rows(detections, y_threshold=30):
            detections = sorted(detections, key=lambda x: x[1])
            rows = []
            current_row = []
            current_y = detections[0][1]
            for det in detections:
                y = det[1]
                if abs(y - current_y) <= y_threshold:
                    current_row.append(det)
                else:
                    rows.append(current_row)
                    current_row = [det]
                    current_y = y
            if current_row:
                rows.append(current_row)
            for row in rows:
                row.sort(key=lambda x: x[0])
            sorted_detections = [det for row in rows for det in row]
            return sorted_detections

        rearranged_detections = group_by_rows(filtered_boxes1)

        def get_class_name(class_number):
            class_map = {
                0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: 'A', 11: 'D', 12: 'E', 13: 'F', 14: 'G',
                15: 'H', 16: 'I', 17: 'K', 18: 'L', 19: 'M',
                20: 'N', 21: 'P', 22: 'R', 23: 'T', 24: 'U',
                25: 'V', 26: 'W', 27: 'X', 28: 'Z', 29: 'd',
                30: 'h', 31: 'meter'
            }
            return class_map.get(class_number, "Invalid class number")

        final = []
        for i in rearranged_detections:
            classes = get_class_name(i[5])
            final.append(classes)

        final_reading = ''.join(map(str, final[1:]))

        print("Formatted Detection Raw Reading:", [get_class_name(i[5]) for i in detection_data[1:]])
        print("Overlapping Filtered Reading:", [get_class_name(i[5]) for i in filtered_boxes1[1:]])
        print("Rearranged Reading:", [get_class_name(i[5]) for i in rearranged_detections[1:]])
        print("\nFinal Reading:", final_reading)


        for box in rearranged_detections:
            x1, y1, x2, y2, confidence, class_id = map(int, box)  
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2) 
            label = f"{class_id}"
            cv2.putText(im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite('detection_image.jpg', im)

        reading = final_reading

        # # Process predictions
        # for i, det in enumerate(pred):  # per image
        #     seen += 1
        #     if webcam:  # batch_size >= 1
        #         p, im0, frame = path[i], im0s[i].copy(), dataset.count
        #         s += f'{i}: '
        #     else:
        #         p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        #     p = Path(p)  # to Path
        #     save_path = str(save_dir / p.name)  # im.jpg
        #     txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        #     s += '%gx%g ' % im.shape[2:]  # print string
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     imc = im0.copy() if save_crop else im0  # for save_crop
        #     annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        #         # Sort the boxes based on the conditions: descending `y`, then ascending `x`
        #         # det[:, :4] contains (x1, y1, x2, y2), we need to sort by y1 descending, and x1 ascending within y1 groups.
        #         det = det[det[:, 1].argsort(descending=True)]  # Sort by y1 (descending)
        #         det = det[det[:, 0].argsort(descending=True)]  # Sort by x1 (ascending)
        #         print(det)
        #         # Print results
        #         for c in det[:, 5].unique():
        #             n = (det[:, 5] == c).sum()  # detections per class
        #             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             print(f"Class: {class_names[int(cls)]}, Confidence: {conf.item()}")
        #             reading += class_names[int(cls)]
        #             print(f"{class_names[int(cls)]}")
        #             if save_txt:  # Write to file
        #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #                 line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        #                 with open(f'{txt_path}.txt', 'a') as f:
        #                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

        #             if save_img or save_crop or view_img:  # Add bbox to image
        #                 c = int(cls)  # integer class
        #                 label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        #                 annotator.box_label(xyxy, label, color=colors(c, True))
        #             if save_crop:
        #                 save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        #     # Stream results
        #     im0 = annotator.result()
        #     if view_img:
        #         if platform.system() == 'Linux' and p not in windows:
        #             windows.append(p)
        #             cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        #             cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        #         cv2.imshow(str(p), im0)
        #         cv2.waitKey(1)  # 1 millisecond

        #     # Save results (image with detections)
        #     if save_img:
        #         if dataset.mode == 'image':
        #             cv2.imwrite(save_path, im0)
        #         else:  # 'video' or 'stream'
        #             if vid_path[i] != save_path:  # new video
        #                 vid_path[i] = save_path
        #                 if isinstance(vid_writer[i], cv2.VideoWriter):
        #                     vid_writer[i].release()  # release previous video writer
        #                 if vid_cap:  # video
        #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #                 else:  # stream
        #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
        #                 save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        #                 vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #             vid_writer[i].write(im0)

        # Print time (inference-only)
        print(f"reading {reading}")
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        return reading

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
