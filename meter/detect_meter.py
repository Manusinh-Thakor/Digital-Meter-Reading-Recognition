
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import LOGGER, check_img_size, non_max_suppression, xyxy2xywh, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_show = True
detection_results = []
imgsz = (640, 640)
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
device = ''

weights = 'last.pt'  
data = 'data.yaml'  

device = select_device(device)
model = DetectMultiBackend(weights, device=device, data=data, fp16=False)

def is_overlapping(box1, box2, overlap_threshold=0.5):
    x1_min, x1_max = box1['x'], box1['x'] + box1['width']
    y1_min, y1_max = box1['y'], box1['y'] + box1['height']
    
    x2_min, x2_max = box2['x'], box2['x'] + box2['width']
    y2_min, y2_max = box2['y'], box2['y'] + box2['height']
    
    overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    overlap_area = overlap_x * overlap_y
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    iou = overlap_area / float(box1_area + box2_area - overlap_area)
    
    return iou > overlap_threshold

def remove_overlapping_detections(detections, overlap_threshold=0.5):
    print("Overlapping Found")
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    filtered_detections = []
    
    for detection in detections:
        is_overlap = False
        for chosen in filtered_detections:
            if is_overlapping(detection['coordinates'], chosen['coordinates'], overlap_threshold):
                is_overlap = True
                break
        if not is_overlap:
            filtered_detections.append(detection)
    
    return filtered_detections

def process_image(source):
    global imgsz
    imgsz = check_img_size(imgsz, s=model.stride)
    dataset = LoadImages(source, img_size=imgsz, stride=model.stride)
    
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        im0s_rgb = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
        annotator = Annotator(im0s, line_width=3)
        
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    xywh = xyxy2xywh(torch.tensor(xyxy).unsqueeze(0)).squeeze().tolist()
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    detection_results.append({
                        'label': model.names[int(cls)],
                        'confidence': float(f"{conf:.2f}"),
                        'coordinates': {
                            'x': int(xywh[0]),
                            'y': int(xywh[1]),
                            'width': int(xywh[2]),
                            'height': int(xywh[3])
                        }
                    })
        # print(f"Detections for {path}:")
        # for result in detection_results:
        #     print(result)

        filtered_results = remove_overlapping_detections(detection_results)
        sorted_detections = sorted(filtered_results, key=lambda d: d['coordinates']['x'])
        meter_reading = ''.join([d['label'] for d in sorted_detections])



        if img_show:
            img = cv2.imread(source)
            for detection in filtered_results:
                x = detection['coordinates']['x']
                y = detection['coordinates']['y']
                label = detection['label']

                # Set the color and font for the text
                color = (0, 255, 0)  # Green
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 2

                # Put text on the image
                cv2.putText(img, label, (x, y - 5), font, scale, color, thickness)

            # Show the image
            cv2.imshow('Detections', img)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()  # Close all OpenCV windows

        #print("Meter reading:", meter_reading)
        return meter_reading

        
# reading = process_image('new2.jpg')
# print("output:",reading)