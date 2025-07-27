import sqlite3
import detect_meter

class MeterDatabase:
    def __init__(self, db_name="meters_database.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS meters (
                    meter_id TEXT PRIMARY KEY,
                    reading INTEGER,
                    meter_digits INTEGER,
                    meter_points INTEGER)
            """)

    def add_meter(self, meter_id, reading, meter_digits, meter_points):
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO meters (meter_id, reading, meter_digits, meter_points)
                VALUES (?, ?, ?, ?)
            """, (meter_id, reading, meter_digits, meter_points))

    
    def update_meter_reading(self, meter_id, new_reading):
        with self.conn:
            self.conn.execute("""
                UPDATE meters
                SET reading = ?
                WHERE meter_id = ?
            """, (new_reading, meter_id))

    def get_meter_info(self, meter_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT reading, meter_digits, meter_points
            FROM meters
            WHERE meter_id = ?
        """, (meter_id,))
        return cursor.fetchone()

    def delete_meter(self, meter_id):
        with self.conn:
            self.conn.execute("DELETE FROM meters WHERE meter_id = ?", (meter_id,))

def add_meter(id,reading,interger,decimal):
    try:
        db.add_meter(id, reading,interger,decimal)
        return "Meter Added"
    except:
        return "Not Added"
    
def update(id,reading):
    try:
        db.update_meter_reading(id,reading)
        return "Meter Updated"
    except:
        return "Not Updated"

def delete_meter(id):
    try:
        db.delete_meter(id)
        return "Meter Deleted"
    except:
        return "Not Deleted"

def verify_reading(last_reading="000000",current_reading="000000"):
    lower_range = 10
    higher_range = 1

    def get_diff(cur,pre):
        cur = int(cur)
        pre = int(pre)
    
        if 91 <= pre <= 99 and 0 <= cur <= 30:
            cur = cur + 100
    
        return cur - pre

    try:
        if (len(last_reading) != len(current_reading)) :
            print("Missing number function called")
            if (len(last_reading) != len(current_reading)) > 2:
                print("More than 2 number missing")
                return last_reading,last_reading
        
            elif last_reading[:-2] == current_reading[:-2]:
                print("Number missing from last 2 digits")
                current_reading = current_reading[:-2]+last_reading[-2:]
                
            else:
                print("Unkown number missing")
                return last_reading,last_reading
                
        l_higher_digits = last_reading[:-2]
        l_lower_digits = last_reading[-2:]
        
        c_higher_digits = current_reading[:-2]
        c_lower_digits = current_reading[-2:]

        lower_diff =  get_diff(c_lower_digits,l_lower_digits)
        higher_diff = get_diff(c_higher_digits,l_higher_digits)
    
        #print(lower_diff,higher_diff)
    
        if lower_diff > lower_range or lower_diff < 0:
            print("Lower digits difference is high or minus")
            c_lower_digits = l_lower_digits[:-1] + c_lower_digits[-1:]
            
            new_lower_diff = get_diff(c_lower_digits,l_lower_digits)
            if new_lower_diff > lower_range or new_lower_diff < 0:
                c_lower_digits = l_lower_digits
            
    
        if higher_diff > higher_range or higher_diff < 0:
            print("Higher digits difference is high or minus")
            c_higher_digits = l_higher_digits
        elif higher_diff == higher_range and int(l_lower_digits)%10 < 8 :
            print("Higher digits updated but lower digits are not in range")
            c_higher_digits = l_higher_digits

        updated_reading = c_higher_digits+c_lower_digits
        return last_reading,updated_reading
            
    except:
        return last_reading,last_reading
        
    
def get_reading(current_meter_id,image):
    try:
        data = db.get_meter_info(current_meter_id)
        Last_reading = data[0]
        #Integer_digits = data[1]
        Decimal_digits = data[2]
        parts = Last_reading.split('.')
        last_reading_value = parts[0]
        #last_reading_decimal_part = parts[1]

        current_reading = detect_meter.process_image(image)

        final_reading = verify_reading(last_reading_value,current_reading[:-(Decimal_digits)])
        final_reading = final_reading +"."+ current_reading[-(Decimal_digits):]

        update(current_meter_id,final_reading)

        print(final_reading)
        return final_reading
    except:
        return "000000.000"

db = MeterDatabase()