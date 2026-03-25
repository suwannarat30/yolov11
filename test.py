print("===== PROGRAM START =====")

import cv2
import time
import serial
import requests
import sys
import numpy as np
from collections import Counter
from ultralytics import YOLO

# =========================
# 🔌 1. Arduino Config
# =========================
PORT = 'COM5'
BAUD = 9600

print("Connecting Arduino...")
try:
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Arduino Connected!")
    arduino_connected = True
except Exception as e:
    print(f"Arduino not connected: {e}")
    arduino_connected = False

# =========================
# 🌐 2. Dashboard Server
# =========================
SERVER_URL = "http://localhost:5000/add"

# =========================
# 🧠 3. โหลด YOLO Model
# =========================
print("Loading YOLO model...")
# ระบุ Path ให้ถูกต้อง (ระวังเรื่อง \ ใน Windows)
model = YOLO(r"D:\New folder\yolov11-1\best.pt")
print("YOLO Loaded!")

# =========================
# 📷 4. ตั้งค่ากล้อง
# =========================
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera error: Could not open video device.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# ⚙️ 5. System Variables
# =========================
working = False
detect_counter = 0
DETECT_THRESHOLD = 10  # ต้องเจอวัตถุเดิมซ้ำ 10 เฟรมถึงจะทำงาน

DETECT_DELAY = 3
last_detection_time = 0
last_labels = []

# 🔥 Cooldown กันระบบทำงานซ้ำซ้อน
cooldown = 5 
last_action_time = 0

# รายชื่อ Class ที่ตรงกับใน Model (แก้ไขเป็น 'pet bottle' ตามรูป)
TARGET_CLASSES = ["pet bottle", "can", "glass"]

print("System Ready. Press 'Q' to exit.")

# =========================
# 🔁 6. MAIN LOOP
# =========================
try:
    while True:
        time.sleep(0.01) # ปรับค่าเพื่อความลื่นไหล

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # ตรวจจับวัตถุ
        results = model(frame, imgsz=640, conf=0.25, verbose=False)
        boxes = results[0].boxes
        display = frame.copy()

        # --- กรณีไม่พบวัตถุเลย ---
        if boxes is None or len(boxes) == 0:
            detect_counter = 0
            working = False
            cv2.imshow("YOLO Waste Detection", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # --- เลือกวัตถุที่ใหญ่ที่สุดในเฟรม (ป้องกันพื้นหลังหลอก) ---
        areas = []
        for box in boxes.xyxy:
            x1, y1, x2, y2 = box
            areas.append((x2 - x1) * (y2 - y1))

        i = areas.index(max(areas))
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])

        area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]

        # ❌ ข้ามถ้าวัตถุใหญ่เกิน 80% ของจอ (อาจเป็นพื้นหลัง)
        if area > frame_area * 0.8:
            continue

        # =========================
        # 🧠 SMART CLASSIFICATION
        # =========================
        # ดึงชื่อคลาสมาเช็ค (ทำเป็นตัวพิมพ์เล็กเพื่อความแม่นยำ)
        raw_label = model.names[cls_id].lower()
        
        # เช็คว่าชื่อตรงกับ Target และความมั่นใจสูงพอไหม
        if raw_label in TARGET_CLASSES and conf >= 0.45:
            label = raw_label
        else:
            # ถ้าไม่ใช่ขวด/กระป๋อง/แก้ว หรือดูไม่ออก ให้เป็น other ทันที
            label = "other"

        # --- เก็บประวัติ 7 เฟรมล่าสุดเพื่อหาค่าที่นิ่งที่สุด (Voting) ---
        last_labels.append(label)
        if len(last_labels) > 7:
            last_labels.pop(0)

        final_label = Counter(last_labels).most_common(1)[0][0]

        # =========================
        # 🖼️ DRAW & DISPLAY
        # =========================
        color = (0, 255, 0) if final_label in TARGET_CLASSES else (0, 0, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"DETECT: {final_label} ({conf:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("YOLO Waste Detection", display)

        # =========================
        # 🚀 TRIGGER SYSTEM (ส่งข้อมูล)
        # =========================
        if not working:
            detect_counter += 1

            # เมื่อตรวจพบวัตถุเดิมซ้ำๆ จนมั่นใจ
            if detect_counter >= DETECT_THRESHOLD:
                
                # ตรวจสอบ Cooldown
                current_time = time.time()
                if current_time - last_action_time < cooldown:
                    continue

                working = True
                detect_counter = 0
                last_action_time = current_time

                print(f"\n[ACTION] Target identified as: {final_label.upper()}")

                # 🌐 1. ส่งไป Dashboard
                try:
                    requests.post(SERVER_URL, json={"type": final_label}, timeout=1)
                except:
                    print("Dashboard: Server offline")

                # 🔥 2. ส่งไป Arduino (ส่งทั้งขยะเป้าหมาย และ 'other')
                if arduino_connected:
                    try:
                        # ล้าง Buffer เก่าก่อนส่ง
                        arduino.reset_input_buffer()
                        arduino.write((final_label + "\n").encode())
                        
                        # รอรับสัญญาณตอบกลับ 'DONE' จาก Arduino
                        start_wait = time.time()
                        print("Waiting for Arduino feedback...")
                        while True:
                            if arduino.in_waiting:
                                res = arduino.readline().decode().strip()
                                if res == "DONE":
                                    print("Arduino: Completed successfully.")
                                    break
                            
                            # ป้องกัน Loop ค้าง (Timeout 7 วินาที)
                            if time.time() - start_wait > 7:
                                print("Arduino: Wait timeout!")
                                break
                    except Exception as e:
                        print(f"Arduino Error: {e}")

                working = False

        # --- Exit Control ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    if arduino_connected:
        arduino.close()
    print("===== PROGRAM END =====")