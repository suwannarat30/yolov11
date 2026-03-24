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
# 🔌 Arduino Config
# =========================
PORT = 'COM5'
BAUD = 9600

print("Connecting Arduino...")

try:
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Arduino Connected!")
    arduino_connected = True
except:
    print("Arduino not connected")
    arduino_connected = False


# =========================
# 🌐 Dashboard Server
# =========================
SERVER_URL = "http://localhost:5000/add"


# =========================
# 🧠 โหลด YOLO
# =========================
print("Loading YOLO model...")
model = YOLO(r"D:\yolov11\Dataset\dataset2\runs\pet-bottle-can-glass-v1\weights\best.pt")
print("YOLO Loaded!")


# =========================
# 📷 เปิดกล้อง
# =========================
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera error")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# =========================
# ⚙️ System Variables
# =========================
working = False
detect_counter = 0
DETECT_THRESHOLD = 10

DETECT_DELAY = 3
last_detection_time = 0

last_labels = []

# 🔥 กันยิงซ้ำ
cooldown = 5
last_action_time = 0

print("System Ready. Press Q or close window to exit.")


# =========================
# 🔁 MAIN LOOP
# =========================
try:

    while True:

        time.sleep(0.03)  # 🔥 ลดความเร็ว (กัน detect รัว)

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=0.25, verbose=False)

        boxes = results[0].boxes
        display = frame.copy()

        # =========================
        # ❌ ไม่พบวัตถุ
        # =========================
        if boxes is None or len(boxes) == 0:

            detect_counter = 0
            working = False

            cv2.imshow("YOLO Waste Detection", display)

            key = cv2.waitKey(1)

            if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

            if key & 0xFF == ord('q'):
                break

            continue


        # =========================
        # 🔍 เลือก object ใหญ่สุด
        # =========================
        areas = []

        for box in boxes.xyxy:
            x1, y1, x2, y2 = box
            areas.append((x2 - x1) * (y2 - y1))

        i = areas.index(max(areas))

        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])

        x1, y1, x2, y2 = map(int, boxes.xyxy[i])

        w = x2 - x1
        h = y2 - y1
        area = w * h

        frame_area = frame.shape[0] * frame.shape[1]

        # =========================
        # ❌ ตัด background (สำคัญมาก)
        # =========================
        if area > frame_area * 0.7:
            continue

        # =========================
        # 🧠 SMART FILTER
        # =========================
        if conf < 0.35 or area < 3000:
            label = "other"
        else:
            label = model.names[cls_id].lower()


        # =========================
        # 🔥 VOTING กันเด้ง
        # =========================
        last_labels.append(label)

        if len(last_labels) > 7:
            last_labels.pop(0)

        final_label = Counter(last_labels).most_common(1)[0][0]


        # =========================
        # 🖼️ DRAW
        # =========================
        color = (0,255,0) if final_label != "other" else (0,0,255)

        cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
        cv2.putText(display,
                    f"{final_label} {conf:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        cv2.imshow("YOLO Waste Detection", display)


        # =========================
        # 🧠 TRIGGER SYSTEM
        # =========================
        if not working:

            detect_counter += 1

            if detect_counter >= DETECT_THRESHOLD:

                # 🔥 กันยิงรัว
                if time.time() - last_action_time < cooldown:
                    continue

                if time.time() - last_detection_time > DETECT_DELAY:

                    # ❌ ไม่ส่ง other ไป Arduino
                    if final_label == "other":
                        continue

                    working = True
                    detect_counter = 0
                    last_detection_time = time.time()
                    last_action_time = time.time()

                    print("\n======================")
                    print("Detected:", final_label)
                    print("======================")

                    # 🌐 Dashboard
                    try:
                        requests.post(SERVER_URL, json={"type": final_label}, timeout=1)
                    except:
                        print("Dashboard error")

                    # 🔥 Arduino
                    if arduino_connected:
                        try:
                            arduino.write((final_label + "\n").encode())

                            start = time.time()

                            while True:

                                if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
                                    sys.exit()

                                if arduino.in_waiting:
                                    res = arduino.readline().decode().strip()
                                    if res == "DONE":
                                        break

                                if time.time() - start > 5:
                                    break

                        except:
                            print("Arduino error")

                    working = False


        # =========================
        # ❌ EXIT CONTROL
        # =========================
        key = cv2.waitKey(1)

        if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

        if key & 0xFF == ord('q'):
            break


# =========================
# 🔚 CLOSE
# =========================
finally:

    cap.release()
    cv2.destroyAllWindows()

    if arduino_connected:
        arduino.close()

    print("===== PROGRAM END =====")