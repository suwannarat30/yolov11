print("===== PROGRAM START =====")

import cv2
import time
import serial
import requests
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
model = YOLO("yolov11/best.pt")
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

fps_time = time.time()

print("System Ready. Press Q or close window to exit.")


# =========================
# 🔁 MAIN LOOP
# =========================
try:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=0.65, verbose=False)

        annotated = results[0].plot()
        cv2.imshow("YOLO Waste Detection", annotated)

        # =========================
        # ⌨️ Keyboard
        # =========================
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
            break


        boxes = results[0].boxes

        # =========================
        # ❌ ไม่พบวัตถุ
        # =========================
        if boxes is None or len(boxes) == 0:
            detect_counter = 0
            working = False
            continue


        # =========================
        # 🔍 เลือกวัตถุใหญ่สุด
        # =========================
        areas = []

        for box in boxes.xyxy:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)

        biggest_index = areas.index(max(areas))


        # =========================
        # 🧠 Frame Voting
        # =========================
        if not working:

            detect_counter += 1

            if detect_counter >= DETECT_THRESHOLD:

                current_time = time.time()

                if current_time - last_detection_time > DETECT_DELAY:

                    working = True
                    detect_counter = 0
                    last_detection_time = current_time

                    cls_id = int(boxes.cls[biggest_index])
                    label = model.names[cls_id].lower()

                    print("\n======================")
                    print("Detected:", label)
                    print("======================")


                    # =========================
                    # 🌐 ส่งไป Dashboard
                    # =========================
                    try:

                        requests.post(
                            SERVER_URL,
                            json={"type": label},
                            timeout=1
                        )

                        print("Dashboard updated")

                    except:
                        print("Dashboard connection error")


                    # =========================
                    # 🔥 ส่งไป Arduino
                    # =========================
                    if arduino_connected:

                        try:

                            command = label + "\n"
                            arduino.write(command.encode())

                            print("Sent to Arduino:", label)

                            start_wait = time.time()

                            while True:

                                if arduino.in_waiting:

                                    response = arduino.readline().decode().strip()

                                    if response == "DONE":
                                        print("Arduino finished")
                                        break

                                # กันค้าง
                                if time.time() - start_wait > 5:
                                    print("Arduino timeout")
                                    break

                        except:
                            print("Arduino communication error")

                    else:
                        print("Arduino skipped (not connected)")

                    working = False


# =========================
# 🔚 CLOSE SYSTEM
# =========================
finally:

    cap.release()
    cv2.destroyAllWindows()

    if arduino_connected:
        arduino.close()

    print("===== PROGRAM END =====")