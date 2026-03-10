from ultralytics import YOLO
import cv2
import time

model = YOLO(r"D:\yolov11\Dataset\runs\detect\train\weights\best.pt")
cap = cv2.VideoCapture(0)

MODE = "ACTIVE"        # ACTIVE | SLEEP
last_object_time = time.time()

STOP_AFTER = 3.0       # วินาทีที่ไม่เจอวัตถุ → เข้า SLEEP
WAKE_CHECK = 1.0       # วินาที ตรวจว่ามีวัตถุกลับมาหรือยัง
last_wake_check = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # ================= ACTIVE MODE =================
    if MODE == "ACTIVE":
        results = model(frame, conf=0.25, imgsz=512)

        if len(results[0].boxes) > 0:
            last_object_time = now
            frame = results[0].plot()
        else:
            # ไม่มีวัตถุ
            if now - last_object_time > STOP_AFTER:
                MODE = "SLEEP"
                print("⏸ SYSTEM SLEEP (no object)")

    # ================= SLEEP MODE =================
    elif MODE == "SLEEP":
        cv2.putText(
            frame,
            "SYSTEM SLEEP",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # ตรวจเป็นช่วง ๆ ไม่ inference ทุกเฟรม
        if now - last_wake_check > WAKE_CHECK:
            last_wake_check = now

            results = model(frame, conf=0.25, imgsz=512)
            if len(results[0].boxes) > 0:
                MODE = "ACTIVE"
                last_object_time = now
                print("▶ SYSTEM RESUME (object detected)")

    cv2.imshow("Bottle Sorting Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
