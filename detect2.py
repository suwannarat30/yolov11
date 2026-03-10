from ultralytics import YOLO
import cv2

model = YOLO(
    r"Dataset/dataset2/runs/pet-bottle-can-glass-v1/weights/best.pt"
)

results = model.predict(
    source=0,
    imgsz=800,        # 🔼 แม่นขึ้นกับ glass
    conf=0.65,        # 🔼 ลด false positive
    iou=0.7,
    show=False,       # เราใช้ cv2.imshow เอง
    save=False,
    save_txt=False,
    save_conf=True,
    vid_stride=1,
    device=0,
    half=True,        # 🔥 เพิ่มประสิทธิภาพ
    stream=True
)

for result in results:
    frame = result.plot()
    cv2.imshow("YOLOv11 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
