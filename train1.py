from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt", task="detect")

    model.train(
        data=r"D:\yolov11\Dataset\dataset1\data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        name="pet-bottle-can-glass-newtrain1",
        project=r"D:\yolov11\Dataset\dataset1\runs1",
        device=0,          # ใช้ GPU
        pretrained=True,   # แนะนำให้ fine-tune
        workers=4,         # กัน multiprocessing error บน Windows
        cache=False,       # dataset ใหญ่ RAM ไม่พอ ห้ามเปิด
    )

if __name__ == "__main__":
    main()
