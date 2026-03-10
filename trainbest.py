from ultralytics import YOLO

def main():
    # โหลดโมเดล best.pt เดิม
    model = YOLO(
        "D:/yolov11/Dataset/dataset1/runs1/"
        "pet-bottle-can-glass-newtrain12/weights/best.pt"
    )

    # เทรนต่อ (fine-tune)
    model.train(
        data="D:/yolov11/Dataset/dataset1/data.yaml",
        epochs=15,
        lr0=0.0003,
        imgsz=640,
        batch=8,          # เหมาะกับ RTX 3050 4GB
        patience=5,
        workers=0,        # 🔥 แก้ multiprocessing error บน Windows
        device=0,
        project="D:/yolov11/Dataset/dataset1/runs1",
        name="pet-bottle-can-glass-finetune1"
    )

if __name__ == "__main__":
    main()
