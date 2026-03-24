from ultralytics import YOLO

def main():

    model = YOLO("yolo11s.pt")   # s แม่นกว่า n ประมาณ 10-15%

    model.train(
        data=r"D:\yolov11\Dataset\dataset3\data.yaml",

        # Training
        epochs=300,
        imgsz=512,
        batch=4,              # สำคัญสำหรับ GPU 4GB
        device=0,
        workers=2,

        # Optimization
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Early stopping
        patience=30,

        # Augmentation (เหมาะกับ ขวด / กระป๋อง / แก้ว)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=20,
        translate=0.1,
        scale=0.5,

        fliplr=0.5,
        flipud=0.1,

        mosaic=1.0,
        mixup=0.05,

        # Performance
        amp=True,
        cos_lr=True,
        rect=True,

        # Stability
        seed=42,
        deterministic=True,

        # Output
        name="bottle-can-glass-v4",
        project=r"D:\yolov11\Dataset\dataset3\runs3"
    )

if __name__ == "__main__":
    main()