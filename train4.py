from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data=r"D:\yolov11\Dataset\dataset4\data.yaml", # ปรับ path ให้ถูกต้อง
        epochs=150,
        imgsz=512,
        batch=4,  # หรือสูงสุดที่เครื่องไหว

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        fliplr=0.5,
        mosaic=0.5,
        mixup=0.05,
        degrees=10,
        scale=0.5,
        translate=0.1,

        patience=25,

        name="pet-bottle-can-glass-v3",
        project=r"D:\yolov11\Dataset\dataset4\runs4", # ปรับ path ให้ถูกต้อง

        device=0,
        pretrained=True,
        workers=0,
        cache=False,
        val=True,
        save_period=10,
        exist_ok=True,
    )

if __name__ == "__main__":
    main()