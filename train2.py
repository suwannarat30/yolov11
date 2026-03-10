from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt", task="detect")

    model.train(
        data=r"D:\yolov11\Dataset\dataset2\data.yaml",
        epochs=80,                      # ⬅ เพิ่ม
        imgsz=518,                      # ⬅ เพิ่ม
        batch=8,                       # ⬅ ถ้า VRAM ไหว
        name="pet-bottle-can-glass-v1",
        project=r"D:\yolov11\Dataset\dataset2\runs",
        device=0,
        pretrained=True,
        workers=4,
        cache=False,
        patience=20,                    # ⬅ เพิ่ม
        seed=42                         # ⬅ งานวิจัย
    )

if __name__ == "__main__":
    main()
