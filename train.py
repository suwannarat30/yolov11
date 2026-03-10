from ultralytics import YOLO

model = YOLO("yolo11n.pt", task="detect")

model.train(
    data=r"D:\yolov11\Dataset\Machine.yolov11\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="pet-bottle-can-glass-newtrain",
    project=r"D:\yolov11\Dataset\Machine.yolov11\runs",
    device=0,
    pretrained=False,
    workers=0,
    cache=True,
)
