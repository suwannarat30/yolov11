from ultralytics import YOLO
import cv2

# ✅ โหลดโมเดล (แก้ path ให้ตรงกับของคุณ)
model = YOLO(r"D:\yolov11\Dataset\dataset1\runs1\pet-bottle-can-glass-newtrain12\weights\best.pt")

# ✅ ตั้งค่าการตรวจจับ
results = model.predict(
    source=0,           # 0 = กล้องหลักของคอมพิวเตอร์
    imgsz=640,          # ขนาดภาพ (สามารถปรับเป็น 800 ได้)
    conf=0.6,           # ✅ ความมั่นใจขั้นต่ำ (ยิ่งสูง ยิ่งกรอง false positive)
    iou=0.7,            # ✅ กรองกรอบที่ซ้อนกันเกินไป
    show=True,          # แสดงภาพสดพร้อมกรอบตรวจจับ
    save=False,         # ถ้าต้องการบันทึกวิดีโอเปลี่ยนเป็น True
    save_txt=False,     # บันทึกไฟล์ตำแหน่ง .txt (ตั้งเป็น True ได้ถ้าต้องการ)
    save_conf=True,     # บันทึกค่าความมั่นใจ
    vid_stride=1,       # ประมวลผลทุกเฟรม (เพิ่มเป็น 2 ถ้าเครื่องช้า)
    device=0,           # ใช้ GPU ถ้ามี
    classes=None,       # ตรวจจับทุกคลาส (หรือใส่เฉพาะเช่น [0,1,2])
    stream=True         # ทำให้เราสามารถวนลูปดูผลลัพธ์ได้แบบสด
)

# ✅ วนลูปแสดงผลจากกล้อง
for result in results:
    frame = result.plot()  # วาดกรอบ/ชื่อคลาสลงภาพ
    cv2.imshow("YOLOv11 Detection", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ ปิดกล้องและหน้าต่างทั้งหมดเมื่อออก
cv2.destroyAllWindows()
