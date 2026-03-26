print("===== PROGRAM START =====")

import cv2
import time
import serial
import requests
import sys
import threading
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, deque

# =========================
# 🔌 Arduino Config
# =========================
PORT = 'COM3'
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
# 🌐 Dashboard (async)
# =========================
SERVER_URL = "http://localhost:5000/add"

def safe_post_async(label_name):
    def _send():
        try:
            requests.post(SERVER_URL, json={"type": label_name}, timeout=1)
        except:
            pass
    threading.Thread(target=_send, daemon=True).start()

# =========================
# ⚙️ CONFIG
# =========================
YOLO_MODEL_PATH = r"D:\yolov11\Dataset\dataset4\runs4\pet-bottle-can-glass-v3\weights\best.pt"
FEATURES_PATH   = r"D:\yolov11\features.npy"
LABELS_PATH     = r"D:\yolov11\labels.npy"

YOLO_CONF             = 0.25
YOLO_LOW_CONF         = 0.55
OTHER_MATCH_THRESHOLD = 0.70
YOLO_SCORE_MIN        = 0.78
MISMATCH_CONF_MAX     = 0.65
BOX_SHRINK            = 0.82
TOP_K                 = 5
SMOOTH_WINDOW         = 7

# Trigger system
DETECT_THRESHOLD  = 10     # ต้องเห็นกี่ frame ถึงยิง
DETECT_DELAY      = 3      # วินาที cooldown หลัง trigger
ACTION_COOLDOWN   = 5      # วินาที กันยิงรัว
ARDUINO_TIMEOUT   = 5      # วินาที รอ DONE จาก Arduino

LOW_CONF_PER_CLASS = {
    "Glass":      0.60,
    "Can":        0.55,
    "Pet_bottle": 0.55,
    "PET-Bottle": 0.55,
}

CLASS_COLORS = {
    "Glass":      (255, 200, 0),
    "Can":        (0, 255, 255),
    "Pet_bottle": (0, 255, 0),
    "PET-Bottle": (0, 255, 0),
    "Other":      (0, 0, 255),
}

# =========================
# 🧠 โหลด YOLO
# =========================
print("Loading YOLO...")
model = YOLO(YOLO_MODEL_PATH)
print("YOLO loaded!")

# =========================
# 🗄️ โหลด Feature DB
# =========================
print("Loading feature DB...")
feature_db = np.load(FEATURES_PATH)
label_db   = np.load(LABELS_PATH)
print(f"DB: {feature_db.shape}, labels: {np.unique(label_db)}")

# =========================
# 🔬 ResNet18 Feature Extractor
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights   = models.ResNet18_Weights.DEFAULT
resnet    = models.resnet18(weights=weights)
extractor = nn.Sequential(*list(resnet.children())[:-1])
extractor.eval().to(device)
preprocess = weights.transforms()

# =========================
# 🔧 Helpers
# =========================
LABEL_MAP = {
    "Can": "Can",
    "can": "Can",

    "Glass": "Glass",
    "glass": "Glass",

    "Pet bottle": "Pet_bottle",
    "Pet_bottle": "Pet_bottle",
    "pet_bottle": "Pet_bottle",
    "PET-Bottle": "Pet_bottle",

    "Other": "Other",
    "other": "Other",
}

# map สำหรับส่งให้ Arduino โดยตรง
ARDUINO_LABEL_MAP = {
    "Pet_bottle": "PET-Bottle",
    "PET-Bottle": "PET-Bottle",

    "Can": "can",
    "can": "can",

    "Glass": "glass",
    "glass": "glass",

    "Other": "other",
    "other": "other",
}

def normalize_label(name):
    return LABEL_MAP.get(str(name), str(name))

def arduino_label(name):
    return ARDUINO_LABEL_MAP.get(str(name), "other")

def shrink_box(x1, y1, x2, y2, scale=BOX_SHRINK):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h   = (x2 - x1) * scale, (y2 - y1) * scale
    return (
        int(cx - w / 2),
        int(cy - h / 2),
        int(cx + w / 2),
        int(cy + h / 2)
    )

def extract_feature(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = preprocess(img_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = extractor(img_tensor).flatten(1).cpu().numpy()[0]

    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat

def get_class_scores(feat):
    sims = cosine_similarity([feat], feature_db)[0]
    scores = {}

    for cls in np.unique(label_db):
        cls_norm = normalize_label(cls)
        idxs = np.where(label_db == cls)[0]
        cls_sims = sims[idxs]
        k = min(TOP_K, len(cls_sims))
        scores[cls_norm] = float(np.mean(np.sort(cls_sims)[-k:]))

    return scores

# =========================
# 🎥 เปิดกล้อง
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# 📊 State Variables
# =========================
label_history       = {}   # box_key → deque
vote_buffer         = []

working             = False
detect_counter      = 0
last_detection_time = 0
last_action_time    = 0

print("System Ready. Press Q or ESC to exit.")

# =========================
# 🔁 Main Loop
# =========================
try:
    while True:
        time.sleep(0.03)

        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        results = model(frame, imgsz=640, conf=YOLO_CONF, verbose=False)
        display = frame.copy()
        boxes = results[0].boxes

        # ── ไม่พบ object ──
        if boxes is None or len(boxes) == 0:
            detect_counter = 0
            working = False

            cv2.imshow("YOLO Waste Detection", display)
            key = cv2.waitKey(1)

            if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key & 0xFF in [ord('q'), 27]:
                break
            continue

        # ── เลือก object ใหญ่สุด ──
        xyxy = boxes.xyxy.cpu().numpy()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
        i = int(np.argmax(areas))

        # ใช้ .cpu().numpy() ก่อน แล้วค่อย map(int,...)
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())

        area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]

        # ตัด background / object ใหญ่หรือเล็กเกินไป
        if area > frame_area * 0.7 or area < 3000:
            cv2.imshow("YOLO Waste Detection", display)
            key = cv2.waitKey(1)

            if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key & 0xFF in [ord('q'), 27]:
                break
            continue

        yolo_conf = float(boxes.conf[i].cpu().numpy())
        yolo_label = normalize_label(model.names[int(boxes.cls[i].cpu().numpy())])

        # ── Rule LOWCONF → Other ทันที ──
        low_thresh = LOW_CONF_PER_CLASS.get(yolo_label, YOLO_LOW_CONF)

        if yolo_conf < low_thresh:
            final_label = "Other"
            rule_used = "LOWCONF"

            color = CLASS_COLORS["Other"]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                f"Other (low:{yolo_conf:.2f})",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        else:
            # ── Shrink crop ──
            sx1, sy1, sx2, sy2 = shrink_box(x1, y1, x2, y2)
            sx1 = max(0, sx1)
            sy1 = max(0, sy1)
            sx2 = min(frame.shape[1], sx2)
            sy2 = min(frame.shape[0], sy2)

            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                cv2.imshow("YOLO Waste Detection", display)
                key = cv2.waitKey(1)

                if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key & 0xFF in [ord('q'), 27]:
                    break
                continue

            # ── Feature DB ──
            feat = extract_feature(crop)
            class_scores = get_class_scores(feat)

            sorted_items = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
            pred1, top1 = sorted_items[0] if sorted_items else (yolo_label, 0.0)
            pred2, top2 = sorted_items[1] if len(sorted_items) > 1 else ("-", 0.0)

            other_score = class_scores.get("Other", 0.0)
            yolo_score_db = class_scores.get(yolo_label, 0.0)

            # ── Decision Rules ──
            final_label = yolo_label
            rule_used = "YOLO"

            if other_score >= OTHER_MATCH_THRESHOLD and other_score > yolo_score_db:
                final_label = "Other"
                rule_used = "A"

            elif pred1 != yolo_label and yolo_score_db < YOLO_SCORE_MIN:
                if other_score >= (top1 - 0.05):
                    final_label = "Other"
                    rule_used = "B"

            elif yolo_conf < MISMATCH_CONF_MAX and pred1 != yolo_label:
                final_label = "Other"
                rule_used = "C"

            # ── Temporal Smoothing ──
            box_key = f"{sx1//50}_{sy1//50}"
            if box_key not in label_history:
                label_history[box_key] = deque(maxlen=SMOOTH_WINDOW)

            label_history[box_key].append(final_label)
            final_label = Counter(label_history[box_key]).most_common(1)[0][0]

            # ── Draw ──
            color = CLASS_COLORS.get(final_label, (255, 255, 255))
            text1 = f"{final_label} ({yolo_conf:.2f})"
            text2 = f"rule={rule_used} os={other_score:.2f} ys={yolo_score_db:.2f} p1={pred1}:{top1:.2f}"

            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(
                display,
                text1,
                (sx1, max(30, sy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            cv2.putText(
                display,
                text2,
                (sx1, min(display.shape[0] - 10, sy2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
            )

        # =========================
        # 🗳️ Voting Buffer (Trigger)
        # =========================
        vote_buffer.append(final_label)
        if len(vote_buffer) > SMOOTH_WINDOW:
            vote_buffer.pop(0)

        voted_label = Counter(vote_buffer).most_common(1)[0][0]

        # =========================
        # 🚀 Trigger System
        # =========================
        if not working:
            detect_counter += 1

            if detect_counter >= DETECT_THRESHOLD:
                now = time.time()

                if (now - last_action_time >= ACTION_COOLDOWN and
                    now - last_detection_time >= DETECT_DELAY):

                    working = True
                    detect_counter = 0
                    last_detection_time = now
                    last_action_time = now

                    print("\n======================")
                    print(f"Detected: {voted_label}")
                    print("======================")

                    # Dashboard
                    safe_post_async(voted_label)

                    # Arduino
                    if arduino_connected:
                        try:
                            send_label = arduino_label(voted_label)

                            print(f"Send to Arduino: {send_label}")
                            arduino.write((send_label + "\n").encode())

                            start = time.time()

                            while True:
                                if cv2.getWindowProperty(
                                    "YOLO Waste Detection",
                                    cv2.WND_PROP_VISIBLE
                                ) < 1:
                                    sys.exit()

                                if arduino.in_waiting:
                                    res = arduino.readline().decode().strip()
                                    print("Arduino:", res)
                                    if res == "DONE":
                                        break

                                if time.time() - start > ARDUINO_TIMEOUT:
                                    print("Arduino timeout")
                                    break

                                cv2.imshow("YOLO Waste Detection", display)
                                cv2.waitKey(1)

                        except Exception as e:
                            print("Arduino error:", e)

                    working = False

        cv2.imshow("YOLO Waste Detection", display)
        key = cv2.waitKey(1)

        if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key & 0xFF in [ord('q'), 27]:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    if arduino_connected:
        try:
            arduino.close()
        except:
            pass

    print("===== PROGRAM END =====")
