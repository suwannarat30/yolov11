"""
╔══════════════════════════════════════════════════════╗
║           Waste Detection — All-in-one               ║
╠══════════════════════════════════════════════════════╣
║  รันด้วย mode ที่ต้องการ:                            ║
║                                                      ║
║  1) python waste_detection.py --mode prepare         ║
║     → crop ROI จาก Roboflow dataset (ทำครั้งเดียว)  ║
║                                                      ║
║  2) python waste_detection.py --mode train           ║
║     → train MobileNet classifier                     ║
║                                                      ║
║  3) python waste_detection.py --mode detect          ║
║     → รันระบบจริง (default)                         ║
║                                                      ║
║  (optional)                                          ║
║  python waste_detection.py --mode collect            ║
║     → เก็บ ROI เพิ่มจากกล้อง                        ║
╚══════════════════════════════════════════════════════╝
"""

print("===== PROGRAM START =====")

import argparse
import os
import sys
import time
import cv2
import numpy as np
import requests
import serial
from pathlib import Path
from collections import Counter
from ultralytics import YOLO


# ══════════════════════════════════════════
#  CONFIG — แก้ตรงนี้
# ══════════════════════════════════════════
YOLO_MODEL_PATH       = r"D:\yolov11\Dataset\dataset2\runs\pet-bottle-can-glass-v1\weights\best.pt"
ROBOFLOW_DATASET_PATH = r"D:\yolov11\Dataset\dataset2"
CLASSIFIER_PATH       = "classifier.pth"
CLASSIFIER_DATA_ROOT  = "classifier_data"
CLASSIFIER_CLASSES    = ["PET-Bottle", "can", "glass"]
CLASSIFIER_THRESHOLD  = 0.60    # < นี้ = other (Softmax gate)

# ── OOD Detection ──
OOD_CENTROIDS_PATH  = "ood_centroids.npy"  # สร้างอัตโนมัติตอน --mode train
OOD_DISTANCE_THRESH = 12.0  # ระยะ L2 เกินนี้ = other
                             # ↑ สูง = ยอมรับมากขึ้น | ↓ ต่ำ = เข้มงวดขึ้น

ARDUINO_PORT = 'COM5'
ARDUINO_BAUD = 9600
SERVER_URL   = "http://localhost:5000/add"


# ══════════════════════════════════════════
#  HSV Color Profiles
# ══════════════════════════════════════════
HSV_PROFILES = {
    "PET-Bottle": [
        {"name": "clear",      "lower": [0,   0,  100], "upper": [180, 60, 255]},
        {"name": "light_blue", "lower": [85,  20, 150], "upper": [130, 90, 255]},
        {"name": "green",      "lower": [35,  30, 100], "upper": [85, 120, 220]},
        {"name": "orange_cap", "lower": [5,  100, 100], "upper": [20, 255, 255]},
    ],
    "glass": [
        {"name": "dark_green", "lower": [35,  40,  30], "upper": [85, 255, 130]},
        {"name": "brown",      "lower": [5,   60,  30], "upper": [25, 255, 140]},
        {"name": "clear",      "lower": [0,   0,  180], "upper": [180, 25, 255]},
    ],
    "can": [
        {"name": "silver",     "lower": [0,   0,  140], "upper": [180, 40, 255]},
        {"name": "gold",       "lower": [15,  30, 140], "upper": [35, 180, 255]},
    ],
}

BOX_COLORS = {
    "PET-Bottle":   (50,  180, 255),
    "glass":        (50,  220, 100),
    "can":          (255, 180,  50),
    "other":        (100, 100, 100),
}


# ══════════════════════════════════════════
#  MODE 1 — PREPARE
# ══════════════════════════════════════════
def run_prepare():
    import yaml

    print("\n[PREPARE] crop ROI จาก Roboflow dataset")
    dataset_path = Path(ROBOFLOW_DATASET_PATH)

    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] ไม่พบ data.yaml ที่ {yaml_path}")
        return
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    class_names = data.get("names", [])
    print(f"[INFO] classes: {class_names}")

    for cls in class_names:
        os.makedirs(os.path.join(CLASSIFIER_DATA_ROOT, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(CLASSIFIER_DATA_ROOT, "val",   cls), exist_ok=True)

    counters = {cls: 0 for cls in class_names}
    skipped  = 0
    PADDING  = 0.05
    MIN_SIZE = 32

    splits = [s for s in ["train", "valid", "test"]
              if (dataset_path / s / "images").exists()]
    print(f"[INFO] splits ที่พบ: {splits}\n")

    for split in splits:
        img_dir   = dataset_path / split / "images"
        lbl_dir   = dataset_path / split / "labels"
        img_files = sorted(list(img_dir.glob("*.jpg")) +
                           list(img_dir.glob("*.png")) +
                           list(img_dir.glob("*.jpeg")))
        print(f"[{split}] {len(img_files)} ภาพ")

        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                skipped += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            H, W = img.shape[:2]

            for line in open(lbl_path).read().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id >= len(class_names):
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = max(0, int((cx - bw/2 - bw*PADDING) * W))
                y1 = max(0, int((cy - bh/2 - bh*PADDING) * H))
                x2 = min(W, int((cx + bw/2 + bw*PADDING) * W))
                y2 = min(H, int((cy + bh/2 + bh*PADDING) * H))
                roi = img[y1:y2, x1:x2]
                if roi.shape[0] < MIN_SIZE or roi.shape[1] < MIN_SIZE:
                    skipped += 1
                    continue
                cls_name = class_names[cls_id]
                n   = counters[cls_name]
                sub = "val" if n % 5 == 4 else "train"
                out = os.path.join(CLASSIFIER_DATA_ROOT, sub, cls_name, f"{n:05d}.jpg")
                cv2.imwrite(out, roi)
                counters[cls_name] += 1

    print("\n" + "="*45)
    total = sum(counters.values())
    for cls, n in counters.items():
        print(f"  {cls:20s}: {n} ภาพ")
    print(f"  {'TOTAL':20s}: {total} ภาพ  (skipped: {skipped})")
    print("="*45)
    for cls, n in counters.items():
        if n < 100:
            print(f"[WARN] {cls} มีแค่ {n} ภาพ — ควรมีอย่างน้อย 100+")
    print(f"\n[OK] บันทึกที่: {os.path.abspath(CLASSIFIER_DATA_ROOT)}/")
    print("ถัดไป: python train_classifier.py --mode train")


# ══════════════════════════════════════════
#  MODE 2 — COLLECT
# ══════════════════════════════════════════
def run_collect():
    print("\n[COLLECT] เก็บ ROI จากกล้อง")
    print("  S=save  N=เปลี่ยน class  Q=ออก\n")

    yolo = YOLO(YOLO_MODEL_PATH)
    cap  = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cls_idx  = 0
    counters = {}
    for c in CLASSIFIER_CLASSES:
        folder = os.path.join(CLASSIFIER_DATA_ROOT, "train", c)
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(CLASSIFIER_DATA_ROOT, "val", c), exist_ok=True)
        counters[c] = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
    print("ภาพที่มีอยู่แล้ว:", counters)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur = CLASSIFIER_CLASSES[cls_idx]
        res = yolo(frame, verbose=False)[0]
        display = frame.copy()
        roi = None
        if res.boxes and len(res.boxes):
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in res.boxes.xyxy]
            i = areas.index(max(areas))
            x1, y1, x2, y2 = map(int, res.boxes.xyxy[i])
            roi = frame[y1:y2, x1:x2]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 100), 2)
            if roi.size > 0:
                thumb = cv2.resize(roi, (120, 120))
                display[8:128, display.shape[1]-128:display.shape[1]-8] = thumb
        cv2.putText(display, f"Class: {cur}  saved: {counters[cur]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        cv2.putText(display, "S=save  N=next class  Q=quit",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Collect ROI", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            cls_idx = (cls_idx + 1) % len(CLASSIFIER_CLASSES)
            print(f"[COLLECT] → {CLASSIFIER_CLASSES[cls_idx]}")
        elif key == ord('s') and roi is not None and roi.size > 0:
            n   = counters[cur]
            sub = "val" if n % 5 == 4 else "train"
            path = os.path.join(CLASSIFIER_DATA_ROOT, sub, cur, f"{n:04d}.jpg")
            cv2.imwrite(path, roi)
            counters[cur] += 1
            print(f"  saved {cur}/{n:04d}.jpg [{sub}]")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[COLLECT] สรุป:", counters)
    print("ถัดไป: python train_classifier.py --mode train")


# ══════════════════════════════════════════
#  MODE 3 — TRAIN
# ══════════════════════════════════════════
def run_train():
    try:
        import torch
        import torch.nn as nn
        from torchvision import datasets, transforms, models
        from torch.utils.data import DataLoader
    except ImportError:
        print("[ERROR] ติดตั้งก่อน: pip install torch torchvision")
        return

    EPOCHS = 25
    BATCH  = 32
    LR     = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[TRAIN] device={DEVICE}  epochs={EPOCHS}")

    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(CLASSIFIER_DATA_ROOT, "train")
    val_path   = os.path.join(CLASSIFIER_DATA_ROOT, "val")
    if not os.path.exists(train_path):
        print(f"[ERROR] ไม่พบ {train_path} — รัน --mode prepare ก่อน")
        return

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_path,   transform=val_tf)
    print(f"  train: {len(train_ds)} ภาพ  val: {len(val_ds)} ภาพ")
    print(f"  classes: {train_ds.classes}")

    net = models.mobilenet_v3_small(weights="DEFAULT")
    net.classifier[3] = nn.Linear(1024, len(CLASSIFIER_CLASSES))
    net = net.to(DEVICE)

    opt     = torch.optim.Adam(net.parameters(), lr=LR)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for ep in range(1, EPOCHS + 1):
        net.train()
        for imgs, labels in DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss_fn(net(imgs), labels).backward()
            opt.step()
        sch.step()

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in DataLoader(val_ds, batch_size=BATCH, num_workers=0):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (net(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total if total else 0
        tag = "  ← best" if acc > best_acc else ""
        print(f"  Epoch {ep:02d}/{EPOCHS}  val_acc: {acc:.3f}{tag}")
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), CLASSIFIER_PATH)

    print(f"\n[TRAIN] เสร็จ! best val_acc: {best_acc:.3f}")
    print(f"  บันทึกที่: {CLASSIFIER_PATH}")

    # คำนวณ OOD centroids หลัง train เสร็จ
    print("\n[OOD] กำลังคำนวณ feature centroids...")
    net.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    net.eval()
    _compute_and_save_centroids(net, train_ds, DEVICE)

    print("ถัดไป: python train_classifier.py --mode detect")


# ══════════════════════════════════════════
#  OOD — คำนวณและบันทึก centroids
# ══════════════════════════════════════════
def _compute_and_save_centroids(net, dataset, device):
    """
    ดึง feature vector จาก avgpool (576-dim) ของแต่ละภาพ
    แล้วคำนวณ centroid ของแต่ละ class บันทึกเป็น ood_centroids.npy
    ตอน detect จะวัดระยะ L2 จาก feature ของวัตถุไปยัง centroid ที่ใกล้ที่สุด
    ถ้าห่างเกิน OOD_DISTANCE_THRESH = other
    """
    import torch
    from torch.utils.data import DataLoader

    features_list = []
    labels_list   = []

    def hook_fn(module, input, output):
        # flatten แต่ละภาพเป็น 1D vector เสมอ ไม่ว่า batch size จะเป็นเท่าไหร่
        out = output.detach().cpu()
        for vec in out.view(out.size(0), -1).numpy():
            features_list.append(vec)

    handle = net.avgpool.register_forward_hook(hook_fn)
    net.eval()
    with torch.no_grad():
        for imgs, labels in DataLoader(dataset, batch_size=64, num_workers=0):
            imgs = imgs.to(device)
            net(imgs)
            labels_list.extend(labels.numpy())

    handle.remove()

    features_arr = np.array(features_list)  # (N, D) — shape สม่ำเสมอแน่นอน
    labels_arr   = np.array(labels_list)

    centroids = {}
    for cls_idx, cls_name in enumerate(CLASSIFIER_CLASSES):
        mask = labels_arr == cls_idx
        if mask.sum() > 0:
            centroids[cls_name] = features_arr[mask].mean(axis=0)
            print(f"  centroid [{cls_name}] จาก {mask.sum()} ภาพ  "
                  f"dim={centroids[cls_name].shape[0]}")

    np.save(OOD_CENTROIDS_PATH, centroids)
    print(f"[OOD] บันทึก → {OOD_CENTROIDS_PATH}  (threshold={OOD_DISTANCE_THRESH})")


# ══════════════════════════════════════════
#  SHARED — analyze_color + calibration
# ══════════════════════════════════════════
def analyze_color(roi_bgr, yolo_class):
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown", 0.0
    profiles = HSV_PROFILES.get(yolo_class, [])
    if not profiles:
        return "n/a", 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    best_name, best_ratio = "unknown", 0.0
    for p in profiles:
        mask  = cv2.inRange(hsv,
                            np.array(p["lower"], dtype=np.uint8),
                            np.array(p["upper"], dtype=np.uint8))
        ratio = np.sum(mask > 0) / mask.size
        if ratio > best_ratio:
            best_ratio, best_name = ratio, p["name"]
    return best_name, round(best_ratio, 2)

_cs = {"active": False, "class": "can", "profile": 0,
       "h_low": 0, "s_low": 0, "v_low": 140,
       "h_high": 180, "s_high": 40, "v_high": 255}

def _nothing(x): pass

def _sync_tb(p):
    w = _cs["win"]
    for k, v in zip(["H Low", "S Low", "V Low", "H High", "S High", "V High"],
                    p["lower"] + p["upper"]):
        cv2.setTrackbarPos(k, w, v)
    _cs.update({"h_low": p["lower"][0], "s_low": p["lower"][1], "v_low": p["lower"][2],
                "h_high": p["upper"][0], "s_high": p["upper"][1], "v_high": p["upper"][2]})

def open_calib():
    win = "HSV Calibration  [S=Save  N=Next profile  Q=Close]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 640, 300)
    for k, mx, val in [("H Low", 180, _cs["h_low"]),   ("S Low", 255, _cs["s_low"]),
                       ("V Low", 255, _cs["v_low"]),   ("H High", 180, _cs["h_high"]),
                       ("S High", 255, _cs["s_high"]), ("V High", 255, _cs["v_high"])]:
        cv2.createTrackbar(k, win, val, mx, _nothing)
    _cs["active"], _cs["win"] = True, win
    print(f"[CALIB] เปิดแล้ว — class:{_cs['class']}  S=save  N=next  Q=ปิด")

def update_calib(roi_bgr):
    if not _cs.get("active"):
        return
    win = _cs["win"]
    hl, sl, vl = [cv2.getTrackbarPos(k, win) for k in ["H Low", "S Low", "V Low"]]
    hh, sh, vh = [cv2.getTrackbarPos(k, win) for k in ["H High", "S High", "V High"]]
    _cs.update({"h_low": hl, "s_low": sl, "v_low": vl,
                "h_high": hh, "s_high": sh, "v_high": vh})

    if roi_bgr is None or roi_bgr.size == 0:
        blank = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "ไม่พบวัตถุ", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow(win, blank)
    else:
        hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([hl, sl, vl], dtype=np.uint8),
                           np.array([hh, sh, vh], dtype=np.uint8))
        ratio = np.sum(mask > 0) / mask.size
        rs = cv2.resize(roi_bgr, (300, 240))
        ms = cv2.cvtColor(cv2.resize(mask, (300, 240)), cv2.COLOR_GRAY2BGR)
        cb = np.hstack([rs, ms])
        pname = HSV_PROFILES.get(_cs["class"], [{}])[_cs["profile"]].get("name", "?")
        info  = (f"Class:{_cs['class']} Profile[{_cs['profile']}]:{pname} "
                 f"Match:{ratio*100:.1f}% lower=({hl},{sl},{vl}) upper=({hh},{sh},{vh})")
        cv2.putText(cb, info, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.imshow(win, cb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cp = HSV_PROFILES.get(_cs["class"])
        pi = _cs["profile"]
        if cp and pi < len(cp):
            cp[pi]["lower"] = [hl, sl, vl]
            cp[pi]["upper"] = [hh, sh, vh]
            print(f"[SAVED] {_cs['class']}.{cp[pi]['name']} lower={[hl,sl,vl]} upper={[hh,sh,vh]}")
    elif key == ord('n'):
        cp = HSV_PROFILES.get(_cs["class"], [])
        if cp:
            _cs["profile"] = (_cs["profile"] + 1) % len(cp)
            _sync_tb(cp[_cs["profile"]])
            print(f"[CALIB] profile → {cp[_cs['profile']]['name']}")
    elif key == ord('q'):
        cv2.destroyWindow(win)
        _cs["active"] = False
        print("[CALIB] ปิดแล้ว")

def cycle_calib_class():
    classes = list(HSV_PROFILES.keys())
    nxt = classes[(classes.index(_cs["class"]) + 1) % len(classes)]
    _cs["class"], _cs["profile"] = nxt, 0
    if _cs.get("active"):
        _sync_tb(HSV_PROFILES[nxt][0])
    print(f"[CALIB] class → {nxt}")


# ══════════════════════════════════════════
#  SHARED — load classifier + OOD centroids
# ══════════════════════════════════════════
def load_classifier():
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"[WARN] ไม่พบ {CLASSIFIER_PATH} — ใช้ YOLO อย่างเดียว")
        return None, None, None
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        net = models.mobilenet_v3_small()
        net.classifier[3] = nn.Linear(1024, len(CLASSIFIER_CLASSES))
        net.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu"))
        net.eval()

        tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # โหลด OOD centroids
        centroids = None
        if os.path.exists(OOD_CENTROIDS_PATH):
            centroids = np.load(OOD_CENTROIDS_PATH, allow_pickle=True).item()
            print(f"[OOD] โหลด centroids สำเร็จ — threshold={OOD_DISTANCE_THRESH}")
        else:
            print(f"[WARN] ไม่พบ {OOD_CENTROIDS_PATH} — OOD ปิดอยู่")
            print("  วิธีสร้าง: รัน --mode train ใหม่อีกครั้ง")

        print(f"[OK] Classifier โหลดแล้ว (softmax_thresh={CLASSIFIER_THRESHOLD})")
        return net, tf, centroids

    except Exception as e:
        print(f"[WARN] โหลด classifier ไม่ได้: {e}")
        return None, None, None


def classify_roi(roi_bgr, clf, tf, centroids):
    """
    3 ชั้นในการตัดสิน:
      ชั้น 1 — Softmax threshold : score < CLASSIFIER_THRESHOLD → other
      ชั้น 2 — OOD distance      : ห่างจาก centroid เกิน OOD_DISTANCE_THRESH → other
      ชั้น 3 — ผ่านทั้งคู่       → คืน class ที่ classifier เลือก

    คืนค่า: (label, clf_score, ood_dist)
    """
    if clf is None or roi_bgr is None or roi_bgr.size == 0:
        return None, 0.0, 999.0

    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        img    = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        tensor = tf(img).unsqueeze(0)

        # hook ดึง feature vector จาก avgpool
        feature_vec = []
        def hook_fn(module, input, output):
            feature_vec.append(output.squeeze().detach().cpu().numpy())
        handle = clf.avgpool.register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = clf(tensor)
            scores = F.softmax(logits, dim=1)[0]

        handle.remove()

        score     = float(scores.max())
        cls_label = CLASSIFIER_CLASSES[int(scores.argmax())]

        # ชั้น 1: Softmax gate
        if score < CLASSIFIER_THRESHOLD:
            return "other", score, 999.0

        # ชั้น 2: OOD feature distance
        ood_dist = 999.0
        if centroids is not None and len(feature_vec) > 0:
            fvec     = feature_vec[0]
            min_dist = min(np.linalg.norm(fvec - centroids[c]) for c in centroids)
            ood_dist = round(float(min_dist), 2)
            if min_dist > OOD_DISTANCE_THRESH:
                return "other", score, ood_dist

        return cls_label, score, ood_dist

    except Exception as e:
        print(f"[ERR] classify_roi: {e}")
        return None, 0.0, 999.0


# ══════════════════════════════════════════
#  MODE 4 — DETECT (main loop)
# ══════════════════════════════════════════
def run_detect():
    print("\n[DETECT] เปิดระบบ")
    print("  Q=ออก | C=เปิด/เปลี่ยน Calibration\n")

    # Arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)
        print("Arduino Connected!")
        arduino_connected = True
    except:
        print("Arduino not connected")
        arduino_connected, arduino = False, None

    # YOLO
    print("Loading YOLO...")
    yolo = YOLO(YOLO_MODEL_PATH)
    print("YOLO Loaded!")

    # Classifier + OOD
    clf, clf_tf, centroids = load_classifier()
    use_twostage = clf is not None
    ood_active   = centroids is not None
    mode_text    = ("two-stage+OOD" if ood_active else "two-stage") if use_twostage else "YOLO only"

    # Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera error")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State
    working          = False
    detect_counter   = 0
    DETECT_THRESHOLD = 10
    DETECT_DELAY     = 3
    last_det_time    = 0
    last_act_time    = 0
    COOLDOWN         = 5
    last_labels      = []
    current_roi      = None

    print(f"System Ready — {mode_text}  |  Q=ออก  C=Calibrate")

    try:
        while True:
            time.sleep(0.03)
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, imgsz=640, conf=0.25, verbose=False)
            boxes   = results[0].boxes
            display = frame.copy()

            # ── ไม่พบวัตถุ ──
            if boxes is None or len(boxes) == 0:
                detect_counter = 0
                working        = False
                current_roi    = None
                update_calib(None)
                cv2.imshow("YOLO Waste Detection", display)
                key = cv2.waitKey(1)
                if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord('c'):
                    open_calib() if not _cs.get("active") else cycle_calib_class()
                continue

            # ── เลือก object ใหญ่สุด ──
            areas           = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy]
            i               = areas.index(max(areas))
            cls_id          = int(boxes.cls[i])
            conf            = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            area            = (x2-x1)*(y2-y1)
            frame_area      = frame.shape[0] * frame.shape[1]

            if area > frame_area * 0.7:
                update_calib(None)
                cv2.imshow("YOLO Waste Detection", display)
                cv2.waitKey(1)
                continue

            current_roi = frame[y1:y2, x1:x2].copy()
            clf_score   = 0.0
            ood_dist    = 999.0

            # ── ตัดสิน label ──
            if area < 3000 or conf < 0.25:
                label = "other"
            elif use_twostage:
                clf_label, clf_score, ood_dist = classify_roi(
                    current_roi, clf, clf_tf, centroids)
                label = clf_label if clf_label is not None else yolo.names[cls_id].lower()
            else:
                label = "other" if conf < 0.35 else yolo.names[cls_id].lower()

            # ── Color Gate (ข้าม PET-Bottle เพราะขวดใสสีไม่นิ่ง) ──
            color_name, color_ratio = analyze_color(current_roi, label)
            if label != "other" and label != "PET-Bottle" and color_ratio < 0.06:
                label      = "other"
                color_name = "no_match"

            # ── Voting ──
            last_labels.append(label)
            if len(last_labels) > 7:
                last_labels.pop(0)
            final_label = Counter(last_labels).most_common(1)[0][0]

            # ── Draw ──
            bc = BOX_COLORS.get(final_label, (100, 100, 100))
            cv2.rectangle(display, (x1, y1), (x2, y2), bc, 2)
            cv2.putText(display, f"{final_label} [{color_name}]",
                        (x1, y1-26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, bc, 2, cv2.LINE_AA)

            if use_twostage:
                ood_txt   = f"ood:{ood_dist:.1f}" if ood_active else "ood:off"
                score_txt = f"clf:{clf_score:.2f}  {ood_txt}  color:{color_ratio:.0%}"
            else:
                score_txt = f"conf:{conf:.2f}  color:{color_ratio:.0%}"

            cv2.putText(display, score_txt,
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, bc, 1, cv2.LINE_AA)
            cv2.putText(display, f"[C] calib:{_cs['class']}  {mode_text}",
                        (8, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow("YOLO Waste Detection", display)
            update_calib(current_roi)

            # ── Trigger ──
            if not working:
                detect_counter += 1
                if detect_counter >= DETECT_THRESHOLD:
                    if time.time() - last_act_time >= COOLDOWN:
                        if time.time() - last_det_time > DETECT_DELAY:
                            if final_label != "other":
                                working        = True
                                detect_counter = 0
                                last_det_time  = time.time()
                                last_act_time  = time.time()

                                print("\n======================")
                                print(f"Detected : {final_label}")
                                print(f"Color    : {color_name} ({color_ratio:.0%})")
                                if use_twostage:
                                    print(f"Clf score: {clf_score:.3f}")
                                if ood_active:
                                    print(f"OOD dist : {ood_dist:.2f}  (thresh={OOD_DISTANCE_THRESH})")
                                print("======================")

                                try:
                                    requests.post(SERVER_URL,
                                                  json={"type": final_label}, timeout=1)
                                except:
                                    print("Dashboard error")

                                if arduino_connected:
                                    try:
                                        arduino.write((final_label + "\n").encode())
                                        start = time.time()
                                        while True:
                                            if cv2.getWindowProperty(
                                                    "YOLO Waste Detection",
                                                    cv2.WND_PROP_VISIBLE) < 1:
                                                sys.exit()
                                            if arduino.in_waiting:
                                                if arduino.readline().decode().strip() == "DONE":
                                                    break
                                            if time.time() - start > 5:
                                                break
                                    except:
                                        print("Arduino error")

                                working = False

            # ── Key ──
            key = cv2.waitKey(1)
            if cv2.getWindowProperty("YOLO Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('c'):
                open_calib() if not _cs.get("active") else cycle_calib_class()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arduino_connected and arduino:
            arduino.close()
        print("===== PROGRAM END =====")


# ══════════════════════════════════════════
#  MODE 5 — CENTROIDS
#  สร้าง OOD centroids จาก classifier.pth ที่มีอยู่
#  โดยไม่ต้อง retrain
# ══════════════════════════════════════════
def run_centroids():
    try:
        import torch
        import torch.nn as nn
        from torchvision import datasets, transforms, models
    except ImportError:
        print("[ERROR] ติดตั้งก่อน: pip install torch torchvision")
        return

    if not os.path.exists(CLASSIFIER_PATH):
        print(f"[ERROR] ไม่พบ {CLASSIFIER_PATH} — รัน --mode train ก่อน")
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[CENTROIDS] โหลด {CLASSIFIER_PATH}  device={DEVICE}")

    net = models.mobilenet_v3_small()
    net.classifier[3] = nn.Linear(1024, len(CLASSIFIER_CLASSES))
    net.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    net = net.to(DEVICE)
    net.eval()

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(CLASSIFIER_DATA_ROOT, "train")
    if not os.path.exists(train_path):
        print(f"[ERROR] ไม่พบ {train_path} — รัน --mode prepare ก่อน")
        return

    train_ds = datasets.ImageFolder(train_path, transform=tf)
    print(f"  ภาพที่ใช้คำนวณ: {len(train_ds)}  classes: {train_ds.classes}")

    _compute_and_save_centroids(net, train_ds, DEVICE)
    print("[OK] พร้อมใช้งาน — รัน --mode detect ได้เลย")


# ══════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waste Detection All-in-one")
    parser.add_argument("--mode",
                        choices=["prepare", "collect", "train", "centroids", "detect"],
                        default="detect",
                        help="prepare=crop ROI | collect=เก็บจากกล้อง | train=train | centroids=สร้าง OOD | detect=รันจริง")
    args = parser.parse_args()

    {"prepare":   run_prepare,
     "collect":   run_collect,
     "train":     run_train,
     "centroids": run_centroids,
     "detect":    run_detect}[args.mode]()