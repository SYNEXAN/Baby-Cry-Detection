import os
import json
import zipfile
from pathlib import Path

V4_DIR = Path(r"data\processed\vision_v4_clean_balanced")
UPLOAD_DIR = Path(r"data\kaggle_upload\vision_v4_clean_balanced")
KERNEL_DIR = Path(r"kaggle_kernel_v4_clean")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
KERNEL_DIR.mkdir(parents=True, exist_ok=True)

def build_dataset_package():
    zip_path = UPLOAD_DIR / "vision_v4_clean_balanced.zip"
    print(f"Zipping {V4_DIR} -> {zip_path} ...")
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(V4_DIR):
            for f in files:
                fp = Path(root) / f
                arc = fp.relative_to(V4_DIR)
                zf.write(fp, arc)
                count += 1
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Packaged {count} files ({size_mb:.2f} MB).")

    ds_meta = {
        "title": "Vision V4 Clean Balanced Kids Adults",
        "id": "thittn/vision-v4-clean-balanced",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(ds_meta, indent=2), encoding="utf-8")
    print("Wrote dataset-metadata.json")

def make_md_cell(source_str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_str.strip().splitlines()],
    }

def make_code_cell(source_str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_str.strip().splitlines()],
    }

def build_notebook():
    cells = [
        make_md_cell("""
# YOLO11-M Training Pipeline: V4 Clean-Balanced Kids vs Adults Detection

## Objective
Train a single pure YOLO11-M detector on the audited, cleaned, and rebalanced 3-source dataset (`thittn/vision-v4-clean-balanced`) with:
- Auto-audited removal of unlabeled foreground person images from Aniruddha
- Removal of tiny noise boxes (<0.8% area) and heavy salt-and-pepper noise images from True-Baby
- Stratified 70/15/15 Zero-Data-Leakage split with 2x Train-only oversampling on high-resolution Kaushi images
- High-resolution training (`imgsz=960`, `batch=8`, `cls=1.0`, `scale=0.25`, `mosaic=0.0`, `mixup=0.0`)
"""),
        make_code_cell("""
# Cell 1: Install Ultralytics and Verify Hardware Environment
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

import os, glob, time, hashlib, shutil, random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import ultralytics
from ultralytics import YOLO

print("=================================================================")
print("ENVIRONMENT AND HARDWARE VERIFICATION")
print("=================================================================")
print(f"PyTorch Version     : {torch.__version__}")
print(f"Ultralytics Version : {ultralytics.__version__}")
print(f"CUDA Available      : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Primary GPU         : {gpu_name} ({vram_gb:.2f} GB VRAM)")
else:
    raise RuntimeError("GPU is required for training.")
"""),
        make_md_cell("## 2. Dataset Discovery and Integrity Audit"),
        make_code_cell("""
# Cell 2: Locate Dataset and Audit Split Counts and Class Distribution
CANDIDATES = [
    Path("/kaggle/input/vision-v4-clean-balanced"),
    Path("/kaggle/input/datasets/thittn/vision-v4-clean-balanced"),
]
DATA_ROOT = None
for c in CANDIDATES:
    if c.exists():
        if (c / "train" / "images").exists():
            DATA_ROOT = c
            break
        zips = list(c.glob("*.zip"))
        if zips:
            import zipfile
            extract_dir = Path("/kaggle/working/dataset_v4")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(extract_dir)
            DATA_ROOT = extract_dir
            break

if DATA_ROOT is None:
    for p in Path("/kaggle/input").rglob("train"):
        if (p / "images").exists():
            DATA_ROOT = p.parent
            break

assert DATA_ROOT is not None and (DATA_ROOT / "train" / "images").exists(), "Dataset not found!"
print(f"Dataset Root Resolved: {DATA_ROOT}")

stats = {}
total_imgs = 0
total_adults = 0
total_kids = 0
for sp in ["train", "valid", "test"]:
    imgs = sorted((DATA_ROOT / sp / "images").iterdir())
    lbls = sorted((DATA_ROOT / sp / "labels").iterdir())
    n_a, n_k = 0, 0
    for lp in lbls:
        for line in lp.read_text().splitlines():
            p = line.strip().split()
            if len(p) >= 5:
                if int(float(p[0])) == 0:
                    n_a += 1
                elif int(float(p[0])) == 1:
                    n_k += 1
    stats[sp] = (len(imgs), n_a, n_k)
    total_imgs += len(imgs)
    total_adults += n_a
    total_kids += n_k
    print(f"Split [{sp:5s}]: {len(imgs):4d} images | adults={n_a:4d} | kids={n_k:4d} | ratio kids:adults = {n_k/max(1,n_a):.2f}:1")

print(f"TOTAL        : {total_imgs:4d} images | adults={total_adults:4d} | kids={total_kids:4d} | ratio kids:adults = {total_kids/max(1,total_adults):.2f}:1")
"""),
        make_md_cell("## 3. Visual Inspection of Training Samples"),
        make_code_cell("""
# Cell 3: Plot Sample Ground Truth Images from Train Split
random.seed(42)
train_imgs = sorted((DATA_ROOT / "train" / "images").iterdir())
picks = random.sample(train_imgs, min(6, len(train_imgs)))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, img_path in zip(axes.ravel(), picks):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    lp = DATA_ROOT / "train" / "labels" / f"{img_path.stem}.txt"
    if lp.exists():
        for line in lp.read_text().splitlines():
            p = line.strip().split()
            if len(p) >= 5:
                c = int(float(p[0]))
                xc, yc, bw, bh = map(float, p[1:5])
                x1, y1 = int((xc - bw/2)*w), int((yc - bh/2)*h)
                x2, y2 = int((xc + bw/2)*w), int((yc + bh/2)*h)
                col = (0, 120, 255) if c == 0 else (255, 180, 0)
                lbl = "adults" if c == 0 else "kids"
                cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, w // 300))
                cv2.putText(img, lbl, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 900), col, 2)
    ax.imshow(img)
    ax.set_title(img_path.name[:32], fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("/kaggle/working/sample_inspection.png", dpi=140)
plt.close()
print("Saved /kaggle/working/sample_inspection.png")
"""),
        make_md_cell("## 4. Training Configuration and Execution (YOLO11-M, imgsz=960, batch=8, cls=1.0)"),
        make_code_cell("""
# Cell 4: Write data_v4.yaml and Execute Training
yaml_path = Path("/kaggle/working/data_v4.yaml")
yaml_content = f\"\"\"path: {DATA_ROOT}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: adults
  1: kids
\"\"\"
yaml_path.write_text(yaml_content, encoding="utf-8")
print(yaml_content)

t0 = time.time()
model = YOLO("yolo11m.pt")
train_results = model.train(
    data=str(yaml_path),
    epochs=45,
    patience=15,
    imgsz=960,
    batch=8,
    device=0,
    workers=4,
    optimizer="AdamW",
    lr0=0.0008,
    lrf=0.01,
    cos_lr=True,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    cls=1.0,
    box=7.5,
    dfl=1.5,
    scale=0.25,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.10,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    fliplr=0.5,
    project="/kaggle/working/runs",
    name="yolo11m_v4_clean_balanced",
    exist_ok=True,
    plots=True,
    save=True,
    verbose=True,
)
train_minutes = (time.time() - t0) / 60.0
print(f"Training completed in {train_minutes:.2f} minutes.")
"""),
        make_md_cell("## 5. Training Metrics Visualization"),
        make_code_cell("""
# Cell 5: Plot Training Curves (Box Loss, Cls Loss, Precision/Recall, mAP)
run_dir = Path("/kaggle/working/runs/yolo11m_v4_clean_balanced")
df = pd.read_csv(run_dir / "results.csv")
df.columns = [c.strip() for c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ep = df["epoch"]

axes[0, 0].plot(ep, df["train/box_loss"], label="Train Box Loss", color="blue", lw=2)
axes[0, 0].plot(ep, df["val/box_loss"], label="Val Box Loss", color="red", ls="--", lw=2)
axes[0, 0].set_title("Bounding Box Loss (CIoU)")
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss"); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

axes[0, 1].plot(ep, df["train/cls_loss"], label="Train Cls Loss", color="blue", lw=2)
axes[0, 1].plot(ep, df["val/cls_loss"], label="Val Cls Loss", color="red", ls="--", lw=2)
axes[0, 1].set_title("Classification Loss (adults vs kids, cls=1.0)")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss"); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

axes[1, 0].plot(ep, df["metrics/precision(B)"], label="Precision", color="green", lw=2)
axes[1, 0].plot(ep, df["metrics/recall(B)"], label="Recall", color="purple", ls="--", lw=2)
axes[1, 0].set_title("Validation Precision and Recall")
axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Score"); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

axes[1, 1].plot(ep, df["metrics/mAP50(B)"], label="mAP@50", color="teal", lw=2)
axes[1, 1].plot(ep, df["metrics/mAP50-95(B)"], label="mAP@50-95", color="orange", lw=2)
axes[1, 1].set_title("Validation Mean Average Precision (mAP)")
axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("mAP"); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

plt.tight_layout()
plt.savefig("/kaggle/working/training_metrics_curves.png", dpi=150)
plt.close()
print("Saved /kaggle/working/training_metrics_curves.png")
"""),
        make_md_cell("## 6. Independent Test Set Evaluation"),
        make_code_cell("""
# Cell 6: Evaluate best.pt on Held-Out Test Split (15%, Zero Leakage)
best_pt_src = run_dir / "weights" / "best.pt"
best_model = YOLO(str(best_pt_src))
test_metrics = best_model.val(
    data=str(yaml_path),
    split="test",
    imgsz=960,
    batch=8,
    conf=0.25,
    iou=0.45,
    agnostic_nms=True,
    plots=True,
)
print("=================================================================")
print("INDEPENDENT TEST SET EVALUATION METRICS")
print("=================================================================")
print(f"Test Precision (B) : {test_metrics.box.mp:.4f}")
print(f"Test Recall (B)    : {test_metrics.box.mr:.4f}")
print(f"Test mAP@50 (B)    : {test_metrics.box.map50:.4f}")
print(f"Test mAP@50-95 (B) : {test_metrics.box.map:.4f}")
"""),
        make_md_cell("## 7. Test Set Visual Inference (Ground Truth vs Prediction)"),
        make_code_cell("""
# Cell 7: Side-by-Side Ground Truth vs Prediction on Test Split
test_imgs = sorted((DATA_ROOT / "test" / "images").iterdir())
# Pick representative samples across kaushi, anir, tb
picks = []
for prefix in ["kaushi_", "anir_", "tb_"]:
    sub = [p for p in test_imgs if p.name.startswith(prefix)]
    if sub:
        picks.extend(random.sample(sub, min(2, len(sub))))
picks = picks[:6]

fig, axes = plt.subplots(len(picks), 2, figsize=(14, 4.5 * len(picks)))
for i, img_path in enumerate(picks):
    orig = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = orig.shape[:2]

    # Ground Truth
    gt_img = orig.copy()
    lp = DATA_ROOT / "test" / "labels" / f"{img_path.stem}.txt"
    if lp.exists():
        for line in lp.read_text().splitlines():
            p = line.strip().split()
            if len(p) >= 5:
                c = int(float(p[0]))
                xc, yc, bw, bh = map(float, p[1:5])
                x1, y1 = int((xc - bw/2)*w), int((yc - bh/2)*h)
                x2, y2 = int((xc + bw/2)*w), int((yc + bh/2)*h)
                col = (0, 180, 0) if c == 0 else (255, 140, 0)
                lbl = "GT: adults" if c == 0 else "GT: kids"
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), col, max(2, w // 300))
                cv2.putText(gt_img, lbl, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 900), col, 2)

    # Prediction
    pred_img = orig.copy()
    res = best_model.predict(str(img_path), imgsz=960, conf=0.35, iou=0.45, agnostic_nms=True, verbose=False)[0]
    for b in res.boxes:
        c = int(b.cls[0])
        cf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        col = (0, 100, 255) if c == 0 else (0, 210, 255)
        lbl = f"{'adults' if c == 0 else 'kids'} {cf:.2f}"
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), col, max(2, w // 300))
        cv2.putText(pred_img, lbl, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 900), col, 2)

    axes[i, 0].imshow(gt_img); axes[i, 0].set_title(f"GT - {img_path.name[:30]}", fontsize=10); axes[i, 0].axis("off")
    axes[i, 1].imshow(pred_img); axes[i, 1].set_title(f"YOLO11-M V4 PRED", fontsize=10); axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("/kaggle/working/test_predictions_comparison.png", dpi=140)
plt.close()
print("Saved /kaggle/working/test_predictions_comparison.png")
"""),
        make_md_cell("## 8. Export Weights and Final Report"),
        make_code_cell("""
# Cell 8: Copy best.pt, Compute MD5, and Write Final Report
dst_pt = Path("/kaggle/working/best.pt")
shutil.copy2(best_pt_src, dst_pt)
md5 = hashlib.md5(dst_pt.read_bytes()).hexdigest()
size_mb = dst_pt.stat().st_size / (1024 * 1024)

report = f\"\"\"# FINAL TRAINING EXPERIMENT REPORT (V4 Clean-Balanced Dataset)
## 1. Environment and Hardware
- Primary GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)
- Framework: Ultralytics {ultralytics.__version__} (PyTorch {torch.__version__})

## 2. Dataset Configuration
- Dataset: thittn/vision-v4-clean-balanced (Cleaned Aniruddha + Balanced True-Baby + 2x Kaushi Train)
- Total Images: {total_imgs} (Train: {stats['train'][0]}, Valid: {stats['valid'][0]}, Test: {stats['test'][0]})
- Target Classes: [0] adults ({total_adults} boxes), [1] kids ({total_kids} boxes)

## 3. Training and Hyperparameters
- Architecture: YOLO11-M
- Epochs: 45 (patience=15, cos_lr=True) | Batch: 8 (nbs=64) | Image Size: 960x960
- Loss & Augmentations: cls=1.0, scale=0.25, mosaic=0.0, mixup=0.0, erasing=0.10
- Training Duration: {train_minutes:.2f} minutes

## 4. Final Evaluation (Independent Test Split)
- Test Precision: {test_metrics.box.mp:.4f}
- Test Recall   : {test_metrics.box.mr:.4f}
- Test mAP@50   : {test_metrics.box.map50:.4f}
- Test mAP@50-95: {test_metrics.box.map:.4f}

## 5. Output Artifacts
- Weights: /kaggle/working/best.pt ({size_mb:.2f} MB, MD5: {md5})
- Curves : /kaggle/working/training_metrics_curves.png
- Grid   : /kaggle/working/test_predictions_comparison.png
\"\"\"
Path("/kaggle/working/experiment_summary.md").write_text(report, encoding="utf-8")
print(report)
print("EXPERIMENT EXECUTION SUCCESSFULLY COMPLETED.")
"""),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    nb_path = KERNEL_DIR / "train-yolo11-m-v4-clean-balanced.ipynb"
    nb_text = json.dumps(nb, indent=2, ensure_ascii=True)
    # Verify strictly ASCII (0 emojis/icons)
    nb_text.encode("ascii")
    nb_path.write_text(nb_text, encoding="utf-8")
    print(f"Wrote ZERO-ICON notebook: {nb_path} ({len(cells)} cells, 100% ASCII verified)")

    kmeta = {
        "id": "thittn/train-yolo11-m-v4-clean-balanced",
        "title": "Train YOLO11-M V4 Clean Balanced",
        "code_file": "train-yolo11-m-v4-clean-balanced.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["thittn/vision-v4-clean-balanced"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01",
        "machine_shape": "NvidiaTeslaT4",
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print("Wrote kernel-metadata.json")

if __name__ == "__main__":
    build_dataset_package()
    build_notebook()
