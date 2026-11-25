"""
Script: create_vision_v6_notebook.py
Purpose: Generate the 13-section compliant training notebook for RT-DETR-L Version 6
         Adheres strictly to KAGGLE_TRAINING_WORKFLOW_POLICY.md and KAGGLE_MCP_POLICY.md.
"""

import os
import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell("""# 👶 Infant & Caregiver Adult Detection - RT-DETR-L (Version 6)
**Architecture:** `RT-DETR-L` (Real-Time Detection Transformer with HGNetv2 Backbone & NMS-Free Head)  
**Dataset:** `thietnguyen/true-baby-dataset` (2,698 images, Balanced Full-Body Adult & Baby)  
**Classes:** `['adult', 'baby_child']`  
**Compliance:** Strict adherence to `KAGGLE_TRAINING_WORKFLOW_POLICY.md` (13 Standardized Sections)
"""))

    # =========================================================================
    # Section 1: Environment and GPU verification
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 🛠️ 1. Environment and GPU Verification"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 1: Environment and GPU verification
import os
import sys
import torch

print("=" * 60)
print("1. ENVIRONMENT & GPU HARDWARE VERIFICATION")
print("=" * 60)
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemError("❌ CRITICAL: GPU acceleration is NOT available. Halting per policy.")

gpu_name = torch.cuda.get_device_name(0)
gpu_props = torch.cuda.get_device_properties(0)
vram_gb = gpu_props.total_memory / (1024 ** 3)
print(f"✅ GPU Device Name: {gpu_name}")
print(f"✅ GPU Total Memory: {vram_gb:.2f} GB")
print(f"✅ CUDA Device Count: {torch.cuda.device_count()}")
assert vram_gb >= 14.0, f"Expected Tesla T4 (>=14GB), found {vram_gb:.2f} GB"
"""))

    # =========================================================================
    # Section 2: Dependency installation and imports
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 📦 2. Dependency Installation and Imports"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 2: Dependency installation and imports
!pip install -q ultralytics matplotlib seaborn pandas opencv-python pyyaml

import ultralytics
import cv2
import yaml
import glob
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

print(f"✅ Ultralytics Version: {ultralytics.__version__}")
ultralytics.checks()
"""))

    # =========================================================================
    # Section 3: Dataset connection and configuration
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 📁 3. Dataset Connection and Configuration"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 3: Dataset connection and configuration
POSSIBLE_INPUT_DIRS = [
    "/kaggle/input/true-baby-dataset",
    "/kaggle/input/datasets/thietnguyen/true-baby-dataset",
    "/kaggle/input/true-baby-dataset-v6"
]

KAGGLE_INPUT_DIR = None
for d in POSSIBLE_INPUT_DIRS:
    if os.path.exists(d):
        KAGGLE_INPUT_DIR = d
        break

if KAGGLE_INPUT_DIR is None:
    # Try finding any true-baby directory
    matches = glob.glob("/kaggle/input/*baby*")
    if matches:
        KAGGLE_INPUT_DIR = matches[0]

assert KAGGLE_INPUT_DIR is not None, "❌ Dataset not found in /kaggle/input!"
print(f"📍 Mounted Dataset Path: {KAGGLE_INPUT_DIR}")

WORK_DATA_DIR = "/kaggle/working/dataset_v6"
os.makedirs(WORK_DATA_DIR, exist_ok=True)

# Check if unzipped or extract zips
if os.path.exists(os.path.join(KAGGLE_INPUT_DIR, "train", "images")):
    print("✅ Dataset already uncompressed in input directory.")
    DATASET_ROOT = KAGGLE_INPUT_DIR
else:
    print("🔄 Extracting train.zip, valid.zip, test.zip to working directory...")
    for split_zip in ["train.zip", "valid.zip", "test.zip"]:
        zip_path = os.path.join(KAGGLE_INPUT_DIR, split_zip)
        split_name = split_zip.replace(".zip", "")
        dest_split = os.path.join(WORK_DATA_DIR, split_name)
        if os.path.exists(zip_path):
            print(f"   Unzipping {split_zip} -> {dest_split}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(dest_split)
    DATASET_ROOT = WORK_DATA_DIR

# Generate data_v6.yaml
data_yaml_dict = {
    "path": DATASET_ROOT,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 2,
    "names": ["adult", "baby_child"]
}

YAML_PATH = "/kaggle/working/data_v6.yaml"
with open(YAML_PATH, "w") as f:
    yaml.dump(data_yaml_dict, f, sort_keys=False)

print(f"✅ Generated dataset config at: {YAML_PATH}")
with open(YAML_PATH) as f:
    print(f.read())
"""))

    # =========================================================================
    # Section 4: Dataset structure and annotation validation
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 🔍 4. Dataset Structure and Annotation Validation"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 4: Dataset structure and annotation validation
splits = ["train", "valid", "test"]
box_counts = {s: {0: 0, 1: 0} for s in splits}
img_counts = {s: 0 for s in splits}

for s in splits:
    lbl_files = glob.glob(os.path.join(DATASET_ROOT, s, "labels", "*.txt"))
    img_counts[s] = len(lbl_files)
    for lf in lbl_files:
        with open(lf) as f:
            for line in f:
                p = line.strip().split()
                if p:
                    c = int(p[0])
                    assert c in [0, 1], f"Invalid class ID {c} in {lf}"
                    box_counts[s][c] += 1
                    # Validate bbox ranges
                    cx, cy, bw, bh = map(float, p[1:5])
                    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                    assert 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0

print("=" * 60)
print("📊 DATASET INTEGRITY VALIDATION")
print("=" * 60)
for s in splits:
    total_b = box_counts[s][0] + box_counts[s][1]
    print(f"Split {s.upper():<6}: {img_counts[s]:<5} images | Adult: {box_counts[s][0]:<5} | Baby: {box_counts[s][1]:<5} | Total: {total_b:<5}")

assert img_counts["train"] > 1500, "Train set must have >1500 images"
assert box_counts["train"][0] > 500, "Train set must have >500 adult boxes"
print("✅ All splits, bounding boxes, and label constraints validated successfully!")
"""))

    # =========================================================================
    # Section 5: Data visualization and sample inspection
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 🖼️ 5. Data Visualization and Sample Inspection"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 5: Data visualization and sample inspection
def visualize_sample_grid(split='train', num_samples=6):
    lbl_files = sorted(glob.glob(os.path.join(DATASET_ROOT, split, "labels", "*.txt")))
    # Sample diverse indices
    step = max(1, len(lbl_files) // num_samples)
    selected = [lbl_files[i * step] for i in range(num_samples)]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    class_names = ["Adult", "Baby/Child"]
    colors = [(0, 230, 115), (255, 60, 60)] # Green for adult, Red for baby
    
    for ax, lf in zip(axes, selected):
        img_f = lf.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep).replace(".txt", ".jpg")
        img = cv2.imread(img_f)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        with open(lf) as f:
            for line in f:
                p = line.strip().split()
                if p:
                    c = int(p[0])
                    cx, cy, bw, bh = map(float, p[1:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    color = colors[c]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, class_names[c], (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
        ax.imshow(img)
        ax.set_title(f"{os.path.basename(img_f)[:25]}...", fontsize=10)
        ax.axis("off")
        
    plt.suptitle("Sample Training Inspections (Green=Adult Full-Body, Red=Baby)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/kaggle/working/dataset_samples_inspection.jpg", dpi=150)
    plt.show()

visualize_sample_grid()
"""))

    # =========================================================================
    # Section 6: Model configuration and initialization
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 🤖 6. Model Configuration and Initialization"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 6: Model configuration and initialization
from ultralytics import RTDETR

print("=" * 60)
print("🤖 INITIALIZING RT-DETR-L MODEL ARCHITECTURE")
print("=" * 60)

# Load official pretrained RT-DETR-L weights
model = RTDETR("rtdetr-l.pt")
print("✅ Loaded official pretrained weights: rtdetr-l.pt")
print(f"Model Task: {model.task}")
"""))

    # =========================================================================
    # Section 7: Training configuration
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### ⚙️ 7. Training Hyperparameter Configuration"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 7: Training configuration
train_config = {
    "data": YAML_PATH,
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 4,
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "lrf": 0.01,
    "warmup_epochs": 3,
    "weight_decay": 0.0001,
    "patience": 15,
    "save": True,
    "save_period": 10,
    "val": True,
    "project": "/kaggle/working/runs/detect",
    "name": "rtdetr_l_baby_adult_v6",
    "exist_ok": True,
    "hsv_h": 0.015,
    "hsv_s": 0.6,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.3,
    "fliplr": 0.5,
    "mosaic": 0.5
}

print("=" * 60)
print("⚙️ TRAINING CONFIGURATION HYPERPARAMETERS")
print("=" * 60)
for k, v in train_config.items():
    print(f"  {k:<16}: {v}")
"""))

    # =========================================================================
    # Section 8: Model training
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 🚀 8. Model Training Execution"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 8: Model training
import time

print("=" * 60)
print("🚀 STARTING RT-DETR-L 50-EPOCH TRAINING ON TESLA T4")
print("=" * 60)

start_time = time.time()
results = model.train(**train_config)
elapsed_time = time.time() - start_time

print("=" * 60)
print(f"🎉 TRAINING FINISHED IN {elapsed_time/60:.2f} MINUTES ({elapsed_time/3600:.2f} HOURS)")
print("=" * 60)
"""))

    # =========================================================================
    # Section 9: Training history and metric visualization
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 📈 9. Training History and Metric Visualization"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 9: Training history and metric visualization
results_csv = "/kaggle/working/runs/detect/rtdetr_l_baby_adult_v6/results.csv"

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Train & Val Loss
    ax = axes[0, 0]
    if "train/loss" in df.columns and "val/loss" in df.columns:
        ax.plot(df["epoch"], df["train/loss"], label="Train Loss", color="blue")
        ax.plot(df["epoch"], df["val/loss"], label="Val Loss", color="red")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True)
        
    # mAP50
    ax = axes[0, 1]
    if "metrics/mAP50(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], color="green", lw=2)
        ax.set_title("mAP @ 0.50 Curve")
        ax.grid(True)
        
    # mAP50-95
    ax = axes[1, 0]
    if "metrics/mAP50-95(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], color="purple", lw=2)
        ax.set_title("mAP @ 0.50:0.95 Curve")
        ax.grid(True)
        
    # Precision & Recall
    ax = axes[1, 1]
    if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="teal")
        ax.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="orange")
        ax.set_title("Precision & Recall Curves")
        ax.legend()
        ax.grid(True)
        
    plt.suptitle("RT-DETR-L Version 6 Training Performance History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = "/kaggle/working/training_performance_curves.png"
    plt.savefig(chart_path, dpi=150)
    plt.show()
    print(f"✅ Saved performance chart: {chart_path}")
else:
    print(f"⚠️ Warning: results.csv not found at {results_csv}")
"""))

    # =========================================================================
    # Section 10: Model evaluation on the test dataset
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 📊 10. Model Evaluation on Test Dataset"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 10: Model evaluation on test dataset
best_weights = "/kaggle/working/runs/detect/rtdetr_l_baby_adult_v6/weights/best.pt"
assert os.path.exists(best_weights), f"best.pt not found at {best_weights}"

eval_model = RTDETR(best_weights)
print("Evaluating on Unseen Test Dataset (split='test')...")
test_metrics = eval_model.val(data=YAML_PATH, split="test", imgsz=640, batch=16, conf=0.25)

print("=" * 60)
print("📊 TEST DATASET FINAL BENCHMARK METRICS")
print("=" * 60)
print(f"Overall mAP50    : {test_metrics.box.map50:.4f}")
print(f"Overall mAP50-95 : {test_metrics.box.map:.4f}")
print(f"Overall Precision: {test_metrics.box.mp:.4f}")
print(f"Overall Recall   : {test_metrics.box.mr:.4f}")

for i, cname in enumerate(["adult", "baby_child"]):
    print(f"  Class [{cname}]: mAP50={test_metrics.box.maps[i]:.4f}")
"""))

    # =========================================================================
    # Section 11: Prediction visualization and result comparison
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 🔍 11. Prediction Visualization on Test Samples"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 11: Prediction visualization on test samples
test_imgs = sorted(glob.glob(os.path.join(DATASET_ROOT, "test", "images", "*.jpg")))
sample_test_imgs = test_imgs[::max(1, len(test_imgs) // 6)][:6]

pred_results = eval_model.predict(source=sample_test_imgs, conf=0.35, imgsz=640, device=0)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, r in zip(axes, pred_results):
    res_bgr = r.plot()
    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(res_rgb)
    ax.set_title(f"{os.path.basename(r.path)[:25]}...", fontsize=10)
    ax.axis("off")

plt.suptitle("RT-DETR-L Predictions on Unseen Test Samples (Conf >= 0.35)", fontsize=14, fontweight="bold")
plt.tight_layout()
test_pred_path = "/kaggle/working/test_predictions_grid.jpg"
plt.savefig(test_pred_path, dpi=150)
plt.show()
print(f"✅ Saved prediction visualization: {test_pred_path}")
"""))

    # =========================================================================
    # Section 12: Model checkpoint and artifact saving
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 💾 12. Model Checkpoint and Artifact Saving"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 12: Model checkpoint and artifact saving
final_weights_dst = "/kaggle/working/rtdetr_l_v6_best.pt"
shutil.copy2(best_weights, final_weights_dst)
print(f"✅ Exported top model weight to: {final_weights_dst}")

meta_summary = {
    "model_name": "RT-DETR-L (Version 6)",
    "weights_path": final_weights_dst,
    "input_resolution": 640,
    "classes": ["adult", "baby_child"],
    "mAP50": float(test_metrics.box.map50),
    "mAP50_95": float(test_metrics.box.map),
    "precision": float(test_metrics.box.mp),
    "recall": float(test_metrics.box.mr),
    "epochs_trained": 50,
    "training_time_minutes": float(elapsed_time / 60)
}

meta_json_path = "/kaggle/working/model_v6_metadata.json"
with open(meta_json_path, "w") as f:
    json.dump(meta_summary, f, indent=2)

print(f"✅ Deployment metadata saved to: {meta_json_path}")
"""))

    # =========================================================================
    # Section 13: Final experiment summary
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### 📋 13. Final Experiment Summary"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 13: Final experiment summary
print("=" * 60)
print("🎉 RT-DETR-L VERSION 6 EXPERIMENT SUMMARY REPORT")
print("=" * 60)
print(f"Hardware        : NVIDIA Tesla T4 ({vram_gb:.2f} GB VRAM)")
print(f"Architecture    : RT-DETR-L (Real-Time Detection Transformer)")
print(f"Dataset Size    : {img_counts['train']} train | {img_counts['valid']} val | {img_counts['test']} test")
print(f"Training Time   : {elapsed_time/60:.2f} minutes")
print(f"Test mAP@0.50   : {test_metrics.box.map50:.4f}")
print(f"Test mAP@0.50:95: {test_metrics.box.map:.4f}")
print(f"Top Weight File : {final_weights_dst}")
print("=" * 60)
print("✅ WORKFLOW COMPLETED IN STRICT COMPLIANCE WITH KAGGLE POLICIES.")
"""))

    nb["cells"] = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    }

    out_file = "kaggle_kernel/train_rtdetr_baby_vision.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✅ Created 13-section notebook: {out_file}")

if __name__ == "__main__":
    build_notebook()
