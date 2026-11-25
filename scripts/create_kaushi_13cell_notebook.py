"""
Script: create_kaushi_13cell_notebook.py
Purpose: Generate a clean 13-cell Kaggle training notebook complying strictly with
         KAGGLE_TRAINING_WORKFLOW_POLICY.md, training YOLO11-M exclusively on
         kaushigihanml/kids-and-adults-detection (no dataset merging, mosaic=0.0, mixup=0.0).
"""

import os
import json

NB_DIR = "kaggle_kernel_kaushi"
KERNEL_SLUG = "thittn/train-yolo11-m-kids-adults-clean"
DATASET_SLUG = "kaushigihanml/kids-and-adults-detection"


def md_cell(src: str):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code_cell(src: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    cells = [
        # CELL 1
        md_cell("# 🔬 CELL 1: ENVIRONMENT & GPU VERIFICATION\nXác thực phần cứng GPU Tesla T4, VRAM, RAM và môi trường PyTorch theo Mục 2 `KAGGLE_TRAINING_WORKFLOW_POLICY.md`."),
        code_cell("""import os, sys, json, torch, psutil

print('=' * 65)
print('STEP 1: HARDWARE ACCELERATOR & ENVIRONMENT VALIDATION')
print('=' * 65)

cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0
device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU Mode'
vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_available else 0
ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

print(f'CUDA Acceleration Available : {cuda_available}')
print(f'GPU Devices Found           : {device_count}')
print(f'Primary GPU Model           : {device_name}')
print(f'Dedicated VRAM              : {vram_gb} GB')
print(f'System RAM                  : {ram_gb} GB')
print(f'CPU Logical Cores           : {psutil.cpu_count(logical=True)}')
print(f'PyTorch Version             : {torch.__version__}')

assert cuda_available, 'ERROR: GPU acceleration is required!'
print('PASS: GPU hardware verification successful!')
"""),

        # CELL 2
        md_cell("# 📦 CELL 2: DEPENDENCY INSTALLATION & IMPORTS\nCài đặt `ultralytics` và nạp các thư viện cần thiết."),
        code_cell("""import os, sys, subprocess, json, time, glob, shutil, hashlib, random

print('Installing ultralytics & dependencies...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'matplotlib', 'seaborn', 'pandas', 'pyyaml'], check=True)

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

print('PASS: Libraries successfully imported. Ultralytics ready.')
"""),

        # CELL 3
        md_cell("# 📂 CELL 3: DATASET CONNECTION & STANDARDIZATION (`kaushigihanml/kids-and-adults-detection`)\n- Kết nối trực tiếp dataset xác thực `kaushigihanml/kids-and-adults-detection` (không ghép thêm nguồn khác)\n- Đảo chuẩn nhãn về hệ thống: `1 (Adult) -> 0 (adult)`, `0 (Kid) -> 1 (baby_child)`\n- Chia độc lập Train / Valid / Test (Zero Data Leakage) vào `/tmp/kids_adults_std`"),
        code_cell("""import os, glob, shutil, yaml, json, random

# Exact verified Kaggle dataset mount paths
CANDIDATE_PATHS = [
    '/kaggle/input/datasets/kaushigihanml/kids-and-adults-detection/children_and_adults',
    '/kaggle/input/kids-and-adults-detection/children_and_adults',
]
SRC_ROOT = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
assert SRC_ROOT is not None, f'ERROR: Verified dataset not found in {CANDIDATE_PATHS}'
print(f'Verified Kaggle Dataset Source: {SRC_ROOT}')

data_root = '/tmp/kids_adults_std'
if os.path.exists(data_root):
    shutil.rmtree(data_root)
for sp in ('train', 'valid', 'test'):
    os.makedirs(os.path.join(data_root, sp, 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_root, sp, 'labels'), exist_ok=True)

# Collect all valid (image, label) pairs from train & val folders of source dataset
orig_train = sorted(glob.glob(os.path.join(SRC_ROOT, 'images', 'train', '*.*')))
orig_val = sorted(glob.glob(os.path.join(SRC_ROOT, 'images', 'val', '*.*')))

def pair_with_label(img_list, split_subdir):
    paired = []
    for ip in img_list:
        bn = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(SRC_ROOT, 'labels', split_subdir, bn + '.txt')
        if os.path.exists(lp):
            paired.append((ip, lp))
    return paired

train_pairs = pair_with_label(orig_train, 'train')
val_pairs = pair_with_label(orig_val, 'val')

# Deterministic Zero-Leakage split:
# Keep 365 from orig_train -> train; 50 from orig_train + 14 from orig_val -> valid (64); 64 from orig_val -> test
rng = random.Random(42)
rng.shuffle(train_pairs)
rng.shuffle(val_pairs)

splits_map = {
    'train': train_pairs[:365],
    'valid': train_pairs[365:] + val_pairs[:14],
    'test':  val_pairs[14:],
}

for sp, pairs in splits_map.items():
    for ip, lp in pairs:
        bn = os.path.basename(ip)
        stem = os.path.splitext(bn)[0]
        valid_lines = []
        with open(lp, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    raw_c = int(p[0])
                    xc, yc, w, h = map(float, p[1:5])
                    # Remap kaushigihanml (0=Kid, 1=Adult) -> Project standard (0=adult, 1=baby_child)
                    if raw_c == 1:
                        std_c = 0
                    elif raw_c == 0:
                        std_c = 1
                    else:
                        continue
                    valid_lines.append(f'{std_c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\\n')
        if not valid_lines:
            continue
        shutil.copy2(ip, os.path.join(data_root, sp, 'images', bn))
        with open(os.path.join(data_root, sp, 'labels', stem + '.txt'), 'w') as f:
            f.writelines(valid_lines)

yaml_config = {
    'path': data_root,
    'train': os.path.join(data_root, 'train', 'images'),
    'val': os.path.join(data_root, 'valid', 'images'),
    'test': os.path.join(data_root, 'test', 'images'),
    'nc': 2,
    'names': {0: 'adult', 1: 'baby_child'},
}
data_yaml_path = '/kaggle/working/data_kaushi.yaml'
with open(data_yaml_path, 'w') as f:
    yaml.dump(yaml_config, f, default_flow_style=False)

print(f'PASS: Standardized dataset ready at {data_root}')
print(json.dumps(yaml_config, indent=2))
"""),

        # CELL 4
        md_cell("# 📊 CELL 4: DATASET STRUCTURE & ANNOTATION VALIDATION\nKiểm định số ảnh, phân bố lớp `adult` vs `baby_child`, tọa độ hợp lệ và 0 rò rỉ dữ liệu."),
        code_cell("""import os, glob

counts = {}
class_counts = {0: 0, 1: 0}
invalid_boxes = 0
split_fnames = {}

for sp in ('train', 'valid', 'test'):
    imgs = sorted(glob.glob(os.path.join(data_root, sp, 'images', '*.*')))
    lbls = sorted(glob.glob(os.path.join(data_root, sp, 'labels', '*.txt')))
    split_fnames[sp] = {os.path.basename(p) for p in imgs}
    c_sp = {0: 0, 1: 0}
    for lf in lbls:
        with open(lf) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0])
                    xc, yc, w, h = map(float, p[1:5])
                    if c in (0, 1):
                        class_counts[c] += 1
                        c_sp[c] += 1
                    else:
                        invalid_boxes += 1
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        invalid_boxes += 1
    counts[sp] = {'images': len(imgs), 'labels': len(lbls), 'c0': c_sp[0], 'c1': c_sp[1]}

total_imgs = sum(v['images'] for v in counts.values())
ov_tv = len(split_fnames['train'] & split_fnames['valid'])
ov_tt = len(split_fnames['train'] & split_fnames['test'])
ov_vt = len(split_fnames['valid'] & split_fnames['test'])

print('=' * 65)
print('DATASET SUMMARY (kaushigihanml/kids-and-adults-detection ONLY)')
print('=' * 65)
for sp, v in counts.items():
    pct = v['images'] / max(total_imgs, 1) * 100
    print(f"{sp.upper():<6}: {v['images']:4d} imgs ({pct:5.1f}%) | [0] adult: {v['c0']:4d} | [1] baby_child: {v['c1']:4d}")
print(f"Total Images: {total_imgs} | [0] adult: {class_counts[0]} | [1] baby_child: {class_counts[1]}")
print(f"Invalid boxes: {invalid_boxes} | Overlaps T&V={ov_tv}, T&Test={ov_tt}, V&Test={ov_vt}")
assert total_imgs > 0 and invalid_boxes == 0 and ov_tv == 0 and ov_tt == 0 and ov_vt == 0
print('PASS: Dataset integrity & Zero Data Leakage verified!')
"""),

        # CELL 5
        md_cell("# 🖼️ CELL 5: DATA VISUALIZATION & SAMPLE INSPECTION\nHiển thị 4 ảnh mẫu thực tế từ tập Train kèm nhãn chuẩn sau khi chuẩn hóa (`0: adult` xanh lá, `1: baby_child` cam)."),
        code_cell("""import os, cv2, glob
import matplotlib.pyplot as plt

colors = {0: (0, 200, 0), 1: (255, 128, 0)}
names = {0: 'adult', 1: 'baby_child'}

train_imgs = sorted(glob.glob(os.path.join(data_root, 'train', 'images', '*.*')))
picked = train_imgs[::max(1, len(train_imgs)//4)][:4]
fig, axes = plt.subplots(1, len(picked), figsize=(20, 5))

for idx, img_p in enumerate(picked):
    img_rgb = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]
    stem = os.path.splitext(os.path.basename(img_p))[0]
    lbl_p = os.path.join(data_root, 'train', 'labels', stem + '.txt')
    if os.path.exists(lbl_p):
        with open(lbl_p) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0]); xc, yc, w, h = map(float, p[1:5])
                    x1, y1 = int((xc - w/2) * w_img), int((yc - h/2) * h_img)
                    x2, y2 = int((xc + w/2) * w_img), int((yc + h/2) * h_img)
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), colors[c], 3)
                    cv2.putText(img_rgb, names[c], (x1+4, max(20, y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[c], 2)
    axes[idx].imshow(img_rgb)
    axes[idx].set_title(os.path.basename(img_p), fontsize=11)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/sample_inspection.png', dpi=150)
plt.show()
print('PASS: Sample inspection grid saved.')
"""),

        # CELL 6
        md_cell("# 🤖 CELL 6: MODEL CONFIGURATION & INITIALIZATION (`YOLO11-M`)\nNạp trọng số tiền huấn luyện `yolo11m.pt` (20.1M parameters)."),
        code_cell("""from ultralytics import YOLO

model = YOLO('yolo11m.pt')
print(f'Model Architecture: YOLO11-M ({type(model.model).__name__})')
print('PASS: Base YOLO11-M weights loaded successfully!')
"""),

        # CELL 7
        md_cell("# ⚙️ CELL 7: TRAINING CONFIGURATION (NO IMAGE STITCHING: `mosaic=0.0`, `mixup=0.0`)\nThiết lập huấn luyện riêng trên bộ dữ liệu sạch ở độ phân giải cao `768×768`, không ghép ảnh (`mosaic=0.0`, `mixup=0.0`) để giữ nguyên độ sắc nét và đẩy cao confidence."),
        code_cell("""import json

train_config = dict(
    data=data_yaml_path,
    epochs=80,
    batch=12,
    imgsz=768,
    device=0,
    project='/kaggle/working/runs',
    name='yolo11m_kids_adults_clean',
    exist_ok=True,
    # Pure single-image training (NO mosaic/mixup stitching as requested)
    mosaic=0.0,
    mixup=0.0,
    erasing=0.10,
    scale=0.4,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.35,
    fliplr=0.5,
    optimizer='AdamW',
    lr0=0.0008,
    lrf=0.01,
    cos_lr=True,
    patience=20,
    save=True,
    plots=True,
    verbose=True,
)

print('Training Configuration Summary:')
print(json.dumps({k: str(v) for k, v in train_config.items()}, indent=2))
print('PASS: Training configuration locked.')
"""),

        # CELL 8
        md_cell("# 🚀 CELL 8: MODEL TRAINING (80 EPOCHS, `imgsz=768`)\nThực thi huấn luyện YOLO11-M trên GPU Tesla T4."),
        code_cell("""import time

start_time = time.time()
print('=' * 65)
print(f'TRAINING STARTED AT: {time.strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 65)

results = model.train(**train_config)

total_train_sec = time.time() - start_time
print('=' * 65)
print(f'TRAINING FINISHED in {total_train_sec / 60:.2f} minutes!')
print('=' * 65)
"""),

        # CELL 9
        md_cell("# 📈 CELL 9: TRAINING HISTORY & METRIC VISUALIZATION\nVẽ biểu đồ hội tụ Box Loss, Cls Loss, Precision/Recall và mAP qua các epoch."),
        code_cell("""import os
import pandas as pd
import matplotlib.pyplot as plt

save_dir = str(model.trainer.save_dir) if hasattr(model, 'trainer') else '/kaggle/working/runs/yolo11m_kids_adults_clean'
csv_path = os.path.join(save_dir, 'results.csv')

assert os.path.exists(csv_path), f'results.csv not found at {csv_path}'
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0, 0].plot(df['epoch'], df['train/box_loss'], 'b-', label='Train Box Loss', lw=2)
axes[0, 0].plot(df['epoch'], df['val/box_loss'], 'r--', label='Val Box Loss', lw=2)
axes[0, 0].set_title('Bounding Box Loss (CIoU)'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train Cls Loss', lw=2)
axes[0, 1].plot(df['epoch'], df['val/cls_loss'], 'r--', label='Val Cls Loss', lw=2)
axes[0, 1].set_title('Classification Loss (Adult vs Baby_Child)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], 'g-', label='Precision', lw=2)
axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], 'm--', label='Recall', lw=2)
axes[1, 0].set_title('Precision & Recall Curves'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], 'teal', label='mAP@50', lw=2.5)
axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], 'orange', label='mAP@50-95', lw=2)
axes[1, 1].set_title('Mean Average Precision (mAP)'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

chart_path = '/kaggle/working/training_metrics_curves.png'
plt.tight_layout()
plt.savefig(chart_path, dpi=200)
plt.show()
print(f'PASS: Training metrics plot saved to {chart_path}')
"""),

        # CELL 10
        md_cell("# 🎯 CELL 10: INDEPENDENT TEST DATASET EVALUATION\nĐánh giá checkpoint tốt nhất `best.pt` trên tập Test độc lập (`64` ảnh)."),
        code_cell("""import os
from ultralytics import YOLO

best_weight_p = os.path.join(save_dir, 'weights', 'best.pt')
assert os.path.exists(best_weight_p), f'Checkpoint not found at {best_weight_p}'

best_model = YOLO(best_weight_p)
print('Running evaluation on independent TEST split...')
test_results = best_model.val(data=data_yaml_path, split='test', imgsz=768)

print('=' * 65)
print('INDEPENDENT TEST SET EVALUATION METRICS')
print('=' * 65)
print(f"Test Precision (B) : {test_results.results_dict.get('metrics/precision(B)', 0):.4f}")
print(f"Test Recall (B)    : {test_results.results_dict.get('metrics/recall(B)', 0):.4f}")
print(f"Test mAP@50 (B)    : {test_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"Test mAP@50-95 (B) : {test_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
print('PASS: Independent test evaluation completed.')
"""),

        # CELL 11
        md_cell("# 🔍 CELL 11: PREDICTION VISUALIZATION & GT COMPARISON\nSo sánh song song Ground Truth (trái) và Dự đoán YOLO11-M (`agnostic_nms=True`, phải) trên 4 ảnh Test đại diện."),
        code_cell("""import os, cv2, glob
import matplotlib.pyplot as plt

test_imgs = sorted(glob.glob(os.path.join(data_root, 'test', 'images', '*.*')))
picked_test = test_imgs[::max(1, len(test_imgs)//4)][:4]
fig, axes = plt.subplots(len(picked_test), 2, figsize=(14, 18))

for idx, img_p in enumerate(picked_test):
    img_bgr = cv2.imread(img_p)
    h_img, w_img = img_bgr.shape[:2]
    gt_vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    stem = os.path.splitext(os.path.basename(img_p))[0]
    lbl_p = os.path.join(data_root, 'test', 'labels', stem + '.txt')
    if os.path.exists(lbl_p):
        with open(lbl_p) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0]); xc, yc, w, h = map(float, p[1:5])
                    x1, y1 = int((xc - w/2) * w_img), int((yc - h/2) * h_img)
                    x2, y2 = int((xc + w/2) * w_img), int((yc + h/2) * h_img)
                    color = (0, 200, 0) if c == 0 else (255, 128, 0)
                    cv2.rectangle(gt_vis, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(gt_vis, f'GT: {names[c]}', (x1+4, max(20, y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    axes[idx, 0].imshow(gt_vis)
    axes[idx, 0].set_title(f'Sample {idx+1} ({os.path.basename(img_p)}) - GT', fontsize=11, fontweight='bold', color='green')
    axes[idx, 0].axis('off')

    pred_res = best_model.predict(img_p, imgsz=768, conf=0.35, iou=0.45, agnostic_nms=True, verbose=False)[0]
    pred_vis = cv2.cvtColor(pred_res.plot(), cv2.COLOR_BGR2RGB)
    axes[idx, 1].imshow(pred_vis)
    axes[idx, 1].set_title(f'Sample {idx+1} - YOLO11-M PREDICTION', fontsize=11, fontweight='bold', color='blue')
    axes[idx, 1].axis('off')

pred_comp_path = '/kaggle/working/test_predictions_comparison.png'
plt.tight_layout()
plt.savefig(pred_comp_path, dpi=200)
plt.show()
print(f'PASS: Visual comparison saved to {pred_comp_path}')
"""),

        # CELL 12
        md_cell("# 💾 CELL 12: MODEL CHECKPOINT & ARTIFACT SAVING\nLưu trọng số tốt nhất ra `/kaggle/working/best.pt` và kiểm tra MD5."),
        code_cell("""import os, shutil, hashlib

final_weight_p = '/kaggle/working/best.pt'
shutil.copy2(best_weight_p, final_weight_p)
size_mb = os.path.getsize(final_weight_p) / (1024 * 1024)
with open(final_weight_p, 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()

print('=' * 65)
print('SAVED ARTIFACT DETAILS')
print('=' * 65)
print(f'Target File Path : {final_weight_p}')
print(f'File Size        : {size_mb:.2f} MB')
print(f'MD5 Checksum     : {md5}')
print('PASS: Checkpoint verified and ready.')
"""),

        # CELL 13
        md_cell("# 📋 CELL 13: FINAL EXPERIMENT SUMMARY\nTổng kết toàn bộ thực nghiệm theo `KAGGLE_TRAINING_WORKFLOW_POLICY.md`."),
        code_cell("""summary_md = f\"\"\"# FINAL TRAINING EXPERIMENT REPORT (Kids & Adults Clean)
## 1. Environment & Hardware
- Primary GPU: {device_name} ({vram_gb} GB VRAM)
- Framework: Ultralytics YOLO11 (PyTorch {torch.__version__})

## 2. Dataset Configuration
- Source: kaushigihanml/kids-and-adults-detection (Standalone, No Merging)
- Total Images: {total_imgs} (Train: {counts['train']['images']}, Valid: {counts['valid']['images']}, Test: {counts['test']['images']})
- Target Classes: [0] adult ({class_counts[0]} boxes), [1] baby_child ({class_counts[1]} boxes)

## 3. Training & Hyperparameters
- Architecture: YOLO11-M (20.1M parameters)
- Epochs: 80 (patience=20) | Batch: 12 | Image Size: 768x768
- Augmentations: mosaic=0.0, mixup=0.0, erasing=0.10 (No image stitching)
- Training Duration: {total_train_sec / 60:.2f} minutes

## 4. Final Evaluation (Test Split)
- Test Precision: {test_results.results_dict.get('metrics/precision(B)', 0):.4f}
- Test Recall   : {test_results.results_dict.get('metrics/recall(B)', 0):.4f}
- Test mAP@50   : {test_results.results_dict.get('metrics/mAP50(B)', 0):.4f}
- Test mAP@50-95: {test_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}

## 5. Output Artifacts
- Weights: /kaggle/working/best.pt ({size_mb:.2f} MB, MD5: {md5})
- Curves : /kaggle/working/training_metrics_curves.png
- Grid   : /kaggle/working/test_predictions_comparison.png
\"\"\"
print(summary_md)
with open('/kaggle/working/experiment_summary.md', 'w') as f:
    f.write(summary_md)
print('EXPERIMENT EXECUTION SUCCESSFULLY COMPLETED.')
"""),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    nb_path = os.path.join(NB_DIR, "train-yolo11-m-kids-adults-clean.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    kmeta = {
        "id": KERNEL_SLUG,
        "title": "Train YOLO11-M Kids Adults Clean",
        "code_file": "train-yolo11-m-kids-adults-clean.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [DATASET_SLUG],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01",
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(NB_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(kmeta, f, indent=2)

    print(f"Generated 13-cell notebook ({len(cells)} total cells) at {nb_path}")


if __name__ == "__main__":
    main()
