"""
Script: package_and_create_3clean_kaggle.py
Purpose:
  1. Re-split vision_3clean_kids_adults with greedy group bin-packing to hit exact 70/15/15
     per source while preserving 100% Zero Data Leakage.
  2. Package into data/kaggle_upload/vision-3clean-kids-adults/dataset.zip.
  3. Generate a clean 13-cell Kaggle training notebook with ZERO icons/emojis anywhere.
"""

import os
import re
import glob
import json
import time
import shutil
import zipfile
import collections

SRC_DIR = "data/processed/vision_3clean_kids_adults"
UP_DIR = "data/kaggle_upload/vision-3clean-kids-adults"
NB_DIR = "kaggle_kernel_3clean"
DATASET_SLUG = "thittn/vision-3clean-kids-adults"
KERNEL_SLUG = "thittn/train-yolo11-m-3clean-kids-adults"


def resplit_exact_70_15_15():
    """Re-balance existing vision_3clean_kids_adults into exact ~70/15/15 per source via group bin-packing."""
    tmp_dir = SRC_DIR + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    for sp in ("train", "valid", "test"):
        os.makedirs(os.path.join(tmp_dir, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, sp, "labels"), exist_ok=True)

    # Gather all files grouped by (source, base_group_key)
    by_src_group = collections.defaultdict(lambda: collections.defaultdict(list))
    for sp in ("train", "valid", "test"):
        for ip in sorted(glob.glob(os.path.join(SRC_DIR, sp, "images", "*.*"))):
            bn = os.path.basename(ip)
            stem = os.path.splitext(bn)[0]
            lp = os.path.join(SRC_DIR, sp, "labels", stem + ".txt")
            if not os.path.exists(lp):
                continue
            parts = stem.split("_", 2)
            src = parts[0]
            raw_stem = parts[2] if len(parts) >= 3 else stem
            gbase = raw_stem.split("_jpg.rf.")[0].split(".rf.")[0]
            gbase = re.sub(r"[_-]\d+$", "", gbase)
            by_src_group[src][gbase].append((ip, lp, bn, stem + ".txt"))

    counts = collections.defaultdict(int)
    for src, gdict in by_src_group.items():
        groups = sorted(gdict.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        total_src = sum(len(v) for _, v in groups)
        targets = {"train": total_src * 0.70, "valid": total_src * 0.15, "test": total_src * 0.15}
        cur = {"train": 0, "valid": 0, "test": 0}
        for gkey, flist in groups:
            # Assign entire group to the split with largest remaining deficit ratio
            best_sp = max(targets.keys(), key=lambda s: (targets[s] - cur[s]) / max(targets[s], 1.0))
            for ip, lp, im_bn, lb_bn in flist:
                shutil.copy2(ip, os.path.join(tmp_dir, best_sp, "images", im_bn))
                shutil.copy2(lp, os.path.join(tmp_dir, best_sp, "labels", lb_bn))
            cur[best_sp] += len(flist)
            counts[best_sp] += len(flist)

    shutil.copy2(os.path.join(SRC_DIR, "data.yaml"), os.path.join(tmp_dir, "data.yaml"))
    shutil.rmtree(SRC_DIR)
    shutil.move(tmp_dir, SRC_DIR)

    tot = sum(counts.values())
    print(f"Exact Group Split Complete (Total={tot}): "
          f"Train={counts['train']} ({counts['train']/tot*100:.1f}%) | "
          f"Valid={counts['valid']} ({counts['valid']/tot*100:.1f}%) | "
          f"Test={counts['test']} ({counts['test']/tot*100:.1f}%)")


def package_dataset():
    os.makedirs(UP_DIR, exist_ok=True)
    shutil.copy2(os.path.join(SRC_DIR, "data.yaml"), os.path.join(UP_DIR, "data.yaml"))
    meta = {
        "title": "Vision 3-Clean Kids and Adults Dataset",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(UP_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    zip_path = os.path.join(UP_DIR, "dataset.zip")
    print(f"Packaging {SRC_DIR} -> {zip_path} ...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for root, _, files in os.walk(SRC_DIR):
            for fn in files:
                if fn.endswith(".json"):
                    continue
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.relpath(fp, SRC_DIR))
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"DONE packaging in {time.time() - t0:.1f}s! Zip size: {size_mb:.2f} MB")


def md_cell(src: str):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code_cell(src: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def create_clean_no_icon_notebook():
    os.makedirs(NB_DIR, exist_ok=True)
    cells = [
        # CELL 1
        md_cell("# CELL 1: ENVIRONMENT AND GPU VERIFICATION\nVerify NVIDIA GPU hardware, CUDA availability, VRAM, RAM, and PyTorch runtime environment."),
        code_cell("""import os, sys, json, torch, psutil

print('=' * 65)
print('STEP 1: HARDWARE ACCELERATOR AND ENVIRONMENT VALIDATION')
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

assert cuda_available, 'ERROR: GPU acceleration is required.'
print('PASS: GPU hardware verification successful.')
"""),

        # CELL 2
        md_cell("# CELL 2: DEPENDENCY INSTALLATION AND IMPORTS\nInstall Ultralytics YOLO11 and import image processing and visualization libraries."),
        code_cell("""import os, sys, subprocess, json, time, glob, shutil, hashlib, collections

print('Installing ultralytics and dependencies...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'matplotlib', 'seaborn', 'pandas', 'pyyaml'], check=True)

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

print('PASS: All libraries successfully imported.')
"""),

        # CELL 3
        md_cell("# CELL 3: DATASET CONNECTION AND LOCAL SSD STAGING\nConnect directly to the verified Kaggle dataset `thittn/vision-3clean-kids-adults` and copy to `/tmp/dataset_3clean` for fast local I/O."),
        code_cell("""import os, shutil, json, yaml, time

CANDIDATE_PATHS = [
    '/kaggle/input/datasets/thittn/vision-3clean-kids-adults/dataset',
    '/kaggle/input/vision-3clean-kids-adults/dataset',
]
VERIFIED_DATASET_SRC = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
assert VERIFIED_DATASET_SRC is not None, f'ERROR: Verified dataset not found in {CANDIDATE_PATHS}'

data_root = '/tmp/dataset_3clean'
if os.path.exists(data_root):
    shutil.rmtree(data_root)

t0 = time.time()
print(f'Copying dataset from {VERIFIED_DATASET_SRC} to {data_root}...')
shutil.copytree(VERIFIED_DATASET_SRC, data_root)
print(f'Copy completed in {time.time() - t0:.1f}s.')

train_img_dir = os.path.join(data_root, 'train', 'images')
val_img_dir = os.path.join(data_root, 'valid', 'images')
test_img_dir = os.path.join(data_root, 'test', 'images')

assert os.path.exists(train_img_dir), f'Missing train directory: {train_img_dir}'
assert os.path.exists(val_img_dir), f'Missing valid directory: {val_img_dir}'
assert os.path.exists(test_img_dir), f'Missing test directory: {test_img_dir}'

yaml_config = {
    'path': data_root,
    'train': train_img_dir,
    'val': val_img_dir,
    'test': test_img_dir,
    'nc': 2,
    'names': {0: 'adults', 1: 'kids'},
}
data_yaml_path = '/kaggle/working/data_3clean.yaml'
with open(data_yaml_path, 'w') as f:
    yaml.dump(yaml_config, f, default_flow_style=False)

print('PASS: Dataset configuration written to', data_yaml_path)
print(json.dumps(yaml_config, indent=2))
"""),

        # CELL 4
        md_cell("# CELL 4: DATASET STRUCTURE AND ANNOTATION VALIDATION\nAudit split ratios (70/15/15), class counts (`0: adults`, `1: kids`), per-source distribution, coordinate validity, and zero data leakage."),
        code_cell("""import os, glob, collections

counts = {}
class_counts = {0: 0, 1: 0}
invalid_boxes = 0
split_fnames = {}

for sp in ('train', 'valid', 'test'):
    img_dir = os.path.join(data_root, sp, 'images')
    lbl_dir = os.path.join(data_root, sp, 'labels')
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.*')))
    lbls = sorted(glob.glob(os.path.join(lbl_dir, '*.txt')))
    split_fnames[sp] = {os.path.basename(p) for p in imgs}
    src_cnt = collections.Counter(os.path.basename(p).split('_')[0] for p in imgs)
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
    counts[sp] = {'images': len(imgs), 'labels': len(lbls), 'c0': c_sp[0], 'c1': c_sp[1], 'sources': dict(src_cnt)}

total_imgs = sum(v['images'] for v in counts.values())
ov_tv = len(split_fnames['train'] & split_fnames['valid'])
ov_tt = len(split_fnames['train'] & split_fnames['test'])
ov_vt = len(split_fnames['valid'] & split_fnames['test'])

print('=' * 65)
print('DATASET SUMMARY: 3-SOURCE CLEAN (KAUSHI + ANIRUDDHA + TRUE-BABY)')
print('=' * 65)
for sp, v in counts.items():
    pct = v['images'] / max(total_imgs, 1) * 100
    print(f"{sp.upper():<6}: {v['images']:4d} imgs ({pct:5.1f}%) | [0] adults: {v['c0']:4d} | [1] kids: {v['c1']:4d} | sources={v['sources']}")
print(f"Total Images: {total_imgs} | [0] adults: {class_counts[0]} | [1] kids: {class_counts[1]}")
print(f"Invalid boxes: {invalid_boxes} | Overlaps T&V={ov_tv}, T&Test={ov_tt}, V&Test={ov_vt}")
assert total_imgs > 0 and invalid_boxes == 0 and ov_tv == 0 and ov_tt == 0 and ov_vt == 0
print('PASS: Dataset integrity and zero leakage verified.')
"""),

        # CELL 5
        md_cell("# CELL 5: GROUND-TRUTH SAMPLE VISUALIZATION\nInspect representative training samples from all three sources with ground-truth bounding boxes (`0: adults` green, `1: kids` orange)."),
        code_cell("""import os, cv2, glob
import matplotlib.pyplot as plt

colors = {0: (0, 200, 0), 1: (255, 128, 0)}
names = {0: 'adults', 1: 'kids'}

train_dir = os.path.join(data_root, 'train', 'images')
sample_imgs = []
for prefix in ('kaushi_*', 'anir_*', 'anir_*', 'tb_*'):
    cands = sorted(glob.glob(os.path.join(train_dir, prefix)))
    if cands:
        pick = cands[len(cands)//3] if (prefix != 'anir_*' or len(sample_imgs) == 1) else cands[(2*len(cands))//3]
        sample_imgs.append(pick)
sample_imgs = sample_imgs[:4]

fig, axes = plt.subplots(1, len(sample_imgs), figsize=(20, 5))
for idx, img_p in enumerate(sample_imgs):
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
    axes[idx].set_title(os.path.basename(img_p)[:26], fontsize=10)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/sample_inspection.png', dpi=150)
plt.show()
print('PASS: Ground-truth sample visualization generated.')
"""),

        # CELL 6
        md_cell("# CELL 6: MODEL CONFIGURATION AND INITIALIZATION\nLoad pretrained YOLO11-M weights (`yolo11m.pt`, 20.1M parameters)."),
        code_cell("""from ultralytics import YOLO

model = YOLO('yolo11m.pt')
print(f'Model Architecture: YOLO11-M ({type(model.model).__name__})')
print('PASS: Base YOLO11-M weights loaded.')
"""),

        # CELL 7
        md_cell("# CELL 7: TRAINING HYPERPARAMETERS\nConfigure 60 epochs at `768x768` resolution with `mosaic=0.0` and `mixup=0.0` to preserve sharp facial/body cues and high prediction confidence."),
        code_cell("""import json

train_config = dict(
    data=data_yaml_path,
    epochs=60,
    batch=12,
    imgsz=768,
    device=0,
    project='/kaggle/working/runs',
    name='yolo11m_3clean_kids_adults',
    exist_ok=True,
    mosaic=0.0,
    mixup=0.0,
    erasing=0.10,
    scale=0.45,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.35,
    fliplr=0.5,
    optimizer='AdamW',
    lr0=0.0008,
    lrf=0.01,
    cos_lr=True,
    patience=15,
    save=True,
    plots=True,
    verbose=True,
)

print('Training Configuration Summary:')
print(json.dumps({k: str(v) for k, v in train_config.items()}, indent=2))
print('PASS: Training configuration locked.')
"""),

        # CELL 8
        md_cell("# CELL 8: MODEL TRAINING EXECUTION\nRun YOLO11-M training on NVIDIA Tesla T4 GPU."),
        code_cell("""import time

start_time = time.time()
print('=' * 65)
print(f'TRAINING STARTED AT: {time.strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 65)

results = model.train(**train_config)

total_train_sec = time.time() - start_time
print('=' * 65)
print(f'TRAINING FINISHED IN {total_train_sec / 60:.2f} MINUTES.')
print('=' * 65)
"""),

        # CELL 9
        md_cell("# CELL 9: TRAINING HISTORY AND METRIC CURVES\nPlot Box Loss, Classification Loss, Precision/Recall, and mAP curves across training epochs."),
        code_cell("""import os
import pandas as pd
import matplotlib.pyplot as plt

save_dir = str(model.trainer.save_dir) if hasattr(model, 'trainer') else '/kaggle/working/runs/yolo11m_3clean_kids_adults'
csv_path = os.path.join(save_dir, 'results.csv')

assert os.path.exists(csv_path), f'ERROR: results.csv not found at {csv_path}'
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0, 0].plot(df['epoch'], df['train/box_loss'], 'b-', label='Train Box Loss', lw=2)
axes[0, 0].plot(df['epoch'], df['val/box_loss'], 'r--', label='Val Box Loss', lw=2)
axes[0, 0].set_title('Bounding Box Loss (CIoU)'); axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train Cls Loss', lw=2)
axes[0, 1].plot(df['epoch'], df['val/cls_loss'], 'r--', label='Val Cls Loss', lw=2)
axes[0, 1].set_title('Classification Loss (adults vs kids)'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], 'g-', label='Precision', lw=2)
axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], 'm--', label='Recall', lw=2)
axes[1, 0].set_title('Precision and Recall'); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Score')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], 'teal', label='mAP@50', lw=2.5)
axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], 'orange', label='mAP@50-95', lw=2)
axes[1, 1].set_title('Mean Average Precision (mAP)'); axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('mAP')
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

chart_path = '/kaggle/working/training_metrics_curves.png'
plt.tight_layout()
plt.savefig(chart_path, dpi=200)
plt.show()
print('PASS: Training curves saved to', chart_path)
"""),

        # CELL 10
        md_cell("# CELL 10: INDEPENDENT TEST SET EVALUATION\nEvaluate the best trained checkpoint (`best.pt`) on the unseen test split."),
        code_cell("""import os
from ultralytics import YOLO

best_weight_p = os.path.join(save_dir, 'weights', 'best.pt')
assert os.path.exists(best_weight_p), f'ERROR: Checkpoint not found at {best_weight_p}'

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
        md_cell("# CELL 11: TEST SET INFERENCE VISUALIZATION (GROUND TRUTH VS PREDICTION)\nRun inference with `best.pt` (`agnostic_nms=True`) on representative samples from all three sources in the test split and display side-by-side comparisons."),
        code_cell("""import os, cv2, glob
import matplotlib.pyplot as plt

test_dir = os.path.join(data_root, 'test', 'images')
picked_test = []
for prefix in ('kaushi_*', 'anir_*', 'anir_*', 'tb_*'):
    cands = sorted(glob.glob(os.path.join(test_dir, prefix)))
    if cands:
        pick = cands[len(cands)//3] if (prefix != 'anir_*' or len(picked_test) == 1) else cands[(2*len(cands))//3]
        picked_test.append(pick)
picked_test = picked_test[:4]

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
    axes[idx, 0].set_title(f'Sample {idx+1} ({os.path.basename(img_p)[:22]}) - GROUND TRUTH', fontsize=11, fontweight='bold', color='green')
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
print('PASS: Test prediction comparison saved to', pred_comp_path)
"""),

        # CELL 12
        md_cell("# CELL 12: MODEL CHECKPOINT AND ARTIFACT SAVING\nCopy `best.pt` to `/kaggle/working/best.pt` and record file size and MD5 checksum."),
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
print('PASS: Checkpoint verified.')
"""),

        # CELL 13
        md_cell("# CELL 13: FINAL EXPERIMENT SUMMARY\nWrite structured training report to `/kaggle/working/experiment_summary.md`."),
        code_cell("""summary_md = f\"\"\"# FINAL TRAINING EXPERIMENT REPORT (3-Source Clean Dataset)
## 1. Environment and Hardware
- Primary GPU: {device_name} ({vram_gb} GB VRAM)
- Framework: Ultralytics YOLO11 (PyTorch {torch.__version__})

## 2. Dataset Configuration
- Dataset: thittn/vision-3clean-kids-adults (kaushigihanml + aniruddha-jmp5a + true-baby)
- Total Images: {total_imgs} (Train: {counts['train']['images']}, Valid: {counts['valid']['images']}, Test: {counts['test']['images']})
- Target Classes: [0] adults ({class_counts[0]} boxes), [1] kids ({class_counts[1]} boxes)

## 3. Training and Hyperparameters
- Architecture: YOLO11-M (20.1M parameters)
- Epochs: 60 (patience=15) | Batch: 12 | Image Size: 768x768
- Augmentations: mosaic=0.0, mixup=0.0, erasing=0.10
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
    nb_path = os.path.join(NB_DIR, "train-yolo11-m-3clean-kids-adults.ipynb")
    raw_json = json.dumps(nb, ensure_ascii=True, indent=2)
    # Verify strictly ASCII (0 emojis / 0 icons anywhere in notebook!)
    assert all(ord(ch) < 128 for ch in raw_json), "Non-ASCII / icon character detected in notebook!"
    with open(nb_path, "w", encoding="utf-8") as f:
        f.write(raw_json)

    kmeta = {
        "id": KERNEL_SLUG,
        "title": "Train YOLO11-M 3-Clean Kids Adults",
        "code_file": "train-yolo11-m-3clean-kids-adults.ipynb",
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
    print(f"Created ZERO-ICON 13-cell notebook ({len(cells)} cells) at {nb_path}")


if __name__ == "__main__":
    resplit_exact_70_15_15()
    package_dataset()
    create_clean_no_icon_notebook()
