"""
Script: package_and_create_v9_kaggle.py
Purpose: Package vision_v9_domain_balanced into Kaggle upload zip and generate
         the 13-cell Kaggle training notebook for YOLO11-M V9 Domain-Balanced.
"""

import os
import json
import time
import shutil
import zipfile

SRC_DIR = "data/processed/vision_v9_domain_balanced"
UP_DIR = "data/kaggle_upload/vision-v9-domain-balanced"
NB_DIR = "kaggle_kernel_v9"
DATASET_SLUG = "thittn/vision-v9-domain-balanced"
KERNEL_SLUG = "thittn/train-yolo11-m-v9-domain-balanced"


def package_dataset():
    os.makedirs(UP_DIR, exist_ok=True)
    shutil.copy2(os.path.join(SRC_DIR, "data.yaml"), os.path.join(UP_DIR, "data.yaml"))
    meta = {
        "title": "Vision V9 Domain-Balanced Dataset",
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


def create_notebook():
    os.makedirs(NB_DIR, exist_ok=True)
    # Reuse the verified 13-cell structure from V7 and update for V9
    base_nb_path = "inspect_yolo_notebook/train-yolo11-m-full-body-v7.ipynb"
    with open(base_nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 3 (index 5): Direct verified V9 dataset path + copy to /tmp/dataset_v9
    nb["cells"][5]["source"] = """import os, shutil, json, yaml, time

VERIFIED_DATASET_SRC = '/kaggle/input/datasets/thittn/vision-v9-domain-balanced/dataset'
if not os.path.exists(VERIFIED_DATASET_SRC):
    # Standard Kaggle mount path for newly created datasets
    VERIFIED_DATASET_SRC = '/kaggle/input/vision-v9-domain-balanced/dataset'
assert os.path.exists(VERIFIED_DATASET_SRC), f'ERROR: Verified V9 dataset path not found: {VERIFIED_DATASET_SRC}'

data_root = '/tmp/dataset_v9'
if os.path.exists(data_root):
    shutil.rmtree(data_root)
t0 = time.time()
print(f'Copying verified V9 dataset from {VERIFIED_DATASET_SRC} to local SSD {data_root}...')
shutil.copytree(VERIFIED_DATASET_SRC, data_root)
print(f'Copy completed in {time.time() - t0:.1f}s!')

train_img_dir = os.path.join(data_root, 'train', 'images')
val_img_dir = os.path.join(data_root, 'valid', 'images')
test_img_dir = os.path.join(data_root, 'test', 'images')

assert os.path.exists(train_img_dir), f'Missing train_img_dir: {train_img_dir}'
assert os.path.exists(val_img_dir), f'Missing val_img_dir: {val_img_dir}'
assert os.path.exists(test_img_dir), f'Missing test_img_dir: {test_img_dir}'

yaml_config = {
    'path': data_root,
    'train': train_img_dir,
    'val': val_img_dir,
    'test': test_img_dir,
    'nc': 2,
    'names': {0: 'adult', 1: 'baby_child'}
}

data_yaml_path = '/kaggle/working/data_v9.yaml'
with open(data_yaml_path, 'w') as f:
    yaml.dump(yaml_config, f, default_flow_style=False)

print(f'PASS: Verified V9 dataset ready at {data_root}')
print(json.dumps(yaml_config, indent=2))
"""

    # Cell 5 (index 9): Multi-source sample inspection (oi, tb, dc)
    nb["cells"][9]["source"] = """import os, cv2, glob
import matplotlib.pyplot as plt

colors = {0: (0, 200, 0), 1: (255, 128, 0)}
names = {0: 'adult', 1: 'baby_child'}

train_dir = os.path.join(data_root, 'train', 'images')
train_imgs = []
for pat in ['oi_*.jpg', 'oi_*.jpg', 'tb_*.jpg', 'dc_*.jpg']:
    m = sorted(glob.glob(os.path.join(train_dir, pat)))
    if m:
        pick = m[len(m)//4] if pat != 'oi_*.jpg' or len(train_imgs) == 0 else m[len(m)//2]
        train_imgs.append(pick)
train_imgs = train_imgs[:4]

fig, axes = plt.subplots(1, len(train_imgs), figsize=(20, 5))
for idx, img_p in enumerate(train_imgs):
    img = cv2.imread(img_p)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]
    lbl_p = img_p.replace('images', 'labels').replace('.jpg', '.txt')
    if os.path.exists(lbl_p):
        with open(lbl_p) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0])
                    xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    x1, y1 = int((xc - w/2) * w_img), int((yc - h/2) * h_img)
                    x2, y2 = int((xc + w/2) * w_img), int((yc + h/2) * h_img)
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), colors.get(c, (255,0,0)), 3)
                    cv2.putText(img_rgb, names.get(c, str(c)), (x1+5, max(20, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(c, (255,0,0)), 2)
    axes[idx].imshow(img_rgb)
    axes[idx].set_title(os.path.basename(img_p)[:24], fontsize=11)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/sample_inspection.png', dpi=150)
plt.show()
print('PASS: Multi-source V9 sample visualization generated.')
"""

    # Cell 7 (index 13): V9 Training Config (erasing=0.20 to preserve facial/head age cues)
    nb["cells"][13]["source"] = """import json

train_config = dict(
    data=data_yaml_path,
    epochs=60,
    batch=16,
    imgsz=640,
    device=0,
    project='/kaggle/working/runs',
    name='yolo11m_domain_balanced_v9',
    exist_ok=True,
    erasing=0.20,         # Moderate occlusion without erasing faces/heads
    mosaic=1.0,
    mixup=0.15,
    scale=0.5,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    close_mosaic=10,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    patience=15,
    save=True,
    plots=True,
    verbose=True
)

print('V9 Training Configuration Summary:')
print(json.dumps({k: str(v) for k, v in train_config.items()}, indent=2))
print('PASS: Training configuration locked.')
"""

    # Cell 9 (index 17): update fallback save_dir name
    nb["cells"][17]["source"] = nb["cells"][17]["source"].replace(
        "yolo11m_fullbody_v7", "yolo11m_domain_balanced_v9"
    )

    # Cell 11 (index 21): Stratified test samples (oi, tb, dc) + agnostic_nms=True
    nb["cells"][21]["source"] = nb["cells"][21]["source"].replace(
        "['tb_*.jpg', 'kpz_Tongdosa_*.jpg', 'kpz_daycarecenter_*.jpg', 'coco_*.jpg']",
        "['oi_*.jpg', 'tb_*.jpg', 'dc_*.jpg']"
    )

    # Cell 13 (index 25): Update report title
    nb["cells"][25]["source"] = nb["cells"][25]["source"].replace(
        "Single Unified Dataset V7 (Balanced Full-Body & Occlusion-Aware)",
        "Domain-Balanced Multi-Age Dataset V9 (OpenImages V7 + True-Baby + Daycare)"
    ).replace(
        "Random Erasing (0.4), Mosaic (1.0), Mixup (0.2)",
        "Random Erasing (0.20), Mosaic (1.0), Mixup (0.15)"
    )

    nb_out = os.path.join(NB_DIR, "train-yolo11-m-v9-domain-balanced.ipynb")
    with open(nb_out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    kmeta = {
        "id": KERNEL_SLUG,
        "title": "Train YOLO11-M V9 Domain-Balanced",
        "code_file": "train-yolo11-m-v9-domain-balanced.ipynb",
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
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(NB_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(kmeta, f, indent=2)
    print(f"Created V9 Kaggle notebook ({len(nb['cells'])} cells) & metadata in {NB_DIR}/")


if __name__ == "__main__":
    package_dataset()
    create_notebook()
