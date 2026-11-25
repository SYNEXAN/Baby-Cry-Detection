"""
Script: build_unified_v8_dataset.py
Purpose: Build the unified Version 8 dataset with Morphology & Invariant Feature Learning
         Components:
           1. Base V7 Dataset (True-Baby, Daycare CCTV kpz2, COCO Adults)
           2. 1,200 CCTV Crop-Zoomed images (large-scale children) mapped by parent scene
           3. 72 Classroom video frames (co-occurrence of teacher and children)
         Target Classes:
           0: adult (Full-body)
           1: baby_child (Full-body)
"""

import os
import glob
import shutil
from collections import defaultdict

V7_DIR = "data/processed/vision_v7_unified"
V8_DIR = "data/processed/vision_v8_unified"
CROP_DIR = "data/raw/kpz_crop_zoomed"
CLASSROOM_IMG_DIR = "data/raw/classroom_video_frames"
CLASSROOM_LBL_DIR = "data/raw/classroom_video_labels"

def clean_and_prepare_dirs():
    if os.path.exists(V8_DIR):
        print(f"Cleaning existing {V8_DIR}...")
        shutil.rmtree(V8_DIR)
    for s in ["train", "valid", "test"]:
        os.makedirs(os.path.join(V8_DIR, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(V8_DIR, s, "labels"), exist_ok=True)

def main():
    print("=" * 75)
    print("🚀 BUILDING SINGLE UNIFIED DATASET V8 (MORPHOLOGY & SCALE-INVARIANT)")
    print("=" * 75)

    clean_and_prepare_dirs()

    stats = defaultdict(lambda: defaultdict(int))
    v7_img_to_split = {}

    # -------------------------------------------------------------------------
    # 1. Copy Base V7 Dataset
    # -------------------------------------------------------------------------
    print("\n📦 [1/3] Copying Base V7 Dataset...")
    for split in ["train", "valid", "test"]:
        src_img_dir = os.path.join(V7_DIR, split, "images")
        src_lbl_dir = os.path.join(V7_DIR, split, "labels")
        dst_img_dir = os.path.join(V8_DIR, split, "images")
        dst_lbl_dir = os.path.join(V8_DIR, split, "labels")

        img_files = glob.glob(os.path.join(src_img_dir, "*.*"))
        for img_p in img_files:
            fname = os.path.basename(img_p)
            base_name = os.path.splitext(fname)[0]
            lbl_fname = base_name + ".txt"
            lbl_p = os.path.join(src_lbl_dir, lbl_fname)

            if not os.path.exists(lbl_p):
                continue

            # Record split mapping for child crops
            v7_img_to_split[fname] = split
            # Also record without prefix if any
            if fname.startswith("kpz_"):
                v7_img_to_split[fname[4:]] = split

            shutil.copy2(img_p, os.path.join(dst_img_dir, fname))
            shutil.copy2(lbl_p, os.path.join(dst_lbl_dir, lbl_fname))

            # Count classes
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(parts[0])
                        stats[split][f"cls_{c}"] += 1

            stats[split]["v7_images"] += 1
            stats[split]["total_images"] += 1

    print(f"   V7 Base copied: Train={stats['train']['v7_images']} | Val={stats['valid']['v7_images']} | Test={stats['test']['v7_images']}")

    # -------------------------------------------------------------------------
    # 2. Add CCTV Crop-Zoomed Images (Scale Inversion)
    # -------------------------------------------------------------------------
    print("\n📦 [2/3] Adding 1,200 CCTV Crop-Zoomed Images...")
    cz_img_files = glob.glob(os.path.join(CROP_DIR, "images", "*.jpg"))
    cz_added = defaultdict(int)

    for img_p in cz_img_files:
        fname = os.path.basename(img_p)
        lbl_p = os.path.join(CROP_DIR, "labels", fname.replace(".jpg", ".txt"))
        if not os.path.exists(lbl_p):
            continue

        # Extract parent kpz filename from cz_XXXX_<original_filename>
        # e.g. cz_0000_daycarecenter_20210609_00646_jpg.rf.eb1db698af0b36c1afd7a30fe91dbd92.jpg
        parts = fname.split("_", 2)
        if len(parts) >= 3:
            parent_fname = parts[2]
        else:
            parent_fname = fname

        # Determine split based on parent
        split = v7_img_to_split.get(parent_fname, None)
        if not split:
            # Check with kpz_ prefix
            split = v7_img_to_split.get(f"kpz_{parent_fname}", "train")

        dst_img_dir = os.path.join(V8_DIR, split, "images")
        dst_lbl_dir = os.path.join(V8_DIR, split, "labels")

        shutil.copy2(img_p, os.path.join(dst_img_dir, fname))
        shutil.copy2(lbl_p, os.path.join(dst_lbl_dir, fname.replace(".jpg", ".txt")))

        with open(lbl_p, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0])
                    stats[split][f"cls_{c}"] += 1

        stats[split]["crop_images"] += 1
        stats[split]["total_images"] += 1
        cz_added[split] += 1

    print(f"   Crop-Zoom added: Train={cz_added['train']} | Val={cz_added['valid']} | Test={cz_added['test']}")

    # -------------------------------------------------------------------------
    # 3. Add Classroom Video Keyframes (Co-occurrence Contrast)
    # -------------------------------------------------------------------------
    print("\n📦 [3/3] Adding Classroom Video Keyframes...")
    classroom_imgs = sorted(glob.glob(os.path.join(CLASSROOM_IMG_DIR, "*.jpg")))
    cr_added = defaultdict(int)

    for img_p in classroom_imgs:
        fname = os.path.basename(img_p)
        base_name = os.path.splitext(fname)[0]
        lbl_p = os.path.join(CLASSROOM_LBL_DIR, base_name + ".txt")
        if not os.path.exists(lbl_p):
            continue

        # Parse frame number (e.g. classroom_frame_0340 -> 340)
        frame_num = int(base_name.split("_")[-1])

    # Assign split chronologically:
        # First 50 frames (approx 0 to 8.2s): Train
        # Next 11 frames (approx 8.3 to 10.0s): Valid
        # Last 11 frames (approx 10.2 to 11.8s): Test
        if frame_num <= 294:
            split = "train"
        elif frame_num <= 360:
            split = "valid"
        else:
            split = "test"

        dst_img_name = f"cr_{fname}"
        dst_lbl_name = f"cr_{fname.replace('.jpg', '.txt')}"
        dst_img_dir = os.path.join(V8_DIR, split, "images")
        dst_lbl_dir = os.path.join(V8_DIR, split, "labels")

        shutil.copy2(img_p, os.path.join(dst_img_dir, dst_img_name))
        shutil.copy2(lbl_p, os.path.join(dst_lbl_dir, dst_lbl_name))

        with open(lbl_p, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0])
                    stats[split][f"cls_{c}"] += 1

        stats[split]["classroom_images"] += 1
        stats[split]["total_images"] += 1
        cr_added[split] += 1

    print(f"   Classroom frames added: Train={cr_added['train']} | Val={cr_added['valid']} | Test={cr_added['test']}")

    # -------------------------------------------------------------------------
    # 4. Generate data.yaml
    # -------------------------------------------------------------------------
    yaml_content = f"""# Ultralytics YOLO11 Dataset Configuration
# Single Unified Dataset Version 8 (Morphology & Scale-Invariant)
# Classes:
#   0: adult (Full-body)
#   1: baby_child (Full-body)

path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: adult
  1: baby_child
"""
    yaml_path = os.path.join(V8_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # -------------------------------------------------------------------------
    # 5. Summary & Verification
    # -------------------------------------------------------------------------
    total_imgs = sum(stats[s]["total_images"] for s in ["train", "valid", "test"])
    print("\n" + "=" * 75)
    print("📊 UNIFIED DATASET V8 SUMMARY")
    print("=" * 75)
    for s in ["train", "valid", "test"]:
        imgs = stats[s]["total_images"]
        pct = (imgs / total_imgs * 100) if total_imgs > 0 else 0
        c0 = stats[s]["cls_0"]
        c1 = stats[s]["cls_1"]
        print(f"  * {s.upper():<6}: {imgs:5d} images ({pct:5.1f}%) | [0] adult: {c0:5d} | [1] baby_child: {c1:5d}")

    print(f"\n  Total Images in Dataset V8: {total_imgs}")
    print(f"  data.yaml generated at: {yaml_path}")
    print("  ✅ DATASET V8 BUILD COMPLETE")

if __name__ == "__main__":
    main()
