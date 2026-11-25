"""
Script: build_balanced_v6.py
Purpose: Build the Version 6 Balanced Vision Dataset for RT-DETR-L.
         Combines:
         1. true-baby dataset (full-body baby & caregiver)
         2. COCO Indoor Full-Body Person (solo adults & indoor caregiver poses)
         3. Roboflow kpz2 daycare CCTV (high-angle caregiver & child surveillance)
         
         Target Classes:
         - Class 0: adult
         - Class 1: baby_child
         
         Zero Leakage: Strict group-based splitting by base video/scene hash.
"""

import os
import glob
import shutil
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import cv2

TB_DIR = "data/processed/vision_baby_adult_resplit"
KPZ2_DIR = "data/raw/roboflow_kpz2"
COCO_LBL_DIR = "data/raw/coco_labels/coco/labels/val2017"
COCO_IMG_BASE = "http://images.cocodataset.org/val2017"
OUT_DIR = "data/processed/vision_balanced_v6"
KAGGLE_DIR = "data/kaggle_upload/true-baby-dataset"
METADATA_JSON = "data/DATASET_V6_METADATA.json"

INDOOR_COCO_CLASSES = {56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73}

def get_base_name(fname):
    if "_jpg.rf." in fname:
        return fname.split("_jpg.rf.")[0]
    return fname.split(".")[0]

def hash_split(name):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "valid"
    else:
        return "test"

def main():
    print("=" * 60)
    print("🚀 BUILDING BALANCED VISION DATASET VERSION 6 (FULL-BODY ADULT & BABY)")
    print("=" * 60)

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
        
    for s in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUT_DIR, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, s, "labels"), exist_ok=True)

    stats = defaultdict(lambda: defaultdict(int))
    scene_types = defaultdict(lambda: defaultdict(int))

    # =========================================================================
    # 1. Process true-baby base dataset
    # =========================================================================
    print("\n📦 [1/3] Processing true-baby base dataset...")
    for s in ["train", "valid", "test"]:
        lbl_files = glob.glob(os.path.join(TB_DIR, s, "labels", "*.txt"))
        for lf in lbl_files:
            fname = os.path.basename(lf)
            img_name = fname.replace(".txt", ".jpg")
            img_src = os.path.join(TB_DIR, s, "images", img_name)
            if not os.path.exists(img_src):
                continue

            new_lines = []
            has_adult = False
            has_baby = False
            with open(lf, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if p:
                        c = int(p[0])
                        target_c = 0 if c == 0 else 1
                        if target_c == 0:
                            has_adult = True
                        else:
                            has_baby = True
                        new_lines.append(f"{target_c} " + " ".join(p[1:]) + "\n")
                        stats[s][f"cls_{target_c}"] += 1

            if not new_lines:
                continue

            lbl_dest = os.path.join(OUT_DIR, s, "labels", f"tb_{fname}")
            with open(lbl_dest, "w") as f:
                f.writelines(new_lines)

            img_dest = os.path.join(OUT_DIR, s, "images", f"tb_{img_name}")
            shutil.copy2(img_src, img_dest)
            stats[s]["images"] += 1

            if has_adult and has_baby:
                scene_types[s]["both"] += 1
            elif has_adult:
                scene_types[s]["adult_only"] += 1
            else:
                scene_types[s]["baby_only"] += 1

        print(f"   [true-baby] {s.upper()}: {stats[s]['images']} imgs | Adult: {stats[s]['cls_0']} | Baby: {stats[s]['cls_1']}")

    # =========================================================================
    # 2. Select & Download COCO Indoor Full-Body Person
    # =========================================================================
    print("\n📦 [2/3] Processing COCO Indoor Full-Body Person...")
    coco_lbls = glob.glob(os.path.join(COCO_LBL_DIR, "*.txt"))
    selected_coco = []

    for lf in coco_lbls:
        with open(lf) as f:
            lines = [l.split() for l in f if l.strip()]
            classes = [int(l[0]) for l in lines]
            p_boxes = [l for l in lines if l[0] == "0"]
            has_indoor = any(c in INDOOR_COCO_CLASSES for c in classes)

            if p_boxes and has_indoor:
                # Prioritize solo adults with substantial body height
                if len(p_boxes) == 1:
                    bh = float(p_boxes[0][4])
                    if bh >= 0.25:  # Good full body or torso sitting
                        selected_coco.append((lf, p_boxes))
                elif 2 <= len(p_boxes) <= 3:
                    selected_coco.append((lf, p_boxes))

    print(f"   Identified {len(selected_coco)} candidate indoor adult images from COCO.")
    # Take up to 450 highest quality indoor adult images
    selected_coco = selected_coco[:450]

    def download_coco_img(item):
        lf, p_boxes = item
        fname = os.path.basename(lf)
        img_name = fname.replace(".txt", ".jpg")
        url = f"{COCO_IMG_BASE}/{img_name}"
        target_s = hash_split(img_name)
        img_dest = os.path.join(OUT_DIR, target_s, "images", f"coco_{img_name}")
        lbl_dest = os.path.join(OUT_DIR, target_s, "labels", f"coco_{fname}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                data = r.read()
                with open(img_dest, "wb") as f_out:
                    f_out.write(data)

            # Write label: only person class 0 (adult)
            with open(lbl_dest, "w") as f_out:
                for box in p_boxes:
                    f_out.write("0 " + " ".join(box[1:5]) + "\n")
            return target_s, len(p_boxes)
        except Exception as e:
            return None

    print(f"   Downloading {len(selected_coco)} COCO images in parallel...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(download_coco_img, selected_coco))

    coco_downloaded = 0
    for r in results:
        if r is not None:
            target_s, box_count = r
            stats[target_s]["images"] += 1
            stats[target_s]["cls_0"] += box_count
            scene_types[target_s]["adult_only"] += 1
            coco_downloaded += 1

    print(f"   ✅ Successfully added {coco_downloaded} COCO indoor full-body adult images.")

    # =========================================================================
    # 3. Process Daycare CCTV Caregiver dataset (kpz2)
    # =========================================================================
    print("\n📦 [3/3] Processing Daycare CCTV Caregiver dataset (kpz2)...")
    kpz2_lbls = glob.glob(os.path.join(KPZ2_DIR, "*", "labels", "*.txt"))
    base_to_files = defaultdict(list)
    for lf in kpz2_lbls:
        fname = os.path.basename(lf)
        bname = get_base_name(fname)
        base_to_files[bname].append(lf)

    # Filter base frames: between 1 and 6 people, contains adult/elderly
    selected_kpz2_bases = {}
    for bname, flist in base_to_files.items():
        sample_file = flist[0]
        with open(sample_file, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]
            if 1 <= len(lines) <= 6:
                classes = [int(l[0]) for l in lines]
                if 0 in classes or 2 in classes:
                    selected_kpz2_bases[bname] = flist

    print(f"   Selected {len(selected_kpz2_bases)} base CCTV caregiver scenes.")
    # Target ~500 images from kpz2 to reach perfect balance
    kpz2_added = 0
    for bname, flist in list(selected_kpz2_bases.items())[:300]:
        target_s = hash_split(bname)
        for lf in flist[:2]:  # At most 2 augmented variants per base frame
            fname = os.path.basename(lf)
            img_name = fname.replace(".txt", ".jpg")
            img_src = lf.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep).replace(".txt", ".jpg")
            if not os.path.exists(img_src):
                candidates = glob.glob(os.path.join(KPZ2_DIR, "*", "images", img_name))
                if candidates:
                    img_src = candidates[0]
                else:
                    continue

            new_lines = []
            has_adult = False
            has_baby = False
            with open(lf, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if p:
                        c = int(p[0])
                        # 0: adult, 2: elderly -> 0 (adult); 1: child -> 1 (baby_child)
                        target_c = 0 if c in [0, 2] else 1
                        if target_c == 0:
                            has_adult = True
                        else:
                            has_baby = True
                        new_lines.append(f"{target_c} " + " ".join(p[1:]) + "\n")
                        stats[target_s][f"cls_{target_c}"] += 1

            if not new_lines:
                continue

            lbl_dest = os.path.join(OUT_DIR, target_s, "labels", f"kpz2_{fname}")
            with open(lbl_dest, "w") as f:
                f.writelines(new_lines)

            img_dest = os.path.join(OUT_DIR, target_s, "images", f"kpz2_{img_name}")
            shutil.copy2(img_src, img_dest)
            stats[target_s]["images"] += 1
            kpz2_added += 1

            if has_adult and has_baby:
                scene_types[target_s]["both"] += 1
            elif has_adult:
                scene_types[target_s]["adult_only"] += 1
            else:
                scene_types[target_s]["baby_only"] += 1

    print(f"   ✅ Successfully added {kpz2_added} daycare CCTV caregiver images.")

    # =========================================================================
    # 4. Summary & Verification
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 VERSION 6 BALANCED DATASET FINAL STATISTICS")
    print("=" * 60)
    total_imgs = sum(stats[s]["images"] for s in ["train", "valid", "test"])
    total_adult = sum(stats[s]["cls_0"] for s in ["train", "valid", "test"])
    total_baby = sum(stats[s]["cls_1"] for s in ["train", "valid", "test"])

    for s in ["train", "valid", "test"]:
        print(f"📁 {s.upper():<6} | Images: {stats[s]['images']:<5} | Adult: {stats[s]['cls_0']:<5} | Baby/Child: {stats[s]['cls_1']:<5}")
        print(f"          | Adult-Only: {scene_types[s]['adult_only']:<4} | Baby-Only: {scene_types[s]['baby_only']:<4} | Both: {scene_types[s]['both']:<4}")

    print("-" * 60)
    print(f"🎯 TOTAL: {total_imgs} Images | Adult: {total_adult} boxes | Baby: {total_baby} boxes")
    print(f"⚖️ Ratio Adult : Baby = {total_adult / (total_adult + total_baby) * 100:.1f}% : {total_baby / (total_adult + total_baby) * 100:.1f}%")

    # Generate data.yaml
    data_yaml = f"""# Version 6 Balanced Full-Body Dataset
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['adult', 'baby_child']
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(data_yaml)

    # Save metadata JSON
    meta = {
        "version": 6,
        "classes": ["adult", "baby_child"],
        "splits": dict(stats),
        "scene_types": dict(scene_types),
        "total_images": total_imgs,
        "total_adult_boxes": total_adult,
        "total_baby_boxes": total_baby,
    }
    with open(METADATA_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n💾 Metadata saved to", METADATA_JSON)
    print("✅ BUILD COMPLETE!")

if __name__ == "__main__":
    main()
