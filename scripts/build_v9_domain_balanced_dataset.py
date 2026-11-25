"""
Script: build_v9_domain_balanced_dataset.py
Purpose: Build Domain-Balanced Multi-Age Dataset V9:
         1. Open Images V7 (Boy/Girl -> baby_child, Man/Woman -> adult; rich co-occurrence)
         2. True-Baby (infants 0-24m in cribs/beds/night-vision; c in (0,2)->0, c==1->1)
         3. Daycare CCTV (500 deduplicated keyframes from daycarecenter_20210609)
         Explicitly DROPS: COCO (adult-only indoor bias), seq_* (corrupted boxes), Tongdosa_* (outdoor temple)
         Zero Data Leakage enforced (70% Train / 15% Valid / 15% Test).
"""

import os
import glob
import shutil
import hashlib
import collections
import cv2
import numpy as np

OUT_DIR = "data/processed/vision_v9_domain_balanced"
OI_IMG_DIR = "data/raw/openimages_child_adult/images"
OI_LBL_DIR = "data/raw/openimages_child_adult/labels"
TB_DIR = "data/processed/vision_baby_adult_resplit"
KPZ2_DIR = "data/raw/roboflow_kpz2"
ART_DIR = r"C:\Users\thiet\.gemini\antigravity-cli\brain\8572cbe7-5a3c-48d4-b5e9-c06fe9bc9ae1"

MAX_DAYCARE_KEYFRAMES = 500


def get_tb_group_key(fname: str) -> str:
    if "_jpg.rf." in fname:
        prefix = fname.split("_jpg.rf.")[0]
        if "-" in prefix:
            parts = prefix.rsplit("-", 1)
            if parts[1].isdigit():
                return parts[0]
        return prefix
    return fname.split(".")[0]


def hash_to_split(group_key: str) -> str:
    h = int(hashlib.md5(group_key.encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "valid"
    return "test"


def clean_and_prepare():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for s in ("train", "valid", "test"):
        os.makedirs(os.path.join(OUT_DIR, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, s, "labels"), exist_ok=True)


def main():
    print("=" * 75)
    print("BUILDING DOMAIN-BALANCED MULTI-AGE DATASET V9 (70/15/15 ZERO LEAKAGE)")
    print("=" * 75)
    clean_and_prepare()

    stats = collections.defaultdict(lambda: collections.defaultdict(int))
    src_stats = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.Counter()))
    areas = collections.defaultdict(list)
    groups_assigned = collections.defaultdict(set)

    # -------------------------------------------------------------------------
    # 1. Open Images V7 (Boy/Girl -> baby_child, Man/Woman -> adult)
    # -------------------------------------------------------------------------
    print("\n[1/3] Adding Open Images V7 (eye-level indoor/outdoor child + adult)...")
    oi_imgs = sorted(glob.glob(os.path.join(OI_IMG_DIR, "*.jpg")))
    assert len(oi_imgs) > 500, f"Too few OpenImages files found ({len(oi_imgs)}) in {OI_IMG_DIR}"

    for img_p in oi_imgs:
        fname = os.path.basename(img_p)
        lbl_p = os.path.join(OI_LBL_DIR, fname.replace(".jpg", ".txt"))
        if not os.path.exists(lbl_p):
            continue
        group_key = "oi_" + os.path.splitext(fname)[0]
        split = hash_to_split(group_key)
        groups_assigned[split].add(group_key)

        valid_lines = []
        with open(lbl_p, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0])
                    xc, yc, w, h = map(float, p[1:5])
                    if c not in (0, 1):
                        continue
                    valid_lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    stats[split][f"cls_{c}"] += 1
                    src_stats[split]["oi"][c] += 1
                    areas[("oi", c)].append(w * h)
        if not valid_lines:
            continue
        shutil.copy2(img_p, os.path.join(OUT_DIR, split, "images", fname))
        with open(os.path.join(OUT_DIR, split, "labels", fname.replace(".jpg", ".txt")), "w") as f:
            f.writelines(valid_lines)
        stats[split]["images"] += 1
        src_stats[split]["oi"]["imgs"] += 1

    # -------------------------------------------------------------------------
    # 2. True-Baby (Infants 0-24 months, cribs/beds/night-vision)
    # -------------------------------------------------------------------------
    print("[2/3] Adding True-Baby (infants 0-24m)...")
    tb_imgs = sorted(glob.glob(os.path.join(TB_DIR, "**", "*.jpg"), recursive=True))
    seen_tb = set()
    for img_p in tb_imgs:
        fname = os.path.basename(img_p)
        if fname in seen_tb:
            continue
        seen_tb.add(fname)
        lbl_p = img_p.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(lbl_p):
            continue
        group_key = "tb_" + get_tb_group_key(fname)
        split = hash_to_split(group_key)
        groups_assigned[split].add(group_key)

        valid_lines = []
        with open(lbl_p, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0])
                    xc, yc, w, h = map(float, p[1:5])
                    if c in (0, 2):
                        target_c = 0
                    elif c == 1:
                        target_c = 1
                    else:
                        continue
                    if w < 0.04 and h < 0.04:
                        continue
                    valid_lines.append(f"{target_c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    stats[split][f"cls_{target_c}"] += 1
                    src_stats[split]["tb"][target_c] += 1
                    areas[("tb", target_c)].append(w * h)
        if not valid_lines:
            continue
        shutil.copy2(img_p, os.path.join(OUT_DIR, split, "images", f"tb_{fname}"))
        with open(os.path.join(OUT_DIR, split, "labels", f"tb_{fname.replace('.jpg', '.txt')}"), "w") as f:
            f.writelines(valid_lines)
        stats[split]["images"] += 1
        src_stats[split]["tb"]["imgs"] += 1

    # -------------------------------------------------------------------------
    # 3. Daycare CCTV (500 deduplicated keyframes, chronological 70/15/15 split)
    # -------------------------------------------------------------------------
    print("[3/3] Adding deduplicated Daycare CCTV keyframes (max 500, no seq_*/Tongdosa_*)...")
    daycare_by_frame = {}
    for img_p in sorted(glob.glob(os.path.join(KPZ2_DIR, "**", "daycarecenter_*.jpg"), recursive=True)):
        fname = os.path.basename(img_p)
        lbl_p = img_p.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(lbl_p):
            continue
        base_frame = fname.split("_jpg.rf.")[0]
        # Keep only 1 copy per unique base frame (deduplicates Roboflow 3x salt-and-pepper copies)
        if base_frame not in daycare_by_frame:
            daycare_by_frame[base_frame] = (img_p, lbl_p, fname)

    sorted_frames = sorted(daycare_by_frame.keys())
    if len(sorted_frames) > MAX_DAYCARE_KEYFRAMES:
        indices = np.linspace(0, len(sorted_frames) - 1, MAX_DAYCARE_KEYFRAMES, dtype=int)
        selected_frames = [sorted_frames[i] for i in indices]
    else:
        selected_frames = sorted_frames

    n = len(selected_frames)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    for idx, bf in enumerate(selected_frames):
        split = "train" if idx < t_end else ("valid" if idx < v_end else "test")
        img_p, lbl_p, fname = daycare_by_frame[bf]
        groups_assigned[split].add("dc_" + bf)

        valid_lines = []
        with open(lbl_p, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0])
                    xc, yc, w, h = map(float, p[1:5])
                    if c in (0, 2):
                        target_c = 0
                    elif c == 1:
                        target_c = 1
                    else:
                        continue
                    valid_lines.append(f"{target_c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    stats[split][f"cls_{target_c}"] += 1
                    src_stats[split]["dc"][target_c] += 1
                    areas[("dc", target_c)].append(w * h)
        if not valid_lines:
            continue
        shutil.copy2(img_p, os.path.join(OUT_DIR, split, "images", f"dc_{fname}"))
        with open(os.path.join(OUT_DIR, split, "labels", f"dc_{fname.replace('.jpg', '.txt')}"), "w") as f:
            f.writelines(valid_lines)
        stats[split]["images"] += 1
        src_stats[split]["dc"]["imgs"] += 1

    # Write data.yaml
    yaml_content = """# Ultralytics YOLO11 Dataset Configuration
# Domain-Balanced Multi-Age Dataset V9 (OpenImages V7 + True-Baby + Daycare Keyframes)
path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: adult
  1: baby_child
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Audit & Leakage Verification
    ov_tv = len(groups_assigned["train"] & groups_assigned["valid"])
    ov_tt = len(groups_assigned["train"] & groups_assigned["test"])
    ov_vt = len(groups_assigned["valid"] & groups_assigned["test"])
    total_imgs = sum(stats[s]["images"] for s in ("train", "valid", "test"))

    print("\n" + "=" * 75)
    print("DATASET V9 DOMAIN-BALANCED SUMMARY & AUDIT")
    print("=" * 75)
    for s in ("train", "valid", "test"):
        imgs = stats[s]["images"]
        pct = imgs / max(total_imgs, 1) * 100
        print(f"  * {s.upper():<6}: {imgs:5d} imgs ({pct:5.1f}%) | [0] adult: {stats[s]['cls_0']:5d} | [1] baby_child: {stats[s]['cls_1']:5d}")
        for src in ("oi", "tb", "dc"):
            c = src_stats[s][src]
            print(f"      - {src:<3}: {c['imgs']:4d} imgs | adult={c[0]:5d} | baby_child={c[1]:5d}")
    print("\n  Box Area Medians:")
    for key in sorted(areas.keys()):
        print(f"    src={key[0]:<3} cls={key[1]}: count={len(areas[key]):5d} | med_area={np.median(areas[key]):.4f}")
    print(f"\n  Leakage Audit -> T&V={ov_tv} | T&Test={ov_tt} | V&Test={ov_vt}")
    assert ov_tv == 0 and ov_tt == 0 and ov_vt == 0, "LEAKAGE DETECTED!"

    # Generate visual audit grid (3x3: 6 OpenImages co-occurrence/child, 2 True-Baby, 1 Daycare)
    os.makedirs(ART_DIR, exist_ok=True)
    samples = []
    oi_train = sorted(glob.glob(os.path.join(OUT_DIR, "train", "images", "oi_*.jpg")))
    # Pick 6 OpenImages images that contain BOTH adult and baby_child
    for ip in oi_train:
        lp = ip.replace("images", "labels").replace(".jpg", ".txt")
        with open(lp) as f:
            cs = {int(l.split()[0]) for l in f if l.strip()}
        if cs == {0, 1}:
            samples.append((ip, lp, "OI Co-occur"))
        if len(samples) == 6:
            break
    for pat, tag, n_pick in (("tb_*.jpg", "True-Baby", 2), ("dc_*.jpg", "Daycare CCTV", 1)):
        cands = sorted(glob.glob(os.path.join(OUT_DIR, "train", "images", pat)))
        for ip in cands[::max(1, len(cands)//(n_pick+1))][:n_pick]:
            samples.append((ip, ip.replace("images", "labels").replace(".jpg", ".txt"), tag))

    tiles = []
    for ip, lp, tag in samples[:9]:
        im = cv2.imread(ip)
        h, w = im.shape[:2]
        with open(lp) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0]); xc, yc, bw, bh = map(float, p[1:5])
                    x1, y1 = max(0, int((xc-bw/2)*w)), max(0, int((yc-bh/2)*h))
                    x2, y2 = min(w-1, int((xc+bw/2)*w)), min(h-1, int((yc+bh/2)*h))
                    col = (0, 220, 0) if c == 0 else (0, 140, 255)
                    lbl = "0:adult" if c == 0 else "1:baby_child"
                    cv2.rectangle(im, (x1, y1), (x2, y2), col, max(2, w // 250))
                    cv2.putText(im, lbl, (x1+2, max(18, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w/1000.0), col, 2)
        tile = cv2.resize(im, (420, 300))
        b = np.full((28, 420, 3), (25, 25, 35), dtype=np.uint8)
        cv2.putText(b, f"[{tag}] {os.path.basename(ip)[:30]}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        tiles.append(np.vstack([b, tile]))
    while len(tiles) < 9:
        tiles.append(np.zeros((328, 420, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])
    out_grid = os.path.join(ART_DIR, "v9_dataset_sample_audit.jpg")
    cv2.imwrite(out_grid, grid)
    print(f"\nSaved visual sample audit grid: {out_grid}")
    print("DATASET V9 BUILD COMPLETE!")


if __name__ == "__main__":
    main()
