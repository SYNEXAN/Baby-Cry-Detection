"""
Script: download_openimages_child_adult.py
Purpose: Download clean, verified, non-ambiguous adult (Man/Woman) and child (Boy/Girl)
         object detection images from Google Open Images V7 (validation + test splits)
         directly into YOLO format (0: adult, 1: baby_child).
"""

import os
import csv
import time
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

META_DIR = "data/raw/openimages_meta"
OUT_IMG_DIR = "data/raw/openimages_child_adult/images"
OUT_LBL_DIR = "data/raw/openimages_child_adult/labels"

URLS = {
    "classes": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}

TARGET_COOCCUR = 1800
TARGET_CHILD_ONLY = 900
TARGET_ADULT_ONLY = 400
MAX_IMG_DIM = 1024
RANDOM_SEED = 42


def download_file(url: str, dst: str):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print(f"  [Cached] {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
        return
    print(f"  [Downloading] {url} -> {dst} ...")
    t0 = time.time()
    urllib.request.urlretrieve(url, dst)
    print(f"  [Done] {os.path.getsize(dst)/1e6:.1f} MB in {time.time()-t0:.1f}s")


def box_iou(b1, b2):
    # b = (xmin, ymin, xmax, ymax)
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def dedup_boxes(boxes, iou_thr=0.70):
    # Keep largest boxes first when same-class boxes heavily overlap
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for b in boxes:
        if all(box_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept


def load_target_mids(class_csv: str):
    name_to_mid = {}
    with open(class_csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                name_to_mid[row[1].strip()] = row[0].strip()
    adult_mids = {name_to_mid["Man"], name_to_mid["Woman"]}
    child_mids = {name_to_mid["Boy"], name_to_mid["Girl"]}
    print(f"Verified OpenImages MIDs -> Adult (Man/Woman): {adult_mids} | Child (Boy/Girl): {child_mids}")
    return adult_mids, child_mids


def parse_split_annotations(csv_path: str, split_name: str, adult_mids: set, child_mids: set):
    per_image = defaultdict(lambda: {0: [], 1: []})
    ambiguous_images = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["LabelName"]
            if mid not in adult_mids and mid not in child_mids:
                continue
            img_id = row["ImageID"]
            # Reject group-of, depictions (drawings/statues/posters), inside-view
            if row.get("IsGroupOf") == "1" or row.get("IsDepiction") == "1" or row.get("IsInside") == "1":
                ambiguous_images.add(img_id)
                continue
            xmin = float(row["XMin"])
            xmax = float(row["XMax"])
            ymin = float(row["YMin"])
            ymax = float(row["YMax"])
            w = xmax - xmin
            h = ymax - ymin
            area = w * h
            # Keep clear, recognizable bodies (avoid microscopic distant specks or full-frame crops)
            if w < 0.05 or h < 0.08 or area < 0.015 or area > 0.92:
                continue
            cls = 0 if mid in adult_mids else 1
            per_image[img_id][cls].append((xmin, ymin, xmax, ymax))

    clean_records = []
    dropped_conflict = 0
    for img_id, cls_dict in per_image.items():
        if img_id in ambiguous_images:
            continue
        adults = dedup_boxes(cls_dict[0], iou_thr=0.70)
        children = dedup_boxes(cls_dict[1], iou_thr=0.70)
        if not adults and not children:
            continue
        # Reject any image where an adult box and child box overlap strongly (teenager/ambiguous label)
        has_conflict = False
        for ab in adults:
            for cb in children:
                if box_iou(ab, cb) > 0.45:
                    has_conflict = True
                    break
            if has_conflict:
                break
        if has_conflict:
            dropped_conflict += 1
            continue
        # Avoid ultra-crowded stadium/street scenes (> 8 boxes)
        if len(adults) + len(children) > 8:
            continue
        clean_records.append((split_name, img_id, adults, children))

    print(f"  [{split_name}] Clean candidates: {len(clean_records)} (dropped {dropped_conflict} adult/child label conflicts)")
    return clean_records


def download_and_save_one(item):
    split_name, img_id, adults, children = item
    img_out = os.path.join(OUT_IMG_DIR, f"oi_{split_name}_{img_id}.jpg")
    lbl_out = os.path.join(OUT_LBL_DIR, f"oi_{split_name}_{img_id}.txt")

    if os.path.exists(img_out) and os.path.exists(lbl_out) and os.path.getsize(img_out) > 2000:
        return True, len(adults), len(children)

    url = f"https://s3.amazonaws.com/open-images-dataset/{split_name}/{img_id}.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None or im.shape[0] < 160 or im.shape[1] < 160:
            return False, 0, 0
        h, w = im.shape[:2]
        if max(h, w) > MAX_IMG_DIM:
            scale = MAX_IMG_DIM / float(max(h, w))
            im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_out, im, [cv2.IMWRITE_JPEG_QUALITY, 92])

        lines = []
        for cls_id, blist in ((0, adults), (1, children)):
            for (xmin, ymin, xmax, ymax) in blist:
                xc = (xmin + xmax) / 2.0
                yc = (ymin + ymax) / 2.0
                bw = xmax - xmin
                bh = ymax - ymin
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        with open(lbl_out, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True, len(adults), len(children)
    except Exception:
        return False, 0, 0


def main():
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LBL_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1: DOWNLOADING OPEN IMAGES V7 METADATA (VALIDATION + TEST)")
    print("=" * 70)
    class_csv = os.path.join(META_DIR, "oidv7-class-descriptions-boxable.csv")
    val_csv = os.path.join(META_DIR, "validation-annotations-bbox.csv")
    test_csv = os.path.join(META_DIR, "test-annotations-bbox.csv")

    download_file(URLS["classes"], class_csv)
    download_file(URLS["validation"], val_csv)
    download_file(URLS["test"], test_csv)

    adult_mids, child_mids = load_target_mids(class_csv)

    print("\n" + "=" * 70)
    print("STEP 2: FILTERING CLEAN CO-OCCURRENCE & CHILD/ADULT CANDIDATES")
    print("=" * 70)
    records = []
    records.extend(parse_split_annotations(val_csv, "validation", adult_mids, child_mids))
    records.extend(parse_split_annotations(test_csv, "test", adult_mids, child_mids))

    cooccur = [r for r in records if r[2] and r[3]]
    child_only = [r for r in records if not r[2] and r[3]]
    adult_only = [r for r in records if r[2] and not r[3]]
    print(f"Total clean pools -> Co-occurrence: {len(cooccur)} | Child-only: {len(child_only)} | Adult-only: {len(adult_only)}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(cooccur)
    rng.shuffle(child_only)
    rng.shuffle(adult_only)

    selected = (
        cooccur[:TARGET_COOCCUR]
        + child_only[:TARGET_CHILD_ONLY]
        + adult_only[:TARGET_ADULT_ONLY]
    )
    print(f"Selected for download: {len(selected)} images "
          f"(cooccur={min(len(cooccur), TARGET_COOCCUR)}, "
          f"child_only={min(len(child_only), TARGET_CHILD_ONLY)}, "
          f"adult_only={min(len(adult_only), TARGET_ADULT_ONLY)})")

    print("\n" + "=" * 70)
    print("STEP 3: DOWNLOADING IMAGES IN PARALLEL (16 THREADS)")
    print("=" * 70)
    t0 = time.time()
    ok_cnt = 0
    fail_cnt = 0
    tot_a = 0
    tot_c = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(download_and_save_one, item) for item in selected]
        for idx, fut in enumerate(as_completed(futs), 1):
            ok, na, nc = fut.result()
            if ok:
                ok_cnt += 1
                tot_a += na
                tot_c += nc
            else:
                fail_cnt += 1
            if idx % 250 == 0 or idx == len(selected):
                print(f"  Progress: {idx}/{len(selected)} | OK={ok_cnt} | Failed={fail_cnt} | Elapsed={time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print(f"DONE! Saved {ok_cnt} clean OpenImages images -> [0] adult: {tot_a} | [1] baby_child: {tot_c}")
    print("=" * 70)


if __name__ == "__main__":
    main()
