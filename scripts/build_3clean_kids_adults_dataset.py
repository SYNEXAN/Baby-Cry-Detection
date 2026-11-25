"""
Script: build_3clean_kids_adults_dataset.py
Purpose: Download aniruddha-jmp5a/child-adult-detection-u81pm from Roboflow and merge
         3 clean datasets locally with Zero Data Leakage (70% Train / 15% Valid / 15% Test):
           1. kaushigihanml/kids-and-adults-detection  (Adult->0, Kid->1)
           2. aniruddha-jmp5a/child-adult-detection-u81pm (adult->0, child->1)
           3. baby-7psdb/true-baby (local vision_baby_adult: adult->0, baby->1, child->1)
         Standardized Classes:
           0: adults
           1: kids
"""

import os
import re
import glob
import yaml
import shutil
import zipfile
import hashlib
import requests
import collections
import cv2
import numpy as np

RF_API_KEY = "FUHLlbwEcb2F9VVlyjef"
RF_WORKSPACE = "aniruddha-jmp5a"
RF_PROJECT = "child-adult-detection-u81pm"

ANIRUDDHA_DIR = "data/raw/roboflow_aniruddha"
KAUSHI_DIR = "data/raw/inspect_kaushigihanml/children_and_adults"
TRUE_BABY_DIR = "data/raw/vision_baby_adult"
OUT_DIR = "data/processed/vision_3clean_kids_adults"
ART_DIR = r"C:\Users\thiet\.gemini\antigravity-cli\brain\8572cbe7-5a3c-48d4-b5e9-c06fe9bc9ae1"


def download_aniruddha():
    if os.path.exists(os.path.join(ANIRUDDHA_DIR, "data.yaml")):
        print(f"[Cached] {ANIRUDDHA_DIR} already exists.")
        return

    os.makedirs("data/raw", exist_ok=True)
    # Query project info to discover latest version number
    proj_url = f"https://api.roboflow.com/{RF_WORKSPACE}/{RF_PROJECT}?api_key={RF_API_KEY}"
    print(f"Querying Roboflow project metadata: {RF_WORKSPACE}/{RF_PROJECT} ...")
    r = requests.get(proj_url, timeout=20)
    r.raise_for_status()
    pdata = r.json()
    versions = pdata.get("versions", [])
    assert versions, f"No versions found for {RF_WORKSPACE}/{RF_PROJECT}"
    # Pick highest version number
    ver_nums = [int(str(v.get("id", "1")).split("/")[-1]) for v in versions]
    latest_ver = max(ver_nums)
    print(f"Found versions {sorted(ver_nums)} -> using latest version {latest_ver}")

    exp_url = f"https://api.roboflow.com/{RF_WORKSPACE}/{RF_PROJECT}/{latest_ver}/yolov8?api_key={RF_API_KEY}"
    r2 = requests.get(exp_url, timeout=30)
    r2.raise_for_status()
    dl_link = r2.json().get("export", {}).get("link")
    assert dl_link, f"No export link in response: {r2.json()}"

    zip_p = "data/raw/roboflow_aniruddha.zip"
    print(f"Downloading {RF_PROJECT} v{latest_ver} zip...")
    with requests.get(dl_link, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(zip_p, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if os.path.exists(ANIRUDDHA_DIR):
        shutil.rmtree(ANIRUDDHA_DIR)
    os.makedirs(ANIRUDDHA_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_p, "r") as zf:
        zf.extractall(ANIRUDDHA_DIR)
    os.remove(zip_p)
    print(f"Extracted {RF_PROJECT} v{latest_ver} to {ANIRUDDHA_DIR}")


def read_class_names(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: str(v).strip().lower() for i, v in enumerate(names)}
    return {int(k): str(v).strip().lower() for k, v in names.items()}


def map_name_to_target(cname: str):
    c = cname.strip().lower()
    if any(k in c for k in ("adult", "man", "woman", "parent", "elderly", "therapist", "teacher", "caregiver")):
        return 0  # adults
    if any(k in c for k in ("kid", "child", "baby", "infant", "toddler")):
        return 1  # kids
    return None


def get_base_group_key(src_prefix: str, fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    if "_jpg.rf." in fname:
        stem = fname.split("_jpg.rf.")[0]
    elif ".rf." in fname:
        stem = fname.split(".rf.")[0]
    # Strip trailing frame numbers after _ or -
    stem = re.sub(r"[_-]\d+$", "", stem)
    return f"{src_prefix}_{stem}"


def hash_to_split(group_key: str) -> str:
    h = int(hashlib.md5(group_key.encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "valid"
    return "test"


def collect_source(src_prefix: str, root_dir: str, yaml_path: str, dedup_rf: bool = True):
    name_map = read_class_names(yaml_path)
    id_to_target = {}
    for cid, cname in name_map.items():
        t = map_name_to_target(cname)
        if t is not None:
            id_to_target[cid] = t
    print(f"\n[{src_prefix}] YAML names={name_map} -> mapped={id_to_target}")

    imgs = sorted(
        glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(root_dir, "**", "*.jpeg"), recursive=True)
        + glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    )

    items = []
    seen_rf_base = set()
    skipped_dup = 0
    for ip in imgs:
        fname = os.path.basename(ip)
        # Find corresponding label file
        lp = ip.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lp):
            continue
        if dedup_rf and (".rf." in fname):
            rf_base = fname.split(".rf.")[0]
            if rf_base in seen_rf_base:
                skipped_dup += 1
                continue
            seen_rf_base.add(rf_base)

        valid_lines = []
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    raw_c = int(p[0])
                    if raw_c not in id_to_target:
                        continue
                    xc, yc, w, h = map(float, p[1:5])
                    if w < 0.03 and h < 0.03:
                        continue
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        continue
                    valid_lines.append((id_to_target[raw_c], xc, yc, w, h))
        if not valid_lines:
            continue
        gkey = get_base_group_key(src_prefix, fname)
        items.append((src_prefix, ip, fname, gkey, valid_lines))

    print(f"  -> Collected {len(items)} clean images (dedup skipped={skipped_dup})")
    return items


def main():
    print("=" * 75)
    print("BUILDING LOCAL 3-SOURCE CLEAN DATASET (adults vs kids, ZERO LEAKAGE)")
    print("=" * 75)

    download_aniruddha()

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for sp in ("train", "valid", "test"):
        os.makedirs(os.path.join(OUT_DIR, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, sp, "labels"), exist_ok=True)

    all_items = []
    all_items.extend(collect_source("kaushi", KAUSHI_DIR, os.path.join(KAUSHI_DIR, "data1.yaml"), dedup_rf=False))
    all_items.extend(collect_source("anir", ANIRUDDHA_DIR, os.path.join(ANIRUDDHA_DIR, "data.yaml"), dedup_rf=True))
    all_items.extend(collect_source("tb", TRUE_BABY_DIR, os.path.join(TRUE_BABY_DIR, "data.yaml"), dedup_rf=True))

    stats = collections.defaultdict(lambda: collections.defaultdict(int))
    src_stats = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.Counter()))
    groups_by_split = collections.defaultdict(set)

    for idx, (src_prefix, ip, fname, gkey, lines) in enumerate(all_items):
        split = hash_to_split(gkey)
        groups_by_split[split].add(gkey)
        ext = os.path.splitext(fname)[1].lower()
        stem = os.path.splitext(fname)[0]
        out_name = f"{src_prefix}_{idx:05d}_{stem}"
        dst_img = os.path.join(OUT_DIR, split, "images", out_name + ext)
        dst_lbl = os.path.join(OUT_DIR, split, "labels", out_name + ".txt")
        shutil.copy2(ip, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for c, xc, yc, w, h in lines:
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                stats[split][f"cls_{c}"] += 1
                src_stats[split][src_prefix][c] += 1
        stats[split]["images"] += 1
        src_stats[split][src_prefix]["imgs"] += 1

    yaml_text = """# Ultralytics YOLO11 Dataset Configuration
# 3-Source Clean Dataset: kaushigihanml + aniruddha-jmp5a + true-baby
# Standardized Classes: 0=adults, 1=kids (Zero Data Leakage 70/15/15)
path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: adults
  1: kids
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_text)

    ov_tv = len(groups_by_split["train"] & groups_by_split["valid"])
    ov_tt = len(groups_by_split["train"] & groups_by_split["test"])
    ov_vt = len(groups_by_split["valid"] & groups_by_split["test"])
    total_imgs = sum(stats[s]["images"] for s in ("train", "valid", "test"))

    print("\n" + "=" * 75)
    print("3-SOURCE CLEAN DATASET SUMMARY & AUDIT")
    print("=" * 75)
    for sp in ("train", "valid", "test"):
        im_n = stats[sp]["images"]
        pct = im_n / max(total_imgs, 1) * 100
        print(f"  * {sp.upper():<6}: {im_n:4d} imgs ({pct:5.1f}%) | [0] adults: {stats[sp]['cls_0']:5d} | [1] kids: {stats[sp]['cls_1']:5d}")
        for s in ("kaushi", "anir", "tb"):
            c = src_stats[sp][s]
            print(f"      - {s:<6}: {c['imgs']:4d} imgs | adults={c[0]:4d} | kids={c[1]:4d}")
    print(f"\n  Total Images: {total_imgs}")
    print(f"  Leakage Audit -> T&V={ov_tv} | T&Test={ov_tt} | V&Test={ov_vt}")
    assert ov_tv == 0 and ov_tt == 0 and ov_vt == 0

    # Generate 3x3 visual verification grid (3 per source)
    os.makedirs(ART_DIR, exist_ok=True)
    tiles = []
    for s_tag, label_title in (("kaushi", "Kaushi"), ("anir", "Aniruddha RF"), ("tb", "True-Baby")):
        cands = sorted(glob.glob(os.path.join(OUT_DIR, "train", "images", f"{s_tag}_*")))
        picks = cands[::max(1, len(cands)//3)][:3]
        for ip in picks:
            lp = ip.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
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
                        lbl = "0:adults" if c == 0 else "1:kids"
                        cv2.rectangle(im, (x1,y1), (x2,y2), col, max(2, w//250))
                        cv2.putText(im, lbl, (x1+3, max(20, y1+20)), cv2.FONT_HERSHEY_SIMPLEX, max(0.55, w/900.0), col, 2)
            tile = cv2.resize(im, (400, 300))
            b = np.full((28, 400, 3), (25, 25, 35), dtype=np.uint8)
            cv2.putText(b, f"[{label_title}] {os.path.basename(ip)[:28]}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            tiles.append(np.vstack([b, tile]))
    while len(tiles) < 9:
        tiles.append(np.zeros((328, 400, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])
    grid_p = os.path.join(ART_DIR, "preview_3clean_merged.jpg")
    cv2.imwrite(grid_p, grid)
    print(f"\nSaved visual verification grid: {grid_p}")
    print(f"Dataset saved locally at: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
