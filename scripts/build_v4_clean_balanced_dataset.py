import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SRC_ROOT = Path(r"data\processed\vision_3clean_kids_adults")
OUT_ROOT = Path(r"data\processed\vision_v4_clean_balanced")
ARTIFACT_DIR = Path(r"C:\Users\thiet\.gemini\antigravity-cli\brain\8572cbe7-5a3c-48d4-b5e9-c06fe9bc9ae1")

def get_source(stem: str) -> str:
    if stem.startswith("kaushi_"):
        return "kaushi"
    if stem.startswith("anir_"):
        return "aniruddha"
    if stem.startswith("tb_"):
        return "true_baby"
    return "other"

def compute_iou(b1, b2):
    # [x1, y1, x2, y2] normalized
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / max(1e-9, a1 + a2 - inter)

def read_labels(lbl_path: Path):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        xc, yc, bw, bh = map(float, p[1:5])
        area = bw * bh
        # Drop tiny noise boxes (<0.8% area or <3% side) and corrupted full-frame boxes (>88% area)
        if area < 0.008 or min(bw, bh) < 0.03 or area > 0.88:
            continue
        boxes.append((cls, xc, yc, bw, bh))
    return boxes

def salt_pepper_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.int16) - med.astype(np.int16))
    return float(np.mean(diff > 45))

def is_grayscale(img_bgr):
    return float(np.mean(np.abs(img_bgr[:, :, 0].astype(np.int16) - img_bgr[:, :, 1].astype(np.int16)))) < 2.0

def main():
    print("=== STEP 1: COLLECT ALL SOURCE IMAGES & FILTER TINY/HUGE BOXES ===")
    records = []
    dropped_empty = 0
    for sp in ["train", "valid", "test"]:
        img_dir = SRC_ROOT / sp / "images"
        lbl_dir = SRC_ROOT / sp / "labels"
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = read_labels(lbl_path)
            if not boxes:
                dropped_empty += 1
                continue
            records.append({
                "src": get_source(img_path.stem),
                "img_path": img_path,
                "stem": img_path.stem,
                "boxes": boxes,
            })
    print(f"Loaded {len(records)} valid records (dropped {dropped_empty} empty/tiny-only images).")

    print("\n=== STEP 2: AUTO-AUDIT MISSING PERSON LABELS (aniruddha & true_baby) ===")
    audit_model = YOLO("yolo11n.pt")
    to_audit = [r for r in records if r["src"] in ("aniruddha", "true_baby")]
    keep_records = [r for r in records if r["src"] == "kaushi"]
    dropped_missing = []

    BATCH = 16
    for i in range(0, len(to_audit), BATCH):
        batch = to_audit[i:i + BATCH]
        paths = [str(r["img_path"]) for r in batch]
        results = audit_model.predict(paths, imgsz=512, conf=0.65, classes=[0], verbose=False)
        for r, res in zip(batch, results):
            gt_xyxy = [
                [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2]
                for (_, xc, yc, bw, bh) in r["boxes"]
            ]
            has_missing = False
            miss_box = None
            h, w = res.orig_shape
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxyn[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area < 0.08:
                    continue
                max_iou = max((compute_iou([x1, y1, x2, y2], g) for g in gt_xyxy), default=0.0)
                if max_iou < 0.15:
                    has_missing = True
                    miss_box = (x1, y1, x2, y2, float(b.conf[0]))
                    break
            if has_missing:
                dropped_missing.append((r, miss_box))
            else:
                keep_records.append(r)

    miss_by_src = defaultdict(int)
    for r, _ in dropped_missing:
        miss_by_src[r["src"]] += 1
    print(f"Dropped missing-label images: total={len(dropped_missing)} | {dict(miss_by_src)}")

    # Save visual audit of dropped missing-label images
    if dropped_missing:
        vis_picks = random.sample(dropped_missing, min(8, len(dropped_missing)))
        tiles = []
        for r, mb in vis_picks:
            im = cv2.imread(str(r["img_path"]))
            if im is None:
                continue
            h, w = im.shape[:2]
            for cls, xc, yc, bw, bh in r["boxes"]:
                x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                col = (255, 100, 0) if cls == 0 else (0, 215, 255)
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            # Draw the unlabeled person in RED
            mx1, my1, mx2, my2, mcf = mb
            cv2.rectangle(im, (int(mx1 * w), int(my1 * h)), (int(mx2 * w), int(my2 * h)), (0, 0, 255), 3)
            cv2.putText(im, f"MISSING PERSON {mcf:.2f}", (int(mx1 * w), max(20, int(my1 * h) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            im = cv2.resize(im, (320, 240))
            cv2.putText(im, r["src"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            tiles.append(im)
        if len(tiles) >= 4:
            row1 = np.hstack(tiles[:4])
            grid = np.vstack([row1, np.hstack(tiles[4:8])]) if len(tiles) >= 8 else row1
            cv2.imwrite(str(ARTIFACT_DIR / "audit_v4_dropped_missing_labels.jpg"), grid)
            print("Saved audit_v4_dropped_missing_labels.jpg")

    print("\n=== STEP 3: FILTER SALT-AND-PEPPER NOISE & BALANCE true_baby ===")
    kaushi_recs = [r for r in keep_records if r["src"] == "kaushi"]
    anir_recs = [r for r in keep_records if r["src"] == "aniruddha"]
    tb_recs = [r for r in keep_records if r["src"] == "true_baby"]

    tb_with_adults = []
    tb_gray_clean = []
    tb_color_clean = []
    tb_noisy_dropped = 0

    for r in tb_recs:
        im = cv2.imread(str(r["img_path"]))
        if im is None:
            continue
        sp_score = salt_pepper_score(im)
        has_adult = any(b[0] == 0 for b in r["boxes"])
        if sp_score > 0.015 and not has_adult:
            tb_noisy_dropped += 1
            continue
        r["sp_score"] = sp_score
        r["area_px"] = im.shape[0] * im.shape[1]
        if has_adult:
            tb_with_adults.append(r)
        elif is_grayscale(im):
            tb_gray_clean.append(r)
        else:
            tb_color_clean.append(r)

    # Sort by lowest salt-and-pepper noise, highest resolution
    tb_gray_clean.sort(key=lambda x: (x["sp_score"], -x["area_px"]))
    tb_color_clean.sort(key=lambda x: (x["sp_score"], -x["area_px"]))

    # Keep ALL tb_with_adults + 200 cleanest grayscale CCTV crib/bed babies + 200 cleanest color babies
    tb_selected = tb_with_adults + tb_gray_clean[:200] + tb_color_clean[:200]
    print(f"true_baby: dropped_noisy={tb_noisy_dropped} | kept={len(tb_selected)} (with_adults={len(tb_with_adults)}, gray_cctv={min(200, len(tb_gray_clean))}, color={min(200, len(tb_color_clean))})")

    all_clean = kaushi_recs + anir_recs + tb_selected
    print(f"\nTotal clean unique images before split: {len(all_clean)} (kaushi={len(kaushi_recs)}, anir={len(anir_recs)}, tb={len(tb_selected)})")

    print("\n=== STEP 4: STRATIFIED 70/15/15 ZERO-LEAKAGE SPLIT + KAUSHI x2 IN TRAIN ONLY ===")
    buckets = defaultdict(list)
    for r in all_clean:
        has_a = int(any(b[0] == 0 for b in r["boxes"]))
        has_k = int(any(b[0] == 1 for b in r["boxes"]))
        buckets[(r["src"], has_a, has_k)].append(r)

    splits = {"train": [], "valid": [], "test": []}
    for key, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_tr = int(round(n * 0.70))
        n_va = int(round(n * 0.15))
        splits["train"].extend(items[:n_tr])
        splits["valid"].extend(items[n_tr:n_tr + n_va])
        splits["test"].extend(items[n_tr + n_va:])

    # Verify Zero Leakage
    tr_stems = {r["stem"] for r in splits["train"]}
    va_stems = {r["stem"] for r in splits["valid"]}
    te_stems = {r["stem"] for r in splits["test"]}
    assert len(tr_stems & va_stems) == 0 and len(tr_stems & te_stems) == 0 and len(va_stems & te_stems) == 0, "DATA LEAKAGE DETECTED!"

    # Write clean dataset
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for sp in ["train", "valid", "test"]:
        (OUT_ROOT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / sp / "labels").mkdir(parents=True, exist_ok=True)

    summary = {}
    for sp in ["train", "valid", "test"]:
        n_imgs = 0
        n_a = 0
        n_k = 0
        kaushi_dups = 0
        for r in splits[sp]:
            ext = r["img_path"].suffix.lower()
            dst_img = OUT_ROOT / sp / "images" / f"{r['stem']}{ext}"
            dst_lbl = OUT_ROOT / sp / "labels" / f"{r['stem']}.txt"
            shutil.copy2(r["img_path"], dst_img)
            lbl_text = "\n".join(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for c, xc, yc, bw, bh in r["boxes"]) + "\n"
            dst_lbl.write_text(lbl_text, encoding="utf-8")
            n_imgs += 1
            a_c = sum(1 for b in r["boxes"] if b[0] == 0)
            k_c = sum(1 for b in r["boxes"] if b[0] == 1)
            n_a += a_c
            n_k += k_c

            # Oversample kaushi x2 ONLY in train split
            if sp == "train" and r["src"] == "kaushi":
                dup_img = OUT_ROOT / sp / "images" / f"dup2_{r['stem']}{ext}"
                dup_lbl = OUT_ROOT / sp / "labels" / f"dup2_{r['stem']}.txt"
                shutil.copy2(r["img_path"], dup_img)
                dup_lbl.write_text(lbl_text, encoding="utf-8")
                n_imgs += 1
                n_a += a_c
                n_k += k_c
                kaushi_dups += 1

        summary[sp] = (n_imgs, n_a, n_k, kaushi_dups)
        print(f"Split [{sp:5s}]: imgs={n_imgs:4d} (kaushi_dup={kaushi_dups:3d}) | adults={n_a:4d} | kids={n_k:4d} | ratio kids:adults = {n_k/max(1,n_a):.2f}:1")

    yaml_text = (
        f"path: {OUT_ROOT.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 2\n"
        "names:\n"
        "  0: adults\n"
        "  1: kids\n"
    )
    (OUT_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"\nDONE! Saved V4 Clean-Balanced dataset to: {OUT_ROOT}")

if __name__ == "__main__":
    main()
