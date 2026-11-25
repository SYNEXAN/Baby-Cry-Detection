import os
import shutil
import zipfile
import json
import time

SRC_DIR = "data/processed/vision_v8_unified"
DST_DIR = "data/kaggle_upload/vision-v8-morphology-dataset"

os.makedirs(DST_DIR, exist_ok=True)

# Copy data.yaml directly
shutil.copy2(os.path.join(SRC_DIR, "data.yaml"), os.path.join(DST_DIR, "data.yaml"))

# Create dataset-metadata.json
meta = {
    "title": "Vision V8 Morphology Invariant Dataset",
    "id": "thittn/vision-v8-morphology-dataset",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(DST_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

zip_path = os.path.join(DST_DIR, "dataset.zip")
print(f"📦 Creating {zip_path} from {SRC_DIR}...")
t0 = time.time()

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
    for root, dirs, files in os.walk(SRC_DIR):
        for f in files:
            if f.endswith(".json"):
                continue
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, SRC_DIR)
            zf.write(full_p, rel_p)

t1 = time.time()
size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"✅ DONE in {t1 - t0:.1f}s! Zip size: {size_mb:.2f} MB")
