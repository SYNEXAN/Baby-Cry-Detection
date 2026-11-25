"""
Script: download_kpz2_dataset.py
Purpose: Download and extract kpz2/child-adult-elderly (Version 2) in YOLOv8 format from Roboflow.
"""

import os
import requests
import zipfile
import shutil

API_KEY = "FUHLlbwEcb2F9VVlyjef"
WORKSPACE = "kpz2"
PROJECT = "child-adult-elderly"
VERSION = "2"
FORMAT = "yolov8"

TARGET_DIR = "data/raw/roboflow_kpz2"
ZIP_PATH = "data/raw/roboflow_kpz2.zip"

def main():
    os.makedirs("data/raw", exist_ok=True)
    
    # 1. Fetch export link from Roboflow API
    api_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/{FORMAT}?api_key={API_KEY}"
    print(f"📡 Querying Roboflow API for export link: {api_url}")
    res = requests.get(api_url)
    res.raise_for_status()
    data = res.json()
    
    download_url = data.get("export", {}).get("link")
    if not download_url:
        raise ValueError(f"Could not retrieve download link. API response: {data}")
        
    print(f"📥 Download URL obtained: {download_url[:60]}...")
    
    # 2. Download zip file with stream
    print("⏳ Downloading zip file...")
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        print(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
    print("\n✅ Download completed successfully!")
    
    # 3. Extract zip file
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    print(f"🔄 Extracting zip to {TARGET_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(TARGET_DIR)
    print("✅ Extraction complete!")
    
    # 4. Check data.yaml
    yaml_file = os.path.join(TARGET_DIR, "data.yaml")
    if os.path.exists(yaml_file):
        print("\n📄 data.yaml contents:")
        with open(yaml_file, "r") as f:
            print(f.read())
            
    # Clean up zip
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("🗑️ Removed temporary zip file.")

if __name__ == "__main__":
    main()
