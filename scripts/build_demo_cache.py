"""
Script: build_demo_cache.py
Purpose: Pre-renders annotated videos and caches summary stats for the 3 demo sample clips.
"""

import os
import sys
import json
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.video_detector import VideoDetector, DEFAULT_MODEL_PATH

DEMO_VIDEOS_DIR = ROOT_DIR / "demo_videos"
CACHE_DIR = ROOT_DIR / "demo_cache"
CACHE_DIR.mkdir(exist_ok=True)

SAMPLES = {
    "baby_alone": str(DEMO_VIDEOS_DIR / "sample_baby_alone.mp4"),
    "adult_caring": str(DEMO_VIDEOS_DIR / "sample_adult_caring.mp4"),
    "daycare": str(DEMO_VIDEOS_DIR / "sample_daycare_indoor.mp4"),
}

def main():
    print("🚀 Initializing detector for demo cache generation...")
    detector = VideoDetector(DEFAULT_MODEL_PATH)
    
    metadata = {}
    for key, path in SAMPLES.items():
        if not os.path.exists(path):
            print(f"⚠️ Warning: Sample video not found: {path}")
            continue
            
        print(f"\n🎬 Pre-rendering {key} ({path}) ...")
        out_video, summary = detector.process_video(
            video_path=path,
            conf_thresh=0.30,
            iou_thresh=0.45,
            frame_skip=2,
            max_duration=10
        )
        
        target_video = CACHE_DIR / f"{key}_annotated.mp4"
        shutil.copy(out_video, target_video)
        print(f"✅ Saved cached video: {target_video}")
        
        metadata[key] = {
            "video_path": str(target_video).replace("\\", "/"),
            "summary": summary
        }
        
    meta_path = CACHE_DIR / "demo_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 Successfully created all demo cache at {meta_path}!")

if __name__ == "__main__":
    main()
