import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from engine.unified_detector import UnifiedMultimodalDetector

print("=== Starting Unified Pipeline Integration Test ===")
detector = UnifiedMultimodalDetector()

test_videos = [
    ("Baby Crying Alone", "demo_videos/sample_baby_crying_alone.mp4"),
    ("Adult Caring Burping", "demo_videos/sample_adult_caring_burping.mp4"),
    ("Baby Alone Ambient Noise", "demo_videos/sample_baby_alone_ambient_noise.mp4")
]

for name, vpath in test_videos:
    print(f"\n--- Testing Scenario: {name} ({vpath}) ---")
    if not os.path.exists(vpath):
        print(f"File not found: {vpath}")
        continue
    
    result = detector.process_video(vpath)
    print(f"Output Video: {result['output_video_path']} (Exists: {os.path.exists(result['output_video_path'])})")
    print(f"Total Frames: {result['total_frames']} | Duration: {result['duration_sec']:.1f}s")
    print(f"Max Baby Count: {result['max_baby']} | Max Adult Count: {result['max_adult']}")
    print(f"Alone Duration: {result['alone_duration_sec']:.1f}s ({result['unattended_pct']:.1f}%)")
    print(f"Audio VAD Status: {result['vad_result']['status']} ({result['vad_result']['label']})")
    print(f"Cry Cause: {result['cry_classification']['vietnamese_cause']} (Confidence: {result['cry_classification']['confidence']*100:.1f}%)")
    print(f"Fusion Verdict: [{result['fusion']['badge']}] {result['fusion']['title']}")
    print(f"Fusion Description: {result['fusion']['description']}")

print("\n=== Integration Test Finished Successfully! ===")
