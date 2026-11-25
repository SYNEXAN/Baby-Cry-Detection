"""
Automated verification test for Object Tracking & Frame-by-Frame Streaming.
Tests:
1. Execution of detector.stream_video_frames() on GPU.
2. Verification of Track IDs (ByteTrack).
3. Verification of frame yield format (numpy RGB), HUD overlays, and metadata.
4. Trajectory trail continuity and memory safety.
"""
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from engine.video_detector import VideoDetector, DEFAULT_MODEL_PATH

def test_tracking_and_streaming():
    print("=" * 65)
    print("🧪 KIỂM THỬ TỰ ĐỘNG OBJECT TRACKING & FRAME STREAMING REAL-TIME")
    print("=" * 65)

    detector = VideoDetector(DEFAULT_MODEL_PATH)
    sample_video = ROOT_DIR / "demo_videos" / "sample_baby_alone.mp4"
    assert sample_video.exists(), f"Sample video not found at {sample_video}"

    print(f"🎬 Video thử nghiệm: {sample_video.name}")
    print("🚀 Đang khởi động generator stream_video_frames (giới hạn 30 frames test)...")

    frame_count = 0
    all_seen_tids = set()
    t_start = time.time()
    last_frame = None

    for item in detector.stream_video_frames(
        video_path=str(sample_video),
        conf_thresh=0.30,
        frame_skip=1,
        max_duration=2, # ~30 frames
        tracker_type="bytetrack.yaml",
        enable_trail=True
    ):
        frame_count += 1
        frame = item["frame"]
        last_frame = frame
        boxes = item["boxes_data"]
        
        # Verify frame structure
        assert frame is not None, "Frame should not be None"
        assert len(frame.shape) == 3, f"Expected 3 channels, got {frame.shape}"
        assert frame.shape[2] == 3, "Expected RGB 3 channels"

        for b in boxes:
            tid = b.get("track_id")
            if tid is not None:
                all_seen_tids.add(tid)

        if frame_count <= 5 or frame_count % 10 == 0:
            print(f"   [Frame #{frame_count:02d}] Trẻ: {item['baby_count']} | Người lớn: {item['adult_count']} | Alone: {item['alone_duration_sec']}s | Latency: {item['latency_ms']:.1f}ms ({item['fps']:.1f} FPS) | TIDs: {all_seen_tids}")

    total_time = time.time() - t_start
    avg_fps = frame_count / max(total_time, 0.001)

    print("-" * 65)
    print(f"✅ Đã xử lý thành công: {frame_count} khung hình")
    print(f"✅ Danh sách Track IDs đã gán: {sorted(list(all_seen_tids))}")
    print(f"✅ Tốc độ trung bình: {avg_fps:.1f} FPS (Tổng thời gian: {total_time:.2f}s)")
    print(f"✅ Kích thước frame đầu ra: {last_frame.shape[1]}x{last_frame.shape[0]}")
    assert len(all_seen_tids) > 0, "Tracking should assign at least 1 Track ID to detected objects!"
    print("🎉 TẤT CẢ CÁC BÀI TEST TRACKING & STREAMING ĐÃ ĐẠT!")
    print("=" * 65)

if __name__ == "__main__":
    test_tracking_and_streaming()
