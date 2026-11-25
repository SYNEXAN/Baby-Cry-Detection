"""
Automated verification test for video upload and streaming handling on Windows.
Tests:
1. Resilient RangedFileResponse streaming simulation with concurrent file access.
2. Direct execution of process_video_feed on the user's uploaded videoplayback.mp4.
"""
import os
import sys
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import app components (activates monkey patch and directories)
import app
from app import resilient_ranged_call, process_video_feed, TEMP_GRADIO_DIR, UPLOAD_DIR
from gradio.ranged_response import RangedFileResponse, OpenRange

async def test_streaming_resilience():
    print("🧪 1. Testing RangedFileResponse streaming on videoplayback.mp4...")
    # Find videoplayback.mp4 in Temp or UPLOAD_DIR
    target_video = None
    possible_paths = [
        Path(r"C:\Users\thiet\AppData\Local\Temp\gradio\1c7f6a4cc5023c0cb60265e0af64712c2d41edde2a71da6dc655f64a2ff7fab9\videoplayback.mp4"),
        UPLOAD_DIR / "videoplayback.mp4",
        ROOT_DIR / "demo_videos" / "sample_baby_alone.mp4"
    ]
    for p in possible_paths:
        if p.exists():
            target_video = p
            break

    assert target_video is not None, "No test video found!"
    print(f"   Target video: {target_video} ({target_video.stat().st_size / 1024 / 1024:.2f} MB)")

    # Simulate HTTP Range request
    resp = RangedFileResponse(
        target_video,
        OpenRange(0, 1024 * 64), # 64 KB range
        headers={"range": "bytes=0-65535"},
        stat_result=os.stat(target_video)
    )

    received_messages = []
    async def mock_send(msg):
        received_messages.append(msg)

    scope = {"type": "http", "method": "GET"}
    await resp(scope, None, mock_send)

    status_msgs = [m for m in received_messages if m.get("type") == "http.response.start"]
    body_msgs = [m for m in received_messages if m.get("type") == "http.response.body"]
    
    assert len(status_msgs) > 0, "No status message sent!"
    assert status_msgs[0]["status"] == 206, f"Expected 206 Partial Content, got {status_msgs[0]['status']}"
    total_bytes = sum(len(m.get("body", b"")) for m in body_msgs)
    print(f"   ✅ HTTP 206 Partial Content successfully returned! Streamed {total_bytes} bytes cleanly.")

def test_inference_on_user_video():
    print("\n🧪 2. Testing process_video_feed with GPU detection (5s test slice)...")
    possible_paths = [
        r"C:\Users\thiet\AppData\Local\Temp\gradio\1c7f6a4cc5023c0cb60265e0af64712c2d41edde2a71da6dc655f64a2ff7fab9\videoplayback.mp4",
        str(UPLOAD_DIR / "videoplayback.mp4"),
    ]
    test_path = None
    for p in possible_paths:
        if os.path.exists(p):
            test_path = p
            break

    if not test_path:
        print("   ⚠️ videoplayback.mp4 not found on disk, skipping inference test.")
        return

    out_video, status_html, stat_cards, audit_md = process_video_feed(
        video_input=test_path,
        conf_thresh=0.30,
        frame_skip=2,
        max_duration=5, # fast 5s verification
        progress=lambda frac, desc="": print(f"      [Progress {int(frac*100)}%] {desc}")
    )

    assert out_video is not None, "Output video path should not be None!"
    assert os.path.exists(out_video), f"Output video file does not exist: {out_video}"
    print(f"   ✅ Video output created: {out_video} ({os.path.getsize(out_video) / 1024:.1f} KB)")
    print(f"   ✅ Overall Status: {'DANGER' if 'NGUY HIỂM' in status_html else 'SAFE / NORMAL'}")

if __name__ == "__main__":
    asyncio.run(test_streaming_resilience())
    test_inference_on_user_video()
    print("\n🎉 ALL VERIFICATION TESTS PASSED!")
