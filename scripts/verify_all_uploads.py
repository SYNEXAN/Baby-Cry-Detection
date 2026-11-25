import time
import requests
import os
import sys
from pathlib import Path
from gradio_client import Client, handle_file

def run_tests():
    print("=" * 70)
    print("STRICT COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Connect to Client
    print("\n[Step 1] Connecting to Gradio server on http://127.0.0.1:7860/ ...")
    client = Client("http://127.0.0.1:7860/")
    print("  -> Connected successfully!")

    # 2. Direct HTTP multipart upload
    video_path = "uploaded_videos/videoplayback.mp4"
    assert os.path.exists(video_path), f"Test video not found: {video_path}"
    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"\n[Step 2] Testing direct HTTP multipart upload ({video_size_mb:.1f} MB) ...")
    t0 = time.time()
    with open(video_path, "rb") as f:
        r = requests.post("http://127.0.0.1:7860/gradio_api/upload", files={"files": ("videoplayback.mp4", f, "video/mp4")})
    upload_dt = time.time() - t0
    assert r.status_code == 200, f"Upload failed with HTTP {r.status_code}: {r.text}"
    uploaded_server_path = r.json()[0]
    print(f"  -> Upload HTTP 200 OK in {upload_dt:.2f}s! Saved at: {uploaded_server_path}")

    # 3. Test Range request on uploaded file
    print("\n[Step 3] Testing Range requests on uploaded video ...")
    range_url = f"http://127.0.0.1:7860/gradio_api/file={uploaded_server_path}"
    r_range = requests.get(range_url, headers={"Range": "bytes=0-1048575"})
    assert r_range.status_code == 206, f"Range request failed with {r_range.status_code}"
    print(f"  -> Range request HTTP 206 Partial Content OK! Content-Length: {len(r_range.content)} bytes")

    # 4. Test stream_video_feed with video_input (gr.Video)
    print("\n[Step 4] Testing stream_video_feed via video_input (gr.Video) ...")
    t0 = time.time()
    job_video = client.submit(
        video_input=handle_file(video_path),
        file_video_input=None,
        conf_thresh=0.3,
        frame_skip=3,
        max_duration=10,
        tracker_name="ByteTrack (Tối ưu tốc độ & GPU)",
        enable_trail=True,
        api_name="/stream_video_feed"
    )
    chunks_video = []
    for chunk in job_video:
        chunks_video.append(chunk)
        print(f"  -> Received frame chunk #{len(chunks_video)}: status={chunk[1][:45].strip()}...")
        if len(chunks_video) >= 2:
            break
    dt_video = time.time() - t0
    assert len(chunks_video) > 0, "No chunks received from video_input!"
    assert chunks_video[0][0] is not None, "Frame image in chunk 0 is None!"
    print(f"  -> video_input PASSED in {dt_video:.2f}s! First frame path: {chunks_video[0][0]}")

    # 5. Test stream_video_feed with file_video_input (gr.File)
    print("\n[Step 5] Testing stream_video_feed via file_video_input (gr.File) ...")
    t0 = time.time()
    job_file = client.submit(
        video_input=None,
        file_video_input=handle_file(video_path),
        conf_thresh=0.3,
        frame_skip=3,
        max_duration=10,
        tracker_name="ByteTrack (Tối ưu tốc độ & GPU)",
        enable_trail=True,
        api_name="/stream_video_feed"
    )
    chunks_file = []
    for chunk in job_file:
        chunks_file.append(chunk)
        print(f"  -> Received frame chunk #{len(chunks_file)}: status={chunk[1][:45].strip()}...")
        if len(chunks_file) >= 2:
            break
    dt_file = time.time() - t0
    assert len(chunks_file) > 0, "No chunks received from file_video_input!"
    assert chunks_file[0][0] is not None, "Frame image in chunk 0 is None!"
    print(f"  -> file_video_input PASSED in {dt_file:.2f}s! First frame path: {chunks_file[0][0]}")

    print("\n" + "=" * 70)
    print("🎉 ALL VERIFICATION TESTS PASSED FLAWLESSLY WITH SOLID EVIDENCE!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
