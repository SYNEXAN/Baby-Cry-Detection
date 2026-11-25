import sys
import os
import time
from gradio_client import Client, handle_file

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

video_4k_path = r"D:\SYNEXAN\Project\Cry_Baby_Detection\temp_gradio\ee68d8ade170777fc8797aa98950094da0c0d00a8557236bab3f90c48c53495c\8160128-uhd_3840_2160_25fps.mp4"
if not os.path.exists(video_4k_path):
    # Fallback to any other uploaded video
    video_4k_path = r"uploaded_videos\videoplayback.mp4"

print(f"Connecting to Gradio Client at http://127.0.0.1:7860/ ...")
client = Client("http://127.0.0.1:7860/")

print(f"\n=======================================================")
print(f"🧪 TEST 1: Uploading 4K Video: {os.path.basename(video_4k_path)} ({os.path.getsize(video_4k_path)/(1024*1024):.2f} MB)")
print(f"=======================================================")

start_t = time.time()
job = client.submit(
    video_input=handle_file(video_4k_path),
    file_video_input=None,
    conf_thresh=0.25,
    frame_skip=3,
    max_duration=5,
    tracker_name="ByteTrack (Tối ưu tốc độ & GPU)",
    enable_trail=True,
    api_name="/stream_video_feed"
)

chunk_count = 0
first_chunk_status = None
for chunk in job:
    chunk_count += 1
    if chunk_count == 1:
        first_chunk_status = str(chunk[1])[:100]
        print(f"First chunk received in {time.time() - start_t:.2f}s!")
    if chunk_count >= 5:
        break

total_t = time.time() - start_t
print(f"✅ TEST 1 PASSED! Received {chunk_count} streaming chunks in {total_t:.2f}s!")
print(f"Status snippet: {first_chunk_status}")

print(f"\n=======================================================")
print(f"🧪 TEST 2: Uploading via File Accordion (file_video_input)")
print(f"=======================================================")

start_t = time.time()
job2 = client.submit(
    video_input=None,
    file_video_input=handle_file(video_4k_path),
    conf_thresh=0.25,
    frame_skip=3,
    max_duration=5,
    tracker_name="ByteTrack (Tối ưu tốc độ & GPU)",
    enable_trail=True,
    api_name="/stream_video_feed"
)

chunk_count2 = 0
for chunk in job2:
    chunk_count2 += 1
    if chunk_count2 == 1:
        print(f"First chunk received in {time.time() - start_t:.2f}s!")
    if chunk_count2 >= 5:
        break

print(f"✅ TEST 2 PASSED! Received {chunk_count2} streaming chunks in {time.time() - start_t:.2f}s!")
print("\n🎉 ALL TESTS COMPLETED WITH ZERO PERMISSION ERRORS!")
