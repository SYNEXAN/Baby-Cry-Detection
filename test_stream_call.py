import sys
from gradio_client import Client, handle_file
import time

print("1. Connecting to Client...", flush=True)
client = Client("http://127.0.0.1:7860/")
print("Connected!", flush=True)

print("\n2. Submitting with video_input (gr.Video)...", flush=True)
t0 = time.time()
job = client.submit(
    video_input=handle_file("uploaded_videos/videoplayback.mp4"),
    file_video_input=None,
    conf_thresh=0.3,
    frame_skip=2,
    max_duration=10,
    tracker_name="ByteTrack (Tối ưu tốc độ & GPU)",
    enable_trail=True,
    api_name="/stream_video_feed"
)
print(f"Job submitted in {time.time()-t0:.2f}s, waiting for chunks...", flush=True)

for i, chunk in enumerate(job):
    print(f"Chunk {i+1} received! Frame type: {type(chunk[0])}, status: {chunk[1][:50]}", flush=True)
    if i >= 2:
        break

print("All done for video_input!", flush=True)
