import time
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from engine.unified_detector import UnifiedMultimodalDetector

def main():
    print("=== SYSTEM & HARDWARE VERIFICATION ===")
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Active Device    : {device}")
    print(f"Torch CPU Threads: {torch.get_num_threads()}")

    model_path = "kaggle_results/v4_clean_balanced/best.pt"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found!")
        return

    print("\n=== BENCHMARK 1: RAW YOLO11-m VISION MODEL INFERENCE ON CPU ===")
    model = YOLO(model_path)
    
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        _ = model.predict(source=dummy_frame, imgsz=960, conf=0.35, iou=0.45, device="cpu", verbose=False)

    # Test imgsz = 960
    latencies_960 = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model.predict(source=dummy_frame, imgsz=960, conf=0.35, iou=0.45, device="cpu", verbose=False)
        latencies_960.append(time.perf_counter() - t0)

    avg_lat_960 = np.mean(latencies_960) * 1000
    fps_960 = 1000.0 / avg_lat_960
    print(f"[imgsz=960] Latency per frame: {avg_lat_960:.1f} ms | Raw Inference FPS: {fps_960:.2f} FPS")

    # Test imgsz = 640
    latencies_640 = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model.predict(source=dummy_frame, imgsz=640, conf=0.35, iou=0.45, device="cpu", verbose=False)
        latencies_640.append(time.perf_counter() - t0)

    avg_lat_640 = np.mean(latencies_640) * 1000
    fps_640 = 1000.0 / avg_lat_640
    print(f"[imgsz=640] Latency per frame: {avg_lat_640:.1f} ms | Raw Inference FPS: {fps_640:.2f} FPS")

    print("\n=== BENCHMARK 2: END-TO-END STREAMING ON REAL TEST VIDEO ===")
    test_video = "VIDEO_TEST/294613747-rivne-ukraine-june-12-2024-chi.mp4"
    if not os.path.exists(test_video):
        video_files = [os.path.join("VIDEO_TEST", f) for f in os.listdir("VIDEO_TEST") if f.endswith(".mp4")]
        if video_files:
            test_video = video_files[0]

    print(f"Testing video: {test_video}")
    detector = UnifiedMultimodalDetector(
        vision_model_path="kaggle_results/v4_clean_balanced/best.pt",
        audio_weights_path="models/best_eletriby_4class_audio_classifier.pt"
    )

    t0 = time.perf_counter()
    frame_count = 0
    yielded_count = 0
    for step in detector.stream_video(test_video, conf_thresh=0.35, iou_thresh=0.45, imgsz=960, frame_skip=4):
        yielded_count += 1
    t_total = time.perf_counter() - t0

    cap = cv2.VideoCapture(test_video)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    effective_throughput_fps = total_video_frames / t_total

    print(f"Total Video Frames      : {total_video_frames} frames ({total_video_frames / video_fps:.2f} s video)")
    print(f"Total Pipeline Time     : {t_total:.2f} s")
    print(f"Effective Processing FPS: {effective_throughput_fps:.2f} FPS (with frame_skip=4)")
    print(f"Playback Speed Factor   : {effective_throughput_fps / video_fps:.2f}x Realtime")

if __name__ == "__main__":
    main()
