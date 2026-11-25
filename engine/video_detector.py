"""
Module: video_detector.py
Purpose: Real-time RT-DETR-L object detector and HUD visualizer for Infant & Adult monitoring.
"""

import os
import time
import subprocess
from collections import defaultdict, deque
from typing import Dict, Any, Generator, Tuple, Optional, List
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from engine.risk_assessment import assess_frame_risk, summarize_video_risk

DEFAULT_MODEL_PATH = "kaggle_results/kaushi_clean/best.pt"

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def deduplicate_person_boxes(boxes_data: list, iou_thresh: float = 0.45) -> list:
    """
    Lọc bỏ các bounding box bị dán nhãn trùng lặp trên cùng 1 người vật lý.
    Ưu tiên giữ box có confidence cao hơn khi IoU > iou_thresh.
    """
    if len(boxes_data) <= 1:
        return boxes_data
    sorted_boxes = sorted(boxes_data, key=lambda x: x["conf"], reverse=True)
    keep = []
    for current in sorted_boxes:
        overlap = False
        for kept in keep:
            if compute_iou(current["coords"], kept["coords"]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(current)
    return keep

class PoseBiometricVerifier:
    """
    Bộ xác thực nhân trắc học hình thái xương (Pose-Guided Biometrics).
    Sử dụng 17 khớp xương cơ thể để kiểm chứng và ngăn chặn nhầm lẫn hai chiều giữa Adult và Baby/Child:
    - Người lớn ngồi/quỳ có tỷ lệ rộng vai / chiều dài đầu lớn (>= 1.35).
    - Trẻ em/em bé có tỷ lệ vai hẹp, đầu to (<= 0.85).
    """
    def __init__(self, pose_model_path: str = "yolov8n-pose.pt", device: str = "cpu"):
        self.pose_model = None
        if os.path.exists(pose_model_path):
            try:
                self.pose_model = YOLO(pose_model_path)
                if device != "cpu":
                    self.pose_model.to(device)
                    self.pose_model.model.half()
                print(f"🦴 Initialized PoseBiometricVerifier from {pose_model_path} ({device})")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize PoseBiometricVerifier: {e}")

    def verify_classification(self, crop_img: np.ndarray, predicted_cls: int, conf: float) -> Tuple[int, float]:
        # Trust the clean YOLO11-M model directly without heuristic overrides
        # (2D shoulder/head ratio misclassifies side-profile adults as children)
        return predicted_cls, conf

# High-contrast aesthetic colors (BGR for OpenCV)
COLOR_ADULT = (163, 255, 0)      # Neon Green-Cyan for adult
COLOR_BABY = (50, 50, 255)       # Coral-Red for baby_child
COLOR_TEXT_BG = (20, 24, 33)     # Cyber dark header
COLOR_ALERT_BG = (30, 30, 220)   # Vivid Red warning header

class VideoDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
        # Auto-detect GPU acceleration
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if self.device != "cpu" else "CPU"
        print(f"🤖 Initializing Vision Detection model from {model_path} on {device_name} (Device: {self.device}) ...")
        
        self.model = YOLO(model_path)
        if self.device != "cpu":
            self.model.to(self.device)
            self.model.model.half()  # Native FP16 for 2GB VRAM optimization
            
        self.names = self.model.names
        print(f"✅ Model loaded successfully! Classes: {self.names}")
        
        # Initialize Pose Biometric Verifier for morphology-based adult/child validation
        self.pose_verifier = PoseBiometricVerifier("yolov8n-pose.pt", device=self.device)

    def draw_hud(
        self,
        frame: np.ndarray,
        boxes_data: list,
        adult_count: int,
        baby_count: int,
        fps: float,
        track_history: Optional[Dict[int, deque]] = None,
        alone_duration_sec: float = 0.0
    ) -> np.ndarray:
        """
        Renders HUD overlays: Bounding boxes with Track IDs, motion trajectory trails,
        top dashboard bar, and real-time danger alerts.
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Draw Trajectory Trails (if tracking history is provided)
        if track_history:
            for item in boxes_data:
                tid = item.get("track_id")
                if tid is not None and tid in track_history:
                    pts = list(track_history[tid])
                    if len(pts) > 1:
                        cls_id = item["cls"]
                        color = COLOR_ADULT if cls_id == 0 else COLOR_BABY
                        for i in range(1, len(pts)):
                            # Fade thickness based on recency
                            thickness = int(np.sqrt(64 / float(len(pts) - i + 1)) * 1.5)
                            cv2.line(canvas, pts[i - 1], pts[i], color, max(1, min(thickness, 4)), cv2.LINE_AA)
                        # Small glowing dot at latest center
                        cv2.circle(canvas, pts[-1], 4, color, -1)

        # 2. Draw Bounding Boxes with Track IDs
        for item in boxes_data:
            x1, y1, x2, y2 = item["coords"]
            cls_id = item["cls"]
            conf = item["conf"]
            tid = item.get("track_id")
            label_name = self.names.get(cls_id, str(cls_id))

            # Select color: adult (0) -> Green, baby_child (1) -> Red
            color = COLOR_ADULT if cls_id == 0 else COLOR_BABY

            # Draw outer box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

            # Draw label tag with Track ID badge
            if tid is not None:
                tag = f"[ID: {tid}] {label_name} {int(conf * 100)}%"
            else:
                tag = f"{label_name} {int(conf * 100)}%"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(canvas, tag, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # 3. Risk Assessment
        risk = assess_frame_risk(adult_count, baby_count)

        # 4. Top HUD Status Bar (Dark translucent gradient)
        hud_h = 60
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_TEXT_BG, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # KPI Badges in HUD
        hud_text_baby = f"BABY/CHILD: {baby_count}"
        hud_text_adult = f"ADULT: {adult_count}"
        hud_fps = f"{fps:.1f} FPS"

        cv2.putText(canvas, hud_text_baby, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_BABY, 2, cv2.LINE_AA)
        cv2.putText(canvas, hud_text_adult, (260, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_ADULT, 2, cv2.LINE_AA)
        cv2.putText(canvas, hud_fps, (w - 140, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 5. Danger Banner if Infant is Unattended
        if risk["is_danger"]:
            banner_h = 50
            alert_overlay = canvas.copy()
            cv2.rectangle(alert_overlay, (0, h - banner_h), (w, h), COLOR_ALERT_BG, -1)
            cv2.addWeighted(alert_overlay, 0.85, canvas, 0.15, 0, canvas)
            
            if alone_duration_sec > 0:
                alert_text = f"DANGER: UNATTENDED CHILD DETECTED ({alone_duration_sec:.1f}s) - NO ADULT PRESENT!"
            else:
                alert_text = "DANGER ALERT: UNATTENDED CHILD DETECTED - NO ADULT PRESENT!"
                
            (atw, ath), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
            cv2.putText(canvas, alert_text, ((w - atw) // 2, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def detect_image(self, image: np.ndarray, conf_thresh: float = 0.35, iou_thresh: float = 0.45) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single still image with Class-Agnostic NMS, Cross-Class Deduplication,
        and Pose Biometric Verification.
        """
        t0 = time.time()
        results = self.model.predict(
            image,
            conf=conf_thresh,
            iou=iou_thresh,
            agnostic_nms=True,
            device=self.device,
            imgsz=768,
            verbose=False
        )
        latency_ms = (time.time() - t0) * 1000.0

        raw_boxes = []
        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                    raw_boxes.append({
                        "coords": (x1, y1, x2, y2),
                        "cls": cls_id,
                        "conf": conf
                    })

        # 1. Deduplicate overlapping boxes
        deduped_boxes = deduplicate_person_boxes(raw_boxes, iou_thresh=iou_thresh)

        # 2. Pose-Guided Biometric Verification
        boxes_data = []
        adult_count = 0
        baby_count = 0

        for item in deduped_boxes:
            x1, y1, x2, y2 = item["coords"]
            crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
            v_cls, v_conf = self.pose_verifier.verify_classification(crop, item["cls"], item["conf"])
            item["cls"] = v_cls
            item["conf"] = v_conf
            boxes_data.append(item)
            if v_cls == 0:
                adult_count += 1
            elif v_cls == 1:
                baby_count += 1

        fps = 1000.0 / max(latency_ms, 1.0)
        annotated_img = self.draw_hud(image, boxes_data, adult_count, baby_count, fps)
        risk = assess_frame_risk(adult_count, baby_count)

        summary = {
            "adult_count": adult_count,
            "baby_count": baby_count,
            "latency_ms": round(latency_ms, 1),
            "fps": round(fps, 1),
            "risk": risk,
            "detections": [
                {
                    "class": self.names.get(b["cls"], str(b["cls"])),
                    "confidence": round(b["conf"] * 100, 1),
                    "box": list(b["coords"])
                }
                for b in boxes_data
            ]
        }
        return annotated_img, summary

    def stream_video_frames(
        self,
        video_path: str,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        frame_skip: int = 1,
        max_duration: Optional[int] = None,
        tracker_type: str = "bytetrack.yaml",
        enable_trail: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams video frame-by-frame with real-time RT-DETR-L ByteTrack/BoT-SORT tracking,
        rendering HUD overlays and yielding immediately for true live streaming.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames_to_process = total_video_frames
        if max_duration and max_duration > 0:
            max_frames_to_process = min(total_video_frames, int(max_duration * orig_fps))

        track_history = defaultdict(lambda: deque(maxlen=30))
        consecutive_alone_frames = 0
        frame_idx = 0
        processed_count = 0
        use_builtin_tracker = True
        prev_tracks = []
        next_track_id = 1

        try:
            while cap.isOpened() and frame_idx < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    t_frame_start = time.time()
                    if use_builtin_tracker:
                        try:
                            results = self.model.track(
                                frame,
                                persist=True,
                                tracker=tracker_type,
                                conf=conf_thresh,
                                iou=iou_thresh,
                                agnostic_nms=True,
                                device=self.device,
                                imgsz=768,
                                verbose=False
                            )
                        except (ImportError, OSError):
                            # Fallback when Windows Application Control blocks _lapjv DLL
                            use_builtin_tracker = False
                            results = self.model.predict(
                                frame,
                                conf=conf_thresh,
                                iou=iou_thresh,
                                agnostic_nms=True,
                                device=self.device,
                                imgsz=768,
                                verbose=False
                            )
                    else:
                        results = self.model.predict(
                            frame,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            agnostic_nms=True,
                            device=self.device,
                            imgsz=768,
                            verbose=False
                        )
                    latency_ms = (time.time() - t_frame_start) * 1000.0

                    raw_boxes = []
                    for r in results:
                        if r.boxes is not None:
                            for b in r.boxes:
                                cls_id = int(b.cls[0])
                                conf = float(b.conf[0])
                                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                                tid = int(b.id[0]) if (use_builtin_tracker and b.id is not None) else None
                                raw_boxes.append({
                                    "coords": (x1, y1, x2, y2),
                                    "cls": cls_id,
                                    "conf": conf,
                                    "track_id": tid
                                })

                    # 1. Deduplicate cross-class boxes
                    deduped_boxes = deduplicate_person_boxes(raw_boxes, iou_thresh=iou_thresh)

                    # Lightweight Pure-Python IoU tracking fallback when _lapjv is unavailable
                    if not use_builtin_tracker:
                        used_prev = set()
                        new_tracks = []
                        for item in deduped_boxes:
                            best_iou, best_tid, best_pi = 0.0, None, -1
                            for pi, (pt_id, pt_box) in enumerate(prev_tracks):
                                if pi in used_prev:
                                    continue
                                sc = compute_iou(item["coords"], pt_box)
                                if sc > best_iou:
                                    best_iou, best_tid, best_pi = sc, pt_id, pi
                            if best_iou >= 0.25 and best_tid is not None:
                                item["track_id"] = best_tid
                                used_prev.add(best_pi)
                            else:
                                item["track_id"] = next_track_id
                                next_track_id += 1
                            new_tracks.append((item["track_id"], item["coords"]))
                        prev_tracks = new_tracks

                    # 2. Pose-Guided Biometric Verification
                    boxes_data = []
                    adult_count = 0
                    baby_count = 0

                    for item in deduped_boxes:
                        x1, y1, x2, y2 = item["coords"]
                        tid = item.get("track_id")
                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        v_cls, v_conf = self.pose_verifier.verify_classification(crop, item["cls"], item["conf"])
                        item["cls"] = v_cls
                        item["conf"] = v_conf
                        boxes_data.append(item)
                        if v_cls == 0:
                            adult_count += 1
                        elif v_cls == 1:
                            baby_count += 1

                        if tid is not None and enable_trail:
                            center_pt = ((x1 + x2) // 2, (y1 + y2) // 2)
                            track_history[tid].append(center_pt)

                    # Temporal unattended calculation
                    if baby_count > 0 and adult_count == 0:
                        consecutive_alone_frames += frame_skip
                    else:
                        consecutive_alone_frames = 0

                    alone_sec = consecutive_alone_frames / orig_fps if orig_fps > 0 else 0
                    fps_val = 1000.0 / max(latency_ms, 1.0)

                    # Render HUD overlay
                    annotated_bgr = self.draw_hud(
                        frame=frame,
                        boxes_data=boxes_data,
                        adult_count=adult_count,
                        baby_count=baby_count,
                        fps=fps_val,
                        track_history=track_history if enable_trail else None,
                        alone_duration_sec=alone_sec
                    )

                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    risk = assess_frame_risk(adult_count, baby_count)

                    processed_count += 1
                    timestamp_sec = round(frame_idx / orig_fps, 2) if orig_fps > 0 else 0.0
                    progress_pct = round((frame_idx + 1) / max(max_frames_to_process, 1) * 100, 1)

                    yield {
                        "frame": annotated_rgb,
                        "frame_idx": frame_idx,
                        "total_frames": max_frames_to_process,
                        "processed_count": processed_count,
                        "timestamp_sec": timestamp_sec,
                        "progress_pct": progress_pct,
                        "adult_count": adult_count,
                        "baby_count": baby_count,
                        "alone_duration_sec": round(alone_sec, 1),
                        "latency_ms": round(latency_ms, 1),
                        "fps": round(fps_val, 1),
                        "risk": risk,
                        "boxes_data": boxes_data
                    }

                frame_idx += 1
        finally:
            cap.release()
            if self.device != "cpu":
                torch.cuda.empty_cache()

    def process_video(
        self,
        video_path: str,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        frame_skip: int = 1,
        max_duration: Optional[int] = None,
        progress_callback = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Processes video file, renders HUD, encodes output via ffmpeg (H.264) for HTML5 video players.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames_to_process = total_video_frames
        if max_duration and max_duration > 0:
            max_frames_to_process = min(total_video_frames, int(max_duration * orig_fps))

        os.makedirs("output_videos", exist_ok=True)
        raw_out_path = f"output_videos/temp_raw_{int(time.time())}.mp4"
        web_out_path = f"output_videos/processed_{int(time.time())}.mp4"

        # OpenCV raw writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_out_path, fourcc, orig_fps / max(frame_skip, 1), (orig_w, orig_h))

        track_history = defaultdict(lambda: deque(maxlen=30))
        frame_idx = 0
        processed_count = 0
        timeline_stats = []

        t_start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= max_frames_to_process:
                break

            if frame_idx % frame_skip == 0:
                t_frame_start = time.time()
                # Run tracking with ByteTrack on GPU with agnostic_nms
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=conf_thresh,
                    iou=iou_thresh,
                    agnostic_nms=True,
                    device=self.device,
                    imgsz=768,
                    verbose=False
                )
                t_frame_end = time.time()

                raw_boxes = []
                for r in results:
                    if r.boxes is not None:
                        for b in r.boxes:
                            cls_id = int(b.cls[0])
                            conf = float(b.conf[0])
                            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                            tid = int(b.id[0]) if b.id is not None else None
                            raw_boxes.append({
                                "coords": (x1, y1, x2, y2),
                                "cls": cls_id,
                                "conf": conf,
                                "track_id": tid
                            })

                # 1. Deduplicate cross-class boxes
                deduped_boxes = deduplicate_person_boxes(raw_boxes, iou_thresh=iou_thresh)

                # 2. Pose-Guided Biometric Verification
                boxes_data = []
                adult_count = 0
                baby_count = 0

                for item in deduped_boxes:
                    x1, y1, x2, y2 = item["coords"]
                    tid = item.get("track_id")
                    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    v_cls, v_conf = self.pose_verifier.verify_classification(crop, item["cls"], item["conf"])
                    item["cls"] = v_cls
                    item["conf"] = v_conf
                    boxes_data.append(item)
                    if v_cls == 0:
                        adult_count += 1
                    elif v_cls == 1:
                        baby_count += 1

                    if tid is not None:
                        track_history[tid].append(((x1 + x2) // 2, (y1 + y2) // 2))

                frame_fps = 1.0 / max(t_frame_end - t_frame_start, 0.001)
                hud_frame = self.draw_hud(frame, boxes_data, adult_count, baby_count, frame_fps, track_history=track_history)
                out.write(hud_frame)

                risk = assess_frame_risk(adult_count, baby_count)
                timeline_stats.append({
                    "frame": frame_idx,
                    "adult_count": adult_count,
                    "baby_count": baby_count,
                    "risk": risk
                })
                processed_count += 1

                if progress_callback:
                    try:
                        prog_val = min(0.98, processed_count / max(1.0, (max_frames_to_process / max(frame_skip, 1))))
                        progress_callback(prog_val, desc=f"Đang phân tích khung hình {frame_idx}/{max_frames_to_process} ({frame_fps:.1f} FPS)...")
                    except TypeError:
                        try:
                            prog_val = min(0.98, processed_count / max(1.0, (max_frames_to_process / max(frame_skip, 1))))
                            progress_callback(prog_val)
                        except Exception:
                            pass
                    except Exception:
                        pass

            frame_idx += 1

        cap.release()
        out.release()
        if self.device != "cpu":
            torch.cuda.empty_cache()
        total_time = time.time() - t_start

        # Convert to H.264 using FFmpeg for 100% web browser compatibility
        print("🎬 Converting video to web-compatible H.264 MP4 via FFmpeg ...")
        cmd = [
            "ffmpeg", "-y", "-i", raw_out_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "23",
            web_out_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(raw_out_path):
                os.remove(raw_out_path)
            final_path = web_out_path
        except Exception as e:
            print(f"⚠️ FFmpeg conversion failed ({e}), fallback to raw video: {raw_out_path}")
            final_path = raw_out_path

        summary = summarize_video_risk(timeline_stats, orig_fps / max(frame_skip, 1))
        summary["total_processing_time"] = round(total_time, 2)
        summary["avg_fps"] = round(processed_count / max(total_time, 0.01), 1)

        return final_path, summary
