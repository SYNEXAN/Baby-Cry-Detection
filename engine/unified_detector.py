"""
Module: engine/unified_detector.py
Purpose: Unified Multi-Modal Pipeline for Baby Cry Reason Classification
and Visual Infant Unattended Risk Detection.

Components:
1. CryVADGate: Acoustic DSP Cry Activity Detector (separates real cry from ambient noise / silence).
2. UnifiedVisionDetector: Object detector for Adult & Baby/Child based on YOLO11-m (version 7 fixed).
3. UnifiedMultimodalDetector: Fuses Vision + Audio streams into real-time safety metrics and clinical advice.
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
import torch
import librosa
from ultralytics import YOLO

from engine.audio_detector import AudioCryClassifier, CLASS_VIETNAMESE, PARENT_ADVICE

# Detection colors (BGR for OpenCV)
COLOR_ADULT = (163, 255, 0)      # Bright Green-Cyan for adult
COLOR_BABY = (50, 50, 255)       # Coral-Red for baby_child
COLOR_HUD_BG = (24, 28, 36)      # Dark slate dashboard background
COLOR_ALERT_BG = (35, 35, 215)   # Red alert header background
COLOR_SAFE_BG = (30, 140, 50)    # Green status header background


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
    Filters redundant overlapping bounding boxes on the same physical person.
    Prioritizes higher confidence detection when IoU > iou_thresh.
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


class CryVADGate:
    """
    Stage 1: Cry Voice Activity Detection (Cry-VAD) Gate.
    Uses Digital Signal Processing (DSP) to discriminate real infant cries
    from ambient room noise (fans, air conditioning) and quiet periods.
    """
    def __init__(
        self,
        sr: int = 16000,
        rms_silence_thresh: float = 0.012,
        flatness_noise_thresh: float = 0.22,
        min_cry_energy_ratio: float = 0.20
    ):
        self.sr = sr
        self.rms_silence_thresh = rms_silence_thresh
        self.flatness_noise_thresh = flatness_noise_thresh
        self.min_cry_energy_ratio = min_cry_energy_ratio

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes audio signal and returns cry detection status and acoustic characteristics.
        """
        if len(audio) == 0:
            return {
                "is_cry": False,
                "status": "SILENCE",
                "label": "Yen tinh / Khong co tin hieu",
                "rms": 0.0,
                "flatness": 1.0,
                "cry_energy_ratio": 0.0
            }

        # 1. Root Mean Square (RMS) energy
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < self.rms_silence_thresh:
            return {
                "is_cry": False,
                "status": "SILENCE",
                "label": "Yen tinh / Khong co tieng khoc",
                "rms": rms,
                "flatness": 1.0,
                "cry_energy_ratio": 0.0
            }

        # 2. Spectral Flatness: White/fan noise has high flatness (> 0.22)
        # Infant cries have prominent harmonics with low flatness (< 0.18)
        flatness_series = librosa.feature.spectral_flatness(y=audio, n_fft=1024, hop_length=512)
        mean_flatness = float(np.mean(flatness_series))

        # 3. Frequency Band Energy Ratio: Cry resonance is concentrated in 800-3500 Hz
        stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=1024)

        cry_band_mask = (freqs >= 800) & (freqs <= 3500)
        low_noise_mask = (freqs < 250)

        total_power = np.sum(stft**2) + 1e-9
        cry_band_power = np.sum(stft[cry_band_mask, :]**2)
        low_noise_power = np.sum(stft[low_noise_mask, :]**2)

        cry_energy_ratio = float(cry_band_power / total_power)
        low_noise_ratio = float(low_noise_power / total_power)

        # Decision Logic:
        # 1. Low frequency noise dominates (> 40% power)
        # 2. Or sound lacks vocal harmonic structure (flatness > 0.15)
        # 3. Or acoustic energy in infant cry resonance band (800-3500 Hz) is deficient (< 30%)
        if low_noise_ratio > 0.40 or mean_flatness > 0.15 or cry_energy_ratio < 0.30:
            return {
                "is_cry": False,
                "status": "AMBIENT_NOISE",
                "label": "Tieng on moi truong / Tieng quat (Khong phai tieng khoc)",
                "rms": rms,
                "flatness": mean_flatness,
                "cry_energy_ratio": cry_energy_ratio
            }

        # Cry detected
        return {
            "is_cry": True,
            "status": "CRY_DETECTED",
            "label": "Phat hien am thanh tieng khoc",
            "rms": rms,
            "flatness": mean_flatness,
            "cry_energy_ratio": cry_energy_ratio
        }


class UnifiedVisionDetector:
    """
    Vision-based human detector using fine-tuned YOLO11-m V4 (v4_clean_balanced).
    Classes:
      0: adults
      1: kids
    """
    def __init__(self, model_path: str = "kaggle_results/v4_clean_balanced/best.pt"):
        if not os.path.exists(model_path):
            # Fallback path if needed
            alt_path = "kaggle_results/version7_fixed/best.pt"
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"Vision model weights not found at: {model_path}")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        if self.device != "cpu":
            self.model.to(self.device)
            self.model.model.half()

        self.names = self.model.names
        self.cls_adult = 0
        self.cls_baby = 1

    def detect_frame(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        imgsz: int = 960
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs object detection on a single video frame and draws clean annotations.
        Uses fixed V4 inference parameters (imgsz=960, conf=0.35, iou=0.45, agnostic_nms=True).
        """
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            imgsz=imgsz,
            conf=conf_thresh,
            iou=iou_thresh,
            agnostic_nms=True,
            verbose=False
        )[0]

        boxes_data = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for b, c, cl in zip(boxes, confs, classes):
                boxes_data.append({
                    "coords": (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                    "conf": float(c),
                    "cls": int(cl)
                })

        # Apply deduplication across overlapping detections
        filtered_boxes = deduplicate_person_boxes(boxes_data, iou_thresh=iou_thresh)

        adult_count = sum(1 for b in filtered_boxes if b["cls"] == self.cls_adult)
        baby_count = sum(1 for b in filtered_boxes if b["cls"] == self.cls_baby)

        is_alone = (baby_count > 0 and adult_count == 0)

        # Draw HUD and Bounding Boxes
        canvas = frame.copy()

        # 1. Bounding Boxes
        for item in filtered_boxes:
            x1, y1, x2, y2 = item["coords"]
            cls_id = item["cls"]
            conf = item["conf"]
            label_name = "NGUOI LON" if cls_id == self.cls_adult else "TRE EM"
            color = COLOR_ADULT if cls_id == self.cls_adult else COLOR_BABY

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            tag = f"{label_name} {int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(canvas, tag, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Top Dashboard Header (Dark bar)
        header_h = 44
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

        txt_baby = f"TRE EM: {baby_count}"
        txt_adult = f"NGUOI LON: {adult_count}"
        status_txt = "CANH BAO: TRE O MOT MINH" if is_alone else ("AN TOAN: CO NGUOI LON" if baby_count > 0 else "PHONG TRONG")
        status_color = COLOR_BABY if is_alone else (COLOR_ADULT if baby_count > 0 else (200, 200, 200))

        cv2.putText(canvas, txt_baby, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_BABY, 2, cv2.LINE_AA)
        cv2.putText(canvas, txt_adult, (160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ADULT, 2, cv2.LINE_AA)
        cv2.putText(canvas, status_txt, (340, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # 3. Bottom Alert Banner if Child is Alone
        if is_alone:
            banner_h = 36
            bot_overlay = canvas.copy()
            cv2.rectangle(bot_overlay, (0, h - banner_h), (w, h), COLOR_ALERT_BG, -1)
            cv2.addWeighted(bot_overlay, 0.85, canvas, 0.15, 0, canvas)

            alert_msg = "CANH BAO: PHAT HIEN TRE DANG O MOT MINH"
            (atw, ath), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(canvas, alert_msg, ((w - atw) // 2, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas, {
            "adult_count": adult_count,
            "baby_count": baby_count,
            "is_alone": is_alone,
            "boxes": filtered_boxes
        }


class UnifiedMultimodalDetector:
    """
    Unified Multi-Modal Pipeline:
    - Audio track extraction and DSP Cry-VAD Gate.
    - SOTA 4-class ResNet18 cry reason inference if cry is detected.
    - YOLO11-m vision tracking for baby and adult unattended risk.
    - Multi-modal risk fusion and parental recommendation engine.
    """
    def __init__(
        self,
        vision_model_path: str = "kaggle_results/v4_clean_balanced/best.pt",
        audio_weights_path: str = "models/best_eletriby_4class_audio_classifier.pt"
    ):
        print(f"Loading Unified Multi-Modal Detector...")
        self.vision_detector = UnifiedVisionDetector(model_path=vision_model_path)
        self.audio_classifier = AudioCryClassifier(weights_path=audio_weights_path)
        self.vad_gate = CryVADGate(sr=16000)
        print(f"Unified Multi-Modal Detector loaded successfully.")

    def extract_audio(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extracts mono 16kHz audio stream from video using FFmpeg.
        Returns None if video has no audio track.
        """
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_wav_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0 or not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) < 1000:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass
            return None

        try:
            audio, _ = librosa.load(temp_wav_path, sr=16000, mono=True)
            return audio
        except Exception:
            return None
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass

    def process_video(
        self,
        video_path: str,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45
    ) -> Dict[str, Any]:
        """
        Full end-to-end multi-modal processing for input video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Step 1: Audio Processing & Cry Gate
        audio_signal = self.extract_audio(video_path)
        audio_present = audio_signal is not None and len(audio_signal) > 0

        vad_result = None
        cry_classification = None

        if audio_present:
            vad_result = self.vad_gate.analyze(audio_signal)
            if vad_result["is_cry"]:
                cry_classification = self.audio_classifier.predict(audio_signal)
            else:
                cry_classification = {
                    "primary_cause": "none",
                    "vietnamese_cause": vad_result["label"],
                    "confidence": 0.0,
                    "probabilities": {
                        "Burping (O hoi)": 0.0,
                        "Needs (Nhu cau)": 0.0,
                        "Physical Pain (Dau bung)": 0.0,
                        "Scared (Hoang so)": 0.0
                    },
                    "advice": "He thong xac nhan khong phai tieng khoc cua tre (am thanh moi truong / tieng on). Khong can can thiep y te.",
                    "sample_rate": 16000,
                    "duration_sec": 5.0,
                    "model_name": self.audio_classifier.model_name
                }
        else:
            vad_result = {
                "is_cry": False,
                "status": "NO_AUDIO",
                "label": "Video khong co luong am thanh",
                "rms": 0.0,
                "flatness": 0.0,
                "cry_energy_ratio": 0.0
            }
            cry_classification = {
                "primary_cause": "none",
                "vietnamese_cause": "Khong co du lieu am thanh",
                "confidence": 0.0,
                "probabilities": {
                    "Burping (O hoi)": 0.0,
                    "Needs (Nhu cau)": 0.0,
                    "Physical Pain (Dau bung)": 0.0,
                    "Scared (Hoang so)": 0.0
                },
                "advice": "Video khong co tap am thanh de phan tich nguyen nhan tieng khoc.",
                "sample_rate": 16000,
                "duration_sec": 5.0,
                "model_name": self.audio_classifier.model_name
            }

        # Step 2: Visual Frame-by-Frame Detection
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Downscale wide (>1280px, e.g. 4K) videos before inference like V4 pipeline
        out_w, out_h = width, height
        if width > 1280:
            scale = 1280 / width
            out_w = 1280
            out_h = int(height * scale)
            out_h = out_h - (out_h % 2)

        temp_out_raw = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
        temp_out_raw_path = temp_out_raw.name
        temp_out_raw.close()

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_writer = cv2.VideoWriter(temp_out_raw_path, fourcc, fps, (out_w, out_h))

        frame_count = 0
        alone_frames = 0
        accompanied_frames = 0
        empty_frames = 0
        max_baby = 0
        max_adult = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if out_w != width:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            annotated_frame, metrics = self.vision_detector.detect_frame(
                frame, conf_thresh=conf_thresh, iou_thresh=iou_thresh, imgsz=960
            )
            out_writer.write(annotated_frame)

            if metrics["is_alone"]:
                alone_frames += 1
            elif metrics["baby_count"] > 0:
                accompanied_frames += 1
            else:
                empty_frames += 1

            max_baby = max(max_baby, metrics["baby_count"])
            max_adult = max(max_adult, metrics["adult_count"])

        cap.release()
        out_writer.release()

        # Step 3: Re-encode video using FFmpeg to H.264 MP4 for browser playback
        final_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        final_mp4_path = final_mp4.name
        final_mp4.close()

        # Combine video with original audio if available
        if audio_present:
            encode_cmd = [
                "ffmpeg", "-y", "-i", temp_out_raw_path, "-i", video_path,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                final_mp4_path
            ]
        else:
            encode_cmd = [
                "ffmpeg", "-y", "-i", temp_out_raw_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
                "-movflags", "+faststart",
                final_mp4_path
            ]

        subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_out_raw_path):
            try:
                os.remove(temp_out_raw_path)
            except OSError:
                pass

        # Step 4: Multi-Modal Risk Fusion
        unattended_pct = (alone_frames / max(1, frame_count)) * 100.0
        alone_duration_sec = alone_frames / max(1.0, fps)
        
        # Vision danger evaluation:
        # 1. Baby present and no adult ever seen -> Unattended Danger
        # 2. Or baby left alone for an extended period (> 25% of time and > 2.5s)
        if max_baby > 0 and max_adult == 0:
            is_vision_danger = True
        elif max_baby > 0 and unattended_pct >= 25.0 and alone_duration_sec >= 2.5:
            is_vision_danger = True
        else:
            is_vision_danger = False
            
        is_crying = vad_result["is_cry"]

        cause_str = cry_classification["vietnamese_cause"]
        primary_cls = cry_classification["primary_cause"]

        if is_vision_danger and is_crying:
            risk_badge = "CRITICAL"
            risk_color = "#DC2626"
            risk_title = "NGUY HIEM: TRE O MOT MINH VA DANG KHOC"
            risk_desc = (
                f"Phat hien tre bi bo lai mot minh ({alone_duration_sec:.1f}s) va dang phat ra tieng khoc "
                f"do: {cause_str}. Canh bao can thiep ngay lap tuc de bao ve an toan cho tre."
            )
        elif is_vision_danger and not is_crying:
            risk_badge = "WARNING"
            risk_color = "#D97706"
            risk_title = "CANH BAO: TRE O MOT MINH (CHUA KHOC)"
            risk_desc = (
                f"Phat hien tre dang o mot minh trong phong ({alone_duration_sec:.1f}s) khong co nguoi lon trong coi. "
                f"Hien tai chua phat hien tieng khoc, tuy nhien can nguoi lon den quan sat ngay de phong tai nan."
            )
        elif not is_vision_danger and is_crying and max_baby > 0:
            risk_badge = "ATTENTION"
            risk_color = "#2563EB"
            risk_title = "CHU Y: TRE DANG KHOC (CO NGUOI LON TRONG COI)"
            risk_desc = (
                f"Co nguoi lon trong phong nhung tre dang khoc vi: {cause_str}. "
                f"Phu huynh vui long tham khao huong dan xu ly ben duoi de cham soc be kip thoi."
            )
        elif max_baby > 0:
            risk_badge = "SAFE"
            risk_color = "#059669"
            risk_title = "AN TOAN: CO NGUOI LON CHAM SOC"
            risk_desc = "Trang thai binh thuong: Co nguoi lon trong nom va khong phat hien dau hieu tre khoc."
        else:
            risk_badge = "EMPTY"
            risk_color = "#64748B"
            risk_title = "PHONG TRONG: KHONG PHAT HIEN TRE"
            risk_desc = "Khong phat hien tre nho trong khung hinh camera."

        return {
            "output_video_path": final_mp4_path,
            "total_frames": frame_count,
            "duration_sec": frame_count / max(1.0, fps),
            "fps": fps,
            "alone_frames": alone_frames,
            "alone_duration_sec": alone_duration_sec,
            "unattended_pct": unattended_pct,
            "max_baby": max_baby,
            "max_adult": max_adult,
            "vad_result": vad_result,
            "cry_classification": cry_classification,
            "fusion": {
                "badge": risk_badge,
                "color": risk_color,
                "title": risk_title,
                "description": risk_desc,
                "advice": cry_classification.get("advice", "")
            }
        }

    def analyze_audio_window(
        self,
        audio_signal: Optional[np.ndarray],
        current_time_sec: float,
        window_sec: float = 2.5
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Continuously analyzes a sliding window of audio corresponding to current video timestamp.
        Window: [max(0, current_time_sec - window_sec) : current_time_sec]
        Provides real-time dynamic cry detection as video progresses.
        """
        if audio_signal is None or len(audio_signal) == 0:
            vad_res = {
                "is_cry": False,
                "status": "NO_AUDIO",
                "label": "Video khong co luong am thanh",
                "rms": 0.0,
                "flatness": 0.0,
                "cry_energy_ratio": 0.0
            }
            cry_cls = {
                "primary_cause": "none",
                "vietnamese_cause": "Khong co du lieu am thanh",
                "confidence": 0.0,
                "probabilities": {
                    "Burping (O hoi)": 0.0,
                    "Needs (Nhu cau)": 0.0,
                    "Physical Pain (Dau bung)": 0.0,
                    "Scared (Hoang so)": 0.0
                },
                "advice": "",
                "sample_rate": 16000,
                "duration_sec": 5.0,
                "model_name": self.audio_classifier.model_name
            }
            return vad_res, cry_cls

        sr = 16000
        end_sample = int(current_time_sec * sr)
        start_sample = max(0, int((current_time_sec - window_sec) * sr))

        if end_sample <= 0:
            end_sample = min(len(audio_signal), int(window_sec * sr))
            start_sample = 0

        audio_slice = audio_signal[start_sample:end_sample]
        if len(audio_slice) < int(0.5 * sr):
            if len(audio_signal) >= int(0.5 * sr):
                audio_slice = audio_signal[:int(window_sec * sr)]
            else:
                audio_slice = np.pad(audio_slice, (0, int(0.5 * sr) - len(audio_slice)))

        vad_result = self.vad_gate.analyze(audio_slice)
        if vad_result["is_cry"]:
            cry_classification = self.audio_classifier.predict(audio_slice)
        else:
            cry_classification = {
                "primary_cause": "none",
                "vietnamese_cause": vad_result["label"],
                "confidence": 0.0,
                "probabilities": {
                    "Burping (O hoi)": 0.0,
                    "Needs (Nhu cau)": 0.0,
                    "Physical Pain (Dau bung)": 0.0,
                    "Scared (Hoang so)": 0.0
                },
                "advice": "",
                "sample_rate": 16000,
                "duration_sec": 5.0,
                "model_name": self.audio_classifier.model_name
            }

        return vad_result, cry_classification

    def stream_video(
        self,
        video_path: str,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        imgsz: int = 960,
        frame_skip: int = 4
    ):
        """
        Real-time streaming generator: yields annotated RGB frames and live safety metrics
        immediately frame-by-frame. Audio is continuously monitored via sliding window
        in background at each step along the video timeline.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Pre-extract full audio track once (< 0.2s)
        audio_signal = self.extract_audio(video_path)

        # Video Frame-by-Frame Streaming with Frame Skip
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        out_w, out_h = width, height
        if width > 1280:
            scale = 1280 / width
            out_w = 1280
            out_h = int(height * scale)
            out_h = out_h - (out_h % 2)

        frame_idx = 0
        processed_count = 0
        alone_processed = 0
        max_baby = 0
        max_adult = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    current_time_sec = frame_idx / max(1.0, fps)
                    
                    # 1. Continuous Audio Sliding Window Analysis (Runs in background for current timestamp)
                    vad_result, cry_classification = self.analyze_audio_window(
                        audio_signal=audio_signal,
                        current_time_sec=current_time_sec,
                        window_sec=2.5
                    )
                    is_crying = vad_result["is_cry"]
                    cause_str = cry_classification["vietnamese_cause"]

                    # 2. Vision Detection
                    if out_w != width:
                        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                    annotated_bgr, metrics = self.vision_detector.detect_frame(
                        frame, conf_thresh=conf_thresh, iou_thresh=iou_thresh, imgsz=imgsz
                    )
                    processed_count += 1
                    if metrics["is_alone"]:
                        alone_processed += 1

                    max_baby = max(max_baby, metrics["baby_count"])
                    max_adult = max(max_adult, metrics["adult_count"])

                    unattended_pct = (alone_processed / max(1, processed_count)) * 100.0
                    alone_duration_sec = (alone_processed * frame_skip) / max(1.0, fps)
                    cur_alone = metrics["is_alone"]
                    cur_baby = metrics["baby_count"]
                    cur_adult = metrics["adult_count"]

                    # 3. Dynamic Multi-Modal Risk Fusion at timestamp t
                    if cur_alone and is_crying:
                        risk_badge = "CRITICAL"
                        risk_color = "#DC2626"
                        risk_title = "NGUY HIEM: TRE O MOT MINH VA DANG KHOC"
                        risk_desc = f"Tre dang o mot minh ({alone_duration_sec:.1f}s) va khoc do: {cause_str}."
                    elif cur_alone and not is_crying:
                        risk_badge = "WARNING"
                        risk_color = "#D97706"
                        risk_title = "CANH BAO: TRE O MOT MINH (CHUA KHOC)"
                        risk_desc = f"Tre dang o mot minh ({alone_duration_sec:.1f}s) khong co nguoi lon ben canh."
                    elif not cur_alone and is_crying and cur_baby > 0:
                        risk_badge = "ATTENTION"
                        risk_color = "#2563EB"
                        risk_title = "CHU Y: TRE DANG KHOC (CO NGUOI LON TRONG COI)"
                        risk_desc = f"Co nguoi lon ben canh, tre dang khoc do: {cause_str}."
                    elif cur_baby > 0 or max_baby > 0:
                        risk_badge = "SAFE"
                        risk_color = "#059669"
                        risk_title = "AN TOAN: CO NGUOI LON CHAM SOC"
                        risk_desc = "Trang thai binh thuong: Co nguoi lon trong nom."
                    else:
                        risk_badge = "EMPTY"
                        risk_color = "#64748B"
                        risk_title = "PHONG TRONG: KHONG PHAT HIEN TRE"
                        risk_desc = "Khong phat hien tre nho trong khung hinh hien tai."

                    frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    yield {
                        "frame_rgb": frame_rgb,
                        "frame_idx": frame_idx + 1,
                        "total_frames": total_frames,
                        "current_time_sec": current_time_sec,
                        "alone_duration_sec": alone_duration_sec,
                        "unattended_pct": unattended_pct,
                        "baby_count": cur_baby,
                        "adult_count": cur_adult,
                        "max_baby": max_baby,
                        "max_adult": max_adult,
                        "vad_result": vad_result,
                        "cry_classification": cry_classification,
                        "fusion": {
                            "badge": risk_badge,
                            "color": risk_color,
                            "title": risk_title,
                            "description": risk_desc
                        }
                    }

                frame_idx += 1
        finally:
            cap.release()
