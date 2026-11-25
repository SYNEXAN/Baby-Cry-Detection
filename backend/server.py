"""
Backend Server: FastAPI REST & SSE Streaming Service
Provides decoupled API endpoints for Infant Safety Monitoring:
- Video upload & Range-supported streaming
- Test video examples inventory
- Server-Sent Events (SSE) real-time multi-modal streaming
- Static file serving for Frontend
"""

import os
import sys
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from engine.unified_detector import UnifiedMultimodalDetector

# Create upload directory
TEMP_UPLOADS = ROOT_DIR / "temp_uploads"
TEMP_UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="He Thong Giam Sat An Toan Tre Em API",
    description="Decoupled Multi-Modal Infant Safety & Cry Analysis API",
    version="1.0.0"
)

# Enable CORS for future remote API server deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
DETECTOR: Optional[UnifiedMultimodalDetector] = None

def get_detector() -> UnifiedMultimodalDetector:
    global DETECTOR
    if DETECTOR is None:
        print("[BACKEND] Initializing UnifiedMultimodalDetector...")
        DETECTOR = UnifiedMultimodalDetector(
            vision_model_path="kaggle_results/v4_clean_balanced/best.pt",
            audio_weights_path="models/best_eletriby_4class_audio_classifier.pt"
        )
        print("[BACKEND] UnifiedMultimodalDetector ready.")
    return DETECTOR


@app.get("/api/examples")
async def get_test_examples():
    """
    Returns inventory of available sample videos in VIDEO_TEST/ with thumbnails.
    """
    video_dir = ROOT_DIR / "VIDEO_TEST"
    candidate_files = [
        "294613747-rivne-ukraine-june-12-2024-chi.mp4",
        "280167556-female-teacher-checking-kids-d.mp4",
        "050870650-preschool-children-are-early-c.mp4",
        "sd_watermarked_preview.mp4"
    ]
    examples = []
    for f in candidate_files:
        full_p = video_dir / f
        if full_p.exists():
            thumb_name = os.path.splitext(f)[0] + ".jpg"
            examples.append({
                "filename": f,
                "path": f"VIDEO_TEST/{f}",
                "thumbnail_url": f"thumbnails/{thumb_name}"
            })
    return JSONResponse(content={"examples": examples})


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Handles video upload and saves to local temporary storage.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Ten tap tin khong hop le.")

    safe_name = os.path.basename(file.filename).replace(" ", "_")
    target_path = TEMP_UPLOADS / safe_name

    contents = await file.read()
    with open(target_path, "wb") as f:
        f.write(contents)

    return JSONResponse(content={
        "status": "success",
        "filename": safe_name,
        "path": f"temp_uploads/{safe_name}",
        "size_mb": round(len(contents) / (1024 * 1024), 2)
    })


@app.get("/api/video")
async def serve_video(path: str = Query(...)):
    """
    Streams video with standard HTTP Range support for smooth HTML5 video playback.
    """
    clean_path = (ROOT_DIR / path).resolve()
    # Security check: must reside inside project directory
    if not str(clean_path).startswith(str(ROOT_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Khong co quyen truy cap tap tin ngoai thu muc.")

    if not clean_path.exists():
        raise HTTPException(status_code=404, detail="Tap tin video khong ton tai.")

    return FileResponse(clean_path, media_type="video/mp4")


@app.get("/api/analyze/stream")
async def stream_analysis(request: Request, video_path: str = Query(...)):
    """
    Server-Sent Events (SSE) endpoint:
    Streams real-time vision frames (base64 JPEG) and dynamic sliding-window audio metrics.
    """
    clean_path = (ROOT_DIR / video_path).resolve()
    if not clean_path.exists():
        raise HTTPException(status_code=404, detail="Video khong ton tai.")

    detector = get_detector()

    async def sse_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        
        # Generator wrapper to yield in non-blocking fashion
        def generate_steps():
            return detector.stream_video(
                str(clean_path),
                conf_thresh=0.35,
                iou_thresh=0.45,
                imgsz=960,
                frame_skip=4
            )

        iterator = iter(generate_steps())

        while True:
            # Check client disconnect
            if await request.is_disconnected():
                print("[BACKEND] Client disconnected, terminating stream.")
                break

            # Run step in executor to avoid blocking event loop
            try:
                step = await loop.run_in_executor(None, next, iterator, None)
            except Exception as e:
                print(f"[BACKEND] Error in stream step: {e}")
                break

            if step is None:
                # Video processing finished
                done_payload = json.dumps({"event": "done"})
                yield f"data: {done_payload}\n\n"
                break

            # Encode frame to JPEG Base64
            frame_bgr = cv2.cvtColor(step["frame_rgb"], cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                continue

            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            # Extract metrics
            fusion = step["fusion"]
            vad = step["vad_result"]
            cry = step["cry_classification"]

            pct_done = min(100, int((step["frame_idx"] / max(1, step["total_frames"])) * 100))
            badge_title = f"[{fusion['badge']}] {fusion['title']} ({pct_done}%)"
            badge_desc = fusion["description"]

            # Metric 1: Vision Monitoring (Live)
            vision_text = (
                f"Frame {step['frame_idx']}/{step['total_frames']} | "
                f"Tre: {step['baby_count']} | Nguoi lon: {step['adult_count']} | "
                f"Mot minh: {step['alone_duration_sec']:.1f}s ({step['unattended_pct']:.0f}%)"
            )

            # Metric 2: Audio Cry Detection (Dynamic)
            if vad["is_cry"]:
                audio_text = "Phat hien am thanh tieng khoc"
            elif vad.get("status") == "AMBIENT_NOISE":
                audio_text = "Tieng on moi truong / Tieng quat (Khong phai tieng khoc)"
            elif vad.get("status") == "SILENCE":
                audio_text = "Yen tinh / Khong co tieng khoc"
            else:
                audio_text = "Video khong co luong am thanh"

            # Metric 3: Cry Cause Diagnosis (Dynamic)
            if vad["is_cry"]:
                cause_text = f"{cry['vietnamese_cause']} ({cry['confidence']*100:.1f}%)"
            else:
                cause_text = vad.get("label", "Yen tinh")

            payload = {
                "event": "step",
                "frame_b64": frame_b64,
                "frame_idx": step["frame_idx"],
                "total_frames": step["total_frames"],
                "pct_done": pct_done,
                "current_time_sec": round(step.get("current_time_sec", 0.0), 2),
                "badge": fusion["badge"],
                "color": fusion["color"],
                "title": badge_title,
                "description": badge_desc,
                "vision_text": vision_text,
                "audio_text": audio_text,
                "cause_text": cause_text,
                "probabilities": cry.get("probabilities", {
                    "Burping (O hoi)": 0.0,
                    "Needs (Nhu cau)": 0.0,
                    "Physical Pain (Dau bung)": 0.0,
                    "Scared (Hoang so)": 0.0
                })
            }

            yield f"data: {json.dumps(payload)}\n\n"
            # Small yield pause to let client render smoothly
            await asyncio.sleep(0.01)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Mount Frontend static files
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
