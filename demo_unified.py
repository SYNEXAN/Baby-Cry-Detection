"""
Module: demo_unified.py
Purpose: Giao Dien Demo Giam Sat An Toan Tre Em & Nhan Dien Tieng Khoc.

Thiet ke toi gian, truc quan:
- Dau vao: 1 Video duy nhat (chua hinh anh va am thanh).
- Nguong co dinh toi uu trong backend, khong co thanh truot cau hinh.
- An toan bo ten mo hinh ky thuat, hien thi thong tin nghiep vu ro rang.
- Danh sach video mau lay truc tiep tu thu muc VIDEO_TEST (co kem luong am thanh).
"""

import os
import sys
import asyncio
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, AsyncIterator

# Setup path and local Gradio temp directory BEFORE importing gradio
# Prevents Windows AppData\Local\Temp file locking PermissionError [Errno 13]
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

TEMP_GRADIO_DIR = ROOT_DIR / "temp_gradio"
TEMP_GRADIO_DIR.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(TEMP_GRADIO_DIR)

import anyio
import gradio as gr
import gradio.processing_utils as pu
from starlette.responses import FileResponse

# Disable blocking synchronous ffmpeg transcoding on upload
pu.video_is_playable = lambda video_filepath: True
pu.convert_video_to_playable_mp4 = lambda video, **kwargs: video

# Windows Resilient FileResponse: retries when files are transiently locked by OS/Defender
_original_starlette_call = FileResponse.__call__

@asynccontextmanager
async def _resilient_starlette_open_file(self) -> AsyncIterator[anyio.AsyncFile[bytes]]:
    file = None
    for attempt in range(50):
        try:
            file = await anyio.open_file(self.path, mode="rb")
            break
        except (PermissionError, OSError) as err:
            if attempt == 49:
                print(f"[ERROR] File locked permanently after 50 attempts: {self.path} ({err})")
                raise
            await asyncio.sleep(0.1)
    try:
        yield file
    finally:
        if file is not None:
            with anyio.CancelScope(shield=True):
                await file.aclose()

async def _resilient_starlette_call(self, scope, receive, send) -> None:
    if self.stat_result is None:
        for attempt in range(50):
            try:
                stat_result = await anyio.to_thread.run_sync(os.stat, self.path)
                self.stat_result = stat_result
                self.set_stat_headers(stat_result)
                break
            except (PermissionError, OSError):
                if attempt == 49:
                    raise
                await asyncio.sleep(0.1)
    return await _original_starlette_call(self, scope, receive, send)

FileResponse._open_file = _resilient_starlette_open_file
FileResponse.__call__ = _resilient_starlette_call

from engine.unified_detector import UnifiedMultimodalDetector

# Global detector instance
DETECTOR: Optional[UnifiedMultimodalDetector] = None

def get_detector() -> UnifiedMultimodalDetector:
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = UnifiedMultimodalDetector(
            vision_model_path="kaggle_results/v4_clean_balanced/best.pt",
            audio_weights_path="models/best_eletriby_4class_audio_classifier.pt"
        )
    return DETECTOR


CUSTOM_CSS = """
/* Responsive full-width container */
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0F172A;
    background-color: #F8FAFC;
    width: 96% !important;
    max-width: 96% !important;
    margin: 0 auto !important;
    padding: 10px 16px !important;
}

/* Header */
.header-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.header-title {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #0F172A;
    margin: 0 0 4px 0;
}
.header-sub {
    font-size: 13px;
    color: #64748B;
    margin: 0;
}

/* Side-by-side horizontal video layout */
.video-grid-row {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 16px !important;
    align-items: stretch !important;
    margin-bottom: 12px !important;
}
.video-grid-row > div {
    min-width: 0 !important;
}
.action-row {
    display: flex !important;
    justify-content: center !important;
    gap: 12px !important;
    margin-bottom: 14px !important;
}

/* Primary Action Button */
button.primary-btn {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 10px 24px !important;
    min-width: 200px !important;
}
button.secondary-btn {
    background-color: #E2E8F0 !important;
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 10px 20px !important;
    min-width: 120px !important;
}
"""

# Fixed backend thresholds (V4 standard)
CONF_THRESH = 0.35
IOU_THRESH = 0.45
IMGSZ = 960
FRAME_SKIP = 4

def analyze_video(
    video_path: Optional[str]
):
    """
    Streams multi-modal infant safety monitoring frame-by-frame in real time.
    """
    if not video_path or not os.path.exists(video_path):
        empty_banner = """
        <div style="background:#F1F5F9; color:#475569; padding:12px 16px; border-radius:6px; font-weight:600;">
        CHUA CO VIDEO: Vui long tai len video hoac chon video mau tu VIDEO_TEST ben duoi.
        </div>
        """
        yield None, empty_banner, "-", "-", "-", {}
        return

    detector = get_detector()
    for step in detector.stream_video(
        video_path=video_path,
        conf_thresh=CONF_THRESH,
        iou_thresh=IOU_THRESH,
        imgsz=IMGSZ,
        frame_skip=FRAME_SKIP
    ):
        fusion = step["fusion"]
        vad = step["vad_result"]
        cry = step["cry_classification"]

        # Status Banner HTML
        bg_color = fusion["color"]
        pct_done = min(100, int((step["frame_idx"] / max(1, step["total_frames"])) * 100))
        badge_title = f"[{fusion['badge']}] {fusion['title']} ({pct_done}%)"
        badge_desc = fusion["description"]

        banner_html = f"""
        <div style="background-color: {bg_color}; color: #FFFFFF; padding: 14px 18px; border-radius: 6px; margin-bottom: 12px;">
            <div style="font-size: 15px; font-weight: 700; margin-bottom: 4px;">{badge_title}</div>
            <div style="font-size: 13px; opacity: 0.95;">{badge_desc}</div>
        </div>
        """

        # Metric 1: Vision Monitoring (Live)
        vision_text = (
            f"Frame {step['frame_idx']}/{step['total_frames']} | "
            f"Tre: {step['baby_count']} | Nguoi lon: {step['adult_count']} | "
            f"Mot minh: {step['alone_duration_sec']:.1f}s ({step['unattended_pct']:.0f}%)"
        )

        # Metric 2: Audio Cry Detection
        if vad["is_cry"]:
            audio_text = "Phat hien am thanh tieng khoc"
        elif vad["status"] == "AMBIENT_NOISE":
            audio_text = "Tieng on moi truong / Tieng quat (Khong phai tieng khoc)"
        elif vad["status"] == "SILENCE":
            audio_text = "Yen tinh / Khong co tieng khoc"
        else:
            audio_text = "Video khong co luong am thanh"

        # Metric 3: Cry Cause Diagnosis
        if vad["is_cry"]:
            cause_text = f"{cry['vietnamese_cause']} ({cry['confidence']*100:.1f}%)"
        else:
            cause_text = vad["label"]

        # Probability Distribution
        probs = cry.get("probabilities", {})

        yield (
            step["frame_rgb"],
            banner_html,
            vision_text,
            audio_text,
            cause_text,
            probs
        )


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="He Thong Giam Sat An Toan Tre Em", fill_width=True) as demo:
        # Header Box
        with gr.Row(elem_classes=["header-box"]):
            with gr.Column():
                gr.HTML("""
                <div class="header-title">HE THONG GIAM SAT AN TOAN TRE EM</div>
                <div class="header-sub">Phat hien tre bi bo lai mot minh va nhan dien nguyen nhan tieng khoc truc tiep tu video.</div>
                """)

        # Main Screen: Side-by-Side Layout (Strictly Horizontal)
        with gr.Row(elem_classes=["video-grid-row"], equal_height=True):
            input_video = gr.Video(
                label="Video Dau Vao (Hinh Anh & Am Thanh)",
                sources=["upload"],
                height=420
            )
            output_frame = gr.Image(
                label="Ket Qua Nhan Dien Truc Tiep (Tung Khung Hinh)",
                interactive=False,
                height=420
            )

        # Action Button Row (Centered)
        with gr.Row(elem_classes=["action-row"]):
            run_btn = gr.Button("Phan Tich Video", variant="primary", elem_classes=["primary-btn"])
            stop_btn = gr.Button("Dung", variant="secondary", elem_classes=["secondary-btn"])

        # Safety Status Banner
        status_banner = gr.HTML("""
        <div style="background:#F1F5F9; color:#475569; padding:12px 16px; border-radius:6px; font-weight:600;">
        TRANG THAI: Vui long chon video mau ben duoi hoac tai len video de phan tich.
        </div>
        """)

        # Metric KPI Cards
        with gr.Row():
            vision_status = gr.Textbox(
                label="Giam Sat Hinh Anh",
                value="-",
                interactive=False
            )
            audio_status = gr.Textbox(
                label="Nhan Dien Am Thanh",
                value="-",
                interactive=False
            )
            cause_status = gr.Textbox(
                label="Nguyen Nhan Tieng Khoc",
                value="-",
                interactive=False
            )

        # Probability Distribution
        with gr.Row():
            prob_label = gr.Label(
                num_top_classes=4,
                label="Xac Suat 4 Nhom Nguyen Nhan Tieng Khoc"
            )

        # Examples Section from VIDEO_TEST/
        gr.Markdown("### Video Mau Kiem Thu (Thu Muc VIDEO_TEST)")
        
        # Candidate test videos from VIDEO_TEST
        test_examples = [
            ["VIDEO_TEST/294613747-rivne-ukraine-june-12-2024-chi.mp4"],
            ["VIDEO_TEST/280167556-female-teacher-checking-kids-d.mp4"],
            ["VIDEO_TEST/050870650-preschool-children-are-early-c.mp4"],
            ["VIDEO_TEST/sd_watermarked_preview.mp4"]
        ]
        
        # Filter existing files
        valid_examples = [ex for ex in test_examples if os.path.exists(ex[0])]

        gr.Examples(
            examples=valid_examples,
            inputs=[input_video],
            cache_examples=False
        )

        # Event Binding (Real-time Streaming + Stop)
        run_event = run_btn.click(
            fn=analyze_video,
            inputs=[input_video],
            outputs=[output_frame, status_banner, vision_status, audio_status, cause_status, prob_label]
        )
        stop_btn.click(fn=None, cancels=[run_event])

    return demo


if __name__ == "__main__":
    app = create_demo()
    app.queue()
    print("Khoi chay Ung Dung Gradio Multi-Modal Demo tai http://127.0.0.1:7860 ...")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
        allowed_paths=[str(ROOT_DIR), str(TEMP_GRADIO_DIR), tempfile.gettempdir()]
    )
