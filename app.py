"""
SYNEXAN AI LABS - AI Baby & Child Safety Video Monitoring System
Hệ Thống AI Giám Sát Video & Phát Hiện Nguy Cơ Trẻ Em Bị Bỏ Lại Một Mình
Framework: Gradio 6
Vision Model: RT-DETR-L (Real-Time Detection Transformer Large, mAP@50 = 80.08%)
Risk Engine: Multi-Class Rule-Based Unattended Child Assessment
"""

import os
import sys
import time
import json
import shutil
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Generator, List
import cv2
import numpy as np
import pandas as pd
import torch
from gradio._vendor import aiofiles
import gradio as gr
from gradio import ranged_response
import gradio.processing_utils as pu

# ---------------------------------------------------------------------------
# Monkey-patch: Windows Asyncio Proactor WinError 10054 ConnectionReset Fix
# Prevents unhandled callback exceptions when clients disconnect / abort
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _silenced_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                if hasattr(self, '_sock') and self._sock is not None:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                self._called_connection_lost = True

        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_connection_lost
    except Exception as patch_e:
        print(f"Warning: Proactor patch not applied: {patch_e}")

# ---------------------------------------------------------------------------
# Monkey-patch: Disable blocking synchronous ffmpeg transcoding on upload
# OpenCV cv2.VideoCapture natively decodes all video codecs (H264, H265, AVI, MKV)
# at 100+ FPS without needing Gradio to freeze the event loop re-encoding.
# ---------------------------------------------------------------------------
pu.video_is_playable = lambda video_filepath: True
pu.convert_video_to_playable_mp4 = lambda video, **kwargs: video

# Ensure root directory in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Dedicated local directories to prevent Windows temp file lock and permission issues
TEMP_GRADIO_DIR = ROOT_DIR / "temp_gradio"
TEMP_GRADIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = ROOT_DIR / "uploaded_videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(TEMP_GRADIO_DIR)

# Auto-cleanup stale temporary frames from previous streaming runs
try:
    _clean_now = time.time()
    _purged = 0
    for _root, _dirs, _files in os.walk(TEMP_GRADIO_DIR, topdown=False):
        for _f in _files:
            if _f.endswith(".webp") or (_clean_now - os.path.getmtime(os.path.join(_root, _f)) > 3600):
                try:
                    os.remove(os.path.join(_root, _f))
                    _purged += 1
                except Exception:
                    pass
        for _d in _dirs:
            try:
                _dp = os.path.join(_root, _d)
                if not os.listdir(_dp):
                    os.rmdir(_dp)
            except Exception:
                pass
    if _purged > 0:
        print(f"🧹 Đã dọn dẹp {_purged} file tạm thời cũ trong temp_gradio.")
except Exception as _clean_err:
    pass

# ---------------------------------------------------------------------------
# Monkey-patch: Windows Resilient FileResponse & RangedFileResponse
# Fixes PermissionError: [Errno 13] Permission denied on large video uploads
# Automatically retries when files are temporarily locked by Windows Defender or OS cache
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager
from typing import AsyncIterator
import anyio
from starlette.responses import FileResponse

original_starlette_open_file = FileResponse._open_file
original_starlette_call = FileResponse.__call__

@asynccontextmanager
async def resilient_starlette_open_file(self) -> AsyncIterator[anyio.AsyncFile[bytes]]:
    file = None
    for attempt in range(50):
        try:
            file = await anyio.open_file(self.path, mode="rb")
            break
        except (PermissionError, OSError) as err:
            if attempt == 49:
                print(f"❌ File locked permanently after 50 attempts: {self.path} ({err})")
                raise
            await asyncio.sleep(0.1)
    try:
        yield file
    finally:
        if file is not None:
            with anyio.CancelScope(shield=True):
                await file.aclose()

async def resilient_starlette_call(self, scope, receive, send) -> None:
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
    return await original_starlette_call(self, scope, receive, send)

FileResponse._open_file = resilient_starlette_open_file
FileResponse.__call__ = resilient_starlette_call

# Also patch legacy ranged_response if present
try:
    original_ranged_call = ranged_response.RangedFileResponse.__call__

    async def resilient_ranged_call(self, scope, receive, send) -> None:
        if self.stat_result is None:
            for attempt in range(50):
                try:
                    self.stat_result = await ranged_response.aio_stat(self.path)
                    break
                except (PermissionError, OSError):
                    if attempt == 49:
                        raise
                    await asyncio.sleep(0.1)
                except FileNotFoundError as fnfe:
                    raise RuntimeError(f"File at path {self.path} does not exist.") from fnfe

        if self.range.start >= self.stat_result.st_size:
            raise ranged_response.HTTPException(
                status_code=416,
                headers={"Content-Range": f"bytes */{self.stat_result.st_size}"},
            )

        byte_range = self.range.clamp(0, self.stat_result.st_size - 1)
        self.set_range_headers(byte_range)

        file = None
        for attempt in range(50):
            try:
                file = await aiofiles.open(self.path, mode="rb")
                break
            except (PermissionError, OSError):
                if attempt == 49:
                    raise
                await asyncio.sleep(0.1)

        try:
            await file.seek(byte_range.start)
            await send(
                {
                    "type": "http.response.start",
                    "status": 206,
                    "headers": self.raw_headers,
                }
            )
            if self.send_header_only:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            else:
                remaining_bytes = len(byte_range)
                if not byte_range:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                    return

                while remaining_bytes > 0:
                    chunk_size = min(self.chunk_size, remaining_bytes)
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        raise RuntimeError("File ended before the requested range was read.")
                    remaining_bytes -= len(chunk)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": remaining_bytes > 0,
                        }
                    )
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, asyncio.CancelledError):
            pass
        finally:
            if file is not None:
                await file.close()

    ranged_response.RangedFileResponse.__call__ = resilient_ranged_call
except Exception as patch_e:
    pass

from engine.video_detector import VideoDetector, DEFAULT_MODEL_PATH
from engine.risk_assessment import RiskLevel
from engine.audio_detector import AudioCryClassifier

# Pre-load detector instance
print("Đang khởi tạo bộ giải mã VideoDetector và nạp trọng số RT-DETR-L...")
detector = VideoDetector(DEFAULT_MODEL_PATH)

# Pre-load audio cry detector
try:
    audio_detector = AudioCryClassifier()
    print(f"Đã nạp thành công bộ phân tích tiếng khóc: {audio_detector.model_name}")
except Exception as _audio_err:
    print(f"Chưa thể tải mô hình âm thanh: {_audio_err}")
    audio_detector = None

# Path definitions
DEMO_VIDEOS_DIR = ROOT_DIR / "demo_videos"
DEMO_IMAGES_DIR = ROOT_DIR / "demo_images"
CACHE_DIR = ROOT_DIR / "demo_cache"
CACHE_META_PATH = CACHE_DIR / "demo_metadata.json"
RESULTS_DIR = ROOT_DIR / "kaggle_results" / "version5" / "runs_rtdetr" / "baby_adult_rtdetr_l"

# Sample files
SAMPLE_VIDEOS = {
    "baby_alone": str(DEMO_VIDEOS_DIR / "sample_baby_alone.mp4"),
    "adult_caring": str(DEMO_VIDEOS_DIR / "sample_adult_caring.mp4"),
    "daycare": str(DEMO_VIDEOS_DIR / "sample_daycare_indoor.mp4"),
}

SAMPLE_IMAGES = {
    "baby_alone": str(DEMO_IMAGES_DIR / "sample_baby_alone.jpg"),
    "child_and_adult": str(DEMO_IMAGES_DIR / "sample_child_and_adult.jpg"),
    "adult_only": str(DEMO_IMAGES_DIR / "sample_adult_only.jpg"),
}

# Load pre-rendered demo cache
CACHED_DEMOS = {}
if CACHE_META_PATH.exists():
    try:
        with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
            CACHED_DEMOS = json.load(f)
            print(f"⚡ Đã nạp thành công bộ nhớ đệm cho {len(CACHED_DEMOS)} video mẫu demo!")
    except Exception as e:
        print(f"⚠️ Lỗi nạp cache demo: {e}")

# ---------------------------------------------------------
# Custom Styling: Modern Clean Light Theme
# ---------------------------------------------------------
CUSTOM_CSS = """
/* Toàn bộ nền và font chữ */
body, .gradio-container {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Thanh Header 1 dòng tối giản phong cách sáng */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-bottom: 16px;
    white-space: nowrap;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
.app-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
    margin: 0;
    white-space: nowrap;
}
.app-badges {
    display: flex;
    gap: 8px;
    align-items: center;
    white-space: nowrap;
}
.app-badge-item {
    font-size: 0.75rem;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 3px 8px;
    color: #475569;
    font-weight: 500;
    white-space: nowrap;
}

/* Thanh trạng thái nguy cơ phong cách sáng */
.status-bar {
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.status-bar.danger {
    background: #fef2f2;
    border: 1px solid #f87171;
    color: #b91c1c;
}
.status-bar.safe {
    background: #f0fdf4;
    border: 1px solid #4ade80;
    color: #15803d;
}
.status-bar.normal {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    color: #334155;
}
.status-sub {
    font-size: 0.8rem;
    font-weight: 400;
    color: #64748b;
    margin-left: 12px;
    white-space: nowrap;
}

/* Khối thống kê HUD phong cách sáng */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
}
@media (max-width: 768px) {
    .stat-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.stat-card .stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 2px;
    white-space: nowrap;
}
.stat-card .stat-value {
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.2;
    white-space: nowrap;
}
.stat-card .stat-sub {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 2px;
    white-space: nowrap;
}

/* Nút bấm & Nhãn form */
.quick-btn {
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #1e293b !important;
    white-space: nowrap !important;
}
.quick-btn:hover {
    background: #e2e8f0 !important;
}
label span {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #334155 !important;
    white-space: nowrap !important;
}

/* Tab Navigation */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #475569 !important;
    white-space: nowrap !important;
}
.tab-nav button.selected {
    color: #0f172a !important;
    border-color: #2563eb !important;
}

/* Ẩn footer Gradio */
footer { display: none !important; }
"""

# ---------------------------------------------------------
# Helper Functions for UI Rendering
# ---------------------------------------------------------
def render_risk_html(status_text: str, is_danger: bool, is_safe: bool, subtitle: str = "") -> str:
    """Generates clean minimalist risk status bar HTML without icons."""
    if is_danger:
        bar_cls = "status-bar danger"
    elif is_safe:
        bar_cls = "status-bar safe"
    else:
        bar_cls = "status-bar normal"

    sub_span = f"<span class='status-sub'>{subtitle}</span>" if subtitle else ""
    return f"""
    <div class="{bar_cls}">
        <span>{status_text}</span>
        {sub_span}
    </div>
    """

def render_stat_cards_html(stat1_lbl: str, stat1_val: str, stat1_sub: str,
                           stat2_lbl: str, stat2_val: str, stat2_sub: str,
                           stat3_lbl: str, stat3_val: str, stat3_sub: str,
                           stat4_lbl: str, stat4_val: str, stat4_sub: str) -> str:
    """Generates clean 4-card metric HUD bar HTML for light theme."""
    return f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">{stat1_lbl}</div>
            <div class="stat-value" style="color: #dc2626;">{stat1_val}</div>
            <div class="stat-sub">{stat1_sub}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">{stat2_lbl}</div>
            <div class="stat-value" style="color: #16a34a;">{stat2_val}</div>
            <div class="stat-sub">{stat2_sub}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">{stat3_lbl}</div>
            <div class="stat-value" style="color: #2563eb;">{stat3_val}</div>
            <div class="stat-sub">{stat3_sub}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">{stat4_lbl}</div>
            <div class="stat-value" style="color: #d97706;">{stat4_val}</div>
            <div class="stat-sub">{stat4_sub}</div>
        </div>
    </div>
    """

# ---------------------------------------------------------
# Helper: Robust File & Video Path Resolution
# ---------------------------------------------------------
def resolve_media_path(val: Any) -> Optional[str]:
    """
    Robustly resolves a local filesystem path from any Gradio file representation:
    string, Path, FileData, tempfile wrapper, dict, or list/tuple.
    Handles nested dicts like {'video': {'path': '...'}} and avoids TypeError on os.path.exists.
    """
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        for item in val:
            res = resolve_media_path(item)
            if res:
                return res
        return None
    if isinstance(val, (str, Path)):
        p = str(val).strip()
        if p and os.path.exists(p):
            return str(Path(p).resolve())
        return p if p else None
    if hasattr(val, "path") and getattr(val, "path"):
        return resolve_media_path(getattr(val, "path"))
    if hasattr(val, "name") and getattr(val, "name"):
        return resolve_media_path(getattr(val, "name"))
    if isinstance(val, dict):
        for key in ("path", "name", "video", "data"):
            if key in val and val[key]:
                res = resolve_media_path(val[key])
                if res and os.path.exists(res):
                    return res
        for v in val.values():
            if isinstance(v, (str, dict, list)):
                res = resolve_media_path(v)
                if res and os.path.exists(res):
                    return res
    return None

# ---------------------------------------------------------
# Handlers: Real-Time Video Streaming & Tracking Generator
# ---------------------------------------------------------
def stream_video_feed(
    video_input: Any,
    file_video_input: Any,
    conf_thresh: float,
    frame_skip: int,
    max_duration: int,
    tracker_name: str,
    enable_trail: bool
) -> Generator[Tuple[Optional[np.ndarray], str, str, pd.DataFrame, str], None, None]:
    """
    Generator that processes video frame-by-frame and yields live output at real-time rate.
    Displays live bounding boxes, Track IDs, and trajectory trails.
    """
    resolved_path = resolve_media_path(file_video_input) or resolve_media_path(video_input)

    if not resolved_path or not os.path.exists(resolved_path):
        yield (
            None,
            render_risk_html("CHƯA CÓ VIDEO", False, False, "Vui lòng chọn hoặc nạp video để bắt đầu."),
            render_stat_cards_html("Trẻ em", "-", "-", "Người lớn", "-", "-", "Một mình", "-", "-", "Tốc độ", "-", "-"),
            pd.DataFrame(columns=["ID", "Đối tượng", "Độ tin cậy", "Tọa độ", "Trạng thái"]),
            ""
        )
        return

    # Staging safely to avoid Windows temp locks
    actual_path = resolved_path
    try:
        source_p = Path(resolved_path)
        if "temp" in str(source_p).lower() or "cache" in str(source_p).lower():
            target_p = UPLOAD_DIR / source_p.name
            if not target_p.exists() or target_p.stat().st_size != source_p.stat().st_size:
                copied = False
                for c_att in range(25):
                    try:
                        shutil.copy2(source_p, target_p)
                        copied = True
                        break
                    except (PermissionError, OSError):
                        if c_att == 24:
                            break
                        time.sleep(0.1)
                if copied:
                    actual_path = str(target_p)
            else:
                actual_path = str(target_p)
    except Exception as stage_err:
        actual_path = resolved_path

    tracker_file = "bytetrack.yaml" if "ByteTrack" in tracker_name else "botsort.yaml"

    # Multimodal Audio Cry Analysis directly from the input video
    audio_summary_md = "**Phân tích âm thanh:** Video không có âm thanh hoặc bộ phân tích chưa sẵn sàng."
    if audio_detector is not None and getattr(audio_detector, "is_loaded", False):
        tmp_wav = None
        try:
            import subprocess, tempfile
            tmp_wav = os.path.join(tempfile.gettempdir(), f"cry_audio_{int(time.time()*1000)}.wav")
            ff_cmd = ["ffmpeg", "-y", "-i", actual_path, "-vn", "-ac", "1", "-ar", "16000", tmp_wav]
            ff_res = subprocess.run(ff_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            audio_target = tmp_wav if (ff_res.returncode == 0 and os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 1000) else actual_path
            ares = audio_detector.predict(audio_target)
            probs_str = " | ".join(f"**{k}**: `{v*100:.1f}%`" for k, v in ares.get("probabilities", {}).items())
            audio_summary_md = (
                f"### Phân tích Âm thanh Tiếng khóc (Từ Video)\n"
                f"- **Nguyên nhân dự đoán:** **{ares.get('vietnamese_cause', 'N/A')}** (`{ares.get('confidence', 0)*100:.1f}%`)\n"
                f"- **Phân bố xác suất:** {probs_str}\n"
                f"- **Gợi ý chăm sóc:** {ares.get('advice', '')}"
            )
        except Exception:
            audio_summary_md = "### Phân tích Âm thanh Tiếng khóc\n- Video không chứa kênh âm thanh hoặc không thể trích xuất âm thanh."
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    max_baby_seen = 0
    max_adult_seen = 0
    total_alone_seconds = 0.0
    all_seen_track_ids = set()
    last_frame = None
    stat_cards_html = ""
    df = pd.DataFrame(columns=["ID", "Đối tượng", "Độ tin cậy", "Tọa độ", "Trạng thái"])

    for item in detector.stream_video_frames(
        video_path=actual_path,
        conf_thresh=conf_thresh,
        iou_thresh=0.45,
        frame_skip=frame_skip,
        max_duration=max_duration,
        tracker_type=tracker_file,
        enable_trail=enable_trail
    ):
        frame = item["frame"]
        baby_count = item["baby_count"]
        adult_count = item["adult_count"]
        alone_sec = item["alone_duration_sec"]
        fps = item["fps"]
        progress_pct = item["progress_pct"]
        timestamp_sec = item["timestamp_sec"]
        boxes_data = item["boxes_data"]

        max_baby_seen = max(max_baby_seen, baby_count)
        max_adult_seen = max(max_adult_seen, adult_count)
        if alone_sec > 0:
            total_alone_seconds += (1.0 / max(fps, 1.0))

        # Build clean status bar
        if baby_count > 0 and adult_count == 0:
            status_text = f"CẢNH BÁO: Trẻ ở một mình ({alone_sec:.1f}s)"
            status_html = render_risk_html(
                status_text,
                is_danger=True,
                is_safe=False,
                subtitle=f"{fps:.1f} FPS | Clip: {timestamp_sec}s ({progress_pct}%)"
            )
        elif baby_count > 0 and adult_count > 0:
            status_text = f"AN TOÀN: Có người lớn ({adult_count})"
            status_html = render_risk_html(
                status_text,
                is_danger=False,
                is_safe=True,
                subtitle=f"{fps:.1f} FPS | Clip: {timestamp_sec}s ({progress_pct}%)"
            )
        else:
            status_text = "BÌNH THƯỜNG"
            status_html = render_risk_html(
                status_text,
                is_danger=False,
                is_safe=False,
                subtitle=f"{fps:.1f} FPS | Clip: {timestamp_sec}s ({progress_pct}%)"
            )

        # Build clean stats HUD
        stat_cards_html = render_stat_cards_html(
            stat1_lbl="Trẻ em",
            stat1_val=str(baby_count),
            stat1_sub=f"Tối đa: {max_baby_seen}",
            stat2_lbl="Người lớn",
            stat2_val=str(adult_count),
            stat2_sub=f"Tối đa: {max_adult_seen}",
            stat3_lbl="Một mình",
            stat3_val=f"{alone_sec:.1f}s",
            stat3_sub=f"Clip: {timestamp_sec}s",
            stat4_lbl="Tốc độ",
            stat4_val=f"{fps:.1f} FPS",
            stat4_sub=f"{item['latency_ms']} ms"
        )

        # Build clean tracking table
        table_rows = []
        for b in boxes_data:
            tid = b.get("track_id")
            cls_name = "Trẻ em" if b["cls"] == 1 else "Người lớn"
            tid_str = f"#{tid}" if tid is not None else "-"
            if tid is not None:
                all_seen_track_ids.add(tid)
            risk_tag = "Một mình" if b["cls"] == 1 and adult_count == 0 else "Bình thường"
            table_rows.append({
                "ID": tid_str,
                "Đối tượng": cls_name,
                "Độ tin cậy": f"{int(b['conf'] * 100)}%",
                "Tọa độ": str(b["coords"]),
                "Trạng thái": risk_tag
            })

        df = pd.DataFrame(table_rows) if table_rows else pd.DataFrame(columns=["ID", "Đối tượng", "Độ tin cậy", "Tọa độ", "Trạng thái"])
        last_frame = frame
        yield frame, status_html, stat_cards_html, df, audio_summary_md

    # Final completion yield (keep last annotated frame visible)
    final_banner = render_risk_html(
        "HOÀN TẤT",
        is_danger=(total_alone_seconds > 0),
        is_safe=(total_alone_seconds == 0),
        subtitle=f"Đã theo dõi {len(all_seen_track_ids)} đối tượng"
    )
    yield last_frame, final_banner, stat_cards_html, df, audio_summary_md

def load_demo_clip(key: str) -> Tuple[Any, Any, Optional[np.ndarray], str, str, pd.DataFrame, str]:
    """Loads demo source video path and resets live streaming HUD."""
    raw_video = SAMPLE_VIDEOS.get(key)
    status_html = render_risk_html("SẴN SÀNG", False, False, "Bấm 'Bắt đầu' để theo dõi.")
    empty_df = pd.DataFrame(columns=["ID", "Đối tượng", "Độ tin cậy", "Tọa độ", "Trạng thái"])
    return gr.Video(value=raw_video, interactive=True), None, None, status_html, "", empty_df, ""

# ---------------------------------------------------------
# Handlers: Image Processing
# ---------------------------------------------------------
def process_single_image(image_input: Optional[np.ndarray],
                         conf_thresh: float) -> Tuple[Optional[np.ndarray], str, str, pd.DataFrame]:
    """Processes uploaded or demo snapshot."""
    if image_input is None:
        return (
            None,
            render_risk_html("CHƯA CÓ ẢNH", False, False, "Vui lòng nạp ảnh để phân tích."),
            render_stat_cards_html("Trẻ em", "-", "-", "Người lớn", "-", "-", "Độ trễ", "-", "-", "Tốc độ", "-", "-"),
            pd.DataFrame(columns=["Đối tượng", "Độ tin cậy", "Tọa độ", "Trạng thái"])
        )

    annotated_img, summary = detector.detect_image(image_input, conf_thresh=conf_thresh, iou_thresh=0.45)
    risk = summary["risk"]

    status_text = "CẢNH BÁO: Trẻ ở một mình" if risk["is_danger"] else ("AN TOÀN: Có người lớn" if risk["level"] == RiskLevel.SAFE else "BÌNH THƯỜNG")
    status_html = render_risk_html(
        status_text,
        is_danger=risk["is_danger"],
        is_safe=(risk["level"] == RiskLevel.SAFE),
        subtitle=f"Độ trễ: {summary['latency_ms']} ms | {summary['fps']} FPS"
    )

    stat_cards_html = render_stat_cards_html(
        stat1_lbl="Trẻ em",
        stat1_val=str(summary['baby_count']),
        stat1_sub="Số lượng",
        stat2_lbl="Người lớn",
        stat2_val=str(summary['adult_count']),
        stat2_sub="Số lượng",
        stat3_lbl="Độ trễ",
        stat3_val=f"{summary['latency_ms']} ms",
        stat3_sub="Thời gian xử lý",
        stat4_lbl="Tốc độ",
        stat4_val=f"{summary['fps']} FPS",
        stat4_sub="Tốc độ suy luận"
    )

    # Format dataframe
    rows = []
    for d in summary.get("detections", []):
        cls_name = "Trẻ em" if d["class"] == "baby_child" else "Người lớn"
        risk_note = "Một mình" if d["class"] == "baby_child" and summary["adult_count"] == 0 else "Bình thường"
        rows.append({
            "Đối tượng": cls_name,
            "Độ tin cậy": f"{d['confidence']}%",
            "Tọa độ": str(d["box"]),
            "Trạng thái": risk_note
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"Đối tượng": "Không có người", "Độ tin cậy": "-", "Tọa độ": "-", "Trạng thái": "Trống"}])

    return annotated_img, status_html, stat_cards_html, df

# ---------------------------------------------------------
# Gradio 6 Blocks Application
# ---------------------------------------------------------
with gr.Blocks(title="Giám Sát An Toàn Trẻ Em") as demo:
    gpu_badge = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    
    # 1. Minimalist Top Header
    gr.HTML(f"""
    <div class="app-header">
        <h1 class="app-title">GIÁM SÁT AN TOÀN TRẺ EM</h1>
        <div class="app-badges">
            <span class="app-badge-item">Hệ thống: Sẵn sàng</span>
            <span class="app-badge-item">{gpu_badge}</span>
        </div>
    </div>
    """)

    # 2. Main Tabs
    with gr.Tabs():
        # ==========================================
        # TAB 1: VIDEO
        # ==========================================
        with gr.TabItem("Video", id="tab_video"):
            with gr.Row():
                # Left Column: Inputs & Controls
                with gr.Column(scale=5):
                    video_input = gr.Video(
                        label="Video nguồn",
                        sources=["upload"],
                        interactive=True
                    )
                    with gr.Accordion("Chọn file khác", open=False):
                        file_video_input = gr.File(
                            label="Tải file từ máy tính",
                            file_types=[".mp4", ".avi", ".mkv", ".mov", ".webm", "video/*"],
                            interactive=True
                        )
                    
                    gr.Markdown("**Mẫu thử:**")
                    with gr.Row():
                        btn_sample_alone = gr.Button("Trẻ một mình", elem_classes=["quick-btn"], size="sm")
                        btn_sample_caring = gr.Button("Có người lớn", elem_classes=["quick-btn"], size="sm")
                        btn_sample_daycare = gr.Button("Lớp học", elem_classes=["quick-btn"], size="sm")

                    with gr.Accordion("Cấu hình", open=False):
                        tracker_dropdown = gr.Dropdown(
                            choices=["ByteTrack", "BoT-SORT"],
                            value="ByteTrack",
                            label="Thuật toán theo dõi"
                        )
                        trail_checkbox = gr.Checkbox(
                            value=True,
                            label="Hiển thị quỹ đạo di chuyển"
                        )
                        conf_slider_video = gr.Slider(minimum=0.15, maximum=0.85, value=0.35, step=0.05, label="Ngưỡng tin cậy")
                        skip_slider = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Bước nhảy khung hình")
                        duration_slider = gr.Slider(
                            minimum=1,
                            maximum=300,
                            value=30,
                            step=1,
                            label="Thời lượng tối đa (giây)"
                        )

                    with gr.Row():
                        btn_run_stream = gr.Button("Bắt đầu", variant="primary", size="lg", scale=3)
                        btn_stop_stream = gr.Button("Dừng", variant="stop", size="lg", scale=1)

                # Right Column: Output Dashboard
                with gr.Column(scale=7):
                    video_status_out = gr.HTML(render_risk_html("CHỜ TÍN HIỆU", False, False, "Nạp video và bấm 'Bắt đầu'."))
                    video_stats_out = gr.HTML(render_stat_cards_html("Trẻ em", "-", "-",
                                                                      "Người lớn", "-", "-",
                                                                      "Một mình", "-", "-",
                                                                      "Tốc độ", "-", "-"))
                    live_frame_output = gr.Image(label="Luồng giám sát", type="numpy")
                    live_tracking_table = gr.Dataframe(headers=["ID", "Đối tượng", "Độ tin cậy", "Tọa độ", "Trạng thái"], label="Danh sách đối tượng")
                    video_audit_out = gr.Markdown("")

            # Tab 1 Event Bindings
            btn_sample_alone.click(
                fn=lambda: load_demo_clip("baby_alone"),
                inputs=[],
                outputs=[video_input, file_video_input, live_frame_output, video_status_out, video_stats_out, live_tracking_table, video_audit_out]
            )
            btn_sample_caring.click(
                fn=lambda: load_demo_clip("adult_caring"),
                inputs=[],
                outputs=[video_input, file_video_input, live_frame_output, video_status_out, video_stats_out, live_tracking_table, video_audit_out]
            )
            btn_sample_daycare.click(
                fn=lambda: load_demo_clip("daycare"),
                inputs=[],
                outputs=[video_input, file_video_input, live_frame_output, video_status_out, video_stats_out, live_tracking_table, video_audit_out]
            )

            stream_event = btn_run_stream.click(
                fn=stream_video_feed,
                inputs=[video_input, file_video_input, conf_slider_video, skip_slider, duration_slider, tracker_dropdown, trail_checkbox],
                outputs=[live_frame_output, video_status_out, video_stats_out, live_tracking_table, video_audit_out]
            )
            btn_stop_stream.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[stream_event]
            )

        # ==========================================
        # TAB 2: IMAGE
        # ==========================================
        with gr.TabItem("Hình ảnh", id="tab_image"):
            with gr.Row():
                with gr.Column(scale=5):
                    image_input = gr.Image(label="Ảnh nguồn", type="numpy")
                    
                    gr.Markdown("**Mẫu thử:**")
                    with gr.Row():
                        btn_img_alone = gr.Button("Trẻ một mình", elem_classes=["quick-btn"], size="sm")
                        btn_img_both = gr.Button("Cả hai", elem_classes=["quick-btn"], size="sm")
                        btn_img_adult = gr.Button("Người lớn", elem_classes=["quick-btn"], size="sm")

                    conf_slider_img = gr.Slider(minimum=0.15, maximum=0.85, value=0.35, step=0.05, label="Ngưỡng tin cậy")
                    btn_run_image = gr.Button("Phân tích", variant="primary", size="lg")

                with gr.Column(scale=7):
                    img_status_out = gr.HTML(render_risk_html("CHỜ TÍN HIỆU", False, False, "Nạp ảnh và bấm 'Phân tích'."))
                    img_stats_out = gr.HTML(render_stat_cards_html("Trẻ em", "-", "-", "Người lớn", "-", "-", "Độ trễ", "-", "-", "Tốc độ", "-", "-"))
                    image_output = gr.Image(label="Kết quả phân tích")
                    img_table_out = gr.Dataframe(headers=["Đối tượng", "Độ tin cậy", "Tọa độ", "Trạng thái"], label="Đối tượng phát hiện")

            # Tab 2 Event Bindings
            btn_img_alone.click(fn=lambda: cv2.cvtColor(cv2.imread(SAMPLE_IMAGES["baby_alone"]), cv2.COLOR_BGR2RGB), inputs=[], outputs=[image_input])
            btn_img_both.click(fn=lambda: cv2.cvtColor(cv2.imread(SAMPLE_IMAGES["child_and_adult"]), cv2.COLOR_BGR2RGB), inputs=[], outputs=[image_input])
            btn_img_adult.click(fn=lambda: cv2.cvtColor(cv2.imread(SAMPLE_IMAGES["adult_only"]), cv2.COLOR_BGR2RGB), inputs=[], outputs=[image_input])

            btn_run_image.click(
                fn=process_single_image,
                inputs=[image_input, conf_slider_img],
                outputs=[image_output, img_status_out, img_stats_out, img_table_out]
            )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
        allowed_paths=[str(ROOT_DIR), str(TEMP_GRADIO_DIR), str(UPLOAD_DIR), str(CACHE_DIR), tempfile.gettempdir()]
    )
