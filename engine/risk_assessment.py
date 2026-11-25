"""
Module: risk_assessment.py
Purpose: Evaluates infant/child unattended risk from detection counts.
"""

from typing import Dict, Any, List

class RiskLevel:
    SAFE = "SAFE"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    EMPTY = "EMPTY"

def assess_frame_risk(adult_count: int, baby_count: int) -> Dict[str, Any]:
    """
    Evaluates the safety risk of a single frame:
    - SAFE: Both adult and baby present, or only adult present.
    - HIGH_RISK: Baby is present, but NO ADULT present in the room!
    - EMPTY: No adult and no baby present.
    """
    if baby_count > 0 and adult_count == 0:
        level = RiskLevel.HIGH_RISK
        message = "🚨 NGUY HIỂM CAO: PHÁT HIỆN TRẺ BỊ BỎ LẠI MỘT MÌNH!"
        color = "#FF3366" # Neon Red
        badge = "UNATTENDED_DANGER"
    elif baby_count > 0 and adult_count > 0:
        level = RiskLevel.SAFE
        message = "✅ AN TOÀN: Có người lớn / người giám hộ trông nom trẻ."
        color = "#00FFA3" # Neon Green
        badge = "SAFE_ACCOMPANIED"
    elif baby_count == 0 and adult_count > 0:
        level = RiskLevel.SAFE
        message = "ℹ️ Phòng có người lớn, không phát hiện trẻ nhỏ."
        color = "#00D2FF" # Cyan
        badge = "ADULT_ONLY"
    else:
        level = RiskLevel.EMPTY
        message = "⚪ Phòng trống: Không phát hiện người hay trẻ nhỏ."
        color = "#94A3B8" # Slate Gray
        badge = "EMPTY_ROOM"

    return {
        "level": level,
        "message": message,
        "color": color,
        "badge": badge,
        "is_danger": level == RiskLevel.HIGH_RISK
    }

def summarize_video_risk(frame_results: List[Dict[str, Any]], fps: float) -> Dict[str, Any]:
    """
    Aggregates full video detection timeline into executive safety metrics.
    """
    total_frames = len(frame_results)
    if total_frames == 0:
        return {}

    duration_sec = total_frames / max(fps, 1.0)
    unattended_frames = sum(1 for f in frame_results if f["risk"]["level"] == RiskLevel.HIGH_RISK)
    safe_frames = sum(1 for f in frame_results if f["risk"]["level"] == RiskLevel.SAFE)
    empty_frames = sum(1 for f in frame_results if f["risk"]["level"] == RiskLevel.EMPTY)

    max_baby = max((f["baby_count"] for f in frame_results), default=0)
    max_adult = max((f["adult_count"] for f in frame_results), default=0)

    unattended_sec = unattended_frames / max(fps, 1.0)
    unattended_pct = (unattended_frames / total_frames) * 100.0

    if unattended_frames > (total_frames * 0.15): # If alone >15% of the video
        overall_status = "🔴 NGUY HIỂM CAO (Cần can thiệp ngay)"
        status_color = "#FF3366"
    elif unattended_frames > 0:
        overall_status = "🟡 CẢNH BÁO (Trẻ có thời điểm bị bỏ lại)"
        status_color = "#FFA500"
    else:
        overall_status = "🟢 AN TOÀN (Có người lớn bảo hộ xuyên suốt)"
        status_color = "#00FFA3"

    return {
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 1),
        "unattended_sec": round(unattended_sec, 1),
        "unattended_pct": round(unattended_pct, 1),
        "max_baby": max_baby,
        "max_adult": max_adult,
        "overall_status": overall_status,
        "status_color": status_color
    }
