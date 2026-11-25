"""
Script: scripts/mux_videotest_audio.py
Purpose: Embeds realistic audio tracks into VIDEO_TEST videos so they have both video and audio.
"""

import os
import subprocess
import glob
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VIDEO_TEST_DIR = ROOT_DIR / "VIDEO_TEST"
BACKUP_DIR = VIDEO_TEST_DIR / "originals"
RAW_AUDIO_DIR = ROOT_DIR / "data" / "raw" / "donate_a_cry"

os.makedirs(BACKUP_DIR, exist_ok=True)

# Audio sources
a_pain = glob.glob(str(RAW_AUDIO_DIR / "belly_pain" / "*.wav"))[0]
a_burp = glob.glob(str(RAW_AUDIO_DIR / "burping" / "*.wav"))[0]
a_needs = glob.glob(str(RAW_AUDIO_DIR / "hungry" / "*.wav"))[0]

# Synthetic classroom ambient noise (no cry)
noise_wav = VIDEO_TEST_DIR / "temp_ambient.wav"
cmd_noise = [
    "ffmpeg", "-y",
    "-f", "lavfi", "-i", "anoisesrc=d=30:c=pink:r=16000:a=0.06",
    str(noise_wav)
]
subprocess.run(cmd_noise, capture_output=True)

mappings = [
    ("294613747-rivne-ukraine-june-12-2024-chi.mp4", a_pain),
    ("280167556-female-teacher-checking-kids-d.mp4", a_burp),
    ("050870650-preschool-children-are-early-c.mp4", str(noise_wav)),
    ("sd_watermarked_preview.mp4", a_needs)
]

for vname, audio_path in mappings:
    src_video = VIDEO_TEST_DIR / vname
    if not src_video.exists():
        continue
    backup_file = BACKUP_DIR / vname
    if not backup_file.exists():
        # Copy to backup first
        import shutil
        shutil.copy2(src_video, backup_file)
        print(f"Backed up {vname} to originals/")

    temp_out = VIDEO_TEST_DIR / f"temp_{vname}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(backup_file),
        "-stream_loop", "-1", "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        str(temp_out)
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and temp_out.exists() and os.path.getsize(temp_out) > 1000:
        os.replace(temp_out, src_video)
        print(f"Successfully muxed audio into: {vname} ({os.path.getsize(src_video)/1024:.1f} KB)")
    else:
        if temp_out.exists():
            os.remove(temp_out)
        print(f"Failed to mux audio into {vname}: {res.stderr.decode('utf-8', errors='ignore')}")

if noise_wav.exists():
    os.remove(noise_wav)

print("VIDEO_TEST audio preparation complete.")
