import os
import subprocess
import glob
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT_DIR / "demo_videos"
RAW_AUDIO_DIR = ROOT_DIR / "data" / "raw" / "donate_a_cry"

os.makedirs(DEMO_DIR, exist_ok=True)

# Sources
v_alone = DEMO_DIR / "sample_baby_alone.mp4"
v_adult = DEMO_DIR / "sample_adult_caring.mp4"

# Audio sources
a_belly_pain = glob.glob(str(RAW_AUDIO_DIR / "belly_pain" / "*.wav"))[0]
a_burping = glob.glob(str(RAW_AUDIO_DIR / "burping" / "*.wav"))[0]
a_hungry = glob.glob(str(RAW_AUDIO_DIR / "hungry" / "*.wav"))[0]

print("Preparing multi-modal demo video samples...")

# Sample 1: Baby Alone + Belly Pain Cry (High Risk + Cry)
out1 = DEMO_DIR / "sample_baby_crying_alone.mp4"
cmd1 = [
    "ffmpeg", "-y",
    "-stream_loop", "-1", "-i", str(v_alone),
    "-i", a_belly_pain,
    "-map", "0:v", "-map", "1:a",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k",
    "-t", "6", str(out1)
]
subprocess.run(cmd1, capture_output=True)
print(f"Created: {out1.name} ({os.path.getsize(out1)/1024:.1f} KB)")

# Sample 2: Adult Caring + Burping Cry (Accompanied + Cry)
out2 = DEMO_DIR / "sample_adult_caring_burping.mp4"
cmd2 = [
    "ffmpeg", "-y",
    "-stream_loop", "-1", "-i", str(v_adult),
    "-i", a_burping,
    "-map", "0:v", "-map", "1:a",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k",
    "-t", "6", str(out2)
]
subprocess.run(cmd2, capture_output=True)
print(f"Created: {out2.name} ({os.path.getsize(out2)/1024:.1f} KB)")

# Sample 3: Baby Alone + Ambient Fan Noise (High Risk + Noise/No Cry)
# Generate 6s synthetic white/fan noise audio
noise_wav = DEMO_DIR / "temp_ambient_noise.wav"
cmd_noise = [
    "ffmpeg", "-y",
    "-f", "lavfi", "-i", "anoisesrc=d=6:c=pink:r=16000:a=0.08",
    str(noise_wav)
]
subprocess.run(cmd_noise, capture_output=True)

out3 = DEMO_DIR / "sample_baby_alone_ambient_noise.mp4"
cmd3 = [
    "ffmpeg", "-y",
    "-stream_loop", "-1", "-i", str(v_alone),
    "-i", str(noise_wav),
    "-map", "0:v", "-map", "1:a",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k",
    "-t", "6", str(out3)
]
subprocess.run(cmd3, capture_output=True)
if os.path.exists(noise_wav):
    os.remove(noise_wav)
print(f"Created: {out3.name} ({os.path.getsize(out3)/1024:.1f} KB)")

print("All multi-modal demo samples prepared successfully.")
