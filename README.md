# Baby-Cry-Detection

Multi-Modal Infant Safety Monitoring and Real-Time Cry Reason Classification System.

This project integrates deep computer vision and procedural acoustic signal processing to monitor infant unattended risk in real time, discriminate infant cries from ambient environment noise, and diagnose 4 primary causes of infant distress.

---

## 1. System Architecture

The platform is designed with a decoupled architecture for maximum responsiveness and seamless remote cloud deployment:

- **Frontend (HTML5 / CSS3 / Vanilla JavaScript):**
  - Pure, lightweight web interface without third-party framework overhead.
  - Side-by-side horizontal multi-modal layout (50%-50% responsive grid).
  - Real-time frame rendering with annotated bounding boxes and HUD diagnostics.
  - Visual video gallery showcasing first-frame thumbnails for instant test playback.
  - Sits in `frontend/` directory, configurable for remote cloud API servers via `API_BASE_URL`.

- **Backend (FastAPI + Uvicorn):**
  - RESTful endpoints for video file uploads and sample video inventory.
  - HTTP Range Request streaming (`/api/video`) supporting smooth HTML5 video scrubbing.
  - Server-Sent Events (SSE) streaming (`/api/analyze/stream`) pushing annotated frames and live multi-modal fusion metrics.

- **Core AI Engine (`engine/`):**
  - **Vision Model (YOLO11-m V4):** Trained on balanced person/infant clinical data. Accurately detects `TRE EM` (infant) and `NGUOI LON` (adults) with bounding boxes and unattended timer logic.
  - **Acoustic Cry-VAD Gate (Stage 1 DSP):** Evaluates RMS energy, spectral flatness, and 800-3500 Hz infant vocal resonance to filter out air conditioners, fan rumble, and silence.
  - **Acoustic Reason Classifier (Stage 2 ResNet18):** 4-class classifier identifying `Burping (O hoi)`, `Needs (Nhu cau)`, `Physical Pain (Dau bung)`, and `Scared (Hoang so)`.
  - **Continuous Sliding Window Monitoring:** Real-time audio window (2.5s) evaluated continuously along the video timeline, ensuring cry events occurring anywhere in the video are immediately detected.

---

## 2. Directory Structure

```text
Baby-Cry-Detection/
├── backend/
│   └── server.py             # FastAPI REST & SSE Streaming service
├── frontend/
│   ├── index.html            # Semantic HTML5 UI
│   ├── css/
│   │   └── style.css         # Slate design system
│   ├── js/
│   │   └── app.js            # SSE client, drag-and-drop upload, DOM manager
│   └── thumbnails/           # First-frame visual thumbnails for sample videos
├── engine/
│   └── unified_detector.py   # Multi-modal detection, VAD, and risk fusion
├── models/
│   └── best_eletriby_4class_audio_classifier.pt # Production ResNet18 audio weights
├── kaggle_results/
│   └── v4_clean_balanced/
│       └── best.pt           # Production YOLO11-m vision weights
├── VIDEO_TEST/               # Verified candidate evaluation videos
├── scripts/                  # CPU benchmarks, training, and analysis scripts
├── requirements.txt          # Python dependencies
├── run_app.py                # Main application runner
└── README.md
```

---

## 3. Installation & Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13)
- FFmpeg installed and accessible in the system PATH

### Installation
```bash
# Clone the repository
git clone https://github.com/SYNEXAN/Baby-Cry-Detection.git
cd Baby-Cry-Detection

# Create virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
Launch the decoupled web server:
```bash
python run_app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 4. Hardware Requirements & Performance

The pipeline operates fully on CPU runtime without mandatory GPU requirements:

| Component | Resolution / Setting | CPU Latency (Intel Core i5) | Inference Throughput |
| :--- | :---: | :---: | :---: |
| **YOLO11-m Vision** | `imgsz=960` | ~605 ms / frame | ~1.65 FPS raw |
| **YOLO11-m Vision** | `imgsz=640` | ~301 ms / frame | ~3.32 FPS raw |
| **DSP Cry-VAD + ResNet18** | 16 kHz Mono (2.5s window) | ~15 ms / window | Continuous background |
| **End-to-End Pipeline** | `frame_skip=4`, `imgsz=960` | - | ~3.71 FPS effective video throughput |

When deployed on a GPU environment (e.g. NVIDIA Tesla T4 or RTX 3060+), the vision model achieves 45 - 65 FPS, providing full real-time monitoring speed.

---

## 5. Security & Privacy Notice

This repository strictly enforces confidentiality:
- All raw datasets (~8 GB audio samples) are excluded via `.gitignore`.
- No credentials, tokens, API keys, or private research papers are stored in this repository.
- Model weights included are dedicated production checkpoints (< 50 MB) for out-of-the-box demonstration.

---

## 6. License

Proprietary - Developed by SYNEXAN Team.
