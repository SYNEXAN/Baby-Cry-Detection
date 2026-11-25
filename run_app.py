"""
Main Application Runner: Launches the decoupled Infant Safety Monitoring Web App.
Serves Frontend & FastAPI Backend on http://127.0.0.1:8000
"""

import sys
import uvicorn
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    print("=" * 60)
    print("  HE THONG GIAM SAT AN TOAN TRE EM & NHAN DIEN TIENG KHOC")
    print("  Decoupled Architecture: HTML5/CSS/JS + FastAPI Backend")
    print("  Truy cap tai: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
