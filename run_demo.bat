@echo off
chcp 65001 > nul
title SYNEXAN AI - Baby Safety Video Monitoring Dashboard
echo =====================================================================
echo    SYNEXAN AI LABS - BABY & CHILD SAFETY VIDEO MONITORING SYSTEM
echo    Model: RT-DETR-L (Version 5, mAP@50 = 80.08%)
echo =====================================================================
echo.
echo [1/2] Dang kiem tra moi truong Python & Torch...
set VENV_PY=D:\SYNEXAN\Project\Fruit-Disease-Detection\.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    echo [ERROR] Khong tim thay Python venv tai: %VENV_PY%
    pause
    exit /b 1
)

echo [2/2] Dang khoi dong Web Dashboard tai http://127.0.0.1:7860 ...
echo.
"%VENV_PY%" app.py
pause
