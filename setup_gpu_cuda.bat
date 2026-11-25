@echo off
chcp 65001 > nul
title SYNEXAN AI LABS - GPU & CUDA Universal Setup
echo =====================================================================
echo    SYNEXAN AI LABS - TRÌNH CÀI ĐẶT & KÍCH HOẠT GPU NVIDIA CHO DỰ ÁN
echo =====================================================================
echo.

:: 1. Xác định môi trường Python
set PY_BIN=
if exist "%~dp0..\Fruit-Disease-Detection\.venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0..\Fruit-Disease-Detection\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

echo [1/3] Môi trường Python được chọn:
echo       %PY_BIN%
echo.

:: 2. Chạy kiểm tra GPU hiện tại
echo [2/3] Đang kiểm tra cấu hình phần cứng và thư viện PyTorch...
"%PY_BIN%" scripts\check_gpu.py
if %ERRORLEVEL% equ 0 (
    echo.
    echo =====================================================================
    echo ✅ GPU đã sẵn sàng hoạt động! Không cần cài đặt lại.
    echo =====================================================================
    pause
    exit /b 0
)

:: 3. Tiến hành nâng cấp PyTorch CUDA nếu chưa có
echo.
echo =====================================================================
echo ⚠️ Phát hiện chưa có PyTorch CUDA. Bắt đầu tải và cài đặt PyTorch CUDA 12.1...
echo    (Quá trình tải khoảng 2.5 GB, vui lòng giữ kết nối Internet)
echo =====================================================================
echo.

"%PY_BIN%" -m pip uninstall -y torch torchvision
"%PY_BIN%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/3] Đang kiểm tra lại sau khi cài đặt...
"%PY_BIN%" scripts\check_gpu.py

echo.
echo =====================================================================
echo 🎉 Hoàn tất thiết lập GPU cho dự án!
echo =====================================================================
pause
