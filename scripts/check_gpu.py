"""
SYNEXAN AI LABS - Universal GPU & CUDA Diagnostic Utility
Script: check_gpu.py
Purpose: Automatically verifies NVIDIA GPU, CUDA runtime, VRAM capacity, and tensor compute.
"""

import sys
import time

def print_header(title: str):
    print("=" * 65)
    print(f"   {title}")
    print("=" * 65)

def check_gpu():
    print_header("SYNEXAN AI LABS - KIỂM TRA PHẦN CỨNG GPU & CUDA")
    print(f"🐍 Python Interpreter: {sys.executable}")
    print(f"🐍 Python Version:     {sys.version.split()[0]}")
    
    try:
        import torch
    except ImportError:
        print("\n❌ LỖI: Chưa cài đặt thư viện 'torch'. Vui lòng chạy setup_gpu_cuda.bat để cài đặt!")
        return False
        
    print(f"📦 PyTorch Version:    {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"⚡ CUDA Khả Dụng:      {'✅ CÓ (TRUE)' if cuda_available else '❌ KHÔNG (FALSE - Đang chạy CPU)'}")
    
    if not cuda_available:
        print("\n⚠️ NGUYÊN NHÂN & HƯỚNG XỬ LÝ:")
        print("  - Bản PyTorch hiện tại là bản CPU-only (không có driver CUDA đi kèm).")
        print("  - Vui lòng chạy tệp 'setup_gpu_cuda.bat' hoặc lệnh sau để cài đặt bản CUDA:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False
        
    # GPU Information
    device_count = torch.cuda.device_count()
    print(f"🎮 Số Lượng GPU:       {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        vram_total_mb = props.total_memory / (1024 ** 2)
        vram_total_gb = vram_total_mb / 1024
        cc = f"{props.major}.{props.minor}"
        print(f"\n--- [GPU {i}: {props.name}] ---")
        print(f"  • Dung lượng VRAM:        {vram_total_mb:.1f} MB (~{vram_total_gb:.2f} GB)")
        print(f"  • Compute Capability:     {cc}")
        print(f"  • Multi-Processors (SMs): {props.multi_processor_count}")
        
    # Tensor Compute Test
    print("\n🧪 [BÀI TEST TÍNH TOÁN MA TRẬN TRÊN GPU (CUDA TENSOR COMPUTE)]")
    try:
        torch.cuda.empty_cache()
        t0 = time.time()
        # Test Float32
        x = torch.randn(2000, 2000, device="cuda:0", dtype=torch.float32)
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        t_fp32 = (time.time() - t0) * 1000
        print(f"  ✅ Phép nhân ma trận FP32 (2000x2000): {t_fp32:.2f} ms")
        
        # Test Float16 (Half-precision)
        t0 = time.time()
        x_half = x.half()
        y_half = torch.matmul(x_half, x_half)
        torch.cuda.synchronize()
        t_fp16 = (time.time() - t0) * 1000
        mem_used_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        print(f"  ✅ Phép nhân ma trận FP16 (2000x2000): {t_fp16:.2f} ms (Nhanh gấp {t_fp32 / max(t_fp16, 0.001):.1f}x)")
        print(f"  📊 VRAM cấp phát trong bài test:     {mem_used_mb:.2f} MB")
        
        print("\n🎉 KẾT LUẬN: GPU & CUDA HOẠT ĐỘNG HOÀN HẢO! SẴN SÀNG CHO MỌI DỰ ÁN AI.")
        return True
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH TÍNH TOÁN GPU: {e}")
        return False

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
