# CẨM NANG THIẾT LẬP & TỐI ƯU HÓA GPU NVIDIA CHO TẤT CẢ DỰ ÁN AI
*(SYNEXAN AI LABS - Universal GPU & CUDA Acceleration Guide)*

Tài liệu này hướng dẫn cách cấu hình, kích hoạt và tối ưu card đồ họa rời **NVIDIA GeForce MX350 (2GB VRAM)** để sử dụng mượt mà, thời gian thực cho tất cả các dự án Trí Tuệ Nhân Tạo hiện tại và tương lai của SYNEXAN (như `Cry_Baby_Detection`, `Fruit-Disease-Detection`, `Flood`, v.v.).

---

## 1. Thông Số Phần Cứng & Môi Trường Chuẩn

| Thành phần | Thông số trên máy tính | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **Card đồ họa (GPU)** | **NVIDIA GeForce MX350** | Kiến trúc Pascal GP107, 640 nhân CUDA |
| **Dung lượng VRAM** | **2,048 MB (2.0 GB)** | Cần chạy chế độ FP16 nửa độ chính xác để tiết kiệm bộ nhớ |
| **Driver NVIDIA** | Version `582.67` | Tương thích CUDA 11.8, 12.1, 12.4, 13.0 |
| **PyTorch khuyên dùng** | **PyTorch 2.5.1 + CUDA 12.1 (`cu121`)** | Tương thích hoàn hảo với Python 3.12 |
| **Môi trường Python chung** | `D:\SYNEXAN\Project\Fruit-Disease-Detection\.venv` | Dùng chung để tiết kiệm dung lượng ổ cứng |

---

## 2. Cách Khởi Động & Kiểm Tra GPU (Chỉ Với 1 Click)

### Cách 1: Sử dụng công cụ tự động 1-Click
Trong thư mục dự án, chỉ cần nhấp đúp tệp:
👉 [setup_gpu_cuda.bat](file:///D:/SYNEXAN/Project/Cry_Baby_Detection/setup_gpu_cuda.bat)
Tệp sẽ tự động kiểm tra xem môi trường đã có GPU chưa. Nếu chưa có, nó sẽ tự động tải và cài đặt bản CUDA mà bạn không cần gõ lệnh.

### Cách 2: Chạy công cụ chẩn đoán chuyên sâu
Mở terminal và gõ:
```powershell
D:\SYNEXAN\Project\Fruit-Disease-Detection\.venv\Scripts\python.exe scripts/check_gpu.py
```
Nếu màn hình xuất hiện:
```
⚡ CUDA Khả Dụng: CÓ (TRUE)
--- [GPU 0: NVIDIA GeForce MX350] ---
• Dung lượng VRAM: 2048.0 MB (~2.00 GB)
✅ Phép nhân ma trận FP16: ... ms (Nhanh gấp ...)
🎉 KẾT LUẬN: GPU & CUDA HOẠT ĐỘNG HOÀN HẢO!
```
$\rightarrow$ GPU đã sẵn sàng 100%.

---

## 3. Quy Tắc Lập Trình Chuẩn (Best Practices) Cho Mọi Dự Án Tương Lai

Do card đồ họa có **2GB VRAM**, để tránh hoàn toàn lỗi tràn bộ nhớ (`CUDA out of memory`), mọi dự án cần tuân theo 4 quy tắc vàng sau:

### Quy tắc 1: Đoạn code tự động nhận GPU thông minh
```python
import torch

# Tự động chọn GPU nếu có, ngược lại fallback về CPU an toàn
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 Thiết bị suy luận: {device}")
```

### Quy tắc 2: Luôn bật chế độ nửa độ chính xác FP16 (`half=True`)
Chế độ FP16 giúp mô hình tiêu thụ ít hơn 50% VRAM (chỉ ~800MB thay vì 1.6GB), đồng thời nhân CUDA tính toán nhanh gấp đôi:
- **Với Ultralytics (YOLOv8, YOLOv11, RT-DETR)**:
  ```python
  from ultralytics import RTDETR
  
  model = RTDETR("weights/best.pt")
  # Suy luận với half=True và batch size = 1
  results = model.predict(image, device=device, half=(device != "cpu"), imgsz=480)
  ```
- **Với PyTorch thuần túy**:
  ```python
  model = MyModel().to(device)
  if device != "cpu":
      model = model.half() # Chuyển trọng số sang FP16
  ```

### Quy tắc 3: Luôn sử dụng `torch.inference_mode()`
Thay vì `torch.no_grad()`, `torch.inference_mode()` giúp tắt hoàn toàn đồ thị tính toán đạo hàm, giải phóng triệt để RAM và tăng tốc thêm 15%:
```python
with torch.inference_mode():
    output = model(input_tensor)
```

### Quy tắc 4: Tự động giải phóng bộ nhớ đệm VRAM khi chuyển video/tác vụ
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## 4. Cách Dùng Chung Môi Trường `.venv` Cho Dự Án Mới
Để không phải tạo thêm nhiều môi trường ảo gây tốn dung lượng ổ cứng (~5GB mỗi venv):
1. Khi tạo dự án mới (ví dụ `D:\SYNEXAN\Project\DuAnMoi`):
2. Trong tệp chạy `run.bat`, chỉ cần trỏ trực tiếp đến Python venv đã có CUDA:
   ```bat
   @echo off
   "D:\SYNEXAN\Project\Fruit-Disease-Detection\.venv\Scripts\python.exe" main.py
   pause
   ```
3. Mọi thư viện Deep Learning (PyTorch, Torchvision, CUDA, Ultralytics, Gradio, OpenCV, NumPy) sẽ được dùng chung ngay lập tức mà không cần cài đặt lại!
