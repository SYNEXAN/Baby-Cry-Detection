# PROJECT HANDOVER CONTEXT & STATE SNAPSHOT
**Dự án**: Cry_Baby_Detection (Hệ Thống Giám Sát An Toàn Trẻ Em & Nhận Diện Tiếng Khóc)
**Thời gian tạo snapshot**: 2026-11-18 16:20:00 (UTC+7)
**Conversation ID gốc**: `30d185aa-58de-4b8e-b83e-50bde963d7f6`

---

## 1. Mục Đích & Vấn Đề Đang Xử Lý
1. **Lỗi nhận diện nhầm người lớn thành em bé (False Alarm)**:
   - Mô hình cũ (V5) nhận nhầm cô giáo ngồi trên thảm cầm bóng bay màu hồng và cô giáo quỳ gối thành `baby_child` với bounding box chỉ khoanh vùng khuôn mặt/thân trên.
   - Hậu quả: Kích hoạt còi báo động `DANGER: UNATTENDED CHILD DETECTED | NO ADULT PRESENT!` sai thực tế.
2. **Giải pháp đang triển khai (Model V7 Retraining)**:
   - Chuyển sang mô hình SOTA **YOLO11-M** (`yolo11m.pt`, 20.1M parameters).
   - Chuẩn hóa Bounding Box **toàn thân (Full-body)**, lọc bỏ triệt để nhãn chỉ khoanh vào mặt.
   - Thêm kỹ thuật **Occlusion Augmentation** (Random Erasing / Cutout 0.4, Mosaic 1.0, Mixup 0.2) để phát hiện người dù bị che khuất bởi đồ chơi, thành cũi, bóng bay.
   - Huấn luyện **60 Epochs** trên Kaggle GPU T4.

---

## 2. Dữ Liệu & Tài Nguyên Huấn Luyện
- **Dataset V7 Hợp Nhất**:
  - Gộp thành 1 tập duy nhất không rò rỉ dữ liệu (**Zero Data Leakage 100%**).
  - Tổng số ảnh: **6,172 ảnh** (Train: 4,607 [70%], Valid: 798 [15%], Test: 767 [15%]).
  - Kaggle Dataset Slug: `thittn/vision-v7-unified-dataset` (Trạng thái: Ready).
- **Tài khoản Kaggle**:
  - Username: `thittn` (file token: `C:\Users\thiet\Downloads\kaggle (6).json`).
  - Đường dẫn cấu hình: `C:\Users\thiet\.kaggle_accounts\thittn\kaggle.json` và `C:\Users\thiet\.kaggle\kaggle.json`.
  - Server MCP: `kaggle_thittn` đã được định nghĩa trong `C:\Users\thiet\.gemini\config\mcp_config.json`.
- **Kaggle Kernel**:
  - Kernel slug: `thittn/train-yolo11-m-full-body-v7`
  - URL: https://www.kaggle.com/code/thittn/train-yolo11-m-full-body-v7
  - Cấu trúc: 13 Cell theo chuẩn `KAGGLE_TRAINING_WORKFLOW_POLICY.md`.

---

## 3. Trạng Thái Hiện Tại Của Hệ Thống
1. **Mô hình V7 trên Kaggle**:
   - Đang hoàn thành 60 Epochs và xuất kết quả (thời lượng GPU ~160 phút).
   - Khi hoàn thành, kết quả sẽ tự động lưu về thư mục `kaggle_results/version7/` gồm `best.pt`, `training_metrics_curves.png`, `test_predictions_comparison.png`.
2. **Kịch bản kiểm thử độc lập**:
   - Script sẵn sàng: `scripts/verify_v7_on_classroom.py`.
   - File ảnh kiểm thử của người dùng: `C:\Users\thiet\.gemini\antigravity-cli\brain\30d185aa-58de-4b8e-b83e-50bde963d7f6\.user_uploaded\uploaded_media_1794978411315.jpg`.
3. **Web Demo Dashboard**:
   - Giao diện đang chạy tại cổng `http://127.0.0.1:7860` (`app.py`).
   - Đã sửa triệt để lỗi upload video (hỗ trợ mọi định dạng MP4, AVI, MKV, dung lượng 87MB+ chạy mượt mà).

---

## 4. Hướng Dẫn Sử Dụng Trong Conversation Mới
Nếu bạn mở một Conversation mới trong Antigravity để làm việc khác, bạn có thể:
1. Gõ `@PROJECT_HANDOVER_CONTEXT.md` để agent trong hội thoại mới lập tức nắm toàn bộ bối cảnh dự án mà không cần giải thích lại từ đầu.
2. Conversation cũ này vẫn tiếp tục duy trì và theo dõi tiến trình huấn luyện Kaggle cho đến khi hoàn tất và nghiệm thu.
