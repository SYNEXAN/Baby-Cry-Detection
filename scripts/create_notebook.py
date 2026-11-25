import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# --- Markdown Header ---
cells.append(nbf.v4.new_markdown_cell("""# 👶 Infant & Adult Detection using RT-DETR (Real-Time Detection Transformer)
**Dataset:** `thietnguyen/true-baby-dataset` (Kaggle Dataset)  
**Classes:** `['adult', 'baby', 'child']`  
**Model Architecture:** `RT-DETR-L` (Transformer-based Object Detection with NMS-Free Head)  
**Optimized for:** High-accuracy indoor nursery monitoring with complex occlusions, low-light, and diverse camera angles.
"""))

# --- Cell 1: Environment & GPU Checks ---
cells.append(nbf.v4.new_markdown_cell("""### 🛠️ Cell 1: Cài đặt thư viện & Kiểm tra cấu hình GPU"""))
cells.append(nbf.v4.new_code_cell("""# Cell 1: Khởi tạo môi trường & Kiểm tra GPU
!pip install -q ultralytics matplotlib seaborn pandas opencv-python

import os
import torch
import ultralytics
ultralytics.checks()

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
"""))

# --- Cell 2: Hardcoded Paths & data.yaml ---
cells.append(nbf.v4.new_markdown_cell("""### 📁 Cell 2: Cấu hình cứng đường dẫn Dataset & Tạo file `baby_rtdetr_data.yaml`"""))
cells.append(nbf.v4.new_code_cell("""# Cell 2: Thiết lập đường dẫn cứng cố định và tạo cấu hình data.yaml
import os
import yaml
import zipfile

# Đường dẫn cứng chính xác do bạn chỉ định
KAGGLE_INPUT_DIR = "/kaggle/input/datasets/thietnguyen/true-baby-dataset"

# 1. Kiểm tra nếu Kaggle đã giải nén sẵn thư mục train/images
if os.path.exists(os.path.join(KAGGLE_INPUT_DIR, "train", "images")):
    DATASET_ROOT = KAGGLE_INPUT_DIR
    print(f"✅ Sử dụng trực tiếp dataset đã giải nén tại: {DATASET_ROOT}")
else:
    # 2. Giải nén train.zip, valid.zip, test.zip sang /kaggle/working/true-baby-dataset
    DATASET_ROOT = "/kaggle/working/true-baby-dataset"
    os.makedirs(DATASET_ROOT, exist_ok=True)
    print(f"🔄 Đang giải nén các tệp zip từ {KAGGLE_INPUT_DIR} sang {DATASET_ROOT} ...")
    for split_zip in ["train.zip", "valid.zip", "test.zip"]:
        zip_file_path = os.path.join(KAGGLE_INPUT_DIR, split_zip)
        if os.path.exists(zip_file_path):
            print(f"Extracting {split_zip} ...")
            with zipfile.ZipFile(zip_file_path, "r") as z:
                z.extractall(DATASET_ROOT)
        else:
            print(f"Warning: Không tìm thấy {zip_file_path}")
    print(f"✅ Đã giải nén hoàn tất tại: {DATASET_ROOT}")

# Xác thực các thư mục cố định
assert os.path.exists(os.path.join(DATASET_ROOT, "train", "images")), f"Lỗi: Không tìm thấy {DATASET_ROOT}/train/images"
assert os.path.exists(os.path.join(DATASET_ROOT, "valid", "images")), f"Lỗi: Không tìm thấy {DATASET_ROOT}/valid/images"
assert os.path.exists(os.path.join(DATASET_ROOT, "test", "images")), f"Lỗi: Không tìm thấy {DATASET_ROOT}/test/images"

data_config = {
    'path': DATASET_ROOT,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': 3,
    'names': ['adult', 'baby', 'child']
}

YAML_PATH = "/kaggle/working/baby_rtdetr_data.yaml"
with open(YAML_PATH, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ Đã tạo file cấu hình chuẩn xác tại: {YAML_PATH}")
with open(YAML_PATH, "r") as f:
    print(f.read())
"""))

# --- Cell 3: Init RT-DETR-L ---
cells.append(nbf.v4.new_markdown_cell("""### 🤖 Cell 3: Khởi tạo mô hình RT-DETR-L (Pretrained Weights)"""))
cells.append(nbf.v4.new_code_cell("""# Cell 3: Khởi tạo mô hình Real-Time Detection Transformer (RT-DETR-L)
from ultralytics import RTDETR

# Nạp pretrained weights từ Ultralytics (tự động tải rtdetr-l.pt ~65MB)
model = RTDETR("rtdetr-l.pt")
print("✅ Mô hình RT-DETR-L đã sẵn sàng để huấn luyện!")
"""))

# --- Cell 4: Training with Advanced Augmentation ---
cells.append(nbf.v4.new_markdown_cell("""### 🚀 Cell 4: Huấn luyện RT-DETR tích hợp toàn diện Data Augmentation (Điều kiện ảnh phức tạp)"""))
cells.append(nbf.v4.new_code_cell("""# Cell 4: Huấn luyện RT-DETR với siêu tham số tối ưu cho phòng giám sát trẻ em
results = model.train(
    data=YAML_PATH,
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    optimizer="AdamW",
    lr0=0.0001,
    weight_decay=0.0001,
    
    # --- DATA AUGMENTATION PARAMETERS CHO MÔI TRƯỜNG THỰC TẾ ---
    hsv_h=0.015,         # Biến thiên màu sắc
    hsv_s=0.7,           # Giả lập giảm màu / hồng ngoại ban đêm
    hsv_v=0.5,           # Giả lập phòng ngủ tối / đèn mờ / ngược sáng
    degrees=10.0,        # Xoay nhẹ góc nghiêng
    translate=0.1,       # Dịch chuyển vị trí đối tượng
    scale=0.5,           # Đa dạng kích thước xa / gần (0.5x - 1.5x)
    shear=2.0,           # Biến dạng góc nhìn camera
    perspective=0.0008,  # Giả lập camera gắn trần nhìn chúc xuống
    fliplr=0.5,          # Lật ngang tư thế bé nằm
    flipud=0.0,          # Không lật ngược đầu
    mosaic=1.0,          # Ghép 4 ảnh học đa tỉ lệ và bối cảnh
    mixup=0.15,          # Trộn ảnh giả lập bóng mờ / che khuất
    erasing=0.4,         # XÓA MẢNG NGẪU NHIÊN: Mô phỏng chăn/gối/thành nôi che khuất
    
    project="/kaggle/working/runs_rtdetr",
    name="baby_adult_rtdetr_l",
    device=0,
    save=True,
    plots=True
)
print("✅ Hoàn thành quá trình huấn luyện RT-DETR-L!")
"""))

# --- Cell 5: Evaluation & Metrics ---
cells.append(nbf.v4.new_markdown_cell("""### 📊 Cell 5: Đánh giá mô hình trên tập Test: mAP50, mAP50-95 & Bảng Precision / Recall / F1 từng lớp"""))
cells.append(nbf.v4.new_code_cell("""# Cell 5: Đánh giá mô hình trên tập Test và xuất bảng số liệu khoa học
import pandas as pd
import numpy as np

# Đánh giá trên tập kiểm thử (Test split)
metrics = model.val(data=YAML_PATH, split="test", project="/kaggle/working/runs_rtdetr", name="eval_test")

print("=" * 65)
print("BÁO CÁO ĐỘ ĐO TOÀN DIỆN MÔ HÌNH RT-DETR TRÊN TẬP TEST")
print("=" * 65)
print(f"Overall mAP@50    : {metrics.box.map50:.4f}")
print(f"Overall mAP@50-95 : {metrics.box.map:.4f}")
print("-" * 65)

# Trích xuất Precision, Recall, F1 theo từng lớp
class_names = ['adult', 'baby', 'child']
report_data = []

for i, c_name in enumerate(class_names):
    p = float(metrics.box.p[i]) if i < len(metrics.box.p) else 0.0
    r = float(metrics.box.r[i]) if i < len(metrics.box.r) else 0.0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
    ap = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0
    
    report_data.append({
        'Class': c_name,
        'Precision': round(p, 4),
        'Recall': round(r, 4),
        'F1-Score': round(f1, 4),
        'AP@50': round(ap50, 4),
        'AP@50-95': round(ap, 4)
    })

df_report = pd.DataFrame(report_data)
display(df_report)
"""))

# --- Cell 6: Visualizations (Loss curves & Confusion Matrix) ---
cells.append(nbf.v4.new_markdown_cell("""### 📈 Cell 6: Biểu đồ Loss theo từng Epoch & Ma trận nhầm lẫn (Confusion Matrix)"""))
cells.append(nbf.v4.new_code_cell("""# Cell 6: Biểu đồ suy giảm Loss & Ma trận nhầm lẫn chuẩn hóa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

exp_dir = "/kaggle/working/runs_rtdetr/baby_adult_rtdetr_l"
csv_path = os.path.join(exp_dir, "results.csv")

if os.path.exists(csv_path):
    df_results = pd.read_csv(csv_path)
    df_results.columns = [c.strip() for c in df_results.columns]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # 1. Đồ thị Box / GIoU Loss
    if 'train/giou_loss' in df_results.columns:
        axes[0].plot(df_results['epoch'], df_results['train/giou_loss'], label='Train GIoU Loss', color='#1f77b4', lw=2)
        axes[0].plot(df_results['epoch'], df_results['val/giou_loss'], label='Val GIoU Loss', color='#ff7f0e', lw=2)
    elif 'train/box_loss' in df_results.columns:
        axes[0].plot(df_results['epoch'], df_results['train/box_loss'], label='Train Box Loss', color='#1f77b4', lw=2)
        axes[0].plot(df_results['epoch'], df_results['val/box_loss'], label='Val Box Loss', color='#ff7f0e', lw=2)
    axes[0].set_title("Quá Trình Huấn Luyện Box/GIoU Loss Qua Từng Epoch", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Đồ thị Classification Loss
    if 'train/cls_loss' in df_results.columns:
        axes[1].plot(df_results['epoch'], df_results['train/cls_loss'], label='Train Cls Loss', color='#2ca02c', lw=2)
        axes[1].plot(df_results['epoch'], df_results['val/cls_loss'], label='Val Cls Loss', color='#d62728', lw=2)
    axes[1].set_title("Quá Trình Huấn Luyện Classification Loss Qua Từng Epoch", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Hiển thị Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix)
cm_path = os.path.join(exp_dir, "confusion_matrix_normalized.png")
if os.path.exists(cm_path):
    plt.figure(figsize=(9, 9))
    img_cm = mpimg.imread(cm_path)
    plt.imshow(img_cm)
    plt.axis('off')
    plt.title("Normalized Confusion Matrix (RT-DETR Baby/Adult Detection)", fontsize=14, fontweight='bold')
    plt.show()
"""))

# --- Cell 7: Visual Inference Grid ---
cells.append(nbf.v4.new_markdown_cell("""### 🔍 Cell 7: Inference trên ảnh Test & Hiển thị Lưới Kết Quả Trực Quan (Predictions Grid)"""))
cells.append(nbf.v4.new_code_cell("""# Cell 7: Dự đoán trên ảnh Test ngẫu nhiên & Hiển thị kết quả trực quan
import glob
import random
import cv2

best_weight = os.path.join(exp_dir, "weights", "best.pt")
best_model = RTDETR(best_weight)

# Lấy 6 ảnh ngẫu nhiên từ tập Test cố định
test_images = sorted(glob.glob(f"{DATASET_ROOT}/test/images/*.jpg"))
random.seed(42)
sample_images = random.sample(test_images, min(6, len(test_images)))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, img_path in enumerate(sample_images):
    # Suy luận với ngưỡng tin cậy 0.35
    preds = best_model.predict(source=img_path, conf=0.35, imgsz=640, verbose=False)[0]
    
    # Lấy ảnh kết quả đã vẽ bounding box (BGR -> RGB)
    res_bgr = preds.plot()
    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    
    axes[idx].imshow(res_rgb)
    axes[idx].set_title(f"Test Image #{idx+1} (Detected: {len(preds.boxes)} objects)", fontsize=11, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
print(f"🎉 Checkpoint tối ưu nhất đã được lưu tại: {best_weight}")
"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.10.12'
    }
}

os.makedirs("notebooks", exist_ok=True)
out_path = os.path.join("notebooks", "train_rtdetr_baby_vision.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created notebook with {len(cells)} cells at {out_path}")
