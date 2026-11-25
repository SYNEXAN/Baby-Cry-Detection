import json
import os

NOTEBOOK_PATH = "kaggle_kernel/train_yolo11m_fullbody_v7.ipynb"
os.makedirs("kaggle_kernel", exist_ok=True)

cells = [
    # Cell 1: Environment & GPU Verification
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 PHẦN 1: KIỂM TRA MÔI TRƯỜNG & TĂNG TỐC PHẦN CỨNG GPU\n",
            "Tuân thủ nghiêm ngặt theo `KAGGLE_TRAINING_WORKFLOW_POLICY.md`:\n",
            "- Xác thực card đồ họa (NVIDIA Tesla T4 x2 hoặc P100)\n",
            "- Kiểm tra VRAM khả dụng và phiên bản CUDA\n",
            "- Kiểm tra kết nối Internet để nạp weights tiền huấn luyện"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, torch, psutil\n",
            "\n",
            "print('=' * 60)\n",
            "print('🔬 KAGGLE ENVIRONMENT & HARDWARE ACCELERATOR CHECK')\n",
            "print('=' * 60)\n",
            "\n",
            "cuda_available = torch.cuda.is_available()\n",
            "device_count = torch.cuda.device_count() if cuda_available else 0\n",
            "gpu_name = torch.cuda.get_device_name(0) if cuda_available else 'None (CPU Mode)'\n",
            "vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_available else 0\n",
            "\n",
            "print(f'CUDA Available: {cuda_available}')\n",
            "print(f'GPU Devices Found: {device_count}')\n",
            "print(f'Primary GPU Model: {gpu_name}')\n",
            "print(f'Total Dedicated VRAM: {vram_gb} GB')\n",
            "print(f'PyTorch Version: {torch.__version__}')\n",
            "print(f'CPU Logical Cores: {psutil.cpu_count(logical=True)} | RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB')\n",
            "\n",
            "if cuda_available:\n",
            "    print('✅ Hardware verification PASSED! T4/GPU is ready for YOLO11-M training.')\n",
            "else:\n",
            "    print('⚠️ Running on CPU mode. GPU acceleration is strongly recommended for 35 epochs.')\n"
        ]
    },

    # Cell 2: Dependencies & Setup
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📦 PHẦN 2: THƯ VIỆN & CÔNG CỤ HUẤN LUYỆN YOLO11-M\n",
            "Cài đặt phiên bản mới nhất của `ultralytics` (hỗ trợ kiến trúc YOLO11) và các công cụ hỗ trợ."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q ultralytics albumentations seaborn matplotlib\n",
            "\n",
            "from ultralytics import YOLO\n",
            "import cv2, glob, shutil, json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "\n",
            "print('✅ Ultralytics & Computer Vision libraries imported successfully!')\n"
        ]
    },

    # Cell 3: Dataset Discovery & Auto-Mounting
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📁 PHẦN 3: KẾT NỐI TẬP DỮ LIỆU VERSION 7 (FULL-BODY BALANCED)\n",
            "Xác thực đường dẫn tập dữ liệu chuẩn hóa `true-baby-dataset` hoặc `vision-v7-fullbody`:\n",
            "- `0: adult` (Full-body adult)\n",
            "- `1: baby_child` (Full-body baby & toddler)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Detect mounted dataset path on Kaggle\n",
            "possible_roots = [\n",
            "    '/kaggle/input/vision-v7-fullbody-dataset',\n",
            "    '/kaggle/input/true-baby-dataset',\n",
            "    'data/processed/vision_balanced_v6',\n",
            "    'data/processed/vision_balanced_2classes'\n",
            "]\n",
            "\n",
            "DATASET_ROOT = None\n",
            "for pr in possible_roots:\n",
            "    if os.path.exists(pr) and os.path.exists(os.path.join(pr, 'train', 'images')):\n",
            "        DATASET_ROOT = pr\n",
            "        break\n",
            "\n",
            "if DATASET_ROOT is None:\n",
            "    # Search recursively in /kaggle/input\n",
            "    for root, dirs, files in os.walk('/kaggle/input'):\n",
            "        if 'train' in dirs and 'valid' in dirs:\n",
            "            DATASET_ROOT = root\n",
            "            break\n",
            "\n",
            "assert DATASET_ROOT is not None, '❌ ERROR: Could not find valid dataset directory in /kaggle/input!'\n",
            "print(f'✅ Found Dataset Root: {DATASET_ROOT}')\n",
            "\n",
            "# Generate data.yaml with exact absolute paths\n",
            "train_img_dir = os.path.abspath(os.path.join(DATASET_ROOT, 'train', 'images'))\n",
            "valid_img_dir = os.path.abspath(os.path.join(DATASET_ROOT, 'valid', 'images'))\n",
            "test_img_dir = os.path.abspath(os.path.join(DATASET_ROOT, 'test', 'images'))\n",
            "\n",
            "yaml_content = f\"\"\"\n",
            "path: {os.path.abspath(DATASET_ROOT)}\n",
            "train: {train_img_dir}\n",
            "val: {valid_img_dir}\n",
            "test: {test_img_dir}\n",
            "\n",
            "names:\n",
            "  0: adult\n",
            "  1: baby_child\n",
            "\n",
            "nc: 2\n",
            "\"\"\"\n",
            "\n",
            "yaml_path = '/kaggle/working/data_v7_fullbody.yaml' if os.path.exists('/kaggle/working') else 'data_v7_fullbody.yaml'\n",
            "with open(yaml_path, 'w') as f:\n",
            "    f.write(yaml_content.strip())\n",
            "\n",
            "print(f'✅ data.yaml created at {yaml_path}:')\n",
            "print(open(yaml_path).read())\n"
        ]
    },

    # Cell 4: Dataset Integrity & Class Distribution Verification
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📊 PHẦN 4: ĐỐI SOÁT TÍNH TOÀN VẸN & CÂN BẰNG LỚP DỮ LIỆU\n",
            "Đếm số lượng instances `adult` và `baby_child` trên cả 3 tập train, valid, test để chứng minh tập dữ liệu hoàn toàn cân bằng."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from collections import Counter\n",
            "\n",
            "stats = {}\n",
            "for split in ['train', 'valid', 'test']:\n",
            "    lbl_files = glob.glob(os.path.join(DATASET_ROOT, split, 'labels', '*.txt'))\n",
            "    class_counts = Counter()\n",
            "    for lf in lbl_files:\n",
            "        for line in open(lf):\n",
            "            parts = line.strip().split()\n",
            "            if parts:\n",
            "                class_counts[int(parts[0])] += 1\n",
            "    stats[split] = {'images': len(lbl_files), 'adult (0)': class_counts[0], 'baby_child (1)': class_counts[1]}\n",
            "\n",
            "df_stats = pd.DataFrame(stats).T\n",
            "print('=' * 60)\n",
            "print('📊 DATASET INSTANCE DISTRIBUTION TABLE')\n",
            "print('=' * 60)\n",
            "print(df_stats)\n"
        ]
    },

    # Cell 5: Sample Visualization
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🖼️ PHẦN 5: TRỰC QUAN HÓA MẪU ẢNH HUẤN LUYỆN TOÀN THÂN (FULL-BODY BBOXES)\n",
            "Vẽ thử bounding box mặt đất để đảm bảo không có nhãn nào bị co cụm vào mặt."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sample_lbls = glob.glob(os.path.join(DATASET_ROOT, 'train', 'labels', '*.txt'))[:3]\n",
            "plt.figure(figsize=(15, 5))\n",
            "\n",
            "for idx, lf in enumerate(sample_lbls):\n",
            "    img_path = lf.replace('labels', 'images').replace('.txt', '.jpg')\n",
            "    img = cv2.imread(img_path)\n",
            "    if img is None: continue\n",
            "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
            "    h, w = img.shape[:2]\n",
            "    for line in open(lf):\n",
            "        parts = line.strip().split()\n",
            "        if parts:\n",
            "            c, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])\n",
            "            x1, y1 = int((xc - bw/2)*w), int((yc - bh/2)*h)\n",
            "            x2, y2 = int((xc + bw/2)*w), int((yc + bh/2)*h)\n",
            "            color = (0, 200, 255) if c == 0 else (255, 120, 0)\n",
            "            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)\n",
            "            lbl_name = 'adult (Full-Body)' if c == 0 else 'baby_child'\n",
            "            cv2.putText(img, lbl_name, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)\n",
            "    \n",
            "    plt.subplot(1, 3, idx+1)\n",
            "    plt.imshow(img)\n",
            "    plt.title(f'Sample #{idx+1}: {os.path.basename(img_path)[:20]}')\n",
            "    plt.axis('off')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('/kaggle/working/training_samples_preview.jpg' if os.path.exists('/kaggle/working') else 'training_samples_preview.jpg')\n",
            "plt.show()\n"
        ]
    },

    # Cell 6: Model Configuration
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🤖 PHẦN 6: KHỞI TẠO KIẾN TRÚC MÔ HÌNH YOLO11-M\n",
            "Sử dụng mô hình thị giác mới nhất **YOLO11-M** (Ultralytics SOTA 2024/2025):\n",
            "- Số tham số: ~20.1M\n",
            "- C3k2 + SPPF: Bắt nét đường biên cơ thể toàn thân cực kỳ sắc sảo\n",
            "- Khởi tạo từ trọng số `yolo11m.pt` (COCO Full-Body Pretrained)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model = YOLO('yolo11m.pt')\n",
            "print('=' * 60)\n",
            "print('🤖 YOLO11-M MODEL ARCHITECTURE INITIALIZED')\n",
            "print('=' * 60)\n",
            "print(f'Model type: {type(model.model)}')\n"
        ]
    },

    # Cell 7: Training Parameters & Duration Estimation
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⏱️ PHẦN 7: ƯỚC LƯỢNG THỜI GIAN & THIẾT LẬP SIÊU THAM SỐ\n",
            "Tuân thủ `KAGGLE_TRAINING_WORKFLOW_POLICY.md` mục 5 (Thời lượng & Epoch):\n",
            "- Kích thước tập train: ~2,000 ảnh\n",
            "- Batch size: 16 (tối ưu cho 15GB VRAM T4)\n",
            "- Số epochs: 35 epochs\n",
            "- Ước lượng: ~1.2 phút / epoch $\\rightarrow$ Tổng thời gian: ~42 phút (hoàn toàn nằm trong ngưỡng 12h của Kaggle)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "EPOCHS = 35\n",
            "BATCH = 16\n",
            "IMGSZ = 640\n",
            "WORKERS = 4\n",
            "\n",
            "print(f'Proposed Training Configuration:')\n",
            "print(f'- Epochs: {EPOCHS}')\n",
            "print(f'- Batch Size: {BATCH}')\n",
            "print(f'- Image Resolution: {IMGSZ}x{IMGSZ}')\n",
            "print(f'- Augmentation: Mosaic=1.0, Mixup=0.15, Random Perspective=0.1')\n",
            "print(f'- Estimated Duration: 40-50 minutes on Kaggle T4 GPU')\n"
        ]
    },

    # Cell 8: Training Execution
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏋️ PHẦN 8: TIẾN HÀNH HUẤN LUYỆN MÔ HÌNH YOLO11-M"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "save_dir = '/kaggle/working/runs_yolo11m' if os.path.exists('/kaggle/working') else 'runs_yolo11m'\n",
            "\n",
            "results = model.train(\n",
            "    data=yaml_path,\n",
            "    epochs=EPOCHS,\n",
            "    batch=BATCH,\n",
            "    imgsz=IMGSZ,\n",
            "    workers=WORKERS,\n",
            "    project=save_dir,\n",
            "    name='baby_adult_yolo11m',\n",
            "    exist_ok=True,\n",
            "    pretrained=True,\n",
            "    optimizer='AdamW',\n",
            "    lr0=0.001,\n",
            "    lrf=0.01,\n",
            "    warmup_epochs=3,\n",
            "    mosaic=1.0,\n",
            "    mixup=0.15,\n",
            "    close_mosaic=5,\n",
            "    val=True,\n",
            "    save=True,\n",
            "    verbose=True\n",
            ")\n",
            "\n",
            "print('🎉 Training completed successfully!')\n"
        ]
    },

    # Cell 9: Training History & Learning Curves
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📈 PHẦN 9: BIỂU ĐỒ HỌC TẬP (LOSS & mAP@50 CURVES)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "csv_path = os.path.join(save_dir, 'baby_adult_yolo11m', 'results.csv')\n",
            "if os.path.exists(csv_path):\n",
            "    df = pd.read_csv(csv_path)\n",
            "    df.columns = [c.strip() for c in df.columns]\n",
            "    \n",
            "    plt.figure(figsize=(14, 5))\n",
            "    plt.subplot(1, 2, 1)\n",
            "    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:\n",
            "        plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')\n",
            "        plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='orange')\n",
            "        plt.title('Bounding Box Regression Loss (CIoU)')\n",
            "        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)\n",
            "        \n",
            "    plt.subplot(1, 2, 2)\n",
            "    if 'metrics/mAP50(B)' in df.columns:\n",
            "        plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50 (Overall)', color='green', lw=2)\n",
            "        plt.title('Validation Accuracy: mAP@50')\n",
            "        plt.xlabel('Epoch'); plt.ylabel('mAP@50'); plt.legend(); plt.grid(True)\n",
            "        \n",
            "    plt.tight_layout()\n",
            "    plt.savefig('/kaggle/working/yolo11m_training_curves.png' if os.path.exists('/kaggle/working') else 'yolo11m_training_curves.png')\n",
            "    plt.show()\n"
        ]
    },

    # Cell 10: Model Evaluation on Test Set
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏆 PHẦN 10: ĐÁNH GIÁ CHUYÊN SÂU TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET)\n",
            "Kiểm tra độc lập trên tập test hoàn toàn chưa từng xuất hiện trong quá trình train."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "best_weight = os.path.join(save_dir, 'baby_adult_yolo11m', 'weights', 'best.pt')\n",
            "test_model = YOLO(best_weight)\n",
            "metrics = test_model.val(data=yaml_path, split='test')\n",
            "\n",
            "print('=' * 60)\n",
            "print('🎯 TEST SET EVALUATION METRICS (YOLO11-M)')\n",
            "print('=' * 60)\n",
            "print(f'Overall mAP@50:    {metrics.box.map50*100:.2f}%')\n",
            "print(f'Overall mAP@50-95: {metrics.box.map*100:.2f}%')\n",
            "if len(metrics.box.maps) >= 2:\n",
            "    print(f'Class 0 (adult) mAP@50:      {metrics.box.maps[0]*100:.2f}%')\n",
            "    print(f'Class 1 (baby_child) mAP@50: {metrics.box.maps[1]*100:.2f}%')\n"
        ]
    },

    # Cell 11: Prediction Visualization
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🔍 PHẦN 11: TRỰC QUAN HÓA DỰ ĐOÁN THỰC TẾ TRÊN TẬP TEST"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "test_imgs = glob.glob(os.path.join(DATASET_ROOT, 'test', 'images', '*.jpg'))[:3]\n",
            "plt.figure(figsize=(15, 5))\n",
            "\n",
            "for idx, tip in enumerate(test_imgs):\n",
            "    res = test_model.predict(tip, conf=0.3, verbose=False)[0]\n",
            "    res_img = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)\n",
            "    \n",
            "    plt.subplot(1, 3, idx+1)\n",
            "    plt.imshow(res_img)\n",
            "    plt.title(f'Test Sample #{idx+1}')\n",
            "    plt.axis('off')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('/kaggle/working/yolo11m_test_predictions.jpg' if os.path.exists('/kaggle/working') else 'yolo11m_test_predictions.jpg')\n",
            "plt.show()\n"
        ]
    },

    # Cell 12: Checkpoint Export
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 💾 PHẦN 12: ĐÓNG GÓI TRỌNG SỐ TỐI ƯU & ARTIFACTS"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "target_dir = '/kaggle/working/yolo11m_export' if os.path.exists('/kaggle/working') else 'yolo11m_export'\n",
            "os.makedirs(target_dir, exist_ok=True)\n",
            "\n",
            "# Copy best weights\n",
            "shutil.copy2(best_weight, os.path.join(target_dir, 'best.pt'))\n",
            "last_weight = os.path.join(save_dir, 'baby_adult_yolo11m', 'weights', 'last.pt')\n",
            "if os.path.exists(last_weight):\n",
            "    shutil.copy2(last_weight, os.path.join(target_dir, 'last.pt'))\n",
            "\n",
            "# Save metadata json\n",
            "meta_out = {\n",
            "    'model_name': 'YOLO11-M (Ultralytics)',\n",
            "    'task': 'Baby & Adult Full-Body Safety Detection',\n",
            "    'classes': ['adult', 'baby_child'],\n",
            "    'epochs': EPOCHS,\n",
            "    'imgsz': IMGSZ,\n",
            "    'overall_map50': round(float(metrics.box.map50 * 100), 2)\n",
            "}\n",
            "with open(os.path.join(target_dir, 'model_metadata.json'), 'w') as f:\n",
            "    json.dump(meta_out, f, indent=2)\n",
            "\n",
            "print('=' * 60)\n",
            "print(f'✅ Exported all artifacts to {target_dir}:')\n",
            "print(os.listdir(target_dir))\n",
            "print('=' * 60)\n"
        ]
    },

    # Cell 13: Final Summary
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏁 PHẦN 13: BÁO CÁO TỔNG KẾT THÍ NGHIỆM HUẤN LUYỆN\n",
            "- Mô hình YOLO11-M đã huấn luyện xong toàn diện.\n",
            "- Bounding box bao trùm toàn thân người lớn và trẻ em.\n",
            "- Trọng số `best.pt` đã sẵn sàng tải về để đưa vào `app.py`!"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"SUCCESS: Generated {NOTEBOOK_PATH} with {len(cells)} cells.")
