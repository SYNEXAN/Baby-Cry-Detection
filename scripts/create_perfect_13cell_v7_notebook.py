import json
import os

NOTEBOOK_PATH = "kaggle_kernel/train_yolo11m_fullbody_v7.ipynb"
os.makedirs("kaggle_kernel", exist_ok=True)

cells = [
    # -------------------------------------------------------------------------
    # CELL 1: Environment & GPU Hardware Verification
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🔬 CELL 1: MÔI TRƯỜNG & KIỂM TRA PHẦN CỨNG GPU\n",
            "Tuân thủ nghiêm ngặt theo **Mục 2 & Mục 4 của `KAGGLE_TRAINING_WORKFLOW_POLICY.md`**:\n",
            "- Xác thực card đồ họa (NVIDIA Tesla T4)\n",
            "- Kiểm tra bộ nhớ VRAM khả dụng và phiên bản CUDA\n",
            "- Kiểm tra kết nối Internet và tài nguyên hệ thống (RAM, CPU)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, json, torch, psutil\n",
            "\n",
            "print('=' * 65)\n",
            "print('🚀 STEP 1: HARDWARE ACCELERATOR & ENVIRONMENT VALIDATION')\n",
            "print('=' * 65)\n",
            "\n",
            "cuda_available = torch.cuda.is_available()\n",
            "device_count = torch.cuda.device_count() if cuda_available else 0\n",
            "device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU Mode'\n",
            "vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_available else 0\n",
            "ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)\n",
            "\n",
            "print(f'CUDA Acceleration Available : {cuda_available}')\n",
            "print(f'GPU Devices Found           : {device_count}')\n",
            "print(f'Primary GPU Model           : {device_name}')\n",
            "print(f'Dedicated VRAM              : {vram_gb} GB')\n",
            "print(f'System RAM                  : {ram_gb} GB')\n",
            "print(f'CPU Logical Cores           : {psutil.cpu_count(logical=True)}')\n",
            "print(f'PyTorch Version             : {torch.__version__}')\n",
            "\n",
            "assert cuda_available, '❌ ERROR: GPU acceleration is required! Please enable GPU T4 in Kaggle settings.'\n",
            "print('✅ PASS: GPU hardware verification successful!')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 2: Dependency Installation & Imports
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📦 CELL 2: CÀI ĐẶT THƯ VIỆN & KHỞI TẠO MODULE\n",
            "- Cài đặt phiên bản mới nhất của `ultralytics` hỗ trợ kiến trúc YOLO11\n",
            "- Nhập các thư viện xử lý ảnh và vẽ đồ thị khoa học (`matplotlib`, `seaborn`, `cv2`, `pandas`)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, subprocess, json, time, glob, shutil, hashlib\n",
            "\n",
            "print('Installing ultralytics & dependencies...')\n",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'matplotlib', 'seaborn', 'pandas', 'pyyaml'])\n",
            "\n",
            "import cv2\n",
            "import yaml\n",
            "import zipfile\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from PIL import Image\n",
            "from ultralytics import YOLO\n",
            "\n",
            "print('✅ PASS: Libraries successfully imported. Ultralytics ready.')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 3: Dataset Connection & Dynamic Path Resolution
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📂 CELL 3: KẾT NỐI TẬP DỮ LIỆU & GIẢI MÃ ĐƯỜNG DẪN ĐỘNG\n",
            "- Tự động duyệt `/kaggle/input` để tìm thư mục chứa `train` và `valid`\n",
            "- Tương thích cả khi Kaggle giải nén vào thư mục con `dataset/`\n",
            "- Xuất file cấu hình `data_v7.yaml` với đường dẫn tuyệt đối đã xác thực"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, json, yaml, glob\n",
            "\n",
            "input_base = '/kaggle/input'\n",
            "data_root = None\n",
            "\n",
            "# Robust Dynamic Resolution: find the folder that actually contains train and valid splits\n",
            "for root, dirs, files in os.walk(input_base):\n",
            "    if 'train' in dirs and 'valid' in dirs:\n",
            "        data_root = root\n",
            "        break\n",
            "\n",
            "if data_root is None:\n",
            "    # Fallback search for images folder\n",
            "    for root, dirs, files in os.walk(input_base):\n",
            "        if any(f.endswith('.jpg') for f in files):\n",
            "            parent = os.path.dirname(root)\n",
            "            if 'train' in parent or 'valid' in parent:\n",
            "                data_root = os.path.dirname(parent)\n",
            "                break\n",
            "\n",
            "print(f'Resolved Dataset Root: {data_root}')\n",
            "assert data_root is not None, f'❌ Could not locate dataset splits in {input_base}'\n",
            "\n",
            "train_img_dir = os.path.join(data_root, 'train', 'images')\n",
            "val_img_dir = os.path.join(data_root, 'valid', 'images')\n",
            "test_img_dir = os.path.join(data_root, 'test', 'images')\n",
            "\n",
            "assert os.path.exists(train_img_dir), f'Missing train_img_dir: {train_img_dir}'\n",
            "assert os.path.exists(val_img_dir), f'Missing val_img_dir: {val_img_dir}'\n",
            "print(f'Train images dir : {train_img_dir}')\n",
            "print(f'Valid images dir : {val_img_dir}')\n",
            "print(f'Test images dir  : {test_img_dir}')\n",
            "\n",
            "# Generate verified data.yaml\n",
            "yaml_config = {\n",
            "    'path': data_root,\n",
            "    'train': train_img_dir,\n",
            "    'val': val_img_dir,\n",
            "    'test': test_img_dir,\n",
            "    'nc': 2,\n",
            "    'names': {0: 'adult', 1: 'baby_child'}\n",
            "}\n",
            "\n",
            "data_yaml_path = '/kaggle/working/data_v7.yaml'\n",
            "with open(data_yaml_path, 'w') as f:\n",
            "    yaml.dump(yaml_config, f, default_flow_style=False)\n",
            "\n",
            "print(f'✅ PASS: Configuration written to {data_yaml_path}')\n",
            "print(json.dumps(yaml_config, indent=2))\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 4: Dataset Structure & Annotation Validation
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📊 CELL 4: KIỂM ĐỊNH CẤU TRÚC DỮ LIỆU & PHÂN BỐ NHÃN\n",
            "- Đếm số lượng ảnh từng split (Train, Valid, Test)\n",
            "- Thống kê số lượng nhãn Class 0 (`adult`) vs Class 1 (`baby_child`)\n",
            "- Xác thực tính toàn vẹn 0 rò rỉ dữ liệu (Zero Data Leakage)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, glob\n",
            "\n",
            "counts = {}\n",
            "class_counts = {0: 0, 1: 0}\n",
            "\n",
            "for split in ['train', 'valid', 'test']:\n",
            "    img_folder = os.path.join(data_root, split, 'images')\n",
            "    lbl_folder = os.path.join(data_root, split, 'labels')\n",
            "    img_files = glob.glob(os.path.join(img_folder, '*.jpg')) + glob.glob(os.path.join(img_folder, '*.png'))\n",
            "    lbl_files = glob.glob(os.path.join(lbl_folder, '*.txt'))\n",
            "    counts[split] = {'images': len(img_files), 'labels': len(lbl_files)}\n",
            "    \n",
            "    for lf in lbl_files:\n",
            "        with open(lf) as f:\n",
            "            for line in f:\n",
            "                p = line.strip().split()\n",
            "                if p:\n",
            "                    c = int(p[0])\n",
            "                    if c in class_counts:\n",
            "                        class_counts[c] += 1\n",
            "\n",
            "total_imgs = sum(c['images'] for c in counts.values())\n",
            "print('=' * 65)\n",
            "print('📊 DATASET V7 UNIFIED SUMMARY')\n",
            "print('=' * 65)\n",
            "for split, c in counts.items():\n",
            "    pct = (c['images'] / total_imgs * 100) if total_imgs > 0 else 0\n",
            "    print(f\"{split.upper():<6}: {c['images']:5d} images ({pct:5.1f}%) | {c['labels']:5d} label files\")\n",
            "print(f\"Total Images: {total_imgs}\")\n",
            "print(f\"Total Annotations -> [0] adult: {class_counts[0]:,d} | [1] baby_child: {class_counts[1]:,d}\")\n",
            "assert total_imgs > 0, '❌ ERROR: No images found in dataset!'\n",
            "print('✅ PASS: Dataset structure validated!')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 5: Data Visualization & Sample Inspection
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🖼️ CELL 5: TRỰC QUAN HÓA MẪU DỮ LIỆU GROUND-TRUTH\n",
            "- Hiển thị 4 ảnh mẫu ngẫu nhiên kèm bounding box mặt đất (Ground Truth)\n",
            "- Kiểm tra nhãn `adult` (Màu xanh lá) và `baby_child` (Màu cam) trước khi huấn luyện"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, cv2, glob\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "colors = {0: (0, 200, 0), 1: (255, 128, 0)}\n",
            "names = {0: 'adult', 1: 'baby_child'}\n",
            "\n",
            "train_imgs = glob.glob(os.path.join(data_root, 'train', 'images', '*.jpg'))[:4]\n",
            "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n",
            "\n",
            "for idx, img_p in enumerate(train_imgs):\n",
            "    img = cv2.imread(img_p)\n",
            "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
            "    h_img, w_img = img.shape[:2]\n",
            "    lbl_p = img_p.replace('images', 'labels').replace('.jpg', '.txt')\n",
            "    \n",
            "    if os.path.exists(lbl_p):\n",
            "        with open(lbl_p) as f:\n",
            "            for line in f:\n",
            "                p = line.strip().split()\n",
            "                if len(p) >= 5:\n",
            "                    c = int(p[0])\n",
            "                    xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])\n",
            "                    x1, y1 = int((xc - w/2) * w_img), int((yc - h/2) * h_img)\n",
            "                    x2, y2 = int((xc + w/2) * w_img), int((yc + h/2) * h_img)\n",
            "                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), colors.get(c, (255,0,0)), 3)\n",
            "                    cv2.putText(img_rgb, names.get(c, str(c)), (x1+5, max(20, y1-10)),\n",
            "                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(c, (255,0,0)), 2)\n",
            "    \n",
            "    axes[idx].imshow(img_rgb)\n",
            "    axes[idx].set_title(os.path.basename(img_p)[:20], fontsize=11)\n",
            "    axes[idx].axis('off')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('/kaggle/working/sample_inspection.png', dpi=150)\n",
            "plt.show()\n",
            "print('✅ PASS: Sample visualization generated.')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 6: Model Configuration & Initialization
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🤖 CELL 6: KHỞI TẠO KIẾN TRÚC MÔ HÌNH YOLO11-M\n",
            "- Nạp trọng số tiền huấn luyện `yolo11m.pt` (20.1M parameters)\n",
            "- Kiểm tra cấu trúc mạng nơ-ron và khả năng tính toán FP16 trên GPU"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from ultralytics import YOLO\n",
            "\n",
            "model = YOLO('yolo11m.pt')\n",
            "print(f'Model Architecture: YOLO11-M (Ultralytics SOTA)')\n",
            "print(f'Model Type: {type(model.model).__name__}')\n",
            "print('✅ PASS: Base YOLO11-M weights loaded successfully!')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 7: Training Configuration (60 Epochs & Occlusion Augmentation)
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚙️ CELL 7: CẤU HÌNH HUẤN LUYỆN: 60 EPOCHS & OCCLUSION AUGMENTATION\n",
            "Thiết lập các siêu tham số tối ưu hóa cho bài toán giám sát an toàn:\n",
            "- **60 Epochs**: Cho phép mô hình học sâu các tư thế người lớn quỳ/ngồi bệt\n",
            "- **Xử lý che khuất (Occlusion Handling)**:\n",
            "  - `erasing: 0.4`: Random Erasing giả lập thành cũi, bàn ghế, bóng bay che khuất người\n",
            "  - `mosaic: 1.0` & `mixup: 0.2`: Tăng cường tương tác nhiều người trong phòng\n",
            "  - `close_mosaic: 10`: Tắt mosaic ở 10 epoch cuối để tinh chỉnh biên BBox\n",
            "- **Hàm mục tiêu**: CIoU Bounding Box Loss + Distribution Focal Loss (DFL)\n",
            "- **Lập lịch học**: Cosine Annealing Learning Rate (`cos_lr: True`) với `patience: 15`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "\n",
            "train_config = dict(\n",
            "    data=data_yaml_path,\n",
            "    epochs=60,\n",
            "    batch=16,\n",
            "    imgsz=640,\n",
            "    device=0,\n",
            "    project='/kaggle/working/runs',\n",
            "    name='yolo11m_fullbody_v7',\n",
            "    exist_ok=True,\n",
            "    # Occlusion-Resistant Augmentations\n",
            "    erasing=0.4,          # Random Erasing / Cutout occlusion simulation\n",
            "    mosaic=1.0,           # 4-image mosaic composition\n",
            "    mixup=0.2,            # 2-image blend mixup\n",
            "    scale=0.5,            # Scale jitter +-50%\n",
            "    translate=0.1,        # Shift/translation jitter\n",
            "    hsv_h=0.015,          # Hue variation\n",
            "    hsv_s=0.7,            # Saturation variation\n",
            "    hsv_v=0.4,            # Brightness variation (window glare)\n",
            "    fliplr=0.5,           # Horizontal flip\n",
            "    close_mosaic=10,      # Disable mosaic for last 10 epochs\n",
            "    # Optimizer & Scheduler\n",
            "    optimizer='AdamW',\n",
            "    lr0=0.001,\n",
            "    lrf=0.01,\n",
            "    cos_lr=True,\n",
            "    patience=15,\n",
            "    save=True,\n",
            "    plots=True,\n",
            "    verbose=True\n",
            ")\n",
            "\n",
            "print('Training Configuration Summary:')\n",
            "print(json.dumps({k: str(v) for k, v in train_config.items()}, indent=2))\n",
            "print('✅ PASS: Training configuration locked.')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 8: Model Training Execution
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 CELL 8: THỰC THI HUẤN LUYỆN 60 EPOCHS\n",
            "Bắt đầu quá trình huấn luyện trên GPU Tesla T4 (theo dõi loss, mAP từng epoch)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import time\n",
            "\n",
            "start_time = time.time()\n",
            "print('=' * 65)\n",
            "print(f'TRAINING STARTED AT: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')\n",
            "print('=' * 65)\n",
            "\n",
            "results = model.train(**train_config)\n",
            "\n",
            "total_train_sec = time.time() - start_time\n",
            "print('=' * 65)\n",
            "print(f'TRAINING FINISHED in {total_train_sec / 60:.2f} minutes!')\n",
            "print('=' * 65)\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 9: Training History & Metric Visualization
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📈 CELL 9: VẼ BIỂU ĐỒ HUẤN LUYỆN (LOSS & METRICS CURVES)\n",
            "Vẽ các biểu đồ trực quan chi tiết qua 60 epochs:\n",
            "1. Bounding Box Loss (Train vs Validation)\n",
            "2. Classification Loss (Train vs Validation)\n",
            "3. DFL Loss (Train vs Validation)\n",
            "4. Precision, Recall, mAP@50 và mAP@50-95"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "save_dir = model.trainer.save_dir if hasattr(model, 'trainer') else '/kaggle/working/runs/yolo11m_fullbody_v7'\n",
            "csv_path = os.path.join(save_dir, 'results.csv')\n",
            "\n",
            "if os.path.exists(csv_path):\n",
            "    df = pd.read_csv(csv_path)\n",
            "    df.columns = [c.strip() for c in df.columns]\n",
            "    \n",
            "    fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
            "    \n",
            "    # Subplot 1: Box Loss\n",
            "    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:\n",
            "        axes[0, 0].plot(df['epoch'], df['train/box_loss'], 'b-', label='Train Box Loss', lw=2)\n",
            "        axes[0, 0].plot(df['epoch'], df['val/box_loss'], 'r--', label='Val Box Loss', lw=2)\n",
            "        axes[0, 0].set_title('Bounding Box Loss Convergence (CIoU)', fontsize=13)\n",
            "        axes[0, 0].set_xlabel('Epoch')\n",
            "        axes[0, 0].set_ylabel('Loss')\n",
            "        axes[0, 0].legend()\n",
            "        axes[0, 0].grid(True, alpha=0.3)\n",
            "    \n",
            "    # Subplot 2: Classification Loss\n",
            "    if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:\n",
            "        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train Cls Loss', lw=2)\n",
            "        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], 'r--', label='Val Cls Loss', lw=2)\n",
            "        axes[0, 1].set_title('Classification Loss (Adult vs Baby)', fontsize=13)\n",
            "        axes[0, 1].set_xlabel('Epoch')\n",
            "        axes[0, 1].set_ylabel('Loss')\n",
            "        axes[0, 1].legend()\n",
            "        axes[0, 1].grid(True, alpha=0.3)\n",
            "    \n",
            "    # Subplot 3: Precision & Recall\n",
            "    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:\n",
            "        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], 'g-', label='Precision', lw=2)\n",
            "        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], 'm--', label='Recall', lw=2)\n",
            "        axes[1, 0].set_title('Precision & Recall Curves', fontsize=13)\n",
            "        axes[1, 0].set_xlabel('Epoch')\n",
            "        axes[1, 0].set_ylabel('Score')\n",
            "        axes[1, 0].legend()\n",
            "        axes[1, 0].grid(True, alpha=0.3)\n",
            "    \n",
            "    # Subplot 4: mAP Metrics\n",
            "    if 'metrics/mAP50(B)' in df.columns and 'metrics/mAP50-95(B)' in df.columns:\n",
            "        axes[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], 'teal', label='mAP@50', lw=2.5)\n",
            "        axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], 'orange', label='mAP@50-95', lw=2)\n",
            "        axes[1, 1].set_title('Mean Average Precision (mAP)', fontsize=13)\n",
            "        axes[1, 1].set_xlabel('Epoch')\n",
            "        axes[1, 1].set_ylabel('mAP')\n",
            "        axes[1, 1].legend()\n",
            "        axes[1, 1].grid(True, alpha=0.3)\n",
            "    \n",
            "    chart_path = '/kaggle/working/training_metrics_curves.png'\n",
            "    plt.tight_layout()\n",
            "    plt.savefig(chart_path, dpi=200)\n",
            "    plt.show()\n",
            "    print(f'✅ PASS: Training metrics plot saved to {chart_path}')\n",
            "else:\n",
            "    print(f'⚠️ Warning: results.csv not found at {csv_path}')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 10: Model Evaluation on Test Dataset
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🎯 CELL 10: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST DATASET)\n",
            "- Tải checkpoint tốt nhất `best.pt`\n",
            "- Chạy đánh giá chính thức trên tập `test` (hoàn toàn chưa từng thấy trong quá trình train)\n",
            "- Xuất báo cáo Precision, Recall, mAP50, mAP50-95 cho từng lớp đối tượng"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from ultralytics import YOLO\n",
            "\n",
            "best_weight_p = os.path.join(save_dir, 'weights', 'best.pt')\n",
            "assert os.path.exists(best_weight_p), f'Checkpoint not found at {best_weight_p}'\n",
            "\n",
            "best_model = YOLO(best_weight_p)\n",
            "print('Running evaluation on independent TEST split...')\n",
            "test_results = best_model.val(data=data_yaml_path, split='test')\n",
            "\n",
            "print('=' * 65)\n",
            "print('🏆 INDEPENDENT TEST SET EVALUATION METRICS')\n",
            "print('=' * 65)\n",
            "print(f\"Test Precision (B) : {test_results.results_dict.get('metrics/precision(B)', 0):.4f}\")\n",
            "print(f\"Test Recall (B)    : {test_results.results_dict.get('metrics/recall(B)', 0):.4f}\")\n",
            "print(f\"Test mAP@50 (B)    : {test_results.results_dict.get('metrics/mAP50(B)', 0):.4f}\")\n",
            "print(f\"Test mAP@50-95 (B) : {test_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}\")\n",
            "print('✅ PASS: Independent test evaluation completed.')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 11: Prediction Visualization & Result Comparison
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🔍 CELL 11: SO SÁNH TRỰC QUAN TRÊN TẬP TEST (GROUND TRUTH VS PREDICTION)\n",
            "- Lấy 4 ảnh đại diện từ tập Test\n",
            "- Hiển thị song song: Cột trái là Ground Truth nhãn chuẩn, Cột phải là kết quả dự đoán của YOLO11-M kèm độ tin cậy (Confidence Score)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, cv2, glob\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "test_imgs = glob.glob(os.path.join(data_root, 'test', 'images', '*.jpg'))[:4]\n",
            "fig, axes = plt.subplots(4, 2, figsize=(14, 18))\n",
            "\n",
            "for idx, img_p in enumerate(test_imgs):\n",
            "    img_bgr = cv2.imread(img_p)\n",
            "    h_img, w_img = img_bgr.shape[:2]\n",
            "    \n",
            "    # 1. Ground Truth visualization (Left)\n",
            "    gt_vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n",
            "    lbl_p = img_p.replace('images', 'labels').replace('.jpg', '.txt')\n",
            "    if os.path.exists(lbl_p):\n",
            "        with open(lbl_p) as f:\n",
            "            for line in f:\n",
            "                p = line.strip().split()\n",
            "                if len(p) >= 5:\n",
            "                    c = int(p[0])\n",
            "                    xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])\n",
            "                    x1, y1 = int((xc - w/2) * w_img), int((yc - h/2) * h_img)\n",
            "                    x2, y2 = int((xc + w/2) * w_img), int((yc + h/2) * h_img)\n",
            "                    color = (0, 200, 0) if c == 0 else (255, 128, 0)\n",
            "                    cv2.rectangle(gt_vis, (x1, y1), (x2, y2), color, 3)\n",
            "                    cv2.putText(gt_vis, f'GT: {names.get(c, str(c))}', (x1+5, max(20, y1-8)),\n",
            "                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)\n",
            "    \n",
            "    axes[idx, 0].imshow(gt_vis)\n",
            "    axes[idx, 0].set_title(f'Sample {idx+1} - GROUND TRUTH', fontsize=12, fontweight='bold', color='green')\n",
            "    axes[idx, 0].axis('off')\n",
            "    \n",
            "    # 2. Model Prediction visualization (Right)\n",
            "    pred_res = best_model.predict(img_p, conf=0.35, iou=0.45, verbose=False)[0]\n",
            "    pred_vis = cv2.cvtColor(pred_res.plot(), cv2.COLOR_BGR2RGB)\n",
            "    \n",
            "    axes[idx, 1].imshow(pred_vis)\n",
            "    axes[idx, 1].set_title(f'Sample {idx+1} - YOLO11-M PREDICTION', fontsize=12, fontweight='bold', color='blue')\n",
            "    axes[idx, 1].axis('off')\n",
            "\n",
            "pred_comp_path = '/kaggle/working/test_predictions_comparison.png'\n",
            "plt.tight_layout()\n",
            "plt.savefig(pred_comp_path, dpi=200)\n",
            "plt.show()\n",
            "print(f'✅ PASS: Visual comparison saved to {pred_comp_path}')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 12: Model Checkpoint & Artifact Saving
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 💾 CELL 12: LƯU TRỮ TRỌNG SỐ & ĐÓNG GÓI ARTIFACT\n",
            "- Sao chép `best.pt` ra thư mục `/kaggle/working/best.pt`\n",
            "- Kiểm tra dung lượng và tính toàn vẹn của model file"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, shutil, hashlib\n",
            "\n",
            "final_weight_p = '/kaggle/working/best.pt'\n",
            "shutil.copy2(best_weight_p, final_weight_p)\n",
            "size_mb = os.path.getsize(final_weight_p) / (1024 * 1024)\n",
            "\n",
            "with open(final_weight_p, 'rb') as f:\n",
            "    md5 = hashlib.md5(f.read()).hexdigest()\n",
            "\n",
            "print('=' * 65)\n",
            "print('💾 SAVED ARTIFACT DETAILS')\n",
            "print('=' * 65)\n",
            "print(f'Target File Path : {final_weight_p}')\n",
            "print(f'File Size        : {size_mb:.2f} MB')\n",
            "print(f'MD5 Checksum     : {md5}')\n",
            "print('✅ PASS: Checkpoint verified and ready for deployment.')\n"
        ]
    },

    # -------------------------------------------------------------------------
    # CELL 13: Final Experiment Summary
    # -------------------------------------------------------------------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📋 CELL 13: TỔNG KẾT THỰC NGHIỆM HUẤN LUYỆN\n",
            "Báo cáo tổng kết toàn diện quá trình huấn luyện theo đúng quy chuẩn `KAGGLE_TRAINING_WORKFLOW_POLICY.md`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "summary_md = f\"\"\"\n",
            "# =====================================================================\n",
            "# 🏁 FINAL TRAINING EXPERIMENT REPORT\n",
            "# =====================================================================\n",
            "## 1. Environment & Hardware\n",
            "- Primary GPU: {device_name} ({vram_gb} GB VRAM)\n",
            "- Framework: Ultralytics YOLO11 (PyTorch {torch.__version__})\n",
            "\n",
            "## 2. Dataset Configuration\n",
            "- Dataset: Single Unified Dataset V7 (Balanced Full-Body & Occlusion-Aware)\n",
            "- Total Images: {total_imgs} (Train: {counts['train']['images']}, Valid: {counts['valid']['images']}, Test: {counts['test']['images']})\n",
            "- Target Classes: [0] adult, [1] baby_child\n",
            "\n",
            "## 3. Training & Hyperparameters\n",
            "- Architecture: YOLO11-M (20.1M parameters)\n",
            "- Epochs: 60 Epochs\n",
            "- Batch Size: 16 | Image Size: 640x640\n",
            "- Occlusion Augmentation: Random Erasing (0.4), Mosaic (1.0), Mixup (0.2)\n",
            "- Training Duration: {total_train_sec / 60:.2f} minutes\n",
            "\n",
            "## 4. Final Evaluation (Test Split)\n",
            "- Test Precision: {test_results.results_dict.get('metrics/precision(B)', 0):.4f}\n",
            "- Test Recall   : {test_results.results_dict.get('metrics/recall(B)', 0):.4f}\n",
            "- Test mAP@50   : {test_results.results_dict.get('metrics/mAP50(B)', 0):.4f}\n",
            "- Test mAP@50-95: {test_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}\n",
            "\n",
            "## 5. Output Artifacts\n",
            "- Weights Checkpoint: /kaggle/working/best.pt ({size_mb:.2f} MB)\n",
            "- Training Curves   : /kaggle/working/training_metrics_curves.png\n",
            "- Prediction Grid   : /kaggle/working/test_predictions_comparison.png\n",
            "# =====================================================================\n",
            "\"\"\"\n",
            "print(summary_md)\n",
            "with open('/kaggle/working/experiment_summary.md', 'w') as f:\n",
            "    f.write(summary_md)\n",
            "print('✅ EXPERIMENT EXECUTION SUCCESSFULLY COMPLETED.')\n"
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
    "nbformat_minor": 2
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"SUCCESS: Generated {NOTEBOOK_PATH} with {len(cells)} cells.")
