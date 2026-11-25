import json
import shutil

nb_path = "notebooks/train_rtdetr_baby_vision.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 4 (Code 2: Path setup & YAML)
cell_4_code = """# Cell 2: Thiết lập đường dẫn cứng cố định và tạo cấu hình data.yaml (Hệ 2 lớp: adult & baby_child)
import os
import yaml
import zipfile

# Các đường dẫn mount khả dĩ trên Kaggle
POSSIBLE_INPUT_DIRS = [
    "/kaggle/input/datasets/thietnguyen/true-baby-dataset",
    "/kaggle/input/true-baby-dataset",
    "/kaggle/input/true-baby-dataset/true-baby-dataset"
]

KAGGLE_INPUT_DIR = None
for d in POSSIBLE_INPUT_DIRS:
    if os.path.exists(d):
        KAGGLE_INPUT_DIR = d
        break

if KAGGLE_INPUT_DIR is None:
    KAGGLE_INPUT_DIR = "/kaggle/input/datasets/thietnguyen/true-baby-dataset"

print(f"📍 Kaggle Input Directory: {KAGGLE_INPUT_DIR}")

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
    'nc': 2,
    'names': ['adult', 'baby_child']
}

YAML_PATH = "/kaggle/working/baby_rtdetr_data.yaml"
with open(YAML_PATH, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ Đã tạo file cấu hình chuẩn xác tại: {YAML_PATH}")
with open(YAML_PATH, "r") as f:
    print(f.read())
"""

# Update Cell 10 (Code 5: Evaluation for 2 classes)
cell_10_code = """# Cell 5: Đánh giá mô hình toàn diện trên tập Validation & Test (Hệ 2 lớp: adult & baby_child)
import pandas as pd
import numpy as np

def evaluate_and_report(split_name):
    print("=" * 70)
    print(f"BÁO CÁO ĐỘ ĐO TOÀN DIỆN MÔ HÌNH RT-DETR TRÊN TẬP {split_name.upper()}")
    print("=" * 70)
    
    metrics = model.val(
        data=YAML_PATH,
        split=split_name,
        project="/kaggle/working/runs_rtdetr",
        name=f"eval_{split_name}"
    )
    
    print(f"Overall mAP@50    : {metrics.box.map50:.4f}")
    print(f"Overall mAP@50-95 : {metrics.box.map:.4f}")
    print("-" * 70)
    
    class_names = ['adult', 'baby_child']
    report_data = []
    
    ap_indices = list(metrics.box.ap_class_index) if hasattr(metrics.box, 'ap_class_index') else []
    
    for cls_id, c_name in enumerate(class_names):
        if cls_id in ap_indices:
            idx = ap_indices.index(cls_id)
            p = float(metrics.box.p[idx])
            r = float(metrics.box.r[idx])
            f1 = float(metrics.box.f1[idx]) if (hasattr(metrics.box, 'f1') and len(metrics.box.f1) > idx) else (2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            ap50 = float(metrics.box.ap50[idx])
            ap = float(metrics.box.ap[idx])
        else:
            p, r, f1, ap50, ap = 0.0, 0.0, 0.0, 0.0, 0.0
            
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
    return metrics, df_report

# Đánh giá cả 2 tập độc lập
metrics_val, df_val = evaluate_and_report("val")
metrics_test, df_test = evaluate_and_report("test")
"""

nb['cells'][4]['source'] = [line + '\n' for line in cell_4_code.split('\n')[:-1]] + [cell_4_code.split('\n')[-1]]
nb['cells'][10]['source'] = [line + '\n' for line in cell_10_code.split('\n')[:-1]] + [cell_10_code.split('\n')[-1]]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

shutil.copy2(nb_path, "kaggle_kernel/train_rtdetr_baby_vision.ipynb")
print("✅ Successfully updated notebook for 2 classes and synced to kaggle_kernel!")
