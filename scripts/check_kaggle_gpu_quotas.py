"""
Script: scripts/check_kaggle_gpu_quotas.py
Mục đích:
Kiểm tra quota GPU trên toàn bộ 6 tài khoản Kaggle được cấu hình trong ~/.kaggle
theo đúng quy định của KAGGLE_MCP_POLICY.md.
"""

import os
import json
import tempfile
import shutil
import subprocess

kaggle_dir = os.path.expanduser("~/.kaggle")
accounts = [
    ("thittn", "kaggle.json"),
    ("thitenguyen", "kaggle_thitenguyen.json"),
    ("thietnguyen", "kaggle_thietnguyen_backup.json"),
    ("jtachitv", "kaggle_jtachitv.json"),
    ("thitnguynthanh", "kaggle_thitnguynthanh.json"),
    ("nguynthanhthit", "kaggle_nguynthanhthit.json")
]

test_workdir = tempfile.mkdtemp()
nb = {
    "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "source": ["print('GPU quota check')"]}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}
with open(os.path.join(test_workdir, "gpu_test.ipynb"), "w") as f:
    json.dump(nb, f)

print("=" * 70)
print("🔍 KIỂM TRA QUOTA GPU TRÊN CÁC TÀI KHOẢN KAGGLE CỦA BẠN")
print("=" * 70)

available_account = None

for username, fname in accounts:
    fpath = os.path.join(kaggle_dir, fname)
    if not os.path.exists(fpath):
        continue

    temp_cfg = tempfile.mkdtemp()
    shutil.copy2(fpath, os.path.join(temp_cfg, "kaggle.json"))

    meta = {
        "id": f"{username}/gpu-quota-check-{username}",
        "title": f"GPU Quota Check {username}",
        "code_file": "gpu_test.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": []
    }
    with open(os.path.join(test_workdir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f)

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = temp_cfg

    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", test_workdir],
        capture_output=True,
        text=True,
        env=env
    )

    output = (res.stdout + " " + res.stderr).strip()

    if "Maximum weekly GPU quota" in output:
        print(f"❌ Tài khoản: {username:16s} -> ĐÃ HẾT QUOTA GPU (30.00h/tuần)")
    elif "Kernel push error" in output:
        print(f"⚠️ Tài khoản: {username:16s} -> Lỗi push: {output[:100]}...")
    elif "successfully" in output.lower() or "Your kernel title does not resolve" in output or res.returncode == 0:
        print(f"✅ TÀI KHOẢN: {username:16s} -> CÒN QUOTA GPU! SẴN SÀNG HUẤN LUYỆN! 🎉")
        if available_account is None:
            available_account = (username, fname)
    else:
        print(f"ℹ️ Tài khoản: {username:16s} -> Kết quả: {output[:100]}...")

    shutil.rmtree(temp_cfg, ignore_errors=True)

shutil.rmtree(test_workdir, ignore_errors=True)

print("=" * 70)
if available_account:
    print(f"🎯 KẾT LUẬN: Đã chọn tài khoản có GPU sẵn sàng: {available_account[0]} ({available_account[1]})")
else:
    print("❌ Không có tài khoản nào còn quota GPU!")
print("=" * 70)
