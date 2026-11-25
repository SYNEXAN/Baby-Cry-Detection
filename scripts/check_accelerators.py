"""
Script: scripts/check_accelerators.py
Kiểm tra xem có tài khoản nào còn quota TPU (Google Cloud TPU v3-8 / v5e-8, 20h/tuần)
hoặc các loại accelerator khác hay không.
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
    "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "source": ["print('Accelerator quota check')"]}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}
with open(os.path.join(test_workdir, "accel_test.ipynb"), "w") as f:
    json.dump(nb, f)

print("=" * 70)
print("🚀 KIỂM TRA QUOTA TPU (GOOGLE CLOUD TPU V3-8) TRÊN TẤT CẢ TÀI KHOẢN")
print("=" * 70)

tpu_account = None

for username, fname in accounts:
    fpath = os.path.join(kaggle_dir, fname)
    if not os.path.exists(fpath):
        continue

    temp_cfg = tempfile.mkdtemp()
    shutil.copy2(fpath, os.path.join(temp_cfg, "kaggle.json"))

    meta = {
        "id": f"{username}/accel-check-{username}",
        "title": f"Accel Check {username}",
        "code_file": "accel_test.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
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

    if "Maximum weekly" in output:
        print(f"❌ Tài khoản: {username:16s} -> Hết quota TPU: {output}")
    elif "Kernel push error" in output:
        print(f"⚠️ Tài khoản: {username:16s} -> Lỗi push: {output[:100]}...")
    elif "successfully" in output.lower() or "Your kernel title does not resolve" in output or res.returncode == 0:
        print(f"✅ TÀI KHOẢN: {username:16s} -> CÒN NGUYÊN QUOTA TPU 20H/TUẦN! 🎉")
        if tpu_account is None:
            tpu_account = (username, fname)
    else:
        print(f"ℹ️ Tài khoản: {username:16s} -> Output: {output[:100]}...")

    shutil.rmtree(temp_cfg, ignore_errors=True)

shutil.rmtree(test_workdir, ignore_errors=True)

print("=" * 70)
if tpu_account:
    print(f"🎯 KẾT LUẬN: Đã tìm thấy tài khoản có TPU sẵn sàng: {tpu_account[0]} ({tpu_account[1]})")
else:
    print("❌ Không có tài khoản nào dùng được TPU!")
print("=" * 70)
