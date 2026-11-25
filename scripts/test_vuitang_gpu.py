"""
Script: scripts/test_vuitang_gpu.py
Kiểm tra khả năng cấp phát GPU cho tài khoản vuitang.
"""

import os
import json
import tempfile
import shutil
import subprocess

test_workdir = tempfile.mkdtemp()
nb = {
    "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "source": ["print('GPU check vuitang')"]}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}
with open(os.path.join(test_workdir, "gpu_check.ipynb"), "w") as f:
    json.dump(nb, f)

meta = {
    "id": "vuitang/tpu-check-unique-01",
    "title": "Tpu Check Unique 01",
    "code_file": "gpu_check.ipynb",
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

res = subprocess.run(["kaggle", "kernels", "push", "-p", test_workdir], capture_output=True, text=True)
print("STDOUT:", res.stdout.strip())
print("STDERR:", res.stderr.strip())
print("Return code:", res.returncode)

shutil.rmtree(test_workdir, ignore_errors=True)
