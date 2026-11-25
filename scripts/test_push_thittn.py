import os
import subprocess
import json
import tempfile
import shutil

acc = 'thittn'
acc_dir = r'C:\Users\thiet\.kaggle_accounts\thittn'
os.makedirs(acc_dir, exist_ok=True)
shutil.copy2(r'C:\Users\thiet\Downloads\kaggle (6).json', os.path.join(acc_dir, 'kaggle.json'))

env = os.environ.copy()
env['KAGGLE_CONFIG_DIR'] = acc_dir

test_dir = 'test_gpu_probe_thittn'
os.makedirs(test_dir, exist_ok=True)

nb_content = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import torch\n", "print('CUDA:', torch.cuda.is_available())\n"]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}
with open(os.path.join(test_dir, "probe.ipynb"), "w") as f:
    json.dump(nb_content, f)

meta = {
    "id": f"{acc}/gpu-probe-{acc}",
    "title": f"GPU Probe {acc}",
    "code_file": "probe.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}
with open(os.path.join(test_dir, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f)

res = subprocess.run(["kaggle", "kernels", "push", "-p", test_dir], env=env, capture_output=True, text=True)
out = (res.stdout + res.stderr).strip()
print("GPU PUSH RESULT:")
print(out)

shutil.rmtree(test_dir, ignore_errors=True)
