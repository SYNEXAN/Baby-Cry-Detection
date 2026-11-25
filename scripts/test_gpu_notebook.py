import os
import json
import subprocess
import shutil

acc = 'jtachitv'
acc_dir = os.path.join(r'C:\Users\thiet\.kaggle_accounts', acc)
env = os.environ.copy()
env['KAGGLE_CONFIG_DIR'] = acc_dir

test_dir = 'test_nb_gpu'
os.makedirs(test_dir, exist_ok=True)

nb_content = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import torch\n", "print(torch.cuda.is_available())\n"]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(os.path.join(test_dir, "test.ipynb"), "w") as f:
    json.dump(nb_content, f)

meta = {
    "id": f"{acc}/gpu-notebook-test",
    "title": "GPU Notebook Test",
    "code_file": "test.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

with open(os.path.join(test_dir, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f)

res = subprocess.run(["kaggle", "kernels", "push", "-p", test_dir], env=env, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
shutil.rmtree(test_dir, ignore_errors=True)
