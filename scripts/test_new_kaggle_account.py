import os
import subprocess
import json
import tempfile
import shutil

test_dir = tempfile.mkdtemp()
src_file = r'C:\Users\thiet\Downloads\kaggle (6).json'
shutil.copy2(src_file, os.path.join(test_dir, 'kaggle.json'))

env = os.environ.copy()
env['KAGGLE_CONFIG_DIR'] = test_dir

cmd = [
    "python", "-c",
    "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print('User:', api.get_config_value('username')); print('Quota:', str(api.quota_view()))"
]
res = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("STDOUT:")
print(res.stdout.strip())
if res.stderr.strip():
    print("STDERR:")
    print(res.stderr.strip())

shutil.rmtree(test_dir, ignore_errors=True)
