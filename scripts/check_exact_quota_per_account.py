import os
import subprocess
import json

accounts = ['thitnguynthanh', 'jtachitv', 'thitenguyen', 'nguynthanhthit', 'thietnguyen']
base = r'C:\Users\thiet\.kaggle_accounts'

for acc in accounts:
    acc_dir = os.path.join(base, acc)
    env = os.environ.copy()
    env['KAGGLE_CONFIG_DIR'] = acc_dir
    
    # We run a small one-line python script via subprocess so KaggleApi class attribute is freshly evaluated!
    cmd = [
        "python", "-c",
        "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print('User:', api.get_config_value('username')); print('Quota:', str(api.quota_view()))"
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(f"\n==========================================")
    print(f"ACCOUNT: {acc}")
    print(f"==========================================")
    print("STDOUT:")
    print(res.stdout.strip())
    if res.stderr.strip():
        print("STDERR:")
        print(res.stderr.strip())
