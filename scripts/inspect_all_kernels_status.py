import os
import subprocess
import json

accounts = ['thitnguynthanh', 'jtachitv', 'thitenguyen', 'nguynthanhthit', 'thietnguyen']
base = r'C:\Users\thiet\.kaggle_accounts'

for acc in accounts:
    acc_dir = os.path.join(base, acc)
    env = os.environ.copy()
    env['KAGGLE_CONFIG_DIR'] = acc_dir
    res = subprocess.run(['kaggle', 'kernels', 'list', '--mine', '--page-size', '10'], env=env, capture_output=True, text=True)
    print(f"\n==========================================")
    print(f"ACCOUNT: {acc}")
    print(f"==========================================")
    lines = res.stdout.strip().split("\n")
    if len(lines) > 2:
        for line in lines[2:]:
            parts = line.split()
            if parts:
                slug = parts[0]
                # Check status of this kernel
                st_res = subprocess.run(['kaggle', 'kernels', 'status', slug], env=env, capture_output=True, text=True)
                st_out = st_res.stdout.strip()
                print(f"  {slug:<45} -> {st_out}")
    else:
        print("  No kernels found.")
