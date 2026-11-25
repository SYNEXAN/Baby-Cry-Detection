import json
import re
import sys

nb_path = 'kaggle_eletriby_kernel/train_eletriby_4class_audio_classifier.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

errors = 0
emojis_found = 0

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    emojis = emoji_pattern.findall(src)
    if emojis:
        print(f"Warning: Emoji found in cell {i}: {emojis}")
        emojis_found += len(emojis)
    if cell['cell_type'] == 'code':
        try:
            compile(src, f"cell_{i}", "exec")
        except Exception as e:
            print(f"Error compiling cell {i}: {e}")
            errors += 1

print(f"Check finished: {errors} compile errors, {emojis_found} emojis found.")
if errors > 0 or emojis_found > 0:
    sys.exit(1)
print("All notebook cells compiled successfully with 0 emojis.")
