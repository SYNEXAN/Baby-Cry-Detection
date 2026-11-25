import json
import re

with open('kaggle_eletriby_results/train-eletriby-4class-audio-classifier.log', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# The log is in json lines or json array
try:
    data = json.loads(text)
    for entry in data:
        if entry.get('stream_name') == 'stdout':
            line = entry.get('data', '')
            if any(k in line for k in ['DATASET', 'Found', 'category', 'files', 'Overlap', 'Leakage', 'weight', 'Epoch', 'Accuracy', 'F1']):
                print(line.strip())
except Exception as e:
    print("Error parsing json log:", e)
