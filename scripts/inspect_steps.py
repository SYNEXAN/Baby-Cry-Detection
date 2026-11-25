import json

with open(r'C:\Users\thiet\.gemini\antigravity-cli\brain\30d185aa-58de-4b8e-b83e-50bde963d7f6\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        idx = data.get('step_index')
        if idx and 2950 <= idx <= 2962:
            print(f"=== STEP {idx} ===")
            if 'content' in data:
                print("CONTENT:", str(data['content'])[:200])
            if 'tool_calls' in data:
                for tc in data['tool_calls']:
                    print("TOOL:", tc.get('name'), tc.get('args'))
