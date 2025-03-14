import json

file_path = '/data-mnt/data/wrm/EVAL/VoiceBench/Qwen2-Audio-7B-Instruct-alpacaeval-test-audio.jsonl'

total_prompt_tokens = 0
total_completion_tokens = 0

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        total_prompt_tokens += data['token_usage']['prompt_tokens']
        total_completion_tokens += data['token_usage']['completion_tokens']

print(f"Total prompt tokens: {total_prompt_tokens}")
print(f"Total completion tokens: {total_completion_tokens}")
print(f"Total tokens: {total_prompt_tokens + total_completion_tokens}")