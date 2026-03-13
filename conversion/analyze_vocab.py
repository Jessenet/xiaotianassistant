"""Analyze vocabulary usage in training data."""
import json
from transformers import AutoTokenizer

MODEL_PATH = "../training/model_cache/models--google--functiongemma-270m-it/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

print(f"Tokenizer type: {type(tokenizer).__name__}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Test encoding
text = "播放音乐"
ids = tokenizer.encode(text)
print(f"encode('{text}'): {ids}")
for tid in ids:
    print(f"  {tid} -> {tokenizer.decode([tid])!r}")

# Count unique tokens in all training data
all_tokens = set()
for split in ["train", "eval"]:
    path = f"../training/data/music_control_{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids = tokenizer.encode(d["text"])
            all_tokens.update(ids)

print(f"\nUnique tokens in training data: {len(all_tokens)}")
print(f"Original vocab: {tokenizer.vocab_size}")
print(f"Usage: {len(all_tokens)/tokenizer.vocab_size*100:.2f}%")
print(f"Max token id: {max(all_tokens)}")
print(f"Min token id: {min(all_tokens)}")

# Add safety margin: include all single-char Chinese tokens + ASCII
safety_tokens = set()
for i in range(tokenizer.vocab_size):
    try:
        tok = tokenizer.decode([i])
        if len(tok) == 1 and ('\u4e00' <= tok <= '\u9fff'):  # CJK Unified
            safety_tokens.add(i)
        if len(tok) == 1 and ord(tok) < 128:  # ASCII
            safety_tokens.add(i)
    except:
        pass

total_needed = all_tokens | safety_tokens
print(f"\nSafety tokens (single CJK + ASCII): {len(safety_tokens)}")
print(f"Total tokens needed: {len(total_needed)}")
print(f"Recommended vocab size: {max(len(total_needed) + 1000, 8000)}")
