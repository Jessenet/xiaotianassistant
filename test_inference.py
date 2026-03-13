#!/usr/bin/env python3
"""Quick local inference test – v6 model with hallucination checks"""
import torch, json, sys
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = r'e:\devAI\boltassistant\training\output\gemma_finetuned_v6\final'
print('Loading model from:', model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.eval()

# ── Test cases ──────────────────────────────────────────────
# (query, expected_function)  expected=None means "any valid"
test_cases = [
    # ===== 正常音乐指令 =====
    ("播放稻香",          "play_song"),
    ("播放后来",          "play_song"),
    ("来一首周杰伦的歌",  "play_song"),
    ("放首歌",            "play_song"),
    ("我想听晴天",        "play_song"),
    ("下一首",            "next_song"),
    ("切换下一首歌",      "next_song"),
    ("播放下一首",        "next_song"),
    ("上一首",            "previous_song"),
    ("播放上一首歌",      "previous_song"),
    ("暂停",              "pause"),
    ("暂停播放",          "pause"),
    ("继续播放",          "resume"),
    ("声音大一点",        "set_volume"),
    ("音量调到50",        "set_volume"),
    ("声音小一点",        "set_volume"),
    
    # ===== 模糊/短输入 – 以前会幻觉成 next_song =====
    ("换",                "none"),    # 单字，不应触发任何音乐功能
    ("切",                "none"),
    ("跳过",              "none"),
    ("下一个",            "none"),
    ("换歌",              "none"),
    ("换别的",            "none"),
    ("换个歌",            "none"),
    ("切一下",            "none"),
    
    # ===== 负样本 – 应该输出 none =====
    ("今天天气怎么样",    "none"),
    ("你好",              "none"),
    ("大模型",            "none"),
    ("帮我订个闹钟",      "none"),
    ("明天几点开会",      "none"),
    ("导航到公司",        "none"),
    ("打电话给张三",      "none"),
    ("发个微信",          "none"),
    ("Python怎么学",      "none"),
    ("海尔25块钱",        "none"),
]

print(f"\n{'='*70}")
print(f"Testing {len(test_cases)} cases on v6 model")
print(f"{'='*70}")

pass_count = 0
fail_count = 0
results = []

for query, expected in test_cases:
    prompt = f'<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n'
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=60, do_sample=False)
    
    new_tokens = out[0][input_ids.shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Parse function name from output
    actual_func = "???"
    try:
        parsed = json.loads(result)
        actual_func = parsed.get("f", "???")
    except:
        # Try to find JSON in output
        if '{' in result:
            try:
                json_str = result[result.index('{'):result.rindex('}')+1]
                parsed = json.loads(json_str)
                actual_func = parsed.get("f", "???")
            except:
                actual_func = f"PARSE_ERR"
        else:
            actual_func = f"NO_JSON"
    
    ok = (actual_func == expected) if expected else True
    status = "✓" if ok else "✗"
    if ok:
        pass_count += 1
    else:
        fail_count += 1
    
    results.append((query, expected, actual_func, ok, result))
    print(f"  {status}  [{actual_func:15s}] (expect={expected:15s})  query={query}")

print(f"\n{'='*70}")
print(f"Results: {pass_count} passed, {fail_count} failed / {len(test_cases)} total")
print(f"{'='*70}")

if fail_count > 0:
    print("\n--- Failed cases ---")
    for query, expected, actual, ok, raw in results:
        if not ok:
            print(f"  Query: {query}")
            print(f"  Expected: {expected}, Got: {actual}")
            print(f"  Raw output: {raw}")
            print()
