from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('conversion/merged_model', trust_remote_code=True)

# 测试不同长度的 function call 输出
test_outputs = [
    '<function_call>{"f": "play_song", "p": {"s": "稻香", "a": "周杰伦"}}</function_call>',
    '<function_call>{"f": "play_song", "p": {"s": "稻香"}}</function_call>',
    '<function_call>{"f": "pause_music", "p": {}}</function_call>',
    '<function_call>{"f": "next_song", "p": {}}</function_call>',
    '<function_call>{"f": "set_volume", "p": {"l": 50}}</function_call>',
]

print("=== Token count analysis ===")
for out in test_outputs:
    tokens = tok.encode(out)
    print(f"\n{out}")
    print(f"  Token count: {len(tokens)}")
    
# 分析 <function_call> 和 </function_call> 标记的 token 开销
fc_open = '<function_call>'
fc_close = '</function_call>'
print(f"\n=== Overhead tokens ===")
print(f"'<function_call>' tokens: {tok.encode(fc_open)} ({len(tok.encode(fc_open))} tokens)")
print(f"'</function_call>' tokens: {tok.encode(fc_close)} ({len(tok.encode(fc_close))} tokens)")

# JSON 部分
json_part = '{"f": "play_song", "p": {"s": "稻香", "a": "周杰伦"}}'
print(f"JSON part tokens: {len(tok.encode(json_part))} tokens")

# 紧凑 JSON (无空格)
json_compact = '{"f":"play_song","p":{"s":"稻香","a":"周杰伦"}}'
print(f"Compact JSON tokens: {len(tok.encode(json_compact))} tokens")

# 更紧凑: 不用 function_call 标签
json_only = '{"f":"play_song","p":{"s":"稻香","a":"周杰伦"}}'
print(f"JSON-only tokens: {len(tok.encode(json_only))} tokens")

# 检查 <end_of_turn> token
eot = '<end_of_turn>'
eot_id = tok.encode(eot)
print(f"\n'<end_of_turn>' token ids: {eot_id}")

# 检查 vocab 中是否有 function_call 相关的特殊 token
special_tokens = [t for t in tok.get_vocab().keys() if 'function' in t.lower() or 'call' in t.lower()]
print(f"\nSpecial tokens with 'function'/'call': {special_tokens[:20]}")
