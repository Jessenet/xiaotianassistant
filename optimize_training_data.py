#!/usr/bin/env python3
"""
优化训练数据: 去除 <function_call>/<\/function_call> 标签

原格式 (每条约 55-70 tokens):
  <start_of_turn>model\n<function_call>{"f":"play_song","p":{"s":"稻香"}}</function_call><end_of_turn>

优化后 (每条约 25-40 tokens, 节省约 30 tokens):
  <start_of_turn>model\n{"f":"play_song","p":{"s":"稻香"}}<end_of_turn>

标签 <function_call> 和 </function_call> 在裁剪后的词表中不是特殊 token,
每个标签被拆成约 15 个字符级 token, 两个标签共消耗约 30 tokens (纯开销)。
去掉后:
  - 模型输出 token 数从 ~62 降至 ~32 (减少约 50%)
  - 推理时间预计从 ~15s 缩短到 ~8s
  - 同时简化了 Android 端的早停检测 (检测 } 即可)
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "training" / "data"

def strip_function_call_tags(text: str) -> str:
    """去除 <function_call> 和 </function_call> 标签, 保留内部 JSON"""
    text = text.replace("<function_call>", "")
    text = text.replace("</function_call>", "")
    return text

def process_file(input_path: Path, output_path: Path):
    """处理单个 JSONL 文件"""
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line.strip())
            if "text" in item:
                item["text"] = strip_function_call_tags(item["text"])
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count

def verify(path: Path, n=3):
    """打印前 n 条验证"""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            item = json.loads(line)
            text = item.get("text", "")
            # 验证没有标签
            assert "<function_call>" not in text, f"Line {i}: still has tag!"
            # 验证有 JSON
            assert '{"f":' in text or '"f": ' in text, f"Line {i}: no JSON!"
            print(f"  [{i}] {text[:150]}")

def main():
    files = [
        "music_control_train_balanced.jsonl",
        "music_control_eval_balanced.jsonl",
        "music_control_train.jsonl",
        "music_control_eval.jsonl",
    ]

    for fname in files:
        input_path = DATA_DIR / fname
        if not input_path.exists():
            print(f"跳过 (不存在): {fname}")
            continue

        # 备份原文件
        backup_path = DATA_DIR / f"{fname}.bak"
        if not backup_path.exists():
            import shutil
            shutil.copy2(str(input_path), str(backup_path))
            print(f"备份: {fname} → {fname}.bak")

        # 处理
        count = process_file(input_path, input_path)
        print(f"✓ {fname}: {count} 条已处理")
        verify(input_path)
        print()

    # Token 对比
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(Path(__file__).parent / "conversion" / "merged_model"),
            trust_remote_code=True
        )
        
        old_sample = '<start_of_turn>model\n<function_call>{"f": "play_song", "p": {"s": "稻香", "a": "周杰伦"}}</function_call><end_of_turn>'
        new_sample = '<start_of_turn>model\n{"f": "play_song", "p": {"s": "稻香", "a": "周杰伦"}}<end_of_turn>'
        
        old_tokens = tok.encode(old_sample)
        new_tokens = tok.encode(new_sample)
        
        print(f"=== Token 对比 ===")
        print(f"原格式: {len(old_tokens)} tokens")
        print(f"新格式: {len(new_tokens)} tokens")
        print(f"节省: {len(old_tokens) - len(new_tokens)} tokens ({(1 - len(new_tokens)/len(old_tokens))*100:.0f}%)")
    except Exception as e:
        print(f"Token 对比跳过: {e}")


if __name__ == "__main__":
    main()
