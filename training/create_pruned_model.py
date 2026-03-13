#!/usr/bin/env python3
"""
创建裁剪版 Gemma3 模型 (词表 + 层数裁剪)

裁剪策略:
  1. 词表裁剪: 262144 → ~12000 (保留训练数据 token + CJK 单字 + 字节级 token + 特殊 token)
  2. 层数裁剪: 18 → 12 层 (每组保留 3 个 sliding + 1 个 full_attention)

使用方法:
  python create_pruned_model.py                         # 默认裁剪 (12 层)
  python create_pruned_model.py --num-layers 10         # 更激进裁剪 (10 层)
  python create_pruned_model.py --skip-layer-prune      # 仅裁剪词表

输出:
  training/model_cache/pruned_model/  (完整 HuggingFace 模型目录)
"""

import os
import sys
import json
import copy
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np

# ── 路径配置 ──
SCRIPT_DIR = Path(__file__).parent
MODEL_SNAPSHOT = SCRIPT_DIR / "model_cache" / "models--google--functiongemma-270m-it" / \
    "snapshots" / "39eccb091651513a5dfb56892d3714c1b5b8276c"
TRAIN_DATA = SCRIPT_DIR / "data" / "music_control_train.jsonl"
EVAL_DATA = SCRIPT_DIR / "data" / "music_control_eval.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "model_cache" / "pruned_model"

# ── 裁剪配置 ──
# 原始 18 层 layer_types: [s,s,s,s,s,F, s,s,s,s,s,F, s,s,s,s,s,F]
# 12 层保留: 每组第 0, 2, 4 个 sliding + full_attention
LAYERS_TO_KEEP_12 = [0, 2, 4, 5, 6, 8, 10, 11, 12, 14, 16, 17]
# 10 层保留: 每组第 0, 3 个 sliding + full_attention, 加 layer 1
LAYERS_TO_KEEP_10 = [0, 1, 3, 5, 6, 9, 11, 14, 17]
# 8 层保留: 前中后均匀选取 sliding + 保留所有 full_attention (layer 5,11,17)
LAYERS_TO_KEEP_8 = [0, 3, 5, 8, 11, 14, 16, 17]


def collect_training_tokens(tokenizer):
    """收集训练数据中使用的所有 token ID"""
    token_counter = Counter()
    total_samples = 0

    for data_file in [TRAIN_DATA, EVAL_DATA]:
        if not data_file.exists():
            print(f"  ⚠ 数据文件不存在: {data_file}")
            continue
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                text = sample.get("text", "")
                if text:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    token_counter.update(ids)
                    total_samples += 1

    print(f"  训练样本: {total_samples}")
    print(f"  使用的唯一 token: {len(token_counter)}")
    return token_counter


def build_keep_set(tokenizer, token_counter, vocab_dict):
    """构建要保留的 token ID 集合"""
    keep_ids = set()

    # 1. 训练数据中使用的 token
    keep_ids.update(token_counter.keys())
    print(f"  训练数据 token: {len(keep_ids)}")

    # 2. 必要的特殊 token
    special_tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '[multimodal]',
                      '<start_of_turn>', '<end_of_turn>']
    for sp in special_tokens:
        if sp in vocab_dict:
            keep_ids.add(vocab_dict[sp])
    # BOS/EOS/PAD token IDs from tokenizer
    for tid in [tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.pad_token_id, tokenizer.unk_token_id]:
        if tid is not None:
            keep_ids.add(tid)
    print(f"  加特殊 token 后: {len(keep_ids)}")

    # 3. 字节级回退 token (0x00-0xFF)
    for tok, tid in vocab_dict.items():
        if tok.startswith('<0x') and tok.endswith('>') and len(tok) == 6:
            keep_ids.add(tid)
    print(f"  加字节级 token 后: {len(keep_ids)}")

    # 4. CJK 单字符 token (U+4E00-U+9FFF)
    cjk_count = 0
    for tok, tid in vocab_dict.items():
        if len(tok) == 1 and 0x4E00 <= ord(tok) <= 0x9FFF:
            keep_ids.add(tid)
            cjk_count += 1
        # ▁ prefix CJK chars (SentencePiece word boundary)
        elif len(tok) == 2 and tok[0] == '\u2581' and 0x4E00 <= ord(tok[1]) <= 0x9FFF:
            keep_ids.add(tid)
            cjk_count += 1
    print(f"  加 CJK 单字 token 后: {len(keep_ids)} (CJK: {cjk_count})")

    # 5. 常用 ASCII 字符 token (单字符 + ▁前缀)
    ascii_count = 0
    for tok, tid in vocab_dict.items():
        if len(tok) == 1 and (tok.isascii() and tok.isprintable()):
            keep_ids.add(tid)
            ascii_count += 1
        elif len(tok) == 2 and tok[0] == '\u2581' and tok[1].isascii() and tok[1].isprintable():
            keep_ids.add(tid)
            ascii_count += 1
    print(f"  加 ASCII 字符后: {len(keep_ids)} (ASCII: {ascii_count})")

    # 6. 常用标点符号
    punctuation = '，。？！、：；""''《》【】（）…——～·' + ',.?!:;"\'-_()[]{}/<>@#$%^&*+=|\\`~'
    for ch in punctuation:
        if ch in vocab_dict:
            keep_ids.add(vocab_dict[ch])
        sp_ch = '\u2581' + ch
        if sp_ch in vocab_dict:
            keep_ids.add(vocab_dict[sp_ch])

    # 7. 数字和常用词
    for tok in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '▁0', '▁1', '▁2', '▁3', '▁4', '▁5', '▁6', '▁7', '▁8', '▁9',
                '\n', '\t', ' ', '▁', '▁▁']:
        if tok in vocab_dict:
            keep_ids.add(vocab_dict[tok])

    # 8. JSON 相关 token (function calling 格式需要)
    json_tokens = ['{', '}', '[', ']', ':', ',', '"', 'true', 'false', 'null',
                   '▁{', '▁}', '▁[', '▁]', '▁:', '▁,', '▁"',
                   'function', 'parameters', 'function_call',
                   '<function_call>', '</function_call>',
                   '▁function', '▁parameters',
                   'play_song', 'pause_music', 'next_song', 'previous_song',
                   'set_volume', 'search_song', 'add_to_playlist',
                   '▁play', '▁pause', '▁next', '▁previous', '▁set', '▁search', '▁add',
                   '_song', '_music', '_volume', '_to', '_playlist',
                   'song_name', 'artist', 'volume', 'query', 'playlist_name',
                   '▁song', '▁artist', '▁volume', '▁query', '▁playlist',
                   # 紧凑 JSON 格式的短 key (f/p/s/a/l)
                   'f', 'p', 's', 'a', 'l', 'q', 'v',
                   '▁f', '▁p', '▁s', '▁a', '▁l', '▁q', '▁v',
                   'resume', '▁resume',]
    for tok in json_tokens:
        if tok in vocab_dict:
            keep_ids.add(vocab_dict[tok])

    print(f"  最终保留 token 数: {len(keep_ids)}")

    # 确保不超过原始 vocab
    keep_ids = keep_ids & set(range(len(vocab_dict)))
    return keep_ids


def prune_tokenizer(tokenizer_json_path, keep_ids, output_path):
    """
    裁剪 BPE tokenizer: 保留指定 token, 移除其他 token 和无关 merge。
    返回 old_id → new_id 映射。
    """
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tj = json.load(f)

    vocab = tj['model']['vocab']  # dict: token_str -> old_id
    merges = tj['model'].get('merges', [])

    # 按 old_id 排序保留的 token, 建立 old→new 映射
    sorted_keep = sorted(keep_ids)
    old_to_new = {}
    new_to_old = {}
    for new_id, old_id in enumerate(sorted_keep):
        old_to_new[old_id] = new_id
        new_to_old[new_id] = old_id

    # 构建新 vocab
    # 反转原始 vocab: old_id -> token_str
    id_to_token = {tid: tok for tok, tid in vocab.items()}
    kept_token_strings = set()
    new_vocab = {}
    for new_id in range(len(sorted_keep)):
        old_id = new_to_old[new_id]
        token_str = id_to_token[old_id]
        new_vocab[token_str] = new_id
        kept_token_strings.add(token_str)

    print(f"  新词表大小: {len(new_vocab)}")

    # 过滤 merges: 保留仅涉及保留 token 的 merge
    # BPE merge: [piece_A, piece_B], 合并结果为 piece_A + piece_B
    new_merges = []
    for merge in merges:
        piece_a, piece_b = merge
        result = piece_a + piece_b
        if (piece_a in kept_token_strings and
            piece_b in kept_token_strings and
            result in kept_token_strings):
            new_merges.append(merge)

    print(f"  原始 merges: {len(merges)} → 保留: {len(new_merges)}")

    # 更新 tokenizer.json
    tj['model']['vocab'] = new_vocab
    tj['model']['merges'] = new_merges

    # 更新 added_tokens
    new_added_tokens = []
    for at in tj.get('added_tokens', []):
        old_id = at['id']
        if old_id in old_to_new:
            at_copy = copy.deepcopy(at)
            at_copy['id'] = old_to_new[old_id]
            new_added_tokens.append(at_copy)
    tj['added_tokens'] = new_added_tokens
    print(f"  added_tokens: {len(new_added_tokens)}")

    # 更新 post_processor 中的 special_tokens ID
    if 'post_processor' in tj and tj['post_processor']:
        pp = tj['post_processor']
        if 'special_tokens' in pp:
            for tok_key, tok_info in pp['special_tokens'].items():
                if 'id' in tok_info:
                    old_id = tok_info['id']
                    if old_id in old_to_new:
                        tok_info['id'] = old_to_new[old_id]
                if 'ids' in tok_info:
                    tok_info['ids'] = [old_to_new.get(i, 0) for i in tok_info['ids']]

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  tokenizer.json 已保存: {output_path} ({size_mb:.1f} MB)")

    return old_to_new, new_to_old


def prune_model_weights(model, old_to_new, new_vocab_size, layers_to_keep, config):
    """
    裁剪模型权重:
    1. Embedding/LM-head: 仅保留 keep 的 token 对应行
    2. Decoder layers: 保留指定层
    """
    print("\n裁剪模型权重...")

    # ── 1. 词表裁剪: embed_tokens ──
    old_embed = model.model.embed_tokens.weight.data  # [old_vocab, hidden]
    hidden_size = old_embed.shape[1]
    print(f"  原始 embed_tokens: {old_embed.shape}")

    # 按 new_id 顺序提取对应 old_id 的行
    new_embed = torch.zeros(new_vocab_size, hidden_size, dtype=old_embed.dtype)
    for new_id in range(new_vocab_size):
        old_id = {v: k for k, v in old_to_new.items()}[new_id]
        new_embed[new_id] = old_embed[old_id]

    model.model.embed_tokens = torch.nn.Embedding(
        new_vocab_size, hidden_size,
        _weight=new_embed,
    )
    print(f"  新 embed_tokens: {new_embed.shape}")

    # ── 2. 词表裁剪: lm_head ──
    # Gemma3 默认 tie_word_embeddings=True, lm_head 共享 embed_tokens 权重
    tie_weights = getattr(config, 'tie_word_embeddings', True)
    if tie_weights:
        print(f"  lm_head: 权重与 embed_tokens 绑定 (tie_word_embeddings=True)")
        # 创建新 lm_head, 权重将在 model.tie_weights() 时绑定
        model.lm_head = torch.nn.Linear(hidden_size, new_vocab_size, bias=False)
        model.lm_head.weight = model.model.embed_tokens.weight  # 手动绑定
    else:
        old_lm_head = model.lm_head.weight.data  # [old_vocab, hidden]
        new_lm_head = torch.zeros(new_vocab_size, hidden_size, dtype=old_lm_head.dtype)
        for new_id in range(new_vocab_size):
            old_id = {v: k for k, v in old_to_new.items()}[new_id]
            new_lm_head[new_id] = old_lm_head[old_id]
        model.lm_head = torch.nn.Linear(hidden_size, new_vocab_size, bias=False)
        model.lm_head.weight.data = new_lm_head
        print(f"  新 lm_head: {new_lm_head.shape}")

    # ── 3. 层裁剪 ──
    if layers_to_keep is not None:
        original_layers = len(model.model.layers)
        new_layers = torch.nn.ModuleList([model.model.layers[i] for i in layers_to_keep])
        model.model.layers = new_layers
        print(f"  层裁剪: {original_layers} → {len(new_layers)} (保留: {layers_to_keep})")

        # 更新 layer_types
        old_layer_types = config.layer_types
        new_layer_types = [old_layer_types[i] for i in layers_to_keep]
        config.num_hidden_layers = len(layers_to_keep)
        config.layer_types = new_layer_types
        print(f"  新 layer_types: {new_layer_types}")

    # 更新 config
    config.vocab_size = new_vocab_size

    # 打印新模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  裁剪后总参数: {total_params:,} ({total_params / 1e6:.1f}M)")

    return model, config


def save_pruned_model(model, config, tokenizer, old_to_new, output_dir):
    """保存裁剪后的模型为完整 HuggingFace 格式"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型权重
    print(f"\n保存模型到: {output_dir}")
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # 更新并保存 config.json (save_pretrained 已保存, 但确保 layer_types 正确)
    config_path = output_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        saved_config = json.load(f)
    saved_config['layer_types'] = config.layer_types
    saved_config['vocab_size'] = config.vocab_size
    saved_config['num_hidden_layers'] = config.num_hidden_layers
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2, ensure_ascii=False)

    # tokenizer 已通过 prune_tokenizer 保存到 output_dir

    # 保存 token 映射 (用于调试)
    mapping_path = output_dir / "token_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"old_to_new": {str(k): v for k, v in old_to_new.items()}},
                  f, ensure_ascii=False)

    # 复制其他必要文件
    for fname in ['special_tokens_map.json', 'tokenizer_config.json',
                   'added_tokens.json', 'generation_config.json', 'chat_template.jinja']:
        src = MODEL_SNAPSHOT / fname
        dst = output_dir / fname
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(str(src), str(dst))

    # 更新 tokenizer_config.json 中的 vocab_size
    tc_path = output_dir / "tokenizer_config.json"
    if tc_path.exists():
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        if 'vocab_size' in tc:
            tc['vocab_size'] = config.vocab_size
        # Update added_tokens_decoder with new IDs
        if 'added_tokens_decoder' in tc:
            new_decoder = {}
            for old_id_str, token_info in tc['added_tokens_decoder'].items():
                old_id = int(old_id_str)
                if old_id in old_to_new:
                    new_decoder[str(old_to_new[old_id])] = token_info
            tc['added_tokens_decoder'] = new_decoder
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)

    # 更新 added_tokens.json
    at_path = output_dir / "added_tokens.json"
    if at_path.exists():
        with open(at_path, "r", encoding="utf-8") as f:
            at = json.load(f)
        new_at = {}
        for tok, old_id in at.items():
            if old_id in old_to_new:
                new_at[tok] = old_to_new[old_id]
        with open(at_path, "w", encoding="utf-8") as f:
            json.dump(new_at, f, indent=2, ensure_ascii=False)

    # 验证文件完整性
    files = list(output_dir.iterdir())
    print(f"  保存的文件:")
    for f in sorted(files):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"    {f.name} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"    {f.name} ({size / 1024:.1f} KB)")


def verify_pruned_model(output_dir):
    """验证裁剪后的模型可以正常加载和推理"""
    print("\n" + "=" * 60)
    print("验证裁剪后的模型")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(str(output_dir), trust_remote_code=True)
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  layer_types: {config.layer_types}")

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)
    print(f"  tokenizer vocab: {len(tokenizer)}")

    model = AutoModelForCausalLM.from_pretrained(
        str(output_dir),
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {total:,} ({total / 1e6:.1f}M)")

    # 测试推理
    test_text = "<start_of_turn>user\n播放音乐<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(test_text, return_tensors="pt")
    print(f"  测试输入: {repr(test_text[:50])}...")
    print(f"  token IDs: {inputs['input_ids'][0].tolist()[:20]}...")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        print(f"  输出 logits shape: {logits.shape}")
        print(f"  预测下一个 token: {next_token} = {repr(tokenizer.decode([next_token]))}")

    print("  ✓ 模型验证通过!")

    # 对比原始模型大小
    original_safetensors = MODEL_SNAPSHOT / "model.safetensors"
    pruned_safetensors = output_dir / "model.safetensors"
    if original_safetensors.exists() and pruned_safetensors.exists():
        orig_mb = original_safetensors.stat().st_size / 1024 / 1024
        pruned_mb = pruned_safetensors.stat().st_size / 1024 / 1024
        ratio = orig_mb / pruned_mb
        print(f"\n  原始模型: {orig_mb:.1f} MB")
        print(f"  裁剪模型: {pruned_mb:.1f} MB")
        print(f"  压缩比:   {ratio:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="创建裁剪版 Gemma3 模型")
    parser.add_argument("--num-layers", type=int, default=8,
                        choices=[8, 10, 12], help="保留层数 (默认 8)")
    parser.add_argument("--skip-layer-prune", action="store_true",
                        help="跳过层裁剪 (仅裁剪词表)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过验证步骤")
    args = parser.parse_args()

    print("=" * 60)
    print("创建裁剪版 Gemma3 模型")
    print("=" * 60)

    # 检查路径
    if not MODEL_SNAPSHOT.exists():
        print(f"✗ 模型快照不存在: {MODEL_SNAPSHOT}")
        sys.exit(1)
    if not TRAIN_DATA.exists():
        print(f"✗ 训练数据不存在: {TRAIN_DATA}")
        sys.exit(1)

    # 确定要保留的层
    if args.skip_layer_prune:
        layers_to_keep = None
        print(f"  层裁剪: 跳过 (保留全部 18 层)")
    elif args.num_layers == 8:
        layers_to_keep = LAYERS_TO_KEEP_8
        print(f"  层裁剪: 18 → {args.num_layers} (保留: {layers_to_keep})")
    elif args.num_layers == 10:
        layers_to_keep = LAYERS_TO_KEEP_10
        print(f"  层裁剪: 18 → {args.num_layers} (保留: {layers_to_keep})")
    else:
        layers_to_keep = LAYERS_TO_KEEP_12
        print(f"  层裁剪: 18 → {args.num_layers} (保留: {layers_to_keep})")

    # ── Step 1: 加载 tokenizer 并分析训练数据 ──
    print("\n[1/5] 加载 tokenizer 并分析训练数据...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_SNAPSHOT), trust_remote_code=True
    )
    vocab_dict = json.load(open(MODEL_SNAPSHOT / "tokenizer.json", encoding="utf-8"))['model']['vocab']

    token_counter = collect_training_tokens(tokenizer)

    # ── Step 2: 构建保留集 ──
    print("\n[2/5] 构建 token 保留集...")
    keep_ids = build_keep_set(tokenizer, token_counter, vocab_dict)
    new_vocab_size = len(keep_ids)
    print(f"  → 新词表大小: {new_vocab_size} (原始: {len(vocab_dict)})")
    print(f"  → 压缩比: {len(vocab_dict) / new_vocab_size:.1f}x")

    # ── Step 3: 裁剪 tokenizer ──
    print("\n[3/5] 裁剪 tokenizer...")
    old_to_new, new_to_old = prune_tokenizer(
        MODEL_SNAPSHOT / "tokenizer.json",
        keep_ids,
        OUTPUT_DIR / "tokenizer.json",
    )

    # ── Step 4: 裁剪模型 ──
    print("\n[4/5] 加载并裁剪模型...")
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(str(MODEL_SNAPSHOT), trust_remote_code=True)
    print(f"  加载模型: {MODEL_SNAPSHOT}")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_SNAPSHOT),
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"  原始参数: {orig_params:,} ({orig_params / 1e6:.1f}M)")

    model, config = prune_model_weights(model, old_to_new, new_vocab_size,
                                         layers_to_keep, config)

    # ── Step 5: 保存 ──
    print("\n[5/5] 保存裁剪后的模型...")
    save_pruned_model(model, config, tokenizer, old_to_new, OUTPUT_DIR)

    # ── 验证 ──
    if not args.skip_verify:
        verify_pruned_model(OUTPUT_DIR)

    # ── 后续步骤提示 ──
    new_params = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 60)
    print("✓ 裁剪完成!")
    print(f"  参数量: {orig_params / 1e6:.1f}M → {new_params / 1e6:.1f}M "
          f"({orig_params / new_params:.1f}x 压缩)")
    print(f"  输出目录: {OUTPUT_DIR}")
    print()
    print("后续步骤:")
    print("  1. 更新 training_config.yaml 的 base_model 为 ./model_cache/pruned_model")
    print("  2. 运行微调: python train_simple.py")
    print("  3. 导出 .pte: python ../conversion/export_pipeline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
