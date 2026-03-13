#!/usr/bin/env python3
"""
统一导出脚本: LoRA 合并 + KV-Cache 封装 + ExecuTorch 导出

合并了原来 export_kvcache.py 和 convert_to_executorch.py 的功能:
  Step 0: 合并 LoRA 适配器到基础模型
  Step 1: 创建 KV-Cache 封装
  Step 2: torch.export + INT8 量化
  Step 3: XNNPACK 委托 + 保存 .pte

使用方法:
  python export_pipeline.py                    # 全流程 (合并 + 导出)
  python export_pipeline.py --skip-merge       # 跳过合并 (已有 merged_model/)
  python export_pipeline.py --no-quantize      # 不量化 (FP32)
  python export_pipeline.py --copy             # 复制到 Android assets

输入:
  基础模型: training/model_cache/pruned_model/ (裁剪后的模型)
  LoRA:     training/output/gemma_finetuned_v2/final/ 或 gemma_finetuned/final/

输出:
  conversion/executorch_output/
    model_kvcache.pte    ExecuTorch 模型
    model_config.json    模型配置
    tokenizer.json       Tokenizer
"""

import os
import sys
import json
import time
import copy
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 基础模型 (裁剪后)
PRUNED_MODEL_DIR = PROJECT_ROOT / "training" / "model_cache" / "pruned_model"
# 原始模型 (未裁剪, 后备)
ORIGINAL_MODEL_CACHE = PROJECT_ROOT / "training" / "model_cache" / \
    "models--google--functiongemma-270m-it"

# LoRA 适配器 或 full fine-tuned 模型 (微调输出)
LORA_DIRS = [
    PROJECT_ROOT / "training" / "output" / "gemma_finetuned_v6" / "final",
    PROJECT_ROOT / "training" / "output" / "gemma_finetuned_v5" / "final",
    PROJECT_ROOT / "training" / "output" / "gemma_finetuned_v4" / "final",
    PROJECT_ROOT / "training" / "output" / "gemma_finetuned_v3" / "final",
    PROJECT_ROOT / "training" / "output" / "gemma_finetuned_v2" / "final",
    PROJECT_ROOT / "training" / "output" / "gemma_finetuned" / "final",
]

# 输出
MERGED_DIR = SCRIPT_DIR / "merged_model"
OUTPUT_DIR = SCRIPT_DIR / "executorch_output"

# ─── 模型配置 ─────────────────────────────────────────────────────────────────
MAX_CACHE_LEN = 80    # KV 缓存最大长度 (prompt ~23 + response ~40 tokens, 80 足够)
DEVICE = "cpu"
DTYPE = torch.float32
QUANTIZE = True       # 是否 INT8 量化

# ─── 速度优化选项 ─────────────────────────────────────────────────────────────
# INT8 动态量化: XNNPACK 原生 int8 matmul, 权重 int8 + 激活动态量化
#   - 模型体积减半 (~100MB)
#   - XNNPACK int8 compute kernel 比 fp32 快 1.5-2x
#   - 无需 FP16 (当前 XNNPACK 版本无 FP16 ConfigPrecisionType)
PRECISION_MODES = {
    "fp32": {"dtype": torch.float32, "quantize": False, "desc": "FP32 (基线)"},
    "int8": {"dtype": torch.float32, "quantize": True, "desc": "INT8 动态量化 (推荐, 1.5-2x 提速)"},
}


# ─── KV Cache 模块 ──────────────────────────────────────────────────────────
class GemmaKVCache(nn.Module):
    """单层 KV 缓存, 使用 register_buffer 支持 ExecuTorch 导出"""
    def __init__(self, max_cache_len, n_kv_heads, head_dim, dtype=torch.float32,
                 cache_dtype=None):
        super().__init__()
        # KV 缓存可以使用低精度 (FP16) 以减少内存带宽, 提高速度
        actual_dtype = cache_dtype if cache_dtype is not None else dtype
        shape = (1, n_kv_heads, max_cache_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=actual_dtype))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=actual_dtype))
        self.max_cache_len = max_cache_len

    def update(self, cache_position, k_val, v_val):
        # 使用 scatter_ 替代 fancy indexing, 支持动态 shape 导出 (batch prefill)
        # cache_position: [seq_len], k_val: [1, n_kv_heads, seq_len, head_dim]
        idx = cache_position.view(1, 1, -1, 1).expand_as(k_val)
        self.k_cache.scatter_(2, idx, k_val)
        self.v_cache.scatter_(2, idx, v_val)
        return self.k_cache, self.v_cache


# ─── KV-Cache 封装模型 ──────────────────────────────────────────────────────
class GemmaWithKVCache(nn.Module):
    """
    自定义 forward, 使用 register_buffer KV 缓存。
    完全绕过 HuggingFace 的 StaticCache/DynamicCache。
    预计算 RoPE cos/sin 存为 buffer, 避免 @torch.no_grad() 导致
    wrap_with_set_grad_enabled 高阶算子进入 torch.export graph,
    executorch to_edge_transform_and_lower 无法处理该算子。
    """
    def __init__(self, hf_model, config, max_cache_len, model_dtype=torch.float32,
                 cache_dtype=None):
        super().__init__()
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, repeat_kv
        self._apply_rotary_pos_emb = apply_rotary_pos_emb
        self._repeat_kv = repeat_kv

        # 从 HF 模型借用子模块 (共享权重)
        self.embed_tokens = hf_model.model.embed_tokens
        self.layers = hf_model.model.layers
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head

        # 配置
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.max_cache_len = max_cache_len
        self.layer_types = config.layer_types
        self.model_dtype = model_dtype

        # ── 预计算 RoPE cos/sin 为 buffer ──
        # Gemma3RotaryEmbedding.forward 有 @torch.no_grad() 装饰器,
        # torch.export 会捕获为 wrap_with_set_grad_enabled 高阶算子,
        # executorch 无法处理, 因此在 __init__ 中一次性计算所有位置的 cos/sin
        #
        # 关键: Gemma3 有两个独立的 RotaryEmbedding 实例:
        #   - rotary_emb:       用于 full_attention 层 (rope_theta=1000000)
        #   - rotary_emb_local: 用于 sliding_attention 层 (rope_local_base_freq=10000)
        # 它们的 inv_freq 不同, 必须分别预计算!
        rotary_emb_global = hf_model.model.rotary_emb
        rotary_emb_local = getattr(hf_model.model, 'rotary_emb_local', rotary_emb_global)

        dummy_x = torch.zeros(1, 1, config.hidden_size, dtype=model_dtype)
        all_positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)

        unique_types = sorted(set(self.layer_types))
        with torch.no_grad():
            for lt in unique_types:
                # 选择正确的 rotary embedding: sliding 用 local, full 用 global
                if lt in ("sliding_attention", "local"):
                    rope = rotary_emb_local
                else:
                    rope = rotary_emb_global
                cos, sin = rope(dummy_x, all_positions)
                # cos/sin shape: [1, max_cache_len, head_dim]
                self.register_buffer(f"rope_cos_{lt}", cos.squeeze(0).to(model_dtype))
                self.register_buffer(f"rope_sin_{lt}", sin.squeeze(0).to(model_dtype))

        self._unique_layer_types = unique_types
        print(f"  预计算 RoPE: {len(unique_types)} 种 layer_type, max_pos={max_cache_len}")
        for lt in unique_types:
            cos_buf = getattr(self, f"rope_cos_{lt}")
            rope_name = "local" if lt in ("sliding_attention", "local") else "global"
            print(f"    {lt}: {rope_name} rope, cos[0,:3]={cos_buf[0,:3].tolist()}")

        # KV 缓存 (可使用 FP16 以减少内存带宽)
        # cache_dtype 默认跟随 model_dtype, 也可单独指定
        actual_cache_dtype = cache_dtype if cache_dtype is not None else model_dtype
        n_layers = config.num_hidden_layers
        n_kv_heads = config.num_key_value_heads
        self.kv_caches = nn.ModuleList([
            GemmaKVCache(max_cache_len, n_kv_heads, config.head_dim,
                         model_dtype, actual_cache_dtype)
            for _ in range(n_layers)
        ])
        print(f"  KV cache dtype: {actual_cache_dtype}")

    def _build_causal_mask(self, cache_position, dtype, device):
        # 2D mask [seq_len, cache_len] — XNNPACK SDPA kernel 要求 2D
        # SDPA 内部会自动 broadcast 到 [B, H, seq_len, cache_len]
        kv_pos = torch.arange(self.max_cache_len, device=device).view(1, -1)
        q_pos = cache_position.view(-1, 1)  # 使用 -1 支持动态 seq_len
        mask = torch.where(kv_pos <= q_pos, 0.0, float("-inf")).to(dtype)
        return mask

    def forward(self, input_ids, cache_position):
        """
        Args:
            input_ids: [1, seq_len]
            cache_position: [seq_len] 绝对位置
        Returns:
            logits: [1, seq_len, vocab_size]
        """
        hidden = self.embed_tokens(input_ids)
        position_ids = cache_position.unsqueeze(0)
        mask = self._build_causal_mask(cache_position, hidden.dtype, hidden.device)

        # 位置编码: 从预计算的 buffer 中按 position 索引 cos/sin
        # 避免调用 rotary_emb.forward (@torch.no_grad → wrap_with_set_grad_enabled)
        # 注意: sliding_attention 和 full_attention 使用不同 rope 频率!
        pos_emb = {}
        for lt in self._unique_layer_types:
            cos_buf = getattr(self, f"rope_cos_{lt}")
            sin_buf = getattr(self, f"rope_sin_{lt}")
            # cos_buf: [max_cache_len, head_dim], 按 cache_position 索引
            cos = cos_buf[cache_position].unsqueeze(0)  # [1, seq_len, head_dim]
            sin = sin_buf[cache_position].unsqueeze(0)
            pos_emb[lt] = (cos, sin)

        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.layer_types[i]
            cos, sin = pos_emb[layer_type]

            # === Attention ===
            residual = hidden
            hidden = decoder_layer.input_layernorm(hidden)

            attn = decoder_layer.self_attn
            input_shape = hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            q = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
            k = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
            v = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)

            q = attn.q_norm(q)
            k = attn.k_norm(k)

            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

            k, v = self.kv_caches[i].update(cache_position, k, v)

            # Expand KV heads 1→4 for uniform head count, then fused SDPA
            k = self._repeat_kv(k, self.num_kv_groups)  # [1,4,cache,256]
            v = self._repeat_kv(v, self.num_kv_groups)  # [1,4,cache,256]

            # SDPA 融合: Q*K^T + scale + mask + softmax + V 在一个 kernel 中
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=self.scaling
            )
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_out = attn.o_proj(attn_out)

            hidden = decoder_layer.post_attention_layernorm(attn_out)
            hidden = residual + hidden

            # === MLP ===
            residual = hidden
            hidden = decoder_layer.pre_feedforward_layernorm(hidden)
            hidden = decoder_layer.mlp(hidden)
            hidden = decoder_layer.post_feedforward_layernorm(hidden)
            hidden = residual + hidden

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits


# ─── Step 0: LoRA 合并 / Full Fine-tuned 复制 ──────────────────────────────
def step0_merge_lora(base_model_dir):
    """合并基础模型 + LoRA 适配器, 或复制 full fine-tuned 模型 → merged_model/"""
    print("=" * 60)
    print("Step 0: 合并/复制微调模型")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 查找微调输出 (优先 full fine-tuned, 然后 LoRA)
    finetuned_dir = None
    is_lora = False
    for d in LORA_DIRS:
        if (d / "adapter_config.json").exists():
            finetuned_dir = d
            is_lora = True
            break
        elif (d / "model.safetensors").exists():
            finetuned_dir = d
            is_lora = False
            break

    if finetuned_dir is None:
        print("⚠ 未找到微调模型, 直接复制基础模型")
        if MERGED_DIR.exists():
            shutil.rmtree(str(MERGED_DIR))
        shutil.copytree(str(base_model_dir), str(MERGED_DIR))
        return True

    print(f"  微调输出: {finetuned_dir}")
    print(f"  类型: {'LoRA adapter' if is_lora else 'Full fine-tuned'}")

    # 检查是否已存在
    merged_safetensors = MERGED_DIR / "model.safetensors"
    if merged_safetensors.exists():
        size_mb = merged_safetensors.stat().st_size / 1024 / 1024
        print(f"\n✓ 已存在合并模型: {merged_safetensors} ({size_mb:.1f} MB)")
        print("  跳过 (如需重新合并, 请删除 merged_model/ 目录)")
        return True

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if is_lora:
            from peft import PeftModel
            print(f"\n加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_dir), torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True,
            )
            print(f"  参数量: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M")
            print(f"加载 LoRA 适配器...")
            model = PeftModel.from_pretrained(base_model, str(finetuned_dir))
            print("合并权重...")
            merged_model = model.merge_and_unload()
            print("✓ 合并完成")
        else:
            print(f"\n加载 full fine-tuned 模型...")
            merged_model = AutoModelForCausalLM.from_pretrained(
                str(finetuned_dir), torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True,
            )
            print(f"  参数量: {sum(p.numel() for p in merged_model.parameters()) / 1e6:.1f}M")

        print(f"保存到: {MERGED_DIR}")
        merged_model.save_pretrained(str(MERGED_DIR), safe_serialization=True)

        # tokenizer: 优先从微调输出取, 否则从基础模型
        tok_dir = finetuned_dir if (finetuned_dir / "tokenizer.json").exists() else base_model_dir
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
        tokenizer.save_pretrained(str(MERGED_DIR))

        # 复制额外文件
        for fname in ['tokenizer_config.json', 'special_tokens_map.json',
                       'added_tokens.json', 'generation_config.json']:
            src = base_model_dir / fname
            dst = MERGED_DIR / fname
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))

        size_mb = merged_safetensors.stat().st_size / 1024 / 1024
        print(f"✓ 模型已准备: {size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ─── Step 1-3: KV-Cache 封装 + 导出 ────────────────────────────────────────
def export_model(model_dir, quantize=True, precision="fp32"):
    """加载模型, 封装 KV-Cache, 量化, 导出 .pte
    
    Args:
        model_dir: 模型目录
        quantize: 是否量化 (被 precision 覆盖)
        precision: 精度模式 - fp32/fp16/int8/fp16_int8
    """
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    # 解析精度配置
    if precision in PRECISION_MODES:
        pconfig = PRECISION_MODES[precision]
        model_dtype = pconfig["dtype"]
        do_quantize = pconfig["quantize"]
        precision_desc = pconfig["desc"]
    else:
        model_dtype = DTYPE
        do_quantize = quantize
        precision_desc = f"custom (dtype={model_dtype}, quantize={do_quantize})"

    print(f"\n{'=' * 60}")
    print(f"精度模式: {precision_desc}")
    print(f"{'=' * 60}")

    # ── 加载模型 ──
    print("\n" + "=" * 60)
    print("[1/6] 加载模型")
    print("=" * 60)
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), config=config, torch_dtype=model_dtype, device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    print(f"  层数: {config.num_hidden_layers}")
    print(f"  词表: {config.vocab_size}")
    print(f"  KV heads: {config.num_key_value_heads}, head_dim: {config.head_dim}")
    print(f"  layer_types: {config.layer_types}")
    print(f"  模型 dtype: {model_dtype}")

    # ── 创建 KV-Cache 封装 ──
    print(f"\n[2/6] 创建 KV-Cache 封装 (max_cache_len={MAX_CACHE_LEN})")
    wrapper = GemmaWithKVCache(model, config, MAX_CACHE_LEN,
                               model_dtype=model_dtype)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    kv_bufs = [n for n, _ in wrapper.named_buffers() if "k_cache" in n or "v_cache" in n]
    total_params = sum(p.numel() for p in wrapper.parameters())
    print(f"  KV cache buffers: {len(kv_bufs)}")
    print(f"  总参数: {total_params:,} ({total_params / 1e6:.1f}M)")

    # ── 验证正确性 ──
    print(f"\n[3/6] 验证 KV-Cache 推理")
    prompt = "<start_of_turn>user\n播放音乐<end_of_turn>\n<start_of_turn>model\n"
    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"]
    prompt_len = prompt_ids.shape[1]

    for c in wrapper.kv_caches:
        c.k_cache.zero_()
        c.v_cache.zero_()

    with torch.no_grad():
        cache_pos = torch.arange(prompt_len)
        logits = wrapper(prompt_ids, cache_pos)
        t = logits[:, -1, :].argmax(dim=-1).item()
        tokens = [t]
        print(f"  Prefill ({prompt_len} tokens) → {t} = {repr(tokenizer.decode([t]))}")

        for step in range(5):
            logits = wrapper(torch.tensor([[t]]), torch.tensor([prompt_len + step]))
            t = logits[:, -1, :].argmax(dim=-1).item()
            tokens.append(t)
            print(f"  Decode {step}: {t} = {repr(tokenizer.decode([t]))}")

    generated = tokenizer.decode(tokens)
    print(f"  生成: {repr(generated)}")

    # ── torch.export ──
    print(f"\n[4/6] torch.export...")
    for c in wrapper.kv_caches:
        c.k_cache.zero_()
        c.v_cache.zero_()

    t0 = time.time()
    # 使用多 token 示例 (seq_len=3) 避免 torch.export 将 seq_len 特化为常量 1
    example_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    example_pos = torch.tensor([0, 1, 2], dtype=torch.long)

    # 尝试动态 shape (支持 batch prefill)
    supports_batch_prefill = False
    dynamic_shapes = None
    try:
        from torch.export import Dim
        seq_dim = Dim("seq_len", min=1, max=MAX_CACHE_LEN)
        dynamic_shapes = {
            "input_ids": {1: seq_dim},
            "cache_position": {0: seq_dim},
        }
        exported = torch.export.export(
            wrapper, args=(example_ids, example_pos),
            dynamic_shapes=dynamic_shapes, strict=False,
        )
        supports_batch_prefill = True
        print(f"  ✓ 动态 shape 导出 (batch prefill, max_seq={MAX_CACHE_LEN})")
    except Exception as e:
        print(f"  动态 shape 失败: {e}")
        # 降级: 使用固定 shape [1,1]
        print(f"  降级到固定 shape [1,1]")
        dynamic_shapes = None
        example_ids = torch.tensor([[1]], dtype=torch.long)
        example_pos = torch.tensor([0], dtype=torch.long)
        # 重置 KV cache
        for c in wrapper.kv_caches:
            c.k_cache.zero_()
            c.v_cache.zero_()
        exported = torch.export.export(
            wrapper, args=(example_ids, example_pos), strict=False,
        )

    # 检查 buffer mutation
    sig = exported.graph_signature
    mutated = [s for s in sig.output_specs if s.kind.name == "BUFFER_MUTATION"]
    print(f"  导出耗时: {time.time()-t0:.1f}s")
    print(f"  Buffer mutations: {len(mutated)}")

    # ── INT8 动态量化 (XNNPACKQuantizer, 推荐方式) ──
    quantize_method = "FP32"
    if do_quantize:
        print(f"\n[5/6] XNNPACK INT8 动态量化 (权重 per-channel + 激活动态)...")
        t0 = time.time()
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config,
            )
            from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

            quantizer = XNNPACKQuantizer()
            quantizer.set_global(
                get_symmetric_quantization_config(
                    is_per_channel=True,   # 权重按 output channel 量化
                    is_dynamic=True,       # 激活值在推理时动态量化
                )
            )

            # prepare: 从 ExportedProgram 提取 GraphModule, 插入观察者节点
            graph_module = exported.module()
            prepared = prepare_pt2e(graph_module, quantizer)

            # 校准: 用几条代表性输入确定量化参数
            print("  校准中...")
            calib_prompts = [
                "播放周杰伦的稻香",
                "暂停音乐",
                "下一首",
                "播放一首轻音乐",
            ]
            for cp in calib_prompts:
                full_prompt = f"<start_of_turn>user\n{cp}<end_of_turn>\n<start_of_turn>model\n"
                enc = tokenizer(full_prompt, return_tensors="pt")
                calib_ids = enc["input_ids"]
                calib_len = calib_ids.shape[1]
                # 重置 KV cache (通过 wrapper 引用)
                for c in wrapper.kv_caches:
                    c.k_cache.zero_()
                    c.v_cache.zero_()
                calib_pos = torch.arange(calib_len, dtype=torch.long)
                with torch.no_grad():
                    prepared(calib_ids, calib_pos)

            # convert: 将观察者替换为量化/反量化节点
            converted = convert_pt2e(prepared)
            print("  重新导出量化模型...")

            # 重置 KV cache 再次导出
            for c in wrapper.kv_caches:
                c.k_cache.zero_()
                c.v_cache.zero_()

            # 重新导出量化后的 GraphModule
            export_kwargs = {"strict": False}
            if dynamic_shapes is not None:
                export_kwargs["dynamic_shapes"] = dynamic_shapes
            exported = torch.export.export(
                converted, args=(example_ids, example_pos), **export_kwargs,
            )

            quantize_method = "INT8 Dynamic (XNNPACKQuantizer)"
            print(f"  ✓ {quantize_method} ({time.time()-t0:.1f}s)")

        except Exception as e:
            print(f"  ✗ XNNPACKQuantizer 失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"  回退到 FP32")
    else:
        print(f"\n[5/6] 跳过量化 (FP32)")

    # ── XNNPACK 委托 + 保存 .pte ──
    print(f"\n[6/6] Edge IR + XNNPACK → .pte")
    import executorch.kernels.quantized
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType

    t0 = time.time()

    if "INT8" in quantize_method:
        precision = ConfigPrecisionType.DYNAMIC_QUANT
        print(f"  XNNPACK precision: DYNAMIC_QUANT (int8 kernels)")
    else:
        precision = ConfigPrecisionType.FP32
        print(f"  XNNPACK precision: FP32")

    # 关键: 使用 to_edge_transform_and_lower 替代 to_edge + to_backend
    # 原因: to_edge 会将 nn.Linear 分解为 permute_copy + mm, 导致 XNNPACK
    #   无法识别 mm 的权重为静态参数 (weight_idx 指向 permute_copy 输出而非 placeholder)
    #   to_edge_transform_and_lower 保持 aten.linear.default 不分解, XNNPACK 直接委托
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner(config_precisions=precision)],
    )
    print(f"  XNNPACK 分区完成, precision={precision}")

    # 输出 XNNPACK 委托统计
    try:
        graph = edge.exported_program().graph
        total_ops = sum(1 for n in graph.nodes if n.op == 'call_function')
        delegated = sum(1 for n in graph.nodes
                        if n.op == 'call_function' and
                        'executorch_call_delegate' in str(n.target))
        non_delegated = total_ops - delegated
        ratio = delegated / total_ops * 100 if total_ops > 0 else 0
        print(f"  算子统计: 总计 {total_ops}, 委托 {delegated} ({ratio:.0f}%), "
              f"未委托 {non_delegated}")

        # 列出未委托的算子类型 (帮助诊断)
        non_del_ops = {}
        for n in graph.nodes:
            if n.op == 'call_function' and 'executorch_call_delegate' not in str(n.target):
                op_name = str(n.target).split('.')[-1] if '.' in str(n.target) else str(n.target)
                non_del_ops[op_name] = non_del_ops.get(op_name, 0) + 1
        if non_del_ops:
            sorted_ops = sorted(non_del_ops.items(), key=lambda x: -x[1])[:10]
            print(f"  未委托 Top10: {', '.join(f'{k}({v})' for k, v in sorted_ops)}")
    except Exception as e:
        print(f"  (委托统计不可用: {e})")

    et_prog = edge.to_executorch()

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    out_path = OUTPUT_DIR / "model_xnnpack.pte"
    with open(str(out_path), "wb") as f:
        et_prog.write_to_file(f)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ 保存: {out_path}")
    print(f"  大小: {size_mb:.1f} MB")
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(f"  量化: {quantize_method}")

    # 保存配置
    config_out = {
        "model_file": "model_xnnpack.pte",
        "max_cache_len": MAX_CACHE_LEN,
        "num_layers": config.num_hidden_layers,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "has_kv_cache": True,
        "kv_cache_mutated": len(mutated) > 0,
        "supports_batch_prefill": supports_batch_prefill,
        "quantization": quantize_method,
        "precision": precision.name if hasattr(precision, 'name') else str(precision),
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "xnnpack_threads": 4,  # 推荐线程数 (Android 端参考)
    }
    config_path = OUTPUT_DIR / "model_config.json"
    with open(str(config_path), "w") as f:
        json.dump(config_out, f, indent=2)

    # 复制 tokenizer
    for fname in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]:
        src = Path(model_dir) / fname
        if src.exists():
            shutil.copy2(str(src), str(OUTPUT_DIR / fname))
            print(f"  ✓ 复制: {fname}")

    return True


def copy_to_assets():
    """复制输出文件到 Android assets"""
    assets_dir = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for fname in ["model_xnnpack.pte", "model_config.json", "tokenizer.json"]:
        src = OUTPUT_DIR / fname
        if src.exists():
            dst = assets_dir / fname
            shutil.copy2(str(src), str(dst))
            size = src.stat().st_size / 1024 / 1024
            print(f"  ✓ {fname} ({size:.1f} MB)" if size > 1 else f"  ✓ {fname}")
            copied += 1

    if copied > 0:
        print(f"✓ 已复制 {copied} 个文件到 Android assets")
    else:
        print("✗ 没有文件可复制")
    return copied > 0


def main():
    parser = argparse.ArgumentParser(
        description="统一导出: LoRA 合并 + KV-Cache + ExecuTorch .pte"
    )
    parser.add_argument("--skip-merge", action="store_true",
                        help="跳过 LoRA 合并 (使用已有 merged_model/)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="不量化 (FP32)")
    parser.add_argument("--precision", type=str, default="int8",
                        choices=["fp32", "int8"],
                        help="精度模式: fp32/int8 (默认 int8, 推荐)")
    parser.add_argument("--copy", action="store_true",
                        help="复制到 Android assets")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="基础模型目录 (默认: pruned_model)")
    args = parser.parse_args()

    if args.copy:
        copy_to_assets()
        return

    print("=" * 60)
    print("统一导出: LoRA 合并 + KV-Cache + ExecuTorch .pte")
    print("=" * 60)

    # 确定基础模型路径
    if args.model_dir:
        base_model_dir = Path(args.model_dir)
    elif PRUNED_MODEL_DIR.exists() and (PRUNED_MODEL_DIR / "config.json").exists():
        base_model_dir = PRUNED_MODEL_DIR
    else:
        # 查找原始模型 snapshot
        snapshots = ORIGINAL_MODEL_CACHE / "snapshots"
        if snapshots.exists():
            snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
            if snapshot_dirs:
                base_model_dir = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
            else:
                print("✗ 找不到模型")
                sys.exit(1)
        else:
            print("✗ 找不到模型")
            sys.exit(1)

    print(f"  基础模型: {base_model_dir}")
    print(f"  精度: {args.precision}")
    if args.no_quantize:
        precision = "fp32"
    else:
        precision = args.precision

    # Step 0: 合并 LoRA
    if not args.skip_merge:
        success = step0_merge_lora(base_model_dir)
        if not success:
            sys.exit(1)
        model_dir = MERGED_DIR
    else:
        if MERGED_DIR.exists() and (MERGED_DIR / "model.safetensors").exists():
            model_dir = MERGED_DIR
        else:
            model_dir = base_model_dir

    # Step 1-3: 导出
    success = export_model(model_dir, quantize=not args.no_quantize,
                           precision=precision)
    if not success:
        sys.exit(1)

    # 复制到 assets
    copy_to_assets()

    print("\n" + "=" * 60)
    print("✓ 导出完成!")
    pte = OUTPUT_DIR / "model_kvcache.pte"
    if pte.exists():
        print(f"  .pte 文件: {pte} ({pte.stat().st_size / 1024 / 1024:.1f} MB)")
    print()
    print("后续步骤:")
    print("  1. 构建 APK: cd android && .\\gradlew assembleDebug")
    print("  2. 安装: adb install -r app/build/outputs/apk/debug/app-debug.apk")
    print('  3. 测试: adb shell am start -n com.xiaotian.assistant/.MainActivity')
    print("=" * 60)


if __name__ == "__main__":
    main()
