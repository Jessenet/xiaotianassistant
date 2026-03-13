#!/usr/bin/env python3
"""Debug: compare HF model vs KV-cache wrapper step-by-step to find divergence."""
import sys
sys.path.insert(0, 'conversion')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from export_pipeline import GemmaWithKVCache, MAX_CACHE_LEN

MODEL_PATH = r'e:\devAI\boltassistant\conversion\merged_model'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
hf_model.eval()

wrapped = GemmaWithKVCache(hf_model, config, MAX_CACHE_LEN, model_dtype=torch.float32)
wrapped.eval()

query = "播放后来"
prompt = f'<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n'
input_ids = tokenizer.encode(prompt)
print(f"Input tokens ({len(input_ids)}): {input_ids}")

# ─── HF model teacher-forcing: feed all tokens, check predictions ───
# First generate correctly with HF
with torch.no_grad():
    ids = torch.tensor([input_ids])
    out = hf_model.generate(ids, max_new_tokens=30, do_sample=False)
    hf_gen_ids = out[0][len(input_ids):].tolist()
    print(f"HF generated: {hf_gen_ids}")
    print(f"HF text: {tokenizer.decode(hf_gen_ids, skip_special_tokens=False)}")

# ─── KV-cache wrapper step-by-step ───
print(f"\n{'='*60}")
print("Step-by-step comparison: HF model (no cache) vs KV-cache wrapper")
print(f"{'='*60}")

# Reset cache by recreating wrapper
wrapped = GemmaWithKVCache(hf_model, config, MAX_CACHE_LEN, model_dtype=torch.float32)
wrapped.eval()

with torch.no_grad():
    # === PREFILL ===
    ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    pos_tensor = torch.arange(len(input_ids), dtype=torch.long)
    
    kv_logits = wrapped(ids_tensor, pos_tensor)
    kv_last_logits = kv_logits[0, -1, :]
    kv_next = torch.argmax(kv_last_logits).item()
    
    # HF: single forward pass with prompt
    hf_out = hf_model(ids_tensor)
    hf_last_logits = hf_out.logits[0, -1, :]
    hf_next = torch.argmax(hf_last_logits).item()
    
    # Compare
    logit_diff = (kv_last_logits - hf_last_logits).abs().max().item()
    top5_kv = torch.topk(kv_last_logits, 5)
    top5_hf = torch.topk(hf_last_logits, 5)
    
    print(f"\nPrefill:")
    print(f"  HF next: {hf_next}, KV next: {kv_next}, match: {hf_next == kv_next}")
    print(f"  Max logit diff: {logit_diff:.6f}")
    print(f"  HF top5: {list(zip(top5_hf.indices.tolist(), [f'{v:.3f}' for v in top5_hf.values.tolist()]))}")
    print(f"  KV top5: {list(zip(top5_kv.indices.tolist(), [f'{v:.3f}' for v in top5_kv.values.tolist()]))}")
    
    if hf_next != kv_next:
        print("  *** DIVERGED AT PREFILL!")
    
    # === DECODE: step by step ===
    current_seq = list(input_ids)
    kv_token = kv_next
    hf_token = hf_next
    
    for step in range(1, 25):
        pos = len(input_ids) + step - 1
        
        # KV-cache: feed single token
        ids_t = torch.tensor([[kv_token]], dtype=torch.long)
        pos_t = torch.tensor([pos], dtype=torch.long)
        kv_logits = wrapped(ids_t, pos_t)
        kv_next_logits = kv_logits[0, 0, :]
        kv_next = torch.argmax(kv_next_logits).item()
        
        # HF: feed FULL sequence (no cache, teacher-forcing with KV-cache's tokens)
        current_seq.append(kv_token)
        hf_ids = torch.tensor([current_seq], dtype=torch.long)
        hf_out = hf_model(hf_ids)
        hf_next_logits = hf_out.logits[0, -1, :]
        hf_next_from_kv = torch.argmax(hf_next_logits).item()
        
        logit_diff = (kv_next_logits - hf_next_logits).abs().max().item()
        mean_diff = (kv_next_logits - hf_next_logits).abs().mean().item()
        
        id_to_tok = {v: k for k, v in tokenizer.get_vocab().items()}
        kv_tok_str = id_to_tok.get(kv_next, '?')
        hf_tok_str = id_to_tok.get(hf_next_from_kv, '?')
        
        match = "✓" if kv_next == hf_next_from_kv else "✗ DIVERGED"
        print(f"\n  Step {step} (pos={pos}): {match}")
        print(f"    KV-cache next: {kv_next} ({repr(kv_tok_str)})")
        print(f"    HF next:       {hf_next_from_kv} ({repr(hf_tok_str)})")
        print(f"    Max logit diff: {logit_diff:.6f}, Mean: {mean_diff:.6f}")
        
        if kv_next != hf_next_from_kv:
            # Show top-5 for both
            top5_kv = torch.topk(kv_next_logits, 5)
            top5_hf = torch.topk(hf_next_logits, 5)
            print(f"    KV top5: {[(i, f'{v:.3f}') for i, v in zip(top5_kv.indices.tolist(), top5_kv.values.tolist())]}")
            print(f"    HF top5: {[(i, f'{v:.3f}') for i, v in zip(top5_hf.indices.tolist(), top5_hf.values.tolist())]}")
            
            # Check specific logit values
            print(f"    KV logit[{kv_next}]={kv_next_logits[kv_next]:.4f}, KV logit[{hf_next_from_kv}]={kv_next_logits[hf_next_from_kv]:.4f}")
            print(f"    HF logit[{kv_next}]={hf_next_logits[kv_next]:.4f}, HF logit[{hf_next_from_kv}]={hf_next_logits[hf_next_from_kv]:.4f}")
            print(f"    STOPPING - found divergence point")
            break
        
        kv_token = kv_next
