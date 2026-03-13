#!/usr/bin/env python3
"""Test inference using the same KV-cache wrapper as ExecuTorch export.
This helps isolate whether the problem is INT8 quantization vs the wrapper itself."""
import sys
sys.path.insert(0, 'conversion')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from export_pipeline import GemmaWithKVCache, MAX_CACHE_LEN

MODEL_PATH = r'e:\devAI\boltassistant\conversion\merged_model'
FINETUNED_PATH = r'e:\devAI\boltassistant\training\output\gemma_finetuned_v5\final'

# Use merged_model if exists, otherwise finetuned
import os
model_path = MODEL_PATH if os.path.exists(os.path.join(MODEL_PATH, 'model.safetensors')) else FINETUNED_PATH

print(f"Loading from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
hf_model.eval()

# Wrap with KV-cache (same as ExecuTorch export)
print(f"\nCreating KV-cache wrapper (max_cache_len={MAX_CACHE_LEN})...")
wrapped = GemmaWithKVCache(hf_model, config, MAX_CACHE_LEN, model_dtype=torch.float32)
wrapped.eval()

# Build vocab map for detokenization
id_to_token = {}
for tok, idx in tokenizer.get_vocab().items():
    id_to_token[idx] = tok

query = "播放后来"
prompt = f'<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n'
input_ids = tokenizer.encode(prompt)
print(f"\nPrompt: {repr(prompt)}")
print(f"Input tokens ({len(input_ids)}): {input_ids}")

# --- Simulate ExecuTorch inference exactly ---
print(f"\n=== KV-Cache Wrapper Inference (FP32, like ExecuTorch) ===")

with torch.no_grad():
    # Batch prefill
    ids_tensor = torch.tensor([input_ids], dtype=torch.long)  # [1, seq_len]
    pos_tensor = torch.arange(len(input_ids), dtype=torch.long)  # [seq_len]
    
    logits = wrapped(ids_tensor, pos_tensor)  # [1, seq_len, vocab]
    last_logits = logits[0, -1, :]  # last token's logits
    next_token = torch.argmax(last_logits).item()
    
    generated = [next_token]
    token_text = id_to_token.get(next_token, f"[{next_token}]")
    print(f"Step 0 (prefill): token={next_token} text={repr(token_text)}")
    
    text_so_far = token_text
    
    # Decode loop
    eos_id = tokenizer.eos_token_id or 1
    stop_ids = {eos_id}
    # Add end_of_turn token
    eot_id = tokenizer.convert_tokens_to_ids('<end_of_turn>')
    if eot_id is not None:
        stop_ids.add(eot_id)
    
    for step in range(1, 80):
        pos = len(input_ids) + step - 1
        if pos >= MAX_CACHE_LEN:
            print(f"Step {step}: CACHE FULL at pos={pos}")
            break
        
        ids_tensor = torch.tensor([[next_token]], dtype=torch.long)  # [1, 1]
        pos_tensor = torch.tensor([pos], dtype=torch.long)  # [1]
        
        logits = wrapped(ids_tensor, pos_tensor)  # [1, 1, vocab]
        next_logits = logits[0, 0, :]
        next_token = torch.argmax(next_logits).item()
        
        if next_token in stop_ids:
            print(f"Step {step}: STOP token {next_token}")
            break
        
        generated.append(next_token)
        token_text = id_to_token.get(next_token, f"[{next_token}]")
        text_so_far += token_text
        
        if step <= 30 or step % 10 == 0:
            print(f"Step {step}: token={next_token} text={repr(token_text)}")
        
        # Check if JSON is complete
        open_count = text_so_far.count('{')
        close_count = text_so_far.count('}')
        if open_count > 0 and open_count == close_count and len(generated) > 5:
            print(f"Step {step}: JSON COMPLETE (braces balanced)")
            break

print(f"\n--- Result ---")
print(f"Generated {len(generated)} tokens: {generated}")
# Detokenize using SentencePiece style (replace ▁ with space)
full_text = ""
for tid in generated:
    t = id_to_token.get(tid, f"[{tid}]")
    full_text += t
full_text = full_text.replace("▁", " ")
print(f"Output text: {full_text}")
