"""
简化版FunctionGemma训练脚本 - 完全避免bitsandbytes
"""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import yaml
from pathlib import Path
import json
from tqdm import tqdm

print("Loading basic dependencies...")
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

print("All dependencies loaded successfully")

def main():
    print("\n" + "="*60)
    print("FunctionGemma Training - Simplified Version")
    print("="*60 + "\n")
    
    # Load config
    with open('training_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['base_model']
    cache_dir = config['model'].get('cache_dir', './model_cache')
    
    # 判断是本地路径还是 HuggingFace model ID
    is_local = os.path.isdir(model_name) or model_name.startswith('./')
    
    print(f"Model: {model_name}")
    print(f"Cache: {cache_dir}")
    print(f"Local model: {is_local}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if not is_local:
        tokenizer_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CRITICAL FIX: Set padding side to 'right' for causal LM training
    tokenizer.padding_side = 'right'
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}, padding_side={tokenizer.padding_side}\n")
    
    # Clear GPU cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    # Load model with fp16 and aggressive memory optimization
    print("Loading model (FP32, Trainer handles FP16 AMP)...")
    model_kwargs = {
        "torch_dtype": torch.float32,
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if not is_local:
        model_kwargs["cache_dir"] = cache_dir
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters\n")
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    use_lora = config.get('lora', {}).get('enabled', True)
    
    if use_lora:
        # Configure LoRA
        print("Configuring LoRA...")
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=None,
        )
        model = get_peft_model(model, lora_config)
    else:
        # Full fine-tuning - 所有参数可训练
        print("Full fine-tuning mode (no LoRA)")
        for param in model.parameters():
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Total: {total_params:,}\n")
    
    # Load dataset (offline mode)
    print("Loading dataset...")
    train_file = Path(config['dataset']['train_file'])
    eval_file = Path(config['dataset']['eval_file'])
    
    # Load locally without internet access
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    dataset = load_dataset('json', data_files={
        'train': str(train_file),
        'validation': str(eval_file)
    }, trust_remote_code=False)
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Eval samples: {len(dataset['validation'])}\n")
    
    # 找到 response 起始标记的 token IDs
    # "<start_of_turn>model\n" -> [bos, <start_of_turn>, m, o, d, e, l, \n]
    response_marker_ids = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
    # 去掉 bos 如果 encode 自动添加了 (取最后几个 token 作为标记)
    if response_marker_ids and response_marker_ids[0] == tokenizer.bos_token_id:
        response_marker_ids = response_marker_ids[1:]
    print(f"Response marker IDs: {response_marker_ids}")
    
    # Tokenize
    print("Tokenizing dataset...")
    def preprocess(examples):
        texts = examples['text']
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=config['model']['max_length'],
            padding='max_length',
            return_tensors=None,
        )
        # Set labels: mask prompt tokens (before model response) with -100
        labels = []
        marker_len = len(response_marker_ids)
        for input_ids in tokenized['input_ids']:
            label_ids = input_ids.copy()
            
            # 找到 response 起始位置 ("<start_of_turn>model\n" 之后)
            response_start = -1
            for i in range(len(input_ids) - marker_len + 1):
                if input_ids[i:i+marker_len] == response_marker_ids:
                    # 找最后一个匹配 (可能 prompt 中也有 model 标记)
                    response_start = i + marker_len
            
            # Mask: prompt tokens -> -100, response tokens -> keep, padding -> -100
            for i in range(len(label_ids)):
                if label_ids[i] == tokenizer.pad_token_id:
                    label_ids[i] = -100  # padding
                elif response_start > 0 and i < response_start:
                    label_ids[i] = -100  # prompt tokens
            
            labels.append(label_ids)
        tokenized['labels'] = labels
        return tokenized
    
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    print("Tokenization complete\n")
    
    # DEBUG: Inspect the first example to verify data processing
    print("DEBUG: Inspecting first training sample...")
    debug_ex = tokenized_dataset['train'][0]
    print(f"Input text length: {len(dataset['train'][0]['text'])}")
    print(f"Input IDs length: {len(debug_ex['input_ids'])}")
    print(f"Labels length: {len(debug_ex['labels'])}")
    print(f"First 50 Input IDs: {debug_ex['input_ids'][:50]}")
    print(f"First 50 Labels: {debug_ex['labels'][:50]}")
    non_masked_labels = [x for x in debug_ex['labels'] if x != -100]
    print(f"Number of non-masked labels: {len(non_masked_labels)}")
    print(f"First 10 non-masked labels: {non_masked_labels[:10]}")
    print(f"Tokenizer Pad ID: {tokenizer.pad_token_id}")
    print(f"Tokenizer EOS ID: {tokenizer.eos_token_id}")
    print("-" * 40)

    # Training arguments
    train_config = config['training']
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=float(train_config['learning_rate']),
        weight_decay=train_config['weight_decay'],
        warmup_steps=train_config['warmup_steps'],
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        eval_steps=train_config['eval_steps'],
        fp16=train_config['fp16'],
        gradient_checkpointing=train_config['gradient_checkpointing'],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
    )
    
    # Start training
    print("="*60)
    print("Starting training...")
    print(f"Total steps: {len(tokenized_dataset['train']) // (train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']) * train_config['num_train_epochs']}")
    print("="*60 + "\n")
    
    try:
        trainer.train()
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        
        # Save final model
        output_dir = Path(train_config['output_dir']) / "final"
        if use_lora:
            # LoRA: 保存 adapter
            model.save_pretrained(output_dir)
        else:
            # Full fine-tuning: 保存完整模型
            model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\nModel saved to: {output_dir}")
        print(f"Mode: {'LoRA adapter' if use_lora else 'Full model'}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        save_dir = Path(train_config['output_dir']) / "interrupted"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Progress saved to: {save_dir}")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
