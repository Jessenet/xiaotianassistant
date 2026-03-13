"""
评估微调后的FunctionGemma模型在function calling任务上的准确率
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def load_model(base_model_name, adapter_path, cache_dir="./model_cache"):
    """加载基础模型和LoRA适配器"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer

def extract_function_call(text):
    """从生成的文本中提取function call JSON"""
    import re
    # 查找 <function_call>...</function_call> 标签
    match = re.search(r'<function_call>(.*?)</function_call>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def evaluate_on_dataset(model, tokenizer, eval_file, max_samples=None):
    """在评估数据集上测试模型"""
    print(f"\nLoading evaluation data from {eval_file}...")
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]
    
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    print(f"Evaluating on {len(eval_data)} samples...\n")
    
    correct = 0
    total = 0
    errors = []
    
    for item in tqdm(eval_data, desc="Evaluating"):
        user_input = item['user_input']
        expected_call = item['function_call']
        
        # 构造输入
        prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
        
        # 生成预测
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码输出
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        predicted_call = extract_function_call(generated)
        
        # 比较预测和期望
        total += 1
        try:
            if predicted_call:
                pred_json = json.loads(predicted_call)
                exp_json = json.loads(expected_call)
                if pred_json == exp_json:
                    correct += 1
                else:
                    errors.append({
                        'input': user_input,
                        'expected': expected_call,
                        'predicted': predicted_call,
                        'reason': 'mismatch'
                    })
            else:
                errors.append({
                    'input': user_input,
                    'expected': expected_call,
                    'predicted': generated[-200:],
                    'reason': 'no_function_call'
                })
        except json.JSONDecodeError as e:
            errors.append({
                'input': user_input,
                'expected': expected_call,
                'predicted': predicted_call or generated[-200:],
                'reason': f'json_error: {str(e)}'
            })
    
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # 显示前5个错误样例
    if errors:
        print(f"First {min(5, len(errors))} errors:")
        for i, err in enumerate(errors[:5], 1):
            print(f"\n--- Error {i} ({err['reason']}) ---")
            print(f"Input: {err['input']}")
            print(f"Expected: {err['expected']}")
            print(f"Predicted: {err['predicted']}")
    
    return accuracy, correct, total, errors

def main():
    # 配置
    BASE_MODEL = "google/functiongemma-270m-it"
    ADAPTER_PATH = "./output/gemma_finetuned_v2/checkpoint-1500"  # 使用v2的最佳checkpoint
    EVAL_FILE = "./data/music_control_eval.jsonl"
    CACHE_DIR = "./model_cache"
    
    print("="*60)
    print("FunctionGemma Model Evaluation")
    print("="*60)
    
    # 加载模型
    model, tokenizer = load_model(BASE_MODEL, ADAPTER_PATH, CACHE_DIR)
    
    # 评估
    accuracy, correct, total, errors = evaluate_on_dataset(
        model, 
        tokenizer, 
        EVAL_FILE,
        max_samples=100  # 先测试100个样本
    )
    
    # 保存错误案例
    if errors:
        with open('./evaluation_errors.json', 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\nError cases saved to evaluation_errors.json")

if __name__ == "__main__":
    main()
