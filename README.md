# BoltAssistant — 端侧智能语音助手

基于 FunctionGemma-270M 词表裁剪 + 全量微调，通过 ExecuTorch XNNPACK 部署在 Android 手机上的离线语音助手。支持音乐播放控制 8 大指令。

开发时运行环境：Android 15，QQ播放器V14.1

## 🎯 项目亮点

- **全 18 层模型方案 (v9)** — 保留原始 18 层不裁剪，仅词表裁剪 262K→10,918，参数 107.3M，推理质量远优于 8 层
- **端侧 1.1s 推理** — ExecuTorch XNNPACK INT8 Dynamic，130.5 MB 模型，Prefill 179ms + Decode 1166ms
- **RoPE 双频率 Bug 修复** — 发现 Gemma3 的 `rotary_emb`(global) 和 `rotary_emb_local`(local) 使用不同频率，KV-Cache 封装必须分别预计算，否则 15/18 层位置编码错误导致乱码输出
- **模型推理直出** — 模型正确生成 `{"f":"play_song","p":{"s":"后来"}}` 等 function call，无需降级到关键词匹配
- **XNNPACK 委派 Bug 修复** — 发现 `to_edge()` 分解 Linear 导致 87% 算力运行在非优化后端，改用 `to_edge_transform_and_lower()` 修复
- **INT8 动态量化** — XNNPACKQuantizer + per-channel weight + dynamic activation
- **SDPA 注意力融合** — `F.scaled_dot_product_attention` 替代手动注意力计算，Prefill 8.6x 加速
- **Token 数优化** — 去除标签格式，token 数 62 → 32（-48%），无标签 JSON 输出
- **离线语音识别** — SenseVoice (Sherpa-ONNX) 中文离线 + Google 在线双引擎，Silero VAD 语音端点检测，硬件降噪三重加持
- **抗噪语音增强** — NoiseSuppressor + AutomaticGainControl + AcousticEchoCanceler，SenseVoice 大规模多条件训练数据，噪声鲁棒性远超 Vosk Kaldi TDNN-F
- **车载噪声适配** — VOICE_COMMUNICATION 音源 + 硬件 DSP 降噪链路，SenseVoice INT8 量化模型 ~230MB
- **QQ Music 搜索播放提速** — 智能轮询替代固定 sleep，搜索到播放全流程 11.5s → **3.1s**（3.7x 加速）
- **语音唤醒** — "你好小天" 唤醒词，编辑距离模糊匹配容忍 1 字识别误差，前台 Service 常驻监听
- **多音乐 App** — QQ 音乐 / 网易云音乐 无障碍服务控制
- **TTS 语音回复** — 每条指令执行后由 Android Google TTS 引擎播报结果（中文）；唤醒时播报"我在"替代提示音；播报期间自动暂停麦克风防回声
- **软件抗噪增强** — 250Hz Butterworth 高通 + 预加重 + 自适应噪底追踪 + SNR 段校验 + 动态 VAD 阈值，应对车载高速风噪/胎噪
- **唤醒零感延迟** — 修复识别线程 self-join 死锁（节省 ~3s），TTS 先排队再暂停麦克风，重启监听 500ms→200ms
- **歌名原文直传** — 关键词命中时跳过 AI 模型，杜绝模型幻觉篡改中文歌名；修复歌名/歌手字段错位
- **界面全内容可滚** — ConstraintLayout → ScrollView+LinearLayout，音乐 App 与无障碍服务并排，支持指令区完整显示

## 推理性能对比

| 指标 | v3 (基线) | v8 (8层+INT8) | **v9 (18层+INT8)** | 说明 |
|------|----------|--------------|-------------------|------|
| **Output Tokens** | 62 | 32 | **23** | v9 JSON 更紧凑 |
| **Prefill** | 1780ms | 171ms | **179ms** | 18 tokens batch |
| **Decode Total** | 14,763ms | 381ms | **1166ms** | 43ms/tok |
| **Decode Speed** | 238ms/tok | 11ms/tok | **43ms/tok** | 18层计算量更大 |
| **总耗时** | 16,588ms | 554ms | **1352ms** | 端到端 |
| 模型参数 | 73.9M (12层) | 51.6M (8层) | **107.3M (18层)** | 全层保留 |
| 模型大小 | 282 MB | 76.4 MB | **130.5 MB** | INT8 Dynamic |
| 推理质量 | ✅ | ✅ | ✅✅ | **v9 无需降级** |

**关键优化历程：**
- **v3→v8**：层裁剪 + 去标签 + SDPA + 委派修复 + INT8，总耗时 16.6s→0.55s (**29.9x 加速**)
- **v8→v9**：回退到全 18 层（不裁剪层），推理质量大幅提升，模型直出正确 function call
- **v9 关键修复**：RoPE 双频率 Bug — Gemma3 的 sliding_attention 和 full_attention 使用不同 RoPE 频率，必须分别预计算
- **v9 当前状态**：1.35s 端侧推理，模型直出无需关键词降级，推理质量为最终部署版本

## 技术架构

```
┌─────────────────────────────────────────────────┐
│                  Android App                     │
│                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ 语音唤醒  │→│SenseVoice/   │→│  文本指令   │  │
│  │ "你好小天" │  │Google 语音识别│  │            │  │
│  └──────────┘  └──────────────┘  └─────┬─────┘  │
│                                        │         │
│                                        ▼         │
│              ┌─────────────────────────────┐     │
│              │   SmartAssistantAgent        │     │
│              │   (ExecuTorch XNNPACK 推理)  │     │
│              │   Pruned Gemma3 51.6M INT8   │     │
│              │   76.4MB + BPE Tokenizer     │     │
│              └─────────────┬───────────────┘     │
│                            │                     │
│              ┌─────────────▼───────────────┐     │
│              │   SmartFunctionExecutor      │     │
│              │   → Android Intent 执行      │     │
│              │   → 无障碍服务控制音乐 App    │     │
│              └─────────────┬───────────────┘     │
│                            │                     │
│              ┌─────────────▼───────────────┐     │
│              │   TtsManager (Android TTS)   │     │
│              │   Google TTS 语音播报结果    │     │
│              │   麦克风自动暂停/恢复         │     │
│              └─────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

## 项目结构

```
boltassistant/
├── training/                              # 模型训练
│   ├── create_pruned_model.py            # 模型裁剪 (词表 + 层数)
│   ├── augment_data.py                   # 数据增强 & 平衡
│   ├── train_simple.py                   # 全量微调训练脚本
│   ├── training_config.yaml              # 训练配置
│   ├── evaluate_model.py                 # 模型评估
│   ├── data/
│   │   ├── music_control_train_balanced.jsonl  # 训练数据 (2,628 条)
│   │   └── music_control_eval_balanced.jsonl   # 评估数据 (571 条)
│   ├── output/
│   │   └── gemma_finetuned_v3/final/     # 全量微调模型
│   └── model_cache/
│       └── pruned_model/                 # 裁剪后的基础模型
├── conversion/                            # 模型导出
│   ├── export_pipeline.py                # ExecuTorch 导出流水线
│   ├── merged_model/                     # 合并后 HF 模型
│   └── executorch_output/
│       ├── model_xnnpack.pte             # ExecuTorch INT8 v8 (76.4 MB, 8层)
│       └── tokenizer.json                # BPE 分词器
├── android/                               # Android 工程
│   ├── app/src/main/
│   │   ├── java/com/xiaotian/assistant/
│   │   │   ├── MainActivity.kt           # 主界面
│   │   │   ├── ExecuTorchModelManager.kt # ExecuTorch 推理 + BPE 分词
│   │   │   ├── SmartAssistantAgent.kt    # AI Agent 协调器
│   │   │   ├── SmartCommand.kt           # 命令定义 (8 个 function)
│   │   │   ├── SmartFunctionExecutor.kt  # 命令执行器
│   │   │   ├── HybridVoiceRecognizer.kt  # SenseVoice+Google 混合识别
│   │   │   ├── SherpaRecognizer.kt       # SenseVoice 离线 ASR (Sherpa-ONNX)
│   │   │   ├── VoiceWakeupService.kt     # 语音唤醒前台服务
│   │   │   ├── TtsManager.kt             # TTS 语音回复（Google TTS，麦克风自动暂停）
│   │   │   ├── QQMusicController.kt      # QQ 音乐控制
│   │   │   └── NeteaseMusicController.kt # 网易云音乐控制
│   │   └── assets/
│   │       ├── model_xnnpack.pte         # ExecuTorch INT8 v8 (76.4 MB, 8层)
│   │       ├── tokenizer.json            # BPE 分词器
│   │       ├── silero_vad.onnx           # Silero VAD 模型 (0.61 MB)
│   │       ├── sherpa-onnx-1.12.25.aar   # Sherpa-ONNX 推理运行时 (libs/)
│   │       └── sherpa-onnx-sense-voice-*/  # SenseVoice INT8 (~230 MB) + tokens.txt
│   └── build.gradle
└── README.md
```

## 支持的命令

| 命令 | 示例 | Function Call |
|------|------|---------------|
| 播放歌曲 | "播放周杰伦的稻香" | `play_song(song_name, artist)` |
| 暂停 | "暂停" | `pause_music()` |
| 继续播放 | "继续播放" | `resume_music()` |
| 下一首 | "下一首" | `next_song()` |
| 上一首 | "上一首" | `previous_song()` |
| 绝对音量 | "音量调到50" | `set_volume(level)` |
| 调高音量 | "大声点" / "调高音量" | `volume_up()` |
| 调低音量 | "小声点" / "调低音量" | `volume_down()` |

## 模型裁剪

从 Google FunctionGemma-270M-IT 出发进行深度裁剪，大幅减少参数量和推理开销：

### 裁剪策略

| 维度 | 原始 | v3 (12层) | v5 (8层) | **v9 (18层)** | 缩减 (v9) |
|------|------|-----------|----------|--------------|----------|
| 词表大小 | 262,144 | 10,919 | 10,920 | **10,918** | **95.8%** |
| 隐藏层数 | 18 | 12 | 8 | **18 (全保留)** | **0%** |
| 总参数量 | 268.1M | 73.9M | 51.6M | **107.3M** | **60.0%** |
| Embedding | 335.5M params | 14.0M params | 7.0M params | **14.0M params** | **97.9%** |
| 模型大小 | 511 MB | 282 MB | 197 MB | **130.5 MB** (INT8) | **74.5%** |

> **v9 方案选择**：v5/v8 的 8 层裁剪虽然推理速度更快 (554ms)，但发现层裁剪对小模型的推理质量影响较大。v9 恢复全 18 层，仅做词表裁剪 + INT8 量化，推理质量显著提升，1.35s 延迟在语音助手场景下完全可接受。

### 词表裁剪细节

1. **收集必要 token**: 扫描训练数据中所有出现的 token
2. **保留特殊 token**: `<bos>`, `<eos>`, `<pad>`, `<start_of_turn>`, `<end_of_turn>` 等
3. **保留 byte fallback**: `<0x00>` ~ `<0xFF>` (256 个) 确保任意字符可编码
4. **重映射 Embedding**: 只保留被选中的 10,919 个 token 的 embedding 权重
5. **输出 token_mapping.json**: 原始 ID → 新 ID 映射表

### 层裁剪

**v3 (12层)**: 原始 18 层均匀跳过 6 层，保留 12 层 `[0,2,4,5,6,8,10,11,12,14,16,17]`

**v5 (8层)**: 进一步裁剪至 8 层 `[0,3,5,8,11,14,16,17]`，保留所有 `full_attention` 层 (5/11/17)，前中后均匀采样 `sliding_attention` 层

- **v5 层分布**: `[S, S, F, S, F, S, S, F]` (S=sliding_attention, F=full_attention)
- 保持 `hidden_size=1536`, `num_attention_heads=4`, `head_dim=256` 不变
- **参数量**: 73.9M → 51.6M (**30% 减少**)
- **模型大小**: 282 MB → 197 MB (**30% 减少**)

**v9 (18层，当前方案)**: 不裁剪层，保留全部 18 层原始架构

- **层分布**: `[S,S,S,S,S,F,S,S,S,S,S,F,S,S,S,S,S,F]` (15 sliding + 3 full)
- 仅词表裁剪 262K → 10,918
- **参数量**: 107.3M，**模型大小**: 409 MB (FP32) / **130.5 MB (INT8)**
- **原因**: 8 层裁剪对小模型推理质量影响较大，full 18 层生成的 function call 更准确，不再需要关键词降级

## 数据准备

### 原始数据问题

原始训练数据存在严重不平衡：

| Function | 原始数量 | 占比 |
|----------|----------|------|
| play_song | 4,508 | 98.2% |
| pause_music | 5 | 0.1% |
| resume_music | 30 | 0.7% |
| next_song | 3 | 0.1% |
| previous_song | 3 | 0.1% |
| set_volume | 40 | 0.9% |

### 数据增强 (augment_data.py)

通过模板生成 + play_song 下采样，得到平衡数据集：

| Function | 增强后数量 | 操作 |
|----------|-----------|------|
| play_song | 2,000 | 从 4,508 下采样 |
| set_volume | 440 | 模板生成 (0~100 各级别) |
| pause_music | 55 | 模板生成 |
| resume_music | 48 | 模板生成 |
| next_song | 44 | 模板生成 |
| previous_song | 41 | 模板生成 |
| **合计** | **2,628** | 训练集 |

评估集: 571 条 (同样平衡)

## 训练

### 训练方案演进

| 版本 | 方案 | 问题 |
|------|------|------|
| v1 | LoRA (r=16, 3.4% 可训练参数) | 参数太少，裁剪后小模型 LoRA 效果差 |
| v2 | LoRA + 4508 条不平衡数据 | 98% play_song，其他功能无法学会 |
| **v3** | **全量微调 + 平衡数据 + Prompt Masking** | ✅ 所有功能均正确 |

### v3 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | 裁剪后 Gemma3 v5 (51.6M 参数, 8层) |
| 微调方法 | **全量微调** (100% 参数可训练) |
| Prompt Masking | ✅ 只在 `<start_of_turn>model\n` 之后计算 loss |
| 学习率 | 3e-5 |
| 训练轮次 | 10 epochs (1,650 steps) |
| 批次大小 | 4 × 4 gradient accumulation = 有效 batch 16 |
| 序列长度 | 256 tokens |
| 精度 | FP32 |
| 训练时长 | ~32 分钟 (RTX 4050) |

### 训练结果

| 指标 | 值 |
|------|-----|
| 初始 train loss | 0.82 |
| 最终 train loss | 0.0 |
| 最终 eval loss | 0.002 |
| 训练步数 | 1,650 |

### Python 验证 (7/7 通过)

```
输入: "播放周杰伦的稻香"  → play_song, song_name:"稻香", artist:"周杰伦"  ✅
输入: "暂停"            → pause_music                                   ✅
输入: "下一首"          → next_song                                     ✅
输入: "上一首"          → previous_song                                 ✅
输入: "音量调到50"       → set_volume, level:50                          ✅
输入: "放一首冷雨夜"     → play_song, song_name:"冷雨夜"                 ✅
输入: "继续播放"         → resume_music                                  ✅
```

## 模型导出

### 导出流水线 (export_pipeline.py)

```
全量微调模型 (training/output/gemma_finetuned_v3/final/)
      │
      ▼ Step 0: 合并模型 (复制到 conversion/merged_model/)
      │
      ▼ Step 1: 包装 KV-Cache (StaticKVCacheWrapper)
      │         max_cache_len=80, head_dim=256
      │         SDPA attention (F.scaled_dot_product_attention)
      │
      ▼ Step 2: torch.export (动态 shape, batch prefill)
      │
      ▼ Step 3: INT8 动态量化 (XNNPACKQuantizer)
      │         per-channel weight + dynamic activation
      │         4 条代表性 prompt 校准
      │
      ▼ Step 4: to_edge_transform_and_lower (XNNPACK 委派)
      │         保留 aten.linear.default，避免 mm 分解
      │
      ▼ model_xnnpack.pte (76.4 MB, INT8 Dynamic, 8层)
```

### 导出结果

| 阶段 | 大小 | 说明 |
|------|------|------|
| 裁剪后 HF 模型 v5 | ~197 MB | FP32 safetensors, 8层 |
| ExecuTorch FP32 .pte | 223.7 MB | XNNPACK FP32 + 正确委派 |
| **ExecuTorch INT8 .pte** | **76.4 MB** | **XNNPACK INT8 Dynamic 最终部署** |

## 推理优化方案

### 1. Token 数优化 (v6) — 去除格式标签

#### 问题分析
- **原格式**: `<function_call>{"f": "play_song", ...}</function_call>`  
  - `<function_call>` 占 16 tokens (字符级编码，非特殊 token)
  - `</function_call>` 占 16 tokens
  - **纯格式开销 32 tokens，占总生成 token 数 52%**

- **新格式**: `{"f": "play_song", ...}` （无标签）
  - JSON 本身 41 tokens (vs 原 71 tokens)
  - **节省 30 tokens，总 token 数 62 → 32 (-48%)**

#### 实现方案

1. **训练数据优化** (training/data/*.jsonl)
   ```python
   # 去除标签，保只保留 JSON
   text = text.replace('<function_call>', '')
   text = text.replace('</function_call>', '')
   ```
   - 所有 4 个 JSONL 文件已处理
   - 备份保存为 .bak 文件

2. **模型微调** (v4)
   - 用无标签数据重新全量微调（10 epochs）
   - 模型自然学到新格式，无需推理时抑制
   - 训练损失正常收敛 (eval_loss 0.0037)

3. **推理输出**
   - v4 模型直接以 `{"` 开头
   - SmartAssistantAgent 支持 3 种解析模式：
     - 标签提取 (旧)
     - 直接 JSON regex (新)
     - 花括号匹配

#### 性能对比
| 指标 | v3 (有标签) | v6 (无标签) |
|------|-----------|-----------|
| Output tokens | 62 | 32 |
| Decode 时间 | 14,763ms | 8,085ms |
| Decode 速度 | 238 ms/token | 252 ms/token |
| **总推理时间** | **16,588ms** | **9,892ms** |

### 2. Early Stop 优化 — JSON 花括号平衡检测

#### 问题与修复
- **BPE 多字符 token**: tokenizer 可能将 `}}` 编码为单个 token (id 360)
- **原逻辑缺陷**: 只检查单字符 token ID，无法识别 `}}`
- **修复方案**: 检查 token 文本中是否包含 '}'，而非精确匹配 ID

#### 实现 (ExecuTorchModelManager.kt)
```kotlin
val tokenText = idToToken[nextToken] ?: ""
if ('}' in tokenText) {
    val openCount = currentText.count { it == '{' }
    val closeCount = currentText.count { it == '}' }
    if (openCount > 0 && openCount == closeCount) {
        // JSON 平衡，立即停止
        break
    }
}
```

#### 效果
- 检测 2 对花括号平衡，在 step 31 停止
- 相比原 step 46，节省 15 tokens

### 3. 模型缓存更新检测

#### 根本问题
- **新旧模型大小相同**: 103,503,440 字节
- **文件大小比对失败**: 代码无法检测模型更新
- **后果**: 手机缓存继续使用旧模型，新模型未应用

#### 解决方案
```kotlin
val needsCopy = !modelFile.exists() ||
    (assetSize > 0 && modelFile.length() != assetSize) ||
    modelFile.lastModified() < getApkLastModified(context)
```

添加 APK lastUpdateTime 检测，确保 APK 更新时自动复制新模型。

### 4. 推理优化总结 (v3→v6)

| 优化项 | 方案 | 效果 |
|--------|------|------|
| **模型层裁剪** | 18 → 12 → 8 层 | **-30% 参数，-30% 大小** |
| **Token 数减少** | 去除标签格式 | **Token 62 → 32 (-48%)** |
| **Early Stop** | JSON 花括号平衡 | 自动提前停止，节省 15 tokens |
| **Batch Prefill** | 批量处理 prompt | **32.8x prefill 加速** |
| **缓存检测** | APK 时间戳 | **自动同步新模型** |
| **累计效果** | 综合优化 | **端到端 40% 加速 (16.6s → 9.9s)** |

### 5. SDPA 注意力融合 (v7) — Prefill 8.6x 加速

#### 问题分析

原始注意力实现为手动四步运算：

```python
# 原始手动注意力 (4次矩阵操作)
attn_weights = torch.matmul(q, k.transpose(-2, -1))   # Q·K^T
attn_weights = attn_weights * self.scaling              # scale
attn_weights = attn_weights + causal_mask               # mask (4D)
attn_weights = F.softmax(attn_weights, dim=-1)          # softmax
attn_output = torch.matmul(attn_weights, v)             # ·V
```

ExecuTorch 导出后每步生成独立算子，XNNPACK 无法融合优化。

#### 优化方案

替换为 PyTorch 原生 `F.scaled_dot_product_attention`，将 5 个操作融合为 1 个 SDPA kernel：

```python
# 融合后 SDPA (单次调用)
# 1. 使用 repeat_kv 扩展 KV heads 以匹配 Q heads (GQA 4:1)
k = repeat_kv(k, self.num_key_value_groups)  # [1,1,cache,256] → [1,4,cache,256]
v = repeat_kv(v, self.num_key_value_groups)

# 2. 2D mask [seq_len, cache_len] 替代 4D mask (XNNPACK 兼容)
attn_output = F.scaled_dot_product_attention(
    q, k, v, attn_mask=mask, scale=self.scaling
)
```

**关键细节**：
- **Mask 降维**: 从 `[1, num_heads, seq, cache]` 4D 改为 `[seq, cache]` 2D，因为 XNNPACK SDPA kernel 不支持 4D mask
- **GQA 处理**: 测试了 `enable_gqa=True` 参数，Prefill 快 8.6x 但 Decode 慢 27%；最终采用显式 `repeat_kv` + SDPA，Prefill 同样 8.6x 快但 Decode 不受影响
- **Causal Mask 生成**: 导出时直接在 wrapper 内部用 `torch.triu(torch.full(..., -inf), diagonal=1)` 构造

#### 性能结果

| 指标 | v6 (手动注意力) | v7 (SDPA) | 提升 |
|------|----------------|-----------|------|
| Prefill (23 tok) | 1803ms (78ms/tok) | **209ms (9ms/tok)** | **8.6x** |
| Decode (per tok) | 252ms | 249ms | ~同 |
| 总耗时 | 9,892ms | **8,192ms** | **1.2x** |

### 6. XNNPACK 委派 Bug 发现与修复 (v8) — 关键性能突破

#### 🔍 问题发现过程

在分析 v7 导出日志时，发现 XNNPACK 委派率异常：

```
[INFO] XNNPACK delegation statistics:
  Delegated ops: 122/938 (13.0%)
  Non-delegated ops: 816/938 (87.0%)
  
  Top non-delegated ops:
    aten_mm_default: 57        ← 最重的矩阵乘法全部未委派！
    aten_add_tensor: 41
    aten_mul_tensor: 40
    ...
```

**57 个 `mm` (矩阵乘法) 操作 —— 模型中计算量最大的操作 —— 全部运行在非优化的 portable 后端！** 这意味着模型 87% 的算力浪费在了没有 SIMD/NEON 优化的 fallback 路径上。

#### 🔬 根因分析

深入 XNNPACK 源码追查原因：

**1. `to_edge()` 的算子分解问题**

```python
# 原代码使用 to_edge()
edge = to_edge(exported)
edge = edge.to_backend(XnnpackPartitioner(...))
```

`to_edge()` 会将 `nn.Linear(x, weight, bias)` 分解为两个算子：
```
aten.linear.default  →  aten.permute_copy.default + aten.mm.default
                         (转置权重)               (矩阵乘法)
```

**2. XNNPACK Partitioner 拒绝 `mm` 的原因**

XNNPACK 的 `GEMMConfig._get_weight_deps()` (在 `gemm_configs.py` 中) 会检查 `mm` 算子的权重输入是否是静态参数：

```python
# XNNPACK 内部检查逻辑
def _get_weight_deps(self, node):
    weight_node = node.args[1]         # mm 的第二个参数 (权重)
    if not is_param_node(weight_node): # 检查是否是模型参数
        return []  # ← 返回空，表示 "不支持"
```

问题在于：分解后 `mm` 的权重输入是 `permute_copy` 的输出（一个 `call_function` 节点），**不是** 原始的 `placeholder` 参数节点。`is_param_node()` 检查失败，XNNPACK 认为 "权重不是静态参数"，拒绝委派。

```
nn.Linear(x, W)
    ↓ to_edge() 分解
permute_copy(W) → mm(x, permuted_W)
                       ↑
          weight 是 permute_copy 的输出 (call_function)
          而非原始参数 (placeholder)
          → is_param_node() = False
          → XNNPACK 拒绝: "Expected weight to be a static param"
```

**3. 影响范围**

Gemma3 模型每层 Transformer 有 7 个 Linear 操作 (Q/K/V/O projection + gate/up/down FFN)。8 层 × 7 = 56 个 Linear + 1 个 lm_head = **57 个 mm 全部未委派**。这些矩阵乘法占模型 90%+ 的计算量。

#### ✅ 修复方案

使用 `to_edge_transform_and_lower()` 替代 `to_edge() + to_backend()`：

```python
# 修复前 (有 Bug)
from executorch.exir import to_edge
edge = to_edge(exported)
edge = edge.to_backend(XnnpackPartitioner(config_precisions=precision))

# 修复后 (正确)
from executorch.exir import to_edge_transform_and_lower
edge = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner(config_precisions=precision)]
)
```

**为什么有效**：`to_edge_transform_and_lower()` 在委派阶段保留 `aten.linear.default` 不做分解。XNNPACK 的 `LinearConfig` 直接匹配 `aten.linear.default`，权重输入仍然是原始 `placeholder` 参数节点，`is_param_node()` 检查通过。

#### 修复验证

```
修复前: mm NOT in delegated list (57 ops on portable backend)
修复后: mm 不再出现在 non-delegated list (all Linear ops delegated via XNNPACK)
```

FP32 + 正确委派测试：
| 指标 | 修复前 (v7) | 修复后 (FP32) | 提升 |
|------|------------|--------------|------|
| Prefill | 209ms | **224ms** | ~同 |
| Decode/tok | 249ms | **11ms** | **22.6x** |
| 总耗时 | 8,192ms | **589ms** | **13.9x** |

**仅修复委派 Bug（不加量化），Decode 就从 249ms/tok 降到 11ms/tok！**

### 7. INT8 动态量化 (v8) — 模型体积 -61%

#### 量化方案

在修复委派 Bug 的基础上，叠加 INT8 动态量化进一步优化：

```python
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

# 配置量化器
quantizer = XNNPACKQuantizer().set_global(
    get_symmetric_quantization_config(
        is_per_channel=True,   # 权重 per-channel 量化 (精度更高)
        is_dynamic=True        # 激活值动态量化 (无需大量校准数据)
    )
)

# 量化流程: prepare → calibrate → convert
graph_module = exported.module()
prepared = prepare_pt2e(graph_module, quantizer)

# 用 4 条代表性 prompt 校准
calibration_prompts = [
    "播放周杰伦的稻香",   # 长查询 (歌曲+歌手)
    "暂停音乐",           # 简短指令
    "下一首",             # 最短指令
    "播放一首轻音乐",     # 模糊查询
]
for prompt in calibration_prompts:
    prepared(input_ids, cache_position)

converted = convert_pt2e(prepared)
```

#### API 踩坑记录

| 尝试 | API | 错误 | 原因 |
|------|-----|------|------|
| ❌ 第1次 | `torch.ao.quantization.quantize_pt2e.prepare_pt2e(exported, ...)` | `'ExportedProgram' has no attribute 'meta'` | 需传入 `GraphModule` 而非 `ExportedProgram` |
| ❌ 第2次 | `torch.ao.quantization.quantize_pt2e.prepare_pt2e(graph_module, ...)` | `quantization_spec must be a QuantizationSpec` | `torch.ao` 已废弃，与 ExecuTorch quantizer 不兼容 |
| ✅ 第3次 | `torchao.quantization.pt2e.quantize_pt2e.prepare_pt2e(graph_module, ...)` | 成功 | `torchao` 版本与 ExecuTorch `XNNPACKQuantizer` 兼容 |

#### 量化效果

| 指标 | FP32 (修复委派) | INT8 Dynamic | 变化 |
|------|----------------|-------------|------|
| 模型大小 | 223.7 MB | **76.4 MB** | **-66%** |
| Prefill | 224ms | **171ms** | -24% |
| Decode/tok | 11ms | **11ms** | 同 |
| 总耗时 | 589ms | **554ms** | -6% |

INT8 量化主要优势在**模型体积**，推理速度与 FP32 + 正确委派基本持平。

### 8. KV Cache 缩减

将 KV Cache 从 128 缩减到 80 个 position：

- **Prompt 长度**: 约 23 tokens
- **最大输出**: 约 40 tokens (JSON 格式)
- **实际需求**: 23 + 40 = 63 tokens
- **安全余量**: 80 (够用且减少内存占用)

### 9. 推理优化总结 (全部版本)

| 优化项 | 版本 | 方案 | 效果 |
|--------|------|------|------|
| **模型层裁剪** | v5 | 18 → 12 → 8 层 | **-30% 参数，-30% 大小** |
| **Token 数减少** | v6 | 去除标签格式 | **Token 62 → 32 (-48%)** |
| **Early Stop** | v5 | JSON 花括号平衡 | 自动提前停止，节省 15 tokens |
| **Batch Prefill** | v4 | 批量处理 prompt | **32.8x prefill 加速** |
| **SDPA 融合** | v7 | `F.scaled_dot_product_attention` | **Prefill 8.6x 加速** |
| **XNNPACK 委派修复** | v8 | `to_edge_transform_and_lower` | **Decode 22.6x 加速** |
| **INT8 动态量化** | v8 | XNNPACKQuantizer | **模型 -61%，76.4MB** |
| **KV Cache 缩减** | v8 | 128→80 | 减少内存占用 |
| **累计效果** | — | 综合优化 | **端到端 29.9x (16.6s→554ms)** |


### ExecuTorch 配置

| 参数 | 值 |
|------|-----|
| ExecuTorch 版本 | 1.1.0 |
| 后端 | XNNPACK (CPU) |
| 量化 | INT8 Dynamic (XNNPACKQuantizer, per-channel weight + dynamic activation) |
| 委派 API | `to_edge_transform_and_lower()` (保留 `aten.linear.default`) |
| KV Cache | Static, max_cache_len=80 |
| XNNPACK 线程数 | 4 |
| 输出格式 | .pte (Portable Tensor Exchange) |

## 推理结果

### 设备信息

| 项目 | 值 |
|------|-----|
| 设备 | Motorola XT2507-5 |
| SoC | MediaTek Dimensity 8300 (MT6897) |
| Android | 15 |

### 性能指标 (v9 当前版本 — 18层)

| 指标 | 数值 |
|------|------|
| 模型大小 | **130.5 MB** (INT8 Dynamic, 18层) |
| 模型参数 | **107.3M** (词表裁剪，全层保留) |
| Prefill | 18-22 tokens, **179-200ms (8-11 ms/token, batch)** |
| Decode | 23-27 tokens, **943-1166ms (41-43 ms/token)** |
| 总推理时间 | **1.1-1.4s** |
| Stop 条件 | JSON 花括号平衡检测 (22-26 步) |
| 优化方案 | **INT8 Dynamic + XNNPACK 委派修复 + SDPA + 全 18 层** |

### 端侧推理示例 (v9)

**测试 1: "播放后来"**
```
Prefill: 200ms (18 tokens, 11ms/tok, batch)
Decode:  943ms (23 tokens, 41ms/tok)
Total:   1147ms

模型输出: {"f": "play_song", "p": {"s": "后来"}}
最终命令: 播放歌曲 - 歌曲: 后来
→ QQ音乐搜索播放, 返回 true ✅
```

**测试 2: "播放别怕我伤心"**
```
Prefill: 179ms (22 tokens, 8ms/tok, batch)
Decode:  1166ms (27 tokens, 43ms/tok)
Total:   1352ms

模型输出: {"f": "play_song", "p": {"s": "别怕我伤心"}}
最终命令: 播放歌曲 - 歌曲: 别怕我伤心
→ QQ音乐搜索播放, 返回 true ✅
```

**测试 3: "播放响起"**
```
Prefill: 160ms (19 tokens, 8ms/tok, batch)
Decode:  1003ms (24 tokens, 41ms/tok)
Total:   1163ms

模型输出: {"f": "play_song", "p": {"s": "响起"}}
→ QQ音乐搜索播放, 返回 true ✅
```

**安全检查: "我有" (噪声误识别)**
```
模型输出: {"f": "play_song", "p": {"s": "有"}}
→ 安全检查: 原始输入无音乐意图，判定为模型幻觉，拦截 ✅
最终命令: 未识别
```

<details>
<summary>v8 历史推理数据 (8层方案)</summary>

| 指标 | 数值 |
|------|------|
| 模型大小 | **76.4 MB** (INT8 Dynamic, 8层) |
| Prefill | 23 tokens, **171 ms (7 ms/token, batch)** |
| Decode | 32 tokens, **381 ms (11 ms/token)** |
| 总推理时间 | **554 ms** |

**"播放周杰伦的稻香"**: Prefill 171ms, Decode 381ms, Total 554ms → `{"f":"play_song","p":{"s":"稻香","a":"周杰伦"}}` ✅

**"暂停音乐"**: Prefill 184ms, Decode 272ms, Total 458ms → `{"f":"pause_music","p":{}}` ✅

</details>


## 许可证

MIT License
