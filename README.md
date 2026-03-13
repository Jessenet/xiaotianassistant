# XiaotianAssistant — 小天智能语音助手

 部署在 Android 手机上的离线语音助手。支持音乐播放控制 8 大指令。

开发时运行环境：Android 15，QQ播放器V14.1


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

**"播放周杰伦的稻香"**: Prefill 171ms, Decode 381ms, Total 554ms → `{"f":"play_song","p":{"s":"稻香","a":"周杰伦"}}` ✅

**"暂停音乐"**: Prefill 184ms, Decode 272ms, Total 458ms → `{"f":"pause_music","p":{}}` ✅

</details>


## 许可证

MIT License
