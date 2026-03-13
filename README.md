# XiaotianAssistant — 小天AI语音助手

Android 手机上的离线AI语音助手，使用LLM大模型解析用户指令，支持音乐播放控制 8 大指令。

开发时运行环境：Android 15，QQ播放器V14.1

## 支持的命令

说你好小天，唤醒语音助手。播放音乐需要授予APP无障碍权限。

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
│              │   Pruned Gemma3 107M INT8   │     │
│              │   130.5 MB + BPE Tokenizer     │     │
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
│       ├── model_xnnpack.pte             # ExecuTorch INT8 v8 (130.5 MB)
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
│   │       ├── model_xnnpack.pte         # ExecuTorch INT8 v8 (130.5 MB)
│   │       ├── tokenizer.json            # BPE 分词器
│   │       ├── silero_vad.onnx           # Silero VAD 模型 (0.61 MB)
│   │       ├── sherpa-onnx-1.12.25.aar   # Sherpa-ONNX 推理运行时 (libs/)
│   │       └── sherpa-onnx-sense-voice-*/  # SenseVoice INT8 (~230 MB) + tokens.txt
│   └── build.gradle
└── README.md
```

## 推理结果

### 设备信息

| 项目 | 值 |
|------|-----|
| 设备 | Motorola |
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

```



## 许可证

MIT License
