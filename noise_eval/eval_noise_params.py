"""
语音识别降噪参数评估工具 (SenseVoice ASR)
==========================================
用途：
  把高速胎噪/风噪样本混入参考语音，在不同 SNR 下评估
  NOISE_GATE_MULTIPLIER 和 PRE_EMPHASIS_ALPHA 的最优组合。

  ASR 引擎已从 Vosk 替换为 Sherpa-ONNX + SenseVoice-Small INT8。

使用方法：
  1. 把你的噪声样本（WAV/MP3/M4A 均可）放到 noise_eval/noise/ 目录
  2. 把带标注的参考语音（静音录制）放到 noise_eval/speech/ 目录，
     同时在 noise_eval/speech/labels.txt 里写对应文本（见格式说明）
  3. 运行: python noise_eval/eval_noise_params.py
  4. 结果保存到 noise_eval/results/

labels.txt 格式（每行 文件名<TAB>正确文本）：
  你好小天.m4a\t你好小天
  播放稻香.m4a\t播放稻香
  下一首.m4a\t下一首

输出：
  - results/grid_search.csv     所有参数组合的 CER 表格
  - results/best_params.json    最优参数值
  - results/snr_curve.png       各 SNR 下准确率折线图
"""

import json
import wave
import struct
import math
import csv
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf

# ─── 配置 ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
NOISE_DIR  = ROOT / "noise"
SPEECH_DIR = ROOT / "speech"
RESULTS_DIR = ROOT / "results"
LABELS_FILE = SPEECH_DIR / "labels.txt"

# SenseVoice 模型路径 (复用 Android assets 中的模型)
ASSETS_DIR = ROOT.parent / "android" / "app" / "src" / "main" / "assets"
SENSE_VOICE_MODEL = ASSETS_DIR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" / "model.int8.onnx"
SENSE_VOICE_TOKENS = ASSETS_DIR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" / "tokens.txt"
SILERO_VAD_MODEL = ASSETS_DIR / "silero_vad.onnx"

SAMPLE_RATE = 16000  # SenseVoice 要求 16kHz

# 测试的 SNR 范围（dB）：-5 到 +20，共 6 档
SNR_LIST = [-5, 0, 5, 10, 15, 20]

# 参数搜索网格 (SenseVoice 不需要 MIN_CONFIDENCE，去掉该维度)
GATE_MULTIPLIERS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]   # NOISE_GATE_MULTIPLIER
PRE_EMPHASIS_ALPHAS = [0.0, 0.90, 0.95, 0.97]          # PRE_EMPHASIS_ALPHA (0=禁用)

# ─── 音频工具 ────────────────────────────────────────────────────────────────

def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """加载任意格式音频，重采样到 target_sr，转单声道 float32 [-1,1]"""
    suffix = path.suffix.lower()
    # M4A/AAC/MP3 等非 PCM 格式：用 imageio-ffmpeg 的静态二进制直接解码
    if suffix in ('.m4a', '.aac', '.mp3', '.ogg', '.opus', '.wma'):
        try:
            import subprocess
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, '-hide_banner', '-loglevel', 'error',
                '-i', str(path),
                '-ar', str(target_sr), '-ac', '1',
                '-f', 's16le', '-'
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors='replace'))
            samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
            return samples / 32768.0
        except Exception as e:
            raise RuntimeError(f"ffmpeg 解码失败 {path.name}: {e}")
    # WAV/FLAC/AIFF 等 PCM 格式：用 soundfile / librosa
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按目标 SNR 混合语音和噪声，返回 float32"""
    # 对齐长度：噪声循环填充或截断
    if len(noise) < len(speech):
        repeats = math.ceil(len(speech) / len(noise))
        noise = np.tile(noise, repeats)
    noise = noise[:len(speech)]

    # 计算 RMS
    speech_rms = np.sqrt(np.mean(speech ** 2)) + 1e-9
    noise_rms  = np.sqrt(np.mean(noise  ** 2)) + 1e-9

    # 按 SNR 缩放噪声
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = speech + noise_scaled
    # 防止削波
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed


def to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1,1] → PCM-16 LE bytes"""
    clipped = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return clipped.tobytes()


def save_wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """生成 WAV 文件 bytes（内存中，不写磁盘）"""
    import io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(to_pcm16(audio))
    return buf.getvalue()


# ─── 软件音频处理（与 SherpaRecognizer.kt 完全对齐）──────────────────────────

def pre_emphasis(samples: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """一阶预加重高通滤波 y[n] = x[n] - alpha * x[n-1]"""
    return np.append(samples[0], samples[1:] - alpha * samples[:-1])


def process_audio_frames(
    audio: np.ndarray,
    noise_gate_multiplier: float,
    pre_emphasis_alpha: float,
    noise_floor_alpha: float = 0.995,
    min_silence_frames: int = 3,
    frame_size: int = 512,          # ~32ms @ 16kHz，与 Kotlin 侧 bufferSize 对应
) -> np.ndarray:
    """
    模拟 SherpaRecognizer.processAudioFrame() 的逐帧处理：
      1. 计算帧 RMS
      2. 自适应更新噪声基底
      3. 噪声门：纯噪声帧用零帧替换（这里选择清零，与 Kotlin 侧"仍喂零帧"语义一致）
      4. 预加重滤波
    """
    result = np.zeros_like(audio)
    noise_floor_rms = 500.0 / 32768.0   # 初始值（与 Kotlin 一致，归一化到 float32）
    silence_count = 0
    prev_sample = 0.0

    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size].copy()

        # 1. RMS
        frame_rms = float(np.sqrt(np.mean(frame ** 2)) + 1e-12)

        # 2. 更新噪声基底（仅在低能量帧时更新）
        if frame_rms < noise_floor_rms * 2.0:
            noise_floor_rms = noise_floor_alpha * noise_floor_rms + (1 - noise_floor_alpha) * frame_rms

        # 3. 噪声门
        is_silence = frame_rms < noise_floor_rms * noise_gate_multiplier
        if is_silence:
            silence_count += 1
            if silence_count >= min_silence_frames:
                # 纯噪声 → 用静音帧（零帧）替换
                result[i:i + frame_size] = 0.0
                continue
        else:
            silence_count = 0

        # 4. 预加重
        filtered = np.empty_like(frame)
        filtered[0] = frame[0] - pre_emphasis_alpha * prev_sample
        filtered[1:] = frame[1:] - pre_emphasis_alpha * frame[:-1]
        prev_sample = frame[-1]
        result[i:i + frame_size] = filtered

    return result


# ─── SenseVoice 推理 ─────────────────────────────────────────────────────────

def create_sensevoice_recognizer() -> "sherpa_onnx.OfflineRecognizer":
    """创建 SenseVoice OfflineRecognizer (与 Android SherpaRecognizer.kt 对齐)"""
    import sherpa_onnx

    if not SENSE_VOICE_MODEL.exists():
        raise FileNotFoundError(
            f"未找到 SenseVoice 模型: {SENSE_VOICE_MODEL}\n"
            f"请确保 Android assets 中的模型文件存在。"
        )
    if not SENSE_VOICE_TOKENS.exists():
        raise FileNotFoundError(f"未找到 tokens 文件: {SENSE_VOICE_TOKENS}")

    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(SENSE_VOICE_MODEL),
        tokens=str(SENSE_VOICE_TOKENS),
        language="zh",
        use_itn=True,
        num_threads=4,
        debug=False,
        provider="cpu",
    )


def recognize_audio(
    audio: np.ndarray,
    recognizer,
    sr: int = SAMPLE_RATE,
) -> str:
    """
    用 SenseVoice OfflineRecognizer 识别处理后的音频。
    返回识别文本。
    """
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, audio.tolist())
    recognizer.decode_stream(stream)
    text = stream.result.text.strip()
    # SenseVoice 可能输出标点，去除以便计算 CER
    text = text.replace(",", "").replace("，", "").replace("。", "").replace("！", "").replace("？", "")
    return text


# ─── 评估指标 ────────────────────────────────────────────────────────────────

def cer(ref: str, hyp: str) -> float:
    """字符错误率 CER（去空格比较，适合中文）"""
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # 动态规划编辑距离
    dp = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref):
        new_dp = [i + 1]
        for j, hc in enumerate(hyp):
            new_dp.append(min(
                dp[j + 1] + 1,       # 删除
                new_dp[j] + 1,        # 插入
                dp[j] + (0 if rc == hc else 1),  # 替换
            ))
        dp = new_dp
    return dp[len(hyp)] / len(ref)


def exact_match(ref: str, hyp: str) -> bool:
    """完全匹配（去除空格后比较）"""
    return ref.replace(" ", "") == hyp.replace(" ", "")


# ─── 主评估流程 ──────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    gate_multiplier: float
    pre_emphasis_alpha: float
    snr_db: float
    avg_cer: float
    exact_acc: float
    n_samples: int


def run_evaluation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 初始化 SenseVoice ──
    print("初始化 SenseVoice OfflineRecognizer...")
    try:
        recognizer = create_sensevoice_recognizer()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    print(f"✓ SenseVoice 模型已加载: {SENSE_VOICE_MODEL.name}")

    # ── 加载标注 ──
    if not LABELS_FILE.exists():
        _create_sample_labels()
        print(f"⚠ 未找到 {LABELS_FILE}，已创建示例文件，请按格式填写后重新运行。")
        return

    labels: dict[str, str] = {}
    for line in LABELS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            labels[parts[0]] = parts[1]
        else:
            print(f"⚠ 跳过格式错误行: {line!r}")

    if not labels:
        print("❌ labels.txt 为空，请填写语音文件和对应文本。")
        return

    # ── 加载语音文件 ──
    speech_files = {}
    for fname, ref_text in labels.items():
        p = SPEECH_DIR / fname
        if not p.exists():
            print(f"⚠ 语音文件不存在，跳过: {p}")
            continue
        try:
            speech_files[fname] = (load_audio(p), ref_text)
        except Exception as e:
            print(f"⚠ 加载失败 {p}: {e}")

    if not speech_files:
        print("❌ 没有可用的语音文件。")
        return
    print(f"✓ 加载了 {len(speech_files)} 个语音样本")

    # ── 先对干净语音做基线识别 ──
    print("\n--- 基线识别（无噪声）---")
    for fname, (speech, ref_text) in speech_files.items():
        hyp = recognize_audio(speech, recognizer)
        match = "✅" if exact_match(ref_text, hyp) else "❌"
        print(f"  {fname}: ref={ref_text!r} → hyp={hyp!r} {match}")

    # ── 加载噪声文件 ──
    noise_files = list(NOISE_DIR.glob("*")) if NOISE_DIR.exists() else []
    noise_files = [f for f in noise_files if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}]

    if not noise_files:
        print(f"\n⚠ {NOISE_DIR} 中没有噪声文件，将生成白噪声作为替代。")
        noise_audio = np.random.randn(SAMPLE_RATE * 10).astype(np.float32) * 0.1
        noise_name = "synthetic_white_noise"
    else:
        print(f"\n✓ 加载了 {len(noise_files)} 个噪声文件")
        parts = []
        for nf in noise_files:
            try:
                parts.append(load_audio(nf))
                print(f"  - {nf.name}: {len(parts[-1])/SAMPLE_RATE:.1f}s")
            except Exception as e:
                print(f"  ⚠ 加载噪声失败 {nf.name}: {e}")
        if not parts:
            noise_audio = np.random.randn(SAMPLE_RATE * 10).astype(np.float32) * 0.1
            noise_name = "synthetic_white_noise"
        else:
            noise_audio = np.concatenate(parts)
            noise_name = "+".join(f.stem for f in noise_files[:3])

    print(f"✓ 噪声总时长: {len(noise_audio)/SAMPLE_RATE:.1f}s\n")

    # ── 网格搜索 ──
    all_results: list[EvalResult] = []
    total_combos = len(GATE_MULTIPLIERS) * len(PRE_EMPHASIS_ALPHAS) * len(SNR_LIST)
    done = 0

    print(f"开始网格搜索: {len(GATE_MULTIPLIERS)}×{len(PRE_EMPHASIS_ALPHAS)}×{len(SNR_LIST)} = {total_combos} 种组合 × {len(speech_files)} 样本")
    print("=" * 70)

    for gate_mult, pe_alpha in itertools.product(GATE_MULTIPLIERS, PRE_EMPHASIS_ALPHAS):
        for snr_db in SNR_LIST:
            cer_scores = []
            exact_scores = []

            for fname, (speech, ref_text) in speech_files.items():
                # 混合噪声
                mixed = mix_at_snr(speech, noise_audio, snr_db)

                # 软件处理（模拟降噪管线）
                processed = process_audio_frames(
                    mixed,
                    noise_gate_multiplier=gate_mult,
                    pre_emphasis_alpha=pe_alpha,
                )

                # SenseVoice 识别
                hyp = recognize_audio(processed, recognizer)

                cer_score = cer(ref_text, hyp)
                cer_scores.append(cer_score)
                exact_scores.append(1.0 if exact_match(ref_text, hyp) else 0.0)

            result = EvalResult(
                gate_multiplier=gate_mult,
                pre_emphasis_alpha=pe_alpha,
                snr_db=snr_db,
                avg_cer=float(np.mean(cer_scores)),
                exact_acc=float(np.mean(exact_scores)),
                n_samples=len(speech_files),
            )
            all_results.append(result)
            done += 1

        # 进度
        combos_done = done // len(SNR_LIST)
        total_param_combos = len(GATE_MULTIPLIERS) * len(PRE_EMPHASIS_ALPHAS)
        pct = combos_done / total_param_combos * 100
        # 打印当前组合在各 SNR 下的平均 CER
        recent = all_results[-len(SNR_LIST):]
        avg = np.mean([r.avg_cer for r in recent])
        print(f"  [{pct:5.1f}%] gate={gate_mult} pe={pe_alpha} → avg_CER={avg:.4f}")

    # ── 保存 CSV ──
    csv_path = RESULTS_DIR / "grid_search.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "noise_gate_multiplier", "pre_emphasis_alpha",
            "snr_db", "avg_cer", "exact_acc", "n_samples"
        ])
        for r in all_results:
            w.writerow([
                r.gate_multiplier, r.pre_emphasis_alpha,
                r.snr_db, f"{r.avg_cer:.4f}", f"{r.exact_acc:.4f}", r.n_samples
            ])
    print(f"\n✓ 详细结果已保存: {csv_path}")

    # ── 找最优参数（按困难场景 SNR≤5 加权平均 CER 排序）──
    hard_snr = {r for r in SNR_LIST if r <= 5}

    param_scores: dict[tuple, list] = {}
    for r in all_results:
        if r.snr_db in hard_snr:
            key = (r.gate_multiplier, r.pre_emphasis_alpha)
            param_scores.setdefault(key, []).append(r.avg_cer)

    best_key = min(param_scores, key=lambda k: np.mean(param_scores[k]))
    best_gate, best_pe = best_key

    best_json = {
        "asr_engine": "SenseVoice-Small INT8 (sherpa-onnx)",
        "noise_gate_multiplier": best_gate,
        "pre_emphasis_alpha": best_pe,
        "evaluated_on_snr_leq_5db": True,
        "noise_source": noise_name,
        "n_speech_samples": len(speech_files),
        "avg_cer_hard_snr": float(np.mean(param_scores[best_key])),
    }
    best_path = RESULTS_DIR / "best_params.json"
    best_path.write_text(json.dumps(best_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print("最优参数（SNR ≤ 5dB 场景加权）：")
    print(f"  NOISE_GATE_MULTIPLIER = {best_gate}")
    print(f"  PRE_EMPHASIS_ALPHA    = {best_pe}")
    print(f"  平均 CER（困难场景）   = {np.mean(param_scores[best_key]):.4f}")
    print(f"\n✓ 最优参数已保存: {best_path}")

    # ── 绘图 ──
    _plot_snr_curve(all_results, best_gate, best_pe)

    # ── 打印对比表 ──
    current_key = (3.0, 0.97)   # 与 SherpaRecognizer.kt 当前值对齐
    print(f"\n{'='*70}")
    print("当前参数 vs 最优参数对比（各 SNR 下平均 CER）：")
    print(f"{'SNR':>6} | {'当前(gate=3.0/pe=0.97)':>24} | {'最优':>10} | {'改善':>8}")
    print("-" * 56)
    for snr in SNR_LIST:
        cur_cer = next((r.avg_cer for r in all_results
                        if r.gate_multiplier == current_key[0]
                        and r.pre_emphasis_alpha == current_key[1]
                        and r.snr_db == snr), None)
        best_cer_val = next((r.avg_cer for r in all_results
                         if r.gate_multiplier == best_gate
                         and r.pre_emphasis_alpha == best_pe
                         and r.snr_db == snr), None)
        if cur_cer is not None and best_cer_val is not None:
            improvement = (cur_cer - best_cer_val) / max(cur_cer, 1e-9) * 100
            print(f"{snr:>4}dB | {cur_cer:>24.4f} | {best_cer_val:>10.4f} | {improvement:>+7.1f}%")
    print("=" * 56)


def _plot_snr_curve(results: list[EvalResult], best_gate: float, best_pe: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib 未安装，跳过绘图。")
        return

    current_key = (3.0, 0.97)
    best_key    = (best_gate, best_pe)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SenseVoice ASR 降噪参数评估 - SNR vs 准确率", fontsize=13)

    for ax, metric, ylabel, title in [
        (axes[0], "avg_cer",   "字符错误率 CER（越低越好）", "CER vs SNR"),
        (axes[1], "exact_acc", "完全匹配准确率（越高越好）", "完全匹配率 vs SNR"),
    ]:
        for key, label, color, ls in [
            (current_key, f"当前参数 (gate={current_key[0]}, pe={current_key[1]})", "steelblue", "-"),
            (best_key,    f"最优参数 (gate={best_key[0]}, pe={best_key[1]})",       "tomato",    "--"),
        ]:
            snrs = []
            vals = []
            for snr in SNR_LIST:
                r = next((r for r in results
                          if r.gate_multiplier == key[0]
                          and r.pre_emphasis_alpha == key[1]
                          and r.snr_db == snr), None)
                if r:
                    snrs.append(snr)
                    vals.append(getattr(r, metric))
            ax.plot(snrs, vals, marker="o", label=label, color=color, linestyle=ls)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=5, color="gray", linestyle=":", alpha=0.5, label="高速噪声典型 SNR")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "snr_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ SNR 曲线图已保存: {plot_path}")


def _create_sample_labels():
    """生成示例 labels.txt"""
    SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    sample = (
        "# 格式：文件名<TAB>正确文本\n"
        "# 示例（把对应 WAV 放到 noise_eval/speech/ 目录）：\n"
        "你好小天.wav\t你好小天\n"
        "播放稻香.wav\t播放稻香\n"
        "下一首.wav\t下一首\n"
        "暂停.wav\t暂停\n"
        "调高音量.wav\t调高音量\n"
    )
    LABELS_FILE.write_text(sample, encoding="utf-8")


# ─── 快速单次测试（不需要噪声文件，用白噪声验证流程）────────────────────────

def quick_test():
    """
    快速验证：用合成正弦波 + 白噪声，检查流程是否正常。
    不需要真实语音/噪声文件。
    """
    recognizer = create_sensevoice_recognizer()

    print("快速流程验证（合成音频）...")
    t = np.linspace(0, 2.0, SAMPLE_RATE * 2)
    # 440Hz + 880Hz 正弦（代替语音）
    speech = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    noise  = (np.random.randn(len(t)) * 0.05).astype(np.float32)

    for snr in [10, 0, -5]:
        mixed = mix_at_snr(speech, noise, snr)
        processed = process_audio_frames(mixed, noise_gate_multiplier=3.0, pre_emphasis_alpha=0.97)
        hyp = recognize_audio(processed, recognizer)
        print(f"  SNR={snr:+3d}dB → 识别结果: {hyp!r}")
    print("✓ 流程验证完成\n")


if __name__ == "__main__":
    import sys

    # 确保目录存在
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "--quick" in sys.argv:
        quick_test()
    else:
        run_evaluation()
