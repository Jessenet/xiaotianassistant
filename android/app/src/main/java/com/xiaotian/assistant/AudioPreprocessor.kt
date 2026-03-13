package com.xiaotian.assistant

import android.util.Log
import kotlin.math.*

/**
 * 车载环境音频预处理器
 *
 * 针对高速行驶场景的风噪和胎噪进行优化：
 *
 * 1. **高通滤波器** (Biquad HPF @ 250Hz)
 *    - 风噪主要能量集中在 <500Hz，胎噪在 100-1000Hz
 *    - 250Hz 高通可大幅衰减低频噪声，语音可懂度主要在 300Hz-3kHz
 *
 * 2. **预加重滤波** (Pre-emphasis, α=0.97)
 *    - 提升高频能量，补偿语音自然频谱倾斜
 *    - 使语音特征在噪声中更突出
 *
 * 3. **噪声底噪追踪** (Noise Floor Tracking)
 *    - 慢速自适应 RMS 估计器，追踪背景噪声水平
 *    - 用于动态调整 VAD 灵敏度和 SNR 判断
 *
 * 4. **SNR 估计**
 *    - 估算语音段的信噪比
 *    - 低 SNR 的语音段可能是噪声误触发，可拒绝
 *
 * @param sampleRate 采样率 (默认 16000)
 * @param highPassCutoff 高通截止频率 Hz (默认 250)
 * @param enablePreEmphasis 是否启用预加重 (默认 true)
 */
class AudioPreprocessor(
    private val sampleRate: Int = 16000,
    private val highPassCutoff: Float = 250f,
    private val enablePreEmphasis: Boolean = true
) {

    companion object {
        private const val TAG = "AudioPreprocessor"

        // 预加重系数 (标准值 0.97)
        private const val PRE_EMPHASIS_ALPHA = 0.97f

        // 噪声底噪追踪参数
        private const val NOISE_FLOOR_ATTACK = 0.001f   // 噪声上升速度（慢）
        private const val NOISE_FLOOR_RELEASE = 0.05f    // 噪声下降速度（较快）
        private const val NOISE_FLOOR_INIT = 0.01f       // 初始噪声估计

        // 帧能量计算
        private const val FRAME_SIZE_MS = 32             // 每帧 32ms
    }

    // ===== 高通 Biquad 滤波器状态 =====
    private var hpB0 = 0.0
    private var hpB1 = 0.0
    private var hpB2 = 0.0
    private var hpA1 = 0.0
    private var hpA2 = 0.0
    // 滤波器延迟线 (Direct Form II Transposed)
    private var hpZ1 = 0.0
    private var hpZ2 = 0.0

    // ===== 预加重状态 =====
    private var preEmphPrev = 0f

    // ===== 噪声追踪状态 =====
    /** 当前估计的噪声底噪 RMS */
    var noiseFloorRms = NOISE_FLOOR_INIT
        private set

    /** 最近一帧的 RMS */
    var currentFrameRms = 0f
        private set

    /** 是否处于高噪声环境 (自动检测) */
    val isHighNoiseEnvironment: Boolean
        get() = noiseFloorRms > 0.02f  // 约 -34dBFS

    /** 推荐的 VAD 阈值 (根据噪声水平动态调整) */
    val recommendedVadThreshold: Float
        get() {
            // 噪声越大，VAD 阈值越高，避免噪声触发 VAD
            // 正常环境: 0.5, 轻度噪声: 0.55, 中度: 0.65, 重度(车载): 0.7-0.8
            return when {
                noiseFloorRms < 0.005f -> 0.45f  // 安静环境，可以灵敏一些
                noiseFloorRms < 0.015f -> 0.50f  // 正常环境
                noiseFloorRms < 0.03f  -> 0.60f  // 轻度噪声
                noiseFloorRms < 0.06f  -> 0.70f  // 中度噪声（市区行驶）
                noiseFloorRms < 0.10f  -> 0.78f  // 重度噪声（高速行驶）
                else                   -> 0.85f  // 极重噪声
            }
        }

    /** 推荐的最短语音时长 (噪声大时要更长，过滤短促噪声) */
    val recommendedMinSpeechDuration: Float
        get() = when {
            noiseFloorRms < 0.015f -> 0.25f  // 正常: 250ms
            noiseFloorRms < 0.03f  -> 0.35f  // 轻度噪声: 350ms
            noiseFloorRms < 0.06f  -> 0.45f  // 中度噪声: 450ms
            else                   -> 0.55f  // 重度噪声: 550ms
        }

    /** 推荐的最短静音时长 (噪声大时要更长，避免噪声间隙误切) */
    val recommendedMinSilenceDuration: Float
        get() = when {
            noiseFloorRms < 0.015f -> 0.30f  // 正常: 300ms
            noiseFloorRms < 0.03f  -> 0.40f  // 轻度噪声: 400ms
            noiseFloorRms < 0.06f  -> 0.50f  // 中度噪声: 500ms
            else                   -> 0.60f  // 重度噪声: 600ms
        }

    init {
        computeHighPassCoefficients()
        Log.d(TAG, "音频预处理器初始化: HPF=${highPassCutoff}Hz, 预加重=$enablePreEmphasis")
    }

    /**
     * 计算 2 阶 Butterworth 高通 Biquad 滤波器系数
     *
     * Butterworth 特性：通带最大平坦，无纹波，-12dB/oct 滚降。
     * 对于 250Hz HPF @ 16kHz:
     *   - 250Hz 以下的风噪/胎噪衰减 >12dB
     *   - 300Hz 以上的语音基本无影响
     */
    private fun computeHighPassCoefficients() {
        val omega0 = 2.0 * PI * highPassCutoff / sampleRate
        val cosW = cos(omega0)
        val sinW = sin(omega0)
        val alpha = sinW / (2.0 * sqrt(2.0))  // Q = 1/√2 for Butterworth

        val b0 = (1.0 + cosW) / 2.0
        val b1 = -(1.0 + cosW)
        val b2 = (1.0 + cosW) / 2.0
        val a0 = 1.0 + alpha
        val a1 = -2.0 * cosW
        val a2 = 1.0 - alpha

        // 归一化
        hpB0 = b0 / a0
        hpB1 = b1 / a0
        hpB2 = b2 / a0
        hpA1 = a1 / a0
        hpA2 = a2 / a0

        Log.d(TAG, "HPF 系数: b=[${hpB0.f3()}, ${hpB1.f3()}, ${hpB2.f3()}] a=[1, ${hpA1.f3()}, ${hpA2.f3()}]")
    }

    /**
     * 处理音频样本（就地修改）
     *
     * 处理链: 高通滤波 → 预加重 → 噪声追踪
     *
     * @param samples Float 采样数组 [-1.0, 1.0]
     * @return 处理后的样本（同一数组，已就地修改）
     */
    fun process(samples: FloatArray): FloatArray {
        // 1. 高通滤波 - 去除低频风噪/胎噪
        applyHighPassFilter(samples)

        // 2. 预加重 - 提升高频语音成分
        if (enablePreEmphasis) {
            applyPreEmphasis(samples)
        }

        // 3. 更新噪声底噪估计
        updateNoiseFloor(samples)

        return samples
    }

    /**
     * 仅做噪声追踪，不修改音频
     * 用于在不想改变音频的情况下仍然追踪噪声水平
     */
    fun trackNoiseOnly(samples: FloatArray) {
        updateNoiseFloor(samples)
    }

    /**
     * 估算一段音频的 SNR (信噪比)
     *
     * @param speechSamples 语音段采样
     * @return 估计 SNR (dB)，越高越可能是真实语音
     */
    fun estimateSnr(speechSamples: FloatArray): Float {
        if (speechSamples.isEmpty()) return 0f

        val speechRms = computeRms(speechSamples)
        if (noiseFloorRms <= 0f || speechRms <= 0f) return 0f

        val snrDb = 20f * log10(speechRms / noiseFloorRms)
        return snrDb
    }

    /**
     * 判断一段语音是否可能是噪声误触发
     *
     * 综合考虑 SNR、持续时间、能量分布：
     * - SNR < 3dB: 很可能是噪声
     * - SNR 3-6dB: 可疑，需要结合其他因素
     * - SNR > 6dB: 大概率是真实语音
     *
     * @param speechSamples 语音段采样
     * @param minSnrDb 最低 SNR 阈值 (默认 3dB)
     * @return true 表示可能是噪声
     */
    fun isLikelyNoiseSegment(speechSamples: FloatArray, minSnrDb: Float = 3f): Boolean {
        val snr = estimateSnr(speechSamples)
        val durationMs = speechSamples.size * 1000 / sampleRate

        // 极短段+低SNR → 大概率噪声
        if (durationMs < 400 && snr < minSnrDb + 3f) {
            Log.d(TAG, "疑似噪声段: ${durationMs}ms, SNR=${snr.f1()}dB (短段+低SNR)")
            return true
        }

        // 任何段低于最低SNR → 噪声
        if (snr < minSnrDb) {
            Log.d(TAG, "疑似噪声段: ${durationMs}ms, SNR=${snr.f1()}dB < ${minSnrDb}dB")
            return true
        }

        // 计算能量变化率 (语音有明显的能量波动，纯噪声较平坦)
        val energyVariation = computeEnergyVariation(speechSamples)
        if (energyVariation < 0.15f && snr < minSnrDb + 6f) {
            Log.d(TAG, "疑似噪声段: 能量波动=${energyVariation.f3()}, SNR=${snr.f1()}dB (平坦能量)")
            return true
        }

        return false
    }

    /**
     * 计算能量变化率
     * 语音信号的能量在时间上有明显波动（元音/辅音交替）,
     * 而风噪/胎噪的能量比较平稳。
     *
     * @return 归一化能量变化率 [0, 1]，越大表示波动越明显
     */
    private fun computeEnergyVariation(samples: FloatArray): Float {
        val frameSize = sampleRate * FRAME_SIZE_MS / 1000
        if (samples.size < frameSize * 3) return 0.5f  // 太短不判断

        val frameEnergies = mutableListOf<Float>()
        var offset = 0
        while (offset + frameSize <= samples.size) {
            var sum = 0f
            for (i in offset until offset + frameSize) {
                sum += samples[i] * samples[i]
            }
            frameEnergies.add(sqrt(sum / frameSize))
            offset += frameSize
        }

        if (frameEnergies.size < 3) return 0.5f

        val mean = frameEnergies.average().toFloat()
        if (mean <= 0f) return 0f

        val variance = frameEnergies.map { (it - mean).pow(2) }.average().toFloat()
        val cv = sqrt(variance) / mean  // 变异系数

        return cv.coerceIn(0f, 1f)
    }

    // ===== 内部 DSP 实现 =====

    /**
     * 2 阶 Butterworth 高通滤波 (Direct Form II Transposed)
     */
    private fun applyHighPassFilter(samples: FloatArray) {
        for (i in samples.indices) {
            val x = samples[i].toDouble()
            val y = hpB0 * x + hpZ1
            hpZ1 = hpB1 * x - hpA1 * y + hpZ2
            hpZ2 = hpB2 * x - hpA2 * y
            samples[i] = y.toFloat()
        }
    }

    /**
     * 预加重: y[n] = x[n] - α * x[n-1]
     * 提升高频 (~6dB/oct)，补偿语音频谱自然倾斜
     */
    private fun applyPreEmphasis(samples: FloatArray) {
        for (i in samples.indices) {
            val current = samples[i]
            samples[i] = current - PRE_EMPHASIS_ALPHA * preEmphPrev
            preEmphPrev = current
        }
    }

    /**
     * 更新噪声底噪估计
     *
     * 使用非对称平滑：
     * - 能量上升时用慢速追踪 (认为可能是语音开始)
     * - 能量下降时用较快追踪 (认为噪声在降低)
     * - 持续低于底噪时快速下降
     */
    private fun updateNoiseFloor(samples: FloatArray) {
        val rms = computeRms(samples)
        currentFrameRms = rms

        // 非对称平滑追踪噪声底噪
        noiseFloorRms = if (rms < noiseFloorRms) {
            // 能量低于当前噪底 → 快速下降
            noiseFloorRms * (1 - NOISE_FLOOR_RELEASE) + rms * NOISE_FLOOR_RELEASE
        } else if (rms < noiseFloorRms * 3f) {
            // 能量略高于噪底（可能是噪声波动）→ 慢速上升
            noiseFloorRms * (1 - NOISE_FLOOR_ATTACK) + rms * NOISE_FLOOR_ATTACK
        } else {
            // 能量远高于噪底（大概率是语音）→ 不更新噪底
            noiseFloorRms
        }
    }

    private fun computeRms(samples: FloatArray): Float {
        if (samples.isEmpty()) return 0f
        var sum = 0f
        for (s in samples) sum += s * s
        return sqrt(sum / samples.size)
    }

    /**
     * 重置所有滤波器状态
     * 在开始新的录音会话时调用
     */
    fun reset() {
        hpZ1 = 0.0
        hpZ2 = 0.0
        preEmphPrev = 0f
        noiseFloorRms = NOISE_FLOOR_INIT
        currentFrameRms = 0f
        Log.d(TAG, "预处理器状态已重置")
    }

    /**
     * 获取当前状态的诊断信息
     */
    fun getDiagnostics(): String {
        val noiseDb = if (noiseFloorRms > 0) 20 * log10(noiseFloorRms) else -100f
        return "噪底=${noiseFloorRms.f4()} (${noiseDb.f1()}dB), " +
               "推荐VAD=${recommendedVadThreshold}, " +
               "高噪=${isHighNoiseEnvironment}, " +
               "推荐最短语音=${recommendedMinSpeechDuration}s"
    }

    // ===== 扩展 =====
    private fun Double.f3() = String.format("%.3f", this)
    private fun Float.f1() = String.format("%.1f", this)
    private fun Float.f3() = String.format("%.3f", this)
    private fun Float.f4() = String.format("%.4f", this)
}
