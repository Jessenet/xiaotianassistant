package com.xiaotian.assistant

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.util.Log
import com.k2fsa.sherpa.onnx.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import kotlin.math.abs

/**
 * Sherpa-ONNX + SenseVoice 离线语音识别器
 *
 * 替代原 VoskRecognizer，使用 SenseVoice (阿里 FunAudioLLM) 声学模型：
 * - 大规模多条件训练数据，噪声鲁棒性远超 Vosk Kaldi TDNN-F（2019）
 * - 支持中/英/日/韩/粤语，INT8 量化模型 ~230MB
 * - 非流式架构，搭配 Silero VAD 检测语音段落后整段识别
 *
 * 硬件音频增强保留：
 * - NoiseSuppressor（硬件降噪）
 * - AutomaticGainControl（自动增益控制）
 * - AcousticEchoCanceler（回声消除）
 * - VOICE_COMMUNICATION 音频源（激活平台 DSP）
 */
class SherpaRecognizer(private val context: Context) {

    // Sherpa-ONNX 识别器 + VAD
    private var offlineRecognizer: OfflineRecognizer? = null
    private var vad: Vad? = null

    // 音频录制
    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null

    // 硬件音频增强效果
    private var noiseSuppressor: NoiseSuppressor? = null
    private var agc: AutomaticGainControl? = null
    private var aec: AcousticEchoCanceler? = null

    // 车载噪声预处理器 (高通滤波 + 预加重 + 自适应噪声追踪)
    private val audioPreprocessor = AudioPreprocessor()

    @Volatile
    private var isRecording = false
    private var isInitialized = false

    // 自适应 VAD 追踪
    private var lastVadAdaptTime = 0L
    private var currentVadThreshold = VAD_THRESHOLD
    private var currentMinSpeech = VAD_MIN_SPEECH_DURATION
    private var currentMinSilence = VAD_MIN_SILENCE_DURATION

    private var onResultListener: ((String) -> Unit)? = null
    private var onErrorListener: ((String) -> Unit)? = null

    companion object {
        private const val TAG = "SherpaRecognizer"
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        // SenseVoice 模型文件路径（assets 相对路径）
        private const val SENSE_VOICE_MODEL_DIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
        private const val SENSE_VOICE_MODEL = "$SENSE_VOICE_MODEL_DIR/model.int8.onnx"
        private const val SENSE_VOICE_TOKENS = "$SENSE_VOICE_MODEL_DIR/tokens.txt"

        // Silero VAD 模型
        private const val VAD_MODEL = "silero_vad.onnx"

        // VAD 参数 (初始值，会根据噪声环境自适应调整)
        private const val VAD_THRESHOLD = 0.5f
        private const val VAD_MIN_SILENCE_DURATION = 0.3f  // 静音 0.3s 后切断
        private const val VAD_MIN_SPEECH_DURATION = 0.25f  // 最短语音段 0.25s
        private const val VAD_WINDOW_SIZE = 512             // Silero VAD 窗口大小
        private const val VAD_MAX_SPEECH_DURATION = 15.0f   // 最长语音段 15s

        // 自适应 VAD 参数更新间隔
        private const val VAD_ADAPT_INTERVAL_MS = 5000L     // 每 5 秒评估一次是否需要调整
        private const val SNR_REJECT_THRESHOLD_DB = 3f      // 低于此 SNR 的语音段丢弃
    }

    /**
     * 初始化 SenseVoice 模型和 Silero VAD
     */
    suspend fun initialize(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "开始初始化 Sherpa-ONNX SenseVoice + Silero VAD...")
            val startTime = System.currentTimeMillis()

            // 1. 初始化 Silero VAD
            val vadConfig = VadModelConfig(
                sileroVadModelConfig = SileroVadModelConfig(
                    model = VAD_MODEL,
                    threshold = VAD_THRESHOLD,
                    minSilenceDuration = VAD_MIN_SILENCE_DURATION,
                    minSpeechDuration = VAD_MIN_SPEECH_DURATION,
                    windowSize = VAD_WINDOW_SIZE,
                    maxSpeechDuration = VAD_MAX_SPEECH_DURATION,
                ),
                sampleRate = SAMPLE_RATE,
                numThreads = 2,
                provider = "cpu",
            )
            vad = Vad(assetManager = context.assets, config = vadConfig)
            Log.d(TAG, "Silero VAD 初始化成功")

            // 2. 初始化 SenseVoice OfflineRecognizer
            val offlineConfig = OfflineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
                modelConfig = OfflineModelConfig(
                    senseVoice = OfflineSenseVoiceModelConfig(
                        model = SENSE_VOICE_MODEL,
                        language = "zh",
                        useInverseTextNormalization = true,
                    ),
                    tokens = SENSE_VOICE_TOKENS,
                    numThreads = 2,
                    debug = false,
                    provider = "cpu",
                ),
                decodingMethod = "greedy_search",
            )
            offlineRecognizer = OfflineRecognizer(
                assetManager = context.assets,
                config = offlineConfig,
            )

            isInitialized = true
            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "SenseVoice + VAD 初始化完成 (${elapsed}ms)")

            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "初始化失败", e)
            Result.failure(e)
        }
    }

    /**
     * 开始语音识别
     */
    fun startListening() {
        if (!isInitialized) {
            onErrorListener?.invoke("SenseVoice 模型未初始化")
            return
        }

        // 如果已在录音，先停止
        if (isRecording) {
            stopRecording()
        }

        try {
            // 1. 创建 AudioRecord
            val minBufSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            if (minBufSize == AudioRecord.ERROR_BAD_VALUE || minBufSize == AudioRecord.ERROR) {
                onErrorListener?.invoke("不支持的音频配置")
                return
            }
            val bufferSize = maxOf(minBufSize * 2, 8192)

            // VOICE_COMMUNICATION 激活平台 DSP 降噪链路
            val audioSource = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                MediaRecorder.AudioSource.VOICE_COMMUNICATION
            } else {
                MediaRecorder.AudioSource.VOICE_RECOGNITION
            }

            audioRecord = AudioRecord(
                audioSource,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                onErrorListener?.invoke("AudioRecord 初始化失败")
                releaseRecordingResources()
                return
            }

            // 2. 附加硬件音频增强
            attachAudioEffects(audioRecord!!.audioSessionId)

            // 3. 开始录音
            audioRecord!!.startRecording()
            isRecording = true

            // 4. 重置 VAD 状态
            vad?.reset()

            // 5. 重置音频预处理器
            audioPreprocessor.reset()

            // 6. 启动识别线程
            recordingThread = Thread({
                recognitionLoop()
            }, "SherpaRecognition").apply {
                priority = Thread.MAX_PRIORITY
                start()
            }

            Log.d(TAG, "SenseVoice 语音识别已启动 [降噪=${noiseSuppressor != null}, AGC=${agc != null}, AEC=${aec != null}, 车载预处理=✓]")

        } catch (e: Exception) {
            Log.e(TAG, "启动识别失败", e)
            releaseRecordingResources()
            onErrorListener?.invoke("启动识别失败: ${e.message}")
        }
    }

    /**
     * 附加硬件音频增强效果
     */
    private fun attachAudioEffects(sessionId: Int) {
        try {
            if (NoiseSuppressor.isAvailable()) {
                noiseSuppressor = NoiseSuppressor.create(sessionId)?.also {
                    it.enabled = true
                    Log.d(TAG, "✓ 硬件降噪 (NoiseSuppressor) 已启用")
                }
            }
        } catch (e: Exception) { Log.w(TAG, "NoiseSuppressor 创建失败", e) }

        try {
            if (AutomaticGainControl.isAvailable()) {
                agc = AutomaticGainControl.create(sessionId)?.also {
                    it.enabled = true
                    Log.d(TAG, "✓ 自动增益控制 (AGC) 已启用")
                }
            }
        } catch (e: Exception) { Log.w(TAG, "AGC 创建失败", e) }

        try {
            if (AcousticEchoCanceler.isAvailable()) {
                aec = AcousticEchoCanceler.create(sessionId)?.also {
                    it.enabled = true
                    Log.d(TAG, "✓ 回声消除 (AEC) 已启用")
                }
            }
        } catch (e: Exception) { Log.w(TAG, "AEC 创建失败", e) }
    }

    /**
     * 识别主循环
     *
     * 流程：AudioRecord → PCM-16 → FloatArray → 高通滤波 + 预加重 → 噪声追踪
     *       → 自适应 VAD → 检测到语音段 → SNR 校验 → SenseVoice 离线识别
     *
     * 车载优化：
     * - 250Hz 高通滤波去除风噪/胎噪低频成分
     * - 预加重提升语音高频特征
     * - 噪声底噪追踪 + 自适应 VAD 阈值
     * - SNR 校验拒绝低信噪比段
     */
    private fun recognitionLoop() {
        // VAD 需要 512 采样的 FloatArray 窗口
        val windowSize = VAD_WINDOW_SIZE
        // AudioRecord 每次读取的采样数（可以大于 windowSize）
        val readSamples = 1600  // 100ms @ 16kHz
        val buffer = ShortArray(readSamples)
        // 累积采样缓冲区，用于拆分成 VAD 窗口
        val sampleBuffer = mutableListOf<Float>()

        // 重置预处理器状态
        audioPreprocessor.reset()
        lastVadAdaptTime = System.currentTimeMillis()

        try {
            while (isRecording) {
                val samplesRead = audioRecord?.read(buffer, 0, buffer.size) ?: break
                if (samplesRead <= 0) {
                    if (isRecording && samplesRead < 0) {
                        Log.e(TAG, "AudioRecord 读取错误: $samplesRead")
                    }
                    continue
                }

                // Short → Float [-1.0, 1.0]
                val floatSamples = FloatArray(samplesRead)
                for (i in 0 until samplesRead) {
                    floatSamples[i] = buffer[i].toFloat() / 32768.0f
                }

                // ★ 车载音频预处理：高通滤波 + 预加重 + 噪声追踪
                audioPreprocessor.process(floatSamples)

                // ★ 自适应 VAD：根据噪声水平动态调整参数
                maybeAdaptVad()

                // 累积到 VAD 缓冲区
                for (i in 0 until samplesRead) {
                    sampleBuffer.add(floatSamples[i])
                }

                // 拆成 VAD windowSize 的块逐个喂给 VAD
                while (sampleBuffer.size >= windowSize) {
                    val window = FloatArray(windowSize)
                    for (i in 0 until windowSize) {
                        window[i] = sampleBuffer.removeAt(0)
                    }

                    val currentVad = vad ?: break
                    currentVad.acceptWaveform(window)

                    // 检查 VAD 是否有已完成的语音段
                    while (!currentVad.empty()) {
                        val segment = currentVad.front()
                        currentVad.pop()

                        // 语音段检测完毕
                        val durationMs = (segment.samples.size * 1000) / SAMPLE_RATE
                        Log.d(TAG, "VAD 检测到语音段: ${durationMs}ms (${segment.samples.size} 采样)")

                        // ★ SNR 校验：拒绝低信噪比的段（可能是噪声触发）
                        if (audioPreprocessor.isLikelyNoiseSegment(
                                segment.samples, SNR_REJECT_THRESHOLD_DB)) {
                            Log.d(TAG, "SNR 校验未通过，丢弃语音段 (噪底=${audioPreprocessor.noiseFloorRms})")
                            continue
                        }

                        val text = recognizeSegment(segment.samples)
                        if (!text.isNullOrBlank()) {
                            Log.d(TAG, "识别结果: $text [噪声环境=${audioPreprocessor.isHighNoiseEnvironment}]")
                            onResultListener?.invoke(text)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            if (isRecording) {
                Log.e(TAG, "识别线程异常", e)
                onErrorListener?.invoke("识别错误: ${e.message}")
            }
        }

        // 处理 VAD 残留（flush）
        try {
            vad?.flush()
            val currentVad = vad
            if (currentVad != null) {
                while (!currentVad.empty()) {
                    val segment = currentVad.front()
                    currentVad.pop()
                    val text = recognizeSegment(segment.samples)
                    if (!text.isNullOrBlank()) {
                        Log.d(TAG, "尾部识别结果: $text")
                        onResultListener?.invoke(text)
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "处理残留语音段失败", e)
        }
    }

    /**
     * 自适应 VAD 参数调整
     *
     * 每隔 VAD_ADAPT_INTERVAL_MS 评估噪声水平，
     * 如果噪声环境变化显著，重建 VAD 以应用新参数。
     * 这使得系统能在安静室内和嘈杂车内之间自动切换。
     */
    private fun maybeAdaptVad() {
        val now = System.currentTimeMillis()
        if (now - lastVadAdaptTime < VAD_ADAPT_INTERVAL_MS) return
        lastVadAdaptTime = now

        val newThreshold = audioPreprocessor.recommendedVadThreshold
        val newMinSpeech = audioPreprocessor.recommendedMinSpeechDuration
        val newMinSilence = audioPreprocessor.recommendedMinSilenceDuration

        // 仅在参数变化显著时重建 VAD
        val thresholdChanged = abs(newThreshold - currentVadThreshold) > 0.05f
        val speechChanged = abs(newMinSpeech - currentMinSpeech) > 0.05f
        val silenceChanged = abs(newMinSilence - currentMinSilence) > 0.05f

        if (thresholdChanged || speechChanged || silenceChanged) {
            Log.i(TAG, "🔧 环境噪声变化，自适应调整 VAD: " +
                    "阈值 ${currentVadThreshold}→${newThreshold}, " +
                    "最短语音 ${currentMinSpeech}→${newMinSpeech}s, " +
                    "最短静音 ${currentMinSilence}→${newMinSilence}s " +
                    "[${audioPreprocessor.getDiagnostics()}]")

            currentVadThreshold = newThreshold
            currentMinSpeech = newMinSpeech
            currentMinSilence = newMinSilence

            recreateVad(newThreshold, newMinSilence, newMinSpeech)
        }
    }

    /**
     * 按新参数重建 Silero VAD
     */
    private fun recreateVad(threshold: Float, minSilence: Float, minSpeech: Float) {
        try {
            vad?.reset()  // 先重置再释放
            vad?.release()
            val vadConfig = VadModelConfig(
                sileroVadModelConfig = SileroVadModelConfig(
                    model = VAD_MODEL,
                    threshold = threshold,
                    minSilenceDuration = minSilence,
                    minSpeechDuration = minSpeech,
                    windowSize = VAD_WINDOW_SIZE,
                    maxSpeechDuration = VAD_MAX_SPEECH_DURATION,
                ),
                sampleRate = SAMPLE_RATE,
                numThreads = 2,
                provider = "cpu",
            )
            vad = Vad(assetManager = context.assets, config = vadConfig)
            Log.d(TAG, "VAD 已用新参数重建: threshold=$threshold, minSilence=$minSilence, minSpeech=$minSpeech")
        } catch (e: Exception) {
            Log.e(TAG, "重建 VAD 失败", e)
        }
    }

    /**
     * 用 SenseVoice 离线识别一段语音
     *
     * @param samples 16kHz Float 采样数组
     * @return 识别文本，失败或空则返回 null
     */
    private fun recognizeSegment(samples: FloatArray): String? {
        val rec = offlineRecognizer ?: return null

        return try {
            val startTime = System.currentTimeMillis()

            val stream = rec.createStream()
            stream.acceptWaveform(samples, SAMPLE_RATE)
            rec.decode(stream)
            val result = rec.getResult(stream)

            val elapsed = System.currentTimeMillis() - startTime
            val durationMs = (samples.size * 1000) / SAMPLE_RATE
            val rtf = if (durationMs > 0) elapsed.toFloat() / durationMs else 0f

            Log.d(TAG, "SenseVoice 识别完成: ${elapsed}ms (RTF=${String.format("%.2f", rtf)}, 音频=${durationMs}ms)")

            // SenseVoice 结果清洗：去除语言/情感/事件标签
            cleanSenseVoiceText(result.text)
        } catch (e: Exception) {
            Log.e(TAG, "SenseVoice 识别失败", e)
            null
        }
    }

    /**
     * 清洗 SenseVoice 输出文本
     *
     * SenseVoice 可能在文本前后添加标签如 <|zh|>, <|NEUTRAL|>, <|Speech|> 等，需要去除。
     * 例如: "<|zh|><|NEUTRAL|><|Speech|>你好小天" → "你好小天"
     *
     * 非语音事件（Noise, BGM, Applause, Laughter, Music 等）直接丢弃，
     * 避免环境噪声被错误识别为指令。
     */
    private fun cleanSenseVoiceText(text: String): String? {
        if (text.isBlank()) return null

        // SenseVoice 事件标签检测: <|zh|><|NEUTRAL|><|Event|>text
        // 非语音事件直接丢弃
        val nonSpeechEvents = listOf("Noise", "BGM", "Applause", "Laughter", "Music", "NON_SPEECH")
        for (event in nonSpeechEvents) {
            if (text.contains("<|$event|>", ignoreCase = true)) {
                Log.d(TAG, "检测到非语音事件 <|$event|>，丢弃: $text")
                return null
            }
        }

        // 去除所有 <|...|> 标签
        var cleaned = text.replace(Regex("<\\|[^|]*\\|>"), "").trim()

        // 去除可能的前导/尾部标点
        cleaned = cleaned.trim('，', '。', '、', '！', '？', ' ')

        return if (cleaned.isNotEmpty()) cleaned else null
    }

    fun stopListening() { stopRecording() }
    fun cancel() { stopRecording() }

    private fun stopRecording() {
        if (!isRecording && recordingThread == null) return

        isRecording = false

        try { audioRecord?.stop() } catch (_: Exception) {}

        // 防止自引用死锁：如果是从识别线程自身的回调中调用 stopRecording
        // （例如唤醒词检测后 TTS.speak → pauseMic → stopListening），
        // 不要 join 自己，否则会阻塞整整 3 秒。
        val callingThread = Thread.currentThread()
        if (recordingThread != null && recordingThread != callingThread) {
            try { recordingThread?.join(1000) } catch (_: InterruptedException) {}
        } else if (recordingThread == callingThread) {
            Log.d(TAG, "跳过 self-join（从识别线程内部调用 stop）")
        }
        recordingThread = null

        releaseRecordingResources()
        Log.d(TAG, "SenseVoice 语音识别已停止")
    }

    private fun releaseRecordingResources() {
        try { noiseSuppressor?.release() } catch (_: Exception) {}
        noiseSuppressor = null
        try { agc?.release() } catch (_: Exception) {}
        agc = null
        try { aec?.release() } catch (_: Exception) {}
        aec = null

        try { audioRecord?.release() } catch (_: Exception) {}
        audioRecord = null
    }

    fun destroy() {
        try {
            stopRecording()
            offlineRecognizer?.release()
            offlineRecognizer = null
            vad?.release()
            vad = null
            isInitialized = false
            Log.d(TAG, "Sherpa-ONNX 资源已释放")
        } catch (e: Exception) {
            Log.e(TAG, "释放资源失败", e)
        }
    }

    fun setOnResultListener(listener: (String) -> Unit) {
        onResultListener = listener
    }

    fun setOnErrorListener(listener: (String) -> Unit) {
        onErrorListener = listener
    }
}
