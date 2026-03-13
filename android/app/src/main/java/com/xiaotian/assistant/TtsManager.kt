package com.xiaotian.assistant

import android.content.Context
import android.media.AudioManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * 语音播报管理器（单例）
 *
 * 功能：
 * - 使用 Android TTS 引擎播报语音回复
 * - 播报期间自动暂停麦克风（防止回声被识别）
 * - 支持队列播报和打断
 * - 线程安全
 */
object TtsManager {

    private const val TAG = "TtsManager"
    private const val GOOGLE_TTS_PACKAGE = "com.google.android.tts"

    private var tts: TextToSpeech? = null
    private val isReady = AtomicBoolean(false)
    private val utteranceId = AtomicInteger(0)

    /**
     * 麦克风控制回调
     * 播报时需要暂停麦克风，播报结束后恢复
     */
    interface MicController {
        fun pauseMic()
        fun resumeMic()
    }

    private var micController: MicController? = null

    /**
     * 初始化 TTS 引擎
     * 在 Application 或 Service.onCreate 中调用
     */
    fun init(context: Context) {
        if (tts != null) return

        val appContext = context.applicationContext

        // 尝试使用 Google TTS 引擎，如果不可用则使用系统默认
        val initListener = TextToSpeech.OnInitListener { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.CHINESE)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "中文语音不支持，尝试使用默认语言")
                    tts?.setLanguage(Locale.getDefault())
                }

                // 设置语速和音调
                tts?.setSpeechRate(1.1f)   // 稍快一点，更自然
                tts?.setPitch(1.0f)

                // 设置播报完成监听
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        Log.d(TAG, "开始播报: $utteranceId")
                    }

                    override fun onDone(utteranceId: String?) {
                        Log.d(TAG, "播报完成: $utteranceId")
                        micController?.resumeMic()
                    }

                    @Deprecated("Deprecated in API level 21")
                    override fun onError(utteranceId: String?) {
                        Log.e(TAG, "播报错误: $utteranceId")
                        micController?.resumeMic()
                    }

                    override fun onError(utteranceId: String?, errorCode: Int) {
                        Log.e(TAG, "播报错误: $utteranceId, code=$errorCode")
                        micController?.resumeMic()
                    }
                })

                isReady.set(true)
                Log.d(TAG, "TTS 引擎初始化成功")
            } else {
                Log.e(TAG, "TTS 引擎初始化失败: status=$status")
            }
        }

        // 优先使用 Google TTS（明确指定引擎，避免系统默认为空导致初始化失败）
        tts = try {
            TextToSpeech(appContext, initListener, GOOGLE_TTS_PACKAGE).also {
                Log.d(TAG, "使用 Google TTS 引擎: $GOOGLE_TTS_PACKAGE")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Google TTS 不可用，使用系统默认引擎", e)
            TextToSpeech(appContext, initListener)
        }
    }

    /**
     * 设置麦克风控制器
     */
    fun setMicController(controller: MicController?) {
        micController = controller
    }

    /**
     * 播报文本（打断当前播报）
     *
     * @param text 要播报的文本
     * @param pauseMic 是否在播报期间暂停麦克风（默认 true）
     */
    fun speak(text: String, pauseMic: Boolean = true) {
        if (!isReady.get()) {
            Log.w(TAG, "TTS 未就绪，跳过播报: $text")
            return
        }

        if (text.isBlank()) return

        val id = "tts_${utteranceId.incrementAndGet()}"

        Log.d(TAG, "播报: $text (id=$id, pauseMic=$pauseMic)")

        // 使用 STREAM_MUSIC（媒体音量）播报，音量更大
        val params = Bundle().apply {
            putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_MUSIC)
        }

        // ★ 先提交 TTS 请求（异步排队），再暂停麦克风
        // 这样 TTS 引擎可以立即开始准备音频，减少感知延迟
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, id)

        if (pauseMic) {
            micController?.pauseMic()
        }
    }

    /**
     * 追加播报（不打断当前）
     */
    fun speakAppend(text: String, pauseMic: Boolean = true) {
        if (!isReady.get() || text.isBlank()) return

        val id = "tts_${utteranceId.incrementAndGet()}"

        if (pauseMic) {
            micController?.pauseMic()
        }

        val params = Bundle().apply {
            putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_MUSIC)
        }
        tts?.speak(text, TextToSpeech.QUEUE_ADD, params, id)
    }

    /**
     * 停止当前播报
     */
    fun stop() {
        tts?.stop()
        micController?.resumeMic()
    }

    /**
     * 是否正在播报
     */
    val isSpeaking: Boolean
        get() = tts?.isSpeaking == true

    /**
     * 释放资源
     */
    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isReady.set(false)
        micController = null
        Log.d(TAG, "TTS 已释放")
    }
}
