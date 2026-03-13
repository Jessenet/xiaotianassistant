package com.xiaotian.assistant

import android.content.Context
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * 混合语音识别器
 * 优先使用 SenseVoice (Sherpa-ONNX) 离线识别，如果不可用则自动切换到 Google 在线识别
 */
class HybridVoiceRecognizer(private val context: Context) {
    
    private var googleRecognizer: VoiceRecognizer? = null
    private var sherpaRecognizer: SherpaRecognizer? = null
    
    private var isSherpaReady = false
    private var currentEngine: RecognizerEngine = RecognizerEngine.NONE
    
    private var onResultListener: ((String) -> Unit)? = null
    private var onErrorListener: ((String) -> Unit)? = null
    private var onStatusListener: ((String) -> Unit)? = null
    
    enum class RecognizerEngine {
        NONE,
        GOOGLE,
        SHERPA
    }
    
    companion object {
        private const val TAG = "HybridVoiceRecognizer"
    }
    
    init {
        // 优先使用 SenseVoice 离线识别
        Log.d(TAG, "优先初始化 SenseVoice (Sherpa-ONNX) 离线识别...")
        currentEngine = RecognizerEngine.NONE  // 先设置为NONE，等待初始化完成
        
        // 检查 Google 是否可用作为备用
        val isGoogleAvailable = SpeechRecognizer.isRecognitionAvailable(context)
        Log.d(TAG, "Google 语音识别可用性: $isGoogleAvailable")
        
        // 后台初始化 SenseVoice（优先）
        CoroutineScope(Dispatchers.IO).launch {
            try {
                initSherpaRecognizer()
                // SenseVoice 初始化成功后，如果Google可用，在后台准备作为备用
                if (isGoogleAvailable && isSherpaReady) {
                    Log.d(TAG, "SenseVoice已就绪，准备Google作为备用")
                    initGoogleRecognizer()
                }
            } catch (e: Exception) {
                Log.e(TAG, "SenseVoice初始化失败，尝试使用Google", e)
                // SenseVoice 失败，尝试使用 Google
                if (isGoogleAvailable) {
                    Log.d(TAG, "切换到Google语音识别")
                    currentEngine = RecognizerEngine.GOOGLE
                    initGoogleRecognizer()
                } else {
                    Log.e(TAG, "所有语音识别引擎都不可用")
                    onErrorListener?.invoke("语音识别不可用: 离线和在线识别均失败")
                }
            }
        }
    }
    
    private fun initGoogleRecognizer() {
        googleRecognizer = VoiceRecognizer(context).apply {
            setOnResultListener { text ->
                Log.d(TAG, "Google 识别结果: $text")
                onResultListener?.invoke(text)
            }
            
            setOnErrorListener { error ->
                Log.e(TAG, "Google 识别错误（备用引擎）: $error")
                // Google作为备用，失败时只报告错误
                onErrorListener?.invoke(error)
            }
        }
    }
    
    private suspend fun initSherpaRecognizer() {
        try {
            Log.d(TAG, "开始初始化 SenseVoice (Sherpa-ONNX)...")
            sherpaRecognizer = SherpaRecognizer(context).apply {
                setOnResultListener { text ->
                    Log.d(TAG, "SenseVoice 识别结果: $text")
                    onResultListener?.invoke(text)
                }
                
                setOnErrorListener { error ->
                    Log.e(TAG, "SenseVoice 识别错误（主引擎）: $error")
                    
                    // 如果SenseVoice失败且Google可用，尝试切换到Google
                    if (googleRecognizer != null && currentEngine == RecognizerEngine.SHERPA) {
                        Log.d(TAG, "SenseVoice识别失败，尝试切换到Google备用")
                        currentEngine = RecognizerEngine.GOOGLE
                        onErrorListener?.invoke("切换到在线识别")
                    } else {
                        onErrorListener?.invoke(error)
                    }
                }
            }
            
            val result = sherpaRecognizer?.initialize()
            if (result?.isSuccess == true) {
                isSherpaReady = true
                Log.d(TAG, "SenseVoice 初始化成功，离线识别已就绪")
                
                // 如果当前还没有可用引擎，切换到 Sherpa
                if (currentEngine == RecognizerEngine.NONE) {
                    currentEngine = RecognizerEngine.SHERPA
                    Log.d(TAG, "★★★ 引擎已设置为: SHERPA (SenseVoice) ★★★")
                    onStatusListener?.invoke("离线识别已就绪")
                }
            } else {
                val errorMsg = result?.exceptionOrNull()?.message ?: "未知错误"
                Log.e(TAG, "SenseVoice 初始化失败: $errorMsg")
                if (currentEngine == RecognizerEngine.NONE) {
                    // 如果当前没有任何可用引擎，通知用户
                    onErrorListener?.invoke("语音识别不可用，请检查模型文件")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "SenseVoice 初始化异常", e)
            sherpaRecognizer = null  // 清空失败的实例
            if (currentEngine == RecognizerEngine.NONE) {
                onErrorListener?.invoke("语音识别初始化失败: ${e.message}")
            }
        }
    }
    
    private fun switchToSherpa() {
        if (!isSherpaReady) {
            onErrorListener?.invoke("离线识别尚未就绪")
            return
        }
        
        currentEngine = RecognizerEngine.SHERPA
        onErrorListener?.invoke("已切换到离线识别模式")
        Log.d(TAG, "已切换到 SenseVoice 引擎")
    }
    
    /**
     * 开始语音识别
     */
    fun startListening() {
        Log.d(TAG, "★★★ startListening 被调用, 当前引擎: $currentEngine, SenseVoice就绪: $isSherpaReady ★★★")
        when (currentEngine) {
            RecognizerEngine.GOOGLE -> {
                Log.d(TAG, "使用 Google 开始识别")
                googleRecognizer?.startListening()
            }
            RecognizerEngine.SHERPA -> {
                if (isSherpaReady) {
                    Log.d(TAG, "★ 使用 SenseVoice 开始识别 ★")
                    sherpaRecognizer?.startListening()
                } else {
                    Log.e(TAG, "SenseVoice尚未就绪")
                    onErrorListener?.invoke("离线识别尚未就绪，请稍候")
                }
            }
            RecognizerEngine.NONE -> {
                Log.e(TAG, "引擎为NONE，语音识别不可用")
                onErrorListener?.invoke("语音识别不可用")
            }
        }
    }
    
    /**
     * 停止语音识别
     */
    fun stopListening() {
        when (currentEngine) {
            RecognizerEngine.GOOGLE -> googleRecognizer?.stopListening()
            RecognizerEngine.SHERPA -> sherpaRecognizer?.stopListening()
            else -> {}
        }
    }
    
    /**
     * 取消识别
     */
    fun cancel() {
        when (currentEngine) {
            RecognizerEngine.GOOGLE -> googleRecognizer?.cancel()
            RecognizerEngine.SHERPA -> sherpaRecognizer?.cancel()
            else -> {}
        }
    }
    
    /**
     * 释放资源
     */
    fun destroy() {
        try {
            googleRecognizer?.destroy()
            sherpaRecognizer?.destroy()
            googleRecognizer = null
            sherpaRecognizer = null
            currentEngine = RecognizerEngine.NONE
            Log.d(TAG, "资源已释放")
        } catch (e: Exception) {
            Log.e(TAG, "释放资源失败", e)
        }
    }
    
    /**
     * 获取当前使用的引擎
     */
    fun getCurrentEngine(): RecognizerEngine = currentEngine
    
    /**
     * 检查 SenseVoice 离线识别是否就绪
     */
    fun isSherpaReady(): Boolean = isSherpaReady
    
    // 设置回调
    
    fun setOnResultListener(listener: (String) -> Unit) {
        onResultListener = listener
    }
    
    fun setOnErrorListener(listener: (String) -> Unit) {
        onErrorListener = listener
    }

    fun setOnStatusListener(listener: (String) -> Unit) {
        onStatusListener = listener
    }
}
