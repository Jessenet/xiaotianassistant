package com.xiaotian.assistant

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

/**
 * 语音识别器
 * 使用Android原生SpeechRecognizer API
 */
class VoiceRecognizer(private val context: Context) {
    
    private var speechRecognizer: SpeechRecognizer? = null
    private var onResultListener: ((String) -> Unit)? = null
    private var onErrorListener: ((String) -> Unit)? = null
    
    companion object {
        private const val TAG = "VoiceRecognizer"
    }
    
    init {
        initSpeechRecognizer()
    }
    
    private fun initSpeechRecognizer() {
        // 获取系统配置的语音识别服务
        val recognitionService = android.provider.Settings.Secure.getString(
            context.contentResolver, "voice_recognition_service")
        Log.d(TAG, "当前语音识别服务: $recognitionService")
        
        val isAvailable = SpeechRecognizer.isRecognitionAvailable(context)
        Log.d(TAG, "检查语音识别可用性: $isAvailable")
        
        // 即使isAvailable返回false，如果系统配置了服务，也尝试创建
        if (isAvailable || recognitionService != null) {
            try {
                // 尝试创建默认的语音识别器
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
                if (speechRecognizer == null) {
                    Log.e(TAG, "创建SpeechRecognizer失败，返回null")
                    Log.e(TAG, "系统识别服务: $recognitionService")
                    onErrorListener?.invoke("无法创建语音识别器\n系统服务: ${recognitionService ?: "未配置"}\n\n可能需要安装Google语音服务")
                } else {
                    speechRecognizer?.setRecognitionListener(recognitionListener)
                    Log.d(TAG, "语音识别器初始化成功，使用服务: $recognitionService")
                }
            } catch (e: Exception) {
                Log.e(TAG, "初始化语音识别器异常", e)
                onErrorListener?.invoke("语音服务异常: ${e.message}\n服务: $recognitionService")
            }
        } else {
            Log.e(TAG, "设备不支持语音识别，且未配置系统服务")
            onErrorListener?.invoke("设备不支持语音识别\n\n请安装以下任一服务：\n• Google 搜索应用\n• Google 语音输入\n\n或联系设备厂商")
        }
    }
    
    /**
     * 开始监听
     */
    fun startListening() {
        if (speechRecognizer == null) {
            Log.e(TAG, "SpeechRecognizer未初始化")
            onErrorListener?.invoke("语音识别器未就绪，请确保已安装语音服务")
            return
        }
        
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, 
                    RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "zh-CN")  // 中文识别
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "zh-CN")
            putExtra(RecognizerIntent.EXTRA_ONLY_RETURN_LANGUAGE_PREFERENCE, "zh-CN")
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }
        
        try {
            Log.d(TAG, "准备启动语音识别...")
            speechRecognizer?.startListening(intent)
            Log.d(TAG, "startListening调用成功，等待回调...")
        } catch (e: Exception) {
            Log.e(TAG, "启动语音识别失败", e)
            onErrorListener?.invoke("启动失败: ${e.message}\n\n提示：此设备可能未安装Google语音服务")
        }
    }
    
    /**
     * 停止监听
     */
    fun stopListening() {
        speechRecognizer?.stopListening()
        Log.d(TAG, "停止语音识别")
    }
    
    /**
     * 取消监听
     */
    fun cancel() {
        speechRecognizer?.cancel()
        Log.d(TAG, "取消语音识别")
    }
    
    /**
     * 设置结果监听器
     */
    fun setOnResultListener(listener: (String) -> Unit) {
        onResultListener = listener
    }
    
    /**
     * 设置错误监听器
     */
    fun setOnErrorListener(listener: (String) -> Unit) {
        onErrorListener = listener
    }
    
    /**
     * 销毁识别器
     */
    fun destroy() {
        speechRecognizer?.destroy()
        speechRecognizer = null
        Log.d(TAG, "语音识别器已销毁")
    }
    
    private val recognitionListener = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            Log.d(TAG, "✓ 回调: onReadyForSpeech - 准备接收语音")
        }
        
        override fun onBeginningOfSpeech() {
            Log.d(TAG, "✓ 回调: onBeginningOfSpeech - 开始说话")
        }
        
        override fun onRmsChanged(rmsdB: Float) {
            // 音量变化，可用于显示波形动画
            // Log.d(TAG, "音量: $rmsdB")
        }
        
        override fun onBufferReceived(buffer: ByteArray?) {
            // 接收到音频数据
            Log.d(TAG, "✓ 回调: onBufferReceived - 接收到音频数据")
        }
        
        override fun onEndOfSpeech() {
            Log.d(TAG, "✓ 回调: onEndOfSpeech - 说话结束")
        }
        
        override fun onError(error: Int) {
            val errorMessage = when (error) {
                SpeechRecognizer.ERROR_AUDIO -> "音频错误"
                SpeechRecognizer.ERROR_CLIENT -> "客户端错误"
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "权限不足"
                SpeechRecognizer.ERROR_NETWORK -> "网络错误"
                SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "网络超时"
                SpeechRecognizer.ERROR_NO_MATCH -> "没有匹配"
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "识别器忙"
                SpeechRecognizer.ERROR_SERVER -> "服务器错误"
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "说话超时"
                else -> "未知错误: $error"
            }
            
            Log.e(TAG, "✓ 回调: onError - 识别错误: $errorMessage (错误码: $error)")
            onErrorListener?.invoke(errorMessage)
        }
        
        override fun onResults(results: Bundle?) {
            Log.d(TAG, "✓ 回调: onResults - 收到最终结果")
            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            if (!matches.isNullOrEmpty()) {
                val result = matches[0]
                Log.d(TAG, "识别结果: $result")
                onResultListener?.invoke(result)
            } else {
                Log.w(TAG, "结果为空")
                onErrorListener?.invoke("没有识别到内容")
            }
        }
        
        override fun onPartialResults(partialResults: Bundle?) {
            // 部分结果，可用于实时显示
            val matches = partialResults?.getStringArrayList(
                SpeechRecognizer.RESULTS_RECOGNITION)
            if (!matches.isNullOrEmpty()) {
                Log.d(TAG, "✓ 回调: onPartialResults - 部分结果: ${matches[0]}")
            }
        }
        
        override fun onEvent(eventType: Int, params: Bundle?) {
            // 其他事件
            Log.d(TAG, "✓ 回调: onEvent - 事件类型: $eventType")
        }
    }
}
