package com.xiaotian.assistant

import android.app.*
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*

/**
 * 语音唤醒服务
 * 持续监听麦克风，检测唤醒词"你好小天"
 * 唤醒后持续监听3分钟，超时后停止监听
 */
class VoiceWakeupService : Service() {
        // 防抖相关变量
        private var lastRecognizedText: String? = null
        private var lastRecognizedTime: Long = 0L
        private val debounceIntervalMs = 1500L // 1.5秒防抖
    
    private var voiceRecognizer: HybridVoiceRecognizer? = null
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var isListening = false
    private var isWakeupMode = true // true: 等待唤醒词, false: 等待指令
    private var commandTimeoutJob: Job? = null // 3分钟超时计时器
    private var commandListenDeadlineMs: Long = 0L
    
    companion object {
        private const val TAG = "VoiceWakeupService"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "voice_wakeup_channel"
        private const val CHANNEL_NAME = "语音唤醒服务"
        
        // 唤醒词（含常见标点变体，stripPunctuation 会统一处理）
        private val WAKEUP_WORDS = listOf(
            "你好小天",
            "小天小天",
            "你好天天",
            "嗨小天",
            "你好，小天",
            "小天，小天"
        )
        
        // 唤醒后持续监听时长 (1分钟)
        private const val COMMAND_LISTEN_TIMEOUT_MS = 1 * 60 * 1000L
        
        // 服务状态
        @Volatile
        var isServiceRunning = false
            private set

        // 服务实例引用（用于外部通知有效指令）
        @Volatile
        private var serviceInstance: VoiceWakeupService? = null
        
        // 回调接口
        interface WakeupCallback {
            fun onWakeupDetected()
            fun onCommandReceived(command: String)
            fun onListeningStateChanged(isListening: Boolean)
        }
        
        private var callback: WakeupCallback? = null
        
        fun setCallback(cb: WakeupCallback?) {
            callback = cb
        }

        /**
         * 外部通知收到有效指令，重置3分钟计时器。
         * 由 MainActivity 在成功执行命令后调用。
         */
        fun notifyValidCommandReceived() {
            serviceInstance?.let {
                it.resetCommandTimeout()
                Log.d(TAG, "收到有效指令通知，重置3分钟计时器")
            }
        }
        
        /**
         * 启动服务
         */
        fun start(context: Context) {
            val intent = Intent(context, VoiceWakeupService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }
        
        /**
         * 停止服务
         */
        fun stop(context: Context) {
            val intent = Intent(context, VoiceWakeupService::class.java)
            context.stopService(intent)
        }
    }
    
    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "服务创建")
        
        // 创建通知渠道
        createNotificationChannel()
        
        // 启动前台服务
        startForeground(NOTIFICATION_ID, createNotification("等待唤醒词..."))
        
        isServiceRunning = true
        serviceInstance = this
        
        // 初始化 TTS 引擎（并设置麦克风控制）
        TtsManager.init(this)
        TtsManager.setMicController(object : TtsManager.MicController {
            override fun pauseMic() {
                Log.d(TAG, "TTS 播报中，暂停麦克风")
                voiceRecognizer?.stopListening()
            }
            override fun resumeMic() {
                Log.d(TAG, "TTS 播报结束，恢复麦克风")
                serviceScope.launch {
                    delay(300)  // 等待音频缓冲区完全清空
                    if (isServiceRunning) {
                        restartListening()
                    }
                }
            }
        })
        
        // 初始化语音识别器
        initVoiceRecognizer()
        
        // 开始监听
        startListening()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "服务启动")
        // 每次启动服务确保重置为唤醒模式
        isWakeupMode = true
        commandListenDeadlineMs = 0L
        updateNotification("🎤 正在监听唤醒词...")
        return START_NOT_STICKY // 服务被杀死后不自动重启，避免意外监听
    }
    
    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "服务销毁")
        
        isServiceRunning = false
        serviceInstance = null
        isListening = false
        
        // 释放 TTS 麦克风控制
        TtsManager.setMicController(null)
        
        // 先取消所有协程（防止 restartListening 在销毁过程中重新启动监听）
        serviceScope.cancel()
        commandTimeoutJob?.cancel()
        commandTimeoutJob = null
        
        // 先停止监听（优雅释放 AudioRecord），再销毁资源
        try {
            voiceRecognizer?.stopListening()
        } catch (e: Exception) {
            Log.w(TAG, "停止监听时异常", e)
        }
        
        // 等待识别线程退出
        try {
            Thread.sleep(100)
        } catch (_: InterruptedException) {}
        
        try {
            voiceRecognizer?.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "销毁识别器时异常", e)
        }
        voiceRecognizer = null
    }
    
    /**
     * 创建通知渠道
     */
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "保持语音唤醒服务运行"
                setShowBadge(false)
            }
            
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }
    
    /**
     * 创建通知
     */
    private fun createNotification(content: String): Notification {
        // 点击通知打开MainActivity
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("语音助手运行中")
            .setContentText(content)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
    
    /**
     * 更新通知
     */
    private fun updateNotification(content: String) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, createNotification(content))
    }
    
    /**
     * 初始化语音识别器
     */
    private fun initVoiceRecognizer() {
        voiceRecognizer = HybridVoiceRecognizer(this)
        
        // 设置识别结果回调
        voiceRecognizer?.setOnResultListener { text ->
            onSpeechRecognized(text)
        }
        
        // 设置错误回调
        voiceRecognizer?.setOnErrorListener { error ->
            Log.w(TAG, "识别错误: $error")
            // 错误后重新开始监听
            serviceScope.launch {
                delay(1000)
                if (isListening) {
                    restartListening()
                }
            }
        }
    }
    
    /**
     * 开始监听
     */
    private fun startListening() {
        if (!isListening) {
            isListening = true
            isWakeupMode = true
            
            Log.d(TAG, "开始监听唤醒词")
            updateNotification("🎤 正在监听唤醒词...")
            callback?.onListeningStateChanged(true)
            
            voiceRecognizer?.startListening()
        }
    }
    
    /**
     * 停止监听
     */
    private fun stopListening() {
        isListening = false
        voiceRecognizer?.stopListening()
        callback?.onListeningStateChanged(false)
    }
    
    /**
     * 重启监听
     */
    private fun restartListening() {
        Log.d(TAG, "重启监听")
        voiceRecognizer?.stopListening()
        
        // 短暂延迟后重新开始（200ms 足够清空音频缓冲区）
        serviceScope.launch {
            delay(200)
            if (isListening) {
                voiceRecognizer?.startListening()
            }
        }
    }
    
    /**
     * 语音识别结果处理
     */
    private fun onSpeechRecognized(text: String) {
        Log.d(TAG, "识别到: $text")

        // 最小长度检查（少于2个字直接丢弃）
        val normalized = stripPunctuation(text)
        if (normalized.length < 2) {
            Log.d(TAG, "输入过短，已丢弃: $text")
            restartListening()
            return
        }

        // 防抖：同一内容短时间内只处理一次
        val now = System.currentTimeMillis()
        if (text == lastRecognizedText && now - lastRecognizedTime < debounceIntervalMs) {
            Log.d(TAG, "防抖丢弃重复输入: $text")
            restartListening()
            return
        }
        lastRecognizedText = text
        lastRecognizedTime = now

        if (isWakeupMode) {
            // 唤醒模式：检测唤醒词
            if (detectWakeupWord(text)) {
                onWakeupDetected()
            } else {
                // 未检测到唤醒词，继续监听
                restartListening()
            }
        } else {
            // 指令模式：检查超时或非法状态
            // 如果 deadline 为 0 但 mode 为 false，说明状态异常（Zombie Mode），强制重置
            // 注意：now 变量在上面已定义
            if (commandListenDeadlineMs == 0L || (commandListenDeadlineMs > 0 && now > commandListenDeadlineMs)) {
                Log.w(TAG, "指令监听状态异常或超时 (deadline=$commandListenDeadlineMs)，忽略输入: $text")
                resetToWakeupMode()
                return
            }
            // 噪声过滤：在指令模式下丢弃疑似噪声的输入，避免误触发"抱歉"
            if (isLikelyNoise(text)) {
                Log.d(TAG, "指令模式噪声过滤，丢弃: $text")
                restartListening()
                return
            }
            // 指令模式：执行用户指令
            onCommandDetected(text)
        }
    }
    
    /**
     * 去除标点符号和空格，只保留中文/英文/数字
     * SenseVoice 可能输出 "你好，小天" / "你好, 小天" / "你好。小天" 等带标点的文本
     */
    private fun stripPunctuation(text: String): String {
        return text.replace(Regex("""[\s\p{Punct}，。！？、；：""''（）【】《》…—·～\u3000]+"""), "")
    }

    // 常见噪声识别产物（SenseVoice 在噪声环境下可能输出的无意义文本）
    // 包含车载场景中风噪、胎噪、路面振动可能被识别为的文本
    private val NOISE_TEXT_PATTERNS = listOf(
        // 常见语气词/无意义音节
        "嗯", "呃", "啊", "哦", "唔", "嘶", "呼", "哈", "呀", "噢", "额",
        "嗯嗯", "啊啊", "哦哦", "哈哈", "呵呵", "嗯啊", "呃呃",
        // 车载噪声常见误识别 (风噪声/胎噪/发动机声/路面声)
        "嗖", "呜", "呜呜", "嗡", "嗡嗡", "嘶嘶", "呼呼", "哗", "哗哗",
        "沙沙", "嘎", "嘎嘎", "咔", "咔咔", "轰", "轰轰",
        "咚", "咚咚", "砰", "砰砰", "滋", "滋滋", "吱", "吱吱",
        // 短促无意义词（高速风噪可能产生的）
        "啦", "的", "了", "吧", "吗", "呢", "啥", "那", "这", "一"
    )

    /**
     * 判断识别文本是否为噪声产物
     * 噪声经过 SenseVoice 可能被识别为无意义的短文本
     * 车载场景下风噪/胎噪更容易产生误识别
     */
    private fun isLikelyNoise(text: String): Boolean {
        val stripped = stripPunctuation(text)
        // 单字直接视为噪声
        if (stripped.length <= 1) return true
        // 匹配常见噪声模式
        if (NOISE_TEXT_PATTERNS.any { stripped.equals(it, ignoreCase = true) }) return true
        // 全部是同一个字的重复（如"啊啊啊啊"、"呜呜呜"）
        if (stripped.length >= 2 && stripped.all { it == stripped[0] }) return true
        // 2-3 个字且全是语气词/象声词（车载噪声特征）
        if (stripped.length in 2..3 && stripped.all { isOnomatopoeia(it) }) return true
        return false
    }

    /** 判断是否为语气词/象声词字符 */
    private fun isOnomatopoeia(c: Char): Boolean {
        return c in "嗯啊哦唔嘶呼哈呀噢额呃嗖呜嗡哗嘎咔轰咚砰滋吱嘀嘟嘭嘣咕噜沙嗒嘀"
    }

    /**
     * 检测唤醒词（精确匹配 + 模糊匹配）
     * 噪声环境下 ASR 可能把"你好小天"识别为"你豪小天"、"你好小甜"等
     * SenseVoice 可能输出 "你好，小天"（带标点），需先去除标点再匹配
     * 模糊匹配允许1个字的差异，减少噪声导致的漏检
     */
    private fun detectWakeupWord(text: String): Boolean {
        val normalizedText = stripPunctuation(text).lowercase()
        
        for (wakeupWord in WAKEUP_WORDS) {
            val normalizedWakeup = stripPunctuation(wakeupWord).lowercase()
            
            // 1. 精确匹配
            if (normalizedText.contains(normalizedWakeup)) {
                Log.d(TAG, "精确匹配唤醒词: $wakeupWord")
                return true
            }
            
            // 2. 模糊匹配（编辑距离 ≤ 1）— 容忍噪声导致的1个字识别错误
            if (normalizedText.length >= normalizedWakeup.length) {
                for (i in 0..normalizedText.length - normalizedWakeup.length) {
                    val sub = normalizedText.substring(i, i + normalizedWakeup.length)
                    if (editDistance(sub, normalizedWakeup) <= 1) {
                        Log.d(TAG, "模糊匹配唤醒词: $wakeupWord (实际识别: $sub, 原文: $text)")
                        return true
                    }
                }
            }
        }
        
        return false
    }
    
    /**
     * 计算两个字符串的编辑距离（Levenshtein Distance）
     */
    private fun editDistance(a: String, b: String): Int {
        val dp = Array(a.length + 1) { IntArray(b.length + 1) }
        for (i in 0..a.length) dp[i][0] = i
        for (j in 0..b.length) dp[0][j] = j
        for (i in 1..a.length) {
            for (j in 1..b.length) {
                dp[i][j] = minOf(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1
                )
            }
        }
        return dp[a.length][b.length]
    }
    
    /**
     * 唤醒检测到
     */
    private fun onWakeupDetected() {
        Log.d(TAG, "==== 已唤醒 ====")
        
        // 切换到指令模式
        isWakeupMode = false
        
        // 更新通知
        updateNotification("✓ 已唤醒，请说出指令...")
        
        // 通知回调
        callback?.onWakeupDetected()
        
        // 语音回复“我在”（替代提示音，播报结束后自动恢复麦克风）
        TtsManager.speak("我在")
        
        // 启动3分钟超时计时器
        resetCommandTimeout()
    }
    
    /**
     * 重置3分钟指令监听超时计时器
     * 每次收到指令后重置，3分钟无输入则返回唤醒模式
     */
    private fun resetCommandTimeout() {
        commandTimeoutJob?.cancel()
        commandListenDeadlineMs = System.currentTimeMillis() + COMMAND_LISTEN_TIMEOUT_MS
        commandTimeoutJob = serviceScope.launch {
            delay(COMMAND_LISTEN_TIMEOUT_MS)
            if (!isWakeupMode) {
                Log.d(TAG, "1分钟无输入，返回唤醒模式")
                resetToWakeupMode()
            }
        }
    }
    
    /**
     * 检测到指令
     */
    private fun onCommandDetected(command: String) {
        Log.d(TAG, "收到指令: $command")
        
        // 更新通知
        updateNotification("正在处理: $command")
        
        // 通知回调
        callback?.onCommandReceived(command)
        
        // 注意：不再自动重置3分钟计时器。
        // 只有当 MainActivity 确认指令有效并成功执行后，
        // 才会通过 notifyValidCommandReceived() 重置计时器。
        
        // 执行指令后，继续监听下一条指令
        serviceScope.launch {
            delay(2000)
            if (!isWakeupMode) {
                Log.d(TAG, "指令处理完毕，继续监听...")
                updateNotification("✓ 指令已处理，继续监听中...")
                restartListening()
            }
        }
    }
    
    /**
     * 重置到唤醒模式
     */
    private fun resetToWakeupMode() {
        Log.d(TAG, "返回唤醒模式")
        commandTimeoutJob?.cancel()
        commandTimeoutJob = null
        commandListenDeadlineMs = 0L
        isWakeupMode = true
        updateNotification("🎤 正在监听唤醒词...")
        callback?.onListeningStateChanged(true)
        restartListening()
    }
    
    /**
     * 播放提示音
     */
    private fun playBeep() {
        try {
            val toneGen = android.media.ToneGenerator(
                android.media.AudioManager.STREAM_NOTIFICATION,
                100
            )
            toneGen.startTone(android.media.ToneGenerator.TONE_PROP_BEEP, 150)
            
            serviceScope.launch {
                delay(200)
                toneGen.release()
            }
        } catch (e: Exception) {
            Log.e(TAG, "播放提示音失败", e)
        }
    }
}
