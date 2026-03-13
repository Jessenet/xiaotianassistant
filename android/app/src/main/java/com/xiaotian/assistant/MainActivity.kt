package com.xiaotian.assistant

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognizerIntent
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.textview.MaterialTextView
import kotlinx.coroutines.launch

/**
 * 主Activity
 * 集成语音识别、模型推理、QQ音乐控制、语音唤醒
 */
class MainActivity : AppCompatActivity() {
    
    private lateinit var voiceRecognizer: HybridVoiceRecognizer
    private lateinit var aiAgent: SmartAssistantAgent
    private lateinit var musicController: IMusicController
    private lateinit var functionExecutor: SmartFunctionExecutor
    
    private lateinit var btnVoice: MaterialButton
    private lateinit var btnAccessibility: MaterialButton
    private lateinit var btnWakeup: MaterialButton
    private lateinit var btnSelectApp: MaterialButton
    private lateinit var tvStatus: MaterialTextView
    private lateinit var tvResult: MaterialTextView
    private lateinit var tvAccessibilityStatus: MaterialTextView
    private lateinit var tvWakeupStatus: MaterialTextView
    private lateinit var tvSelectedApp: MaterialTextView
    
    private var isWakeupServiceRunning = false
    
    // === 助手激活状态管理 ===
    // 只有在助手被唤醒（1分钟内）或用户点击语音按钮时才处理命令
    private var isAssistantActive = false          // 助手是否处于激活状态（唤醒后1分钟内）
    private var activeDeadlineMs: Long = 0L         // 激活状态截止时间戳
    private var isManualInput = false               // 用户手动点击语音按钮（单次输入）
    private var activeTimeoutJob: kotlinx.coroutines.Job? = null  // 自动休眠定时器
    
    companion object {
        private const val REQUEST_RECORD_AUDIO_PERMISSION = 200
        private const val REQUEST_SPEECH_RECOGNITION = 201
        private const val ASSISTANT_ACTIVE_TIMEOUT_MS = 1 * 60 * 1000L  // 1分钟自动休眠
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        try {
            android.util.Log.d("MainActivity", "=== onCreate 开始 ===")
            setContentView(R.layout.activity_main)
            android.util.Log.d("MainActivity", "setContentView 完成")
            
            android.util.Log.d("MainActivity", "开始请求权限")
            
            // 请求权限
            requestPermissions()
            
            android.util.Log.d("MainActivity", "开始初始化组件")
            
            // 初始化MusicAppManager
            MusicAppManager.init(this)
            
            // 先初始化组件(包括musicController)
            initComponents()
            android.util.Log.d("MainActivity", "initComponents 完成")
            
            // 再初始化视图(需要用到musicController)
            initViews()
            android.util.Log.d("MainActivity", "initViews 完成")
            
            // 视图初始化后才能更新状态
            android.util.Log.d("MainActivity", "更新初始状态")
            updateStatus("正在初始化AI代理...")
            updateAccessibilityStatus()
            
            // 确保同步服务状态（防止服务在后台运行但UI不同步）
            isWakeupServiceRunning = VoiceWakeupService.isServiceRunning
            updateWakeupStatus()
            updateSelectedApp()
            
            setupListeners()
            
            // 检查是否通过 adb intent 传入了测试文本
            handleTestIntent(intent)
            
            android.util.Log.d("MainActivity", "=== onCreate 完成 ===")
            
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "!!! onCreate 失败 !!!", e)
            Toast.makeText(this, "启动失败: ${e.message}", Toast.LENGTH_LONG).show()
            throw e
        }
    }
    
    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        intent?.let { handleTestIntent(it) }
    }
    
    /**
     * 处理通过 adb 传入的测试文本
     * 用法: adb shell am start -n com.xiaotian.assistant/.MainActivity --es test_query "播放周杰伦的稻香"
     */
    private fun handleTestIntent(intent: Intent?) {
        val testQuery = intent?.getStringExtra("test_query") ?: return
        android.util.Log.d("MainActivity", "收到测试指令: $testQuery")
        
        // 等模型加载完成后执行
        lifecycleScope.launch {
            // 等待 aiAgent 变量赋值（最多等5秒）
            var waited = 0
            while (!::aiAgent.isInitialized && waited < 5) {
                kotlinx.coroutines.delay(1000)
                waited++
            }
            if (!::aiAgent.isInitialized) {
                android.util.Log.e("MainActivity", "aiAgent 未初始化，放弃测试")
                return@launch
            }
            
            // 等待模型实际加载完成（最多等120秒）
            val modelManager = ExecuTorchModelManager.getInstance(this@MainActivity)
            waited = 0
            while (!modelManager.isLoaded && waited < 120) {
                if (waited % 10 == 0) {
                    android.util.Log.d("MainActivity", "等待模型加载... ${waited}s")
                }
                kotlinx.coroutines.delay(1000)
                waited++
            }
            
            if (!modelManager.isLoaded) {
                android.util.Log.e("MainActivity", "模型加载超时(${waited}s)，放弃测试")
                return@launch
            }
            
            android.util.Log.d("MainActivity", "模型已加载，等待${waited}s，开始测试推理: $testQuery")
            // ADB 测试模式：临时激活助手以允许处理
            isManualInput = true
            processVoiceInput(testQuery)
        }
    }
    
    private fun initViews() {
        try {
            android.util.Log.d("MainActivity", "查找视图控件...")
            btnVoice = findViewById(R.id.btn_voice)
            btnAccessibility = findViewById(R.id.btn_accessibility)
            btnWakeup = findViewById(R.id.btn_wakeup)
            btnSelectApp = findViewById(R.id.btn_select_app)
            tvStatus = findViewById(R.id.tv_status)
            tvResult = findViewById(R.id.tv_result)
            tvAccessibilityStatus = findViewById(R.id.tv_accessibility_status)
            tvWakeupStatus = findViewById(R.id.tv_wakeup_status)
            tvSelectedApp = findViewById(R.id.tv_selected_app)
            android.util.Log.d("MainActivity", "视图控件查找完成")
            
            // 初始化音乐APP管理器(必须在获取musicController之前)
            android.util.Log.d("MainActivity", "初始化音乐APP管理器")
            MusicAppManager.init(this)
            
            android.util.Log.d("MainActivity", "initViews 所有步骤完成")
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "initViews 失败", e)
            throw e
        }
    }
    
    private fun requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_RECORD_AUDIO_PERMISSION
            )
        }
    }
    
    private fun initComponents() {
        try {
            android.util.Log.d("MainActivity", "--- initComponents 开始 ---")
            
            // 初始化 TTS 语音播报
            android.util.Log.d("MainActivity", "初始化TTS引擎")
            TtsManager.init(this)
            
            // 初始化混合语音识别器（自动切换SenseVoice/Google）
            android.util.Log.d("MainActivity", "创建HybridVoiceRecognizer")
            voiceRecognizer = HybridVoiceRecognizer(this)
            
            // 获取当前选择的音乐控制器
            android.util.Log.d("MainActivity", "获取音乐控制器")
            musicController = MusicAppManager.getCurrentController()
            
            // 初始化函数执行器 (多域版本)
            android.util.Log.d("MainActivity", "创建SmartFunctionExecutor")
            functionExecutor = SmartFunctionExecutor(this, musicController)
            
            // 延迟初始化AI代理 - 避免启动时崩溃
            android.util.Log.d("MainActivity", "启动AI代理后台加载")
            lifecycleScope.launch {
                try {
                    android.util.Log.d("MainActivity", "AI代理加载协程开始")
                    aiAgent = SmartAssistantAgent(this@MainActivity)
                    android.util.Log.d("MainActivity", "SmartAssistantAgent创建成功")
                    
                    val result = aiAgent.initialize { progress ->
                        runOnUiThread {
                            updateStatus("加载模型: $progress%")
                        }
                    }
                    
                    if (result.isSuccess) {
                        updateStatus("初始化完成，AI已就绪")
                    } else {
                        val errorMsg = result.exceptionOrNull()?.message ?: "未知错误"
                        updateStatus("模型加载失败: $errorMsg")
                        Toast.makeText(this@MainActivity, "AI模型加载失败，可能需要重启应用", Toast.LENGTH_LONG).show()
                    }
                } catch (e: Exception) {
                    val errorMsg = "${e.javaClass.simpleName}: ${e.message}"
                    updateStatus("AI初始化失败: $errorMsg")
                    android.util.Log.e("MainActivity", "AI初始化失败", e)
                    Toast.makeText(this@MainActivity, "AI初始化失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
            
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "组件初始化失败", e)
            throw e  // 向上抛出异常让onCreate的catch处理
        }
    }
    
    private fun setupListeners() {
        // 语音按钮点击（单次识别）
        btnVoice.setOnClickListener {
            isManualInput = true  // 标记为手动输入，允许单次命令处理
            android.util.Log.d("MainActivity", "用户点击语音按钮，设置 isManualInput=true")
            startVoiceRecognition()
        }
        
        // 无障碍服务按钮
        btnAccessibility.setOnClickListener {
            AccessibilityHelper.openAccessibilitySettings(this)
        }
        
        // 语音唤醒按钮
        btnWakeup.setOnClickListener {
            toggleWakeupService()
        }
        
        // 音乐APP选择按钮
        btnSelectApp.setOnClickListener {
            showAppSelectionDialog()
        }
        
        // 语音识别结果回调
        voiceRecognizer.setOnResultListener { text ->
            onVoiceRecognized(text)
        }
        
        // 语音识别错误回调
        voiceRecognizer.setOnErrorListener { error ->
            updateStatus("语音识别错误: $error")
            
            // 如果是严重错误，显示详细提示
            if (error.contains("不可用") || error.contains("不支持") || error.contains("未就绪")) {
                android.app.AlertDialog.Builder(this)
                    .setTitle("语音识别不可用")
                    .setMessage("您的设备可能未安装 Google 语音服务。\n\n" +
                            "解决方法：\n" +
                            "1. 打开 Google Play 商店\n" +
                            "2. 搜索并安装 \"Google 语音服务\"\n" +
                            "3. 重启本应用")
                    .setPositiveButton("确定", null)
                    .show()
            }
        }

        // 语音识别状态回调（非错误）
        voiceRecognizer.setOnStatusListener { status ->
            updateStatus(status)
        }
        
        // 设置唤醒服务回调
        VoiceWakeupService.setCallback(object : VoiceWakeupService.Companion.WakeupCallback {
            override fun onWakeupDetected() {
                runOnUiThread {
                    // 唤醒助手：激活1分钟命令窗口
                    activateAssistant()
                    updateStatus("✓ 已唤醒，请说出指令（1分钟内有效）...")
                    tvWakeupStatus.text = "🎤 等待指令中..."
                    // 语音回复"我在"已在 VoiceWakeupService.onWakeupDetected() 中播报，
                    // 此处不再重复播报，避免用户听到两次"我在"
                }
            }
            
            override fun onCommandReceived(command: String) {
                runOnUiThread {
                    // 只有助手激活时才处理命令
                    if (isAssistantActive && System.currentTimeMillis() < activeDeadlineMs) {
                        updateStatus("收到指令: $command")
                        onVoiceRecognized(command)
                    } else {
                        android.util.Log.w("MainActivity", "助手未激活，忽略指令: $command")
                        updateStatus("助手未激活，已忽略指令")
                    }
                }
            }
            
            override fun onListeningStateChanged(isListening: Boolean) {
                runOnUiThread {
                    if (isAssistantActive) {
                        val remainSec = ((activeDeadlineMs - System.currentTimeMillis()) / 1000).coerceAtLeast(0)
                        tvWakeupStatus.text = "🎤 等待指令中... (${remainSec}s)"
                    } else {
                        val status = if (isListening) "正在监听唤醒词..." else "已停止"
                        tvWakeupStatus.text = "🎤 $status"
                    }
                }
            }
        })
    }
    
    private fun startVoiceRecognition() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED) {
            
            updateStatus("准备启动语音识别...")
            
            // 使用HybridVoiceRecognizer（支持SenseVoice离线和Google在线）
            try {
                android.util.Log.d("MainActivity", "调用HybridVoiceRecognizer.startListening()")
                voiceRecognizer.startListening()
                updateStatus("正在监听...")
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "启动语音识别失败", e)
                updateStatus("启动失败: ${e.message}")
                Toast.makeText(this, "语音识别启动失败: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        } else {
            Toast.makeText(this, "需要麦克风权限", Toast.LENGTH_SHORT).show()
            requestPermissions()
        }
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == REQUEST_SPEECH_RECOGNITION) {
            if (resultCode == RESULT_OK && data != null) {
                val results = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
                if (!results.isNullOrEmpty()) {
                    val recognizedText = results[0]
                    android.util.Log.d("MainActivity", "语音识别成功: $recognizedText")
                    updateStatus("识别成功")
                    onVoiceRecognized(recognizedText)
                } else {
                    android.util.Log.d("MainActivity", "语音识别结果为空")
                    updateStatus("未识别到内容")
                }
            } else {
                android.util.Log.d("MainActivity", "语音识别取消或失败, resultCode: $resultCode")
                updateStatus("识别取消")
            }
        }
    }
    
    private fun processVoiceInput(text: String) {
        lifecycleScope.launch {
            try {
                // === 状态门控：只有助手激活或手动输入时才处理命令 ===
                val isActiveByWake = isAssistantActive && System.currentTimeMillis() < activeDeadlineMs
                val isActiveByManual = isManualInput
                
                if (!isActiveByWake && !isActiveByManual) {
                    android.util.Log.w("MainActivity", "助手未激活，拒绝处理: $text (isAssistantActive=$isAssistantActive, isManualInput=$isManualInput)")
                    updateStatus("助手未激活，请先唤醒或点击语音按钮")
                    return@launch
                }
                
                android.util.Log.d("MainActivity", "助手已激活，处理指令: $text (wake=$isActiveByWake, manual=$isActiveByManual)")
                
                // 手动输入：处理完一次后清除标记
                if (isActiveByManual) {
                    isManualInput = false
                }
                
                // 检查AI代理是否已初始化
                if (!::aiAgent.isInitialized) {
                    updateStatus("AI代理尚未初始化，请稍候...")
                    updateResult("模型正在加载中...")
                    Toast.makeText(this@MainActivity, "AI模型正在加载中，请稍候", Toast.LENGTH_SHORT).show()
                    return@launch
                }
                
                // AI模型处理指令
                updateStatus("正在分析指令...")
                android.util.Log.d("MainActivity", "开始调用AI处理: $text")
                val result = aiAgent.processCommand(text)
                
                android.util.Log.d("MainActivity", "AI处理结果: isSuccess=${result.isSuccess}")
                
                if (result.isSuccess) {
                    val command = result.getOrNull()!!
                    android.util.Log.d("MainActivity", "获取到命令: ${command.describe()}")
                    updateResult("识别的命令: ${command.describe()}")
                    
                    // 执行命令
                    updateStatus("正在执行...")
                    android.util.Log.d("MainActivity", "开始执行命令")
                    val execResult = functionExecutor.execute(command)
                    
                    android.util.Log.d("MainActivity", "执行结果: ${execResult.success}")
                    updateStatus(if (execResult.success) execResult.message else "执行失败: ${execResult.message}")
                    
                    // 语音播报执行结果（仅成功时播报，失败时静默避免打扰）
                    if (execResult.success) {
                        TtsManager.speak(execResult.message)
                        // 每次成功执行有效命令后重新计时1分钟
                        VoiceWakeupService.notifyValidCommandReceived()
                        if (isAssistantActive) {
                            activateAssistant()
                        }
                    }
                } else {
                    val error = result.exceptionOrNull()
                    android.util.Log.e("MainActivity", "AI处理失败", error)
                    updateStatus("AI处理失败: ${error?.message ?: "未知错误"}")
                    updateResult("无法理解指令")
                    // 不再语音播报错误，避免噪声误识别时频繁打扰用户
                }
                
            } catch (e: Exception) {
                updateStatus("处理失败: ${e.message}")
                android.util.Log.e("MainActivity", "语音处理失败", e)
            }
        }
    }
    
    private fun onVoiceRecognized(text: String) {
        updateResult("识别结果: $text")
        processVoiceInput(text)
    }
    
    /**
     * 切换唤醒服务
     */
    private fun toggleWakeupService() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "需要麦克风权限", Toast.LENGTH_SHORT).show()
            requestPermissions()
            return
        }
        
        if (isWakeupServiceRunning) {
            // 停止服务
            VoiceWakeupService.stop(this)
            isWakeupServiceRunning = false
            btnWakeup.text = "启动语音唤醒"
            btnWakeup.icon = getDrawable(android.R.drawable.ic_media_play)
            tvWakeupStatus.text = "未启动"
            updateStatus("语音唤醒已停止")
        } else {
            // 启动服务
            VoiceWakeupService.start(this)
            isWakeupServiceRunning = true
            btnWakeup.text = "停止语音唤醒"
            btnWakeup.icon = getDrawable(android.R.drawable.ic_media_pause)
            tvWakeupStatus.text = "🎤 正在监听唤醒词..."
            updateStatus("语音唤醒已启动，说\"你好小天\"唤醒")
        }
    }
    
    /**
     * 更新唤醒服务状态
     */
    private fun updateWakeupStatus() {
        isWakeupServiceRunning = VoiceWakeupService.isServiceRunning
        
        if (isWakeupServiceRunning) {
            btnWakeup.text = "停止语音唤醒"
            btnWakeup.icon = getDrawable(android.R.drawable.ic_media_pause)
            tvWakeupStatus.text = "🎤 正在监听唤醒词..."
        } else {
            btnWakeup.text = "启动语音唤醒"
            btnWakeup.icon = getDrawable(android.R.drawable.ic_media_play)
            tvWakeupStatus.text = "未启动"
        }
    }
    
    private fun updateStatus(status: String) {
        runOnUiThread {
            tvStatus.text = status
        }
    }
    
    private fun updateResult(result: String) {
        runOnUiThread {
            tvResult.text = result
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        when (requestCode) {
            REQUEST_RECORD_AUDIO_PERMISSION -> {
                if (grantResults.isNotEmpty() && 
                    grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "权限已授予", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "需要麦克风权限才能使用语音功能", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
    
    /**
     * 显示音乐APP选择对话框
     */
    private fun showAppSelectionDialog() {
        val availableApps = MusicAppManager.getAvailableApps()
        
        if (availableApps.isEmpty()) {
            Toast.makeText(this, "未检测到已安装的音乐APP", Toast.LENGTH_SHORT).show()
            return
        }
        
        val appNames = availableApps.map { it.displayName }.toTypedArray()
        val currentApp = MusicAppManager.getSelectedApp()
        val currentIndex = availableApps.indexOf(currentApp)
        
        android.app.AlertDialog.Builder(this)
            .setTitle("选择音乐APP")
            .setSingleChoiceItems(appNames, currentIndex) { dialog, which ->
                val selectedApp = availableApps[which]
                MusicAppManager.setSelectedApp(selectedApp)
                
                // 重新初始化控制器
                musicController = MusicAppManager.getCurrentController()
                functionExecutor = SmartFunctionExecutor(this@MainActivity, musicController)
                
                updateSelectedApp()
                updateAccessibilityStatus()
                
                Toast.makeText(this, "已切换到${selectedApp.displayName}", Toast.LENGTH_SHORT).show()
                dialog.dismiss()
            }
            .setNegativeButton("取消", null)
            .show()
    }
    
    /**
     * 更新选择的音乐APP显示
     */
    private fun updateSelectedApp() {
        val currentApp = MusicAppManager.getSelectedApp()
        tvSelectedApp.text = "当前: ${currentApp.displayName}"
    }
    
    override fun onResume() {
        super.onResume()
        
        // 同步服务状态，防止UI与实际状态不一致
        isWakeupServiceRunning = VoiceWakeupService.isServiceRunning
        
        updateAccessibilityStatus()
        updateWakeupStatus()
        updateSelectedApp()
    }
    
    private fun updateAccessibilityStatus() {
        val isEnabled = AccessibilityHelper.isAccessibilityServiceEnabled(this)
        val statusText = if (isEnabled) {
            "✓ 无障碍服务已启用"
        } else {
            "✗ 无障碍服务未启用（推荐启用以获得更好的控制）"
        }
        tvAccessibilityStatus.text = statusText
        
        // 更新控制器设置
        if (musicController is QQMusicController) {
            (musicController as QQMusicController).setUseAccessibilityService(isEnabled)
        } else if (musicController is NeteaseMusicController) {
            (musicController as NeteaseMusicController).setUseAccessibilityService(isEnabled)
        }
    }
    
    // === 助手激活状态管理方法 ===
    
    /**
     * 激活助手（唤醒后1分钟内可接收指令）
     */
    private fun activateAssistant() {
        isAssistantActive = true
        activeDeadlineMs = System.currentTimeMillis() + ASSISTANT_ACTIVE_TIMEOUT_MS
        android.util.Log.d("MainActivity", "助手已激活，有效期至: ${java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date(activeDeadlineMs))}")
        
        // 取消之前的定时器
        activeTimeoutJob?.cancel()
        
        // 启动自动休眠定时器
        activeTimeoutJob = lifecycleScope.launch {
            // 每10秒更新一次剩余时间显示
            while (System.currentTimeMillis() < activeDeadlineMs) {
                val remainSec = ((activeDeadlineMs - System.currentTimeMillis()) / 1000).coerceAtLeast(0)
                runOnUiThread {
                    if (isAssistantActive) {
                        tvWakeupStatus.text = "🎤 等待指令中... (${remainSec}s)"
                    }
                }
                kotlinx.coroutines.delay(10_000)
            }
            
            // 超时：自动休眠
            deactivateAssistant()
        }
    }
    
    /**
     * 停用助手（回到待唤醒状态）
     */
    private fun deactivateAssistant() {
        isAssistantActive = false
        activeDeadlineMs = 0L
        isManualInput = false
        activeTimeoutJob?.cancel()
        activeTimeoutJob = null
        android.util.Log.d("MainActivity", "助手已休眠")
        
        runOnUiThread {
            updateStatus("助手已休眠")
            if (isWakeupServiceRunning) {
                tvWakeupStatus.text = "🎤 正在监听唤醒词..."
            } else {
                tvWakeupStatus.text = "未启动"
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        activeTimeoutJob?.cancel()
        VoiceWakeupService.setCallback(null)
        TtsManager.shutdown()
        aiAgent.release()
        try {
            voiceRecognizer.destroy()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}