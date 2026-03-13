package com.xiaotian.assistant

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import kotlinx.coroutines.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * QQ音乐无障碍服务
 * 通过模拟点击来精确控制QQ音乐播放器
 */
class QQMusicAccessibilityService : AccessibilityService() {
    
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    companion object {
        private const val TAG = "QQMusicAccessibility"
        
        // QQ音乐包名
        private const val QQMUSIC_PACKAGE = "com.tencent.qqmusic"
        
        // 服务实例（用于外部调用）
        @Volatile
        private var instance: QQMusicAccessibilityService? = null
        
        // 当前QQ音乐活动的类名（通过 onAccessibilityEvent 跟踪）
        @Volatile
        private var currentQQMusicActivity: String = ""
        
        // QQ音乐播放器Activity
        private const val PLAYER_ACTIVITY = "com.tencent.qqmusic.business.playernew.view.NewPlayerActivity"
        
        fun getInstance(): QQMusicAccessibilityService? = instance
        
        fun isServiceEnabled(): Boolean = instance != null
        
        // 控件ID和文本（可能需要根据实际QQ音乐版本调整）
        private val PLAY_BUTTON_IDS = arrayOf(
            "play_btn",
            "btn_play",
            "play_pause_btn",
            "player_play_btn"
        )
        
        private val PAUSE_BUTTON_IDS = arrayOf(
            "pause_btn",
            "btn_pause",
            "play_pause_btn",
            "player_play_btn"
        )
        
        private val NEXT_BUTTON_IDS = arrayOf(
            "next_btn",
            "btn_next",
            "player_next_btn"
        )
        
        private val PREV_BUTTON_IDS = arrayOf(
            "prev_btn",
            "btn_prev",
            "previous_btn",
            "player_prev_btn"
        )
        
        private val SEARCH_BUTTON_IDS = arrayOf(
            "search_btn",
            "btn_search",
            "search_icon"
        )
    }
    
    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.d(TAG, "无障碍服务已连接")
    }
    
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return
        
        // 记录QQ音乐的界面变化
        if (event.packageName == QQMUSIC_PACKAGE) {
            when (event.eventType) {
                AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> {
                    val className = event.className?.toString() ?: ""
                    Log.d(TAG, "QQ音乐界面变化: $className")
                    // 跟踪当前Activity（只记录Activity类名，忽略弹窗等View类名）
                    if (className.contains("Activity")) {
                        currentQQMusicActivity = className
                    }
                }
                AccessibilityEvent.TYPE_VIEW_CLICKED -> {
                    Log.d(TAG, "QQ音乐点击事件")
                }
            }
        }
    }
    
    override fun onInterrupt() {
        Log.d(TAG, "服务中断")
    }
    
    override fun onDestroy() {
        super.onDestroy()
        instance = null
        serviceScope.cancel()
        Log.d(TAG, "无障碍服务已销毁")
    }
    
    /**
     * 搜索并播放歌曲
     */
    fun searchAndPlay(songName: String, artist: String? = null): Boolean {
        return try {
            val query = if (artist != null) "$artist $songName" else songName
            Log.d(TAG, "搜索并播放歌曲: $query")
            
            // 步骤1：打开QQ音乐
            Log.d(TAG, "步骤1: 打开QQ音乐...")
            if (!openQQMusic()) {
                Log.w(TAG, "无法打开QQ音乐")
                return false
            }
            
            // 步骤1.5：确保回到主界面（从播放器/其他页面返回）
            if (!ensureOnMainScreen()) {
                Log.w(TAG, "无法回到主界面")
                return false
            }
            
            // 步骤2：点击顶部搜索按钮，打开搜索界面
            Log.d(TAG, "步骤2: 点击顶部搜索按钮...")
            if (!clickSearchButton()) {
                Log.w(TAG, "未找到搜索按钮")
                return false
            }
            
            Thread.sleep(500) // 等待搜索界面打开
            
            // 步骤3：将关键词输入顶部搜索框
            Log.d(TAG, "步骤3: 输入关键词到搜索框: $query")
            if (!inputSearchText(query)) {
                Log.w(TAG, "无法输入搜索文本")
                return false
            }
            
            Thread.sleep(200) // 等待输入完成
            
            // 步骤4：回车触发搜索
            Log.d(TAG, "步骤4: 回车触发搜索...")
            if (!sendEnterKey()) {
                Log.w(TAG, "回车触发失败，尝试备用触发方式")
                if (!triggerSearch()) {
                    Log.w(TAG, "无法触发搜索，但将继续尝试")
                }
            }
            
            // 搜索结果加载等待移到clickFirstSearchResult()中统一处理
            
            // 步骤5：点击第一个搜索结果播放
            Log.d(TAG, "步骤5: 点击第一个搜索结果...")
            if (!clickFirstSearchResult()) {
                Log.w(TAG, "未找到搜索结果")
                return false
            }
            
            // QQ音乐会自动开始播放
            Log.d(TAG, "✓ 搜索并播放完成！")
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "搜索播放失败", e)
            false
        }
    }

    /**
     * 发送回车键触发搜索
     * 优先使用无障碍API，不依赖shell权限
     */
    private fun sendEnterKey(): Boolean {
        return try {
            // 方法1: API 30+ 使用 IME_ENTER（最可靠，不需要shell权限）
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                val rootNode = rootInActiveWindow
                if (rootNode != null) {
                    val editText = findFocusedEditNode(rootNode) ?: findEditableNode(rootNode)
                    if (editText != null) {
                        val result = editText.performAction(
                            AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.id
                        )
                        if (result) {
                            Log.d(TAG, "通过 IME_ENTER 触发搜索成功")
                            Thread.sleep(200)
                            return true
                        }
                    }
                }
            }

            // 方法2: 查找键盘上的"搜索"按钮并点击（通常在屏幕下半部分）
            val rootNode = rootInActiveWindow
            if (rootNode != null) {
                val searchBtns = rootNode.findAccessibilityNodeInfosByText("搜索")
                val screenHeight = resources.displayMetrics.heightPixels
                for (btn in searchBtns) {
                    val bounds = Rect()
                    btn.getBoundsInScreen(bounds)
                    // 键盘/输入法上的搜索按钮在屏幕下半部分
                    if (bounds.top > screenHeight / 2 && btn.isClickable) {
                        btn.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                        Log.d(TAG, "通过键盘搜索按钮触发搜索成功")
                        Thread.sleep(200)
                        return true
                    }
                }
            }

            // 方法3: 回退到 shell 命令发送回车
            Runtime.getRuntime().exec("input keyevent 66")
            Thread.sleep(200)
            Log.d(TAG, "通过 shell 命令发送回车键")
            true
        } catch (e: Exception) {
            Log.e(TAG, "发送回车键失败", e)
            false
        }
    }
    
    /**
     * 点击播放/暂停按钮
     */
    fun clickPlayPause(): Boolean {
        return try {
            val rootNode = rootInActiveWindow ?: run {
                Log.w(TAG, "无法获取根节点")
                return false
            }
            
            Log.d(TAG, "查找播放/暂停按钮...")
            Log.d(TAG, "当前界面: ${rootNode.packageName}")
            
            // 等待界面稳定
            Thread.sleep(500)
            
            // 方法0：优先查找右下角播放按钮（QQ音乐播放按钮通常在右下角）
            Log.d(TAG, "查找右下角播放按钮...")
            val bottomRightButton = findBottomRightPlayButton(rootNode)
            if (bottomRightButton != null) {
                Log.d(TAG, "找到右下角播放按钮")
                if (tryMultipleClickMethods(bottomRightButton)) {
                    return true
                }
            }
            
            // 方法1：通过ID查找（QQ音乐常见的播放按钮ID）
            val playPauseIds = arrayOf(
                // QQ音乐常见ID
                "play_pause_btn", "btn_play_pause", "play_btn", "pause_btn",
                "player_play_btn", "player_pause_btn", "bottom_play_btn",
                "play_pause", "iv_play", "iv_pause", "play_pause_iv",
                // 通用播放按钮ID
                "play", "pause", "btn_play", "btn_pause", "playButton", "pauseButton"
            )
            
            for (id in playPauseIds) {
                val nodes = rootNode.findAccessibilityNodeInfosByViewId("$QQMUSIC_PACKAGE:id/$id")
                if (nodes.isNotEmpty()) {
                    val node = nodes[0]
                    Log.d(TAG, "通过ID找到播放按钮: $id, clickable=${node.isClickable}, " +
                            "enabled=${node.isEnabled}, bounds=${getBoundsString(node)}")
                    
                    // 尝试多种点击方式
                    if (tryMultipleClickMethods(node)) {
                        Log.d(TAG, "成功点击播放按钮（ID: $id）")
                        return true
                    }
                }
            }
            
            // 方法2：通过ContentDescription查找（精确匹配）
            val playDescriptions = arrayOf("播放", "play", "Play", "暂停", "pause", "Pause")
            for (desc in playDescriptions) {
                val nodes = rootNode.findAccessibilityNodeInfosByText(desc)
                for (node in nodes) {
                    val nodeDesc = node.contentDescription?.toString() ?: ""
                    // 精确匹配描述，避免匹配到"正在播放"等
                    if (nodeDesc == desc && node.isClickable) {
                        Log.d(TAG, "通过ContentDescription精确匹配到播放按钮: $desc")
                        if (tryMultipleClickMethods(node)) {
                            return true
                        }
                    }
                }
            }
            
            // 方法3：查找底部播放栏的播放按钮（通常在屏幕下方）
            Log.d(TAG, "尝试查找底部播放栏...")
            val bottomPlayButton = findBottomPlayButton(rootNode)
            if (bottomPlayButton != null) {
                Log.d(TAG, "找到底部播放按钮")
                if (tryMultipleClickMethods(bottomPlayButton)) {
                    return true
                }
            }
            
            // 方法4：查找中央的大播放按钮（通常在播放页面中央）
            Log.d(TAG, "尝试查找中央播放按钮...")
            val centerPlayButton = findCenterPlayButton(rootNode)
            if (centerPlayButton != null) {
                Log.d(TAG, "找到中央播放按钮")
                if (tryMultipleClickMethods(centerPlayButton)) {
                    return true
                }
            }
            
            // 方法5：查找所有ImageButton/ImageView，按位置和大小筛选
            Log.d(TAG, "尝试智能查找播放按钮...")
            val playButton = findPlayButtonByCharacteristics(rootNode)
            if (playButton != null) {
                Log.d(TAG, "通过特征找到播放按钮")
                if (tryMultipleClickMethods(playButton)) {
                    return true
                }
            }
            
            Log.w(TAG, "未找到播放/暂停按钮")
            // 打印所有可能的播放相关节点
            printPlayRelatedNodes(rootNode)
            false
        } catch (e: Exception) {
            Log.e(TAG, "点击播放/暂停失败", e)
            false
        }
    }
    
    /**
     * 尝试多种点击方法
     */
    private fun tryMultipleClickMethods(node: AccessibilityNodeInfo): Boolean {
        // 方法1：标准点击
        if (node.isClickable && node.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
            Log.d(TAG, "标准点击成功")
            Thread.sleep(300)
            return true
        }
        
        // 方法2：点击父节点
        var parent = node.parent
        var depth = 0
        while (parent != null && depth < 3) {
            if (parent.isClickable && parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                Log.d(TAG, "点击父节点成功（深度: $depth）")
                Thread.sleep(300)
                return true
            }
            parent = parent.parent
            depth++
        }
        
        // 方法3：手势点击
        if (performGestureClick(node)) {
            Log.d(TAG, "手势点击成功")
            Thread.sleep(300)
            return true
        }
        
        Log.w(TAG, "所有点击方法都失败")
        return false
    }
    
    /**
     * 查找右下角播放按钮（QQ音乐播放按钮通常在此位置）
     */
    private fun findBottomRightPlayButton(root: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        val rootBounds = Rect()
        root.getBoundsInScreen(rootBounds)
        val screenWidth = rootBounds.width()
        val screenHeight = rootBounds.height()
        
        // 右下角区域：屏幕右侧30%，底部30%
        val rightThreshold = screenWidth * 0.7f
        val bottomThreshold = screenHeight * 0.7f
        
        Log.d(TAG, "屏幕尺寸: ${screenWidth}x${screenHeight}, 右下角区域: x>${rightThreshold}, y>${bottomThreshold}")
        
        val buttons = mutableListOf<AccessibilityNodeInfo>()
        findNodesByClassName(root, buttons, "android.widget.ImageButton", 0, 8)
        findNodesByClassName(root, buttons, "android.widget.ImageView", 0, 8)
        findNodesByClassName(root, buttons, "android.view.View", 0, 8)
        
        val candidates = mutableListOf<Pair<AccessibilityNodeInfo, Int>>()
        
        for (button in buttons) {
            val bounds = getBounds(button)
            if (bounds == null || bounds.width() <= 0 || bounds.height() <= 0) continue
            
            // 必须在右下角区域
            if (bounds.centerX() > rightThreshold && bounds.centerY() > bottomThreshold) {
                val desc = button.contentDescription?.toString() ?: ""
                val viewId = button.viewIdResourceName ?: ""
                
                // 计算得分
                var score = 0
                
                // 位置得分：越接近右下角得分越高
                val rightness = (bounds.centerX() - rightThreshold) / (screenWidth - rightThreshold)
                val bottomness = (bounds.centerY() - bottomThreshold) / (screenHeight - bottomThreshold)
                score += ((rightness + bottomness) * 5).toInt()
                
                // 大小得分：40-100dp的按钮
                val size = Math.max(bounds.width(), bounds.height())
                if (size in 80..250) score += 3
                
                // 可点击性
                if (button.isClickable) score += 2
                if (button.isEnabled) score += 1
                
                // 描述包含播放相关词
                if (desc.contains("播放", ignoreCase = true) || desc.contains("play", ignoreCase = true)) {
                    score += 10
                }
                
                // ID包含play相关
                if (viewId.contains("play", ignoreCase = true)) {
                    score += 5
                }
                
                if (score > 0) {
                    Log.d(TAG, "右下角候选按钮: score=$score, bounds=$bounds, desc=$desc, id=$viewId")
                    candidates.add(Pair(button, score))
                }
            }
        }
        
        // 返回得分最高的候选
        return candidates.maxByOrNull { it.second }?.first
    }
    
    /**
     * 查找底部播放栏的播放按钮
     */
    private fun findBottomPlayButton(root: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        val screenHeight = root.window?.let {
            val bounds = Rect()
            root.getBoundsInScreen(bounds)
            bounds.height()
        } ?: 2000
        
        val bottomThreshold = screenHeight * 0.7f  // 屏幕下方30%
        
        val buttons = mutableListOf<AccessibilityNodeInfo>()
        findNodesByClassName(root, buttons, "android.widget.ImageButton", 0, 8)
        findNodesByClassName(root, buttons, "android.widget.ImageView", 0, 8)
        
        for (button in buttons) {
            if (!button.isClickable && !button.isEnabled) continue
            
            val bounds = Rect()
            button.getBoundsInScreen(bounds)
            
            // 在底部区域，且大小合适（40-120dp）
            if (bounds.centerY() > bottomThreshold && bounds.width() in 80..300) {
                val desc = button.contentDescription?.toString() ?: ""
                if (desc.contains("播放", ignoreCase = true) || 
                    desc.contains("play", ignoreCase = true)) {
                    return button
                }
            }
        }
        
        return null
    }
    
    /**
     * 查找中央的大播放按钮
     */
    private fun findCenterPlayButton(root: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        val bounds = Rect()
        root.getBoundsInScreen(bounds)
        val centerX = bounds.centerX()
        val centerY = bounds.centerY()
        
        val buttons = mutableListOf<AccessibilityNodeInfo>()
        findNodesByClassName(root, buttons, "android.widget.ImageButton", 0, 8)
        findNodesByClassName(root, buttons, "android.widget.ImageView", 0, 8)
        
        // 查找靠近屏幕中央的大按钮
        for (button in buttons) {
            val buttonBounds = Rect()
            button.getBoundsInScreen(buttonBounds)
            
            val distanceFromCenter = Math.sqrt(
                Math.pow((buttonBounds.centerX() - centerX).toDouble(), 2.0) +
                Math.pow((buttonBounds.centerY() - centerY).toDouble(), 2.0)
            )
            
            // 大小在100-300dp，且距离中心不太远
            val size = Math.max(buttonBounds.width(), buttonBounds.height())
            if (size in 200..600 && distanceFromCenter < 400) {
                Log.d(TAG, "找到中央区域大按钮: size=$size, distance=$distanceFromCenter")
                return button
            }
        }
        
        return null
    }
    
    /**
     * 通过特征查找播放按钮
     */
    private fun findPlayButtonByCharacteristics(root: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        val buttons = mutableListOf<AccessibilityNodeInfo>()
        findNodesByClassName(root, buttons, "android.widget.ImageButton", 0, 8)
        findNodesByClassName(root, buttons, "android.widget.ImageView", 0, 8)
        
        // 按优先级排序：大小合适、可点击、有相关描述
        val candidates = buttons.mapNotNull { button ->
            val bounds = Rect()
            button.getBoundsInScreen(bounds)
            val size = Math.max(bounds.width(), bounds.height())
            val desc = button.contentDescription?.toString() ?: ""
            val id = button.viewIdResourceName ?: ""
            
            var score = 0
            
            // 大小评分（80-200dp最佳）
            if (size in 160..400) score += 3
            else if (size in 80..500) score += 1
            
            // 可点击性评分
            if (button.isClickable) score += 2
            if (button.isEnabled) score += 1
            
            // 描述评分
            if (desc.contains("播放", ignoreCase = true) || desc.contains("play", ignoreCase = true)) score += 5
            if (desc.contains("暂停", ignoreCase = true) || desc.contains("pause", ignoreCase = true)) score += 5
            
            // ID评分
            if (id.contains("play", ignoreCase = true) || id.contains("pause", ignoreCase = true)) score += 3
            
            if (score > 0) {
                Triple(button, score, "size=$size, desc=$desc, id=$id, clickable=${button.isClickable}")
            } else null
        }.sortedByDescending { it.second }
        
        if (candidates.isNotEmpty()) {
            val (button, score, info) = candidates[0]
            Log.d(TAG, "找到最佳候选播放按钮: score=$score, $info")
            return button
        }
        
        return null
    }
    
    /**
     * 获取节点边界信息（字符串格式）
     */
    private fun getBoundsString(node: AccessibilityNodeInfo): String {
        val bounds = Rect()
        node.getBoundsInScreen(bounds)
        return "Rect(${bounds.left}, ${bounds.top} - ${bounds.right}, ${bounds.bottom})"
    }
    
    /**
     * 打印播放相关节点（调试用）
     */
    private fun printPlayRelatedNodes(root: AccessibilityNodeInfo) {
        Log.d(TAG, "=== 播放相关节点 ===")
        val buttons = mutableListOf<AccessibilityNodeInfo>()
        findNodesByClassName(root, buttons, "android.widget.ImageButton", 0, 8)
        findNodesByClassName(root, buttons, "android.widget.ImageView", 0, 8)
        
        for ((index, button) in buttons.take(10).withIndex()) {
            val bounds = Rect()
            button.getBoundsInScreen(bounds)
            val desc = button.contentDescription?.toString() ?: ""
            val id = button.viewIdResourceName ?: ""
            
            Log.d(TAG, "按钮${index + 1}: id=$id, desc=$desc, " +
                    "size=${bounds.width()}x${bounds.height()}, " +
                    "clickable=${button.isClickable}, enabled=${button.isEnabled}")
        }
    }
    
    /**
     * 查找指定类名的所有节点
     */
    private fun findNodesByClassName(
        node: AccessibilityNodeInfo?,
        result: MutableList<AccessibilityNodeInfo>,
        className: String,
        depth: Int,
        maxDepth: Int
    ) {
        if (node == null || depth > maxDepth) return
        
        if (node.className?.toString() == className) {
            result.add(node)
        }
        
        for (i in 0 until node.childCount) {
            findNodesByClassName(node.getChild(i), result, className, depth + 1, maxDepth)
        }
    }
    
    /**
     * 点击下一首按钮
     */
    fun clickNext(): Boolean {
        return try {
            val rootNode = rootInActiveWindow ?: return false
            
            val nextNode = findNodeByIds(rootNode, NEXT_BUTTON_IDS)
                ?: findNodeByContentDescription(rootNode, "下一首", "next")
            
            if (nextNode != null) {
                return performClick(nextNode)
            }
            
            Log.w(TAG, "未找到下一首按钮")
            false
        } catch (e: Exception) {
            Log.e(TAG, "点击下一首失败", e)
            false
        }
    }
    
    /**
     * 点击上一首按钮
     */
    fun clickPrevious(): Boolean {
        return try {
            val rootNode = rootInActiveWindow ?: return false
            
            val prevNode = findNodeByIds(rootNode, PREV_BUTTON_IDS)
                ?: findNodeByContentDescription(rootNode, "上一首", "previous", "prev")
            
            if (prevNode != null) {
                return performClick(prevNode)
            }
            
            Log.w(TAG, "未找到上一首按钮")
            false
        } catch (e: Exception) {
            Log.e(TAG, "点击上一首失败", e)
            false
        }
    }
    
    /**
     * 打开QQ音乐应用
     */
    private fun openQQMusic(): Boolean {
        return try {
            val intent = packageManager.getLaunchIntentForPackage(QQMUSIC_PACKAGE)
            if (intent != null) {
                intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                startActivity(intent)
                true
            } else {
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "打开QQ音乐失败", e)
            false
        }
    }
    
    /**
     * 确保当前停留在QQ音乐主界面（有搜索栏的页面）
     * 如果在播放器页面或其他子页面，通过BACK键返回主界面
     * @return true 已在主界面，false 超时失败
     */
    private fun ensureOnMainScreen(): Boolean {
        val maxWaitMs = 18000L
        val checkIntervalMs = 500L
        var waited = 0L
        var backPressed = false
        
        while (waited < maxWaitMs) {
            Thread.sleep(checkIntervalMs)
            waited += checkIntervalMs
            
            val rootNode = rootInActiveWindow ?: continue
            val pkg = rootNode.packageName?.toString() ?: continue
            
            if (pkg != QQMUSIC_PACKAGE) {
                if (waited % 2000 == 0L) {
                    Log.d(TAG, "等待QQ音乐界面... 当前活跃: $pkg (${waited}ms)")
                }
                continue
            }
            
            // QQ音乐已在前台，检查是否在主界面
            val hasBottomNav = rootNode.findAccessibilityNodeInfosByText("首页")?.isNotEmpty() == true
            val hasSearchEntry = findSearchEntry(rootNode)
            
            if (hasBottomNav && hasSearchEntry) {
                Log.d(TAG, "已在QQ音乐主界面 (等待了 ${waited}ms)")
                Thread.sleep(300)
                return true
            }
            
            // 检测是否在播放器界面或其他非主界面子页面
            val isPlayerScreen = currentQQMusicActivity == PLAYER_ACTIVITY || isInPlayingScreen()
            val isNonMainScreen = !hasBottomNav && !hasSearchEntry
            
            if ((isPlayerScreen || isNonMainScreen) && !backPressed) {
                Log.d(TAG, "检测到非主界面 (activity=$currentQQMusicActivity, player=$isPlayerScreen), 按返回键...")
                performGlobalAction(GLOBAL_ACTION_BACK)
                backPressed = true
                Thread.sleep(800) // 等待界面切换动画
                continue
            }
            
            // 已经按过返回键但仍不在主界面，可能有多层页面
            if (isNonMainScreen && backPressed && waited % 2000 == 0L) {
                Log.d(TAG, "仍不在主界面，再次按返回键 (${waited}ms)")
                performGlobalAction(GLOBAL_ACTION_BACK)
                Thread.sleep(800)
            }
        }
        
        // 超时后最终检查
        val rootNode = rootInActiveWindow
        if (rootNode?.packageName?.toString() == QQMUSIC_PACKAGE) {
            val hasSearch = findSearchEntry(rootNode)
            if (hasSearch) {
                Log.d(TAG, "超时但已在有搜索入口的界面，继续")
                return true
            }
            Log.w(TAG, "超时，QQ音乐已打开但未找到搜索入口")
            return true  // 仍然尝试继续，让后续步骤处理
        }
        
        Log.e(TAG, "超时，QQ音乐未打开")
        return false
    }
    
    /**
     * 检查当前界面是否有搜索入口
     */
    private fun findSearchEntry(rootNode: AccessibilityNodeInfo): Boolean {
        // 通过 ID 查找
        val searchIds = SEARCH_BUTTON_IDS + arrayOf("searchItem", "search_edit_text", "search_input")
        for (id in searchIds) {
            val nodes = rootNode.findAccessibilityNodeInfosByViewId("$QQMUSIC_PACKAGE:id/$id")
            if (nodes?.isNotEmpty() == true) return true
        }
        // 通过描述查找
        val descs = rootNode.findAccessibilityNodeInfosByText("搜索")
        if (descs != null) {
            for (node in descs) {
                // 确保是 QQ 音乐的搜索入口，而非其他应用
                val viewId = node.viewIdResourceName ?: ""
                if (viewId.startsWith(QQMUSIC_PACKAGE)) return true
                val desc = node.contentDescription?.toString() ?: ""
                if (desc.contains("搜索") && desc.length <= 20) return true
            }
        }
        return false
    }

    /**
     * 检测是否在播放界面
     * 通过检测播放控件的存在来判断
     */
    private fun isInPlayingScreen(): Boolean {
        return try {
            val rootNode = rootInActiveWindow ?: return false
            
            Log.d(TAG, "检测是否在播放界面...")
            
            // 方法1：检测播放界面特有的大型播放控件
            val rootBounds = Rect()
            rootNode.getBoundsInScreen(rootBounds)
            val screenWidth = rootBounds.width()
            val screenHeight = rootBounds.height()
            
            val buttons = mutableListOf<AccessibilityNodeInfo>()
            findNodesByClassName(rootNode, buttons, "android.widget.ImageButton", 0, 8)
            findNodesByClassName(rootNode, buttons, "android.widget.ImageView", 0, 8)
            
            // 在屏幕中下部（50%-90%高度）查找大型播放按钮（尺寸 > 150px）
            for (button in buttons) {
                val bounds = getBounds(button) ?: continue
                val size = Math.max(bounds.width(), bounds.height())
                val centerY = bounds.centerY()
                
                // 大型播放控件（通常在播放界面中央或底部）
                if (size > 150 && centerY > screenHeight * 0.5 && centerY < screenHeight * 0.9) {
                    val desc = button.contentDescription?.toString() ?: ""
                    val viewId = button.viewIdResourceName ?: ""
                    
                    if (desc.contains("播放", ignoreCase = true) || 
                        desc.contains("暂停", ignoreCase = true) ||
                        desc.contains("play", ignoreCase = true) ||
                        desc.contains("pause", ignoreCase = true) ||
                        viewId.contains("play", ignoreCase = true) ||
                        viewId.contains("pause", ignoreCase = true)) {
                        Log.d(TAG, "检测到播放界面（大型播放控件）: size=$size, desc=$desc, id=$viewId")
                        return true
                    }
                }
            }
            
            // 方法2：检测播放界面特有的节点ID
            val playScreenIds = arrayOf(
                "player_play_btn", "player_pause_btn", "player_cover",
                "player_album", "player_lyric", "lyric_view"
            )
            for (id in playScreenIds) {
                val nodes = rootNode.findAccessibilityNodeInfosByViewId("$QQMUSIC_PACKAGE:id/$id")
                if (nodes.isNotEmpty()) {
                    Log.d(TAG, "检测到播放界面（特有ID）: $id")
                    return true
                }
            }
            
            Log.d(TAG, "未检测到播放界面特征")
            false
        } catch (e: Exception) {
            Log.e(TAG, "检测播放界面失败", e)
            false
        }
    }
    
    /**
     * 点击屏幕左上角的返回/下拉按钮
     * 用于从播放界面返回主界面
     */
    private fun clickTopLeftBackButton(): Boolean {
        return try {
            val rootNode = rootInActiveWindow ?: run {
                Log.w(TAG, "无法获取根节点")
                return false
            }
            
            Log.d(TAG, "查找左上角返回按钮...")
            
            val buttons = mutableListOf<AccessibilityNodeInfo>()
            findNodesByClassName(rootNode, buttons, "android.widget.ImageButton", 0, 5)
            findNodesByClassName(rootNode, buttons, "android.widget.ImageView", 0, 5)
            
            val rootBounds = Rect()
            rootNode.getBoundsInScreen(rootBounds)
            val screenWidth = rootBounds.width()
            val screenHeight = rootBounds.height()
            
            // 左上角区域：左侧20%，顶部15%
            val leftThreshold = screenWidth * 0.2f
            val topThreshold = screenHeight * 0.15f
            
            Log.d(TAG, "屏幕尺寸: ${screenWidth}x${screenHeight}, 左上角区域: x<${leftThreshold}, y<${topThreshold}")
            
            val candidates = mutableListOf<Pair<AccessibilityNodeInfo, Int>>()
            
            for (button in buttons) {
                val bounds = getBounds(button) ?: continue
                
                // 必须在左上角区域
                if (bounds.centerX() < leftThreshold && bounds.centerY() < topThreshold) {
                    val desc = button.contentDescription?.toString() ?: ""
                    val viewId = button.viewIdResourceName ?: ""
                    
                    var score = 0
                    
                    // 位置得分：越接近左上角得分越高
                    val leftness = 1.0f - (bounds.centerX() / leftThreshold)
                    val topness = 1.0f - (bounds.centerY() / topThreshold)
                    score += ((leftness + topness) * 10).toInt()
                    
                    // 大小得分：30-80dp的按钮
                    val size = Math.max(bounds.width(), bounds.height())
                    if (size in 60..160) score += 5
                    
                    // 可点击性
                    if (button.isClickable) score += 3
                    if (button.isEnabled) score += 2
                    
                    // 描述包含返回相关词
                    if (desc.contains("返回", ignoreCase = true) || 
                        desc.contains("back", ignoreCase = true) ||
                        desc.contains("下拉", ignoreCase = true) ||
                        desc.contains("收起", ignoreCase = true)) {
                        score += 15
                    }
                    
                    // ID包含back/返回相关
                    if (viewId.contains("back", ignoreCase = true) ||
                        viewId.contains("return", ignoreCase = true) ||
                        viewId.contains("arrow", ignoreCase = true)) {
                        score += 10
                    }
                    
                    if (score > 0) {
                        Log.d(TAG, "左上角候选按钮: score=$score, bounds=$bounds, desc=$desc, id=$viewId")
                        candidates.add(Pair(button, score))
                    }
                }
            }
            
            // 返回得分最高的候选并点击
            val bestCandidate = candidates.maxByOrNull { it.second }
            if (bestCandidate != null) {
                Log.d(TAG, "选择最佳候选按钮（score=${bestCandidate.second}）")
                if (tryMultipleClickMethods(bestCandidate.first)) {
                    Log.d(TAG, "成功点击左上角返回按钮")
                    return true
                }
            }
            
            // 如果没有找到明确的返回按钮，尝试点击左上角固定位置
            Log.d(TAG, "未找到明确返回按钮，尝试点击左上角固定位置...")
            val clickX = screenWidth * 0.1f  // 屏幕宽的10%
            val clickY = screenHeight * 0.05f  // 屏幕高的5%
            Log.d(TAG, "点击位置: ($clickX, $clickY)")
            
            if (tapScreenPosition(clickX, clickY)) {
                Log.d(TAG, "成功点击左上角位置")
                return true
            }
            
            Log.w(TAG, "未找到或无法点击左上角返回按钮")
            false
        } catch (e: Exception) {
            Log.e(TAG, "点击返回按钮失败", e)
            false
        }
    }
    
    /**
     * 点击搜索按钮
     */
    private fun clickSearchButton(): Boolean {
        val rootNode = rootInActiveWindow ?: run {
            Log.w(TAG, "无法获取根节点")
            return false
        }
        
        Log.d(TAG, "开始查找搜索按钮... rootPackage=${rootNode.packageName}")
        
        // 方法1：优先通过ID查找（包括QQ音乐的searchItem）
        val searchIds = SEARCH_BUTTON_IDS + arrayOf("searchItem", "search_edit_text", "search_input")
        val searchNode = findNodeByIds(rootNode, searchIds)
        if (searchNode != null) {
            Log.d(TAG, "通过ID找到搜索入口: ${searchNode.viewIdResourceName}, clickable=${searchNode.isClickable}")
            val result = performClick(searchNode)
            Log.d(TAG, "点击搜索入口结果: $result")
            return true
        }
        
        // 方法2：通过ContentDescription查找搜索栏
        // QQ音乐搜索栏的ContentDescription格式为 "搜索 [推荐文本]"，长度会超过10
        // 策略：收集所有候选节点，优先选择屏幕顶部的（搜索栏在顶部，推荐卡片在中下部）
        val nodesByDesc = findAllNodesByContentDescription(rootNode, "搜索", "search")
        val searchCandidates = mutableListOf<Pair<AccessibilityNodeInfo, Int>>() // node, topY
        for (node in nodesByDesc) {
            val desc = node.contentDescription?.toString() ?: ""
            val viewId = node.viewIdResourceName ?: ""
            // 必须属于QQ音乐包名，排除其他应用的搜索节点
            if (viewId.isNotEmpty() && !viewId.startsWith(QQMUSIC_PACKAGE)) {
                Log.d(TAG, "跳过非QQ音乐搜索节点: id=$viewId, desc=$desc")
                continue
            }
            if (desc.contains("搜索", ignoreCase = true) || desc.contains("search", ignoreCase = true)) {
                val bounds = Rect()
                node.getBoundsInScreen(bounds)
                Log.d(TAG, "搜索候选: desc=${desc.take(30)}, bounds=$bounds, id=$viewId, clickable=${node.isClickable}")
                searchCandidates.add(Pair(node, bounds.top))
            }
        }
        
        if (searchCandidates.isNotEmpty()) {
            // 选择屏幕最顶部的搜索节点（搜索栏总是在页面顶部）
            val topCandidate = searchCandidates.minByOrNull { it.second }!!
            Log.d(TAG, "选择最顶部的搜索节点: top=${topCandidate.second}, desc=${topCandidate.first.contentDescription?.toString()?.take(30)}")
            performClick(topCandidate.first)
            return true
        }
        
        // 方法3：查找文本为"搜索"的节点
        val nodesByText = rootNode.findAccessibilityNodeInfosByText("搜索")
        for (node in nodesByText) {
            val text = node.text?.toString() ?: ""
            val viewId = node.viewIdResourceName ?: ""
            // 只接受QQ音乐的节点
            if (viewId.isNotEmpty() && !viewId.startsWith(QQMUSIC_PACKAGE)) continue
            if (text.contains("搜索") && (node.isClickable || text.length <= 5)) {
                Log.d(TAG, "通过文本找到搜索按钮: text=$text, id=$viewId")
                performClick(node)
                return true
            }
        }
        
        Log.w(TAG, "未找到合适的搜索按钮")
        printSearchRelatedNodes(rootNode, 0)
        return false
    }
    
    /**
     * 触发搜索（点击搜索框或搜索按钮）
     */
    private fun triggerSearch(): Boolean {
        val rootNode = rootInActiveWindow ?: run {
            Log.w(TAG, "无法获取根节点")
            return false
        }
        
        Log.d(TAG, "查找搜索触发按钮...")

        // 方法0：优先点击顶部搜索框（符合“点击搜索框”的要求）
        val searchEditIds = arrayOf(
            "search_edit", "edit_search", "search_input", "et_search",
            "search_box", "searchEdit", "editSearch", "search_text",
            "et_input", "input", "edit_text", "et_keyword"
        )
        var searchEditText: AccessibilityNodeInfo? = null
        for (id in searchEditIds) {
            val nodes = rootNode.findAccessibilityNodeInfosByViewId("$QQMUSIC_PACKAGE:id/$id")
            if (nodes.isNotEmpty()) {
                searchEditText = nodes[0]
                Log.d(TAG, "通过ViewId找到搜索框: $id")
                break
            }
        }

        if (searchEditText == null) {
            val allEditTexts = mutableListOf<AccessibilityNodeInfo>()
            findAllNodesByClassName(rootNode, allEditTexts, "android.widget.EditText", 0, 10)
            if (allEditTexts.isNotEmpty()) {
                searchEditText = allEditTexts.minByOrNull { node ->
                    val bounds = Rect()
                    node.getBoundsInScreen(bounds)
                    bounds.top
                }
                Log.d(TAG, "通过EditText列表找到顶部搜索框")
            }
        }

        if (searchEditText != null) {
            if (searchEditText.isClickable) {
                searchEditText.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            } else {
                performGestureClick(searchEditText)
            }
            searchEditText.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            Thread.sleep(200)

            // 尝试发送回车以触发搜索
            try {
                Runtime.getRuntime().exec("input keyevent KEYCODE_ENTER")
                Thread.sleep(200)
                return true
            } catch (e: Exception) {
                Log.e(TAG, "点击搜索框后发送回车失败", e)
            }
        }
        
        // 方法1：查找搜索按钮/图标（常见ID）
        val searchTriggerIds = arrayOf(
            "search_btn", "btn_search", "search_button", 
            "search_icon", "iv_search", "search_confirm",
            "btn_confirm", "search_enter"
        )
        
        for (id in searchTriggerIds) {
            val nodes = rootNode.findAccessibilityNodeInfosByViewId("$QQMUSIC_PACKAGE:id/$id")
            if (nodes.isNotEmpty()) {
                val node = nodes[0]
                Log.d(TAG, "通过ID找到搜索触发按钮: $id")
                if (performClick(node)) {
                    return true
                }
            }
        }
        
        // 方法2：查找可点击的搜索相关节点（图标、按钮）
        val searchNodes = mutableListOf<AccessibilityNodeInfo>()
        findNodesByClassName(rootNode, searchNodes, "android.widget.ImageView", 0, 5)
        findNodesByClassName(rootNode, searchNodes, "android.widget.ImageButton", 0, 5)
        findNodesByClassName(rootNode, searchNodes, "android.widget.Button", 0, 5)
        
        for (node in searchNodes) {
            val desc = node.contentDescription?.toString() ?: ""
            val viewId = node.viewIdResourceName ?: ""
            
            // 查找包含"搜索"或"search"的可点击节点
            if ((desc.contains("搜索", ignoreCase = true) || 
                 desc.contains("search", ignoreCase = true) ||
                 viewId.contains("search", ignoreCase = true)) && 
                node.isClickable) {
                Log.d(TAG, "找到搜索触发节点: desc=$desc, id=$viewId")
                if (performClick(node)) {
                    return true
                }
            }
        }
        
        // 方法3：发送回车键
        Log.d(TAG, "尝试发送回车键触发搜索...")
        try {
            Runtime.getRuntime().exec("input keyevent KEYCODE_ENTER")
            Thread.sleep(200)
            return true
        } catch (e: Exception) {
            Log.e(TAG, "发送回车键失败", e)
        }
        
        Log.w(TAG, "未找到搜索触发方式")
        return false
    }
    
    /**
     * 查找所有匹配ContentDescription的节点
     */
    private fun findAllNodesByContentDescription(
        root: AccessibilityNodeInfo,
        vararg descriptions: String
    ): List<AccessibilityNodeInfo> {
        val result = mutableListOf<AccessibilityNodeInfo>()
        for (desc in descriptions) {
            val nodes = root.findAccessibilityNodeInfosByText(desc)
            result.addAll(nodes)
        }
        return result
    }
    
    /**
     * 打印所有搜索相关节点（调试用）
     */
    private fun printSearchRelatedNodes(node: AccessibilityNodeInfo?, depth: Int) {
        if (node == null || depth > 10) return
        
        val indent = "  ".repeat(depth)
        val desc = node.contentDescription?.toString() ?: ""
        val text = node.text?.toString() ?: ""
        
        if (desc.contains("搜索", ignoreCase = true) || text.contains("搜索", ignoreCase = true)) {
            Log.d(TAG, "${indent}[搜索相关] class=${node.className}, text=$text, desc=$desc, " +
                    "clickable=${node.isClickable}, id=${node.viewIdResourceName}")
        }
        
        for (i in 0 until node.childCount) {
            printSearchRelatedNodes(node.getChild(i), depth + 1)
        }
    }
    
    /**
     * 输入搜索文本
     */
    private fun inputSearchText(text: String): Boolean {
        val rootNode = rootInActiveWindow ?: run {
            Log.w(TAG, "无法获取根节点")
            return false
        }
        
        Log.d(TAG, "▶▶▶ 开始查找搜索框 ◀◀◀")
        Log.d(TAG, "当前界面: ${rootNode.packageName}")
        
        // 等待界面稳定
        Thread.sleep(800)
        
        var editText: AccessibilityNodeInfo? = null
        
        // 策略1: 通过ViewId查找（最精确）
        Log.d(TAG, "策略1: 尝试通过ViewId查找搜索框...")
        val searchIds = arrayOf(
            "search_edit", "edit_search", "search_input", "et_search", 
            "search_box", "searchEdit", "editSearch", "search_text",
            "et_input", "input", "edit_text", "et_keyword"
        )
        for (id in searchIds) {
            val nodes = rootNode.findAccessibilityNodeInfosByViewId("$QQMUSIC_PACKAGE:id/$id")
            if (nodes.isNotEmpty()) {
                editText = nodes[0]
                Log.d(TAG, "✓ 通过ViewId找到搜索框: $id")
                break
            }
        }
        
        // 策略2: 查找所有EditText，选择最靠近顶部的
        if (editText == null) {
            Log.d(TAG, "策略2: 查找所有EditText...")
            val allEditTexts = mutableListOf<AccessibilityNodeInfo>()
            findAllNodesByClassName(rootNode, allEditTexts, "android.widget.EditText", 0, 10)
            
            if (allEditTexts.isNotEmpty()) {
                Log.d(TAG, "找到 ${allEditTexts.size} 个EditText")
                // 选择y坐标最小的（最顶部的）
                editText = allEditTexts.minByOrNull { node ->
                    val bounds = Rect()
                    node.getBoundsInScreen(bounds)
                    bounds.top
                }
                Log.d(TAG, "✓ 选择最顶部的EditText")
            }
        }
        
        // 策略3: 通过hint文本查找
        if (editText == null) {
            Log.d(TAG, "策略3: 尝试通过hint文本查找...")
            val hints = arrayOf("搜索", "Search", "搜索歌曲", "搜索音乐", "输入歌曲")
            for (hint in hints) {
                val nodes = rootNode.findAccessibilityNodeInfosByText(hint)
                for (node in nodes) {
                    if (node.className?.contains("EditText") == true || node.isEditable) {
                        editText = node
                        Log.d(TAG, "✓ 通过hint找到搜索框: $hint")
                        break
                    }
                }
                if (editText != null) break
            }
        }
        
        // 策略4: 查找可编辑节点
        if (editText == null) {
            Log.d(TAG, "策略4: 递归查找可编辑节点...")
            editText = findEditableNode(rootNode)
            if (editText != null) {
                Log.d(TAG, "✓ 找到可编辑节点")
            }
        }
        
        // 策略5: 查找有焦点的节点
        if (editText == null) {
            Log.d(TAG, "策略5: 查找有焦点的节点...")
            editText = findFocusedEditNode(rootNode)
            if (editText != null) {
                Log.d(TAG, "✓ 找到有焦点的输入节点")
            }
        }
        
        // 策略6: 查找屏幕顶部的可点击文本框
        if (editText == null) {
            Log.d(TAG, "策略6: 查找顶部区域的文本框...")
            editText = findTopTextBox(rootNode)
            if (editText != null) {
                Log.d(TAG, "✓ 找到顶部文本框")
            }
        }
        
        if (editText == null) {
            Log.e(TAG, "❌ 所有策略都未找到搜索框")
            // 打印界面结构帮助调试
            Log.d(TAG, "========== 界面结构分析 ==========")
            printDetailedNodeTree(rootNode, 0, 5)
            return false
        }
        
        // 找到搜索框，开始输入
        Log.d(TAG, "▶ 找到搜索框，准备输入文本: $text")
        Log.d(TAG, "  - 类名: ${editText.className}")
        Log.d(TAG, "  - ViewId: ${editText.viewIdResourceName}")
        Log.d(TAG, "  - Hint: ${editText.hintText}")
        Log.d(TAG, "  - 当前文本: ${editText.text}")
        Log.d(TAG, "  - 可编辑: ${editText.isEditable}")
        Log.d(TAG, "  - 可点击: ${editText.isClickable}")
        
        // 多种输入方法，按成功率排序
        
        // 方法1: 先清空再设置（最可靠）
        Log.d(TAG, "输入方法1: 清空后设置文本")
        if (tryInputMethod1(editText, text)) {
            return true
        }
        
        // 方法2: 点击后设置
        Log.d(TAG, "输入方法2: 点击后设置文本")
        if (tryInputMethod2(editText, text)) {
            return true
        }
        
        // 方法3: 剪贴板粘贴
        Log.d(TAG, "输入方法3: 剪贴板粘贴")
        if (tryInputMethod3(editText, text)) {
            return true
        }
        
        // 方法4: 逐字符输入（最后的手段）
        Log.d(TAG, "输入方法4: 逐字符输入")
        if (tryInputMethod4(editText, text)) {
            return true
        }
        
        Log.e(TAG, "❌ 所有输入方法都失败")
        return false
    }
    
    /**
     * 查找所有指定类名的节点
     */
    private fun findAllNodesByClassName(
        node: AccessibilityNodeInfo?,
        result: MutableList<AccessibilityNodeInfo>,
        className: String,
        depth: Int,
        maxDepth: Int
    ) {
        if (node == null || depth > maxDepth) return
        
        if (node.className?.toString() == className) {
            result.add(node)
        }
        
        for (i in 0 until node.childCount) {
            findAllNodesByClassName(node.getChild(i), result, className, depth + 1, maxDepth)
        }
    }
    
    /**
     * 查找有焦点的可编辑节点
     */
    private fun findFocusedEditNode(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (node == null) return null
        
        if ((node.isFocused || node.isAccessibilityFocused) && 
            (node.isEditable || node.className?.contains("EditText") == true)) {
            return node
        }
        
        for (i in 0 until node.childCount) {
            val result = findFocusedEditNode(node.getChild(i))
            if (result != null) return result
        }
        
        return null
    }
    
    /**
     * 查找顶部区域的文本框
     */
    private fun findTopTextBox(root: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        val rootBounds = Rect()
        root.getBoundsInScreen(rootBounds)
        val topThreshold = rootBounds.height() * 0.3f  // 顶部30%区域
        
        val candidates = mutableListOf<AccessibilityNodeInfo>()
        findAllEditableInRegion(root, candidates, topThreshold.toInt(), 0, 8)
        
        // 返回最顶部的
        return candidates.minByOrNull { node ->
            val bounds = Rect()
            node.getBoundsInScreen(bounds)
            bounds.top
        }
    }
    
    /**
     * 查找指定区域的可编辑节点
     */
    private fun findAllEditableInRegion(
        node: AccessibilityNodeInfo?,
        result: MutableList<AccessibilityNodeInfo>,
        maxY: Int,
        depth: Int,
        maxDepth: Int
    ) {
        if (node == null || depth > maxDepth) return
        
        val bounds = Rect()
        node.getBoundsInScreen(bounds)
        
        if (bounds.top < maxY && (node.isEditable || node.className?.contains("EditText") == true)) {
            result.add(node)
        }
        
        for (i in 0 until node.childCount) {
            findAllEditableInRegion(node.getChild(i), result, maxY, depth + 1, maxDepth)
        }
    }
    
    /**
     * 输入方法1: 清空后设置
     */
    private fun tryInputMethod1(editText: AccessibilityNodeInfo, text: String): Boolean {
        return try {
            // 聚焦
            editText.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            Thread.sleep(200)
            
            // 先清空
            val clearArgs = android.os.Bundle()
            clearArgs.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "")
            editText.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, clearArgs)
            Thread.sleep(100)
            
            // 再设置新文本
            val setArgs = android.os.Bundle()
            setArgs.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
            val success = editText.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, setArgs)
            
            if (success) {
                Thread.sleep(300)
                editText.refresh()
                val currentText = editText.text?.toString() ?: ""
                if (currentText.contains(text)) {
                    Log.d(TAG, "✓ 方法1成功: $currentText")
                    return true
                }
            }
            false
        } catch (e: Exception) {
            Log.e(TAG, "方法1异常", e)
            false
        }
    }
    
    /**
     * 输入方法2: 点击后设置
     */
    private fun tryInputMethod2(editText: AccessibilityNodeInfo, text: String): Boolean {
        return try {
            // 点击激活
            if (editText.isClickable) {
                editText.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            } else {
                performGestureClick(editText)
            }
            Thread.sleep(400)
            
            // 聚焦
            editText.refresh()
            editText.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            Thread.sleep(200)
            
            // 设置文本
            val args = android.os.Bundle()
            args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
            val success = editText.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
            
            if (success) {
                Thread.sleep(300)
                editText.refresh()
                val currentText = editText.text?.toString() ?: ""
                if (currentText.contains(text)) {
                    Log.d(TAG, "✓ 方法2成功: $currentText")
                    return true
                }
            }
            false
        } catch (e: Exception) {
            Log.e(TAG, "方法2异常", e)
            false
        }
    }
    
    /**
     * 输入方法3: 剪贴板粘贴
     */
    private fun tryInputMethod3(editText: AccessibilityNodeInfo, text: String): Boolean {
        return try {
            // 复制到剪贴板
            val clipboard = getSystemService(android.content.Context.CLIPBOARD_SERVICE) 
                as android.content.ClipboardManager
            val clip = android.content.ClipData.newPlainText("search", text)
            clipboard.setPrimaryClip(clip)
            Thread.sleep(100)
            
            // 点击并聚焦
            if (editText.isClickable) {
                editText.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            } else {
                performGestureClick(editText)
            }
            Thread.sleep(300)
            
            editText.refresh()
            editText.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            Thread.sleep(200)
            
            // 先清空 (ACTION_SELECT_ALL = 0x00020000)
            editText.performAction(0x00020000)
            Thread.sleep(100)
            
            // 粘贴
            val success = editText.performAction(AccessibilityNodeInfo.ACTION_PASTE)
            
            if (success) {
                Thread.sleep(300)
                editText.refresh()
                val currentText = editText.text?.toString() ?: ""
                if (currentText.contains(text)) {
                    Log.d(TAG, "✓ 方法3成功: $currentText")
                    return true
                }
            }
            false
        } catch (e: Exception) {
            Log.e(TAG, "方法3异常", e)
            false
        }
    }
    
    /**
     * 输入方法4: 逐字符输入（使用输入法）
     */
    private fun tryInputMethod4(editText: AccessibilityNodeInfo, text: String): Boolean {
        return try {
            // 点击聚焦
            performGestureClick(editText)
            Thread.sleep(500)
            
            // 使用adb输入（需要root或调试权限）
            Runtime.getRuntime().exec("input text ${text.replace(" ", "%s")}")
            Thread.sleep(500)
            
            editText.refresh()
            val currentText = editText.text?.toString() ?: ""
            if (currentText.contains(text.substring(0, minOf(3, text.length)))) {
                Log.d(TAG, "✓ 方法4成功: $currentText")
                return true
            }
            false
        } catch (e: Exception) {
            Log.e(TAG, "方法4异常", e)
            false
        }
    }
    
    /**
     * 打印详细的节点树（用于调试）
     */
    private fun printDetailedNodeTree(node: AccessibilityNodeInfo?, depth: Int, maxDepth: Int) {
        if (node == null || depth > maxDepth) return
        
        val indent = "  ".repeat(depth)
        val bounds = Rect()
        node.getBoundsInScreen(bounds)
        
        if (node.isEditable || node.className?.contains("Edit") == true || 
            node.className?.contains("Input") == true) {
            Log.d(TAG, "${indent}[${node.className}]")
            Log.d(TAG, "${indent}  ID: ${node.viewIdResourceName}")
            Log.d(TAG, "${indent}  Text: ${node.text}")
            Log.d(TAG, "${indent}  Hint: ${node.hintText}")
            Log.d(TAG, "${indent}  Editable: ${node.isEditable}")
            Log.d(TAG, "${indent}  Bounds: $bounds")
        }
        
        for (i in 0 until node.childCount) {
            printDetailedNodeTree(node.getChild(i), depth + 1, maxDepth)
        }
    }
    
    /**
     * 查找可编辑节点
     */
    private fun findEditableNode(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (node == null) return null
        
        if (node.isEditable || node.className?.contains("EditText") == true) {
            return node
        }
        
        for (i in 0 until node.childCount) {
            val result = findEditableNode(node.getChild(i))
            if (result != null) return result
        }
        
        return null
    }
    
    /**
     * 查找EditText附近的搜索按钮
     */
    private fun findSearchButtonNearEditText(
        root: AccessibilityNodeInfo,
        editText: AccessibilityNodeInfo
    ): AccessibilityNodeInfo? {
        // 查找EditText的父节点
        var parent = editText.parent
        var depth = 0
        
        while (parent != null && depth < 3) {
            // 在父节点的子节点中查找搜索按钮
            for (i in 0 until parent.childCount) {
                val child = parent.getChild(i) ?: continue
                val desc = child.contentDescription?.toString() ?: ""
                val text = child.text?.toString() ?: ""
                
                if ((desc.contains("搜索") || text.contains("搜索")) && child.isClickable) {
                    Log.d(TAG, "找到EditText附近的搜索按钮: desc=$desc, text=$text")
                    return child
                }
            }
            
            parent = parent.parent
            depth++
        }
        
        return null
    }
    
    /**
     * 打印可交互的节点（用于调试）
     */
    private fun printInteractiveNodes(node: AccessibilityNodeInfo?, depth: Int) {
        if (node == null || depth > 3) return
        
        val indent = "  ".repeat(depth)
        if (node.isClickable || node.isFocusable || node.isEditable || node.className?.contains("Edit") == true) {
            Log.d(TAG, "${indent}节点: class=${node.className}, text=${node.text}, desc=${node.contentDescription}, clickable=${node.isClickable}, editable=${node.isEditable}")
        }
        
        for (i in 0 until node.childCount) {
            printInteractiveNodes(node.getChild(i), depth + 1)
        }
    }
    
    /**
     * 点击第一个搜索结果
     * 先切换到"歌曲"Tab（纯歌曲列表，无可变内容），然后点击第一首歌
     */
    private fun clickFirstSearchResult(): Boolean {
        val screenHeight = resources.displayMetrics.heightPixels
        val screenWidth = resources.displayMetrics.widthPixels
        Log.d(TAG, "屏幕尺寸: ${screenWidth}x${screenHeight}")
        
        // 智能等待搜索结果加载（轮询检测TabBar出现，最长2000ms）
        Log.d(TAG, "等待搜索结果加载...")
        var rootNode: AccessibilityNodeInfo? = null
        val searchWaitStart = System.currentTimeMillis()
        val maxSearchWait = 2000L
        while (System.currentTimeMillis() - searchWaitStart < maxSearchWait) {
            Thread.sleep(200)
            rootNode = rootInActiveWindow
            if (rootNode != null) {
                // 检测搜索结果TabBar是否已出现（HorizontalScrollView id=ba9）
                val tabBar = rootNode.findAccessibilityNodeInfosByViewId("com.tencent.qqmusic:id/ba9")
                if (!tabBar.isNullOrEmpty()) {
                    Log.d(TAG, "搜索结果已加载 (${System.currentTimeMillis() - searchWaitStart}ms)")
                    break
                }
            }
        }
        if (rootNode == null) rootNode = rootInActiveWindow
        
        // 策略1（优先）: 点击顶部"歌曲"Tab切换到纯歌曲列表，再点击第一首歌
        // "歌曲"Tab在顶部固定TabBar中（HorizontalScrollView id=ba9, y≈257-341），
        // 切换后歌曲列表紧跟TabBar下方，无歌手卡片/推荐歌单等可变内容
        Log.d(TAG, "策略1: 切换到歌曲Tab后点击第一首歌...")
        if (switchToSongTabAndClickFirst(rootNode, screenWidth, screenHeight)) {
            return true
        }
        
        // 策略2: 在当前Tab（综合）内找"单曲"标签定位，在其正下方点击第一条歌曲
        Log.d(TAG, "策略2: 通过单曲标签定位歌曲结果...")
        if (clickFirstSongBelowTab(rootNode, screenWidth, screenHeight)) {
            return true
        }
        
        // 策略3: 通过 item_layout 资源ID找到歌曲条目，跳过歌手栏
        Log.d(TAG, "策略3: 通过 item_layout 资源ID定位...")
        if (clickFirstSongByItemLayout(rootNode, screenWidth, screenHeight)) {
            return true
        }
        
        // 策略4: 直接坐标点击 - 基于UI分析，第一首歌在屏幕约33%位置
        Log.d(TAG, "策略4: 直接坐标点击...")
        return clickAtFixedSongPosition(screenWidth, screenHeight)
    }
    
    /**
     * 策略1: 切换到"歌曲"Tab，然后点击第一首歌
     * 顶部TabBar结构: HorizontalScrollView (id=ba9, y≈257-341)
     *   -> FrameLayout -> 多个RelativeLayout (desc="综合 按钮", "歌曲 按钮", ...)
     * "歌曲"Tab切换后只有纯歌曲列表，歌曲紧跟TabBar下方，位置固定可靠
     */
    private fun switchToSongTabAndClickFirst(rootNode: AccessibilityNodeInfo?, screenWidth: Int, screenHeight: Int): Boolean {
        if (rootNode == null) return false
        
        // 查找"歌曲"Tab节点
        // 方法1: 通过findAccessibilityNodeInfosByText查找
        val songTabNodes = rootNode.findAccessibilityNodeInfosByText("歌曲")
        var songTab: AccessibilityNodeInfo? = null
        var tabBarBottom = -1
        
        if (!songTabNodes.isNullOrEmpty()) {
            for (node in songTabNodes) {
                val desc = node.contentDescription?.toString() ?: ""
                val bounds = Rect()
                node.getBoundsInScreen(bounds)
                Log.d(TAG, "  歌曲候选节点: desc='$desc', bounds=$bounds, class=${node.className}, clickable=${node.isClickable}")
                
                // 排除搜索框占位文本（desc包含"搜索"的是搜索栏，不是Tab按钮）
                if (desc.contains("搜索")) {
                    Log.d(TAG, "  ✗ 跳过：搜索栏占位文本")
                    continue
                }
                
                // 顶部TabBar中的"歌曲"Tab特征:
                // 1. 在屏幕上方 (y < 500)
                // 2. 高度较小 (< 120px) — 是Tab按钮
                // 3. content-desc包含"歌曲" 或 文本包含"歌曲" 且 不包含"单曲"
                if (bounds.top < 500 && bounds.height() < 120) {
                    // 优先选content-desc精确匹配的
                    if (desc.contains("歌曲")) {
                        songTab = node
                        tabBarBottom = bounds.bottom
                        Log.d(TAG, "  ✓ 通过contentDesc匹配到歌曲Tab: bounds=$bounds")
                        break
                    }
                    // 文本匹配但排除"单曲"
                    val text = node.text?.toString() ?: ""
                    if (text == "歌曲" && !text.contains("单曲")) {
                        songTab = node
                        tabBarBottom = bounds.bottom
                        Log.d(TAG, "  ✓ 通过文本匹配到歌曲Tab: bounds=$bounds")
                    }
                }
            }
        }
        
        // 方法2: 如果没找到，尝试遍历HorizontalScrollView的子节点
        if (songTab == null) {
            Log.d(TAG, "  文本搜索未找到歌曲Tab，尝试遍历TabBar子节点...")
            songTab = findSongTabInScrollView(rootNode, screenWidth)
            if (songTab != null) {
                val bounds = Rect()
                songTab.getBoundsInScreen(bounds)
                tabBarBottom = bounds.bottom
            }
        }
        
        if (songTab == null) {
            Log.d(TAG, "  未找到歌曲Tab")
            return false
        }
        
        // 点击"歌曲"Tab
        Log.d(TAG, "  点击歌曲Tab...")
        var clicked = false
        
        // 尝试ACTION_CLICK
        if (songTab.isClickable) {
            clicked = songTab.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            Log.d(TAG, "  歌曲Tab ACTION_CLICK: $clicked")
        }
        
        // 尝试点击父节点
        if (!clicked) {
            val parent = songTab.parent
            if (parent != null && parent.isClickable) {
                clicked = parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                Log.d(TAG, "  歌曲Tab 父节点 ACTION_CLICK: $clicked")
                // 注意：不用父节点的bounds更新tabBarBottom
                // 父节点可能是整个页面容器，其bottom远大于Tab实际位置
                // tabBarBottom 保持从匹配到的歌曲Tab节点获取的值
            }
        }
        
        // 手势点击
        if (!clicked) {
            val tabBounds = Rect()
            songTab.getBoundsInScreen(tabBounds)
            val tapX = tabBounds.centerX().toFloat()
            val tapY = tabBounds.centerY().toFloat()
            Log.d(TAG, "  手势点击歌曲Tab: ($tapX, $tapY)")
            clicked = tapScreenPosition(tapX, tapY)
        }
        
        if (!clicked) {
            Log.d(TAG, "  无法点击歌曲Tab")
            return false
        }
        
        // 智能等待Tab切换和内容加载（轮询检测，最长1000ms）
        Log.d(TAG, "  等待歌曲Tab内容加载...")
        var newRoot: AccessibilityNodeInfo? = null
        val tabWaitStart = System.currentTimeMillis()
        val maxTabWait = 1000L
        while (System.currentTimeMillis() - tabWaitStart < maxTabWait) {
            Thread.sleep(150)
            newRoot = rootInActiveWindow
            if (newRoot != null) {
                // Tab切换后UI会刷新，检测RecyclerView内容是否就绪
                val recyclers = mutableListOf<AccessibilityNodeInfo>()
                findNodesByResourceId(newRoot, recyclers, "com.tencent.qqmusic:id/bb2", 0, 5)
                if (recyclers.any { node ->
                    val b = Rect()
                    node.getBoundsInScreen(b)
                    b.width() > 0 && b.height() > 100
                }) {
                    Log.d(TAG, "  歌曲Tab内容已加载 (${System.currentTimeMillis() - tabWaitStart}ms)")
                    break
                }
            }
        }
        if (newRoot == null) newRoot = rootInActiveWindow
        
        // 确定TabBar底部位置（切换后重新获取更准确的值）
        if (tabBarBottom <= 0) {
            tabBarBottom = (screenHeight * 0.13).toInt() // 默认值约341px
        }
        
        // 在"歌曲"Tab中，歌曲列表紧跟TabBar下方
        // 先尝试通过item_layout找第一首歌
        if (newRoot != null) {
            val itemLayouts = mutableListOf<AccessibilityNodeInfo>()
            findNodesByResourceId(newRoot, itemLayouts, "com.tencent.qqmusic:id/item_layout", 0, 15)
            Log.d(TAG, "  歌曲Tab中找到 ${itemLayouts.size} 个 item_layout")
            
            if (itemLayouts.isNotEmpty()) {
                val sorted = itemLayouts
                    .map { node ->
                        val bounds = Rect()
                        node.getBoundsInScreen(bounds)
                        Pair(node, bounds)
                    }
                    .filter { (_, bounds) -> bounds.top >= tabBarBottom - 20 }
                    .sortedBy { (_, bounds) -> bounds.top }
                
                if (sorted.isNotEmpty()) {
                    val (node, bounds) = sorted[0]
                    val clickX = bounds.centerX().toFloat()
                    val clickY = bounds.centerY().toFloat()
                    Log.d(TAG, "  点击歌曲Tab中的第一首歌: ($clickX, $clickY) bounds=$bounds")
                    if (tapScreenPosition(clickX, clickY)) {
                        Thread.sleep(300)
                        Log.d(TAG, "  ✓ 策略1成功！已点击歌曲Tab中的第一首歌")
                        return true
                    }
                }
            }
        }
        
        // item_layout未找到时，直接在TabBar下方145px处点击（歌曲Tab中此位置固定可靠）
        val clickX = screenWidth / 2f
        val clickY = tabBarBottom + 145f
        Log.d(TAG, "  在TabBar下方点击第一首歌: ($clickX, $clickY) [TabBar底部=$tabBarBottom + 145px]")
        if (tapScreenPosition(clickX, clickY)) {
            Thread.sleep(300)
            Log.d(TAG, "  ✓ 策略1成功！歌曲Tab坐标点击完成")
            return true
        }
        
        return false
    }
    
    /**
     * 在HorizontalScrollView中查找"歌曲"Tab
     * 遍历TabBar的子节点，通过contentDescription或文本匹配
     */
    private fun findSongTabInScrollView(rootNode: AccessibilityNodeInfo, screenWidth: Int): AccessibilityNodeInfo? {
        // 查找HorizontalScrollView（顶部TabBar容器）
        val scrollViews = mutableListOf<AccessibilityNodeInfo>()
        findAllNodesByClassName(rootNode, scrollViews, "android.widget.HorizontalScrollView", 0, 8)
        
        for (sv in scrollViews) {
            val svBounds = Rect()
            sv.getBoundsInScreen(svBounds)
            // TabBar特征: 在屏幕顶部 (y < 500)，宽度接近全屏
            if (svBounds.top < 500 && svBounds.width() > screenWidth * 0.8) {
                Log.d(TAG, "  找到顶部TabBar: bounds=$svBounds")
                // 遍历其所有后代节点
                val candidate = findSongTabRecursive(sv)
                if (candidate != null) return candidate
            }
        }
        return null
    }
    
    /**
     * 递归查找contentDescription或文本包含"歌曲"的节点
     */
    private fun findSongTabRecursive(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (node == null) return null
        val desc = node.contentDescription?.toString() ?: ""
        val text = node.text?.toString() ?: ""
        if (desc.contains("歌曲") && !desc.contains("单曲")) {
            return node
        }
        if (text == "歌曲") {
            return node
        }
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            val found = findSongTabRecursive(child)
            if (found != null) return found
        }
        return null
    }
    
    /**
     * 策略2: 找到"单曲"Tab标签，在其下方点击第一条歌曲（综合Tab内的备选方案）
     * 注意: 单曲标签位置会随上方内容变化，不够稳定，优先使用策略1(歌曲Tab)
     */
    private fun clickFirstSongBelowTab(rootNode: AccessibilityNodeInfo?, screenWidth: Int, screenHeight: Int): Boolean {
        if (rootNode == null) return false
        
        // 查找"单曲"文本节点
        val tabNodes = rootNode.findAccessibilityNodeInfosByText("单曲")
        if (tabNodes.isNullOrEmpty()) {
            Log.d(TAG, "  未找到'单曲'标签")
            return false
        }
        
        // 找高度较小的那个"单曲"节点（Tab标签，不是歌单内的"单曲"文字）
        var tabBottom = -1
        for (node in tabNodes) {
            val bounds = Rect()
            node.getBoundsInScreen(bounds)
            // Tab标签特征: 高度小于100px，在屏幕上半部
            if (bounds.height() < 100 && bounds.top < screenHeight * 0.4) {
                Log.d(TAG, "  找到单曲Tab: bounds=$bounds")
                // 向上找父容器，获取整个Tab栏的底部
                var parentBottom = bounds.bottom
                var parent = node.parent
                if (parent != null) {
                    val parentBounds = Rect()
                    parent.getBoundsInScreen(parentBounds)
                    // 父容器应该是Tab行，宽度接近屏幕宽
                    if (parentBounds.width() > screenWidth * 0.5 && parentBounds.height() < screenHeight * 0.15) {
                        parentBottom = parentBounds.bottom
                        Log.d(TAG, "  Tab栏容器: bounds=$parentBounds")
                        // 再找上一层到整个tab bar区域
                        val grandParent = parent.parent
                        if (grandParent != null) {
                            val gpBounds = Rect()
                            grandParent.getBoundsInScreen(gpBounds)
                            if (gpBounds.width() > screenWidth * 0.8 && gpBounds.height() < screenHeight * 0.15) {
                                parentBottom = gpBounds.bottom
                                Log.d(TAG, "  Tab栏外层容器: bounds=$gpBounds")
                            }
                        }
                    }
                }
                tabBottom = maxOf(tabBottom, parentBottom)
            }
        }
        
        if (tabBottom <= 0) {
            Log.d(TAG, "  无法确定Tab栏底部位置")
            return false
        }
        
        // 在Tab栏下方点击第一首歌的中心位置
        // 根据UI分析：每首歌高度约291px(773-1064)，中心在Tab底部下方约145px
        val clickX = screenWidth / 2f
        val clickY = tabBottom + 145f
        Log.d(TAG, "  点击第一首歌: ($clickX, $clickY) [Tab底部=$tabBottom + 145px]")
        
        if (tapScreenPosition(clickX, clickY)) {
            Thread.sleep(1500)
            if (checkPlayingStarted()) {
                Log.d(TAG, "  ✓ 策略1成功！歌曲已播放")
                return true
            }
            // 即使没检测到播放控件变化，点击位置大概率正确
            Log.d(TAG, "  点击已执行，虽未检测到播放控件变化")
            return true
        }
        return false
    }
    
    /**
     * 策略3: 通过 item_layout 资源ID 找到歌曲条目
     */
    private fun clickFirstSongByItemLayout(rootNode: AccessibilityNodeInfo?, screenWidth: Int, screenHeight: Int): Boolean {
        if (rootNode == null) return false
        
        // 先找"单曲"Tab的位置作为参考线
        var tabBarY = screenHeight * 0.25 // 默认值
        val tabNodes = rootNode.findAccessibilityNodeInfosByText("单曲")
        if (!tabNodes.isNullOrEmpty()) {
            for (node in tabNodes) {
                val bounds = Rect()
                node.getBoundsInScreen(bounds)
                if (bounds.height() < 100 && bounds.top < screenHeight * 0.4) {
                    tabBarY = bounds.bottom.toDouble()
                    break
                }
            }
        }
        
        // 查找所有 item_layout 节点
        val itemLayouts = mutableListOf<AccessibilityNodeInfo>()
        findNodesByResourceId(rootNode, itemLayouts, "com.tencent.qqmusic:id/item_layout", 0, 15)
        
        Log.d(TAG, "  找到 ${itemLayouts.size} 个 item_layout 节点")
        
        // 按Y位置排序，选择Tab栏下方的第一个
        val candidates = itemLayouts
            .map { node ->
                val bounds = Rect()
                node.getBoundsInScreen(bounds)
                Pair(node, bounds)
            }
            .filter { (_, bounds) -> bounds.top >= tabBarY - 10 }
            .sortedBy { (_, bounds) -> bounds.top }
        
        Log.d(TAG, "  Tab栏下方有 ${candidates.size} 个候选歌曲")
        
        if (candidates.isNotEmpty()) {
            val (_, firstBounds) = candidates[0]
            val clickX = firstBounds.centerX().toFloat()
            val clickY = firstBounds.centerY().toFloat()
            Log.d(TAG, "  点击第一个歌曲item: center=($clickX, $clickY) bounds=$firstBounds")
            
            // 先尝试点击其可点击的父节点
            val firstNode = candidates[0].first
            var clickTarget: AccessibilityNodeInfo? = firstNode
            // item_layout的父节点通常是可点击的LinearLayout
            val parent = firstNode.parent
            if (parent != null && parent.isClickable) {
                clickTarget = parent
                Log.d(TAG, "  使用父节点 ACTION_CLICK: ${parent.className}")
                if (parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                    Log.d(TAG, "  ✓ 父节点 ACTION_CLICK 成功")
                    return true
                }
            }
            
            // ACTION_CLICK 失败，用手势点击
            if (tapScreenPosition(clickX, clickY)) {
                Thread.sleep(1500)
                Log.d(TAG, "  ✓ 手势点击歌曲 item 完成")
                return true
            }
        }
        
        return false
    }
    
    /**
     * 通过 resource-id 查找节点
     */
    private fun findNodesByResourceId(node: AccessibilityNodeInfo?, result: MutableList<AccessibilityNodeInfo>, id: String, depth: Int, maxDepth: Int) {
        if (node == null || depth > maxDepth) return
        if (node.viewIdResourceName == id) {
            result.add(node)
        }
        for (i in 0 until node.childCount) {
            findNodesByResourceId(node.getChild(i), result, id, depth + 1, maxDepth)
        }
    }
    
    /**
     * 策略4: 直接在固定坐标位置点击
     */
    private fun clickAtFixedSongPosition(screenWidth: Int, screenHeight: Int): Boolean {
        // 基于实际UI分析的位置: 第一首歌中心约在屏幕33.8%处
        val positions = arrayOf(
            Pair(0.50f, 0.34f),  // 第一首歌中心
            Pair(0.50f, 0.40f),  // 稍下
            Pair(0.35f, 0.34f),  // 偏左
            Pair(0.35f, 0.40f),  // 偏左+稍下
        )
        
        for ((index, pos) in positions.withIndex()) {
            val tapX = screenWidth * pos.first
            val tapY = screenHeight * pos.second
            Log.d(TAG, "  固定坐标点击${index+1}: ($tapX, $tapY)")
            
            if (tapScreenPosition(tapX, tapY)) {
                Thread.sleep(1500)
                if (checkPlayingStarted()) {
                    Log.d(TAG, "  ✓ 固定坐标点击成功")
                    return true
                }
            }
        }
        
        Log.d(TAG, "  固定坐标点击完成（大概率已生效）")
        return true
    }
    
    /**
     * 检查是否已开始播放
     */
    private fun checkPlayingStarted(): Boolean {
        val currentRoot = rootInActiveWindow ?: return false
        // 检查搜索框是否消失（说明进入了播放页）
        val editTexts = mutableListOf<AccessibilityNodeInfo>()
        findAllNodesByClassName(currentRoot, editTexts, "android.widget.EditText", 0, 10)
        if (editTexts.isEmpty()) {
            return true
        }
        // 检查播放控件
        val pauseNodes = currentRoot.findAccessibilityNodeInfosByText("暂停")
        if (pauseNodes.isNotEmpty()) return true
        return false
    }
    
    /**
     * 尝试通过无障碍节点的 ACTION_CLICK 点击搜索结果
     */
    private fun tryClickResultByNode(rootNode: AccessibilityNodeInfo, screenWidth: Int, screenHeight: Int): Boolean {
        // 查找列表容器
        val listContainers = mutableListOf<AccessibilityNodeInfo>()
        findAllNodesByClassName(rootNode, listContainers, "androidx.recyclerview.widget.RecyclerView", 0, 15)
        findAllNodesByClassName(rootNode, listContainers, "android.widget.ListView", 0, 15)
        
        val sortedContainers = listContainers
            .map { node ->
                val bounds = Rect()
                node.getBoundsInScreen(bounds)
                Pair(node, bounds)
            }
            .filter { (_, bounds) ->
                bounds.top > screenHeight * 0.08 &&
                bounds.height() > screenHeight * 0.15
            }
            .sortedByDescending { (_, bounds) -> bounds.width() * bounds.height() }
        
        Log.d(TAG, "找到 ${sortedContainers.size} 个列表容器")
        
        for ((listNode, listBounds) in sortedContainers) {
            var targetList = listNode
            if (listNode.childCount <= 2) {
                val inner = findInnerRecyclerView(listNode)
                if (inner != null) targetList = inner
            }
            
            for (i in 0 until minOf(targetList.childCount, 10)) {
                val child = targetList.getChild(i) ?: continue
                val childBounds = Rect()
                child.getBoundsInScreen(childBounds)
                
                if (childBounds.height() < 50) continue
                if (childBounds.height() < 100 && childBounds.width() > screenWidth * 0.7) {
                    val txt = getAllText(child)
                    if (txt.contains("单曲") || txt.contains("专辑") ||
                        txt.contains("歌单") || txt.contains("MV")) continue
                }
                if (!hasTextContent(child)) continue
                
                Log.d(TAG, "  子项$i: h=${childBounds.height()}, text=${getAllText(child).take(40)}")
                
                // 只尝试 ACTION_CLICK（最直接有效的方式）
                if (child.isClickable && child.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                    Log.d(TAG, "  ✓ 子项$i ACTION_CLICK 成功")
                    Thread.sleep(500)
                    return true
                }
                val clickable = findClickableChild(child)
                if (clickable != null && clickable.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                    Log.d(TAG, "  ✓ 子项$i 内部节点 ACTION_CLICK 成功")
                    Thread.sleep(500)
                    return true
                }
                // 不在这里使用手势点击，留给坐标点击方法处理
            }
        }
        
        Log.d(TAG, "节点 ACTION_CLICK 未成功")
        return false
    }
    
    /**
     * 通过屏幕坐标多点点击搜索结果
     * QQ音乐搜索结果页布局：顶部搜索栏~10%，标签栏~15-20%，结果列表~20%-85%
     * 在屏幕中间区域多个Y位置尝试点击，每次点击同步等待完成
     */
    private fun clickSearchResultByCoordinates(screenWidth: Int, screenHeight: Int): Boolean {
        Log.d(TAG, "===== 开始多点坐标点击 =====")
        
        // 策略1: 智能定位 - 找到歌手栏/标签栏的底部位置，在其下方点击歌曲结果
        Log.d(TAG, "  策略1: 智能定位歌手栏下方...")
        if (clickBelowArtistBar(screenWidth, screenHeight)) {
            return true
        }
        
        // 策略2: 先滚动列表把歌手栏滑走，再坐标点击
        Log.d(TAG, "  策略2: 先滚动列表跳过歌手栏...")
        scrollSearchResultList()
        Thread.sleep(1000)
        
        // 滚动后再次尝试智能定位
        Log.d(TAG, "  滚动后再次智能定位...")
        if (clickBelowArtistBar(screenWidth, screenHeight)) {
            return true
        }
        
        // 策略3: 多点坐标盲扫
        Log.d(TAG, "  策略3: 多点坐标盲扫...")
        data class TapPoint(val xRatio: Float, val yRatio: Float, val desc: String)
        
        val tapPoints = listOf(
            // 滑动后搜索结果应该在屏幕中上部，从25%开始覆盖
            TapPoint(0.40f, 0.25f, "居中-25%"),
            TapPoint(0.40f, 0.35f, "居中-35%"),
            TapPoint(0.40f, 0.45f, "居中-45%"),
            TapPoint(0.40f, 0.55f, "居中-55%"),
            TapPoint(0.40f, 0.65f, "居中-65%"),
            // 偏左点击
            TapPoint(0.25f, 0.30f, "左侧-30%"),
            TapPoint(0.25f, 0.40f, "左侧-40%"),
            TapPoint(0.25f, 0.50f, "左侧-50%"),
            TapPoint(0.25f, 0.60f, "左侧-60%"),
        )
        
        for ((index, point) in tapPoints.withIndex()) {
            val tapX = screenWidth * point.xRatio
            val tapY = screenHeight * point.yRatio
            Log.d(TAG, "  坐标点击${index+1}/${tapPoints.size} [${point.desc}]: ($tapX, $tapY)")
            
            val success = tapScreenPosition(tapX, tapY)
            Log.d(TAG, "  手势结果: $success")
            
            if (success) {
                // 等待UI响应
                Thread.sleep(1200)
                
                // 检查是否触发了播放
                val currentRoot = rootInActiveWindow
                if (currentRoot != null) {
                    // 检查是否出现了播放相关控件
                    val pauseNodes = currentRoot.findAccessibilityNodeInfosByText("暂停")
                    val progressNodes = mutableListOf<AccessibilityNodeInfo>()
                    findAllNodesByClassName(currentRoot, progressNodes, "android.widget.SeekBar", 0, 10)
                    
                    if (pauseNodes.isNotEmpty() || progressNodes.isNotEmpty()) {
                        Log.d(TAG, "  ✓ 检测到播放控件，歌曲已开始播放！")
                        return true
                    }
                    
                    // 检测界面是否跳转（搜索输入框消失了 = 进入了播放页或歌曲详情）
                    val editTexts = mutableListOf<AccessibilityNodeInfo>()
                    findAllNodesByClassName(currentRoot, editTexts, "android.widget.EditText", 0, 10)
                    if (editTexts.isEmpty()) {
                        Log.d(TAG, "  ✓ 搜索框已消失，可能已进入播放页")
                        return true
                    }
                }
                Log.d(TAG, "  点击${index+1}后未检测到变化，继续...")
            }
        }
        
        Log.d(TAG, "  多点坐标点击全部完成")
        return true // 坐标点击大概率已生效
    }
    
    /**
     * 在容器节点内查找内层 RecyclerView
     */
    private fun findInnerRecyclerView(node: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            val className = child.className?.toString() ?: ""
            if (className.contains("RecyclerView") || className.contains("ListView")) {
                return child
            }
            // 再往下找一层
            for (j in 0 until child.childCount) {
                val grandChild = child.getChild(j) ?: continue
                val gcClassName = grandChild.className?.toString() ?: ""
                if (gcClassName.contains("RecyclerView") || gcClassName.contains("ListView")) {
                    return grandChild
                }
            }
        }
        return null
    }
    
    /**
     * 递归获取节点及其子节点的所有文本内容
     */
    private fun getAllText(node: AccessibilityNodeInfo?, depth: Int = 0): String {
        if (node == null || depth > 4) return ""
        val sb = StringBuilder()
        if (!node.text.isNullOrEmpty()) sb.append(node.text).append(" ")
        if (!node.contentDescription.isNullOrEmpty()) sb.append(node.contentDescription).append(" ")
        for (i in 0 until node.childCount) {
            sb.append(getAllText(node.getChild(i), depth + 1))
        }
        return sb.toString().trim()
    }
    
    /**
     * 直接点击屏幕上的指定坐标位置（同步等待完成）
     */
    private fun tapScreenPosition(x: Float, y: Float): Boolean {
        return try {
            dispatchGestureSync(x, y)
        } catch (e: Exception) {
            Log.e(TAG, "屏幕坐标点击失败", e)
            false
        }
    }
    
    /**
     * 滚动搜索结果列表（把歌手栏滑走，露出歌曲结果）
     * 使用 dispatchGesture 直接滑动（和 tapScreenPosition 完全相同的方式，已验证可用）
     */
    private fun scrollSearchResultList() {
        val screenWidth = resources.displayMetrics.widthPixels
        val screenHeight = resources.displayMetrics.heightPixels
        
        // 方法1: 使用 dispatchGesture 直接滑动（和点击手势完全相同的调用方式）
        // tapScreenPosition/dispatchGestureSync 已验证可从后台线程执行
        Log.d(TAG, "  滚动: 使用 dispatchGesture 滑动手势...")
        for (i in 1..3) {
            try {
                val latch = CountDownLatch(1)
                var success = false
                
                val path = Path()
                path.moveTo(screenWidth / 2f, screenHeight * 0.70f)
                path.lineTo(screenWidth / 2f, screenHeight * 0.20f)
                
                val gesture = GestureDescription.Builder()
                    .addStroke(GestureDescription.StrokeDescription(path, 0, 400))
                    .build()
                
                val callback = object : GestureResultCallback() {
                    override fun onCompleted(gestureDescription: GestureDescription?) {
                        Log.d(TAG, "  第${i}次滑动手势完成")
                        success = true
                        latch.countDown()
                    }
                    override fun onCancelled(gestureDescription: GestureDescription?) {
                        Log.w(TAG, "  第${i}次滑动手势被取消")
                        latch.countDown()
                    }
                }
                
                // 直接调用 dispatchGesture，和 dispatchGestureSync 一样
                dispatchGesture(gesture, callback, null)
                val completed = latch.await(3, TimeUnit.SECONDS)
                
                if (!completed) {
                    Log.w(TAG, "  第${i}次滑动超时")
                }
                
                if (success) {
                    Log.d(TAG, "  ✓ 第${i}次滑动成功")
                    Thread.sleep(600)
                } else {
                    Log.w(TAG, "  第${i}次滑动失败, 尝试备用方法")
                    break
                }
            } catch (e: Exception) {
                Log.e(TAG, "  第${i}次滑动异常", e)
                break
            }
        }
        
        // 备用方法: ACTION_SCROLL_FORWARD
        val rootNode = rootInActiveWindow
        if (rootNode != null) {
            val scrollable = findScrollableNode(rootNode)
            if (scrollable != null) {
                Log.d(TAG, "  备用: ACTION_SCROLL_FORWARD on ${scrollable.className}")
                scrollable.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
                Thread.sleep(500)
                scrollable.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
                Thread.sleep(500)
            }
        }
    }
    
    /**
     * 智能定位歌手栏下方，点击第一条歌曲结果
     * 通过遍历界面节点，找到包含"歌手"、"歌手:"或歌手头像等标志性元素的区域
     * 然后在该区域下方点击第一个歌曲结果
     */
    private fun clickBelowArtistBar(screenWidth: Int, screenHeight: Int): Boolean {
        val rootNode = rootInActiveWindow ?: return false
        
        // 收集所有节点的位置和文本信息
        var artistBarBottom = -1
        var tabBarBottom = -1
        
        // 查找歌手栏: 包含"歌手"、"歌手:"、"歌手："的节点
        val artistKeywords = arrayOf("歌手", "歌手:", "歌手：", "演唱者", "singer")
        for (keyword in artistKeywords) {
            val nodes = rootNode.findAccessibilityNodeInfosByText(keyword)
            for (node in nodes) {
                val bounds = Rect()
                node.getBoundsInScreen(bounds)
                // 歌手栏通常在屏幕上部 (10%-60%)
                if (bounds.top > screenHeight * 0.08 && bounds.top < screenHeight * 0.65) {
                    Log.d(TAG, "    找到歌手相关节点: text=${node.text}, bounds=$bounds")
                    // 取最大的bottom作为歌手栏底部
                    // 歌手栏可能有头像/描述等，取其父容器的底部更准确
                    var checkNode: AccessibilityNodeInfo? = node
                    var maxBottom = bounds.bottom
                    // 向上找父节点，找到合适的歌手栏容器
                    for (depth in 0..3) {
                        val parent = checkNode?.parent ?: break
                        val parentBounds = Rect()
                        parent.getBoundsInScreen(parentBounds)
                        // 父节点不能太大（不能是整个屏幕）
                        if (parentBounds.height() < screenHeight * 0.35 && parentBounds.width() > screenWidth * 0.5) {
                            maxBottom = maxOf(maxBottom, parentBounds.bottom)
                            Log.d(TAG, "    歌手栏父容器($depth): bounds=$parentBounds")
                        }
                        checkNode = parent
                    }
                    artistBarBottom = maxOf(artistBarBottom, maxBottom)
                }
            }
        }
        
        // 查找标签栏: 包含"单曲"、"专辑"的节点
        val tabKeywords = arrayOf("单曲", "专辑")
        for (keyword in tabKeywords) {
            val nodes = rootNode.findAccessibilityNodeInfosByText(keyword)
            for (node in nodes) {
                val bounds = Rect()
                node.getBoundsInScreen(bounds)
                // 标签栏太小（高度<100px）说明是Tab标签
                if (bounds.height() < 150 && bounds.top > screenHeight * 0.05 && bounds.top < screenHeight * 0.3) {
                    tabBarBottom = maxOf(tabBarBottom, bounds.bottom)
                    Log.d(TAG, "    找到标签栏: text=${node.text}, bounds=$bounds")
                }
            }
        }
        
        // 确定点击起始Y位置
        val clickStartY = when {
            artistBarBottom > 0 -> {
                Log.d(TAG, "    歌手栏底部: $artistBarBottom")
                artistBarBottom + 30 // 在歌手栏下方30px开始点击
            }
            tabBarBottom > 0 -> {
                Log.d(TAG, "    标签栏底部: $tabBarBottom (未找到歌手栏)")
                tabBarBottom + 80 // 在标签栏下方80px（跳过可能的歌手栏）
            }
            else -> {
                Log.d(TAG, "    未找到歌手栏和标签栏")
                return false
            }
        }
        
        Log.d(TAG, "    从Y=$clickStartY 开始点击歌曲结果")
        
        // 在歌手栏下方区域点击，间隔50px尝试多个位置
        val clickX = screenWidth * 0.35f
        for (offset in arrayOf(20, 60, 110, 170, 240)) {
            val tapY = (clickStartY + offset).toFloat()
            if (tapY > screenHeight * 0.85f) break // 不要点到底部导航栏
            
            Log.d(TAG, "    智能点击: ($clickX, $tapY) offset=$offset")
            val success = tapScreenPosition(clickX, tapY)
            if (success) {
                Thread.sleep(1200)
                val currentRoot = rootInActiveWindow
                if (currentRoot != null) {
                    val pauseNodes = currentRoot.findAccessibilityNodeInfosByText("暂停")
                    val progressNodes = mutableListOf<AccessibilityNodeInfo>()
                    findAllNodesByClassName(currentRoot, progressNodes, "android.widget.SeekBar", 0, 10)
                    if (pauseNodes.isNotEmpty() || progressNodes.isNotEmpty()) {
                        Log.d(TAG, "    ✓ 智能定位成功！检测到播放控件")
                        return true
                    }
                    val editTexts = mutableListOf<AccessibilityNodeInfo>()
                    findAllNodesByClassName(currentRoot, editTexts, "android.widget.EditText", 0, 10)
                    if (editTexts.isEmpty()) {
                        Log.d(TAG, "    ✓ 搜索框消失，可能已进入播放页")
                        return true
                    }
                }
            }
        }
        
        Log.d(TAG, "    智能定位未成功")
        return false
    }
    
    /**
     * 递归查找可滚动的节点
     */
    private fun findScrollableNode(node: AccessibilityNodeInfo?, depth: Int = 0): AccessibilityNodeInfo? {
        if (node == null || depth > 10) return null
        if (node.isScrollable) return node
        for (i in 0 until node.childCount) {
            val result = findScrollableNode(node.getChild(i), depth + 1)
            if (result != null) return result
        }
        return null
    }
    
    /**
     * 打印详细节点树（用于调试搜索结果界面结构）
     */
    private fun printNodeTreeDetailed(node: AccessibilityNodeInfo?, depth: Int, maxDepth: Int) {
        if (node == null || depth > maxDepth) return
        val indent = "  ".repeat(depth)
        val bounds = Rect()
        node.getBoundsInScreen(bounds)
        val text = node.text?.toString()?.take(20) ?: ""
        val desc = node.contentDescription?.toString()?.take(20) ?: ""
        val id = node.viewIdResourceName ?: ""
        Log.d(TAG, "${indent}[${node.className}] id=$id text=$text desc=$desc " +
                "click=${node.isClickable} bounds=$bounds children=${node.childCount}")
        for (i in 0 until node.childCount) {
            printNodeTreeDetailed(node.getChild(i), depth + 1, maxDepth)
        }
    }
    
    /**
     * 递归检查节点或其子节点是否包含文本内容
     */
    private fun hasTextContent(node: AccessibilityNodeInfo?, depth: Int = 0): Boolean {
        if (node == null || depth > 5) return false
        if (!node.text.isNullOrEmpty()) return true
        if (!node.contentDescription.isNullOrEmpty()) return true
        for (i in 0 until node.childCount) {
            if (hasTextContent(node.getChild(i), depth + 1)) return true
        }
        return false
    }
    
    /**
     * 查找可点击的子节点
     */
    private fun findClickableChild(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (node == null) return null
        
        if (node.isClickable) {
            return node
        }
        
        for (i in 0 until node.childCount) {
            val result = findClickableChild(node.getChild(i))
            if (result != null) return result
        }
        
        return null
    }
    
    /**
     * 安全获取节点边界（返回null如果边界无效）
     */
    private fun getBounds(node: AccessibilityNodeInfo): Rect? {
        try {
            val bounds = Rect()
            node.getBoundsInScreen(bounds)
            
            // 验证边界是否有效
            if (bounds.left < 0 || bounds.top < 0 || 
                bounds.right <= bounds.left || bounds.bottom <= bounds.top) {
                return null
            }
            
            return bounds
        } catch (e: Exception) {
            return null
        }
    }
    
    /**
     * 查找所有可点击节点
     */
    private fun findAllClickableNodes(
        node: AccessibilityNodeInfo?,
        result: MutableList<AccessibilityNodeInfo>,
        depth: Int,
        maxDepth: Int
    ) {
        if (node == null || depth > maxDepth) return
        
        if (node.isClickable && node.isVisibleToUser) {
            result.add(node)
        }
        
        for (i in 0 until node.childCount) {
            findAllClickableNodes(node.getChild(i), result, depth + 1, maxDepth)
        }
    }
    
    /**
     * 根据资源ID查找节点
     */
    private fun findNodeByIds(root: AccessibilityNodeInfo, ids: Array<String>): AccessibilityNodeInfo? {
        for (id in ids) {
            val nodes = root.findAccessibilityNodeInfosByViewId("$QQMUSIC_PACKAGE:id/$id")
            if (nodes.isNotEmpty()) {
                return nodes[0]
            }
        }
        return null
    }
    
    /**
     * 根据内容描述查找节点
     */
    private fun findNodeByContentDescription(
        root: AccessibilityNodeInfo,
        vararg descriptions: String
    ): AccessibilityNodeInfo? {
        for (desc in descriptions) {
            val nodes = root.findAccessibilityNodeInfosByText(desc)
            if (nodes.isNotEmpty()) {
                return nodes[0]
            }
        }
        return null
    }
    
    /**
     * 根据类名查找节点
     */
    private fun findNodeByClassName(root: AccessibilityNodeInfo, className: String): AccessibilityNodeInfo? {
        if (root.className == className) {
            return root
        }
        
        for (i in 0 until root.childCount) {
            val child = root.getChild(i) ?: continue
            val result = findNodeByClassName(child, className)
            if (result != null) {
                return result
            }
        }
        
        return null
    }
    
    /**
     * 执行点击操作
     */
    private fun performClick(node: AccessibilityNodeInfo): Boolean {
        if (node.isClickable) {
            return node.performAction(AccessibilityNodeInfo.ACTION_CLICK)
        }
        
        // 如果节点不可点击，尝试点击父节点
        var parent = node.parent
        while (parent != null) {
            if (parent.isClickable) {
                return parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            }
            parent = parent.parent
        }
        
        // 如果节点不可点击，尝试模拟触摸
        return performGestureClick(node)
    }
    
    /**
     * 通过手势模拟点击（同步等待完成）
     */
    private fun performGestureClick(node: AccessibilityNodeInfo): Boolean {
        return try {
            val rect = Rect()
            node.getBoundsInScreen(rect)
            
            if (rect.left < 0 || rect.top < 0 || rect.right <= rect.left || rect.bottom <= rect.top) {
                Log.w(TAG, "手势点击失败: 节点边界无效 $rect")
                return false
            }
            
            val x = rect.exactCenterX()
            val y = rect.exactCenterY()
            
            Log.d(TAG, "手势点击位置: ($x, $y), 边界: $rect")
            dispatchGestureSync(x, y)
        } catch (e: Exception) {
            Log.e(TAG, "手势点击失败", e)
            false
        }
    }
    
    /**
     * 同步发送手势点击并等待完成
     */
    private fun dispatchGestureSync(x: Float, y: Float): Boolean {
        val latch = CountDownLatch(1)
        var success = false
        
        val path = Path()
        path.moveTo(x, y)
        
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 150))
            .build()
        
        val callback = object : AccessibilityService.GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription?) {
                Log.d(TAG, "手势点击完成: ($x, $y)")
                success = true
                latch.countDown()
            }
            override fun onCancelled(gestureDescription: GestureDescription?) {
                Log.w(TAG, "手势点击被取消: ($x, $y)")
                success = false
                latch.countDown()
            }
        }
        
        dispatchGesture(gesture, callback, null)
        
        // 等最多2秒
        latch.await(2, TimeUnit.SECONDS)
        
        if (success) {
            Thread.sleep(300) // 等待UI响应
        }
        return success
    }
    
    /**
     * 打印节点树（用于调试）
     */
    fun printNodeTree() {
        val root = rootInActiveWindow
        if (root != null) {
            Log.d(TAG, "=== 节点树 ===")
            printNode(root, 0)
        }
    }
    
    private fun printNode(node: AccessibilityNodeInfo, depth: Int) {
        val indent = "  ".repeat(depth)
        Log.d(TAG, "$indent${node.className} | ${node.viewIdResourceName} | ${node.contentDescription} | ${node.text}")
        
        for (i in 0 until node.childCount) {
            val child = node.getChild(i)
            if (child != null) {
                printNode(child, depth + 1)
            }
        }
    }
}
