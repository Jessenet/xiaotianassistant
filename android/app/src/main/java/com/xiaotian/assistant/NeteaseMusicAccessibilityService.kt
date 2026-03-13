package com.xiaotian.assistant

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import kotlinx.coroutines.*

/**
 * 网易云音乐无障碍服务
 * 通过模拟点击来精确控制网易云音乐播放器
 */
class NeteaseMusicAccessibilityService : AccessibilityService() {
    
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    companion object {
        private const val TAG = "NeteaseMusicAccess"
        
        // 网易云音乐包名
        private const val NETEASE_PACKAGE = "com.netease.cloudmusic"
        
        // 服务实例
        @Volatile
        private var instance: NeteaseMusicAccessibilityService? = null
        
        fun getInstance(): NeteaseMusicAccessibilityService? = instance
        
        fun isServiceEnabled(): Boolean = instance != null
        
        // 控件ID（根据网易云音乐版本可能需要调整）
        private val PLAY_BUTTON_IDS = arrayOf(
            "play",
            "play_button",
            "btn_play",
            "player_play",
            "playButton"
        )
        
        private val PAUSE_BUTTON_IDS = arrayOf(
            "pause",
            "pause_button",
            "btn_pause",
            "player_pause"
        )
        
        private val NEXT_BUTTON_IDS = arrayOf(
            "next",
            "next_button",
            "btn_next",
            "player_next",
            "nextButton"
        )
        
        private val PREV_BUTTON_IDS = arrayOf(
            "prev",
            "previous",
            "previous_button",
            "btn_previous",
            "player_prev"
        )
        
        private val SEARCH_BUTTON_IDS = arrayOf(
            "search",
            "search_button",
            "btn_search",
            "searchButton"
        )
    }
    
    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.d(TAG, "网易云音乐无障碍服务已连接")
    }
    
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return
        
        if (event.packageName == NETEASE_PACKAGE) {
            when (event.eventType) {
                AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> {
                    Log.d(TAG, "网易云音乐界面变化: ${event.className}")
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
        Log.d(TAG, "网易云音乐无障碍服务已销毁")
    }
    
    /**
     * 搜索并播放歌曲
     */
    fun searchAndPlay(songName: String, artist: String? = null): Boolean {
        return try {
            Log.d(TAG, "搜索歌曲: $songName, 艺术家: $artist")
            
            // 1. 打开网易云音乐
            if (!openNeteaseMusic()) {
                Log.w(TAG, "无法打开网易云音乐")
                return false
            }
            
            Thread.sleep(1500)
            
            // 2. 找到并点击搜索按钮
            if (!clickSearchButton()) {
                Log.w(TAG, "未找到搜索按钮")
                return false
            }
            
            Thread.sleep(500)
            
            // 3. 输入搜索内容
            val query = if (artist != null) "$artist $songName" else songName
            if (!inputSearchText(query)) {
                Log.w(TAG, "无法输入搜索文本")
                return false
            }
            
            Thread.sleep(1000)
            
            // 4. 点击第一个搜索结果
            if (!clickFirstSearchResult()) {
                Log.w(TAG, "未找到搜索结果")
                return false
            }
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "搜索播放失败", e)
            false
        }
    }
    
    /**
     * 点击播放/暂停按钮
     */
    fun clickPlayPause(): Boolean {
        return try {
            val rootNode = rootInActiveWindow ?: return false
            
            val playPauseNode = findNodeByIds(rootNode, PLAY_BUTTON_IDS + PAUSE_BUTTON_IDS)
                ?: findNodeByContentDescription(rootNode, "播放", "暂停", "play", "pause")
            
            if (playPauseNode != null) {
                return performClick(playPauseNode)
            }
            
            Log.w(TAG, "未找到播放/暂停按钮")
            false
        } catch (e: Exception) {
            Log.e(TAG, "点击播放/暂停失败", e)
            false
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
    
    private fun openNeteaseMusic(): Boolean {
        return try {
            val intent = packageManager.getLaunchIntentForPackage(NETEASE_PACKAGE)
            if (intent != null) {
                intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                startActivity(intent)
                true
            } else {
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "打开网易云音乐失败", e)
            false
        }
    }
    
    private fun clickSearchButton(): Boolean {
        val rootNode = rootInActiveWindow ?: return false
        
        val searchNode = findNodeByIds(rootNode, SEARCH_BUTTON_IDS)
            ?: findNodeByContentDescription(rootNode, "搜索", "search")
        
        return if (searchNode != null) {
            performClick(searchNode)
        } else {
            false
        }
    }
    
    private fun inputSearchText(text: String): Boolean {
        val rootNode = rootInActiveWindow ?: return false
        
        val editText = findNodeByClassName(rootNode, "android.widget.EditText")
        if (editText != null) {
            editText.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            
            val arguments = android.os.Bundle()
            arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
            editText.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
            
            Thread.sleep(300)
            editText.refresh()
            
            return true
        }
        
        return false
    }
    
    private fun clickFirstSearchResult(): Boolean {
        val rootNode = rootInActiveWindow ?: return false
        
        val listNode = findNodeByClassName(rootNode, "androidx.recyclerview.widget.RecyclerView")
            ?: findNodeByClassName(rootNode, "android.widget.ListView")
        
        if (listNode != null && listNode.childCount > 0) {
            val firstItem = listNode.getChild(0)
            if (firstItem != null) {
                return performClick(firstItem)
            }
        }
        
        return false
    }
    
    private fun findNodeByIds(root: AccessibilityNodeInfo, ids: Array<String>): AccessibilityNodeInfo? {
        for (id in ids) {
            val nodes = root.findAccessibilityNodeInfosByViewId("$NETEASE_PACKAGE:id/$id")
            if (nodes.isNotEmpty()) {
                return nodes[0]
            }
        }
        return null
    }
    
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
    
    private fun performClick(node: AccessibilityNodeInfo): Boolean {
        if (node.isClickable) {
            return node.performAction(AccessibilityNodeInfo.ACTION_CLICK)
        }
        
        var parent = node.parent
        while (parent != null) {
            if (parent.isClickable) {
                return parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            }
            parent = parent.parent
        }
        
        return performGestureClick(node)
    }
    
    private fun performGestureClick(node: AccessibilityNodeInfo): Boolean {
        return try {
            val rect = Rect()
            node.getBoundsInScreen(rect)
            
            val x = rect.exactCenterX()
            val y = rect.exactCenterY()
            
            val path = Path()
            path.moveTo(x, y)
            
            val gestureBuilder = GestureDescription.Builder()
            gestureBuilder.addStroke(GestureDescription.StrokeDescription(path, 0, 100))
            
            dispatchGesture(gestureBuilder.build(), null, null)
            true
        } catch (e: Exception) {
            Log.e(TAG, "手势点击失败", e)
            false
        }
    }
}
