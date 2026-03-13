package com.xiaotian.assistant

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast

/**
 * QQ音乐控制器
 * 支持两种控制方式：
 * 1. 标准媒体控制（Intent和媒体按键）
 * 2. 无障碍服务（更精确的控制）
 */
class QQMusicController(private val context: Context) : IMusicController {
    
    private val audioManager: AudioManager = 
        context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    
    // 是否优先使用无障碍服务
    private var useAccessibilityService = true
    
    companion object {
        private const val TAG = "QQMusicController"
        
        // QQ音乐包名
        const val QQMUSIC_PACKAGE = "com.tencent.qqmusic"
        
        // 媒体控制Action
        private const val ACTION_PLAY = "com.android.music.musicservicecommand.play"
        private const val ACTION_PAUSE = "com.android.music.musicservicecommand.pause"
        private const val ACTION_NEXT = "com.android.music.musicservicecommand.next"
        private const val ACTION_PREVIOUS = "com.android.music.musicservicecommand.previous"
        
        // 标准媒体按键事件
        private const val ACTION_MEDIA_BUTTON = "android.intent.action.MEDIA_BUTTON"
    }
    
    /**
     * 检查QQ音乐是否已安装
     */
    private fun isQQMusicInstalled(): Boolean {
        return try {
            context.packageManager.getPackageInfo(QQMUSIC_PACKAGE, 0)
            true
        } catch (e: Exception) {
            Log.w(TAG, "QQ音乐未安装")
            false
        }
    }
    
    override fun getAppName(): String = "QQ音乐"
    
    override fun getPackageName(): String = QQMUSIC_PACKAGE
    
    override fun isAppInstalled(): Boolean {
        return try {
            context.packageManager.getPackageInfo(QQMUSIC_PACKAGE, 0)
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * 设置是否优先使用无障碍服务
     */
    override fun setUseAccessibilityService(use: Boolean) {
        useAccessibilityService = use
    }
    
    /**
     * 播放歌曲
     * 优先尝试无障碍服务，失败则使用Intent
     */
    override fun playSong(songName: String, artist: String?): Boolean {
        return try {
            if (!isAppInstalled()) {
                Toast.makeText(context, "请先安装QQ音乐", Toast.LENGTH_SHORT).show()
                return false
            }
            
            // 构建搜索查询
            val query = if (artist != null) {
                "$artist $songName"
            } else {
                songName
            }
            
            Log.d(TAG, "播放歌曲: $query")
            Toast.makeText(context, "正在搜索: $query", Toast.LENGTH_SHORT).show()
            
            // 在后台线程执行（searchAndPlay 有大量 sleep，必须在后台）
            Thread {
                try {
                    // 尝试获取无障碍服务实例
                    var service = AccessibilityHelper.getServiceInstance()
                    
                    if (service == null) {
                        Log.d(TAG, "无障碍服务尚未连接，先打开QQ音乐并等待服务连接...")
                        // 先打开QQ音乐（利用等待时间让 QQ 加载）
                        val launchIntent = context.packageManager.getLaunchIntentForPackage(QQMUSIC_PACKAGE)
                        if (launchIntent != null) {
                            launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                            context.startActivity(launchIntent)
                        }
                        
                        // 等待无障碍服务连接（最多 12 秒，每 500ms 检查一次）
                        for (i in 0 until 24) {
                            Thread.sleep(500)
                            service = AccessibilityHelper.getServiceInstance()
                            if (service != null) {
                                Log.d(TAG, "无障碍服务已连接 (等待了 ${(i + 1) * 500}ms)")
                                break
                            }
                        }
                    }
                    
                    if (service != null) {
                        Log.d(TAG, "使用无障碍服务 searchAndPlay: song=$songName, artist=$artist")
                        val success = service.searchAndPlay(songName, artist)
                        Log.d(TAG, "无障碍服务返回结果: $success")
                        Handler(Looper.getMainLooper()).post {
                            if (success) {
                                Toast.makeText(context, "正在播放: $query", Toast.LENGTH_SHORT).show()
                            } else {
                                Toast.makeText(context, "播放失败，请重试", Toast.LENGTH_SHORT).show()
                            }
                        }
                    } else {
                        Log.w(TAG, "无障碍服务不可用，请在系统设置中启用")
                        Handler(Looper.getMainLooper()).post {
                            Toast.makeText(context, "请在设置中启用无障碍服务以自动搜索播放", Toast.LENGTH_LONG).show()
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "后台 searchAndPlay 失败", e)
                }
            }.start()
            
            true  // 立即返回，避免主线程阻塞
        } catch (e: Exception) {
            Log.e(TAG, "播放歌曲失败", e)
            Toast.makeText(context, "播放失败: ${e.message}", Toast.LENGTH_SHORT).show()
            false
        }
    }
    
    /**
     * 暂停播放
     * 优先使用无障碍服务，失败则使用媒体按键
     */
    override fun pause(): Boolean {
        return try {
            Log.d(TAG, "暂停播放")
            
            // 优先使用无障碍服务
            if (useAccessibilityService && AccessibilityHelper.isServiceRunning()) {
                val service = AccessibilityHelper.getServiceInstance()
                if (service?.clickPlayPause() == true) {
                    Toast.makeText(context, "已暂停", Toast.LENGTH_SHORT).show()
                    return true
                }
            }
            
            // 降级到媒体按键
            sendMediaButtonEvent(android.view.KeyEvent.KEYCODE_MEDIA_PAUSE)
            Toast.makeText(context, "已暂停", Toast.LENGTH_SHORT).show()
            true
        } catch (e: Exception) {
            Log.e(TAG, "暂停失败", e)
            false
        }
    }
    
    /**
     * 继续播放
     * 优先使用无障碍服务，失败则使用媒体按键
     */
    override fun resume(): Boolean {
        return try {
            Log.d(TAG, "继续播放")
            
            // 优先使用无障碍服务
            if (useAccessibilityService && AccessibilityHelper.isServiceRunning()) {
                val service = AccessibilityHelper.getServiceInstance()
                if (service?.clickPlayPause() == true) {
                    Toast.makeText(context, "继续播放", Toast.LENGTH_SHORT).show()
                    return true
                }
            }
            
            // 降级到媒体按键
            sendMediaButtonEvent(android.view.KeyEvent.KEYCODE_MEDIA_PLAY)
            Toast.makeText(context, "继续播放", Toast.LENGTH_SHORT).show()
            true
        } catch (e: Exception) {
            Log.e(TAG, "继续播放失败", e)
            false
        }
    }
    
    /**
     * 下一首
     */
    override fun next(): Boolean {
        return try {
            Log.d(TAG, "下一首")
            
            // 优先使用无障碍服务
            if (useAccessibilityService && AccessibilityHelper.isServiceRunning()) {
                val service = AccessibilityHelper.getServiceInstance()
                if (service?.clickNext() == true) {
                    Toast.makeText(context, "下一首", Toast.LENGTH_SHORT).show()
                    return true
                }
            }
            
            // 降级到媒体按键
            sendMediaButtonEvent(android.view.KeyEvent.KEYCODE_MEDIA_NEXT)
            Toast.makeText(context, "下一首", Toast.LENGTH_SHORT).show()
            true
        } catch (e: Exception) {
            Log.e(TAG, "下一首失败", e)
            false
        }
    }
    
    /**
     * 上一首
     */
    override fun previous(): Boolean {
        return try {
            Log.d(TAG, "上一首")
            sendMediaButtonEvent(android.view.KeyEvent.KEYCODE_MEDIA_PREVIOUS)
            Toast.makeText(context, "上一首", Toast.LENGTH_SHORT).show()
            true
        } catch (e: Exception) {
            Log.e(TAG, "上一首失败", e)
            false
        }
    }
    
    /**
     * 设置音量
     */
    override fun setVolume(level: Int): Boolean {
        return try {
            val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
            val targetVolume = (level * maxVolume / 100).coerceIn(0, maxVolume)
            
            Log.d(TAG, "设置音量: $level% -> $targetVolume/$maxVolume")
            audioManager.setStreamVolume(
                AudioManager.STREAM_MUSIC,
                targetVolume,
                AudioManager.FLAG_SHOW_UI
            )
            
            Toast.makeText(context, "音量: $level%", Toast.LENGTH_SHORT).show()
            true
        } catch (e: Exception) {
            Log.e(TAG, "设置音量失败", e)
            false
        }
    }

    /**
     * 调高音量 (每次 +5%)
     */
    override fun volumeUp(): Boolean {
        return try {
            val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
            val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
            val step = Math.max(1, (maxVolume * 5 / 100))  // 5% of max, at least 1
            val newVolume = (currentVolume + step).coerceAtMost(maxVolume)
            audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, newVolume, AudioManager.FLAG_SHOW_UI)
            val pct = newVolume * 100 / maxVolume
            Log.d(TAG, "调高音量: $currentVolume -> $newVolume/$maxVolume ($pct%)")
            Toast.makeText(context, "音量: $pct%", Toast.LENGTH_SHORT).show()
            true
        } catch (e: Exception) {
            Log.e(TAG, "调高音量失败", e)
            false
        }
    }

    /**
     * 调低音量 (每次 -5%)
     */
    override fun volumeDown(): Boolean {
        return try {
            val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
            val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
            val step = Math.max(1, (maxVolume * 5 / 100))  // 5% of max, at least 1
            val newVolume = (currentVolume - step).coerceAtLeast(0)
            audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, newVolume, AudioManager.FLAG_SHOW_UI)
            val pct = newVolume * 100 / maxVolume
            Log.d(TAG, "调低音量: $currentVolume -> $newVolume/$maxVolume ($pct%)")
            Toast.makeText(context, "音量: $pct%", Toast.LENGTH_SHORT).show()
            true
        } catch (e: Exception) {
            Log.e(TAG, "调低音量失败", e)
            false
        }
    }
    
    /**
     * 发送媒体按键事件
     */
    private fun sendMediaButtonEvent(keyCode: Int) {
        val downEvent = android.view.KeyEvent(
            android.view.KeyEvent.ACTION_DOWN,
            keyCode
        )
        val upEvent = android.view.KeyEvent(
            android.view.KeyEvent.ACTION_UP,
            keyCode
        )
        
        // 方法1: 使用AudioManager
        audioManager.dispatchMediaKeyEvent(downEvent)
        audioManager.dispatchMediaKeyEvent(upEvent)
        
        // 方法2: 发送广播（备用）
        try {
            val intent = Intent(ACTION_MEDIA_BUTTON).apply {
                putExtra(Intent.EXTRA_KEY_EVENT, downEvent)
                setPackage(QQMUSIC_PACKAGE)
            }
            context.sendBroadcast(intent)
            
            intent.putExtra(Intent.EXTRA_KEY_EVENT, upEvent)
            context.sendBroadcast(intent)
        } catch (e: Exception) {
            Log.w(TAG, "广播媒体按键失败", e)
        }
    }
    
    /**
     * 通过Deep Link打开QQ音乐（如果支持）
     */
    fun openQQMusicWithDeepLink(songId: String): Boolean {
        return try {
            val uri = Uri.parse("qqmusic://songDetail?songid=$songId")
            val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Deep Link打开失败", e)
            false
        }
    }
    
    /**
     * 通过URL Schema打开QQ音乐搜索
     */
    fun searchInQQMusic(keyword: String): Boolean {
        return try {
            // QQ音乐的URL Schema（可能需要根据实际情况调整）
            val encodedKeyword = Uri.encode(keyword)
            val uri = Uri.parse("qqmusic://search?key=$encodedKeyword")
            
            val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            
            context.startActivity(intent)
            true
        } catch (e: Exception) {
            Log.e(TAG, "URL Schema搜索失败", e)
            
            // 降级到普通搜索
            playSong(keyword)
        }
    }
}
