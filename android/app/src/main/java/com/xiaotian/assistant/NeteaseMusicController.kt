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
 * 网易云音乐控制器
 * 支持标准媒体控制和无障碍服务控制
 */
class NeteaseMusicController(private val context: Context) : IMusicController {
    
    private val audioManager: AudioManager = 
        context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    
    private var useAccessibilityService = true
    
    companion object {
        private const val TAG = "NeteaseMusicController"
        
        // 网易云音乐包名
        const val NETEASE_PACKAGE = "com.netease.cloudmusic"
        
        // 媒体控制Action
        private const val ACTION_PLAY = "com.android.music.musicservicecommand.play"
        private const val ACTION_PAUSE = "com.android.music.musicservicecommand.pause"
        private const val ACTION_NEXT = "com.android.music.musicservicecommand.next"
        private const val ACTION_PREVIOUS = "com.android.music.musicservicecommand.previous"
        private const val ACTION_MEDIA_BUTTON = "android.intent.action.MEDIA_BUTTON"
    }
    
    override fun getAppName(): String = "网易云音乐"
    
    override fun getPackageName(): String = NETEASE_PACKAGE
    
    override fun isAppInstalled(): Boolean {
        return try {
            context.packageManager.getPackageInfo(NETEASE_PACKAGE, 0)
            true
        } catch (e: Exception) {
            Log.w(TAG, "网易云音乐未安装")
            false
        }
    }
    
    override fun setUseAccessibilityService(use: Boolean) {
        useAccessibilityService = use
    }
    
    override fun playSong(songName: String, artist: String?): Boolean {
        return try {
            if (!isAppInstalled()) {
                Toast.makeText(context, "请先安装网易云音乐", Toast.LENGTH_SHORT).show()
                return false
            }
            
            val query = if (artist != null) {
                "$artist $songName"
            } else {
                songName
            }
            
            Log.d(TAG, "播放歌曲: $query")
            
            // 优先使用无障碍服务
            if (useAccessibilityService && NeteaseMusicAccessibilityService.isServiceEnabled()) {
                val service = NeteaseMusicAccessibilityService.getInstance()
                if (service != null) {
                    Log.d(TAG, "使用无障碍服务搜索播放")
                    Toast.makeText(context, "正在搜索: $query", Toast.LENGTH_SHORT).show()
                    // 在后台线程执行，避免ANR
                    Thread {
                        try {
                            val success = service.searchAndPlay(songName, artist)
                            Log.d(TAG, "无障碍服务返回结果: $success")
                            Handler(Looper.getMainLooper()).post {
                                if (success) {
                                    Toast.makeText(context, "正在播放: $query", Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(context, "播放失败，请重试", Toast.LENGTH_SHORT).show()
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "后台searchAndPlay失败", e)
                        }
                    }.start()
                    return true
                }
            }
            
            // 降级到Intent方式
            // 方法1: 使用搜索Intent
            val searchIntent = Intent(Intent.ACTION_SEARCH).apply {
                setPackage(NETEASE_PACKAGE)
                putExtra("query", query)
                putExtra("keyword", query)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            
            try {
                context.startActivity(searchIntent)
                Toast.makeText(context, "正在搜索: $query", Toast.LENGTH_SHORT).show()
                return true
            } catch (e: Exception) {
                Log.w(TAG, "搜索Intent失败，尝试其他方法", e)
            }
            
            // 方法2: 启动网易云音乐主界面
            val launchIntent = context.packageManager.getLaunchIntentForPackage(NETEASE_PACKAGE)
            if (launchIntent != null) {
                launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(launchIntent)
                Toast.makeText(context, "已打开网易云音乐，请手动搜索: $query", Toast.LENGTH_LONG).show()
                return true
            }
            
            false
        } catch (e: Exception) {
            Log.e(TAG, "播放歌曲失败", e)
            Toast.makeText(context, "播放失败: ${e.message}", Toast.LENGTH_SHORT).show()
            false
        }
    }
    
    override fun pause(): Boolean {
        return try {
            Log.d(TAG, "暂停播放")
            
            // 优先使用无障碍服务
            if (useAccessibilityService && NeteaseMusicAccessibilityService.isServiceEnabled()) {
                val service = NeteaseMusicAccessibilityService.getInstance()
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
    
    override fun resume(): Boolean {
        return try {
            Log.d(TAG, "继续播放")
            
            // 优先使用无障碍服务
            if (useAccessibilityService && NeteaseMusicAccessibilityService.isServiceEnabled()) {
                val service = NeteaseMusicAccessibilityService.getInstance()
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
    
    override fun next(): Boolean {
        return try {
            Log.d(TAG, "下一首")
            
            // 优先使用无障碍服务
            if (useAccessibilityService && NeteaseMusicAccessibilityService.isServiceEnabled()) {
                val service = NeteaseMusicAccessibilityService.getInstance()
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
    
    override fun previous(): Boolean {
        return try {
            Log.d(TAG, "上一首")
            
            // 优先使用无障碍服务
            if (useAccessibilityService && NeteaseMusicAccessibilityService.isServiceEnabled()) {
                val service = NeteaseMusicAccessibilityService.getInstance()
                if (service?.clickPrevious() == true) {
                    Toast.makeText(context, "上一首", Toast.LENGTH_SHORT).show()
                    return true
                }
            }
            
            // 降级到媒体按键
            sendMediaButtonEvent(android.view.KeyEvent.KEYCODE_MEDIA_PREVIOUS)
            Toast.makeText(context, "上一首", Toast.LENGTH_SHORT).show()
            true
        } catch (e: Exception) {
            Log.e(TAG, "上一首失败", e)
            false
        }
    }
    
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
        
        // 使用AudioManager
        audioManager.dispatchMediaKeyEvent(downEvent)
        audioManager.dispatchMediaKeyEvent(upEvent)
        
        // 发送广播（备用）
        try {
            val intent = Intent(ACTION_MEDIA_BUTTON).apply {
                putExtra(Intent.EXTRA_KEY_EVENT, downEvent)
                setPackage(NETEASE_PACKAGE)
            }
            context.sendBroadcast(intent)
            
            intent.putExtra(Intent.EXTRA_KEY_EVENT, upEvent)
            context.sendBroadcast(intent)
        } catch (e: Exception) {
            Log.w(TAG, "广播媒体按键失败", e)
        }
    }
    
    /**
     * 通过URL Schema打开网易云音乐搜索
     */
    fun searchInNeteaseMusic(keyword: String): Boolean {
        return try {
            // 网易云音乐的URL Schema
            val encodedKeyword = Uri.encode(keyword)
            val uri = Uri.parse("orpheus://search?s=$encodedKeyword")
            
            val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            
            context.startActivity(intent)
            true
        } catch (e: Exception) {
            Log.e(TAG, "URL Schema搜索失败", e)
            playSong(keyword)
        }
    }
}
