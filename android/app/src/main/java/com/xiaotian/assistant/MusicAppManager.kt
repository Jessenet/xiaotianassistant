package com.xiaotian.assistant

import android.content.Context
import android.content.SharedPreferences

/**
 * 音乐APP管理器
 * 管理多个音乐应用的切换和选择
 */
object MusicAppManager {
    
    private const val PREFS_NAME = "music_app_prefs"
    private const val KEY_SELECTED_APP = "selected_app"
    
    enum class MusicApp(val displayName: String, val packageName: String) {
        QQ_MUSIC("QQ音乐", "com.tencent.qqmusic"),
        NETEASE_MUSIC("网易云音乐", "com.netease.cloudmusic")
    }
    
    private lateinit var prefs: SharedPreferences
    private lateinit var context: Context
    private var currentController: IMusicController? = null
    
    /**
     * 初始化
     */
    fun init(context: Context) {
        this.context = context.applicationContext
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }
    
    /**
     * 获取当前选择的音乐APP
     */
    fun getSelectedApp(): MusicApp {
        val appName = prefs.getString(KEY_SELECTED_APP, MusicApp.QQ_MUSIC.name)
        return try {
            MusicApp.valueOf(appName ?: MusicApp.QQ_MUSIC.name)
        } catch (e: Exception) {
            MusicApp.QQ_MUSIC
        }
    }
    
    /**
     * 设置选择的音乐APP
     */
    fun setSelectedApp(app: MusicApp) {
        prefs.edit().putString(KEY_SELECTED_APP, app.name).apply()
        currentController = null // 清除缓存，下次重新创建
    }
    
    /**
     * 获取当前的音乐控制器
     */
    fun getCurrentController(): IMusicController {
        if (currentController == null) {
            currentController = when (getSelectedApp()) {
                MusicApp.QQ_MUSIC -> QQMusicController(context)
                MusicApp.NETEASE_MUSIC -> NeteaseMusicController(context)
            }
        }
        return currentController!!
    }
    
    /**
     * 获取所有可用的音乐APP（已安装）
     */
    fun getAvailableApps(): List<MusicApp> {
        val availableApps = mutableListOf<MusicApp>()
        
        for (app in MusicApp.values()) {
            val controller = createController(app)
            if (controller.isAppInstalled()) {
                availableApps.add(app)
            }
        }
        
        return availableApps
    }
    
    /**
     * 创建指定APP的控制器
     */
    private fun createController(app: MusicApp): IMusicController {
        return when (app) {
            MusicApp.QQ_MUSIC -> QQMusicController(context)
            MusicApp.NETEASE_MUSIC -> NeteaseMusicController(context)
        }
    }
    
    /**
     * 检查指定APP是否已安装
     */
    fun isAppInstalled(app: MusicApp): Boolean {
        return createController(app).isAppInstalled()
    }
    
    /**
     * 获取所有支持的APP列表
     */
    fun getAllApps(): List<MusicApp> {
        return MusicApp.values().toList()
    }
}
