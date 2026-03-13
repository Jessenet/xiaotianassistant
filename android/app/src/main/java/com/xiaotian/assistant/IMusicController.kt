package com.xiaotian.assistant

/**
 * 音乐控制器接口
 * 统一不同音乐APP的控制方式
 */
interface IMusicController {
    /**
     * 播放歌曲
     */
    fun playSong(songName: String, artist: String? = null): Boolean
    
    /**
     * 暂停播放
     */
    fun pause(): Boolean
    
    /**
     * 继续播放
     */
    fun resume(): Boolean
    
    /**
     * 下一首
     */
    fun next(): Boolean
    
    /**
     * 上一首
     */
    fun previous(): Boolean
    
    /**
     * 设置音量
     */
    fun setVolume(level: Int): Boolean

    /**
     * 调高音量
     */
    fun volumeUp(): Boolean

    /**
     * 调低音量
     */
    fun volumeDown(): Boolean
    
    /**
     * 检查APP是否已安装
     */
    fun isAppInstalled(): Boolean
    
    /**
     * 获取APP名称
     */
    fun getAppName(): String
    
    /**
     * 获取包名
     */
    fun getPackageName(): String
    
    /**
     * 设置是否使用无障碍服务
     */
    fun setUseAccessibilityService(use: Boolean)
}
