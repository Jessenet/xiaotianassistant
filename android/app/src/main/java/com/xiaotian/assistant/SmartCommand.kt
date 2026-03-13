package com.xiaotian.assistant

/**
 * 音乐控制命令
 *
 * 支持音乐播放控制功能:
 * play_song, pause_music, resume_music, next_song, previous_song, set_volume
 */
sealed class SmartCommand {

    // ── 音乐控制 ──────────────────────────────
    data class PlaySong(
        val artist: String = "",
        val song: String = ""
    ) : SmartCommand()

    object PauseMusic : SmartCommand()
    object ResumeMusic : SmartCommand()
    object NextSong : SmartCommand()
    object PreviousSong : SmartCommand()

    data class SetVolume(val volume: Int) : SmartCommand()
    object VolumeUp : SmartCommand()
    object VolumeDown : SmartCommand()

    // ── 问候/闲聊 ─────────────────────────────
    data class Greeting(val text: String) : SmartCommand()

    // ── 未识别 ────────────────────────────────
    data class Unknown(val text: String) : SmartCommand()

    /**
     * 获取命令中文描述
     */
    fun describe(): String = when (this) {
        is PlaySong -> "播放歌曲 - 歌手: $artist, 歌曲: $song"
        is PauseMusic -> "暂停音乐"
        is ResumeMusic -> "继续播放"
        is NextSong -> "下一首"
        is PreviousSong -> "上一首"
        is SetVolume -> "设置音量: $volume"
        is VolumeUp -> "调高音量"
        is VolumeDown -> "调低音量"
        is Greeting -> "问候: $text"
        is Unknown -> "未识别: $text"
    }

    /**
     * 是否为音乐控制命令
     */
    fun isMusicCommand(): Boolean = when (this) {
        is PlaySong, is PauseMusic, is ResumeMusic,
        is NextSong, is PreviousSong, is SetVolume,
        is VolumeUp, is VolumeDown -> true
        else -> false
    }

    /**
     * 是否为需要执行的命令（排除问候和未识别）
     */
    fun isActionable(): Boolean = when (this) {
        is Greeting, is Unknown -> false
        else -> true
    }
}
