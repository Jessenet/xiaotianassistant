package com.xiaotian.assistant

import android.content.Context
import android.util.Log

/**
 * 音乐控制函数执行器
 *
 * 执行 SmartCommand 对应的音乐播放操作:
 * - 委托给 IMusicController 进行实际控制
 */
class SmartFunctionExecutor(
    private val context: Context,
    private val musicController: IMusicController
) {

    companion object {
        private const val TAG = "SmartFunctionExecutor"
    }

    /**
     * 执行 SmartCommand
     *
     * @return ExecutionResult 包含执行状态和反馈文本
     */
    fun execute(command: SmartCommand): ExecutionResult {
        return try {
            Log.d(TAG, "执行命令: ${command.describe()}")

            when (command) {
                // ── 音乐控制 ──
                is SmartCommand.PlaySong -> executePlaySong(command)
                is SmartCommand.PauseMusic -> executeSimple("暂停音乐") { musicController.pause() }
                is SmartCommand.ResumeMusic -> executeSimple("继续播放") { musicController.resume() }
                is SmartCommand.NextSong -> executeSimple("下一首") { musicController.next() }
                is SmartCommand.PreviousSong -> executeSimple("上一首") { musicController.previous() }
                is SmartCommand.SetVolume -> executeSetVolume(command)
                is SmartCommand.VolumeUp -> executeSimple("调高音量") { musicController.volumeUp() }
                is SmartCommand.VolumeDown -> executeSimple("调低音量") { musicController.volumeDown() }

                // ── 问候 ──
                is SmartCommand.Greeting -> {
                    Log.d(TAG, "问候语: ${command.text}")
                    ExecutionResult(true, "你好！我可以帮你播放音乐，试试说\"播放稻香\"")
                }

                // ── 未识别 ──
                is SmartCommand.Unknown -> {
                    Log.w(TAG, "未识别的命令: ${command.text}")
                    ExecutionResult(false, "抱歉，我不太理解您的意思")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "执行命令失败", e)
            ExecutionResult(false, "执行失败: ${e.message}")
        }
    }

    /**
     * 执行并兼容旧版 FunctionExecutor (返回 Boolean)
     */
    fun executeLegacy(command: SmartCommand): Boolean {
        return execute(command).success
    }

    // ══════════════════════════════════════════════
    //  音乐控制
    // ══════════════════════════════════════════════

    private fun executePlaySong(cmd: SmartCommand.PlaySong): ExecutionResult {
        val success = when {
            cmd.song.isEmpty() && cmd.artist.isEmpty() -> {
                Log.w(TAG, "歌曲名和艺术家都为空")
                false
            }
            cmd.song.isEmpty() -> musicController.playSong(cmd.artist)
            cmd.artist.isNotEmpty() -> musicController.playSong(cmd.song, cmd.artist)
            else -> musicController.playSong(cmd.song)
        }

        val msg = if (success) {
            buildString {
                append("正在播放")
                if (cmd.artist.isNotEmpty() && cmd.song.isNotEmpty()) {
                    append(" ${cmd.artist} 的 ${cmd.song}")
                } else if (cmd.artist.isNotEmpty()) {
                    append(" ${cmd.artist} 的歌")
                } else if (cmd.song.isNotEmpty()) {
                    append(" ${cmd.song}")
                }
            }
        } else "播放失败"

        return ExecutionResult(success, msg)
    }

    private fun executeSetVolume(cmd: SmartCommand.SetVolume): ExecutionResult {
        if (cmd.volume !in 0..100) {
            return ExecutionResult(false, "音量超出范围 (0-100)")
        }
        val success = musicController.setVolume(cmd.volume)
        return ExecutionResult(success, if (success) "音量已设置为 ${cmd.volume}" else "设置音量失败")
    }

    // ══════════════════════════════════════════════
    //  辅助方法
    // ══════════════════════════════════════════════

    private fun executeSimple(description: String, action: () -> Boolean): ExecutionResult {
        val success = action()
        return ExecutionResult(success, if (success) description else "${description}失败")
    }

    /**
     * 执行结果
     */
    data class ExecutionResult(
        val success: Boolean,
        val message: String
    )
}
