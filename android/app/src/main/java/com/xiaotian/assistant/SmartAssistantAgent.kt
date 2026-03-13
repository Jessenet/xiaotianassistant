package com.xiaotian.assistant

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * 智能音乐助手代理 - 音乐控制 Function Calling
 *
 * 使用 ExecuTorch 运行时进行推理, 支持 Qualcomm/MediaTek NPU 加速
 * 专注于音乐播放控制功能
 */
class SmartAssistantAgent(private val context: Context) {

    private val modelManager: ExecuTorchModelManager by lazy {
        ExecuTorchModelManager.getInstance(context)
    }

    private val gson = Gson()

    companion object {
        private const val TAG = "SmartAssistantAgent"
    }

    /**
     * 初始化 (加载 ExecuTorch 模型, 自动选择 NPU/CPU 后端)
     */
    suspend fun initialize(progressCallback: ((Int) -> Unit)? = null): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                progressCallback?.invoke(10)
                Log.d(TAG, "开始加载 ExecuTorch LLM 模型...")

                val result = modelManager.initialize()

                if (result.isSuccess) {
                    progressCallback?.invoke(100)
                    Log.d(TAG, "ExecuTorch 模型加载成功 (后端: ${modelManager.getBackendInfo()})")
                } else {
                    Log.e(TAG, "模型加载失败", result.exceptionOrNull())
                }

                result
            } catch (e: Exception) {
                Log.e(TAG, "初始化失败", e)
                Result.failure(e)
            }
        }
    }

    /**
     * 处理用户命令 (返回 SmartCommand)
     */
    // 问候/打招呼模式，这类输入不应该触发任何音乐操作
    private val GREETING_PATTERNS = listOf(
        "你好", "您好", "嗨", "哈喽", "hello", "hi ",
        "早上好", "下午好", "晚上好", "晚安",
        "谢谢", "感谢", "再见", "拜拜", "bye"
    )

    // 音乐意图关键词 - 用户原始输入中至少要包含其中之一，才认为是有效的音乐命令
    // 注意：避免使用单字如"放""听""停"，太宽泛会误匹配非音乐语句
    private val MUSIC_INTENT_KEYWORDS = listOf(
        "播放", "放歌", "放一首", "放首", "放一个", "play",
        "听歌", "听一首", "听首", "想听", "来一首", "来首", "唱",
        "暂停", "pause", "停止播放", "停止", "继续播放", "继续", "resume",
        "下一首", "上一首", "下一个", "上一个", "切歌",
        "音量", "volume", "大声", "小声", "静音",
        "调高", "调低", "声音大", "声音小"
    )

    /**
     * 判断用户输入是否为纯问候/闲聊（不含任何音乐指令意图）
     */
    private fun isGreeting(text: String): Boolean {
        val lower = text.lowercase().trim()
        // 如果包含音乐意图关键词，则不是纯问候
        if (MUSIC_INTENT_KEYWORDS.any { lower.contains(it) }) return false
        // 检查是否匹配问候模式
        return GREETING_PATTERNS.any { lower.contains(it) }
    }

    /**
     * 检查用户原始输入是否包含音乐操作意图
     */
    private fun hasMusicIntent(text: String): Boolean {
        val lower = text.lowercase().trim()
        return MUSIC_INTENT_KEYWORDS.any { lower.contains(it) }
    }

    suspend fun processCommand(userQuery: String): Result<SmartCommand> {
        return withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "处理命令: $userQuery")

                // 前置检查：纯问候/闲聊，直接返回，不调用模型
                if (isGreeting(userQuery)) {
                    Log.d(TAG, "检测到问候语，跳过AI模型")
                    val cmd = SmartCommand.Greeting(userQuery)
                    Log.d(TAG, "✓ 最终命令: ${cmd.describe()}")
                    return@withContext Result.success(cmd)
                }

                var command: SmartCommand? = null
                var useModelFailed = false

                try {
                    // 1. 优先使用 ExecuTorch LLM 模型推理 (18层,107M,INT8)
                    val result = modelManager.generate(userQuery)

                    if (result.isSuccess) {
                        val generatedText = result.getOrNull()!!
                        Log.d(TAG, "模型输出: $generatedText")

                        // 2. 解析 function calling 结果
                        command = parseFunctionCall(generatedText)

                        if (command is SmartCommand.Unknown) {
                            Log.w(TAG, "模型返回未知命令, 使用关键词降级")
                            useModelFailed = true
                        }
                    } else {
                        Log.w(TAG, "模型生成失败, 使用关键词降级")
                        useModelFailed = true
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "模型推理异常, 使用关键词降级", e)
                    useModelFailed = true
                }

                // 降级: 关键词匹配
                if (useModelFailed || command == null) {
                    command = parseByKeywords(userQuery)
                }

                // 安全检查：如果模型返回了任何音乐命令，但用户原始输入完全没有音乐意图，
                // 视为模型幻觉，降级为 Unknown（防止 "大模型" 等非音乐输入误触发上一首/下一首等）
                if (command!!.isMusicCommand() && !hasMusicIntent(userQuery)) {
                    Log.w(TAG, "模型返回 ${command!!.describe()} 但原始输入无音乐意图，判定为幻觉: $userQuery")
                    command = SmartCommand.Unknown(userQuery)
                }

                // 安全检查：如果模型返回了 SetVolume，但用户原始输入是相对音量指令，
                // 覆盖为 VolumeUp / VolumeDown
                if (command is SmartCommand.SetVolume) {
                    val lower = userQuery.lowercase()
                    when {
                        lower.contains("调高") || lower.contains("大声") || lower.contains("大点") ||
                        lower.contains("音量高") || lower.contains("声音大") || lower.contains("加大") -> {
                            Log.d(TAG, "模型返回 SetVolume 但用户意图是调高音量，覆盖为 VolumeUp")
                            command = SmartCommand.VolumeUp
                        }
                        lower.contains("调低") || lower.contains("小声") || lower.contains("小点") ||
                        lower.contains("音量低") || lower.contains("声音小") || lower.contains("减小") -> {
                            Log.d(TAG, "模型返回 SetVolume 但用户意图是调低音量，覆盖为 VolumeDown")
                            command = SmartCommand.VolumeDown
                        }
                    }
                }

                Log.d(TAG, "✓ 最终命令: ${command.describe()}")
                Result.success(command)

            } catch (e: Exception) {
                Log.e(TAG, "处理命令失败", e)
                Result.failure(e)
            }
        }
    }

    // ══════════════════════════════════════════════
    //  Function Call 解析
    // ══════════════════════════════════════════════

    /**
     * 解析模型输出的 function_call JSON
     *
     * 模型输出格式 (紧凑):
     *   <function_call>{"f":"play_song","p":{"s":"稻香","a":"周杰伦"}}</function_call>
     * 兼容旧格式:
     *   <function_call>{"function":"play_song","arguments":{"artist":"周杰伦","song":"稻香"}}</function_call>
     */
    private fun parseFunctionCall(text: String): SmartCommand {
        try {
            // 提取 <function_call>...</function_call> 中的 JSON
            val jsonStr = extractFunctionCallJson(text) ?: return SmartCommand.Unknown(text)

            val json = gson.fromJson(jsonStr, JsonObject::class.java)
            // 兼容紧凑 "f" 和完整 "function"
            val functionName = json.get("f")?.asString
                ?: json.get("function")?.asString
                ?: return SmartCommand.Unknown(text)
            // 兼容 "p" / "arguments" / "parameters"
            val args = json.getAsJsonObject("p")
                ?: json.getAsJsonObject("arguments")
                ?: json.getAsJsonObject("parameters")
                ?: JsonObject()

            return when (functionName) {
                // 音乐控制 (兼容紧凑 key: s=song, a=artist, l=level)
                "play_song" -> SmartCommand.PlaySong(
                    artist = args.optString("a").ifEmpty { args.optString("artist").ifEmpty { args.optString("artist_name") } },
                    song = args.optString("s").ifEmpty { args.optString("song").ifEmpty { args.optString("song_name").ifEmpty { args.optString("title") } } },
                )
                "pause_music" -> SmartCommand.PauseMusic
                "resume_music" -> SmartCommand.ResumeMusic
                "next_song" -> SmartCommand.NextSong
                "previous_song" -> SmartCommand.PreviousSong
                "volume_up" -> SmartCommand.VolumeUp
                "volume_down" -> SmartCommand.VolumeDown
                "set_volume" -> SmartCommand.SetVolume(
                    volume = args.optInt("l", args.optInt("level", args.optInt("volume", 50))),
                )

                else -> {
                    Log.w(TAG, "未知 function: $functionName")
                    SmartCommand.Unknown("$functionName: $args")
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "解析 function call 失败: $text", e)
            return SmartCommand.Unknown(text)
        }
    }

    /**
     * 从模型输出中提取 function_call JSON
     */
    private fun extractFunctionCallJson(text: String): String? {
        // 模式1: <function_call>{...}</function_call> (贪婪匹配嵌套花括号)
        val tagPattern = Regex("""<function_call>\s*(\{.+\})\s*(?:</function_call>)?""", RegexOption.DOT_MATCHES_ALL)
        tagPattern.find(text)?.let { return it.groupValues[1].trim() }

        // 模式2: 直接 JSON — 兼容紧凑 "f"/"p" 和完整 "function"/"arguments"/"parameters"
        val jsonPattern = Regex("""\{[^{}]*"(?:f|function)"\s*:\s*"[^"]+"\s*,\s*"(?:p|arguments|parameters)"\s*:\s*\{[^{}]*\}\s*\}""")
        jsonPattern.find(text)?.let { return it.value }

        // 模式3: 模型可能输出不完整的 JSON
        val braceStart = text.indexOf('{')
        if (braceStart >= 0) {
            val braceEnd = text.lastIndexOf('}')
            if (braceEnd > braceStart) {
                return text.substring(braceStart, braceEnd + 1)
            }
        }

        return null
    }

    // ══════════════════════════════════════════════
    //  关键词匹配降级
    // ══════════════════════════════════════════════

    private fun parseByKeywords(text: String): SmartCommand {
        val lowerText = text.lowercase()

        return when {
            // 音乐控制（精确匹配播放意图，避免"放"单字误触发）
            lowerText.contains("播放") || lowerText.contains("play") ||
            lowerText.contains("放歌") || lowerText.contains("放一首") ||
            lowerText.contains("放首") || lowerText.contains("放一个") -> {
                val (artist, song) = extractArtistAndSong(text)
                SmartCommand.PlaySong(artist, song)
            }
            lowerText.contains("暂停") || lowerText.contains("pause") -> SmartCommand.PauseMusic
            lowerText.contains("继续") || lowerText.contains("resume") -> SmartCommand.ResumeMusic
            lowerText.contains("下一首") || lowerText.contains("下一个") -> SmartCommand.NextSong
            lowerText.contains("上一首") || lowerText.contains("上一个") -> SmartCommand.PreviousSong
            lowerText.contains("调高") || lowerText.contains("大声") || lowerText.contains("大点") ||
            lowerText.contains("音量高") || lowerText.contains("声音大") || lowerText.contains("加大") -> {
                SmartCommand.VolumeUp
            }
            lowerText.contains("调低") || lowerText.contains("小声") || lowerText.contains("小点") ||
            lowerText.contains("音量低") || lowerText.contains("声音小") || lowerText.contains("减小") -> {
                SmartCommand.VolumeDown
            }
            lowerText.contains("音量") || lowerText.contains("volume") -> {
                val vol = Regex("\\d+").find(text)?.value?.toIntOrNull() ?: 50
                SmartCommand.SetVolume(vol)
            }

            else -> SmartCommand.Unknown(text)
        }
    }

    private fun extractArtistAndSong(text: String): Pair<String, String> {
        var content = text
            .replace(Regex("(请|帮我|给我|想要|要|我要)?(播放|放一首|放首|放歌|放一个)", RegexOption.IGNORE_CASE), "")
            .replace(Regex("(听一首|听首|听歌|想听|来一首|来首)", RegexOption.IGNORE_CASE), "")
            .replace(Regex("play", RegexOption.IGNORE_CASE), "")
            .trim()

        return if (content.contains("的")) {
            val parts = content.split("的", limit = 2)
            val possibleArtist = parts[0].trim()
            val possibleSong = parts.getOrElse(1) { "" }.trim()
            if (possibleSong.isNotEmpty()) {
                // "周杰伦的稻香" → artist=周杰伦, song=稻香
                Pair(possibleArtist, possibleSong)
            } else {
                // "稻香的" → song=稻香
                Pair("", possibleArtist)
            }
        } else {
            // 没有"的"分隔符，整段作为歌名（不是歌手）
            // "别怕我伤心" → artist="", song="别怕我伤心"
            Pair("", content)
        }
    }



    /**
     * 释放资源
     */
    fun release() {
        modelManager.release()
    }

    // ── JsonObject 扩展工具 ──

    private fun JsonObject.optString(key: String, default: String = ""): String {
        return try { get(key)?.asString ?: default } catch (_: Exception) { default }
    }

    private fun JsonObject.optInt(key: String, default: Int = 0): Int {
        return try { get(key)?.asInt ?: default } catch (_: Exception) { default }
    }
}
