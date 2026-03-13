package com.xiaotian.assistant

import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.BufferedReader
import java.io.InputStreamReader
import com.google.gson.Gson
import com.google.gson.JsonObject

/**
 * ExecuTorch LLM 推理管理器
 *
 * 使用 PyTorch ExecuTorch 运行时加载 .pte 模型进行推理。
 * 支持多后端自动切换:
 *   1. QNN (Qualcomm 骁龙 NPU) - 骁龙 8 Gen1+ 设备
 *   2. MTK (MediaTek 天玑 NPU) - 天玑 9000+ 设备
 *   3. XNNPACK (CPU 通用) - 所有 ARM 设备 (保底)
 *
 * 线程安全: 单例模式，所有推理操作在 IO 调度器上执行
 */
class ExecuTorchModelManager private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ExecuTorchModelManager"

        // 模型文件名 (按 NPU 优先级排序)
        private const val MODEL_KVCACHE = "model_kvcache.pte"  // KV-cache 版本 (优先)
        private const val MODEL_QNN = "model_qnn.pte"          // Qualcomm QNN
        private const val MODEL_MTK = "model_mtk.pte"          // MediaTek NeuroPilot
        private const val MODEL_XNNPACK = "model_xnnpack.pte"  // XNNPACK CPU (旧版, 无缓存)

        // 配置
        private const val TOKENIZER_ASSET = "tokenizer.json"
        private const val MODEL_CONFIG_ASSET = "model_config.json"
        private const val MAX_OUTPUT_TOKENS = 512
        private const val MAX_SEQ_LEN = 2048
        private const val LOG_DECODE_EVERY_N_STEPS = 50
        private const val ENABLE_PER_STEP_LOGS = false

        @Volatile
        private var instance: ExecuTorchModelManager? = null

        fun getInstance(context: Context): ExecuTorchModelManager {
            return instance ?: synchronized(this) {
                instance ?: ExecuTorchModelManager(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }

    // ExecuTorch Module 实例
    private var module: Module? = null

    // 当前使用的后端
    private var activeBackend: String = "unknown"

    // KV-cache 模式标志
    private var hasKVCache: Boolean = false
    private var maxCacheLen: Int = 128
    private var supportsBatchPrefill: Boolean = false
    private var xnnpackThreads: Int = 4  // XNNPACK 线程数 (从 model_config.json 读取)

    // ══ 性能优化: 预分配 Tensor 避免 GC 抖动 ══
    // Decode 阶段每步重复使用, 避免每次分配新对象
    private val decodeIdBuffer = LongArray(1)
    private val decodePosBuffer = LongArray(1)
    private val decodeIdShape = longArrayOf(1, 1)
    private val decodePosShape = longArrayOf(1)

    // Tokenizer (BPE, 基于 tokenizer.json)
    private var vocabMap: HashMap<String, Int> = HashMap(16384)  // 预分配容量避免 rehash
    private var idToToken: HashMap<Int, String> = HashMap(16384)
    private var bpeMerges: List<Pair<String, String>> = emptyList()  // BPE 合并规则 (按优先级排序)
    private var mergeRank: HashMap<Pair<String, String>, Int> = HashMap(32768)  // pair → rank
    private var bosTokenId: Int = 2   // <bos> default for Gemma
    private var eosTokenId: Int = 1   // <eos> default for Gemma
    private var stopTokenIds: Set<Int> = emptySet()
    // 预计算的特殊 token 集合 (避免每次 tokenize 重建)
    private val specialTokenList = listOf(
        "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>",
        "<pad>", "<unk>", "<mask>", "[multimodal]", "\n"
    )
    private val specialTokenSet = specialTokenList.toSet()

    // Prompt 模板
    private var promptTemplate: String = "<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"

    // 早停检测: 支持两种格式
    // 1. <function_call>{...}</function_call> (旧格式)
    // 2. {"f":"...","p":{...}} (新格式, 无标签, 检测 JSON 花括号平衡)
    private val FUNCTION_CALL_END = "</function_call>"
    private var closeBracketTokenId: Int = -1  // > 的 token id
    private var closeBraceTokenId: Int = -1   // } 的 token id, 用于 JSON 结束检测

    // 推理锁
    private val inferenceLock = Object()

    // 模型是否就绪
    val isLoaded: Boolean get() = module != null

    /**
     * 初始化 (检测芯片 → 选择最优后端 → 加载模型)
     */
    suspend fun initialize(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            if (isLoaded) {
                Log.i(TAG, "Model already loaded (backend=$activeBackend)")
                return@withContext Result.success(Unit)
            }

            // Step 1: 检测设备芯片, 选择最优后端
            val chipInfo = detectChipset()
            Log.i(TAG, "Chipset detected: $chipInfo")

            // Step 2: 按优先级尝试加载模型
            val modelFile = selectAndCopyModel(chipInfo)
            if (modelFile == null) {
                return@withContext Result.failure(
                    IllegalStateException(
                        "没有找到可用的模型文件。请先完成模型转换 (convert_to_executorch.py) 并将 .pte 放入 assets/"
                    )
                )
            }

            // Step 3: 加载 tokenizer
            loadTokenizer()

            // Step 4: 加载 ExecuTorch 模型
            Log.i(TAG, "Loading ExecuTorch model: ${modelFile.name} (backend=$activeBackend)")
            module = Module.load(modelFile.absolutePath)
            
            // 尝试设置线程数 (优化 CPU 推理速度)
            try {
                // 使用反射调用 setNumThreads (如果存在) - 适用于部分版本的 ExecuTorch/PyTorch Android
                // XNNPACK 后端通常受益于 4 线程 (大核数量)
                val setNumThreads = module?.javaClass?.getMethod("setNumThreads", Int::class.javaPrimitiveType)
                if (setNumThreads != null) {
                    setNumThreads.invoke(module, xnnpackThreads)
                    Log.i(TAG, "Success: Set numThreads to $xnnpackThreads")
                } else {
                     Log.i(TAG, "Note: setNumThreads method not found, using default threading")
                }
            } catch (e: Exception) {
                // 备用: 设置环境变量 (XNNPACK 会读取)
                try {
                    val field = Runtime::class.java.getDeclaredField("availableProcessors")
                    // XNNPACK respects OMP_NUM_THREADS
                } catch (_: Exception) {}
                Log.w(TAG, "Failed to set numThreads: ${e.message}")
            }

            Log.i(TAG, "ExecuTorch model loaded successfully (backend=$activeBackend)")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            Result.failure(e)
        }
    }

    /**
     * 生成文本 (Function Calling 推理)
     *
     * 输入: 用户指令 (如 "播放周杰伦的稻香")
     * 输出: Function Call JSON (如 <function_call>{"function":"play_song",...}</function_call>)
     */
    suspend fun generate(userQuery: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            val mod = module
            if (mod == null) {
                return@withContext Result.failure(RuntimeException("Model not loaded"))
            }

            // 构建 prompt
            val prompt = buildPrompt(userQuery)
            Log.d(TAG, "Generating with prompt (${prompt.length} chars)...")

            val startTime = System.currentTimeMillis()

            // Tokenize
            val inputIds = tokenize(prompt)
            Log.d(TAG, "Input tokens: ${inputIds.size}")

            // 推理
            val result = synchronized(inferenceLock) {
                generateTokens(mod, inputIds)
            }

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "Generation completed in ${elapsed}ms: $result")

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Generation failed", e)
            Result.failure(e)
        }
    }

    /**
     * 释放模型资源
     */
    fun release() {
        try {
            module?.destroy()
            module = null
            activeBackend = "unknown"
            Log.i(TAG, "Model released")
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing model", e)
        }
    }

    // ══════════════════════════════════════════════
    //  芯片检测 & 后端选择
    // ══════════════════════════════════════════════

    data class ChipInfo(
        val isQualcomm: Boolean,
        val isMediaTek: Boolean,
        val socModel: String,
        val hardware: String,
    )

    private fun detectChipset(): ChipInfo {
        val hardware = Build.HARDWARE.lowercase()
        val board = Build.BOARD.lowercase()
        val soc = Build.SOC_MODEL.lowercase()
        val manufacturer = Build.SOC_MANUFACTURER.lowercase()

        val isQualcomm = manufacturer.contains("qualcomm") ||
                hardware.contains("qcom") ||
                board.contains("sm") ||
                soc.contains("sm") ||
                hardware.contains("kona") ||
                hardware.contains("lahaina") ||
                hardware.contains("taro") ||
                hardware.contains("kalama") ||
                hardware.contains("pineapple")

        val isMediaTek = manufacturer.contains("mediatek") ||
                hardware.contains("mt") ||
                soc.contains("mt") ||
                hardware.contains("dimensity")

        return ChipInfo(
            isQualcomm = isQualcomm,
            isMediaTek = isMediaTek,
            socModel = Build.SOC_MODEL,
            hardware = Build.HARDWARE,
        )
    }

    /**
     * 按优先级选择并复制模型文件
     *
     * 优先级:
     *   骁龙设备: QNN → XNNPACK
     *   天玑设备: MTK → XNNPACK
     *   其他设备: XNNPACK
     */
    private fun selectAndCopyModel(chipInfo: ChipInfo): File? {
        val candidates = mutableListOf<String>()

        // KV-cache 版本优先 (最快)
        // XNNPACK+KVCache 优先 (model_config.json 会覆盖 hasKVCache)
        candidates.add(MODEL_XNNPACK)
        candidates.add(MODEL_KVCACHE)

        if (chipInfo.isQualcomm) {
            candidates.add(MODEL_QNN)
            Log.i(TAG, "Qualcomm chip detected (${chipInfo.socModel}), prefer QNN backend")
        }
        if (chipInfo.isMediaTek) {
            candidates.add(MODEL_MTK)
            Log.i(TAG, "MediaTek chip detected (${chipInfo.socModel}), prefer MTK backend")
        }

        for (modelName in candidates) {
            val hasAsset = try {
                context.assets.open(modelName).use { it.read(); true }
            } catch (e: Exception) {
                false
            }

            if (hasAsset) {
                val modelFile = File(context.filesDir, modelName)
                // 检查 assets 中模型大小 (用于检测更新)
                val assetSize = try {
                    context.assets.openFd(modelName).use { it.length }
                } catch (e: Exception) {
                    // 压缩 assets 无法用 openFd, 用流计算
                    try {
                        context.assets.open(modelName).use { input ->
                            var total = 0L
                            val buf = ByteArray(8192)
                            var n: Int
                            while (input.read(buf).also { n = it } != -1) { total += n }
                            total
                        }
                    } catch (_: Exception) { -1L }
                }

                val needsCopy = !modelFile.exists() ||
                        (assetSize > 0 && modelFile.length() != assetSize) ||
                        modelFile.lastModified() < getApkLastModified(context)
                if (needsCopy) {
                    if (modelFile.exists()) {
                        Log.i(TAG, "Model updated in assets (local=${modelFile.length()}, asset=$assetSize), re-copying")
                        modelFile.delete()
                    }
                    Log.i(TAG, "Copying model from assets: $modelName")
                    copyFromAssets(modelName, modelFile)
                }

                // 验证文件大小
                if (modelFile.length() < 1024 * 1024) {
                    Log.e(TAG, "Model file too small: ${modelFile.length()} bytes, skipping")
                    modelFile.delete()
                    continue
                }

                activeBackend = when (modelName) {
                    MODEL_KVCACHE -> "XNNPACK+KVCache"
                    MODEL_QNN -> "QNN (Qualcomm NPU)"
                    MODEL_MTK -> "MTK (MediaTek NPU)"
                    MODEL_XNNPACK -> "XNNPACK (CPU)"
                    else -> modelName
                }
                hasKVCache = (modelName == MODEL_KVCACHE)
                Log.i(TAG, "Selected model: $modelName (${modelFile.length() / 1024 / 1024}MB), backend=$activeBackend")
                return modelFile
            }
        }

        return null
    }

    // ══════════════════════════════════════════════
    //  Tokenizer
    // ══════════════════════════════════════════════

    private fun loadTokenizer() {
        try {
            // 尝试加载 tokenizer.json
            val tokenizerJson = context.assets.open(TOKENIZER_ASSET).use { input ->
                BufferedReader(InputStreamReader(input)).readText()
            }

            val gson = Gson()
            val root = gson.fromJson(tokenizerJson, JsonObject::class.java)

            // 解析词表 (tokenizer.json -> model -> vocab)
            val vocab = HashMap<String, Int>(16384)
            val reverse = HashMap<Int, String>(16384)

            if (root.has("model") && root.getAsJsonObject("model").has("vocab")) {
                val vocabObj = root.getAsJsonObject("model").getAsJsonObject("vocab")
                for ((key, value) in vocabObj.entrySet()) {
                    val id = value.asInt
                    vocab[key] = id
                    reverse[id] = key
                }
            } else if (root.has("added_tokens")) {
                // 备用格式
                val addedTokens = root.getAsJsonArray("added_tokens")
                for (element in addedTokens) {
                    val obj = element.asJsonObject
                    val content = obj.get("content").asString
                    val id = obj.get("id").asInt
                    vocab[content] = id
                    reverse[id] = content
                }
            }

            vocabMap = vocab
            idToToken = reverse

            // 解析 BPE 合并规则 (支持 array 和 string 两种格式)
            if (root.has("model") && root.getAsJsonObject("model").has("merges")) {
                val mergesArr = root.getAsJsonObject("model").getAsJsonArray("merges")
                val mergesList = mutableListOf<Pair<String, String>>()
                val rankMap = HashMap<Pair<String, String>, Int>(mergesArr.size() * 2)
                for (i in 0 until mergesArr.size()) {
                    val elem = mergesArr[i]
                    val pair: Pair<String, String>? = if (elem.isJsonArray) {
                        // 数组格式: ["播", "放"]
                        val arr = elem.asJsonArray
                        if (arr.size() == 2) Pair(arr[0].asString, arr[1].asString) else null
                    } else {
                        // 字符串格式: "播 放"
                        val parts = elem.asString.split(" ", limit = 2)
                        if (parts.size == 2) Pair(parts[0], parts[1]) else null
                    }
                    if (pair != null) {
                        mergesList.add(pair)
                        rankMap[pair] = i
                    }
                }
                bpeMerges = mergesList
                mergeRank = rankMap
                Log.i(TAG, "BPE merges loaded: ${mergesList.size}")
            }

            // 查找特殊 token ID
            bosTokenId = vocab["<bos>"] ?: vocab["<s>"] ?: 2
            eosTokenId = vocab["<eos>"] ?: vocab["</s>"] ?: 1
            val endOfTurnId = vocab["<end_of_turn>"]
            stopTokenIds = setOfNotNull(eosTokenId, endOfTurnId)
            closeBracketTokenId = vocab[">"] ?: -1
            closeBraceTokenId = vocab["}"] ?: -1

            Log.i(TAG, "Tokenizer loaded: ${vocab.size} tokens, ${bpeMerges.size} merges, bos=$bosTokenId, eos=$eosTokenId, stops=$stopTokenIds")

        } catch (e: Exception) {
            Log.w(TAG, "Failed to load tokenizer.json, using fallback", e)
        }

        // 加载模型配置
        try {
            val configJson = context.assets.open(MODEL_CONFIG_ASSET).use { input ->
                BufferedReader(InputStreamReader(input)).readText()
            }
            val gson = Gson()
            val config = gson.fromJson(configJson, JsonObject::class.java)
            if (config.has("prompt_template")) {
                promptTemplate = config.get("prompt_template").asString
            }
            if (config.has("has_kv_cache")) {
                hasKVCache = config.get("has_kv_cache").asBoolean
            }
            if (config.has("max_cache_len")) {
                maxCacheLen = config.get("max_cache_len").asInt
            }
            if (config.has("supports_batch_prefill")) {
                supportsBatchPrefill = config.get("supports_batch_prefill").asBoolean
            }
            if (config.has("xnnpack_threads")) {
                xnnpackThreads = config.get("xnnpack_threads").asInt
            }
            Log.i(TAG, "Model config loaded (hasKVCache=$hasKVCache, maxCacheLen=$maxCacheLen, batchPrefill=$supportsBatchPrefill, threads=$xnnpackThreads)")
        } catch (e: Exception) {
            Log.d(TAG, "No model_config.json, using defaults")
        }
    }

    /**
     * BPE Tokenizer: 先处理特殊 token, 再按字节拆分, 最后应用 BPE 合并
     */
    private fun tokenize(text: String): LongArray {
        if (vocabMap.isEmpty()) {
            Log.w(TAG, "Tokenizer not loaded, using byte-level fallback")
            val bytes = text.toByteArray(Charsets.UTF_8)
            return LongArray(bytes.size) { bytes[it].toLong() and 0xFF }
        }

        val tokens = mutableListOf<Long>()
        tokens.add(bosTokenId.toLong())

        // 按特殊 token 分割文本, 保留特殊 token
        val segments = splitBySpecialTokens(text, specialTokenList)

        for (segment in segments) {
            if (segment in specialTokenSet) {
                // 特殊 token: 直接查找 vocab ID
                val id = vocabMap[segment]
                if (id != null) {
                    tokens.add(id.toLong())
                }
            } else {
                // 普通文本: 应用 normalizer (空格→▁), 然后 BPE 编码
                val normalized = segment.replace(" ", "▁")
                val bpeTokens = bpeEncode(normalized)
                tokens.addAll(bpeTokens)
            }
        }

        val result = tokens.toLongArray()
        Log.d(TAG, "Tokenized ${text.length} chars → ${result.size} tokens: ${result.take(30).toList()}")
        return result
    }

    /**
     * 按特殊 token 分割文本, 返回交替的 [普通文本, 特殊token, 普通文本, ...] 序列
     */
    private fun splitBySpecialTokens(text: String, specials: List<String>): List<String> {
        val result = mutableListOf<String>()
        var remaining = text

        while (remaining.isNotEmpty()) {
            var bestIdx = Int.MAX_VALUE
            var bestSpecial = ""

            for (sp in specials) {
                val idx = remaining.indexOf(sp)
                if (idx in 0 until bestIdx) {
                    bestIdx = idx
                    bestSpecial = sp
                }
            }

            if (bestSpecial.isEmpty()) {
                // 没有更多特殊 token
                if (remaining.isNotEmpty()) result.add(remaining)
                break
            }

            if (bestIdx > 0) {
                result.add(remaining.substring(0, bestIdx))
            }
            result.add(bestSpecial)
            remaining = remaining.substring(bestIdx + bestSpecial.length)
        }

        return result
    }

    /**
     * BPE 编码: 文本 → 字符 token 序列 → 应用合并规则
     * Gemma BPE 用字符级初始 token (非字节级), 未知字符用 <0xHH> 字节回退
     */
    private fun bpeEncode(text: String): List<Long> {
        if (text.isEmpty()) return emptyList()

        // Step 1: 将文本拆分为初始 symbol 列表 (字符级)
        val symbols = mutableListOf<String>()
        for (ch in text) {
            val charStr = ch.toString()
            if (vocabMap.containsKey(charStr)) {
                symbols.add(charStr)
            } else {
                // 字符不在 vocab 中, 用字节级回退 <0xHH>
                val bytes = charStr.toByteArray(Charsets.UTF_8)
                for (b in bytes) {
                    val hexStr = String.format("<0x%02X>", b.toInt() and 0xFF)
                    symbols.add(hexStr)
                }
            }
        }

        // Step 2: 迭代应用 BPE 合并 (按优先级)
        if (bpeMerges.isNotEmpty()) {
            while (symbols.size >= 2) {
                // 找当前序列中 rank 最小 (最高优先级) 的相邻对
                var bestRank = Int.MAX_VALUE
                var bestIdx = -1
                for (i in 0 until symbols.size - 1) {
                    val pair = Pair(symbols[i], symbols[i + 1])
                    val rank = mergeRank[pair]
                    if (rank != null && rank < bestRank) {
                        bestRank = rank
                        bestIdx = i
                    }
                }
                if (bestIdx < 0) break  // 没有可合并的对了

                // 合并
                val merged = symbols[bestIdx] + symbols[bestIdx + 1]
                symbols[bestIdx] = merged
                symbols.removeAt(bestIdx + 1)
            }
        }

        // Step 3: 将 symbol 转换为 token ID
        val result = mutableListOf<Long>()
        for (sym in symbols) {
            val id = vocabMap[sym]
            if (id != null) {
                result.add(id.toLong())
            } else {
                // 未知 token, 尝试逐字节回退
                Log.w(TAG, "Unknown BPE token: ${sym.take(20)}")
                val unkId = vocabMap["<unk>"] ?: 3
                result.add(unkId.toLong())
            }
        }

        return result
    }

    /**
     * 将 token ID 序列解码为文本
     * 处理 BPE byte-level tokens (<0xHH>) 和 SentencePiece ▁ 前缀
     */
    private fun detokenize(tokenIds: List<Int>): String {
        if (idToToken.isEmpty()) return ""

        val bytes = mutableListOf<Byte>()
        val sb = StringBuilder()

        for (id in tokenIds) {
            val token = idToToken[id] ?: continue
            // 跳过特殊 token
            if (token in listOf("<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
                    "[multimodal]", "<start_of_turn>", "<end_of_turn>")) continue

            // 解析 token 内容中的字节级子 token
            var i = 0
            while (i < token.length) {
                if (i + 5 < token.length && token[i] == '<' && token[i + 1] == '0' && token[i + 2] == 'x') {
                    // <0xHH> 格式的字节 token
                    val endIdx = token.indexOf('>', i)
                    if (endIdx > i) {
                        val hex = token.substring(i + 3, endIdx)
                        try {
                            bytes.add(hex.toInt(16).toByte())
                        } catch (_: Exception) {
                            // 不是有效 hex
                            sb.append(token.substring(i, endIdx + 1))
                        }
                        i = endIdx + 1
                        continue
                    }
                }
                // 普通字符 - 先 flush 字节缓冲
                if (bytes.isNotEmpty()) {
                    sb.append(String(bytes.toByteArray(), Charsets.UTF_8))
                    bytes.clear()
                }
                if (token[i] == '▁') {
                    sb.append(' ')
                } else {
                    sb.append(token[i])
                }
                i++
            }
        }
        // 最终 flush
        if (bytes.isNotEmpty()) {
            sb.append(String(bytes.toByteArray(), Charsets.UTF_8))
        }

        return sb.toString().trim()
    }

    // ══════════════════════════════════════════════
    //  推理
    // ══════════════════════════════════════════════

    /**
     * 自回归生成 token (自动选择 KV-cache 或旧版推理路径)
     */
    private fun generateTokens(mod: Module, inputIds: LongArray): String {
        return if (hasKVCache) {
            generateTokensKVCache(mod, inputIds)
        } else {
            generateTokensLegacy(mod, inputIds)
        }
    }

    /**
     * KV-Cache 推理路径
     *
     * 模型接口: forward(input_ids[1,seq], cache_position[seq]) → logits[1,seq,vocab]
     * 缓存内置在模型 register_buffer 中, 通过 in-place 操作自动更新
     *
     * 流程:
     *   1. Prefill: 批量处理所有 prompt tokens (一次 forward 调用)
     *   2. Decode:  逐 token 生成 (位置 N, N+1, ...)
     *
     * 性能优化:
     *   - 批量 prefill: 将所有 prompt tokens 合并为一次 forward, 大幅减少调用次数
     *   - 高优先级线程: 提升 CPU 调度优先级以尽量使用大核
     */
    private fun generateTokensKVCache(mod: Module, inputIds: LongArray): String {
        val generatedTokens = mutableListOf<Int>()
        val promptLen = inputIds.size

        Log.d(TAG, "KV-Cache inference: prompt=$promptLen tokens, maxCache=$maxCacheLen")

        // 限制 prompt 长度: 保留至少 2 个位置给生成的 token
        val maxPrompt = maxOf(maxCacheLen - 2, 1)
        val effectiveLen = minOf(promptLen, maxPrompt)

        // 提升线程优先级以利用大核
        val origPriority = android.os.Process.getThreadPriority(android.os.Process.myTid())
        try {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to boost thread priority: ${e.message}")
        }

        // ── Prefill: 批量或逐 token 处理 prompt ──
        var lastLogits: FloatArray? = null
        val prefillStart = System.currentTimeMillis()

        if (supportsBatchPrefill) {
            // ── Batch Prefill: 一次 forward 处理所有 prompt tokens ──
            val batchIds = LongArray(effectiveLen) { inputIds[promptLen - effectiveLen + it.toInt()] }
            val batchPos = LongArray(effectiveLen) { it.toLong() }

            val idTensor = Tensor.fromBlob(batchIds, longArrayOf(1, effectiveLen.toLong()))
            val posTensor = Tensor.fromBlob(batchPos, longArrayOf(effectiveLen.toLong()))

            val output = mod.forward(
                EValue.from(idTensor),
                EValue.from(posTensor)
            )
            lastLogits = output[0].toTensor().dataAsFloatArray
            // logits shape: [1, effectiveLen, vocab] — 取最后一个 token 的 logits
            val vocabSize = lastLogits.size / effectiveLen
            val lastTokenLogits = FloatArray(vocabSize)
            System.arraycopy(lastLogits, (effectiveLen - 1) * vocabSize, lastTokenLogits, 0, vocabSize)
            lastLogits = lastTokenLogits
        } else {
            // ── 逐 token Prefill (fallback) ──
            val idShape = longArrayOf(1, 1)
            val posShape = longArrayOf(1)

            for (i in 0 until effectiveLen) {
                val idTensor = Tensor.fromBlob(
                    longArrayOf(inputIds[promptLen - effectiveLen + i]),
                    idShape
                )
                val posTensor = Tensor.fromBlob(
                    longArrayOf(i.toLong()),
                    posShape
                )
                val output = mod.forward(
                    EValue.from(idTensor),
                    EValue.from(posTensor)
                )
                lastLogits = output[0].toTensor().dataAsFloatArray
            }
        }

        val prefillMs = System.currentTimeMillis() - prefillStart
        Log.d(TAG, "Prefill done: ${effectiveLen} tokens in ${prefillMs}ms" +
            " (${if (effectiveLen > 0) prefillMs / effectiveLen else 0}ms/token" +
            ", batch=${supportsBatchPrefill})")

        if (lastLogits == null) {
            Log.e(TAG, "Empty prompt!")
            return ""
        }

        // 第一个生成 token
        var nextToken = argmax(lastLogits)
        if (nextToken in stopTokenIds) {
            Log.d(TAG, "Stop token at first position")
            return ""
        }
        generatedTokens.add(nextToken)
        Log.d(TAG, "First token: $nextToken (${idToToken[nextToken] ?: "?"})")

        // ── 早停检测: 增量拼接文本, 检测 </function_call> 结尾 ──
        // 当检测到完整的 </function_call> 时立即停止, 省掉 <end_of_turn> 等后续 token
        val textBuilder = StringBuilder()
        val firstTokenText = idToToken[nextToken] ?: ""
        textBuilder.append(firstTokenText)

        // ── Decode: 逐 token 生成 (使用预分配 Tensor 避免 GC 抖动) ──
        val decodeStart = System.currentTimeMillis()

        for (step in 1 until MAX_OUTPUT_TOKENS) {
            val pos = effectiveLen + step - 1  // 当前写入的缓存位置
            if (pos >= maxCacheLen) {
                Log.d(TAG, "Cache full at step $step")
                break
            }

            // 复用预分配的 buffer, 避免每步分配新 LongArray
            decodeIdBuffer[0] = nextToken.toLong()
            decodePosBuffer[0] = pos.toLong()

            val idTensor = Tensor.fromBlob(decodeIdBuffer, decodeIdShape)
            val posTensor = Tensor.fromBlob(decodePosBuffer, decodePosShape)

            val output = mod.forward(
                EValue.from(idTensor),
                EValue.from(posTensor)
            )

            val logits = output[0].toTensor().dataAsFloatArray
            nextToken = argmax(logits)

            if (nextToken in stopTokenIds) {
                Log.d(TAG, "Stop token $nextToken at step $step")
                break
            }

            generatedTokens.add(nextToken)

            // ── 早停: 检测 JSON 结束或 </function_call> 结束 ──
            val tokenText = idToToken[nextToken] ?: ""
            textBuilder.append(tokenText)

            // 快速路径: 在 token 包含 } 或 > 时检查, 避免每步做字符串操作
            // 注意: BPE 可能将 }} 或 "}} 合并为单个 token, 不能只匹配单字符 token ID
            if (generatedTokens.size > 10) {
                if ('}' in tokenText) {
                    // 检测花括号平衡 (支持新格式纯 JSON 和旧格式 <function_call>)
                    val currentText = textBuilder.toString()
                    val openCount = currentText.count { it == '{' }
                    val closeCount = currentText.count { it == '}' }
                    if (openCount > 0 && openCount == closeCount) {
                        Log.d(TAG, "Early stop: JSON complete at step $step, braces balanced ($openCount pairs)")
                        break
                    }
                } else if ('>' in tokenText) {
                    // 旧格式: 检测 </function_call> 结尾
                    if (textBuilder.endsWith(FUNCTION_CALL_END)) {
                        Log.d(TAG, "Early stop: </function_call> detected at step $step")
                        break
                    }
                }
            }

            if (ENABLE_PER_STEP_LOGS && step % LOG_DECODE_EVERY_N_STEPS == 0) {
                val elapsed = System.currentTimeMillis() - decodeStart
                Log.d(TAG, "Decode step $step: ${elapsed / step}ms/token")
            }

            // Function call 通常 <100 tokens
            if (generatedTokens.size > 100) {
                Log.d(TAG, "Max generation length reached")
                break
            }
        }

        val decodeMs = System.currentTimeMillis() - decodeStart
        val totalMs = System.currentTimeMillis() - prefillStart
        val totalTokens = generatedTokens.size

        // 恢复线程优先级
        try {
            android.os.Process.setThreadPriority(origPriority)
        } catch (e: Exception) { /* ignore */ }

        Log.i(TAG, "KV-Cache generation done: $totalTokens tokens, " +
                "total=${totalMs}ms, decode=${decodeMs}ms, " +
                "speed=${if (totalTokens > 0) (decodeMs / totalTokens) else 0}ms/token")

        return detokenize(generatedTokens)
    }

    /**
     * Argmax over logits array
     */
    private fun argmax(logits: FloatArray): Int {
        var maxIdx = 0
        var maxVal = Float.NEGATIVE_INFINITY
        for (j in logits.indices) {
            if (logits[j] > maxVal) {
                maxVal = logits[j]
                maxIdx = j
            }
        }
        return maxIdx
    }

    /**
     * 旧版推理路径 (无 KV-cache, 每步处理完整序列)
     *
     * 模型接口: forward(input_ids[1,64], attention_mask[1,64]) → logits[1,64,vocab]
     */
    private fun generateTokensLegacy(mod: Module, inputIds: LongArray): String {
        val generatedTokens = mutableListOf<Int>()

        val MAX_INPUT_LEN = 64

        val paddedInput = LongArray(MAX_INPUT_LEN) { 0L }
        val attentionMask = LongArray(MAX_INPUT_LEN) { 0L }
        val actualLen = minOf(inputIds.size, MAX_INPUT_LEN)
        
        for (i in 0 until actualLen) {
            paddedInput[MAX_INPUT_LEN - actualLen + i] = inputIds[i]
            attentionMask[MAX_INPUT_LEN - actualLen + i] = 1L
        }

        var currentPaddedInput = paddedInput.clone()
        var currentAttMask = attentionMask.clone()

        for (step in 0 until MAX_OUTPUT_TOKENS) {
            val inputTensor = Tensor.fromBlob(
                currentPaddedInput,
                longArrayOf(1, MAX_INPUT_LEN.toLong())
            )
            val maskTensor = Tensor.fromBlob(
                currentAttMask,
                longArrayOf(1, MAX_INPUT_LEN.toLong())
            )

            val output = mod.forward(
                EValue.from(inputTensor),
                EValue.from(maskTensor)
            )
            val outputTensor = output[0].toTensor()

            val logits = outputTensor.dataAsFloatArray
            val vocabSize = logits.size / MAX_INPUT_LEN
            
            val lastValidPos = MAX_INPUT_LEN - 1
            val offset = lastValidPos * vocabSize
            val lastTokenLogits = logits.sliceArray(offset until offset + vocabSize)

            val maxIdx = argmax(lastTokenLogits)

            if (maxIdx in stopTokenIds) {
                Log.d(TAG, "Stop token $maxIdx at step $step")
                break
            }

            generatedTokens.add(maxIdx)
            if (ENABLE_PER_STEP_LOGS && step % LOG_DECODE_EVERY_N_STEPS == 0) {
                Log.d(TAG, "Step $step: token=$maxIdx (${idToToken[maxIdx] ?: "?"})")
            }

            currentPaddedInput = LongArray(MAX_INPUT_LEN) { i ->
                if (i < MAX_INPUT_LEN - 1) currentPaddedInput[i + 1] else maxIdx.toLong()
            }
            currentAttMask = LongArray(MAX_INPUT_LEN) { i ->
                if (i < MAX_INPUT_LEN - 1) currentAttMask[i + 1] else 1L
            }

            if (generatedTokens.size > 100) {
                Log.d(TAG, "Max generation length reached")
                break
            }
        }

        return detokenize(generatedTokens)
    }

    // ══════════════════════════════════════════════
    //  Prompt 构建
    // ══════════════════════════════════════════════

    /**
     * 构建 FunctionGemma prompt
     *
     * 与训练数据格式完全一致:
     *   <start_of_turn>user
     *   {query}
     *   <end_of_turn>
     *   <start_of_turn>model
     */
    private fun buildPrompt(userQuery: String): String {
        return promptTemplate.replace("{query}", userQuery)
    }

    // ══════════════════════════════════════════════
    //  文件工具
    // ══════════════════════════════════════════════

    private fun copyFromAssets(assetName: String, destFile: File) {
        context.assets.open(assetName).use { input ->
            FileOutputStream(destFile).use { output ->
                val buffer = ByteArray(8 * 1024 * 1024) // 8MB buffer
                var bytesRead: Int
                var totalCopied = 0L
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalCopied += bytesRead
                    if (totalCopied % (50 * 1024 * 1024) == 0L) {
                        Log.d(TAG, "  Copied ${totalCopied / 1024 / 1024}MB...")
                    }
                }
                Log.i(TAG, "Model copied: ${totalCopied / 1024 / 1024}MB")
            }
        }
    }

    /** 获取 APK 最后更新时间, 用于检测模型是否需要重新复制 */
    private fun getApkLastModified(ctx: android.content.Context): Long {
        return try {
            val info = ctx.packageManager.getPackageInfo(ctx.packageName, 0)
            info.lastUpdateTime
        } catch (e: Exception) {
            0L
        }
    }

    /**
     * 获取当前使用的后端信息
     */
    fun getBackendInfo(): String = activeBackend
}
