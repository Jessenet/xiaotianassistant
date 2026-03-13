#!/bin/bash
# Linux/Mac 部署脚本 - 更新于 2026-02-24
# 双模型架构: SenseVoice (ASR) + ExecuTorch FunctionGemma (NLU)

echo "========================================="
echo "  语音AI音乐助手 部署脚本 (Linux/Mac)"
echo "========================================="
echo ""
echo "双模型架构:"
echo "  ASR:  Sherpa-ONNX + SenseVoice-Small INT8 (~230 MB)"
echo "  NLU:  ExecuTorch XNNPACK FunctionGemma INT8 (~76 MB)"
echo "  VAD:  Silero VAD (~0.6 MB)"
echo ""

ASSETS_DIR="android/app/src/main/assets"
NLU_SRC="conversion/executorch_output"
ASR_MODEL_DIR="${ASSETS_DIR}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

# 检查 Python 环境
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)

echo "选择部署模式:"
echo "  1. 快速部署 (使用已有的 ExecuTorch 模型)"
echo "  2. 重新导出模型 (合并 LoRA + ExecuTorch 导出，约 10-20 分钟)"
echo "  3. 完整流程 (训练 + 导出，约 1-2 小时)"
echo "  4. 下载 SenseVoice ASR 模型"
echo ""
read -p "请选择 (1/2/3/4): " choice

case $choice in
    # ─── 模式 1: 快速部署 ───────────────────────────────────────
    1)
        echo ""
        echo "=== 快速部署模式 ==="
        echo ""

        echo "检查 NLU 模型 (ExecuTorch)..."
        if [ ! -f "${NLU_SRC}/model_xnnpack.pte" ]; then
            echo "错误: 未找到 ExecuTorch 模型，请使用模式 2 重新导出"
            exit 1
        fi

        echo "复制 NLU 模型到 Android assets..."
        mkdir -p "${ASSETS_DIR}"
        cp -f "${NLU_SRC}/model_xnnpack.pte"    "${ASSETS_DIR}/"
        cp -f "${NLU_SRC}/tokenizer.json"       "${ASSETS_DIR}/"
        cp -f "${NLU_SRC}/tokenizer_config.json" "${ASSETS_DIR}/"
        cp -f "${NLU_SRC}/model_config.json"    "${ASSETS_DIR}/"

        echo ""
        echo "NLU 模型大小:"
        ls -lh "${ASSETS_DIR}/model_xnnpack.pte"

        echo ""
        echo "检查 ASR 模型 (SenseVoice)..."
        if [ ! -f "${ASR_MODEL_DIR}/model.int8.onnx" ]; then
            echo "警告: 未找到 SenseVoice ASR 模型"
            echo "请使用模式 4 下载，或手动下载到:"
            echo "  ${ASR_MODEL_DIR}/model.int8.onnx"
            echo "  ${ASR_MODEL_DIR}/tokens.txt"
        fi
        if [ ! -f "${ASSETS_DIR}/silero_vad.onnx" ]; then
            echo "警告: 未找到 Silero VAD 模型"
            echo "请手动下载 silero_vad.onnx 到 ${ASSETS_DIR}/"
        fi
        ;;

    # ─── 模式 2: 重新导出 ──────────────────────────────────────
    2)
        echo ""
        echo "=== 模型导出模式 (ExecuTorch) ==="
        echo ""
        cd conversion

        echo "运行统一导出脚本 (合并 LoRA + ExecuTorch INT8 量化)..."
        $PYTHON export_pipeline.py --copy

        if [ $? -ne 0 ]; then
            echo "错误: 模型导出失败"
            cd ..
            exit 1
        fi

        cd ..
        echo "模型已导出并复制到 Android assets"
        ;;

    # ─── 模式 3: 完整流程 ──────────────────────────────────────
    3)
        echo ""
        echo "=== 完整训练 + 导出模式 ==="
        echo "警告: 此过程需要 1-2 小时 (含 GPU 训练)，确定继续吗？"
        read -p "输入 YES 继续: " confirm
        if [ "$confirm" != "YES" ]; then
            echo "已取消"
            exit 0
        fi

        echo ""
        echo "步骤 1/2: 训练模型..."
        cd training
        $PYTHON train_simple.py

        if [ $? -ne 0 ]; then
            echo "错误: 模型训练失败"
            cd ..
            exit 1
        fi
        cd ..

        echo ""
        echo "步骤 2/2: 导出 ExecuTorch 模型..."
        cd conversion
        $PYTHON export_pipeline.py --copy

        if [ $? -ne 0 ]; then
            echo "错误: 模型导出失败"
            cd ..
            exit 1
        fi
        cd ..
        ;;

    # ─── 模式 4: 下载 ASR 模型 ────────────────────────────────
    4)
        echo ""
        echo "=== 下载 SenseVoice ASR 模型 ==="
        echo ""

        mkdir -p "${ASR_MODEL_DIR}"

        BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main"

        echo "下载 model.int8.onnx (~230 MB)，请耐心等待..."
        if command -v wget &> /dev/null; then
            wget -O "${ASR_MODEL_DIR}/model.int8.onnx" "${BASE_URL}/model.int8.onnx"
        else
            curl -L -o "${ASR_MODEL_DIR}/model.int8.onnx" "${BASE_URL}/model.int8.onnx"
        fi

        if [ $? -ne 0 ]; then
            echo "错误: model.int8.onnx 下载失败"
            exit 1
        fi

        echo "下载 tokens.txt..."
        if command -v wget &> /dev/null; then
            wget -O "${ASR_MODEL_DIR}/tokens.txt" "${BASE_URL}/tokens.txt"
        else
            curl -L -o "${ASR_MODEL_DIR}/tokens.txt" "${BASE_URL}/tokens.txt"
        fi

        if [ $? -ne 0 ]; then
            echo "错误: tokens.txt 下载失败"
            exit 1
        fi

        echo ""
        echo "SenseVoice 模型下载完成:"
        ls -lh "${ASR_MODEL_DIR}/model.int8.onnx"
        ls -lh "${ASR_MODEL_DIR}/tokens.txt"

        echo ""
        echo "检查 Silero VAD 模型..."
        if [ ! -f "${ASSETS_DIR}/silero_vad.onnx" ]; then
            echo "下载 silero_vad.onnx..."
            if command -v wget &> /dev/null; then
                wget -O "${ASSETS_DIR}/silero_vad.onnx" "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            else
                curl -L -o "${ASSETS_DIR}/silero_vad.onnx" "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            fi
        else
            echo "  silero_vad.onnx 已存在"
        fi

        echo ""
        echo "ASR 模型准备就绪！"
        ;;

    *)
        echo "无效选择，已取消"
        exit 1
        ;;
esac

# ─── 编译 Android APK ──────────────────────────────────────────
echo ""
echo "========================================="
echo "  编译 Android 应用"
echo "========================================="
echo ""
echo "1. 跳过编译 (仅复制模型)"
echo "2. 编译 Debug APK"
echo "3. 编译 + 安装到设备"
echo ""
read -p "请选择 (1/2/3): " build_choice

if [ "$build_choice" = "2" ] || [ "$build_choice" = "3" ]; then
    echo ""
    echo "编译 Debug APK..."
    cd android
    ./gradlew assembleDebug

    if [ $? -ne 0 ]; then
        echo "错误: 编译失败"
        cd ..
        exit 1
    fi

    echo ""
    echo "APK 位置: android/app/build/outputs/apk/debug/app-debug.apk"

    if [ "$build_choice" = "3" ]; then
        echo ""
        echo "安装到设备..."
        adb install -r app/build/outputs/apk/debug/app-debug.apk
        if [ $? -ne 0 ]; then
            echo "警告: 安装失败，请检查设备连接"
        else
            echo "安装成功！"
        fi
    fi
    cd ..
fi

echo ""
echo "========================================="
echo "  部署完成！"
echo "========================================="
echo ""
echo "支持 8 大语音指令:"
echo "  播放歌曲 / 暂停 / 继续 / 上一首 / 下一首"
echo "  设置音量 / 调高音量 / 调低音量"
echo ""
echo "应用功能:"
echo "  - SenseVoice 离线语音识别 (中/英/日/韩/粤)"
echo "  - ExecuTorch AI 语义理解 (FunctionGemma)"
echo "  - 语音唤醒 ('你好小天')"
echo "  - 支持 QQ 音乐 / 网易云音乐"
echo "  - 无障碍服务精确控制"
echo ""
echo "下一步:"
echo "  1. 安装 APK: adb install android/app/build/outputs/apk/debug/app-debug.apk"
echo "  2. 授予麦克风和无障碍权限"
echo "  3. 说 '你好小天' 唤醒，开始语音控制"
echo ""
echo "详细文档: README.md / android/AI_INTEGRATION_GUIDE.md"
echo ""
