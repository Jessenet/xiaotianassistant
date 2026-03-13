@echo off
REM Android SDK 命令行工具自动安装脚本
REM 运行此脚本需要管理员权限

echo ========================================
echo Android SDK 命令行工具安装脚本
echo ========================================
echo.

set SDK_ROOT=C:\Android
set CMDLINE_TOOLS_URL=https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip
set DOWNLOAD_FILE=%TEMP%\android-cmdline-tools.zip

echo 步骤 1/4: 下载 Android SDK 命令行工具...
echo 下载地址: %CMDLINE_TOOLS_URL%
echo.

REM 使用PowerShell下载
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%CMDLINE_TOOLS_URL%' -OutFile '%DOWNLOAD_FILE%' -UseBasicParsing}"

if not exist "%DOWNLOAD_FILE%" (
    echo 错误: 下载失败
    echo 请手动下载: https://developer.android.com/studio#command-tools
    pause
    exit /b 1
)

echo ✓ 下载完成
echo.

echo 步骤 2/4: 解压文件...
if not exist "%SDK_ROOT%" mkdir "%SDK_ROOT%"
if not exist "%SDK_ROOT%\cmdline-tools" mkdir "%SDK_ROOT%\cmdline-tools"

REM 解压到临时目录
powershell -Command "& {Expand-Archive -Path '%DOWNLOAD_FILE%' -DestinationPath '%SDK_ROOT%\cmdline-tools' -Force}"

REM 重命名为latest
if exist "%SDK_ROOT%\cmdline-tools\cmdline-tools" (
    if exist "%SDK_ROOT%\cmdline-tools\latest" rmdir /s /q "%SDK_ROOT%\cmdline-tools\latest"
    move "%SDK_ROOT%\cmdline-tools\cmdline-tools" "%SDK_ROOT%\cmdline-tools\latest"
)

echo ✓ 解压完成
echo.

echo 步骤 3/4: 安装必需的 SDK 组件...
echo 这可能需要几分钟...
echo.

cd /d "%SDK_ROOT%\cmdline-tools\latest\bin"

REM 接受许可证
echo y | sdkmanager.bat --sdk_root=%SDK_ROOT% --licenses

REM 安装必需组件
call sdkmanager.bat --sdk_root=%SDK_ROOT% "platforms;android-34"
call sdkmanager.bat --sdk_root=%SDK_ROOT% "build-tools;34.0.0"
call sdkmanager.bat --sdk_root=%SDK_ROOT% "platform-tools"
call sdkmanager.bat --sdk_root=%SDK_ROOT% "tools"

echo ✓ SDK 组件安装完成
echo.

echo 步骤 4/4: 创建 local.properties...
cd /d e:\devAI\boltassistant\android
echo sdk.dir=C:/Android > local.properties

echo ✓ local.properties 已创建
echo.

echo ========================================
echo 安装完成！
echo ========================================
echo.
echo SDK 位置: %SDK_ROOT%
echo Android 34 已安装
echo Build Tools 34.0.0 已安装
echo.
echo 现在可以编译 Android APK 了！
echo 运行命令: cd android && gradlew.bat assembleDebug
echo.

REM 清理下载文件
del /f /q "%DOWNLOAD_FILE%"

pause
