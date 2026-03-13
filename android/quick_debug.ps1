# Bolt Android 快速调试脚本

$adb = "C:\Android\platform-tools\adb.exe"
$apk = "E:\devAI\boltassistant\android\app\build\outputs\apk\debug\app-debug.apk"

Write-Host "`n=== Bolt 调试工具 ===`n" -ForegroundColor Cyan

# 检查设备
Write-Host "[1] 检查设备..." -ForegroundColor Yellow
& $adb devices

$devices = & $adb devices | Select-String "device$"
if ($devices.Count -eq 0) {
    Write-Host "`n未检测到设备。请连接手机并启用USB调试，然后按回车..." -ForegroundColor Red
    Read-Host
    & $adb wait-for-device
}

# 获取设备信息
Write-Host "`n[2] 设备信息:" -ForegroundColor Yellow
Write-Host "型号: $(&$adb shell getprop ro.product.model)"
Write-Host "Android: $(&$adb shell getprop ro.build.version.release)"

# 安装APK
Write-Host "`n[3] 安装APK..." -ForegroundColor Yellow
Write-Host "APK: $apk"
& $adb install -r $apk

# 清空日志
Write-Host "`n[4] 清空旧日志..." -ForegroundColor Yellow
& $adb logcat -c

# 启动APP
Write-Host "`n[5] 启动应用..." -ForegroundColor Yellow
& $adb shell am start -n com.xiaotian.assistant/.MainActivity

Start-Sleep -Seconds 2

# 查看日志
Write-Host "`n[6] 实时日志 (Ctrl+C停止):`n" -ForegroundColor Cyan
& $adb logcat | Select-String -Pattern "Bolt|MainActivity|FATAL|AndroidRuntime|Exception"
