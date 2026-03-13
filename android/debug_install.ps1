# Bolt Android APP 调试脚本
# 自动安装、启动并收集日志

$adb = "C:\Android\platform-tools\adb.exe"
$apk = "E:\devAI\boltassistant\android\app\build\outputs\apk\debug\app-debug.apk"
$package = "com.xiaotian.assistant"
$activity = "$package/.MainActivity"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Bolt APP 调试工具" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. 检查设备连接
Write-Host "[1/6] 检查设备连接..." -ForegroundColor Yellow
$devices = & $adb devices | Select-String "device$"
if ($devices.Count -eq 0) {
    Write-Host "× 未检测到设备，请:" -ForegroundColor Red
    Write-Host "  1. 连接USB线" -ForegroundColor White
    Write-Host "  2. 启用开发者选项 → USB调试" -ForegroundColor White
    Write-Host "  3. 允许电脑调试授权" -ForegroundColor White
    Write-Host "`n等待设备连接..." -ForegroundColor Yellow
    & $adb wait-for-device
    Write-Host "√ 设备已连接!" -ForegroundColor Green
} else {
    Write-Host "√ 设备已连接" -ForegroundColor Green
}

# 2. 显示设备信息
Write-Host "`n[2/6] 获取设备信息..." -ForegroundColor Yellow
$model = & $adb shell getprop ro.product.model
$version = & $adb shell getprop ro.build.version.release
$sdk = & $adb shell getprop ro.build.version.sdk
$memory = & $adb shell cat /proc/meminfo | Select-String "MemTotal" | ForEach-Object { $_ -replace '\D+(\d+).*', '$1' }
$memoryMB = [math]::Round([int]$memory / 1024)

Write-Host "  设备型号: $model" -ForegroundColor White
Write-Host "  Android版本: $version (API $sdk)" -ForegroundColor White
Write-Host "  内存: $memoryMB MB" -ForegroundColor White

# 3. 卸载旧版本
Write-Host "`n[3/6] 检查现有安装..." -ForegroundColor Yellow
$installed = & $adb shell pm list packages | Select-String $package
if ($installed) {
    Write-Host "  发现旧版本，正在卸载..." -ForegroundColor Yellow
    & $adb uninstall $package | Out-Null
    Write-Host "√ 旧版本已卸载" -ForegroundColor Green
} else {
    Write-Host "  未安装旧版本" -ForegroundColor White
}

# 4. 安装新APK
Write-Host "`n[4/6] 安装APK..." -ForegroundColor Yellow
if (-Not (Test-Path $apk)) {
    Write-Host "× APK文件不存在: $apk" -ForegroundColor Red
    exit 1
}

$apkSize = [math]::Round((Get-Item $apk).Length / 1MB, 2)
Write-Host "  APK大小: $apkSize MB" -ForegroundColor White
Write-Host "  正在安装 (大文件需要时间)..." -ForegroundColor Yellow

$installResult = & $adb install -r $apk 2>&1
if ($installResult -like "*Success*") {
    Write-Host "√ APK安装成功!" -ForegroundColor Green
} else {
    Write-Host "× 安装失败:" -ForegroundColor Red
    Write-Host $installResult -ForegroundColor Red
    exit 1
}

# 5. 清空日志并启动APP
Write-Host "`n[5/6] 启动应用..." -ForegroundColor Yellow
& $adb logcat -c
Write-Host "  日志已清空" -ForegroundColor White
Write-Host "  正在启动 MainActivity..." -ForegroundColor Yellow

$startResult = & $adb shell am start -n $activity 2>&1
if ($startResult -like "*Error*") {
    Write-Host "× 启动失败:" -ForegroundColor Red
    Write-Host $startResult -ForegroundColor Red
} else {
    Write-Host "√ 应用已启动!" -ForegroundColor Green
}

Start-Sleep -Seconds 2

# 6. 实时显示日志
Write-Host "`n[6/6] 查看实时日志 (Ctrl+C 停止)..." -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

try {
    & $adb logcat | ForEach-Object {
        $line = $_
        
        # 高亮关键日志
        if ($line -match "BoltApplication|MainActivity|FATAL|AndroidRuntime") {
            if ($line -match "FATAL|AndroidRuntime.*FATAL") {
                Write-Host $line -ForegroundColor Red
            } elseif ($line -match "ERROR|失败|Exception") {
                Write-Host $line -ForegroundColor Yellow
            } elseif ($line -match "完成|成功|SUCCESS") {
                Write-Host $line -ForegroundColor Green
            } else {
                Write-Host $line -ForegroundColor Cyan
            }
        }
    }
} catch {
    Write-Host "`n日志监控已停止" -ForegroundColor Yellow
}
