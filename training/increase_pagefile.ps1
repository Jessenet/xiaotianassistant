# 增加Windows虚拟内存页面文件
# 需要管理员权限运行

Write-Host "检查当前页面文件设置..." -ForegroundColor Cyan

# 获取当前页面文件设置
$computerSystem = Get-WmiObject Win32_ComputerSystem -EnableAllPrivileges
$currentPageFile = Get-WmiObject -Class Win32_PageFileSetting

if ($currentPageFile) {
    Write-Host "当前页面文件:" -ForegroundColor Yellow
    $currentPageFile | Format-Table Name, InitialSize, MaximumSize -AutoSize
} else {
    Write-Host "系统管理页面文件（自动）" -ForegroundColor Yellow
}

Write-Host "`n建议操作:" -ForegroundColor Green
Write-Host "1. 打开 系统属性 > 高级 > 性能设置 > 高级 > 虚拟内存"
Write-Host "2. 取消选中 '自动管理所有驱动器的分页文件大小'"
Write-Host "3. 选择系统驱动器，选择 '自定义大小'"
Write-Host "4. 初始大小: 16384 MB (16GB)"
Write-Host "5. 最大大小: 32768 MB (32GB)"
Write-Host "6. 点击 '设置' 然后 '确定'"
Write-Host "7. 重启计算机使更改生效"

Write-Host "`n或者运行以下命令（需要管理员权限）:" -ForegroundColor Cyan
Write-Host @"
# 禁用自动管理
`$computerSystem = Get-WmiObject Win32_ComputerSystem -EnableAllPrivileges
`$computerSystem.AutomaticManagedPagefile = `$false
`$computerSystem.Put()

# 删除现有页面文件设置
Get-WmiObject -Class Win32_PageFileSetting | Remove-WmiObject

# 创建新的页面文件设置（C盘）
`$pageFile = ([wmiclass]"Win32_PageFileSetting").CreateInstance()
`$pageFile.Name = "C:\\pagefile.sys"
`$pageFile.InitialSize = 16384  # 16GB
`$pageFile.MaximumSize = 32768  # 32GB
`$pageFile.Put()

Write-Host "页面文件已配置，请重启计算机" -ForegroundColor Green
"@

Write-Host "`n按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
