# Download FlatBuffers Compiler (flatc.exe) for Windows
$flatcVersion = "24.3.25"
$downloadUrl = "https://github.com/google/flatbuffers/releases/download/v$flatcVersion/Windows.flatc.binary.zip"
$outputZip = "$PSScriptRoot\flatc.zip"
$flatcDir = "$PSScriptRoot\.flatc"
$flatcExe = "$flatcDir\flatc.exe"

Write-Host "============================================================"
Write-Host "Downloading FlatBuffers Compiler (flatc v$flatcVersion)"
Write-Host "============================================================"

# Create directory
New-Item -ItemType Directory -Force -Path $flatcDir | Out-Null

# Download
Write-Host "`nDownloading flatc.exe..."
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $outputZip -UseBasicParsing
    Write-Host "Download complete: $outputZip"
} catch {
    Write-Host "Download failed: $_"
    exit 1
}

# Extract
Write-Host "`nExtracting..."
try {
    Expand-Archive -Path $outputZip -DestinationPath $flatcDir -Force
    Write-Host "Extraction complete: $flatcDir"
} catch {
    Write-Host "Extraction failed: $_"
    exit 1
}

# Cleanup
Remove-Item $outputZip -Force

# Verify
if (Test-Path $flatcExe) {
    Write-Host "`nflatc.exe installed: $flatcExe"
    Write-Host "`nAdding to PATH (current session only)..."
    
    # Add to PATH temporarily
    $env:PATH = "$flatcDir;$env:PATH"
    Write-Host "PATH updated"
    
    # Test
    Write-Host "`nTesting flatc:"
    & $flatcExe --version
} else {
    Write-Host "`nInstallation failed: flatc.exe not found"
    exit 1
}

Write-Host "`n============================================================"
Write-Host "flatc installation complete!"
Write-Host "============================================================"
