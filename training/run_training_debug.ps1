Set-Location "E:\devAI\boltassistant\training"
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

Write-Host "Starting training debug..."
python -u train_simple.py
