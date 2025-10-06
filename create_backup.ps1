$src = 'D:\binance_ats_clone\obaidur-binance-ats-main'
$dst = "D:\binance-autobot-backup-$(Get-Date -Format yyyyMMdd).zip"

try {
    # Remove previous file if exists (optional)
    if (Test-Path $dst) {
        Remove-Item $dst -Force
    }

    # Create zip (use wildcard to include all files/folders inside source)
    Compress-Archive -Path (Join-Path $src '*') -DestinationPath $dst -Force -CompressionLevel Optimal

    # Verify creation
    if (Test-Path $dst) {
        Write-Output "BACKUP_OK:$dst"
        exit 0
    } else {
        Write-Output "BACKUP_FAIL"
        exit 2
    }
}
catch {
    Write-Output "BACKUP_FAIL: $($_.Exception.Message)"
    exit 1
}
