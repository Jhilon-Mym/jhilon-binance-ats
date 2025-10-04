<#
.SYNOPSIS
  Incremental backup script for the project.

.DESCRIPTION
  Creates full or incremental zip backups. On incremental runs the script
  finds files modified since the last successful backup and stores them in a
  timestamped zip. A small metadata file tracks the last backup time.

.USAGE
  # Test mode (lists files that would be archived)
  .\incremental_backup.ps1 -Test

  # Create an incremental archive (default)
  .\incremental_backup.ps1

  # Force a full backup
  .\incremental_backup.ps1 -Full

  # Change source or destination
  .\incremental_backup.ps1 -SourcePath 'C:\path\to\repo' -DestDir 'D:\backups'

.NOTES
  - Designed for PowerShell on Windows. Uses Compress-Archive to create zips.
  - The script writes a small metadata file `last_backup.txt` under the
    destination folder to track the last-run timestamp (UTC).
#>

param(
    [string]$SourcePath = '',
    [string]$DestDir = '',
    [switch]$Full,
    [switch]$Test,
    [int]$RetentionDays = 90
)

Set-StrictMode -Version Latest

function Write-Log { param($m) Write-Output "[$((Get-Date).ToString('s'))] $m" }

# Resolve script and default project root (two levels up from script)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if([string]::IsNullOrWhiteSpace($SourcePath)){
    $defaultRoot = Resolve-Path -Path (Join-Path $scriptDir '..')
    $SourcePath = $defaultRoot.ProviderPath
} else {
    $rp = Resolve-Path -Path $SourcePath -ErrorAction SilentlyContinue
    if($rp){
        if($rp -is [System.Array] -or $rp -is [System.Management.Automation.PathInfo[]]){ $SourcePath = $rp[0].ProviderPath }
        elseif($rp -is [System.Management.Automation.PathInfo]){ $SourcePath = $rp.ProviderPath }
        else { $SourcePath = $rp.ToString() }
    } else {
        # leave as provided (may be relative path)
        $SourcePath = $SourcePath
    }
}

if([string]::IsNullOrWhiteSpace($DestDir)){
    $rd = Resolve-Path -Path (Join-Path $SourcePath '..') -ErrorAction SilentlyContinue
    if($rd){
        if($rd -is [System.Array] -or $rd -is [System.Management.Automation.PathInfo[]]){ $DestDir = $rd[0].ProviderPath }
        elseif($rd -is [System.Management.Automation.PathInfo]){ $DestDir = $rd.ProviderPath }
        else { $DestDir = $rd.ToString() }
    } else { $DestDir = (Join-Path $SourcePath '..') }
} else {
    $rd = Resolve-Path -Path $DestDir -ErrorAction SilentlyContinue
    if($rd){
        if($rd -is [System.Array] -or $rd -is [System.Management.Automation.PathInfo[]]){ $DestDir = $rd[0].ProviderPath }
        elseif($rd -is [System.Management.Automation.PathInfo]){ $DestDir = $rd.ProviderPath }
        else { $DestDir = $rd.ToString() }
    } else { $DestDir = $DestDir }
}

$metaDir = Join-Path $DestDir 'incremental_backups'
if(-not (Test-Path $metaDir)){ New-Item -ItemType Directory -Path $metaDir -Force | Out-Null }

$lastFile = Join-Path $metaDir 'last_backup.txt'

Write-Log "Source: $SourcePath"
Write-Log "Destination: $DestDir"

if($Full -or -not (Test-Path $lastFile)){
    Write-Log "Performing full backup (forced or first run)."
    $since = [datetime]::MinValue
} else {
    $text = Get-Content $lastFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if([string]::IsNullOrWhiteSpace($text)){
        $since = [datetime]::MinValue
    } else {
        $since = [datetime]::Parse($text)
    }
    Write-Log "Performing incremental backup since $since (UTC)."
}

# Collect files changed
Push-Location $SourcePath
try{
    $allFiles = Get-ChildItem -Path $SourcePath -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Attributes -notmatch 'ReparsePoint' }
    if($since -eq [datetime]::MinValue){
        $changed = $allFiles
    } else {
        # Convert to UTC comparison for safety
        $sinceUtc = [datetime]::SpecifyKind($since, 'Utc')
        $changed = $allFiles | Where-Object { ($_.LastWriteTimeUtc) -gt $sinceUtc }
    }
    $changed = @($changed)
    $count = $changed.Count
    Write-Log "$count files changed/collected for backup."

    if($Test){
        if($count -eq 0){ Write-Log 'No files to archive.'; Pop-Location; exit 0 }
        Write-Log 'Files that would be archived:'
        $changed | ForEach-Object { Write-Output $_.FullName }
        Pop-Location; exit 0
    }

    if($count -eq 0){ Write-Log 'No changes since last backup. Nothing to do.'; Pop-Location; exit 0 }

    $ts = Get-Date -Format yyyyMMdd_HHmmss
    $type = if($since -eq [datetime]::MinValue){ 'full' } else { 'inc' }
    $zipName = "backup_${type}_$ts.zip"
    $zipPath = Join-Path $metaDir $zipName

    # Build relative paths so zip keeps repo structure
    $relativePaths = $changed | ForEach-Object { $_.FullName.Substring($SourcePath.Length+1) }
    # Change into source dir and compress by relative paths to preserve structure
    Push-Location $SourcePath
    try{
        Write-Log "Creating zip: $zipPath (files: $count)"
        Compress-Archive -Path $relativePaths -DestinationPath $zipPath -Force
    } finally { Pop-Location }

    if(Test-Path $zipPath){
        Write-Log "Archive created: $zipPath"
        # Update last backup to now (UTC)
        $now = (Get-Date).ToUniversalTime().ToString('u')
        Set-Content -Path $lastFile -Value $now -Encoding UTF8
        Write-Log "Updated last backup timestamp to $now"
    } else {
        Write-Log "Failed to create archive: $zipPath"
        Pop-Location; exit 1
    }
}
finally{ Pop-Location }

# Prune old backups
try{
    $cutoff = (Get-Date).AddDays(-1 * [int]$RetentionDays)
    $old = Get-ChildItem -Path $metaDir -Filter 'backup_*.zip' -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt $cutoff }
    if($old -and $old.Count -gt 0){
        Write-Log "Pruning $($old.Count) old backup(s) older than $RetentionDays days."
        $old | ForEach-Object { Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue }
    }
} catch { Write-Log "Prune error: $_" }

Write-Log 'Done.'
