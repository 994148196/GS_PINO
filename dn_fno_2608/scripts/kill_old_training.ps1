# Kill the orphaned 12-run scaling study (old run_scaling.sh task whose
# TaskStop failed on Windows). Matches only processes whose command line
# references train_dn_fno / run_scaling, so nothing else is touched.
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('python.exe', 'bash.exe') -and
    $_.CommandLine -match 'train_dn_fno|run_scaling'
}
if (-not $targets) {
    Write-Output "no matching processes found"
    exit 0
}
foreach ($t in $targets) {
    $short = $t.CommandLine
    if ($short.Length -gt 120) { $short = $short.Substring(0, 120) }
    $msg = "killing PID {0} ({1}): {2}" -f $t.ProcessId, $t.Name, $short
    Write-Output $msg
    Stop-Process -Id $t.ProcessId -Force
}
Start-Sleep -Seconds 2
$left = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('python.exe', 'bash.exe') -and
    $_.CommandLine -match 'train_dn_fno|run_scaling'
}
if ($left) {
    Write-Output ("STILL ALIVE: " + ($left.ProcessId -join ', '))
    exit 1
}
Write-Output "all old-training processes killed"
