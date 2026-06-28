# Standalone, shared SOCKS5 tunnel (Windows / PowerShell).
#
# Opens `ssh -D <port> -N <host>` — a SOCKS5 proxy on 127.0.0.1:<port> that
# routes traffic out through the VPS (unrestricted region). Kept SEPARATE from
# the collectors on purpose: the one tunnel is shared by every process that
# points at the proxy (the Rust app, the Python collector, curl, ...).
#
# Idempotent: if the port is already open it exits immediately (re-uses the
# existing tunnel). Otherwise it runs the tunnel and auto-restarts on drop.
#
#   pwsh scripts/tunnel.ps1                 # host=collector-vps, port=1080
#   pwsh scripts/tunnel.ps1 -SshHost my-vps -Port 1080
#
# Prereq: a `collector-vps` entry in ~/.ssh/config (see crypto_collector
# doc/vps-setup.md). Verify once up:
#   curl --proxy socks5://127.0.0.1:1080 https://api.ipify.org

param(
    [string]$SshHost = "collector-vps",
    [int]$Port = 1080
)

function Test-Port([string]$h, [int]$p) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $iar = $c.BeginConnect($h, $p, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(800)
        $connected = $ok -and $c.Connected
        $c.Close()
        return $connected
    } catch {
        return $false
    }
}

if (Test-Port "127.0.0.1" $Port) {
    Write-Host "SOCKS5 tunnel already up on 127.0.0.1:$Port - sharing the existing tunnel."
    exit 0
}

Write-Host "Starting shared SOCKS5 tunnel: ssh -D $Port -N $SshHost"
$backoff = 2
while ($true) {
    $start = Get-Date
    & ssh -D $Port -N `
        -o ExitOnForwardFailure=yes `
        -o ServerAliveInterval=15 `
        -o ServerAliveCountMax=3 `
        $SshHost
    $code = $LASTEXITCODE
    $uptime = [int](New-TimeSpan -Start $start).TotalSeconds
    if ($uptime -gt 60) { $backoff = 2 }   # stable run -> reset backoff
    Write-Host "tunnel exited (code=$code) after ${uptime}s - restarting in ${backoff}s"
    Start-Sleep -Seconds $backoff
    $backoff = [Math]::Min($backoff * 2, 60)
}
