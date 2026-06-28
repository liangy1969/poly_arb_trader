# scripts/

## SOCKS5 tunnel (shared)

Binance Global is geo-restricted, so the Binance collector routes through a
SOCKS5 proxy at `127.0.0.1:1080` (see `DESIGN_TRADING_SYSTEM.md` §5). The proxy
is an SSH dynamic forward to a VPS in an unrestricted region.

**The tunnel is deliberately a separate, standalone process** — one tunnel is
shared by every consumer (the Rust app, the Python collector, `curl`, …). No
process opens its own; they all just dial `127.0.0.1:1080`.

### Run it

```powershell
pwsh scripts/tunnel.ps1            # Windows
```
```bash
scripts/tunnel.sh                  # POSIX
```

Both are **idempotent** (skip if the port is already up) and **auto-restart**
the tunnel on drop. Leave it running; start the app/collectors separately.

### Prerequisite

A `collector-vps` entry in `~/.ssh/config` pointing at your VPS (full setup in
the crypto_collector repo's `doc/vps-setup.md`):

```
Host collector-vps
    HostName <your-vps-ip>
    User collector
    IdentityFile ~/.ssh/collector_vps
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

### Verify

```bash
curl --proxy socks5://127.0.0.1:1080 https://api.ipify.org   # prints the VPS IP
```

### Consumers

- **Rust:** `BinanceCfg.socks_proxy = Some("127.0.0.1:1080")` (default). Set it
  to `None` (or `BINANCE_DIRECT=1` in the probe) to bypass the proxy when
  running on the VPS itself.
- The app never spawns or supervises the tunnel — that stays out-of-band so it
  can be shared.
