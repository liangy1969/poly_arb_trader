#!/usr/bin/env bash
# Standalone, shared SOCKS5 tunnel (POSIX).
#
# Opens `ssh -D <port> -N <host>` — a SOCKS5 proxy on 127.0.0.1:<port> routed
# out through the VPS. Kept SEPARATE from the collectors on purpose: the one
# tunnel is shared by every process pointing at the proxy.
#
# Idempotent (skips if the port is already open); auto-restarts on drop.
#
#   scripts/tunnel.sh                 # host=collector-vps, port=1080
#   scripts/tunnel.sh my-vps 1080
#
# Prereq: a `collector-vps` entry in ~/.ssh/config. Verify once up:
#   curl --proxy socks5://127.0.0.1:1080 https://api.ipify.org
set -u

SSH_HOST="${1:-collector-vps}"
PORT="${2:-1080}"

if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
    exec 3>&- 3<&-
    echo "SOCKS5 tunnel already up on 127.0.0.1:${PORT} - sharing the existing tunnel."
    exit 0
fi

echo "Starting shared SOCKS5 tunnel: ssh -D ${PORT} -N ${SSH_HOST}"
backoff=2
while true; do
    start=$(date +%s)
    ssh -D "${PORT}" -N \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        "${SSH_HOST}"
    code=$?
    uptime=$(( $(date +%s) - start ))
    [ "${uptime}" -gt 60 ] && backoff=2
    echo "tunnel exited (code=${code}) after ${uptime}s - restarting in ${backoff}s"
    sleep "${backoff}"
    if [ $(( backoff * 2 )) -gt 60 ]; then backoff=60; else backoff=$(( backoff * 2 )); fi
done
