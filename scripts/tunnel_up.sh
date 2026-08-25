#!/bin/bash
# Idempotent Tokyo SOCKS tunnels (binance geo-block relay).
#
#   1081 — TRADING. Every price feed the trader acts on rides this one.
#   1082 — LOGGING-ONLY venue-latency probe (bybit/okx/binance top-of-book).
#
# The probe gets its OWN tunnel on purpose. A single `ssh -D` multiplexes all
# of its channels over ONE TCP connection with shared SSH windowing, so nine
# high-rate book streams sharing 1081 could add latency to the feed the trader
# actually trades on. Separate tunnel => separate TCP connection => the probe
# cannot starve trading. Losing 1082 costs only log rows.
#
# Cron: every 10 min + @reboot (idempotent, so re-running is free).
HOST="ubuntu@ec2-43-206-116-243.ap-northeast-1.compute.amazonaws.com"
KEY="/home/ubuntu/.ssh/collector.pem"

up() {
  local port="$1" tag="$2"
  pgrep -f "ssh -D ${port}" > /dev/null && return 0
  nohup ssh -D "${port}" -N -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=20 -o ExitOnForwardFailure=yes \
    -i "${KEY}" "${HOST}" \
    > "/tmp/tunnel_${port}.log" 2>&1 &
  echo "started ${tag} tunnel on ${port}"
}

up 1081 trading
up 1082 venuelat
