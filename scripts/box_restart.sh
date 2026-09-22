#!/bin/bash
# Stop the live traders by PID (this script's own command line does not contain the yaml patterns), then relaunch
# via scripts/start_collectors.sh (which only starts what is not running). Run on trade-poly from ~/poly_arb_trader.
# Usage: box_restart.sh            -> restart BTC and ETH
#        box_restart.sh btc | eth  -> restart one of them
cd ~/poly_arb_trader || exit 1
case "${1:-all}" in
  btc) PATS=('kalshi-trade\.yaml$') ;;
  eth) PATS=('kalshi-eth\.yaml$') ;;
  *)   PATS=('kalshi-trade\.yaml$' 'kalshi-eth\.yaml$') ;;
esac
for pat in "${PATS[@]}"; do
  for pid in $(pgrep -f "$pat"); do
    echo "stopping $pid: $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-80)"
    kill "$pid"
  done
done
sleep 3
for pat in "${PATS[@]}"; do
  for pid in $(pgrep -f "$pat"); do echo "still up, SIGKILL $pid"; kill -9 "$pid"; done
done
sleep 1
bash scripts/start_collectors.sh
sleep 2
pgrep -af "release/live" | cut -c1-100
