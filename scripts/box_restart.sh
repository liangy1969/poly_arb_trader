#!/bin/bash
# Stop the BTC and ETH live traders by PID (this script's own command line does not contain the yaml
# patterns), then relaunch both via scripts/start_collectors.sh. Run on trade-poly from ~/poly_arb_trader.
cd ~/poly_arb_trader || exit 1
for pat in 'kalshi-trade\.yaml$' 'kalshi-eth\.yaml$'; do
  for pid in $(pgrep -f "$pat"); do
    echo "stopping $pid: $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-80)"
    kill "$pid"
  done
done
sleep 3
for pat in 'kalshi-trade\.yaml$' 'kalshi-eth\.yaml$'; do
  for pid in $(pgrep -f "$pat"); do echo "still up, SIGKILL $pid"; kill -9 "$pid"; done
done
sleep 1
bash scripts/start_collectors.sh
sleep 2
pgrep -af "release/live" | cut -c1-100
