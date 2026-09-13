#!/usr/bin/env bash
# PRE-REGISTERED re-test of the taker cell: stability-only gate win 5 s / range 0.5c / delta 0.005 (also .0025/.01 for context)
# on the untouched days 2026-09-05..12 with the standard 1s model (prune5) and the full lake 1s model (lake5). Bar: |t| > 2 on the new slice.
PY="D:/ProgramData/Anaconda3/envs/pytorch2/python.exe"
cd E:/poly/crypto_trader || exit 1
for i in $(seq 1 400); do n=0; for d in 09 11 12; do [ -f data/samples/2026-09-$d.csv.gz ] && [ -f data/nowcast/lake_bars/2026-09-$d.bars.npz ] && n=$((n+1)); done; [ "$n" -ge 3 ] && break; sleep 15; done
sleep 20; echo "inputs ready $(date +%H:%M:%S)"
S="$(ls data/samples/2026-09-0[5-9].csv.gz data/samples/2026-09-1[0-2].csv.gz | paste -sd,)"; echo "$S" | tr ',' '\n' | wc -l
mkdir -p out/oos2_w5r05
"$PY" tools/analyze_online.py --samples "$S" --model prune5=models/resid-prune5-btc.json --model lake5=models/resid-lake5-btc.json \
  --deltas 0.0025,0.005,0.01 --tte 300:60 --latency-ms 100 --chase-c 0.01 --gate stable --gate-grid "open=-100;win=5;rng=0.5" --episode-arm \
  --out-dir out/oos2_w5r05 > out/oos2_w5r05.log 2>&1
echo "run done $(date +%H:%M:%S)"; grep -iE "traceback|error|nostrike" out/oos2_w5r05.log | head -3; grep -E "^events:" out/oos2_w5r05.log
"$PY" tools/nowcast/cells_days.py out/oos2_w5r05/trades.csv 5
