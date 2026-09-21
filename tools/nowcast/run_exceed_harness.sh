#!/bin/bash
# Exceedance classifier through the standing harness (kind=exceed): 22 days 08-27..09-17 (VAL 08-27..31 | TEST 09-01..04 |
# OOS 09-05..17), 50 ms signal, |P(up)-P(down)| cuts as --deltas, --gate none, tte 300:60, taker exits at 1 s and settlement.
cd /e/poly/crypto_trader
PY=/d/ProgramData/Anaconda3/envs/pytorch2/python.exe
S=$(ls data/samples/2026-08-2[7-9].csv.gz data/samples/2026-08-3[01].csv.gz data/samples/2026-09-0[1-9].csv.gz data/samples/2026-09-1[0-7].csv.gz | paste -sd,)
M="--model x1=models/exceed-1s-x1-btc.json --model x15=models/exceed-1s-x1.5-btc.json"
COMMON="--deltas 0.26,0.34,0.42,0.48,0.53 --tte 300:60 --gate none --rearm-eps 0.2 --exit-h 0,1 --close-eps 0.001 --close-cap 5 --cap 0"
$PY tools/analyze_online.py --samples "$S" $M $COMMON --latency-ms 100 --chase-c 0.01 --out-dir out/exceed_all > out/exceed_all.log 2>&1 &
$PY tools/analyze_online.py --samples "$S" $M $COMMON --latency-ms 100 --chase-c 0.001 --px-tails 0.10,0.90 --out-dir out/exceed_tails > out/exceed_tails.log 2>&1 &
$PY tools/analyze_online.py --samples "$S" $M $COMMON --latency-ms 0 --px-tails 0.10,0.90 --out-dir out/exceed_tails_lat0 > out/exceed_tails_lat0.log 2>&1 &
wait
echo ALL DONE
for d in exceed_all exceed_tails exceed_tails_lat0; do echo "=== $d"; $PY tools/nowcast/exceed_slices.py out/$d/trades.csv 10; done
