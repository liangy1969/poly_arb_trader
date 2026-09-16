#!/usr/bin/env bash
# Multi-day maker simulation: for each tape day make the standard 1 s fair dump (if missing) and run the strategy variants; then aggregate.
SP="C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
PY="D:/ProgramData/Anaconda3/envs/pytorch2/python.exe"
cd E:/poly/crypto_trader || exit 1
DAYS="2026-09-10 2026-09-11 2026-09-12 2026-09-13 2026-09-14 2026-09-15"
for i in $(seq 1 400); do ok=1; for d in $DAYS; do [ -f data/samples/$d.csv.gz ] && [ -f data/nowcast/lake_bars/$d.bars.npz ] && ls data/samples/trades-$d.csv* >/dev/null 2>&1 || ok=0; done; [ $ok = 1 ] && break; sleep 30; done
echo "inputs ready $(date +%H:%M:%S)"
for d in $DAYS; do
  tag=${d:5:2}${d:8:2}; DUMP="$SP/dump_prune5_$tag.csv"
  if [ ! -f "$DUMP" ]; then
    mkdir -p out/mm_${tag}p
    "$PY" tools/analyze_online.py --samples data/samples/$d.csv.gz --model prune5=models/resid-prune5-btc.json --deltas 0.01 --tte 300:60 --latency-ms 0 --chase-c 0 --gate none --episode-arm --dump-fair "$DUMP" --out-dir out/mm_${tag}p > out/mm_${tag}p.log 2>&1
    echo "dump $d done $(date +%H:%M:%S) $(grep -c . "$DUMP") rows"
  fi
  C="tools/nowcast/maker_queue_sim.py $d $DUMP --lat-dist --quiet --lone 0"
  "$PY" $C --policy naive --tag naive > "$SP/qd_${tag}_naive.log" 2>&1 &
  "$PY" $C --policy pull --tag pull > "$SP/qd_${tag}_pull.log" 2>&1 &
  "$PY" $C --policy pull --jump 0.5 --tag pull_jump > "$SP/qd_${tag}_pull_jump.log" 2>&1 &
  "$PY" $C --policy pull --imb-pull 0.30 --tag pull_imb > "$SP/qd_${tag}_pull_imb.log" 2>&1 &
  "$PY" $C --policy pull --jump 0.5 --imb-pull 0.30 --tag pull_jump_imb > "$SP/qd_${tag}_pull_jump_imb.log" 2>&1 &
  "$PY" $C --policy fav --tag fav > "$SP/qd_${tag}_fav.log" 2>&1 &
  wait; echo "sims $d done $(date +%H:%M:%S)"; grep -l "Traceback" "$SP"/qd_${tag}_*.log 2>/dev/null
done
echo "=== AGGREGATE over $DAYS ==="
"$PY" tools/nowcast/maker_days.py naive pull pull_jump pull_imb pull_jump_imb fav -- $DAYS
echo "=== AGGREGATE excluding 09-11 (lake gap 13-20 UTC) ==="
"$PY" tools/nowcast/maker_days.py naive pull pull_jump pull_imb pull_jump_imb fav -- 2026-09-10 2026-09-12 2026-09-13 2026-09-14 2026-09-15
echo "all done $(date +%H:%M:%S)"
