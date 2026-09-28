#!/bin/bash
# Maker strategies driven by the exceedance classifier (raw-perp model), queue-fill simulator on the Kalshi trade tape.
# 1) per-day score dump: analyze_online --dump-exceed (raw model, 50 ms rows) -> CSV (ticker, ts_ms, fair, mid) with
#    fair = mid + s/100, so the simulator's g (cents) = s = P(up) - P(down).
# 2) grid: region tails|1c x policy naive | pull theta .2/.34/.48 | fav theta .2/.34/.48; measured latency distribution;
#    hold to settlement (plus 6/30 s marks). Aggregate with maker_days.py.
cd /e/poly/crypto_trader
PY=/d/ProgramData/Anaconda3/envs/pytorch2/python.exe
DAYS="2026-09-10 2026-09-11 2026-09-12 2026-09-13 2026-09-14 2026-09-15 2026-09-16 2026-09-17 2026-09-22 2026-09-23 2026-09-25 2026-09-26"
mkdir -p out/xmaker
dump() {
  d=$1
  [ -f out/xmaker/dump_$d.csv ] && return
  $PY tools/analyze_online.py --samples "$(ls data/samples/$d.csv* | head -1)" --model raw=models/exceed-1s-x1-raw-btc.json --deltas 0.48 --tte 300:60 --gate none --latency-ms 0 --cap 0 --dump-exceed out/xmaker/dump_$d.npz --out-dir out/xmaker/h_$d > out/xmaker/h_$d.log 2>&1
  $PY - "$d" <<'EOF'
import sys, re, numpy as np, pandas as pd
d = sys.argv[1]; z = np.load("out/xmaker/dump_%s.npz" % d, allow_pickle=True)
tk = [re.sub(r"^kalshi\.|\.YES$", "", t) for t in z["ticker"].astype(str)]
s = z["p_up"] - z["p_dn"]
pd.DataFrame({"ticker": tk, "ts_ms": z["ts"], "fair": z["mid"] + s / 100.0, "mid": z["mid"]}).to_csv("out/xmaker/dump_%s.csv" % d, index=False)
print(d, "dump rows", len(tk))
EOF
}
N=0
for d in $DAYS; do dump $d & N=$((N+1)); if [ $N -ge 4 ]; then wait; N=0; fi; done; wait
sim() {  # day tag args...
  d=$1; tag=$2; shift 2
  $PY tools/nowcast/maker_queue_sim.py $d out/xmaker/dump_$d.csv --lat-dist --tte 60,300 --tag x_$tag "$@" > out/xmaker/sim_${d}_$tag.log 2>&1
}
CELLS="naive:--policy naive pull20:--policy pull --theta 0.20 pull34:--policy pull --theta 0.34 pull48:--policy pull --theta 0.48 fav20:--policy fav --theta 0.20 fav34:--policy fav --theta 0.34 fav48:--policy fav --theta 0.48"
N=0
for reg in tails 1c; do
  for d in $DAYS; do
    for c in naive pull20 pull34 pull48 fav20 fav34 fav48; do
      case $c in
        naive) args="--policy naive";; pull20) args="--policy pull --theta 0.20";; pull34) args="--policy pull --theta 0.34";; pull48) args="--policy pull --theta 0.48";;
        fav20) args="--policy fav --theta 0.20";; fav34) args="--policy fav --theta 0.34";; fav48) args="--policy fav --theta 0.48";;
      esac
      sim $d ${reg}_$c --region $reg $args &
      N=$((N+1)); if [ $N -ge 10 ]; then wait; N=0; fi
    done
  done
done
wait
echo ALL DONE
for reg in tails 1c; do
  echo "######## region $reg"
  $PY tools/nowcast/maker_days.py x_${reg}_naive x_${reg}_pull20 x_${reg}_pull34 x_${reg}_pull48 x_${reg}_fav20 x_${reg}_fav34 x_${reg}_fav48 -- $DAYS
done
