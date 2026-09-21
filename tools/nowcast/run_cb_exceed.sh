#!/bin/bash
# std (43) vs std+cb (43 + 10 cb lags + sigma_cb) exceedance classifiers on the v10 rows, 3 seeds, h 1 s / 5 s, X 1/2/3 c.
cd /e/poly/crypto_trader
PY=/d/ProgramData/Anaconda3/envs/pytorch2/python.exe
SP="C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
run() {  # seed tag extra
  $PY tools/nowcast/train_exceed.py --h mid5,mid25 --x 1,2,3 --seed $1 --tag $2 --save-pred data/nowcast/exceed $3 > "$SP/exceed_$2_s$1.log" 2>&1
}
run 0 k43 "" & run 0 k43cb "--cb" & run 1 k43 "" & run 1 k43cb "--cb" &
wait
run 2 k43 "" & run 2 k43cb "--cb" &
wait
for f in "$SP"/exceed_k43_s*.log "$SP"/exceed_k43cb_s*.log; do echo "=== $f"; cat "$f"; done
