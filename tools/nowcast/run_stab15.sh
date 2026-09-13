#!/usr/bin/env bash
# 3 s-target model (standard 43 feats): 6 seeds with checkpoints -> models/resid-prune15-btc.json -> stability-only grid 08-27..09-04
SP="C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"; PY="D:/ProgramData/Anaconda3/envs/pytorch2/python.exe"; FEAT="hist,kbs,kas,kmi,pbs,lf_flow_tfi_1s,lf_flow_tfi_5s,lf_flow_tfi_30s,lf_flow_vol_1s,lf_flow_n_1s,lf_flow_n_5s,lf_depth_band5,lf_depth_band10,lf_depth_band25,lf_depth_depth10,lf_depth_spread_bps,lf_depth_dimb5_1s,lf_depth_dimb5_5s,lf_depth_dband10_1s"; LAGS="1,2,3,4,5,7,10,15,20,30"
cd E:/poly/crypto_trader || exit 1
echo "=== 3 s model, 6 seeds with checkpoints $(date) ===" >> "$SP/campaign.log"
for sd in 0 1 2 3 4 5; do
  timeout 1800 "$PY" -u "$SP/train_outcome.py" --cpu --arch mlp --market residual --feat "$FEAT" --lookback 6 --lags "$LAGS" --target mid15 --seed $sd     --tag hz15ck --save-pred "$SP/prd_hz15ck_s$sd.npz" --save-model data/nowcast/resid_models3/prune15_s$sd.pt 2>&1 | grep -vE "Warning" | cut -c1-110 | tee -a "$SP/campaign.log"
done
F=""; for sd in 0 1 2 3 4 5; do F="$F $SP/prd_hz15ck_s$sd.npz"; done
"$PY" "$SP/target_mse.py" hz15_x6 mid15 $F | cut -c1-235 | tee -a "$SP/campaign.log"; "$PY" "$SP/ens_outcome.py" hz15_x6 $F | cut -c1-200 | tee -a "$SP/campaign.log"
"$PY" - <<'PYEOF'
import json
js = {"kind": "resid", "note": "3 s-target twin of the standard 43-input nowcast (feat/lags as resid-prune5-btc.json, target mid15 = mid +3 s), 6 MLP seeds; lake features need data/nowcast/lake_bars. Checkpoints data/nowcast/resid_models3/prune15_s*.pt (2026-09-13).",
      "members": ["../data/nowcast/resid_models3/prune15_s%d.pt" % sd for sd in range(6)], "extras": []}
json.dump(js, open("models/resid-prune15-btc.json", "w"), indent=1); print("json written")
PYEOF
S9="$(ls data/samples/2026-08-2[7-9].csv.gz data/samples/2026-08-3[01].csv.gz data/samples/2026-09-0[1-4].csv.gz | paste -sd,)"
mkdir -p out/pr15_stabonly
"$PY" tools/analyze_online.py --samples "$S9" --model prune15=models/resid-prune15-btc.json --deltas 0.0025,0.005,0.01,0.02 --tte 300:60   --latency-ms 100 --chase-c 0.01 --gate stable --gate-grid "open=-100;win=0.5,1,2,3,5;rng=0.5,1,2,3" --episode-arm   --fresh-from 2026-09-01 --out-dir out/pr15_stabonly > out/pr15_stabonly.log 2>&1
echo "grid done $(date +%H:%M:%S)"; grep -iE "traceback|error" out/pr15_stabonly.log | head -3
"$PY" "$SP/pos_cells.py" out/pr15_stabonly/trades.csv 20
