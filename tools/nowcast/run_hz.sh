#!/usr/bin/env bash
# Horizon test: the standard 43-input feature set trained on +3 s (mid15) and +5 s (mid25) targets, 3 seeds; reference +1 s (mid5)
SP="C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"; PY="D:/ProgramData/Anaconda3/envs/pytorch2/python.exe"; FEAT="hist,kbs,kas,kmi,pbs,lf_flow_tfi_1s,lf_flow_tfi_5s,lf_flow_tfi_30s,lf_flow_vol_1s,lf_flow_n_1s,lf_flow_n_5s,lf_depth_band5,lf_depth_band10,lf_depth_band25,lf_depth_depth10,lf_depth_spread_bps,lf_depth_dimb5_1s,lf_depth_dimb5_5s,lf_depth_dband10_1s"; LAGS="1,2,3,4,5,7,10,15,20,30"
cd E:/poly/crypto_trader || exit 1
for i in $(seq 1 200); do [ "$(ls "$SP/midmove_v10_fut/"*.fut.npz 2>/dev/null | wc -l)" -ge 38 ] && break; sleep 15; done; sleep 10
echo "=== HORIZON test: standard 43 feats, targets mid5 (1s) / mid15 (3s) / mid25 (5s), 3 seeds $(date) ===" >> "$SP/campaign.log"
for tg in mid15 mid25 mid5; do for sd in 0 1 2; do
  timeout 1800 "$PY" -u "$SP/train_outcome.py" --cpu --arch mlp --market residual --feat "$FEAT" --lookback 6 --lags "$LAGS" --target $tg --seed $sd     --tag "hz_$tg" --save-pred "$SP/prd_hz_${tg}_s$sd.npz" 2>&1 | grep -vE "Warning" | cut -c1-110 | tee -a "$SP/campaign.log"
done; done
echo "--- horizon: 3-seed ensembles, forecast ratio (MSE(y-model)/MSE(y-mid)) + outcome BCE vs market ---" | tee -a "$SP/campaign.log"
for tg in mid5 mid15 mid25; do
  F=""; for sd in 0 1 2; do F="$F $SP/prd_hz_${tg}_s$sd.npz"; done
  "$PY" "$SP/target_mse.py" "hz_$tg" $tg $F | cut -c1-235; "$PY" "$SP/ens_outcome.py" "hz_$tg" $F | cut -c1-200
done | tee -a "$SP/campaign.log"
echo "horizon done $(date +%H:%M:%S)"
