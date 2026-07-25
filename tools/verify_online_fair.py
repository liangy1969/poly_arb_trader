#!/usr/bin/env python3
"""Online-vs-offline fair/calibrator replay — Level 2 of the consistency
ladder (DESIGN_LIVE_SIM_CONSISTENCY.md). Run after every model deploy.

Inputs:
  argv[1] = fairlog/fit lines (grep -E "fairlog|fit" data/trader-events.log)
  argv[2] = sampler slice CSV(.gz) covering the same window (box data/samples)
  argv[3] = model JSON (2p family; default = the live model)

Checks (pass bars in the design doc §Level 2):
  A. SURFACE+FEATURE replay: for each fairlog line, fair_off = model(px_log,
     cb_log, tte_log, pmom_from_sampler, db_log, dr_log) vs fair_log.
     (Surface is golden-proven; this isolates the feature reconstruction.)
  B. FULL OFFLINE-SIM replay: fair_off2 = model(px_sampler, cb_sampler, ...) —
     the offline pipeline end-to-end vs the online output (bundles feed diffs;
     the fat tail on fast moves is genuine feed timing — check dpx).
  C. CALIB check: logged per-boundary (db, dr) vs an offline rolling refit on
     the sampler rows (same 120/60 protocol).
"""
import sys, re, csv, gzip, json, math
import numpy as np
sys.path.insert(0, "tools")
import sim_50ms as sim

MODEL = sys.argv[3] if len(sys.argv) > 3 else "models/fair-2pmom-seqft3e3-btc.json"
M = json.load(open(MODEL))
print(f"model: {MODEL}")
RP, RC, BSC = M["rho_bar"], M["rho_cb"], M["b_scale"]
MU = np.array(M["mu"]); SD = np.array(M["sd"]); KS = [int(x[4:]) for x in M["extras"]]
W = [(np.array(l["w"]), np.array(l["b"])) for l in M["layers"]]

def logit_np(px, cb, tte, pmoms, b, db_s):  # db_s = dr (shared)
    tau = min(max(tte / 900.0, 1e-4), 1.0)
    sp = math.exp(RP + db_s); sc = math.exp(RC + db_s)
    h = np.array([((px - b) / sp) / math.sqrt(tau), ((cb - b) / sc) / math.sqrt(tau), math.log(tau)]
                 + list(np.clip((np.array(pmoms) - MU) / SD, -5, 5)))
    for i, (w, bb) in enumerate(W):
        h = w @ h + bb
        if i < 2: h = np.tanh(h)
    return float(h[0])

# ---- parse log lines ----
# tracing: 2026-07-22T11:25:03.123456Z  INFO fairlog: kalshi.KXBTC15M-...  tte=...
FAIR_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}T[\d:.]+Z).*fairlog\S* (\S+) tte=([\d.]+) px=([\d.]+) cb=([\d.]+) "
    r"mid=([\d.]+) fair=([\d.]+) db=([+-][\d.]+) dr=([+-][\d.]+)")
CAL_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}T[\d:.]+Z).*calib (\S+) tte=([\d.]+)s rows=(\d+) db=([+-][\d.]+) dr=([+-][\d.]+)")

def iso_ms(s):
    from datetime import datetime, timezone
    s = s.rstrip("Z")
    if "." in s:
        base, frac = s.split("."); frac = (frac + "000000")[:6]
        dt = datetime.strptime(base + "." + frac, "%Y-%m-%dT%H:%M:%S.%f")
    else:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

fairs, cals = [], []
for ln in open(sys.argv[1], errors="ignore"):
    m = FAIR_RE.search(ln)
    if m:
        fairs.append({"ts": iso_ms(m.group(1)), "inst": m.group(2), "tte": float(m.group(3)),
                      "px": float(m.group(4)), "cb": float(m.group(5)), "mid": float(m.group(6)),
                      "fair": float(m.group(7)), "db": float(m.group(8)), "dr": float(m.group(9))})
        continue
    m = CAL_RE.search(ln)
    if m:
        cals.append({"ts": iso_ms(m.group(1)), "inst": m.group(2), "tte": float(m.group(3)),
                     "rows": int(m.group(4)), "db": float(m.group(5)), "dr": float(m.group(6))})
print(f"parsed: {len(fairs)} fairlog lines, {len(cals)} calib lines")
if not fairs:
    sys.exit("no fairlog lines")

def tick_of(inst):
    t = inst.split(".")[-2] if inst.endswith(".YES") else inst.split(".")[-1]
    return t

# ---- sampler slice: 1s grid per ticker ----
S = sim.load_samples(sys.argv[2])
grid = {}
for t, d in S.items():
    o = np.argsort(d["ts"]);  # ensure sorted
    for k in d: d[k] = d[k][o]
    tsec = d["ts"] / 1000.0; secn = np.floor(tsec).astype(np.int64)
    _, ki = np.unique(secn[::-1], return_index=True); keep = np.sort(len(secn) - 1 - ki)
    g = {k: d[k][keep] for k in d}
    g["tsc"] = tsec[keep]
    pm = []
    for k in KS:
        idx = np.searchsorted(g["tsc"], g["tsc"] - k, side="right") - 1; ok = idx >= 0
        ok[ok] &= (g["tsc"][ok] - g["tsc"][idx[ok]]) <= k + 2
        lag = np.full(len(g["tsc"]), np.nan); lag[ok] = g["spot"][idx[ok]]
        pm.append((g["spot"] - lag) / lag)
    g["pmom"] = np.stack(pm, 1)
    grid[t] = g
print(f"sampler tickers: {len(grid)}")

# strikes for check C
meta = sim.fetch_meta(sorted({tick_of(f['inst']) for f in fairs}), "data/samples/meta_cache.json")

# ---- A + B ----
dA, dB_, dpx, dcb = [], [], [], []
for f in fairs:
    t = tick_of(f["inst"]); g = grid.get(t)
    mm = meta.get(t)
    if g is None or not mm or mm.get("strike") is None: continue
    strike = float(mm["strike"]); b = strike + BSC * f["db"]
    i = np.searchsorted(g["ts"], f["ts"] + 1, side="right") - 1
    if i < 0 or abs(g["ts"][i] - f["ts"]) > 1500: continue
    if np.isnan(g["pmom"][i]).any() or np.isnan(g["cbmid"][i]) or not (g["spot"][i] > 0): continue
    fa = 1 / (1 + math.exp(-logit_np(f["px"], f["cb"], f["tte"], g["pmom"][i], b, f["dr"])))
    fb = 1 / (1 + math.exp(-logit_np(g["spot"][i], g["cbmid"][i], f["tte"], g["pmom"][i], b, f["dr"])))
    dA.append(fa - f["fair"]); dB_.append(fb - f["fair"])
    dpx.append(g["spot"][i] - f["px"]); dcb.append(g["cbmid"][i] - f["cb"])

def rep(name, d):
    d = np.abs(np.array(d))
    if not len(d): print(f"{name}: n=0"); return
    print(f"{name}: n={len(d)} median={np.median(d):.5f} p90={np.percentile(d,90):.5f} max={d.max():.5f}")

print("\nA. |fair_off(logged px/cb + sampler pmom) - fair_online|   <- %mom reconstruction")
rep("  dfair_A", dA)
print("B. |fair_off(sampler px/cb/pmom) - fair_online|             <- full offline pipeline")
rep("  dfair_B", dB_)
rep("  dpx($)", dpx); rep("  dcb($)", dcb)

# ---- C: calib refit on sampler rows ----
print("\nC. calib (db,dr): logged vs offline rolling refit (same 120/60)")
import torch, torch.nn as nn
class Surf(nn.Module):
    def __init__(s, ne):
        super().__init__(); s.net = nn.Sequential(nn.Linear(3 + ne, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(s, zp, zc, lt, ex):
        return s.net(torch.cat([zp.unsqueeze(-1), zc.unsqueeze(-1), lt.unsqueeze(-1), ex], -1)).squeeze(-1)
net = Surf(len(KS))
for lin, (w, bb) in zip([m for m in net.net if isinstance(m, nn.Linear)], W):
    lin.weight.data = torch.tensor(w, dtype=torch.float32); lin.bias.data = torch.tensor(bb, dtype=torch.float32)
net.eval()
def logits_t(perp, cb, tte, ex, b, dr):
    tau = (tte / 900.0).clamp(1e-4, 1.0)
    sp = torch.exp(RP + dr); sc = torch.exp(RC + dr)
    return net(((perp - b) / sp) / tau.sqrt(), ((cb - b) / sc) / tau.sqrt(), tau.log(), ex)
def fit_off(g, mask, strike, init, steps):
    db = torch.zeros(1, requires_grad=True); dr = torch.zeros(1, requires_grad=True)
    if init is not None:
        with torch.no_grad(): db[0], dr[0] = init
    o2 = torch.optim.Adam([db, dr], lr=0.05)
    tp = torch.tensor(g["spot"][mask], dtype=torch.float32); tc = torch.tensor(g["cbmid"][mask], dtype=torch.float32)
    tt = torch.tensor(g["tte"][mask], dtype=torch.float32)
    ee = torch.tensor(np.clip((g["pmom"][mask] - MU) / SD, -5, 5), dtype=torch.float32)
    mid = torch.tensor(np.clip((g["ybid"][mask] + g["yask"][mask]) / 2.0, 0.01, 0.99), dtype=torch.float32)
    for _ in range(steps):
        o2.zero_grad()
        lo = logits_t(tp, tc, tt, ee, strike + BSC * db, dr)
        nn.functional.binary_cross_entropy_with_logits(lo, mid).backward(); o2.step()
    return float(db), float(dr)
byev = {}
for c in cals: byev.setdefault(tick_of(c["inst"]), []).append(c)
for t, cs in sorted(byev.items()):
    g = grid.get(t); mm = meta.get(t)
    if g is None or not mm or mm.get("strike") is None: continue
    strike = float(mm["strike"]); init = None; rows = []
    ok = (g["spot"] > 0) & ~np.isnan(g["cbmid"]) & ~np.isnan(g["pmom"]).any(1)
    gg = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in g.items()}
    for c in sorted(cs, key=lambda x: -x["tte"]):
        B = c["tte"]
        m = (gg["tte"] > B) & (gg["tte"] <= B + 120.0)
        if m.sum() < 20: continue
        db_o, dr_o = fit_off(gg, m, strike, init, 150 if init is None else 60)
        init = (db_o, dr_o)
        rows.append((B, c["db"], db_o, c["dr"], dr_o))
    for B, dbl, dbo, drl, dro in rows:
        print(f"  {t} B={B:.0f}: db {dbl:+.4f} vs {dbo:+.4f} (d{dbl-dbo:+.4f}) | dr {drl:+.4f} vs {dro:+.4f} (d{drl-dro:+.4f})")
