"""Turnaround model: on TAIL rows (YES mid < px or > 1-px), predict P(settle = CHEAP side) at settlement
horizon. The market's own cheap-side mid is the anchor and THE bar: report dBCE = model - market with a
DAY-clustered t (negative = model beats the quote). Known-good trick: the head learns a logit DELTA on the
market anchor (--raw trains an unanchored control head).

Rows come from the persistent v10 set (data/nowcast/midmove_v10). Features are flag-selectable groups, all
expressed in CHEAP-side convention (high-tail rows are mirrored so "up" always means "toward the cheap win"):
  hist  mid-history logit diffs at LAGS (mirrored)
  perp  perp raw-dollar lags (mirrored)
  kz    strike geometry: perp-K, cb-K ($ and /sigma*sqrt(tte), mirrored) + gaussian-fair-minus-mid (mirrored)
  mx    book snapshot (imb/micro mirrored, sizes as-is)
  lake  the standard 13 lf_* columns from data/nowcast/midmove_v10_lake (tfi/vol mirrored)
Always-on ctx: log tte, sigma, spread, cheap logit, mid(1-mid), side flag.

Split by day: --split tr_end,va_end (test = after). Eval: pooled + day-t dBCE on val/test, per tte/price
slices, and a one-entry-per-(event,side) trade sim at --edges thresholds (buy cheap ask of the NEXT row
+1c fee model, hold to settle) with an event-clustered t.
Usage: train_turnaround.py [--feat hist,perp,kz] [--tte-min 40 --tte-max 600] [--px 0.10] [--seed 0] ...
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="E:/poly/crypto_trader/data/nowcast/midmove_v10")
ap.add_argument("--feat", default="hist,perp,kz", help="feature groups: hist,perp,kz,mx,lake")
ap.add_argument("--px", type=float, default=0.10, help="tail definition: YES mid < px or > 1-px")
ap.add_argument("--tte-min", type=float, default=40.0)
ap.add_argument("--tte-max", type=float, default=600.0)
ap.add_argument("--split", default="2026-10-10,2026-10-17")
ap.add_argument("--test-end", default="2100-01-01", help="drop days after this from the test split")
ap.add_argument("--target", default="settle", choices=("settle", "up30", "up5", "up1"), help="label: settlement, or cheap-side mid +>=1c at 30s/5s/1s (delta heads train UNANCHORED; sims still settle-P&L)")
ap.add_argument("--raw", action="store_true", help="unanchored head (control); default learns a delta on the market logit")
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--delta-l2", type=float, default=0.0, help="L2 penalty on the delta output (shrink toward the market anchor)")
ap.add_argument("--event-weight", action="store_true", help="weight the loss so every (event,side) counts equally, not every row")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=8192)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--seeds", default="", help="comma list: train an ensemble in one run (overrides --seed)")
ap.add_argument("--export", default="", help="write the Rust-loadable ensemble json (members + mu/sd + train-quantile thresholds + test vectors)")
ap.add_argument("--edges", default="0.01,0.02,0.03,0.05", help="sim entry thresholds on (p_model - p_market)")
ap.add_argument("--eval-tte", default="300:60,600:60", help="eval windows MAX:MIN, comma list")
ap.add_argument("--save-pred", default="", help="write test-row predictions to this npz")
a = ap.parse_args()
torch.set_num_threads(4); torch.manual_seed(a.seed); np.random.seed(a.seed)
EPS = 1e-4
LAGS = np.array([1, 2, 3, 5, 10, 25, 50, 100, 200, 300])  # 0.2 s .. 60 s


def lg(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def load_day(p):
    z = np.load(p)
    ctx = z["ctx"]
    mid, tte, sig, spr, bid, ask = (ctx[:, i] for i in range(6))
    w = (tte >= a.tte_min) & (tte <= a.tte_max) & ((mid < a.px) | (mid > 1 - a.px)) & (bid > 0) & (ask > bid)
    if not w.any():
        return None
    mid, tte, sig, spr, bid, ask = mid[w], tte[w], sig[w], spr[w], bid[w], ask[w]
    lo = mid < 0.5                      # low tail: cheap side = YES
    m = np.where(lo, 1.0, -1.0)         # mirror: +1 keeps, -1 flips direction-sensitive features
    settle = z["settle"][w]
    y = np.where(lo, settle, 1 - settle).astype(np.float32)          # cheap side wins
    if a.target == "settle":
        yt, ok_t = y, np.ones(len(y), bool)
    else:
        J = {"up1": 1, "up5": 2, "up30": 3}[a.target]                # fut horizons: 0.2s,1s,5s,30s
        fm = z["fut"][w].reshape(len(y), 4, 3)[:, J, 0]
        later = np.where(lo, fm, 1 - fm)
        yt = (later - np.where(lo, mid, 1 - mid) >= 0.01).astype(np.float32)
        ok_t = np.isfinite(fm)
    f3 = z["fut"][w].reshape(len(y), 4, 3)
    # cheap-side ask at +1s and +5s (NaN when the book was not sane there)
    ask1 = np.where(lo, f3[:, 1, 2], 1 - f3[:, 1, 1]).astype(np.float32)
    ask5 = np.where(lo, f3[:, 2, 2], 1 - f3[:, 2, 1]).astype(np.float32)
    p_mkt = np.where(lo, mid, 1 - mid).astype(np.float32)            # market prob of cheap win (mid)
    cheap_ask = np.where(lo, ask, 1 - bid).astype(np.float32)        # buy price of the cheap side
    F = []
    groups = a.feat.split(",")
    if "hist" in groups:
        mh = z["mh"][w][:, LAGS - 1].astype(np.float32)
        F.append(m[:, None] * (lg(mid)[:, None] - lg(mh)))           # cheap-logit gain since each lag
    if "perp" in groups:
        ph = z["ph"][w][:, LAGS - 1].astype(np.float32)              # perp(t-lag) - perp(now), $ (fp16-safe diffs)
        F.append(m[:, None] * ph)
    if "kz" in groups:
        kz = z["kz"][w].astype(np.float32)
        F.append(np.column_stack([m * kz[:, 0], m * kz[:, 1], m * kz[:, 2], m * kz[:, 3], m * kz[:, 4], m * kz[:, 5]]))
    if "mx" in groups:
        mx = z["mx"][w].astype(np.float32)
        mx2 = mx.copy()
        for c in (2, 3, 6, 10):                                       # imb / micro / perp-imb / lastdir flip with side
            if c < mx2.shape[1]:
                mx2[:, c] = m * mx2[:, c]
        F.append(mx2)
    if "lake" in groups:
        lp = os.path.join(os.path.dirname(a.data), "midmove_v10_lake", os.path.basename(p).replace(".npz", ".lake.npz"))
        zl = np.load(lp)
        ln = [str(x) for x in zl["names"]]
        want = ["flow_tfi_1s", "flow_tfi_5s", "flow_tfi_30s", "flow_vol_1s", "flow_n_1s", "flow_n_5s",
                "depth_band5", "depth_band10", "depth_band25", "depth_depth10", "depth_spread_bps",
                "depth_dimb5_1s", "depth_dimb5_5s", "depth_dband10_1s"]
        idx = [ln.index(n) for n in want if n in ln]
        lk = zl["lk"][w][:, idx].astype(np.float32)
        for j, n in enumerate([ln[i] for i in idx]):                  # signed flow features mirror with side
            if n.startswith("flow_tfi") or "dimb" in n or "dband" in n:
                lk[:, j] = m * lk[:, j]
        F.append(lk)
    F.append(np.column_stack([np.log(np.maximum(tte, 1.0)), sig, 100 * spr, lg(p_mkt), mid * (1 - mid), lo.astype(np.float32)]))
    X = np.concatenate(F, 1).astype(np.float32)
    tk = z["tk"][w].astype(np.int64)
    return X[ok_t], y[ok_t], p_mkt[ok_t], cheap_ask[ok_t], tte[ok_t].astype(np.float32), tk[ok_t], z["ts"][w][ok_t].astype(np.int64), yt[ok_t], ask1[ok_t], ask5[ok_t]


days = sorted(glob.glob(os.path.join(a.data, "*.npz")))
tr_end, va_end = a.split.split(",")
split = {"tr": [], "va": [], "te": []}
for p in days:
    d = os.path.basename(p)[:10]
    k = "tr" if d <= tr_end else "va" if d <= va_end else "te" if d <= a.test_end else None
    if k:
        split[k].append(p)

D = {}
for k, ps in split.items():
    Xs, ys, pm, ca, tt, tks, tss, dys, yts, a1s, a5s = [], [], [], [], [], [], [], [], [], [], []
    for di, p in enumerate(ps):
        r = load_day(p)
        if r is None:
            continue
        X, y, p_mkt, cheap_ask, tte, tk, ts, yt, ask1, ask5 = r
        Xs.append(X); ys.append(y); pm.append(p_mkt); ca.append(cheap_ask); tt.append(tte)
        tks.append(di * 10000 + tk); tss.append(ts); yts.append(yt); a1s.append(ask1); a5s.append(ask5)
        dys.append(np.full(len(y), di, np.int32))
    X = np.concatenate(Xs); fin = np.isfinite(X).all(1)
    D[k] = [X[fin], np.concatenate(ys)[fin], np.concatenate(pm)[fin], np.concatenate(ca)[fin],
            np.concatenate(tt)[fin], np.concatenate(tks)[fin], np.concatenate(tss)[fin], np.concatenate(dys)[fin],
            np.concatenate(yts)[fin], np.concatenate(a1s)[fin], np.concatenate(a5s)[fin]]

mu, sd = D["tr"][0].mean(0), D["tr"][0].std(0) + 1e-6
for k in D:
    D[k][0][:] = (D[k][0] - mu) / sd
NF = D["tr"][0].shape[1]
print("feat=%s NF=%d px=%.2f tte %g-%g | rows tr %d va %d te %d | base cheap-win tr %.3f te %.3f" % (
    a.feat, NF, a.px, a.tte_min, a.tte_max, len(D["tr"][0]), len(D["va"][0]), len(D["te"][0]),
    D["tr"][1].mean(), D["te"][1].mean()), flush=True)

if a.target != "settle":
    a.raw = True
Xt = torch.from_numpy(D["tr"][0]); Yt = torch.from_numpy(D["tr"][8 if a.target != "settle" else 1])
At = torch.from_numpy(lg(D["tr"][2]))                                  # market anchor logit
bce = nn.BCEWithLogitsLoss()
if a.event_weight:
    cnt = pd.Series(D["tr"][5]).groupby(D["tr"][5]).transform("size").to_numpy()
    Wt = torch.from_numpy((len(cnt) / cnt / len(np.unique(D["tr"][5]))).astype(np.float32))


def predict_net(net, k):
    net.eval(); out = []
    with torch.no_grad():
        Xk = torch.from_numpy(D[k][0]); Ak = torch.from_numpy(lg(D[k][2]))
        for i in range(0, len(Xk), 65536):
            z_ = net(Xk[i:i + 65536]).squeeze(-1)
            logit = z_ if a.raw else Ak[i:i + 65536] + z_
            out.append(torch.sigmoid(logit).numpy())
    return np.concatenate(out)


def predict(k):
    return np.mean([predict_net(n_, k) for n_ in NETS], 0)


def bce_np(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = nn.Sequential(nn.Linear(NF, a.hidden), nn.ReLU(), nn.Dropout(a.dropout),
                        nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Linear(a.hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    best_dv, best_state = float("inf"), None
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        net.train()
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            z_ = net(Xt[j]).squeeze(-1)
            logit = z_ if a.raw else At[j] + z_
            if a.event_weight:
                raw = nn.functional.binary_cross_entropy_with_logits(logit, Yt[j], reduction="none")
                loss = (raw * Wt[j]).mean() + a.delta_l2 * (z_ ** 2).mean()
            else:
                loss = bce(logit, Yt[j]) + a.delta_l2 * (z_ ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
        # epoch-select on VAL metric (dBCE vs market for settle; plain target BCE for delta heads)
        pv = predict_net(net, "va")
        if a.target == "settle":
            dv = (bce_np(D["va"][1], pv) - bce_np(D["va"][1], D["va"][2])).mean()
        else:
            dv = bce_np(D["va"][8], pv).mean()
        print("s%d ep %d loss %.5f | val %.5f" % (seed, ep, tot / len(perm), dv), flush=True)
        if dv < best_dv:
            best_dv, best_state = dv, {k2: v2.clone() for k2, v2 in net.state_dict().items()}
    net.load_state_dict(best_state)
    print("s%d best val %.5f" % (seed, best_dv), flush=True)
    return net


SEEDS = [int(x) for x in str(a.seeds).split(",")] if a.seeds else [a.seed]
NETS = [train_one(s_) for s_ in SEEDS]

if a.export:
    import json
    ptr = predict("tr")
    thresholds = {("q%02d" % (100 * q)): float(np.quantile(ptr, q)) for q in (0.90, 0.95, 0.98, 0.99)}
    members = []
    for n_ in NETS:
        layers = [{"w": lin.weight.detach().numpy().tolist(), "b": lin.bias.detach().numpy().tolist()}
                  for lin in n_ if isinstance(lin, nn.Linear)]
        members.append({"layers": layers})
    rng = np.random.RandomState(7)
    ix = rng.choice(len(D["te"][0]), 8, replace=False)
    raw_x = (D["te"][0][ix] * sd + mu).tolist()                        # pre-standardization features
    pte = predict("te")
    tvs = [{"x": raw_x[i_], "p": float(pte[ix[i_]])} for i_ in range(len(ix))]
    js = {"kind": "turnaround", "target": a.target, "feat": a.feat, "lags": LAGS.tolist(),
          "px": a.px, "tte": [a.tte_min, a.tte_max], "raw_head": bool(a.raw),
          "note": "cheap-side %s head, %d seeds %s, split %s (test-end %s); thresholds = train-pred quantiles"
                  % (a.target, len(SEEDS), SEEDS, a.split, a.test_end),
          "mu": mu.astype(float).tolist(), "sd": sd.astype(float).tolist(),
          "thresholds": thresholds, "members": members, "test_vectors": tvs}
    json.dump(js, open(a.export, "w"))
    print("exported %s | thresholds %s" % (a.export, thresholds), flush=True)


def day_t(v, day):
    g = pd.Series(v).groupby(day).mean()
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 and g.std(ddof=1) > 0 else float("nan")


if a.target != "settle":
    from sklearn.metrics import roc_auc_score
    ptr = predict("tr")
    for k in ("va", "te"):
        X, y, pmk, cask, tte, tk, ts, dy, yt, ask1, ask5 = D[k]
        pm_ = predict(k)
        print("\n== %s target=%s: AUC %.3f (base %.3f) | SETTLE-P&L sims by train-quantile score cut:" % (
            k.upper(), a.target, roc_auc_score(yt, pm_), yt.mean()), flush=True)
        nd_ = len(np.unique(dy))
        for wmax, wmin in [tuple(map(float, s_.split(":"))) for s_ in a.eval_tte.split(",")]:
            for q in (0.90, 0.95, 0.98, 0.99):
                thr = np.quantile(ptr, q)
                m_ = (tte <= wmax) & (tte >= wmin) & (pm_ >= thr) & (cask <= a.px + 0.02)
                if m_.sum() == 0:
                    continue
                df = pd.DataFrame({"tk": tk[m_], "ts": ts[m_], "px": cask[m_], "y": y[m_]})
                e = df.sort_values("ts").drop_duplicates("tk")
                fee = 0.07 * e.px * (1 - e.px)
                net_ = e.y * (1 - e.px) - (1 - e.y) * e.px - fee
                g = net_.groupby(e.tk.values).sum()
                t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 and g.std(ddof=1) > 0 else float("nan")
                print("   %3.0f:%3.0f q%.2f: n=%4d (%4.1f/d) win %4.1f%% px %.3f | net %+6.2f$ (%+5.1fc/tr) ev-t %+5.2f" % (
                    wmax, wmin, q, len(e), len(e) / nd_, 100 * e.y.mean(), e.px.mean(), net_.sum(), 100 * net_.mean(), t), flush=True)
    if a.save_pred:
        X, y, pmk, cask, tte, tk, ts, dy, yt, ask1, ask5 = D["te"]
        np.savez_compressed(a.save_pred, p=predict("te"), y=y, yt=yt, p_mkt=pmk, cheap_ask=cask, tte=tte, tk=tk, ts=ts, day=dy, ask1=ask1, ask5=ask5)
        print("saved", a.save_pred)
    raise SystemExit(0)
for k in ("va", "te"):
    X, y, pmk, cask, tte, tk, ts, dy, yt, ask1, ask5 = D[k]
    pm_ = predict(k)
    d = bce_np(y, pm_) - bce_np(y, pmk)
    print("\n== %s: dBCE model-market %+.5f (market %.5f) DAY-t %+.2f (%d days)" % (
        k.upper(), d.mean(), bce_np(y, pmk).mean(), day_t(d, dy), len(np.unique(dy))), flush=True)
    for wmax, wmin in [tuple(map(float, s.split(":"))) for s in a.eval_tte.split(",")]:
        m_ = (tte <= wmax) & (tte >= wmin)
        if m_.sum() < 100:
            continue
        print("   tte %3.0f-%3.0fs: n=%6d dBCE %+.5f t %+.2f" % (
            wmax, wmin, m_.sum(), d[m_].mean(), day_t(d[m_], dy[m_])))
    # one-entry-per-(event) trade sim: enter at the FIRST row with edge >= thr inside each eval window
    for wmax, wmin in [tuple(map(float, s.split(":"))) for s in a.eval_tte.split(",")]:
        for thr in [float(x) for x in a.edges.split(",")]:
            m_ = (tte <= wmax) & (tte >= wmin) & (pm_ - pmk >= thr) & (cask <= a.px + 0.02)
            if m_.sum() == 0:
                print("   SIM %3.0f:%3.0f edge>=%.2f: no entries"); continue
            df = pd.DataFrame({"tk": tk[m_], "ts": ts[m_], "px": cask[m_], "y": y[m_], "dy": dy[m_]})
            e = df.sort_values("ts").drop_duplicates("tk")
            fee = 0.07 * e.px * (1 - e.px)
            net_ = e.y * (1 - e.px) - (1 - e.y) * e.px - fee
            g = net_.groupby(e.tk.values).sum()
            t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 and g.std(ddof=1) > 0 else float("nan")
            print("   SIM %3.0f:%3.0f edge>=%.2f: n=%4d (%4.1f/d) win %4.1f%% px %.3f | net %+6.2f$ (%+5.1fc/tr) ev-t %+5.2f" % (
                wmax, wmin, thr, len(e), len(e) / max(1, len(np.unique(dy))), 100 * e.y.mean(), e.px.mean(),
                net_.sum(), 100 * net_.mean(), t), flush=True)
if a.save_pred:
    X, y, pmk, cask, tte, tk, ts, dy, yt, ask1, ask5 = D["te"]
    np.savez_compressed(a.save_pred, p=predict("te"), y=y, p_mkt=pmk, cheap_ask=cask, tte=tte, tk=tk, ts=ts, day=dy, ask1=ask1, ask5=ask5)
    print("saved", a.save_pred)
