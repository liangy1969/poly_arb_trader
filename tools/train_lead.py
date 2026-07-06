#!/usr/bin/env python3
"""Train a LEAD model: predict the Kalshi mid n seconds ahead, directly.

Architecture (residual on the martingale — g can only ADD signal):
    pred_logit(t+n) = logit(mid(t)) + g( z'(t), log tau(t), logit(mid(t)) )
    g: 3 -> 32 -> 32 -> 1 tanh MLP, output scaled by tanh to +-1.5 logits.
z' uses the FROZEN base surface's per-event (b,s), fitted causally on the
event's first 10 minutes (tte>300) exactly as in deployment; g is trained
ONLY on trade-window rows (tte<=300) of the TRAIN half and evaluated on the
untouched TEST half (chronological 50/50 split). fit-the-way-you-predict.

Reports per horizon: BCE(martingale) vs BCE(model) on train and test halves,
plus gap->move slope and capture. If g cannot beat the martingale out of
sample, (perp distance, tte, current mid) contains no learnable lead.

Usage: python train_lead.py <samples1[,samples2..]> <fair_model.json> [n_list]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NO_TUNE", "1")
import sim_50ms as S  # noqa: E402

CLIP = 0.005
FIT_TTE_S = 300
EPOCHS = 60
LR = 1e-3
WD = 1e-4
G_CLAMP = 1.5


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def main():
    samples = sys.argv[1].split(",")
    model_path = sys.argv[2]
    horizons = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "5,10,30").split(",")]

    js = json.load(open(model_path))
    net, mode, clamp = S.build_surface(js)
    fwd = S.make_fwd(net, mode, clamp)
    rho = torch.tensor(float(js["rho_bar"]))
    bs = float(js.get("b_scale", 50.0))

    ev = {}
    for p in samples:
        for t, d in S.load_samples(p).items():
            if t in ev:
                ev[t] = {k: np.concatenate([ev[t][k], d[k]]) for k in d}
            else:
                ev[t] = d
    meta = json.load(open(os.path.join(os.path.dirname(samples[0]), "meta_cache.json")))
    usable = [t for t, d in ev.items()
              if meta.get(t, {}).get("strike") is not None and meta[t].get("result") in ("yes", "no")
              and (d["tte"] > FIT_TTE_S).sum() >= 2000 and (d["tte"] <= FIT_TTE_S).sum() >= 1000]
    usable.sort(key=lambda t: ev[t]["ts"][0])
    n_ev = len(usable)
    train_ev, test_ev = usable[: n_ev // 2], usable[n_ev // 2 :]
    print(f"events={n_ev} train={len(train_ev)} test={len(test_ev)}")

    # Per-event trade-window features (1s grid), (b,s) fitted causally as in prod.
    feats = {}
    for t in usable:
        d = ev[t]
        strike = float(meta[t]["strike"])
        fit = d["tte"] > FIT_TTE_S
        sl = slice(None, None, 20)
        mid_f = ((d["ybid"] + d["yask"]) / 2)[fit][sl]
        db, dr = S.fit_event(fwd, d["spot"][fit][sl], d["tte"][fit][sl], mid_f, strike, rho, bs)
        pred = d["tte"] <= FIT_TTE_S
        ts = d["ts"][pred][sl] / 1000.0
        tte = d["tte"][pred][sl]
        mid = ((d["ybid"] + d["yask"]) / 2)[pred][sl]
        b = float(strike + bs * db)
        s = float(torch.exp(rho + dr))
        tau = np.clip(tte / 900.0, 1e-4, 1.0)
        zp = ((d["spot"][pred][sl] - b) / s) / np.sqrt(tau)
        o = np.argsort(ts)
        feats[t] = {"ts": ts[o], "zp": zp[o], "ltau": np.log(tau[o]), "mid": mid[o]}

    for n in horizons:
        # rows: x=[zp, ltau, logit(mid)], base=logit(mid), y=mid(t+n)
        def rows_of(events):
            X, B, Y = [], [], []
            for t in events:
                f = feats[t]
                j = np.searchsorted(f["ts"], f["ts"] + n - 0.25)
                ok = j < len(f["ts"])
                if ok.sum() < 30:
                    continue
                lm = logit(f["mid"][ok])
                X.append(np.stack([f["zp"][ok], f["ltau"][ok], lm / 4.0], axis=1))
                B.append(lm)
                Y.append(np.clip(f["mid"][j[ok]], CLIP, 1 - CLIP))
            return (np.concatenate(X), np.concatenate(B), np.concatenate(Y))

        Xtr, Btr, Ytr = rows_of(train_ev)
        Xte, Bte, Yte = rows_of(test_ev)
        g = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        opt = torch.optim.Adam(g.parameters(), lr=LR, weight_decay=WD)
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        Bt = torch.tensor(Btr, dtype=torch.float32)
        Yt = torch.tensor(Ytr, dtype=torch.float32)

        def pred_logit(gnet, X, Bl):
            corr = G_CLAMP * torch.tanh(gnet(X).squeeze(-1) / G_CLAMP)
            return Bl + corr

        for ep in range(EPOCHS):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 8192):
                ix = perm[i : i + 8192]
                lo = pred_logit(g, Xt[ix], Bt[ix])
                loss = nn.functional.binary_cross_entropy_with_logits(lo, Yt[ix])
                opt.zero_grad()
                loss.backward()
                opt.step()

        def ev_bce(X, Bl, Y):
            with torch.no_grad():
                lo = pred_logit(g, torch.tensor(X, dtype=torch.float32), torch.tensor(Bl, dtype=torch.float32))
                p = torch.sigmoid(lo).numpy()
            yb = np.clip(Y, CLIP, 1 - CLIP)
            mart = 1 / (1 + np.exp(-Bl))
            def bce(q):
                q = np.clip(q, CLIP, 1 - CLIP)
                return float(np.mean(-(yb * np.log(q) + (1 - yb) * np.log(1 - q))))
            gap = p - mart
            move = Y - mart
            big = np.abs(gap) > 0.05
            cap = float(np.mean(np.sign(gap[big]) * move[big])) if big.sum() > 20 else float("nan")
            slope = float(np.polyfit(gap, move, 1)[0]) if len(gap) > 100 else float("nan")
            return bce(mart), bce(p), slope, cap

        m_tr, p_tr, sl_tr, cap_tr = ev_bce(Xtr, Btr, Ytr)
        m_te, p_te, sl_te, cap_te = ev_bce(Xte, Bte, Yte)
        print(f"\nn={n}s  TRAIN: mart={m_tr:.4f} model={p_tr:.4f} diff={p_tr - m_tr:+.4f} slope={sl_tr:+.3f} cap={cap_tr:+.4f}")
        print(f"n={n}s  TEST : mart={m_te:.4f} model={p_te:.4f} diff={p_te - m_te:+.4f} slope={sl_te:+.3f} cap={cap_te:+.4f}   (diff<0 = learned lead)")


if __name__ == "__main__":
    main()
