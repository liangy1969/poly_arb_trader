"""Phase 1 maker simulator with QUEUE-based fills (DESIGN_MARKET_MAKING.md §2 Phase 1 / §2c rules).

One resting order of size S per side at the touch. On posting we join BEHIND the displayed size at that
price (queue model 'back'; 'half' = behind half of it). The queue ahead shrinks by the print volume at our
price on our side, and is bounded above by the displayed size at our price at every sampler row (cancels
ahead of us). We fill when prints at our price exceed the queue ahead (partial fills allowed), or when a
sweep prints beyond our price on our side (our level was consumed). Decisions are taken on the 50 ms
sampler rows and take effect after a latency (cancel 300 ms, post 150 ms). Rules: R0 tte window + 1c-tick
region, R1 price = touch, R2 side selection (naive | pull | favourable-only), R3 inventory cap + threshold
skew, R4 asymmetric hysteresis (pull now, re-join after the flag has cleared with margin for 1 s), R5
touch-move re-join + lone-quote pull. P&L: per fill mark-to-mid at 6/30/60 s; per market the net inventory
is held to settlement (meta_cache outcome).
Usage: maker_queue_sim.py DAY DUMP.csv [--policy naive|pull|fav] [--queue back|half] [--size S] [--theta 0.25]
       [--q 5] [--lat-cancel 300] [--lat-post 150] [--lone 100] [--region 1c|all] [--tte 90,300]"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("day"); ap.add_argument("dump")
ap.add_argument("--policy", default="pull"); ap.add_argument("--queue", default="back")
ap.add_argument("--size", type=float, default=1.0); ap.add_argument("--theta", type=float, default=0.25)
ap.add_argument("--q", type=float, default=5.0); ap.add_argument("--lat-cancel", type=int, default=300); ap.add_argument("--lat-post", type=int, default=150)
ap.add_argument("--lone", type=float, default=100.0); ap.add_argument("--region", default="1c"); ap.add_argument("--tte", default="90,300")
ap.add_argument("--rejoin-hold", type=int, default=1000); ap.add_argument("--quiet", action="store_true")
ap.add_argument("--max-join", type=float, default=1e9, help="post only when the displayed size at the touch is <= this (front-of-queue variant)")
ap.add_argument("--lat-dist", action="store_true", help="sample cancel/post latency from the measured live distribution (lognormal, median 10 ms, p90 ~46 ms, p99 ~160 ms) instead of constants")
ap.add_argument("--sweep-guard", type=float, default=0.0, help="print-driven pull: cancel when prints at our price in the last 100 ms consumed >= this fraction of the queue ahead (0 = off)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--imb-pull", type=float, default=0.0, help="queue-imbalance pull: cancel our side when own-side share of the touch sizes < this (Gould-Bonart / Lehalle-Mounjid)")
ap.add_argument("--imb-post", type=float, default=0.0, help="post only when own-side share of the touch sizes >= this")
ap.add_argument("--fresh-ms", type=int, default=0, help="post only within this many ms after the touch level formed (front-of-queue by timing; 0 = off)")
ap.add_argument("--jump", type=float, default=0.0, help="anticipate the move: when g toward us > this (cents) and the opposite touch level just cleared, post at that price (ahead of the current touch), first in the new queue; 0 = off")
ap.add_argument("--jump-hold", type=int, default=500, help="ms the jump target stays valid after the level cleared")
ap.add_argument("--perp-pull", type=float, default=0.0, help="perp-tick pull: cancel a side when the binance perp moved >= this many bps against it within --perp-win ms (lake prints, tick cadence); 0 = off")
ap.add_argument("--perp-win", type=int, default=300); ap.add_argument("--perp-delay", type=int, default=40, help="ms from perp exchange time to the box")
ap.add_argument("--perp-rejoin", type=int, default=1500, help="ms after a perp pull before the side may re-post")
a = ap.parse_args()
TTE_LO, TTE_HI = (float(x) for x in a.tte.split(","))
EPSP = 5e-5

tps = sorted(glob.glob(f"data/samples/trades-{a.day}.csv*"))
T = pd.concat([pd.read_csv(f) for f in tps], ignore_index=True).drop_duplicates("trade_id")
T["created_time"] = pd.to_datetime(T["created_time"], utc=True, errors="coerce")
T = T[T.created_time.notna() & T.yes_price.notna() & T["count"].notna()].copy()
T["ts_ms"] = T.created_time.astype("int64") // 1_000_000
T["taker_yes"] = T.taker_side.astype(str).str.startswith("y")
sp = glob.glob(f"data/samples/{a.day}.csv*")[0]
S = pd.read_csv(sp, usecols=["ts_ms", "ticker", "tte_ms", "ybid", "yask", "ybid_sz", "yask_sz"], dtype=str, on_bad_lines="skip")
for c in ("ts_ms", "tte_ms", "ybid", "yask", "ybid_sz", "yask_sz"):
    S[c] = pd.to_numeric(S[c], errors="coerce")
S = S.dropna(); S = S[(S.yask > S.ybid) & (S.yask - S.ybid <= 0.10)].sort_values(["ticker", "ts_ms"])
D = pd.read_csv(a.dump, usecols=["ticker", "ts_ms", "fair", "mid"]).drop_duplicates(["ticker", "ts_ms"]).sort_values(["ticker", "ts_ms"])
meta = json.load(open("data/samples/meta_cache.json"))
PERP = None
if a.perp_pull > 0:
    fs = sorted(glob.glob(f"E:/crypto/data/parquet/stream=trade/venue=binance/market=USDT_PERP/symbol=BTCUSDT/date={a.day}/*.parquet"))
    P_ = pd.concat([pd.read_parquet(f, columns=["exch_ts_ns", "price"]) for f in fs]).sort_values("exch_ts_ns")
    pts_ = (P_.exch_ts_ns.to_numpy() // 1_000_000) + a.perp_delay; ppx_ = P_.price.to_numpy()
    j_ = np.searchsorted(pts_, pts_ - a.perp_win, side="right") - 1
    mv_ = np.where(j_ >= 0, 1e4 * (ppx_ - ppx_[np.clip(j_, 0, len(ppx_) - 1)]) / ppx_, 0.0)      # bps move over the trailing window
    trig = np.abs(mv_) >= a.perp_pull
    PERP = (pts_[trig], mv_[trig])
    print("perp-pull: %d perp prints, %d triggers >= %.1f bps / %d ms" % (len(pts_), trig.sum(), a.perp_pull, a.perp_win))
BID, ASK = 0, 1
RNG = np.random.default_rng(a.seed)


def lat(kind):
    """decision-to-effect latency in ms for a cancel or a post"""
    if a.lat_dist:
        return 5.0 + 10.0 * float(np.exp(1.2 * RNG.standard_normal()))
    return float(a.lat_cancel if kind == "cancel" else a.lat_post)


def policy_on(pol, g, theta_side):
    if pol == "naive":
        return True
    if pol == "pull":
        return g >= -theta_side
    if pol == "fav":
        return g > theta_side
    raise ValueError(pol)


def sim_market(tk, B, P, Dk):
    ts = B.ts_ms.to_numpy(); tte = B.tte_ms.to_numpy() / 1000.0
    yb, ya, ybs, yas = (B[c].to_numpy() for c in ("ybid", "yask", "ybid_sz", "yask_sz"))
    mid = 0.5 * (yb + ya)
    if len(Dk):
        dts = Dk.ts_ms.to_numpy(); jd = np.searchsorted(dts, ts, side="right") - 1
        okd = (jd >= 0) & (ts - dts[np.clip(jd, 0, len(dts) - 1)] <= 400)
        gbid = np.full(len(ts), np.nan); gbid[okd] = 100 * (Dk.fair.to_numpy()[jd[okd]] - Dk.mid.to_numpy()[jd[okd]])
    else:
        gbid = np.full(len(ts), np.nan)
    pts, ppx, pcnt, ptk = P.ts_ms.to_numpy(), P.yes_price.to_numpy(), P["count"].to_numpy(), P.taker_yes.to_numpy()
    st = [dict(on=False, price=np.nan, ahead=0.0, rem=0.0, pend=[], off_reason=None, clear_since=None, posted_ts=0, burst=[], jumped=False) for _ in range(2)]
    jump_px = [np.nan, np.nan]; jump_until = [0, 0]
    q = 0.0; fills = []; presence_ms = [0, 0]; actions = 0
    level_since = [ts[0], ts[0]]; last_touch = [yb[0], ya[0]]
    perp_block = [0, 0]
    if PERP is not None:
        lo_, hi_ = np.searchsorted(PERP[0], ts[0]), np.searchsorted(PERP[0], ts[-1], side="right")
        ev_ts, ev_mv = PERP[0][lo_:hi_], PERP[1][lo_:hi_]
    else:
        ev_ts, ev_mv = np.zeros(0), np.zeros(0)
    ei = 0
    pi = 0; n = len(ts)

    def apply_pending(side, now):
        s_ = st[side]; keep = []
        nonlocal actions
        for eff, act, px in s_["pend"]:
            if eff <= now:
                if act == "post" and not s_["on"]:
                    s_["on"] = True; s_["price"] = px; s_["rem"] = a.size; s_["posted_ts"] = eff
                    s_["jumped"] = bool(a.jump > 0 and abs(px - jump_px[side]) < EPSP) if not np.isnan(jump_px[side]) else False
                    # queue ahead = displayed size at that price at posting time (row at/before eff)
                    k = np.searchsorted(ts, eff, side="right") - 1
                    k = max(k, 0)
                    lvl = (ybs[k] if side == BID else yas[k]) if abs((yb[k] if side == BID else ya[k]) - px) < EPSP else 0.0   # at a jump level nobody displays yet -> first in line
                    s_["ahead"] = lvl if a.queue == "back" else 0.5 * lvl
                    actions += 1
                elif act == "cancel" and s_["on"]:
                    s_["on"] = False; s_["price"] = np.nan; s_["rem"] = 0.0; actions += 1
            else:
                keep.append((eff, act, px))
        s_["pend"] = keep

    def record_fill(side, t, px, qty, kind):
        nonlocal q
        sign = 1.0 if side == BID else -1.0
        q += sign * qty
        fills.append((t, side, px, qty, ("jump-" + kind) if st[side]["jumped"] else kind, t - st[side]["posted_ts"]))

    for i in range(n):
        t = ts[i]
        # prints since the previous row (strictly after ts[i-1], up to and including t)
        while pi < len(pts) and pts[pi] <= t:
            tp, px, cnt, ty = pts[pi], ppx[pi], pcnt[pi], ptk[pi]; pi += 1
            side = ASK if ty else BID
            apply_pending(side, tp)
            s_ = st[side]
            if not s_["on"]:
                continue
            if abs(px - s_["price"]) < EPSP:
                if a.sweep_guard > 0:
                    # print-driven pull: the level is being eaten -> cancel at tape speed (before the rest of the sweep reaches us)
                    s_["burst"] = [(t_, c_) for (t_, c_) in s_["burst"] if tp - t_ <= 100] + [(tp, cnt)]
                    burst = sum(c_ for _, c_ in s_["burst"])
                    if burst >= a.sweep_guard * max(s_["ahead"], 1.0) and not any(k_ == "cancel" for _, k_, _ in s_["pend"]):
                        s_["pend"].append((tp + lat("cancel"), "cancel", np.nan)); s_["off_reason"] = "sweep_guard"; s_["clear_since"] = None
                if s_["ahead"] >= cnt:
                    s_["ahead"] -= cnt
                else:
                    qty = min(s_["rem"], cnt - s_["ahead"]); s_["ahead"] = 0.0
                    if qty > 0:
                        record_fill(side, tp, s_["price"], qty, "queue"); s_["rem"] -= qty
                        if s_["rem"] <= 1e-9:
                            s_["on"] = False; s_["price"] = np.nan; s_["off_reason"] = "filled"; s_["clear_since"] = tp
            elif (side == BID and px < s_["price"] - EPSP) or (side == ASK and px > s_["price"] + EPSP):
                # the sweep printed beyond our level: our level was consumed first
                qty = s_["rem"]
                if qty > 0:
                    record_fill(side, tp, s_["price"], qty, "sweep")
                s_["on"] = False; s_["price"] = np.nan; s_["rem"] = 0.0; s_["off_reason"] = "filled"; s_["clear_since"] = tp
        # perp-tick pulls since the previous row (tick cadence): a down move threatens the bid, an up move the ask
        while ei < len(ev_ts) and ev_ts[ei] <= t:
            te, mv = ev_ts[ei], ev_mv[ei]; ei += 1
            side = BID if mv < 0 else ASK
            s_ = st[side]
            perp_block[side] = te + a.perp_rejoin
            if s_["on"] and not any(k_ == "cancel" for _, k_, _ in s_["pend"]):
                s_["pend"].append((te + lat("cancel"), "cancel", np.nan)); s_["off_reason"] = "perp"; s_["clear_since"] = None
        for side in (BID, ASK):
            apply_pending(side, t)
        eligible = (TTE_LO <= tte[i] <= TTE_HI) and (a.region == "all" or 0.10 <= mid[i] <= 0.90)
        closing = tte[i] < 45.0
        for side in (BID, ASK):
            s_ = st[side]
            if s_["on"]:
                presence_ms[side] += 50
            touch = yb[i] if side == BID else ya[i]
            lvl = ybs[i] if side == BID else yas[i]
            if abs(touch - last_touch[side]) >= EPSP:
                level_since[side] = t; last_touch[side] = touch          # a new touch level formed on this side
            # jump target: the opposite touch level just cleared in our direction (ask moved up for a bid; bid moved down for an ask)
            if a.jump > 0 and i > 0:
                if side == BID and ya[i] > ya[i - 1] + EPSP:
                    jump_px[side] = ya[i - 1]; jump_until[side] = t + a.jump_hold
                if side == ASK and yb[i] < yb[i - 1] - EPSP:
                    jump_px[side] = yb[i - 1]; jump_until[side] = t + a.jump_hold
            opp = yas[i] if side == BID else ybs[i]
            share = lvl / max(lvl + opp, 1e-9)                          # own-side share of the touch sizes
            gs = gbid[i] if side == BID else -gbid[i]
            adds = (side == BID and q >= 0) or (side == ASK and q <= 0)      # this side would increase |q|
            theta_side = a.theta * max(0.0, 1.0 - abs(q) / a.q) if adds else a.theta
            want = eligible and not closing and not (adds and abs(q) >= a.q) and not np.isnan(gs) and policy_on(a.policy, gs, theta_side)
            if not s_["on"] and t < perp_block[side]:
                want = False                                             # perp pull cool-down
            if not s_["on"] and lvl > a.max_join:
                want = False                                             # front-of-queue variant: only join thin levels
            if not s_["on"] and a.imb_post > 0 and share < a.imb_post:
                want = False                                             # imbalance post gate: only post behind the bigger queue
            if not s_["on"] and a.fresh_ms > 0 and t - level_since[side] > a.fresh_ms:
                want = False                                             # freshness gate: only join a level that just formed
            if s_["on"] and a.imb_pull > 0 and share < a.imb_pull:
                want = False                                             # imbalance pull: our queue is the small one -> it gets consumed first
            if closing and adds:
                want = False
            if closing or tte[i] < TTE_LO:
                want = want and not adds          # only the reducing side below the window
            # bound the queue ahead by what is displayed at our price now
            if s_["on"]:
                s_["ahead"] = min(s_["ahead"], lvl if abs(touch - s_["price"]) < EPSP else s_["ahead"])
            # jump price: ahead of the current touch, only while it does not cross and the model still says the move is toward us
            jp = np.nan
            if a.jump > 0 and want and t <= jump_until[side] and not np.isnan(jump_px[side]) and not np.isnan(gs) and gs > a.jump:
                cand = jump_px[side]
                if (side == BID and cand < ya[i] - EPSP and cand > yb[i] - EPSP) or (side == ASK and cand > yb[i] + EPSP and cand < ya[i] + EPSP):
                    jp = cand
            target_px = jp if not np.isnan(jp) else touch
            has_pending = bool(s_["pend"])
            if s_["on"] and not has_pending:
                reason = None
                if not want:
                    reason = "flag" if (eligible and not closing) else "window"
                elif abs(target_px - s_["price"]) >= EPSP and (np.isnan(jp) or target_px > s_["price"] + EPSP if side == BID else True) and (np.isnan(jp) or target_px < s_["price"] - EPSP if side == ASK else True):
                    reason = "move"                                          # R5: touch moved (or a jump level opened ahead of us): re-post at target
                elif lvl < a.lone and np.isnan(jp) and not s_["jumped"]:
                    reason = "lone"                                          # R5: alone / thin at the touch
                if reason:
                    lc = lat("cancel"); s_["pend"].append((t + lc, "cancel", np.nan)); s_["off_reason"] = reason; s_["clear_since"] = None
                    if reason == "move" and want:
                        s_["pend"].append((t + lc + lat("post"), "post", target_px))
            elif not s_["on"] and not has_pending and want:
                # R4 re-join hysteresis after a flag pull: the side's own condition with margin, held rejoin_hold ms
                if s_["off_reason"] == "flag":
                    margin_ok = policy_on(a.policy, gs, theta_side * 0.5) if a.policy != "naive" else True
                    if not margin_ok:
                        s_["clear_since"] = None
                    else:
                        if s_["clear_since"] is None:
                            s_["clear_since"] = t
                        if t - s_["clear_since"] >= a.rejoin_hold:
                            s_["pend"].append((t + lat("post"), "post", target_px)); s_["off_reason"] = None
                else:
                    s_["pend"].append((t + lat("post"), "post", target_px)); s_["off_reason"] = None
    # mark fills
    out = []
    m = meta.get(tk) or {}
    res = m.get("result") if isinstance(m, dict) else None
    settle = 1.0 if res == "yes" else 0.0 if res == "no" else np.nan
    for (t, side, px, qty, kind, age) in fills:
        sign = 1.0 if side == BID else -1.0
        marks = []
        for h in (6, 30, 60):
            j = min(np.searchsorted(ts, t + 1000 * h, side="left"), n - 1); marks.append(100 * sign * (mid[j] - px))
        out.append(dict(ticker=tk, ts_ms=t, side=side, px=px, qty=qty, kind=kind, age_ms=age, pnl6=marks[0], pnl30=marks[1], pnl60=marks[2], pnl_settle=100 * sign * (settle - px)))
    return out, q, presence_ms, actions


rows, inv, pres, acts = [], {}, {}, {}
for tk, B in S.groupby("ticker"):
    P = T[T.ticker == tk].sort_values("ts_ms")
    if len(P) == 0:
        continue
    Dk = D[D.ticker == tk]
    f, q, pm, ac = sim_market(tk, B, P, Dk)
    rows.extend(f); inv[tk] = q; pres[tk] = pm; acts[tk] = ac
F = pd.DataFrame(rows)
mk = list(inv.keys())
def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return v.mean(), (v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 3 and v.std() > 0 else 0.0), len(v)
per_mkt = F.groupby("ticker").agg(n=("qty", "size"), qty=("qty", "sum"), pnl6=("pnl6", "mean"), pnl30=("pnl30", "mean"), pnl_settle_tot=("pnl_settle", lambda x: (x * F.loc[x.index, "qty"]).sum())) if len(F) else pd.DataFrame()
tot = per_mkt.reindex(mk).fillna({"n": 0, "qty": 0, "pnl_settle_tot": 0})
pres_min = np.array([(pres[t][0] + pres[t][1]) / 60000.0 for t in mk])
print("policy=%s queue=%s size=%g theta=%.2f Q=%g lat %s lone<%g sweep-guard %.2f imb-pull %.2f imb-post %.2f fresh %dms region=%s tte=%s | markets %d" % (
    a.policy, a.queue, a.size, a.theta, a.q, "measured-dist" if a.lat_dist else "%d/%d ms" % (a.lat_cancel, a.lat_post), a.lone, a.sweep_guard, a.imb_pull, a.imb_post, a.fresh_ms, a.region, a.tte, len(mk)))
nf, tnf, _ = clus(tot["n"]); ntot = int(tot["n"].sum())
print("fills: total %d | per market %.1f | contracts per market %.2f | side-minutes in book per market %.1f -> fills per side-minute %.2f | actions per market %.0f" % (
    ntot, nf, tot["qty"].mean(), pres_min.mean(), ntot / max(pres_min.sum(), 1e-9), np.mean(list(acts.values()))))
if len(F):
    for col, lab in (("pnl6", "P&L/fill @6s"), ("pnl30", "@30s"), ("pnl60", "@60s"), ("pnl_settle", "settle per contract")):
        m_, t_, n_ = clus(F.groupby("ticker")[col].mean()); print("  %-22s %+6.2fc  (market-clustered t %+5.2f, n=%d markets with fills; raw mean %+6.2fc over %d fills)" % (lab, m_, t_, n_, F[col].mean(), len(F)))
    m_, t_, n_ = clus(tot["pnl_settle_tot"]); print("  per-MARKET P&L to settlement (net inventory held): %+6.2fc per market (t %+5.2f, n=%d incl. zero-fill markets) | inventory at expiry: mean |q| %.2f" % (m_, t_, n_, np.mean([abs(v) for v in inv.values()])))
    print("  fills by side: bid %d ask %d | markets with 0 fills: %d" % ((F.side == BID).sum(), (F.side == ASK).sum(), (tot["n"] == 0).sum()))
    for k in ("queue", "sweep", "jump-queue", "jump-sweep"):
        h = F[F.kind == k]
        if len(h):
            print("  fill type %-6s %5.1f%% | P&L @6s %+6.2fc @30s %+6.2fc settle %+6.2fc | age since post: median %.1fs p10 %.1fs" % (k, 100 * len(h) / len(F), h.pnl6.mean(), h.pnl30.mean(), h.pnl_settle.mean(), h.age_ms.median() / 1000, h.age_ms.quantile(.1) / 1000))
    for lo, hi in ((0, 1000), (1000, 5000), (5000, 1e12)):
        h = F[(F.age_ms >= lo) & (F.age_ms < hi)]
        if len(h):
            print("  age %5.0f-%5.0fs %5.1f%% | P&L @6s %+6.2fc @30s %+6.2fc" % (lo / 1000, min(hi, 1e6) / 1000, 100 * len(h) / len(F), h.pnl6.mean(), h.pnl30.mean()))
if not a.quiet and len(F):
    F.to_parquet(f"data/nowcast/qsim_{a.day}_{a.policy}_{a.queue}.parquet")
