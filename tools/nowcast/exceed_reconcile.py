"""Live-vs-harness reconciliation of the ARMED exceed trader (DESIGN_LIVE_SIM_CONSISTENCY level 3): for a live window,
(1) crossings: every live `signal` vs the harness's zero-latency crossings (same cut/gate) within +-2 s on the same
market; (2) fills: every live fill (chase log) vs the harness's 100 ms fill on the matched crossing — price, outcome,
net; (3) totals: live realized P&L (from the executor TRADE/settlement outcomes given in a small CSV or inferred from
the harness outcomes on matched markets) vs harness P&L on the same window.
Usage: exceed_reconcile.py LIVE_LOG HARNESS_LAT0_TRADES HARNESS_LAT100_TRADES --from 'YYYY-MM-DDTHH:MM' --to '...' [--label X]"""
import argparse
import json
import re

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("live"); ap.add_argument("lat0"); ap.add_argument("lat100")
ap.add_argument("--from", dest="t0", required=True); ap.add_argument("--to", dest="t1", required=True); ap.add_argument("--label", default="")
ap.add_argument("--model", default="x1"); ap.add_argument("--delta", type=float, default=0.48)
a = ap.parse_args()
T0 = pd.Timestamp(a.t0, tz="UTC"); T1 = pd.Timestamp(a.t1, tz="UTC")
norm = lambda t: re.sub(r"^kalshi\.|\.(YES|NO)$", "", t)

# ── live: signals, executor dispositions, chase fills ──
sigs, disp, fills = [], {}, {}
rx_disp = re.compile(r"exec: \[(exceed-\d+)\] (Entering|Holding|Abandoned|Rejected|Shadow)[^\n]*?(kalshi\.\S+)?")
rx_chase = re.compile(r"CHASE trade=(exceed-\d+) signal_ask=([\d.]+) fill_ask=([\d.]+) drift_c=([+-]?[\d.]+) ask_sz=([\d.]+) filled=(true|false)")
for line in open(a.live, encoding="utf-8", errors="replace"):
    ts = pd.Timestamp(line[:27].rstrip("Z"), tz="UTC") if line[:4].isdigit() else None
    if ts is None or not (T0 <= ts <= T1):
        continue
    if '"strategy":"exceed"' in line:
        js = json.loads(line[line.index("{"):]); r = dict(re.findall(r"(\w+)=([+-]?[\d.]+)", js["reason"]))
        sigs.append(dict(tid="exceed-%d" % js["ts_ns"], ts_ms=js["ts_ns"] // 1_000_000, tk=norm(js["target"]), dir=js["direction"], s=float(r["s"]), mid=float(r["mid"]), tte=float(r["tte"]), yes_px=js["trigger"]["yes_price"]))
        continue
    m = rx_disp.search(line)
    if m:
        tid, what = m.group(1), m.group(2)
        if what == "Rejected":
            why = re.search(r"Rejected \S+ (.*)", line).group(1).strip()
            what = "Rejected:" + ("bucket" if "bucket" in why else "cap" if "cap" in why else "cooldown" if "cooldown" in why else why[:20])
        disp.setdefault(tid, []).append(what)
        continue
    m = rx_chase.search(line)
    if m:
        fills[m.group(1)] = dict(signal_ask=float(m.group(2)), fill_ask=float(m.group(3)), drift_c=float(m.group(4)), filled=m.group(6) == "true")
S = pd.DataFrame(sigs)
S["disp"] = S.tid.map(lambda t: "/".join(dict.fromkeys(disp.get(t, ["none"]))))
S["fill_px"] = S.tid.map(lambda t: fills.get(t, {}).get("fill_ask", np.nan))
S["filled"] = S.tid.map(lambda t: fills.get(t, {}).get("filled", False))

# ── harness ──
def load(path):
    df = pd.read_csv(path, usecols=["model", "delta", "ticker", "ts_ms", "date", "side_yes", "gap", "mid0", "tte", "cost", "won", "net", "slip"])
    df = df[(df.model == a.model) & np.isclose(df.delta, a.delta)].copy()
    df["tk"] = df.ticker.map(norm); df["t"] = pd.to_datetime(df.ts_ms, unit="ms", utc=True)
    return df[(df.t >= T0) & (df.t <= T1)].sort_values("ts_ms")
H0, H1 = load(a.lat0), load(a.lat100)
print("=== %s window %s .. %s | model %s cut %.2f" % (a.label, a.t0, a.t1, a.model, a.delta))
print("live: %d signals (%d filled, %d sent) | harness: %d crossings (latency 0), %d fills (100 ms + chase)" % (len(S), int(S.filled.sum()), int(S.disp.str.contains("Entering").sum()), len(H0), len(H1)))


def nearest(df, tk, ts_ms, tol=2000):
    g = df[df.tk == tk]
    if g.empty:
        return None
    i = (g.ts_ms - ts_ms).abs().idxmin()
    return g.loc[i] if abs(int(g.loc[i, "ts_ms"]) - ts_ms) <= tol else None


# (1) crossings
print("\n--- crossings: live signal -> harness crossing within 2 s (same market)")
matched = 0; used = set()
for r in S.itertuples():
    h = nearest(H0, r.tk, r.ts_ms)
    tag = "NONE"
    if h is not None:
        matched += 1; used.add(h.name)
        tag = "%+d ms dir %+d |s| %.3f" % (h.ts_ms - r.ts_ms, 1 if h.side_yes else -1, h.gap)
    print("  %s %s dir %+d s %+.3f mid %.4f | %-28s | harness: %s" % (pd.Timestamp(r.ts_ms, unit="ms").strftime("%m-%d %H:%M:%S.%f")[:-3], r.tk[-12:], r.dir, r.s, r.mid, r.disp[:28], tag))
print("  matched %d/%d live signals; harness-only crossings: %d" % (matched, len(S), len(H0) - len(used)))
for h in H0[~H0.index.isin(used)].itertuples():
    print("    harness-only %s %s dir %+d |s| %.3f mid %.4f tte %.0f" % (h.t.strftime("%m-%d %H:%M:%S.%f")[:-3], h.tk[-12:], 1 if h.side_yes else -1, h.gap, h.mid0, h.tte))

# (2) fills and P&L on matched live fills
print("\n--- fills: live fill vs harness 100 ms fill on the matched crossing (net = settle - cost - fee, cents)")
rows = []
for r in S[S.filled].itertuples():
    h = nearest(H1, r.tk, r.ts_ms)
    hn = h.net * 100 if h is not None else np.nan
    print("  %s %s dir %+d live fill %.3f | harness fill %s net %s" % (pd.Timestamp(r.ts_ms, unit="ms").strftime("%m-%d %H:%M:%S")[:-3] if False else pd.Timestamp(r.ts_ms, unit="ms").strftime("%m-%d %H:%M:%S"), r.tk[-12:], r.dir, r.fill_px,
          "%.3f" % h.cost if h is not None else "NONE (missed or no crossing)", "%+.1fc" % hn if h is not None else "-"))
    rows.append((r.tk, r.dir, r.fill_px, h.cost if h is not None else np.nan, hn))
R = pd.DataFrame(rows, columns=["tk", "dir", "live_px", "h_px", "h_net"])
print("  live fills %d; with a harness fill on the same crossing %d; mean |price diff| %.2fc" % (len(R), R.h_px.notna().sum(), 100 * (R.live_px - R.h_px).abs().mean()))
print("\n--- harness totals in the window (100 ms fills): n %d, net %+.1fc/trade, sum %+.2f$, win %.0f%%" % (len(H1), 100 * H1.net.mean() if len(H1) else 0, H1.net.sum(), 100 * H1.won.mean() if len(H1) else 0))
