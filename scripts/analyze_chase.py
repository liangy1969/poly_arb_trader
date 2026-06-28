"""Chase-tradeoff analysis.

Reads a live-run log containing `CHASE` probe lines (one per signal's first entry
attempt: the drift over the 250ms taker delay = the chase a fill needed) and
`TRADE {json}` lines (the per-trade P&L). Reconstructs, for every candidate
chase_c, the fill rate and P&L by thresholding drift — because in the sim a fill
at chase c happens iff drift <= c, and the fill price is the touch (so a lower
chase that still fills gets the same trade).

    py scripts/analyze_chase.py /tmp/chase_study.log
"""
import json
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CHASE = re.compile(
    r"CHASE trade=(\S+) signal_ask=([\d.]+) fill_ask=([-\d.]+) "
    r"drift_c=([-\d.]+) ask_sz=([\d.]+) filled=(true|false)"
)

def main(path):
    drift = {}      # trade_id -> drift_c (cents)
    filled = {}     # trade_id -> bool
    pnl = {}        # trade_id -> pnl_net
    outcome = {}    # trade_id -> outcome

    for raw in open(path, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw)
        m = CHASE.search(line)
        if m:
            tid = m.group(1)
            drift[tid] = float(m.group(4))
            filled[tid] = m.group(6) == "true"
            continue
        i = line.find("TRADE {")
        if i >= 0:
            try:
                rec = json.loads(line[line.find("{", i):])
            except json.JSONDecodeError:
                continue
            pnl[rec["trade_id"]] = rec["pnl_net"]
            outcome[rec["trade_id"]] = rec["outcome"]

    sigs = sorted(drift)  # every signal that reached an entry attempt
    n = len(sigs)
    if n == 0:
        print("no CHASE probe lines found")
        return

    drifts = sorted(drift[t] for t in sigs)
    def pct(p):
        return drifts[min(int(p * (n - 1)), n - 1)]
    n_fill_total = sum(filled.values())

    print(f"signals (entry attempts): {n}")
    print(f"actually filled (drift<=chase ceiling 10c): {n_fill_total} ({100*n_fill_total/n:.0f}%)")
    print("drift over 250ms taker delay (cents): "
          f"min={drifts[0]:.2f} p10={pct(.10):.2f} p25={pct(.25):.2f} "
          f"p50={pct(.50):.2f} p75={pct(.75):.2f} p90={pct(.90):.2f} max={drifts[-1]:.2f}")
    print()

    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    print(f"{'chase_c¢':>8} {'fill_rate':>9} {'n_fills':>7} "
          f"{'tot_pnl':>9} {'pnl/fill':>9} {'pnl/signal':>10} {'avg_entry_drift¢':>16}")
    for c in grid:
        hit = [t for t in sigs if drift[t] <= c + 1e-9]
        nf = len(hit)
        # pnl is known for these (run ceiling 10c >= c, so all `hit` filled live)
        ps = [pnl[t] for t in hit if t in pnl]
        tot = sum(ps)
        adrift = sum(drift[t] for t in hit) / nf if nf else 0.0
        print(f"{c:>8.1f} {nf/n:>9.2%} {nf:>7d} "
              f"{tot:>9.3f} {tot/nf if nf else 0:>9.4f} {tot/n:>10.4f} {adrift:>16.2f}")

    print()
    print("note: sim has no real predictive edge, so absolute P&L is "
          "spread+fee+microstructure (mostly negative). The decision signals are "
          "(a) the fill-rate curve and (b) how pnl/fill degrades as chase rises "
          "(adverse selection on marginal fills). Combine with the backtest's "
          "edge estimate to choose chase_c.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/chase_study.log")
