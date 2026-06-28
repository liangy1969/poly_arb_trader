"""Hold-period return analysis.

Reads a live-run log with `HOLD` probe lines — one per (entry, hold offset),
carrying the **realized** exit return (book sweep for the held size + taker fee
on both legs). Aggregates by hold to show return-vs-hold, so one run reveals the
hold that maximizes net return.

    py scripts/analyze_hold.py /tmp/hold_study.log
"""
import collections
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HOLD = re.compile(
    r"HOLD trade=(\S+) inst=\S+ h=(\d+) qty=[\d.]+ filled=([\d.]+) "
    r"entry=[\d.]+ exit=([\d.]+) gross_c=([-\d.]+) net_c=([-\d.]+)"
)


def main(path):
    by_h = collections.defaultdict(lambda: {"gross": [], "net": [], "exit": []})
    trades = set()
    for raw in open(path, encoding="utf-8", errors="replace"):
        m = HOLD.search(ANSI.sub("", raw))
        if not m:
            continue
        trades.add(m.group(1))
        h = int(m.group(2))
        by_h[h]["gross"].append(float(m.group(5)))
        by_h[h]["net"].append(float(m.group(6)))
        by_h[h]["exit"].append(float(m.group(4)))

    if not by_h:
        print("no HOLD probe lines found")
        return

    print(f"entries sampled: {len(trades)}")
    print("per-share return by hold (cents). gross = exit_vwap - entry; "
          "net = gross - taker fee (both legs); slippage is in the swept exit_vwap.\n")
    print(f"{'hold_ms':>8} {'n':>5} {'gross_c':>9} {'net_c(mean)':>12} {'net_c(med)':>11} {'avg_exit':>9}")
    best = None
    for h in sorted(by_h):
        g, n, ex = by_h[h]["gross"], by_h[h]["net"], by_h[h]["exit"]
        mean_net = statistics.mean(n)
        print(f"{h:>8} {len(n):>5} {statistics.mean(g):>9.2f} {mean_net:>12.2f} "
              f"{statistics.median(n):>11.2f} {statistics.mean(ex):>9.3f}")
        if best is None or mean_net > best[1]:
            best = (h, mean_net)
    print(f"\nbest mean net return: hold={best[0]}ms ({best[1]:+.2f}c/share)")
    print("note: sim P&L (real book reaction, optimistic fills). Use the *shape* "
          "to pick the hold; confirm the absolute level against the backtest.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hold_study.log")
