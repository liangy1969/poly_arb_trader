"""Maker-exit shadow analysis.

Reads a live-run log with `MAKER` probe lines — one per exit, comparing a
Post-Only maker exit (post at the ask, rest, fill if the bid lifts it, else cross)
against the taker cross we currently do. Reports maker fill rate, the outcome mix
(reject / maker_fill / timeout), and the per-share gain over taker.

    py scripts/analyze_maker.py /tmp/study.log
"""
import collections
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
MAKER = re.compile(
    r"MAKER trade=(\S+) inst=\S+ outcome=(\w+) offer=[\d.]+ bid0=[\d.]+ cross=[\d.]+ "
    r"maker_net=[-\d.]+ taker_net=[-\d.]+ gain_c=([-\d.]+)"
)


def main(path):
    rows = []  # (outcome, gain_c)
    for raw in open(path, encoding="utf-8", errors="replace"):
        m = MAKER.search(ANSI.sub("", raw))
        if m:
            rows.append((m.group(2), float(m.group(3))))
    if not rows:
        print("no MAKER probe lines found")
        return

    n = len(rows)
    by = collections.defaultdict(list)
    for o, g in rows:
        by[o].append(g)
    gains = [g for _, g in rows]
    n_fill = len(by.get("maker_fill", []))

    print(f"exits sampled: {n}")
    print(f"maker fill rate: {n_fill}/{n} = {100*n_fill/n:.0f}%")
    print()
    print(f"{'outcome':>12} {'n':>5} {'share':>6} {'mean_gain_c':>12}")
    for o in ("maker_fill", "reject", "timeout"):
        g = by.get(o, [])
        if g:
            print(f"{o:>12} {len(g):>5} {100*len(g)/n:>5.0f}% {statistics.mean(g):>12.2f}")
    print()
    print(f"NET maker-vs-taker gain: mean={statistics.mean(gains):+.2f}c/share  "
          f"median={statistics.median(gains):+.2f}c/share  total={sum(gains):+.1f}c-shares")
    print("\nnote: maker_fill is fee-free at the ask (the win); reject is a wash "
          "(both cross at the same bid); timeout rests the full window then falls "
          "to a taker cross at the then-bid. Fills assume the full size lifts when "
          "the bid touches the offer (optimistic). Confirm vs backtest + real fills.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/study.log")
