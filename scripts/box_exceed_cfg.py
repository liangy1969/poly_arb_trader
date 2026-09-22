"""Box config switch (run on trade-poly from ~/poly_arb_trader): BTC trader -> exceed strategy in DRY RUN
(signals + probes, no orders); ETH trader -> executor OFF (feature collection only). Backs both files up first.
Idempotent: re-running after the switch is a no-op."""
import datetime
import os
import shutil

TS = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
EXCEED_BLOCK = """  # ── EXCEED (2026-09-21, user): 1 s exceedance classifier X=1c, |P(up)-P(down)| >= 0.48, tails
  # (mid < .10 or > .90), one fire per episode (re-arm at 0.2), hold-to-settle. DRY RUN: executor.dry_run.
  # Offline twin: analyze_online --model x=models/exceed-1s-x1-btc.json --gate none --deltas 0.48
  # --px-tails 0.10,0.90 --rearm-eps 0.2 --tte 300:60 (zero-latency tails settle cell +3.6/+7.5/+5.9c VAL/TEST/OOS;
  # 100 ms + one-tick chase ~0). Log streams: exceed / exceedfeat in data/trader-events.log.
  exceed:
    model_path: "models/exceed-1s-x1-rs-btc.json"
    reference: "binance.usdt_perp.BTCUSDT"
    depth_instrument: "binance.usdt_perp.BTCUSDT.depth"
    cut: 0.48
    rearm_eps: 0.2
    px_mode: "tails"
    px_lo: 0.10
    px_hi: 0.90
    entry_min_tte_s: 60
    entry_max_tte_s: 300
    max_entries_per_event: 255
    stale_ms: 1500
    max_spread: 0.15
    ref_max_age_ms: 5000
    hold_ms: 0
    ttl_ms: 500
    eval_min_ms: 50
    log_every_s: 1.0
    feat_log_every_s: 10.0
"""


def edit(path, reps, inserts=()):
    s = open(path, encoding="utf-8").read()
    changed = False
    for old, new, count in reps:
        n = s.count(old)
        if n == 0 and s.count(new) >= 1:
            print("  already applied:", old.strip()[:60]); continue
        assert n == count, (path, old, n)
        s = s.replace(old, new); changed = True
    for anchor, block in inserts:
        if block.strip().splitlines()[-1] in s:
            print("  block already present"); continue
        assert s.count(anchor) == 1, (path, anchor)
        s = s.replace(anchor, anchor + block); changed = True
    if changed:
        shutil.copyfile(path, "%s.bak-preexceed-%s" % (path, TS))
        open(path, "w", encoding="utf-8").write(s)
        print("  wrote", path, "(backup .bak-preexceed-%s)" % TS)
    else:
        print("  no change", path)


print("BTC config/kalshi-trade.yaml")
edit("config/kalshi-trade.yaml",
     [('  strategy: "fair_ride"\n', '  strategy: "exceed"\n', 1),
      ("  dry_run: false\n", "  dry_run: true                # *** DRY RUN (2026-09-21, user): signals + probes only, NO orders ***\n", 1)],
     [('  ring_horizon_ms: 10000\n', EXCEED_BLOCK)])
print("ETH config/kalshi-eth.yaml")
edit("config/kalshi-eth.yaml",
     [("  enabled: true                # *** LIVE TRADING — REAL MONEY, Kalshi mainnet (2026-08-10, user) ***",
       "  enabled: false               # OFF (2026-09-21, user): feature collection only — sampler keeps running", 1)])
for p in ("config/kalshi-trade.yaml", "config/kalshi-eth.yaml"):
    s = open(p).read()
    print(p, "strategy:", [l.strip() for l in s.splitlines() if l.strip().startswith("strategy:")],
          "| executor enabled/dry_run:", [l.strip()[:40] for l in s.splitlines() if l.strip().startswith(("enabled:", "dry_run:")) and "executor" not in l][:6])
