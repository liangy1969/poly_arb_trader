"""One scheduled cycle of the cross-market system. Intended to run every 15 minutes
(Windows scheduled task 'xmarket-cycle').

Every cycle: poll news feeds, snapshot tracked prices, run the consistency monitor.
Every 8th cycle (~2 hours): run the news agent on fresh headlines (uses the claude CLI,
so it is deliberately throttled). A cycle counter lives in data/xmarket/cycle_count.txt;
all output appends to data/xmarket/cycle.log.
"""
import os
import subprocess
import sys
import time

ROOT = r"E:\poly\crypto_trader"
PY = sys.executable
OUT = os.path.join(ROOT, "data", "xmarket")


def run(script, *args):
    r = subprocess.run([PY, os.path.join(ROOT, "tools", "xmarket", script), *args],
                       capture_output=True, text=True, timeout=1200, cwd=ROOT, encoding="utf-8")
    return (r.stdout or "") + (r.stderr or "")


def main():
    os.makedirs(OUT, exist_ok=True)
    cpath = os.path.join(OUT, "cycle_count.txt")
    n = 0
    if os.path.exists(cpath):
        try:
            n = int(open(cpath).read().strip())
        except Exception:
            n = 0
    n += 1
    open(cpath, "w").write(str(n))
    log = open(os.path.join(OUT, "cycle.log"), "a", encoding="utf-8")
    log.write("\n=== cycle %d @ %s ===\n" % (n, time.strftime("%Y-%m-%d %H:%M:%S")))
    for script, args in (("news_fetch.py", ()), ("snapshot_prices.py", ()), ("consistency_monitor.py", ())):
        try:
            out = run(script, *args)
        except Exception as e:
            out = "FAILED: %s" % e
        log.write("-- %s --\n%s\n" % (script, out.strip()[-2000:]))
    if n % 8 == 0:
        try:
            out = run("news_agent.py", "--max-headlines", "25")
        except Exception as e:
            out = "FAILED: %s" % e
        log.write("-- news_agent.py --\n%s\n" % out.strip()[-2000:])
    if n % 96 == 0:   # ~daily: correlation agent data pass + thesis validation
        for script, args in (("corr_agent.py", ("analyze",)), ("corr_agent.py", ("validate",))):
            try:
                out = run(script, *args)
            except Exception as e:
                out = "FAILED: %s" % e
            log.write("-- %s %s --\n%s\n" % (script, args[0], out.strip()[-2000:]))
    if n % 32 == 0:   # ~every 8 hours: score the news agent
        try:
            out = run("news_eval.py", "--horizon", "60")
        except Exception as e:
            out = "FAILED: %s" % e
        log.write("-- news_eval.py --\n%s\n" % out.strip()[-2000:])
    log.close()
    print("cycle %d done" % n)


if __name__ == "__main__":
    main()
