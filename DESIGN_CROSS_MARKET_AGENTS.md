# Cross-market correlation & arbitrage agent system

Goal: find money in RELATIONS between prediction markets, instead of beating a
single market's own quote (which this program has repeatedly shown is hard).
Three layers, ordered by how cheap the evidence is:

## Layer 1 — Hard cross-venue arbitrage (deterministic, no news)
The same real-world event is often listed on both Kalshi and Polymarket
(elections, Fed decisions, CPI, sports, "BTC above X on date D"). If the YES
price on one venue plus the NO price on the other sums to less than $1 after
fees, that is a locked profit regardless of outcome.
- `tools/xmarket/fetch_catalogs.py` — pull the full public catalogs: Kalshi
  (/trade-api/v2/markets, paginated) and Polymarket (gamma-api, public).
- `tools/xmarket/match_pairs.py` — match same-event pairs: first a
  deterministic pass (close dates + shared numbers/strikes + keyword overlap),
  then an optional LLM confirmation pass for fuzzy titles.
- `tools/xmarket/spread_scan.py` — for each matched pair, compute the
  executable two-leg arb after fees (Kalshi taker 0.07·p·(1−p); Polymarket
  ~0 fee but wider spreads; use top-of-book, not mid) and report a ranked
  list. Re-run on a schedule; log history so we learn which spreads are real
  vs stale-book illusions.
Known reality check: this space is picked over by dedicated bots; the edge,
if any, lives in less-liquid pairs and fresh listings. The scanner measures
rather than assumes.

## Layer 2 — Correlation discovery (statistical, no news)
- `tools/xmarket/snapshot_prices.py` — periodic (e.g. 5-minute) snapshots of
  every active market's top-of-book on both venues into a local parquet/csv
  store (the box or this machine; small data).
- `tools/xmarket/corr_track.py` — rolling correlation matrix over price
  CHANGES (never levels — levels of trending markets fake correlation),
  cross-venue lead-lag at the snapshot cadence, and a report of the top
  stable pairs that are NOT the same event (those are layer 1's job).
Output: candidate related pairs with history, feeding layer 3's hypotheses
and a future stat-arb study (trade the laggard when the leader moves).

## Layer 3 — News agents (causal hypotheses + event-time reaction)
Agents run through the local `claude` CLI in headless mode (claude -p), so
they use the existing subscription; no separate API key. All agent outputs
are logged JSON; nothing trades.
- `tools/xmarket/news_fetch.py` — poll RSS/public feeds (Reuters, AP,
  CoinDesk, Treasury/Fed calendars) into a deduplicated headline store.
- `tools/xmarket/news_agent.py` — for each fresh headline batch, ask the LLM:
  which active markets (from the catalogs) does this plausibly move, which
  direction, how strongly, and which OTHER markets should co-move. Store as
  (headline, market, direction, confidence) rows.
- `tools/xmarket/news_eval.py` — the honesty layer: after the fact, score the
  agent's calls against actual market moves (did the named market move in the
  named direction more than base rate?). No deployment until this shows real
  precision on a few hundred calls.
Trading concept to evaluate later, only if layers 1-3 produce evidence:
when news moves market A and the agent + correlation history say B should
follow, buy B's lagging side — an event-time lead-lag trade with a thesis.

## Discipline
- Everything logs; nothing trades without the usual fresh-evidence bar and
  the user's explicit arm instruction.
- Each layer's output is a measurable artifact (spread history, correlation
  stability, agent precision) — kill criteria stated before building more.
