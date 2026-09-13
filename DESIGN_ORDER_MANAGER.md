# DESIGN: Order manager (OMS)

Status: IMPLEMENTED 2026-09-12, **disabled by default**, not deployed. Code:
`crates/executor/src/order_manager.rs` (state machine + handle, unit-tested),
`crates/executor/src/order_manager_kalshi.rs` (Kalshi private WS + REST sync feed),
wiring in `module.rs` (`executor.order_manager.enabled`, kalshi adapter only).

## 1. Purpose

One component owns "what are OUR orders and positions right now" and exposes it to
the executor and to the future maker rule (DESIGN_MARKET_MAKING.md §2c). Today the
executor learns about its own orders only from the synchronous create-order
response, the cancel response and per-cycle REST polls in the exit reconciler. A
resting-quote strategy needs millisecond-level knowledge of fills, remaining size
and inventory, a locked side while an order's state is unsettled, and a periodic
resync that always wins.

## 2. Inputs

| source | what | latency | role |
|---|---|---|---|
| WS `user_orders` | order created / filled / canceled / executed: `status`, `initial_count_fp`, `fill_count_fp`, `remaining_count_fp`, `client_order_id`, `last_updated_ts_ms` | ms | order state |
| WS `fill` | one message per match: `trade_id`, `order_id`, `count_fp`, `yes_price_dollars`, `is_taker`, `book_side`, `ts_ms` | ms | fills, position estimate |
| WS `market_positions` | `position_fp` after each change | ms | authoritative position push |
| write path | `place` registers the intent before the REST call (WS ack may beat the HTTP response; linked by `client_order_id`); `cancel` books the engine-synchronous `reduced_by` | sync | local truth for our own actions |
| REST `GET /portfolio/orders?status=resting` (paginated) | full resting list | every `sync_ms` (1 s) | resync; a managed order missing after `unknown_grace_ms` becomes `Unknown` |
| REST `GET /portfolio/positions` | signed net position per market | every `sync_ms` | THE position; overwrites the estimate |
| REST `GET /portfolio/fills?min_ts` | fills since the last one seen − 10 s | every `fills_sync_ms` (5 s) and after every WS (re)connect | gap recovery, deduped by fill id |
| REST `GET /portfolio/orders/{id}` | one order | each cycle for every `Unknown` order | resolution |

The WS connection is signed like the collector's (RSA-PSS over `ts + GET + /trade-api/ws/v2`),
subscribes once to the three channels without a market filter, pings every 20 s,
reconnects with backoff, and triggers a full REST sync + fills gap-fetch on every
(re)connect.

## 3. State machine (`OmsState`, pure)

Per order: `order_id`, `client_id`, `ticker`, `side` (YES-book `Bid` = buy YES / sell
NO, `Ask` = sell YES / buy NO), `yes_price`, `initial`, `filled`, `remaining`, `status`,
`managed` (placed through the manager), `fills_sum`.

Status: `Pending` (sent, no id) → `Resting` → `Canceling` → `Canceled` | `Executed`;
`Unknown` from a cancel transport error, a cancel 404 (`Gone`: terminal but the fill
count is not known), a placement transport error, or vanishing from the resting list.

Rules:
- `filled` = max(reported cumulative, sum of deduped fills); it never decreases;
  observations older than the record's `updated_ms` are ignored.
- a `resting` report while a cancel is in flight keeps `Canceling` (the lock holds until
  the cancel resolves);
- `Canceled{reduced_by}` ⇒ `filled = initial − reduced_by`, `remaining = 0`;
- a side is **locked** while any managed order on it is `Pending` / `Canceling` /
  `Unknown`: `place` refuses with `SideLocked`. This is the oversell rule.
- position: `auth` = venue number (poll or push) with its timestamp; `est` = `auth` +
  signed fills seen since; every poll overwrites `est`. Fills on orders the manager did
  not place (executor IOC entries, manual) are tracked as unmanaged and count toward
  `est` but never lock a side.
- write budget: `max_actions_per_s` (default 4) across place + cancel.

## 4. Interface (`OrderManager`, cloneable handle)

Reads (synchronous, lock-free for the caller): `resting(ticker)`, `order(id)`,
`position(ticker) -> {auth, auth_age_ms, est}`, `side_locked(ticker, side)`, `health()`.
Writes: `place(intent, ticker, side) -> order_id | SideLocked | Budget | Rejected |
Unknown`, `cancel(order_id) -> CancelOutcome`, `cancel_all(ticker)`.
Events: `subscribe() -> broadcast::Receiver<OmsEvent>` with `OrderUpdate`, `Fill`,
`Position`, `Unknown`, `Resolved`, `WsUp/WsDown`, `Synced`.
Every WS order/fill and every resolution is logged at info under target `oms`
(lands in trader-events.log); a health line every `health_log_s`.

## 5. Config

```yaml
executor:
  order_manager:
    enabled: false        # flip to true to run it as a read-only observer of the live account
    ws: true
    sync_ms: 1000
    fills_sync_ms: 5000
    unknown_grace_ms: 2000
    max_actions_per_s: 4.0
    health_log_s: 30
```

## 6. Rollout

1. Enable as an observer on the live box (no rule places through it yet): its view of
   the executor's own IOC entries and reconciler closes must match the reconciler's
   authoritative polls. Requires a trader restart (BTC blind ~50 min: vsurge + spnorm
   warm-ups). Validate: fills seen on WS before the reconciler's poll, `Unknown` count
   0, `Synced` every second, position `est == auth` at every poll.
2. The maker rule (Phase 2 of DESIGN_MARKET_MAKING.md) reads `position` / `side_locked`
   and writes through `place` / `cancel`; the executor's existing entry/exit paths are
   untouched.
3. Open items to verify on the first live run: the `market_positions` subscribe form
   without a market filter (falls back to the REST position poll if the server rejects
   it); the WS envelope (`type` + `msg`) for `user_order`; fractional `count` on
   post-only orders (`place_resting` now sends 2-decimal fixed point, min 0.01).
