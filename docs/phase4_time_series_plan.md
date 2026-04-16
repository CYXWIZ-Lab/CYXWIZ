# Phase 4 — Time-Series Training Pipeline Plan

**Status:** Phase 4 v1 **COMPLETE** (2026-04-16, same-day execution after
the architecture pass). Sessions A, B, C, D all shipped in a single day.
See "What actually shipped" below for the landed surface and "Deferred
to Phase 4.x" for the still-open items.

**Precedent:** Phase 3 Text (shipped 2026-04-14), architecture discussion 2026-04-16.

## What actually shipped (2026-04-16)

Seven commits on `Nodes_Implementation` between `52808745` and `272dca2e`:

**Foundation (Session A):**
- `52808745` node_executors framework + IPipelineOperator interface
  (Cat-1 base class, four-band enum, PipelineOperatorFactory)
- `8b6055b0` PipelineMaterializer wired into training launch — walks
  the graph forward from DataInput, applies Cat-1 ops in BFS order,
  registers the transformed table as `<source>__materialized`

**Phase 4 v1 core (Session B + C):**
- `aacbadcd` TimeSeriesWindow + TimeSeriesSplit operators,
  ArrowDatasetBatcher partition + regression modes, FileCategory::
  TimeSeries, airline_passengers dataset, v1 smoke graph
- `5da46ede` Two runtime fixes surfaced during smoke verify
  (TrainingManager input_size override + loss gradient reshape)

**Phase 4 extensions (Session D):**
- `2f9bea40` LogTransform + Differencing operators, ts_column_utils.h
  shared helpers, v2 smoke graph
- `82498bb2` ReplaceColumnWithFloat field-type fix (int→float column
  retyping on SetColumn)
- `272dca2e` TimeSeriesFeatures operator (lag + rolling mean) +
  TimeSeriesWindow multivariate extension (feature_cols param), v3
  smoke graph with full chain

**Operators delivered (all Cat-1, real Arrow-table transforms):**

| Operator | Band | Params | Row effect |
|---|---|---|---|
| `LogTransform` | 1 | value_col | preserves |
| `Differencing` | 1 | value_col, lag, order | drops lag × order |
| `TimeSeriesFeatures` | 1 | value_col, lag_values, rolling_windows | drops max(max_lag, max_window - 1) |
| `TimeSeriesWindow` | 1 | value_col, feature_cols, input_width, label_width=1, shift | produces N - span + 1 windows |
| `TimeSeriesSplit` | 2 | train_ratio, val_ratio, test_ratio | adds `__partition__` int8 |

**Smoke graphs (all in `examples/cyxgraph/timeseries/`):**

| Graph | Pipeline | Final val_loss / 50 ep |
|---|---|---|
| `airline_passengers_dense.cyxgraph` | Window → Split → Dense | 5,598 (raw scale) |
| `airline_passengers_dense_v2.cyxgraph` | Log → Diff(12) → Window → Split → Dense | 0.0011 |
| `airline_passengers_dense_v3.cyxgraph` | Log → Diff(12) → Features(lag_1 + roll_7) → MVar Window → Split → Dense | 0.0007 |

**Overshoots vs original plan:**

- Multivariate forecasting was deferred to Phase 4.x in the plan but
  shipped in Session D via `TimeSeriesWindow.feature_cols`. Phase 4 v1
  is no longer univariate-only. "Multiple value columns at a time"
  in the strict sense (multiple targets) is still deferred.
- The single smoke test in the plan became three (v1/v2/v3) showing
  convergence improving with richer preprocessing.



## Goal

Minimum viable time-series forecasting training pipeline — user loads a CSV
with a time column, windows it, chronologically splits, trains a model to
forecast next step(s). **Every new node is a real pipeline operation from
day one.** No config-extractor shortcuts like Phase 3 took (see tofix.md
"TextTokenizer is a config extractor, not a pipeline operation").

## Architectural context

This plan was written AFTER the 2026-04-16 architecture pass that defined
the four pipeline bands and the four tool categories (see CLAUDE.md
"Pipeline Architecture: Four Bands" and "Node vs Panel"). Phase 4 is the
first feature work designed in that framework; its nodes MUST respect
the band contracts and the "nodes are real operations" rule.

## Existing surface (as of 2026-04-15, before Phase 4 started)

**Dead NodeTypes (already in the enum, need executors + dialogs):**
- ~~`TimeSeriesWindow`~~ **LIVE** — Cat-1 Band 1 operator (`aacbadcd`,
  multivariate in `272dca2e`)
- ~~`TimeSeriesFeatures`~~ **LIVE** — Cat-1 Band 1 operator (`272dca2e`)
- ~~`TimeSeriesSplit`~~ **LIVE** — Cat-1 Band 2 operator (`aacbadcd`)
- `TimeSeriesCSV` — data source (legacy, still dead — replaced by
  `FileCategory::TimeSeries` on `DataInput`)
- `TimeSeriesDecomposition` — trend/seasonal/residual (Phase 5)
- `ACFNode` / `PACFNode` — autocorrelation (Phase 5)
- `StationarityTest` / `SeasonalityDetector` — tests (Phase 5)
- `ARIMAForecaster` / `ExponentialSmoothing` — classical (Phase 5)

**New NodeTypes added during Phase 4:**
- `LogTransform` — Cat-1 Band 1 operator (`2f9bea40`)
- `Differencing` — Cat-1 Band 1 operator (`2f9bea40`)

**Live panels already implementing the underlying logic:**
- `acf_pacf_panel.h`, `decomposition_panel.h`, `forecasting_panel.h`,
  `seasonality_panel.h`, `stationarity_panel.h`

**Backend layers usable for time-series models:**
- `Conv1DLayer` exists (`cyxwiz-backend/include/cyxwiz/layer.h:592`) —
  causal/dilated support uncertain, may need verification.
- `LSTMLayer` / `GRULayer` exist but **LSTM is broken** (no CPU Backward,
  AF path has a column-major bug). Treat as unusable for Phase 4 v1.
- `DenseLayer`, `FlattenLayer` exist and work correctly.

## Four-band mapping for time-series

| Band | Time-series nodes | Cacheable? |
|---|---|---|
| **1. Data prep (stateless)** | `DataInput(TimeSeries)`, `LogTransform`, `Differencing`, `TimeSeriesWindow`, `TimeSeriesFeatures` (lag/rolling) | Yes |
| **2. Partitioning** | `TimeSeriesSplit` (chronological 80/10/10, NO shuffle) | Yes |
| **3. Iteration + phase-aware** | `DataLoader` (shuffle OK after windowing — windows are iid) | No — per run |
| **4. Model + optimization** | `Flatten`, `Dense`, `Conv1D`, `MSELoss`, `Adam`, `Output` (LSTM/GRU deferred) | No |

**Hard constraint:** windowing must happen in Band 1 — BEFORE the split —
otherwise windows straddle the train/val boundary and leak temporal info.

## Minimum viable smoke test

**Dataset:** Airline Passengers (`airline_passengers.csv`, 144 monthly totals
1949-1960). Classic Box-Jenkins toy, clear trend + seasonality, tiny enough
to train in <1 minute. Public domain, ~3KB. Bundle in `examples/datasets/`.

**Graph:**
```
DataInput(TimeSeries, time_col=month, value_col=passengers)
  → LogTransform              (stabilize variance)
  → Differencing(lag=12)      (remove yearly seasonality)
  → TimeSeriesWindow(
        input_width=24,
        label_width=1,
        shift=1)
  → TimeSeriesSplit(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        method=chronological)
  → DataLoader(batch_size=16, shuffle=true, epochs=50)
  → Flatten
  → Dense(32) → ReLU → Dropout(0.2)
  → Dense(1)
  → MSELoss
  → Adam(lr=1e-3)
  → Output
```

**Design rationale for the graph:**
- **Uses Flatten + Dense (feedforward)** on purpose to avoid the broken
  LSTM. Phase 4 v1 should not be blocked on the LSTM CPU-backward fix.
- **Shuffle=true in DataLoader is OK here** because windowing already made
  each window an iid training example. Shuffling windows ≠ breaking
  temporal order; the model still sees past → future within each window.
- **TimeSeriesSplit is chronological**, providing the real guarantee
  against temporal leakage. DataLoader shuffle is fine after that.
- **Differencing(lag=12)** handles the yearly seasonality; log handles the
  exponential growth. What's left is (hopefully) stationary enough for the
  feedforward model to learn.

**Success criterion:** val MSE drops over 50 epochs. Not chasing SOTA
accuracy — verifying the pipeline works end-to-end, same bar as v1 text
("64.65% = fix confirmed").

## New nodes to build

Each as a **real operation** with Arrow table input → Arrow table output,
registered via the `node_executors/` framework:

| Node | Band | Input schema | Output schema |
|---|---|---|---|
| `DataInput(TimeSeries)` | 1 | (file) | `{time: timestamp, values: float[]}` long format |
| `LogTransform` | 1 | `{values: float[]}` | `{values: float[]}` (log1p) |
| `Differencing(lag=N)` | 1 | `{values: float[]}` | `{values: float[]}` (first N rows dropped) |
| `TimeSeriesWindow(in, out, shift)` | 1 | `{values: float[]}` | `{window_start_time: timestamp, input: list<float>[in], target: list<float>[out]}` one row per window |
| `TimeSeriesSplit(ratios)` | 2 | windowed rows | same schema + partition marker column, chronologically split |
| `DataLoader` (existing) | 3 | partitioned windows | batches of `(input, target)` tensors |

**New NodeTypes outside the existing enum that may be needed:**
- `LogTransform` — could extend existing `Normalize` with a method param,
  or be a new node. Design call open (not blocking).
- `Differencing` — new node, or rolled into `TimeSeriesFeatures`.

## Open design questions (answered with "my picks" — user to confirm)

**Q1: `DataInput(TimeSeries)` vs `TimeSeriesCSV` — one node or two?**
- My pick: **extend `DataInput` with `FileCategory::TimeSeries`.**
  Consistency with the Text/Image/Audio pattern, fewer NodeTypes, users
  already know `DataInput`.

**Q2: `TimeSeriesWindow` — one node or two?**
- My pick: **one node.** `window(input_width, label_width, shift)` produces
  `(input_slice, target_slice)` pairs in one step. Matches
  `tf.keras.utils.timeseries_dataset_from_array`. Can add a two-node
  version later if users want finer control.

**Q3: What happens to the time column after windowing?**
- My pick: **keep as a metadata column** `window_start_time` on each
  windowed row. Ignored by model layers, used by the Output node for
  forecast-vs-time plotting at inference.

**Q4: Dataset for the smoke test?**
- My pick: **bundle `airline_passengers.csv` in `examples/datasets/`.**
  Tiny, public domain, self-contained smoke test with no network
  dependency.

**Q5: Chronological split — flag on `DataSplit` or separate `TimeSeriesSplit`?**
- My pick: **keep them separate.** `DataSplit` keeps its random-shuffle
  default; `TimeSeriesSplit` is a distinct NodeType (already in the enum)
  with chronological semantics. A separate node name prevents the
  "why isn't my DataSplit shuffling" FAQ.

**Q6: Finish `node_executors/` framework first, or build Phase 4 inline?**
- My pick: **finish the framework first** as a foundation commit. Land
  KMeans as the first worked concrete executor (scaffolded already).
  Then Phase 4 nodes plug into a clean executor API instead of
  reinventing the pattern per-node.

## Deferred to Phase 4.x (still open after Phase 4 v1 shipped)

- **LSTM / GRU for time-series** — blocked on LSTM CPU-backward tofix fix.
  Without LSTM working the `test_02_sentiment_lstm.cyxgraph` pattern
  for time-series is also blocked.
- **Transformer for time-series** (Informer, Autoformer) — no Transformer
  Module wrapper exists yet, parallel to the LSTM situation.
- **Multi-step forecasting** (`label_width > 1`) — currently enforced
  to 1 in `TimeSeriesWindowOperator::Configure`. Needs a multi-output
  regression path in `ArrowDatasetBatcher` (label tensor `[batch, H]`
  for horizon H) and a corresponding `MSELoss` target shape that
  matches.
- **Multi-target multivariate** — multiple columns as TARGETS simultaneously.
  The `feature_cols` param only enriches the INPUT side; `y` still
  comes from a single `value_col`. Needs a `target_cols` param and
  matching batcher/loss changes.
- **Exogenous variables / known-future covariates** (holiday flags,
  promo flags) — different from lag features: these carry values for
  the forecast horizon itself, not just the input window.
- **Probabilistic / quantile forecasting** — quantile loss,
  DeepAR-style outputs. Needs new loss functions.
- **Walk-forward cross-validation** — proper time-series CV is harder
  than a single chronological split. `TimeSeriesSplit` is just one
  chronological split right now.
- **ARIMA / Exponential Smoothing** — classical methods, would need
  Python / statsmodels bridge. Blocked on a broader Python bridge
  design, not Phase 4 scope.
- **ACF / PACF / Stationarity / Decomposition as pipeline nodes** —
  keep as panels for now (Category 2 introspection). Users run them
  during data exploration, not during training. Panel form already
  works; node form is valuable but not blocking.
- **Rolling aggregations beyond mean** (std / min / max / median) —
  `TimeSeriesFeatures` v1 ships mean only. Adding std/min/max is a
  few lines but blocked behind "is this the right shape".
- **TimeSeriesFeatures: differencing features** — the original
  NodeType description mentioned "lag / rolling / **differencing**
  features". Since we shipped `Differencing` as its own node, the
  differencing-inside-Features capability is redundant and was
  dropped.
- **Forecast plotting** — `window_start_time` metadata column on
  windowed rows (Q3 answer) is not yet implemented. Needed when
  `Output` node plots forecasts vs time.

## Execution order — as originally planned vs what happened

| Session | Plan estimate | Actually shipped |
|---|---|---|
| A (foundation) | ~1 session | 2 commits (`52808745`, `8b6055b0`) |
| B (v1 core) | ~1-2 sessions | 1 commit (`aacbadcd`) |
| C (smoke verify) | ~0.5-1 session | 1 fix commit (`5da46ede`) |
| D+ (extensions) | ~1 session each | 3 commits (`2f9bea40`, `82498bb2`, `272dca2e`) |

All four sessions shipped same-day (2026-04-16) with seven commits
totaling ~2300 insertions. No blockers hit.

**Deferred to Phase 4.x** — see "Deferred to Phase 4.x" section above
for the full list. Main categories: sequence models (LSTM/Transformer,
blocked on LSTM CPU backward fix), multi-step horizons, classical
stats (ARIMA), exogenous covariates, probabilistic forecasting, and
introspection nodes (ACF/PACF/Stationarity as graph nodes — they're
panels today).

## Dependencies and coupling

- **`node_executors/` framework** — the Phase 4 foundation. Currently
  scaffolded (untracked: `node_executor.h`, `node_executor_factory.h`,
  `kmeans_executor.{cpp,h}`) but not committed. Session A lands this.
- **Fix B (TextTokenizer as real operation)** — not a blocker for Phase 4,
  but Phase 4 establishes the pattern that Fix B will follow. If both
  are landed close together, the architecture is consistent.
- **LSTM CPU Backward fix** — blocks Phase 4.x (sequence models), NOT
  blocking for Phase 4 v1 (feedforward smoke test).

## Regression checks required

- v1 `mental_health_sentiment_classifier.cyxgraph` must still train to
  64.65% val acc after Phase 4 lands (should be untouched but verify).
- v2 `mental_health_sentiment_classifier_v2.cyxgraph` must still train to
  ~60.72% val acc.
- LSTM smoke test `test_02_sentiment_lstm.cyxgraph` must still compile
  (it'll still fail to learn due to the known LSTM bug, but it should
  not crash or regress).

## Entry point for the next session

Phase 4 v1 is complete. The obvious next steps:

1. **LSTM CPU Backward fix** (tofix.md "LSTM Layer — Broken AF Forward +
   Missing CPU Backward") — unblocks sequence models for time-series,
   text, and any other domain. This is a `cyxwiz-backend/src/algorithms/
   layer.cpp` fix, not an engine change.
2. **Fix B: TextTokenizer as real operator** (tofix.md) — uses the
   IPipelineOperator framework that Phase 4 validated. Would close
   the Phase 3 config-extractor shortcut.
3. **Tool-to-Node Migration** (tofix.md) — ~40 dead NodeTypes still
   waiting on the operator framework. Phase 4 removed 3 of them
   (TimeSeriesWindow / Features / Split); 37ish remain. Recommended
   order: text analytics → linear algebra → clustering → signal
   processing → statistics → ML algorithms → introspection nodes.
4. Phase 4.x items above if sequence models become the priority.

The IPipelineOperator framework, PipelineMaterializer dispatch,
ArrowDatasetBatcher regression + partition modes, and the
shared `ts_column_utils.h` all graduate to "general purpose"
infrastructure after Phase 4. Future Cat-1 operators should follow
the same pattern: factory-registered, schema-aware Apply, Configure
reads string params, spdlog summary, v1/v2 smoke graphs demonstrating
convergence.
