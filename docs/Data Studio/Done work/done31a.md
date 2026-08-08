# Data Studio done31a - LSTM ArrayFire performance phase 2 benchmark

Status: completed scoped ticket extracted from `tofix31.md`.

Goal: make recurrent performance work measurement-gated before changing LSTM
runtime behavior again.

## Scope

- Add an opt-in repeated-run benchmark mode to the existing
  `test_recurrent_af_profile_smoke` target.
- Keep the default path as a cheap one-pass smoke test.
- Report forward/backward min, median, max, and mean timings.
- Preserve the existing placement-policy assertions for the small CUDA-eligible
  LSTM shape.
- Use the benchmark output to pick exactly one hotspot before any optimization.

## Non-goals

- No pinned-memory implementation.
- No TransformerDecoder seq2seq implementation.
- No pretrained transformer import/fine-tuning implementation.
- No broad recurrent rewrite before a measured hotspot is selected.

## Current implementation slice

- `CYXWIZ_RECURRENT_PROFILE_RUNS=N` enables repeated measured runs.
- `CYXWIZ_RECURRENT_PROFILE_WARMUP=N` controls warmup runs when measured runs
  are greater than one.
- `CYXWIZ_RECURRENT_PROFILE_BATCH=N`, `CYXWIZ_RECURRENT_PROFILE_SEQ=N`,
  `CYXWIZ_RECURRENT_PROFILE_INPUT=N`, `CYXWIZ_RECURRENT_PROFILE_HIDDEN=N`,
  and `CYXWIZ_RECURRENT_PROFILE_LAYERS=N` select the measured LSTM shape.
- `CYXWIZ_RECURRENT_PROFILE_MATRIX=1` runs three representative shapes in one
  process: small, medium, and longer-sequence.
- With no environment variables set, the test remains a single measured smoke.

## Current local measurement

Default one-pass smoke on ArrayFire CUDA:

- forward: about 84 ms
- backward: about 159 ms

Repeated benchmark with `CYXWIZ_RECURRENT_PROFILE_RUNS=10` and
`CYXWIZ_RECURRENT_PROFILE_WARMUP=2` on the same default shape:

- forward median: about 5.0 ms
- backward median: about 6.8 ms

Interpretation: the small-shape steady-state path is not showing the earlier
large backward hotspot; the first pass is dominated by one-time backend/JIT/setup
cost. Next measurements should use shape overrides before selecting a runtime
optimization target.

Medium shape benchmark with `CYXWIZ_RECURRENT_PROFILE_BATCH=16`,
`CYXWIZ_RECURRENT_PROFILE_SEQ=16`, `CYXWIZ_RECURRENT_PROFILE_INPUT=32`,
`CYXWIZ_RECURRENT_PROFILE_HIDDEN=16`, `CYXWIZ_RECURRENT_PROFILE_RUNS=10`, and
`CYXWIZ_RECURRENT_PROFILE_WARMUP=2`:

- forward median: about 14.3 ms
- backward median: about 20.7 ms

Interpretation: backward is the larger steady-state segment on this shape, but
the gap is modest. The next implementation step should inspect whether the
backward path is paying avoidable synchronization/materialization costs before
changing math behavior.

## Next decision point

Run the benchmark on the target machine and compare:

- forward median
- backward median
- min/max spread
- whether the active backend is CUDA, OpenCL, ArrayFire CPU, or unavailable

Only after that should the next change choose one hotspot from:

- input projection
- per-step gate math
- AF-to-Tensor cache materialization
- CPU backward

## Profiling knobs added after measurement

- `CYXWIZ_LSTM_AF_BACKWARD_EVAL_INTERVAL=N` controls how often the ArrayFire
  LSTM backward timestep loop materializes intermediate expressions.
- Default is `1`, preserving prior behavior.
- Larger values are for profiling only until benchmarks prove the JIT expression
  chain remains stable and faster on the target shape.

## Eval-barrier experiment

Medium shape with default eval interval `1`:

- forward median: about 12.3 ms
- backward median: about 20.5 ms
- backward mean: about 20.1 ms

Same shape with `CYXWIZ_LSTM_AF_BACKWARD_EVAL_INTERVAL=4`:

- forward median: about 13.4 ms
- backward median: about 19.7 ms
- backward mean: about 20.3 ms

Interpretation: reducing per-timestep ArrayFire eval barriers did not produce a
clear improvement on this shape. Keep the default interval at `1`; do not claim
this as an optimization yet.

Longer sequence shape with `CYXWIZ_RECURRENT_PROFILE_BATCH=16`,
`CYXWIZ_RECURRENT_PROFILE_SEQ=32`, `CYXWIZ_RECURRENT_PROFILE_INPUT=32`,
`CYXWIZ_RECURRENT_PROFILE_HIDDEN=16`, `CYXWIZ_RECURRENT_PROFILE_RUNS=8`, and
`CYXWIZ_RECURRENT_PROFILE_WARMUP=2`:

| Eval interval | Forward median | Backward median | Backward mean | Result |
|---|---:|---:|---:|---|
| 1 | about 16.3 ms | about 27.3 ms | about 28.3 ms | baseline |
| 2 | about 17.4 ms | about 24.9 ms | about 26.6 ms | modest improvement |
| 4 | about 17.5 ms | about 22.3 ms | about 24.8 ms | best measured candidate |
| 8 | about 18.0 ms | about 33.1 ms | about 31.8 ms | worse |

Interpretation: eval interval `4` is a promising longer-sequence profiling
candidate, but the runtime default remains `1` until broader shapes prove it is
stable and consistently faster.

## Validation

- `cmake --build build --config Debug --target test_recurrent_af_profile_smoke`
- `build\bin\Debug\test_recurrent_af_profile_smoke.exe`
- Medium shape benchmark with `CYXWIZ_RECURRENT_PROFILE_RUNS=10`,
  `CYXWIZ_RECURRENT_PROFILE_WARMUP=2`, `CYXWIZ_RECURRENT_PROFILE_BATCH=16`,
  `CYXWIZ_RECURRENT_PROFILE_SEQ=16`, `CYXWIZ_RECURRENT_PROFILE_INPUT=32`,
  `CYXWIZ_RECURRENT_PROFILE_HIDDEN=16`
- Same medium shape benchmark with
  `CYXWIZ_LSTM_AF_BACKWARD_EVAL_INTERVAL=4`
- Matrix benchmark mode with `CYXWIZ_RECURRENT_PROFILE_MATRIX=1`

## Latest matrix validation (2026-06-29)

Focused target build passed:

```powershell
cmake --build build --config Debug --target test_recurrent_af_profile_smoke
```

Default smoke passed on ArrayFire CUDA with the default backward eval interval:

- Shape: batch=8, seq=8, input=16, hidden=8, layers=1
- `CYXWIZ_LSTM_AF_BACKWARD_EVAL_INTERVAL`: 1
- Forward: 107.239 ms, single measured cold run
- Backward: 79.5746 ms, single measured cold run

Matrix mode passed with `CYXWIZ_RECURRENT_PROFILE_MATRIX=1`, `CYXWIZ_RECURRENT_PROFILE_RUNS=6`, and `CYXWIZ_RECURRENT_PROFILE_WARMUP=2`.

Default backward eval interval (`1`):

| batch | seq | input | hidden | forward median ms | backward median ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 16 | 8 | 10.3791 | 12.7722 |
| 16 | 16 | 32 | 16 | 14.3463 | 22.3252 |
| 16 | 32 | 32 | 16 | 19.8531 | 30.6645 |

Backward eval interval `4` profiling experiment:

| batch | seq | input | hidden | forward median ms | backward median ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 16 | 8 | 7.3890 | 11.3472 |
| 16 | 16 | 32 | 16 | 15.5667 | 20.7394 |
| 16 | 32 | 32 | 16 | 25.0637 | 37.2959 |

Current readout: interval `4` is not consistently better across the matrix. It can improve smaller/medium backward medians, but regresses the longer-sequence shape in this run. Keep production/default behavior at interval `1`; keep interval override as a profiling-only diagnostic until a larger benchmark proves an adaptive rule.
