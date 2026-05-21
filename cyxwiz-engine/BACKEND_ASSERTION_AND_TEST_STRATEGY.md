# Backend Assertion and Test Strategy

## Why This Exists

The current CyxWiz engine has useful test coverage in a few places, but many
algorithm paths still rely on manual inspection and debugger output.

The Studio Debugger should not replace tests. It should help us define the
right assertions, capture trace evidence, and reproduce failures.

## What We Need To Protect

### 1. Data and preprocessing invariants

Assert these early:
- input file exists and is readable
- required columns exist
- label count is valid
- train / val / test ratios sum correctly
- vocab file exists before `TextVocabulary` uses it
- token ids stay inside the vocab range
- padding length matches the configured max length

### 2. Shape and wiring invariants

Assert these during compile and debug:
- upstream and downstream tensor shapes are compatible
- text graphs use the configured sequence length
- embedding input ids are integer tensors
- recurrent layers return the expected output shape
- loss input shape matches the selected loss function

### 3. Training invariants

Assert these during execution:
- loss is finite
- gradient norms are finite
- at least one gradient is non-zero when learning should happen
- batch size is positive and consistent
- no silent shape truncation or unexpected broadcast

### 4. Export and replay invariants

Assert these when saving or replaying:
- graph serialization preserves node parameters
- trace replay matches the recorded node order
- saved vocab and metadata stay aligned

## Where To Add Tests First

### Highest priority

1. `tests/test_debug_executor.cpp`
2. `tests/test_hdf5_loading.cpp`
3. text preprocessing paths in `src/core/text_dataset_batcher.cpp`
4. shape inference in `src/core/graph_compiler.cpp`

### Second priority

1. `src/core/data_registry.cpp`
2. `src/core/pipeline_executor.cpp`
3. `src/core/model_builder.cpp`
4. `src/core/model_exporter.cpp`

## How The Studio Debugger Helps

The debugger should produce a trace that we can treat as a regression record.

For each debug run, capture:
- node id
- node type
- input shape
- output shape
- warnings
- errors
- runtime status

Then use that trace to define expected behavior for a known-good graph.

Example assertions from a trace:
- `DataInput -> TextTokenizer -> TextVocabulary -> TextPadding` happens in that order
- `TextVocabulary` resolves the expected vocab file
- `Embedding` receives integer ids with the expected padded length
- `GRU` output shape matches the configured hidden size
- no trace row reports a shape mismatch

## Test Layers

### Unit tests

Use for small, deterministic logic:
- shape inference
- vocab loading
- split validation
- loss / gradient sanity checks

### Regression tests

Use for known-bug fixes:
- text shape mismatch regressions
- GRU / LSTM output shape regressions
- vocab path regressions
- debug executor trace matching regressions

### Integration tests

Use for end-to-end flows:
- text sentiment training
- MNIST training
- HDF5 dataset loading
- debugger session completion

## First Practical Targets

Start with the paths that already caused issues:
- text vocab loading
- text shape inference
- runtime trace shape matching
- recurrent layer output prediction

Those are the places where the debugger already exposed real bugs.

## Exit Criteria

This backend work is not done when the code compiles.
It is done when:
- important invariants have assertions or tests
- debugger traces can be replayed as regressions
- known shape and vocab failures have fixed test coverage
- future breaks fail loudly instead of drifting silently
