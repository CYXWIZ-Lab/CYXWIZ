# Pin-Connectivity Compile-Gate Test Fixtures

These cyxgraphs exist to verify the 2026-04-17 pin-connectivity checks
in `GraphCompiler::Compile` (commit `8c468316`). Each fixture is a
deliberately-broken graph that the compile gate must reject with a
specific error.

Both fixtures are derived from `examples/cyxgraph/mnist_mlp.cyxgraph`
— the only difference is one bad wire (or one missing wire) on the
loss node.

## Fixtures

### `01_targets_disconnected.cyxgraph`

The `DataInput.Labels → CrossEntropy.Targets` link is removed.
`Loss.Targets` is required but has no incoming connection.

**Expected compile error:**
```
Required input pin 'Targets' on node 'CrossEntropy' has no incoming
connection
```

This catches `ValidateRequiredInputsConnected`.

### `02_targets_wrong_source.cyxgraph`

`Loss.Targets` IS wired — but to `fc3.Output` (the model's prediction
tensor) instead of `DataInput.Labels`. The required-input check passes
because the pin has an incoming link; the reachability check fails
because that link's upstream chain never touches a `PinType::Labels`
output pin.

**Expected compile error:**
```
Loss node 'CrossEntropy' has its Targets pin wired, but the upstream
chain never passes through a Labels-typed pin (DataInput.Labels,
DataSplit.*Labels, or DataLoader.Labels). The model is being trained
against the wrong stream.
```

This catches `ValidateLossTargetsReachLabels`.

### `03_predictions_wrong_source.cyxgraph`

`Loss.Predictions` IS wired — but to `DataInput.Labels` (the label
stream) instead of the model's output. The required-input check
passes; the reachability check fails because the upstream chain
never touches a model layer or `Output` node.

**Expected compile error:**
```
Loss node 'CrossEntropy' has its Predictions pin wired, but the
upstream chain never passes through a model layer or Output node.
The loss is being computed against a non-prediction tensor (often
the Labels stream wired by mistake, or a raw DataInput tensor with
no model in between).
```

This catches `ValidateLossPredictionsReachModel`. Note this fixture
ALSO has the same Labels-→-Targets wire as a normal graph, so the
Targets check passes — only Predictions is wrong.

## Why these matter

Before the pin-connectivity compile-gate landed, both fixtures would
compile cleanly and start training. The runtime path
(`TrainingExecutor`) reads the dataset and labels from `DataRegistry`
by name, ignoring the graph topology — so a graph with broken wires
trained against whatever the registry held, producing meaningless
gradients and metrics with no error message.

The compile gate now refuses to launch training for these graphs.
The architectural fix that makes `TrainingExecutor` actually walk the
pins is still pending (`tofix.md` → "TrainingExecutor should walk pin
connections, not registry lookups"), but the canvas is now the source
of truth at compile time.

## How to use

1. Open the engine.
2. **File → Open** one of the `.cyxgraph` files in this folder.
3. Hit **Compile** (F7).
4. Confirm the popup shows the expected error message above.
5. Confirm **Train** (F5) refuses to launch (the pre-flight gate in
   `MainWindow::StartTrainingFromGraph` re-runs `BuildCompileResult`
   and rejects on Error-level issues).

If either fixture *succeeds* compile, the corresponding check has
regressed.
