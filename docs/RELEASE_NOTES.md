# CyxWiz Studio release notes

Notable user-facing changes, newest first. Engineering detail lives in the
private Data Studio tickets (`tofix*/track*`).

## Unreleased

### Changed

- **Training Dashboard points are the mean since the previous point**, not
  the epoch's running mean, so the curves no longer jump at each epoch start;
  the training cards' "best" names a whole epoch, never one batch.
- **Data Studio ML readiness follows the flow back from each Data Split
  input**: through a Quality Analyzer to its Data Input, or to a step's run
  result (SQL, Text Clean, ...), and says which one has not been loaded or run.
- **Canvas groups follow the zoom**, and a group's name no longer sits on its
  first node's title.
- **Plugin SDK: `cyxwiz::image::RandAugmentOp` is now `AugmentOp`** in
  `cyxwiz/image_augmentation.h`, with a new `Invert` value: AutoAugment,
  TrivialAugment and AugMix pick from the same list as RandAugment. A plugin
  that names `RandAugmentOp` needs the new name; `kRandAugmentOps` (14)
  still counts RandAugment's own ops, `kAugmentOps` (15) the whole list.
- **Image Rotate fills the corners outside the image with 0 through
  torchvision's fill mask** (RandomRotation's default), which changes
  bilinear borders slightly.
- **Recurrent layers (RNN, LSTM, GRU) now reset their state at the start of
  every forward call.** Each batch starts from a zero hidden (and cell)
  state, matching PyTorch and TensorFlow defaults and making CPU and GPU
  training numerically identical. Previously the CPU path carried the
  final state of the previous batch into the next one; training numerics
  may shift slightly if a workflow relied on that. Streaming inference or
  deliberate truncated BPTT can still seed a call explicitly through the
  layer's hidden/cell state setters, which apply to the next forward call
  only.
- Bidirectional LSTM (and RNN) training no longer fail closed: both run as
  split forward/reverse branches, like GRU, on the CPU reference or the
  native neural provider.

### Added

- **Example: cats vs dogs ResNet from scratch**
  (`examples/cyxgraph/cats_dogs_resnet_scratch.cyxgraph`, guide in
  `docs/usage/image_classifier_resnet.md`): 92.6 % test accuracy on one
  GTX 1050 Ti.
- **Python: `BCEWithLogitsLoss(reduction, pos_weight, label_smoothing)`** and
  a `label_smoothing` property.
- **BCE with Logits has `label_smoothing`**: training targets become
  target x (1 - e) + e / 2, the two-class form of CrossEntropy's label
  smoothing; checked against PyTorch. Plugin SDK: `BCEWithLogitsLoss` takes it
  as a third constructor argument (`GetLabelSmoothing` / `SetLabelSmoothing`).
- `KNOWN_ISSUES.md` ships in every release package, next to `LICENSE`. It lists known problems
  in the GPU runtimes and what to do about them. The first entry is an ArrayFire oneAPI bug in
  indexed min / max along rows (arrayfire/arrayfire#3732), which CyxWiz already works around.
- Environment switch `CYXWIZ_DISABLE_NEURAL_PROVIDERS=1` forces every
  recurrent layer onto the portable path for the process, as an escape
  hatch when diagnosing a GPU provider.
- Native neural providers for recurrent training on NVIDIA (CUDA) and
  OpenCL GPUs, selected per run by the device you choose. Stacked and
  bidirectional RNN, LSTM and GRU train on the GPU; the placement audit
  shows which path each layer took.
- On OpenCL devices, recurrent layers with fewer than 16 hidden units
  stay on the portable path because the OpenCL provider measured slower
  than the CPU there; the placement audit says so
  (`opencl_provider_below_retention_floor`). NVIDIA CUDA has no such floor.
- The OpenCL provider is verified on NVIDIA and on Intel integrated
  graphics (UHD 630). When a machine has several OpenCL GPUs, the run's
  selected device decides which one serves it.
