# CyxWiz Studio release notes

Notable user-facing changes, newest first. Engineering detail lives in the
private Data Studio tickets (`tofix*/track*`).

## Unreleased

### Changed

- **Tools > Model is one window** (guide in `docs/usage/model_tools.md`,
  example `examples/cyxgraph/model_tools`) with tabs Summary, Learning
  rate, Explain, Health and Projector; its menu items open it on a tab.
  - Summary counts parameters and multiply-adds the way PyTorch does
    (checked on five models), with the memory of a batch's activations.
  - Learning rate runs a real range test on a copy of the model and
    suggests a rate (steepest fall or lowest / 10); **Use** writes it into
    the optimizer node.
  - Explain shows Grad-CAM, saliency and Integrated Gradients for any kept
    test row, as Captum computes them, on the model's own device.
  - Health is recorded during every training: weight and gradient size by
    epoch, a weight histogram and dead ReLU units; the Training Dashboard
    warns and links to it.
  - Projector maps a layer's features of the test rows (PCA, t-SNE, UMAP).
  - The Model Summary, Architecture Diagram, Learning Rate Finder, Grad-CAM,
    NAS and DNN Inference windows are gone (the diagram is the canvas; NAS
    had no search; DNN Inference ran fixed demo networks). The Grad-CAM and
    Saliency Map nodes are no longer in the node list.
- **Results > Test has Errors, Threshold and Calibration**: the mistakes by
  pair with their images, thresholds by F1, recall or cost, a confidence
  floor for several classes, and temperature / Platt / isotonic calibration
  (checked against scikit-learn). A threshold and a calibration can be kept
  with the model: the next test applies it, checkpoints and exports carry
  it, and the local inference server returns calibrated probabilities.
- A graph that reads a CSV of pixel columns into a CNN through a leading
  Reshape (784 -> 28,28,1) trains, and compiles before its data is loaded.
- **pycyxwiz.model**: the Model window from Python on a `Sequential`:
  `summary` (the same multiply-add counter), `lr_range_test` /
  `LearningRateRange`, `saliency`, `smooth_grad`, `integrated_gradients`,
  and the calibration fits (`reliability`, `fit_temperature`, `fit_platt`,
  `fit_isotonic`); numpy optional. Example `examples/python/model_tools.py`.
- **Plugin SDK**: new headers `cyxwiz/calibration.h`,
  `cyxwiz/learning_rate_range.h` and `cyxwiz/model_cost.h`
  (`CountMultiplyAdds`, `ModuleCostInput`); the convolution, dense,
  recurrent and attention modules gain a non-virtual `CostInput`.
  Layout changes: `SequentialModel` gains a member (`SetForwardObserver`),
  and `GradCAMResult` is rewritten (the `overlay` field is gone; maps are
  the gradient at the layer's output, as pytorch-grad-cam). A plugin that
  subclasses or holds a `SequentialModel` by value, or reads
  `GradCAMResult`, must be rebuilt against these headers.

- **Tools > Optimization and Calculus is one window** (guide in
  `docs/usage/optimization_calculus.md`). Its six menu items open it on
  Optimizers, Curvature, Linear, Quadratic, Derivative or Integral. It takes
  typed formulas with exact derivatives, follows the training optimizers on
  a formula's contour map, solves linear and quadratic programs of any size
  with HiGHS (shadow prices, infeasible and unbounded in words), and gives
  adaptive integrals with an error estimate. The six separate windows are
  gone.
- The Template nodes GD Visualizer, Convexity Analyzer, LP Solver, QP
  Solver, Differentiation and Integration are no longer in the node list. They had no executor; the window does that
  work, and the Portfolio Weights node brings a program into a graph.
- **Plugin SDK (`cyxwiz/optimization.h`) is rewritten**: `LinearProgram`,
  `QuadraticProgram`, `ProgramResult`, `Optimization::SolveLinearProgram`,
  `SolveQuadraticProgram`, `PortfolioProgram` and `FollowOptimizer`.
  Removed: `GradientDescent`, `MomentumGD`, `Adam`, `RMSprop`,
  `AnalyzeConvexity`, `ComputeHessian`, `SolveLP`, `SolveLPWithBounds`,
  `SolveQP`, `SolveUnconstrainedQP`, the numerical derivative and integral
  functions, the test functions (Rosenbrock, Sphere, ...), `VectorNorm` and
  their result structs. Their replacements are in `cyxwiz/calculus.h` (`Calculus::
  Integrate`, `Differentiate`, `ErrorByStep`, `CurvatureAt`,
  `CurvatureOverRange`) and `cyxwiz/expression.h` (`Expression`).

- **Tools > Text opens Data Studio's Explore > Text** (Tokenization: Tokens,
  Word Frequency: Words, TF-IDF Analysis: Terms, Word Embeddings: Embeddings,
  Sentiment Analysis: Sentiment; Data > Tokenization: Tokens). The five
  separate text windows are gone; Language Model Generation stays a window.
- **The Sentiment Analysis node is VADER** (vaderSentiment 3.3.2, checked
  rule for rule): it adds `<text>_compound`, `<text>_positive`,
  `<text>_neutral`, `<text>_negative` and `<text>_sentiment` ("positive" from
  0.05, "negative" at -0.05 and below, else "neutral") beside the input
  columns and keeps the label and the split. The Simple and AFINN methods are
  gone: a saved graph's `method` and `label_col` settings are ignored, and a
  graph that read the old output columns needs the new names.
- **The TF-IDF and Count Vectorizer nodes split words by one rule** shared
  with Explore > Text: letters and digits of any script, lower-cased, two or
  more long, an apostrophe inside a word kept ("don't"); the English stop
  words are scikit-learn's `ENGLISH_STOP_WORDS` and their contractions. The
  feature columns can differ from before: refit a saved vectorizer state
  rather than reusing it.
- **Text Clean keeps the letters, digits and marks of every script** when it
  removes special characters (it kept ASCII only).
- **The t-SNE node's `learning_rate` defaults to 0 = auto** (scikit-learn's
  rows / 48, at least 50); a typed value still wins. Small data now
  separates in 3D as it does in scikit-learn.
- The Template nodes Word Frequency and the second Tokenizer are no longer in
  the node list (Explore > Text > Words and the Tokenizer node do that work).
- **Plugin SDK (`cyxwiz/text_processing.h`)**: `Stem` is the Porter (1980)
  algorithm (NLTK's ORIGINAL_ALGORITHM); `GetStopwords` returns
  scikit-learn's list and the contractions; `ComputeTFIDF` / `ComputeTF` use
  raw counts on `Words`. Removed: `SentimentResult` and the lexicon sentiment
  functions (use `Vader`), the sample text generators, and
  `CreateOneHotEmbeddings` / `CreateRandomEmbeddings`. Added: `Words`,
  `NGrams`, `Terms`, `Vader` and `VaderScores`.

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

- **The Portfolio Weights node** (Analytics): the least risky mix of assets
  from a table of returns, with an optional target return and long only,
  fitted on the Train rows; one row per asset with its weight, mean, sd and
  share of the risk. Quadratic > From a returns table in the window reads
  the same problem and adds this node (example:
  `examples/cyxgraph/portfolio/`).
- **Python: `pycyxwiz.calculus`** (`Expression`, `integrate`, `derivative`,
  `step_errors`, `curvature`, `curvature_over_range`, `gauss_legendre`) and
  **`pycyxwiz.optimize`** (`linprog`, `qp`, `portfolio`, `follow`), checked
  against scipy and torch (example: `examples/python/optimization_calculus.py`).

- SDK: `cyxwiz::TimeSeriesStop` (time_series.h): while one lives, the time
  series fits of the thread (exponential smoothing, ARIMA, STL, the ADF
  test) ask its stop query between passes and end with "Cancelled.".
- **Data Studio Explore > Text**: one text column, grouped by a category or
  the label, in five looks: Words, Terms (TF-IDF: what sets each group apart,
  a row's terms and the rows most like it, the text quality), Sentiment
  (VADER per row and group), Tokens (the Tokenizer node's tokenizer: lengths,
  the rows a limit cuts, the vocabulary's coverage) and Embeddings (word2vec /
  fastText skip-gram or CBOW, GloVe, from a sample by default; nearest words).
  Each look adds its node to the graph. Guide in `docs/usage/text.md`.
- **The word map**: word vectors in 3D (t-SNE, UMAP or PCA, k-means clusters
  named by their words), the word looked up and its neighbours ringed and
  joined, labels clear of each other, in a window of its own; also the **Map**
  tab of the Word Embeddings node's dialog.
- **Python: `pycyxwiz.text`** with `words`, `terms`, `english_stop_words`,
  `stem` and `vader`, the same rules as the tab and the nodes (example:
  `examples/python/text_words_and_sentiment.py`).
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

### Fixed

- The Calculator bound a leading minus tighter than `^`, so `-2^2` gave 4 and
  `exp(-x^2)` gave exp(x^2). It now reads formulas through the same parser
  as the Optimization and Calculus window: `-2^2` is -4.
- Training with NAdam no longer stops with "NAdam has invalid continuation
  state" after about a thousand steps: the product of its momentum terms
  reaches 0 in float32 there, which is a valid state (PyTorch carries on
  the same way).
- Language Model Generation: a Generate from IDs run started with Details
  open now finishes when Details is closed (the panel stayed "Generating");
  Regenerate, Generate again and Continue further are off while another run
  or a speed measurement runs, or training runs on the session's model, with
  the reason on hover; Regenerate keeps the reply when the prompt cannot be
  built, and Send keeps a chat message that could not start.
- Data Studio Signal > Spectrogram labels its frequencies beside the cells,
  not on top of them.
- The TF-IDF and Count Vectorizer nodes and Explore > Text's Terms count a
  large column in a fraction of the memory: each term is kept once, and a
  row holds its term ids and counts, not a table of strings (300,000 rows of
  words and word pairs, min_df 1: 1.8 GB instead of 6.3 GB, and 29 s instead
  of 102 s, with the same figures).
- Linear algebra on a table of many rows no longer runs the GPU out of
  memory: the SVD (and rank, condition, least squares) of a tall matrix
  comes from its small Gram matrix instead of the full SVD, whose U is rows
  x rows (160,000 rows: 200 GB). Tools > Linear Algebra's From a table and
  the new steps use it.
- The PCA node learns from the Train rows only (it learned from every row,
  validation and test rows included, and dropped the split column when it
  output pc_* alone): its means, deviations and components come from the
  Train rows, every row is projected, the split is carried, and it can save
  and reuse its fitted values like the scalers.
- Data Studio Series and Forecast: Stop ends the fit running in the
  background (it ran on to its end, up to minutes on a long series).
- Training Dashboard: the Active model card no longer shows the internal
  provenance key (`restored_best_checkpoint`, wrapped mid-word) under the
  State line that says it in words.
