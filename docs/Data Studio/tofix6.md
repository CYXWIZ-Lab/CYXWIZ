# Machine Learning Model Training Pipeline From Scratch

This note captures the normal end-to-end pipeline for training a
machine learning model from scratch.

The goal is to keep one simple reference that engineers, product, and
workflow designers can map into CyxWiz nodes, training flows, and UI.

This document also covers:

- the machine learning pipeline mapped to CyxWiz nodes
- the difference between a classic ML pipeline and a deep learning
  pipeline
- a stage-by-stage mapping to the current CyxWiz node surface
- what is currently missing, partial, or misleading

---

## Standard Pipeline

### 1. Define the problem

Decide:

- what you want to predict
- whether this is regression, classification, clustering, forecasting,
  ranking, or anomaly detection
- what the target variable is
- what success metric matters
- what constraints matter: speed, accuracy, memory, explainability,
  privacy, latency

Without this step, the rest of the pipeline is usually built on weak
assumptions.

---

### 2. Collect data

Gather raw data from the source systems:

- files
- databases
- APIs
- sensors
- logs
- third-party datasets

If this is supervised learning, labels must exist or be created.

---

### 3. Inspect and understand the data

Before training, inspect:

- data types
- feature distributions
- missing values
- duplicated rows
- class imbalance
- outliers
- label quality
- leakage risk

This is where many bad modeling decisions should be stopped early.

---

### 4. Clean and prepare the data

Typical preparation steps:

- remove duplicates
- fix invalid records
- handle missing values
- encode categorical variables
- scale or normalize numeric features if needed
- align column types
- remove obvious leakage columns

This stage should produce a model-ready dataset.

---

### 5. Split the data

Create separate datasets for:

- training
- validation
- test

Common split patterns:

- `70 / 15 / 15`
- `80 / 10 / 10`

For time-series data, splits should respect time order.

---

### 6. Do feature engineering

Transform raw columns into more useful signals:

- select relevant features
- create derived features
- aggregate values
- create lag/window features for time series
- reduce noise
- optionally reduce dimensionality with methods like PCA

This step often matters more than changing the model itself.

---

### 7. Choose the model

Pick a model based on the problem and data type.

Examples:

- `Linear Regression` for numeric prediction
- `Logistic Regression` for binary classification
- `Random Forest` for tabular prediction
- `Gradient Boosting / XGBoost` for strong tabular baselines
- `Neural Networks` for more complex nonlinear tasks
- sequence models for text, audio, and advanced forecasting

The first model should usually be a clear baseline, not the most
complex possible option.

---

### 8. Train the model

Training means:

- feed the model the training data
- compute predictions
- compute loss
- optimize the parameters
- repeat over epochs or fit iterations

This is the core optimization step.

---

### 9. Validate and tune

Use the validation set to:

- compare models
- tune hyperparameters
- detect overfitting
- detect underfitting
- choose the best training configuration

This is where model selection happens.

---

### 10. Test the final model

Once the model is chosen, evaluate it on the held-out test set.

This gives the best estimate of real generalization performance.

The test set should not be used for repeated tuning.

---

### 11. Save artifacts

Save everything needed to reproduce and deploy:

- trained model
- preprocessing steps
- feature list
- label mappings
- hyperparameters
- metrics
- dataset/version metadata

This is necessary for reproducibility and deployment.

---

### 12. Deploy or use for inference

Use the trained model in one of these forms:

- batch prediction
- local inference
- API service
- embedded application
- internal enterprise workflow

Training only has value once predictions can be used.

---

### 13. Monitor and retrain

After deployment, track:

- data drift
- model drift
- prediction quality
- latency
- failure rates

Retrain when the data distribution or business target changes.

---

## Short Version

The short pipeline is:

`Problem -> Data -> Clean -> Split -> Feature Engineering -> Train -> Validate -> Test -> Deploy -> Monitor`

---

## CyxWiz Node-Level Mapping

In CyxWiz terms, the ideal pipeline should look like:

`Data Input -> Data Prep -> Split -> Feature Processing -> Model -> Loss -> Optimizer -> Train -> Evaluate -> Save Model`

More explicitly:

1. `Data Input`
2. `Cleaning / Preprocessing Nodes`
3. `Train/Val/Test Split`
4. `Feature Engineering Nodes`
5. `Model Layer Nodes`
6. `Loss Node`
7. `Optimizer Node`
8. `Training Execution`
9. `Evaluation Nodes`
10. `Model Save / Export`

---

## Machine Learning Pipeline Mapped to CyxWiz Nodes

This is the practical mapping for a normal supervised ML workflow in
CyxWiz.

### Stage 1. Problem definition

CyxWiz status:

- no dedicated "Problem Type" node
- this is currently implied by the chosen graph shape:
  - regression model + `MSELoss`
  - classification model + `CrossEntropyLoss` or `BCELoss`
  - time-series graph + `TimeSeriesWindow`

Current issue:

- the engine does not make problem type explicit enough
- users infer task type indirectly from node combinations

Recommendation:

- add explicit templates or graph presets for:
  - regression
  - binary classification
  - multiclass classification
  - time-series forecasting

---

### Stage 2. Data collection / input

Current CyxWiz nodes and UI:

- `DataInput`
- `DatasetInput`
- Data Studio data-source nodes
- `SaveDataset`
- `ExportCSV`

Current strengths:

- `DataInputDialog` supports multiple file categories:
  - tabular
  - image
  - audio
  - text
  - time series
- the engine can load into `DataRegistry`

Current gaps / misleading parts:

- video is visibly present in the UI but not actually supported
- some built-in dataset sources and cloud/dataset nodes still lead to
  placeholder execution paths
- Data Studio and training graph input flows still feel like separate
  systems

---

### Stage 3. Data inspection / understanding

Current CyxWiz nodes and UI:

- `DataInputDialog` preview
- profiling tab in `DataInputDialog`
- `DataProfiler` style functionality
- table viewers / data preview panels

Current gaps / misleading parts:

- profiling support is not yet trustworthy enough as a graph-stage
  decision tool
- some runtime data analysis nodes are still passthrough or partial
- there is no strong "inspect before training" workflow baked into the
  graph

Recommendation:

- make data profiling a first-class graph and UI step
- surface missing values, class balance, column types, label preview,
  and shape summary before compile/train

---

### Stage 4. Data cleaning and preprocessing

Current CyxWiz nodes:

- `FillMissing`
- `RemoveDuplicates`
- `DetectOutliers`
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `LabelEncoder`
- `OrdinalEncoder`
- `TargetEncoder`

Current strengths:

- these nodes match the shape of a real ML preprocessing pipeline
- several preprocessing nodes already have operator-backed
  implementations

Current gaps / misleading parts:

- some Data Studio preprocessing support is split between:
  - legacy `PipelineExecutor`
  - newer `PipelineOperatorFactory`
- not all preprocessing nodes are guaranteed to run through the newer
  path
- unsupported or placeholder behavior is still too easy to reach

---

### Stage 5. Split the data

Current CyxWiz nodes:

- `Train/Val/Test Split`
- `TimeSeriesSplit`

Current strengths:

- the idea is correct
- time-series split exists as a distinct concept

Current gaps / misleading parts:

- split behavior is not yet uniformly modeled across all dataset modes
- Arrow and Parquet training flows still have parity risks
- split semantics are not clearly surfaced enough for users who need
  leakage-safe pipelines

---

### Stage 6. Feature engineering

Current CyxWiz nodes:

- `PCANode`
- `TimeSeriesWindow`
- `TimeSeriesFeatures`
- `LogTransform`
- `Differencing`
- `TextTokenizer`
- `TFIDFVectorizer`
- `CountVectorizer`
- clustering and signal-processing feature nodes

Current strengths:

- this is one of the stronger conceptual areas in the product
- many of these now have real operator-backed implementations

Current gaps / misleading parts:

- the audited legacy `PipelineExecutor` placeholder branches now fail
  closed instead of returning fake success, but old duplicate function
  bodies and operator-backed implementations still coexist
- multi-path execution makes support truth unclear
- some advanced nodes exist in UI/metadata before their runtime story is
  fully converged

---

### Stage 7. Model selection / model definition

Current CyxWiz nodes for classic ML:

- `LinearRegressionNode`
- `PolynomialRegressionNode`
- UI/documented surface for:
  - `LogisticRegression`
  - `DecisionTree`
  - `RandomForest`
  - `GradientBoosting`
  - `SVM`
  - `KNN`
  - `NaiveBayes`

Current CyxWiz nodes for neural networks / deep learning:

- `Dense`
- `Dropout`
- `BatchNorm`
- `Flatten`
- activation nodes such as:
  - `ReLU`
  - `Sigmoid`
  - `Tanh`
  - `Softmax`
- higher-end documented / visible nodes such as:
  - `Conv2D`
  - `MaxPool2D`
  - `AvgPool2D`
  - `GlobalMaxPool`
  - `GlobalAvgPool`
  - `LayerNorm`
  - `GroupNorm`
  - `InstanceNorm`
  - `MultiHeadAttention`

Current strengths:

- the frontend story for model construction is ambitious
- backend layer support is broader than the training graph currently
  exposes
- recurrent layer `input_size` is now treated as graph-derived UI truth:
  the side-panel property is hidden for `LSTM`, `GRU`, and `RNN`, while
  compile/model-building derive the real value from the previous node
  output shape such as `Embedding.embedding_dim`
- new `Dense` nodes now use the same default output width across metadata
  and node creation (`units=64`), avoiding two different defaults for the
  same layer

Current gaps / misleading parts:

- classic ML model nodes beyond linear/polynomial are still weak in
  runtime truth; audited legacy placeholder paths now fail closed, but
  canonical implementations are still missing
- several deep learning nodes are visible and documented but not
  reliably supported end to end by `GraphCompiler -> ModelBuilder ->
  TrainingExecutor`
- CNN/Bidirectional and unavailable attention surfaces are now
  blocked or guarded rather than silently accepted; implementation is
  still pending before they can become real training nodes
- recurrent GPU placement remains conservative: current `GRU` CUDA is
  CPU-routed by compiler/runtime policy until a probe or fused/native
  GPU recurrent implementation lands

---

### Stage 8. Loss definition

Current CyxWiz nodes:

- `MSELoss`
- `CrossEntropyLoss`
- `BCELoss`
- `BCEWithLogits`
- `L1Loss`
- `SmoothL1Loss`
- `HuberLoss`
- `NLLLoss`

Current strengths:

- the loss surface is reasonably complete for first-pass workflows

Current gaps / misleading parts:

- compile gates still catch missing target/label wiring late
- loss selection is not always well-guided by task type

Recommendation:

- tie loss suggestions to graph/task templates

---

### Stage 9. Optimizer definition

Current CyxWiz nodes and UI:

- `SGD`
- `Adam`
- `AdamW`
- `RMSprop`
- `Adagrad`
- `NAdam`
- dedicated optimizer settings panel

Current strengths:

- the UI presents a modern optimizer story
- `GraphCompiler` maps `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`,
  and `NAdam` into backend optimizer types
- `BuildSequentialFromConfig` constructs the optimizer through
  `CreateOptimizer(config.GetOptimizerType(), config.learning_rate)`, so
  optimizer execution belongs to the training path, not `PipelineExecutor`

Current gaps / misleading parts:

- the central supported-training-backend capability table currently names
  supported model layers, not loss/optimizer/control nodes, so it should not
  be treated as a complete training-node support matrix
- learning-rate schedulers and regularization control nodes are still blocked
  as unsupported training controls until they are connected to execution
- the optimizer settings panel exposes richer optimizer-specific parameters
  than the current graph compiler forwards into model construction; the
  training path currently preserves the selected optimizer and learning rate

---

### Stage 10. Training execution

Current CyxWiz path:

- `GraphCompiler`
- `TrainingManager`
- `TrainingExecutor`
- `BuildSequentialFromConfig`

Current strengths:

- there is a real compiled training path
- training and debug share the same model-building path

Current gaps / misleading parts:

- training support is narrower than the visible node surface
- compile success does not always mean faithful graph realization
- some nodes are accepted by compile logic but ignored or dropped by
  model building
- model-building must keep owning derived tensor dimensions. Users should
  edit architectural intent such as `hidden_size`, `embedding_dim`, and
  `return_sequences`; connected layer input sizes should come from graph
  shape propagation rather than hand-entered defaults

---

### Stage 11. Evaluation / validation / tuning

Current CyxWiz nodes and panels:

- `ConfusionMatrix`
- `ROCCurve`
- `LearningCurves`
- `FeatureImportance`
- `CrossValidation`
- training plot panel
- test and evaluation panels

Current strengths:

- the UI has the right analytics vocabulary

Current gaps / misleading parts:

- several evaluation nodes are still placeholder-success paths in
  `PipelineExecutor`
- some evaluation capability lives more in panels/tools than in truthful
  graph execution
- cross-validation is not yet a trustworthy graph-level training stage

---

### Stage 12. Save / export / deployment handoff

Current CyxWiz nodes and UI:

- `SaveDataset`
- `ExportCSV`
- model save callback in main window
- export model dialog / toolbar actions

Current strengths:

- the engine clearly wants save/export to be part of the workflow

Current gaps / misleading parts:

- dataset export is clearer than model export
- some export surfaces are present in UI before full backend/export-path
  completion
- the line between "save graph", "save weights", and "export deployable
  model" is not yet clean enough

Current backend truth:

- `Save Model` in the main window calls `TrainingManager::SaveModel`, which
  preserves the last trained `SequentialModel` and writes the model through
  `SequentialModel::Save`;
- `Export Model` uses `ExportDialog -> ModelExporter`, which is the richer
  deployment/export path;
- `.cyxmodel` export is implemented by `CyxModelFormat`, currently using a
  directory-based package layout with manifest, config, optional graph/history,
  and weights;
- `.safetensors` export writes model parameter tensors;
- ONNX export is only available when `CYXWIZ_HAS_ONNX_EXPORT` is compiled, not
  merely when ONNX runtime/import support is present;
- GGUF export is intentionally disabled and returns a planned/future-release
  error if called directly.

Progress note:

- the export dialog now uses the same ONNX availability macro as
  `ModelExporter`, so ONNX export is not shown as available when only ONNX
  inference/runtime support is present.

---

## Difference Between a Classic ML Pipeline and a Deep Learning Pipeline

The two pipelines are related, but they are not the same.

### Classic ML pipeline

Classic ML is usually:

- more dependent on explicit feature engineering
- more common for tabular data
- often easier to train, debug, and explain
- often built from:
  - regression
  - tree models
  - SVM
  - KNN
  - Naive Bayes

Typical shape:

`Data -> Clean -> Encode/Scale -> Split -> Feature Engineering -> Train Classical Model -> Evaluate`

The human or the preprocessing pipeline does more of the feature work.

### Deep learning pipeline

Deep learning is usually:

- more dependent on network architecture
- more compute intensive
- more common for images, audio, text, and large-scale sequence data
- less reliant on hand-crafted feature engineering
- more reliant on:
  - layers
  - activations
  - losses
  - optimizers
  - training stability

Typical shape:

`Data -> Tensor/Sequence Preparation -> Split -> Model Architecture -> Loss -> Optimizer -> Train for Epochs -> Validate -> Export`

The model learns internal representations instead of depending as much
on manual feature engineering.

### Practical difference inside CyxWiz

Classic ML in CyxWiz should emphasize:

- dataset loading
- cleaning
- encoding
- scaling
- feature engineering
- classical model node
- evaluation

Deep learning in CyxWiz should emphasize:

- dataset loading
- shape-safe batching
- architecture graph
- activation/loss/optimizer choices
- training dashboard
- debug visibility
- checkpoint/export flow

Current product issue:

- CyxWiz currently presents a stronger deep-learning node vocabulary
  than the current end-to-end training graph fully supports
- at the same time, many classical ML / analytics nodes exist but still
  have placeholder or split runtime paths

So both sides need support cleanup, but in different ways.

---

## Current CyxWiz Mapping by Pipeline Type

### A. Classic ML pipeline mapped to current CyxWiz

Ideal graph:

`DataInput -> FillMissing / RemoveDuplicates / Encoding / Scaling -> Train/Val/Test Split -> PCA / Feature Nodes -> LinearRegression / LogisticRegression / Tree / SVM / KNN -> Evaluation -> Save`

What currently maps well:

- `DataInput`
- preprocessing nodes
- `Train/Val/Test Split`
- feature nodes like:
  - `PCANode`
  - `TimeSeriesFeatures`
  - `TFIDFVectorizer`
  - `CountVectorizer`
- `LinearRegressionNode`
- `PolynomialRegressionNode`

What is missing or misleading:

- many classical ML nodes exist in surface area but not as trustworthy
  runtime-supported graph nodes
- evaluation nodes overstate real graph execution support
- support truth varies too much by execution path

### B. Deep learning pipeline mapped to current CyxWiz

Ideal graph:

`DataInput -> Split / Batch Prep -> Dense / Conv / Attention / Normalization / Activation -> Loss -> Optimizer -> Train -> Validate -> Save/Export`

What currently maps well:

- `Dense`
- basic activations
- `Dropout`
- `MSELoss`
- `CrossEntropyLoss`
- `Adam`, `SGD`, `AdamW`
- shared training/debug compiled flow

What is missing or misleading:

- `Conv2D`, pooling, attention, and some normalization nodes are more
  visible than they are truly supported end to end
- optimizer surface is ahead of compile/runtime truth
- the frontend suggests broader architecture freedom than the current
  model-builder path can safely execute

---

## Stage-by-Stage Gap Summary

### Strong enough today

- `DataInput` as the entry point
- basic preprocessing and scaling concepts
- time-series feature/windowing direction
- basic dense-network training path
- standard loss-node surface

### Present but still inconsistent

- split handling across different dataset/storage modes
- feature-engineering execution across old/new runtime paths
- model export clarity
- evaluation as graph-execution truth

### Present but misleading

- several classical ML nodes remain visible even though their audited
  legacy runtime path now fails closed
- several deep-learning architecture nodes remain visible/documented even
  though the compiler now blocks them until backend support lands
- some Data Studio nodes look executable but are now blocked or UI-only
  rather than real graph transforms

### Missing or needs stronger product truth

- explicit task-type templates
- clear distinction between classic ML workflows and deep learning
  workflows
- one authoritative support matrix for nodes
- complete fail-closed behavior for every unsupported graph node, beyond
  the audited legacy executor and training-compiler slices
- visible user messaging when a node is:
  - real
  - partial
  - blocked
  - UI-only

---

## Engineering Relevance

This pipeline should be the baseline reference for:

- frontend workflow design
- default templates
- training graph validation
- onboarding examples
- documentation
- node grouping and naming
- support-matrix cleanup
- truthful node availability

Any CyxWiz training experience that breaks this mental model without a
clear reason will be harder for users to trust and learn.

The immediate product/design lesson is:

- CyxWiz should stop presenting one undifferentiated "all ML" story
- it should present at least two clear workflow lanes:
  - `Classic ML`
  - `Deep Learning`

Then each lane should only expose nodes and templates that the current
runtime can actually support truthfully.
