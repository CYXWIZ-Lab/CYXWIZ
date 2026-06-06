# Tool Panel to Graph Migration Plan

This document is the current source-of-truth for reducing standalone analysis
panel sprawl without removing useful tools prematurely.

It complements the older high-level Tool-to-Node design notes in
`cyxwiz_studio_upgrade_design.md`. That design describes the intended product
direction. This file records the current implementation boundary observed in
the engine.

## Product Rule

If a tool transforms data, trains/evaluates a model, computes reusable
features, or produces pipeline output, its durable configuration belongs in the
graph.

Standalone panels are acceptable only when they are clearly:

- inspectors over existing project state
- development utilities
- monitoring/debugging surfaces
- file/project management surfaces
- one-off calculators whose output is not part of a reproducible pipeline

Do not add another standalone analysis panel when an equivalent `NodeType` and
pipeline operator already exist.

## Already Node-Backed Workflows

These areas already have graph-facing nodes, and several also have registered
pipeline operators:

| Area | Node-backed coverage | Current standalone panels |
| --- | --- | --- |
| Clustering | `KMeansCluster`, `DBSCANCluster`, `HierarchicalCluster`, `GMMCluster` | K-Means, DBSCAN, hierarchical, GMM, cluster evaluation |
| Dimensionality reduction | `PCANode` | Dim reduction |
| Preprocessing | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `LabelEncoder`, `OrdinalEncoder`, `TargetEncoder`, `OutlierDetector`, `LogTransform` | normalization, standardization, log transform, outlier detection, feature scaling |
| Text | `TextTokenizer`, `TokenizerNode`, `TFIDFVectorizer`, `CountVectorizer`, `WordFrequencyNode`, `SentimentAnalyzer`, `WordEmbeddings` | tokenization, TF-IDF, word frequency, sentiment, embeddings |
| Signal processing | `FFTNode`, `Convolution1D`, `FilterDesigner`, `WaveletTransform` | FFT, convolution, filter designer, wavelet |
| Time series | `TimeSeriesWindow`, `TimeSeriesFeatures`, `TimeSeriesSplit`, `TimeSeriesDecomposition`, `ACFNode`, `PACFNode`, `StationarityTest`, `SeasonalityDetector`, `ARIMAForecaster`, `ExponentialSmoothing` | decomposition, ACF/PACF, stationarity, seasonality, forecasting |
| Regression and metrics | `LinearRegressionNode`, `PolynomialRegressionNode`, `RegressionMetricsNode` | regression, learning curves, cross validation |
| Matrix analytics | `MatrixCalculator`, `SVDNode`, `QRDecomposition`, `CholeskyDecomposition`, `EigenDecomposition` | matrix calculator, SVD, QR, Cholesky, eigen decomposition |
| Explainability and visualization | `FeatureImportanceNode`, `GradCAMNode`, `SaliencyMapNode`, `BarChart` | feature importance, Grad-CAM, chart/visualization panels |

For these areas, future UI work should prefer one of:

- add/open the matching node in the current graph
- show a read-only inspector for the selected graph node output
- keep the panel as a transient preview, but persist configuration on the node

## Panels That Can Stay Standalone

These are not pipeline steps and should not be forced into graph nodes:

| Panel type | Reason |
| --- | --- |
| Asset browser, table viewer, data explorer | Project/data inspection and navigation |
| Script editor, command window, variable explorer | Development workflow |
| Profiler, memory panels, task progress, test results | Runtime monitoring/debugging |
| Plugin manager, wallet, job status, P2P training | System/service management |
| Calculator, unit converter, random/hash, JSON/regex utilities | Ad hoc utility output, not durable graph state |
| Annotation editor | Dedicated labeling workspace; graph integration should consume its exported datasets |

## Migration Rules

1. If a standalone panel has an equivalent node and pipeline operator, do not
   add new persistent settings to the panel. Add missing settings to the node
   metadata instead.
2. If a panel has an equivalent node but no pipeline operator, keep it
   standalone only as a temporary preview and prioritize the operator.
3. If a panel launches from a menu and has a graph equivalent, the menu entry
   should eventually offer `Add Node to Graph` before `Open Standalone Panel`.
4. Graph-backed tools must persist parameters in `MLNode::parameters`, not
   panel-local state.
5. Panel-local state is acceptable for layout, selected row, preview page,
   temporary filters, and render-only toggles.

## Next Bounded Implementation Tasks

1. Add a graph-node handoff path for one mature family first, preferably text
   or clustering because both have multiple existing pipeline operators.
2. Add a small menu/command annotation for graph-backed tools so command-palette
   search can distinguish durable graph workflows from transient utilities.
3. Audit standalone panels that duplicate parameters already present in
   `NodeMetadataRegistry` and stop expanding those duplicated controls.
4. Add a regression test around `PipelineOperatorFactory::GetSupportedTypes`
   so newly migrated tools are visibly registered.
