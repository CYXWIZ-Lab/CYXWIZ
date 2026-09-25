#pragma once

// The graph data model: node and pin types and the saved node/link records.
// GUI-free, so the training core (graph compiler, model builder, executors)
// can use it without the editor; gui/node_editor.h includes it (TOFIX118 P2).
// Numeric NodeType values are persisted in .cyxgraph files: append only.

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace gui {

// Node category for organization and UI display (Unified Canvas Phase 1)
enum class NodeCategory {
    // Data I/O (Read & Write)
    DataSources,      // CSV, SQL, HDF5, API + Export CSV, Parquet, SQL, JSON
    Database,         // PostgreSQL, MySQL, SQLite, MongoDB (Coming Soon)
    CloudStorage,     // AWS S3, Azure Blob, Google Cloud (Coming Soon)
    DataTransform,    // Filter, Join, GroupBy, Sort, etc.

    // Analytics & Visualization
    Analytics,        // Stats, Visualize, Sample, Correlation
    Visualization,    // 3D Scatter, Maps, Network Graphs (Coming Soon)

    // ML Layers
    Layers,           // Dense, Conv2D, LSTM, etc. (ML layers)
    Activation,       // ReLU, Sigmoid, Softmax, etc.
    Pooling,          // MaxPool, AvgPool, etc.
    Normalization,    // BatchNorm, LayerNorm, etc.
    Attention,        // MultiHeadAttention, Transformer, etc.
    Recurrent,        // RNN, LSTM, GRU, etc.
    ShapeOps,         // Reshape, Permute, Squeeze, etc.
    MergeOps,         // Concatenate, Add, Multiply, etc.
    Upsampling,       // ConvTranspose, Upsample, PixelShuffle

    // Training & Models
    Training,         // Optimizer, Loss, LR Scheduler
    Regularization,   // L1, L2, Dropout
    ModelIO,          // Model Reader/Writer, Checkpoints (Coming Soon)
    MLServices,       // AutoML, HuggingFace, MLflow (Coming Soon)
    Explainability,   // SHAP, LIME (Coming Soon)

    // Data Processing
    Preprocessing,    // Normalize, Scale, Encode (existing nodes)
    DataPipeline,     // DatasetInput, DataLoader, Augmentation
    TextProcessing,   // Tokenizer, Vocabulary, Padding
    TimeSeries,       // Window, Features, Split
    Audio,            // AudioInput, Spectrogram, MFCC
    JsonXml,          // JSON Path, XML Reader (Coming Soon)

    // Specialized
    DNN,              // Pre-trained models, detection, pose
    RL,               // Gym, ReplayBuffer, Policy, Value
    BigData,          // Spark, Dask, Ray, Kafka (Coming Soon)

    // Workflow & UI
    Workflow,         // Loop, IF Switch, Try/Catch (Coming Soon)
    Widgets,          // Interactive inputs (Coming Soon)
    Reporting,        // PDF, HTML Reports (Coming Soon)
    Utility,          // Lambda, Identity, Constant
    Signal,           // Sliders, Sine, Scope (for simulation)

    Plugin,           // Plugin-defined custom nodes
    Unknown           // Fallback
};

// Node types for ML model building
enum class NodeType {
    // ===== Core Layers =====
    Dense,

    // Convolutional Layers
    Conv1D,
    Conv2D,
    Conv3D,
    DepthwiseConv2D,

    // Pooling Layers
    MaxPool2D,
    AvgPool2D,
    GlobalMaxPool,
    GlobalAvgPool,
    AdaptiveAvgPool,

    // Normalization Layers
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,

    // Regularization
    Dropout,
    Flatten,

    // ===== Recurrent Layers =====
    RNN,
    LSTM,
    GRU,
    Bidirectional,
    TimeDistributed,
    Embedding,

    // ===== Attention & Transformer =====
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    LinearAttention,      // O(n) linear attention (Performer/Linear Transformer)
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,

    // ===== Activation Functions =====
    ReLU,
    LeakyReLU,
    PReLU,
    ELU,
    SELU,
    GELU,
    Swish,
    Mish,
    Sigmoid,
    Tanh,
    Softmax,

    // ===== Shape Operations =====
    Reshape,
    Permute,
    Squeeze,
    Unsqueeze,
    View,
    Split,

    // ===== Merge Operations =====
    Concatenate,
    Add,
    Multiply,
    Average,

    // ===== Tensor Reductions =====
    TensorSum,
    TensorMean,
    TensorMax,
    TensorMin,
    TensorProd,
    TensorVar,
    TensorStd,

    // ===== Tensor Broadcast / Unary Math =====
    TensorBroadcastTo,
    TensorExpand,
    TensorPow,
    TensorSqrt,
    TensorExp,
    TensorLog,
    TensorAbs,
    TensorSign,
    TensorClip,

    // ===== Tensor Linalg / Masks =====
    TensorDot,
    TensorBatchMatMul,
    TensorCompare,
    TensorLogicalMask,
    TensorIndexSelect,

    // ===== Output =====
    Output,

    // ===== Loss Functions =====
    MSELoss,
    CrossEntropyLoss,
    BCELoss,
    BCEWithLogits,
    L1Loss,
    SmoothL1Loss,
    HuberLoss,
    NLLLoss,

    // ===== Optimizers =====
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    NAdam,

    // ===== Learning Rate Schedulers =====
    StepLR,
    CosineAnnealing,
    ReduceOnPlateau,
    ExponentialLR,
    WarmupScheduler,

    // ===== Regularization Nodes =====
    L1Regularization,
    L2Regularization,
    ElasticNet,

    // ===== Utility Nodes =====
    Lambda,
    Identity,
    Constant,
    Parameter,

    // ===== Signal / Control Nodes =====
    SignalSlider,       // Interactive slider outputting scalar value
    SineWave,           // Sine wave generator (amplitude, frequency, phase)
    StepSignal,         // Step function (0 before t, value after t)
    RampSignal,         // Linear ramp from start to end value
    SignalScope,        // Real-time signal plotter (input visualization)

    // ===== Data Pipeline Nodes =====
    DatasetInput,       // Load dataset from DataRegistry
    DataLoader,         // Batch iterator with shuffle/drop_last
    Augmentation,       // Transform pipeline for data augmentation
    DataSplit,          // Train/val/test splitter
    TensorReshape,      // Reshape tensor dimensions (legacy, use Reshape)
    Normalize,          // Normalize values (mean/std, domain-aware)
    OneHotEncode,       // Label encoding

    // ===== Image Transform Nodes (Phase 1) =====
    // Single-responsibility preprocessing nodes for image data. Each
    // handles one transform. The graph compiler extracts them in order
    // and builds an ImageTransformPipeline passed to the image batcher.
    Resize,             // Resize image to target width/height
    CenterCrop,         // Crop from center to target size
    RandomCrop,         // Random crop (augmentation)
    HorizontalFlip,     // Random horizontal flip (augmentation)
    VerticalFlip,       // Random vertical flip (augmentation)
    ImageRotate,        // Random rotation by angle (augmentation)
    ColorJitter,        // Random brightness/contrast/saturation/hue
    ImageGaussianBlur,  // Gaussian blur with kernel size + sigma
    Grayscale,          // Convert to single-channel grayscale

    // ===== Composite Nodes =====
    Subgraph,           // Encapsulated subgraph (collapsible)

    // ===== DNN Inference Nodes =====
    DNNModelLoad,       // Load pre-trained DNN model
    DNNDetect,          // Object detection (YOLO, SSD)
    DNNClassify,        // Image classification
    DNNPoseEstimate,    // Pose estimation (OpenPose)
    DNNFaceDetect,      // Face detection
    DNNPreprocess,      // DNN preprocessing pipeline

    // Pre-trained Model Shortcuts
    PretrainedYOLO,     // YOLOv4 with default config
    PretrainedMobileNet,// MobileNet classifier
    PretrainedOpenPose, // OpenPose body estimation
    PretrainedFaceNet,  // Face detector

    // Post-processing Nodes
    NonMaxSuppression,  // NMS for detections
    ArgMax,             // Classification argmax
    TopK,               // Top-K predictions
    ThresholdFilter,    // Filter by confidence

    // ===== Text Processing Nodes =====
    TextCleanNode,      // Clean one text column
    TextTokenizer,      // Tokenize text -> integer sequences
    TextVocabulary,     // Manage word<->index vocabulary
    TextPadding,        // Pad/truncate sequences to fixed length

    // ===== Upsampling Layers =====
    ConvTranspose2D,    // Learnable transposed convolution
    Upsample,           // Nearest/bilinear interpolation upsampling
    PixelShuffle,       // Sub-pixel convolution rearrangement

    // ===== Time-Series Nodes =====
    TimeSeriesWindow,   // Sliding window for sequential data
    TimeSeriesFeatures, // Lag, rolling, differencing features
    TimeSeriesLag,      // Add lag columns for one numeric time-series column
    TimeSeriesSplit,    // Chronological train/val/test split
    LogTransform,       // log1p stabilization for exponential growth
    Differencing,       // Subtract lagged values to remove trend/seasonality

    // ===== Audio Processing Nodes =====
    AudioInput,         // Load audio dataset
    Spectrogram,        // Compute spectrogram from waveform
    MelSpectrogram,     // Compute mel-spectrogram
    MFCC,               // Extract MFCC features
    AudioAugmentation,  // Time stretch, pitch shift, noise

    // ===== RL Nodes =====
    GymEnvironment,     // OpenAI Gym environment connector
    ReplayBufferNode,   // Experience replay buffer
    PolicyNetwork,      // Actor network for RL
    ValueNetwork,       // Critic network for RL
    RLTraining,         // RL training loop controller

    // ===== Smart I/O Nodes (Unified - replaces individual format nodes) =====
    DataInput,          // Universal data input with smart dialog (auto-detects format)
    DataOutput,         // Universal data export with smart dialog (supports all formats)
    DeployToNodeEditorNode, // Mark a dataset ready for Node Editor deployment
    DataConvert,        // Convert datasets between supported file formats

    // ===== Legacy Data Source Nodes (kept for compatibility) =====
    CSVFile,            // Load CSV file into Arrow table
    SQLQuery,           // Execute SQL query, return Arrow table
    HDF5Dataset,        // Load HDF5 dataset into Arrow
    ParquetFile,        // Load Parquet file into Arrow
    JSONFile,           // Load JSON file into Arrow
    ExcelFile,          // Load Excel file into Arrow
    RESTAPISource,      // Fetch data from REST API

    // Additional I/O formats (supported by engine)
    TSVFile,            // Load TSV file into Arrow
    TXTFile,            // Load plain text file
    ImageCSVDataset,    // Images folder + CSV labels
    StreamingDataset,   // Stream large datasets
    ARFFFile,           // Weka ARFF format
    FashionMNISTDataset,// Fashion-MNIST dataset
    CIFAR100Dataset,    // CIFAR-100 dataset

    // Additional I/O formats (Arrow, NumPy, Domain-specific)
    FeatherFile,        // Apache Arrow Feather format
    ArrowIPCFile,       // Arrow IPC binary format
    NumPyFile,          // NumPy .npy/.npz files
    AudioFolderDataset, // Audio files with class folders
    TimeSeriesCSV,      // Time series CSV with windowing
    TextCorpusDataset,  // Text corpus for NLP

    // ===== Data Transform Nodes (Unified Canvas Phase 1) =====
    FilterRows,         // Filter rows by SQL WHERE condition
    SelectColumns,      // Select specific columns
    JoinTables,         // Join two datasets (inner/left/right/outer)
    GroupByAggregate,   // Group by columns with aggregations
    SortRows,           // Sort rows by columns (ascending/descending)
    FillMissingValues,  // Handle missing values (mean/median/mode/constant)
    RemoveDuplicateRows,// Remove duplicate rows
    PivotTable,         // Pivot wide to long or long to wide
    UnionTables,        // Stack multiple datasets (UNION ALL)
    RenameColumns,      // Rename columns

    // ===== Analytics Nodes (Unified Canvas Phase 1) =====
    DescribeStats,      // Compute statistical summary (count, mean, std, etc.)
    VisualizeData,      // Create plots (scatter, bar, line, histogram)
    SampleRows,         // Sample random rows from dataset
    CorrelationMatrix,  // Compute correlation matrix
    ValueCounts,        // Count unique values per column
    CrossTabulation,    // Cross-tabulation (contingency table)

    // ===== Data Export Nodes (Unified Canvas Phase 1) =====
    ExportCSV,          // Export dataset to CSV
    ExportParquet,      // Export dataset to Parquet
    ExportSQL,          // Write dataset to SQL database
    ExportJSON,         // Export dataset to JSON
    ExportExcel,        // Export dataset to Excel (.xlsx)

    // ===== KNIME-Style Table Manipulation Nodes =====
    RowToColumnNames,   // Promote a row to column headers
    TableSplitter,      // Split table at specified row
    CellExtractor,      // Extract value from specific cell
    CellUpdater,        // Update value in specific cell
    TableCropper,       // Crop table to specified dimensions
    ColumnAppender,     // Append columns from multiple tables
    RowAppender,        // Append rows from multiple tables (alias for UnionTables)
    Unpivot,            // Unpivot wide to long format
    StringManipulation, // String operations (replace, trim, upper, lower)
    MathFormula,        // Apply math formula to columns
    RuleEngine,         // Apply if-then-else rules to create/modify columns

    // ===== Machine Learning Algorithms (Phase 4 - Tool-to-Node Migration) =====
    // Clustering
    KMeansCluster,      // K-Means clustering algorithm
    DBSCANCluster,      // DBSCAN density-based clustering
    HierarchicalCluster,// Hierarchical/Agglomerative clustering
    GMMCluster,         // Gaussian Mixture Model clustering

    // Dimensionality Reduction
    PCANode,            // Principal Component Analysis
    TSNENode,           // t-SNE visualization
    UMAPNode,           // UMAP dimensionality reduction

    // Classification
    DecisionTreeClassifier,  // Decision Tree classifier
    RandomForestClassifier,  // Random Forest ensemble
    GradientBoostingClassifier, // Gradient Boosted Trees
    SVMClassifier,      // Support Vector Machine classifier
    KNNClassifier,      // K-Nearest Neighbors classifier
    NaiveBayesClassifier, // Naive Bayes classifier
    LogisticRegressionNode, // Logistic Regression

    // Regression
    LinearRegressionNode,   // Linear Regression
    PolynomialRegressionNode, // Polynomial Regression
    SVMRegressor,       // Support Vector Regression

    // ===== Model Evaluation Nodes (Phase 4) =====
    ConfusionMatrixNode,    // Confusion matrix visualization
    ROCCurveNode,       // ROC curve and AUC computation
    PRCurveNode,        // Precision-Recall curve
    LearningCurvesNode, // Training/validation learning curves
    FeatureImportanceNode, // Feature importance analysis
    CrossValidationNode,   // K-Fold cross-validation
    RegressionMetricsNode,  // Regression metrics (MSE, RMSE, MAE, R²)

    // ===== Data Preprocessing Nodes (Phase 4) =====
    StandardScaler,     // Z-score standardization (mean=0, std=1)
    MinMaxScaler,       // Min-Max scaling to [0,1]
    RobustScaler,       // Robust scaling using median/IQR
    LabelEncoder,       // Encode categorical labels as integers
    OrdinalEncoder,     // Encode ordinal categories
    TargetEncoder,      // Target-based encoding
    BinningNode,        // Bin continuous numeric columns
    PolynomialFeaturesNode, // Generate polynomial features for one numeric column
    // (TrainTestSplit removed — use DataSplit, which supports 3-way train/val/test)

    // ===== Advanced Preprocessing Nodes (Phase 3 - UI Consolidation) =====
    OutlierDetector,    // Detect/remove outliers (IQR, Z-score, Isolation Forest)
    ImagePreprocessor,  // Image resize, crop, normalize pipeline
    QualityAnalyzer,    // Image quality filtering (blur, brightness, contrast)
    DataValidator,      // Schema validation and data quality checks

    // ===== Dataset Source Nodes (Phase 4 - UI Consolidation) =====
    ImageFolderDataset, // Load images from folder with class labels
    MNISTDataset,       // Load MNIST handwritten digits dataset
    CIFAR10Dataset,     // Load CIFAR-10 image classification dataset
    HuggingFaceDataset, // Load dataset from HuggingFace Hub
    KaggleDataset,      // Load dataset from Kaggle

    // ===== Advanced Augmentation Nodes (Phase 6 - UI Consolidation) =====
    AugmentationPreset, // Predefined augmentation pipelines (ImageNet, CIFAR, Medical, Self-Supervised)
    GeometricTransform, // Geometric transforms (rotate, flip, crop, perspective, affine)
    ColorTransform,     // Color transforms (brightness, contrast, saturation, hue, gamma)
    MorphologyTransform,// Morphological operations (dilate, erode, blur, sharpen, edge)
    AdvancedAugment,    // Advanced augmentation (Cutout, MixUp, CutMix, RandAugment, AutoAugment)

    // ===== Signal Processing Nodes (Phase 4) =====
    FFTNode,            // Fast Fourier Transform
    IFFTNode,           // Inverse FFT
    FilterDesigner,     // Design FIR/IIR filters
    Convolution1D,      // 1D signal convolution
    WaveletTransform,   // Discrete Wavelet Transform

    // ===== Text Analytics Nodes (Phase 4) =====
    TFIDFVectorizer,    // TF-IDF text vectorization
    CountVectorizer,    // Bag-of-words vectorization
    WordEmbeddings,     // Word embeddings (Word2Vec, GloVe)
    SentimentAnalyzer,  // Sentiment analysis
    NamedEntityRecognizer, // NER extraction

    // ===== Utility Nodes (Phase 4) =====
    CalculatorNode,     // Math expression calculator
    UnitConverter,      // Unit conversion utility
    RegexTester,        // Regular expression tester
    JSONPathExtractor,  // Extract data using JSONPath
    DataProfiler,       // Comprehensive data profiling


    // ===== Linear Algebra Nodes (Phase 5 - Tool-to-Node Migration) =====
    SVDNode,            // Singular Value Decomposition
    QRDecomposition,    // QR matrix decomposition
    CholeskyDecomposition, // Cholesky factorization
    EigenDecomposition, // Eigenvalue/eigenvector decomposition
    MatrixCalculator,   // General matrix operations

    // ===== Time Series Analysis Nodes (Phase 5) =====
    TimeSeriesDecomposition, // Trend/Seasonal/Residual decomposition
    ACFNode,            // Autocorrelation function
    PACFNode,           // Partial autocorrelation function
    StationarityTest,   // ADF/KPSS stationarity tests
    SeasonalityDetector,// Detect seasonal patterns
    ARIMAForecaster,    // ARIMA time series forecasting
    ExponentialSmoothing, // Holt-Winters forecasting

    // ===== Statistics Nodes (Phase 5) =====
    HypothesisTest,     // t-test, ANOVA, Chi-Square
    DistributionFitter, // Fit probability distributions

    // ===== Deep Learning Interpretation Nodes (Phase 5) =====
    GradCAMNode,        // Grad-CAM visualization
    SaliencyMapNode,    // Gradient saliency maps

    // ===== Optimization Nodes (Phase 5) =====
    GradientDescentViz, // Visualize optimization paths
    ConvexityAnalyzer,  // Analyze function convexity
    LPSolver,           // Linear programming solver
    QPSolver,           // Quadratic programming solver
    NumericalDifferentiation, // Numerical derivatives
    NumericalIntegration,     // Numerical integration

    // ===== Additional Text Processing (Phase 5) =====
    WordFrequencyNode,  // Word frequency analysis
    TokenizerNode,      // Text tokenization

    // ===== Visualization Nodes (Phase 8 - Plot Types) =====
    // Basic 2D Plots
    LinePlot,           // Line plot (plot())
    ScatterPlot,        // Scatter plot (scatter())
    BarChart,           // Bar chart (bar/barh)
    Histogram,          // Histogram distribution (hist())
    PieChart,           // Pie chart (pie())
    AreaPlot,           // Filled area plot (fill_between())

    // Advanced 2D Plots
    BoxPlot,            // Box plot statistics (boxplot())
    ViolinPlot,         // Violin plot distribution (violinplot())
    ErrorBarPlot,       // Error bar plot (errorbar())
    StepPlot,           // Step plot (step())
    HexbinPlot,         // Hexagonal binning (hexbin())

    // Heatmaps & Matrices
    Heatmap,            // Heatmap visualization
    ContourPlot,        // Contour plot (contour/contourf())
    Imshow,             // Image display (imshow())

    // 3D Plots
    Plot3D,             // 3D line plot
    Scatter3D,          // 3D scatter plot
    SurfacePlot,        // 3D surface plot
    WireframePlot,      // 3D wireframe plot

    // Specialized Plots
    PolarPlot,          // Polar coordinate plot
    QuiverPlot,         // Vector field plot (quiver())
    StreamPlot,         // Streamline plot (streamplot())
    SpectrogramPlot,    // Spectrogram visualization
    NetworkGraph,       // Network/graph visualization

    // Plugin-defined nodes (sentinel — actual type resolved via string lookup)
    PluginCustom,

    // ===== Sequence Tagging / NER Contract Nodes =====
    // Appended to preserve existing serialized numeric NodeType ids.
    NERSequenceBuilder, // Build word/POS/tag sequence samples
    TokenVocabulary,    // Token id vocabulary for sequence tagging
    POSVocabulary,      // Optional POS id vocabulary for sequence tagging
    NERTagVocabulary,   // BIO tag vocabulary for sequence tagging
    SequenceTagOutput,  // Decode/export token-level tag predictions

    // ===== Metric Learning / Siamese Contract Nodes =====
    // Appended to preserve existing serialized numeric NodeType ids.
    PairDatasetBuilder,     // Build aligned pair samples for metric learning
    TripletDatasetBuilder,  // Build anchor/positive/negative samples
    SharedEncoder,          // Declare one shared encoder parameter set
    SiameseBranch,          // Reference a shared encoder branch
    ContrastiveLoss,        // Contrastive pair loss contract
    CosineEmbeddingLoss,    // Cosine embedding pair loss contract
    TripletLoss,            // Triplet loss contract
    PairMetrics,            // Pair distance metric output
    RetrievalMetrics,       // Embedding retrieval metric output
    EmbeddingOutput,        // Embedding inference output
    PairScoreOutput,        // Pair score inference output

    // ===== Appended Training Loss Nodes =====
    FocalLoss,          // Class-imbalance-aware focal cross-entropy loss
    SoftDiceLoss,       // Soft Dice probability-mask loss
    TverskyLoss,        // Tversky probability-mask loss
    JaccardLoss,        // Jaccard/IoU probability-mask loss
    ClassificationMetricsNode, // Classification accuracy/precision/recall/F1 metrics

    // ===== Appended Classic ML Inference Nodes =====
    TreeModelPredictor, // Apply a saved tree-family model artifact

    // ===== Appended Time-Series Baseline Nodes =====
    // Appended to preserve existing serialized numeric NodeType ids.
    SeasonalNaive,      // Repeat the latest seasonal cycle over future horizons
    TimeSeriesSegment,  // Validate timestamps and assign gap-safe segment IDs

    // ===== Appended Classical Regression Inference Node =====
    // Appended to preserve existing serialized numeric NodeType ids.
    RegressionModelPredictor, // Apply a fitted linear/polynomial artifact

    // ===== Appended Data Studio Check step (TOFIX101 package E) =====
    // Appended to preserve existing serialized numeric NodeType ids.
    RowCountCheck,      // Pass through; stop the run when a row count is wrong

    // Special sentinel value
    Unknown
};

// Attribute for node pins (inputs/outputs)
enum class PinType {
    Tensor,      // General tensor data
    Labels,      // Label tensor (for classification)
    Parameters,  // Model parameters
    Loss,        // Loss value
    Optimizer,   // Optimizer state
    Dataset      // Dataset handle reference
    // Note: Shape is metadata (node parameter), not a data flow type
};

// Pin capacity constants for variadic connections
constexpr int PIN_UNLIMITED = -1;  // No limit on connections
constexpr int PIN_SINGLE = 1;      // Standard single connection

// Node pin structure
struct NodePin {
    int id;
    PinType type;
    std::string name;
    bool is_input;  // true = input pin, false = output pin

    // Hover tooltip text — what does this pin carry, where does it
    // come from, what should connect to it. Empty string = no extra
    // tooltip line (the generic name/type/connection-count tooltip
    // still renders). Author this on the data-chain pins so users
    // know "Labels here is the y in (X, y) ground truth, sourced
    // from the DataInput's label_column setting" without having to
    // read the source.
    std::string description;

    // Variadic pin support - enables multiple connections to a single pin
    bool is_variadic = false;        // True for pins accepting multiple connections
    int min_connections = 0;         // Minimum required connections (0 = optional)
    int max_connections = PIN_SINGLE; // Maximum allowed (-1 = unlimited)

    // Visual indicators
    bool is_required = true;         // Visual indicator for required pins

    // Shape information (for smart validation and inference)
    std::vector<size_t> shape;       // Tensor shape flowing through pin
    bool shape_valid = false;        // Whether shape has been computed
};

// Visual node structure
struct MLNode {
    int id;
    NodeType type;
    NodeCategory category;  // Unified Canvas Phase 1: Category for UI organization
    std::string name;
    std::string description;  // KNIME-style: bound text displayed below node, moves with node
    std::vector<NodePin> inputs;
    std::vector<NodePin> outputs;

    // Node-specific parameters (e.g., units for Dense layer)
    std::map<std::string, std::string> parameters;

    // Position for pattern insertion (optional - set by InstantiatePattern)
    float initial_pos_x = 0.0f;
    float initial_pos_y = 0.0f;
    bool has_initial_position = false;  // True if position should be applied when inserting

    // Dynamic pins (for plugin nodes like MuJoCo Plant that change pins based on parameters)
    bool has_dynamic_pins = false;
    std::string dynamic_pin_trigger;     // Parameter name that triggers pin rebuild
    std::string resolved_config;         // Last resolved config value (to detect changes)
    std::string plugin_qualified_name;   // "plugin_id:type_name" for calling ResolveDynamicPins
};

inline bool IsGeneratedDenseName(const std::string& name) {
    constexpr const char* prefix = "Dense (";
    constexpr size_t prefix_length = 7;
    if (name.size() <= prefix_length + 1 ||
        name.compare(0, prefix_length, prefix) != 0 ||
        name.back() != ')') {
        return false;
    }

    for (size_t index = prefix_length; index + 1 < name.size(); ++index) {
        if (name[index] < '0' || name[index] > '9') {
            return false;
        }
    }
    return true;
}

inline std::string EffectiveNodeName(const MLNode& node) {
    if (node.type != NodeType::Dense || !IsGeneratedDenseName(node.name)) {
        return node.name;
    }

    const auto units = node.parameters.find("units");
    if (units == node.parameters.end() || units->second.empty()) {
        return node.name;
    }
    for (const char character : units->second) {
        if (character < '0' || character > '9') {
            return node.name;
        }
    }
    return "Dense (" + units->second + ")";
}

inline void RefreshGeneratedNodeName(MLNode& node) {
    if (IsGeneratedDenseName(node.name)) {
        node.name = EffectiveNodeName(node);
    }
}

// Connection/Link types for visual differentiation
enum class LinkType {
    TensorFlow,         // Standard data flow (default)
    ResidualSkip,       // Residual/Skip connection (additive)
    DenseSkip,          // DenseNet-style skip (concatenative)
    AttentionFlow,      // Attention Q/K/V connections
    GradientFlow,       // Gradient backprop (for visualization)
    ParameterFlow,      // Parameter updates
    LossFlow            // Loss to optimizer
};

// Connection between nodes
struct NodeLink {
    int id;
    int from_node;
    int from_pin;
    int to_node;
    int to_pin;
    LinkType type = LinkType::TensorFlow;  // Connection type for visual styling
};

// Subgraph data for encapsulated node groups
struct SubgraphData {
    int subgraph_node_id;                    // ID of the parent subgraph node
    std::vector<MLNode> internal_nodes;      // Nodes inside the subgraph
    std::vector<NodeLink> internal_links;    // Links between internal nodes
    std::vector<int> input_pin_mappings;     // External input pin -> internal node pin
    std::vector<int> output_pin_mappings;    // Internal node pin -> external output pin
    bool expanded = false;                   // Whether subgraph is expanded (visible)
};

}  // namespace gui
