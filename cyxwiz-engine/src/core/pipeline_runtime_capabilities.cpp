#include "pipeline_runtime_capabilities.h"

#include <algorithm>

namespace cyxwiz {

const std::vector<PipelineOperatorRuntimeCapability>&
GetPipelineOperatorRuntimeCapabilities() {
    static const std::vector<PipelineOperatorRuntimeCapability> capabilities = {
        {"Identity", gui::NodeType::Identity},
        {"TimeSeriesWindow", gui::NodeType::TimeSeriesWindow},
        {"TimeSeriesSplit", gui::NodeType::TimeSeriesSplit},
        {"TimeSeriesFeatures", gui::NodeType::TimeSeriesFeatures},
        {"LogTransform", gui::NodeType::LogTransform},
        {"Differencing", gui::NodeType::Differencing},
        {"TextTokenizer", gui::NodeType::TextTokenizer},
        {"TFIDFVectorizer", gui::NodeType::TFIDFVectorizer},
        {"CountVectorizer", gui::NodeType::CountVectorizer},
        {"SentimentAnalyzer", gui::NodeType::SentimentAnalyzer},
        {"PCANode", gui::NodeType::PCANode},
        {"KMeansCluster", gui::NodeType::KMeansCluster},
        {"DBSCANCluster", gui::NodeType::DBSCANCluster},
        {"HierarchicalCluster", gui::NodeType::HierarchicalCluster},
        {"GMMCluster", gui::NodeType::GMMCluster},
        {"FFTNode", gui::NodeType::FFTNode},
        {"Convolution1D", gui::NodeType::Convolution1D},
        {"FilterDesigner", gui::NodeType::FilterDesigner},
        {"LinearRegressionNode", gui::NodeType::LinearRegressionNode},
        {"PolynomialRegressionNode", gui::NodeType::PolynomialRegressionNode},
        {"StandardScaler", gui::NodeType::StandardScaler},
        {"MinMaxScaler", gui::NodeType::MinMaxScaler},
        {"RobustScaler", gui::NodeType::RobustScaler},
        {"LabelEncoder", gui::NodeType::LabelEncoder},
        {"OrdinalEncoder", gui::NodeType::OrdinalEncoder},
        {"TargetEncoder", gui::NodeType::TargetEncoder},
        {"OutlierDetector", gui::NodeType::OutlierDetector},
        {"TimeSeriesDecomposition", gui::NodeType::TimeSeriesDecomposition},
        {"ACFNode", gui::NodeType::ACFNode},
        {"PACFNode", gui::NodeType::PACFNode},
        {"StationarityTest", gui::NodeType::StationarityTest},
        {"SeasonalityDetector", gui::NodeType::SeasonalityDetector},
        {"ARIMAForecaster", gui::NodeType::ARIMAForecaster},
        {"ExponentialSmoothing", gui::NodeType::ExponentialSmoothing},
    };
    return capabilities;
}

const std::vector<PipelineFailClosedRuntimeCapability>&
GetPipelineFailClosedRuntimeCapabilities() {
    static const std::vector<PipelineFailClosedRuntimeCapability> capabilities = {
        {"PCA", "legacy PCA graph execution is not implemented; old passthrough behavior is disabled"},
        {"TSNENode", "legacy t-SNE graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::TSNENode},
        {"UMAPNode", "legacy UMAP graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::UMAPNode},
        {"DecisionTreeClassifier", "legacy decision-tree graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::DecisionTreeClassifier},
        {"RandomForestClassifier", "legacy random-forest graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::RandomForestClassifier},
        {"GradientBoostingClassifier", "legacy gradient-boosting graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::GradientBoostingClassifier},
        {"SVMClassifier", "legacy SVM graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::SVMClassifier},
        {"KNNClassifier", "legacy KNN graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::KNNClassifier},
        {"NaiveBayesClassifier", "legacy Naive Bayes graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::NaiveBayesClassifier},
        {"LogisticRegressionNode", "legacy logistic-regression graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::LogisticRegressionNode},
        {"SVMRegressor", "legacy SVM regressor graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::SVMRegressor},
        {"ConfusionMatrixNode", "confusion-matrix graph execution is not implemented in PipelineExecutor",
         gui::NodeType::ConfusionMatrixNode},
        {"ROCCurveNode", "ROC-curve graph execution is not implemented in PipelineExecutor",
         gui::NodeType::ROCCurveNode},
        {"PRCurveNode", "precision-recall curve graph execution is not implemented in PipelineExecutor",
         gui::NodeType::PRCurveNode},
        {"LearningCurvesNode", "learning-curve graph execution is not implemented in PipelineExecutor",
         gui::NodeType::LearningCurvesNode},
        {"FeatureImportanceNode", "feature-importance graph execution is not implemented in PipelineExecutor",
         gui::NodeType::FeatureImportanceNode},
        {"CrossValidationNode", "cross-validation graph execution is not implemented in PipelineExecutor",
         gui::NodeType::CrossValidationNode},
        {"RegressionMetricsNode", "regression-metrics graph execution is not implemented in PipelineExecutor",
         gui::NodeType::RegressionMetricsNode},
        {"TrainTestSplit", "legacy TrainTestSplit graph execution is not implemented; old passthrough behavior is disabled"},
        {"ImagePreprocessor", "legacy ImagePreprocessor graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::ImagePreprocessor},
        {"QualityAnalyzer", "legacy QualityAnalyzer graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::QualityAnalyzer},
        {"DataValidator", "legacy DataValidator graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::DataValidator},
        {"ImageFolderDataset", "legacy ImageFolderDataset graph execution is not implemented; placeholder metadata creation is disabled",
         gui::NodeType::ImageFolderDataset},
        {"MNISTDataset", "legacy MNISTDataset graph execution is not implemented; placeholder metadata creation is disabled",
         gui::NodeType::MNISTDataset},
        {"CIFAR10Dataset", "legacy CIFAR10Dataset graph execution is not implemented; placeholder metadata creation is disabled",
         gui::NodeType::CIFAR10Dataset},
        {"HuggingFaceDataset", "legacy HuggingFaceDataset graph execution is not implemented; placeholder metadata creation is disabled",
         gui::NodeType::HuggingFaceDataset},
        {"KaggleDataset", "legacy KaggleDataset graph execution is not implemented; placeholder metadata creation is disabled",
         gui::NodeType::KaggleDataset},
        {"AugmentationPreset", "legacy AugmentationPreset graph execution is not implemented",
         gui::NodeType::AugmentationPreset},
        {"GeometricTransform", "legacy GeometricTransform graph execution is not implemented",
         gui::NodeType::GeometricTransform},
        {"ColorTransform", "legacy ColorTransform graph execution is not implemented",
         gui::NodeType::ColorTransform},
        {"MorphologyTransform", "legacy MorphologyTransform graph execution is not implemented",
         gui::NodeType::MorphologyTransform},
        {"AdvancedAugment", "legacy AdvancedAugment graph execution is not implemented",
         gui::NodeType::AdvancedAugment},
        {"IFFTNode", "legacy IFFT graph execution is not implemented",
         gui::NodeType::IFFTNode},
        {"WaveletTransform", "legacy WaveletTransform graph execution is not implemented",
         gui::NodeType::WaveletTransform},
        {"WordEmbeddings", "word-embedding graph execution is not implemented in PipelineExecutor",
         gui::NodeType::WordEmbeddings},
        {"NamedEntityRecognizer", "NER graph execution is not implemented in PipelineExecutor",
         gui::NodeType::NamedEntityRecognizer},
        {"NERSequenceBuilder", "NER sequence builder is a training contract node; PipelineExecutor support is not wired yet",
         gui::NodeType::NERSequenceBuilder, std::nullopt, false},
        {"TokenVocabulary", "token vocabulary node is a sequence-training contract node; PipelineExecutor support is not wired yet",
         gui::NodeType::TokenVocabulary},
        {"POSVocabulary", "POS vocabulary node is a sequence-training contract node; PipelineExecutor support is not wired yet",
         gui::NodeType::POSVocabulary},
        {"NERTagVocabulary", "NER tag vocabulary node is a sequence-training contract node; PipelineExecutor support is not wired yet",
         gui::NodeType::NERTagVocabulary},
        {"DNNModelLoad", "DNN model loading is not implemented in PipelineExecutor",
         gui::NodeType::DNNModelLoad},
        {"DNNDetect", "DNN object detection is not implemented in PipelineExecutor",
         gui::NodeType::DNNDetect},
        {"PretrainedYOLO", "pretrained YOLO execution is not implemented in PipelineExecutor",
         gui::NodeType::PretrainedYOLO},
        {"GymEnvironment", "RL environment graph execution is not implemented in PipelineExecutor",
         gui::NodeType::GymEnvironment},
        {"ReplayBuffer", "RL replay-buffer graph execution is not implemented in PipelineExecutor",
         gui::NodeType::ReplayBufferNode},
        {"PolicyNetwork", "RL policy-network graph execution is not implemented in PipelineExecutor",
         gui::NodeType::PolicyNetwork},
        {"ValueNetwork", "RL value-network graph execution is not implemented in PipelineExecutor",
         gui::NodeType::ValueNetwork},
        {"CalculatorNode", "calculator graph execution is not implemented in PipelineExecutor",
         gui::NodeType::CalculatorNode},
        {"UnitConverter", "unit-converter graph execution is not implemented in PipelineExecutor",
         gui::NodeType::UnitConverter},
        {"RegexTester", "regex graph execution is not implemented in PipelineExecutor",
         gui::NodeType::RegexTester},
        {"JSONPathExtractor", "JSONPath graph execution is not implemented in PipelineExecutor",
         gui::NodeType::JSONPathExtractor},
        {"DataProfiler", "DataProfiler is a panel/report workflow, not a real PipelineExecutor transform",
         gui::NodeType::DataProfiler},
        {"ParquetInput", "legacy ParquetInput execution is not implemented; use DataInput with type=parquet"},
        {"CellExtractor", "legacy CellExtractor graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::CellExtractor},
        {"CellUpdater", "legacy CellUpdater graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::CellUpdater},
        {"ColumnAppender", "legacy ColumnAppender graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::ColumnAppender},
        {"RowAppender", "legacy RowAppender graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::RowAppender},
        {"Unpivot", "legacy Unpivot graph execution is not implemented; old passthrough behavior is disabled",
         gui::NodeType::Unpivot},
        {"TableSplitter", "legacy TableSplitter needs pin-aware multi-output routing; PipelineExecutor can only carry one dataset per node",
         gui::NodeType::TableSplitter},
        {"ExportExcel", "legacy ExportExcel graph execution is not implemented; fake success is disabled",
         gui::NodeType::ExportExcel},
        {"ExportJSON", "legacy ExportJSON graph execution is not implemented; fake success is disabled",
         gui::NodeType::ExportJSON},
        {"RuleEngine", "legacy RuleEngine execution ignores rules and is not implemented truthfully",
         gui::NodeType::RuleEngine},
        {"ExcelInput", "Excel input loading is not implemented; use DataInput with csv, parquet, feather, arrow, or ipc until a real Excel Arrow loader exists",
         gui::NodeType::ExcelFile},
    };
    return capabilities;
}

const std::vector<PipelineLegacyRuntimeCapability>&
GetPipelineLegacyRuntimeCapabilities() {
    static const std::vector<PipelineLegacyRuntimeCapability> capabilities = {
        {"FileInput", gui::NodeType::CSVFile},
        {"DataInput", gui::NodeType::DataInput},
        {"DataOutput", gui::NodeType::DataOutput},
        {"DataConvert", gui::NodeType::DataConvert},
        {"FilterRows", gui::NodeType::FilterRows},
        {"SelectColumns", gui::NodeType::SelectColumns},
        {"RemoveDuplicateRows", gui::NodeType::RemoveDuplicateRows},
        {"RemoveDuplicates", gui::NodeType::RemoveDuplicateRows},
        {"BinningNode", gui::NodeType::BinningNode},
        {"PolynomialFeaturesNode", gui::NodeType::PolynomialFeaturesNode},
        {"TimeSeriesLag", gui::NodeType::TimeSeriesLag},
        {"SaveDataset", std::nullopt,
         PipelineLegacyDispatchKind::SaveDataset,
         "legacy saved-pipeline graphs use this output node name; canonical "
         "metadata is DataOutput"},
        {"FillMissing", gui::NodeType::FillMissingValues},
        {"SortRows", gui::NodeType::SortRows},
        {"Join", gui::NodeType::JoinTables},
        {"GroupBy", gui::NodeType::GroupByAggregate},
        {"DeployToNodeEditor", std::nullopt,
         PipelineLegacyDispatchKind::DeployToNodeEditor,
         "legacy saved-pipeline graphs use this handoff node name; no "
         "browser-visible typed metadata exists"},
        {"TextClean", std::nullopt, PipelineLegacyDispatchKind::TextClean,
         "legacy saved-pipeline graphs use pre-operator text preprocessing "
         "names"},
        {"TextTokenize", std::nullopt,
         PipelineLegacyDispatchKind::TextTokenize,
         "legacy saved-pipeline graphs use pre-operator text preprocessing "
         "names"},
        {"TextVectorize", gui::NodeType::CountVectorizer},
        {"TSWindow", gui::NodeType::TimeSeriesWindow},
        {"TSFeatures", gui::NodeType::TimeSeriesFeatures},
        {"TSLag", gui::NodeType::TimeSeriesLag},
        {"TSDiff", gui::NodeType::Differencing},
        {"PolynomialFeatures", gui::NodeType::PolynomialFeaturesNode},
        {"Binning", gui::NodeType::BinningNode},
        {"ExportCSV", gui::NodeType::ExportCSV},
        {"RowToColumnNames", gui::NodeType::RowToColumnNames},
        {"TableCropper", gui::NodeType::TableCropper},
        {"StringManipulation", gui::NodeType::StringManipulation},
        {"MathFormula", gui::NodeType::MathFormula},
        {"RenameColumns", gui::NodeType::RenameColumns},
    };
    return capabilities;
}

const std::vector<PipelineSourceRuntimeCapability>&
GetPipelineSourceRuntimeCapabilities() {
    static const std::vector<PipelineSourceRuntimeCapability> capabilities = {
        {"FileInput"},
        {"DataInput"},
        {"DataConvert"},
        {"ExcelInput"},
        {"ImageFolderDataset"},
        {"MNISTDataset"},
        {"CIFAR10Dataset"},
        {"HuggingFaceDataset"},
        {"KaggleDataset"},
        {"ParquetInput"},
    };
    return capabilities;
}

const std::vector<PipelineInputArityRuntimeCapability>&
GetPipelineInputArityRuntimeCapabilities() {
    static const std::vector<PipelineInputArityRuntimeCapability> capabilities = {
        {"Join", 2},
    };
    return capabilities;
}

const std::vector<PipelineRequiredParameterRuntimeCapability>&
GetPipelineRequiredParameterRuntimeCapabilities() {
    static const std::vector<PipelineRequiredParameterRuntimeCapability> capabilities = {
        {"FileInput", {"path"}},
        {"ExcelInput", {"path"}},
        {"DataOutput", {"file_path"}},
        {"DataConvert", {"input_path", "output_path"}},
        {"ExportCSV", {"file_path"}},
        {"FilterRows", {"condition"}},
        {"SelectColumns", {"columns"}},
        {"SortRows", {"columns"}},
        {"Join", {"on_column"}},
        {"GroupBy", {"group_columns", "aggregations"}},
        {"TextClean", {"text_column"}},
        {"TextTokenize", {"text_column"}},
        {"TextVectorize", {"text_column"}},
        {"TSWindow", {"target_column"}},
        {"TSFeatures", {"columns"}},
        {"TimeSeriesLag", {"columns"}},
        {"TSLag", {"columns"}},
        {"TSDiff", {"columns"}},
        {"PolynomialFeaturesNode", {"columns"}},
        {"PolynomialFeatures", {"columns"}},
        {"BinningNode", {"columns"}},
        {"Binning", {"columns"}},
        {"StringManipulation", {"column"}},
        {"MathFormula", {"formula"}},
        {"RenameColumns", {"mapping"}},
        {"TimeSeriesWindow", {"value_col"}},
        {"TimeSeriesFeatures", {"value_col"}},
        {"LogTransform", {"value_col"}},
        {"Differencing", {"value_col"}},
        {"TextTokenizer", {"text_col"}},
        {"TFIDFVectorizer", {"text_col"}},
        {"CountVectorizer", {"text_col"}},
        {"SentimentAnalyzer", {"text_col"}},
        {"FFTNode", {"signal_col"}},
        {"Convolution1D", {"signal_col", "kernel"}},
        {"FilterDesigner", {"signal_col"}},
        {"LinearRegressionNode", {"feature_cols", "target_col"}},
        {"PolynomialRegressionNode", {"feature_col", "target_col"}},
        {"LabelEncoder", {"column"}},
        {"OrdinalEncoder", {"columns"}},
        {"TargetEncoder", {"columns", "target_col"}},
        {"TimeSeriesDecomposition", {"signal_col", "period"}},
        {"ACFNode", {"signal_col"}},
        {"PACFNode", {"signal_col"}},
        {"StationarityTest", {"signal_col"}},
        {"SeasonalityDetector", {"signal_col"}},
        {"ARIMAForecaster", {"signal_col"}},
        {"ExponentialSmoothing", {"signal_col"}},
    };
    return capabilities;
}

const std::vector<PipelineAllowedParameterValuesRuntimeCapability>&
GetPipelineAllowedParameterValuesRuntimeCapabilities() {
    static const std::vector<PipelineAllowedParameterValuesRuntimeCapability> capabilities = {
        {"FileInput", "format", "auto", {"auto", "csv", "parquet"}},
        {"DataInput", "source_type", "file", {"file", "folder"}},
        {"DataInput", "type", "auto", {"auto", "csv", "tsv", "parquet", "feather", "arrow", "ipc"}},
        {"DataInput", "file_type", "auto", {"auto", "csv", "tsv", "parquet", "feather", "arrow", "ipc"}},
        {"DataOutput", "format", "csv", {"csv", "parquet"}},
        {"DataOutput", "file_type", "csv", {"csv", "parquet"}},
        {"DataConvert", "input_format", "auto", {"auto", "csv", "tsv", "parquet", "feather", "arrow", "ipc"}},
        {"DataConvert", "output_format", "auto", {"auto", "csv", "tsv", "parquet", "feather", "arrow", "ipc"}},
        {"DataConvert", "compression", "snappy", {"none", "snappy", "gzip", "zstd", "brotli"}},
        {"SaveDataset", "format", "csv", {"csv", "parquet"}},
        {"SaveDataset", "file_type", "csv", {"csv", "parquet"}},
        {"FillMissing", "strategy", "mean", {"mean", "median", "mode", "constant"}},
        {"SortRows", "order", "asc", {"asc", "desc"}},
        {"SortRows", "ascending", "true", {"true", "false"}},
        {"Join", "join_type", "inner", {"inner", "left", "right", "outer"}},
        {"BinningNode", "method", "equal_width", {"equal_width", "equal_freq", "equal_frequency"}},
        {"Binning", "method", "equal_width", {"equal_width", "equal_freq", "equal_frequency"}},
        {"TextTokenize", "method", "word", {"word", "sentence", "character"}},
        {"TextTokenizer", "tokenizer_type", "1", {"0", "1", "2"}},
        {"TextVectorize", "method", "count", {"count"}},
        {"StringManipulation", "operation", "trim", {"trim", "upper", "lower", "replace", "substring"}},
        {"CountVectorizer", "norm", "l2", {"l1", "l2", "none"}},
        {"TFIDFVectorizer", "norm", "l2", {"l1", "l2", "none"}},
        {"SentimentAnalyzer", "method", "vader", {"simple", "vader", "afinn"}},
        {"KMeansCluster", "init", "kmeans++", {"random", "kmeans++"}},
        {"DBSCANCluster", "metric", "euclidean", {"euclidean", "manhattan", "cosine"}},
        {"HierarchicalCluster", "linkage", "ward", {"ward", "complete", "average", "single"}},
        {"HierarchicalCluster", "metric", "euclidean", {"euclidean", "manhattan", "cosine"}},
        {"GMMCluster", "covariance_type", "full", {"full", "tied", "diag", "spherical"}},
        {"OrdinalEncoder", "categories", "auto", {"auto"}},
        {"OutlierDetector", "method", "iqr", {"iqr", "zscore"}},
        {"FilterDesigner", "filter_type", "lowpass", {"lowpass", "highpass", "bandpass", "bandstop"}},
        {"TimeSeriesDecomposition", "method", "additive", {"additive", "multiplicative"}},
        {"TimeSeriesDecomposition", "algorithm", "classical", {"classical", "stl"}},
        {"ExponentialSmoothing", "method", "simple", {"simple", "holt", "holt_winters"}},
    };
    return capabilities;
}

const std::vector<PipelineIntegerParameterRuntimeCapability>&
GetPipelineIntegerParameterRuntimeCapabilities() {
    static const std::vector<PipelineIntegerParameterRuntimeCapability> capabilities = {
        {"TSWindow", "window_size", 1, false},
        {"TSWindow", "stride", 1, false},
        {"TSFeatures", "rolling_window", 1, false},
        {"TimeSeriesLag", "lag_periods", 1, true},
        {"TSLag", "lag_periods", 1, true},
        {"TSDiff", "order", 1, false},
        {"PolynomialFeaturesNode", "degree", 2, false},
        {"PolynomialFeatures", "degree", 2, false},
        {"BinningNode", "n_bins", 1, false},
        {"Binning", "n_bins", 1, false},
        {"RowToColumnNames", "row_index", 0, false},
        {"TableCropper", "start_row", 0, false},
        {"TableCropper", "end_row", -1, false},
        {"TimeSeriesWindow", "input_width", 1, false},
        {"TimeSeriesWindow", "shift", 1, false},
        {"TimeSeriesFeatures", "lag_values", 1, true},
        {"TimeSeriesFeatures", "rolling_windows", 1, true},
        {"Differencing", "lag", 1, false},
        {"Differencing", "order", 1, false},
        {"TextTokenizer", "max_length", 1, false},
        {"TextTokenizer", "min_word_freq", 1, false},
        {"TextTokenizer", "max_vocab_size", 1, false},
        {"TextTokenizer", "pad_value", 0, false},
        {"CountVectorizer", "max_features", 1, false},
        {"TFIDFVectorizer", "max_features", 1, false},
        {"PCANode", "n_components", 1, false},
        {"KMeansCluster", "n_clusters", 1, false},
        {"KMeansCluster", "max_iter", 1, false},
        {"KMeansCluster", "n_init", 1, false},
        {"DBSCANCluster", "min_samples", 1, false},
        {"HierarchicalCluster", "n_clusters", 1, false},
        {"GMMCluster", "n_components", 1, false},
        {"GMMCluster", "max_iter", 1, false},
        {"GMMCluster", "n_init", 1, false},
        {"FilterDesigner", "order", 1, false},
        {"PolynomialRegressionNode", "degree", 1, false},
        {"TimeSeriesDecomposition", "period", 2, false},
        {"ACFNode", "max_lag", -1, false, {0}},
        {"ACFNode", "lags", -1, false, {0}},
        {"PACFNode", "max_lag", -1, false, {0}},
        {"PACFNode", "lags", -1, false, {0}},
        {"StationarityTest", "max_lags", -1, false},
        {"SeasonalityDetector", "min_period", 2, false},
    };
    return capabilities;
}

const std::vector<PipelineFloatParameterRuntimeCapability>&
GetPipelineFloatParameterRuntimeCapabilities() {
    static const std::vector<PipelineFloatParameterRuntimeCapability> capabilities = {
        {"TimeSeriesSplit", "train_ratio", 0.0, 1.0},
        {"TimeSeriesSplit", "val_ratio", 0.0, 1.0},
        {"TimeSeriesSplit", "test_ratio", 0.0, 1.0},
        {"RobustScaler", "quantile_min", 0.0, 100.0},
        {"RobustScaler", "quantile_max", 0.0, 100.0},
        {"TargetEncoder", "smoothing", 0.0, std::nullopt},
        {"OutlierDetector", "threshold", 0.0, std::nullopt, false},
        {"DBSCANCluster", "eps", 0.0, std::nullopt, false},
        {"FFTNode", "sample_rate", 0.0, std::nullopt, false},
        {"FilterDesigner", "cutoff", 0.0, std::nullopt, false},
        {"FilterDesigner", "cutoff_high", 0.0, std::nullopt, false},
        {"FilterDesigner", "sample_rate", 0.0, std::nullopt, false},
    };
    return capabilities;
}

const std::vector<PipelineUnsupportedTrainingNodeCapability>&
GetPipelineUnsupportedSequentialModelLayerCapabilities() {
    static const std::vector<PipelineUnsupportedTrainingNodeCapability> capabilities = {
        {gui::NodeType::Conv2D,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::MaxPool2D,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::AvgPool2D,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::GlobalMaxPool,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::GlobalAvgPool,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::ConvTranspose2D,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::Upsample,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::PixelShuffle,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::PolicyNetwork,
         "sketches reinforcement-learning policy training but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::ValueNetwork,
         "sketches reinforcement-learning value training but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::LayerNorm,
         "visible in model analysis but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::GroupNorm,
         "visible in model analysis but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::InstanceNorm,
         "visible in model analysis but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::MultiHeadAttention,
         "visible in model analysis but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::SelfAttention,
         "visible in model analysis but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::CrossAttention,
         "visible in model analysis but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::LinearAttention,
         "visible in model analysis but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::RNN,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
        {gui::NodeType::Bidirectional,
         "recognized by the graph compiler but is not supported by ModelBuilder/SequentialModel yet"},
    };
    return capabilities;
}

const std::vector<PipelineUnsupportedTrainingNodeCapability>&
GetPipelineUnsupportedTrainingControlCapabilities() {
    static const std::vector<PipelineUnsupportedTrainingNodeCapability> capabilities = {
        {gui::NodeType::StepLR,
         "configurable in the editor but is not connected to training execution yet"},
        {gui::NodeType::CosineAnnealing,
         "configurable in the editor but is not connected to training execution yet"},
        {gui::NodeType::ReduceOnPlateau,
         "configurable in the editor but is not connected to training execution yet"},
        {gui::NodeType::ExponentialLR,
         "configurable in the editor but is not connected to training execution yet"},
        {gui::NodeType::WarmupScheduler,
         "configurable in the editor but is not connected to training execution yet"},
        {gui::NodeType::L1Regularization,
         "configurable in the editor but is not connected to training execution yet"},
        {gui::NodeType::L2Regularization,
         "configurable in the editor but is not connected to training execution yet"},
        {gui::NodeType::ElasticNet,
         "configurable in the editor but is not connected to training execution yet"},
    };
    return capabilities;
}

const std::vector<PipelineSupportedTrainingNodeCapability>&
GetPipelineSupportedTrainingBackendCapabilities() {
    static const std::vector<PipelineSupportedTrainingNodeCapability> capabilities = {
        {gui::NodeType::Dense,
         "compiled by GraphCompiler and executed by TrainingExecutor"},
        {gui::NodeType::Dropout,
         "compiled by GraphCompiler and executed by TrainingExecutor"},
        {gui::NodeType::BatchNorm,
         "compiled by GraphCompiler and executed by TrainingExecutor"},
        {gui::NodeType::LSTM,
         "compiled by GraphCompiler and executed by TrainingExecutor"},
        {gui::NodeType::GRU,
         "compiled by GraphCompiler and executed by TrainingExecutor"},
    };
    return capabilities;
}

const std::vector<PipelineMaterializerStorageBackendCapability>&
GetPipelineMaterializerStorageBackendCapabilities() {
    static const std::vector<PipelineMaterializerStorageBackendCapability> capabilities = {
        {PipelineStorageBackend::ArrowTable,
         PipelineMaterializerStorageSupport::ArrowTableOnly,
         true,
         "in-memory Arrow tables are the only PipelineMaterializer storage backend"},
        {PipelineStorageBackend::ParquetBacked,
         PipelineMaterializerStorageSupport::None,
         false,
         "PipelineMaterializer does not rewrite disk-backed Parquet row groups yet"},
        {PipelineStorageBackend::ImageDataset,
         PipelineMaterializerStorageSupport::None,
         false,
         "PipelineMaterializer only applies Arrow-table operators; image datasets use domain batchers"},
        {PipelineStorageBackend::AudioDataset,
         PipelineMaterializerStorageSupport::None,
         false,
         "PipelineMaterializer only applies Arrow-table operators; audio datasets use domain batchers"},
        {PipelineStorageBackend::TextDataset,
         PipelineMaterializerStorageSupport::None,
         false,
         "PipelineMaterializer only applies Arrow-table operators; text datasets use domain batchers"},
    };
    return capabilities;
}

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(const std::string& legacy_type_name) {
    const auto with_validation_axes =
        [&legacy_type_name](PipelineRuntimeSupport support) {
            support.source_node = IsPipelineSourceRuntimeNode(legacy_type_name);
            support.required_input_count =
                ResolvePipelineRequiredInputCount(legacy_type_name);
            support.required_parameters =
                ResolvePipelineRequiredParameters(legacy_type_name);
            support.allowed_parameter_values =
                ResolvePipelineAllowedParameterValues(legacy_type_name);
            support.integer_parameters =
                ResolvePipelineIntegerParameters(legacy_type_name);
            support.float_parameters =
                ResolvePipelineFloatParameters(legacy_type_name);
            return support;
        };

    if (auto operator_type = ResolvePipelineOperatorRuntimeType(legacy_type_name);
        operator_type.has_value()) {
        auto support = with_validation_axes({
            PipelineRuntimeSupportMode::OperatorBacked,
            PipelineRuntimeFailMode::Real,
            operator_type,
            operator_type,
            nullptr,
            PipelineMaterializerStorageSupport::ArrowTableOnly,
            true,
            true});
        support.implementation_owner =
            PipelineRuntimeImplementationOwner::PipelineOperatorFactory;
        return support;
    }

    const auto& fail_closed_capabilities =
        GetPipelineFailClosedRuntimeCapabilities();
    auto fail_closed_it = std::find_if(
        fail_closed_capabilities.begin(),
        fail_closed_capabilities.end(),
        [&legacy_type_name](
            const PipelineFailClosedRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    if (fail_closed_it != fail_closed_capabilities.end()) {
        auto support = PipelineRuntimeSupport{
            PipelineRuntimeSupportMode::FailClosed,
            PipelineRuntimeFailMode::HardFail,
            fail_closed_it->node_type,
            std::nullopt,
            fail_closed_it->reason,
            PipelineMaterializerStorageSupport::None,
            false,
            false};
        support.metadata_node_type = fail_closed_it->metadata_node_type;
        if (!support.metadata_node_type.has_value()) {
            support.metadata_node_type = fail_closed_it->node_type;
        }
        support.implementation_owner = PipelineRuntimeImplementationOwner::None;
        return with_validation_axes(std::move(support));
    }

    const auto& legacy_capabilities = GetPipelineLegacyRuntimeCapabilities();
    auto legacy_it = std::find_if(
        legacy_capabilities.begin(),
        legacy_capabilities.end(),
        [&legacy_type_name](const PipelineLegacyRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    if (legacy_it != legacy_capabilities.end()) {
        auto support = with_validation_axes({
            PipelineRuntimeSupportMode::LegacyExecutor,
            PipelineRuntimeFailMode::Real,
            legacy_it->node_type,
            std::nullopt,
            nullptr,
            PipelineMaterializerStorageSupport::None,
            false,
            true});
        support.implementation_owner =
            PipelineRuntimeImplementationOwner::PipelineExecutor;
        support.legacy_dispatch_kind = legacy_it->dispatch_kind;
        return support;
    }

    return {};
}

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(gui::NodeType node_type) {
    const char* legacy_type_name = ResolvePipelineRuntimeLegacyTypeName(node_type);
    if (legacy_type_name == nullptr) {
        return {};
    }
    return ResolvePipelineRuntimeSupport(legacy_type_name);
}

const char* PipelineStorageBackendName(PipelineStorageBackend backend) {
    switch (backend) {
    case PipelineStorageBackend::ArrowTable:
        return "ArrowTable";
    case PipelineStorageBackend::ParquetBacked:
        return "ParquetBacked";
    case PipelineStorageBackend::ImageDataset:
        return "ImageDataset";
    case PipelineStorageBackend::AudioDataset:
        return "AudioDataset";
    case PipelineStorageBackend::TextDataset:
        return "TextDataset";
    case PipelineStorageBackend::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

const char* PipelineRuntimeSupportModeName(PipelineRuntimeSupportMode mode) {
    switch (mode) {
    case PipelineRuntimeSupportMode::LegacyExecutor:
        return "legacy_executor";
    case PipelineRuntimeSupportMode::OperatorBacked:
        return "operator_backed";
    case PipelineRuntimeSupportMode::FailClosed:
        return "fail_closed";
    case PipelineRuntimeSupportMode::Unknown:
        return "unknown";
    }
    return "unknown";
}

const char* PipelineRuntimeFailModeName(PipelineRuntimeFailMode fail_mode) {
    switch (fail_mode) {
    case PipelineRuntimeFailMode::Real:
        return "real";
    case PipelineRuntimeFailMode::HardFail:
        return "hard_fail";
    case PipelineRuntimeFailMode::Simulated:
        return "simulated";
    case PipelineRuntimeFailMode::Passthrough:
        return "passthrough";
    case PipelineRuntimeFailMode::Unknown:
        return "unknown";
    }
    return "unknown";
}

const char* PipelineRuntimeImplementationOwnerName(
    PipelineRuntimeImplementationOwner owner) {
    switch (owner) {
    case PipelineRuntimeImplementationOwner::None:
        return "none";
    case PipelineRuntimeImplementationOwner::PipelineExecutor:
        return "pipeline_executor";
    case PipelineRuntimeImplementationOwner::PipelineOperatorFactory:
        return "pipeline_operator_factory";
    case PipelineRuntimeImplementationOwner::Unknown:
        return "unknown";
    }
    return "unknown";
}

const char* PipelineMaterializerStorageSupportName(
    PipelineMaterializerStorageSupport support) {
    switch (support) {
    case PipelineMaterializerStorageSupport::None:
        return "none";
    case PipelineMaterializerStorageSupport::ArrowTableOnly:
        return "arrow_table_only";
    }
    return "unknown";
}

const char* PipelineTrainingBackendSupportModeName(
    PipelineTrainingBackendSupportMode mode) {
    switch (mode) {
    case PipelineTrainingBackendSupportMode::Allowed:
        return "allowed";
    case PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer:
        return "unsupported_sequential_model_layer";
    case PipelineTrainingBackendSupportMode::UnsupportedTrainingControl:
        return "unsupported_training_control";
    }
    return "unknown";
}

PipelineMaterializerStorageBackendCapability
ResolvePipelineMaterializerStorageBackendSupport(PipelineStorageBackend backend) {
    const auto& capabilities = GetPipelineMaterializerStorageBackendCapabilities();
    auto it = std::find_if(capabilities.begin(), capabilities.end(),
        [backend](const PipelineMaterializerStorageBackendCapability& capability) {
            return capability.backend == backend;
        });
    if (it != capabilities.end()) {
        return *it;
    }
    return {backend,
            PipelineMaterializerStorageSupport::None,
            false,
            "storage backend is unknown to PipelineMaterializer"};
}

std::optional<gui::NodeType>
ResolvePipelineOperatorRuntimeType(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineOperatorRuntimeCapabilities();
    auto it = std::find_if(capabilities.begin(), capabilities.end(),
        [&legacy_type_name](const PipelineOperatorRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    if (it == capabilities.end()) {
        return std::nullopt;
    }
    return it->node_type;
}

bool IsPipelineOperatorRuntimeNode(const std::string& legacy_type_name) {
    return ResolvePipelineOperatorRuntimeType(legacy_type_name).has_value();
}

std::optional<gui::NodeType>
ResolvePipelineRuntimeNodeType(const std::string& legacy_type_name) {
    if (auto operator_type = ResolvePipelineOperatorRuntimeType(legacy_type_name);
        operator_type.has_value()) {
        return operator_type;
    }

    const auto& fail_closed_capabilities =
        GetPipelineFailClosedRuntimeCapabilities();
    auto fail_closed_it = std::find_if(
        fail_closed_capabilities.begin(),
        fail_closed_capabilities.end(),
        [&legacy_type_name](
            const PipelineFailClosedRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    if (fail_closed_it != fail_closed_capabilities.end()) {
        if (fail_closed_it->node_type.has_value()) {
            return fail_closed_it->node_type;
        }
        return fail_closed_it->metadata_node_type;
    }

    const auto& legacy_capabilities = GetPipelineLegacyRuntimeCapabilities();
    auto legacy_it = std::find_if(
        legacy_capabilities.begin(),
        legacy_capabilities.end(),
        [&legacy_type_name](const PipelineLegacyRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    if (legacy_it == legacy_capabilities.end()) {
        return std::nullopt;
    }
    return legacy_it->node_type;
}

const char* ResolvePipelineRuntimeLegacyTypeName(gui::NodeType node_type) {
    const auto& operator_capabilities = GetPipelineOperatorRuntimeCapabilities();
    auto operator_it = std::find_if(
        operator_capabilities.begin(),
        operator_capabilities.end(),
        [node_type](const PipelineOperatorRuntimeCapability& capability) {
            return capability.node_type == node_type;
        });
    if (operator_it != operator_capabilities.end()) {
        return operator_it->legacy_type_name;
    }

    const auto& fail_closed_capabilities =
        GetPipelineFailClosedRuntimeCapabilities();
    auto fail_closed_it = std::find_if(
        fail_closed_capabilities.begin(),
        fail_closed_capabilities.end(),
        [node_type](const PipelineFailClosedRuntimeCapability& capability) {
            return capability.node_type == node_type ||
                   capability.metadata_node_type == node_type;
        });
    if (fail_closed_it != fail_closed_capabilities.end()) {
        return fail_closed_it->legacy_type_name;
    }

    const auto& legacy_capabilities = GetPipelineLegacyRuntimeCapabilities();
    auto legacy_it = std::find_if(
        legacy_capabilities.begin(),
        legacy_capabilities.end(),
        [node_type](const PipelineLegacyRuntimeCapability& capability) {
            return capability.node_type == node_type;
        });
    if (legacy_it == legacy_capabilities.end()) {
        return nullptr;
    }
    return legacy_it->legacy_type_name;
}

const char* ResolvePipelineFailClosedReason(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineFailClosedRuntimeCapabilities();
    auto it = std::find_if(capabilities.begin(), capabilities.end(),
        [&legacy_type_name](const PipelineFailClosedRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    if (it == capabilities.end()) {
        return nullptr;
    }
    return it->reason;
}

bool IsPipelineFailClosedRuntimeNode(const std::string& legacy_type_name) {
    return ResolvePipelineFailClosedReason(legacy_type_name) != nullptr;
}

bool IsPipelineLegacyRuntimeNode(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineLegacyRuntimeCapabilities();
    auto it = std::find_if(capabilities.begin(), capabilities.end(),
        [&legacy_type_name](const PipelineLegacyRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    return it != capabilities.end();
}

bool IsPipelineSourceRuntimeNode(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineSourceRuntimeCapabilities();
    auto it = std::find_if(capabilities.begin(), capabilities.end(),
        [&legacy_type_name](const PipelineSourceRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    return it != capabilities.end();
}

std::optional<int> ResolvePipelineRequiredInputCount(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineInputArityRuntimeCapabilities();
    auto it = std::find_if(capabilities.begin(), capabilities.end(),
        [&legacy_type_name](const PipelineInputArityRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    if (it == capabilities.end()) {
        return std::nullopt;
    }
    return it->required_input_count;
}

std::vector<const char*>
ResolvePipelineRequiredParameters(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineRequiredParameterRuntimeCapabilities();
    auto it = std::find_if(capabilities.begin(), capabilities.end(),
        [&legacy_type_name](const PipelineRequiredParameterRuntimeCapability& capability) {
            return legacy_type_name == capability.legacy_type_name;
        });
    if (it == capabilities.end()) {
        return {};
    }
    return it->required_parameters;
}

std::vector<PipelineAllowedParameterValuesRuntimeCapability>
ResolvePipelineAllowedParameterValues(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineAllowedParameterValuesRuntimeCapabilities();
    std::vector<PipelineAllowedParameterValuesRuntimeCapability> result;
    for (const auto& capability : capabilities) {
        if (legacy_type_name == capability.legacy_type_name) {
            result.push_back(capability);
        }
    }
    return result;
}

std::vector<PipelineIntegerParameterRuntimeCapability>
ResolvePipelineIntegerParameters(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineIntegerParameterRuntimeCapabilities();
    std::vector<PipelineIntegerParameterRuntimeCapability> result;
    for (const auto& capability : capabilities) {
        if (legacy_type_name == capability.legacy_type_name) {
            result.push_back(capability);
        }
    }
    return result;
}

std::vector<PipelineFloatParameterRuntimeCapability>
ResolvePipelineFloatParameters(const std::string& legacy_type_name) {
    const auto& capabilities = GetPipelineFloatParameterRuntimeCapabilities();
    std::vector<PipelineFloatParameterRuntimeCapability> result;
    for (const auto& capability : capabilities) {
        if (legacy_type_name == capability.legacy_type_name) {
            result.push_back(capability);
        }
    }
    return result;
}

const char* ResolvePipelineUnsupportedSequentialModelLayerReason(gui::NodeType node_type) {
    const auto support = ResolvePipelineTrainingBackendSupport(node_type);
    return support.mode ==
               PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer
        ? support.reason
        : nullptr;
}

const char* ResolvePipelineUnsupportedTrainingControlReason(gui::NodeType node_type) {
    const auto support = ResolvePipelineTrainingBackendSupport(node_type);
    return support.mode ==
               PipelineTrainingBackendSupportMode::UnsupportedTrainingControl
        ? support.reason
        : nullptr;
}

bool IsPipelineUnsupportedSequentialModelLayer(gui::NodeType node_type) {
    return ResolvePipelineUnsupportedSequentialModelLayerReason(node_type) != nullptr;
}

bool IsPipelineUnsupportedTrainingControlNode(gui::NodeType node_type) {
    return ResolvePipelineUnsupportedTrainingControlReason(node_type) != nullptr;
}

bool IsPipelineSupportedTrainingBackendNode(gui::NodeType node_type) {
    const auto& capabilities = GetPipelineSupportedTrainingBackendCapabilities();
    auto it = std::find_if(
        capabilities.begin(),
        capabilities.end(),
        [node_type](const PipelineSupportedTrainingNodeCapability& capability) {
            return capability.node_type == node_type;
        });
    return it != capabilities.end();
}

PipelineTrainingBackendSupport
ResolvePipelineTrainingBackendSupport(gui::NodeType node_type) {
    const auto& layer_capabilities =
        GetPipelineUnsupportedSequentialModelLayerCapabilities();
    auto layer_it = std::find_if(
        layer_capabilities.begin(),
        layer_capabilities.end(),
        [node_type](const PipelineUnsupportedTrainingNodeCapability& capability) {
            return capability.node_type == node_type;
        });
    if (layer_it != layer_capabilities.end()) {
        return {PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer,
                false,
                false,
                layer_it->reason};
    }

    const auto& control_capabilities =
        GetPipelineUnsupportedTrainingControlCapabilities();
    auto control_it = std::find_if(
        control_capabilities.begin(),
        control_capabilities.end(),
        [node_type](const PipelineUnsupportedTrainingNodeCapability& capability) {
            return capability.node_type == node_type;
        });
    if (control_it != control_capabilities.end()) {
        return {PipelineTrainingBackendSupportMode::UnsupportedTrainingControl,
                false,
                false,
                control_it->reason};
    }

    const auto& supported_capabilities =
        GetPipelineSupportedTrainingBackendCapabilities();
    auto supported_it = std::find_if(
        supported_capabilities.begin(),
        supported_capabilities.end(),
        [node_type](const PipelineSupportedTrainingNodeCapability& capability) {
            return capability.node_type == node_type;
        });
    if (supported_it != supported_capabilities.end()) {
        return {PipelineTrainingBackendSupportMode::Allowed,
                true,
                true,
                supported_it->reason};
    }

    return {};
}

} // namespace cyxwiz
