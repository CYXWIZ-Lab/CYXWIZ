#include "pipeline_runtime_capabilities.h"

#include <algorithm>

namespace cyxwiz {

const std::vector<PipelineOperatorRuntimeCapability>&
GetPipelineOperatorRuntimeCapabilities() {
    static const std::vector<PipelineOperatorRuntimeCapability> capabilities = {
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
        {"ARIMAForecaster", gui::NodeType::ARIMAForecaster},
        {"ExponentialSmoothing", gui::NodeType::ExponentialSmoothing},
    };
    return capabilities;
}

const std::vector<PipelineFailClosedRuntimeCapability>&
GetPipelineFailClosedRuntimeCapabilities() {
    static const std::vector<PipelineFailClosedRuntimeCapability> capabilities = {
        {"PCA", "legacy PCA execution is still a passthrough placeholder"},
        {"TSNENode", "legacy t-SNE execution is still a passthrough placeholder",
         gui::NodeType::TSNENode},
        {"UMAPNode", "legacy UMAP execution is still a passthrough placeholder"},
        {"DecisionTreeClassifier", "legacy decision-tree execution is still a passthrough placeholder",
         gui::NodeType::DecisionTreeClassifier},
        {"RandomForestClassifier", "legacy random-forest execution is still a passthrough placeholder",
         gui::NodeType::RandomForestClassifier},
        {"GradientBoostingClassifier", "legacy gradient-boosting execution is still a passthrough placeholder",
         gui::NodeType::GradientBoostingClassifier},
        {"SVMClassifier", "legacy SVM execution is still a passthrough placeholder",
         gui::NodeType::SVMClassifier},
        {"KNNClassifier", "legacy KNN execution is still a passthrough placeholder",
         gui::NodeType::KNNClassifier},
        {"NaiveBayesClassifier", "legacy Naive Bayes execution is still a passthrough placeholder",
         gui::NodeType::NaiveBayesClassifier},
        {"LogisticRegressionNode", "legacy logistic-regression execution is still a passthrough placeholder",
         gui::NodeType::LogisticRegressionNode},
        {"SVMRegressor", "legacy SVM regressor execution is still a passthrough placeholder"},
        {"ConfusionMatrixNode", "confusion-matrix graph execution is not implemented in PipelineExecutor",
         gui::NodeType::ConfusionMatrixNode},
        {"ROCCurveNode", "ROC-curve graph execution is not implemented in PipelineExecutor",
         gui::NodeType::ROCCurveNode},
        {"PRCurveNode", "precision-recall curve graph execution is not implemented in PipelineExecutor"},
        {"LearningCurvesNode", "learning-curve graph execution is not implemented in PipelineExecutor",
         gui::NodeType::LearningCurvesNode},
        {"FeatureImportanceNode", "feature-importance graph execution is not implemented in PipelineExecutor",
         gui::NodeType::FeatureImportanceNode},
        {"CrossValidationNode", "cross-validation graph execution is not implemented in PipelineExecutor",
         gui::NodeType::CrossValidationNode},
        {"RegressionMetricsNode", "regression-metrics graph execution is not implemented in PipelineExecutor"},
        {"TrainTestSplit", "legacy TrainTestSplit execution is still a passthrough placeholder"},
        {"ImagePreprocessor", "legacy ImagePreprocessor execution is still a passthrough placeholder"},
        {"QualityAnalyzer", "legacy QualityAnalyzer execution is still a passthrough placeholder"},
        {"DataValidator", "legacy DataValidator execution is still a passthrough placeholder"},
        {"ImageFolderDataset", "legacy ImageFolderDataset execution creates placeholder metadata only"},
        {"MNISTDataset", "legacy MNISTDataset execution creates placeholder metadata only"},
        {"CIFAR10Dataset", "legacy CIFAR10Dataset execution creates placeholder metadata only"},
        {"HuggingFaceDataset", "legacy HuggingFaceDataset execution creates placeholder metadata only"},
        {"KaggleDataset", "legacy KaggleDataset execution creates placeholder metadata only"},
        {"AugmentationPreset", "legacy AugmentationPreset execution is still a placeholder"},
        {"GeometricTransform", "legacy GeometricTransform execution is still a placeholder"},
        {"ColorTransform", "legacy ColorTransform execution is still a placeholder"},
        {"MorphologyTransform", "legacy MorphologyTransform execution is still a placeholder"},
        {"AdvancedAugment", "legacy AdvancedAugment execution is still a placeholder"},
        {"IFFTNode", "legacy IFFT execution is still a placeholder",
         gui::NodeType::IFFTNode},
        {"WaveletTransform", "legacy WaveletTransform execution is still a placeholder",
         gui::NodeType::WaveletTransform},
        {"WordEmbeddings", "word-embedding graph execution is not implemented in PipelineExecutor"},
        {"NamedEntityRecognizer", "NER graph execution is not implemented in PipelineExecutor"},
        {"DNNModelLoad", "DNN model loading is not implemented in PipelineExecutor",
         gui::NodeType::DNNModelLoad},
        {"DNNDetect", "DNN object detection is not implemented in PipelineExecutor",
         gui::NodeType::DNNDetect},
        {"PretrainedYOLO", "pretrained YOLO execution is not implemented in PipelineExecutor",
         gui::NodeType::PretrainedYOLO},
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
        {"CellExtractor", "legacy CellExtractor execution is still a passthrough placeholder",
         gui::NodeType::CellExtractor},
        {"CellUpdater", "legacy CellUpdater execution is still a passthrough placeholder",
         gui::NodeType::CellUpdater},
        {"ColumnAppender", "legacy ColumnAppender execution is still a passthrough placeholder",
         gui::NodeType::ColumnAppender},
        {"RowAppender", "legacy RowAppender execution is still a passthrough placeholder",
         gui::NodeType::RowAppender},
        {"Unpivot", "legacy Unpivot execution is still a passthrough placeholder",
         gui::NodeType::Unpivot},
        {"TableSplitter", "legacy TableSplitter needs pin-aware multi-output routing; PipelineExecutor can only carry one dataset per node",
         gui::NodeType::TableSplitter},
        {"ExportExcel", "legacy ExportExcel execution is still a fake-success placeholder",
         gui::NodeType::ExportExcel},
        {"ExportJSON", "legacy ExportJSON execution is still a fake-success placeholder",
         gui::NodeType::ExportJSON},
        {"RuleEngine", "legacy RuleEngine execution ignores rules and is not implemented truthfully",
         gui::NodeType::RuleEngine},
    };
    return capabilities;
}

const std::vector<PipelineLegacyRuntimeCapability>&
GetPipelineLegacyRuntimeCapabilities() {
    static const std::vector<PipelineLegacyRuntimeCapability> capabilities = {
        {"FileInput"},
        {"DataInput"},
        {"DataOutput"},
        {"FilterRows"},
        {"SelectColumns"},
        {"RemoveDuplicates"},
        {"SaveDataset"},
        {"FillMissing"},
        {"SortRows"},
        {"Join"},
        {"GroupBy"},
        {"DeployToNodeEditor"},
        {"TextClean"},
        {"TextTokenize"},
        {"TextVectorize"},
        {"TSWindow"},
        {"TSFeatures"},
        {"TSLag"},
        {"TSDiff"},
        {"PolynomialFeatures"},
        {"Binning"},
        {"ExcelInput"},
        {"ExportCSV"},
        {"RowToColumnNames"},
        {"TableCropper"},
        {"StringManipulation"},
        {"MathFormula"},
        {"RenameColumns"},
    };
    return capabilities;
}

const std::vector<PipelineSourceRuntimeCapability>&
GetPipelineSourceRuntimeCapabilities() {
    static const std::vector<PipelineSourceRuntimeCapability> capabilities = {
        {"FileInput"},
        {"DataInput"},
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
        {"ExportCSV", {"file_path"}},
        {"FilterRows", {"condition"}},
        {"SelectColumns", {"columns"}},
        {"SortRows", {"columns"}},
        {"Join", {"on_column"}},
        {"GroupBy", {"group_columns", "aggregations"}},
        {"PolynomialFeatures", {"columns"}},
        {"Binning", {"columns"}},
        {"StringManipulation", {"column"}},
        {"MathFormula", {"formula"}},
    };
    return capabilities;
}

const std::vector<PipelineAllowedParameterValuesRuntimeCapability>&
GetPipelineAllowedParameterValuesRuntimeCapabilities() {
    static const std::vector<PipelineAllowedParameterValuesRuntimeCapability> capabilities = {
        {"DataInput", "source_type", "file", {"file", "folder", "ml_dataset"}},
        {"DataOutput", "format", "csv", {"csv", "parquet", "json"}},
        {"FillMissing", "strategy", "mean", {"mean", "median", "mode", "constant"}},
        {"SortRows", "order", "asc", {"asc", "desc"}},
        {"SortRows", "ascending", "true", {"true", "false"}},
        {"Join", "join_type", "inner", {"inner", "left", "right", "outer"}},
        {"Binning", "method", "equal_width", {"equal_width", "equal_freq"}},
        {"StringManipulation", "operation", "trim", {"trim", "upper", "lower", "replace", "substring"}},
    };
    return capabilities;
}

const std::vector<PipelineIntegerParameterRuntimeCapability>&
GetPipelineIntegerParameterRuntimeCapabilities() {
    static const std::vector<PipelineIntegerParameterRuntimeCapability> capabilities = {
        {"TSWindow", "window_size", 1, false},
        {"TSWindow", "stride", 1, false},
        {"TSFeatures", "rolling_window", 1, false},
        {"TSLag", "lag_periods", 1, true},
        {"TSDiff", "order", 1, false},
        {"PolynomialFeatures", "degree", 2, false},
        {"Binning", "n_bins", 1, false},
        {"RowToColumnNames", "row_index", 0, false},
        {"TableCropper", "start_row", 0, false},
        {"TableCropper", "end_row", -1, false},
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
            return support;
        };

    if (auto operator_type = ResolvePipelineOperatorRuntimeType(legacy_type_name);
        operator_type.has_value()) {
        return with_validation_axes({
            PipelineRuntimeSupportMode::OperatorBacked,
            PipelineRuntimeFailMode::Real,
            operator_type,
            nullptr,
            PipelineMaterializerStorageSupport::ArrowTableOnly,
            true,
            true});
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
            std::nullopt,
            fail_closed_it->reason,
            PipelineMaterializerStorageSupport::None,
            false,
            false};
        support.metadata_node_type = fail_closed_it->metadata_node_type;
        return with_validation_axes(std::move(support));
    }

    if (IsPipelineLegacyRuntimeNode(legacy_type_name)) {
        return with_validation_axes({
            PipelineRuntimeSupportMode::LegacyExecutor,
            PipelineRuntimeFailMode::Real,
            std::nullopt,
            nullptr,
            PipelineMaterializerStorageSupport::None,
            false,
            true});
    }

    return {};
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

    return {};
}

} // namespace cyxwiz
