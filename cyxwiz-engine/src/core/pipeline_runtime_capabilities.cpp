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

} // namespace cyxwiz
