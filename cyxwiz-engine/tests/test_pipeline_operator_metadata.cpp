#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_metadata_registry.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::string TypeId(gui::NodeType type) {
    return std::to_string(static_cast<int>(type));
}

bool HasInput(const cyxwiz::NodeMetadata* meta,
              const std::string& name,
              bool required) {
    if (!meta) return false;
    for (const auto& input : meta->inputs) {
        if (input.name == name && input.required == required) {
            return true;
        }
    }
    return false;
}

bool HasInputType(const cyxwiz::NodeMetadata* meta,
                  const std::string& name,
                  gui::PinType type) {
    if (!meta) return false;
    for (const auto& input : meta->inputs) {
        if (input.name == name && input.type == type) {
            return true;
        }
    }
    return false;
}

bool HasOutputType(const cyxwiz::NodeMetadata* meta,
                   const std::string& name,
                   gui::PinType type) {
    if (!meta) return false;
    for (const auto& output : meta->outputs) {
        if (output.name == name && output.type == type) {
            return true;
        }
    }
    return false;
}

bool HasParameter(const cyxwiz::NodeMetadata* meta,
                  const std::string& name) {
    if (!meta) return false;
    for (const auto& param : meta->parameters) {
        if (param.name == name) {
            return true;
        }
    }
    return false;
}

bool HasEnumValue(const cyxwiz::NodeMetadata* meta,
                  const std::string& param_name,
                  const std::string& enum_value) {
    if (!meta) return false;
    for (const auto& param : meta->parameters) {
        if (param.name != param_name) continue;
        for (const auto& value : param.enum_values) {
            if (value == enum_value) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

int main() {
    auto& metadata = cyxwiz::NodeMetadataRegistry::Instance();
    metadata.Initialize();

    auto& factory = cyxwiz::PipelineOperatorFactory::Instance();
    const auto supported = factory.GetSupportedTypes();
    Check(!supported.empty(), "PipelineOperatorFactory should register operators");

    for (auto type : supported) {
        auto op = factory.Create(type);
        Check(op != nullptr, "factory returned null for type " + TypeId(type));

        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr, "missing metadata for factory type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "factory type " + TypeId(type) + " is not marked implemented");
        Check(meta->category != gui::NodeCategory::Unknown,
              "factory type " + TypeId(type) + " has unknown category");
    }

    const std::vector<gui::NodeType> supported_model_nodes = {
        gui::NodeType::LSTM,
        gui::NodeType::GRU,
    };
    for (auto type : supported_model_nodes) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr, "missing supported model metadata for type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "supported model type " + TypeId(type) + " should be marked implemented");
    }

    const std::vector<gui::NodeType> fail_closed_nodes = {
        gui::NodeType::TSNENode,
        gui::NodeType::DecisionTreeClassifier,
        gui::NodeType::RandomForestClassifier,
        gui::NodeType::GradientBoostingClassifier,
        gui::NodeType::SVMClassifier,
        gui::NodeType::KNNClassifier,
        gui::NodeType::NaiveBayesClassifier,
        gui::NodeType::LogisticRegressionNode,
        gui::NodeType::ConfusionMatrixNode,
        gui::NodeType::ROCCurveNode,
        gui::NodeType::LearningCurvesNode,
        gui::NodeType::CrossValidationNode,
        gui::NodeType::FeatureImportanceNode,
        gui::NodeType::DataProfiler,
        gui::NodeType::Conv2D,
        gui::NodeType::MaxPool2D,
        gui::NodeType::AvgPool2D,
        gui::NodeType::GlobalMaxPool,
        gui::NodeType::GlobalAvgPool,
        gui::NodeType::ConvTranspose2D,
        gui::NodeType::Upsample,
        gui::NodeType::PixelShuffle,
        gui::NodeType::DNNModelLoad,
        gui::NodeType::DNNDetect,
        gui::NodeType::PretrainedYOLO,
        gui::NodeType::IFFTNode,
        gui::NodeType::WaveletTransform,
        gui::NodeType::CalculatorNode,
        gui::NodeType::UnitConverter,
        gui::NodeType::RegexTester,
        gui::NodeType::JSONPathExtractor,
    };
    for (auto type : fail_closed_nodes) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr, "missing fail-closed metadata for type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              "fail-closed type " + TypeId(type) + " should not be marked implemented");
    }

    const auto* compare = metadata.GetMetadata(gui::NodeType::TensorCompare);
    Check(compare != nullptr, "missing TensorCompare metadata");
    Check(HasInput(compare, "A", true),
          "TensorCompare should expose required A input");
    Check(HasInput(compare, "B", false),
          "TensorCompare should expose optional B input");
    Check(HasEnumValue(compare, "op", "=="),
          "TensorCompare should expose tensor compare operators");

    const auto* logical = metadata.GetMetadata(gui::NodeType::TensorLogicalMask);
    Check(logical != nullptr, "missing TensorLogicalMask metadata");
    Check(HasInput(logical, "A", true),
          "TensorLogicalMask should expose required A input");
    Check(HasInput(logical, "B", false),
          "TensorLogicalMask should expose optional B input");
    Check(HasEnumValue(logical, "op", "not"),
          "TensorLogicalMask should keep unary not");
    Check(HasEnumValue(logical, "op", "and") &&
          HasEnumValue(logical, "op", "or"),
          "TensorLogicalMask should expose binary and/or");

    const auto* ts_window = metadata.GetMetadata(gui::NodeType::TimeSeriesWindow);
    Check(ts_window != nullptr, "missing TimeSeriesWindow metadata");
    Check(HasInputType(ts_window, "Data", gui::PinType::Dataset),
          "TimeSeriesWindow should expose Dataset input");
    Check(HasOutputType(ts_window, "Windowed", gui::PinType::Dataset),
          "TimeSeriesWindow should expose one windowed Dataset output");
    Check(HasParameter(ts_window, "value_col") &&
          HasParameter(ts_window, "feature_cols") &&
          HasParameter(ts_window, "time_col") &&
          HasParameter(ts_window, "input_width") &&
          HasParameter(ts_window, "label_width") &&
          HasParameter(ts_window, "shift"),
          "TimeSeriesWindow should expose canonical operator parameters");

    const auto* ts_features = metadata.GetMetadata(gui::NodeType::TimeSeriesFeatures);
    Check(ts_features != nullptr, "missing TimeSeriesFeatures metadata");
    Check(HasInputType(ts_features, "Data", gui::PinType::Dataset),
          "TimeSeriesFeatures should expose Dataset input");
    Check(HasOutputType(ts_features, "Enriched", gui::PinType::Dataset),
          "TimeSeriesFeatures should expose one enriched Dataset output");
    Check(HasParameter(ts_features, "value_col") &&
          HasParameter(ts_features, "lag_values") &&
          HasParameter(ts_features, "rolling_windows") &&
          HasParameter(ts_features, "rolling_aggregations"),
          "TimeSeriesFeatures should expose canonical operator parameters");

    const auto* ts_split = metadata.GetMetadata(gui::NodeType::TimeSeriesSplit);
    Check(ts_split != nullptr, "missing TimeSeriesSplit metadata");
    Check(HasInputType(ts_split, "Data", gui::PinType::Dataset),
          "TimeSeriesSplit should expose Dataset input");
    Check(HasOutputType(ts_split, "Partitioned", gui::PinType::Dataset),
          "TimeSeriesSplit should expose one partitioned Dataset output");
    Check(HasParameter(ts_split, "train_ratio") &&
          HasParameter(ts_split, "val_ratio") &&
          HasParameter(ts_split, "test_ratio"),
          "TimeSeriesSplit should expose train/val/test ratio parameters");

    std::cout << "Pipeline operator metadata drift guard passed\n";
    return 0;
}
