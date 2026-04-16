#include "pipeline_operator_factory.h"
#include "differencing_operator.h"
#include "identity_operator.h"
#include "log_transform_operator.h"
#include "text_tokenizer_operator.h"
#include "tfidf_vectorizer_operator.h"
#include "time_series_features_operator.h"
#include "time_series_split_operator.h"
#include "time_series_window_operator.h"

namespace cyxwiz {

PipelineOperatorFactory& PipelineOperatorFactory::Instance() {
    static PipelineOperatorFactory instance;
    return instance;
}

PipelineOperatorFactory::PipelineOperatorFactory() {
    RegisterDefaults();
}

void PipelineOperatorFactory::RegisterDefaults() {
    RegisterCreator(gui::NodeType::Identity, []() {
        return std::make_unique<IdentityOperator>();
    });
    // Phase 4 Time-Series operators.
    RegisterCreator(gui::NodeType::TimeSeriesWindow, []() {
        return std::make_unique<TimeSeriesWindowOperator>();
    });
    RegisterCreator(gui::NodeType::TimeSeriesSplit, []() {
        return std::make_unique<TimeSeriesSplitOperator>();
    });
    RegisterCreator(gui::NodeType::LogTransform, []() {
        return std::make_unique<LogTransformOperator>();
    });
    RegisterCreator(gui::NodeType::Differencing, []() {
        return std::make_unique<DifferencingOperator>();
    });
    RegisterCreator(gui::NodeType::TimeSeriesFeatures, []() {
        return std::make_unique<TimeSeriesFeaturesOperator>();
    });
    // Fix B (Phase 3 cleanup): TextTokenizer as a real Cat-1 operator.
    // Inert for legacy text graphs that go through RegisterTextDataset
    // (PipelineMaterializer skips non-Arrow datasets); active when
    // DataInput is wired to register text CSVs as Arrow tables.
    RegisterCreator(gui::NodeType::TextTokenizer, []() {
        return std::make_unique<TextTokenizerOperator>();
    });
    // Tool-to-Node text analytics: TFIDFVectorizer.
    RegisterCreator(gui::NodeType::TFIDFVectorizer, []() {
        return std::make_unique<TFIDFVectorizerOperator>();
    });
}

std::unique_ptr<IPipelineOperator> PipelineOperatorFactory::Create(
    gui::NodeType type) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) {
        return nullptr;
    }
    return it->second();
}

void PipelineOperatorFactory::RegisterCreator(gui::NodeType type, Creator creator) {
    creators_[type] = std::move(creator);
}

bool PipelineOperatorFactory::HasOperator(gui::NodeType type) const {
    return creators_.find(type) != creators_.end();
}

std::vector<gui::NodeType> PipelineOperatorFactory::GetSupportedTypes() const {
    std::vector<gui::NodeType> out;
    out.reserve(creators_.size());
    for (const auto& kv : creators_) {
        out.push_back(kv.first);
    }
    return out;
}

} // namespace cyxwiz
