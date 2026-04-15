#include "pipeline_operator_factory.h"
#include "identity_operator.h"

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
