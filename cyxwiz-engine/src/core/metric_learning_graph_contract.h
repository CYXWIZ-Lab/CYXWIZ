#pragma once

#include "compiled_graph_plan.h"

#include <string>
#include <map>
#include <vector>

namespace cyxwiz {

enum class MetricLearningGraphKind {
    None,
    PairTraining,
    TripletTraining,
    EmbeddingExport,
    PairScoring,
};

struct MetricLearningGraphContract {
    bool detected = false;
    bool executable = false;
    MetricLearningGraphKind kind = MetricLearningGraphKind::None;

    std::vector<int> pair_dataset_builder_node_ids;
    std::vector<int> triplet_dataset_builder_node_ids;
    std::vector<int> shared_encoder_node_ids;
    std::vector<int> siamese_branch_node_ids;
    std::vector<int> pair_loss_node_ids;
    std::vector<int> triplet_loss_node_ids;
    std::vector<int> pair_metric_node_ids;
    std::vector<int> retrieval_metric_node_ids;
    std::vector<int> embedding_output_node_ids;
    std::vector<int> pair_score_output_node_ids;

    std::vector<std::string> blockers;

    bool HasSharedEncoder() const {
        return !shared_encoder_node_ids.empty();
    }

    bool HasPairLoss() const {
        return !pair_loss_node_ids.empty();
    }

    bool HasTripletLoss() const {
        return !triplet_loss_node_ids.empty();
    }

    bool HasInferenceOutput() const {
        return !embedding_output_node_ids.empty() ||
               !pair_score_output_node_ids.empty();
    }
};

inline const char* MetricLearningGraphKindName(
    MetricLearningGraphKind kind) {
    switch (kind) {
        case MetricLearningGraphKind::PairTraining:
            return "PairTraining";
        case MetricLearningGraphKind::TripletTraining:
            return "TripletTraining";
        case MetricLearningGraphKind::EmbeddingExport:
            return "EmbeddingExport";
        case MetricLearningGraphKind::PairScoring:
            return "PairScoring";
        case MetricLearningGraphKind::None:
        default:
            return "None";
    }
}

inline void AddBlocker(MetricLearningGraphContract& contract,
                       const std::string& blocker) {
    for (const auto& existing : contract.blockers) {
        if (existing == blocker) {
            return;
        }
    }
    contract.blockers.push_back(blocker);
}

inline void AddNodeId(std::vector<int>& node_ids, int node_id) {
    for (int existing : node_ids) {
        if (existing == node_id) {
            return;
        }
    }
    node_ids.push_back(node_id);
}

inline MetricLearningGraphKind InferMetricLearningGraphKind(
    const MetricLearningGraphContract& contract) {
    if (!contract.triplet_dataset_builder_node_ids.empty() ||
        !contract.triplet_loss_node_ids.empty()) {
        return MetricLearningGraphKind::TripletTraining;
    }
    if (!contract.pair_score_output_node_ids.empty()) {
        return MetricLearningGraphKind::PairScoring;
    }
    if (!contract.embedding_output_node_ids.empty() ||
        !contract.retrieval_metric_node_ids.empty()) {
        return MetricLearningGraphKind::EmbeddingExport;
    }
    if (!contract.pair_dataset_builder_node_ids.empty() ||
        !contract.pair_loss_node_ids.empty() ||
        !contract.pair_metric_node_ids.empty()) {
        return MetricLearningGraphKind::PairTraining;
    }
    return MetricLearningGraphKind::None;
}

inline void RecordMetricLearningNode(MetricLearningGraphContract& contract,
                                     gui::NodeType type,
                                     const std::string& name,
                                     const std::map<std::string, std::string>& parameters,
                                     int node_id) {
    switch (type) {
        case gui::NodeType::PairDatasetBuilder:
            AddNodeId(contract.pair_dataset_builder_node_ids, node_id);
            return;
        case gui::NodeType::TripletDatasetBuilder:
            AddNodeId(contract.triplet_dataset_builder_node_ids, node_id);
            return;
        case gui::NodeType::SharedEncoder:
            AddNodeId(contract.shared_encoder_node_ids, node_id);
            return;
        case gui::NodeType::SiameseBranch:
            AddNodeId(contract.siamese_branch_node_ids, node_id);
            return;
        case gui::NodeType::ContrastiveLoss:
        case gui::NodeType::CosineEmbeddingLoss:
            AddNodeId(contract.pair_loss_node_ids, node_id);
            return;
        case gui::NodeType::TripletLoss:
            AddNodeId(contract.triplet_loss_node_ids, node_id);
            return;
        case gui::NodeType::PairMetrics:
            AddNodeId(contract.pair_metric_node_ids, node_id);
            return;
        case gui::NodeType::RetrievalMetrics:
            AddNodeId(contract.retrieval_metric_node_ids, node_id);
            return;
        case gui::NodeType::EmbeddingOutput:
            AddNodeId(contract.embedding_output_node_ids, node_id);
            return;
        case gui::NodeType::PairScoreOutput:
            AddNodeId(contract.pair_score_output_node_ids, node_id);
            return;
        default:
            break;
    }

    if (name == "PairDatasetBuilder") {
        AddNodeId(contract.pair_dataset_builder_node_ids, node_id);
    } else if (name == "TripletDatasetBuilder") {
        AddNodeId(contract.triplet_dataset_builder_node_ids, node_id);
    } else if (name == "SharedEncoder") {
        AddNodeId(contract.shared_encoder_node_ids, node_id);
    } else if (name == "SiameseBranch") {
        AddNodeId(contract.siamese_branch_node_ids, node_id);
    } else if (name == "ContrastiveLoss" ||
               name == "CosineEmbeddingLoss") {
        AddNodeId(contract.pair_loss_node_ids, node_id);
    } else if (name == "TripletLoss") {
        AddNodeId(contract.triplet_loss_node_ids, node_id);
    } else if (name == "PairMetrics") {
        AddNodeId(contract.pair_metric_node_ids, node_id);
    } else if (name == "RetrievalMetrics") {
        AddNodeId(contract.retrieval_metric_node_ids, node_id);
    } else if (name == "EmbeddingOutput") {
        AddNodeId(contract.embedding_output_node_ids, node_id);
    } else if (name == "PairScoreOutput") {
        AddNodeId(contract.pair_score_output_node_ids, node_id);
    }

    if (parameters.count("sample_a_column") > 0 ||
        parameters.count("sample_b_column") > 0 ||
        parameters.count("pair_label_column") > 0 ||
        parameters.count("pair_id_column") > 0) {
        AddNodeId(contract.pair_dataset_builder_node_ids, node_id);
    }
    if (parameters.count("anchor_column") > 0 ||
        parameters.count("positive_column") > 0 ||
        parameters.count("negative_column") > 0 ||
        parameters.count("triplet_id_column") > 0) {
        AddNodeId(contract.triplet_dataset_builder_node_ids, node_id);
    }
    if (parameters.count("shared_encoder") > 0 ||
        parameters.count("tied_weights") > 0) {
        AddNodeId(contract.shared_encoder_node_ids, node_id);
    }
}

inline MetricLearningGraphContract AnalyzeMetricLearningGraphContract(
    const CompiledGraphPlan& plan) {
    MetricLearningGraphContract contract;
    if (!plan.available) {
        return contract;
    }

    for (const auto& node : plan.nodes) {
        RecordMetricLearningNode(
            contract, node.type, node.name, node.parameters, node.node_id);
    }

    contract.kind = InferMetricLearningGraphKind(contract);
    contract.detected = contract.kind != MetricLearningGraphKind::None ||
                        !contract.shared_encoder_node_ids.empty() ||
                        !contract.siamese_branch_node_ids.empty();
    if (!contract.detected) {
        return contract;
    }

    if (contract.shared_encoder_node_ids.size() != 1) {
        AddBlocker(contract,
                   contract.shared_encoder_node_ids.empty()
                       ? "missing SharedEncoder ownership node"
                       : "metric-learning graphs must select exactly one SharedEncoder");
    }

    if (contract.HasPairLoss()) {
        if (contract.pair_dataset_builder_node_ids.empty()) {
            AddBlocker(contract,
                       "pair losses require a PairDatasetBuilder typed batch source");
        }
        if (contract.siamese_branch_node_ids.size() < 2) {
            AddBlocker(contract,
                       "pair losses require two SiameseBranch embedding branches");
        }
    }

    if (contract.HasTripletLoss()) {
        if (contract.triplet_dataset_builder_node_ids.empty()) {
            AddBlocker(contract,
                       "TripletLoss requires a TripletDatasetBuilder typed batch source");
        }
        if (contract.siamese_branch_node_ids.size() < 3) {
            AddBlocker(contract,
                       "TripletLoss requires anchor, positive, and negative SiameseBranch embeddings");
        }
    }

    if (!contract.pair_score_output_node_ids.empty() &&
        contract.siamese_branch_node_ids.size() < 2) {
        AddBlocker(contract,
                   "PairScoreOutput requires two embedding branches or routed embeddings");
    }

    if (!contract.embedding_output_node_ids.empty() &&
        contract.siamese_branch_node_ids.empty()) {
        AddBlocker(contract,
                   "EmbeddingOutput requires a routed embedding tensor from the shared encoder");
    }

    if (contract.HasPairLoss() || contract.HasTripletLoss()) {
        AddBlocker(contract,
                   "visual graph executor routing for metric-learning losses is not implemented");
    }
    if (contract.HasInferenceOutput()) {
        AddBlocker(contract,
                   "visual graph/runtime routing for metric-learning outputs is not implemented");
    }
    AddBlocker(contract,
               "visual shared-encoder graph execution is not implemented");

    contract.executable = false;
    return contract;
}

}  // namespace cyxwiz
