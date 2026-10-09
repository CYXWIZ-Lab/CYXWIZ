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
    std::vector<int> pair_loss_node_ids;
    std::vector<int> triplet_loss_node_ids;
    std::vector<int> pair_metric_node_ids;
    std::vector<int> retrieval_metric_node_ids;
    std::vector<int> embedding_output_node_ids;
    std::vector<int> pair_score_output_node_ids;

    // A node matched by name or legacy column parameters rather than by type.
    bool has_sketch_nodes = false;

    std::vector<std::string> blockers;

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

inline size_t RecordedNodeCount(const MetricLearningGraphContract& contract) {
    return contract.pair_dataset_builder_node_ids.size() + contract.triplet_dataset_builder_node_ids.size() +
           contract.pair_loss_node_ids.size() + contract.triplet_loss_node_ids.size() +
           contract.pair_metric_node_ids.size() + contract.retrieval_metric_node_ids.size() +
           contract.embedding_output_node_ids.size() + contract.pair_score_output_node_ids.size();
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

    const size_t recorded_before = RecordedNodeCount(contract);
    if (name == "PairDatasetBuilder") {
        AddNodeId(contract.pair_dataset_builder_node_ids, node_id);
    } else if (name == "TripletDatasetBuilder") {
        AddNodeId(contract.triplet_dataset_builder_node_ids, node_id);
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
    if (RecordedNodeCount(contract) != recorded_before) contract.has_sketch_nodes = true;
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
    contract.detected = contract.kind != MetricLearningGraphKind::None;
    if (!contract.detected) {
        return contract;
    }

    // Triplet training (TOFIX140 A5): the typed builder and loss train
    // through TripletBatchSampler and StackedTripletLoss.
    if (contract.kind == MetricLearningGraphKind::TripletTraining && !contract.has_sketch_nodes &&
        !contract.triplet_dataset_builder_node_ids.empty() && contract.HasTripletLoss() &&
        contract.pair_dataset_builder_node_ids.empty() && !contract.HasPairLoss() &&
        contract.pair_metric_node_ids.empty() && contract.retrieval_metric_node_ids.empty() &&
        !contract.HasInferenceOutput()) {
        contract.executable = true;
        return contract;
    }

    if (contract.HasPairLoss()) {
        if (contract.pair_dataset_builder_node_ids.empty()) {
            AddBlocker(contract,
                       "pair losses require a PairDatasetBuilder typed batch source");
        }
    }

    if (contract.HasTripletLoss()) {
        if (contract.triplet_dataset_builder_node_ids.empty()) {
            AddBlocker(contract,
                       "TripletLoss requires a TripletDatasetBuilder typed batch source");
        }
    }

    if (contract.HasPairLoss()) {
        AddBlocker(contract,
                   "visual graph executor routing for pair metric-learning losses is not implemented");
    }
    if (contract.HasInferenceOutput()) {
        AddBlocker(contract,
                   "visual graph/runtime routing for metric-learning outputs is not implemented");
    }
    AddBlocker(contract,
               "visual metric-learning graph execution is not implemented");

    contract.executable = false;
    return contract;
}

}  // namespace cyxwiz
