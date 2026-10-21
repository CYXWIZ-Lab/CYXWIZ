#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz {
struct NodeMetadata;
}

namespace cyxwiz::test {

struct NodeWorkflowEvidenceCatalog;

struct NodeContractInventorySummary {
    std::size_t node_count = 0;
    std::size_t implemented_count = 0;
    std::size_t blocked_count = 0;
    std::size_t resource_template_count = 0;
    std::size_t unassigned_owning_part_count = 0;
    std::size_t unclassified_workflow_lane_count = 0;
    std::size_t missing_representative_workflow_count = 0;
    std::size_t documented_usage_hint_count = 0;
    std::size_t workflow_manifest_artifact_count = 0;
    std::size_t authored_workflow_reference_count = 0;
    std::size_t automated_workflow_evidence_count = 0;
    std::size_t live_release_workflow_evidence_count = 0;
    std::size_t workflow_manifest_error_count = 0;
    std::size_t speculative_with_workflow_reference_count = 0;
    std::size_t speculative_without_workflow_reference_count = 0;
    std::size_t graph_import_name_count = 0;
    std::size_t legacy_compatibility_import_name_count = 0;
    std::size_t unclassified_frontend_primitive_gap_count = 0;
};

std::string BuildNodeContractInventoryJson(
    const std::vector<const NodeMetadata*>& nodes,
    const NodeWorkflowEvidenceCatalog& workflow_evidence,
    NodeContractInventorySummary& summary);

bool WriteNodeContractInventoryJson(
    const std::filesystem::path& output_path,
    const std::string& json,
    std::string& error);

} // namespace cyxwiz::test
