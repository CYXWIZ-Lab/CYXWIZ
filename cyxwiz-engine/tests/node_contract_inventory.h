#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz {
struct NodeMetadata;
}

namespace cyxwiz::test {

struct NodeContractInventorySummary {
    std::size_t node_count = 0;
    std::size_t implemented_count = 0;
    std::size_t blocked_count = 0;
    std::size_t resource_template_count = 0;
    std::size_t unassigned_owning_part_count = 0;
    std::size_t unclassified_workflow_lane_count = 0;
    std::size_t missing_representative_workflow_count = 0;
    std::size_t graph_import_name_count = 0;
    std::size_t legacy_compatibility_import_name_count = 0;
    std::size_t unclassified_frontend_primitive_gap_count = 0;
};

std::string BuildNodeContractInventoryJson(
    const std::vector<const NodeMetadata*>& nodes,
    NodeContractInventorySummary& summary);

bool WriteNodeContractInventoryJson(
    const std::filesystem::path& output_path,
    const std::string& json,
    std::string& error);

} // namespace cyxwiz::test
