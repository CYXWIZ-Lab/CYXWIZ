#pragma once

#include "pipeline_runtime_capabilities.h"

namespace cyxwiz {
struct NodeMetadata;

enum class DataStudioActionState { Unavailable, ExploreOnly, RuntimeBacked };

struct DataStudioCapabilityRequest {
    gui::NodeType node_type = gui::NodeType::Unknown;
    std::map<std::string, std::string> parameters;
    std::vector<PipelineStorageBackend> input_storage;
};

struct DataStudioCapability {
    DataStudioActionState state = DataStudioActionState::Unavailable;
    std::string display_name;
    std::string reason;
};

// Read-only structural view of existing owners; no new operation registry.
// RuntimeBacked means this operation passes capability/configuration checks.
// It does NOT qualify schema, resource limits, a whole recipe, or publication.
// ExploreOnly currently describes the existing QueryEditor consumer only.
DataStudioCapability ResolveDataStudioCapability(
    const DataStudioCapabilityRequest& request, const NodeMetadata* metadata);
const char* DataStudioActionStateName(DataStudioActionState state);
} // namespace cyxwiz
