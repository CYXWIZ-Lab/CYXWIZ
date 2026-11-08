// The Nodes panel's Favourites and Recent: saved in the Engine settings and
// reloaded, and restored into the node registry (unknown types dropped,
// recent kept newest first without repeats).

#include "core/engine_config.h"
#include "core/node_metadata_registry.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using cyxwiz::NodeMetadataRegistry;
using cyxwiz::NodeType;
using cyxwiz::core::EngineConfig;
using cyxwiz::core::NodesPanelConfig;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

}  // namespace

int main() {
    // Engine settings round trip, with the count kept in range.
    const auto path = std::filesystem::temp_directory_path() /
                      ("cyxwiz_nodes_panel_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                       ".json");
    auto& config = EngineConfig::Instance();
    NodesPanelConfig expected;
    expected.favourites = {static_cast<int>(NodeType::Dense), static_cast<int>(NodeType::ReLU)};
    expected.recent = {static_cast<int>(NodeType::Conv2D), static_cast<int>(NodeType::Dense)};
    expected.show_recent = false;
    expected.recent_count = 5;
    config.SetNodesPanel(expected);
    Check(config.Save(path), "the settings save");
    config.SetNodesPanel({});
    Check(config.Load(path), "the settings load");
    Check(config.GetNodesPanel() == expected, "favourites, recent, show and count survive a save and load");

    NodesPanelConfig too_many = expected;
    too_many.recent_count = 500;
    too_many.recent.assign(30, static_cast<int>(NodeType::Dense));
    config.SetNodesPanel(too_many);
    Check(config.GetNodesPanel().recent_count == cyxwiz::core::kNodesPanelMaxRecent, "the count is capped");
    Check(config.GetNodesPanel().recent.size() == static_cast<size_t>(cyxwiz::core::kNodesPanelMaxRecent),
          "the recent list is capped");
    std::error_code error;
    std::filesystem::remove(path, error);

    // Restored into the registry.
    auto& registry = NodeMetadataRegistry::Instance();
    registry.Initialize();
    const auto unknown = static_cast<NodeType>(999999);
    registry.SetUserNodes({NodeType::Dense, unknown}, {NodeType::ReLU, NodeType::ReLU, unknown, NodeType::Dense});
    const auto favourites = registry.FavoriteTypes();
    Check(favourites.size() == 1 && favourites[0] == NodeType::Dense, "an unknown favourite is dropped");
    Check(registry.GetMetadata(NodeType::Dense)->is_favorite, "the card shows the favourite");
    const auto recent = registry.RecentTypes();
    Check(recent == std::vector<NodeType>{NodeType::ReLU, NodeType::Dense}, "recent drops repeats and unknown types");

    // Adding a node puts it first; the star toggles.
    registry.RecordUsage(NodeType::Dense);
    Check(registry.RecentTypes().front() == NodeType::Dense, "the node just added is first");
    registry.ToggleFavorite(NodeType::Dense);
    Check(registry.FavoriteTypes().empty() && !registry.GetMetadata(NodeType::Dense)->is_favorite,
          "the star removes the favourite");

    // Clear Recent keeps the favourites.
    registry.ToggleFavorite(NodeType::ReLU);
    registry.SetUserNodes(registry.FavoriteTypes(), {});
    Check(registry.RecentTypes().empty() && registry.FavoriteTypes().size() == 1, "Clear Recent keeps the favourites");

    if (g_failures) {
        std::printf("%d Nodes panel checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("Nodes panel user nodes passed\n");
    return EXIT_SUCCESS;
}
