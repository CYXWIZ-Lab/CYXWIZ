#pragma once

#include "node_metadata.h"
#include <unordered_map>
#include <vector>
#include <set>
#include <string>
#include <mutex>

namespace cyxwiz {

/**
 * NodeMetadataRegistry - Singleton registry for all node metadata
 * 
 * Provides node discovery, search, and usage tracking for the Node Browser Panel.
 * Supports both implemented nodes and template nodes (future/coming soon).
 */
class NodeMetadataRegistry {
public:
    /**
     * Get the singleton instance
     */
    static NodeMetadataRegistry& Instance();

    /**
     * Check if the registry has been initialized
     */
    bool IsInitialized() const { return initialized_; }

    /**
     * Initialize the registry with all node definitions
     */
    void Initialize();

    /**
     * Get metadata for a specific node type
     * @return Pointer to metadata or nullptr if not found
     */
    const NodeMetadata* GetMetadata(NodeType type) const;

    /**
     * Get all nodes in a category
     * @param category The category to query
     * @param include_templates Include template nodes (Coming Soon)
     * @return Vector of metadata pointers
     */
    std::vector<const NodeMetadata*> GetByCategory(NodeCategory category, bool include_templates = false) const;

    /**
     * Search nodes by query string (matches name, keywords, description)
     * @param query Search query (case-insensitive)
     * @param include_templates Include template nodes in results
     * @return Vector of matching metadata pointers
     */
    std::vector<const NodeMetadata*> Search(const std::string& query, bool include_templates = false) const;

    /**
     * Get list of all categories with at least one node
     * @return Vector of categories in display order
     */
    std::vector<NodeCategory> GetCategories() const;

    /**
     * Get most frequently used nodes
     * @param count Maximum number of nodes to return
     * @return Vector of metadata pointers sorted by usage
     */
    std::vector<const NodeMetadata*> GetMostUsed(size_t count) const;

    /**
     * Get recently used nodes
     * @param count Maximum number of nodes to return
     * @return Vector of metadata pointers in recent-first order
     */
    std::vector<const NodeMetadata*> GetRecent(size_t count) const;

    /**
     * Get user-favorited nodes
     * @return Vector of favorite node metadata pointers
     */
    std::vector<const NodeMetadata*> GetFavorites() const;

    /**
     * Record usage of a node (for most-used tracking)
     * @param type The node type that was used
     */
    void RecordUsage(NodeType type);

    /**
     * Add node to recent list
     * @param type The node type to add
     */
    void AddToRecent(NodeType type);

    /**
     * Toggle favorite status for a node
     * @param type The node type to toggle
     */
    void ToggleFavorite(NodeType type);

    /**
     * Load template nodes from JSON files
     * @param directory Path to template JSON files
     */
    void LoadTemplates(const std::string& directory);

    /**
     * Save user preferences (favorites, usage counts)
     */
    void SaveUserPreferences();

    /**
     * Load user preferences
     */
    void LoadUserPreferences();

    /**
     * Register a node (used during initialization)
     * @param metadata The node metadata to register
     */
    void RegisterNode(NodeMetadata metadata);

    /**
     * Get total node count
     */
    size_t GetNodeCount() const { return metadata_.size(); }

private:
    NodeMetadataRegistry() = default;
    ~NodeMetadataRegistry() = default;
    NodeMetadataRegistry(const NodeMetadataRegistry&) = delete;
    NodeMetadataRegistry& operator=(const NodeMetadataRegistry&) = delete;

    // Initialize all built-in node definitions
    void InitializeBuiltinNodes();

    // Apply runtime capability truth to browser-visible metadata.
    void ApplyRuntimeCapabilityStatus();

    // Initialize data source nodes (I/O)
    void InitializeDataSourceNodes();

    // Initialize data transform nodes (Manipulation)
    void InitializeDataTransformNodes();

    // Initialize analytics nodes
    void InitializeAnalyticsNodes();

    // Initialize ML layer nodes
    void InitializeLayerNodes();

    // Initialize training nodes (optimizers, losses, schedulers)
    void InitializeTrainingNodes();

    // Initialize activation nodes
    void InitializeActivationNodes();

    // Initialize DNN nodes
    void InitializeDNNNodes();

    // Initialize text processing nodes
    void InitializeTextNodes();

    // Initialize time series nodes
    void InitializeTimeSeriesNodes();

    // Initialize audio nodes
    void InitializeAudioNodes();

    // Initialize RL nodes
    void InitializeRLNodes();

    // Initialize export nodes
    void InitializeExportNodes();

    // Initialize KNIME-style table manipulation nodes
    void InitializeKNIMENodes();

    // Initialize utility nodes
    void InitializeUtilityNodes();

    // Tool-to-Node Migration (Phase 5)
    void InitializeLinearAlgebraNodes();
    void InitializeTimeSeriesAnalysisNodes();
    void InitializeStatisticsNodes();
    void InitializeInterpretationNodes();
    void InitializeOptimizationNodes();
    void InitializeAdditionalTextNodes();
    void InitializeVisualizationNodes();  // BarChart / Histogram / LinePlot / ...

    // Check if a node matches search query
    bool MatchesQuery(const NodeMetadata& metadata, const std::string& query) const;

    // Thread safety
    mutable std::mutex mutex_;

    // Storage
    std::unordered_map<NodeType, NodeMetadata> metadata_;
    std::vector<NodeType> recent_nodes_;  // Most recent first
    std::set<NodeType> favorites_;
    bool initialized_ = false;

    // Category display order
    static const std::vector<NodeCategory> category_order_;

    // Maximum recent nodes to track
    static constexpr size_t MAX_RECENT = 10;
};

} // namespace cyxwiz
