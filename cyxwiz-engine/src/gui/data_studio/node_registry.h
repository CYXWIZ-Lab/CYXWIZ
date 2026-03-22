#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace cyxwiz {

/**
 * NodeRegistry - Catalog of available pipeline node types
 *
 * Phase 1, Week 1: Core Infrastructure
 *
 * This component maintains a registry of all available node types that
 * can be used in the data transformation pipeline. Each node type has:
 * - Type identifier (e.g., "FilterRows")
 * - Display name (e.g., "Filter Rows")
 * - Category (e.g., "Transformations", "Data Sources")
 * - Parameter schema (what inputs the node accepts)
 * - Input/Output port configuration
 *
 * Node Categories:
 * - Data Sources: FileInput, DatabaseInput, APIInput
 * - Transformations: FilterRows, SelectColumns, Join, Aggregate
 * - Data Quality: RemoveDuplicates, FillMissing, DetectOutliers
 * - Output: SaveDataset, ExportCSV, ExportParquet
 *
 * Architecture:
 *   NodeRegistry (Catalog) <- PipelineCanvas (UI) <- User
 */
class NodeRegistry {
public:
    /**
     * Get the singleton instance
     */
    static NodeRegistry& Instance();

    /**
     * Node type definition
     */
    struct NodeType {
        std::string type_id;           // Unique identifier
        std::string display_name;      // User-friendly name
        std::string category;          // Category for grouping
        std::string description;       // Short description
        std::vector<std::string> input_ports;   // Input port names
        std::vector<std::string> output_ports;  // Output port names
        std::map<std::string, std::string> parameters;  // Parameter name -> type
    };

    /**
     * Register a new node type
     */
    void RegisterNodeType(const NodeType& node_type);

    /**
     * Get all registered node types
     */
    const std::vector<NodeType>& GetAllNodeTypes() const;

    /**
     * Get node types by category
     */
    std::vector<NodeType> GetNodeTypesByCategory(const std::string& category) const;

    /**
     * Get all categories
     */
    std::vector<std::string> GetCategories() const;

    /**
     * Find a node type by type_id
     */
    const NodeType* FindNodeType(const std::string& type_id) const;

    /**
     * Check if a node type exists
     */
    bool HasNodeType(const std::string& type_id) const;

private:
    NodeRegistry();
    ~NodeRegistry() = default;

    // Prevent copying
    NodeRegistry(const NodeRegistry&) = delete;
    NodeRegistry& operator=(const NodeRegistry&) = delete;

    std::vector<NodeType> node_types_;

    /**
     * Register built-in node types
     */
    void RegisterBuiltInNodes();
};

} // namespace cyxwiz
