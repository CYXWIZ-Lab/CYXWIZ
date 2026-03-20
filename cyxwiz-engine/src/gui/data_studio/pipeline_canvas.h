#pragma once

#include <imgui.h>
#include <imnodes.h>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace cyxwiz {

// Forward declaration
class PipelineExecutor;

/**
 * PipelineCanvas - Visual node-based data transformation pipeline editor
 *
 * Phase 1, Week 1: Core Infrastructure
 *
 * This is the main visual canvas where users drag-and-drop data transformation nodes
 * and connect them to create data pipelines (similar to KNIME, RapidMiner, Orange).
 *
 * Features:
 * - ImNodes-based visual node editor
 * - Drag-and-drop node creation from palette
 * - Connect nodes with links (data flow)
 * - Node context menu (right-click)
 * - Pipeline validation
 * - Execute pipeline button
 *
 * Architecture:
 *   PipelineCanvas (UI) -> PipelineExecutor (Engine) -> DuckDB/Arrow (Backend)
 */
class PipelineCanvas {
public:
    PipelineCanvas();
    ~PipelineCanvas();

    /**
     * Render the pipeline canvas
     * Called every frame from DataStudioPanel
     */
    void Render();

    /**
     * Get the current pipeline state for saving
     */
    std::string SerializePipeline() const;

    /**
     * Load a pipeline from JSON
     */
    bool LoadPipeline(const std::string& json);

    /**
     * Clear all nodes and links
     */
    void Clear();

    /**
     * Execute the current pipeline
     * Returns true if execution started successfully
     */
    bool ExecutePipeline();

private:
    struct Node {
        int id;
        std::string type;        // "FileInput", "Filter", "Join", etc.
        std::string name;        // User-friendly name
        ImVec2 position;
        std::map<std::string, std::string> parameters;  // Node-specific config
    };

    struct Link {
        int id;
        int start_node_id;
        int end_node_id;
        int start_attr;  // Output pin
        int end_attr;    // Input pin
    };

    // ImNodes context (separate from main node editor)
    ImNodesContext* context_;

    // Pipeline state
    std::vector<Node> nodes_;
    std::vector<Link> links_;
    int next_node_id_;
    int next_link_id_;

    // UI state
    bool show_node_palette_;
    int selected_node_id_;

    // Pipeline execution
    std::unique_ptr<PipelineExecutor> executor_;

    // Rendering helpers
    void RenderNodePalette();
    void RenderNode(const Node& node);
    void RenderContextMenu();
    void HandleNodeCreation();
    void HandleLinkCreation();
    void HandleNodeDeletion();

    // Node management
    void AddNode(const std::string& type, ImVec2 position);
    void DeleteNode(int node_id);
    void DeleteLink(int link_id);

    // Pipeline validation
    bool ValidatePipeline() const;
    bool HasCycles() const;
};

} // namespace cyxwiz
