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
     * Save pipeline to a file (Phase 7)
     * @param filepath Path to save the pipeline JSON file
     * @return true if save succeeded
     */
    bool SavePipelineToFile(const std::string& filepath);

    /**
     * Load pipeline from a file (Phase 7)
     * @param filepath Path to the pipeline JSON file
     * @return true if load succeeded
     */
    bool LoadPipelineFromFile(const std::string& filepath);

    /**
     * Set pipeline name (for UI and serialization)
     */
    void SetPipelineName(const std::string& name) { pipeline_name_ = name; }

    /**
     * Get pipeline name
     */
    const std::string& GetPipelineName() const { return pipeline_name_; }

    /**
     * Clear all nodes and links
     */
    void Clear();

    /**
     * Execute the current pipeline
     * Returns true if execution started successfully
     */
    bool ExecutePipeline();

    /**
     * Check if deployment is ready (Phase 5 Week 7)
     */
    bool IsDeploymentReady() const;

    /**
     * Get the deployment dataset name (Phase 5 Week 7)
     */
    std::string GetDeploymentDataset() const;

    /**
     * Clear deployment status after successful deployment (Phase 5 Week 7)
     */
    void ClearDeploymentStatus();

    /**
     * Check if deployment was requested by user click (Phase 5 Week 7)
     */
    bool IsDeploymentRequested() const { return deployment_requested_; }

    /**
     * Clear deployment request flag (Phase 5 Week 7)
     */
    void ClearDeploymentRequest() { deployment_requested_ = false; }

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
    std::string pipeline_name_;  // Phase 7: Pipeline name

    // UI state
    bool show_node_palette_;
    int selected_node_id_;

    // Deployment state (Phase 5 Week 7)
    bool deployment_requested_;

    // Pipeline execution
    std::unique_ptr<PipelineExecutor> executor_;

    // Rendering helpers
    void RenderNodePalette();
    void RenderNode(const Node& node);
    void RenderToolbar();  // Phase 7: Toolbar with Save/Load buttons
    void RenderContextMenu();
    void HandleNodeCreation();
    void HandleLinkCreation();
    void HandleNodeDeletion();

    // Phase 7: Tooltip helper
    std::string GetNodeTooltip(const std::string& node_type) const;

    // Node management
    void AddNode(const std::string& type, ImVec2 position);
    void DeleteNode(int node_id);
    void DeleteLink(int link_id);

    // Pipeline validation
    bool ValidatePipeline() const;
    bool HasCycles() const;
};

} // namespace cyxwiz
