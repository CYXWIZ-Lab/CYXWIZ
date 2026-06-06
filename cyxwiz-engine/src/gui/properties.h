#pragma once

#include <string>
#include <map>
#include <vector>
#include <deque>
#include <functional>
#include <memory>
#include "../core/node_metadata.h"
#include "../core/node_executors/node_executor_factory.h"
#include "node_config_dialog.h"
#include "properties_shape_info.h"

namespace gui {

// Forward declarations
enum class NodeType;
struct MLNode;
struct NodeLink;
class NodeEditor;

using LayerParameters = properties_shape::LayerParameters;
using NodeShapeInfo = properties_shape::NodeShapeInfo;

class Properties {
public:
    Properties();
    ~Properties();

    void Render();

    // Set the currently selected node to display properties for
    void SetSelectedNode(MLNode* node);
    void ClearSelection();

    // Set the node editor reference for graph access
    void SetNodeEditor(NodeEditor* editor) { node_editor_ = editor; }

    // Visibility control for sidebar integration
    bool* GetVisiblePtr() { return &show_window_; }

    // Force recomputation of shapes (call when graph changes)
    void InvalidateShapes() { shapes_valid_ = false; }

private:
    void RenderNodeProperties(MLNode& node);
    void RenderShapeInfo(const NodeShapeInfo& shape_info);

    // Enhanced property sections (Phase 3)
    void RenderGeneralSection(MLNode& node);
    void RenderParametersSection(MLNode& node, const cyxwiz::NodeMetadata* metadata);
    void RenderAdvancedSection(MLNode& node);
    void RenderPresetsSection(MLNode& node);

    // Node executor integration (Phase: Node Executor Architecture)
    void RenderExecutorSection(MLNode& node);
    bool HasNodeExecutor(NodeType type);
    void SetupExecutorInputData(cyxwiz::INodeExecutor* executor, MLNode& node);

    // Metadata-driven parameter rendering
    void RenderParameter(MLNode& node, const cyxwiz::ParameterDefinition& param);

    // Preset management
    void SavePreset(const MLNode& node, const std::string& name);
    void LoadPreset(MLNode& node, const std::string& name);
    std::vector<std::string> GetPresetsForNodeType(NodeType type);

    // Shape inference methods
    NodeShapeInfo ComputeNodeShape(int node_id);
    std::vector<size_t> GetInputShapeFromDataset();
    std::vector<size_t> InferOutputShape(
        NodeType type,
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& params);
    LayerParameters ComputeLayerParameters(
        NodeType type,
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& params);

    // Helper to format shape as string
    std::string FormatShape(const std::vector<size_t>& shape);
    std::string FormatShapeMatrix(const std::vector<size_t>& shape, size_t batch_size);
    size_t GetBatchSize();

    bool show_window_;
    MLNode* selected_node_ = nullptr;
    NodeEditor* node_editor_ = nullptr;

    // Shape caching
    bool shapes_valid_ = false;
    std::map<int, NodeShapeInfo> cached_shapes_;

    // Scope data buffers (node_id → time/value ring buffer)
    struct ScopeBuffer {
        std::deque<float> times;
        std::deque<float> values;
        int max_samples = 500;
        void Push(float t, float v) {
            times.push_back(t);
            values.push_back(v);
            while (static_cast<int>(times.size()) > max_samples) {
                times.pop_front();
                values.pop_front();
            }
        }
        void Clear() { times.clear(); values.clear(); }
    };
    std::map<int, ScopeBuffer> scope_buffers_;
    float scope_demo_time_ = 0.0f;  // Demo animation timer

    // Phase 3: Enhanced properties state
    std::map<std::string, std::string> validation_errors_;  // param_name -> error message
    bool section_general_open_ = true;
    bool section_parameters_open_ = true;
    bool section_advanced_open_ = false;
    bool section_presets_open_ = false;
    char preset_name_buffer_[64] = {};

    // KNIME-style configuration dialogs
    std::unique_ptr<NodeConfigDialog> active_dialog_;
    void RenderOpenDialogButton(MLNode& node);
};

} // namespace gui
