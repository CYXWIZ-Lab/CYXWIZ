#include "node_editor.h"
#include "panels/script_editor.h"
#include "properties.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <map>
#include <nlohmann/json.hpp>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace gui {

NodeEditor::NodeEditor()
    : show_window_(true),
      next_node_id_(1),
      next_pin_id_(1),
      next_link_id_(1),
      show_context_menu_(false),
      context_menu_node_id_(-1),
      selected_node_id_(-1),
      selected_framework_(CodeFramework::PyTorch),
      editor_context_(nullptr),
      script_editor_(nullptr),
      properties_panel_(nullptr) {

    // Create ImNodes editor context
    editor_context_ = ImNodes::EditorContextCreate();
    ImNodes::EditorContextSet(editor_context_);

    // Configure ImNodes
    ImNodesIO& io = ImNodes::GetIO();
    io.LinkDetachWithModifierClick.Modifier = &ImGui::GetIO().KeyCtrl;
    io.MultipleSelectModifier.Modifier = &ImGui::GetIO().KeyCtrl;

    // Enable left-click drag for panning (always enabled, no modifier needed)
    static bool always_true = true;
    io.EmulateThreeButtonMouse.Modifier = &always_true;

    ImNodesStyle& style = ImNodes::GetStyle();
    style.Flags |= ImNodesStyleFlags_GridLines;

    // Create a comprehensive sample neural network graph
    // This demonstrates a complete MNIST-style classifier

    // Layer 1: Input layer
    MLNode input = CreateNode(NodeType::Input, "Input Layer");
    nodes_.push_back(input);
    ImNodes::SetNodeGridSpacePos(input.id, ImVec2(100.0f, 100.0f));

    // Layer 2: First dense layer
    MLNode dense1 = CreateNode(NodeType::Dense, "Dense (128)");
    nodes_.push_back(dense1);
    ImNodes::SetNodeGridSpacePos(dense1.id, ImVec2(300.0f, 100.0f));

    // Layer 3: ReLU activation
    MLNode relu1 = CreateNode(NodeType::ReLU, "ReLU");
    nodes_.push_back(relu1);
    ImNodes::SetNodeGridSpacePos(relu1.id, ImVec2(500.0f, 100.0f));

    // Layer 4: Dropout for regularization
    MLNode dropout1 = CreateNode(NodeType::Dropout, "Dropout (0.2)");
    nodes_.push_back(dropout1);
    ImNodes::SetNodeGridSpacePos(dropout1.id, ImVec2(700.0f, 100.0f));

    // Layer 5: Second dense layer
    MLNode dense2 = CreateNode(NodeType::Dense, "Dense (64)");
    nodes_.push_back(dense2);
    ImNodes::SetNodeGridSpacePos(dense2.id, ImVec2(300.0f, 300.0f));

    // Layer 6: Second ReLU activation
    MLNode relu2 = CreateNode(NodeType::ReLU, "ReLU");
    nodes_.push_back(relu2);
    ImNodes::SetNodeGridSpacePos(relu2.id, ImVec2(500.0f, 300.0f));

    // Layer 7: Output layer
    MLNode output = CreateNode(NodeType::Output, "Output (10)");
    nodes_.push_back(output);
    ImNodes::SetNodeGridSpacePos(output.id, ImVec2(700.0f, 300.0f));

    // Create connections between layers
    // Input -> Dense1
    CreateLink(input.outputs[0].id, dense1.inputs[0].id, input.id, dense1.id);

    // Dense1 -> ReLU1
    CreateLink(dense1.outputs[0].id, relu1.inputs[0].id, dense1.id, relu1.id);

    // ReLU1 -> Dropout1
    CreateLink(relu1.outputs[0].id, dropout1.inputs[0].id, relu1.id, dropout1.id);

    // Dropout1 -> Dense2
    CreateLink(dropout1.outputs[0].id, dense2.inputs[0].id, dropout1.id, dense2.id);

    // Dense2 -> ReLU2
    CreateLink(dense2.outputs[0].id, relu2.inputs[0].id, dense2.id, relu2.id);

    // ReLU2 -> Output
    CreateLink(relu2.outputs[0].id, output.inputs[0].id, relu2.id, output.id);

    spdlog::info("Created sample neural network graph with {} nodes and {} connections",
                 nodes_.size(), links_.size());
}

NodeEditor::~NodeEditor() {
    if (editor_context_) {
        ImNodes::EditorContextFree(editor_context_);
    }
}

void NodeEditor::Render() {
    if (!show_window_) return;

    // Set the editor context for this node editor instance
    ImNodes::EditorContextSet(editor_context_);

    if (ImGui::Begin("Node Editor", &show_window_)) {
        ShowToolbar();

        ImGui::Separator();

        ImNodes::BeginNodeEditor();

        RenderNodes();

        // Handle mouse wheel zoom
        if (ImGui::IsWindowHovered()) {
            float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f) {
                // Zoom by adjusting the panning offset
                ImVec2 panning = ImNodes::EditorContextGetPanning();
                float zoom_delta = wheel * 50.0f;  // Zoom speed

                ImNodes::EditorContextResetPanning(ImVec2(
                    panning.x + zoom_delta,
                    panning.y + zoom_delta
                ));
            }
        }

        // Handle right-click context menu
        if (ImNodes::IsEditorHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            // Store mouse position for node placement
            // Convert from screen space to grid space manually
            ImVec2 mouse_pos = ImGui::GetMousePos();
            ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
            ImVec2 panning = ImNodes::EditorContextGetPanning();

            context_menu_pos_ = ImVec2(
                mouse_pos.x - canvas_pos.x - panning.x,
                mouse_pos.y - canvas_pos.y - panning.y
            );

            ImGui::OpenPopup("NodeContextMenu");
        }

        if (ImGui::BeginPopup("NodeContextMenu")) {
            ShowContextMenu();
            ImGui::EndPopup();
        }

        ImNodes::EndNodeEditor();

        // Process any pending node additions (deferred to avoid modifying nodes_ during ImNodes rendering)
        for (const auto& pending : pending_nodes_) {
            MLNode node = CreateNode(pending.type, pending.name);
            spdlog::info("Creating deferred node: type={}, name={}, id={} at grid position x={} y={}",
                        static_cast<int>(pending.type), pending.name, node.id,
                        pending.position.x, pending.position.y);

            nodes_.push_back(node);

            // Set node position to where the context menu was opened
            ImNodes::SetNodeGridSpacePos(node.id, pending.position);

            spdlog::info("Added node: {} (ID: {}) at position x={} y={}",
                        pending.name, node.id, pending.position.x, pending.position.y);
        }
        pending_nodes_.clear();

        // Handle interactions AFTER EndNodeEditor() - this is when ImNodes processes them
        HandleInteractions();

        // Update properties panel with selected node
        const int num_selected = ImNodes::NumSelectedNodes();

        if (properties_panel_) {
            if (num_selected == 1) {
                int selected_nodes[1];
                ImNodes::GetSelectedNodes(selected_nodes);
                int new_selected_id = selected_nodes[0];

                // Only log if selection changed
                if (new_selected_id != selected_node_id_) {
                    spdlog::info("Node selection changed to ID: {}", new_selected_id);
                    selected_node_id_ = new_selected_id;

                    // Find the selected node and pass it to the properties panel
                    MLNode* selected = nullptr;
                    for (auto& node : nodes_) {
                        if (node.id == selected_node_id_) {
                            selected = &node;
                            spdlog::info("Found selected node: id={}, type={}, name={}",
                                         node.id, static_cast<int>(node.type), node.name);
                            break;
                        }
                    }

                    if (selected) {
                        spdlog::info("About to call SetSelectedNode with node id={}", selected->id);
                        properties_panel_->SetSelectedNode(selected);
                        spdlog::info("SetSelectedNode completed successfully");
                    } else {
                        spdlog::warn("Selected node not found in nodes vector!");
                    }
                } else {
                    selected_node_id_ = new_selected_id;
                    // Selection unchanged, update silently
                    MLNode* selected = nullptr;
                    for (auto& node : nodes_) {
                        if (node.id == selected_node_id_) {
                            selected = &node;
                            break;
                        }
                    }
                    if (selected) {
                        properties_panel_->SetSelectedNode(selected);
                    }
                }
            } else if (num_selected == 0 && selected_node_id_ != -1) {
                spdlog::info("Node deselected");
                selected_node_id_ = -1;
                properties_panel_->ClearSelection();
            }
        }
    }
    ImGui::End();
}

void NodeEditor::ShowToolbar() {
    // File operations
    if (ImGui::Button("Save Graph")) {
        ShowSaveDialog();
    }
    ImGui::SameLine();

    if (ImGui::Button("Load Graph")) {
        ShowLoadDialog();
    }
    ImGui::SameLine();

    ImGui::Text("|");
    ImGui::SameLine();

    if (ImGui::Button("Add Dense Layer")) {
        AddNode(NodeType::Dense, "Dense Layer");
    }
    ImGui::SameLine();

    if (ImGui::Button("Add ReLU")) {
        AddNode(NodeType::ReLU, "ReLU");
    }
    ImGui::SameLine();

    if (ImGui::Button("Clear All")) {
        ClearGraph();
    }
    ImGui::SameLine();

    ImGui::Text("|");
    ImGui::SameLine();

    // Framework selection
    const char* frameworks[] = { "PyTorch", "TensorFlow", "Keras", "PyCyxWiz" };
    int current_framework = static_cast<int>(selected_framework_);
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::Combo("##Framework", &current_framework, frameworks, 4)) {
        selected_framework_ = static_cast<CodeFramework>(current_framework);
        spdlog::info("Code generation framework changed to: {}", frameworks[current_framework]);
    }
    ImGui::SameLine();

    if (ImGui::Button("Generate Code")) {
        GeneratePythonCode();
    }

    ImGui::SameLine();
    ImGui::Text("| Nodes: %zu | Links: %zu", nodes_.size(), links_.size());
}

void NodeEditor::RenderNodes() {
    // Render all nodes
    for (const auto& node : nodes_) {
        // Set node color based on type
        ImNodes::PushColorStyle(ImNodesCol_TitleBar, GetNodeColor(node.type));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, GetNodeColor(node.type));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, GetNodeColor(node.type));

        ImNodes::BeginNode(node.id);

        // Node title bar
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(node.name.c_str());
        ImNodes::EndNodeTitleBar();

        // Input pins
        for (const auto& pin : node.inputs) {
            ImNodes::BeginInputAttribute(pin.id);
            ImGui::TextUnformatted(pin.name.c_str());
            ImNodes::EndInputAttribute();
        }

        // Display key parameter based on node type
        ImGui::Spacing();
        switch (node.type) {
            case NodeType::Input: {
                auto it = node.parameters.find("shape");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Shape: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::Dense: {
                auto it = node.parameters.find("units");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Units: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::Conv2D: {
                auto it = node.parameters.find("filters");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Filters: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::MaxPool2D: {
                auto it = node.parameters.find("pool_size");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Pool: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::Dropout: {
                auto it = node.parameters.find("rate");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Rate: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::BatchNorm: {
                auto it = node.parameters.find("momentum");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Momentum: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::Output: {
                auto it = node.parameters.find("classes");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Classes: %s", it->second.c_str());
                }
                break;
            }
            default:
                // For activation layers and other nodes without parameters, show nothing
                break;
        }
        ImGui::Spacing();

        // Output pins
        for (const auto& pin : node.outputs) {
            ImNodes::BeginOutputAttribute(pin.id);
            const float text_width = ImGui::CalcTextSize(pin.name.c_str()).x;
            ImGui::Indent(120.0f + ImGui::CalcTextSize(pin.name.c_str()).x - text_width);
            ImGui::TextUnformatted(pin.name.c_str());
            ImNodes::EndOutputAttribute();
        }

        ImNodes::EndNode();

        // Pop color styles
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }

    // Render all links
    for (const auto& link : links_) {
        ImNodes::Link(link.id, link.from_pin, link.to_pin);
    }
}

void NodeEditor::HandleInteractions() {
    // Handle new link creation
    // Use the extended version that provides both node IDs and pin IDs
    int from_node, from_pin, to_node, to_pin;
    if (ImNodes::IsLinkCreated(&from_node, &from_pin, &to_node, &to_pin)) {
        NodeLink link;
        link.id = next_link_id_++;
        link.from_node = from_node;
        link.from_pin = from_pin;
        link.to_node = to_node;
        link.to_pin = to_pin;

        links_.push_back(link);
        spdlog::info("Created link {} from node {} pin {} to node {} pin {}",
                    link.id, from_node, from_pin, to_node, to_pin);
    }

    // Handle link deletion
    int deleted_link_id;
    if (ImNodes::IsLinkDestroyed(&deleted_link_id)) {
        auto it = std::find_if(links_.begin(), links_.end(),
            [deleted_link_id](const NodeLink& link) {
                return link.id == deleted_link_id;
            });

        if (it != links_.end()) {
            spdlog::info("Deleted link {}", deleted_link_id);
            links_.erase(it);
        }
    }

    // Handle node deletion (Delete key)
    const int num_selected_nodes = ImNodes::NumSelectedNodes();

    // Debug: Log selection count (only when it changes)
    static int last_selected_count = 0;
    if (num_selected_nodes != last_selected_count) {
        spdlog::info("Selected nodes: {}", num_selected_nodes);
        last_selected_count = num_selected_nodes;
    }

    if (num_selected_nodes > 0 && ImGui::IsKeyReleased(ImGuiKey_Delete)) {
        std::vector<int> selected_nodes(num_selected_nodes);
        ImNodes::GetSelectedNodes(selected_nodes.data());

        spdlog::info("Deleting {} selected nodes", num_selected_nodes);
        for (int node_id : selected_nodes) {
            DeleteNode(node_id);
        }
    }
}

void NodeEditor::ShowContextMenu() {
    ImGui::Text("Add Node:");
    ImGui::Separator();

    if (ImGui::MenuItem("Input Layer")) {
        AddNode(NodeType::Input, "Input");
        ImGui::CloseCurrentPopup();
    }

    if (ImGui::BeginMenu("Dense Layers")) {
        if (ImGui::MenuItem("Dense (64 units)")) {
            AddNode(NodeType::Dense, "Dense (64)");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Dense (128 units)")) {
            AddNode(NodeType::Dense, "Dense (128)");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Dense (256 units)")) {
            AddNode(NodeType::Dense, "Dense (256)");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Activation Functions")) {
        if (ImGui::MenuItem("ReLU")) {
            AddNode(NodeType::ReLU, "ReLU");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Sigmoid")) {
            AddNode(NodeType::Sigmoid, "Sigmoid");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Tanh")) {
            AddNode(NodeType::Tanh, "Tanh");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Softmax")) {
            AddNode(NodeType::Softmax, "Softmax");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Convolutional")) {
        if (ImGui::MenuItem("Conv2D")) {
            AddNode(NodeType::Conv2D, "Conv2D");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("MaxPool2D")) {
            AddNode(NodeType::MaxPool2D, "MaxPool2D");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Regularization")) {
        if (ImGui::MenuItem("Dropout")) {
            AddNode(NodeType::Dropout, "Dropout (0.5)");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("BatchNorm")) {
            AddNode(NodeType::BatchNorm, "BatchNorm");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    if (ImGui::MenuItem("Flatten")) {
        AddNode(NodeType::Flatten, "Flatten");
        ImGui::CloseCurrentPopup();
    }

    if (ImGui::MenuItem("Output Layer")) {
        AddNode(NodeType::Output, "Output");
        ImGui::CloseCurrentPopup();
    }
}

void NodeEditor::AddNode(NodeType type, const std::string& name) {
    // Queue the node for deferred addition (after ImNodes::EndNodeEditor())
    pending_nodes_.push_back({type, name, context_menu_pos_});
    spdlog::info("Queued node for addition: type={}, name={} at position x={} y={}",
                 static_cast<int>(type), name, context_menu_pos_.x, context_menu_pos_.y);
}

MLNode NodeEditor::CreateNode(NodeType type, const std::string& name) {
    MLNode node;
    node.id = next_node_id_++;
    node.type = type;
    node.name = name;

    // Create pins based on node type
    switch (type) {
        case NodeType::Input: {
            // Input node has no input pins, only output
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["shape"] = "(None, 784)";
            break;
        }

        case NodeType::Dense: {
            // Dense layer has input and output
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Extract units from name (e.g., "Dense (128)")
            size_t start = name.find('(');
            size_t end = name.find(')');
            if (start != std::string::npos && end != std::string::npos) {
                node.parameters["units"] = name.substr(start + 1, end - start - 1);
            } else {
                node.parameters["units"] = "128";
            }
            break;
        }

        case NodeType::ReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
        case NodeType::LeakyReLU: {
            // Activation functions
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
        }

        case NodeType::Output: {
            // Output node has input only
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            node.parameters["units"] = "10";
            break;
        }

        case NodeType::Conv2D: {
            // Conv2D layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["filters"] = "32";
            node.parameters["kernel_size"] = "3";
            node.parameters["stride"] = "1";
            node.parameters["padding"] = "same";
            node.parameters["activation"] = "relu";
            break;
        }

        case NodeType::MaxPool2D: {
            // MaxPool2D layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["pool_size"] = "2";
            node.parameters["stride"] = "2";
            break;
        }

        case NodeType::Flatten: {
            // Flatten layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
        }

        case NodeType::Dropout: {
            // Dropout layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["rate"] = "0.5";
            break;
        }

        case NodeType::BatchNorm: {
            // Batch Normalization layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["momentum"] = "0.99";
            node.parameters["epsilon"] = "0.001";
            break;
        }

        default:
            // Default: input and output pins
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
    }

    return node;
}

void NodeEditor::DeleteNode(int node_id) {
    // Delete node
    auto node_it = std::find_if(nodes_.begin(), nodes_.end(),
        [node_id](const MLNode& node) {
            return node.id == node_id;
        });

    if (node_it != nodes_.end()) {
        spdlog::info("Deleting node: {} (ID: {})", node_it->name, node_id);

        // Delete all links connected to this node
        links_.erase(
            std::remove_if(links_.begin(), links_.end(),
                [node_id](const NodeLink& link) {
                    return link.from_node == node_id || link.to_node == node_id;
                }),
            links_.end());

        nodes_.erase(node_it);
    }
}

void NodeEditor::ClearGraph() {
    nodes_.clear();
    links_.clear();
    next_node_id_ = 1;
    next_pin_id_ = 1;
    next_link_id_ = 1;
    spdlog::info("Cleared node graph");
}

void NodeEditor::CreateLink(int from_pin, int to_pin, int from_node, int to_node) {
    NodeLink link;
    link.id = next_link_id_++;
    link.from_pin = from_pin;
    link.to_pin = to_pin;
    link.from_node = from_node;
    link.to_node = to_node;
    links_.push_back(link);
}

void NodeEditor::GeneratePythonCode() {
    // Validate graph before generating code
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Graph validation failed: {}", error_message);
        // TODO: Show error dialog to user
        return;
    }

    GenerateCodeForFramework(selected_framework_);
}

void NodeEditor::GenerateCodeForFramework(CodeFramework framework) {
    spdlog::info("Generating code from node graph...");

    if (nodes_.empty()) {
        spdlog::warn("No nodes in graph - cannot generate code");
        return;
    }

    // Get topologically sorted node order
    std::vector<int> sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to perform topological sort - graph may have cycles");
        return;
    }

    std::string code;
    const char* framework_name = "";

    // Generate code based on selected framework
    switch (framework) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            break;
        default:
            spdlog::error("Unknown framework selected");
            return;
    }

    spdlog::info("Generated {} code ({} lines)", framework_name, std::count(code.begin(), code.end(), '\n'));

    // Send to Script Editor panel
    if (script_editor_) {
        script_editor_->LoadGeneratedCode(code, framework_name);
        script_editor_->SetVisible(true);  // Show the script editor
        spdlog::info("Code sent to Script Editor panel");
    } else {
        spdlog::warn("Script Editor panel not available - logging code instead");
        spdlog::info("{} code:\n{}", framework_name, code);
    }
}

std::string NodeEditor::GeneratePyTorchCode(const std::vector<int>& sorted_ids) {
    std::string code;

    // Header
    code += "# Auto-generated PyTorch model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import torch\n";
    code += "import torch.nn as nn\n";
    code += "import torch.nn.functional as F\n";
    code += "import torch.optim as optim\n\n";

    // Model class
    code += "class GeneratedModel(nn.Module):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip Input and Output nodes in __init__ (they don't have layers)
        if (node->type == NodeType::Input || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToPythonLayer(*node);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Forward pass
    code += "    def forward(self, x):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        switch (node->type) {
            case NodeType::Input:
                code += "        # Input layer (x is already the input)\n";
                break;

            case NodeType::Dense:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::ReLU:
                code += "        x = F.relu(x)\n";
                break;

            case NodeType::Sigmoid:
                code += "        x = torch.sigmoid(x)\n";
                break;

            case NodeType::Tanh:
                code += "        x = torch.tanh(x)\n";
                break;

            case NodeType::Softmax:
                code += "        x = F.softmax(x, dim=1)\n";
                break;

            case NodeType::Dropout:
                code += "        x = F.dropout(x, p=0.5, training=self.training)\n";
                break;

            case NodeType::Flatten:
                code += "        x = torch.flatten(x, 1)\n";
                break;

            case NodeType::Output:
                code += "        # Output layer\n";
                break;

            default:
                break;
        }
    }
    code += "        return x\n\n";

    // Training code
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n";
    code += "    print(model)\n\n";

    code += "    # Loss and optimizer\n";
    code += "    criterion = nn.CrossEntropyLoss()\n";
    code += "    optimizer = optim.Adam(model.parameters(), lr=0.001)\n\n";

    code += "    # TODO: Add your training data here\n";
    code += "    # Example training loop:\n";
    code += "    # for epoch in range(num_epochs):\n";
    code += "    #     for batch_idx, (data, target) in enumerate(train_loader):\n";
    code += "    #         optimizer.zero_grad()\n";
    code += "    #         output = model(data)\n";
    code += "    #         loss = criterion(output, target)\n";
    code += "    #         loss.backward()\n";
    code += "    #         optimizer.step()\n";

    return code;
}

std::string NodeEditor::GenerateTensorFlowCode(const std::vector<int>& sorted_ids) {
    std::string code;

    // Header
    code += "# Auto-generated TensorFlow model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import tensorflow as tf\n";
    code += "from tensorflow.keras import layers, models, optimizers\n\n";

    // Model class using tf.keras
    code += "class GeneratedModel(tf.keras.Model):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip Input and Output nodes in __init__
        if (node->type == NodeType::Input || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToTensorFlowLayer(*node, layer_idx);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Call method (forward pass in TensorFlow)
    code += "    def call(self, x, training=False):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        switch (node->type) {
            case NodeType::Input:
                code += "        # Input layer (x is already the input)\n";
                break;

            case NodeType::Dense:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::ReLU:
                code += "        x = tf.nn.relu(x)\n";
                break;

            case NodeType::Sigmoid:
                code += "        x = tf.nn.sigmoid(x)\n";
                break;

            case NodeType::Tanh:
                code += "        x = tf.nn.tanh(x)\n";
                break;

            case NodeType::Softmax:
                code += "        x = tf.nn.softmax(x)\n";
                break;

            case NodeType::Dropout:
                code += "        x = tf.keras.layers.Dropout(0.5)(x, training=training)\n";
                break;

            case NodeType::Flatten:
                code += "        x = tf.keras.layers.Flatten()(x)\n";
                break;

            case NodeType::Output:
                code += "        # Output layer\n";
                break;

            default:
                break;
        }
    }
    code += "        return x\n\n";

    // Training code
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n";
    code += "    model.build(input_shape=(None, 784))  # Adjust input shape as needed\n";
    code += "    model.summary()\n\n";

    code += "    # Compile model\n";
    code += "    model.compile(\n";
    code += "        optimizer='adam',\n";
    code += "        loss='sparse_categorical_crossentropy',\n";
    code += "        metrics=['accuracy']\n";
    code += "    )\n\n";

    code += "    # TODO: Add your training data here\n";
    code += "    # Example training:\n";
    code += "    # model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)\n";

    return code;
}

std::string NodeEditor::GenerateKerasCode(const std::vector<int>& sorted_ids) {
    std::string code;

    // Header
    code += "# Auto-generated Keras model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "from tensorflow import keras\n";
    code += "from tensorflow.keras import layers\n\n";

    // Sequential model approach
    code += "# Build model using Sequential API\n";
    code += "model = keras.Sequential([\n";

    bool first_layer = true;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip Input node
        if (node->type == NodeType::Input) {
            continue;
        }

        std::string layer_code = NodeTypeToKerasLayer(*node);
        if (!layer_code.empty()) {
            if (!first_layer) {
                code += ",\n";
            }
            code += "    " + layer_code;
            first_layer = false;
        }
    }

    code += "\n])\n\n";

    // Model summary and compilation
    code += "# Model configuration\n";
    code += "model.build(input_shape=(None, 784))  # Adjust input shape as needed\n";
    code += "model.summary()\n\n";

    code += "# Compile model\n";
    code += "model.compile(\n";
    code += "    optimizer='adam',\n";
    code += "    loss='sparse_categorical_crossentropy',\n";
    code += "    metrics=['accuracy']\n";
    code += ")\n\n";

    code += "# TODO: Add your training data here\n";
    code += "# Example training:\n";
    code += "# history = model.fit(\n";
    code += "#     x_train, y_train,\n";
    code += "#     epochs=10,\n";
    code += "#     batch_size=32,\n";
    code += "#     validation_split=0.2\n";
    code += "# )\n";

    return code;
}

std::string NodeEditor::GeneratePyCyxWizCode(const std::vector<int>& sorted_ids) {
    std::string code;

    // Header
    code += "# Auto-generated PyCyxWiz model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import pycyxwiz as cx\n";
    code += "import numpy as np\n\n";

    // Model class using pycyxwiz
    code += "class GeneratedModel:\n";
    code += "    def __init__(self):\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip Input and Output nodes in __init__
        if (node->type == NodeType::Input || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToPyCyxWizLayer(*node);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Forward method
    code += "    def forward(self, x):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        switch (node->type) {
            case NodeType::Input:
                code += "        # Input layer (x is already the input tensor)\n";
                break;

            case NodeType::Dense:
                code += "        x = self.layer" + std::to_string(layer_idx++) + ".forward(x)\n";
                break;

            case NodeType::ReLU:
                code += "        x = cx.relu(x)\n";
                break;

            case NodeType::Sigmoid:
                code += "        x = cx.sigmoid(x)\n";
                break;

            case NodeType::Tanh:
                code += "        x = cx.tanh(x)\n";
                break;

            case NodeType::Softmax:
                code += "        x = cx.softmax(x)\n";
                break;

            case NodeType::Dropout:
                code += "        x = cx.dropout(x, p=0.5)\n";
                break;

            case NodeType::Flatten:
                code += "        x = cx.flatten(x)\n";
                break;

            case NodeType::Output:
                code += "        # Output layer\n";
                break;

            default:
                break;
        }
    }
    code += "        return x\n\n";

    code += "    def train(self, x_train, y_train, epochs=10, learning_rate=0.001):\n";
    code += "        \"\"\"Training loop using CyxWiz backend\"\"\"\n";
    code += "        optimizer = cx.Adam(learning_rate=learning_rate)\n";
    code += "        loss_fn = cx.CrossEntropyLoss()\n\n";
    code += "        for epoch in range(epochs):\n";
    code += "            # Forward pass\n";
    code += "            predictions = self.forward(x_train)\n";
    code += "            loss = loss_fn(predictions, y_train)\n\n";
    code += "            # Backward pass\n";
    code += "            loss.backward()\n";
    code += "            optimizer.step()\n";
    code += "            optimizer.zero_grad()\n\n";
    code += "            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')\n\n";

    // Training setup
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Initialize CyxWiz backend\n";
    code += "    cx.initialize()\n\n";
    code += "    # Select device (GPU if available)\n";
    code += "    device = cx.get_device(cx.DeviceType.CUDA if cx.cuda_available() else cx.DeviceType.CPU)\n";
    code += "    cx.set_device(device)\n";
    code += "    print(f'Using device: {device.name()}')\n\n";

    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n\n";

    code += "    # TODO: Load your training data here\n";
    code += "    # x_train = cx.Tensor(your_data)\n";
    code += "    # y_train = cx.Tensor(your_labels)\n";
    code += "    # model.train(x_train, y_train, epochs=10)\n";

    return code;
}

std::string NodeEditor::NodeTypeToPythonLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            // Note: input size needs to be determined from graph connections
            code = "nn.Linear(in_features=AUTO, out_features=" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "nn.MaxPool2d(kernel_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "nn.BatchNorm2d(num_features=AUTO)";
            break;

        case NodeType::Dropout: {
            code = "nn.Dropout(p=0.5)";
            break;
        }

        default:
            // Activation functions and others don't need layers in __init__
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToTensorFlowLayer(const MLNode& node, int layer_idx) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "layers.Conv2D(32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "layers.MaxPool2D(pool_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "layers.BatchNormalization()";
            break;

        case NodeType::Dropout:
            code = "layers.Dropout(0.5)";
            break;

        default:
            // Activation functions and others don't need layers in __init__
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToKerasLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "layers.Conv2D(32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "layers.MaxPool2D(pool_size=2)";
            break;

        case NodeType::Flatten:
            code = "layers.Flatten()";
            break;

        case NodeType::Dropout:
            code = "layers.Dropout(0.5)";
            break;

        case NodeType::BatchNorm:
            code = "layers.BatchNormalization()";
            break;

        case NodeType::ReLU:
            code = "layers.ReLU()";
            break;

        case NodeType::Sigmoid:
            code = "layers.Activation('sigmoid')";
            break;

        case NodeType::Tanh:
            code = "layers.Activation('tanh')";
            break;

        case NodeType::Softmax:
            code = "layers.Activation('softmax')";
            break;

        case NodeType::Output: {
            std::string units = "10";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ", activation='softmax')";
            break;
        }

        default:
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToPyCyxWizLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            // Note: pycyxwiz Dense layer requires input size determination from graph
            code = "cx.Dense(in_features=AUTO, out_features=" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "cx.Conv2D(in_channels=1, out_channels=32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "cx.MaxPool2D(kernel_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "cx.BatchNorm()";
            break;

        case NodeType::Dropout:
            code = "cx.Dropout(p=0.5)";
            break;

        default:
            // Activation functions handled in forward pass
            code = "";
            break;
    }

    return code;
}

std::vector<int> NodeEditor::TopologicalSort() {
    std::vector<int> result;
    std::map<int, int> in_degree;
    std::map<int, std::vector<int>> adj_list;

    // Initialize in-degree for all nodes
    for (const auto& node : nodes_) {
        in_degree[node.id] = 0;
        adj_list[node.id] = {};
    }

    // Build adjacency list and calculate in-degrees
    for (const auto& link : links_) {
        adj_list[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Find all nodes with in-degree 0 (starting nodes)
    std::vector<int> queue;
    for (const auto& [node_id, degree] : in_degree) {
        if (degree == 0) {
            queue.push_back(node_id);
        }
    }

    // Process nodes
    while (!queue.empty()) {
        int current = queue.front();
        queue.erase(queue.begin());
        result.push_back(current);

        // Reduce in-degree for neighbors
        for (int neighbor : adj_list[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push_back(neighbor);
            }
        }
    }

    // Check if all nodes were processed (no cycles)
    if (result.size() != nodes_.size()) {
        spdlog::error("Graph has cycles - cannot generate code");
        return {};
    }

    return result;
}

const MLNode* NodeEditor::FindNodeById(int node_id) const {
    for (const auto& node : nodes_) {
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

// ========== Color-Coding Implementation ==========
unsigned int NodeEditor::GetNodeColor(NodeType type) {
    switch (type) {
        // Input/Output - Distinct Blue Gradient
        case NodeType::Input:
            return IM_COL32(41, 128, 185, 255);   // Strong Blue
        case NodeType::Output:
            return IM_COL32(52, 152, 219, 255);   // Lighter Blue

        // Core Layers - Green Tones
        case NodeType::Dense:
            return IM_COL32(39, 174, 96, 255);    // Emerald Green

        // Convolutional Layers - Purple Tones
        case NodeType::Conv2D:
            return IM_COL32(142, 68, 173, 255);   // Purple
        case NodeType::MaxPool2D:
            return IM_COL32(155, 89, 182, 255);   // Light Purple

        // Activation Functions - Warm Orange/Yellow
        case NodeType::ReLU:
            return IM_COL32(243, 156, 18, 255);   // Orange
        case NodeType::Sigmoid:
            return IM_COL32(241, 196, 15, 255);   // Gold
        case NodeType::Tanh:
            return IM_COL32(230, 126, 34, 255);   // Carrot
        case NodeType::Softmax:
            return IM_COL32(211, 84, 0, 255);     // Dark Orange
        case NodeType::LeakyReLU:
            return IM_COL32(235, 152, 78, 255);   // Light Orange

        // Regularization - Red/Pink Tones
        case NodeType::Dropout:
            return IM_COL32(231, 76, 60, 255);    // Red
        case NodeType::BatchNorm:
            return IM_COL32(236, 112, 99, 255);   // Light Red

        // Utility Layers - Teal
        case NodeType::Flatten:
            return IM_COL32(22, 160, 133, 255);   // Teal

        // Loss Functions - Dark Red
        case NodeType::MSELoss:
            return IM_COL32(192, 57, 43, 255);    // Dark Red
        case NodeType::CrossEntropyLoss:
            return IM_COL32(169, 50, 38, 255);    // Darker Red

        // Optimizers - Dark Blue/Gray
        case NodeType::SGD:
            return IM_COL32(44, 62, 80, 255);     // Dark Blue Gray
        case NodeType::Adam:
            return IM_COL32(52, 73, 94, 255);     // Blue Gray
        case NodeType::AdamW:
            return IM_COL32(69, 90, 100, 255);    // Light Blue Gray

        default:
            return IM_COL32(127, 140, 141, 255);  // Neutral Gray
    }
}

// ========== Graph Validation ==========
bool NodeEditor::ValidateGraph(std::string& error_message) {
    if (nodes_.empty()) {
        error_message = "Graph is empty. Add nodes first.";
        return false;
    }

    bool has_input = false;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::Input) {
            has_input = true;
            break;
        }
    }

    if (!has_input) {
        error_message = "Graph must have at least one Input node.";
        return false;
    }

    return true;
}

// ========== Save/Load Implementation ==========

bool NodeEditor::SaveGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        json j;
        j["version"] = "1.0";
        j["framework"] = static_cast<int>(selected_framework_);

        // Serialize nodes
        json nodes_array = json::array();
        for (const auto& node : nodes_) {
            json node_json;
            node_json["id"] = node.id;
            node_json["type"] = static_cast<int>(node.type);
            node_json["name"] = node.name;
            node_json["parameters"] = node.parameters;

            // Save node position
            ImVec2 pos = ImNodes::GetNodeGridSpacePos(node.id);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;
            links_array.push_back(link_json);
        }
        j["links"] = links_array;

        // Write to file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for writing: {}", filepath);
            return false;
        }

        file << j.dump(4);  // Pretty print with 4-space indent
        current_file_path_ = filepath;
        spdlog::info("Graph saved to: {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error saving graph: {}", e.what());
        return false;
    }
}

bool NodeEditor::LoadGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for reading: {}", filepath);
            return false;
        }

        json j;
        file >> j;

        // Clear existing graph
        ClearGraph();

        // Update next IDs to avoid conflicts
        int max_node_id = 0;
        int max_pin_id = 0;
        int max_link_id = 0;

        // Load framework
        if (j.contains("framework")) {
            selected_framework_ = static_cast<CodeFramework>(j["framework"].get<int>());
        }

        // Load nodes
        for (const auto& node_json : j["nodes"]) {
            MLNode node;
            node.id = node_json["id"];
            node.type = static_cast<NodeType>(node_json["type"].get<int>());
            node.name = node_json["name"];

            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            }

            // Recreate pins based on node type
            MLNode template_node = CreateNode(node.type, node.name);
            node.inputs = template_node.inputs;
            node.outputs = template_node.outputs;

            // Update max IDs
            max_node_id = std::max(max_node_id, node.id);
            for (const auto& pin : node.inputs) {
                max_pin_id = std::max(max_pin_id, pin.id);
            }
            for (const auto& pin : node.outputs) {
                max_pin_id = std::max(max_pin_id, pin.id);
            }

            nodes_.push_back(node);

            // Restore node position
            if (node_json.contains("pos_x") && node_json.contains("pos_y")) {
                float pos_x = node_json["pos_x"];
                float pos_y = node_json["pos_y"];
                ImNodes::SetNodeGridSpacePos(node.id, ImVec2(pos_x, pos_y));
            }
        }

        // Load links
        for (const auto& link_json : j["links"]) {
            NodeLink link;
            link.id = link_json["id"];
            link.from_node = link_json["from_node"];
            link.from_pin = link_json["from_pin"];
            link.to_node = link_json["to_node"];
            link.to_pin = link_json["to_pin"];
            links_.push_back(link);

            max_link_id = std::max(max_link_id, link.id);
        }

        // Update next IDs
        next_node_id_ = max_node_id + 1;
        next_pin_id_ = max_pin_id + 1;
        next_link_id_ = max_link_id + 1;

        current_file_path_ = filepath;
        spdlog::info("Graph loaded from: {} ({} nodes, {} links)",
                     filepath, nodes_.size(), links_.size());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error loading graph: {}", e.what());
        return false;
    }
}

#ifdef _WIN32
void NodeEditor::ShowSaveDialog() {
    char szFile[260] = {0};

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "CyxWiz Graph Files\0*.cyxwiz\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    ofn.lpstrDefExt = "cyxwiz";
    ofn.lpstrTitle = "Save Neural Network Graph";

    if (GetSaveFileNameA(&ofn)) {
        if (SaveGraph(szFile)) {
            spdlog::info("Graph successfully saved");
        }
    }
}

void NodeEditor::ShowLoadDialog() {
    char szFile[260] = {0};

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "CyxWiz Graph Files\0*.cyxwiz\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    ofn.lpstrTitle = "Load Neural Network Graph";

    if (GetOpenFileNameA(&ofn)) {
        if (LoadGraph(szFile)) {
            spdlog::info("Graph successfully loaded");
        }
    }
}
#else
// For non-Windows platforms, implement later or use a cross-platform dialog library
void NodeEditor::ShowSaveDialog() {
    SaveGraph("model.cyxwiz");
}

void NodeEditor::ShowLoadDialog() {
    LoadGraph("model.cyxwiz");
}
#endif

} // namespace gui
