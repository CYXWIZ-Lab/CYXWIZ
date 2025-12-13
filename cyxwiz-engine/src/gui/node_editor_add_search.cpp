// node_editor_add_search.cpp
// Node add search functionality - allows users to quickly find and add nodes via search box

#include "node_editor.h"
#include "icons.h"
#include <imgui.h>
#include <imnodes.h>
#include <algorithm>
#include <cctype>

namespace gui {

// Fuzzy match algorithm - returns a score (higher is better match, 0 = no match)
// Inspired by fzf/sublime text matching
int NodeEditor::FuzzyMatch(const std::string& pattern, const std::string& str) {
    if (pattern.empty()) return 1;  // Empty pattern matches everything
    if (str.empty()) return 0;

    // Convert both to lowercase for case-insensitive matching
    std::string pattern_lower, str_lower;
    pattern_lower.reserve(pattern.size());
    str_lower.reserve(str.size());

    for (char c : pattern) pattern_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (char c : str) str_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    int score = 0;
    size_t pattern_idx = 0;
    size_t last_match_idx = 0;
    bool consecutive = true;

    for (size_t i = 0; i < str_lower.size() && pattern_idx < pattern_lower.size(); ++i) {
        if (str_lower[i] == pattern_lower[pattern_idx]) {
            // Character matched
            score += 10;  // Base score for match

            // Bonus for consecutive matches
            if (consecutive && i == last_match_idx + 1) {
                score += 15;
            }

            // Bonus for matching at start of word (after space, underscore, or at beginning)
            if (i == 0 || str_lower[i-1] == ' ' || str_lower[i-1] == '_' || str_lower[i-1] == '/') {
                score += 20;
            }

            // Bonus for matching uppercase in original (CamelCase)
            if (i < str.size() && std::isupper(static_cast<unsigned char>(str[i]))) {
                score += 10;
            }

            last_match_idx = i;
            consecutive = true;
            pattern_idx++;
        } else {
            consecutive = false;
        }
    }

    // All pattern characters must be found
    if (pattern_idx < pattern_lower.size()) {
        return 0;  // No match
    }

    // Bonus for shorter strings (more relevant)
    score += static_cast<int>(100 - std::min(str.size(), size_t(100)));

    // Bonus for exact substring match
    if (str_lower.find(pattern_lower) != std::string::npos) {
        score += 50;
    }

    return score;
}

// Initialize the searchable nodes list with all available node types
void NodeEditor::InitializeSearchableNodes() {
    if (searchable_nodes_initialized_) return;

    all_searchable_nodes_.clear();

    // Helper lambda to add a node
    auto addNode = [this](NodeType type, const std::string& name, const std::string& category, const std::string& keywords = "") {
        all_searchable_nodes_.push_back({type, name, category, keywords});
    };

    // Output
    addNode(NodeType::Output, "Output", "Input/Output", "data out end result");

    // Data Pipeline
    addNode(NodeType::DatasetInput, "DatasetInput", "Data Pipeline", "dataset mnist cifar load input data");
    addNode(NodeType::DataLoader, "DataLoader", "Data Pipeline", "batch iterator shuffle loader");
    addNode(NodeType::Augmentation, "Augmentation", "Data Pipeline", "transform augment preprocess");
    addNode(NodeType::DataSplit, "DataSplit", "Data Pipeline", "train val test split partition");
    addNode(NodeType::TensorReshape, "TensorReshape", "Data Pipeline", "reshape tensor dimensions");
    addNode(NodeType::Normalize, "Normalize", "Data Pipeline", "normalize mean std scale");
    addNode(NodeType::OneHotEncode, "OneHotEncode", "Data Pipeline", "one hot encoding labels categorical");

    // Layers > Dense/Linear
    addNode(NodeType::Dense, "Dense", "Layers > Dense/Linear", "fully connected linear fc nn");
    addNode(NodeType::Embedding, "Embedding", "Layers > Dense/Linear", "word vector lookup nlp");

    // Layers > Convolutional
    addNode(NodeType::Conv1D, "Conv1D", "Layers > Convolution", "convolution 1d sequence temporal");
    addNode(NodeType::Conv2D, "Conv2D", "Layers > Convolution", "convolution 2d image cnn filter");
    addNode(NodeType::Conv3D, "Conv3D", "Layers > Convolution", "convolution 3d volume video");
    addNode(NodeType::DepthwiseConv2D, "DepthwiseConv2D", "Layers > Convolution", "depthwise separable mobile");

    // Layers > Pooling
    addNode(NodeType::MaxPool2D, "MaxPool2D", "Layers > Pooling", "max pooling downsample");
    addNode(NodeType::AvgPool2D, "AvgPool2D", "Layers > Pooling", "average pooling downsample mean");
    addNode(NodeType::GlobalMaxPool, "GlobalMaxPool", "Layers > Pooling", "global max pooling");
    addNode(NodeType::GlobalAvgPool, "GlobalAvgPool", "Layers > Pooling", "global average pooling gap");
    addNode(NodeType::AdaptiveAvgPool, "AdaptiveAvgPool", "Layers > Pooling", "adaptive pooling output size");

    // Layers > Normalization
    addNode(NodeType::BatchNorm, "BatchNorm", "Layers > Normalization", "batch normalization bn");
    addNode(NodeType::LayerNorm, "LayerNorm", "Layers > Normalization", "layer normalization ln transformer");
    addNode(NodeType::GroupNorm, "GroupNorm", "Layers > Normalization", "group normalization gn");
    addNode(NodeType::InstanceNorm, "InstanceNorm", "Layers > Normalization", "instance normalization style");

    // Layers > Regularization & Reshape
    addNode(NodeType::Dropout, "Dropout", "Layers > Regularization", "regularization prevent overfit");
    addNode(NodeType::Flatten, "Flatten", "Layers > Reshape", "flatten reshape 1d vector");

    // Shape Operations
    addNode(NodeType::Reshape, "Reshape", "Shape Operations", "reshape view dimensions");
    addNode(NodeType::Permute, "Permute", "Shape Operations", "permute transpose axes");
    addNode(NodeType::Squeeze, "Squeeze", "Shape Operations", "squeeze remove dimension");
    addNode(NodeType::Unsqueeze, "Unsqueeze", "Shape Operations", "unsqueeze add dimension expand");
    addNode(NodeType::View, "View", "Shape Operations", "view reshape tensor");
    addNode(NodeType::Split, "Split", "Shape Operations", "split divide chunk");

    // Activations
    addNode(NodeType::ReLU, "ReLU", "Activations", "relu rectified linear activation");
    addNode(NodeType::LeakyReLU, "LeakyReLU", "Activations", "leaky relu activation negative slope");
    addNode(NodeType::PReLU, "PReLU", "Activations", "prelu parametric relu learnable");
    addNode(NodeType::ELU, "ELU", "Activations", "elu exponential linear activation");
    addNode(NodeType::SELU, "SELU", "Activations", "selu scaled exponential linear");
    addNode(NodeType::GELU, "GELU", "Activations", "gelu gaussian error linear bert gpt");
    addNode(NodeType::Swish, "Swish", "Activations", "swish silu activation efficient");
    addNode(NodeType::Mish, "Mish", "Activations", "mish activation smooth");
    addNode(NodeType::Sigmoid, "Sigmoid", "Activations", "sigmoid logistic activation");
    addNode(NodeType::Tanh, "Tanh", "Activations", "tanh hyperbolic tangent activation");
    addNode(NodeType::Softmax, "Softmax", "Activations", "softmax probability distribution classification");

    // Recurrent
    addNode(NodeType::RNN, "RNN", "Recurrent", "rnn recurrent neural network vanilla");
    addNode(NodeType::LSTM, "LSTM", "Recurrent", "lstm long short term memory rnn sequence");
    addNode(NodeType::GRU, "GRU", "Recurrent", "gru gated recurrent unit rnn sequence");
    addNode(NodeType::Bidirectional, "Bidirectional", "Recurrent", "bidirectional forward backward rnn");
    addNode(NodeType::TimeDistributed, "TimeDistributed", "Recurrent", "time distributed wrapper sequence");

    // Attention & Transformer
    addNode(NodeType::MultiHeadAttention, "MultiHeadAttention", "Attention", "multi head attention transformer mha");
    addNode(NodeType::SelfAttention, "SelfAttention", "Attention", "self attention query key value");
    addNode(NodeType::CrossAttention, "CrossAttention", "Attention", "cross attention encoder decoder");
    addNode(NodeType::LinearAttention, "LinearAttention", "Attention", "linear attention performer efficient");
    addNode(NodeType::TransformerEncoder, "TransformerEncoder", "Transformer", "transformer encoder layer bert");
    addNode(NodeType::TransformerDecoder, "TransformerDecoder", "Transformer", "transformer decoder layer gpt");
    addNode(NodeType::PositionalEncoding, "PositionalEncoding", "Transformer", "positional encoding sinusoidal");

    // Merge Operations
    addNode(NodeType::Add, "Add", "Merge Operations", "add sum residual skip connection");
    addNode(NodeType::Multiply, "Multiply", "Merge Operations", "multiply element wise product");
    addNode(NodeType::Concatenate, "Concatenate", "Merge Operations", "concat join merge axis");
    addNode(NodeType::Average, "Average", "Merge Operations", "average mean merge");

    // Loss Functions
    addNode(NodeType::MSELoss, "MSELoss", "Loss Functions", "mse mean squared error l2 regression");
    addNode(NodeType::CrossEntropyLoss, "CrossEntropyLoss", "Loss Functions", "cross entropy classification softmax");
    addNode(NodeType::BCELoss, "BCELoss", "Loss Functions", "bce binary cross entropy sigmoid");
    addNode(NodeType::BCEWithLogits, "BCEWithLogits", "Loss Functions", "bce with logits binary classification");
    addNode(NodeType::L1Loss, "L1Loss", "Loss Functions", "l1 mae mean absolute error");
    addNode(NodeType::SmoothL1Loss, "SmoothL1Loss", "Loss Functions", "smooth l1 huber robust");
    addNode(NodeType::HuberLoss, "HuberLoss", "Loss Functions", "huber loss robust regression");
    addNode(NodeType::NLLLoss, "NLLLoss", "Loss Functions", "nll negative log likelihood");

    // Optimizers
    addNode(NodeType::SGD, "SGD", "Optimizers", "sgd stochastic gradient descent momentum");
    addNode(NodeType::Adam, "Adam", "Optimizers", "adam adaptive moment estimation");
    addNode(NodeType::AdamW, "AdamW", "Optimizers", "adamw weight decay decoupled");
    addNode(NodeType::RMSprop, "RMSprop", "Optimizers", "rmsprop root mean square propagation");
    addNode(NodeType::Adagrad, "Adagrad", "Optimizers", "adagrad adaptive gradient");
    addNode(NodeType::NAdam, "NAdam", "Optimizers", "nadam nesterov adam");

    // Schedulers
    addNode(NodeType::StepLR, "StepLR", "Schedulers", "step lr learning rate decay");
    addNode(NodeType::ExponentialLR, "ExponentialLR", "Schedulers", "exponential lr decay gamma");
    addNode(NodeType::CosineAnnealing, "CosineAnnealing", "Schedulers", "cosine annealing warm restart");
    addNode(NodeType::ReduceOnPlateau, "ReduceOnPlateau", "Schedulers", "reduce plateau patience factor");
    addNode(NodeType::WarmupScheduler, "WarmupScheduler", "Schedulers", "warmup linear learning rate");

    // Regularization Nodes
    addNode(NodeType::L1Regularization, "L1Regularization", "Regularization", "l1 regularization lasso sparse");
    addNode(NodeType::L2Regularization, "L2Regularization", "Regularization", "l2 regularization ridge weight decay");
    addNode(NodeType::ElasticNet, "ElasticNet", "Regularization", "elastic net l1 l2 combined");

    // Utilities
    addNode(NodeType::Lambda, "Lambda", "Utilities", "lambda custom function apply");
    addNode(NodeType::Identity, "Identity", "Utilities", "identity passthrough skip");
    addNode(NodeType::Constant, "Constant", "Utilities", "constant fixed value");
    addNode(NodeType::Parameter, "Parameter", "Utilities", "parameter learnable tensor");

    // Composite
    addNode(NodeType::Subgraph, "Subgraph", "Composite", "subgraph module encapsulate block");

    searchable_nodes_initialized_ = true;
}

// Update filtered results based on current search query
void NodeEditor::UpdateNodeAddSearchResults() {
    filtered_nodes_.clear();

    std::string query(node_add_search_.search_buffer);

    // If query is empty, show all nodes
    if (query.empty()) {
        for (auto& node : all_searchable_nodes_) {
            filtered_nodes_.push_back({100, &node});  // Default score
        }
    } else {
        // Score each node
        for (auto& node : all_searchable_nodes_) {
            // Match against name, category, and keywords
            int name_score = FuzzyMatch(query, node.name);
            int category_score = FuzzyMatch(query, node.category) / 2;  // Lower weight for category
            int keyword_score = FuzzyMatch(query, node.keywords) / 2;   // Lower weight for keywords

            int total_score = std::max({name_score, category_score, keyword_score});

            if (total_score > 0) {
                filtered_nodes_.push_back({total_score, &node});
            }
        }

        // Sort by score (highest first)
        std::sort(filtered_nodes_.begin(), filtered_nodes_.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // Limit to top 15 results
    if (filtered_nodes_.size() > 15) {
        filtered_nodes_.resize(15);
    }

    // Reset selection if out of bounds
    if (node_add_search_.selected_index >= static_cast<int>(filtered_nodes_.size())) {
        node_add_search_.selected_index = 0;
    }
}

// Render the node add search UI (top-right of canvas)
void NodeEditor::ShowNodeAddSearch() {
    // Initialize searchable nodes on first call
    if (!searchable_nodes_initialized_) {
        InitializeSearchableNodes();
        UpdateNodeAddSearchResults();
    }

    // Get the content region bounds (the actual canvas area inside the window)
    ImVec2 content_min = ImGui::GetWindowContentRegionMin();
    ImVec2 content_max = ImGui::GetWindowContentRegionMax();
    ImVec2 window_pos = ImGui::GetWindowPos();

    // Calculate canvas bounds in screen coordinates
    ImVec2 canvas_pos(window_pos.x + content_min.x, window_pos.y + content_min.y);
    ImVec2 canvas_size(content_max.x - content_min.x, content_max.y - content_min.y);

    // Position search box in top-right corner of the canvas content area
    // Add offset to position below the toolbar rows (approx 70px for two rows of buttons)
    float search_width = 250.0f;
    float search_height = 28.0f;
    float margin = 10.0f;
    float toolbar_offset = 70.0f;  // Offset to position below the toolbars

    ImVec2 search_pos(canvas_pos.x + canvas_size.x - search_width - margin, canvas_pos.y + toolbar_offset + margin);

    // Create a floating window for the search box that renders on top of ImNodes
    ImGui::SetNextWindowPos(search_pos);
    ImGui::SetNextWindowSize(ImVec2(search_width, search_height));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));

    ImGuiWindowFlags search_window_flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoDocking;

    // Declare at function scope so it's accessible for the "close dropdown" check
    bool input_focused = false;

    if (ImGui::Begin("##NodeSearchBox", nullptr, search_window_flags)) {
        // Push style for search input
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.2f, 0.2f, 0.24f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.25f, 0.25f, 0.3f, 0.5f));

        // Search icon
        ImGui::SetCursorPos(ImVec2(8, 6));
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_FA_MAGNIFYING_GLASS);

        // Input field
        ImGui::SetCursorPos(ImVec2(28, 0));
        ImGui::SetNextItemWidth(search_width - 36);

    // Handle keyboard focus
    if (node_add_search_.just_activated) {
        ImGui::SetKeyboardFocusHere();
        node_add_search_.just_activated = false;
    }

    bool text_changed = ImGui::InputTextWithHint(
        "##node_add_search",
        "Search nodes...",
        node_add_search_.search_buffer,
        sizeof(node_add_search_.search_buffer),
        ImGuiInputTextFlags_EnterReturnsTrue
    );

    // Check if input is active
    bool input_active = ImGui::IsItemActive();
    input_focused = ImGui::IsItemFocused();  // Assign to function-scope variable

    // Update search state
    if (input_active || input_focused) {
        node_add_search_.is_active = true;
        node_add_search_.show_results = true;
    }

    // Update results when text changes
    if (text_changed || ImGui::IsItemEdited()) {
        UpdateNodeAddSearchResults();
        node_add_search_.selected_index = 0;
    }

    // Handle keyboard navigation
    if (node_add_search_.is_active && node_add_search_.show_results) {
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            node_add_search_.selected_index = std::min(
                node_add_search_.selected_index + 1,
                static_cast<int>(filtered_nodes_.size()) - 1
            );
        }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            node_add_search_.selected_index = std::max(node_add_search_.selected_index - 1, 0);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            node_add_search_.show_results = false;
            node_add_search_.is_active = false;
            node_add_search_.search_buffer[0] = '\0';
        }
        if ((ImGui::IsKeyPressed(ImGuiKey_Enter) || text_changed) && !filtered_nodes_.empty()) {
            // Add the selected node
            SearchableNode* selected = filtered_nodes_[node_add_search_.selected_index].second;

            // Set position for new node (center of visible canvas)
            ImVec2 editor_pan = ImNodes::EditorContextGetPanning();
            context_menu_pos_ = ImVec2(
                canvas_size.x / 2 - editor_pan.x,
                canvas_size.y / 2 - editor_pan.y
            );

            AddNode(selected->type, selected->name);

            // Reset search state
            node_add_search_.show_results = false;
            node_add_search_.is_active = false;
            node_add_search_.search_buffer[0] = '\0';
            UpdateNodeAddSearchResults();
        }
    }

        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);
    }
    ImGui::End();
    ImGui::PopStyleColor(1);
    ImGui::PopStyleVar(2);

    // Render dropdown results - only show when user has typed something
    std::string query(node_add_search_.search_buffer);
    bool has_search_text = !query.empty();

    if (node_add_search_.show_results && has_search_text && !filtered_nodes_.empty()) {
        ImVec2 dropdown_pos(search_pos.x, search_pos.y + search_height + 2);
        float dropdown_width = search_width;
        float item_height = 32.0f;
        float dropdown_height = std::min(item_height * filtered_nodes_.size() + 8, 400.0f);

        // Create a separate floating window for dropdown results
        ImGui::SetNextWindowPos(dropdown_pos);
        ImGui::SetNextWindowSize(ImVec2(dropdown_width, dropdown_height));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.14f, 0.96f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.24f, 0.24f, 0.28f, 1.0f));

        ImGuiWindowFlags dropdown_flags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav;

        if (ImGui::Begin("##NodeSearchDropdown", nullptr, dropdown_flags)) {
            for (size_t i = 0; i < filtered_nodes_.size(); ++i) {
                SearchableNode* node = filtered_nodes_[i].second;
                bool is_selected = (static_cast<int>(i) == node_add_search_.selected_index);

                ImVec2 item_pos = ImGui::GetCursorScreenPos();
                ImVec2 item_size(dropdown_width - 8, item_height - 2);

                // Highlight selected item
                if (is_selected) {
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();
                    draw_list->AddRectFilled(
                        item_pos,
                        ImVec2(item_pos.x + item_size.x, item_pos.y + item_size.y),
                        IM_COL32(60, 100, 180, 200),
                        3.0f
                    );
                }

                // Handle click
                ImGui::InvisibleButton(("node_result_" + std::to_string(i)).c_str(), item_size);
                if (ImGui::IsItemClicked()) {
                    // Add the clicked node
                    ImVec2 editor_pan = ImNodes::EditorContextGetPanning();
                    context_menu_pos_ = ImVec2(
                        canvas_size.x / 2 - editor_pan.x,
                        canvas_size.y / 2 - editor_pan.y
                    );

                    AddNode(node->type, node->name);

                    // Reset search state
                    node_add_search_.show_results = false;
                    node_add_search_.is_active = false;
                    node_add_search_.search_buffer[0] = '\0';
                    UpdateNodeAddSearchResults();
                }
                if (ImGui::IsItemHovered()) {
                    node_add_search_.selected_index = static_cast<int>(i);
                }

                // Draw node name and category
                ImGui::SetCursorScreenPos(ImVec2(item_pos.x + 8, item_pos.y + 4));
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "%s", node->name.c_str());

                ImGui::SetCursorScreenPos(ImVec2(item_pos.x + 8, item_pos.y + 18));
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.6f, 1.0f), "%s", node->category.c_str());

                ImGui::SetCursorScreenPos(ImVec2(item_pos.x, item_pos.y + item_height));
            }
        }
        ImGui::End();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);
    }

    // Close dropdown when clicking outside
    if (node_add_search_.show_results && !input_focused && ImGui::IsMouseClicked(0)) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        ImVec2 dropdown_check_pos(search_pos.x, search_pos.y);
        float dropdown_height = 32.0f * filtered_nodes_.size() + search_height + 10;

        if (mouse_pos.x < dropdown_check_pos.x || mouse_pos.x > dropdown_check_pos.x + search_width ||
            mouse_pos.y < dropdown_check_pos.y || mouse_pos.y > dropdown_check_pos.y + dropdown_height) {
            node_add_search_.show_results = false;
            node_add_search_.is_active = false;
        }
    }
}

} // namespace gui
