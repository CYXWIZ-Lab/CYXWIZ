#include "graph_compiler.h"
#include "data_registry.h"
#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <queue>
#include <set>
#include <sstream>
#include <stack>

namespace cyxwiz {

namespace {
// Local helpers for issue collection. Kept in an anonymous namespace so
// they don't pollute the cyxwiz public surface.

void AddIssue(TrainingConfiguration& config, IssueLevel level,
              const std::string& message, int node_id = -1,
              const std::string& node_name = "") {
    ValidationIssue issue;
    issue.level = level;
    issue.message = message;
    issue.node_id = node_id;
    issue.node_name = node_name;
    config.issues.push_back(std::move(issue));
}

const char* IssueLevelLabel(IssueLevel level) {
    switch (level) {
        case IssueLevel::Error:   return "ERROR";
        case IssueLevel::Warning: return "WARN";
        case IssueLevel::Info:    return "INFO";
    }
    return "?";
}

// Build the legacy single-string error_message from the issues list so
// existing callers that only look at error_message keep working.
std::string JoinErrorMessages(const std::vector<ValidationIssue>& issues) {
    std::ostringstream out;
    bool first = true;
    for (const auto& i : issues) {
        if (i.level != IssueLevel::Error) continue;
        if (!first) out << "; ";
        first = false;
        if (!i.node_name.empty()) {
            out << "[" << i.node_name << "] ";
        }
        out << i.message;
    }
    return out.str();
}
} // anonymous namespace

TrainingConfiguration GraphCompiler::Compile(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    TrainingConfiguration config;

    // === Structural validation ===
    // Collect all structural errors at once (the old ValidateGraph stopped
    // at the first one). is_valid is determined at the end of Compile by
    // checking whether config.issues contains any Error-level entries.
    const gui::MLNode* dataset_node = FindDatasetInputNode(nodes);
    const gui::MLNode* loss_node = FindLossNode(nodes);
    const gui::MLNode* optimizer_node = FindOptimizerNode(nodes);

    if (nodes.empty()) {
        AddIssue(config, IssueLevel::Error, "Graph is empty - add nodes to create a model");
    } else {
        if (!dataset_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have a DataInput or DatasetInput node");
        }
        if (!loss_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have a loss function (MSELoss, CrossEntropyLoss, etc.)");
        }
        if (!optimizer_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have an optimizer (SGD, Adam, or AdamW)");
        }
        bool has_model_layer = false;
        for (const auto& node : nodes) {
            if (IsModelLayer(node.type)) { has_model_layer = true; break; }
        }
        if (!has_model_layer) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have at least one model layer (Dense, Conv2D, etc.)");
        }
        if (HasCycle(nodes, links)) {
            AddIssue(config, IssueLevel::Error,
                     "Graph contains a cycle - remove circular connections");
        }
    }

    // Extract dataset configuration
    if (dataset_node) {
        // dataset_name parameter is what DataInputDialog::Apply writes when
        // a dataset is loaded. The legacy "dataset" key was used by the
        // older DatasetInput dialog and is now obsolete — fall back to it
        // only if dataset_name is empty, so older project files still load.
        if (dataset_node->parameters.count("dataset_name") &&
            !dataset_node->parameters.at("dataset_name").empty()) {
            config.dataset_name = dataset_node->parameters.at("dataset_name");
        } else if (dataset_node->parameters.count("dataset")) {
            config.dataset_name = dataset_node->parameters.at("dataset");
        }

        // === New error checks tied to the dataset node ===

        // Check 1: data_loaded must be "true". DataInputDialog::Apply sets
        // this when (and only when) the load actually succeeded. If false,
        // the user has the node configured but never clicked Apply, or the
        // load failed, or an async load is still running.
        const std::string& loaded_param = dataset_node->parameters.count("data_loaded")
            ? dataset_node->parameters.at("data_loaded")
            : std::string("false");
        if (loaded_param != "true") {
            AddIssue(config, IssueLevel::Error,
                     "Data is not loaded - open the node and click Apply",
                     dataset_node->id, dataset_node->name);
        }

        // Check 2: dataset must be present in DataRegistry under that name.
        // Catches the case where the user loaded data, deleted the
        // dataset elsewhere, and never re-applied. Also catches stale
        // project state where data_loaded=true but registry was wiped on
        // project close.
        if (!config.dataset_name.empty()) {
            auto& reg = DataRegistry::Instance();
            bool in_registry = reg.IsArrowDataset(config.dataset_name) ||
                               reg.IsParquetBackedDataset(config.dataset_name) ||
                               reg.IsImageDataset(config.dataset_name);
            if (!in_registry && loaded_param == "true") {
                AddIssue(config, IssueLevel::Error,
                         "Dataset '" + config.dataset_name +
                         "' is marked loaded but missing from registry - "
                         "re-apply the DataInput node",
                         dataset_node->id, dataset_node->name);
            }
        } else if (loaded_param == "true") {
            // data_loaded=true but no dataset_name. Inconsistent state.
            AddIssue(config, IssueLevel::Error,
                     "Data marked loaded but no dataset_name parameter set",
                     dataset_node->id, dataset_node->name);
        }

        // Check 3: label column. Required for tabular supervised training,
        // but image datasets get labels from folder structure or CSV mapping
        // automatically — don't emit a misleading warning for them.
        auto cat_it_lbl = dataset_node->parameters.find("file_category");
        bool is_image_dataset = (cat_it_lbl != dataset_node->parameters.end() &&
                                 cat_it_lbl->second == "image");

        const std::string label_col = dataset_node->parameters.count("label_column")
            ? dataset_node->parameters.at("label_column")
            : std::string();
        if (label_col.empty() && !is_image_dataset) {
            AddIssue(config, IssueLevel::Warning,
                     "No label column selected - training will use the last "
                     "column as label by default",
                     dataset_node->id, dataset_node->name);
        }

        // Extract input shape from dataset node
        if (dataset_node->parameters.count("shape")) {
            // Parse shape string like "[28, 28, 1]"
            std::string shape_str = dataset_node->parameters.at("shape");
            // Simple parsing - remove brackets and split by comma
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                config.input_shape.push_back(std::stoul(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                config.input_shape.push_back(std::stoul(shape_str));
            }
        }

        // Calculate flattened input size
        config.input_size = 1;
        for (size_t dim : config.input_shape) {
            config.input_size *= dim;
        }
    }

    // Extract split ratios from DataSplit node if present
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::DataSplit) {
            config.has_data_split = true;
            try {
                if (node.parameters.count("train_ratio"))
                    config.train_ratio = std::stof(node.parameters.at("train_ratio"));
                if (node.parameters.count("val_ratio"))
                    config.val_ratio = std::stof(node.parameters.at("val_ratio"));
                if (node.parameters.count("test_ratio"))
                    config.test_ratio = std::stof(node.parameters.at("test_ratio"));
                if (node.parameters.count("seed"))
                    config.split_seed = std::stoi(node.parameters.at("seed"));
            } catch (const std::exception& e) {
                spdlog::warn("GraphCompiler: DataSplit param parse error ({}) - using defaults", e.what());
            }
            break;
        }
    }
    if (config.has_data_split) {
        spdlog::info("GraphCompiler: DataSplit node found - train={:.2f}, val={:.2f}, test={:.2f}, seed={}",
                     config.train_ratio, config.val_ratio, config.test_ratio, config.split_seed);
    } else {
        spdlog::info("GraphCompiler: No DataSplit node - using defaults (train=0.80, val=0.10, test=0.10)");
    }

    // Extract training-loop config from DataLoader node if present.
    // DataLoader owns batch_size / epochs / shuffle / drop_last / num_workers —
    // all the "how do I iterate training" hyperparameters.
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::DataLoader) {
            config.has_data_loader = true;
            try {
                if (node.parameters.count("batch_size"))
                    config.batch_size = std::stoi(node.parameters.at("batch_size"));
                if (node.parameters.count("epochs"))
                    config.epochs = std::stoi(node.parameters.at("epochs"));
                if (node.parameters.count("shuffle"))
                    config.shuffle = (node.parameters.at("shuffle") == "true");
                if (node.parameters.count("drop_last"))
                    config.drop_last = (node.parameters.at("drop_last") == "true");
                if (node.parameters.count("num_workers"))
                    config.num_workers = std::stoi(node.parameters.at("num_workers"));
            } catch (const std::exception& e) {
                spdlog::warn("GraphCompiler: DataLoader param parse error ({}) - using defaults", e.what());
            }
            break;
        }
    }
    if (config.has_data_loader) {
        spdlog::info("GraphCompiler: DataLoader node found - batch_size={}, epochs={}, shuffle={}, drop_last={}, num_workers={}",
                     config.batch_size, config.epochs, config.shuffle, config.drop_last, config.num_workers);
        if (config.num_workers > 0) {
            spdlog::warn("GraphCompiler: num_workers={} requested but not yet implemented - batching runs single-threaded",
                         config.num_workers);
        }
    } else {
        spdlog::info("GraphCompiler: No DataLoader node - using defaults (batch_size=32, epochs=10, shuffle=true, drop_last=false)");
    }

    // Get topologically sorted node IDs
    std::vector<int> sorted_ids = TopologicalSort(nodes, links);

    // Check if we're using CrossEntropyLoss (which applies softmax internally)
    bool using_cross_entropy = (loss_node && loss_node->type == gui::NodeType::CrossEntropyLoss);

    // Log all nodes in the graph for debugging
    spdlog::info("GraphCompiler: Processing {} nodes (using_cross_entropy={})", sorted_ids.size(), using_cross_entropy);
    for (int node_id : sorted_ids) {
        const gui::MLNode* n = FindNodeById(node_id, nodes);
        if (n) {
            spdlog::debug("GraphCompiler: Node[{}] '{}' type={}", n->id, n->name, static_cast<int>(n->type));
        }
    }

    // Extract model layers and preprocessing in execution order
    std::vector<size_t> current_shape = config.input_shape;

    for (int node_id : sorted_ids) {
        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;

        // Handle preprocessing nodes
        if (IsPreprocessing(node->type)) {
            spdlog::info("GraphCompiler: Found preprocessing node '{}' (type={})", node->name, static_cast<int>(node->type));
            ExtractPreprocessing(*node, config);
            if (node->type == gui::NodeType::Normalize) {
                spdlog::info("GraphCompiler: Normalization enabled - mean={}, std={}",
                             config.preprocessing.norm_mean, config.preprocessing.norm_std);
            }
            continue;
        }

        // Skip Softmax when using CrossEntropyLoss (it applies softmax internally)
        // This prevents double-softmax which kills gradients
        if (node->type == gui::NodeType::Softmax && using_cross_entropy) {
            spdlog::warn("GraphCompiler: Softmax layer skipped - CrossEntropyLoss applies softmax internally");
            continue;
        }

        // Handle model layers and activations
        if (IsModelLayer(node->type) || IsActivation(node->type)) {
            CompiledLayer layer = ExtractLayerConfig(*node);
            layer.input_shape = current_shape;

            // Infer output shape
            layer.output_shape = InferOutputShape(layer, current_shape);
            current_shape = layer.output_shape;

            config.layers.push_back(layer);

            // Track output size from last Dense layer
            if (node->type == gui::NodeType::Dense) {
                config.output_size = layer.units;
            }
        }
    }

    // Extract loss configuration
    if (loss_node) {
        config.loss_type = loss_node->type;
        config.loss_params = loss_node->parameters;
    }

    // Extract optimizer configuration
    if (optimizer_node) {
        config.optimizer_type = optimizer_node->type;

        if (optimizer_node->parameters.count("learning_rate"))
            config.learning_rate = std::stof(optimizer_node->parameters.at("learning_rate"));
        if (optimizer_node->parameters.count("lr"))
            config.learning_rate = std::stof(optimizer_node->parameters.at("lr"));
        if (optimizer_node->parameters.count("momentum"))
            config.momentum = std::stof(optimizer_node->parameters.at("momentum"));
        if (optimizer_node->parameters.count("beta1"))
            config.beta1 = std::stof(optimizer_node->parameters.at("beta1"));
        if (optimizer_node->parameters.count("beta2"))
            config.beta2 = std::stof(optimizer_node->parameters.at("beta2"));
        if (optimizer_node->parameters.count("weight_decay"))
            config.weight_decay = std::stof(optimizer_node->parameters.at("weight_decay"));
    }

    // Set one-hot encoding if we have classification (CrossEntropy loss)
    if (config.loss_type == gui::NodeType::CrossEntropyLoss) {
        config.preprocessing.has_onehot = true;

        // Try to get num_classes from Output node's "classes" parameter
        const gui::MLNode* output_node = FindOutputNode(nodes);
        if (output_node) {
            auto it = output_node->parameters.find("classes");
            if (it != output_node->parameters.end() && !it->second.empty()) {
                try {
                    config.preprocessing.num_classes = std::stoul(it->second);
                    spdlog::info("GraphCompiler: num_classes={} from Output node",
                                 config.preprocessing.num_classes);
                } catch (...) {
                    config.preprocessing.num_classes = 0;  // Will fall back below
                }
            }
        }

        // Fallback to output_size if Output node doesn't specify classes
        if (config.preprocessing.num_classes == 0) {
            config.preprocessing.num_classes = config.output_size;
            spdlog::warn("GraphCompiler: num_classes not specified in Output node, using output_size={}",
                         config.output_size);
        }
    }

    // === Post-compile sanity checks ===
    // These need the values populated by the layer-extraction passes
    // above, so they live at the end of Compile().

    // DataSplit ratios should sum to ~1.0. Drift > 0.05 is almost
    // certainly a typo or stale state from the user adjusting one
    // ratio without rebalancing the others.
    if (config.has_data_split) {
        float sum = config.train_ratio + config.val_ratio + config.test_ratio;
        if (std::abs(sum - 1.0f) > 0.05f) {
            std::ostringstream msg;
            msg << "DataSplit ratios sum to " << sum
                << " (expected 1.0) - train=" << config.train_ratio
                << ", val=" << config.val_ratio
                << ", test=" << config.test_ratio;
            AddIssue(config, IssueLevel::Warning, msg.str());
        }
        if (config.val_ratio < 1e-6f) {
            AddIssue(config, IssueLevel::Warning,
                     "Validation split is 0 - training will run without validation metrics");
        }
    }

    // batch_size must fit in the train set. We can only check the upper
    // bound when the dataset is actually loaded — otherwise the row count
    // is unknown and the existing data_loaded check above already fired.
    if (config.batch_size <= 0) {
        AddIssue(config, IssueLevel::Error,
                 "batch_size must be positive (got " +
                 std::to_string(config.batch_size) + ")");
    } else if (dataset_node && !config.dataset_name.empty()) {
        auto& reg = DataRegistry::Instance();
        int64_t total_rows = 0;
        if (auto arrow_ds = reg.GetArrowDataset(config.dataset_name)) {
            total_rows = arrow_ds->GetNumRows();
        } else if (auto pq_ds = reg.GetParquetBackedDataset(config.dataset_name)) {
            total_rows = pq_ds->GetNumRows();
        }
        if (total_rows > 0) {
            int64_t train_rows = static_cast<int64_t>(total_rows * config.train_ratio);
            if (config.batch_size > train_rows) {
                AddIssue(config, IssueLevel::Error,
                         "batch_size (" + std::to_string(config.batch_size) +
                         ") is larger than the train split (" +
                         std::to_string(train_rows) + " rows)");
            } else if (config.batch_size > train_rows / 2) {
                AddIssue(config, IssueLevel::Warning,
                         "batch_size (" + std::to_string(config.batch_size) +
                         ") is more than half the train split (" +
                         std::to_string(train_rows) + " rows) - few iterations per epoch");
            }
        }
    }

    // === Image-specific checks ===
    // Detect the data domain from the DataInput node's file_category
    // parameter. If it's an image dataset, enforce image-specific rules
    // that prevent crashes at training time (OOM, shape mismatch).
    if (dataset_node) {
        auto cat_it = dataset_node->parameters.find("file_category");
        bool is_image = (cat_it != dataset_node->parameters.end() &&
                         cat_it->second == "image");

        if (is_image) {
            config.preprocessing_domain = PreprocessingDomain::Image;

            // Check 1: Resize node required for image datasets.
            // Without it, images load at their native (variable) size,
            // which can't be batched AND can cause massive tensors that
            // OOM the GPU. This is the most critical image check.
            bool has_resize = false;
            for (const auto& node : nodes) {
                if (node.type == gui::NodeType::Resize) {
                    has_resize = true;
                    break;
                }
            }
            if (!has_resize) {
                AddIssue(config, IssueLevel::Error,
                         "Image datasets require a Resize node to set target "
                         "dimensions. Without it, images load at their native "
                         "size which can exceed GPU memory. Add a Resize node "
                         "between DataInput and the first layer (e.g. "
                         "Resize 64x64 for fast training, 224x224 for accuracy).",
                         dataset_node->id, dataset_node->name);
            }

            // Check 2: Flatten required before Dense layers for images.
            // Images are 3D (H×W×C); Dense expects a 1D flat vector.
            if (!config.layers.empty()) {
                auto first_layer = config.layers[0];
                if (first_layer.type == gui::NodeType::Dense) {
                    bool has_flatten_before_dense = false;
                    for (const auto& node : nodes) {
                        if (node.type == gui::NodeType::Flatten) {
                            has_flatten_before_dense = true;
                            break;
                        }
                    }
                    if (!has_flatten_before_dense) {
                        AddIssue(config, IssueLevel::Error,
                                 "First model layer is Dense but no Flatten node "
                                 "found. Image data is 3D (H x W x C) and needs "
                                 "Flatten before Dense layers.",
                                 first_layer.node_id, first_layer.name);
                    }
                }
            }

            // Check 3: Memory estimation for image models.
            // Estimate total model memory (weights + gradients + optimizer)
            // and warn if it's likely to exceed reasonable GPU memory.
            if (has_resize && config.image_preprocessing.target_width > 0) {
                size_t img_features = static_cast<size_t>(
                    config.image_preprocessing.target_width) *
                    config.image_preprocessing.target_height * 3;
                size_t total_params = 0;
                size_t prev_size = img_features;
                for (const auto& layer : config.layers) {
                    if (layer.type == gui::NodeType::Dense) {
                        total_params += prev_size * layer.units + layer.units;
                        prev_size = layer.units;
                    } else if (layer.type == gui::NodeType::Flatten) {
                        // prev_size stays the same (just reshaped)
                    }
                }
                // Memory = weights × 4 (weights + grads + adam_m + adam_v)
                // + batch activations (batch_size × largest_layer)
                size_t model_bytes = total_params * 4 * sizeof(float);
                size_t batch_bytes = static_cast<size_t>(config.batch_size) *
                    img_features * sizeof(float);
                size_t total_est = model_bytes + batch_bytes;

                double est_mb = total_est / (1024.0 * 1024.0);

                std::ostringstream mem_msg;
                mem_msg << "Estimated GPU memory: " << std::fixed
                        << std::setprecision(0) << est_mb << " MB ("
                        << total_params << " parameters, batch_size="
                        << config.batch_size << ", input="
                        << config.image_preprocessing.target_width << "x"
                        << config.image_preprocessing.target_height << "x3)";

                // Real GPU usage is ~3-5x the raw parameter estimate due to
                // CUDA context (~300 MB), forward/backward activations,
                // matmul temporaries, and optimizer state. Use a 4x
                // multiplier for a realistic bound.
                double realistic_mb = est_mb * 4.0;

                if (realistic_mb > 3000) {
                    // Suggest a smaller size that would fit
                    int safe_features = 64 * 64 * 3;  // 12288
                    mem_msg << ". WARNING: real GPU usage is ~"
                            << std::setprecision(0) << realistic_mb
                            << " MB (incl. activations + optimizer + CUDA "
                            << "overhead). This will likely OOM on your GPU. "
                            << "Set the Resize node to a smaller size "
                            << "(e.g. 64x64 for " << safe_features
                            << " features)";
                    AddIssue(config, IssueLevel::Error, mem_msg.str());
                } else if (realistic_mb > 2000) {
                    mem_msg << ". This is close to typical GPU memory limits "
                            << "— consider reducing Resize dimensions or "
                            << "batch_size if training crashes";
                    AddIssue(config, IssueLevel::Warning, mem_msg.str());
                } else {
                    AddIssue(config, IssueLevel::Info, mem_msg.str());
                }
            }

            // Check 4: Pipeline ordering — Normalize before Resize is
            // almost certainly wrong (normalization stats are scale-
            // dependent). Walk the TOPOLOGICAL order (from links), not
            // the raw node vector (creation order), so the check reflects
            // the actual data flow.
            {
                std::vector<int> topo = TopologicalSort(nodes, links);
                bool seen_normalize = false;
                for (int nid : topo) {
                    const gui::MLNode* n = FindNodeById(nid, nodes);
                    if (!n) continue;
                    if (n->type == gui::NodeType::Normalize) {
                        seen_normalize = true;
                    }
                    if (n->type == gui::NodeType::Resize && seen_normalize) {
                        AddIssue(config, IssueLevel::Warning,
                                 "Normalize appears before Resize in the pipeline. "
                                 "Normalization is scale-dependent — resizing after "
                                 "normalization gives wrong per-pixel values. Move "
                                 "Resize before Normalize.");
                        break;
                    }
                }
            }
        }
    }

    // Final verdict: is_valid is the absence of any Error-level issue.
    // Warnings and Info don't block training.
    config.is_valid = !config.HasErrors();
    config.error_message = JoinErrorMessages(config.issues);

    spdlog::info("GraphCompiler: Compiled {} layers, input_size={}, output_size={}, "
                 "issues: {} errors / {} warnings / {} info, valid={}",
                 config.layers.size(), config.input_size, config.output_size,
                 config.CountIssues(IssueLevel::Error),
                 config.CountIssues(IssueLevel::Warning),
                 config.CountIssues(IssueLevel::Info),
                 config.is_valid);
    for (const auto& issue : config.issues) {
        spdlog::info("  [{}] {}{}",
                     IssueLevelLabel(issue.level),
                     issue.node_name.empty() ? "" : ("[" + issue.node_name + "] "),
                     issue.message);
    }

    return config;
}

bool GraphCompiler::ValidateGraph(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::string& error)
{
    // Check for empty graph
    if (nodes.empty()) {
        error = "Graph is empty - add nodes to create a model";
        return false;
    }

    // Check for dataset input
    if (!FindDatasetInputNode(nodes)) {
        error = "Graph must have a DatasetInput node";
        return false;
    }

    // Check for at least one model layer
    bool has_model_layer = false;
    for (const auto& node : nodes) {
        if (IsModelLayer(node.type)) {
            has_model_layer = true;
            break;
        }
    }
    if (!has_model_layer) {
        error = "Graph must have at least one model layer (Dense, Conv2D, etc.)";
        return false;
    }

    // Check for loss function
    if (!FindLossNode(nodes)) {
        error = "Graph must have a loss function (MSELoss or CrossEntropyLoss)";
        return false;
    }

    // Check for optimizer
    if (!FindOptimizerNode(nodes)) {
        error = "Graph must have an optimizer (SGD, Adam, or AdamW)";
        return false;
    }

    // Check for cycles
    if (HasCycle(nodes, links)) {
        error = "Graph contains a cycle - remove circular connections";
        return false;
    }

    return true;
}

std::vector<int> GraphCompiler::TopologicalSort(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    // Build adjacency list and in-degree count
    std::map<int, std::vector<int>> adj;
    std::map<int, int> in_degree;

    for (const auto& node : nodes) {
        adj[node.id] = {};
        in_degree[node.id] = 0;
    }

    for (const auto& link : links) {
        adj[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Kahn's algorithm
    std::queue<int> queue;
    for (const auto& node : nodes) {
        if (in_degree[node.id] == 0) {
            queue.push(node.id);
        }
    }

    std::vector<int> sorted;
    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();
        sorted.push_back(node_id);

        for (int neighbor : adj[node_id]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    return sorted;
}

const gui::MLNode* GraphCompiler::FindNodeById(int id, const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

std::vector<int> GraphCompiler::GetConnectedNodes(
    int from_node_id,
    const std::vector<gui::NodeLink>& links) const
{
    std::vector<int> connected;
    for (const auto& link : links) {
        if (link.from_node == from_node_id) {
            connected.push_back(link.to_node);
        }
    }
    return connected;
}

std::vector<int> GraphCompiler::GetInputNodes(
    int to_node_id,
    const std::vector<gui::NodeLink>& links) const
{
    std::vector<int> inputs;
    for (const auto& link : links) {
        if (link.to_node == to_node_id) {
            inputs.push_back(link.from_node);
        }
    }
    return inputs;
}

const gui::MLNode* GraphCompiler::FindDatasetInputNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        // Both smart DataInput and legacy DatasetInput nodes are valid data sources
        if (node.type == gui::NodeType::DataInput || node.type == gui::NodeType::DatasetInput) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindLossNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::MSELoss ||
            node.type == gui::NodeType::CrossEntropyLoss ||
            node.type == gui::NodeType::BCELoss ||
            node.type == gui::NodeType::BCEWithLogits ||
            node.type == gui::NodeType::L1Loss ||
            node.type == gui::NodeType::SmoothL1Loss ||
            node.type == gui::NodeType::HuberLoss ||
            node.type == gui::NodeType::NLLLoss) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindOptimizerNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::SGD ||
            node.type == gui::NodeType::Adam ||
            node.type == gui::NodeType::AdamW) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindOutputNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::Output) {
            return &node;
        }
    }
    return nullptr;
}

bool GraphCompiler::IsModelLayer(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::Conv2D:
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::Flatten:
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::ConvTranspose2D:
        case gui::NodeType::Upsample:
        case gui::NodeType::PixelShuffle:
        case gui::NodeType::PolicyNetwork:
        case gui::NodeType::ValueNetwork:
            return true;
        default:
            return false;
    }
}

bool GraphCompiler::IsActivation(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Table-driven preprocessing registry
//
// Each entry maps a NodeType to a preprocessing domain and an optional
// extraction function. IsPreprocessing and ExtractPreprocessing both
// consult this table instead of maintaining separate switch statements.
//
// Adding a new preprocessing node for any domain is a single line here
// plus a static extraction function. No switch cases to update in
// multiple places.
// ---------------------------------------------------------------------------

namespace {

using ExtractorFn = void(*)(const gui::MLNode&, TrainingConfiguration&);

struct PreprocessingNodeSpec {
    gui::NodeType type;
    PreprocessingDomain domain;
    ExtractorFn extractor;  // nullptr = recognized but not yet wired
};

// --- Tabular extractors (migrated from the old switch) ---

static void ExtractNormalize(const gui::MLNode& node, TrainingConfiguration& config) {
    config.preprocessing.has_normalization = true;
    if (node.parameters.count("mean"))
        config.preprocessing.norm_mean = std::stof(node.parameters.at("mean"));
    if (node.parameters.count("std"))
        config.preprocessing.norm_std = std::stof(node.parameters.at("std"));
}

static void ExtractReshape(const gui::MLNode& node, TrainingConfiguration& config) {
    config.preprocessing.has_reshape = true;
    if (node.parameters.count("shape")) {
        std::string shape_str = node.parameters.at("shape");
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

        size_t pos = 0;
        while ((pos = shape_str.find(',')) != std::string::npos) {
            config.preprocessing.reshape_dims.push_back(std::stoi(shape_str.substr(0, pos)));
            shape_str.erase(0, pos + 1);
        }
        if (!shape_str.empty()) {
            config.preprocessing.reshape_dims.push_back(std::stoi(shape_str));
        }
    }
}

static void ExtractOneHot(const gui::MLNode& node, TrainingConfiguration& config) {
    config.preprocessing.has_onehot = true;
    if (node.parameters.count("num_classes"))
        config.preprocessing.num_classes = std::stoul(node.parameters.at("num_classes"));
}

// --- Image extractors (Phase 1) ---

static void ExtractImageResize(const gui::MLNode& node, TrainingConfiguration& config) {
    config.image_preprocessing.resize_mode = ResizeMode::Exact;
    if (node.parameters.count("width"))
        config.image_preprocessing.target_width = std::stoi(node.parameters.at("width"));
    if (node.parameters.count("height"))
        config.image_preprocessing.target_height = std::stoi(node.parameters.at("height"));
    if (node.parameters.count("mode")) {
        std::string mode = node.parameters.at("mode");
        if (mode == "exact")       config.image_preprocessing.resize_mode = ResizeMode::Exact;
        else if (mode == "fit")    config.image_preprocessing.resize_mode = ResizeMode::AspectFit;
        else if (mode == "fill")   config.image_preprocessing.resize_mode = ResizeMode::AspectFill;
        else if (mode == "center") config.image_preprocessing.resize_mode = ResizeMode::Center;
    }
    spdlog::info("GraphCompiler: Resize {}x{} mode={}",
                 config.image_preprocessing.target_width,
                 config.image_preprocessing.target_height,
                 static_cast<int>(config.image_preprocessing.resize_mode));
}

static void ExtractImageNormalize(const gui::MLNode& node, TrainingConfiguration& config) {
    // Domain-aware Normalize: when upstream is image data, populate
    // both the legacy GraphPreprocessingConfig (for backward-compat
    // with the tabular training path) and the image-specific config.
    // The image batcher reads image_preprocessing; the tabular batcher
    // reads config.preprocessing. Both work.
    if (config.preprocessing_domain == PreprocessingDomain::Image) {
        config.image_preprocessing.enable_denoise = false;  // not related but clear the field
        // Image normalize is handled by the ImageTransformPipeline;
        // the legacy fields are also set so existing tabular code paths
        // don't break if they run accidentally.
    }
    // Always populate the legacy tabular fields (backward-compat).
    config.preprocessing.has_normalization = true;
    if (node.parameters.count("mean"))
        config.preprocessing.norm_mean = std::stof(node.parameters.at("mean"));
    if (node.parameters.count("std"))
        config.preprocessing.norm_std = std::stof(node.parameters.at("std"));
}

static void ExtractGrayscale(const gui::MLNode& /*node*/, TrainingConfiguration& config) {
    config.image_preprocessing.convert_to_grayscale = true;
}

static void ExtractImageGaussianBlur(const gui::MLNode& node, TrainingConfiguration& config) {
    config.image_preprocessing.blur_config.enabled = true;
    config.image_preprocessing.blur_config.type = BlurType::Gaussian;
    if (node.parameters.count("kernel_size"))
        config.image_preprocessing.blur_config.kernel_size = std::stoi(node.parameters.at("kernel_size"));
    if (node.parameters.count("sigma"))
        config.image_preprocessing.blur_config.sigma = std::stof(node.parameters.at("sigma"));
}

// --- Audio / Text / TimeSeries extractors (Phase 2-5 — deferred) ---

// --- The table ---

static const PreprocessingNodeSpec kPreprocessingSpecs[] = {
    // Tabular (existing, migrated from switch)
    {gui::NodeType::Normalize,          PreprocessingDomain::Tabular,     ExtractImageNormalize},
    {gui::NodeType::TensorReshape,      PreprocessingDomain::Tabular,     ExtractReshape},
    {gui::NodeType::OneHotEncode,       PreprocessingDomain::Tabular,     ExtractOneHot},
    // General (domain-agnostic data pipeline nodes — no extraction needed)
    {gui::NodeType::DataSplit,          PreprocessingDomain::General,     nullptr},
    {gui::NodeType::DataLoader,         PreprocessingDomain::General,     nullptr},
    // Image (Phase 1)
    {gui::NodeType::Resize,             PreprocessingDomain::Image,       ExtractImageResize},
    {gui::NodeType::CenterCrop,         PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::RandomCrop,         PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::HorizontalFlip,     PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::VerticalFlip,       PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ImageRotate,        PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ColorJitter,        PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ImageGaussianBlur,  PreprocessingDomain::Image,       ExtractImageGaussianBlur},
    {gui::NodeType::Grayscale,          PreprocessingDomain::Image,       ExtractGrayscale},
    {gui::NodeType::Augmentation,       PreprocessingDomain::Image,       nullptr},
    // Audio (Phase 2)
    {gui::NodeType::AudioInput,         PreprocessingDomain::Audio,       nullptr},
    {gui::NodeType::Spectrogram,        PreprocessingDomain::Audio,       nullptr},
    {gui::NodeType::MelSpectrogram,     PreprocessingDomain::Audio,       nullptr},
    {gui::NodeType::MFCC,               PreprocessingDomain::Audio,       nullptr},
    {gui::NodeType::AudioAugmentation,  PreprocessingDomain::Audio,       nullptr},
    // Text (Phase 3)
    {gui::NodeType::TextTokenizer,      PreprocessingDomain::Text,        nullptr},
    {gui::NodeType::TextVocabulary,     PreprocessingDomain::Text,        nullptr},
    {gui::NodeType::TextPadding,        PreprocessingDomain::Text,        nullptr},
    // TimeSeries (Phase 4)
    {gui::NodeType::TimeSeriesWindow,   PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::TimeSeriesFeatures, PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::TimeSeriesSplit,    PreprocessingDomain::TimeSeries,  nullptr},
};

} // anonymous namespace

bool GraphCompiler::IsPreprocessing(gui::NodeType type) const {
    for (const auto& spec : kPreprocessingSpecs) {
        if (spec.type == type) return true;
    }
    return false;
}

CompiledLayer GraphCompiler::ExtractLayerConfig(const gui::MLNode& node) const {
    CompiledLayer layer;
    layer.type = node.type;
    layer.node_id = node.id;
    layer.name = node.name;
    layer.parameters = node.parameters;

    // Extract specific parameters
    switch (node.type) {
        case gui::NodeType::Dense:
            if (node.parameters.count("units"))
                layer.units = std::stoi(node.parameters.at("units"));
            break;

        case gui::NodeType::Conv2D:
            if (node.parameters.count("filters"))
                layer.filters = std::stoi(node.parameters.at("filters"));
            if (node.parameters.count("kernel_size"))
                layer.kernel_size = std::stoi(node.parameters.at("kernel_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            if (node.parameters.count("padding"))
                layer.padding = std::stoi(node.parameters.at("padding"));
            break;

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            if (node.parameters.count("pool_size"))
                layer.pool_size = std::stoi(node.parameters.at("pool_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            break;

        case gui::NodeType::BatchNorm:
            if (node.parameters.count("eps"))
                layer.eps = std::stof(node.parameters.at("eps"));
            if (node.parameters.count("momentum"))
                layer.momentum = std::stof(node.parameters.at("momentum"));
            break;

        case gui::NodeType::Dropout:
            if (node.parameters.count("rate"))
                layer.dropout_rate = std::stof(node.parameters.at("rate"));
            break;

        case gui::NodeType::LeakyReLU:
            if (node.parameters.count("negative_slope"))
                layer.negative_slope = std::stof(node.parameters.at("negative_slope"));
            break;

        case gui::NodeType::ELU:
            if (node.parameters.count("alpha"))
                layer.alpha = std::stof(node.parameters.at("alpha"));
            break;

        case gui::NodeType::ConvTranspose2D:
            if (node.parameters.count("in_channels"))
                layer.in_channels = std::stoi(node.parameters.at("in_channels"));
            if (node.parameters.count("out_channels"))
                layer.filters = std::stoi(node.parameters.at("out_channels"));
            if (node.parameters.count("kernel_size"))
                layer.kernel_size = std::stoi(node.parameters.at("kernel_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            if (node.parameters.count("padding"))
                layer.padding = std::stoi(node.parameters.at("padding"));
            if (node.parameters.count("output_padding"))
                layer.output_padding = std::stoi(node.parameters.at("output_padding"));
            break;

        case gui::NodeType::Upsample:
            if (node.parameters.count("scale_factor"))
                layer.scale_factor = std::stoi(node.parameters.at("scale_factor"));
            if (node.parameters.count("mode"))
                layer.upsample_mode = std::stoi(node.parameters.at("mode"));
            break;

        case gui::NodeType::PixelShuffle:
            if (node.parameters.count("upscale_factor"))
                layer.scale_factor = std::stoi(node.parameters.at("upscale_factor"));
            break;

        default:
            break;
    }

    return layer;
}

void GraphCompiler::ExtractPreprocessing(
    const gui::MLNode& node,
    TrainingConfiguration& config) const
{
    for (const auto& spec : kPreprocessingSpecs) {
        if (spec.type == node.type) {
            if (spec.extractor) {
                spec.extractor(node, config);
            }
            return;
        }
    }
}

std::vector<size_t> GraphCompiler::InferOutputShape(
    const CompiledLayer& layer,
    const std::vector<size_t>& input_shape) const
{
    std::vector<size_t> output_shape;

    switch (layer.type) {
        case gui::NodeType::Dense:
            // Dense: [...] -> [units]
            output_shape = {static_cast<size_t>(layer.units)};
            break;

        case gui::NodeType::Conv2D:
            // Conv2D: [H, W, C] -> [(H + 2*padding - kernel_size) / stride + 1, W', filters]
            if (input_shape.size() >= 2) {
                size_t out_h = (input_shape[0] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                size_t out_w = (input_shape[1] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                output_shape = {out_h, out_w, static_cast<size_t>(layer.filters)};
            }
            break;

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            // Pool2D: [H, W, C] -> [H/pool_size, W/pool_size, C]
            if (input_shape.size() >= 3) {
                int stride = layer.stride > 0 ? layer.stride : layer.pool_size;
                size_t out_h = (input_shape[0] - layer.pool_size) / stride + 1;
                size_t out_w = (input_shape[1] - layer.pool_size) / stride + 1;
                output_shape = {out_h, out_w, input_shape[2]};
            }
            break;

        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
            // Global pooling: [H, W, C] -> [C]
            if (input_shape.size() >= 3) {
                output_shape = {input_shape[2]};  // Just channels remain
            } else if (!input_shape.empty()) {
                output_shape = {input_shape.back()};
            }
            break;

        case gui::NodeType::Flatten:
            // Flatten: [H, W, C] -> [H*W*C]
            {
                size_t flat_size = 1;
                for (size_t dim : input_shape) flat_size *= dim;
                output_shape = {flat_size};
            }
            break;

        // Layers/activations that don't change shape
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            output_shape = input_shape;
            break;

        default:
            output_shape = input_shape;
            break;
    }

    return output_shape;
}

bool GraphCompiler::HasCycle(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const
{
    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& node : nodes) {
        adj[node.id] = {};
    }
    for (const auto& link : links) {
        adj[link.from_node].push_back(link.to_node);
    }

    // DFS-based cycle detection
    std::set<int> white;  // Not visited
    std::set<int> gray;   // Currently visiting
    std::set<int> black;  // Fully processed

    for (const auto& node : nodes) {
        white.insert(node.id);
    }

    std::function<bool(int)> dfs = [&](int node_id) -> bool {
        white.erase(node_id);
        gray.insert(node_id);

        for (int neighbor : adj[node_id]) {
            if (gray.count(neighbor)) {
                // Back edge found - cycle exists
                return true;
            }
            if (white.count(neighbor) && dfs(neighbor)) {
                return true;
            }
        }

        gray.erase(node_id);
        black.insert(node_id);
        return false;
    };

    while (!white.empty()) {
        int node_id = *white.begin();
        if (dfs(node_id)) {
            return true;
        }
    }

    return false;
}

bool GraphCompiler::IsFullyConnected(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const
{
    if (nodes.empty()) return true;

    // BFS from first node
    std::set<int> visited;
    std::queue<int> queue;

    // Build undirected adjacency
    std::map<int, std::set<int>> adj;
    for (const auto& node : nodes) {
        adj[node.id] = {};
    }
    for (const auto& link : links) {
        adj[link.from_node].insert(link.to_node);
        adj[link.to_node].insert(link.from_node);
    }

    queue.push(nodes[0].id);
    visited.insert(nodes[0].id);

    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();

        for (int neighbor : adj[node_id]) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }

    return visited.size() == nodes.size();
}

} // namespace cyxwiz
