#include "node_editor.h"
#include "panels/script_editor.h"
#include "../core/async_task_manager.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>
#include <set>
#include <map>
#include <queue>
#include <functional>

namespace gui {

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
    spdlog::info("Generating code from node graph (async)...");

    if (nodes_.empty()) {
        spdlog::warn("No nodes in graph - cannot generate code");
        return;
    }

    // Get topologically sorted node order (do this synchronously for validation)
    std::vector<int> sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to perform topological sort - graph may have cycles");
        return;
    }

    // Copy graph data for thread safety
    std::vector<MLNode> nodes_copy = nodes_;
    std::vector<NodeLink> links_copy = links_;
    size_t total_nodes = sorted_ids.size();

    // Determine framework name
    std::string framework_name;
    switch (framework) {
        case CodeFramework::PyTorch: framework_name = "PyTorch"; break;
        case CodeFramework::TensorFlow: framework_name = "TensorFlow"; break;
        case CodeFramework::Keras: framework_name = "Keras"; break;
        case CodeFramework::PyCyxWiz: framework_name = "PyCyxWiz"; break;
        default: framework_name = "Unknown"; break;
    }

    // Store result for completion callback
    auto result = std::make_shared<std::string>();
    auto fw_name = std::make_shared<std::string>(framework_name);

    // Capture script_editor_ for completion callback
    auto script_editor = script_editor_;

    // Run code generation async
    cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Generate " + framework_name + " Code",
        [this, framework, sorted_ids, nodes_copy, total_nodes, result, fw_name](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.0f, "Starting code generation...");

            std::string code;

            // Generate code based on selected framework
            task.ReportProgress(0.1f, "Generating " + *fw_name + " code...");

            switch (framework) {
                case CodeFramework::PyTorch:
                    code = GeneratePyTorchCode(sorted_ids);
                    break;
                case CodeFramework::TensorFlow:
                    code = GenerateTensorFlowCode(sorted_ids);
                    break;
                case CodeFramework::Keras:
                    code = GenerateKerasCode(sorted_ids);
                    break;
                case CodeFramework::PyCyxWiz:
                    code = GeneratePyCyxWizCode(sorted_ids);
                    break;
                default:
                    task.MarkFailed("Unknown framework selected");
                    return;
            }

            if (task.ShouldStop()) {
                task.MarkFailed("Code generation cancelled");
                return;
            }

            task.ReportProgress(0.9f, "Finalizing...");

            // Store result
            *result = std::move(code);

            task.ReportProgress(1.0f, "Complete!");
            spdlog::info("Generated {} code ({} lines)", *fw_name, std::count(result->begin(), result->end(), '\n'));
        },
        // Progress callback (optional - can be used for detailed UI updates)
        nullptr,
        // Completion callback - runs on main thread
        [script_editor, result, fw_name](bool success, const std::string& error) {
            if (success && script_editor) {
                script_editor->LoadGeneratedCode(*result, fw_name->c_str());
                script_editor->SetVisible(true);
                spdlog::info("Code sent to Script Editor panel");
            } else if (!success) {
                spdlog::error("Code generation failed: {}", error);
            } else {
                spdlog::warn("Script Editor panel not available");
            }
        }
    );
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

        // Skip DatasetInput and Output nodes in __init__ (they don't have layers)
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
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
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input)\n";
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

        // Skip DatasetInput and Output nodes in __init__
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
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
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input)\n";
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

        // Skip DatasetInput node
        if (node->type == NodeType::DatasetInput) {
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

        // Skip DatasetInput and Output nodes in __init__
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
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
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input tensor)\n";
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

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string feature_map = "elu";
            std::string eps = "1e-6";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("feature_map");
            if (it != node.parameters.end()) feature_map = it->second;
            it = node.parameters.find("eps");
            if (it != node.parameters.end()) eps = it->second;
            // Linear attention with O(n) complexity (Performer-style)
            // Requires: pip install performer-pytorch or custom implementation
            code = "LinearAttention(dim=" + embed_dim + ", heads=" + num_heads +
                   ", dim_head=" + embed_dim + "//" + num_heads +
                   ", feature_map='" + feature_map + "', eps=" + eps + ")";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "nn.MultiheadAttention(embed_dim=" + embed_dim + ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "nn.LayerNorm(" + normalized_shape + ")";
            break;
        }

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "nn.Embedding(num_embeddings=" + num_embeddings + ", embedding_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "nn.GELU()";
            break;

        case NodeType::ReLU:
            code = "nn.ReLU()";
            break;

        default:
            // Activation functions and others don't need layers in __init__
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToTensorFlowLayer(const MLNode& node, int /*layer_idx*/) {
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

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            // TensorFlow doesn't have native linear attention - use MultiHeadAttention or custom layer
            // Comment indicates O(n) linear attention should be used
            code = "# LinearAttention (O(n)) - requires tensorflow-addons or custom impl\n"
                   "        layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")  # Replace with linear attention";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "layers.LayerNormalization()";
            break;
        }

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "layers.Embedding(input_dim=" + num_embeddings + ", output_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "layers.Activation('gelu')";
            break;

        case NodeType::ReLU:
            code = "layers.ReLU()";
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

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            // Keras uses same MultiHeadAttention as TensorFlow
            code = "# LinearAttention (O(n)) - requires custom implementation\n"
                   "        layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")  # Replace with linear attention";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm:
            code = "layers.LayerNormalization()";
            break;

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "layers.Embedding(input_dim=" + num_embeddings + ", output_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "layers.Activation('gelu')";
            break;

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

        // ===== Attention & Transformer Layers =====
        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "cx.LinearAttention(dim=" + embed_dim + ", heads=" + num_heads + ")";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "cx.MultiHeadAttention(embed_dim=" + embed_dim + ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::SelfAttention: {
            std::string embed_dim = "512";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            code = "cx.SelfAttention(embed_dim=" + embed_dim + ")";
            break;
        }

        case NodeType::CrossAttention: {
            std::string embed_dim = "512";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            code = "cx.CrossAttention(embed_dim=" + embed_dim + ")";
            break;
        }

        case NodeType::TransformerEncoder: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string ff_dim = "2048";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("ff_dim");
            if (it != node.parameters.end()) ff_dim = it->second;
            code = "cx.TransformerEncoder(d_model=" + embed_dim + ", nhead=" + num_heads + ", dim_feedforward=" + ff_dim + ")";
            break;
        }

        case NodeType::TransformerDecoder: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string ff_dim = "2048";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("ff_dim");
            if (it != node.parameters.end()) ff_dim = it->second;
            code = "cx.TransformerDecoder(d_model=" + embed_dim + ", nhead=" + num_heads + ", dim_feedforward=" + ff_dim + ")";
            break;
        }

        // ===== Normalization Layers =====
        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "cx.LayerNorm(normalized_shape=" + normalized_shape + ")";
            break;
        }

        case NodeType::GroupNorm: {
            std::string num_groups = "32";
            std::string num_channels = "256";
            auto it = node.parameters.find("num_groups");
            if (it != node.parameters.end()) num_groups = it->second;
            it = node.parameters.find("num_channels");
            if (it != node.parameters.end()) num_channels = it->second;
            code = "cx.GroupNorm(num_groups=" + num_groups + ", num_channels=" + num_channels + ")";
            break;
        }

        case NodeType::InstanceNorm:
            code = "cx.InstanceNorm()";
            break;

        // ===== Embedding Layer =====
        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "cx.Embedding(num_embeddings=" + num_embeddings + ", embedding_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::PositionalEncoding: {
            std::string max_len = "5000";
            std::string d_model = "512";
            auto it = node.parameters.find("max_len");
            if (it != node.parameters.end()) max_len = it->second;
            it = node.parameters.find("d_model");
            if (it != node.parameters.end()) d_model = it->second;
            code = "cx.PositionalEncoding(d_model=" + d_model + ", max_len=" + max_len + ")";
            break;
        }

        // ===== Activation Functions =====
        case NodeType::ReLU:
            code = "cx.ReLU()";
            break;

        case NodeType::GELU:
            code = "cx.GELU()";
            break;

        case NodeType::LeakyReLU: {
            std::string negative_slope = "0.01";
            auto it = node.parameters.find("negative_slope");
            if (it != node.parameters.end()) negative_slope = it->second;
            code = "cx.LeakyReLU(negative_slope=" + negative_slope + ")";
            break;
        }

        case NodeType::Swish:
            code = "cx.Swish()";
            break;

        case NodeType::Mish:
            code = "cx.Mish()";
            break;

        case NodeType::Sigmoid:
            code = "cx.Sigmoid()";
            break;

        case NodeType::Tanh:
            code = "cx.Tanh()";
            break;

        case NodeType::Softmax: {
            std::string dim = "-1";
            auto it = node.parameters.find("dim");
            if (it != node.parameters.end()) dim = it->second;
            code = "cx.Softmax(dim=" + dim + ")";
            break;
        }

        // ===== Recurrent Layers =====
        case NodeType::LSTM: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            code = "cx.LSTM(input_size=" + input_size + ", hidden_size=" + hidden_size + ", num_layers=" + num_layers + ")";
            break;
        }

        case NodeType::GRU: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            code = "cx.GRU(input_size=" + input_size + ", hidden_size=" + hidden_size + ", num_layers=" + num_layers + ")";
            break;
        }

        // ===== Shape Operations =====
        case NodeType::Flatten:
            code = "cx.Flatten()";
            break;

        case NodeType::Reshape:
            code = "cx.Reshape(shape=AUTO)";  // Shape determined from graph
            break;

        // ===== Merge Operations =====
        case NodeType::Add:
            code = "cx.Add()";
            break;

        case NodeType::Concatenate: {
            std::string dim = "1";
            auto it = node.parameters.find("dim");
            if (it != node.parameters.end()) dim = it->second;
            code = "cx.Concatenate(dim=" + dim + ")";
            break;
        }

        default:
            // Other node types handled in forward pass or not yet implemented
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


} // namespace gui
