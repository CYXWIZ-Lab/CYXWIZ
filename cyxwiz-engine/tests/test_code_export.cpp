// The graph code export (core/code_export, TOFIX228 1.5a): GUI-free, the same
// text for the same graph, the PyTorch module each node becomes, and the
// refusals (empty graph, cycle, a layer the export cannot write).
#include "../src/core/code_export.h"
#include "../src/core/graph_node_factory.h"

#include <iostream>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node = 1;
    int next_pin = 1;
    int next_link = 1;

    int Add(gui::NodeType type, const std::string& name, std::map<std::string, std::string> params = {}) {
        gui::MLNode node = gui::CreateGraphNode(type, name, next_node, next_pin);
        for (auto& [key, value] : params) node.parameters[key] = value;
        nodes.push_back(node);
        return node.id;
    }
    const gui::MLNode& Node(int id) const {
        for (const auto& node : nodes)
            if (node.id == id) return node;
        throw std::runtime_error("no node " + std::to_string(id));
    }
    // From output `out` of `from` to input `in` of `to`.
    void Connect(int from, int to, size_t out = 0, size_t in = 0) {
        gui::NodeLink link;
        link.id = next_link++;
        link.from_node = from;
        link.from_pin = Node(from).outputs.at(out).id;
        link.to_node = to;
        link.to_pin = Node(to).inputs.at(in).id;
        links.push_back(link);
    }
};

// Data -> Dense 16 -> ReLU -> Dropout -> Dense 3 -> CrossEntropy -> AdamW.
Graph Mlp() {
    Graph g;
    const int data = g.Add(gui::NodeType::DataInput, "Data", {{"shape", "[8]"}});
    const int hidden = g.Add(gui::NodeType::Dense, "Hidden", {{"units", "16"}});
    const int relu = g.Add(gui::NodeType::ReLU, "ReLU");
    const int drop = g.Add(gui::NodeType::Dropout, "Dropout", {{"rate", "0.25"}});
    const int head = g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}});
    const int loss = g.Add(gui::NodeType::CrossEntropyLoss, "Loss");
    const int opt = g.Add(gui::NodeType::AdamW, "AdamW",
                          {{"learning_rate", "0.002"}, {"weight_decay", "0.05"}, {"grad_clip_norm", "1.5"}});
    g.Connect(data, hidden);
    g.Connect(hidden, relu);
    g.Connect(relu, drop);
    g.Connect(drop, head);
    g.Connect(head, loss);
    g.Connect(loss, opt);
    return g;
}

void CheckPyTorch() {
    const Graph g = Mlp();
    cyxwiz::CodeExportOptions plain;
    plain.timestamp = false;
    const auto first = cyxwiz::ExportGraphCode(g.nodes, g.links, cyxwiz::CodeFramework::PyTorch, plain);
    const auto second = cyxwiz::ExportGraphCode(g.nodes, g.links, cyxwiz::CodeFramework::PyTorch, plain);
    Check(!first.code.empty(), "PyTorch: the MLP exports");
    Check(first.code == second.code, "PyTorch: unstamped, the same graph gives the same text");
    Check(Contains(first.code, "# Generated at: (not stamped)"), "PyTorch: the unstamped header says so");
    Check(Contains(first.code, "class GeneratedModel(nn.Module):"), "PyTorch: GeneratedModel");
    Check(Contains(first.code, "F.dropout(x, p=0.25, training=self.training)"), "PyTorch: Dropout's rate");
    Check(Contains(first.code, "optim.AdamW(") && Contains(first.code, "lr=0.002") &&
              Contains(first.code, "weight_decay=0.05"),
          "PyTorch: the optimizer node's AdamW, lr and decay");
    Check(Contains(first.code, "clip_grad_norm_(model.parameters(), 1.5)"), "PyTorch: the clip norm");

    // Each parameter layer is one module, named in model order.
    Check(first.modules.size() == 2, "PyTorch: two modules (the two Dense), got " +
                                         std::to_string(first.modules.size()));
    if (first.modules.size() == 2) {
        Check(g.Node(first.modules[0].node_id).name == "Hidden" && first.modules[0].name == "layer0",
              "PyTorch: Hidden is self.layer0");
        Check(g.Node(first.modules[1].node_id).name == "Head" && first.modules[1].name == "layer1",
              "PyTorch: Head is self.layer1");
        Check(Contains(first.code, "self.layer0 = ") && Contains(first.code, "self.layer1 = "),
              "PyTorch: both modules are in __init__");
    }

    const auto stamped = cyxwiz::ExportGraphCode(g.nodes, g.links, cyxwiz::CodeFramework::PyTorch);
    Check(!Contains(stamped.code, "(not stamped)") && Contains(stamped.code, "# Generated at: 20"),
          "PyTorch: by default the header has the local time");
}

void CheckOtherFrameworks() {
    const Graph g = Mlp();
    for (const auto& [framework, name, marker] :
         std::vector<std::tuple<cyxwiz::CodeFramework, std::string, std::string>>{
             {cyxwiz::CodeFramework::TensorFlow, "TensorFlow", "tensorflow"},
             {cyxwiz::CodeFramework::Keras, "Keras", "keras"},
             {cyxwiz::CodeFramework::PyCyxWiz, "PyCyxWiz", "pycyxwiz"}}) {
        const auto out = cyxwiz::ExportGraphCode(g.nodes, g.links, framework);
        Check(Contains(out.code, marker), name + ": exports (names " + marker + ")");
        Check(out.modules.empty(), name + ": no PyTorch module list");
    }
}

void CheckRefusals() {
    Check(cyxwiz::ExportGraphCode({}, {}, cyxwiz::CodeFramework::PyTorch).code.empty(), "an empty graph exports nothing");

    Graph cycle;
    const int a = cycle.Add(gui::NodeType::Dense, "A", {{"units", "4"}});
    const int b = cycle.Add(gui::NodeType::Dense, "B", {{"units", "4"}});
    cycle.Connect(a, b);
    cycle.Connect(b, a);
    Check(cyxwiz::ExportGraphCode(cycle.nodes, cycle.links, cyxwiz::CodeFramework::PyTorch).code.empty(),
          "a graph with a cycle exports nothing");

    Graph blocked = Mlp();
    const int conv3d = blocked.Add(gui::NodeType::Conv3D, "Volume");
    blocked.Connect(blocked.nodes.front().id, conv3d);
    const auto out = cyxwiz::ExportGraphCode(blocked.nodes, blocked.links, cyxwiz::CodeFramework::PyTorch);
    Check(Contains(out.code, "raise ValueError(") && Contains(out.code, "Volume") && Contains(out.code, "Conv3D"),
          "a layer the export cannot write: a script that raises with the node and the reason");
}

// Data [6, 3] -> recurrent -> Dense 1 -> `loss` -> Adam.
Graph Recurrent(gui::NodeType recurrent, const std::string& return_sequences, gui::NodeType loss,
                std::map<std::string, std::string> loss_params = {}) {
    Graph g;
    const int data = g.Add(gui::NodeType::DataInput, "Data", {{"shape", "[6,3]"}});
    const int rnn = g.Add(recurrent, "Recurrent",
                          {{"input_size", "3"}, {"hidden_size", "5"}, {"return_sequences", return_sequences}});
    const int head = g.Add(gui::NodeType::Dense, "Head", {{"units", "1"}});
    const int loss_node = g.Add(loss, "Loss", std::move(loss_params));
    const int opt = g.Add(gui::NodeType::Adam, "Adam");
    g.Connect(data, rnn);
    g.Connect(rnn, head);
    g.Connect(head, loss_node);
    g.Connect(loss_node, opt);
    return g;
}

std::string PyTorchOf(const Graph& g) {
    return cyxwiz::ExportGraphCode(g.nodes, g.links, cyxwiz::CodeFramework::PyTorch).code;
}

// The Engine hands the next layer the last time step unless the recurrent
// layer returns sequences; the criterion is the graph's loss node.
void CheckRecurrentAndLoss() {
    const std::string last_step = "x = x[:, -1, :]  # return_sequences=false: the last time step";
    for (const auto type : {gui::NodeType::LSTM, gui::NodeType::GRU, gui::NodeType::RNN}) {
        const std::string name = type == gui::NodeType::LSTM ? "LSTM" : type == gui::NodeType::GRU ? "GRU" : "RNN";
        Check(Contains(PyTorchOf(Recurrent(type, "false", gui::NodeType::MSELoss)), last_step),
              name + " without return_sequences: the last time step goes on");
        Check(!Contains(PyTorchOf(Recurrent(type, "true", gui::NodeType::MSELoss)), last_step),
              name + " with return_sequences: the whole sequence goes on");
    }

    const std::string mse = PyTorchOf(Recurrent(gui::NodeType::GRU, "false", gui::NodeType::MSELoss));
    Check(Contains(mse, "criterion = nn.MSELoss(reduction='mean')"), "MSE loss node: nn.MSELoss");
    Check(Contains(mse, "loss = criterion(output, target.view_as(output))"),
          "MSE: the target takes the output's shape (no [batch, batch] broadcast)");
    Check(!Contains(mse, "CrossEntropyLoss"), "MSE: no cross-entropy criterion");

    const std::string huber =
        PyTorchOf(Recurrent(gui::NodeType::GRU, "false", gui::NodeType::HuberLoss, {{"beta", "0.5"}}));
    Check(Contains(huber, "criterion = nn.HuberLoss(reduction='mean', delta=0.5)"), "Huber: its delta");

    const std::string ce = PyTorchOf(Recurrent(gui::NodeType::GRU, "false", gui::NodeType::CrossEntropyLoss,
                                               {{"label_smoothing", "0.1"}, {"ignore_index", "0"}}));
    Check(Contains(ce, "criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0, label_smoothing=0.1)") &&
              Contains(ce, "loss = criterion(output, target)"),
          "cross-entropy: ignore_index and label_smoothing, integer targets as they are");

    // A per-token head: [batch, seq, classes] against [batch, seq] targets,
    // the loss over every token.
    Graph tagger;
    const int tokens = tagger.Add(gui::NodeType::DataInput, "Data", {{"shape", "[6,3]"}});
    const int lstm = tagger.Add(gui::NodeType::LSTM, "Recurrent",
                                {{"input_size", "3"}, {"hidden_size", "5"}, {"return_sequences", "true"}});
    const int tags = tagger.Add(gui::NodeType::TimeDistributed, "Tags", {{"units", "4"}});
    const int token_loss = tagger.Add(gui::NodeType::CrossEntropyLoss, "Loss", {{"ignore_index", "-100"}});
    const int tagger_opt = tagger.Add(gui::NodeType::Adam, "Adam");
    tagger.Connect(tokens, lstm);
    tagger.Connect(lstm, tags);
    tagger.Connect(tags, token_loss);
    tagger.Connect(token_loss, tagger_opt);
    Check(Contains(PyTorchOf(tagger), "loss = criterion(output.flatten(0, -2), target.flatten())"),
          "a per-token head: the cross-entropy over every token");

    const std::string bce = PyTorchOf(Recurrent(gui::NodeType::GRU, "false", gui::NodeType::BCEWithLogits,
                                                {{"pos_weight", "3.0"}}));
    Check(Contains(bce, "criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(3.0))") &&
              Contains(bce, "loss = criterion(output, target.view_as(output).float())"),
          "BCE with logits: pos_weight, float targets shaped like the output");

    Graph reshaped;
    const int flat = reshaped.Add(gui::NodeType::DataInput, "Data", {{"shape", "[18]"}});
    const int steps = reshaped.Add(gui::NodeType::Reshape, "Steps", {{"shape", "6,3"}});
    const int over_steps = reshaped.Add(gui::NodeType::GRU, "Over the steps", {{"hidden_size", "4"}});
    reshaped.Connect(flat, steps);
    reshaped.Connect(steps, over_steps);
    Check(Contains(PyTorchOf(reshaped), "x = x.reshape(x.size(0), *[6, 3])"),
          "Reshape: the per-sample shape, the batch first (it was left out of the forward)");

    const std::string focal = PyTorchOf(Recurrent(gui::NodeType::GRU, "false", gui::NodeType::FocalLoss));
    Check(Contains(focal, "criterion = None  # Loss: PyTorch has no built-in loss for it; write it here"),
          "focal loss: no built-in criterion, said so (not a silent cross-entropy)");
}

// The committed graphs' export (TOFIX233 A): the width a layer reads comes
// from the graph, the frameworks' own spellings, and a graph with no model.
void CheckWidthsAndNoModel() {
    Graph text;
    const int ids = text.Add(gui::NodeType::DataInput, "Ids");
    const int embed = text.Add(gui::NodeType::Embedding, "Embed", {{"num_embeddings", "50"}, {"embedding_dim", "12"}});
    const int drop = text.Add(gui::NodeType::Dropout, "Drop", {{"rate", "0.1"}});
    const int norm = text.Add(gui::NodeType::LayerNorm, "Norm");
    const int lstm = text.Add(gui::NodeType::LSTM, "Recurrent", {{"hidden_size", "5"}});
    text.Connect(ids, embed);
    text.Connect(embed, drop);
    text.Connect(drop, norm);
    text.Connect(norm, lstm);
    const std::string pytorch = PyTorchOf(text);
    Check(Contains(pytorch, "nn.LayerNorm(normalized_shape=[12], eps="),
          "LayerNorm AUTO: the embedding's width, through Dropout");
    Check(Contains(pytorch, "nn.LSTM(input_size=12, hidden_size=5,"),
          "recurrent input_size: the width the graph gives it (the node has none)");
    Check(!Contains(pytorch, "LazyLayerNorm"), "no lazy LayerNorm helper where every width is known (the text as before)");
    const std::string pycyxwiz =
        cyxwiz::ExportGraphCode(text.nodes, text.links, cyxwiz::CodeFramework::PyCyxWiz).code;
    Check(Contains(pycyxwiz, "cx.LSTM(input_size=12,") && Contains(pycyxwiz, "bidirectional=False"),
          "PyCyxWiz: the same width, and a Python bool when bidirectional is not set");

    Graph rows;
    const int data = rows.Add(gui::NodeType::DataInput, "Data");  // its width set by preprocessing
    const int rows_norm = rows.Add(gui::NodeType::LayerNorm, "Norm");
    const int head = rows.Add(gui::NodeType::Dense, "Head", {{"units", "3"}});
    rows.Connect(data, rows_norm);
    rows.Connect(rows_norm, head);
    const std::string lazy = PyTorchOf(rows);
    Check(Contains(lazy, "LazyLayerNorm(eps=1e-5, elementwise_affine=True)") &&
              Contains(lazy, "class LazyLayerNorm(nn.modules.lazy.LazyModuleMixin, nn.LayerNorm):"),
          "LayerNorm AUTO on the data's width: read on the first forward pass, as nn.LazyLinear");

    Graph simple;
    const int steps = simple.Add(gui::NodeType::DataInput, "Steps", {{"shape", "[6,3]"}});
    const int rnn = simple.Add(gui::NodeType::RNN, "RNN",
                               {{"hidden_size", "4"}, {"nonlinearity", "Tanh"}, {"bidirectional", "true"}});
    simple.Connect(steps, rnn);
    const std::string rnn_code = PyTorchOf(simple);
    Check(Contains(rnn_code, "nn.RNN(input_size=3, hidden_size=4,"),
          "recurrent input_size from the data's shape: [steps, features]");
    Check(Contains(rnn_code, "nonlinearity='tanh'"), "an RNN's activation in the case torch takes");
    Check(Contains(rnn_code, "bidirectional=True)"), "a bidirectional RNN stays bidirectional");

    Graph analysis;
    const int table = analysis.Add(gui::NodeType::DataInput, "Table", {{"shape", "[4]"}});
    const int pca = analysis.Add(gui::NodeType::PCANode, "PCA");
    analysis.Connect(table, pca);
    Check(Contains(PyTorchOf(analysis), "raise ValueError('the graph has no model layers"),
          "a data or analysis pipeline: no model to write, said so (not an empty model and a failing optimizer)");
}

// Branches and image layers (ORA-20, ORA-26): a residual block, a pool.
void CheckBranchesAndImages() {
    const std::string chain = PyTorchOf(Mlp());
    const std::regex saved_tensor("\n        v[0-9]+ = ");
    Check(!std::regex_search(chain, saved_tensor), "a plain chain keeps no named tensors (its code as before)");

    Graph g;
    const int data = g.Add(gui::NodeType::DataInput, "Data", {{"shape", "[60]"}});
    const int image = g.Add(gui::NodeType::Reshape, "Image", {{"shape", "6,5,2"}});
    const int stem = g.Add(gui::NodeType::Conv2D, "Stem",
                           {{"filters", "4"}, {"kernel_size", "3"}, {"padding", "same"}, {"stride", "1"}});
    const int conv = g.Add(gui::NodeType::Conv2D, "Conv", {{"filters", "4"}, {"kernel_size", "5"}, {"padding", "valid"}});
    const int norm = g.Add(gui::NodeType::BatchNorm, "Norm");
    const int pool = g.Add(gui::NodeType::MaxPool2D, "Pool", {{"pool_size", "3"}, {"stride", "1"}, {"padding", "1"}});
    const int add = g.Add(gui::NodeType::Add, "Residual");
    const int flat = g.Add(gui::NodeType::Flatten, "Flatten");
    const int head = g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}});
    g.Connect(data, image);
    g.Connect(image, stem);
    g.Connect(stem, conv);
    g.Connect(conv, norm);
    g.Connect(norm, pool);
    g.Connect(pool, add, 0, 0);
    g.Connect(stem, add, 0, 1);
    g.Connect(add, flat);
    g.Connect(flat, head);
    const std::string code = PyTorchOf(g);
    Check(Contains(code, "x = x.reshape(x.size(0), *[6, 5, 2]).permute(0, 3, 1, 2)"),
          "rows reshaped to an image: height, width, channel, then channels first for torch");
    Check(Contains(code, "nn.LazyConv2d(out_channels=4, kernel_size=3, stride=1, padding=1, bias=True)"),
          "Conv2D: its filters, kernel, stride and same padding as the Engine resolves it");
    Check(Contains(code, "nn.LazyConv2d(out_channels=4, kernel_size=5, stride=1, padding=0, bias=True)"),
          "Conv2D: valid padding is 0");
    Check(Contains(code, "nn.MaxPool2d(kernel_size=3, stride=1, padding=1)"), "MaxPool2D: its settings");
    Check(Contains(code, "nn.LazyBatchNorm2d("), "BatchNorm after a convolution: 2-D");
    const std::string saved = "        v" + std::to_string(stem) + " = x\n";
    Check(Contains(code, saved), "the stem's output, read twice, is kept");
    Check(Contains(code, "x = v" + std::to_string(pool) + " + v" + std::to_string(stem) + "\n"),
          "Add combines the branch and the skip");
    // Every module is called, in order: 5 of them, layer0 .. layer4.
    bool all_called = true;
    for (int i = 0; i < 5; ++i)
        all_called = all_called && Contains(code, "x = self.layer" + std::to_string(i) + "(x)");
    Check(all_called, "the forward calls every module __init__ made (the convolutions and the pool too)");
}

// The pools as the Engine sizes them, and a model layer the export cannot
// write: refused, naming it (LeNet-5's average pools were left out and the
// export trained another network).
void CheckPoolsAndUnwrittenLayers() {
    Graph g;
    const int data = g.Add(gui::NodeType::DataInput, "Data", {{"shape", "[64]"}});
    const int image = g.Add(gui::NodeType::Reshape, "Image", {{"shape", "8,8,1"}});
    const int conv = g.Add(gui::NodeType::Conv2D, "Conv", {{"filters", "4"}, {"kernel_size", "3"}, {"padding", "same"}});
    const int average = g.Add(gui::NodeType::AvgPool2D, "Average", {{"pool_size", "3"}});
    const int maximum = g.Add(gui::NodeType::MaxPool2D, "Max", {{"pool_size", "3"}, {"padding", "same"}});
    const int global = g.Add(gui::NodeType::GlobalAvgPool, "Global");
    const int head = g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}});
    for (auto& node : g.nodes)
        if (node.id == average) node.parameters.erase("stride");  // not set: the window
    g.Connect(data, image);
    g.Connect(image, conv);
    g.Connect(conv, average);
    g.Connect(average, maximum);
    g.Connect(maximum, global);
    g.Connect(global, head);
    const std::string code = PyTorchOf(g);
    Check(Contains(code, "nn.AvgPool2d(kernel_size=3, stride=3, padding=0)"),
          "AvgPool2D: written, its stride the window when the node has none");
    Check(Contains(code, "nn.MaxPool2d(kernel_size=3, stride=2, padding=1)"),
          "MaxPool2D: the node's stride, \"same\" padding resolved as the Engine does");
    Check(Contains(code, "nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())"),
          "GlobalAvgPool on a feature map: one value per channel, as rows");
    bool all_called = true;
    for (int i = 0; i < 5; ++i)
        all_called = all_called && Contains(code, "x = self.layer" + std::to_string(i) + "(x)");
    Check(all_called, "the forward calls the convolution, both pools, the global pool and the head");

    Graph sequence;
    const int tokens = sequence.Add(gui::NodeType::DataInput, "Tokens");
    const int embed = sequence.Add(gui::NodeType::Embedding, "Embed", {{"num_embeddings", "20"}, {"embedding_dim", "6"}});
    const int over_time = sequence.Add(gui::NodeType::GlobalMaxPool, "Over time");
    const int classes = sequence.Add(gui::NodeType::Dense, "Classes", {{"units", "2"}});
    sequence.Connect(tokens, embed);
    sequence.Connect(embed, over_time);
    sequence.Connect(over_time, classes);
    Check(Contains(PyTorchOf(sequence), "x = x.amax(dim=1)"), "GlobalMaxPool on a sequence: the max over time");

    // A feature map reached through an Identity and a Concatenate is still a
    // feature map; a Data Input of [H, W, C] is one; a pool after a Dense is
    // neither, and is refused rather than pooled over the wrong axes.
    Graph merged;
    const int pixels = merged.Add(gui::NodeType::DataInput, "Pixels", {{"shape", "[8, 8, 3]"}});
    const int left = merged.Add(gui::NodeType::Conv2D, "Left", {{"filters", "4"}, {"padding", "same"}});
    const int right = merged.Add(gui::NodeType::Conv2D, "Right", {{"filters", "4"}, {"padding", "same"}});
    const int join = merged.Add(gui::NodeType::Concatenate, "Join");
    const int pass = merged.Add(gui::NodeType::Identity, "Pass");
    const int pooled = merged.Add(gui::NodeType::GlobalAvgPool, "Pooled");
    const int scores = merged.Add(gui::NodeType::Dense, "Scores", {{"units", "2"}});
    merged.Connect(pixels, left);
    merged.Connect(pixels, right);
    merged.Connect(left, join, 0, 0);
    merged.Connect(right, join, 0, 1);
    merged.Connect(join, pass);
    merged.Connect(pass, pooled);
    merged.Connect(pooled, scores);
    const std::string merged_code = PyTorchOf(merged);
    Check(Contains(merged_code, "nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())") &&
              !Contains(merged_code, "x.mean(dim=1)"),
          "GlobalAvgPool after a Concatenate and an Identity of feature maps: a feature-map pool");

    Graph image_rows;
    const int image_data = image_rows.Add(gui::NodeType::DataInput, "Images", {{"shape", "[8, 8, 3]"}});
    const int image_pool = image_rows.Add(gui::NodeType::GlobalMaxPool, "Pool");
    const int image_head = image_rows.Add(gui::NodeType::Dense, "Head", {{"units", "2"}});
    image_rows.Connect(image_data, image_pool);
    image_rows.Connect(image_pool, image_head);
    Check(Contains(PyTorchOf(image_rows), "nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())"),
          "GlobalMaxPool straight after a Data Input of [H, W, C]: a feature-map pool");

    Graph unknown;
    const int features = unknown.Add(gui::NodeType::DataInput, "Features", {{"shape", "[10]"}});
    const int hidden = unknown.Add(gui::NodeType::Dense, "Hidden", {{"units", "6"}});
    const int odd_pool = unknown.Add(gui::NodeType::GlobalAvgPool, "Odd pool");
    const int odd_head = unknown.Add(gui::NodeType::Dense, "Odd head", {{"units", "2"}});
    unknown.Connect(features, hidden);
    unknown.Connect(hidden, odd_pool);
    unknown.Connect(odd_pool, odd_head);
    Check(Contains(PyTorchOf(unknown), "raise ValueError('Odd pool: code export cannot tell whether it pools"),
          "a global pool on neither a feature map nor a sequence is refused, by name");

    Graph adaptive;
    const int grid = adaptive.Add(gui::NodeType::DataInput, "Grid", {{"shape", "[8, 8, 1]"}});
    const int shrink = adaptive.Add(gui::NodeType::AdaptiveAvgPool, "Shrink", {{"output_size", "0"}});
    adaptive.Connect(grid, shrink);
    Check(Contains(PyTorchOf(adaptive), "raise ValueError('Shrink: output_size must be positive"),
          "an Adaptive Avg Pool the Engine cannot size is refused with its reason");

    Graph upsampled;
    const int rows = upsampled.Add(gui::NodeType::DataInput, "Rows", {{"shape", "[16]"}});
    const int map = upsampled.Add(gui::NodeType::Reshape, "Map", {{"shape", "4,4,1"}});
    const int up = upsampled.Add(gui::NodeType::Upsample, "Twice as large");
    const int flat = upsampled.Add(gui::NodeType::Flatten, "Flatten");
    const int out = upsampled.Add(gui::NodeType::Dense, "Out", {{"units", "2"}});
    upsampled.Connect(rows, map);
    upsampled.Connect(map, up);
    upsampled.Connect(up, flat);
    upsampled.Connect(flat, out);
    const std::string refused = PyTorchOf(upsampled);
    Check(Contains(refused, "raise ValueError('Twice as large: code export does not write ") &&
              !Contains(refused, "class GeneratedModel"),
          "a model layer the export has no code for is refused, by name, not left out of the model");
}

// A Split sends each output to its own reader (both read the whole tensor
// before, TOFIX233 A19); TensorFlow and Keras write a straight chain only and
// refuse a graph that branches, naming where (they wrote another network
// before, A17).
void CheckSplitAndChainOnlyExports() {
    Graph g;
    const int data = g.Add(gui::NodeType::DataInput, "Data", {{"shape", "[8]"}});
    const int shared = g.Add(gui::NodeType::Dense, "Shared", {{"units", "16"}});
    const int cut = g.Add(gui::NodeType::Split, "Cut", {{"split_size", "6"}, {"dim", "1"}});
    const int a = g.Add(gui::NodeType::Dense, "Head A", {{"units", "8"}});
    const int b = g.Add(gui::NodeType::Dense, "Head B", {{"units", "8"}});
    const int join = g.Add(gui::NodeType::Concatenate, "Join", {{"dim", "1"}});
    const int classes = g.Add(gui::NodeType::Dense, "Classes", {{"units", "4"}});
    g.Connect(data, shared);
    g.Connect(shared, cut);
    g.Connect(cut, a, 0, 0);
    g.Connect(cut, b, 1, 0);
    g.Connect(a, join, 0, 0);
    g.Connect(b, join, 0, 1);
    g.Connect(join, classes);
    const std::string code = PyTorchOf(g);
    const std::string v = "v" + std::to_string(cut);
    Check(Contains(code, "        " + v + "_0, " + v + "_1 = torch.split(x, [6, x.size(1) - 6], dim=1)\n"),
          "Split: torch.split with the node's split_size and dim");
    Check(Contains(code, "        x = " + v + "_0\n        x = self.layer1(x)\n"), "head A reads Output 1");
    Check(Contains(code, "        x = " + v + "_1\n        x = self.layer2(x)\n"), "head B reads Output 2");
    Check(!Contains(code, "        " + v + " = x"), "the Split is not kept whole");
    Check(Contains(code, "x = torch.cat([v" + std::to_string(a) + ", v" + std::to_string(b) + "], dim=1)") ||
              Contains(code, "x = torch.cat([v" + std::to_string(a) + ", x], dim=1)"),
          "the heads are concatenated");

    // A Split read straight by a merge: its parts
    Graph halves;
    const int in = halves.Add(gui::NodeType::DataInput, "Data", {{"shape", "[10]"}});
    const int halve = halves.Add(gui::NodeType::Split, "Halve", {{"split_size", "5"}, {"dim", "-1"}});
    const int sum = halves.Add(gui::NodeType::Add, "Sum");
    const int out = halves.Add(gui::NodeType::Dense, "Out", {{"units", "2"}});
    halves.Connect(in, halve);
    halves.Connect(halve, sum, 0, 0);
    halves.Connect(halve, sum, 1, 1);
    halves.Connect(sum, out);
    const std::string halves_code = PyTorchOf(halves);
    const std::string h = "v" + std::to_string(halve);
    Check(Contains(halves_code, "torch.split(x, [5, x.size(-1) - 5], dim=-1)") &&
              Contains(halves_code, "x = " + h + "_0 + " + h + "_1\n"),
          "a merge reads the Split's two parts");

    // A Split of a Split: the inner one reads its part
    Graph twice;
    const int t_in = twice.Add(gui::NodeType::DataInput, "Data", {{"shape", "[9]"}});
    const int outer = twice.Add(gui::NodeType::Split, "Outer", {{"split_size", "3"}, {"dim", "1"}});
    const int inner = twice.Add(gui::NodeType::Split, "Inner", {{"split_size", "2"}, {"dim", "1"}});
    const int t_head = twice.Add(gui::NodeType::Dense, "Head", {{"units", "2"}});
    twice.Connect(t_in, outer);
    twice.Connect(outer, inner, 1, 0);
    twice.Connect(inner, t_head, 0, 0);
    const std::string twice_code = PyTorchOf(twice);
    const std::string o = "v" + std::to_string(outer), i = "v" + std::to_string(inner);
    Check(Contains(twice_code, "        x = " + o + "_1\n        " + i + "_0, " + i + "_1 = torch.split(x, [2,"),
          "a Split of a Split's Output 2 splits that part");

    // A Split read by the loss: the forward returns that part, not its input
    Graph tail;
    const int l_in = tail.Add(gui::NodeType::DataInput, "Data", {{"shape", "[6]"}});
    const int l_dense = tail.Add(gui::NodeType::Dense, "Wide", {{"units", "10"}});
    const int l_cut = tail.Add(gui::NodeType::Split, "Logits", {{"split_size", "4"}, {"dim", "1"}});
    const int l_loss = tail.Add(gui::NodeType::CrossEntropyLoss, "Loss");
    tail.Connect(l_in, l_dense);
    tail.Connect(l_dense, l_cut);
    tail.Connect(l_cut, l_loss, 0, 0);
    Check(Contains(PyTorchOf(tail), "        x = v" + std::to_string(l_cut) + "_0\n        return x\n"),
          "a Split the loss reads: the forward returns its Output 1");

    // What the compiler refuses is refused here too, by name
    const auto split_refused = [](std::map<std::string, std::string> params, const std::string& shape,
                                  const std::string& why) {
        Graph r;
        const int in = r.Add(gui::NodeType::DataInput, "Data", {{"shape", shape}});
        const int cut = r.Add(gui::NodeType::Split, "Cut", std::move(params));
        const int head = r.Add(gui::NodeType::Dense, "Head", {{"units", "2"}});
        r.Connect(in, cut);
        r.Connect(cut, head, 0, 0);
        return Contains(PyTorchOf(r), "Cut: " + why);
    };
    Check(split_refused({{"split_size", "2"}, {"dim", "0"}}, "[6]", "Split dim 0 is the batch"),
          "a Split of the batch is refused, as the compiler refuses it");
    Check(split_refused({{"split_size", "0"}, {"dim", "1"}}, "[6]", "Split split_size must be at least 1"),
          "a Split of no entries is refused");
    // Image data: torch holds channels first, so a Split's parts would differ
    Graph cnn;
    const int pixels = cnn.Add(gui::NodeType::DataInput, "Pixels", {{"shape", "[8, 8, 3]"}});
    const int conv = cnn.Add(gui::NodeType::Conv2D, "Conv", {{"filters", "4"}, {"padding", "same"}});
    const int flat = cnn.Add(gui::NodeType::Flatten, "Flatten");
    const int f_cut = cnn.Add(gui::NodeType::Split, "Cut", {{"split_size", "10"}, {"dim", "1"}});
    const int f_head = cnn.Add(gui::NodeType::Dense, "Head", {{"units", "2"}});
    cnn.Connect(pixels, conv);
    cnn.Connect(conv, flat);
    cnn.Connect(flat, f_cut);
    cnn.Connect(f_cut, f_head, 0, 0);
    Check(Contains(PyTorchOf(cnn), "Cut: code export cannot write a Split of image data yet"),
          "a Split of a flattened feature map is refused");
    // A graph the branch walk does not take: a Split there is refused
    Graph inputs;
    const int i_one = inputs.Add(gui::NodeType::DataInput, "One", {{"shape", "[4]"}});
    const int i_other = inputs.Add(gui::NodeType::DataInput, "Other", {{"shape", "[4]"}});
    const int i_sum = inputs.Add(gui::NodeType::Add, "Sum");
    const int i_cut = inputs.Add(gui::NodeType::Split, "Cut", {{"split_size", "2"}, {"dim", "1"}});
    const int i_head = inputs.Add(gui::NodeType::Dense, "Head", {{"units", "2"}});
    inputs.Connect(i_one, i_sum, 0, 0);
    inputs.Connect(i_other, i_sum, 0, 1);
    inputs.Connect(i_sum, i_cut);
    inputs.Connect(i_cut, i_head, 0, 0);
    Check(Contains(PyTorchOf(inputs), "Cut: code export writes a Split only in a graph with one input"),
          "a Split in a graph with two inputs is refused");

    // TensorFlow and Keras: chains only
    const auto refused = [](const Graph& graph, cyxwiz::CodeFramework framework, const std::string& node) {
        const std::string text = cyxwiz::ExportGraphCode(graph.nodes, graph.links, framework).code;
        return Contains(text, node + ": code export writes ") && Contains(text, " only as a straight chain of layers");
    };
    for (const auto framework : {cyxwiz::CodeFramework::TensorFlow, cyxwiz::CodeFramework::Keras}) {
        const std::string name = framework == cyxwiz::CodeFramework::TensorFlow ? "TensorFlow" : "Keras";
        Check(refused(g, framework, "Cut"), name + ": a Split is refused by name");
        // A residual: the first place it branches is the layer both paths read
        Graph residual;
        const int x = residual.Add(gui::NodeType::DataInput, "Data", {{"shape", "[4]"}});
        const int stem = residual.Add(gui::NodeType::Dense, "Stem", {{"units", "4"}});
        const int deep = residual.Add(gui::NodeType::Dense, "Deep", {{"units", "4"}});
        const int skip = residual.Add(gui::NodeType::Add, "Skip");
        const int top = residual.Add(gui::NodeType::Dense, "Top", {{"units", "2"}});
        residual.Connect(x, stem);
        residual.Connect(stem, deep);
        residual.Connect(deep, skip, 0, 0);
        residual.Connect(stem, skip, 0, 1);
        residual.Connect(skip, top);
        Check(refused(residual, framework, "Stem"), name + ": a residual is refused where it branches");
        // Two inputs merged: the merge is the first place it branches
        Graph two;
        const int one = two.Add(gui::NodeType::DataInput, "One", {{"shape", "[4]"}});
        const int other = two.Add(gui::NodeType::DataInput, "Other", {{"shape", "[4]"}});
        const int both = two.Add(gui::NodeType::Add, "Both");
        const int last = two.Add(gui::NodeType::Dense, "Last", {{"units", "2"}});
        two.Connect(one, both, 0, 0);
        two.Connect(other, both, 0, 1);
        two.Connect(both, last);
        Check(refused(two, framework, "Both"), name + ": a merge is refused by name");
        // Two inputs into a two-input op that is not a merge
        Graph dot;
        const int d_a = dot.Add(gui::NodeType::DataInput, "A", {{"shape", "[4]"}});
        const int d_b = dot.Add(gui::NodeType::DataInput, "B", {{"shape", "[4]"}});
        const int d_dot = dot.Add(gui::NodeType::TensorDot, "Dot");
        const int d_out = dot.Add(gui::NodeType::Dense, "Out", {{"units", "2"}});
        dot.Connect(d_a, d_dot, 0, 0);
        dot.Connect(d_b, d_dot, 0, 1);
        dot.Connect(d_dot, d_out);
        Check(refused(dot, framework, "Dot") &&
                  Contains(cyxwiz::ExportGraphCode(dot.nodes, dot.links, framework).code, "it reads two tensors"),
              name + ": a two-input op is refused by name");
        // A Plot on the data reads it, but the model is still a chain
        Graph watched = Mlp();
        const int plot = watched.Add(gui::NodeType::Plot, "Look");
        watched.Connect(watched.nodes.front().id, plot);
        Check(!Contains(cyxwiz::ExportGraphCode(watched.nodes, watched.links, framework).code, "only as a straight chain"),
              name + ": a Plot reading the data does not make the chain a branch");
        // A layer off the chain that leads nowhere would still be written into it
        Graph dead_end = Mlp();
        const int side = dead_end.Add(gui::NodeType::Dense, "Side", {{"units", "5"}});
        dead_end.Connect(dead_end.nodes[1].id, side);
        Check(refused(dead_end, framework, "Hidden"), name + ": a layer branching off the chain is refused");
        Graph fan;
        const int f_in = fan.Add(gui::NodeType::DataInput, "Data", {{"shape", "[4]"}});
        const int trunk = fan.Add(gui::NodeType::Dense, "Trunk", {{"units", "4"}});
        const int left = fan.Add(gui::NodeType::Dense, "Left", {{"units", "2"}});
        const int right = fan.Add(gui::NodeType::Dense, "Right", {{"units", "2"}});
        fan.Connect(f_in, trunk);
        fan.Connect(trunk, left);
        fan.Connect(trunk, right);
        Check(refused(fan, framework, "Trunk"), name + ": a layer two layers read is refused by name");
        const Graph chain = Mlp();
        Check(!Contains(cyxwiz::ExportGraphCode(chain.nodes, chain.links, framework).code, "only as a straight chain"),
              name + ": a straight chain is still written");
    }
}

}  // namespace

int main() {
    std::cout << "test_code_export\n";
    try {
        CheckPyTorch();
        CheckOtherFrameworks();
        CheckRefusals();
        CheckRecurrentAndLoss();
        CheckBranchesAndImages();
        CheckWidthsAndNoModel();
        CheckPoolsAndUnwrittenLayers();
        CheckSplitAndChainOnlyExports();
    } catch (const std::exception& e) {
        std::cout << "  FAIL exception: " << e.what() << "\n";
        ++g_failures;
    }
    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
