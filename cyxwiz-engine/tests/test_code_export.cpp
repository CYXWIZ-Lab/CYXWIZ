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
    reshaped.Connect(flat, steps);
    Check(Contains(PyTorchOf(reshaped), "x = x.reshape(x.size(0), *[6, 3])"),
          "Reshape: the per-sample shape, the batch first (it was left out of the forward)");

    const std::string focal = PyTorchOf(Recurrent(gui::NodeType::GRU, "false", gui::NodeType::FocalLoss));
    Check(Contains(focal, "criterion = None  # Loss: PyTorch has no built-in loss for it; write it here"),
          "focal loss: no built-in criterion, said so (not a silent cross-entropy)");
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

}  // namespace

int main() {
    std::cout << "test_code_export\n";
    try {
        CheckPyTorch();
        CheckOtherFrameworks();
        CheckRefusals();
        CheckRecurrentAndLoss();
        CheckBranchesAndImages();
    } catch (const std::exception& e) {
        std::cout << "  FAIL exception: " << e.what() << "\n";
        ++g_failures;
    }
    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
