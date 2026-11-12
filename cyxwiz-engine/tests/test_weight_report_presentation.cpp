// The Data Loader dialog's words about a weights file (TOFIX178 1.9d): the
// matcher's report for a small graph, and the card and report rows made of it.
#include "core/weight_match.h"
#include "core/weight_report_presentation.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void Check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

cyxwiz::weight_file::Entry Entry(const std::string& name, std::vector<size_t> shape, bool is_float = true) {
    cyxwiz::weight_file::Entry entry;
    entry.name = name;
    entry.shape = std::move(shape);
    entry.dtype = is_float ? "F32" : "I64";
    entry.is_float = is_float;
    size_t n = 1;
    for (const size_t d : entry.shape) n *= d;
    if (is_float) {
        entry.values.resize(n);
        for (size_t i = 0; i < n; ++i) entry.values[i] = static_cast<float>(i);
    }
    return entry;
}

cyxwiz::weight_match::LayerSlot Slot(const std::string& node, gui::NodeType type, const std::string& label,
                                     const std::string& prefix,
                                     std::vector<std::pair<std::string, std::vector<size_t>>> parameters) {
    cyxwiz::weight_match::LayerSlot slot;
    slot.node_name = node;
    slot.type = type;
    slot.type_label = label;
    slot.key_prefix = prefix;
    slot.parameters = std::move(parameters);
    return slot;
}

}  // namespace

int main() {
    using cyxwiz::weight_match::Match;
    // A file as torchvision names things: conv1 [out 2, in 3, 1 x 2 kernel],
    // bn1 with its counter, fc for 1000 classes, and a layer the graph lacks.
    cyxwiz::weight_file::WeightFile file;
    file.format = "PyTorch state-dict";
    file.entries = {Entry("conv1.weight", {2, 3, 1, 2}),
                    Entry("bn1.weight", {2}), Entry("bn1.bias", {2}), Entry("bn1.running_mean", {2}),
                    Entry("bn1.running_var", {2}), Entry("bn1.num_batches_tracked", {}, false),
                    Entry("fc.weight", {1000, 4}), Entry("fc.bias", {1000}),
                    Entry("aux.weight", {5})};
    const std::vector<cyxwiz::weight_match::LayerSlot> graph = {
        Slot("conv1", gui::NodeType::Conv2D, "Conv2D", "layer0.", {{"weights", {1, 2, 3, 2}}}),
        Slot("bn1", gui::NodeType::BatchNorm, "BatchNorm", "layer1.",
             {{"beta", {2}}, {"gamma", {2}}, {"running_mean", {2}}, {"running_var", {2}}}),
        Slot("fc", gui::NodeType::Dense, "Linear", "layer4.", {{"bias", {2}}, {"weight", {2, 4}}}),
        Slot("extra", gui::NodeType::Dense, "Linear", "layer5.", {{"weight", {3, 3}}}),
    };
    const auto report = Match(file, graph);
    Check(report.loaded == 2 && report.fresh == 2, "two layers load, two start fresh");
    Check(report.tensors.size() == 5, "five tensors are set (kernels and four BatchNorm tensors)");
    // The kernel's layout: [out, in, kh, kw] -> [kh, kw, in, out]. Value at
    // PyTorch (o, i, h, w) is o*6 + i*2 + w; the Engine's index is (h, w, i, o).
    bool kernels = false;
    for (const auto& tensor : report.tensors) {
        if (tensor.key != "layer0.weights") continue;
        kernels = tensor.shape == std::vector<size_t>({1, 2, 3, 2}) && tensor.values.size() == 12;
        for (size_t w = 0; kernels && w < 2; ++w)
            for (size_t i = 0; kernels && i < 3; ++i)
                for (size_t o = 0; kernels && o < 2; ++o)
                    kernels = tensor.values[(w * 3 + i) * 2 + o] == static_cast<float>(o * 6 + i * 2 + w);
    }
    Check(kernels, "the convolution kernels arrive in the Engine's layout");
    bool gamma = false;
    for (const auto& tensor : report.tensors) gamma = gamma || tensor.key == "layer1.gamma";
    Check(gamma, "BatchNorm's weight becomes gamma");
    Check(report.Summary() == "2 of 4 layers take weights from the file; 2 start fresh: fc (shape differs: the file "
                              "has [1000], the graph [2]), extra (not in the file)",
          "summary: " + report.Summary());

    const auto view = cyxwiz::BuildWeightReportView(report, "resnet.pth");
    Check(view.file_line == "resnet.pth: PyTorch state-dict, 9 entries", "file line: " + view.file_line);
    Check(view.loaded_line == "2 of 4 layers take their weights", "loaded line: " + view.loaded_line);
    Check(view.fresh_lines.size() == 2 &&
              view.fresh_lines[0] == "fc starts fresh: shape differs: the file has [1000], the graph [2]" &&
              view.fresh_lines[1] == "extra starts fresh: not in the file",
          "fresh lines name the layer and the reason");
    Check(view.unused_line == "4 entries in the file are not used", "unused line: " + view.unused_line);
    Check(view.layers == 4 && view.loaded == 2 && view.fresh == 2 && view.file_only == 4, "the chip counts");
    Check(view.rows.size() == 8, "four layer rows and four file-only rows");
    if (view.rows.size() == 8) {
        Check(view.rows[0].node == "conv1" && view.rows[0].layer == "Conv2D" &&
                  view.rows[0].entries == "conv1.weight [2, 3, 1, 2]" && view.rows[0].shapes == "[1, 2, 3, 2]" &&
                  view.rows[0].result == "Loaded",
              "the convolution's row");
        Check(view.rows[1].shapes == "[2] x 4" && view.rows[1].result == "Loaded", "BatchNorm's four tensors on one row");
        Check(view.rows[2].result == "Starts fresh: shape differs: the file has [1000], the graph [2]", "the head's row");
        Check(view.rows[3].entries.empty() && view.rows[3].result == "Starts fresh: not in the file",
              "a layer the file lacks");
        Check(view.rows[4].entries == "bn1.num_batches_tracked []" &&
                  view.rows[4].result == "Not used: a counter PyTorch keeps, not a weight",
              "the counter: " + view.rows[4].result);
        Check(view.rows[5].result == "Not used: its layer starts fresh" && view.rows[6].result == view.rows[5].result,
              "the head's two entries");
        Check(view.rows[7].entries == "aux.weight [5]" && view.rows[7].result == "Not used: no node is named aux",
              "an entry of a layer the graph lacks");
        int all = 0, loaded = 0, fresh = 0, file_only = 0;
        for (const auto& row : view.rows) {
            all += cyxwiz::WeightReportRowShown(row, 0);
            loaded += cyxwiz::WeightReportRowShown(row, 1);
            fresh += cyxwiz::WeightReportRowShown(row, 2);
            file_only += cyxwiz::WeightReportRowShown(row, 3);
        }
        Check(all == 4 && loaded == 2 && fresh == 2 && file_only == 4, "the filters show the counted rows");
    }

    // More than three fresh layers: three named, the rest counted.
    std::vector<cyxwiz::weight_match::LayerSlot> many;
    for (int i = 0; i < 5; ++i)
        many.push_back(Slot("d" + std::to_string(i), gui::NodeType::Dense, "Linear", "layer" + std::to_string(i) + ".",
                            {{"weight", {2, 2}}}));
    many.push_back(graph[0]);
    const auto crowded = cyxwiz::BuildWeightReportView(Match(file, many), "resnet.pth");
    Check(crowded.fresh_lines.size() == 4 && crowded.fresh_lines[3] == "and 2 more (see the match report)",
          "a long list of fresh layers is cut after three");
    Check(crowded.loaded_line == "1 of 6 layers take their weights", "one loaded: " + crowded.loaded_line);

    if (failures) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "Weight report presentation passed: " << checks << " checks\n";
    return 0;
}
