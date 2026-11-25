// Every shipped pattern (cyxwiz-engine/resources/patterns) is inserted the way
// the Pattern Browser inserts it - PatternLibrary with the registry's node
// factory - and then:
// - a full graph compiles (its Data Input given the pattern's sample shape,
//   since no data is loaded here) and its model builds, and the small ones
//   train a few steps;
// - a block compiles inside a small graph (data -> block -> Flatten -> Dense
//   -> loss -> optimizer) and its model builds (a block whose sample shape
//   does not fit its layers compiled but failed at Train);
// - a pattern that cannot be used yet refuses to insert and says why;
// - a network written from torchvision (ResNet-18, the MobileNets) keeps every
//   layer that holds weights named as torchvision names it, so the published
//   weight file fills it, and compiles with its backbone frozen too.
// Every template link must land on a real pin: the library skips a link to a
// missing pin with only a log line, so a stale pattern would insert broken.
// (TOFIX178 4b: before this no test inserted or compiled a shipped pattern.)

#include "../src/core/arrow_dataset.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/model_builder.h"
#include "../src/core/rl_graph_compile.h"
#include "../src/core/training_executor.h"
#include "../src/gui/data_input_capabilities.h"
#include "../src/gui/loaders/data_loader.h"
#include "../src/gui/patterns/pattern_library.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

// Graph-only: no file loaders.
namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string&) { return FileCategory::Tabular; }
}  // namespace cyxwiz::loaders

namespace fs = std::filesystem;
using gui::patterns::Pattern;
using gui::patterns::PatternKind;
using gui::patterns::PatternLibrary;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

fs::path FindPatternsDir() {
    for (auto dir = fs::current_path(); !dir.empty(); dir = dir.parent_path()) {
        const fs::path candidate = dir / "cyxwiz-engine" / "resources" / "patterns";
        if (fs::is_directory(candidate)) return candidate;
        if (dir.parent_path() == dir) break;
    }
    return {};
}

std::map<std::string, std::string> Defaults(const Pattern& pattern) {
    std::map<std::string, std::string> defaults;
    for (const auto& parameter : pattern.parameters) defaults[parameter.name] = parameter.default_value;
    return defaults;
}

std::string Substitute(std::string value, const std::map<std::string, std::string>& defaults) {
    for (const auto& [name, default_value] : defaults) {
        const std::string token = "$" + name;
        for (size_t at = value.find(token); at != std::string::npos; at = value.find(token, at + default_value.size()))
            value.replace(at, token.size(), default_value);
    }
    return value;
}

struct Inserted {
    bool ok = false;
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node_id = 1;
    int next_pin_id = 1;
    int next_link_id = 1;
};

Inserted Insert(PatternLibrary& library, const Pattern& pattern,
                const std::map<std::string, std::string>& parameters = {}) {
    Inserted inserted;
    int next_node_id = 1;
    inserted.ok = library.InstantiatePatternWithCreator(
        pattern.id, parameters, inserted.nodes, inserted.links, next_node_id, inserted.next_link_id, ImVec2(0, 0),
        [&inserted](gui::NodeType type, const std::string& name) {
            int scratch_id = 0;
            return gui::CreateGraphNode(type, name, scratch_id, inserted.next_pin_id);
        });
    inserted.next_node_id = next_node_id;
    return inserted;
}

gui::MLNode* FindType(std::vector<gui::MLNode>& nodes, gui::NodeType type) {
    for (auto& node : nodes)
        if (node.type == type) return &node;
    return nullptr;
}

const gui::NodePin* PinOf(const gui::MLNode& node, bool input, gui::PinType type) {
    for (const auto& pin : input ? node.inputs : node.outputs)
        if (pin.type == type) return &pin;
    return nullptr;
}

void Link(Inserted& g, const gui::MLNode& from, const gui::NodePin* from_pin, const gui::MLNode& to,
          const gui::NodePin* to_pin) {
    Check(from_pin != nullptr && to_pin != nullptr, "wrapper link " + from.name + " -> " + to.name + " has its pins");
    if (from_pin == nullptr || to_pin == nullptr) return;
    g.links.push_back({g.next_link_id++, from.id, from_pin->id, to.id, to_pin->id});
}

gui::MLNode& Add(Inserted& g, gui::NodeType type, const std::string& name,
                 std::map<std::string, std::string> parameters = {}) {
    gui::MLNode node = gui::CreateGraphNode(type, name, g.next_node_id, g.next_pin_id);
    node.parameters = std::move(parameters);
    g.nodes.push_back(std::move(node));
    return g.nodes.back();
}

// A block between a data line and a classifier head, so it compiles as a graph.
void WrapBlock(Inserted& g, const std::string& sample_shape) {
    std::set<int> has_input, has_output;
    for (const auto& link : g.links) {
        has_input.insert(link.to_node);
        has_output.insert(link.from_node);
    }
    std::vector<int> entries, exits;
    for (const auto& node : g.nodes) {
        if (!has_input.count(node.id)) entries.push_back(node.id);
        if (!has_output.count(node.id)) exits.push_back(node.id);
    }
    Check(entries.size() == 1 && exits.size() == 1, "a block has one entry and one exit");
    if (entries.size() != 1 || exits.size() != 1) return;
    const int entry = entries[0], exit = exits[0];

    const gui::MLNode data = Add(g, gui::NodeType::DataInput, "Data",
                                 {{"file_category", "tabular"}, {"shape", sample_shape}, {"label_column", "label"}});
    const gui::MLNode split = Add(g, gui::NodeType::DataSplit, "Split");
    const gui::MLNode loader = Add(g, gui::NodeType::DataLoader, "Loader", {{"batch_size", "4"}});
    const gui::MLNode flatten = Add(g, gui::NodeType::Flatten, "Flatten");
    const gui::MLNode head = Add(g, gui::NodeType::Dense, "Head", {{"units", "3"}});
    const gui::MLNode loss = Add(g, gui::NodeType::CrossEntropyLoss, "Loss");
    const gui::MLNode adam = Add(g, gui::NodeType::Adam, "Adam");
    const gui::MLNode output = Add(g, gui::NodeType::Output, "Output", {{"num_classes", "3"}});
    const gui::MLNode* entry_node = nullptr;
    const gui::MLNode* exit_node = nullptr;
    for (const auto& node : g.nodes) {
        if (node.id == entry) entry_node = &node;
        if (node.id == exit) exit_node = &node;
    }
    using gui::PinType;
    Link(g, data, PinOf(data, false, PinType::Dataset), split, PinOf(split, true, PinType::Dataset));
    Link(g, split, PinOf(split, false, PinType::Dataset), loader, PinOf(loader, true, PinType::Dataset));
    Link(g, loader, PinOf(loader, false, PinType::Tensor), *entry_node, &entry_node->inputs.front());
    Link(g, *exit_node, &exit_node->outputs.front(), flatten, PinOf(flatten, true, PinType::Tensor));
    Link(g, flatten, PinOf(flatten, false, PinType::Tensor), head, PinOf(head, true, PinType::Tensor));
    Link(g, head, PinOf(head, false, PinType::Tensor), loss, PinOf(loss, true, PinType::Tensor));
    Link(g, loader, PinOf(loader, false, PinType::Labels), loss, PinOf(loss, true, PinType::Labels));
    Link(g, loss, PinOf(loss, false, PinType::Loss), adam, &adam.inputs.front());
    Link(g, adam, &adam.outputs.front(), output, &output.inputs.front());
}

bool HoldsWeights(gui::NodeType type) {
    return type == gui::NodeType::Conv2D || type == gui::NodeType::DepthwiseConv2D ||
           type == gui::NodeType::BatchNorm || type == gui::NodeType::Dense ||
           type == gui::NodeType::SqueezeExcitation;
}

std::optional<cyxwiz::TrainingConfiguration> Compile(const std::string& name, const Inserted& g) {
    cyxwiz::TrainingConfiguration config;
    try {
        config = cyxwiz::GraphCompiler{}.Compile(g.nodes, g.links, /*allow_unloaded_data=*/true);
    } catch (const std::exception& e) {
        Check(false, name + ": compile threw: " + e.what());
        return std::nullopt;
    }
    bool clean = true;
    for (const auto& issue : config.issues) {
        if (issue.level != cyxwiz::IssueLevel::Error) continue;
        Check(false, name + ": [" + issue.node_name + "] " + issue.message + " " + issue.error_code);
        clean = false;
    }
    Check(config.is_valid, name + ": the graph is valid");
    if (!clean || !config.is_valid) return std::nullopt;
    // Train builds the model from the compiled layers: the widths must fit.
    // (An RL graph trains its policy through the RL trainer, not this builder.)
    if (config.rl) return config;
    const auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), name + ": the model builds: " + built.error_message);
    if (!built.ok()) return std::nullopt;
    return config;
}

// Rows of `width` features in [0, 1) (or token ids below `vocabulary`) and a
// label in 0..classes-1.
std::shared_ptr<cyxwiz::ArrowDataset> SyntheticRows(const std::string& name, size_t width, int classes,
                                                    int vocabulary) {
    const size_t rows = 24;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (size_t f = 0; f < width; ++f) {
        arrow::FloatBuilder builder;
        for (size_t r = 0; r < rows; ++r) {
            const float wave = 0.5f + 0.5f * std::sin(0.37f * static_cast<float>(r * width + f));
            (void)builder.Append(vocabulary > 0 ? std::floor(wave * static_cast<float>(vocabulary - 1)) : wave);
        }
        std::shared_ptr<arrow::Array> column;
        (void)builder.Finish(&column);
        columns.push_back(column);
        fields.push_back(arrow::field("x" + std::to_string(f), arrow::float32()));
    }
    arrow::Int32Builder labels;
    for (size_t r = 0; r < rows; ++r) (void)labels.Append(static_cast<int32_t>(r % static_cast<size_t>(classes)));
    std::shared_ptr<arrow::Array> label_column;
    (void)labels.Finish(&label_column);
    columns.push_back(label_column);
    fields.push_back(arrow::field("label", arrow::int32()));
    return std::make_shared<cyxwiz::ArrowDataset>(arrow::Table::Make(arrow::schema(fields), columns), name);
}

// The brief trainings run on ArrayFire CPU: they check the pattern, not the
// device, and a GPU run would wait while another process trains on the card.
void SelectArrayFireCpu() {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = std::find_if(devices.begin(), devices.end(), [](const cyxwiz::DeviceInfo& device) {
        return device.type == cyxwiz::DeviceType::CPU;
    });
    Check(cpu != devices.end(), "an ArrayFire CPU route is found");
    if (cpu == devices.end()) return;
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(cyxwiz::DeviceType::CPU, cpu->device_id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(cyxwiz::ExecutionPolicy::Strict);
}

void TrainBriefly(const Pattern& pattern, cyxwiz::TrainingConfiguration config, int classes, int vocabulary) {
    const std::string name = pattern.id;
    config.dataset_name = "pattern_" + name;
    config.epochs = 1;
    config.batch_size = 8;
    config.validation_freq = 0;
    config.save_best_checkpoint = false;
    config.num_workers = 0;
    config.log_interval = 0;
    const auto dataset = SyntheticRows(config.dataset_name, config.input_size, classes, vocabulary);
    SelectArrayFireCpu();
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    cyxwiz::TrainingMetrics last;
    executor.Train(config.epochs, config.batch_size, nullptr, nullptr,
                   [&](const cyxwiz::TrainingMetrics& metrics) { last = metrics; });
    Check(last.is_complete, name + ": trains (" + last.status_message + ")");
    Check(std::isfinite(last.train_loss) && last.train_loss > 0.0f,
          name + ": a finite loss (" + std::to_string(last.train_loss) + ")");
    if (last.is_complete) std::cout << "  " << name << ": trained one epoch, loss " << last.train_loss << "\n";
}

// Every layer that holds weights keeps its torchvision name (the weight file
// matches by node name); the head takes num_classes, the rest take
// freeze_backbone, and the frozen graph compiles.
void CheckPublishedNames(PatternLibrary& library, const Pattern& pattern, size_t weighted,
                         const std::string& head, const std::string& shape) {
    const std::string& id = pattern.id;
    std::set<std::string> template_names;
    for (const auto& node : pattern.template_data.nodes) template_names.insert(node.name);
    Inserted g = Insert(library, pattern, {{"num_classes", "7"}, {"freeze_backbone", "true"}});
    Check(g.ok, id + ": inserts with num_classes 7 and the backbone frozen");
    size_t count = 0;
    bool head_found = false;
    for (const auto& node : g.nodes) {
        if (!HoldsWeights(node.type)) continue;
        ++count;
        Check(template_names.count(node.name) && node.name.find(' ') == std::string::npos,
              id + ": '" + node.name + "' keeps its torchvision name");
        const auto frozen = node.parameters.find("frozen");
        if (node.name == head) {
            head_found = true;
            Check(node.parameters.count("units") && node.parameters.at("units") == "7",
                  id + ": the head takes num_classes");
            Check(frozen == node.parameters.end() || frozen->second == "false",
                  id + ": the head is never frozen (frozen = '" +
                      (frozen == node.parameters.end() ? std::string("unset") : frozen->second) + "')");
        } else {
            Check(frozen != node.parameters.end() && frozen->second == "true",
                  id + ": '" + node.name + "' takes freeze_backbone");
        }
    }
    Check(head_found, id + ": the head is '" + head + "'");
    Check(count == weighted, id + ": " + std::to_string(count) + " layers with weights, torchvision has " +
                                 std::to_string(weighted));
    if (gui::MLNode* data = FindType(g.nodes, gui::NodeType::DataInput)) data->parameters["shape"] = shape;
    if (Compile(id + " (frozen)", g))
        std::cout << "  " << id << ": " << count << " layers named as torchvision's; frozen backbone compiles\n";
}

}  // namespace

int main() {
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    hooks.labels_from_structure = [](const std::string& category) -> std::optional<bool> {
        return category == "text" || category == "image" || category == "audio";
    };
    hooks.preprocessing_domain = [](const std::string& category) -> std::optional<cyxwiz::PreprocessingDomain> {
        if (category == "text") return cyxwiz::PreprocessingDomain::Text;
        if (category == "image") return cyxwiz::PreprocessingDomain::Image;
        return cyxwiz::PreprocessingDomain::Tabular;
    };
    cyxwiz::SetGraphCompilerDatasetHooks(std::move(hooks));
    // The Engine's built-in CartPole, as the Environment node lists it.
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, const std::map<std::string, std::string>&,
                                      cyxwiz::plugin::PluginEnvironmentInfo& info, std::string&) {
        if (key != "cyxwiz.builtin:CartPole-v1") return false;
        info.id = "CartPole-v1";
        info.name = "Cart Pole";
        info.observation_size = 4;
        info.discrete_actions = true;
        info.action_size = 1;
        info.action_count = 2;
        info.max_episode_steps = 500;
        return true;
    });
    cyxwiz::test::InstallQualifiedRouteSnapshot();  // a test run trains on this PC's devices

    const fs::path dir = FindPatternsDir();
    Check(!dir.empty(), "the shipped patterns folder is found");
    auto& library = PatternLibrary::Instance();
    std::vector<std::string> ids;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".cyxgraph") continue;
        Check(library.LoadPatternFromFile(entry.path().string()), entry.path().filename().string() + ": loads");
        ids.push_back(entry.path().stem().string());
    }
    Check(ids.size() >= 31, "every shipped pattern is read (" + std::to_string(ids.size()) + ")");

    // Small enough to train here: {classes, vocabulary (0: numbers)}.
    const std::map<std::string, std::pair<int, int>> train = {
        {"mlp_basic", {10, 0}}, {"cnn_basic", {10, 0}}, {"lenet5", {10, 0}}, {"rnn_basic", {2, 200}},
        // Decoder -> Layer Norm -> Flatten -> head: the head reads the whole
        // flattened sequence (it was built for one token's width).
        {"transformer_decoder", {5000, 5000}}};

    // Written by scripts/examples/torch_to_cyxgraph.py: {layers with weights, the head}, as
    // docs/usage/pretrained_weights.md lists them.
    const std::map<std::string, std::pair<size_t, std::string>> published = {
        {"resnet18", {41, "fc"}}, {"mobilenet_v2", {105, "classifier.1"}}, {"mobilenet_v3_small", {79, "classifier.3"}}};

    int graphs = 0, blocks = 0, not_ready = 0;
    for (const auto& id : ids) {
        const Pattern* pattern = library.GetPattern(id);
        Check(pattern != nullptr, id + ": the file's id is its file name");
        if (pattern == nullptr) continue;
        const auto defaults = Defaults(*pattern);
        Check(!pattern->description.empty(), id + ": has a description");
        // Custom is where a category the Engine does not know lands: a shipped pattern names a known one.
        Check(pattern->category != gui::patterns::PatternCategory::Custom, id + ": its category is a known one");
        std::set<std::string> node_ids;
        for (const auto& node : pattern->template_data.nodes) {
            Check(node_ids.insert(node.id).second, id + ": node id '" + node.id + "' is used once");
            // An image folder pattern says "one subfolder per class": Data Input opens on that
            // layout, not on a flat folder that needs a labels CSV.
            const auto category = node.params.find("file_category");
            if (node.type == "DataInput" && category != node.params.end() && category->second == "image") {
                const auto layout = node.params.find("image_layout");
                Check(layout != node.params.end() &&
                          layout->second ==
                              std::to_string(static_cast<int>(gui::data_input::ImageLayout::ClassSubdirs)),
                      id + ": Data Input opens on class subfolders");
            }
        }

        Inserted g = Insert(library, *pattern);
        if (!pattern->needs.empty()) {
            ++not_ready;
            Check(!g.ok && g.nodes.empty() && g.links.empty(), id + ": not ready, so it does not insert");
            std::cout << "  " << id << ": not ready - " << pattern->needs << "\n";
            continue;
        }
        Check(g.ok, id + ": inserts");
        if (!g.ok) continue;
        Check(g.links.size() == pattern->template_data.links.size(),
              id + ": every link lands on a pin (" + std::to_string(g.links.size()) + " of " +
                  std::to_string(pattern->template_data.links.size()) + ")");
        Check(!pattern->sample_shape.empty() || FindType(g.nodes, gui::NodeType::DataInput) == nullptr,
              id + ": says what one sample looks like");
        const std::string shape = Substitute(pattern->sample_shape, defaults);

        if (pattern->kind == PatternKind::Block) {
            ++blocks;
            Check(FindType(g.nodes, gui::NodeType::DataInput) == nullptr, id + ": a block brings no data line");
            WrapBlock(g, shape);
            if (Compile(id, g)) std::cout << "  " << id << ": block compiles in a graph\n";
            continue;
        }
        ++graphs;
        if (gui::MLNode* data = FindType(g.nodes, gui::NodeType::DataInput)) data->parameters["shape"] = shape;
        const auto config = Compile(id, g);
        if (!config) continue;
        std::cout << "  " << id << ": graph compiles (" << config->layers.size() << " layers)\n";
        if (const auto it = train.find(id); it != train.end())
            TrainBriefly(*pattern, *config, it->second.first, it->second.second);
        if (const auto it = published.find(id); it != published.end())
            CheckPublishedNames(library, *pattern, it->second.first, it->second.second, shape);
    }
    std::cout << graphs << " graphs, " << blocks << " blocks, " << not_ready << " not ready\n";

    // The Residual Block on a feature map with fewer channels than `filters`:
    // the skip Add reads the block input through its Identity, so the compiler
    // refuses the two shapes (it used to compile and fail at the first batch).
    if (const Pattern* residual = library.GetPattern("resnet_block")) {
        Inserted g = Insert(library, *residual);
        WrapBlock(g, "[16, 16, 32]");
        const auto config = cyxwiz::GraphCompiler{}.Compile(g.nodes, g.links, /*allow_unloaded_data=*/true);
        bool refused = false;
        for (const auto& issue : config.issues) {
            refused |= issue.level == cyxwiz::IssueLevel::Error &&
                       issue.message.find("needs inputs of one shape: [16, 16, 64] and [16, 16, 32]") != std::string::npos;
        }
        Check(!config.is_valid && refused, "resnet_block on 32 channels: the skip Add's shapes are refused");
    }
    if (failures > 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "shipped patterns insert and compile\n";
    return 0;
}
