// A CNN trained on real images through the Engine's training path
// (TOFIX140 A1/A1b): the cats/dogs graph is compiled like the Studio does,
// then trained with the objects TrainingManager::StartTrainingImage builds -
// an ImageDatasetBatcher on an image folder with class subfolders (default
// D:/tmp/gui129/catdog_small, or argv[1]) and a TrainingExecutor - and the
// loss per epoch is printed. Not a ctest entry: the images live outside the repo.
//
//   test_cnn_image_training_smoke [folder] [epochs]
//   (CYXWIZ_TEST_ARRAYFIRE_BACKEND / CYXWIZ_OPENCL_TEST_DEVICE pick the device)
#include "../src/core/data_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/image_dataset_batcher.h"
#include "../src/core/training_executor.h"
#include "../src/gui/loaders/data_loader.h"
#include "computation_truth/test_device_selection.h"
#include "route_qualification_test_fixture.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Graph-only: no file loaders.
namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string& text) {
    return text == "image" ? FileCategory::Image : FileCategory::Tabular;
}
}  // namespace cyxwiz::loaders

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

gui::NodePin Pin(int id, gui::PinType type, const std::string& name, bool is_input) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
    return pin;
}

gui::MLNode Layer(int id, gui::NodeType type, const std::string& name,
                  std::map<std::string, std::string> parameters = {}) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = {Pin(id * 100 + 1, gui::PinType::Tensor, "Input", true)};
    node.outputs = {Pin(id * 100 + 2, gui::PinType::Tensor, "Output", false)};
    node.parameters = std::move(parameters);
    return node;
}

gui::NodeLink Link(int id, int from_node, int from_pin, int to_node, int to_pin) {
    gui::NodeLink link;
    link.id = id;
    link.from_node = from_node;
    link.from_pin = from_pin;
    link.to_node = to_node;
    link.to_pin = to_pin;
    return link;
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const fs::path folder = argc > 1 ? fs::path(argv[1]) : fs::path("D:/tmp/gui129/catdog_small");
    const int epochs = argc > 2 ? std::atoi(argv[2]) : 8;
    if (!fs::is_directory(folder)) {
        std::cout << "image folder not found (" << folder.string() << "); skipped\n";
        return 0;
    }
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested device");
    // A verified device route, as Preferences > Devices > Verify records it.
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    // The image folder: one subfolder per class.
    cyxwiz::DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = folder.string();
    entry.layout = 0;
    for (const auto& dir : fs::directory_iterator(folder)) {
        if (!dir.is_directory()) continue;
        entry.class_names.push_back(dir.path().filename().string());
        for (const auto& file : fs::directory_iterator(dir.path())) {
            if (file.is_regular_file()) ++entry.num_images;
        }
    }
    entry.num_classes = entry.class_names.size();
    Check(entry.num_classes == 2 && entry.num_images > 0, "two class folders with images");

    // The graph: the CNN of test_cnn_graph_training (the image batcher scales pixels).
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.preprocessing_domain = [](const std::string& c) -> std::optional<cyxwiz::PreprocessingDomain> {
        if (c == "image") return cyxwiz::PreprocessingDomain::Image;
        return std::nullopt;
    };
    hooks.labels_from_structure = [](const std::string& c) -> std::optional<bool> {
        if (c == "image") return true;
        return std::nullopt;
    };
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Cats and dogs";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "catdog_smoke"}, {"file_category", "image"},
                       {"folder_path", folder.string()}, {"image_layout", "0"}};
    std::vector<gui::MLNode> nodes = {
        data,
        Layer(2, gui::NodeType::Resize, "Resize 64", {{"width", "64"}, {"height", "64"}, {"mode", "exact"}}),
        Layer(4, gui::NodeType::Conv2D, "Conv 16", {{"filters", "16"}, {"kernel_size", "3"}, {"padding", "same"}}),
        Layer(5, gui::NodeType::ReLU, "ReLU"),
        Layer(6, gui::NodeType::MaxPool2D, "Pool", {{"pool_size", "2"}, {"stride", "2"}}),
        Layer(7, gui::NodeType::Conv2D, "Conv 32", {{"filters", "32"}, {"kernel_size", "3"}, {"padding", "same"}}),
        Layer(8, gui::NodeType::ReLU, "ReLU 2"),
        Layer(9, gui::NodeType::MaxPool2D, "Pool 2", {{"pool_size", "2"}, {"stride", "2"}}),
        Layer(10, gui::NodeType::Flatten, "Flatten"),
        Layer(11, gui::NodeType::Dense, "Dense 2", {{"units", "2"}}),
    };
    gui::MLNode loss;
    loss.id = 12;
    loss.type = gui::NodeType::CrossEntropyLoss;
    loss.name = "Cross entropy";
    loss.inputs = {Pin(1201, gui::PinType::Tensor, "Predictions", true), Pin(1202, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(1203, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = 13;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(1301, gui::PinType::Loss, "Loss", true)};
    adam.parameters = {{"learning_rate", "0.001"}};
    nodes.push_back(loss);
    nodes.push_back(adam);
    std::vector<gui::NodeLink> links;
    int link_id = 1;
    links.push_back(Link(link_id++, 1, 102, 2, 201));
    links.push_back(Link(link_id++, 2, 202, 4, 401));
    for (int id = 4; id < 11; ++id) links.push_back(Link(link_id++, id, id * 100 + 2, id + 1, (id + 1) * 100 + 1));
    links.push_back(Link(link_id++, 11, 1102, 12, 1201));
    links.push_back(Link(link_id++, 1, 103, 12, 1202));
    links.push_back(Link(link_id++, 12, 1203, 13, 1301));

    auto config = cyxwiz::GraphCompiler{}.Compile(nodes, links, true);
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error)
            std::cerr << "  error: " << issue.node_name << ": " << issue.message << "\n";
    }
    Check(config.is_valid, "the CNN graph compiles");
    config.epochs = epochs;
    config.batch_size = 32;

    // What TrainingManager::StartTrainingImage builds.
    auto batcher = std::make_unique<cyxwiz::ImageDatasetBatcher>(
        entry, config.image_preprocessing, config.batch_size, config.train_ratio, config.shuffle,
        config.num_workers, static_cast<uint32_t>(config.dataloader_seed));
    batcher->SetDropLast(config.drop_last);
    Check(batcher->GetNumSamples() > 0, "the batcher sees the images");
    if (config.preprocessing.has_normalization) {
        batcher->SetNormalization(config.preprocessing.norm_mean, config.preprocessing.norm_std);
    }
    if (config.preprocessing.has_onehot && config.preprocessing.num_classes > 0) {
        batcher->SetOneHotEncoding(config.preprocessing.num_classes);
    }
    cyxwiz::TrainingExecutor executor(config, std::move(batcher));

    std::mutex mutex;
    std::vector<std::pair<float, float>> per_epoch;  // loss, accuracy
    std::atomic<bool> complete{false};
    const auto start = std::chrono::steady_clock::now();
    executor.Train(
        epochs, config.batch_size, nullptr,
        [&](int epoch, float train_loss, float train_acc, float val_loss, float val_acc, float epoch_time) {
            std::lock_guard<std::mutex> lock(mutex);
            per_epoch.emplace_back(train_loss, train_acc);
            std::cout << "  epoch " << epoch << ": loss " << train_loss << ", accuracy " << train_acc
                      << " | val loss " << val_loss << ", val accuracy " << val_acc << " (" << epoch_time
                      << " s)" << std::endl;
        },
        [&](const cyxwiz::TrainingMetrics&) { complete.store(true); });
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::lock_guard<std::mutex> lock(mutex);
    Check(complete.load(), "training completes");
    Check(static_cast<int>(per_epoch.size()) == epochs, "one report per epoch");
    std::cout << "trained " << epochs << " epochs on " << entry.num_images << " images in " << seconds
              << " s; loss " << per_epoch.front().first << " -> " << per_epoch.back().first << "\n";
    Check(per_epoch.back().first < per_epoch.front().first, "the loss falls");
    return 0;
}
