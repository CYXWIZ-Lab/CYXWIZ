// Image transform nodes in a training graph (TOFIX140 image transforms).
//
// The compiler turns Resize -> transforms -> Normalize into the image
// batcher's device plan and the model's input shape, and refuses the wiring
// and settings the transforms cannot honour. The batcher then runs the plan on
// real image files: random transforms change Train batches only, Random Crop
// centres outside training, and the batch keeps its [N, H, W, C] shape.
#include "../src/core/data_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/image_dataset_batcher.h"
#include "../src/gui/loaders/data_loader.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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

namespace fs = std::filesystem;
using Params = std::map<std::string, std::string>;

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
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

gui::MLNode Layer(int id, gui::NodeType type, const std::string& name, Params parameters = {}) {
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

struct Step {
    gui::NodeType type;
    std::string name;
    Params parameters;
};

// Data Input (images) -> steps... -> Conv2D -> ReLU -> Flatten -> Dense 2 -> Cross entropy -> Adam.
cyxwiz::TrainingConfiguration Compile(const std::vector<Step>& steps,
                                      gui::NodeType loss_type = gui::NodeType::CrossEntropyLoss) {
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Images";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "image_transform_graph"}, {"file_category", "image"},
                       {"folder_path", "images"}, {"image_layout", "0"}};
    std::vector<gui::MLNode> nodes = {data};
    int id = 2;
    for (const auto& step : steps) nodes.push_back(Layer(id++, step.type, step.name, step.parameters));
    nodes.push_back(Layer(id++, gui::NodeType::Conv2D, "Conv 4", {{"filters", "4"}, {"kernel_size", "3"}, {"padding", "same"}}));
    nodes.push_back(Layer(id++, gui::NodeType::ReLU, "ReLU"));
    nodes.push_back(Layer(id++, gui::NodeType::Flatten, "Flatten"));
    nodes.push_back(Layer(id++, gui::NodeType::Dense, "Dense 2", {{"units", "2"}}));
    const int last_layer = id - 1;
    gui::MLNode loss;
    loss.id = id++;
    loss.type = loss_type;
    loss.name = "Loss";
    loss.inputs = {Pin(loss.id * 100 + 1, gui::PinType::Tensor, "Predictions", true),
                   Pin(loss.id * 100 + 2, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(loss.id * 100 + 3, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = id++;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(adam.id * 100 + 1, gui::PinType::Loss, "Loss", true)};
    adam.parameters = {{"learning_rate", "0.001"}};
    nodes.push_back(loss);
    nodes.push_back(adam);

    std::vector<gui::NodeLink> links;
    int link_id = 1;
    links.push_back(Link(link_id++, 1, 102, 2, 201));
    for (int n = 2; n < last_layer; ++n) links.push_back(Link(link_id++, n, n * 100 + 2, n + 1, (n + 1) * 100 + 1));
    links.push_back(Link(link_id++, last_layer, last_layer * 100 + 2, loss.id, loss.id * 100 + 1));
    links.push_back(Link(link_id++, 1, 103, loss.id, loss.id * 100 + 2));
    links.push_back(Link(link_id++, loss.id, loss.id * 100 + 3, adam.id, adam.id * 100 + 1));
    return cyxwiz::GraphCompiler{}.Compile(nodes, links, true);
}

std::string Errors(const cyxwiz::TrainingConfiguration& config) {
    std::string text;
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error) text += issue.node_name + ": " + issue.message + "\n";
    }
    return text;
}

void CheckRefused(const std::vector<Step>& steps, const std::string& node, const std::string& needle,
                  const std::string& what, gui::NodeType loss_type = gui::NodeType::CrossEntropyLoss) {
    const auto config = Compile(steps, loss_type);
    bool found = false;
    for (const auto& issue : config.issues) {
        found = found || (issue.level == cyxwiz::IssueLevel::Error && issue.node_name == node &&
                          issue.message.find(needle) != std::string::npos);
    }
    Check(!config.is_valid && found, what + " is refused on '" + node + "'; errors:\n" + Errors(config));
}

const Step kResize{gui::NodeType::Resize, "Resize 8", {{"width", "8"}, {"height", "8"}, {"mode", "exact"}}};
const Step kNormalize{gui::NodeType::Normalize, "Normalize", {{"mean", "0.5"}, {"std", "0.5"}}};

void CheckCompiler() {
    const auto config = Compile({
        kResize,
        {gui::NodeType::RandomCrop, "Random Crop 6", {{"width", "6"}, {"height", "6"}}},
        {gui::NodeType::HorizontalFlip, "Flip", {{"probability", "0.5"}}},
        {gui::NodeType::ImageRotate, "Rotate", {{"max_angle", "10"}, {"interpolation", "bilinear"}}},
        {gui::NodeType::ColorJitter, "Jitter", {{"brightness", "0.3"}, {"hue", "0"}}},
        {gui::NodeType::Grayscale, "Gray", {}},
        kNormalize,
    });
    Check(config.is_valid, "Resize -> transforms -> Normalize compiles; errors:\n" + Errors(config));
    Check(config.input_shape == std::vector<size_t>({6, 6, 1}),
          "the model's input follows Random Crop 6 and Grayscale: [6, 6, 1]");
    Check(config.input_size == 36, "input_size is 6 x 6 x 1");
    const auto& ops = config.image_augmentation.ops;
    Check(ops.size() == 5, "five transforms compiled in graph order");
    using cyxwiz::image::ImageOpKind;
    Check(ops[0].kind == ImageOpKind::RandomCrop && ops[0].height == 6 && ops[0].width == 6, "Random Crop 6 x 6");
    Check(ops[1].kind == ImageOpKind::HorizontalFlip && ops[1].probability == 0.5f, "Horizontal Flip p 0.5");
    Check(ops[2].kind == ImageOpKind::Rotate && ops[2].max_angle == 10.0f &&
              ops[2].interpolation == cyxwiz::image::Interpolation::Bilinear,
          "Image Rotate 10 degrees, bilinear");
    Check(ops[3].kind == ImageOpKind::ColorJitter && ops[3].brightness == 0.3f && ops[3].hue == 0.0f &&
              ops[3].contrast == 0.2f,
          "Color Jitter brightness 0.3, hue off, contrast at its default");
    Check(ops[4].kind == ImageOpKind::Grayscale, "Grayscale last");
    Check(config.preprocessing.has_normalization && config.preprocessing.norm_mean == 0.5f,
          "Normalize still reaches the batcher");
    Check(!config.layers.empty() && config.layers.front().input_shape == std::vector<size_t>({6, 6, 1}),
          "the first Conv2D sees [6, 6, 1]");

    const auto cifar = Compile({
        kResize,
        {gui::NodeType::RandomCrop, "Random Crop 8 pad 2", {{"width", "8"}, {"height", "8"}, {"padding", "2"}}},
        {gui::NodeType::MorphologyTransform, "Close", {{"operation", "close"}, {"kernel_size", "3"}}},
        kNormalize,
    });
    Check(cifar.is_valid, "padded Random Crop and Morphology compile; errors:\n" + Errors(cifar));
    Check(cifar.input_shape == std::vector<size_t>({8, 8, 3}), "a full-size padded crop keeps [8, 8, 3]");
    Check(cifar.image_augmentation.ops.size() == 2 && cifar.image_augmentation.ops[0].padding == 2 &&
              cifar.image_augmentation.ops[1].kind == ImageOpKind::Morphology &&
              cifar.image_augmentation.ops[1].morphology == cyxwiz::image::MorphologyOp::Close,
          "Random Crop padding 2, then Morphology close");
    const auto erasing = Compile({
        kResize,
        {gui::NodeType::AdvancedAugment, "Erasing", {{"method", "random_erasing"}, {"probability", "0.7"},
                                                     {"scale_max", "0.2"}, {"value", "0.5"}}},
        kNormalize,
    });
    Check(erasing.is_valid, "Advanced Augment random_erasing compiles; errors:\n" + Errors(erasing));
    Check(erasing.image_augmentation.ops.size() == 1 &&
              erasing.image_augmentation.ops[0].kind == ImageOpKind::Erase &&
              erasing.image_augmentation.ops[0].erase_method == cyxwiz::image::EraseMethod::RandomErasing &&
              erasing.image_augmentation.ops[0].probability == 0.7f &&
              erasing.image_augmentation.ops[0].scale_max == 0.2f && erasing.image_augmentation.ops[0].value == 0.5f,
          "Advanced Augment settings reach the plan");
    CheckRefused({kResize, {gui::NodeType::AdvancedAugment, "Old method", {{"method", "Cutout"}}}},
                 "Old method", "method must be cutout, random_erasing, mixup or cutmix",
                 "an unknown Advanced Augment method");

    const auto mixing = Compile({
        kResize,
        {gui::NodeType::HorizontalFlip, "Flip", {}},
        {gui::NodeType::AdvancedAugment, "CutMix", {{"method", "cutmix"}, {"alpha", "0.4"}, {"probability", "1"}}},
        kNormalize,
    });
    Check(mixing.is_valid, "cutmix with Cross Entropy compiles; errors:\n" + Errors(mixing));
    Check(mixing.image_augmentation.ops.size() == 1 &&
              mixing.image_augmentation.mix == cyxwiz::image::BatchMix::CutMix &&
              mixing.image_augmentation.mix_alpha == 0.4f && mixing.image_augmentation.mix_probability == 1.0f,
          "cutmix becomes the plan's batch mix, not a per-image op");
    CheckRefused({kResize, {gui::NodeType::AdvancedAugment, "MixUp", {{"method", "mixup"}}},
                  {gui::NodeType::AdvancedAugment, "Second mix", {{"method", "cutmix"}}}},
                 "Second mix", "Only one mixup or cutmix per graph", "two batch mixes");
    CheckRefused({kResize, {gui::NodeType::AdvancedAugment, "MixUp MSE", {{"method", "mixup"}}}},
                 "MixUp MSE", "use Cross Entropy", "mixup without Cross Entropy", gui::NodeType::MSELoss);
    CheckRefused({kResize, {gui::NodeType::AdvancedAugment, "Alpha 0", {{"method", "mixup"}, {"alpha", "0"}}}},
                 "Alpha 0", "alpha must be greater than 0", "alpha 0");
    CheckRefused({kResize, {gui::NodeType::MorphologyTransform, "Old blur", {{"operation", "blur"}}}},
                 "Old blur", "retired; use Image Gaussian Blur", "the retired Morphology 'blur' operation");

    CheckRefused({kResize, {gui::NodeType::CenterCrop, "Crop 10", {{"width", "10"}, {"height", "10"}}}},
                 "Crop 10", "larger than the 8 x 8 image", "a crop larger than the Resize size");
    CheckRefused({kResize, kNormalize, {gui::NodeType::HorizontalFlip, "Late flip", {}}},
                 "Late flip", "between Resize and Normalize", "a transform after Normalize");
    CheckRefused({kResize, {gui::NodeType::Grayscale, "Gray", {}},
                  {gui::NodeType::Resize, "Second resize", {{"width", "4"}, {"height", "4"}}}},
                 "Second resize", "Resize must come before the image transforms", "Resize after a transform");
    CheckRefused({kResize, {gui::NodeType::ImageGaussianBlur, "Blur 4", {{"kernel_size", "4"}}}},
                 "Blur 4", "positive odd", "an even blur kernel");
    CheckRefused({kResize, {gui::NodeType::ColorJitter, "Hue 0.7", {{"hue", "0.7"}}}},
                 "Hue 0.7", "hue must be between 0 and 0.5", "hue above 0.5");
    CheckRefused({kResize, {gui::NodeType::HorizontalFlip, "Flip x", {{"probability", "often"}}}},
                 "Flip x", "probability must be a float", "a probability that is not a number");
}

void WriteLe(std::ofstream& out, uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out.put(static_cast<char>((value >> (8 * i)) & 0xff));
}

// A 4 x 4 BMP whose pixel (row, column) is (40 * column, 40 * row, 200), top row first.
void WriteGradientBmp(const fs::path& path) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    Check(out.good(), "BMP fixture " + path.string());
    const uint32_t side = 4, row_bytes = 12, data = 54;
    out.write("BM", 2);
    WriteLe(out, data + row_bytes * side, 4);
    WriteLe(out, 0, 4);
    WriteLe(out, data, 4);
    WriteLe(out, 40, 4);
    WriteLe(out, side, 4);
    WriteLe(out, side, 4);
    WriteLe(out, 1, 2);
    WriteLe(out, 24, 2);
    for (int i = 0; i < 6; ++i) WriteLe(out, i == 1 ? row_bytes * side : (i < 4 && i > 1 ? 2835 : 0), 4);
    for (int bmp_row = 0; bmp_row < 4; ++bmp_row) {  // BMP rows are stored bottom first
        const int row = 3 - bmp_row;
        for (int column = 0; column < 4; ++column) {
            out.put(static_cast<char>(200));
            out.put(static_cast<char>(40 * row));
            out.put(static_cast<char>(40 * column));
        }
    }
}

float Red(int column) { return 40.0f * column / 255.0f; }

void CheckBatcher() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_image_transform_graph";
    fs::remove_all(root);
    for (const char* name : {"a/0.bmp", "a/1.bmp", "b/2.bmp", "b/3.bmp"}) WriteGradientBmp(root / name);
    cyxwiz::DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = root.string();
    entry.layout = 0;
    entry.num_images = 4;
    entry.num_classes = 2;
    cyxwiz::ImagePreprocessingConfig resize;
    resize.resize_mode = cyxwiz::ResizeMode::Exact;
    resize.target_width = 4;
    resize.target_height = 4;

    // Every Train image flipped, then a 2 x 2 crop: Train crops vary, validation
    // takes the centre (rows 1-2, columns 1-2) unflipped.
    cyxwiz::image::ImageOp flip;
    flip.kind = cyxwiz::image::ImageOpKind::HorizontalFlip;
    flip.probability = 1.0f;
    cyxwiz::image::ImageOp crop;
    crop.kind = cyxwiz::image::ImageOpKind::RandomCrop;
    crop.height = 2;
    crop.width = 2;
    cyxwiz::ImageDatasetBatcher batcher(entry, resize, 2, 0.5f, false, 0, 11);
    cyxwiz::image::ImageAugmentation flip_crop;
    flip_crop.ops = {flip, crop};
    batcher.SetImageTransforms(flip_crop);

    batcher.SetPhase(cyxwiz::BatcherPhase::Val);
    batcher.Reset();
    const auto val = batcher.GetNextBatch();
    Check(val.size == 2 && val.data.Shape() == std::vector<size_t>({2, 2, 2, 3}),
          "validation batch is [2, 2, 2, 3]");
    const float* v = val.data.ReadData<float>();
    for (size_t s = 0; s < 2; ++s) {
        const float* sample = v + s * 12;
        Check(std::fabs(sample[0] - Red(1)) < 1e-6f && std::fabs(sample[3] - Red(2)) < 1e-6f,
              "validation: centre crop, not flipped (red channel follows the column)");
        Check(std::fabs(sample[1] - 40.0f / 255.0f) < 1e-6f, "validation: the crop starts at row 1");
    }

    batcher.SetPhase(cyxwiz::BatcherPhase::Train);
    batcher.Reset();
    const auto train = batcher.GetNextBatch();

    // MixUp on every Train batch: one-hot labels come out mixed (rows still sum
    // to 1), validation labels stay one-hot.
    cyxwiz::image::ImageAugmentation mixup;
    mixup.mix = cyxwiz::image::BatchMix::MixUp;
    mixup.mix_probability = 1.0f;
    cyxwiz::ImageDatasetBatcher mixing(entry, resize, 2, 0.5f, false, 0, 11);
    mixing.SetOneHotEncoding(2);
    mixing.SetImageTransforms(mixup);
    mixing.SetPhase(cyxwiz::BatcherPhase::Val);
    mixing.Reset();
    const auto plain = mixing.GetNextBatch();
    const float* plain_labels = plain.labels.ReadData<float>();
    Check(plain_labels[0] + plain_labels[1] == 1.0f && (plain_labels[0] == 0.0f || plain_labels[0] == 1.0f),
          "validation labels stay one-hot");
    mixing.SetPhase(cyxwiz::BatcherPhase::Train);
    mixing.Reset();
    const auto mixed = mixing.GetNextBatch();
    Check(mixed.labels.Shape() == std::vector<size_t>({2, 2}), "mixed labels keep [N, C]");
    const float* mixed_labels = mixed.labels.ReadData<float>();
    for (size_t s = 0; s < 2; ++s) {
        Check(std::fabs(mixed_labels[2 * s] + mixed_labels[2 * s + 1] - 1.0f) < 1e-5f,
              "mixed label rows still sum to 1");
    }
    Check(train.size == 2, "training batch has two images");
    const float* t = train.data.ReadData<float>();
    for (size_t s = 0; s < 2; ++s) {
        const float* sample = t + s * 12;
        Check(sample[0] > sample[3], "training: flipped, so red falls from left to right");
    }
}

}  // namespace

int main() {
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

    CheckCompiler();
    CheckBatcher();
    std::cout << "image transform graph contract passed (" << checks << " checks)\n";
    return 0;
}
