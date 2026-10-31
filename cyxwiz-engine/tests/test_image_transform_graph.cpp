// Image transform nodes in a training graph (TOFIX140 image transforms).
//
// The compiler turns Resize -> transforms -> Normalize into the image
// batcher's device plan and the model's input shape, and refuses the wiring
// and settings the transforms cannot honour. The batcher then runs the plan on
// real image files: random transforms change Train batches only, Random Crop
// centres outside training, and the batch keeps its [N, H, W, C] shape.
// The Quality Analyzer: training refuses a missing, stale or unusable
// analysis, and the batcher leaves the rejected files out.
// The image split (TOFIX178 1.5): the compile counts the parts the batcher
// makes, says them, warns of a class too small for a part and refuses a
// wired dataset with other classes.
#include "../src/core/data_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/image_dataset_batcher.h"
#include "../src/core/image_index.h"
#include "../src/core/image_split_counts.h"
#include "../src/core/image_quality_analysis.h"
#include "../src/core/normalize_statistics.h"
#include "../src/gui/loaders/data_loader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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

namespace cyxwiz {
// graph_compiler.cpp: the image split part of a compile.
void CompileImageSplit(TrainingConfiguration& config, const std::vector<gui::MLNode>& nodes);
}  // namespace cyxwiz

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
                                      gui::NodeType loss_type = gui::NodeType::CrossEntropyLoss,
                                      const std::string& dataset = "image_transform_graph") {
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Images";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", dataset}, {"file_category", "image"},
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
                  const std::string& what, gui::NodeType loss_type = gui::NodeType::CrossEntropyLoss,
                  const std::string& dataset = "image_transform_graph") {
    const auto config = Compile(steps, loss_type, dataset);
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
                 "Old method",
                 "method must be cutout, random_erasing, mixup, cutmix, randaugment, autoaugment, trivialaugment or augmix",
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
    const auto rand = Compile({
        kResize,
        {gui::NodeType::AdvancedAugment, "RandAugment", {{"method", "randaugment"}, {"num_ops", "3"},
                                                         {"magnitude", "12"}, {"probability", "1"}}},
        kNormalize,
    });
    Check(rand.is_valid && rand.image_augmentation.ops.size() == 1 &&
              rand.image_augmentation.ops[0].kind == ImageOpKind::RandAugment &&
              rand.image_augmentation.ops[0].num_ops == 3 && rand.image_augmentation.ops[0].magnitude == 12,
          "randaugment settings reach the plan");
    CheckRefused({kResize, {gui::NodeType::AdvancedAugment, "Too strong", {{"method", "randaugment"},
                                                                           {"magnitude", "40"}}}},
                 "Too strong", "magnitude must be between 0 and 30", "a magnitude past 30");
    const auto preset = Compile({kResize, {gui::NodeType::AugmentationPreset, "CIFAR", {{"preset", "cifar_cutout"}}},
                                 kNormalize});
    const auto& preset_ops = preset.image_augmentation.ops;
    Check(preset.is_valid && preset_ops.size() == 3 && preset_ops[0].kind == ImageOpKind::RandomCrop &&
              preset_ops[0].height == 8 && preset_ops[0].width == 8 && preset_ops[0].padding == 4 &&
              preset_ops[1].kind == ImageOpKind::HorizontalFlip && preset_ops[2].kind == ImageOpKind::Erase &&
              preset_ops[2].cutout_size == 16 && preset_ops[2].probability == 1.0f,
          "cifar_cutout expands to Random Crop 8 pad 4, Horizontal Flip, cutout 16");
    Check(preset.input_shape == std::vector<size_t>({8, 8, 3}), "the cifar crop keeps the Resize size");
    const auto rand_preset = Compile({kResize, {gui::NodeType::AugmentationPreset, "RA", {{"preset", "randaugment"}}}});
    Check(rand_preset.is_valid && rand_preset.image_augmentation.ops.size() == 2 &&
              rand_preset.image_augmentation.ops[1].kind == ImageOpKind::RandAugment,
          "randaugment preset expands to Horizontal Flip, RandAugment");
    // The augmentation steps of TOFIX178 1.6b: settings reach the plan; a
    // Random Resized Crop of 0 x 0 keeps the size coming in.
    const auto steps = Compile({
        kResize,
        {gui::NodeType::RandomResizedCrop, "Resized crop", {}},
        {gui::NodeType::RandomResizedCrop, "Resized crop 5x6", {{"width", "6"}, {"height", "5"}, {"interpolation", "nearest"}}},
        {gui::NodeType::RandomAffine, "Affine", {{"degrees", "20"}, {"shear_x_max", "4"}, {"fill", "0.5"}}},
        {gui::NodeType::RandomPerspective, "Perspective", {{"distortion", "0.3"}}},
        {gui::NodeType::GaussianNoise, "Noise", {{"sigma", "0.05"}, {"clip", "false"}}},
        {gui::NodeType::RandomGrayscale, "Random gray", {{"probability", "0.2"}}},
        {gui::NodeType::AdvancedAugment, "AutoAugment", {{"method", "autoaugment"}, {"policy", "svhn"}}},
        {gui::NodeType::AdvancedAugment, "Trivial", {{"method", "trivialaugment"}, {"magnitude_bins", "20"}}},
        {gui::NodeType::AdvancedAugment, "AugMix", {{"method", "augmix"}, {"severity", "5"}, {"chain_depth", "2"}}},
        kNormalize,
    });
    Check(steps.is_valid, "the 1.6b steps compile; errors:\n" + Errors(steps));
    const auto& s16 = steps.image_augmentation.ops;
    Check(s16.size() == 9, "nine augmentation steps in graph order");
    Check(s16[0].kind == ImageOpKind::RandomResizedCrop && s16[0].height == 8 && s16[0].width == 8 &&
              s16[0].scale_min == 0.08f && s16[0].ratio_max == 1.3333f &&
              s16[0].interpolation == cyxwiz::image::Interpolation::Bilinear,
          "Random Resized Crop 0 x 0 keeps 8 x 8, torchvision's scale and ratio, bilinear");
    Check(s16[1].height == 5 && s16[1].width == 6 && s16[1].interpolation == cyxwiz::image::Interpolation::Nearest,
          "Random Resized Crop 5 x 6, nearest");
    Check(steps.input_shape == std::vector<size_t>({5, 6, 3}), "the model's input follows the resized crop");
    Check(s16[2].kind == ImageOpKind::RandomAffine && s16[2].max_angle == 20.0f && s16[2].translate_x == 0.1f &&
              s16[2].zoom_min == 0.9f && s16[2].zoom_max == 1.1f && s16[2].shear_x_max == 4.0f && s16[2].fill == 0.5f,
          "Random Affine settings with the node's starting values");
    Check(s16[3].kind == ImageOpKind::RandomPerspective && s16[3].distortion == 0.3f && s16[3].probability == 0.5f &&
              s16[3].interpolation == cyxwiz::image::Interpolation::Bilinear,
          "Random Perspective distortion 0.3, bilinear");
    Check(s16[4].kind == ImageOpKind::GaussianNoise && s16[4].sigma == 0.05f && !s16[4].clip && s16[4].noise_mean == 0.0f,
          "Gaussian Noise sigma 0.05, not clipped");
    Check(s16[5].kind == ImageOpKind::RandomGrayscale && s16[5].probability == 0.2f, "Random Grayscale p 0.2");
    Check(s16[6].kind == ImageOpKind::AutoAugment && s16[6].policy == cyxwiz::image::AutoAugmentPolicy::Svhn,
          "AutoAugment with the SVHN policy");
    Check(s16[7].kind == ImageOpKind::TrivialAugment && s16[7].magnitude_bins == 20, "TrivialAugment 20 bins");
    Check(s16[8].kind == ImageOpKind::AugMix && s16[8].severity == 5 && s16[8].chain_depth == 2 &&
              s16[8].mixture_width == 3 && s16[8].interpolation == cyxwiz::image::Interpolation::Bilinear,
          "AugMix severity 5, depth 2, three chains, bilinear");
    CheckRefused({kResize, {gui::NodeType::AdvancedAugment, "MNIST policy", {{"method", "autoaugment"},
                                                                             {"policy", "mnist"}}}},
                 "MNIST policy", "policy must be imagenet, cifar10 or svhn", "an unknown AutoAugment policy");
    CheckRefused({kResize, {gui::NodeType::RandomAffine, "Shear", {{"shear_y_min", "10"}}}},
                 "Shear", "min <= max", "a reversed shear range");
    CheckRefused({kResize, kNormalize, {gui::NodeType::GaussianNoise, "Late noise", {}}},
                 "Late noise", "between Resize and Normalize", "noise after Normalize");
    CheckRefused({kResize, {gui::NodeType::RandomResizedCrop, "Area", {{"scale_max", "2"}}}},
                 "Area", "scale_max <= 1", "a resized crop area past the image");
    CheckRefused({kResize, {gui::NodeType::AugmentationPreset, "ImageNet", {{"preset", "ImageNet"}}}},
                 "ImageNet", "preset must be flip, cifar, cifar_cutout or randaugment", "an unknown preset");
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

    // OpenCV steps (TOFIX178 1.3f): placed among the transforms in canvas
    // order, done once, so before the first random one.
    const auto cv = Compile({kResize,
                             {gui::NodeType::ImageContrast, "CLAHE", {{"clip_limit", "3"}, {"tile_grid", "4"}}},
                             {gui::NodeType::CenterCrop, "Crop 6", {{"width", "6"}, {"height", "6"}}},
                             {gui::NodeType::ImageEdges, "Edges", {{"method", "sobel"}}},
                             {gui::NodeType::HorizontalFlip, "Flip", {}},
                             kNormalize});
    Check(cv.is_valid, "CLAHE, Crop, Edges, Flip compiles; errors:\n" + Errors(cv));
    Check(cv.image_cv_steps.size() == 2 && cv.image_cv_steps[0].position == 0 && cv.image_cv_steps[1].position == 1 &&
              cv.image_cv_steps[0].clip_limit == 3.0f && cv.image_cv_steps[0].tile_grid == 4 &&
              cv.image_cv_steps[1].edge_method == cyxwiz::CvEdgeMethod::Sobel,
          "two OpenCV steps with their settings and places");
    Check(cv.input_shape == std::vector<size_t>({6, 6, 1}) && !cv.layers.empty() &&
              cv.layers.front().input_shape == std::vector<size_t>({6, 6, 1}),
          "Edges gives one channel: the Conv2D sees [6, 6, 1]");
    bool told = false;
    for (const auto& issue : cv.issues)
        told = told || (issue.level == cyxwiz::IssueLevel::Info && issue.node_name == "Edges" &&
                        issue.message.find("one channel") != std::string::npos);
    Check(told, "Edges says it gives one channel");
    const auto denoise = Compile({kResize, {gui::NodeType::ImageDenoise, "Denoise", {}}});
    bool warned = false;
    for (const auto& issue : denoise.issues)
        warned = warned || (issue.level == cyxwiz::IssueLevel::Warning && issue.node_name == "Denoise" &&
                            issue.message.find("ms an image") != std::string::npos);
    Check(denoise.is_valid && warned, "Denoise compiles with a warning on its cost");
    CheckRefused({kResize, {gui::NodeType::HorizontalFlip, "Flip", {}}, {gui::NodeType::ImageContrast, "Late CLAHE", {}}},
                 "Late CLAHE", "move it before Horizontal Flip", "an OpenCV step after a random one");
    CheckRefused({kResize, {gui::NodeType::ImagePyramid, "Pyramid", {{"levels", "2"}}}},
                 "Pyramid", "under 5 pixels a side: use a larger Resize", "a pyramid too deep for 8 x 8");
    CheckRefused({kResize, {gui::NodeType::ImageWarp, "Warp", {{"source_points", "0,0, 1,0, 1.5,1, 0,1"}}}},
                 "Warp", "Corner 3 of the source is outside the image", "a corner outside the image");
    CheckRefused({kResize, {gui::NodeType::ImageWarp, "Warp 3", {{"source_points", "0,0, 1,0"}}}},
                 "Warp 3", "must be 8 numbers", "too few points");
    CheckRefused({kResize, {gui::NodeType::ImageSmooth, "Smooth", {{"method", "gaussian"}}}},
                 "Smooth", "method must be box, median or bilateral", "an unknown smoothing");
    CheckRefused({kResize, kNormalize, {gui::NodeType::ImageSharpen, "Late sharpen", {}}},
                 "Late sharpen", "between Resize and Normalize", "an OpenCV step after Normalize");
    CheckRefused({kResize, {gui::NodeType::ImageSharpen, "Sharpen", {}},
                  {gui::NodeType::Resize, "Second resize", {{"width", "4"}, {"height", "4"}}}},
                 "Second resize", "Resize must come before the image transforms", "Resize after an OpenCV step");

    // Normalize per channel (TOFIX178 1.4).
    const auto per_channel = Compile({kResize, {gui::NodeType::Normalize, "Normalize",
                                                {{"statistics", "per_channel"},
                                                 {"channel_mean", "0.5282,0.4924,0.4528"},
                                                 {"channel_std", "0.2346, 0.2270, 0.2304"}}}});
    Check(cyxwiz::ChannelSummary("per channel", {0.5282f, 0.4924f, 0.4528f}, {0.2346f, 0.2270f, 0.2304f}) ==
              "per channel, mean 0.528 / 0.492 / 0.453, std 0.235 / 0.227 / 0.230",
          "the per-channel summary the compile report and the Data Input dialog show");
    Check(per_channel.is_valid, "per channel compiles; errors:\n" + Errors(per_channel));
    Check(per_channel.preprocessing.norm_channel_mean == std::vector<float>({0.5282f, 0.4924f, 0.4528f}) &&
              per_channel.preprocessing.norm_channel_std == std::vector<float>({0.2346f, 0.2270f, 0.2304f}) &&
              per_channel.preprocessing.norm_channels == 3 && per_channel.preprocessing.norm_node_id > 0,
          "per channel: three means and stds for RGB");
    const auto imagenet = Compile({kResize, {gui::NodeType::Normalize, "Normalize", {{"statistics", "imagenet"}}}});
    bool noted = false;
    for (const auto& issue : imagenet.issues)
        noted = noted || (issue.level == cyxwiz::IssueLevel::Info && issue.message.find("pretrained on ImageNet") != std::string::npos);
    Check(imagenet.is_valid && imagenet.preprocessing.norm_channel_mean == std::vector<float>({0.485f, 0.456f, 0.406f}) &&
              imagenet.preprocessing.norm_channel_std == std::vector<float>({0.229f, 0.224f, 0.225f}) && noted,
          "ImageNet: torchvision's values, with a note");
    const auto same = Compile({kResize, kNormalize});
    Check(same.is_valid && same.preprocessing.norm_channel_mean.empty() && same.preprocessing.norm_statistics == "same",
          "same for every channel: one pair, as before");
    CheckRefused({kResize, {gui::NodeType::Grayscale, "Gray", {}},
                  {gui::NodeType::Normalize, "Normalize", {{"statistics", "per_channel"},
                                                           {"channel_mean", "0.5,0.5,0.5"}, {"channel_std", "0.2,0.2,0.2"}}}},
                 "Normalize", "3 means for 1 channel", "three means after Grayscale");
    CheckRefused({kResize, {gui::NodeType::Normalize, "Normalize", {{"statistics", "per_channel"},
                                                                    {"channel_mean", "0.5,0.5,0.5"}, {"channel_std", "0.2,0,0.2"}}}},
                 "Normalize", "Std G is 0", "a std of 0");
    CheckRefused({kResize, {gui::NodeType::Normalize, "Normalize", {{"statistics", "per_channel"}}}},
                 "Normalize", "a mean and a std for each channel", "per channel without values");
    CheckRefused({kResize, {gui::NodeType::Grayscale, "Gray", {}},
                  {gui::NodeType::Normalize, "Normalize", {{"statistics", "imagenet"}}}},
                 "Normalize", "ImageNet statistics are for RGB images", "ImageNet on one channel");
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
    cyxwiz::ImageDatasetBatcher batcher(
        cyxwiz::PrepareImageSet(entry, cyxwiz::PreparedImageSpecFrom(resize)).set, 2, {0.5f, 0.5f, 0.0f, false, 11}, false, 11);
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
    cyxwiz::ImageDatasetBatcher mixing(
        cyxwiz::PrepareImageSet(entry, cyxwiz::PreparedImageSpecFrom(resize)).set, 2, {0.5f, 0.5f, 0.0f, false, 11}, false, 11);
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

    // Normalize per channel (TOFIX178 1.4): each channel its own mean and std.
    cyxwiz::ImageDatasetBatcher normalized(
        cyxwiz::PrepareImageSet(entry, cyxwiz::PreparedImageSpecFrom(resize)).set, 2, {0.5f, 0.5f, 0.0f, false, 11}, false, 11);
    normalized.SetNormalization(0.5f, 0.5f);
    normalized.SetChannelNormalization({0.1f, 0.2f, 0.3f}, {0.5f, 0.25f, 0.125f});
    normalized.SetPhase(cyxwiz::BatcherPhase::Val);
    normalized.Reset();
    const auto n_batch = normalized.GetNextBatch();
    const float* nv = n_batch.data.ReadData<float>();
    double worst = 0.0;
    for (size_t s = 0; s < 2; ++s)
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column) {
                const float* px = nv + s * 48 + (row * 4 + column) * 3;
                const float rgb[3] = {Red(column), 40.0f * row / 255.0f, 200.0f / 255.0f};
                const float mean[3] = {0.1f, 0.2f, 0.3f}, sd[3] = {0.5f, 0.25f, 0.125f};
                for (int c = 0; c < 3; ++c) worst = std::max<double>(worst, std::fabs(px[c] - (rgb[c] - mean[c]) / sd[c]));
            }
    Check(worst < 1e-5, "per channel normalize on the device: (x - mean[c]) / std[c] (worst " + std::to_string(worst) + ")");
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

// A side x side 24-bit BMP; pixel(row, column) gives (r, g, b), top row first.
void WriteBmp(const fs::path& path, int side, const std::function<std::array<int, 3>(int, int)>& pixel) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    Check(out.good(), "BMP fixture " + path.string());
    const uint32_t row_bytes = static_cast<uint32_t>((side * 3 + 3) / 4 * 4), data = 54;
    out.write("BM", 2);
    WriteLe(out, data + row_bytes * side, 4);
    WriteLe(out, 0, 4);
    WriteLe(out, data, 4);
    WriteLe(out, 40, 4);
    WriteLe(out, static_cast<uint32_t>(side), 4);
    WriteLe(out, static_cast<uint32_t>(side), 4);
    WriteLe(out, 1, 2);
    WriteLe(out, 24, 2);
    for (int i = 0; i < 6; ++i) WriteLe(out, i == 1 ? row_bytes * side : (i < 4 && i > 1 ? 2835 : 0), 4);
    for (int row = side - 1; row >= 0; --row) {  // BMP rows are stored bottom first
        for (int column = 0; column < side; ++column) {
            const auto rgb = pixel(row, column);
            out.put(static_cast<char>(rgb[2]));
            out.put(static_cast<char>(rgb[1]));
            out.put(static_cast<char>(rgb[0]));
        }
        for (uint32_t pad = static_cast<uint32_t>(side) * 3; pad < row_bytes; ++pad) out.put(0);
    }
}

// Sharp, textured scene number `seed`.
std::array<int, 3> Texture(int seed, int row, int column) {
    const uint32_t h = static_cast<uint32_t>(seed * 7919 + row * 131 + column * 31) * 2654435761u;
    const int v = 30 + static_cast<int>((h >> 8) % 196);
    return {v, (v * 3 + seed * 40) % 226 + 15, 255 - v};
}

void CheckQualityAnalyzer() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_quality_analyzer_graph";
    fs::remove_all(root);
    const int side = 16;
    for (int i = 0; i < 3; ++i) {
        WriteBmp(root / "a" / (std::to_string(i) + ".bmp"), side, [i](int r, int c) { return Texture(i, r, c); });
        WriteBmp(root / "b" / (std::to_string(i) + ".bmp"), side, [i](int r, int c) { return Texture(10 + i, r, c); });
    }
    const fs::path flat = root / "b" / "flat.bmp";  // flat grey: blurry and low contrast
    WriteBmp(flat, side, [](int, int) { return std::array<int, 3>{128, 128, 128}; });
    const fs::path copy = root / "b" / "copy.bmp";  // a copy of a/0
    WriteBmp(copy, side, [](int r, int c) { return Texture(0, r, c); });

    cyxwiz::DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = root.string();
    entry.layout = 0;
    entry.num_images = 8;
    entry.num_classes = 2;
    entry.class_names = {"a", "b"};
    cyxwiz::DataRegistry::Instance().RegisterImageDataset("quality_graph", entry);

    const Step quality{gui::NodeType::QualityAnalyzer, "Quality", {}};
    const Step resize16{gui::NodeType::Resize, "Resize 16", {{"width", "16"}, {"height", "16"}, {"mode", "exact"}}};
    const auto ce = gui::NodeType::CrossEntropyLoss;

    const std::string first_key = cyxwiz::ImageQualityKey(entry, side, side);
    std::error_code ec;
    fs::remove(cyxwiz::ImageQualityCacheDir() / (first_key + ".json"), ec);
    CheckRefused({quality, resize16}, "Quality", "not analyzed yet", "training without an analysis", ce, "quality_graph");

    const auto analysis = cyxwiz::AnalyzeImageQuality(entry, side, side, nullptr, nullptr);
    Check(analysis && analysis->files.size() == 8 && analysis->key == first_key, "all 8 images measured");
    std::string error;
    Check(cyxwiz::SaveImageQualityAnalysis(*analysis, &error), "analysis saved: " + error);
    const auto loaded = cyxwiz::LoadImageQualityAnalysis(first_key);
    Check(loaded && loaded->metrics.size() == 8 && loaded->metrics[3].hash == analysis->metrics[3].hash,
          "the saved analysis reads back");

    // Defaults: the flat image fails blur and contrast, the copy is a duplicate.
    const auto config = Compile({quality, resize16}, ce, "quality_graph");
    Check(config.is_valid, "an analyzed dataset trains; errors:\n" + Errors(config));
    std::vector<std::string> excluded = config.image_excluded_files;
    std::sort(excluded.begin(), excluded.end());
    std::vector<std::string> expected = {copy.string(), flat.string()};
    std::sort(expected.begin(), expected.end());
    Check(excluded == expected, "the flat image and the copy are left out (" + std::to_string(excluded.size()) + ")");
    bool told = false;
    for (const auto& issue : config.issues) {
        told = told || issue.message.find("leaves out 2 of 8 images") != std::string::npos;
    }
    Check(told, "the compile says how many images are left out");

    const auto verdict = cyxwiz::JudgeImageQualityAnalysis(*loaded, cyxwiz::image::ImageQualityChecks{});
    Check(verdict.rejected == 2 && verdict.duplicates == 1 && verdict.multiple == 1 &&
              verdict.class_total == std::vector<size_t>({3, 5}) &&
              verdict.class_rejected == std::vector<size_t>({0, 2}),
          "per reason and per class counts");

    // A check turned off changes the verdict without a new analysis.
    const auto lenient = Compile(
        {{gui::NodeType::QualityAnalyzer, "Quality", {{"duplicate_check", "false"}}}, resize16}, ce, "quality_graph");
    Check(lenient.is_valid && lenient.image_excluded_files == std::vector<std::string>({flat.string()}),
          "duplicates off: only the flat image is left out");

    CheckRefused({{gui::NodeType::QualityAnalyzer, "Quality", {{"brightness_min", "230"}}}, resize16}, "Quality",
                 "darkest < brightest", "darkest above brightest", ce, "quality_graph");
    CheckRefused({quality, {gui::NodeType::QualityAnalyzer, "Quality 2", {}}, resize16}, "Quality 2",
                 "Only one Quality Analyzer", "a second Quality Analyzer", ce, "quality_graph");
    CheckRefused({{gui::NodeType::QualityAnalyzer, "Quality",
                   {{"blur_min", "1e9"}, {"brightness_check", "false"}, {"contrast_check", "false"}}},
                  resize16},
                 "Quality", "leave out all 8 images", "checks that leave nothing", ce, "quality_graph");

    // Another Resize size, or another image: the analysis is stale.
    const Step analyzed{gui::NodeType::QualityAnalyzer, "Quality", {{"analysis_key", first_key}}};
    CheckRefused({analyzed, {gui::NodeType::Resize, "Resize 12", {{"width", "12"}, {"height", "12"}, {"mode", "exact"}}}},
                 "Quality", "changed since the analysis", "a new Resize size", ce, "quality_graph");
    WriteBmp(root / "a" / "new.bmp", side, [](int r, int c) { return Texture(99, r, c); });
    CheckRefused({analyzed, resize16}, "Quality", "changed since the analysis", "an added image", ce, "quality_graph");
    fs::remove(root / "a" / "new.bmp");
    Check(Compile({analyzed, resize16}, ce, "quality_graph").is_valid, "removing it again matches the analysis");

    // The batcher leaves the files out and splits the rest: 6 images, 3 + 3.
    cyxwiz::ImagePreprocessingConfig resize;
    resize.resize_mode = cyxwiz::ResizeMode::Exact;
    resize.target_width = side;
    resize.target_height = side;
    cyxwiz::ImageDatasetBatcher batcher(
        cyxwiz::PrepareImageSet(entry, cyxwiz::PreparedImageSpecFrom(resize)).set, 8, {0.5f, 0.5f, 0.0f, false, 7}, false, 7);
    Check(batcher.GetNumSamples() + batcher.SamplesIn(cyxwiz::BatcherPhase::Val) == 8, "8 images before the filter");
    batcher.ExcludeFiles(config.image_excluded_files);
    Check(batcher.GetNumSamples() == 3 && batcher.SamplesIn(cyxwiz::BatcherPhase::Val) == 3, "6 images after it, split 3 + 3");

    // As the canvas builds it: Data Input -> Quality Analyzer -> Data Split ->
    // Data Loader -> Resize -> model, the Loader's labels into the loss. The
    // filter keeps the Data Input as the Split's source.
    {
        int next_node = 1, next_pin = 1;
        const auto make = [&](gui::NodeType type, const std::string& name) {
            return gui::CreateGraphNode(type, name, next_node, next_pin);
        };
        auto data = make(gui::NodeType::DataInput, "Images");
        data.parameters["dataset_name"] = "quality_graph";
        data.parameters["file_category"] = "image";
        data.parameters["folder_path"] = root.string();
        data.parameters["image_layout"] = "0";
        auto filter = make(gui::NodeType::QualityAnalyzer, "Quality");
        auto split = make(gui::NodeType::DataSplit, "Split");
        auto loader = make(gui::NodeType::DataLoader, "Loader");
        auto size = make(gui::NodeType::Resize, "Resize 16");
        size.parameters = {{"width", "16"}, {"height", "16"}, {"mode", "exact"}};
        auto flatten = make(gui::NodeType::Flatten, "Flatten");
        auto dense = make(gui::NodeType::Dense, "Dense 2");
        dense.parameters["units"] = "2";
        auto loss = make(gui::NodeType::CrossEntropyLoss, "Loss");
        auto adam = make(gui::NodeType::Adam, "Adam");
        Check(filter.inputs.size() == 1 && filter.inputs[0].type == gui::PinType::Dataset &&
                  filter.outputs.size() == 1 && filter.outputs[0].type == gui::PinType::Dataset,
              "the Quality Analyzer takes and passes on a Dataset");
        std::vector<gui::NodeLink> wires;
        int link_id = 1;
        const auto wire = [&](const gui::MLNode& from, size_t out, const gui::MLNode& to, size_t in) {
            wires.push_back(Link(link_id++, from.id, from.outputs.at(out).id, to.id, to.inputs.at(in).id));
        };
        wire(data, 0, filter, 0);
        wire(filter, 0, split, 0);
        wire(split, 0, loader, 0);
        wire(loader, 0, size, 0);
        wire(size, 0, flatten, 0);
        wire(flatten, 0, dense, 0);
        wire(dense, 0, loss, 0);
        wire(loader, 1, loss, 1);
        wire(loss, 0, adam, 0);
        const auto wired = cyxwiz::GraphCompiler{}.Compile(
            {data, filter, split, loader, size, flatten, dense, loss, adam}, wires, true);
        std::string origin_errors;
        for (const auto& issue : wired.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error && issue.message.find("originate") != std::string::npos)
                origin_errors += issue.message + "\n";
        }
        Check(origin_errors.empty(), "the Split sees the Data Input through the filter:\n" + origin_errors);
        Check(wired.image_excluded_files.size() == 2,
              "the canvas graph leaves out the same 2 files; errors:\n" + Errors(wired));
    }

    fs::remove(cyxwiz::ImageQualityCacheDir() / (first_key + ".json"), ec);
    cyxwiz::DataRegistry::Instance().UnregisterImageDataset("quality_graph");
}

// Registers an image folder and its index, as loading it in the Engine does.
void RegisterIndexed(const std::string& name, const fs::path& root, size_t images,
                     const std::vector<std::string>& classes) {
    cyxwiz::DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = root.string();
    entry.layout = 0;
    entry.num_images = images;
    entry.num_classes = classes.size();
    entry.class_names = classes;
    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.RegisterImageDataset(name, entry);
    const auto index = cyxwiz::BuildImageIndex(cyxwiz::ListImageSet(entry), {});
    Check(index && static_cast<size_t>(index->num_rows()) == images, name + " is indexed");
    registry.RegisterArrowTable(index, cyxwiz::DataRegistry::ImageIndexName(name));
}

bool Says(const cyxwiz::TrainingConfiguration& config, cyxwiz::IssueLevel level, const std::string& needle) {
    for (const auto& issue : config.issues)
        if (issue.level == level && issue.message.find(needle) != std::string::npos) return true;
    return false;
}

void CheckImageSplit() {
    // 10 images of 'a', 3 of 'b' (one unreadable), 2 of 'b' in a holdout, 2 of 'c' in another.
    const fs::path root = fs::temp_directory_path() / "cyxwiz_image_split";
    fs::remove_all(root);
    for (int i = 0; i < 10; ++i)
        WriteBmp(root / "train" / "a" / (std::to_string(i) + ".bmp"), 4, [i](int r, int c) { return Texture(i, r, c); });
    for (int i = 0; i < 2; ++i)
        WriteBmp(root / "train" / "b" / (std::to_string(i) + ".bmp"), 4, [i](int r, int c) { return Texture(20 + i, r, c); });
    {
        std::ofstream broken(root / "train" / "b" / "broken.bmp", std::ios::binary);
        broken << "not an image";
    }
    for (const char* cls : {"a", "b"})
        WriteBmp(root / "holdout" / cls / "0.bmp", 4, [](int r, int c) { return Texture(30, r, c); });
    for (const char* cls : {"a", "c"})
        WriteBmp(root / "other" / cls / "0.bmp", 4, [](int r, int c) { return Texture(40, r, c); });
    RegisterIndexed("split_train", root / "train", 13, {"a", "b"});
    RegisterIndexed("split_holdout", root / "holdout", 2, {"a", "b"});
    RegisterIndexed("split_other", root / "other", 2, {"a", "c"});

    // The counts are the batcher's parts, class by class.
    const cyxwiz::SampleSplitSettings split{0.6f, 0.2f, 0.2f, true, 5};
    const auto counts = cyxwiz::CountImageSplit("split_train", split, {});
    Check(counts.known && counts.images == 12 && counts.unreadable == 1 && counts.left_out == 0,
          "12 readable images are counted, the broken one is not (" + counts.reason + ")");
    Check(counts.parts[0] + counts.parts[1] + counts.parts[2] == 12 && counts.per_class.size() == 2,
          "the parts hold every readable image");
    const auto* entry = cyxwiz::DataRegistry::Instance().GetImageDatasetEntry("split_train");
    cyxwiz::ImagePreprocessingConfig preprocessing;
    preprocessing.resize_mode = cyxwiz::ResizeMode::Exact;
    preprocessing.target_width = 4;
    preprocessing.target_height = 4;
    const auto images = cyxwiz::PrepareImageSet(*entry, cyxwiz::PreparedImageSpecFrom(preprocessing)).set;
    Check(images != nullptr, "the split images are prepared");
    cyxwiz::ImageDatasetBatcher batcher(images, 4, split, true, 3);
    const cyxwiz::BatcherPhase phases[3] = {cyxwiz::BatcherPhase::Train, cyxwiz::BatcherPhase::Val,
                                           cyxwiz::BatcherPhase::Test};
    for (size_t part = 0; part < 3; ++part) {
        size_t of_a = 0;
        for (const size_t image : batcher.Part(phases[part]))
            of_a += fs::path(batcher.Files().at(image)).parent_path().filename() == "a" ? 1 : 0;
        Check(batcher.SamplesIn(phases[part]) == counts.parts[part] && of_a == counts.per_class[0][part],
              "part " + std::to_string(part) + " counts what the batcher trains on");
    }
    // Images a Quality node leaves out are taken out before the split.
    const auto fewer = cyxwiz::CountImageSplit("split_train", split, {batcher.Files().front()});
    Check(fewer.known && fewer.images == 11 && fewer.left_out == 1, "a left-out image is not split");

    // The compile says the parts; class 'b' (2 images) is too small for 10 %.
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "split_train";
    config.train_ratio = 0.8f;
    config.val_ratio = 0.1f;
    config.test_ratio = 0.1f;
    config.stratified = true;
    config.split_seed = 42;
    cyxwiz::CompileImageSplit(config, {});
    const auto& parts = config.image_split.parts;
    Check(config.image_split.known &&
              Says(config, cyxwiz::IssueLevel::Info,
                   "Images: " + std::to_string(parts[0]) + " train \xC2\xB7 " + std::to_string(parts[1]) +
                       " validation \xC2\xB7 " + std::to_string(parts[2]) + " test, by class (seed 42)."),
          "the compile says the image parts");
    Check(config.dataset_roles.manifest.train_rows == static_cast<int64_t>(parts[0]) &&
              config.dataset_roles.manifest.test_rows == static_cast<int64_t>(parts[2]),
          "the run manifest gets the image parts");
    Check(Says(config, cyxwiz::IssueLevel::Warning, "Class 'b' has 2 images: its validation part is empty"),
          "a class too small for the validation share is warned about");
    Check(!Says(config, cyxwiz::IssueLevel::Warning, "Class 'a'"), "a class large enough is not warned about");

    // A wired Validation dataset is that part whole; other classes are refused.
    cyxwiz::TrainingConfiguration wired = config;
    wired.issues.clear();
    wired.dataset_roles.dev.dataset_name = "split_holdout";
    wired.val_ratio = 0.0f;
    wired.train_ratio = 0.9f;
    cyxwiz::CompileImageSplit(wired, {});
    Check(wired.image_split_dev.known && wired.image_split_dev.parts[2] == 2 &&
              wired.dataset_roles.manifest.dev_rows == 2 &&
              !Says(wired, cyxwiz::IssueLevel::Error, "classes"),
          "a wired Validation dataset with the training classes is read whole (known " +
              std::to_string(wired.image_split_dev.known) + ", " + std::to_string(wired.image_split_dev.parts[2]) +
              " images, manifest " + std::to_string(wired.dataset_roles.manifest.dev_rows) + "); errors:\n" +
              Errors(wired));
    // Training leaves out what the compile counted as unreadable before the
    // split, so its parts are the counted ones.
    Check(config.image_unreadable_files.size() == 1 &&
              fs::path(config.image_unreadable_files[0]).filename() == "broken.bmp",
          "the compile hands training the unreadable files");

    // A file whose header reads but whose pixels are cut off: the index counts
    // it, decoding fails; training drops it from its own part and every
    // other image stays where the compile counted it.
    {
        const fs::path cut = root / "cut";
        for (int i = 0; i < 10; ++i)
            WriteBmp(cut / "a" / (std::to_string(i) + ".bmp"), 4, [i](int r, int c) { return Texture(i, r, c); });
        WriteBmp(cut / "a" / "truncated.bmp", 4, [](int r, int c) { return Texture(7, r, c); });
        fs::resize_file(cut / "a" / "truncated.bmp", 60);  // the headers (54 bytes) and a few pixels
        RegisterIndexed("split_cut", cut, 11, {"a"});
        const cyxwiz::SampleSplitSettings cut_split{0.6f, 0.2f, 0.2f, true, 9};
        const auto counted = cyxwiz::CountImageSplit("split_cut", cut_split, {});
        const auto* cut_entry = cyxwiz::DataRegistry::Instance().GetImageDatasetEntry("split_cut");
        const auto cut_images = cyxwiz::PrepareImageSet(*cut_entry, cyxwiz::PreparedImageSpecFrom(preprocessing)).set;
        if (counted.known && counted.unreadable == 0 && cut_images && cut_images->Unreadable().size() == 1) {
            cyxwiz::ImageDatasetBatcher cut_batcher(cut_images, 4, cut_split, true, 3);
            cut_batcher.ExcludeFiles(counted.unreadable_files);
            size_t smaller = 0, same = 0;
            for (size_t part = 0; part < 3; ++part) {
                const size_t n = cut_batcher.SamplesIn(phases[part]);
                smaller += n + 1 == counted.parts[part] ? 1 : 0;
                same += n == counted.parts[part] ? 1 : 0;
            }
            Check(smaller == 1 && same == 2, "an image that does not decode leaves only its own part");
        } else {
            std::cout << "note: this OpenCV decodes or the index rejects the truncated BMP; case skipped\n";
        }
    }

    cyxwiz::TrainingConfiguration other = config;
    other.issues.clear();
    other.dataset_roles.test.dataset_name = "split_other";
    cyxwiz::CompileImageSplit(other, {});
    Check(Says(other, cyxwiz::IssueLevel::Error,
               "The Test Dataset's classes (a, c) are not the training classes (a, b)"),
          "a wired Test dataset with other classes is refused");
    fs::remove_all(root);
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
    CheckQualityAnalyzer();
    CheckImageSplit();
    std::cout << "image transform graph contract passed (" << checks << " checks)\n";
    return 0;
}
