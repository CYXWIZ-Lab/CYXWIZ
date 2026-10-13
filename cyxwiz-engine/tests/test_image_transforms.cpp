// Image transform nodes (TOFIX140 image transforms) against torchvision.
//
// Each fixture case (fixtures/image_transforms_torchvision.json from
// generate_image_transform_fixtures.py) applies one transform with fixed
// per-sample settings through torchvision.transforms.functional; the backend
// applies the same op with the same settings on the ArrayFire device and must
// match on every ArrayFire backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND). Then: the random draws (ranges, train-only, reproducible per seed),
// the shape rule and its refusals, and a whole plan with Normalize.
#include "computation_truth/test_device_selection.h"

#include <cyxwiz/image_augmentation.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using cyxwiz::image::ImageOp;
using cyxwiz::image::ImageOpDraws;
using cyxwiz::image::ImageOpKind;
using cyxwiz::image::ImageShape;

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

ImageShape ShapeOf(const json& rows) {
    const auto shape = rows.at("shape").get<std::vector<size_t>>();  // [N, H, W, C]
    return {shape[1], shape[2], shape[3]};
}

cyxwiz::Tensor RowsOf(const json& rows) {
    const auto shape = rows.at("shape").get<std::vector<size_t>>();
    const auto values = rows.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor({shape[0], shape[1] * shape[2] * shape[3]}, values.data(),
                          cyxwiz::DataType::Float32);
}

ImageOp OpOf(const json& spec) {
    static const std::map<std::string, ImageOpKind> kinds = {
        {"center_crop", ImageOpKind::CenterCrop},   {"random_crop", ImageOpKind::RandomCrop},
        {"horizontal_flip", ImageOpKind::HorizontalFlip}, {"vertical_flip", ImageOpKind::VerticalFlip},
        {"rotate", ImageOpKind::Rotate},            {"color_jitter", ImageOpKind::ColorJitter},
        {"gaussian_blur", ImageOpKind::GaussianBlur}, {"grayscale", ImageOpKind::Grayscale},
        {"morphology", ImageOpKind::Morphology},  {"erase", ImageOpKind::Erase},
        {"randaugment", ImageOpKind::RandAugment},
    };
    static const std::map<std::string, cyxwiz::image::MorphologyOp> operations = {
        {"erode", cyxwiz::image::MorphologyOp::Erode},     {"dilate", cyxwiz::image::MorphologyOp::Dilate},
        {"open", cyxwiz::image::MorphologyOp::Open},       {"close", cyxwiz::image::MorphologyOp::Close},
        {"gradient", cyxwiz::image::MorphologyOp::Gradient}, {"tophat", cyxwiz::image::MorphologyOp::TopHat},
        {"blackhat", cyxwiz::image::MorphologyOp::BlackHat},
    };
    ImageOp op;
    op.kind = kinds.at(spec.at("kind").get<std::string>());
    op.height = spec.value("height", 0);
    op.width = spec.value("width", 0);
    op.padding = spec.value("padding", 0);
    op.value = spec.value("value", 0.0f);
    if (spec.contains("operation")) op.morphology = operations.at(spec.at("operation").get<std::string>());
    op.kernel_size = spec.value("kernel_size", 5);
    op.sigma = spec.value("sigma", 1.0f);
    if (spec.value("interpolation", std::string("nearest")) == "bilinear") {
        op.interpolation = cyxwiz::image::Interpolation::Bilinear;
    }
    return op;
}

ImageOpDraws DrawsOf(const json& spec) {
    ImageOpDraws draws;
    draws.top = spec.value("top", std::vector<int>{});
    draws.left = spec.value("left", std::vector<int>{});
    draws.box_height = spec.value("box_height", std::vector<int>{});
    draws.box_width = spec.value("box_width", std::vector<int>{});
    draws.apply = spec.value("apply", std::vector<int>{});
    draws.angle = spec.value("angle", std::vector<float>{});
    draws.factors = spec.value("factors", std::vector<std::vector<float>>{});
    draws.order = spec.value("order", std::vector<std::vector<int>>{});
    return draws;
}

void CheckRows(const cyxwiz::Tensor& actual, const json& expected, float tolerance, const std::string& what) {
    const auto shape = expected.at("shape").get<std::vector<size_t>>();
    const auto values = expected.at("values").get<std::vector<float>>();
    Check(actual.Shape() == std::vector<size_t>({shape[0], shape[1] * shape[2] * shape[3]}),
          what + ": output rows have the torchvision shape");
    const float* data = actual.ReadData<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        Check(std::fabs(data[i] - values[i]) <= tolerance,
              what + ": element " + std::to_string(i) + " is " + std::to_string(data[i]) + ", torchvision " +
                  std::to_string(values[i]));
    }
}

void CheckFixtures(const std::filesystem::path& path) {
    std::ifstream in(path);
    Check(in.good(), "fixture file " + path.string());
    const json fixture = json::parse(in);
    for (const auto& item : fixture.at("cases")) {
        const std::string name = item.at("name").get<std::string>();
        const ImageOp op = OpOf(item.at("op"));
        const ImageShape input = ShapeOf(item.at("input"));
        Check(cyxwiz::image::ImageShapeAfter(op, input) == ShapeOf(item.at("expected")),
              name + ": shape rule matches torchvision");
        const auto out = cyxwiz::image::ApplyImageOp(op, DrawsOf(item.at("draws")), input, RowsOf(item.at("input")));
        CheckRows(out, item.at("expected"), 2e-5f, name);
        std::cout << "  " << name << ": matches torchvision\n";
    }
}

cyxwiz::Tensor MatrixOf(const json& m) {
    const auto shape = m.at("shape").get<std::vector<size_t>>();
    const auto values = m.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckMixFixtures(const std::filesystem::path& path) {
    std::ifstream in(path);
    const json fixture = json::parse(in);
    for (const auto& item : fixture.at("mix_cases")) {
        const std::string name = item.at("name").get<std::string>();
        const auto method = item.at("op").at("method").get<std::string>() == "mixup"
            ? cyxwiz::image::BatchMix::MixUp : cyxwiz::image::BatchMix::CutMix;
        const json& d = item.at("draws");
        cyxwiz::image::BatchMixDraw draw;
        draw.apply = true;
        draw.lambda = d.at("lambda").get<float>();
        draw.top = d.value("top", 0);
        draw.left = d.value("left", 0);
        draw.height = d.value("height", 0);
        draw.width = d.value("width", 0);
        cyxwiz::Tensor rows = RowsOf(item.at("input"));
        cyxwiz::Tensor labels = MatrixOf(item.at("labels"));
        cyxwiz::image::ApplyBatchMix(method, draw, ShapeOf(item.at("input")), rows, labels);
        CheckRows(rows, item.at("expected"), 2e-6f, name + " images");
        const auto expected = item.at("expected_labels").at("values").get<std::vector<float>>();
        const float* got = labels.ReadData<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            Check(std::fabs(got[i] - expected[i]) < 1e-6f, name + ": label " + std::to_string(i) + " is " +
                                                               std::to_string(got[i]) + ", torchvision " +
                                                               std::to_string(expected[i]));
        }
        std::cout << "  " << name << ": images and labels match torchvision" << std::endl;
    }
}

void CheckRandAugmentTable(const std::filesystem::path& path) {
    std::ifstream in(path);
    const json fixture = json::parse(in);
    for (const auto& [size, table] : fixture.at("randaugment_magnitudes").items()) {
        const size_t x = size.find('x');
        const size_t h = std::stoul(size.substr(0, x)), w = std::stoul(size.substr(x + 1));
        for (int op = 0; op < cyxwiz::image::kRandAugmentOps; ++op) {
            for (int bin = 0; bin < cyxwiz::image::kRandAugmentBins; ++bin) {
                const float expected = table.at(op).at(bin).get<float>();
                const float got = cyxwiz::image::RandAugmentMagnitude(
                    static_cast<cyxwiz::image::RandAugmentOp>(op), bin, h, w);
                Check(got == expected, "RandAugment magnitude op " + std::to_string(op) + " bin " +
                                           std::to_string(bin) + " at " + size + " is " + std::to_string(got) +
                                           ", torchvision " + std::to_string(expected));
            }
        }
    }
}

void CheckRandAugmentDraws() {
    const ImageShape shape{32, 40, 3};
    std::mt19937 rng(21);
    ImageOp op;
    op.kind = ImageOpKind::RandAugment;
    op.num_ops = 2;
    op.magnitude = 9;
    op.probability = 1.0f;
    const auto draws = cyxwiz::image::DrawImageOp(op, shape, 7000, true, rng);
    std::vector<int> counts(cyxwiz::image::kRandAugmentOps, 0);
    int negative = 0, signed_picks = 0;
    for (size_t i = 0; i < 7000; ++i) {
        Check(draws.order[i].size() == 2 && draws.factors[i].size() == 2, "two picks per image");
        for (size_t p = 0; p < 2; ++p) {
            const auto chosen = static_cast<cyxwiz::image::RandAugmentOp>(draws.order[i][p]);
            ++counts[static_cast<size_t>(draws.order[i][p])];
            const float base = cyxwiz::image::RandAugmentMagnitude(chosen, 9, 32, 40);
            Check(std::fabs(draws.factors[i][p]) == base, "a pick uses the op's magnitude at bin 9");
            if (cyxwiz::image::RandAugmentSigned(chosen) && base != 0.0f) {
                ++signed_picks;
                negative += draws.factors[i][p] < 0.0f ? 1 : 0;
            }
        }
    }
    for (int c : counts) Check(c > 800 && c < 1200, "each of the 14 ops is picked about equally (" + std::to_string(c) + ")");
    Check(std::fabs(static_cast<double>(negative) / signed_picks - 0.5) < 0.03, "signed ops flip sign half the time");
    Check(cyxwiz::image::DrawImageOp(op, shape, 3, false, rng).order.empty(), "RandAugment passes through outside training");
}

void CheckMixDraws() {
    const ImageShape shape{32, 40, 3};
    std::mt19937 rng(5);
    double sum = 0.0;
    for (int i = 0; i < 4000; ++i) {
        const auto draw = cyxwiz::image::DrawBatchMix(cyxwiz::image::BatchMix::MixUp, 1.0f, 1.0f, shape, rng);
        Check(draw.apply && draw.lambda >= 0.0f && draw.lambda <= 1.0f, "MixUp lambda is in [0, 1]");
        sum += draw.lambda;
    }
    Check(std::fabs(sum / 4000.0 - 0.5) < 0.03, "MixUp lambda from Beta(1, 1) averages 0.5");
    for (int i = 0; i < 500; ++i) {
        const auto draw = cyxwiz::image::DrawBatchMix(cyxwiz::image::BatchMix::CutMix, 1.0f, 1.0f, shape, rng);
        Check(draw.top + draw.height <= 32 && draw.left + draw.width <= 40, "CutMix boxes fit the image");
        Check(std::fabs(draw.lambda - (1.0f - static_cast<float>(draw.height * draw.width) / (32.0f * 40.0f))) < 1e-6f,
              "CutMix lambda is one minus the pasted area");
    }
    int applied = 0;
    for (int i = 0; i < 2000; ++i) {
        applied += cyxwiz::image::DrawBatchMix(cyxwiz::image::BatchMix::MixUp, 1.0f, 0.3f, shape, rng).apply ? 1 : 0;
    }
    Check(applied > 520 && applied < 680, "a batch is mixed with mix_probability (" + std::to_string(applied) + ")");

    // In a plan: training mixes images and labels, validation leaves both.
    cyxwiz::image::ImageAugmentation plan;
    plan.mix = cyxwiz::image::BatchMix::MixUp;
    const ImageShape tiny{2, 2, 1};
    const float pixels[] = {0, 0, 0, 0, 1, 1, 1, 1};
    const float onehot[] = {1, 0, 0, 1};
    const cyxwiz::Tensor rows({2, 4}, pixels, cyxwiz::DataType::Float32);
    cyxwiz::Tensor val_labels({2, 2}, onehot, cyxwiz::DataType::Float32);
    std::mt19937 plan_rng(9);
    const auto val = plan.Apply(rows, tiny, false, plan_rng, &val_labels);
    Check(val.ReadData<float>()[0] == 0.0f && val_labels.ReadData<float>()[0] == 1.0f,
          "validation batches are not mixed");
    cyxwiz::Tensor train_labels({2, 2}, onehot, cyxwiz::DataType::Float32);
    const auto train = plan.Apply(rows, tiny, true, plan_rng, &train_labels);
    const float lambda = train_labels.ReadData<float>()[0];
    Check(lambda > 0.0f && lambda < 1.0f && std::fabs(train.ReadData<float>()[0] - (1.0f - lambda)) < 1e-6f,
          "training mixes image and label with the same lambda");
}

void CheckDraws() {
    const ImageShape shape{6, 7, 3};
    std::mt19937 rng(140);
    ImageOp crop;
    crop.kind = ImageOpKind::RandomCrop;
    crop.height = 4;
    crop.width = 5;
    const auto train = cyxwiz::image::DrawImageOp(crop, shape, 200, true, rng);
    bool varied = false;
    for (size_t i = 0; i < 200; ++i) {
        Check(train.top[i] >= 0 && train.top[i] <= 2 && train.left[i] >= 0 && train.left[i] <= 2,
              "Random Crop positions stay inside the image");
        varied = varied || train.top[i] != train.top[0] || train.left[i] != train.left[0];
    }
    Check(varied, "Random Crop positions vary across samples");
    const auto eval = cyxwiz::image::DrawImageOp(crop, shape, 3, false, rng);
    Check(eval.top == std::vector<int>({1, 1, 1}) && eval.left == std::vector<int>({1, 1, 1}),
          "Random Crop centres outside training (torchvision center_crop offsets)");

    ImageOp padded = crop;
    padded.height = 6;
    padded.width = 7;
    padded.padding = 2;
    const auto padded_train = cyxwiz::image::DrawImageOp(padded, shape, 200, true, rng);
    for (size_t i = 0; i < 200; ++i) {
        Check(padded_train.top[i] >= 0 && padded_train.top[i] <= 4 && padded_train.left[i] >= 0 &&
                  padded_train.left[i] <= 4,
              "padded Random Crop positions stay inside the padded image");
    }
    const auto padded_eval = cyxwiz::image::DrawImageOp(padded, shape, 1, false, rng);
    Check(padded_eval.top == std::vector<int>({2}) && padded_eval.left == std::vector<int>({2}),
          "padded Random Crop of the full size is the unchanged image outside training");

    ImageOp flip;
    flip.kind = ImageOpKind::HorizontalFlip;
    flip.probability = 0.25f;
    const auto flips = cyxwiz::image::DrawImageOp(flip, shape, 4000, true, rng);
    int flipped = 0;
    for (int apply : flips.apply) flipped += apply;
    Check(flipped > 850 && flipped < 1150, "flips follow the probability (" + std::to_string(flipped) + "/4000)");
    Check(cyxwiz::image::DrawImageOp(flip, shape, 4, false, rng).apply.empty(), "flips pass through outside training");

    ImageOp rotate;
    rotate.kind = ImageOpKind::Rotate;
    rotate.max_angle = 20.0f;
    rotate.probability = 1.0f;
    for (float angle : cyxwiz::image::DrawImageOp(rotate, shape, 500, true, rng).angle) {
        Check(angle >= -20.0f && angle <= 20.0f, "rotation angles stay within max_angle");
    }

    ImageOp jitter;
    jitter.kind = ImageOpKind::ColorJitter;
    jitter.brightness = 0.4f;
    jitter.hue = 0.1f;
    const auto jitters = cyxwiz::image::DrawImageOp(jitter, shape, 300, true, rng);
    for (size_t i = 0; i < 300; ++i) {
        Check(jitters.order[i].size() == 2, "jitter applies only its active adjustments");
        Check(jitters.factors[i][0] >= 0.6f && jitters.factors[i][0] <= 1.4f, "brightness factor in [1-b, 1+b]");
        Check(jitters.factors[i][1] == 1.0f && jitters.factors[i][2] == 1.0f, "inactive factors stay neutral");
        Check(jitters.factors[i][3] >= -0.1f && jitters.factors[i][3] <= 0.1f, "hue shift in [-hue, hue]");
    }

    const ImageShape big{40, 50, 3};
    ImageOp erasing;
    erasing.kind = ImageOpKind::Erase;
    erasing.erase_method = cyxwiz::image::EraseMethod::RandomErasing;
    erasing.probability = 1.0f;
    const auto erased = cyxwiz::image::DrawImageOp(erasing, big, 300, true, rng);
    for (size_t i = 0; i < 300; ++i) {
        const int h = erased.box_height[i], w = erased.box_width[i];
        Check(h > 0 && w > 0 && h < 40 && w < 50, "Random Erasing boxes are inside and smaller than the image");
        Check(erased.top[i] + h <= 40 && erased.left[i] + w <= 50, "Random Erasing boxes fit");
        const double area = static_cast<double>(h) * w / (40.0 * 50.0);
        Check(area > 0.01 && area < 0.36, "Random Erasing area follows scale (rounded): " + std::to_string(area));
    }
    ImageOp cutout = erasing;
    cutout.erase_method = cyxwiz::image::EraseMethod::Cutout;
    cutout.cutout_size = 16;
    cutout.probability = 0.5f;
    const auto cut = cyxwiz::image::DrawImageOp(cutout, big, 2000, true, rng);
    int applied = 0;
    for (size_t i = 0; i < 2000; ++i) {
        Check(cut.box_height[i] <= 16 && cut.box_width[i] <= 16 && cut.top[i] + cut.box_height[i] <= 40 &&
                  cut.left[i] + cut.box_width[i] <= 50,
              "Cutout squares are at most cutout_size and clipped to the image");
        applied += cut.box_height[i] > 0 ? 1 : 0;
    }
    Check(applied > 900 && applied < 1100, "Cutout follows its probability (" + std::to_string(applied) + ")");
    Check(cyxwiz::image::DrawImageOp(cutout, big, 3, false, rng).box_height.empty(),
          "erasing passes through outside training");

    std::mt19937 first(7), second(7);
    Check(cyxwiz::image::DrawImageOp(jitter, shape, 50, true, first).factors ==
              cyxwiz::image::DrawImageOp(jitter, shape, 50, true, second).factors,
          "draws are reproducible for one seed");
}

void CheckRefusals() {
    const ImageShape shape{6, 7, 3};
    const auto refused = [&](ImageOp op, const std::string& needle, const std::string& what) {
        const std::string reason = cyxwiz::image::ValidateImageOp(op, shape);
        Check(reason.find(needle) != std::string::npos, what + " is refused (got '" + reason + "')");
    };
    ImageOp crop;
    crop.kind = ImageOpKind::CenterCrop;
    crop.height = 7;
    crop.width = 5;
    refused(crop, "larger than the 6 x 7 image", "a crop taller than the image");
    crop.height = 0;
    refused(crop, "positive width and height", "a zero crop");
    crop.kind = ImageOpKind::RandomCrop;
    crop.height = 11;
    crop.width = 7;
    crop.padding = 2;
    refused(crop, "larger than the 10 x 11 padded image", "a crop larger than the padded image");
    ImageOp erase;
    erase.kind = ImageOpKind::Erase;
    erase.erase_method = cyxwiz::image::EraseMethod::RandomErasing;
    erase.scale_min = 0.5f;
    erase.scale_max = 0.2f;
    refused(erase, "scale_min <= scale_max", "a reversed erasing scale");
    erase.scale_max = 0.6f;
    erase.value = 2.0f;
    refused(erase, "pixel value between 0 and 1", "an erasing value outside [0, 1]");
    ImageOp randaugment;
    randaugment.kind = ImageOpKind::RandAugment;
    randaugment.magnitude = 31;
    refused(randaugment, "magnitude must be between 0 and 30", "a magnitude past the 31 bins");
    ImageOp morphology;
    morphology.kind = ImageOpKind::Morphology;
    morphology.kernel_size = 2;
    refused(morphology, "positive odd", "an even morphology kernel");
    ImageOp blur;
    blur.kind = ImageOpKind::GaussianBlur;
    blur.kernel_size = 4;
    refused(blur, "positive odd", "an even blur kernel");
    blur.kernel_size = 13;
    refused(blur, "too large", "a blur kernel wider than the reflect padding allows");
    blur.kernel_size = 3;
    blur.sigma = 0.0f;
    refused(blur, "sigma", "sigma 0");
    ImageOp jitter;
    jitter.kind = ImageOpKind::ColorJitter;
    jitter.hue = 0.6f;
    refused(jitter, "hue", "hue above 0.5");
    ImageOp flip;
    flip.kind = ImageOpKind::VerticalFlip;
    flip.probability = 1.5f;
    refused(flip, "probability", "a probability above 1");
}

void CheckPlan() {
    // Random Crop 4x5, Grayscale, then Normalize: on the device in one pass.
    cyxwiz::image::ImageAugmentation plan;
    ImageOp crop;
    crop.kind = ImageOpKind::RandomCrop;
    crop.height = 4;
    crop.width = 5;
    ImageOp gray;
    gray.kind = ImageOpKind::Grayscale;
    plan.ops = {crop, gray};
    plan.normalize = true;
    plan.mean = 0.5f;
    plan.std_dev = 0.25f;
    const ImageShape input{6, 7, 3};
    Check(plan.ShapeAfter(input) == ImageShape{4, 5, 1}, "the plan's shape follows its ops");

    std::vector<float> pixels(2 * 6 * 7 * 3);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<float>((i * 37) % 101) / 100.0f;
    const cyxwiz::Tensor rows({2, 6 * 7 * 3}, pixels.data(), cyxwiz::DataType::Float32);
    std::mt19937 rng(3);
    const auto validation = plan.Apply(rows, input, false, rng);
    Check(validation.Shape() == std::vector<size_t>({2, 20}), "validation rows have the planned size");
    const float* out = validation.ReadData<float>();
    // Validation centres the crop at (1, 1); pixel (0, 0) of sample 0 is source (1, 1).
    const float* source = pixels.data() + (1 * 7 + 1) * 3;
    const float expected = ((0.2989f * source[0] + 0.587f * source[1] + 0.114f * source[2]) - 0.5f) / 0.25f;
    Check(std::fabs(out[0] - expected) < 1e-5f, "validation: centre crop, grayscale, then Normalize");

    std::mt19937 again(3);
    const auto repeat = plan.Apply(rows, input, false, again);
    Check(std::equal(out, out + 40, repeat.ReadData<float>()), "the plan is deterministic outside training");
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "image_transforms_torchvision.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_IMAGE_TRANSFORMS_FIXTURE);
}

}  // namespace

int main(int, char** argv) {
    // CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu: the same ArrayFire code on each backend.
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested device");
    CheckFixtures(FixturePath(argv[0]));
    CheckMixFixtures(FixturePath(argv[0]));
    CheckDraws();
    CheckMixDraws();
    CheckRandAugmentTable(FixturePath(argv[0]));
    CheckRandAugmentDraws();
    CheckRefusals();
    CheckPlan();
    std::cout << "image transforms match torchvision (" << checks << " checks)\n";
    return 0;
}
