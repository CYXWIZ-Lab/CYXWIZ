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
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
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
        {"randaugment", ImageOpKind::RandAugment},  {"autoaugment", ImageOpKind::AutoAugment},
        {"trivialaugment", ImageOpKind::TrivialAugment}, {"augmix", ImageOpKind::AugMix},
        {"random_resized_crop", ImageOpKind::RandomResizedCrop}, {"random_affine", ImageOpKind::RandomAffine},
        {"random_perspective", ImageOpKind::RandomPerspective}, {"gaussian_noise", ImageOpKind::GaussianNoise},
        {"random_grayscale", ImageOpKind::RandomGrayscale},
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
    op.fill = spec.value("fill", 0.0f);
    op.noise_mean = spec.value("noise_mean", 0.0f);
    op.clip = spec.value("clip", true);
    op.mixture_width = spec.value("mixture_width", 3);
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
    draws.corners = spec.value("corners", std::vector<std::vector<int>>{});
    draws.weights = spec.value("weights", std::vector<std::vector<float>>{});
    draws.noise = spec.value("noise", std::vector<float>{});
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

void CheckAugmentTables(const std::filesystem::path& path) {
    using cyxwiz::image::AugmentOp;
    using cyxwiz::image::AugmentSpace;
    static const std::map<std::string, AugmentOp> ops = {
        {"Identity", AugmentOp::Identity},     {"ShearX", AugmentOp::ShearX},     {"ShearY", AugmentOp::ShearY},
        {"TranslateX", AugmentOp::TranslateX}, {"TranslateY", AugmentOp::TranslateY}, {"Rotate", AugmentOp::Rotate},
        {"Brightness", AugmentOp::Brightness}, {"Color", AugmentOp::Color},       {"Contrast", AugmentOp::Contrast},
        {"Sharpness", AugmentOp::Sharpness},   {"Posterize", AugmentOp::Posterize}, {"Solarize", AugmentOp::Solarize},
        {"AutoContrast", AugmentOp::AutoContrast}, {"Equalize", AugmentOp::Equalize}, {"Invert", AugmentOp::Invert},
    };
    static const std::map<std::string, AugmentSpace> spaces = {
        {"standard", AugmentSpace::Standard}, {"wide", AugmentSpace::Wide}, {"augmix", AugmentSpace::AugMix}};
    std::ifstream in(path);
    const json fixture = json::parse(in);
    for (const auto& table : fixture.at("augment_magnitudes")) {
        const auto space = spaces.at(table.at("space").get<std::string>());
        const int bins = table.at("bins").get<int>();
        const size_t h = table.at("height").get<size_t>(), w = table.at("width").get<size_t>();
        const std::string where = table.at("space").get<std::string>() + " " + std::to_string(bins) + " bins at " +
                                  std::to_string(h) + "x" + std::to_string(w);
        for (const auto& [name, entry] : table.at("ops").items()) {
            const AugmentOp op = ops.at(name);
            Check(cyxwiz::image::AugmentSigned(op) == entry.at("signed").get<bool>(), name + " signed, " + where);
            const auto values = entry.at("values").get<std::vector<float>>();
            for (int bin = 0; bin < bins; ++bin) {
                const float got = cyxwiz::image::AugmentMagnitude(space, op, bins, bin, h, w);
                Check(got == values[static_cast<size_t>(bin)], name + " magnitude bin " + std::to_string(bin) + ", " +
                                                                   where + ": " + std::to_string(got) + ", torchvision " +
                                                                   std::to_string(values[static_cast<size_t>(bin)]));
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
    std::vector<int> counts(cyxwiz::image::kAugmentOps, 0);
    int negative = 0, signed_picks = 0;
    for (size_t i = 0; i < 7000; ++i) {
        Check(draws.order[i].size() == 2 && draws.factors[i].size() == 2, "two picks per image");
        for (size_t p = 0; p < 2; ++p) {
            const auto chosen = static_cast<cyxwiz::image::AugmentOp>(draws.order[i][p]);
            ++counts[static_cast<size_t>(draws.order[i][p])];
            const float base = cyxwiz::image::AugmentMagnitude(cyxwiz::image::AugmentSpace::Standard, chosen, 31, 9, 32, 40);
            Check(std::fabs(draws.factors[i][p]) == base, "a pick uses the op's magnitude at bin 9");
            if (cyxwiz::image::AugmentSigned(chosen) && base != 0.0f) {
                ++signed_picks;
                negative += draws.factors[i][p] < 0.0f ? 1 : 0;
            }
        }
    }
    for (size_t id = 0; id < counts.size(); ++id) {
        const int c = counts[id];
        Check(id < 14 ? c > 800 && c < 1200 : c == 0, "each of RandAugment's 14 ops is picked about equally (" +
                                                          std::to_string(c) + ")");
    }
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

// The 1.6b steps' draws: the ranges torchvision draws from, train-only, and
// the policies' picks.
void CheckNewDraws() {
    const ImageShape shape{32, 40, 3};
    std::mt19937 rng(178);

    ImageOp crop;
    crop.kind = ImageOpKind::RandomResizedCrop;
    crop.height = 24;
    crop.width = 24;
    crop.scale_min = 0.08f;
    crop.scale_max = 1.0f;
    crop.ratio_min = 0.75f;
    crop.ratio_max = 4.0f / 3.0f;
    Check(cyxwiz::image::ImageShapeAfter(crop, shape) == ImageShape{24, 24, 3}, "Random Resized Crop's output size");
    const auto boxes = cyxwiz::image::DrawImageOp(crop, shape, 500, true, rng);
    bool varied = false;
    for (size_t i = 0; i < 500; ++i) {
        const int h = boxes.box_height[i], w = boxes.box_width[i];
        Check(h > 0 && w > 0 && boxes.top[i] + h <= 32 && boxes.left[i] + w <= 40, "resized crop boxes fit the image");
        const double area = static_cast<double>(h) * w / (32.0 * 40.0);
        Check(area > 0.06 && area < 1.01, "resized crop area follows scale (rounded): " + std::to_string(area));
        varied = varied || h != boxes.box_height[0];
    }
    Check(varied, "resized crop boxes vary");
    const auto centred = cyxwiz::image::DrawImageOp(crop, shape, 2, false, rng);
    // 40 / 32 = 1.25 is inside 3:4..4:3: the whole image outside training.
    Check(centred.top[0] == 0 && centred.left[0] == 0 && centred.box_height[0] == 32 && centred.box_width[0] == 40,
          "Random Resized Crop takes the whole image outside training");
    crop.ratio_min = 1.5f;
    crop.ratio_max = 2.0f;
    const auto wide = cyxwiz::image::DrawImageOp(crop, shape, 1, false, rng);
    Check(wide.box_width[0] == 40 && wide.box_height[0] == 27 && wide.top[0] == 2,
          "outside training: the centred box of the nearest allowed aspect");
    crop.ratio_min = 100.0f;
    crop.ratio_max = 200.0f;
    const auto thin = cyxwiz::image::DrawImageOp(crop, shape, 1, false, rng);
    Check(thin.box_height[0] == 1 && thin.box_width[0] == 40, "a ratio past the image still keeps a one-pixel box");
    const std::vector<float> pixels(32 * 40 * 3, 0.5f);
    const cyxwiz::Tensor one({1, 32 * 40 * 3}, pixels.data(), cyxwiz::DataType::Float32);
    const auto refused_apply = [&](const ImageOp& op, const cyxwiz::image::ImageOpDraws& draws, const std::string& what) {
        bool threw = false;
        try {
            cyxwiz::image::ApplyImageOp(op, draws, shape, one);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        Check(threw, what + " is refused");
    };
    auto outside = thin;
    outside.top[0] = 5;
    outside.box_height[0] = 30;
    refused_apply(crop, outside, "a resized crop box past the image");
    auto empty_box = thin;
    empty_box.box_height[0] = 0;
    refused_apply(crop, empty_box, "an empty resized crop box");

    ImageOp affine;
    affine.kind = ImageOpKind::RandomAffine;
    affine.max_angle = 10.0f;
    affine.translate_x = 0.1f;
    affine.translate_y = 0.2f;
    affine.zoom_min = 0.9f;
    affine.zoom_max = 1.1f;
    affine.shear_x_min = -5.0f;
    affine.shear_x_max = 5.0f;
    for (const auto& f : cyxwiz::image::DrawImageOp(affine, shape, 400, true, rng).factors) {
        Check(std::fabs(f[0]) <= 10.0f && std::fabs(f[1]) <= 4.0f && std::fabs(f[2]) <= 6.0f,
              "affine angle and whole-pixel shifts in range");
        Check(f[1] == std::round(f[1]) && f[2] == std::round(f[2]), "affine shifts are whole pixels");
        Check(f[3] >= 0.9f && f[3] <= 1.1f && std::fabs(f[4]) <= 5.0f && f[5] == 0.0f, "affine scale and shears in range");
    }
    Check(cyxwiz::image::DrawImageOp(affine, shape, 3, false, rng).factors.empty(), "affine passes through outside training");

    ImageOp perspective;
    perspective.kind = ImageOpKind::RandomPerspective;
    perspective.distortion = 0.5f;
    perspective.probability = 0.3f;
    const auto corners = cyxwiz::image::DrawImageOp(perspective, shape, 3000, true, rng);
    int applied = 0;
    for (size_t i = 0; i < 3000; ++i) {
        if (!corners.apply[i]) continue;
        ++applied;
        const auto& c = corners.corners[i];
        // half 20 x 16, distortion 0.5: corners move in by at most 10 x 8 pixels.
        Check(c[0] <= 10 && c[1] <= 8 && c[2] >= 29 && c[3] <= 8 && c[4] >= 29 && c[5] >= 23 && c[6] <= 10 && c[7] >= 23,
              "perspective corners move inwards by at most distortion of half a side");
    }
    Check(applied > 820 && applied < 980, "perspective follows its probability (" + std::to_string(applied) + ")");
    cyxwiz::image::ImageOpDraws line;
    line.apply = {1};
    line.corners = {{0, 0, 10, 10, 20, 20, 30, 30}};
    refused_apply(perspective, line, "perspective corners on one line");
    // A 2 x 2 image at full distortion can draw corners with no mapping: those
    // samples stay as they are, and every applied one solves.
    perspective.distortion = 1.0f;
    perspective.probability = 1.0f;
    const ImageShape tiny{2, 2, 1};
    const auto tiny_draws = cyxwiz::image::DrawImageOp(perspective, tiny, 400, true, rng);
    int tiny_applied = 0;
    for (int a : tiny_draws.apply) tiny_applied += a;
    Check(tiny_applied > 0 && tiny_applied < 400, "tiny perspective skips only corners with no mapping (" +
                                                       std::to_string(tiny_applied) + ")");
    const std::vector<float> tiny_pixels(400 * 4, 0.5f);
    const cyxwiz::Tensor tiny_rows({400, 4}, tiny_pixels.data(), cyxwiz::DataType::Float32);
    cyxwiz::image::ApplyImageOp(perspective, tiny_draws, tiny, tiny_rows);  // throws if one did not solve

    ImageOp gray;
    gray.kind = ImageOpKind::RandomGrayscale;
    gray.probability = 0.1f;
    int grays = 0;
    for (int a : cyxwiz::image::DrawImageOp(gray, shape, 4000, true, rng).apply) grays += a;
    Check(grays > 330 && grays < 470, "random grayscale follows its probability (" + std::to_string(grays) + ")");

    // Noise drawn on the device: mean and spread of N(mean, sigma), clip off.
    ImageOp noise;
    noise.kind = ImageOpKind::GaussianNoise;
    noise.noise_mean = 0.05f;
    noise.sigma = 0.2f;
    noise.clip = false;
    const std::vector<float> half(64 * 32 * 40 * 3, 0.5f);
    const cyxwiz::Tensor rows({64, 32 * 40 * 3}, half.data(), cyxwiz::DataType::Float32);
    const auto noisy = cyxwiz::image::ApplyImageOp(noise, cyxwiz::image::DrawImageOp(noise, shape, 64, true, rng), shape, rows);
    const float* v = noisy.ReadData<float>();
    double sum = 0.0, squares = 0.0;
    for (size_t i = 0; i < half.size(); ++i) {
        sum += v[i] - 0.5;
        squares += (v[i] - 0.5) * (v[i] - 0.5);
    }
    const double mean = sum / half.size(), sd = std::sqrt(squares / half.size() - mean * mean);
    Check(std::fabs(mean - 0.05) < 0.003 && std::fabs(sd - 0.2) < 0.003,
          "device noise has the mean and sigma asked for (" + std::to_string(mean) + ", " + std::to_string(sd) + ")");
    const auto clean = cyxwiz::image::ApplyImageOp(noise, cyxwiz::image::DrawImageOp(noise, shape, 64, false, rng), shape, rows);
    Check(clean.ReadData<float>()[0] == 0.5f, "noise passes through outside training");
    cyxwiz::image::ImageOpDraws short_noise;
    short_noise.apply = {1};
    short_noise.noise.assign(10, 0.0f);
    refused_apply(noise, short_noise, "recorded noise shorter than the image");

    ImageOp autoaugment;
    autoaugment.kind = ImageOpKind::AutoAugment;
    autoaugment.probability = 1.0f;
    autoaugment.policy = cyxwiz::image::AutoAugmentPolicy::Svhn;
    int inverts = 0;
    const auto picks = cyxwiz::image::DrawImageOp(autoaugment, shape, 2000, true, rng);
    for (const auto& order : picks.order) {
        Check(order.size() == 2, "AutoAugment draws a sub-policy of two");
        for (int op : order) inverts += op == static_cast<int>(cyxwiz::image::AugmentOp::Invert) ? 1 : 0;
    }
    Check(inverts > 450, "the SVHN policy inverts often (" + std::to_string(inverts) + ")");

    ImageOp trivial;
    trivial.kind = ImageOpKind::TrivialAugment;
    trivial.probability = 1.0f;
    trivial.magnitude_bins = 31;
    std::vector<int> counts(cyxwiz::image::kAugmentOps, 0);
    for (const auto& order : cyxwiz::image::DrawImageOp(trivial, shape, 7000, true, rng).order) ++counts[static_cast<size_t>(order[0])];
    for (size_t op = 0; op < counts.size(); ++op) {
        Check(op < 14 ? counts[op] > 400 && counts[op] < 600 : counts[op] == 0,
              "TrivialAugment picks its 14 ops about equally (" + std::to_string(counts[op]) + ")");
    }

    ImageOp augmix;
    augmix.kind = ImageOpKind::AugMix;
    augmix.probability = 1.0f;
    augmix.mixture_width = 3;
    augmix.chain_depth = -1;
    augmix.severity = 3;
    const auto mixes = cyxwiz::image::DrawImageOp(augmix, shape, 500, true, rng);
    for (size_t i = 0; i < 500; ++i) {
        const auto& w = mixes.weights[i];
        Check(w.size() == 4 && std::fabs(w[0] + w[1] + w[2] + w[3] - 1.0f) < 1e-5f, "AugMix weights sum to one");
        Check(mixes.order[i].size() == 9, "AugMix: three chains of up to three ops");
        for (size_t k = 0; k < 9; ++k) {
            const int op = mixes.order[i][k];
            Check(op != static_cast<int>(cyxwiz::image::AugmentOp::Invert) && (k % 3 == 0 ? op != 0 : true),
                  "AugMix chains start with an op and never invert");
        }
    }
    augmix.probability = 0.0f;
    const auto plain = cyxwiz::image::DrawImageOp(augmix, shape, 2, true, rng);
    Check(plain.weights[0][0] == 1.0f && plain.weights[0][1] == 0.0f, "AugMix not applied keeps the image");
    Check(cyxwiz::image::DrawImageOp(augmix, shape, 2, false, rng).order.empty(), "AugMix passes through outside training");
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
    ImageOp resized;
    resized.kind = ImageOpKind::RandomResizedCrop;
    resized.height = 4;
    resized.width = 4;
    resized.scale_min = 0.5f;
    resized.scale_max = 1.5f;
    refused(resized, "scale_max <= 1", "a resized crop area above the image");
    ImageOp affine;
    affine.kind = ImageOpKind::RandomAffine;
    affine.shear_x_min = 5.0f;
    refused(affine, "min <= max", "a reversed shear range");
    affine.shear_x_min = 0.0f;
    affine.translate_x = 1.5f;
    refused(affine, "fraction of the image", "a translation past the image");
    ImageOp perspective;
    perspective.kind = ImageOpKind::RandomPerspective;
    perspective.distortion = 1.2f;
    refused(perspective, "distortion", "a distortion above 1");
    ImageOp noise;
    noise.kind = ImageOpKind::GaussianNoise;
    noise.sigma = -0.1f;
    refused(noise, "sigma", "a negative noise sigma");
    noise.sigma = 0.1f;
    noise.noise_mean = std::numeric_limits<float>::quiet_NaN();
    refused(noise, "mean must be a number", "a noise mean that is not a number");
    ImageOp augmix;
    augmix.kind = ImageOpKind::AugMix;
    augmix.chain_depth = 0;
    refused(augmix, "chain_depth", "a chain depth of 0");
    ImageOp trivial;
    trivial.kind = ImageOpKind::TrivialAugment;
    trivial.magnitude_bins = 1;
    refused(trivial, "magnitude_bins", "one magnitude bin");
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
    CheckAugmentTables(FixturePath(argv[0]));
    CheckRandAugmentDraws();
    CheckNewDraws();
    CheckRefusals();
    CheckPlan();
    std::cout << "image transforms match torchvision (" << checks << " checks)\n";
    return 0;
}
