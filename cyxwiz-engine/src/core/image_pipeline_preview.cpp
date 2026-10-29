#include "image_pipeline_preview.h"

#include <cyxwiz/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace cyxwiz {

namespace {

const char* kTimes = " \xC3\x97 ";

std::string Number(float v) {
    char out[32];
    std::snprintf(out, sizeof(out), "%g", v);
    return out;
}

const char* ModeName(ResizeMode mode) {
    switch (mode) {
        case ResizeMode::AspectFit: return "fit";
        case ResizeMode::AspectFill: return "fill";
        case ResizeMode::Center: return "center";
        default: return "exact";
    }
}

// Rows [1, shape.Size()] in 0..1 -> RGBA bytes.
PreviewPicture Picture(const Tensor& rows, const image::ImageShape& shape) {
    PreviewPicture p;
    p.width = static_cast<int>(shape.width);
    p.height = static_cast<int>(shape.height);
    const float* values = rows.ReadData<float>();
    const size_t pixels = shape.height * shape.width;
    p.rgba.assign(pixels * 4, 255);
    for (size_t i = 0; i < pixels; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            const float v = values[i * shape.channels + (shape.channels >= 3 ? c : 0)];
            p.rgba[i * 4 + c] = static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
        }
    }
    return p;
}

// What one draw did, in words.
std::string DrawText(const image::ImageOp& op, const image::ImageOpDraws& d) {
    using K = image::ImageOpKind;
    switch (op.kind) {
        case K::HorizontalFlip: return !d.apply.empty() && d.apply[0] ? "flipped" : "not flipped";
        case K::VerticalFlip: return !d.apply.empty() && d.apply[0] ? "flipped upside down" : "not flipped upside down";
        case K::Rotate:
            return d.angle.empty() || d.angle[0] == 0.0f ? "not rotated" : "rotated " + Number(std::round(d.angle[0] * 10.0f) / 10.0f) + " degrees";
        case K::RandomCrop:
            return d.top.empty() ? "cropped" : "cropped at " + std::to_string(d.top[0]) + ", " + std::to_string(d.left[0]);
        case K::ColorJitter: {
            if (d.factors.empty() || d.factors[0].size() < 4) return "colours changed";
            const auto& f = d.factors[0];
            return "brightness " + Number(std::round(f[0] * 100.0f) / 100.0f) + ", contrast " +
                   Number(std::round(f[1] * 100.0f) / 100.0f);
        }
        case K::Erase:
            return d.box_height.empty() || d.box_height[0] == 0 ? "nothing erased"
                                                                 : "erased " + std::to_string(d.box_height[0]) + kTimes +
                                                                       std::to_string(d.box_width[0]);
        case K::RandAugment: return std::to_string(op.num_ops) + " random ops";
        default: return "";
    }
}

}  // namespace

std::string ImageStepsKey(const PreparedImageSpec& spec, const image::ImageAugmentation& augmentation,
                          const std::vector<CvImageStep>& cv_steps) {
    std::string key = std::to_string(spec.width) + "x" + std::to_string(spec.height) + ModeName(spec.mode);
    for (const auto& op : augmentation.ops) {
        key += "|" + std::to_string(static_cast<int>(op.kind));
        for (float v : {static_cast<float>(op.height), static_cast<float>(op.width), static_cast<float>(op.padding),
                        op.probability, op.max_angle, static_cast<float>(op.interpolation), op.brightness, op.contrast,
                        op.saturation, op.hue, static_cast<float>(op.kernel_size), op.sigma,
                        static_cast<float>(op.morphology), static_cast<float>(op.erase_method),
                        static_cast<float>(op.cutout_size), op.scale_min, op.scale_max, op.ratio_min, op.ratio_max,
                        op.value, static_cast<float>(op.num_ops), static_cast<float>(op.magnitude)})
            key += "," + Number(v);
    }
    for (const auto& step : cv_steps)
        key += "|cv" + std::to_string(step.position) + CvImageStepIdentity(step).dump();
    return key;
}

ImagePipelinePreview PreviewImagePipeline(const std::string& path, const PreparedImageSpec& spec,
                                          const image::ImageAugmentation& augmentation,
                                          const std::vector<CvImageStep>& cv_steps, int draws, uint32_t seed) {
    ImagePipelinePreview out;
    if (spec.width <= 0 || spec.height <= 0) {
        out.error = "The graph has no Resize node.";
        return out;
    }
    image::ImageShape shape{static_cast<size_t>(spec.height), static_cast<size_t>(spec.width), 3};
    std::vector<uint8_t> decoded(shape.Size());
    if (!DecodePreparedImage(path, spec, decoded.data())) {
        out.error = "This file cannot be read as an image.";
        return out;
    }
    try {
        std::vector<float> rows(decoded.size());
        for (size_t i = 0; i < rows.size(); ++i) rows[i] = decoded[i] / 255.0f;
        Tensor x({1, shape.Size()}, rows.data());
        out.decoded = Picture(x, shape);
        out.decoded.title = "Resize " + std::to_string(spec.width) + kTimes + std::to_string(spec.height);
        out.decoded.detail = std::string(ModeName(spec.mode)) + ", once";

        const ImageTransformSplit split = SplitFixedImageTransforms(augmentation, cv_steps);
        std::mt19937 rng(seed);
        for (const auto& step : split.fixed) {
            const auto* cv_step = std::get_if<CvImageStep>(&step);
            const std::string why = cv_step ? ValidateCvImageStep(*cv_step, shape)
                                            : image::ValidateImageOp(std::get<image::ImageOp>(step), shape);
            if (!why.empty()) {
                out.error = FixedImageStepText(step) + ": " + why;
                return out;
            }
            if (cv_step) {
                // On uint8, as the prepared set stores it.
                const float* values = x.ReadData<float>();
                std::vector<uint8_t> in(shape.Size());
                for (size_t i = 0; i < in.size(); ++i)
                    in[i] = static_cast<uint8_t>(std::lround(std::clamp(values[i], 0.0f, 1.0f) * 255.0f));
                const image::ImageShape after = CvImageShapeAfter(*cv_step, shape);
                std::vector<uint8_t> result(after.Size());
                std::string failed;
                if (!ApplyCvImageStep(*cv_step, shape, in.data(), result.data(), &failed)) {
                    out.error = failed;
                    return out;
                }
                std::vector<float> pixels(result.size());
                for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = result[i] / 255.0f;
                x = Tensor({1, after.Size()}, pixels.data());
                shape = after;
            } else {
                const auto& op = std::get<image::ImageOp>(step);
                const auto none = image::DrawImageOp(op, shape, 1, false, rng);
                x = image::ApplyImageOp(op, none, shape, x);
                shape = image::ImageShapeAfter(op, shape);
            }
            PreviewPicture p = Picture(x, shape);
            p.title = FixedImageStepText(step);
            p.detail = std::to_string(shape.width) + kTimes + std::to_string(shape.height) +
                       (cv_step ? ", once on the CPU" : ", once");
            out.fixed.push_back(std::move(p));
        }
        for (const auto& op : split.per_batch.ops) out.random_ops.push_back(ImageOpText(op));
        if (split.per_batch.ops.empty()) return out;
        for (int d = 0; d < draws; ++d) {
            std::mt19937 draw_rng(seed + 1 + static_cast<uint32_t>(d));
            Tensor y = x;
            image::ImageShape s = shape;
            std::string detail;
            for (const auto& op : split.per_batch.ops) {
                const std::string why = image::ValidateImageOp(op, s);
                if (!why.empty()) {
                    out.error = ImageOpText(op) + ": " + why;
                    return out;
                }
                const auto drawn = image::DrawImageOp(op, s, 1, true, draw_rng);
                y = image::ApplyImageOp(op, drawn, s, y);
                s = image::ImageShapeAfter(op, s);
                const std::string text = DrawText(op, drawn);
                if (!text.empty()) detail += (detail.empty() ? "" : ", ") + text;
            }
            PreviewPicture p = Picture(y, s);
            p.title = "Draw " + std::to_string(d + 1);
            p.detail = detail;
            out.draws.push_back(std::move(p));
        }
    } catch (const std::exception& e) {
        out.error = e.what();
    }
    return out;
}

}  // namespace cyxwiz
