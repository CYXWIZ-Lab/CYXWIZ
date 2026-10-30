#include "image_recipe_presentation.h"

#include "image_cv_steps.h"

#include <cstdio>

namespace cyxwiz::image_recipe {

namespace {

const char* kTimes = " \xC3\x97 ";

std::string ShapeText(const image::ImageShape& s) {
    return std::to_string(s.width) + kTimes + std::to_string(s.height) + kTimes + std::to_string(s.channels);
}

std::string Number(float value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%g", value);
    return text;
}

std::string ResizeText(const ImagePreprocessingConfig& p) {
    const char* mode = p.resize_mode == ResizeMode::Exact        ? "exact"
                       : p.resize_mode == ResizeMode::AspectFit  ? "fit, padded"
                       : p.resize_mode == ResizeMode::AspectFill ? "fill, cropped"
                       : p.resize_mode == ResizeMode::Center     ? "centre crop"
                                                                 : "";
    return "Resize " + std::to_string(p.target_width) + kTimes + std::to_string(p.target_height) +
           (*mode ? std::string(", ") + mode : std::string());
}

std::string Name(const Input& in, int id, const std::string& fallback) {
    const std::string name = in.name_of && id >= 0 ? in.name_of(id) : std::string();
    return name.empty() ? fallback : name;
}

}  // namespace

View Build(const Input& in) {
    View v;
    if (!in.config) {
        v.reason = "The graph is compiling; its image steps show here once it has.";
        return v;
    }
    const TrainingConfiguration& c = *in.config;
    const ImagePreprocessingConfig& p = c.image_preprocessing;
    if (p.resize_mode == ResizeMode::None || p.target_width <= 0 || p.target_height <= 0) {
        v.reason = "The images need a Resize after the Data Loader before any image step.";
        return v;
    }

    image::ImageShape shape{static_cast<size_t>(p.target_height), static_cast<size_t>(p.target_width), 3};
    Row resize;
    resize.node_id = in.resize_node_id;
    resize.name = Name(in, in.resize_node_id, "Resize");
    resize.what = ResizeText(p);
    resize.group = Group::Resize;
    resize.output = shape;
    resize.output_text = ShapeText(shape);
    v.rows.push_back(resize);

    // The device ops and the OpenCV steps in canvas order (MergeImageSteps),
    // keeping each op's index; the steps are fixed until the first random one
    // (SplitFixedImageTransforms).
    const auto& ops = c.image_augmentation.ops;
    const auto& cv = c.image_cv_steps;
    bool fixed = true;
    size_t next_cv = 0;
    const auto add_cv = [&](const CvImageStep& step, size_t ops_through) {
        shape = CvImageShapeAfter(step, shape);
        Row row;
        row.node_id = step.node_id;
        row.name = Name(in, step.node_id, step.node_name);
        row.what = CvImageStepText(step);
        row.group = Group::Once;
        row.opencv = true;
        row.output = shape;
        row.output_text = ShapeText(shape);
        row.ops_through = ops_through;
        v.rows.push_back(row);
    };
    for (size_t i = 0; i <= ops.size(); ++i) {
        for (; next_cv < cv.size() && cv[next_cv].position <= i; ++next_cv) add_cv(cv[next_cv], i);
        if (i == ops.size()) break;
        const image::ImageOp& op = ops[i];
        const int node = i < c.image_op_node_ids.size() ? c.image_op_node_ids[i] : -1;
        const bool random = image::IsRandomImageOp(op.kind);
        fixed = fixed && !random;
        shape = image::ImageShapeAfter(op, shape);
        // A preset's ops (one node) are one row.
        if (!v.rows.empty() && node >= 0 && v.rows.back().node_id == node && !v.rows.back().opencv &&
            v.rows.back().group != Group::Resize) {
            Row& last = v.rows.back();
            last.what += ", " + ImageOpText(op);
            last.random = last.random || random;
            if (!fixed) last.group = Group::PerBatch;
            last.output = shape;
            last.output_text = ShapeText(shape);
            last.ops_through = i + 1;
            continue;
        }
        Row row;
        row.node_id = node;
        row.name = Name(in, node, ImageOpText(op));
        row.what = ImageOpText(op);
        row.group = fixed ? Group::Once : Group::PerBatch;
        row.random = random;
        row.output = shape;
        row.output_text = ShapeText(shape);
        row.ops_through = i + 1;
        v.rows.push_back(row);
    }
    for (; next_cv < cv.size(); ++next_cv) add_cv(cv[next_cv], ops.size());

    const auto& aug = c.image_augmentation;
    if (aug.mix != image::BatchMix::None) {
        Row mix;
        mix.node_id = c.image_mix_node_id;
        mix.what = std::string(aug.mix == image::BatchMix::MixUp ? "MixUp" : "CutMix") + " (alpha " +
                   Number(aug.mix_alpha) + ", p " + Number(aug.mix_probability) + "), images and their labels";
        mix.name = Name(in, c.image_mix_node_id, mix.what);
        mix.group = Group::PerBatch;
        mix.random = true;
        mix.output = shape;
        mix.output_text = ShapeText(shape);
        mix.ops_through = ops.size();
        v.rows.push_back(mix);
    }

    const auto& pre = c.preprocessing;
    if (pre.has_normalization) {
        Row norm;
        norm.node_id = pre.norm_node_id;
        norm.name = Name(in, pre.norm_node_id, "Normalize");
        if (!pre.norm_channel_mean.empty()) {
            std::string means, stds;
            for (size_t k = 0; k < pre.norm_channel_mean.size(); ++k) {
                means += (k ? " / " : "") + Number(pre.norm_channel_mean[k]);
                stds += (k ? " / " : "") + Number(k < pre.norm_channel_std.size() ? pre.norm_channel_std[k] : 1.0f);
            }
            norm.what = "mean " + means + " \xC2\xB7 std " + stds + " (per channel)";
        } else {
            norm.what = "mean " + Number(pre.norm_mean) + " \xC2\xB7 std " + Number(pre.norm_std);
        }
        norm.group = Group::Normalize;
        norm.output = shape;
        norm.output_text = ShapeText(shape);
        norm.ops_through = ops.size();
        v.rows.push_back(norm);
    }

    for (const Row& row : v.rows) {
        if (row.group == Group::Once) ++v.once;
        else if (row.group == Group::PerBatch) ++v.per_batch;
    }
    const std::string input = "[" + std::to_string(shape.height) + ", " + std::to_string(shape.width) + ", " +
                              std::to_string(shape.channels) + "]";
    v.subtitle = (in.dataset.empty() ? std::string("The images") : in.dataset) + " to model input " + input +
                 " \xC2\xB7 " + std::to_string(v.once) + (v.once == 1 ? " step" : " steps") + " once, " +
                 std::to_string(v.per_batch) + " per batch";
    v.known = true;
    return v;
}

std::string MoveRefusal(const View& view, size_t from, long long after) {
    if (from >= view.rows.size()) return "There is no such step.";
    const Row& moved = view.rows[from];
    if (moved.group == Group::Resize || moved.group == Group::Normalize)
        return moved.name + " stays " + (moved.group == Group::Resize ? "first" : "last") + ".";
    if (after >= static_cast<long long>(view.rows.size())) return "There is no such step.";
    if (after >= 0 && view.rows[static_cast<size_t>(after)].group == Group::Normalize)
        return "Image steps come before " + view.rows[static_cast<size_t>(after)].name + ".";
    // The order after the move; an OpenCV step may not follow a random one.
    std::vector<const Row*> order;
    for (size_t i = 0; i < view.rows.size(); ++i)
        if (i != from) order.push_back(&view.rows[i]);
    size_t at = 0;  // insert position in `order`
    if (after < 0) {
        at = order.empty() || order.front()->group != Group::Resize ? 0 : 1;
    } else {
        const Row* target = &view.rows[static_cast<size_t>(after)];
        for (size_t i = 0; i < order.size(); ++i)
            if (order[i] == target) at = i + 1;
    }
    order.insert(order.begin() + static_cast<std::ptrdiff_t>(at), &moved);
    bool random_before = false;
    for (const Row* row : order) {
        if (row->opencv && random_before)
            return row->name + " is done once with the images: it must come before the first random step.";
        random_before = random_before || row->random;
    }
    return {};
}

}  // namespace cyxwiz::image_recipe
