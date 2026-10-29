#include "image_profile_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <map>

namespace cyxwiz::image_profile {

namespace {

const char* kTimes = " \xC3\x97 ";
const char* kDot = " \xC2\xB7 ";

std::string Thousands(uint64_t n) {
    std::string digits = std::to_string(n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return digits;
}

std::string Bytes(uint64_t bytes) {
    char out[32];
    if (bytes >= 1000ull * 1000 * 1000) std::snprintf(out, sizeof(out), "%.1f GB", bytes / 1e9);
    else if (bytes >= 1000ull * 1000) std::snprintf(out, sizeof(out), "%.0f MB", bytes / 1e6);
    else if (bytes >= 1000) std::snprintf(out, sizeof(out), "%.0f KB", bytes / 1e3);
    else std::snprintf(out, sizeof(out), "%llu bytes", static_cast<unsigned long long>(bytes));
    return out;
}

std::string ChannelsName(int channels) {
    switch (channels) {
        case 1: return "grey";
        case 3: return "colour";
        case 4: return "colour + alpha";
        default: return std::to_string(channels) + " channels";
    }
}

std::string Median(std::vector<int> values) {
    if (values.empty()) return "-";
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return std::to_string(values[values.size() / 2]);
}

std::string ResizeNote(const Input& in, size_t enlarged) {
    const std::string size = std::to_string(in.resize_width) + kTimes + std::to_string(in.resize_height);
    const std::string smaller = enlarged == 0 ? std::string()
                                              : " " + Thousands(enlarged) + (enlarged == 1 ? " image is" : " images are") +
                                                    " smaller than " + size + " on a side and " +
                                                    (enlarged == 1 ? "is" : "are") + " enlarged.";
    if (in.resize_mode == "fit") return "fit: the whole image is kept and padded with black to " + size + "." + smaller;
    if (in.resize_mode == "fill") return "fill: covers " + size + " and crops the long side." + smaller;
    if (in.resize_mode == "center")
        return "center: no scaling; the middle " + size + " is kept, smaller images are padded.";
    return "exact: stretched to " + size + ", aspect is not kept." + smaller;
}

}  // namespace

Histogram MakeHistogram(const std::vector<int>& values, size_t max_bins) {
    Histogram h;
    int max = 0;
    for (int v : values) max = std::max(max, v);
    if (max <= 0 || max_bins == 0) return h;
    static const int kSteps[] = {1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000};
    int step = kSteps[std::size(kSteps) - 1];
    for (int s : kSteps) {
        if (static_cast<size_t>(max / s + 1) <= max_bins) {
            step = s;
            break;
        }
    }
    const int bins = max / step + 1;
    for (int i = 0; i <= bins; ++i) h.edges.push_back(i * step);
    h.counts.assign(static_cast<size_t>(bins), 0);
    for (int v : values) {
        if (v < 0) continue;
        ++h.counts[static_cast<size_t>(std::min(v / step, bins - 1))];
    }
    h.max_count = *std::max_element(h.counts.begin(), h.counts.end());
    return h;
}

View Build(const Input& in) {
    View v;
    const size_t total = in.labels.size();
    size_t unreadable = 0, square = 0, enlarged = 0;
    uint64_t bytes = 0;
    std::map<int, size_t> by_channels;
    std::vector<int> widths, heights;
    widths.reserve(total);
    heights.reserve(total);
    for (size_t i = 0; i < total; ++i) {
        if (i < in.bytes.size() && in.bytes[i] > 0) bytes += static_cast<uint64_t>(in.bytes[i]);
        const int w = i < in.widths.size() ? in.widths[i] : -1;
        const int h = i < in.heights.size() ? in.heights[i] : -1;
        if (w <= 0 || h <= 0) {
            ++unreadable;
            continue;
        }
        widths.push_back(w);
        heights.push_back(h);
        if (w == h) ++square;
        if (in.resize_width > 0 && (w < in.resize_width || h < in.resize_height)) ++enlarged;
        if (i < in.channels.size()) ++by_channels[in.channels[i]];
    }

    // Tiles: images, classes, on disk, unreadable, colour.
    v.tiles.push_back({Thousands(total), total == 1 ? "image" : "images"});
    v.tiles.push_back({Thousands(in.class_names.size()), in.class_names.size() == 1 ? "class" : "classes"});
    v.tiles.push_back({Bytes(bytes), "on disk"});
    v.tiles.push_back({Thousands(unreadable), "unreadable"});
    if (by_channels.size() == 1) {
        const int c = by_channels.begin()->first;
        v.tiles.push_back({c == 1 ? "Grey" : c == 3 ? "RGB" : c == 4 ? "RGBA" : ChannelsName(c), "all " + Thousands(total - unreadable)});
    } else if (!by_channels.empty()) {
        std::string mix;
        for (auto it = by_channels.rbegin(); it != by_channels.rend(); ++it) {
            if (!mix.empty()) mix += ", ";
            mix += Thousands(it->second) + " " + ChannelsName(it->first);
        }
        v.tiles.push_back({"Mixed", mix});
    }

    // Classes, in index order.
    v.classes_title = in.labels_source.empty() ? "Classes" + std::string(kDot) + "from the class folders"
                                               : "Classes" + std::string(kDot) + "from " + in.labels_source;
    std::vector<size_t> counts(in.class_names.size(), 0);
    for (int label : in.labels)
        if (label >= 0 && static_cast<size_t>(label) < counts.size()) ++counts[static_cast<size_t>(label)];
    for (size_t c = 0; c < counts.size(); ++c) {
        ClassRow row;
        row.label = static_cast<int>(c);
        row.name = in.class_names[c];
        row.count = counts[c];
        row.share = total ? static_cast<float>(counts[c]) / static_cast<float>(total) : 0.0f;
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%.0f%%", row.share * 100.0f);
        row.text = Thousands(counts[c]) + kDot + pct;
        v.classes.push_back(std::move(row));
    }
    if (counts.size() >= 2) {
        const auto [lo, hi] = std::minmax_element(counts.begin(), counts.end());
        if (*lo == 0) {
            v.balance = "A class has no images.";
        } else {
            const double ratio = static_cast<double>(*hi) / static_cast<double>(*lo);
            char text[96];
            if (ratio <= 1.5) std::snprintf(text, sizeof(text), "Balanced.");
            else std::snprintf(text, sizeof(text), "Imbalanced: the largest class is %.1f%sthe smallest.", ratio, "\xC3\x97 ");
            v.balance = text;
        }
    }

    // Original sizes and the Resize.
    if (!widths.empty()) {
        const auto [wmin, wmax] = std::minmax_element(widths.begin(), widths.end());
        const auto [hmin, hmax] = std::minmax_element(heights.begin(), heights.end());
        v.sizes.push_back({"Width", std::to_string(*wmin) + " to " + std::to_string(*wmax) + " px, median " + Median(widths)});
        v.sizes.push_back({"Height", std::to_string(*hmin) + " to " + std::to_string(*hmax) + " px, median " + Median(heights)});
        v.sizes.push_back({"Square", Thousands(square) + (square == 1 ? " image" : " images")});
    }
    if (in.resize_width > 0) {
        v.sizes.push_back({"Resize " + std::to_string(in.resize_width) + kTimes + std::to_string(in.resize_height),
                           ResizeNote(in, enlarged)});
    } else {
        v.sizes.push_back({"Resize", "This graph has no Resize node: training needs one to decode the images."});
    }
    v.widths = MakeHistogram(widths);
    v.heights = MakeHistogram(heights);

    // Prepared sets.
    for (const auto& s : in.sets) {
        SetRow row;
        row.size = std::to_string(s.width) + kTimes + std::to_string(s.height) + kTimes + std::to_string(s.channels);
        row.resize = s.mode;
        for (const auto& f : s.fixed) row.fixed += (row.fixed.empty() ? "" : ", ") + f;
        row.bytes = Bytes(s.bytes);
        row.in_use = s.in_use;
        v.sets.push_back(std::move(row));
    }
    // Channel statistics.
    // The measured set's size: the fixed steps (crops, pyramids) can change the Resize size.
    const std::string set_size = in.stats_width > 0
                                     ? std::to_string(in.stats_width) + kTimes + std::to_string(in.stats_height)
                                     : std::to_string(in.resize_width) + kTimes + std::to_string(in.resize_height);
    v.stats_title = "Channel statistics";
    v.can_measure = in.resize_width > 0 && in.stats_step == Input::StatsStep::Idle;
    const size_t c = in.channel_mean.size();
    if (c == 1) v.channels = {"grey"};
    else if (c == 3) v.channels = {"R", "G", "B"};
    else for (size_t i = 0; i < c; ++i) v.channels.push_back("channel " + std::to_string(i + 1));
    if (in.resize_width <= 0) {
        v.stats_note = "Needs the graph's Resize: the statistics are of the images as training reads them.";
    } else if (!in.stats_error.empty()) {
        v.stats_note = in.stats_error;
    } else if (in.stats_step == Input::StatsStep::Preparing) {
        v.stats_note = in.stats_total ? "Preparing images: " + Thousands(in.stats_done) + " of " + Thousands(in.stats_total)
                                      : "Preparing the images...";
    } else if (in.stats_step == Input::StatsStep::Measuring) {
        v.stats_note = "Measuring" + (in.stats_device.empty() ? std::string() : " on " + in.stats_device) + "...";
    } else if (c == 0) {
        v.stats_note = "Mean and spread of each channel over every pixel of the prepared set (Resize " + set_size +
                       ", then the fixed transforms), as Normalize receives them. Measure prepares the set first "
                       "when training has not; training then reuses it.";
    } else {
        v.stats_title += std::string(kDot) + set_size + ", " + Thousands(in.stats_images) + " images" +
                         (in.stats_device.empty() ? std::string() : ", on " + in.stats_device);
        StatsRow mean{"mean", {}}, sd{"std", {}};
        char text[32];
        for (size_t i = 0; i < c; ++i) {
            std::snprintf(text, sizeof(text), "%.4f", in.channel_mean[i]);
            mean.values.push_back(text);
            std::snprintf(text, sizeof(text), "%.4f", in.channel_std[i]);
            sd.values.push_back(text);
        }
        v.stats = {mean, sd};
        if (in.normalize && in.norm_std != 0.0f) {
            std::string centre, spread;
            for (size_t i = 0; i < c; ++i) {
                std::snprintf(text, sizeof(text), "%.2f", (in.channel_mean[i] - in.norm_mean) / in.norm_std);
                centre += (i ? ", " : "") + std::string(text) + (c > 1 ? " (" + v.channels[i] + ")" : "");
                std::snprintf(text, sizeof(text), "%.2f", in.channel_std[i] / in.norm_std);
                spread += (i ? ", " : "") + std::string(text);
            }
            char head[96];
            std::snprintf(head, sizeof(head), "Normalize uses mean %g, std %g: ", in.norm_mean, in.norm_std);
            v.stats_note = head + std::string("the model's inputs centre at ") + centre + " with spread " + spread +
                           ", where 0 and 1 are the aim.";
        } else {
            v.stats_note = "The graph has no Normalize node: the model's inputs are these values in 0 to 1.";
        }
    }

    // Quality.
    using QS = Input::QualityStep;
    v.quality_title = "Quality";
    v.can_measure_quality = in.resize_width > 0 && (in.quality_step == QS::NotMeasured || in.quality_step == QS::Measured);
    v.quality_button = in.quality_step == QS::Measured ? "Measure again" : "Measure quality";
    if (in.resize_width <= 0) {
        v.quality_note = "Needs the graph's Resize: quality is measured on the images as training reads them.";
    } else if (!in.quality_error.empty()) {
        v.quality_note = in.quality_error;
    } else if (in.quality_step == QS::Looking) {
        v.quality_note = "Looking for an earlier measurement...";
    } else if (in.quality_step == QS::NotMeasured) {
        v.quality_note = "Not measured yet. Measures blur, brightness, contrast and near duplicates of every image at " +
                         set_size + ", as the Quality Analyzer does. Training is not changed: only a Quality node on the "
                         "canvas leaves images out.";
    } else if (in.quality_step == QS::Measuring) {
        v.quality_note = in.quality_total ? "Measuring: " + Thousands(in.quality_done) + " of " + Thousands(in.quality_total) + " images"
                                          : "Measuring...";
    } else {
        v.quality_title += std::string(kDot) + "measured at " + set_size +
                           (in.quality_device.empty() ? std::string() : " on " + in.quality_device) + ", default checks";
        char pct[32];
        std::snprintf(pct, sizeof(pct), "%.1f%%", total ? 100.0 * static_cast<double>(in.rejected) / static_cast<double>(total) : 0.0);
        v.quality_summary = Thousands(in.rejected) + (in.rejected == 1 ? " image fails" : " images fail") + " a check (" + pct +
                            ")" + (in.multiple ? "; " + Thousands(in.multiple) + " fail more than one." : ".");
        const std::pair<const char*, size_t> reasons[] = {{"blurry", in.blurry}, {"low contrast", in.low_contrast},
                                                          {"dark", in.dark}, {"bright", in.bright},
                                                          {"duplicate", in.duplicates}};
        for (const auto& [reason, count] : reasons) {
            QualityRow row;
            row.reason = reason;
            row.count = count;
            std::snprintf(pct, sizeof(pct), "%.1f%%", total ? 100.0 * static_cast<double>(count) / static_cast<double>(total) : 0.0);
            row.text = Thousands(count) + kDot + pct;
            row.of_rejected = in.rejected ? static_cast<float>(count) / static_cast<float>(in.rejected) : 0.0f;
            v.quality.push_back(std::move(row));
        }
        std::string by_class;
        for (size_t k = 0; k < in.class_rejected.size() && k < in.class_names.size(); ++k) {
            by_class += (by_class.empty() ? "By class: " : ", ") + in.class_names[k] + kDot + Thousands(in.class_rejected[k]);
            if (k < in.class_total.size()) by_class += " of " + Thousands(in.class_total[k]);
        }
        v.quality_note = (by_class.empty() ? std::string() : by_class + ". ") +
                         "Checks: blur at least 700, brightness 50 to 220, contrast at least 0.12, duplicates within 4 bits. "
                         "Training leaves images out only through a Quality node on the canvas.";
    }

    if (!in.sets_listed) v.sets_note = "Reading cache/images...";
    else if (in.sets.empty()) v.sets_note = "None yet: training decodes the images once at the Resize size and keeps them here.";
    else v.sets_note = "Kept 30 days, at most 10 GB in all. A new Resize size or fixed transform makes a new set the first time training starts.";
    return v;
}

}  // namespace cyxwiz::image_profile
