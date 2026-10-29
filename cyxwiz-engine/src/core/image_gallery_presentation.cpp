#include "image_gallery_presentation.h"

#include <algorithm>
#include <cstdio>
#include <climits>
#include <iterator>

namespace cyxwiz::image_gallery {

namespace {

std::string Thousands(uint64_t n) {
    std::string digits = std::to_string(n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return digits;
}

std::string Kilobytes(int64_t bytes) {
    char out[32];
    if (bytes >= 1000 * 1000) std::snprintf(out, sizeof(out), "%.1f MB", bytes / 1e6);
    else std::snprintf(out, sizeof(out), "%.1f KB", bytes / 1e3);
    return out;
}

std::string ClassName(const Input& in, int label) {
    if (label >= 0 && static_cast<size_t>(label) < in.class_names.size()) return in.class_names[static_cast<size_t>(label)];
    return label >= 0 ? std::to_string(label) : "no label";
}

const char* ChannelsText(int channels) {
    switch (channels) {
        case 1: return "grey";
        case 3: return "colour";
        case 4: return "colour + alpha";
        default: return "";
    }
}

// The reasons in a verdict ("blurry, low contrast").
std::vector<std::string> Reasons(const std::string& verdict) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at < verdict.size()) {
        const size_t comma = verdict.find(", ", at);
        out.push_back(verdict.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
        if (comma == std::string::npos) break;
        at = comma + 2;
    }
    return out;
}

const char* kReasons[] = {"blurry", "low contrast", "dark", "bright", "duplicate"};

bool Keeps(const std::string& filter, const Row& row) {
    if (filter == "all") return true;
    if (filter == "unreadable") return !row.readable;
    if (filter == "fails") return row.measured && !row.verdict.empty();
    if (filter.rfind("reason:", 0) == 0) {
        const auto reasons = Reasons(row.verdict);
        return std::find(reasons.begin(), reasons.end(), filter.substr(7)) != reasons.end();
    }
    if (filter.rfind("class:", 0) == 0) return std::to_string(row.label) == filter.substr(6);
    return true;
}

}  // namespace

View Build(const Input& in) {
    View v;

    // Chips: all, each class, unreadable (only when some are).
    size_t unreadable = 0;
    std::vector<size_t> per_class(in.class_names.size(), 0);
    for (const auto& row : in.rows) {
        if (!row.readable) ++unreadable;
        if (row.label >= 0 && static_cast<size_t>(row.label) < per_class.size()) ++per_class[static_cast<size_t>(row.label)];
    }
    std::string filter = in.filter;
    v.chips.push_back({"all", "All", in.rows.size(), {}, false});
    for (size_t c = 0; c < per_class.size(); ++c) {
        const std::string name = in.class_names[c];
        v.chips.push_back({"class:" + std::to_string(c), name == std::to_string(c) ? "Class " + name : name, per_class[c], {}, false});
    }
    if (unreadable > 0) v.chips.push_back({"unreadable", "Unreadable", unreadable, {}, false});
    // Quality, once measured: the images that fail, then each reason that occurs.
    size_t fails = 0;
    std::vector<size_t> per_reason(std::size(kReasons), 0);
    for (const auto& row : in.rows) {
        if (!row.measured) continue;
        v.quality = true;
        if (row.verdict.empty()) continue;
        ++fails;
        for (const auto& reason : Reasons(row.verdict))
            for (size_t r = 0; r < per_reason.size(); ++r)
                if (reason == kReasons[r]) ++per_reason[r];
    }
    if (v.quality) {
        v.chips.push_back({"fails", "Fails a check", fails, {}, false});
        for (size_t r = 0; r < per_reason.size(); ++r)
            if (per_reason[r]) v.chips.push_back({std::string("reason:") + kReasons[r], kReasons[r], per_reason[r], {}, false});
    }
    if (std::none_of(v.chips.begin(), v.chips.end(), [&](const Chip& c) { return c.key == filter; })) filter = "all";
    for (auto& chip : v.chips) {
        chip.on = chip.key == filter;
        chip.count_text = Thousands(chip.count);
    }

    // The matching rows, sorted.
    std::vector<size_t> order;
    order.reserve(in.rows.size());
    for (size_t i = 0; i < in.rows.size(); ++i)
        if (Keeps(filter, in.rows[i])) order.push_back(i);
    const auto area = [&](size_t i) {
        const Row& r = in.rows[i];
        return r.readable && r.width > 0 && r.height > 0 ? static_cast<long long>(r.width) * r.height : -1LL;
    };
    if (in.sort == Sort::Smallest)
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const long long x = area(a), y = area(b);
            return (x < 0 ? LLONG_MAX : x) < (y < 0 ? LLONG_MAX : y);  // unreadable last
        });
    else if (in.sort == Sort::Largest)
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return area(a) > area(b); });
    else if (in.sort == Sort::Blurriest)
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const Row& x = in.rows[a];
            const Row& y = in.rows[b];
            if (x.measured != y.measured) return x.measured;  // not measured last
            return x.measured && x.blur < y.blur;
        });
    v.matching = order.size();

    // The page.
    const size_t per_page = std::max<size_t>(1, in.per_page);
    v.pages = (order.size() + per_page - 1) / per_page;
    v.page = v.pages == 0 ? 0 : std::min(in.page, v.pages - 1);
    const size_t begin = v.page * per_page;
    const size_t end = std::min(order.size(), begin + per_page);
    for (size_t k = begin; k < end; ++k) {
        v.shown.push_back(order[k]);
        v.tags.push_back(ClassName(in, in.rows[order[k]].label));
        const auto reasons = Reasons(in.rows[order[k]].verdict);
        v.flags.push_back(reasons.empty() ? std::string()
                                          : reasons[0] + (reasons.size() > 1 ? " +" + std::to_string(reasons.size() - 1) : ""));
    }
    v.has_previous = v.page > 0;
    v.has_next = v.page + 1 < v.pages;
    v.showing = order.empty() ? "No image matches."
                              : Thousands(begin + 1) + " to " + Thousands(end) + " of " + Thousands(order.size());

    // The selected image, while the filter keeps it.
    if (in.selected >= 0 && static_cast<size_t>(in.selected) < in.rows.size() &&
        Keeps(filter, in.rows[static_cast<size_t>(in.selected)])) {
        v.selected = in.selected;
        const Row& r = in.rows[static_cast<size_t>(in.selected)];
        v.detail_title = r.file;
        v.detail.push_back({"Row", Thousands(static_cast<uint64_t>(in.selected) + 1) + " of " + Thousands(in.rows.size())});
        const std::string name = ClassName(in, r.label);
        v.detail.push_back({"Label", name == std::to_string(r.label) ? name : std::to_string(r.label) + " \xC2\xB7 " + name});
        if (r.readable) {
            std::string size = std::to_string(r.width) + " \xC3\x97 " + std::to_string(r.height) + " px";
            if (*ChannelsText(r.channels)) size += std::string(", ") + ChannelsText(r.channels);
            v.detail.push_back({"Original", size});
        } else {
            v.detail.push_back({"Original", "cannot be read as an image"});
        }
        v.detail.push_back({"On disk", Kilobytes(r.bytes)});
        if (r.measured) {
            char text[32];
            std::snprintf(text, sizeof(text), "%.0f", r.blur);
            v.detail.push_back({"Blur", text});
            std::snprintf(text, sizeof(text), "%.0f", r.brightness);
            v.detail.push_back({"Brightness", text});
            std::snprintf(text, sizeof(text), "%.2f", r.contrast);
            v.detail.push_back({"Contrast", text});
            v.detail.push_back({"Verdict", r.verdict.empty() ? "passes the checks" : r.verdict});
            if (!r.duplicate_of.empty()) v.detail.push_back({"Same as", r.duplicate_of});
        }
        v.detail.push_back({"Path", r.path});
    }
    return v;
}

}  // namespace cyxwiz::image_gallery
