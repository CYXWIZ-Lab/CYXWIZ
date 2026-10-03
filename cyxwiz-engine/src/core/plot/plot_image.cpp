#include "plot_image.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <sstream>

namespace cyxwiz::plot {

namespace {

int SquareRoot(size_t n) {
    const auto r = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(n))));
    return r * r == n ? static_cast<int>(r) : 0;
}

std::string LabelText(const SourceColumn& c, size_t row) {
    if (!c.numeric) return row < c.text.size() ? c.text[row] : std::string();
    if (row >= c.numbers.size() || !std::isfinite(c.numbers[row])) return "missing";
    std::ostringstream out;
    out << c.numbers[row];
    return out.str();
}

constexpr size_t kChunkRows = 2048;

}  // namespace

ImageLayout ImageLayoutFor(const PlotSpec& spec, size_t n) {
    ImageLayout l;
    if (n == 0) {
        l.problem = "Choose the pixel columns.";
        return l;
    }
    int channels = spec.image_channels;
    if (channels == 0) channels = !SquareRoot(n) && n % 3 == 0 && SquareRoot(n / 3) ? 3 : 1;
    if (n % static_cast<size_t>(channels) != 0) {
        l.problem = std::to_string(n) + " columns do not split into " + std::to_string(channels) + " channels.";
        return l;
    }
    const size_t pixels = n / static_cast<size_t>(channels);
    int w = spec.image_width;
    if (w <= 0) w = SquareRoot(pixels);
    if (w <= 0) w = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(pixels))));  // not a square: rows to fit
    l.width = w;
    l.height = static_cast<int>((pixels + static_cast<size_t>(w) - 1) / static_cast<size_t>(w));
    l.channels = channels;
    return l;
}

Prepared PrepareImage(const PlotSpec& spec, const std::vector<size_t>& chosen,
                      const std::function<Source(const std::vector<size_t>&)>& read) {
    Prepared p;
    p.spec = spec;
    p.img_rows = chosen.size();
    p.rows_selected = chosen.size();
    const ImageLayout layout = ImageLayoutFor(spec, spec.y_columns.size());
    if (!layout.problem.empty()) {
        p.problem = layout.problem;
        return p;
    }
    p.img_w = layout.width;
    p.img_h = layout.height;
    p.img_channels = layout.channels;
    if (chosen.empty()) {
        p.problem = "No rows to show.";
        return p;
    }
    const size_t pixels = static_cast<size_t>(p.img_w) * static_cast<size_t>(p.img_h);
    const size_t n = spec.y_columns.size();
    const size_t per_channel = n / static_cast<size_t>(p.img_channels);
    // The picture of one row of a source: channels last, missing as NaN.
    const auto picture_of = [&](const Source& src, size_t r, std::vector<double>& out) {
        out.assign(pixels * static_cast<size_t>(p.img_channels), NAN);
        for (size_t k = 0; k < n; ++k) {
            const SourceColumn* c = src.Find(spec.y_columns[k]);
            if (!c || !c->numeric || r >= c->numbers.size()) continue;
            size_t pixel, channel;
            if (p.img_channels == 3 && spec.image_planar) {
                channel = k / per_channel;
                pixel = k % per_channel;
            } else {
                channel = k % static_cast<size_t>(p.img_channels);
                pixel = k / static_cast<size_t>(p.img_channels);
            }
            if (pixel < pixels) out[pixel * static_cast<size_t>(p.img_channels) + channel] = c->numbers[r];
        }
    };
    std::vector<std::vector<double>> raw;
    std::vector<double> one;
    if (spec.image_mode == PlotSpec::ImageMode::MeanPerClass) {
        if (spec.color_column.empty()) {
            p.problem = "Choose a label column for the mean per class.";
            return p;
        }
        // Sums per label, read in chunks of rows.
        std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> sums;  // label -> (sum, count) per value
        for (size_t start = 0; start < chosen.size(); start += kChunkRows) {
            const std::vector<size_t> rows(chosen.begin() + static_cast<std::ptrdiff_t>(start),
                                           chosen.begin() + static_cast<std::ptrdiff_t>(std::min(chosen.size(), start + kChunkRows)));
            const Source src = read(rows);
            const SourceColumn* label = src.Find(spec.color_column);
            if (!label) {
                p.problem = "Column '" + spec.color_column + "' is not in the table.";
                return p;
            }
            for (size_t r = 0; r < rows.size(); ++r) {
                picture_of(src, r, one);
                auto& [sum, count] = sums[LabelText(*label, r)];
                if (sum.empty()) {
                    sum.assign(one.size(), 0.0);
                    count.assign(one.size(), 0.0);
                }
                for (size_t k = 0; k < one.size(); ++k)
                    if (std::isfinite(one[k])) {
                        sum[k] += one[k];
                        count[k] += 1.0;
                    }
            }
        }
        // Labels that are numbers in numeric order.
        std::vector<std::string> names;
        for (const auto& [name, v] : sums) names.push_back(name);
        std::stable_sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
            char* ea = nullptr;
            char* eb = nullptr;
            const double da = std::strtod(a.c_str(), &ea), db = std::strtod(b.c_str(), &eb);
            const bool na = ea && *ea == '\0' && !a.empty(), nb = eb && *eb == '\0' && !b.empty();
            if (na && nb) return da < db;
            if (na != nb) return na;
            return a < b;
        });
        for (const auto& name : names) {
            auto& [sum, count] = sums[name];
            std::vector<double> mean(sum.size(), NAN);
            double rows_in = 0;
            for (size_t k = 0; k < sum.size(); ++k) {
                if (count[k] > 0) mean[k] = sum[k] / count[k];
                rows_in = std::max(rows_in, count[k]);
            }
            raw.push_back(std::move(mean));
            Prepared::Picture pic;
            pic.label = name;
            pic.row = 0;
            pic.count = static_cast<size_t>(rows_in);
            p.pictures.push_back(std::move(pic));
        }
        p.label = {DataLabel::State::Exact, chosen.size(), chosen.size()};
    } else {
        std::vector<size_t> places;
        if (spec.image_mode == PlotSpec::ImageMode::OneRow) {
            places.push_back(std::min(static_cast<size_t>(std::max(1, spec.image_row)), chosen.size()) - 1);
        } else {
            for (size_t i = 0; i < chosen.size() && i < static_cast<size_t>(spec.gallery_max); ++i) places.push_back(i);
        }
        std::vector<size_t> rows;
        for (size_t i : places) rows.push_back(chosen[i]);
        const Source src = read(rows);
        const SourceColumn* label = spec.color_column.empty() ? nullptr : src.Find(spec.color_column);
        for (size_t r = 0; r < rows.size(); ++r) {
            picture_of(src, r, one);
            raw.push_back(one);
            Prepared::Picture pic;
            pic.row = places[r] + 1;
            if (label) pic.label = LabelText(*label, r);
            p.pictures.push_back(std::move(pic));
        }
        p.label = rows.size() < chosen.size() && spec.image_mode == PlotSpec::ImageMode::Gallery
                      ? DataLabel{DataLabel::State::Truncated, rows.size(), chosen.size()}
                      : DataLabel{DataLabel::State::Exact, rows.size(), rows.size()};
    }
    // The value range, then each picture scaled to 0..1.
    double lo = 0, hi = 1;
    if (spec.image_range == PlotSpec::ImageRange::Byte) {
        hi = 255;
    } else if (spec.image_range == PlotSpec::ImageRange::Auto) {
        bool first = true;
        for (const auto& pic : raw)
            for (double v : pic) {
                if (!std::isfinite(v)) continue;
                lo = first ? v : std::min(lo, v);
                hi = first ? v : std::max(hi, v);
                first = false;
            }
        if (hi <= lo) hi = lo + 1;
    }
    p.img_lo = lo;
    p.img_hi = hi;
    for (size_t i = 0; i < raw.size(); ++i) {
        auto& out = p.pictures[i].pix;
        out.resize(raw[i].size());
        for (size_t k = 0; k < raw[i].size(); ++k)
            out[k] = std::isfinite(raw[i][k]) ? static_cast<float>(std::clamp((raw[i][k] - lo) / (hi - lo), 0.0, 1.0)) : NAN;
    }
    return p;
}

}  // namespace cyxwiz::plot
