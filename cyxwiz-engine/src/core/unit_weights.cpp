#include "unit_weights.h"

#include "count_text.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cyxwiz::unit_weights {

namespace {

constexpr const char* kTimes = " \xC3\x97 ";  // " × "

// A row-major [rows, columns] matrix as the column-major (columns, rows)
// grid it already is: row r is column r.
af::array Grid(const Tensor& weights) {
    const auto& shape = weights.Shape();
    if (shape.size() != 2) throw std::runtime_error("the weights are not a [units, inputs] matrix");
    return af::moddims(weights.GetRowMajorFlatArray().as(f32), static_cast<dim_t>(shape[1]),
                       static_cast<dim_t>(shape[0]));
}

}  // namespace

std::optional<ImageShape> ReadsImage(size_t dense_inputs, size_t image_height, size_t image_width,
                                     size_t image_channels, bool only_parameter_free_layers_before) {
    if (!only_parameter_free_layers_before || image_height < 2 || image_width < 2) return std::nullopt;
    if (image_channels != 1 && image_channels != 3) return std::nullopt;
    const ImageShape image{image_height, image_width, image_channels};
    if (dense_inputs != image.Pixels()) return std::nullopt;
    return image;
}

std::vector<float> RowNorms(const Tensor& weights) {
    const af::array grid = Grid(weights);
    const af::array norms = af::sqrt(af::sum(grid * grid, 0));
    std::vector<float> out(static_cast<size_t>(grid.dims(1)));
    if (!out.empty()) norms.host(out.data());
    return out;
}

std::vector<size_t> Order(const std::vector<float>& norms, bool by_norm, size_t count) {
    std::vector<size_t> units(norms.size());
    std::iota(units.begin(), units.end(), size_t{0});
    if (by_norm)
        std::stable_sort(units.begin(), units.end(), [&](size_t a, size_t b) {
            const float na = std::isfinite(norms[a]) ? norms[a] : -1.0f;
            const float nb = std::isfinite(norms[b]) ? norms[b] : -1.0f;
            return na > nb;
        });
    if (count > 0 && count < units.size()) units.resize(count);
    return units;
}

std::vector<float> Rows(const Tensor& weights, const std::vector<size_t>& units) {
    if (units.empty()) return {};
    const af::array grid = Grid(weights);
    std::vector<unsigned> index;
    index.reserve(units.size());
    for (size_t u : units) {
        if (u >= static_cast<size_t>(grid.dims(1))) throw std::runtime_error("no such unit");
        index.push_back(static_cast<unsigned>(u));
    }
    const af::array picked =
        af::lookup(grid, af::array(static_cast<dim_t>(index.size()), index.data()), 1);  // (inputs, units)
    std::vector<float> out(static_cast<size_t>(picked.elements()));
    picked.host(out.data());  // column-major: one unit's row after another
    return out;
}

Picture Draw(const float* row, const ImageShape& image) {
    Picture picture;
    const size_t n = image.Pixels();
    picture.pixels.resize(n);
    float low = 0.0f, high = 0.0f, largest = 0.0f;
    bool first = true;
    for (size_t k = 0; k < n; ++k) {
        const float v = std::isfinite(row[k]) ? row[k] : 0.0f;
        picture.pixels[k] = v;
        low = first ? v : std::min(low, v);
        high = first ? v : std::max(high, v);
        largest = std::max(largest, std::fabs(v));
        first = false;
    }
    picture.low = low;
    picture.high = high;
    for (float& v : picture.pixels) {
        if (image.channels == 1)
            v = largest > 0.0f ? v / largest : 0.0f;
        else
            v = high > low ? (v - low) / (high - low) : 0.5f;
    }
    return picture;
}

std::string Lead(const ImageShape& image, size_t units, size_t shown, bool by_norm) {
    const std::string size = std::to_string(image.height) + kTimes + std::to_string(image.width);
    std::string text = "Each unit's " + CountText(image.Pixels()) + " input weights drawn as the " + size +
                       (image.channels == 3 ? " colour image they read: bright where a colour raises the unit, "
                                              "dark where it lowers it"
                                            : " image they read: red where a bright pixel raises the unit, blue "
                                              "where it lowers it") +
                       ", each on its own scale.";
    if (shown < units)
        text += " The " + std::to_string(shown) + (by_norm ? " with the largest weight norm first." : " first units.");
    else
        text += by_norm ? " The largest weight norm first." : " In unit order.";
    return text;
}

const std::string& Hint() {
    static const std::string hint =
        "This Dense layer reads the image flattened, one weight per pixel and colour, so a unit's weights are a "
        "picture of what it looks for: the unit's value is the image times this picture, summed, plus its bias. "
        "Strokes and blobs mean the unit has learned a shape; speckle means it has not learned much yet (a Local "
        "Debug run has the starting weights). A larger weight norm means a stronger unit. Click a unit to see it "
        "large here, then its top images from the data.";
    return hint;
}

}  // namespace cyxwiz::unit_weights
