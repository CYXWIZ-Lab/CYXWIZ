#pragma once

// Image plots (TOFIX134 P2b group 3, approved board 10): the pixel columns
// of a row as a picture, for any table. One row, a gallery of the chosen
// rows, or the mean picture of each label value. The pixels are read only
// for the rows needed (a reader gives them), so a wide table such as
// MNIST's 784 columns is never held whole.

#include "plot_prepare.h"

#include <functional>
#include <string>
#include <vector>

namespace cyxwiz::plot {

struct ImageLayout {
    int width = 0, height = 0, channels = 1;
    std::string problem;  // "" when the columns make a picture
};

// The picture's shape from the number of pixel columns: square when the
// count is a square, 3 channels when asked or when count / 3 is a square,
// else spec.image_width wide (rows to fit).
ImageLayout ImageLayoutFor(const PlotSpec& spec, size_t pixel_columns);

// Builds the pictures. `chosen` are the chosen rows' places in the table
// (0-based, in order); `read(rows)` returns a Source with the pixel columns
// (spec.y_columns) and the label column (spec.color_column) for those rows.
Prepared PrepareImage(const PlotSpec& spec, const std::vector<size_t>& chosen,
                      const std::function<Source(const std::vector<size_t>&)>& read);

}  // namespace cyxwiz::plot
