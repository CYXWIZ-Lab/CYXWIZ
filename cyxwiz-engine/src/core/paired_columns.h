#pragma once

// Two columns of text cells as numbers that belong together (TOFIX134 P0
// item 3): a scatter plot or a correlation needs x and y from the same row.
// Filtering each column on its own shifted every pair after the first row
// that one of them skipped. Pure, no ImGui.

#include <string>
#include <vector>

namespace cyxwiz::chartdata {

// The whole cell as a finite number (spaces around it allowed); "12abc",
// "", "nan" and "inf" are not numbers.
bool ParseFinite(const std::string& text, double* out);

// The rows where both columns are finite numbers, in row order.
void PairedNumbers(const std::vector<std::vector<std::string>>& rows, int x_column, int y_column,
                   std::vector<double>* xs, std::vector<double>* ys);

}  // namespace cyxwiz::chartdata
