#pragma once

// The built-in world map (TOFIX134 P2b group 5, approved board 12): Natural
// Earth 1:110m country outlines (public domain), shipped with the Engine so
// maps work offline. Pure data: no ImGui.

#include <string>
#include <vector>

namespace cyxwiz::plot {

struct Country {
    std::string name;       // "United States of America"
    std::string name_long;  // "United States"
    std::string iso_a3;     // "USA" ("" for a few, e.g. Kosovo)
    std::string iso_a2;     // "US"
    std::string continent;
    // Outer rings, longitude / latitude in degrees (x0, y0, x1, y1, ...).
    std::vector<std::vector<double>> rings;
    double lon_min = 0, lon_max = 0, lat_min = 0, lat_max = 0;
};

// Every country, sorted by name.
const std::vector<Country>& WorldCountries();

// The country a region text names: its name, long or formal name, ISO A3 or
// A2 code, or a common alias ("UK", "USA", "Czech Republic"); case, spaces
// and punctuation ignored. -1 when none.
int FindCountry(const std::string& text);

// The country under a point (the smallest when outlines nest); -1 at sea.
int CountryAt(double lon, double lat);

}  // namespace cyxwiz::plot
