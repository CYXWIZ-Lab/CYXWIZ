#pragma once

// Open Model File (TOFIX129 B5 E-1): what a probed model file holds and
// whether its graph can be opened in Studio. Pure data in and out (no ImGui).

#include "model_format.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::model_open {

struct View {
    bool readable = false;
    std::string error;             // why the file could not be read
    std::string format_line;       // "CyxWiz package v3"
    std::vector<std::pair<std::string, std::string>> rows;  // Name, Parameters, Trained, Contains, Size
    std::string layers;            // "embedding [5000, 100], lstm ..." (capped)
    bool can_open_graph = false;
    std::string open_disabled_reason;
    std::string note;
};

// The view of a probed file.
View BuildView(const ProbeResult& probe);

// The layers with their shapes, at most `limit` of them, then "and N more".
std::string LayersLine(const ProbeResult& probe, size_t limit = 40);

}  // namespace cyxwiz::model_open
