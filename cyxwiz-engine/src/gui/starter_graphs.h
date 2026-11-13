#pragma once

// The example graphs the Engine offers to start from: the start page's task
// starters and the empty Studio canvas's "Open an example graph" (TOFIX129
// A3-3). Each is a file under examples/cyxgraph next to the Engine; one that
// is not found is left out.

#include <string>
#include <vector>

namespace gui {

struct StarterGraph {
    std::string title;        // "Multiclass image classification"
    std::string description;  // "MNIST MLP graph for a tabular digit dataset with ten classes."
    std::string domain;       // "Multiclass classification"
    std::string icon;         // its icon glyph
    std::string path;         // the file, absolute
};

// The starters found next to this Engine, in their fixed order.
std::vector<StarterGraph> FindStarterGraphs();

}  // namespace gui
