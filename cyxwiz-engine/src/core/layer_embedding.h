#pragma once

// Embedding (TOFIX178 6.4b, approved mockup boards Main and Dense): a flat
// layer output over many items projected to 2-D with Explore > Projection's
// methods (cyxwiz::DimensionalityReduction: PCA, t-SNE, UMAP), coloured by
// class. Pure data in and out on the host; the items and their outputs come
// from the run's rebuilt model.

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::layer_embedding {

enum class Method { PCA, TSNE, UMAP };

struct Projection {
    std::vector<float> x;
    std::vector<float> y;
    std::string about;  // "PCA, 2 components, 61 % of the variance", "t-SNE, perplexity 30", ...
    std::string error;  // set when it could not be computed
};
// `rows` items x features, row-major. t-SNE's perplexity and UMAP's
// neighbours are kept below the item count.
Projection Project(const std::vector<float>& rows, size_t items, size_t features, Method method);

// The points scaled to [0, 1] on each axis (a constant axis sits at 0.5).
struct Scaled {
    std::vector<float> x;
    std::vector<float> y;
};
Scaled ToUnit(const Projection& projection);

// The point nearest (px, py) within `radius` (all in unit coordinates, the
// same aspect as drawn: `aspect` = height / width of the plot).
std::optional<size_t> Nearest(const Scaled& points, float px, float py, float radius, float aspect);

// Each class and how many items have it, in label order (0, "cat", 160);
// items without a class (-1) are left out.
struct ClassCount {
    int label = -1;
    std::string name;
    size_t count = 0;
};
std::vector<ClassCount> ClassCounts(const std::vector<int>& labels, const std::vector<std::string>& names);

const char* MethodName(Method method);

}  // namespace cyxwiz::layer_embedding
