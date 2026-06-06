#pragma once

#include "../core/node_metadata.h"
#include <functional>
#include <map>
#include <string>

namespace gui {

struct MLNode;

namespace properties_metadata {

using FallbackRenderer = std::function<void(MLNode&)>;
using InvalidateCallback = std::function<void()>;

void RenderParametersContent(
    MLNode& node,
    const cyxwiz::NodeMetadata* metadata,
    std::map<std::string, std::string>& validation_errors,
    const FallbackRenderer& render_fallback,
    const InvalidateCallback& invalidate);

} // namespace properties_metadata

} // namespace gui
