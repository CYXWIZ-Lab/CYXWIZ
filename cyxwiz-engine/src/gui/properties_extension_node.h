#pragma once

// Properties of an installed extension node other than the MuJoCo Plant
// (TOFIX125 P2 step 2.8): typed settings and the gradient check.

#include "properties_node_editors.h"
#include "../core/extension_node_contract.h"

namespace gui::properties_node_editors {

void RenderInstalledExtensionProperties(MLNode& node, const cyxwiz::ExtensionNodeDescriptor& descriptor,
                                        RenderNodePropertiesContext& context);

}  // namespace gui::properties_node_editors
