// server_address_widgets.h - the rows under a server address (TOFIX168 step
// 3b): each service, where it is and whether it answers. Settings > Network
// and the sign-in dialog draw them the same way.
#pragma once

#include "common/server_address_view.h"

namespace cyxwiz::servernode::gui {

void RenderServerRows(const cyxwiz::protocol::ServerAddressView& view);

}  // namespace cyxwiz::servernode::gui
