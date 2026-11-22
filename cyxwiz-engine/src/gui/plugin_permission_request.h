#pragma once

namespace gui {

// Draws the plugin permission request that PluginManager's request queue
// shows now: what the plugin wants, a tick per permission, "Allow selected"
// (off until something is ticked) and "Start without these" (also Esc).
// Call once per frame from the main window.
void RenderPluginPermissionRequest();

}  // namespace gui
