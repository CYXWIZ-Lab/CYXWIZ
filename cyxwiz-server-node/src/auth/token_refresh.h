#pragma once

// POST <api>/auth/refresh with a still-valid sign-in token and return the new
// token (TOFIX136 E2). Used by the GUI (AuthManager) and the daemon
// (NodeClient), which keeps its own copy of the token for days.

#include <optional>
#include <string>

namespace cyxwiz::servernode::auth {

// `api_base_url` like "http://127.0.0.1:3002/api". nullopt on any failure.
std::optional<std::string> RefreshTokenWithApi(const std::string& api_base_url, const std::string& token);

}  // namespace cyxwiz::servernode::auth
