#pragma once

// Whether each service behind the one server address answers (TOFIX168
// step 3b). Blocking: run it off the UI thread.

#include "server_address_view.h"

#include <chrono>
#include <functional>
#include <string>

namespace cyxwiz::protocol {

// The app's own HTTP client: GET url, the status (0 when nothing answered)
// and the body. Each app passes its httplib-based getter (HTTPS included).
using HttpGetFn = std::function<int(const std::string& url, std::string* body)>;

// A front door answers for every service at its /health. Without one, central
// and the gateway are asked over gRPC and the API at its /health.
ServerReach ProbeServer(const std::string& typed, ServiceSet set, const HttpGetFn& http_get,
                        std::chrono::milliseconds timeout = std::chrono::seconds(3));

// The same for addresses already resolved (the separate addresses: no
// front door). An empty gateway is not asked.
ServerReach ProbeEndpoints(const ServerEndpoints& endpoints, ServiceSet set, const HttpGetFn& http_get,
                           std::chrono::milliseconds timeout = std::chrono::seconds(3));

}  // namespace cyxwiz::protocol
