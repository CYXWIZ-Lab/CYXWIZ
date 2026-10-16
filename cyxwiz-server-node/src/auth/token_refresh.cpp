#include "auth/token_refresh.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::auth {

std::optional<std::string> RefreshTokenWithApi(const std::string& api_base_url, const std::string& token) {
    // Split "http://host:port/base" into the server and the path prefix.
    std::string rest = api_base_url;
    const auto scheme_end = rest.find("://");
    const std::string scheme = scheme_end == std::string::npos ? "http" : rest.substr(0, scheme_end);
    if (scheme_end != std::string::npos) rest = rest.substr(scheme_end + 3);
    const auto slash = rest.find('/');
    const std::string server = slash == std::string::npos ? rest : rest.substr(0, slash);
    std::string base_path = slash == std::string::npos ? "" : rest.substr(slash);
    while (!base_path.empty() && base_path.back() == '/') base_path.pop_back();

    try {
        httplib::Client client(scheme + "://" + server);
        client.set_connection_timeout(10);
        client.set_read_timeout(30);
        const httplib::Headers headers = {{"Authorization", "Bearer " + token}};
        auto res = client.Post(base_path + "/auth/refresh", headers, "", "application/json");
        if (!res) {
            spdlog::warn("Token refresh: the sign-in service at {} did not answer", api_base_url);
            return std::nullopt;
        }
        if (res->status != 200) {
            spdlog::warn("Token refresh refused (HTTP {})", res->status);
            return std::nullopt;
        }
        const auto body = nlohmann::json::parse(res->body);
        if (!body.contains("token")) return std::nullopt;
        return body["token"].get<std::string>();
    } catch (const std::exception& e) {
        spdlog::warn("Token refresh failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace cyxwiz::servernode::auth
