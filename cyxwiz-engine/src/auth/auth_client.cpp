// auth_client.cpp - Authentication client implementation for CyxWiz Engine
#include "auth/auth_client.h"
#include "core/jwt_expiry.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <ctime>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#elif defined(__APPLE__)
#include <unistd.h>
#include <pwd.h>
#else
// Linux
#include <unistd.h>
#include <pwd.h>
#endif

using json = nlohmann::json;

namespace {

// Parse URL into host, port, and base path
struct ParsedUrl {
    std::string host;
    int port;
    std::string base_path;
};

ParsedUrl ParseUrl(const std::string& url) {
    ParsedUrl result;
    result.port = 80;
    result.base_path = "";

    std::string work = url;
    if (work.substr(0, 8) == "https://") {
        result.port = 443;
        work = work.substr(8);
    } else if (work.substr(0, 7) == "http://") {
        work = work.substr(7);
    }

    // Extract path
    size_t path_pos = work.find('/');
    if (path_pos != std::string::npos) {
        result.base_path = work.substr(path_pos);  // Keep the path (e.g., "/api")
        work = work.substr(0, path_pos);
    }

    // Check for port
    size_t port_pos = work.find(':');
    if (port_pos != std::string::npos) {
        result.host = work.substr(0, port_pos);
        result.port = std::stoi(work.substr(port_pos + 1));
    } else {
        result.host = work;
    }

    return result;
}

} // anonymous namespace

namespace cyxwiz::auth {

AuthClient& AuthClient::Instance() {
    static AuthClient instance;
    return instance;
}

AuthClient::AuthClient() {
    spdlog::info("AuthClient initialized with API URL: {}", api_base_url_);
    refresher_ = std::thread([this] { RefreshLoop(); });
}

AuthClient::~AuthClient() {
    {
        std::lock_guard<std::mutex> lock(refresher_mutex_);
        stop_refresher_ = true;
    }
    refresher_cv_.notify_all();
    if (refresher_.joinable()) refresher_.join();
}

void AuthClient::RefreshLoop() {
    std::unique_lock<std::mutex> lock(refresher_mutex_);
    while (!stop_refresher_) {
        refresher_cv_.wait_for(lock, std::chrono::minutes(5), [this] { return stop_refresher_; });
        if (stop_refresher_) break;
        const std::string token = GetJwtToken();
        if (token.empty()) continue;
        const long long left = cyxwiz::JwtExpiry(token) - static_cast<long long>(std::time(nullptr));
        if (left < cyxwiz::kJwtRefreshBeforeSeconds) {
            lock.unlock();
            if (!RefreshJwtToken() && left <= 0) {
                spdlog::warn("Sign-in expired; please sign in again");
                Logout();
            }
            lock.lock();
        }
    }
}

void AuthClient::SetApiBaseUrl(const std::string& url) {
    std::lock_guard<std::mutex> lock(mutex_);
    api_base_url_ = url;
    spdlog::info("AuthClient API URL set to: {}", url);
}

std::string AuthClient::GetApiBaseUrl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return api_base_url_;
}

void AuthClient::SetState(AuthState state) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == state) return;
        state_ = state;
    }

    spdlog::info("Auth state changed to: {}", AuthStateToString(state));

    if (on_state_changed_) {
        on_state_changed_(state);
    }
}

std::future<AuthResult> AuthClient::LoginWithEmail(const std::string& email, const std::string& password) {
    return std::async(std::launch::async, [this, email, password]() {
        SetState(AuthState::LoggingIn);

        json body;
        body["email"] = email;
        body["password"] = password;  // Send plain password - server handles hashing

        return DoLogin("/auth/login", body.dump());
    });
}

AuthResult AuthClient::DoLogin(const std::string& endpoint, const std::string& json_body) {
    AuthResult result;

    try {
        auto parsed = ParseUrl(api_base_url_);
        bool is_https = api_base_url_.substr(0, 5) == "https";
        std::string full_endpoint = parsed.base_path + endpoint;  // e.g., "/api" + "/auth/login"

        spdlog::debug("Attempting login to {}:{}{}", parsed.host, parsed.port, full_endpoint);

        httplib::Result res;

        if (is_https) {
            httplib::SSLClient client(parsed.host, parsed.port);
            client.set_connection_timeout(10);
            client.set_read_timeout(30);
            res = client.Post(full_endpoint, json_body, "application/json");
        } else {
            httplib::Client client(parsed.host, parsed.port);
            client.set_connection_timeout(10);
            client.set_read_timeout(30);
            res = client.Post(full_endpoint, json_body, "application/json");
        }

        if (!res) {
            result.error = "Network error: Could not connect to server";
            spdlog::error("Login failed: {}", result.error);
            SetState(AuthState::Offline);
            return result;
        }

        spdlog::debug("Login response status: {}", res->status);

        if (res->status == 200) {
            try {
                json response = json::parse(res->body);

                if (response.contains("token")) {
                    // Update user data under lock
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        jwt_token_ = response["token"].get<std::string>();

                        // Parse user info from response
                        if (response.contains("user")) {
                            auto& user = response["user"];
                            user_info_.id = user.value("id", "");
                            user_info_.email = user.value("email", "");
                            user_info_.username = user.value("username", "");
                            user_info_.name = user.value("name", "");
                            user_info_.role = user.value("role", "user");
                        }

                        result.success = true;
                        result.user_info = user_info_;
                    }
                    // Lock released - now safe to call SetState

                    SetState(AuthState::Authenticated);

                    // Save session
                    SaveSession();

                    spdlog::info("Login successful for user: {}", result.user_info.email);
                } else {
                    result.error = "Invalid response: missing token";
                    SetState(AuthState::Offline);
                }
            } catch (const json::exception& e) {
                result.error = "Invalid JSON response";
                spdlog::error("JSON parse error: {}", e.what());
                SetState(AuthState::Offline);
            }
        } else {
            // Parse error message from response
            try {
                json error_response = json::parse(res->body);
                result.error = error_response.value("message",
                    error_response.value("error", "Login failed"));
            } catch (...) {
                result.error = "Login failed with status " + std::to_string(res->status);
            }
            spdlog::error("Login failed: {} (status {})", result.error, res->status);
            SetState(AuthState::Offline);
        }
    } catch (const std::exception& e) {
        result.error = std::string("Network error: ") + e.what();
        spdlog::error("Login exception: {}", e.what());
        SetState(AuthState::Offline);
    }

    return result;
}

void AuthClient::Logout() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jwt_token_.clear();
        user_info_ = UserInfo{};
    }

    ClearSavedSession();
    SetState(AuthState::Offline);

    spdlog::info("User logged out");
}

bool AuthClient::IsAuthenticated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !jwt_token_.empty() && state_ == AuthState::Authenticated;
}

std::string AuthClient::GetJwtToken() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return jwt_token_;
}

UserInfo AuthClient::GetUserInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return user_info_;
}

AuthState AuthClient::GetState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

bool AuthClient::RefreshJwtToken() {
    std::string token;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (jwt_token_.empty()) return false;
        token = jwt_token_;
    }

    try {
        auto parsed = ParseUrl(api_base_url_);
        bool is_https = api_base_url_.substr(0, 5) == "https";
        std::string full_endpoint = parsed.base_path + "/auth/refresh";

        httplib::Headers headers = {
            {"Authorization", "Bearer " + token}
        };

        httplib::Result res;

        if (is_https) {
            httplib::SSLClient client(parsed.host, parsed.port);
            client.set_connection_timeout(10);
            client.set_read_timeout(30);
            res = client.Post(full_endpoint, headers, "", "application/json");
        } else {
            httplib::Client client(parsed.host, parsed.port);
            client.set_connection_timeout(10);
            client.set_read_timeout(30);
            res = client.Post(full_endpoint, headers, "", "application/json");
        }

        if (res && res->status == 200) {
            json response = json::parse(res->body);
            if (response.contains("token")) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    jwt_token_ = response["token"].get<std::string>();
                    if (response.contains("user")) {
                        const auto& user = response["user"];
                        user_info_.id = user.value("id", user_info_.id);
                        user_info_.email = user.value("email", user_info_.email);
                        user_info_.username = user.value("username", user_info_.username);
                        user_info_.name = user.value("name", user_info_.name);
                        user_info_.role = user.value("role", user_info_.role);
                    }
                }
                SaveSession();
                spdlog::info("Sign-in token refreshed");
                return true;
            }
        }
        if (res) spdlog::warn("Token refresh refused (HTTP {})", res->status);
    } catch (const std::exception& e) {
        spdlog::error("Failed to refresh JWT token: {}", e.what());
    }

    return false;
}

std::string AuthClient::GetTokenStoragePath() const {
    std::string config_dir;

#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
        config_dir = std::string(path) + "\\CyxWiz\\Engine";
    }
#elif defined(__APPLE__)
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        home = pw ? pw->pw_dir : "/tmp";
    }
    config_dir = std::string(home) + "/Library/Application Support/CyxWiz/Engine";
#else
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        home = pw ? pw->pw_dir : "/tmp";
    }
    config_dir = std::string(home) + "/.config/cyxwiz/engine";
#endif

    std::filesystem::create_directories(config_dir);
    return config_dir + "/session.json";
}

bool AuthClient::LoadSavedSession() {
    std::string path = GetTokenStoragePath();

    if (!std::filesystem::exists(path)) {
        return false;
    }

    try {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        json data = json::parse(file);

        std::lock_guard<std::mutex> lock(mutex_);
        jwt_token_ = data.value("jwt_token", "");

        if (data.contains("user_info")) {
            auto& ui = data["user_info"];
            user_info_.id = ui.value("id", "");
            user_info_.email = ui.value("email", "");
            user_info_.username = ui.value("username", "");
            user_info_.name = ui.value("name", "");
            user_info_.role = ui.value("role", "user");
        }

        const long long now = static_cast<long long>(std::time(nullptr));
        if (!jwt_token_.empty() && cyxwiz::JwtExpiry(jwt_token_) <= now) {
            spdlog::info("Saved sign-in for {} has expired; sign in again", user_info_.email);
            jwt_token_.clear();
            user_info_ = UserInfo{};
            return false;
        }
        if (!jwt_token_.empty()) {
            state_ = AuthState::Authenticated;
            spdlog::info("Restored saved session for: {}", user_info_.email);
            return true;
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to load saved session: {}", e.what());
    }

    return false;
}

bool AuthClient::SaveSession() {
    std::string path = GetTokenStoragePath();

    try {
        json data;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            data["jwt_token"] = jwt_token_;
            data["user_info"] = {
                {"id", user_info_.id},
                {"email", user_info_.email},
                {"username", user_info_.username},
                {"name", user_info_.name},
                {"role", user_info_.role}
            };
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            spdlog::error("Failed to open session file for writing: {}", path);
            return false;
        }

        file << data.dump(2);
        spdlog::debug("Session saved to: {}", path);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to save session: {}", e.what());
    }

    return false;
}

void AuthClient::ClearSavedSession() {
    std::string path = GetTokenStoragePath();
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
        spdlog::debug("Saved session cleared");
    }
}

void AuthClient::SetOnAuthStateChanged(AuthStateCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    on_state_changed_ = std::move(callback);
}

void AuthClient::OpenRegistrationPage() {
    const char* url = "https://cyxwiz.com/register";

#ifdef _WIN32
    ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    std::string cmd = "open \"" + std::string(url) + "\"";
    system(cmd.c_str());
#else
    std::string cmd = "xdg-open \"" + std::string(url) + "\"";
    system(cmd.c_str());
#endif

    spdlog::info("Opened registration page in browser");
}

const char* AuthStateToString(AuthState state) {
    switch (state) {
        case AuthState::Offline: return "Offline";
        case AuthState::LoggingIn: return "Logging In";
        case AuthState::Authenticated: return "Authenticated";
        default: return "Unknown";
    }
}

} // namespace cyxwiz::auth
