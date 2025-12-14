// auth_manager.cpp - Authentication manager implementation
#include "auth/auth_manager.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <sstream>

// OpenSSL for SHA-256 hashing
#include <openssl/sha.h>
#include <openssl/evp.h>

// Device detection from cyxwiz-backend
#include <cyxwiz/device.h>

// MetricsCollector for detailed hardware detection (DXGI + NVML/ADL)
#include "core/metrics_collector.h"

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#elif defined(__APPLE__)
#include <unistd.h>
#include <pwd.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#else
// Linux
#include <unistd.h>
#include <pwd.h>
#include <sys/sysinfo.h>
#endif

using json = nlohmann::json;

namespace {

// SHA-256 hash function for password protection
std::string HashPassword(const std::string& password, const std::string& email) {
    // Use email as salt to prevent rainbow table attacks
    std::string salted = password + ":" + email + ":cyxwiz_salt_v1";

    unsigned char hash[SHA256_DIGEST_LENGTH];
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();

    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, salted.c_str(), salted.length());
    EVP_DigestFinal_ex(ctx, hash, nullptr);
    EVP_MD_CTX_free(ctx);

    // Convert to hex string
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }

    return ss.str();
}

} // anonymous namespace

namespace cyxwiz::servernode::auth {

AuthManager& AuthManager::Instance() {
    static AuthManager instance;
    return instance;
}

AuthManager::AuthManager() {
    spdlog::info("AuthManager initialized with API URL: {}", api_base_url_);
}

AuthManager::~AuthManager() = default;

void AuthManager::SetApiBaseUrl(const std::string& url) {
    std::lock_guard<std::mutex> lock(mutex_);
    api_base_url_ = url;
    spdlog::info("AuthManager API URL set to: {}", url);
}

std::string AuthManager::GetApiBaseUrl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return api_base_url_;
}

void AuthManager::SetState(AuthState state) {
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

AuthState AuthManager::GetState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

bool AuthManager::IsAuthenticated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ >= AuthState::Authenticated;
}

std::string AuthManager::GetJwtToken() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return jwt_token_;
}

std::string AuthManager::GetNodeToken() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return node_token_;
}

UserInfo AuthManager::GetUserInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return user_info_;
}

void AuthManager::SetOnAuthStateChanged(AuthStateCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    on_state_changed_ = std::move(callback);
}

std::future<AuthResult> AuthManager::LoginWithEmail(const std::string& email, const std::string& password) {
    return std::async(std::launch::async, [this, email, password]() -> AuthResult {
        SetState(AuthState::LoggingIn);

        json body;
        body["email"] = email;
        body["password"] = password;

        auto result = DoLogin("/auth/login", body.dump());

        if (result.success) {
            FetchUserProfile();
            SetState(AuthState::Authenticated);
            SaveSession();
        } else {
            SetState(AuthState::Offline);
        }

        return result;
    });
}

std::future<AuthResult> AuthManager::LoginWithWallet(const std::string& wallet_address, const std::string& signature) {
    return std::async(std::launch::async, [this, wallet_address, signature]() -> AuthResult {
        SetState(AuthState::LoggingIn);

        json body;
        body["wallet_address"] = wallet_address;
        body["signature"] = signature;

        auto result = DoLogin("/auth/wallet/login", body.dump());

        if (result.success) {
            FetchUserProfile();
            SetState(AuthState::Authenticated);
            SaveSession();
        } else {
            SetState(AuthState::Offline);
        }

        return result;
    });
}

std::future<std::string> AuthManager::RequestWalletNonce(const std::string& wallet_address) {
    return std::async(std::launch::async, [this, wallet_address]() -> std::string {
        std::string base_url;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            base_url = api_base_url_;
        }

        try {
            // Parse URL
            std::string host = base_url;
            std::string port = "80";
            bool use_ssl = false;

            if (host.find("https://") == 0) {
                host = host.substr(8);
                port = "443";
                use_ssl = true;
            } else if (host.find("http://") == 0) {
                host = host.substr(7);
            }

            // Extract port if present
            auto colon_pos = host.find(':');
            auto slash_pos = host.find('/');
            if (colon_pos != std::string::npos) {
                if (slash_pos != std::string::npos) {
                    port = host.substr(colon_pos + 1, slash_pos - colon_pos - 1);
                    host = host.substr(0, colon_pos);
                } else {
                    port = host.substr(colon_pos + 1);
                    host = host.substr(0, colon_pos);
                }
            } else if (slash_pos != std::string::npos) {
                host = host.substr(0, slash_pos);
            }

            // Create client
            httplib::Client client(host + ":" + port);
            client.set_connection_timeout(10);
            client.set_read_timeout(30);

            json body;
            body["wallet_address"] = wallet_address;

            auto res = client.Post("/api/auth/wallet/nonce", body.dump(), "application/json");

            if (res && res->status == 200) {
                auto j = json::parse(res->body);
                return j.value("nonce", "");
            } else {
                spdlog::error("Failed to request nonce: {}", res ? res->status : 0);
                return "";
            }
        } catch (const std::exception& e) {
            spdlog::error("Exception requesting nonce: {}", e.what());
            return "";
        }
    });
}

AuthResult AuthManager::DoLogin(const std::string& endpoint, const std::string& json_body) {
    AuthResult result;

    std::string base_url;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        base_url = api_base_url_;
    }

    try {
        // Parse URL - extract scheme, host, port, and base path
        std::string scheme = "http";
        std::string host = base_url;
        int port = 80;
        std::string base_path;
        bool use_ssl = false;

        // Extract scheme
        if (host.find("https://") == 0) {
            scheme = "https";
            host = host.substr(8);
            port = 443;
            use_ssl = true;
        } else if (host.find("http://") == 0) {
            host = host.substr(7);
            // Warn about insecure connection for non-localhost
            if (host.find("127.0.0.1") != 0 && host.find("localhost") != 0) {
                spdlog::warn("WARNING: Using unencrypted HTTP connection. Password may be visible in network traffic!");
            }
        }

        // Extract path (e.g., "/api" from "localhost:8080/api")
        auto slash_pos = host.find('/');
        if (slash_pos != std::string::npos) {
            base_path = host.substr(slash_pos);
            host = host.substr(0, slash_pos);
        }

        // Extract port if present
        auto colon_pos = host.find(':');
        if (colon_pos != std::string::npos) {
            port = std::stoi(host.substr(colon_pos + 1));
            host = host.substr(0, colon_pos);
        }

        spdlog::info("Connecting to {}://{}:{}{} (SSL: {})", scheme, host, port, base_path, use_ssl ? "yes" : "no");

        // Construct full path: base_path + endpoint (e.g., "/api" + "/auth/login")
        std::string full_path = base_path + endpoint;

        // Log request (mask password in logs)
        spdlog::info("POST {}", full_path);

        httplib::Result res;

        if (use_ssl) {
            // Use SSL client for HTTPS
            httplib::SSLClient ssl_client(host, port);
            ssl_client.set_connection_timeout(10);
            ssl_client.set_read_timeout(30);
            ssl_client.enable_server_certificate_verification(true);
            res = ssl_client.Post(full_path.c_str(), json_body, "application/json");
        } else {
            // Use regular client for HTTP
            httplib::Client client(host, port);
            client.set_connection_timeout(10);
            client.set_read_timeout(30);
            res = client.Post(full_path.c_str(), json_body, "application/json");
        }

        if (!res) {
            // Get detailed error from httplib
            auto err = res.error();
            std::string err_detail;
            switch (err) {
                case httplib::Error::Connection: err_detail = "Connection failed"; break;
                case httplib::Error::BindIPAddress: err_detail = "Bind IP address failed"; break;
                case httplib::Error::Read: err_detail = "Read error"; break;
                case httplib::Error::Write: err_detail = "Write error"; break;
                case httplib::Error::ExceedRedirectCount: err_detail = "Exceed redirect count"; break;
                case httplib::Error::Canceled: err_detail = "Canceled"; break;
                case httplib::Error::SSLConnection: err_detail = "SSL connection failed"; break;
                case httplib::Error::SSLLoadingCerts: err_detail = "SSL loading certs failed"; break;
                case httplib::Error::SSLServerVerification: err_detail = "SSL server verification failed"; break;
                case httplib::Error::UnsupportedMultipartBoundaryChars: err_detail = "Unsupported multipart boundary"; break;
                case httplib::Error::Compression: err_detail = "Compression error"; break;
                case httplib::Error::ConnectionTimeout: err_detail = "Connection timeout"; break;
                default: err_detail = "Unknown error"; break;
            }
            result.error = "Network error: " + err_detail;
            spdlog::error("Login failed: {} (httplib error: {})", result.error, static_cast<int>(err));
            return result;
        }

        if (res->status == 200) {
            auto j = json::parse(res->body);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                jwt_token_ = j.value("token", "");

                if (j.contains("user")) {
                    auto& user = j["user"];
                    user_info_.id = user.value("id", user.value("_id", ""));
                    user_info_.email = user.value("email", "");
                    user_info_.username = user.value("username", "");
                    user_info_.name = user.value("name", "");
                    user_info_.wallet_address = user.value("wallet_address", "");
                    user_info_.role = user.value("role", "user");
                }
            }

            result.success = true;
            result.user_info = user_info_;
            spdlog::info("Login successful for user: {}", user_info_.email);

        } else {
            // Handle error responses
            try {
                auto j = json::parse(res->body);
                result.error = j.value("error", j.value("message", "Unknown error"));
            } catch (...) {
                switch (res->status) {
                    case 401: result.error = "Invalid email or password"; break;
                    case 403: result.error = "Account suspended"; break;
                    case 404: result.error = "User not found"; break;
                    case 429: result.error = "Too many attempts. Please wait."; break;
                    default: result.error = "Server error (" + std::to_string(res->status) + ")";
                }
            }
            spdlog::error("Login failed: {} (HTTP {})", result.error, res->status);
        }

    } catch (const std::exception& e) {
        result.error = std::string("Connection error: ") + e.what();
        spdlog::error("Login exception: {}", e.what());
    }

    return result;
}

bool AuthManager::FetchUserProfile() {
    std::string jwt;
    std::string user_id;
    std::string base_url;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        jwt = jwt_token_;
        user_id = user_info_.id;
        base_url = api_base_url_;
    }

    if (jwt.empty() || user_id.empty()) {
        return false;
    }

    try {
        // Parse URL (same as DoLogin)
        std::string host = base_url;
        std::string port = "80";

        if (host.find("https://") == 0) {
            host = host.substr(8);
            port = "443";
        } else if (host.find("http://") == 0) {
            host = host.substr(7);
        }

        auto colon_pos = host.find(':');
        auto slash_pos = host.find('/');
        if (colon_pos != std::string::npos) {
            if (slash_pos != std::string::npos) {
                port = host.substr(colon_pos + 1, slash_pos - colon_pos - 1);
                host = host.substr(0, colon_pos);
            } else {
                port = host.substr(colon_pos + 1);
                host = host.substr(0, colon_pos);
            }
        } else if (slash_pos != std::string::npos) {
            host = host.substr(0, slash_pos);
        }

        httplib::Client client(host + ":" + port);
        client.set_connection_timeout(10);
        client.set_read_timeout(30);

        httplib::Headers headers = {
            {"Authorization", "Bearer " + jwt}
        };

        auto res = client.Get(("/api/users/" + user_id).c_str(), headers);

        if (res && res->status == 200) {
            auto j = json::parse(res->body);

            std::lock_guard<std::mutex> lock(mutex_);
            user_info_.email = j.value("email", user_info_.email);
            user_info_.username = j.value("username", user_info_.username);
            user_info_.name = j.value("name", user_info_.name);
            user_info_.wallet_address = j.value("wallet_address", user_info_.wallet_address);
            user_info_.role = j.value("role", user_info_.role);

            spdlog::info("User profile fetched: {} ({})", user_info_.username, user_info_.email);
            return true;
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to fetch user profile: {}", e.what());
    }

    return false;
}

void AuthManager::Logout() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jwt_token_.clear();
        node_token_.clear();
        user_info_ = UserInfo{};
    }

    ClearSavedSession();
    SetState(AuthState::Offline);
    spdlog::info("User logged out");
}

bool AuthManager::RefreshJwtToken() {
    // TODO: Implement token refresh
    // POST /api/auth/refresh with current token
    return false;
}

bool AuthManager::RefreshNodeToken() {
    // TODO: Implement node token refresh
    // POST /api/nodes/:id/token
    return false;
}

std::string AuthManager::GetTokenStoragePath() const {
    std::filesystem::path config_dir;

#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, path))) {
        config_dir = path;
        config_dir /= "CyxWiz";
        config_dir /= "server-node";
    }
#elif defined(__APPLE__)
    const char* home = std::getenv("HOME");
    if (home) {
        config_dir = home;
        config_dir /= "Library/Application Support/CyxWiz/server-node";
    }
#else
    const char* config_home = std::getenv("XDG_CONFIG_HOME");
    if (config_home) {
        config_dir = config_home;
    } else {
        const char* home = std::getenv("HOME");
        if (home) {
            config_dir = home;
            config_dir /= ".config";
        }
    }
    config_dir /= "cyxwiz/server-node";
#endif

    if (!config_dir.empty()) {
        std::filesystem::create_directories(config_dir);
        return (config_dir / "auth.json").string();
    }

    return "auth.json";
}

bool AuthManager::LoadSavedSession() {
    try {
        std::string path = GetTokenStoragePath();
        std::ifstream file(path);

        if (!file.is_open()) {
            return false;
        }

        json j;
        file >> j;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            jwt_token_ = j.value("jwt_token", "");
            node_token_ = j.value("node_token", "");
            user_info_.id = j.value("user_id", "");

            // Load node registration data
            node_id_ = j.value("node_id", "");
            node_api_key_ = j.value("node_api_key", "");
            node_registered_ = j.value("node_registered", false);
        }

        if (!jwt_token_.empty()) {
            // Try to fetch user profile to verify token is still valid
            if (FetchUserProfile()) {
                if (node_registered_) {
                    SetState(AuthState::Connected);
                    spdlog::info("Restored saved session for user: {} (node: {})", user_info_.email, node_id_);
                } else {
                    SetState(AuthState::Authenticated);
                    spdlog::info("Restored saved session for user: {}", user_info_.email);
                }
                return true;
            } else {
                // Token might be expired
                spdlog::warn("Saved session token is invalid or expired");
                ClearSavedSession();
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to load saved session: {}", e.what());
    }

    return false;
}

bool AuthManager::SaveSession() {
    try {
        std::string path = GetTokenStoragePath();

        json j;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            j["jwt_token"] = jwt_token_;
            j["node_token"] = node_token_;
            j["user_id"] = user_info_.id;
            j["remember_me"] = true;

            // Save node registration data
            j["node_id"] = node_id_;
            j["node_api_key"] = node_api_key_;
            j["node_registered"] = node_registered_;

            // Get current timestamp
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            char buf[32];
            std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&time));
            j["last_login"] = buf;
        }

        std::ofstream file(path);
        file << j.dump(2);
        file.close();

        spdlog::debug("Session saved to {}", path);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to save session: {}", e.what());
        return false;
    }
}

void AuthManager::ClearSavedSession() {
    try {
        std::string path = GetTokenStoragePath();
        std::filesystem::remove(path);
        spdlog::debug("Cleared saved session");
    } catch (const std::exception& e) {
        spdlog::error("Failed to clear saved session: {}", e.what());
    }
}

void AuthManager::OpenRegistrationPage() {
    const char* url = "https://cyxwiz.com/register";

#ifdef _WIN32
    ShellExecuteA(NULL, "open", url, NULL, NULL, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    std::string cmd = std::string("open ") + url;
    std::system(cmd.c_str());
#else
    std::string cmd = std::string("xdg-open ") + url;
    std::system(cmd.c_str());
#endif

    spdlog::info("Opened registration page in browser");
}

const char* AuthStateToString(AuthState state) {
    switch (state) {
        case AuthState::Offline:       return "Offline";
        case AuthState::LoggingIn:     return "Logging In";
        case AuthState::Authenticated: return "Authenticated";
        case AuthState::Registering:   return "Registering Node";
        case AuthState::Connecting:    return "Connecting";
        case AuthState::Connected:     return "Connected";
        default:                       return "Unknown";
    }
}

// ============================================================================
// Node Registration via REST API
// ============================================================================

HardwareSpecs AuthManager::DetectHardware() {
    HardwareSpecs specs;

    // Detect CPU
    int cpu_cores = 0;
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    cpu_cores = static_cast<int>(sysinfo.dwNumberOfProcessors);
#else
    cpu_cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#endif
    specs.cpu = std::to_string(cpu_cores) + " cores";

    // Detect RAM
    int64_t total_ram = 0;
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    total_ram = static_cast<int64_t>(memInfo.ullTotalPhys);
#elif defined(__APPLE__)
    // macOS: Use sysctl to get total physical memory
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
        total_ram = memsize;
    }
#else
    // Linux
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    total_ram = static_cast<int64_t>(memInfo.totalram) * memInfo.mem_unit;
#endif
    double ram_gb = total_ram / (1024.0 * 1024.0 * 1024.0);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << ram_gb << " GB";
    specs.ram = ss.str();

    // GPU detection using cyxwiz-backend (ArrayFire)
    specs.gpu = "None";
    try {
        auto devices = cyxwiz::Device::GetAvailableDevices();
        std::vector<std::string> gpu_names;

        for (const auto& dev : devices) {
            // Include CUDA, OpenCL, Metal, and Vulkan GPUs
            if (dev.type != cyxwiz::DeviceType::CPU) {
                gpu_names.push_back(dev.name);
                spdlog::debug("Found GPU: {} ({} MB)", dev.name, dev.memory_total / (1024 * 1024));
            }
        }

        if (!gpu_names.empty()) {
            // Join multiple GPU names with " + "
            specs.gpu = gpu_names[0];
            for (size_t i = 1; i < gpu_names.size(); ++i) {
                specs.gpu += " + " + gpu_names[i];
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to detect GPU via cyxwiz backend: {}", e.what());
    }

    // Storage detection placeholder
    specs.storage = "Unknown";

    // Detect IP address
#ifdef _WIN32
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        struct addrinfo hints, *result;
        ZeroMemory(&hints, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        if (getaddrinfo(hostname, NULL, &hints, &result) == 0) {
            struct sockaddr_in* addr = (struct sockaddr_in*)result->ai_addr;
            char ip_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &addr->sin_addr, ip_str, INET_ADDRSTRLEN);
            specs.ip_address = ip_str;
            freeaddrinfo(result);
        } else {
            specs.ip_address = "127.0.0.1";
        }
    } else {
        specs.ip_address = "127.0.0.1";
    }
#else
    struct ifaddrs *ifAddrStruct = nullptr;
    getifaddrs(&ifAddrStruct);
    specs.ip_address = "127.0.0.1";
    for (struct ifaddrs *ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
            char addressBuffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr, addressBuffer, INET_ADDRSTRLEN);
            std::string addr(addressBuffer);
            if (addr != "127.0.0.1" && ifa->ifa_name[0] != 'l') {
                specs.ip_address = addr;
                break;
            }
        }
    }
    if (ifAddrStruct) freeifaddrs(ifAddrStruct);
#endif

    return specs;
}

DetectedHardware AuthManager::DetectDetailedHardware() {
    DetectedHardware hw;

    // Use MetricsCollector for accurate hardware detection (DXGI + NVML/ADL/D3DKMT)
    core::MetricsCollector collector;
    collector.StartCollection(100);  // Start briefly to collect data
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // Wait for initial collection
    auto metrics = collector.GetCurrentMetrics();
    collector.StopCollection();

    // Populate GPUs from MetricsCollector (uses DXGI enumeration)
    for (const auto& gpu : metrics.gpus) {
        GpuInfo info;
        info.device_id = static_cast<uint32_t>(gpu.device_id);
        info.name = gpu.name;
        info.vendor = gpu.vendor;
        info.vram_mb = gpu.vram_total_bytes / (1024 * 1024);
        info.cuda_version = "";  // TODO: Get from NVML if available
        info.driver_version = "";
        hw.gpus.push_back(info);

        spdlog::info("Detected GPU {}: {} ({}) - {} MB VRAM",
                     info.device_id, info.name, info.vendor, info.vram_mb);
    }

    // RAM from metrics
    hw.ram_total_mb = metrics.ram_total_bytes / (1024 * 1024);

    // CPU detection
    CpuInfo cpu;
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    cpu.cores = sysInfo.dwNumberOfProcessors / 2;  // Physical cores estimate
    if (cpu.cores < 1) cpu.cores = 1;
    cpu.threads = sysInfo.dwNumberOfProcessors;

    // Get CPU name from registry
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char cpuName[256] = {0};
        DWORD bufSize = sizeof(cpuName);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                            (LPBYTE)cpuName, &bufSize) == ERROR_SUCCESS) {
            cpu.name = cpuName;
            // Trim whitespace
            size_t start = cpu.name.find_first_not_of(" ");
            size_t end = cpu.name.find_last_not_of(" ");
            if (start != std::string::npos) {
                cpu.name = cpu.name.substr(start, end - start + 1);
            }
        }

        DWORD mhz = 0;
        bufSize = sizeof(mhz);
        if (RegQueryValueExA(hKey, "~MHz", nullptr, nullptr,
                            (LPBYTE)&mhz, &bufSize) == ERROR_SUCCESS) {
            cpu.frequency_mhz = mhz;
        }
        RegCloseKey(hKey);
    }

    // OS info
    hw.os = "Windows";

    // Hostname
    char hostname[256] = {0};
    DWORD size = sizeof(hostname);
    GetComputerNameA(hostname, &size);
    hw.hostname = hostname;
#else
    cpu.cores = static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_ONLN)) / 2;
    if (cpu.cores < 1) cpu.cores = 1;
    cpu.threads = static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_ONLN));
    cpu.name = "Unknown CPU";
    cpu.frequency_mhz = 0;
    hw.os = "Linux";

    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname));
    hw.hostname = hostname;
#endif

    hw.cpus.push_back(cpu);
    spdlog::info("Detected CPU: {} ({} cores / {} threads)",
                 cpu.name, cpu.cores, cpu.threads);

    // IP address (reuse from legacy detection)
    auto legacy = DetectHardware();
    hw.ip_address = legacy.ip_address;

    spdlog::info("Detected RAM: {} MB, OS: {}, Hostname: {}",
                 hw.ram_total_mb, hw.os, hw.hostname);

    return hw;
}

std::future<NodeRegistrationResult> AuthManager::RegisterNodeWithApi(
    const std::string& node_name,
    const std::string& node_type)
{
    return std::async(std::launch::async, [this, node_name, node_type]() -> NodeRegistrationResult {
        NodeRegistrationResult result;

        std::string base_url;
        std::string jwt;
        std::string owner_id;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            base_url = api_base_url_;
            jwt = jwt_token_;
            owner_id = user_info_.id;
        }

        if (jwt.empty() || owner_id.empty()) {
            result.error = "Not authenticated. Please login first.";
            return result;
        }

        SetState(AuthState::Registering);

        // Detect detailed hardware (uses MetricsCollector for accurate GPU enumeration)
        auto hw = DetectDetailedHardware();

        try {
            // Parse URL
            std::string scheme = "http";
            std::string host = base_url;
            int port = 80;
            std::string base_path;
            bool use_ssl = false;

            if (host.find("https://") == 0) {
                scheme = "https";
                host = host.substr(8);
                port = 443;
                use_ssl = true;
            } else if (host.find("http://") == 0) {
                host = host.substr(7);
            }

            auto slash_pos = host.find('/');
            if (slash_pos != std::string::npos) {
                base_path = host.substr(slash_pos);
                host = host.substr(0, slash_pos);
            }

            auto colon_pos = host.find(':');
            if (colon_pos != std::string::npos) {
                port = std::stoi(host.substr(colon_pos + 1));
                host = host.substr(0, colon_pos);
            }

            // Build structured hardware JSON
            json gpus_json = json::array();
            for (const auto& gpu : hw.gpus) {
                gpus_json.push_back({
                    {"device_id", gpu.device_id},
                    {"name", gpu.name},
                    {"vendor", gpu.vendor},
                    {"vram_mb", gpu.vram_mb}
                });
            }

            json cpus_json = json::array();
            for (const auto& cpu : hw.cpus) {
                cpus_json.push_back({
                    {"name", cpu.name},
                    {"cores", cpu.cores},
                    {"threads", cpu.threads},
                    {"frequency_mhz", cpu.frequency_mhz}
                });
            }

            // Build request JSON with structured hardware data
            json body;
            body["owner_id"] = owner_id;
            body["hostname"] = hw.hostname;
            body["os"] = hw.os;
            body["ip_address"] = hw.ip_address;
            body["hardware"] = {
                {"gpus", gpus_json},
                {"cpus", cpus_json},
                {"ram_total_mb", hw.ram_total_mb}
            };

            // Include node_id if we have one from Central Server
            // This allows Web API to use the same ID as Central Server
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!node_id_.empty()) {
                    body["node_id"] = node_id_;
                    spdlog::info("Including existing node_id from Central Server: {}", node_id_);
                }
            
            }

            // Include legacy specs for backward compatibility
            std::string gpu_summary;
            for (size_t i = 0; i < hw.gpus.size(); i++) {
                if (i > 0) gpu_summary += " + ";
                gpu_summary += hw.gpus[i].name;
            }
            if (gpu_summary.empty()) gpu_summary = "None";

            std::string cpu_summary;
            if (!hw.cpus.empty()) {
                cpu_summary = hw.cpus[0].name;
            }

            body["specs"] = {
                {"cpu", cpu_summary},
                {"gpu", gpu_summary},
                {"ram", std::to_string(hw.ram_total_mb / 1024) + " GB"},
                {"storage", "Unknown"}
            };

            std::string full_path = base_path + "/machines/register";
            spdlog::info("Registering machine '{}' at {}", hw.hostname, full_path);
            spdlog::info("Hardware: {} GPUs, {} CPUs, {} MB RAM",
                         hw.gpus.size(), hw.cpus.size(), hw.ram_total_mb);

            httplib::Result res;

            if (use_ssl) {
                httplib::SSLClient ssl_client(host, port);
                ssl_client.set_connection_timeout(10);
                ssl_client.set_read_timeout(30);
                ssl_client.enable_server_certificate_verification(true);

                httplib::Headers headers = {
                    {"Authorization", "Bearer " + jwt},
                    {"Content-Type", "application/json"}
                };
                res = ssl_client.Post(full_path.c_str(), headers, body.dump(), "application/json");
            } else {
                httplib::Client client(host, port);
                client.set_connection_timeout(10);
                client.set_read_timeout(30);

                httplib::Headers headers = {
                    {"Authorization", "Bearer " + jwt},
                    {"Content-Type", "application/json"}
                };
                res = client.Post(full_path.c_str(), headers, body.dump(), "application/json");
            }

            if (!res) {
                result.error = "Network error: Failed to connect to API";
                spdlog::error("Node registration failed: {}", result.error);
                SetState(AuthState::Authenticated);
                return result;
            }

            if (res->status == 200) {
                auto j = json::parse(res->body);

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    node_api_key_ = j.value("api_key", "");
                    if (j.contains("node") && j["node"].contains("node_id")) {
                        node_id_ = j["node"]["node_id"].get<std::string>();
                    }
                    node_registered_ = true;
                }

                result.success = true;
                result.node_id = node_id_;
                result.api_key = node_api_key_;

                spdlog::info("Node registered successfully!");
                spdlog::info("  Node ID: {}", result.node_id);
                spdlog::info("  API Key: {}...", result.api_key.substr(0, 20));

                SetState(AuthState::Connected);
                SaveSession();

            } else {
                try {
                    auto j = json::parse(res->body);
                    result.error = j.value("error", "Unknown error");
                } catch (...) {
                    result.error = "Server error (" + std::to_string(res->status) + ")";
                }
                spdlog::error("Node registration failed: {} (HTTP {})", result.error, res->status);
                SetState(AuthState::Authenticated);
            }

        } catch (const std::exception& e) {
            result.error = std::string("Exception: ") + e.what();
            spdlog::error("Node registration exception: {}", e.what());
            SetState(AuthState::Authenticated);
        }

        return result;
    });
}

bool AuthManager::SendHeartbeatToApi() {
    std::string base_url;
    std::string node_id;
    std::string api_key;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!node_registered_) {
            return false;
        }
        base_url = api_base_url_;
        node_id = node_id_;
        api_key = node_api_key_;
    }

    try {
        // Parse URL
        std::string host = base_url;
        int port = 80;
        std::string base_path;
        bool use_ssl = false;

        if (host.find("https://") == 0) {
            host = host.substr(8);
            port = 443;
            use_ssl = true;
        } else if (host.find("http://") == 0) {
            host = host.substr(7);
        }

        auto slash_pos = host.find('/');
        if (slash_pos != std::string::npos) {
            base_path = host.substr(slash_pos);
            host = host.substr(0, slash_pos);
        }

        auto colon_pos = host.find(':');
        if (colon_pos != std::string::npos) {
            port = std::stoi(host.substr(colon_pos + 1));
            host = host.substr(0, colon_pos);
        }

        // Detect hardware specs (including GPU)
        auto specs = DetectHardware();

        // Build request
        json body;
        body["api_key"] = api_key;
        body["current_load"] = 0;  // TODO: Get actual CPU load
        body["ip_address"] = specs.ip_address;

        // Include specs in heartbeat so GPU info gets updated
        body["specs"] = {
            {"cpu", specs.cpu},
            {"gpu", specs.gpu},
            {"ram", specs.ram},
            {"storage", specs.storage}
        };

        std::string full_path = base_path + "/machines/" + node_id_ + "/heartbeat";

        httplib::Result res;

        if (use_ssl) {
            httplib::SSLClient ssl_client(host, port);
            ssl_client.set_connection_timeout(5);
            ssl_client.set_read_timeout(10);
            res = ssl_client.Post(full_path.c_str(), body.dump(), "application/json");
        } else {
            httplib::Client client(host, port);
            client.set_connection_timeout(5);
            client.set_read_timeout(10);
            res = client.Post(full_path.c_str(), body.dump(), "application/json");
        }

        if (res && res->status == 200) {
            spdlog::debug("Heartbeat sent successfully");
            return true;
        } else {
            spdlog::warn("Heartbeat failed: HTTP {}", res ? res->status : 0);
            return false;
        }

    } catch (const std::exception& e) {
        spdlog::error("Heartbeat exception: {}", e.what());
        return false;
    }
}

std::string AuthManager::GetNodeId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return node_id_;
}

std::string AuthManager::GetNodeApiKey() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return node_api_key_;
}

bool AuthManager::IsNodeRegistered() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return node_registered_;
}


bool AuthManager::SyncNodeIdWithWebApi(const std::string& central_server_node_id) {
    // If the node_id is the same as what we already have, no need to sync
    std::string current_node_id;
    std::string api_key;
    std::string base_url;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_node_id = node_id_;
        api_key = node_api_key_;
        base_url = api_base_url_;
    }
    
    if (central_server_node_id.empty()) {
        spdlog::warn("SyncNodeIdWithWebApi: No Central Server node_id provided");
        return false;
    }
    
    if (current_node_id == central_server_node_id) {
        spdlog::info("SyncNodeIdWithWebApi: node_id already matches Central Server");
        return true;
    }
    
    if (api_key.empty()) {
        spdlog::error("SyncNodeIdWithWebApi: No API key available");
        return false;
    }
    
    spdlog::info("SyncNodeIdWithWebApi: Updating node_id from {} to {}", 
                 current_node_id, central_server_node_id);
    
    try {
        // Parse the base URL to get host and port
        std::string host;
        int port = 80;
        std::string protocol = "http";
        
        std::string url = base_url;
        if (url.find("https://") == 0) {
            protocol = "https";
            url = url.substr(8);
            port = 443;
        } else if (url.find("http://") == 0) {
            url = url.substr(7);
        }
        
        // Remove /api suffix if present
        if (url.find("/api") != std::string::npos) {
            url = url.substr(0, url.find("/api"));
        }
        
        // Parse host:port
        size_t colon_pos = url.find(':');
        if (colon_pos != std::string::npos) {
            host = url.substr(0, colon_pos);
            port = std::stoi(url.substr(colon_pos + 1));
        } else {
            host = url;
        }
        
        // Create HTTP client
        std::unique_ptr<httplib::Client> cli;
        if (protocol == "https") {
            cli = std::make_unique<httplib::Client>(host, port);
            cli->enable_server_certificate_verification(false);
        } else {
            cli = std::make_unique<httplib::Client>(host, port);
        }
        cli->set_connection_timeout(10);
        cli->set_read_timeout(10);
        
        // Build the endpoint: PUT /api/machines/{old_node_id}/node-id
        std::string endpoint = "/api/machines/" + current_node_id + "/node-id";
        
        // Build request body
        json body;
        body["api_key"] = api_key;
        body["new_node_id"] = central_server_node_id;
        
        auto res = cli->Put(endpoint.c_str(), body.dump(), "application/json");
        
        if (!res) {
            spdlog::error("SyncNodeIdWithWebApi: Request failed (no response)");
            return false;
        }
        
        if (res->status == 200) {
            // Update local node_id
            {
                std::lock_guard<std::mutex> lock(mutex_);
                node_id_ = central_server_node_id;
            }
            
            // Save updated session
            SaveSession();
            
            spdlog::info("SyncNodeIdWithWebApi: Successfully synced node_id to {}", 
                         central_server_node_id);
            return true;
        } else {
            spdlog::error("SyncNodeIdWithWebApi: Failed with status {} - {}", 
                         res->status, res->body);
            return false;
        }
        
    } catch (const std::exception& e) {
        spdlog::error("SyncNodeIdWithWebApi exception: {}", e.what());
        return false;
    }
}

} // namespace cyxwiz::servernode::auth
