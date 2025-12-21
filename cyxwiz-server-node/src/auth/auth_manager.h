// auth_manager.h - Authentication manager for CyxWiz API
#pragma once

#include <string>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <vector>
#include <cstdint>

namespace cyxwiz::servernode::auth {

// Authentication states
enum class AuthState {
    Offline,         // Not authenticated
    LoggingIn,       // Login request in progress
    Authenticated,   // JWT received, not connected to Central Server
    Registering,     // Getting node token from Central Server
    Connecting,      // gRPC connection in progress
    Connected        // Fully connected to Central Server
};

// User information retrieved after login
struct UserInfo {
    std::string id;
    std::string email;
    std::string username;
    std::string name;
    std::string wallet_address;
    std::string role;  // "user", "pro", "admin"
};

// Login result
struct AuthResult {
    bool success = false;
    std::string error;
    UserInfo user_info;
};

// Node registration result
struct NodeRegistrationResult {
    bool success = false;
    std::string node_id;
    std::string api_key;
    std::string error;
};

// Hardware specifications for node registration (legacy format)
struct HardwareSpecs {
    std::string cpu;
    std::string gpu;
    std::string ram;
    std::string storage;
    std::string ip_address;
};

// Detailed GPU information for structured hardware data
struct GpuInfo {
    uint32_t device_id;
    std::string name;
    std::string vendor;  // "NVIDIA", "AMD", "Intel"
    uint64_t vram_mb;
    std::string cuda_version;
    std::string driver_version;
};

// Detailed CPU information
struct CpuInfo {
    std::string name;
    uint32_t cores;
    uint32_t threads;
    uint32_t frequency_mhz;
};

// Structured hardware data for proper API registration
struct DetectedHardware {
    std::vector<CpuInfo> cpus;
    std::vector<GpuInfo> gpus;
    uint64_t ram_total_mb;
    std::string os;
    std::string hostname;
    std::string ip_address;
};

// Authentication manager - singleton for handling CyxWiz API auth
class AuthManager {
public:
    static AuthManager& Instance();

    // Prevent copying
    AuthManager(const AuthManager&) = delete;
    AuthManager& operator=(const AuthManager&) = delete;

    // Configuration
    void SetApiBaseUrl(const std::string& url);
    std::string GetApiBaseUrl() const;

    // Login methods
    std::future<AuthResult> LoginWithEmail(const std::string& email, const std::string& password);
    std::future<AuthResult> LoginWithWallet(const std::string& wallet_address, const std::string& signature);

    // Request nonce for wallet login
    std::future<std::string> RequestWalletNonce(const std::string& wallet_address);

    // Logout
    void Logout();

    // Token management
    bool IsAuthenticated() const;
    std::string GetJwtToken() const;
    std::string GetNodeToken() const;
    UserInfo GetUserInfo() const;
    AuthState GetState() const;

    // Token refresh
    bool RefreshJwtToken();
    bool RefreshNodeToken();

    // Node registration via REST API
    std::future<NodeRegistrationResult> RegisterNodeWithApi(
        const std::string& node_name,
        const std::string& node_type = "server");
    bool SendHeartbeatToApi(bool is_online = true);
    std::string GetNodeId() const;
    std::string GetNodeApiKey() const;
    bool IsNodeRegistered() const;

    // Sync node ID from Central Server to Web API
    bool SyncNodeIdWithWebApi(const std::string& node_id);

    // Hardware detection
    static HardwareSpecs DetectHardware();  // Legacy simple format
    static DetectedHardware DetectDetailedHardware();  // New structured format

    // Persistence
    bool LoadSavedSession();
    bool SaveSession();
    void ClearSavedSession();

    // Callbacks
    using AuthStateCallback = std::function<void(AuthState)>;
    void SetOnAuthStateChanged(AuthStateCallback callback);

    // Open registration page in browser
    static void OpenRegistrationPage();

private:
    AuthManager();
    ~AuthManager();

    void SetState(AuthState state);

    // HTTP request helpers
    AuthResult DoLogin(const std::string& endpoint, const std::string& json_body);
    bool FetchUserProfile();

    // Token storage path
    std::string GetTokenStoragePath() const;

    // State
    mutable std::mutex mutex_;
    AuthState state_ = AuthState::Offline;
    std::string api_base_url_ = "http://127.0.0.1:3002/api";
    std::string jwt_token_;
    std::string node_token_;
    UserInfo user_info_;
    AuthStateCallback on_state_changed_;

    // Node registration state (REST API)
    std::string node_id_;
    std::string node_api_key_;
    bool node_registered_ = false;
};

// Helper to get auth state as string
const char* AuthStateToString(AuthState state);

} // namespace cyxwiz::servernode::auth
