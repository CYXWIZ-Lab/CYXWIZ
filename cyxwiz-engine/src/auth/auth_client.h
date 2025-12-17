// auth_client.h - Authentication client for CyxWiz Engine
#pragma once

#include <string>
#include <functional>
#include <future>
#include <memory>
#include <mutex>

namespace cyxwiz::auth {

// Authentication states
enum class AuthState {
    Offline,         // Not authenticated
    LoggingIn,       // Login request in progress
    Authenticated    // JWT received
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

// Authentication client - singleton for handling CyxWiz API auth
class AuthClient {
public:
    static AuthClient& Instance();

    // Prevent copying
    AuthClient(const AuthClient&) = delete;
    AuthClient& operator=(const AuthClient&) = delete;

    // Configuration
    void SetApiBaseUrl(const std::string& url);
    std::string GetApiBaseUrl() const;

    // Login methods
    std::future<AuthResult> LoginWithEmail(const std::string& email, const std::string& password);

    // Logout
    void Logout();

    // Token management
    bool IsAuthenticated() const;
    std::string GetJwtToken() const;
    UserInfo GetUserInfo() const;
    AuthState GetState() const;

    // Token refresh
    bool RefreshJwtToken();

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
    AuthClient();
    ~AuthClient();

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
    UserInfo user_info_;
    AuthStateCallback on_state_changed_;
};

// Helper to get auth state as string
const char* AuthStateToString(AuthState state);

} // namespace cyxwiz::auth
