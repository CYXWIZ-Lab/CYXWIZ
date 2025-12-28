# CyxWiz Server Node Connection Architecture

## Overview

This document describes how Server Nodes authenticate and connect to the CyxWiz Central Server. Authentication is handled through the CyxWiz Web API, which acts as the identity provider for the entire platform.

## Connection Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Server Node   │     │   CyxWiz API    │     │  Central Server │
│      (GUI)      │     │   (Web Auth)    │     │   (Rust gRPC)   │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │  1. Login Request     │                       │
         │  (email/password)     │                       │
         ├──────────────────────>│                       │
         │                       │                       │
         │  2. JWT Token +       │                       │
         │     User Info         │                       │
         │<──────────────────────┤                       │
         │                       │                       │
         │                       │  3. Register Node     │
         │                       │  (user_id, node_id)   │
         │                       ├──────────────────────>│
         │                       │                       │
         │                       │  4. Node Token        │
         │                       │<──────────────────────┤
         │                       │                       │
         │  5. Node Token        │                       │
         │<──────────────────────┤                       │
         │                       │                       │
         │  6. gRPC Connect      │                       │
         │  (with Node Token)    │                       │
         ├──────────────────────────────────────────────>│
         │                       │                       │
         │  7. Connection        │                       │
         │     Established       │                       │
         │<──────────────────────────────────────────────┤
         │                       │                       │
```

## Authentication Methods

### 1. Email/Password Login

Standard authentication for users with CyxWiz accounts.

```
POST /api/auth/login
{
  "email": "user@example.com",
  "password": "your-password"
}
```

### 2. Wallet Login (Solana)

For users who prefer wallet-based authentication:

1. Request nonce: `POST /api/auth/wallet/nonce`
2. Sign message with wallet
3. Submit signature: `POST /api/auth/wallet/login`

### 3. Registration

New users are directed to the website to complete registration:
- URL: `https://cyxwiz.com/register`
- Opens in system browser
- After registration, user returns to Server Node GUI to login

## Token Types

### JWT Token (User Auth)
- Issued by: CyxWiz API
- Purpose: Authenticate user identity
- Contains: user_id, email, role, wallet_address
- Expiry: 24 hours (configurable)
- Used for: API calls, initial node registration

### Node Token (Node Auth)
- Issued by: Central Server (via API)
- Purpose: Authenticate Server Node for gRPC connections
- Contains: user_id, node_id, capabilities, rate_limits
- Expiry: 7 days (auto-refresh)
- Used for: gRPC metadata, job assignment authorization

## Server Node States

```
┌──────────────┐
│   OFFLINE    │  (Initial state, no auth)
└──────┬───────┘
       │ User clicks "Login"
       v
┌──────────────┐
│ LOGGING_IN   │  (Awaiting API response)
└──────┬───────┘
       │ JWT received
       v
┌──────────────┐
│ AUTHENTICATED│  (User logged in, not connected)
└──────┬───────┘
       │ Node registration initiated
       v
┌──────────────┐
│ REGISTERING  │  (Getting node token)
└──────┬───────┘
       │ Node token received
       v
┌──────────────┐
│ CONNECTING   │  (gRPC connection in progress)
└──────┬───────┘
       │ gRPC channel established
       v
┌──────────────┐
│  CONNECTED   │  (Ready to accept jobs)
└──────────────┘
```

## Data Retrieved on Login

After successful authentication, the Server Node retrieves:

| Field           | Source      | Purpose                          |
|-----------------|-------------|----------------------------------|
| user_id         | JWT Token   | Link node to user account        |
| email           | User API    | Display in GUI                   |
| username        | User API    | Display in GUI                   |
| name            | User API    | Display in GUI                   |
| wallet_address  | User API    | Earnings payout destination      |
| role            | JWT Token   | Permission level (user/pro/admin)|

## Wallet Address Importance

The wallet address is critical for:
1. **Earnings Payout**: Where compute earnings are sent
2. **Staking**: Node reputation/priority staking
3. **Identity Verification**: Optional additional auth factor

If no wallet is linked:
- Node can still connect and run jobs
- Earnings accumulate in platform balance
- User prompted to link wallet for payouts

## Implementation Components

### Server Node GUI (`cyxwiz-server-node/src/gui/`)

```cpp
// auth_manager.h
class AuthManager {
public:
    // Login methods
    std::future<AuthResult> LoginWithEmail(const std::string& email, const std::string& password);
    std::future<AuthResult> LoginWithWallet(const std::string& wallet_address, const std::string& signature);

    // Token management
    bool IsAuthenticated() const;
    std::string GetJwtToken() const;
    std::string GetNodeToken() const;
    UserInfo GetUserInfo() const;

    // Session
    void Logout();
    bool RefreshTokens();

    // Callbacks
    void SetOnAuthStateChanged(std::function<void(AuthState)> callback);

private:
    std::string jwt_token_;
    std::string node_token_;
    UserInfo user_info_;
    AuthState state_ = AuthState::Offline;
};

// user_info.h
struct UserInfo {
    std::string id;
    std::string email;
    std::string username;
    std::string name;
    std::string wallet_address;
    std::string role;
};
```

### Login Panel (`login_panel.h`)

```cpp
class LoginPanel {
public:
    void Render();

private:
    void RenderLoginForm();
    void RenderLoggedInState();
    void RenderWalletLogin();

    // Form state
    char email_[256] = "";
    char password_[256] = "";
    bool show_password_ = false;
    bool remember_me_ = false;

    // Login state
    bool is_logging_in_ = false;
    std::string error_message_;

    // Auth manager reference
    AuthManager* auth_manager_;
};
```

### HTTP Client (`http_client.h`)

```cpp
class HttpClient {
public:
    struct Response {
        int status_code;
        std::string body;
        std::map<std::string, std::string> headers;
    };

    std::future<Response> Get(const std::string& url, const Headers& headers = {});
    std::future<Response> Post(const std::string& url, const std::string& body, const Headers& headers = {});
    std::future<Response> Put(const std::string& url, const std::string& body, const Headers& headers = {});
    std::future<Response> Delete(const std::string& url, const Headers& headers = {});

private:
    // Using cpp-httplib or libcurl
};
```

## API Endpoints Used

| Endpoint               | Method | When Used                        |
|------------------------|--------|----------------------------------|
| /api/auth/login        | POST   | Email/password login             |
| /api/auth/wallet/nonce | POST   | Request wallet sign message      |
| /api/auth/wallet/login | POST   | Submit wallet signature          |
| /api/users/:id         | GET    | Fetch user profile after login   |
| /api/nodes/register    | POST   | Register node with Central Server|
| /api/nodes/:id/token   | POST   | Refresh node token               |

## Configuration Storage

Credentials and tokens stored in:
- **Windows**: `%APPDATA%/CyxWiz/server-node/auth.json`
- **Linux**: `~/.config/cyxwiz/server-node/auth.json`
- **macOS**: `~/Library/Application Support/CyxWiz/server-node/auth.json`

```json
{
  "jwt_token": "eyJhbGciOiJIUzI1NiIs...",
  "node_token": "node_token_here...",
  "user_id": "674f1234abcd5678ef901234",
  "remember_me": true,
  "last_login": "2025-12-08T10:30:00Z"
}
```

**Security Notes**:
- Tokens encrypted at rest using platform keychain when available
- Passwords never stored locally
- JWT token auto-cleared on expiry

## Error Handling

| Error Code | Meaning                    | User Action                    |
|------------|----------------------------|--------------------------------|
| 401        | Invalid credentials        | Check email/password           |
| 403        | Account suspended          | Contact support                |
| 404        | User not found             | Register first                 |
| 429        | Rate limited               | Wait and retry                 |
| 500        | Server error               | Retry later                    |
| NETWORK    | Connection failed          | Check internet connection      |

## Future Enhancements

1. **OAuth Integration**: Google, GitHub login
2. **Hardware Key Support**: YubiKey, Ledger
3. **Multi-Factor Auth**: TOTP, SMS
4. **Session Management**: View/revoke active sessions
5. **Team Accounts**: Organization-level node management

## Security Considerations

1. **HTTPS Only**: All API calls over TLS
2. **Token Rotation**: Regular refresh of node tokens
3. **Rate Limiting**: Prevent brute force attacks
4. **IP Binding**: Optional node token IP restriction
5. **Audit Logging**: Track all auth events

## Design Decisions (Resolved)

| Decision | Resolution |
|----------|------------|
| Offline Mode | **Yes** - Support local-only training without Central Server |
| Remember Me | **Always** - Tokens persist indefinitely |
| API Base URL | `http://localhost:8080/api` (dev), configurable for prod |
| Node Registration | Via `/api/nodes/register` endpoint |
| Central Server Notification | Internal API call from CyxWiz API |

## Open Questions

1. How to handle token expiry during long-running jobs?
2. Should wallet login auto-create account or require registration?
3. Multi-node per account limits?
