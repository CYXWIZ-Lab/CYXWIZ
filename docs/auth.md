# CyxWiz Authentication & Identification Implementation Plan

## Overview

Implement end-to-end authentication across the CyxWiz ecosystem, connecting Engine, Server Node, Central Server, Website, and Admin App.

**Architecture:**
- **Website** (MongoDB + NextAuth): Source of truth for user data
- **Admin App** (MongoDB + NextAuth): Admin authentication with RBAC
- **Central Server** (PostgreSQL + MongoDB read): Validates tokens, looks up users
- **Engine/Server Node** (C++): Authenticate via Website API

---

## Auth Flow Diagrams

### Flow 1: User Login (Engine/Server Node)
```
Engine → POST /api/auth/login (Website) → JWT Token
Engine stores JWT locally
Engine → gRPC SubmitJob (Central Server, JWT in header)
Central Server → Validates JWT → Looks up user in MongoDB → Processes request
```

### Flow 2: Server Node Registration
```
Server Node → Login to Website → JWT Token
Server Node → gRPC RegisterNode (Central Server, JWT in header)
Central Server → Validates JWT → Gets user_id from JWT
Central Server → Creates node in PostgreSQL with user_id link
Central Server → Node can now receive jobs, payments go to user's wallet
```

### Flow 3: Admin API Access
```
Admin App → NextAuth login → Admin JWT (ADMIN_AUTH_SECRET)
Admin App → REST API /api/v1/* (Central Server, Admin JWT in header)
Central Server → Validates Admin JWT → Checks admin role → Returns data
```

### Flow 4: P2P Connection (Engine → Server Node)
```
Central Server assigns job → Generates P2P JWT (job_id, node_id, user_id)
Central Server → Returns P2P JWT to Engine
Engine → Connects to Server Node with P2P JWT
Server Node → Validates P2P JWT → Accepts connection
```

---

## Phase 1: Central Server - MongoDB Integration

### 1.1 Add MongoDB Dependency

**File**: `cyxwiz-central-server/Cargo.toml`
```toml
mongodb = { version = "2.8", features = ["bson-uuid-1"] }
```

### 1.2 MongoDB Configuration

**File**: `cyxwiz-central-server/src/config.rs`

Add `MongoConfig` struct:
```rust
pub struct MongoConfig {
    pub url: String,              // mongodb://localhost:27017
    pub database: String,         // cyxwiz
    pub user_collection: String,  // users
}
```

### 1.3 Create MongoDB Client

**File to CREATE**: `cyxwiz-central-server/src/database/mongo.rs`

```rust
// Read-only MongoDB client for user lookups
pub struct MongoClient { ... }

impl MongoClient {
    pub async fn new(config: &MongoConfig) -> Result<Self>;
    pub async fn get_user_by_id(&self, user_id: &str) -> Result<Option<MongoUser>>;
    pub async fn get_user_by_wallet(&self, wallet: &str) -> Result<Option<MongoUser>>;
    pub async fn get_user_balance(&self, user_id: &str) -> Result<f64>;
}

// User struct matching Website's MongoDB schema
pub struct MongoUser {
    pub id: ObjectId,
    pub email: String,
    pub username: String,
    pub wallet_address: Option<String>,
    pub cyx_wallet: Option<CyxWallet>,
    pub wallet: WalletBalance,
    pub reputation: Reputation,
}
```

---

## Phase 2: Central Server - Auth Middleware

### 2.1 gRPC Auth Interceptor

**File to CREATE**: `cyxwiz-central-server/src/auth/middleware.rs`

```rust
pub struct AuthContext {
    pub user_id: String,
    pub wallet_address: String,
    pub email: String,
    pub is_admin: bool,
    pub admin_role: Option<String>,  // super_admin, admin, moderator, support
}

pub struct AuthInterceptor {
    jwt_manager: Arc<JWTManager>,
    mongo_client: Arc<MongoClient>,
}

impl AuthInterceptor {
    /// Extract and validate JWT from gRPC metadata
    pub async fn authenticate(&self, metadata: &MetadataMap) -> Result<AuthContext, Status>;

    /// Validate admin token (different secret)
    pub async fn authenticate_admin(&self, metadata: &MetadataMap) -> Result<AuthContext, Status>;
}
```

### 2.2 REST Auth Middleware

**File to CREATE**: `cyxwiz-central-server/src/api/rest/auth_middleware.rs`

```rust
use axum::{middleware::Next, http::Request, response::Response};

pub async fn require_auth<B>(
    State(state): State<AppState>,
    request: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode>;

pub async fn require_admin<B>(
    State(state): State<AppState>,
    request: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode>;
```

### 2.3 Update JWT Manager

**File**: `cyxwiz-central-server/src/auth/jwt.rs`

Add methods:
```rust
impl JWTManager {
    /// Validate Website-issued JWT (AUTH_SECRET)
    pub fn verify_website_token(&self, token: &str) -> Result<UserAuthClaims>;

    /// Validate Admin-issued JWT (ADMIN_AUTH_SECRET)
    pub fn verify_admin_token(&self, token: &str, admin_secret: &str) -> Result<AdminAuthClaims>;
}

pub struct AdminAuthClaims {
    pub sub: String,      // admin user ID
    pub email: String,
    pub role: String,     // super_admin, admin, moderator, support
    pub permissions: Vec<String>,
}
```

---

## Phase 3: Central Server - Service Updates

### 3.1 Update Node Service

**File**: `cyxwiz-central-server/src/api/grpc/node_service.rs`

Changes:
1. Require authentication for `RegisterNode`
2. Extract `user_id` from JWT
3. Look up user's wallet from MongoDB
4. Store `user_id` on node record

```rust
async fn register_node(&self, request: Request<RegisterNodeRequest>) -> Result<...> {
    // 1. Validate JWT
    let auth_context = self.auth_interceptor.authenticate(request.metadata()).await?;

    // 2. Get user's wallet from MongoDB
    let user = self.mongo_client.get_user_by_id(&auth_context.user_id).await?;
    let wallet = user.cyx_wallet.map(|w| w.public_key).or(user.wallet_address);

    // 3. Create node with user_id link
    let node = Node {
        user_id: Some(auth_context.user_id),
        wallet_address: wallet.unwrap_or_default(),
        ...
    };
}
```

### 3.2 Update Job Service

**File**: `cyxwiz-central-server/src/api/grpc/job_service.rs`

Changes:
1. Require authentication for `SubmitJob`
2. Get user's wallet and balance from MongoDB
3. Verify sufficient balance before creating escrow

```rust
async fn submit_job(&self, request: Request<SubmitJobRequest>) -> Result<...> {
    // 1. Validate JWT
    let auth_context = self.auth_interceptor.authenticate(request.metadata()).await?;

    // 2. Get balance from MongoDB
    let balance = self.mongo_client.get_user_balance(&auth_context.user_id).await?;

    // 3. Check balance >= estimated cost
    if balance < estimated_cost {
        return Err(Status::failed_precondition("Insufficient balance"));
    }

    // 4. Create escrow with user's wallet
    ...
}
```

### 3.3 Update REST API

**File**: `cyxwiz-central-server/src/api/rest/mod.rs`

Apply auth middleware to protected routes:
```rust
let admin_routes = Router::new()
    .route("/api/v1/nodes/live", get(nodes::list_live_nodes))
    .route("/api/v1/scheduler/queue", get(scheduler::get_queue))
    .route("/api/v1/blockchain/wallet", get(blockchain::get_wallet))
    .layer(middleware::from_fn_with_state(state.clone(), require_admin));
```

---

## Phase 4: Database Schema Updates

### 4.1 Add user_id to nodes table

**Migration**: `migrations/XXXXXX_add_user_id_to_nodes.sql`
```sql
ALTER TABLE nodes ADD COLUMN user_id TEXT;
ALTER TABLE nodes ADD COLUMN device_id TEXT UNIQUE;
CREATE INDEX idx_nodes_user_id ON nodes(user_id);
```

### 4.2 Update Node Model

**File**: `cyxwiz-central-server/src/database/models.rs`
```rust
pub struct Node {
    // ... existing fields ...
    pub user_id: Option<String>,   // MongoDB user ObjectId
    pub device_id: Option<String>, // Hardware-based unique ID
}
```

---

## Phase 5: Website - Login Endpoint for Desktop Clients

### 5.1 Create Desktop Login Endpoint

**File to CREATE**: `cyxwiz_web/apps/web/src/app/api/auth/login/route.ts`

```typescript
import { NextResponse } from "next/server";
import argon2 from "argon2";
import jwt from "jsonwebtoken";

export async function POST(request: Request) {
  const { email, password } = await request.json();

  // Validate credentials
  const user = await User.findOne({ email }).select("+password");
  if (!user || !await argon2.verify(user.password, password)) {
    return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
  }

  // Generate JWT with same secret as NextAuth
  const token = jwt.sign(
    { sub: user._id.toString(), email: user.email, username: user.username },
    process.env.AUTH_SECRET!,
    { expiresIn: "7d" }
  );

  return NextResponse.json({ token, user: { id, email, username, cyxWallet } });
}
```

### 5.2 Wallet Nonce Endpoint (for wallet-based login)

**File to CREATE**: `cyxwiz_web/apps/web/src/app/api/auth/wallet/nonce/route.ts`

```typescript
export async function POST(request: Request) {
  const { wallet_address } = await request.json();

  // Generate random nonce
  const nonce = crypto.randomBytes(32).toString("hex");
  const message = `Sign this message to authenticate with CyxWiz: ${nonce}`;

  // Store nonce temporarily (Redis or DB)
  await storeNonce(wallet_address, nonce);

  return NextResponse.json({ nonce, message });
}
```

---

## Phase 6: Admin App - Pass JWT to Central Server

### 6.1 Update Central Server Client

**File**: `cyxwiz_web/apps/admin/src/lib/central-server.ts`

```typescript
import { auth } from "@/lib/auth";
import { encode } from "next-auth/jwt";

export async function fetchCentralServer<T>(endpoint: string): Promise<T> {
  const session = await auth();
  if (!session?.user) throw new Error("Not authenticated");

  // Generate JWT for Central Server
  const token = await encode({
    token: {
      id: session.user.id,
      email: session.user.email,
      role: session.user.role,
      permissions: session.user.permissions,
    },
    secret: process.env.ADMIN_AUTH_SECRET!,
  });

  const response = await fetch(`${CENTRAL_SERVER_URL}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
    },
    cache: "no-store",
  });

  if (!response.ok) throw new Error(`Central Server error: ${response.statusText}`);
  return response.json();
}
```

---

## Phase 7: Engine/Server Node - JWT Integration

### 7.1 Engine - Add JWT to gRPC Calls

**File**: `cyxwiz-engine/src/network/grpc_client.cpp`

```cpp
// Add authorization header to all gRPC calls
grpc::ClientContext context;
std::string auth_header = "Bearer " + AuthClient::GetInstance().GetJwtToken();
context.AddMetadata("authorization", auth_header);

stub->SubmitJob(&context, request, &response);
```

### 7.2 Server Node - Add JWT to RegisterNode

**File**: `cyxwiz-server-node/src/network/node_client.cpp`

```cpp
// In RegisterNode call
request.set_authentication_token(AuthManager::GetInstance().GetJwtToken());
```

### 7.3 Server Node - Validate P2P JWT

**File to CREATE**: `cyxwiz-server-node/src/auth/p2p_validator.cpp`

```cpp
class P2PValidator {
public:
    bool ValidateToken(const std::string& token, const std::string& expected_job_id);
    std::string GetUserId(const std::string& token);
    std::string GetNodeId(const std::string& token);
};
```

---

## Environment Configuration

### Shared Secrets

All components must use the same JWT secrets:

```bash
# Website (.env)
AUTH_SECRET=your-shared-jwt-secret-32-chars-min

# Admin App (.env)
ADMIN_AUTH_SECRET=your-admin-jwt-secret-32-chars-min

# Central Server (config.toml)
[jwt]
user_secret = "your-shared-jwt-secret-32-chars-min"      # Same as AUTH_SECRET
admin_secret = "your-admin-jwt-secret-32-chars-min"      # Same as ADMIN_AUTH_SECRET
p2p_secret = "your-p2p-jwt-secret-32-chars-min"          # For P2P tokens

[mongo]
url = "mongodb://readonlyuser:password@localhost:27017/cyxwiz"
database = "cyxwiz"
user_collection = "users"
```

---

## Implementation Order

### Sprint 1: Central Server Foundation
1. Add MongoDB dependency to Cargo.toml
2. Create `src/database/mongo.rs` with MongoUser struct and queries
3. Update `src/config.rs` with MongoConfig
4. Create database migration for user_id on nodes

### Sprint 2: Auth Middleware
1. Create `src/auth/middleware.rs` with AuthInterceptor
2. Update `src/auth/jwt.rs` with verify_website_token and verify_admin_token
3. Create `src/api/rest/auth_middleware.rs`
4. Apply middleware to REST routes

### Sprint 3: Service Updates
1. Update NodeService for authenticated registration
2. Update JobService for authenticated job submission
3. Add user_id to Node model and queries
4. Test gRPC auth flow

### Sprint 4: Website Endpoints
1. Create `/api/auth/login` for desktop clients
2. Create `/api/auth/wallet/nonce` and `/api/auth/wallet/verify`
3. Test Engine → Website → JWT flow

### Sprint 5: Admin Integration
1. Update Admin's `central-server.ts` to pass JWT
2. Test Admin → Central Server authenticated calls

### Sprint 6: Client Updates
1. Update Engine gRPC client to include JWT
2. Update Server Node RegisterNode to include JWT
3. Create P2P token validator in Server Node
4. End-to-end testing

---

## Critical Files

### Central Server (Rust)
- `src/database/mongo.rs` - NEW: MongoDB client
- `src/auth/middleware.rs` - NEW: Auth interceptor
- `src/auth/jwt.rs` - MODIFY: Add token validation methods
- `src/api/grpc/node_service.rs` - MODIFY: Require auth
- `src/api/grpc/job_service.rs` - MODIFY: Require auth
- `src/api/rest/mod.rs` - MODIFY: Apply auth middleware
- `src/config.rs` - MODIFY: Add MongoConfig
- `src/database/models.rs` - MODIFY: Add user_id to Node

### Website
- `apps/web/src/app/api/auth/login/route.ts` - NEW
- `apps/web/src/app/api/auth/wallet/nonce/route.ts` - NEW
- `apps/web/src/app/api/auth/wallet/verify/route.ts` - NEW

### Admin App
- `apps/admin/src/lib/central-server.ts` - MODIFY: Pass JWT

### Engine (C++)
- `src/network/grpc_client.cpp` - MODIFY: Add JWT header
- `src/auth/auth_client.cpp` - Verify existing implementation

### Server Node (C++)
- `src/network/node_client.cpp` - MODIFY: Add JWT to RegisterNode
- `src/auth/p2p_validator.cpp` - NEW: Validate P2P tokens

---

## Summary

| Component | New Files | Modified Files |
|-----------|-----------|----------------|
| Central Server | 3 | 6 |
| Website | 3 | 0 |
| Admin App | 0 | 1 |
| Engine | 0 | 2 |
| Server Node | 1 | 1 |
| **Total** | **7** | **10** |
