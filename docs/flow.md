# CyxWiz Authentication Flow

## Complete Authentication Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              CyxWiz Authentication System                             │
└──────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │   MongoDB    │
                                    │   (Users)    │
                                    └──────┬───────┘
                                           │
         ┌─────────────────┬───────────────┼───────────────┬─────────────────┐
         │                 │               │               │                 │
         ▼                 ▼               ▼               ▼                 │
  ┌─────────────┐   ┌─────────────┐  ┌───────────┐  ┌───────────────┐        │
  │   Website   │   │  Admin App  │  │ Rust API  │  │Central Server │        │
  │  (NextAuth) │   │  (NextAuth) │  │  Server   │  │    (Rust)     │        │
  │ AUTH_SECRET │   │ADMIN_SECRET │  │JWT_SECRET │  │ Validates All │        │
  └──────┬──────┘   └──────┬──────┘  └─────┬─────┘  └───────┬───────┘        │
         │                 │               │                │                │
         ▼                 ▼               ▼                │                │
    ┌─────────┐       ┌─────────┐    ┌──────────┐           │                │
    │ Browser │       │ Browser │    │ Engine / │           │                │
    │(Cookie) │       │(Cookie) │    │Server Node│          │                │
    └─────────┘       └─────────┘    │(stores JWT)│         │                │
                                     └─────┬──────┘         │                │
                                           │                │                │
                                           ▼                ▼                │
                                     ┌────────────────────────────┐          │
                                     │   gRPC with Auth Header    │          │
                                     │  "Bearer <JWT_SECRET jwt>" │──────────┘
                                     └────────────────────────────┘
```

---

## Flow 1: Web Browser Login (Website/NextAuth)

```
┌──────────┐         ┌──────────────┐         ┌─────────┐
│  Browser │         │   Website    │         │ MongoDB │
└────┬─────┘         │  (NextAuth)  │         └────┬────┘
     │               └──────┬───────┘              │
     │                      │                      │
     │  1. POST /api/auth/signin                   │
     │      {email, password}                      │
     │ ────────────────────►│                      │
     │                      │                      │
     │                      │  2. Find user        │
     │                      │ ────────────────────►│
     │                      │                      │
     │                      │  3. User doc         │
     │                      │ ◄────────────────────│
     │                      │                      │
     │                      │  4. Verify password  │
     │                      │     (argon2)         │
     │                      │                      │
     │  5. Set-Cookie:      │                      │
     │     next-auth.session│                      │
     │     (HttpOnly JWT)   │                      │
     │ ◄────────────────────│                      │
     │                      │                      │
```

**JWT Payload (Website):**
```json
{
  "sub": "user_mongo_id",
  "email": "user@example.com",
  "username": "username",
  "cyxWalletAddress": "Abc123...xyz",
  "iat": 1234567890,
  "exp": 1234567890
}
```

---

## Flow 2: Desktop Client Login (Engine/Server Node)

**IMPLEMENTED: Rust API Server at `cyxwiz_web/apps/api`**

```
┌──────────┐         ┌──────────────┐         ┌─────────┐
│  Engine  │         │  Rust API    │         │ MongoDB │
│  or      │         │   Server     │         └────┬────┘
│Server Node│        │ :8080        │              │
└────┬─────┘         └──────┬───────┘              │
     │                      │                      │
     │  1. POST /api/auth/login                    │
     │     {email, password}                       │
     │ ────────────────────►│                      │
     │                      │                      │
     │                      │  2. Find user        │
     │                      │ ────────────────────►│
     │                      │                      │
     │                      │  3. User doc         │
     │                      │ ◄────────────────────│
     │                      │                      │
     │                      │  4. Verify password  │
     │                      │     (argon2)         │
     │                      │                      │
     │                      │  5. Generate JWT     │
     │                      │     (JWT_SECRET)     │
     │                      │                      │
     │  6. Response:        │                      │
     │     { token: "jwt",  │                      │
     │       user: {        │                      │
     │         id, email,   │                      │
     │         username,    │                      │
     │         cyxWallet,   │                      │
     │         externalWallet│                     │
     │       }              │                      │
     │     }                │                      │
     │ ◄────────────────────│                      │
     │                      │                      │
     │  7. Store token      │                      │
     │     locally          │                      │
     │                      │                      │
```

**JWT Payload (Rust API → Desktop Clients):**
```json
{
  "sub": "user_mongo_id",
  "email": "user@example.com",
  "role": "user",
  "iat": 1234567890,
  "exp": 1234567890
}
```

Note: Wallet address is NOT in the JWT. Central Server looks it up from MongoDB using `sub` (user_id).

**Alternative: Wallet Login**
```
┌──────────┐         ┌──────────────┐
│  Engine  │         │  Rust API    │
└────┬─────┘         └──────┬───────┘
     │                      │
     │  1. POST /api/auth/wallet/nonce
     │     {wallet_address} │
     │ ────────────────────►│
     │                      │
     │  2. Response:        │
     │     { nonce, message }
     │ ◄────────────────────│
     │                      │
     │  3. Sign message     │
     │     with wallet      │
     │                      │
     │  4. POST /api/auth/wallet/login
     │     {wallet_address, │
     │      signature, nonce}
     │ ────────────────────►│
     │                      │
     │  5. Response:        │
     │     { token, user }  │
     │ ◄────────────────────│
     │                      │
```

---

## Flow 3: Engine → Central Server (Job Submission)

```
┌──────────┐         ┌───────────────┐         ┌─────────┐
│  Engine  │         │Central Server │         │ MongoDB │
│ (C++)    │         │   (Rust)      │         └────┬────┘
└────┬─────┘         └───────┬───────┘              │
     │                       │                      │
     │  1. gRPC: SubmitJob   │                      │
     │     metadata: {       │                      │
     │       authorization:  │                      │
     │       "Bearer <JWT>"  │                      │
     │     }                 │                      │
     │ ─────────────────────►│                      │
     │                       │                      │
     │                       │  2. Validate JWT     │
     │                       │     (JWT_SECRET)     │
     │                       │                      │
     │                       │  3. Extract user_id  │
     │                       │     from claims.sub  │
     │                       │                      │
     │                       │  4. Lookup user      │
     │                       │     + wallet address │
     │                       │ ────────────────────►│
     │                       │                      │
     │                       │  5. User + wallet    │
     │                       │ ◄────────────────────│
     │                       │                      │
     │                       │  6. Check balance    │
     │                       │                      │
     │                       │  7. Create job       │
     │                       │     (PostgreSQL)     │
     │                       │                      │
     │                       │  8. Create escrow    │
     │                       │     (Solana)         │
     │                       │                      │
     │  9. Response:         │                      │
     │     { job_id: "..." } │                      │
     │ ◄─────────────────────│                      │
     │                       │                      │
```

---

## Flow 4: Server Node → Central Server (Registration)

```
┌─────────────┐      ┌───────────────┐         ┌─────────┐
│ Server Node │      │Central Server │         │ MongoDB │
│   (C++)     │      │   (Rust)      │         └────┬────┘
└──────┬──────┘      └───────┬───────┘              │
       │                     │                      │
       │  1. gRPC: RegisterNode                     │
       │     metadata: {     │                      │
       │       authorization:│                      │
       │       "Bearer <JWT>"│                      │
       │     }               │                      │
       │     body: {         │                      │
       │       node_info,    │                      │
       │       devices       │                      │
       │     }               │                      │
       │ ───────────────────►│                      │
       │                     │                      │
       │                     │  2. Validate JWT     │
       │                     │     (JWT_SECRET)     │
       │                     │                      │
       │                     │  3. Extract user_id  │
       │                     │     from claims.sub  │
       │                     │                      │
       │                     │  4. Lookup wallet    │
       │                     │ ────────────────────►│
       │                     │                      │
       │                     │  5. Wallet address   │
       │                     │ ◄────────────────────│
       │                     │                      │
       │                     │  6. Create node      │
       │                     │     with user_id     │
       │                     │     (PostgreSQL)     │
       │                     │                      │
       │  7. Response:       │                      │
       │     { node_id,      │                      │
       │       session_token }                      │
       │ ◄───────────────────│                      │
       │                     │                      │
       │                     │                      │
       │  8. Heartbeat loop  │                      │
       │     (every 10s)     │                      │
       │     with auth       │                      │
       │ ───────────────────►│                      │
       │                     │                      │
```

---

## Flow 5: Admin App → Central Server (REST API)

```
┌───────────┐        ┌──────────────┐        ┌───────────────┐
│ Admin App │        │  Admin App   │        │Central Server │
│ (Browser) │        │  (NextAuth)  │        │  REST API     │
└─────┬─────┘        └──────┬───────┘        └───────┬───────┘
      │                     │                        │
      │  1. Login           │                        │
      │     (Google OAuth)  │                        │
      │ ───────────────────►│                        │
      │                     │                        │
      │  2. Set-Cookie      │                        │
      │     (admin JWT)     │                        │
      │ ◄───────────────────│                        │
      │                     │                        │
      │  3. Request page    │                        │
      │ ───────────────────►│                        │
      │                     │                        │
      │                     │  4. Server component   │
      │                     │     generates JWT      │
      │                     │     (ADMIN_AUTH_SECRET)│
      │                     │                        │
      │                     │  5. GET /api/v1/nodes  │
      │                     │     Authorization:     │
      │                     │     Bearer <admin_jwt> │
      │                     │ ──────────────────────►│
      │                     │                        │
      │                     │                        │  6. Validate
      │                     │                        │     admin JWT
      │                     │                        │
      │                     │                        │  7. Check role
      │                     │                        │     permissions
      │                     │                        │
      │                     │  8. Response: nodes[]  │
      │                     │ ◄──────────────────────│
      │                     │                        │
      │  9. Rendered HTML   │                        │
      │ ◄───────────────────│                        │
      │                     │                        │
```

---

## Flow 6: P2P Connection (Engine → Server Node)

```
┌──────────┐     ┌───────────────┐     ┌─────────────┐
│  Engine  │     │Central Server │     │ Server Node │
└────┬─────┘     └───────┬───────┘     └──────┬──────┘
     │                   │                    │
     │  (Job assigned)   │                    │
     │                   │                    │
     │                   │  1. Generate P2P   │
     │                   │     JWT token      │
     │                   │     (P2P_SECRET)   │
     │                   │                    │
     │  2. GetJobStatus  │                    │
     │     returns:      │                    │
     │     { p2p_token,  │                    │
     │       node_addr } │                    │
     │ ◄─────────────────│                    │
     │                   │                    │
     │                   │                    │
     │  3. Direct connection to node          │
     │     Authorization: Bearer <p2p_token>  │
     │ ──────────────────────────────────────►│
     │                                        │
     │                   │                    │  4. Validate
     │                   │                    │     P2P token
     │                   │                    │
     │                   │                    │  5. Extract
     │                   │                    │     job_id,
     │                   │                    │     user_id
     │                                        │
     │  6. Accept connection                  │
     │ ◄──────────────────────────────────────│
     │                                        │
     │  7. Stream training data               │
     │ ◄─────────────────────────────────────►│
     │                                        │
```

**P2P JWT Payload:**
```json
{
  "sub": "user_mongo_id",
  "job_id": "job_123",
  "node_id": "node_456",
  "type": "p2p",
  "iat": 1234567890,
  "exp": 1234567890
}
```

---

## JWT Secrets Configuration

| Component | Secret | Purpose |
|-----------|--------|---------|
| Website | `AUTH_SECRET` | NextAuth session JWT signing |
| Admin App | `ADMIN_AUTH_SECRET` | Admin NextAuth JWT signing |
| Rust API | `JWT_SECRET` | Desktop client login JWT signing |
| Central Server | `jwt.user_secret` | Validate user JWTs (**must match Rust API's JWT_SECRET**) |
| Central Server | `jwt.admin_secret` | Validate admin JWTs (same as ADMIN_AUTH_SECRET) |
| Central Server | `jwt.p2p_secret` | Generate/validate P2P tokens |
| Server Node | (receives P2P) | Validates P2P tokens from Central Server |

**IMPORTANT:** The `JWT_SECRET` (Rust API) and `jwt.user_secret` (Central Server) MUST be the same value!

---

## Implementation Status

| Component | Login | JWT Storage | gRPC Auth | Status |
|-----------|-------|-------------|-----------|--------|
| Website | NextAuth | HttpOnly Cookie | N/A | COMPLETE |
| Admin App | NextAuth | HttpOnly Cookie | REST calls | COMPLETE |
| **Rust API** | `/api/auth/login` | Returns JWT | N/A | **COMPLETE** |
| Central Server | N/A | Validates tokens | Interceptor | COMPLETE |
| Engine | Rust API | Local storage | Header metadata | COMPLETE |
| Server Node | Rust API | Local storage | Header metadata | COMPLETE |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CyxWiz Services                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌──────────────────┐          │
│  │   Website   │     │  Admin App  │     │    Rust API      │          │
│  │  (NextAuth) │     │  (NextAuth) │     │   Server :8080   │          │
│  │    :3000    │     │    :3001    │     │                  │          │
│  └──────┬──────┘     └──────┬──────┘     └────────┬─────────┘          │
│         │                   │                     │                     │
│         │ Browser           │ Browser             │ Desktop Clients     │
│         │ Sessions          │ Sessions            │ (Engine/Server Node)│
│         │                   │                     │                     │
│         └─────────┬─────────┴─────────────────────┘                     │
│                   │                                                     │
│                   ▼                                                     │
│           ┌──────────────┐                                              │
│           │   MongoDB    │ ◄── Shared user database                     │
│           └──────────────┘                                              │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                    Central Server :50051                       │     │
│  │                                                                │     │
│  │  - Validates JWTs from Rust API (jwt.user_secret)             │     │
│  │  - Validates Admin JWTs (jwt.admin_secret)                    │     │
│  │  - Generates P2P tokens (jwt.p2p_secret)                      │     │
│  │  - gRPC: JobService, NodeService, JobStatusService            │     │
│  │  - REST: /api/v1/* (admin dashboard)                          │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Environment Configuration

### Rust API Server (.env)
```bash
PORT=8080
MONGODB_URI=mongodb://localhost:27017/cyxwiz
JWT_SECRET=your-shared-jwt-secret-32-chars-min
CENTRAL_SERVER_URL=http://localhost:50051
```

### Central Server (config.toml)
```toml
[jwt]
# MUST match Rust API's JWT_SECRET
user_secret = "your-shared-jwt-secret-32-chars-min"
admin_secret = "your-admin-jwt-secret-32-chars-min"
p2p_secret = "your-p2p-jwt-secret-32-chars-min"
p2p_token_expiration_seconds = 3600

[mongo]
url = "mongodb://localhost:27017"
database = "cyxwiz"
user_collection = "users"
```

### Desktop Clients (Engine/Server Node)
```
CYXWIZ_API_URL=http://localhost:8080
CYXWIZ_CENTRAL_SERVER=localhost:50051
```

---

## Rust API Endpoints Reference

**Base URL:** `http://localhost:8080`

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Email/password login → returns JWT |
| POST | `/api/auth/wallet/nonce` | Get nonce for wallet signature |
| POST | `/api/auth/wallet/login` | Wallet signature login → returns JWT |

### Users
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/users` | Create new user |
| GET | `/api/users/:id` | Get user by ID |
| GET | `/api/users/:id/models` | Get user's models |
| PUT | `/api/users/:id/wallet` | Update external wallet |
| DELETE | `/api/users/:id/wallet` | Remove external wallet |

### Machines (Server Nodes)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/machines/register` | Register new machine |
| GET | `/api/machines/:id` | Get machine by ID |
| DELETE | `/api/machines/:id` | Delete machine |
| POST | `/api/machines/:id/heartbeat` | Send heartbeat |
| PUT | `/api/machines/:id/allocations` | Update resource allocations |
| PUT | `/api/machines/:id/node-id` | Update Central Server node ID |
| GET | `/api/users/:user_id/machines` | Get user's machines |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List all models |
| POST | `/api/models` | Create new model |
| GET | `/api/models/id/:id` | Get model by ID |
| PUT | `/api/models/id/:id` | Update model |
| DELETE | `/api/models/id/:id` | Delete model |
| GET | `/api/models/*slug` | Get model by slug |

---

## Design Decisions

### Why wallet_address is NOT in the JWT

The JWT from the Rust API contains only:
- `sub` (user_id)
- `email`
- `role`
- `exp`, `iat`

**Benefits of looking up wallet from MongoDB:**
1. **Single source of truth** - MongoDB is the authoritative source for wallet data
2. **Wallet changes don't require new JWT** - User can update wallet without re-login
3. **Smaller token size** - Less data transmitted with every request
4. **Security** - Wallet address can't be forged in JWT

**The Central Server:**
1. Validates JWT signature using `jwt.user_secret`
2. Extracts `user_id` from `claims.sub`
3. Queries MongoDB for user's wallet address
4. Uses wallet for escrow/payment operations
