# Security Model

This document describes the security architecture, authentication mechanisms, and best practices for the CyxWiz platform.

## Security Overview

CyxWiz implements multiple layers of security to protect:

1. **User Data** - Models, datasets, credentials
2. **Compute Resources** - Node hardware from malicious code
3. **Financial Assets** - Token balances and transactions
4. **Network Integrity** - Communication between components

## Authentication

### JWT Token Authentication

All gRPC communications between components use JWT (JSON Web Tokens) for authentication.

**Token Structure:**
```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "node_id or user_id",
    "iss": "cyxwiz-central-server",
    "iat": 1702000000,
    "exp": 1702086400,
    "scope": ["job:execute", "metrics:report"],
    "node_id": "uuid-...",
    "session_id": "uuid-..."
  },
  "signature": "..."
}
```

**Token Types:**

| Token Type | Lifetime | Purpose |
|------------|----------|---------|
| Session Token | 24 hours | Node registration session |
| Job Token | Job duration | Authorization for specific job |
| P2P Token | 1 hour | Direct Engine-Node communication |
| Refresh Token | 7 days | Obtaining new session tokens |

### Authentication Flow

```
1. Node Registration
   +------------------+                    +------------------+
   |   Server Node    |                    | Central Server   |
   +------------------+                    +------------------+
           |                                       |
           | RegisterNodeRequest                   |
           | { public_key, hardware_info }         |
           |-------------------------------------->|
           |                                       |
           |                    Validate request   |
           |                    Generate node_id   |
           |                    Sign JWT           |
           |                                       |
           | RegisterNodeResponse                  |
           | { node_id, session_token }            |
           |<--------------------------------------|
           |                                       |

2. Authenticated Requests
           |                                       |
           | HeartbeatRequest                      |
           | Header: Authorization: Bearer <JWT>   |
           |-------------------------------------->|
           |                                       |
           |                    Validate JWT       |
           |                    Check expiration   |
           |                    Verify signature   |
           |                                       |
```

### Key Management

**Central Server:**
- JWT signing key stored in environment variable or secrets manager
- Key rotation recommended every 30 days
- Old keys remain valid until token expiration

**Server Node:**
- Generates RSA key pair on first run
- Public key sent during registration
- Private key stored locally (encrypted)
- Used for signing completion proofs

## Authorization

### Role-Based Access Control

| Role | Permissions |
|------|-------------|
| **User** | Submit jobs, view own jobs, manage wallet |
| **Node** | Accept jobs, report progress, claim payments |
| **Admin** | Manage nodes, view all jobs, configure system |

### Resource-Level Permissions

```rust
// Example authorization check in Central Server
fn authorize_job_access(user_id: &str, job: &Job) -> Result<(), Error> {
    // Owner can always access
    if job.user_id == user_id {
        return Ok(());
    }

    // Assigned node can access
    if let Some(node_id) = &job.assigned_node {
        if is_node_request() && get_node_id() == node_id {
            return Ok(());
        }
    }

    Err(Error::Unauthorized)
}
```

## Encryption

### Transport Layer Security (TLS)

All network communications use TLS 1.3:

**gRPC Connections:**
```
Engine <--TLS--> Central Server <--TLS--> Server Node
```

**Certificate Management:**
- Self-signed certificates for development
- Let's Encrypt for production
- Mutual TLS (mTLS) for node authentication

### Data Encryption

**At Rest:**
| Data Type | Encryption |
|-----------|------------|
| Database | AES-256 (PostgreSQL TDE) |
| Model Files | AES-256-GCM |
| Wallet Keys | Argon2 + AES-256 |
| Logs | Not encrypted (no sensitive data) |

**In Transit:**
| Protocol | Encryption |
|----------|------------|
| gRPC | TLS 1.3 |
| REST API | HTTPS |
| IPFS | TLS + Content Hash Verification |
| Solana RPC | HTTPS |

## Sandboxing

### Docker-Based Isolation

Server Nodes execute untrusted training code in Docker containers:

```dockerfile
# Sandboxed training container
FROM python:3.10-slim

# Non-root user
RUN useradd -m -s /bin/bash trainer
USER trainer

# Limited resources
# (set via docker run --memory, --cpus)

# No network access during training
# (set via docker run --network none)

# Read-only filesystem
# (except /tmp and /output)

ENTRYPOINT ["python", "/job/train.py"]
```

**Container Restrictions:**

| Resource | Limit |
|----------|-------|
| Memory | Job-specific (e.g., 16GB) |
| CPU | Job-specific (e.g., 4 cores) |
| Disk | 50GB per job |
| Network | Disabled during training |
| Capabilities | All dropped |
| Syscalls | Seccomp whitelist |

### Resource Limits

```cpp
// Server Node job executor
struct JobLimits {
    size_t max_memory_bytes;      // 16GB default
    int max_cpu_cores;            // 4 default
    int max_gpu_memory_mb;        // 8GB default
    int max_execution_seconds;    // 3600 default (1 hour)
    size_t max_output_bytes;      // 1GB default
};

bool EnforceJobLimits(const JobLimits& limits) {
    // Set cgroup limits
    SetMemoryLimit(limits.max_memory_bytes);
    SetCpuLimit(limits.max_cpu_cores);

    // Start watchdog timer
    StartTimeoutWatchdog(limits.max_execution_seconds);

    return true;
}
```

## Blockchain Security

### Escrow System

Payment security through smart contracts:

```
1. Job Submission
   User ---> CreateEscrow(amount) ---> Solana
                                          |
                                    Escrow Account
                                    (Locked funds)

2. Job Completion
   Node completes job
   Central Server verifies
   Central Server ---> ReleaseEscrow ---> Solana
                                             |
                                       Node Wallet
                                       (Payment received)

3. Dispute Resolution
   If disputed:
   - Funds remain in escrow
   - Arbitration process
   - Manual resolution
```

### Transaction Signing

All Solana transactions require multi-step verification:

```rust
async fn release_escrow(job: &Job) -> Result<Signature, Error> {
    // 1. Verify job completion
    verify_job_completion(job)?;

    // 2. Verify node signature
    verify_node_signature(&job.completion_proof)?;

    // 3. Build transaction
    let tx = Transaction::new_signed_with_payer(
        &[release_escrow_instruction(
            &job.escrow_account,
            &job.node_wallet,
            job.payment_amount,
        )],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );

    // 4. Send and confirm
    let signature = client.send_and_confirm_transaction(&tx).await?;

    Ok(signature)
}
```

## Input Validation

### Model Definition Validation

```cpp
bool ValidateModelDefinition(const std::string& json) {
    // Parse JSON
    auto model = nlohmann::json::parse(json);

    // Check required fields
    if (!model.contains("layers")) return false;

    // Validate layer types
    for (auto& layer : model["layers"]) {
        if (!IsValidLayerType(layer["type"])) return false;
    }

    // Check for circular dependencies
    if (HasCircularDependency(model)) return false;

    // Validate parameter ranges
    for (auto& layer : model["layers"]) {
        if (!ValidateLayerParams(layer)) return false;
    }

    return true;
}
```

### gRPC Request Validation

```rust
impl JobService for JobServiceImpl {
    async fn submit_job(
        &self,
        request: Request<SubmitJobRequest>,
    ) -> Result<Response<SubmitJobResponse>, Status> {
        let req = request.into_inner();

        // Validate job type
        if req.config.job_type == JobType::Unknown as i32 {
            return Err(Status::invalid_argument("Invalid job type"));
        }

        // Validate payment amount
        if req.config.payment_amount < MIN_PAYMENT {
            return Err(Status::invalid_argument("Payment too low"));
        }

        // Validate dataset URI
        if !is_valid_ipfs_uri(&req.config.dataset_uri) {
            return Err(Status::invalid_argument("Invalid dataset URI"));
        }

        // Proceed with validated request
        // ...
    }
}
```

## Audit Logging

### Security Events

All security-relevant events are logged:

```rust
#[derive(Debug, Serialize)]
struct SecurityEvent {
    timestamp: DateTime<Utc>,
    event_type: SecurityEventType,
    actor_id: String,
    actor_type: ActorType,
    resource: String,
    action: String,
    outcome: Outcome,
    ip_address: Option<String>,
    details: serde_json::Value,
}

enum SecurityEventType {
    Authentication,
    Authorization,
    DataAccess,
    Configuration,
    Payment,
}
```

**Logged Events:**
| Event | Details Captured |
|-------|------------------|
| Login attempt | User ID, IP, success/failure, method |
| Token generation | Token type, subject, expiration |
| Job submission | User ID, job config, payment |
| Node registration | Node ID, hardware info, location |
| Payment transaction | Amount, from, to, tx hash |
| Configuration change | Changed fields, old/new values |

### Log Storage

```
Security logs stored:
- Local: /var/log/cyxwiz/security.log (7 days)
- Remote: Elasticsearch (90 days)
- Archive: S3 (indefinite, encrypted)

Log format: JSON Lines
Rotation: Daily
Compression: gzip
```

## Threat Model

### Identified Threats

| Threat | Mitigation |
|--------|------------|
| **Malicious Node** | Sandboxed execution, verification |
| **Man-in-the-Middle** | TLS encryption, certificate pinning |
| **Token Theft** | Short expiration, refresh rotation |
| **Replay Attack** | Nonce in requests, timestamp validation |
| **DoS Attack** | Rate limiting, connection limits |
| **Data Exfiltration** | Network isolation, output limits |
| **Payment Fraud** | Escrow, multi-sig, verification |

### Security Boundaries

```
+----------------------------------------------------------+
|                     Trusted Zone                          |
|                                                           |
|  +------------------+        +------------------+         |
|  | Central Server   |<------>|    Database      |         |
|  +------------------+        +------------------+         |
|           ^                                               |
+-----------|-------------------------------------------------+
            | TLS + JWT
+-----------v-------------------------------------------------+
|                    Semi-Trusted Zone                        |
|                                                             |
|  +------------------+        +------------------+           |
|  |  Server Node     |        |  Engine Client   |           |
|  |  (Authenticated) |        |  (Authenticated) |           |
|  +------------------+        +------------------+           |
|           |                                                 |
+-----------|-------------------------------------------------+
            | Docker Sandbox
+-----------v-------------------------------------------------+
|                    Untrusted Zone                           |
|                                                             |
|  +------------------+                                       |
|  | Training Code    |  <- No network, limited resources    |
|  | (User-provided)  |                                       |
|  +------------------+                                       |
+------------------------------------------------------------+
```

## Best Practices

### For Operators

1. **Rotate JWT signing keys** every 30 days
2. **Enable TLS** for all connections
3. **Monitor security logs** daily
4. **Update dependencies** weekly
5. **Backup encryption keys** securely
6. **Use strong passwords** (16+ characters)
7. **Enable 2FA** where supported

### For Node Operators

1. **Keep Docker updated** (latest stable)
2. **Use dedicated hardware** for nodes
3. **Monitor resource usage** for anomalies
4. **Secure wallet private keys** offline
5. **Enable firewall** (only required ports)

### For Users

1. **Protect API keys** - don't commit to Git
2. **Verify model hashes** after download
3. **Review node reputation** before submitting
4. **Use escrow** for large payments
5. **Keep wallet backups** secure

---

**Next**: [Architecture](architecture.md) | [Developer Guide](../developer/index.md)
