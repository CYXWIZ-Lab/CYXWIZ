# Central Server Configuration

Complete configuration reference for the CyxWiz Central Server.

## Configuration File

The Central Server uses TOML configuration files. Default location: `config.toml` in the working directory.

### Full Configuration Example

```toml
# CyxWiz Central Server Configuration

[server]
# gRPC server settings
grpc_address = "0.0.0.0:50051"
# REST API settings
rest_address = "0.0.0.0:8080"
# Maximum concurrent connections
max_connections = 1000
# Request timeout in seconds
request_timeout = 60
# Enable TLS
tls_enabled = false
tls_cert_path = "certs/server.crt"
tls_key_path = "certs/server.key"

[database]
# Database URL (PostgreSQL or SQLite)
url = "postgresql://user:password@localhost:5432/cyxwiz"
# For SQLite: url = "sqlite:cyxwiz.db?mode=rwc"
# Connection pool settings
max_connections = 20
min_connections = 5
# Connection timeout in seconds
connect_timeout = 30
# Idle connection timeout
idle_timeout = 600
# Enable SQL logging (debug)
log_queries = false

[redis]
# Redis URL for caching
url = "redis://127.0.0.1:6379"
# Connection pool size
pool_size = 10
# Key prefix
prefix = "cyxwiz:"
# Default TTL in seconds
default_ttl = 3600
# Enable mock (in-memory) when Redis unavailable
fallback_to_mock = true

[blockchain]
# Network selection
network = "devnet"  # devnet, testnet, mainnet-beta
# Solana RPC endpoint
solana_rpc_url = "https://api.devnet.solana.com"
# WebSocket URL for subscriptions
solana_ws_url = "wss://api.devnet.solana.com"
# Payer keypair path
payer_keypair_path = "~/.config/solana/id.json"
# Program IDs
escrow_program_id = "11111111111111111111111111111111"
token_program_id = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
# CYXWIZ token mint
token_mint = "CYXWiz111111111111111111111111111111111111"
# Transaction confirmation timeout
confirmation_timeout = 30
# Enable payment processing
payments_enabled = true

[scheduler]
# Job polling interval in milliseconds
polling_interval_ms = 5000
# Maximum jobs per node
max_jobs_per_node = 5
# Job timeout in seconds (default 1 hour)
job_timeout = 3600
# Queue size limit
max_queue_size = 10000
# Enable job prioritization
enable_priority = true
# Node selection algorithm
selection_algorithm = "weighted_score"  # round_robin, least_loaded, weighted_score

[jwt]
# JWT secret for token signing (use strong random value in production)
secret = "your-very-secret-key-change-in-production"
# Token expiration in seconds
expiration = 86400  # 24 hours
# P2P token expiration (for Engine<->Node communication)
p2p_token_expiration = 3600  # 1 hour
# Issuer name
issuer = "cyxwiz-central-server"
# Enable JWT validation
enabled = true

[logging]
# Log level: trace, debug, info, warn, error
level = "info"
# Log format: json, pretty
format = "pretty"
# Log file path (empty for stdout only)
file = "logs/server.log"
# Log rotation
max_size_mb = 100
max_files = 10
# Enable structured logging
structured = true

[monitoring]
# Enable metrics collection
enabled = true
# Prometheus metrics endpoint
metrics_endpoint = "/metrics"
# Health check endpoint
health_endpoint = "/health"
# Metrics collection interval in seconds
collection_interval = 15

[tui]
# Terminal UI settings
enabled = false  # Enable with --tui flag
# Refresh rate in milliseconds
refresh_rate = 500
# Color theme: default, dark, light
theme = "default"

[limits]
# Rate limiting
requests_per_minute = 1000
requests_per_second = 100
# Maximum payload size in bytes
max_payload_size = 104857600  # 100 MB
# Maximum model size for submission
max_model_size = 1073741824  # 1 GB
# Maximum concurrent jobs per user
max_jobs_per_user = 10

[features]
# Feature flags
enable_pool_mining = false
enable_federated_learning = false
enable_model_marketplace = false
enable_terminal_access = false
```

## Environment Variables

Configuration can be overridden with environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `CYXWIZ_SERVER_GRPC_ADDRESS` | gRPC bind address | `0.0.0.0:50051` |
| `CYXWIZ_SERVER_REST_ADDRESS` | REST bind address | `0.0.0.0:8080` |
| `CYXWIZ_DATABASE_URL` | Database connection string | `postgresql://...` |
| `CYXWIZ_REDIS_URL` | Redis connection string | `redis://127.0.0.1:6379` |
| `CYXWIZ_JWT_SECRET` | JWT signing secret | Random string |
| `CYXWIZ_BLOCKCHAIN_NETWORK` | Solana network | `devnet` |
| `CYXWIZ_LOG_LEVEL` | Logging level | `debug` |
| `RUST_LOG` | Rust logging (overrides) | `cyxwiz=debug` |

### Priority Order

1. Environment variables (highest)
2. Command-line arguments
3. Configuration file
4. Default values (lowest)

## Command-Line Arguments

```bash
# Basic usage
cyxwiz-central-server [OPTIONS]

Options:
  -c, --config <PATH>      Configuration file path [default: config.toml]
  --grpc-address <ADDR>    gRPC server address
  --rest-address <ADDR>    REST API address
  --database-url <URL>     Database connection URL
  --tui                    Enable terminal UI mode
  --log-level <LEVEL>      Log level (trace/debug/info/warn/error)
  -h, --help               Print help
  -V, --version            Print version
```

### Examples

```bash
# Use custom config
cyxwiz-central-server -c /etc/cyxwiz/server.toml

# Override database
cyxwiz-central-server --database-url "postgresql://prod:pass@db.example.com/cyxwiz"

# Enable TUI mode
cyxwiz-central-server --tui

# Debug logging
RUST_LOG=debug cyxwiz-central-server
```

## Database Configuration

### PostgreSQL (Production)

```toml
[database]
url = "postgresql://user:password@localhost:5432/cyxwiz"
max_connections = 20
min_connections = 5
```

**Setup:**
```bash
# Create database
createdb cyxwiz

# Run migrations
cyxwiz-central-server migrate
```

### SQLite (Development)

```toml
[database]
url = "sqlite:cyxwiz.db?mode=rwc"
max_connections = 1  # SQLite single-threaded
```

## Redis Configuration

### With Redis

```toml
[redis]
url = "redis://127.0.0.1:6379"
pool_size = 10
```

### Without Redis (Mock)

```toml
[redis]
url = ""
fallback_to_mock = true  # Uses in-memory cache
```

## Blockchain Configuration

### Devnet (Testing)

```toml
[blockchain]
network = "devnet"
solana_rpc_url = "https://api.devnet.solana.com"
payments_enabled = true
```

### Mainnet (Production)

```toml
[blockchain]
network = "mainnet-beta"
solana_rpc_url = "https://api.mainnet-beta.solana.com"
# Use private RPC for production
# solana_rpc_url = "https://your-rpc-provider.com"
payer_keypair_path = "/secure/path/to/keypair.json"
payments_enabled = true
```

## TLS Configuration

### Self-Signed (Development)

```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes
```

```toml
[server]
tls_enabled = true
tls_cert_path = "certs/server.crt"
tls_key_path = "certs/server.key"
```

### Let's Encrypt (Production)

```toml
[server]
tls_enabled = true
tls_cert_path = "/etc/letsencrypt/live/example.com/fullchain.pem"
tls_key_path = "/etc/letsencrypt/live/example.com/privkey.pem"
```

## Scheduler Algorithms

### Round Robin

```toml
[scheduler]
selection_algorithm = "round_robin"
```
Simple rotation through available nodes.

### Least Loaded

```toml
[scheduler]
selection_algorithm = "least_loaded"
```
Assigns to node with fewest active jobs.

### Weighted Score

```toml
[scheduler]
selection_algorithm = "weighted_score"
```
Considers: reputation, capacity, load, and job requirements.

## Logging Configuration

### Development

```toml
[logging]
level = "debug"
format = "pretty"
file = ""  # stdout only
```

### Production

```toml
[logging]
level = "info"
format = "json"
file = "logs/server.log"
max_size_mb = 100
max_files = 10
structured = true
```

### Per-Module Logging

```bash
# Via environment variable
RUST_LOG="cyxwiz_central_server=debug,cyxwiz_central_server::scheduler=trace"
```

## Configuration Validation

The server validates configuration on startup:

```bash
# Validate config without starting
cyxwiz-central-server --config config.toml --validate
```

Common validation errors:
- Invalid database URL format
- JWT secret too short (< 32 chars)
- Missing required fields
- Invalid network name
- Unreachable Redis/Database

## Hot Reload

Some configuration can be reloaded without restart:

```bash
# Send SIGHUP to reload config
kill -HUP $(pgrep cyxwiz-central-server)
```

Hot-reloadable:
- Log level
- Rate limits
- Feature flags

Requires restart:
- Database URL
- Server addresses
- JWT secret
- Blockchain settings

---

**Next**: [gRPC Services](grpc/index.md) | [Deployment](deployment.md)
