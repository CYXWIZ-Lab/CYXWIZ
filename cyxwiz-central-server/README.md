# CyxWiz Central Server

The CyxWiz Central Server is the orchestration layer of the CyxWiz decentralized ML compute platform. It coordinates job distribution, node registration, and payment processing via Solana blockchain.

## Features

- **gRPC Services**: JobService and NodeService for client/node communication
- **Job Scheduling**: Intelligent matching algorithm based on node capabilities and reputation
- **Blockchain Integration**: Solana-based payment escrow and distribution
- **Node Registry**: Track and manage compute nodes in the network
- **REST API**: Dashboard endpoints for monitoring network health
- **Redis Cache**: Fast job queue and node status caching
- **PostgreSQL Database**: Persistent storage for jobs, nodes, payments, and metrics

## Architecture

```
┌─────────────┐              ┌──────────────────┐              ┌─────────────┐
│   Engine    │─────gRPC────▶│  Central Server  │◀────gRPC────│ Server Node │
│  (Client)   │              │                  │              │  (Compute)  │
└─────────────┘              │  ┌────────────┐  │              └─────────────┘
                             │  │  Scheduler │  │
                             │  │   Matcher  │  │
                             │  └────────────┘  │
                             │  ┌────────────┐  │
                             │  │  Database  │  │
                             │  │  (Postgres)│  │
                             │  └────────────┘  │
                             │  ┌────────────┐  │
                             │  │   Redis    │  │
                             │  │   Cache    │  │
                             │  └────────────┘  │
                             │  ┌────────────┐  │
                             │  │  Solana    │  │
                             │  │  Client    │  │
                             │  └────────────┘  │
                             └──────────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │ Solana Blockchain│
                             │  (Escrow/Payment)│
                             └──────────────────┘
```

## Prerequisites

- **Rust**: 1.70+ (install from https://rustup.rs/)
- **PostgreSQL**: 14+ (running locally or remote)
- **Redis**: 6+ (running locally or remote)
- **Solana CLI** (optional, for blockchain testing): Install from https://docs.solana.com/cli/install-solana-cli-tools

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/cyxwiz/cyxwiz.git
cd cyxwiz/cyxwiz-central-server
```

### 2. Install dependencies

```bash
cargo build
```

### 3. Setup PostgreSQL database

```bash
# Create database
createdb cyxwiz

# Or connect to existing PostgreSQL and create database
psql -U postgres
CREATE DATABASE cyxwiz;
CREATE USER cyxwiz WITH PASSWORD 'cyxwiz';
GRANT ALL PRIVILEGES ON DATABASE cyxwiz TO cyxwiz;
\q
```

### 4. Setup Redis

```bash
# On macOS
brew install redis
brew services start redis

# On Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server

# On Windows
# Download from https://redis.io/download
```

### 5. Configure the server

Edit `config.toml` with your settings:

```toml
[database]
url = "postgres://cyxwiz:cyxwiz@localhost/cyxwiz"

[redis]
url = "redis://localhost:6379"

[blockchain]
solana_rpc_url = "https://api.devnet.solana.com"
payer_keypair_path = "~/.config/solana/id.json"
network = "devnet"
```

### 6. Generate Solana keypair (for blockchain integration)

```bash
# Install Solana CLI
sh -c "$(curl -sSfL https://release.solana.com/stable/install)"

# Generate keypair
solana-keygen new --outfile ~/.config/solana/id.json

# Get some devnet SOL for testing
solana airdrop 2 --url devnet
```

### 7. Run database migrations

Migrations run automatically on server start, but you can also run them manually:

```bash
cargo install sqlx-cli --no-default-features --features postgres
sqlx migrate run
```

## Running the Server

### Development mode

```bash
cargo run
```

### Production mode

```bash
cargo build --release
./target/release/cyxwiz-central-server
```

### With Terminal UI (TUI) - Real-time Monitoring

```bash
# Run with integrated TUI
cargo run --release -- --tui

# Or use the compiled binary
./target/release/cyxwiz-central-server --tui
```

The TUI provides:
- Real-time dashboard with stats and graphs
- Node monitoring with health indicators
- Job tracking with progress bars
- Blockchain transaction viewer
- Live server logs
- Configuration display

**See [TUI_GUIDE.md](TUI_GUIDE.md) for full documentation.**

### With custom config

```bash
CYXWIZ_DATABASE__URL=postgres://user:pass@host/db cargo run
```

Environment variables override `config.toml` using the prefix `CYXWIZ_` and double underscores for nesting (e.g., `CYXWIZ_SERVER__GRPC_ADDRESS`).

## API Endpoints

### gRPC Services

#### JobService (Port 50051)

- `SubmitJob(SubmitJobRequest) → SubmitJobResponse`
  - Submit a new ML training job
- `GetJobStatus(GetJobStatusRequest) → GetJobStatusResponse`
  - Query the status of a job
- `CancelJob(CancelJobRequest) → CancelJobResponse`
  - Cancel a pending or running job
- `StreamJobUpdates(GetJobStatusRequest) → stream JobUpdateStream`
  - Real-time job progress updates (not implemented yet)
- `ListJobs(ListJobsRequest) → ListJobsResponse`
  - List user's jobs (not implemented yet)

#### NodeService (Port 50051)

- `RegisterNode(RegisterNodeRequest) → RegisterNodeResponse`
  - Register a new compute node
- `Heartbeat(HeartbeatRequest) → HeartbeatResponse`
  - Keep-alive heartbeat from nodes
- `ReportProgress(ReportProgressRequest) → ReportProgressResponse`
  - Report job progress from node
- `ReportCompletion(ReportCompletionRequest) → ReportCompletionResponse`
  - Report job completion with results
- `GetNodeMetrics(GetNodeMetricsRequest) → GetNodeMetricsResponse`
  - Query node performance metrics

### REST API (Port 8080)

#### Health Check

```bash
curl http://localhost:8080/api/health
```

Response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "service": "cyxwiz-central-server"
}
```

#### Network Statistics

```bash
curl http://localhost:8080/api/stats
```

Response:
```json
{
  "total_nodes": 100,
  "online_nodes": 85,
  "total_jobs": 1523,
  "pending_jobs": 12,
  "running_jobs": 8,
  "completed_jobs": 1503,
  "total_compute_hours": 45234.5
}
```

#### List Nodes

```bash
curl http://localhost:8080/api/nodes
```

#### Get Node Details

```bash
curl http://localhost:8080/api/nodes/<node_id>
```

#### List Jobs

```bash
curl http://localhost:8080/api/jobs?page=1&limit=50
```

#### Get Job Details

```bash
curl http://localhost:8080/api/jobs/<job_id>
```

## Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

```bash
cargo test --test '*'
```

### Testing with grpcurl

```bash
# Install grpcurl
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List services
grpcurl -plaintext localhost:50051 list

# Call JobService
grpcurl -plaintext -d '{"job_id": "123"}' \
  localhost:50051 cyxwiz.protocol.JobService/GetJobStatus
```

## Blockchain Integration

### Payment Flow

1. **Job Submission**
   - User submits job via Engine
   - Central Server estimates cost
   - Creates escrow on Solana blockchain
   - Locks user's CYXWIZ tokens

2. **Job Assignment**
   - Scheduler matches job to best node
   - Node begins execution

3. **Job Execution**
   - Node reports progress every minute
   - Can claim partial payment (streaming)

4. **Job Completion**
   - Node submits results
   - Central Server verifies
   - Releases payment:
     - 90% to node
     - 10% platform fee

5. **Job Failure**
   - Refund user
   - Penalize node reputation
   - Retry job if within limit

### Solana Smart Contracts

The Central Server interacts with three main Solana programs:

1. **JobEscrow**: Lock and release funds
2. **PaymentStreaming**: Real-time payment claims
3. **NodeStaking**: Reputation and slashing

#### Deploying Smart Contracts (Future)

```bash
cd ../cyxwiz-blockchain
anchor build
anchor deploy --provider.cluster devnet
```

Update `config.toml` with the deployed program ID.

## Job Scheduler

### Matching Algorithm

The scheduler uses a scoring system to match jobs to nodes:

```
match_score =
  0.4 * reputation_score +
  0.3 * availability (1 - current_load) +
  0.2 * capability_match +
  0.1 * uptime_percentage
```

Nodes are ranked by score, and the highest-scoring capable node is assigned the job.

### Cost Estimation

```
cost_per_second = base_rate * gpu_multiplier * (1 + ram_factor)
total_cost = cost_per_second * estimated_duration_seconds
```

Where:
- `base_rate = 10,000` (in CYXWIZ smallest units)
- `gpu_multiplier = 10.0` if GPU required
- `ram_factor = required_ram_gb * 0.01`

## Database Schema

### Tables

- **nodes**: Registered compute nodes
- **jobs**: Submitted ML training jobs
- **payments**: Blockchain payment records
- **node_metrics**: Time-series performance data

### Enums

- **node_status**: `online`, `offline`, `busy`, `maintenance`
- **job_status**: `pending`, `assigned`, `running`, `completed`, `failed`, `cancelled`
- **payment_status**: `pending`, `locked`, `streaming`, `completed`, `failed`, `refunded`

## Monitoring

### Logs

Logs are written to stdout in JSON format. Use `RUST_LOG` to control verbosity:

```bash
RUST_LOG=cyxwiz_central_server=debug cargo run
```

### Metrics

Expose metrics via Prometheus (TODO):

```bash
curl http://localhost:8080/metrics
```

## Troubleshooting

### Database connection failed

```
Error: Failed to connect to database
```

**Solution**: Ensure PostgreSQL is running and `config.toml` has correct credentials.

```bash
psql -U cyxwiz -d cyxwiz
```

### Redis connection failed

```
Error: Failed to connect to Redis
```

**Solution**: Ensure Redis is running:

```bash
redis-cli ping
# Should return: PONG
```

### Solana blockchain errors

```
Error: Failed to create escrow
```

**Solution**: Check Solana RPC is accessible and keypair has funds:

```bash
solana balance --url devnet
solana airdrop 2 --url devnet
```

### Migration errors

```
Error: relation "nodes" does not exist
```

**Solution**: Run migrations:

```bash
sqlx migrate run
```

## Development

### Project Structure

```
src/
├── api/
│   ├── grpc/
│   │   ├── job_service.rs    # JobService implementation
│   │   └── node_service.rs   # NodeService implementation
│   └── rest/
│       └── dashboard.rs       # REST API endpoints
├── blockchain/
│   ├── solana_client.rs       # Solana RPC client
│   └── payment_processor.rs   # Payment logic
├── cache/
│   └── mod.rs                 # Redis cache wrapper
├── database/
│   ├── models.rs              # Database models
│   ├── queries.rs             # SQL queries
│   └── mod.rs
├── scheduler/
│   ├── job_queue.rs           # Job queue processor
│   └── matcher.rs             # Node matching algorithm
├── config.rs                  # Configuration
├── error.rs                   # Error types
└── main.rs                    # Entry point
```

### Adding a New Feature

1. **Add database model** (if needed) in `database/models.rs`
2. **Add queries** in `database/queries.rs`
3. **Implement business logic** in appropriate module
4. **Add gRPC/REST endpoint** in `api/`
5. **Write tests** in module or `tests/`
6. **Update documentation**

### Code Style

```bash
cargo fmt
cargo clippy
```

## Deployment

### Docker

```bash
docker build -t cyxwiz-central-server .
docker run -p 50051:50051 -p 8080:8080 \
  -e CYXWIZ_DATABASE__URL=postgres://... \
  cyxwiz-central-server
```

### Kubernetes

See `k8s/` directory for manifests (TODO).

## Security Considerations

- **Authentication**: Implement JWT tokens for gRPC (TODO)
- **TLS/SSL**: Enable TLS for gRPC in production
- **Rate Limiting**: Implemented via Redis
- **Input Validation**: All inputs are validated
- **SQL Injection**: Using parameterized queries (sqlx)
- **Secrets**: Store keypairs securely, never commit to git

## Performance Tuning

- **Database connections**: Adjust `max_connections` in config
- **Redis pool**: Adjust `pool_size` for high traffic
- **Scheduler interval**: Lower `job_poll_interval_ms` for faster assignment
- **gRPC concurrency**: Tune `max_connections` for load

## Roadmap

- [ ] Implement job streaming (real-time updates)
- [ ] Add JWT authentication
- [ ] Deploy smart contracts on Solana mainnet
- [ ] Add Prometheus metrics
- [ ] Implement job cancellation refunds
- [ ] Add node slashing mechanism
- [ ] Support distributed training (multi-node jobs)
- [ ] Add Docker support
- [ ] Kubernetes deployment manifests

## Contributing

See main repository CONTRIBUTING.md

## License

Apache 2.0 - See LICENSE file

## Support

- **Issues**: https://github.com/cyxwiz/cyxwiz/issues
- **Discord**: https://discord.gg/cyxwiz
- **Email**: support@cyxwiz.com

## Credits

Built with:
- [Tonic](https://github.com/hyperium/tonic) - gRPC framework
- [Axum](https://github.com/tokio-rs/axum) - Web framework
- [SQLx](https://github.com/launchbadge/sqlx) - Database toolkit
- [Redis-rs](https://github.com/redis-rs/redis-rs) - Redis client
- [Solana](https://solana.com) - Blockchain platform
