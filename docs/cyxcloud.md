# CyxCloud: Decentralized Cloud Storage Platform

## Executive Summary

CyxCloud is a **decentralized cloud storage platform** - think Google Drive or Dropbox, but powered by a distributed network of storage providers. Users with spare disk space can contribute to the network and earn CYXWIZ tokens, while consumers get secure, redundant, and affordable cloud storage.

**Key Value Propositions:**
- **For Storage Providers**: Monetize unused disk space (like Airbnb for storage)
- **For Consumers**: Affordable, private, censorship-resistant cloud storage
- **For Developers**: S3-compatible API for any application
- **For CyxWiz Ecosystem**: Native ML dataset storage with optimized loading

CyxCloud is **NOT** limited to ML - it's a general-purpose cloud storage that happens to have ML-optimized features for our ecosystem.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CyxWiz Ecosystem                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐         ┌──────────────────┐         ┌──────────────────┐ │
│  │   Engine     │◄───────►│  Central Server  │◄───────►│   Server Node    │ │
│  │ (Desktop)    │         │  (Orchestrator)  │         │   (Compute)      │ │
│  └──────┬───────┘         └────────┬─────────┘         └────────┬─────────┘ │
│         │                          │                             │          │
│         │     ┌────────────────────┴────────────────────┐        │          │
│         │     │                                         │        │          │
│         ▼     ▼                                         ▼        ▼          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         CyxCloud Network                                ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       ││
│  │  │ Storage │  │ Storage │  │ Storage │  │ Storage │  │ Storage │       ││
│  │  │ Node 1  │  │ Node 2  │  │ Node 3  │  │ Node 4  │  │ Node N  │       ││
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       ││
│  │       │            │            │            │            │             ││
│  │       └────────────┴─────┬──────┴────────────┴────────────┘             ││
│  │                          │                                              ││
│  │                 ┌────────┴────────┐                                     ││
│  │                 │  Blockchain     │                                     ││
│  │                 │  (Solana)       │                                     ││
│  │                 │  - Metadata     │                                     ││
│  │                 │  - Ownership    │                                     ││
│  │                 │  - Payments     │                                     ││
│  │                 │  - Integrity    │                                     ││
│  │                 └─────────────────┘                                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Storage Node (cyxcloud-node)

Individual storage providers in the network. Can be:
- Home NAS servers
- Data center racks
- Cloud VPS instances
- Dedicated storage servers

**Responsibilities:**
- Store encrypted data chunks
- Serve data requests
- Participate in RAID reconstruction
- Report health metrics to Central Server
- Earn CYXWIZ tokens for storage/bandwidth

### 2. CyxCloud Gateway

API layer for data access:
- REST API for uploads/downloads
- Streaming API for large datasets
- gRPC for internal ecosystem communication
- WebSocket for real-time sync

### 3. Blockchain Layer (Solana)

On-chain components:
- **Data Registry**: CID → metadata mapping
- **Ownership Registry**: Who owns what data
- **Access Control**: Who can read/write
- **Payment Streams**: Storage fees, bandwidth fees
- **Reputation System**: Node reliability scores

### 4. Coordination Layer

Managed by Central Server:
- Node discovery and health monitoring
- Data placement optimization
- Load balancing
- Replication factor management
- Geographic distribution

---

## Data Flow Scenarios

### Scenario 1: Local Training (Engine Direct Access)

```
┌────────┐     1. Request dataset      ┌──────────────┐
│ Engine │ ──────────────────────────► │   CyxCloud   │
│        │                             │   Gateway    │
│        │ ◄────────────────────────── │              │
└────────┘     2. Stream data          └──────────────┘
                  (lazy load)                 │
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                              ┌──────────┐       ┌──────────┐
                              │ Storage  │       │ Storage  │
                              │ Node A   │       │ Node B   │
                              └──────────┘       └──────────┘
```

### Scenario 2: Distributed Training (Server Node)

```
┌────────┐  1. Create model     ┌─────────────┐
│ Engine │ ────────────────────►│   Central   │
│        │  + data location     │   Server    │
└────────┘  (CID/URI)           └──────┬──────┘
                                       │
                    2. Allocate compute node
                    + forward job metadata
                                       │
                                       ▼
                                ┌──────────────┐
                                │  Server Node │
                                │  (Compute)   │
                                └──────┬───────┘
                                       │
                    3. Request data by CID
                    (knows location from metadata)
                                       │
                                       ▼
                                ┌──────────────┐
                                │   CyxCloud   │
                                │   Network    │
                                └──────────────┘
                                       │
                    4. Stream training data
                    (batch loading, prefetch)
                                       │
                                       ▼
                                ┌──────────────┐
                                │  Server Node │
                                │  (Training)  │
                                └──────────────┘
```

### Scenario 3: Data Marketplace

```
┌──────────────┐                              ┌──────────────┐
│ Data Seller  │                              │  Data Buyer  │
└──────┬───────┘                              └──────┬───────┘
       │                                             │
       │ 1. Upload dataset                           │
       │ 2. Set price/license                        │
       ▼                                             │
┌──────────────┐                                     │
│   CyxCloud   │                                     │
│   Network    │                                     │
└──────┬───────┘                                     │
       │                                             │
       │ 3. Register on blockchain                   │
       ▼                                             │
┌──────────────┐     4. Browse/Search     ┌─────────┴────────┐
│  Blockchain  │ ◄────────────────────────│   Marketplace    │
│  - Metadata  │                          │   (Web/Engine)   │
│  - Pricing   │ ─────────────────────────►                  │
│  - Licensing │     5. Purchase (CYXWIZ) └──────────────────┘
└──────────────┘
       │
       │ 6. Grant access token
       ▼
┌──────────────┐
│  Data Buyer  │ ─────► Can now access dataset
└──────────────┘
```

---

## Technical Specifications

### Content Addressing

Using IPFS-style Content Identifiers (CID):

```
cyx://Qm[base58-encoded-multihash]/path/to/file

Example:
cyx://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG/mnist/train.csv
```

**Benefits:**
- Deduplication (same data = same CID)
- Integrity verification (hash-based)
- Location-independent addressing
- Cache-friendly

### Data Chunking Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Original Dataset (10GB)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                    Split into chunks
                              │
                              ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Chunk  │ │ Chunk  │ │ Chunk  │ │ Chunk  │ │ Chunk  │  ...
│  1MB   │ │  1MB   │ │  1MB   │ │  1MB   │ │  1MB   │
│ CID: A │ │ CID: B │ │ CID: C │ │ CID: D │ │ CID: E │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
     │          │          │          │          │
     │    Reed-Solomon Erasure Coding (k=10, m=4)
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Node 1 │ │ Node 2 │ │ Node 3 │ │ Node 4 │ │ Node 5 │
│ Shard  │ │ Shard  │ │ Shard  │ │ Shard  │ │ Parity │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

**Chunk Size Options:**
- Small files (<1MB): Store whole
- Medium files (1MB-1GB): 1MB chunks
- Large files (>1GB): 4MB chunks
- Streaming datasets: 64KB chunks (for batch loading)

### RAID-like Redundancy

**Erasure Coding Parameters:**
- `k` = 10 (data shards)
- `m` = 4 (parity shards)
- Can recover from any 4 node failures
- 1.4x storage overhead (vs 3x for simple replication)

```rust
// Reed-Solomon configuration
struct ErasureConfig {
    data_shards: usize,      // k = 10
    parity_shards: usize,    // m = 4
    shard_size: usize,       // bytes per shard
}

// Can tolerate up to `parity_shards` failures
// Need only `data_shards` to reconstruct
```

### Data Loading Strategies

#### 1. Lazy Loading
```python
# Only load data when accessed
dataset = CyxDataset("cyx://QmXXX/imagenet")
# No data downloaded yet

for batch in dataset:  # Downloads on-demand
    train(model, batch)
```

#### 2. LRU Cache
```python
# Keep recently used chunks in memory/disk
cache = LRUCache(max_size_gb=10)

# Automatic eviction of least-recently-used
chunk = cache.get_or_fetch(cid)
```

#### 3. Batch Prefetching
```python
# Prefetch next N batches while training current
prefetcher = BatchPrefetcher(
    dataset=dataset,
    prefetch_count=4,      # 4 batches ahead
    num_workers=2          # parallel download threads
)

for batch in prefetcher:
    train(model, batch)  # Next batches downloading in background
```

#### 4. Streaming Mode
```python
# Never store full dataset, stream chunks
stream = dataset.stream(
    batch_size=32,
    shuffle_buffer=1000,  # Shuffle within buffer
    prefetch=2
)

for batch in stream:
    train(model, batch)
```

---

## Tech Stack

### Ecosystem Alignment

CyxCloud's tech stack is designed to seamlessly integrate with the existing CyxWiz ecosystem:

| Layer | CyxWiz Ecosystem | CyxCloud | Rationale |
|-------|-----------------|----------|-----------|
| **Systems Language** | Rust (Central Server) | **Rust** | Consistent, memory-safe, async-native |
| **Client Bindings** | C++ + Python (pybind11) | **Rust + PyO3** (Python) | Engine integration via Rust FFI or pybind11 wrapper |
| **IPC/RPC** | gRPC + Protobuf | **gRPC + Protobuf** | Same protocol definitions, code reuse |
| **Blockchain** | Solana | **Solana** | Existing wallet, token (CYXWIZ), contracts |
| **Build System** | CMake + Cargo | **Cargo** (pure Rust) | Workspace with multiple crates |

### Complete Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CyxCloud Tech Stack                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  NETWORKING                                                      │
│  ├── libp2p (Rust)         P2P discovery, DHT, NAT traversal    │
│  ├── QUIC (quinn)          Fast UDP transport                   │
│  └── tonic                  gRPC for ecosystem comms            │
│                                                                  │
│  STORAGE                                                         │
│  ├── RocksDB (rust-rocksdb) Local chunk storage                 │
│  ├── sled                   Metadata DB (pure Rust alternative) │
│  └── reed-solomon-erasure   Erasure coding                      │
│                                                                  │
│  API                                                             │
│  ├── Axum                   REST API (S3-compatible)            │
│  ├── tonic                  gRPC services                       │
│  └── tokio-tungstenite      WebSocket for real-time sync        │
│                                                                  │
│  CRYPTO                                                          │
│  ├── ring / rustls          TLS, encryption                     │
│  ├── blake3                 Content hashing                     │
│  ├── aes-gcm                Data encryption                     │
│  └── ed25519-dalek          Signatures                          │
│                                                                  │
│  BLOCKCHAIN                                                      │
│  ├── solana-sdk             Wallet, transactions                │
│  ├── anchor-lang            Smart contracts                     │
│  └── solana-client          RPC client                          │
│                                                                  │
│  CLIENT SDKs                                                     │
│  ├── cyxcloud-client (Rust) Core SDK                            │
│  ├── PyO3                   Python bindings                     │
│  └── cyxcloud.h (C)         C/C++ FFI for Engine                │
│                                                                  │
│  OBSERVABILITY                                                   │
│  ├── tracing                Structured logging                  │
│  ├── metrics                Prometheus metrics                  │
│  └── opentelemetry          Distributed tracing                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Technologies

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **P2P Networking** | libp2p (Rust) | Battle-tested, used by IPFS/Filecoin |
| **Content Addressing** | IPFS CID v1 | Industry standard, self-describing |
| **Erasure Coding** | reed-solomon-erasure (Rust) | Fast, proven algorithm |
| **Blockchain** | Solana | High throughput, low fees, existing integration |
| **Storage Backend** | RocksDB | Fast KV store for chunks |
| **API Layer** | Axum (Rust) | Fast async HTTP, gRPC via tonic |
| **Encryption** | AES-256-GCM | Client-side encryption |
| **Hashing** | BLAKE3 | Faster than SHA-256, cryptographically secure |

### Cargo Workspace Dependencies

```toml
# cyxcloud/Cargo.toml
[workspace]
members = [
    "cyxcloud-core",       # Shared types, CID, crypto
    "cyxcloud-node",       # Storage node daemon
    "cyxcloud-gateway",    # API gateway
    "cyxcloud-client",     # Client SDK
    "cyxcloud-python",     # PyO3 bindings
    "cyxcloud-cli",        # CLI tool
    "cyxcloud-contracts",  # Solana programs
]

[workspace.dependencies]
# Networking
libp2p = { version = "0.53", features = ["tokio", "quic", "kad", "identify"] }
quinn = "0.10"
tonic = "0.10"
prost = "0.12"

# Storage
rocksdb = "0.21"
reed-solomon-erasure = "6.0"

# API
axum = { version = "0.7", features = ["ws", "multipart"] }
tower = "0.4"
hyper = "1.0"

# Crypto
blake3 = "1.5"
aes-gcm = "0.10"
ring = "0.17"
ed25519-dalek = "2.0"

# Blockchain
solana-sdk = "1.17"
solana-client = "1.17"
anchor-lang = "0.29"

# Python
pyo3 = { version = "0.20", features = ["extension-module"] }

# Async
tokio = { version = "1", features = ["full"] }
futures = "0.3"

# Observability
tracing = "0.1"
tracing-subscriber = "0.3"
metrics = "0.21"
```

### Language Choices

| Component | Language | Reason |
|-----------|----------|--------|
| Storage Node | Rust | Performance, memory safety, ecosystem consistency |
| Gateway | Rust | Async I/O, performance |
| Smart Contracts | Rust (Anchor) | Solana native |
| Client SDK | Rust + Python bindings | Integrate with Engine |
| CLI | Rust | Single binary distribution |

### Integration with CyxWiz Engine (C++)

**Option A: Rust FFI with C header** (recommended for performance-critical paths)
```cpp
// cyxwiz-engine/include/cyxcloud/client.h
extern "C" {
    typedef struct CyxCloudClient CyxCloudClient;

    CyxCloudClient* cyxcloud_client_new(const char* gateway_url);
    void cyxcloud_client_free(CyxCloudClient* client);

    int cyxcloud_upload(CyxCloudClient* client, const char* path, char** cid_out);
    int cyxcloud_download(CyxCloudClient* client, const char* cid, const char* dest);
    int cyxcloud_stream(CyxCloudClient* client, const char* cid, StreamCallback callback);
}
```

**Option B: gRPC client in C++** (reuse existing infrastructure)
```cpp
// Use same gRPC pattern as Central Server communication
#include "cyxcloud.grpc.pb.h"

auto channel = grpc::CreateChannel("gateway.cyxcloud.io:443", creds);
auto stub = cyxcloud::DataService::NewStub(channel);

// Stream data for training
grpc::ClientContext context;
cyxcloud::StreamDataRequest request;
request.set_dataset_cid("QmXXX...");
request.set_batch_size(32);

auto reader = stub->StreamData(&context, request);
cyxcloud::DataChunk chunk;
while (reader->Read(&chunk)) {
    // Process training batch
}
```

### Open Source Libraries by Purpose

| Purpose | Library | Why |
|---------|---------|-----|
| P2P | **libp2p** | IPFS/Filecoin battle-tested |
| DHT | **libp2p-kad** | Kademlia for peer discovery |
| Erasure | **reed-solomon-erasure** | Fast, pure Rust |
| Content ID | **cid** (Rust crate) | IPFS-compatible CID |
| S3 API | **s3s** or custom Axum | S3-compatible endpoints |
| Storage | **RocksDB** | LSM tree, proven at scale |
| Streams | **Apache Kafka protocol** via **rdkafka** | For real-time data streams |

### Open Source Inspiration

| Project | What to Learn |
|---------|---------------|
| **IPFS** | Content addressing, DHT, libp2p |
| **Filecoin** | Proof of storage, deal-making |
| **Storj** | Erasure coding, satellite architecture |
| **Sia** | Reed-Solomon implementation, contracts |
| **Arweave** | Permanent storage model |
| **OrbitDB** | P2P database over IPFS |
| **SeaweedFS** | Fast blob storage, master/volume architecture |
| **MinIO** | S3-compatible API, erasure coding |

### Why This Stack Works

1. **Rust everywhere** - Same language as Central Server, easy to share code
2. **gRPC/Protobuf** - Reuse existing `.proto` definitions from `cyxwiz-protocol`
3. **Solana** - Same blockchain, token, wallet infrastructure
4. **PyO3** - Like pybind11 but for Rust, integrates with existing Python ecosystem
5. **C FFI** - Easy integration with Engine's C++ codebase

---

## Project Structure

```
cyxcloud/
├── Cargo.toml                    # Workspace root
├── README.md
│
├── cyxcloud-node/               # Storage node daemon
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── storage/             # Local storage engine
│       │   ├── mod.rs
│       │   ├── rocks_backend.rs # RocksDB storage
│       │   ├── chunk_manager.rs # Chunk operations
│       │   └── gc.rs            # Garbage collection
│       ├── network/             # P2P networking
│       │   ├── mod.rs
│       │   ├── peer_manager.rs
│       │   ├── dht.rs           # Distributed hash table
│       │   └── protocol.rs      # Wire protocol
│       ├── erasure/             # Reed-Solomon coding
│       │   ├── mod.rs
│       │   ├── encoder.rs
│       │   └── decoder.rs
│       └── metrics/             # Telemetry
│
├── cyxcloud-gateway/            # API gateway service
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── api/
│       │   ├── rest.rs          # REST endpoints
│       │   ├── grpc.rs          # gRPC for ecosystem
│       │   └── streaming.rs     # Data streaming
│       ├── routing/             # Request routing
│       └── cache/               # Edge caching
│
├── cyxcloud-client/             # Client SDK
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── dataset.rs           # Dataset abstraction
│       ├── loader.rs            # Data loading strategies
│       ├── cache.rs             # LRU cache
│       └── prefetch.rs          # Batch prefetching
│
├── cyxcloud-python/             # Python bindings
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs               # PyO3 bindings
│
├── cyxcloud-contracts/          # Solana smart contracts
│   ├── Cargo.toml
│   └── programs/
│       ├── registry/            # Data registry program
│       ├── marketplace/         # Buy/sell data
│       └── staking/             # Node staking
│
├── cyxcloud-cli/                # Command-line interface
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
│
└── cyxcloud-core/               # Shared library
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── cid.rs               # Content ID handling
        ├── crypto.rs            # Encryption/hashing
        ├── types.rs             # Common types
        └── config.rs            # Configuration
```

---

## API Design

### REST API (Gateway)

```yaml
# Upload dataset
POST /api/v1/datasets
Content-Type: multipart/form-data
Authorization: Bearer <token>

Response:
{
  "cid": "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
  "size": 1073741824,
  "chunks": 1024,
  "replication_factor": 3
}

# Download dataset
GET /api/v1/datasets/{cid}
Range: bytes=0-1048575  # Supports range requests

# Stream dataset
GET /api/v1/datasets/{cid}/stream?batch_size=32&format=numpy

# Get metadata
GET /api/v1/datasets/{cid}/metadata

Response:
{
  "cid": "QmXXX",
  "name": "MNIST",
  "size": 52428800,
  "format": "csv",
  "schema": {
    "features": ["pixel_0", "pixel_1", ..., "pixel_783"],
    "label": "digit"
  },
  "owner": "CYX1abc...xyz",
  "price": 0,  // Free
  "license": "CC-BY-4.0",
  "created_at": "2024-01-15T10:30:00Z"
}

# List user's datasets
GET /api/v1/users/me/datasets

# Marketplace search
GET /api/v1/marketplace/search?q=image+classification&min_size=1GB
```

### gRPC API (Internal Ecosystem)

```protobuf
syntax = "proto3";
package cyxcloud;

service DataService {
  // Get dataset metadata
  rpc GetMetadata(GetMetadataRequest) returns (DatasetMetadata);

  // Stream data chunks
  rpc StreamData(StreamDataRequest) returns (stream DataChunk);

  // Get specific chunk by CID
  rpc GetChunk(GetChunkRequest) returns (DataChunk);

  // Prefetch chunks (hint to cache)
  rpc Prefetch(PrefetchRequest) returns (PrefetchResponse);
}

service StorageService {
  // Upload data
  rpc Upload(stream DataChunk) returns (UploadResponse);

  // Get storage stats
  rpc GetStats(Empty) returns (StorageStats);

  // Pin data (prevent GC)
  rpc Pin(PinRequest) returns (PinResponse);
}

message DatasetMetadata {
  string cid = 1;
  string name = 2;
  uint64 size = 3;
  string format = 4;
  DataSchema schema = 5;
  string owner = 6;
  uint64 price_lamports = 7;
  string license = 8;
}

message DataChunk {
  string cid = 1;
  uint32 index = 2;
  bytes data = 3;
  uint64 offset = 4;
  uint64 total_size = 5;
}

message StreamDataRequest {
  string dataset_cid = 1;
  uint64 offset = 2;
  uint64 limit = 3;
  uint32 batch_size = 4;
  bool shuffle = 5;
}
```

### Python SDK

```python
import cyxcloud

# Initialize client
client = cyxcloud.Client(
    gateway="https://gateway.cyxcloud.io",
    wallet="~/.cyxwiz/wallet.json"
)

# Upload dataset
dataset = client.upload(
    path="./data/my_dataset",
    name="My Custom Dataset",
    description="Training data for image classification",
    price=0,  # Free
    license="MIT"
)
print(f"Uploaded: cyx://{dataset.cid}")

# Load dataset (lazy)
dataset = cyxcloud.Dataset("cyx://QmXXX/imagenet")

# Configure loading strategy
loader = dataset.loader(
    batch_size=32,
    shuffle=True,
    prefetch=4,
    cache_size_gb=10,
    num_workers=4
)

# Use in training
for batch in loader:
    images, labels = batch
    # train...

# Marketplace
results = client.marketplace.search(
    query="medical imaging",
    min_samples=10000,
    max_price=100  # CYXWIZ tokens
)

for dataset in results:
    print(f"{dataset.name}: {dataset.price} CYXWIZ")

# Purchase dataset
client.marketplace.purchase(dataset_cid="QmXXX")
```

---

## Integration with CyxWiz Ecosystem

### Engine Integration

```cpp
// In cyxwiz-engine
#include <cyxcloud/client.h>

// Load remote dataset
auto dataset = cyxcloud::Dataset::from_uri("cyx://QmXXX/mnist");

// Configure for local training
dataset.set_cache_dir("~/.cyxwiz/cache");
dataset.set_prefetch(4);

// Use with DataInput node
node_editor.set_data_source(dataset);
```

### Server Node Integration

```cpp
// In cyxwiz-server-node
#include <cyxcloud/client.h>

void JobExecutor::execute_training_job(const Job& job) {
    // Extract data CID from job metadata
    auto data_cid = job.metadata().data_cid();

    // Create streaming loader (no full download)
    auto loader = cyxcloud::StreamingLoader(data_cid, {
        .batch_size = job.batch_size(),
        .prefetch = 4,
        .cache_size_mb = 1024  // 1GB local cache
    });

    // Train with streaming data
    for (auto& batch : loader) {
        model.train_step(batch);

        // Report progress to Central Server
        report_progress(job.id(), loader.progress());
    }
}
```

### Central Server Coordination

```rust
// In cyxwiz-central-server
impl CentralServer {
    async fn submit_training_job(&self, request: JobRequest) -> Result<JobId> {
        // 1. Validate data CID exists and user has access
        let data_meta = self.cyxcloud_client
            .get_metadata(&request.data_cid)
            .await?;

        self.verify_access(&request.user_id, &data_meta)?;

        // 2. Find optimal compute node (consider data locality)
        let node = self.scheduler.find_optimal_node(
            &request.requirements,
            &data_meta.location_hints  // Prefer nodes near data
        ).await?;

        // 3. Submit job with data reference
        let job = Job {
            id: JobId::new(),
            model_definition: request.model,
            data_cid: request.data_cid,
            data_size: data_meta.size,
            assigned_node: node.id,
            // ...
        };

        self.node_client.submit_job(&node, &job).await?;

        Ok(job.id)
    }
}
```

---

## Data Integrity & Security

### Client-Side Encryption

```
┌────────────────────────────────────────────────────────────┐
│                    User's Machine                          │
│                                                            │
│  Raw Data ──► AES-256-GCM Encrypt ──► Encrypted Chunks    │
│                     │                                      │
│              User's Key (never leaves device)              │
└────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────────┐
│                  CyxCloud Network                          │
│                                                            │
│  Storage nodes only see encrypted data                     │
│  Cannot read contents without user's key                   │
└────────────────────────────────────────────────────────────┘
```

### Access Control

```rust
// On-chain access control
struct DatasetAccess {
    owner: Pubkey,

    // Access levels
    public_read: bool,              // Anyone can read
    public_list: bool,              // Visible in marketplace

    // Whitelist
    allowed_readers: Vec<Pubkey>,   // Specific addresses
    allowed_compute_nodes: Vec<Pubkey>,  // Can use for training

    // Token-gated access
    required_token: Option<Pubkey>, // Must hold NFT/token
    required_balance: u64,          // Minimum token balance
}
```

### Integrity Verification

```
Every chunk has:
1. CID (content hash) - verifies data integrity
2. Merkle proof - verifies chunk belongs to dataset
3. Node signature - verifies source authenticity

┌─────────────────────────────────────────┐
│           Merkle Root (Dataset CID)      │
└────────────────────┬────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────┴────┐             ┌────┴────┐
    │ Hash AB │             │ Hash CD │
    └────┬────┘             └────┬────┘
         │                       │
    ┌────┴────┐             ┌────┴────┐
    │         │             │         │
┌───┴───┐ ┌───┴───┐     ┌───┴───┐ ┌───┴───┐
│Chunk A│ │Chunk B│     │Chunk C│ │Chunk D│
└───────┘ └───────┘     └───────┘ └───────┘

Client can verify any chunk belongs to dataset
by checking Merkle proof against root CID.
```

---

## Become a Storage Provider

Anyone with spare disk space can join the CyxCloud network and earn CYXWIZ tokens.

### Requirements

| Tier | Storage | Bandwidth | Uptime | Monthly Earnings* |
|------|---------|-----------|--------|-------------------|
| **Lite** | 100 GB+ | 10 Mbps | 90%+ | ~$5-15 |
| **Standard** | 1 TB+ | 50 Mbps | 95%+ | ~$20-50 |
| **Pro** | 10 TB+ | 100 Mbps | 99%+ | ~$100-300 |
| **Enterprise** | 100 TB+ | 1 Gbps | 99.9%+ | ~$500-2000 |

*Earnings depend on network demand, location, and performance

### Supported Hardware

```
Recommended setups:

Home NAS:
├── Synology DS920+ with 4x 8TB drives (RAID 5)
├── QNAP TS-453D with SSD cache
└── DIY with TrueNAS + ZFS

Dedicated Server:
├── Dell PowerEdge R740xd (24 bay)
├── Supermicro storage chassis
└── HPE ProLiant DL380

Cloud VPS (resell excess):
├── Hetzner dedicated servers
├── OVH storage VPS
└── Any provider with unmetered bandwidth
```

### Setup Guide

```bash
# 1. Install CyxCloud node
curl -sSL https://get.cyxcloud.io | bash

# 2. Configure storage path and allocation
cyxcloud config set storage.path /mnt/storage
cyxcloud config set storage.allocated 500GB

# 3. Link your wallet
cyxcloud wallet link <your_solana_address>

# 4. Start the node
cyxcloud node start

# 5. Stake tokens (required for trust)
cyxcloud stake deposit 100  # Minimum 100 CYXWIZ

# Check earnings
cyxcloud earnings
# Today: 2.34 CYXWIZ (~$0.47)
# This month: 45.67 CYXWIZ (~$9.13)
# Total: 234.56 CYXWIZ (~$46.91)
```

### Provider Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  🖥️ CyxCloud Node Dashboard          Status: 🟢 Online          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Storage                          Bandwidth                     │
│  ━━━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━        │
│  Used: 423 GB / 500 GB           Upload: 45 Mbps                │
│  [██████████████████░░] 84%      Download: 23 Mbps              │
│                                                                 │
│  Chunks stored: 423,567          Requests today: 12,345         │
│  Unique files: 8,234             Data served: 234 GB            │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  💰 Earnings                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   Today        This Week      This Month      Total     │   │
│  │   2.34 CYXWIZ  15.67 CYXWIZ   45.67 CYXWIZ   234.56    │   │
│  │   ($0.47)      ($3.13)        ($9.13)        ($46.91)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  📊 Performance Score: 94/100                                   │
│  ├── Uptime: 99.7% ✓                                           │
│  ├── Response time: 45ms ✓                                     │
│  ├── Bandwidth: Good ✓                                         │
│  └── Stake: 100 CYXWIZ (minimum met)                           │
│                                                                 │
│  [💸 Withdraw Earnings] [⚙️ Settings] [📈 Analytics]            │
└─────────────────────────────────────────────────────────────────┘
```

### Staking & Slashing

```
Why stake?
- Proves commitment to the network
- Higher stake = higher priority for data placement
- Penalty for bad behavior (slashing)

Slashing conditions:
- Data loss (chunk unrecoverable): -10% stake
- Extended downtime (>24h): -5% stake
- Serving corrupted data: -50% stake
- Repeated failures: Node banned
```

---

## Economic Model

### Storage Pricing

```
Base Rate: X CYXWIZ / GB / month

Factors:
- Replication factor (higher = more expensive)
- Geographic distribution (global = premium)
- Retrieval speed tier (hot/warm/cold)
- Bandwidth usage

Example:
  10GB dataset, 3x replication, hot tier, US region
  = 10 * 0.01 * 3 * 1.5 * 1.0 = 0.45 CYXWIZ/month
```

### Node Rewards

```
Storage Nodes earn:
1. Storage fees (proportional to stored data)
2. Bandwidth fees (per GB transferred)
3. Availability bonus (uptime > 99.9%)
4. Retrieval speed bonus (fast response)

Penalties:
- Data loss: Slashed stake
- Downtime: Reduced rewards
- Slow response: Lower priority for new data
```

### Data Marketplace

```
Seller sets:
- Price per access (one-time or subscription)
- License type (personal/commercial/research)
- Usage restrictions

Platform takes:
- 5% marketplace fee
- Paid to treasury for network development

Buyer pays:
- Dataset price
- Gas fees (minimal on Solana)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (8-10 weeks)
- [ ] Storage node daemon (basic chunk storage)
- [ ] Content addressing (CID generation/verification)
- [ ] P2P networking (peer discovery, chunk transfer)
- [ ] Gateway API (upload/download)
- [ ] Basic CLI

### Phase 2: Redundancy & Reliability (6-8 weeks)
- [ ] Reed-Solomon erasure coding
- [ ] Multi-node replication
- [ ] Health monitoring
- [ ] Automatic repair (reconstruct lost chunks)
- [ ] Node reputation system

### Phase 3: Ecosystem Integration (6-8 weeks)
- [ ] Python SDK
- [ ] Engine integration (C++ client)
- [ ] Server Node integration
- [ ] Central Server coordination
- [ ] gRPC APIs

### Phase 4: Blockchain & Economy (6-8 weeks)
- [ ] Solana smart contracts
- [ ] Payment streams
- [ ] Access control on-chain
- [ ] Marketplace UI
- [ ] Staking mechanism

### Phase 5: Advanced Features (4-6 weeks)
- [ ] Client-side encryption
- [ ] Batch prefetching optimization
- [ ] Geographic routing
- [ ] Dataset versioning
- [ ] Collaborative datasets

---

## Third-Party Integration (S3-Compatible API)

CyxCloud provides an **S3-compatible API** so any application that works with AWS S3, Google Cloud Storage, or MinIO can use CyxCloud with minimal code changes.

### S3-Compatible Endpoints

```
Endpoint: https://s3.cyxcloud.io
Region: auto (routes to nearest nodes)

Supported Operations:
- PutObject / GetObject / DeleteObject
- ListBuckets / ListObjects
- CreateBucket / DeleteBucket
- Multipart uploads
- Presigned URLs
- Object versioning
```

### Integration Examples

#### AWS SDK (Python)
```python
import boto3

# Just change endpoint - same code works!
s3 = boto3.client(
    's3',
    endpoint_url='https://s3.cyxcloud.io',
    aws_access_key_id='your_cyxcloud_key',
    aws_secret_access_key='your_cyxcloud_secret'
)

# Upload file
s3.upload_file('local_file.zip', 'my-bucket', 'remote_file.zip')

# Download file
s3.download_file('my-bucket', 'remote_file.zip', 'downloaded.zip')
```

#### rclone (Command Line)
```bash
# Configure rclone
rclone config create cyxcloud s3 \
    provider=Other \
    endpoint=https://s3.cyxcloud.io \
    access_key_id=YOUR_KEY \
    secret_access_key=YOUR_SECRET

# Sync folder
rclone sync /local/folder cyxcloud:my-bucket/

# Mount as drive
rclone mount cyxcloud:my-bucket /mnt/cyxcloud
```

#### Docker Registry
```bash
# Use CyxCloud as Docker image storage
docker run -d -p 5000:5000 \
    -e REGISTRY_STORAGE=s3 \
    -e REGISTRY_STORAGE_S3_BUCKET=docker-images \
    -e REGISTRY_STORAGE_S3_REGION=auto \
    -e REGISTRY_STORAGE_S3_REGIONENDPOINT=https://s3.cyxcloud.io \
    registry:2
```

#### Backup Tools (Restic, Duplicati)
```bash
# Restic backup to CyxCloud
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

restic -r s3:s3.cyxcloud.io/backups init
restic -r s3:s3.cyxcloud.io/backups backup /home/user
```

### Use Cases for Third-Party Apps

| Application | How CyxCloud Helps |
|-------------|-------------------|
| **Backup Software** | Encrypted offsite backups with redundancy |
| **Media Servers** | Store video/music libraries (Plex, Jellyfin) |
| **Game Servers** | World saves, player data |
| **Databases** | Cold storage, backups |
| **CI/CD Pipelines** | Artifact storage |
| **Content Delivery** | Static assets, downloads |
| **Scientific Research** | Large dataset sharing |
| **Healthcare** | HIPAA-compliant medical records |

---

## Consumer Features (Google Drive Alternative)

### Desktop Sync Client

```
┌────────────────────────────────────────────────────────┐
│              CyxCloud Desktop App                       │
├────────────────────────────────────────────────────────┤
│  ☁️ CyxCloud                                           │
│  ├── 📁 Documents        ✓ Synced                      │
│  ├── 📁 Photos           ↻ Syncing (23%)               │
│  ├── 📁 Projects         ✓ Synced                      │
│  └── 📁 Shared with me   ✓ Synced                      │
│                                                        │
│  Storage: 45.2 GB / 100 GB used                        │
│  ━━━━━━━━━━━━━━━━━━━━━━░░░░░░░░░░                      │
│                                                        │
│  [⬆️ Upload] [📁 Open Folder] [⚙️ Settings]            │
└────────────────────────────────────────────────────────┘
```

**Features:**
- Automatic folder sync (like Google Drive / Dropbox)
- Selective sync (choose which folders to keep local)
- Smart sync (download on-demand, free up space)
- Conflict resolution
- Version history

### Web Interface

```
┌─────────────────────────────────────────────────────────────────┐
│  🌐 CyxCloud                        [Search...]    👤 Account   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📁 My Files  │  🔗 Shared  │  ⭐ Starred  │  🗑️ Trash          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Name              │ Size    │ Modified    │ Shared      │   │
│  ├───────────────────┼─────────┼─────────────┼─────────────┤   │
│  │ 📁 Documents      │ 1.2 GB  │ Today       │ Private     │   │
│  │ 📁 Projects       │ 5.4 GB  │ Yesterday   │ Team        │   │
│  │ 📄 report.pdf     │ 2.3 MB  │ Dec 5       │ Link shared │   │
│  │ 🖼️ photo.jpg      │ 4.1 MB  │ Dec 3       │ Private     │   │
│  │ 📊 data.csv       │ 156 MB  │ Dec 1       │ Private     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [+ New] [⬆️ Upload] [📥 Download] [🔗 Share] [🗑️ Delete]       │
└─────────────────────────────────────────────────────────────────┘
```

### Mobile Apps (iOS/Android)

- Photo/video auto-backup
- Offline access to selected files
- Share files via link
- Document scanner
- Media gallery

### Sharing & Collaboration

```
Share Options:
┌─────────────────────────────────────────────────┐
│  Share "project_files"                          │
├─────────────────────────────────────────────────┤
│                                                 │
│  👤 Add people:                                 │
│  ┌─────────────────────────────────────────┐   │
│  │ alice@email.com                     ✕   │   │
│  │ bob@company.com                     ✕   │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  Permission: [🔽 Can edit ▾]                    │
│                                                 │
│  ─────────── or ───────────                     │
│                                                 │
│  🔗 Get shareable link                          │
│  ┌─────────────────────────────────────────┐   │
│  │ https://cyxcloud.io/s/abc123xyz         │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ☐ Allow download                               │
│  ☐ Password protect                             │
│  ☐ Set expiration date                          │
│                                                 │
│  [Cancel]                      [Share]          │
└─────────────────────────────────────────────────┘
```

---

## Comparison with Alternatives

| Feature | CyxCloud | Google Drive | Dropbox | OneDrive | IPFS | Storj |
|---------|----------|--------------|---------|----------|------|-------|
| **Decentralized** | ✅ Yes | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **S3 Compatible** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Desktop Sync** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Mobile Apps** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **File Sharing** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **End-to-End Encryption** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Censorship Resistant** | ✅ Yes | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Earn by Hosting** | ✅ CYXWIZ | ❌ No | ❌ No | ❌ No | ❌ No | ✅ STORJ |
| **ML Optimized** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |
| **Data Marketplace** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |
| **Free Tier** | ✅ 5GB | ✅ 15GB | ✅ 2GB | ✅ 5GB | ∞ | ✅ 25GB |
| **Price/TB/mo** | ~$4 | $10 | $10 | $10 | Free* | $4 |

*IPFS doesn't guarantee persistence without pinning services

---

## Advanced Features

### 1. CDN / Edge Caching Layer

For global low-latency access, CyxCloud includes a CDN layer with edge Points of Presence (PoPs).

```
┌─────────────────────────────────────────────────────────────────┐
│                      CyxCloud CDN Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   🌍 Edge PoPs (Points of Presence)                             │
│                                                                  │
│   North America          Europe              Asia-Pacific        │
│   ├── 🇺🇸 US-East       ├── 🇩🇪 Frankfurt   ├── 🇯🇵 Tokyo       │
│   ├── 🇺🇸 US-West       ├── 🇬🇧 London      ├── 🇸🇬 Singapore   │
│   ├── 🇺🇸 US-Central    ├── 🇫🇷 Paris       ├── 🇦🇺 Sydney      │
│   └── 🇨🇦 Toronto       ├── 🇳🇱 Amsterdam   ├── 🇰🇷 Seoul       │
│                         └── 🇪🇸 Madrid      └── 🇮🇳 Mumbai      │
│                                                                  │
│   South America          Africa              Middle East         │
│   ├── 🇧🇷 São Paulo     ├── 🇿🇦 Johannesburg├── 🇦🇪 Dubai       │
│   └── 🇦🇷 Buenos Aires  └── 🇳🇬 Lagos       └── 🇮🇱 Tel Aviv    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

How it works:
1. User requests file from cyx://QmXXX/data.csv
2. Gateway routes to nearest edge PoP
3. If cached → serve immediately (<50ms)
4. If not cached → fetch from storage nodes, cache for future

Cache tiers:
- Hot (SSD): Frequently accessed, <10ms
- Warm (HDD): Moderate access, <50ms
- Cold (Archive): Rare access, <500ms
```

**Configuration:**
```python
# Force specific region
dataset = cyxcloud.Dataset(
    "cyx://QmXXX/data",
    preferred_region="eu-west",
    cache_tier="hot"
)

# Enable aggressive prefetching
dataset.enable_cdn_prefetch(ahead_chunks=10)
```

---

### 2. Data Versioning (Git for Datasets)

Full version control for datasets - track changes, branch, merge, time-travel.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dataset Version History                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Dataset: imagenet-cleaned                                       │
│  Current: v3.0.0 (cyx://QmABC...)                                │
│                                                                  │
│  ──●── v3.0.0 (HEAD) "Added 50k new validation images"          │
│    │   Dec 9, 2024 · 156 GB · +50,234 files                     │
│    │                                                             │
│  ──●── v2.1.0 "Fixed mislabeled categories"                     │
│    │   Nov 15, 2024 · 142 GB · ~1,234 files modified            │
│    │                                                             │
│  ──●── v2.0.0 "Major restructure - new category system"         │
│    │   Oct 1, 2024 · 140 GB · restructured                      │
│    │                                                             │
│  ──●── v1.0.0 "Initial release"                                 │
│        Aug 1, 2024 · 120 GB                                      │
│                                                                  │
│  [View diff] [Checkout version] [Create branch]                  │
└─────────────────────────────────────────────────────────────────┘
```

**Python API:**
```python
import cyxcloud

dataset = cyxcloud.Dataset("cyx://QmXXX/imagenet")

# View history
for version in dataset.history():
    print(f"{version.tag}: {version.message} ({version.date})")

# Create new version
dataset.add("new_images/")
dataset.commit("Added 10k new training images")
dataset.tag("v3.1.0")

# Time travel - load old version
old_data = dataset.checkout("v1.0.0")

# Diff between versions
changes = dataset.diff("v2.0.0", "v3.0.0")
print(f"Added: {changes.added} files")
print(f"Modified: {changes.modified} files")
print(f"Deleted: {changes.deleted} files")

# Branch for experiments
dataset.branch("experiment-augmentation")
# ... make changes ...
dataset.merge("main")  # Merge back

# Fork dataset (create your own copy)
my_fork = dataset.fork("my-imagenet-variant")
```

**Storage efficiency:**
```
Deduplication + Delta encoding

v1.0: 120 GB (full storage)
v2.0: +20 GB delta (only changes stored)
v3.0: +16 GB delta

Total storage: 156 GB (not 416 GB!)
```

---

### 3. Federated Learning Support

Train models on distributed data WITHOUT moving the data. Critical for privacy-sensitive applications.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Federated Learning Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Data stays at source - only model updates are shared          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Hospital A  │  │  Hospital B  │  │  Hospital C  │          │
│  │              │  │              │  │              │          │
│  │  📊 Patient  │  │  📊 Patient  │  │  📊 Patient  │          │
│  │     Data     │  │     Data     │  │     Data     │          │
│  │  (private)   │  │  (private)   │  │  (private)   │          │
│  │              │  │              │  │              │          │
│  │  🔄 Local    │  │  🔄 Local    │  │  🔄 Local    │          │
│  │   Training   │  │   Training   │  │   Training   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         │  Model weights  │  Model weights  │                   │
│         │  (encrypted)    │  (encrypted)    │                   │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │   Aggregator    │                            │
│                  │   (Central)     │                            │
│                  │                 │                            │
│                  │  Combine model  │                            │
│                  │  updates using  │                            │
│                  │  FedAvg/FedProx │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │  Global Model   │                            │
│                  │  (improved)     │                            │
│                  └─────────────────┘                            │
│                                                                  │
│   ✅ Data never leaves the hospital                             │
│   ✅ Compliant with HIPAA, GDPR                                 │
│   ✅ Each participant benefits from collective learning         │
└─────────────────────────────────────────────────────────────────┘
```

**Setup Federated Training:**
```python
# In CyxWiz Engine - create federated job
from cyxwiz import FederatedTraining

fed_job = FederatedTraining(
    model=my_model,
    aggregation="fedavg",  # or "fedprox", "scaffold"
    rounds=100,
    min_participants=3,
    privacy={
        "differential_privacy": True,
        "epsilon": 1.0,
        "secure_aggregation": True
    }
)

# Register data sources (they keep their data)
fed_job.add_participant("hospital_a", data_cid="cyx://QmAAA/patients")
fed_job.add_participant("hospital_b", data_cid="cyx://QmBBB/patients")
fed_job.add_participant("hospital_c", data_cid="cyx://QmCCC/patients")

# Start federated training
result = fed_job.train()
# Each hospital trains locally, only gradients are shared
```

**Use cases:**
- 🏥 Healthcare: Train on patient data across hospitals
- 🏦 Finance: Fraud detection across banks
- 📱 Mobile: Learn from user data on devices
- 🏢 Enterprise: Cross-department analytics

---

### 4. Hybrid Compute + Storage Nodes

Nodes that provide BOTH compute and storage get data locality benefits.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Node Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Traditional (separate):          Hybrid (combined):           │
│                                                                  │
│   ┌──────────┐    Network    ┌──────────────────────────────┐  │
│   │ Compute  │◄─────────────►│        Hybrid Node           │  │
│   │ Node     │    transfer   │                              │  │
│   │ (GPU)    │               │  ┌────────┐    ┌──────────┐  │  │
│   └──────────┘               │  │  GPU   │◄──►│ Storage  │  │  │
│        │                     │  │Compute │    │ 100TB    │  │  │
│   Network                    │  │        │    │ RAID     │  │  │
│   latency                    │  └────────┘    └──────────┘  │  │
│        │                     │                              │  │
│   ┌──────────┐               │  Local I/O = Fast!           │  │
│   │ Storage  │               │                              │  │
│   │ Node     │               │  Earnings:                   │  │
│   │ (HDD)    │               │  💰 Storage fees             │  │
│   └──────────┘               │  💰 Compute fees             │  │
│                              │  💰 Locality bonus           │  │
│   ❌ Slow network transfer   └──────────────────────────────┘  │
│   ✅ Fast local access                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Benefits:
- Train on data without downloading (already local)
- Lower latency = faster training
- Earn from BOTH storage and compute
- Central Server prioritizes jobs to nodes with data
```

**Configuration:**
```yaml
# cyxcloud-node.yaml
mode: hybrid

storage:
  path: /mnt/raid
  allocated: 50TB

compute:
  gpus: [0, 1, 2, 3]  # 4x RTX 4090
  max_jobs: 4

locality_bonus: true  # Prefer jobs for data we store
```

**Job scheduling with locality:**
```rust
// Central Server scheduler
fn find_optimal_node(job: &Job) -> Node {
    let data_cid = &job.data_cid;

    // First: try nodes that HAVE the data locally
    if let Some(node) = find_node_with_data(data_cid) {
        return node;  // Zero network transfer!
    }

    // Second: try nodes NEAR nodes with data
    if let Some(node) = find_node_near_data(data_cid) {
        return node;  // Minimal transfer
    }

    // Fallback: any available node
    find_any_available_node()
}
```

---

### 5. Data Lineage & Provenance

Track the complete history of data: where it came from, how it was transformed, who touched it.

```
┌─────────────────────────────────────────────────────────────────┐
│              Data Lineage: medical_xrays_processed               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📥 SOURCE                                                       │
│  │                                                               │
│  ├── hospital_a_raw_scans                                       │
│  │   └── Origin: St. Mary's Hospital PACS                       │
│  │   └── Date: 2024-01-15                                       │
│  │   └── Records: 50,234 DICOM files                            │
│  │   └── IRB Approval: #2024-0123                               │
│  │                                                               │
│  ▼ TRANSFORM: anonymize_dicom_v2                                │
│  │   └── Script: cyx://QmXXX/scripts/anonymize.py               │
│  │   └── Date: 2024-01-16 14:32:00 UTC                          │
│  │   └── Operator: alice@research.org                           │
│  │   └── Actions:                                               │
│  │       ├── Removed: PatientName, PatientID, DOB               │
│  │       ├── Hashed: AccessionNumber (SHA-256)                  │
│  │       └── Retained: Modality, BodyPart, StudyDate            │
│  │                                                               │
│  ▼ TRANSFORM: quality_filter_v1                                 │
│  │   └── Script: cyx://QmYYY/scripts/filter.py                  │
│  │   └── Date: 2024-01-17 09:15:00 UTC                          │
│  │   └── Actions:                                               │
│  │       ├── Removed: 1,234 low-quality images                  │
│  │       └── Kept: 48,990 images (97.5%)                        │
│  │                                                               │
│  ▼ TRANSFORM: normalize_resize                                  │
│  │   └── Script: cyx://QmZZZ/scripts/preprocess.py              │
│  │   └── Date: 2024-01-18 11:00:00 UTC                          │
│  │   └── Actions:                                               │
│  │       ├── Resized: 512x512 → 256x256                         │
│  │       ├── Normalized: [0, 255] → [0, 1]                      │
│  │       └── Format: DICOM → PNG                                │
│  │                                                               │
│  ▼ CURRENT: medical_xrays_processed                             │
│      └── CID: cyx://QmABC123.../                                │
│      └── Size: 12.4 GB                                          │
│      └── Files: 48,990                                          │
│      └── Compliance: ✅ HIPAA, ✅ GDPR                          │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  🔐 Verification:                                               │
│  ├── All transformations cryptographically signed               │
│  ├── Hash chain verified: ✅                                    │
│  └── Audit log immutable (on-chain)                             │
│                                                                  │
│  [Export Lineage Report] [Verify Chain] [View Audit Log]        │
└─────────────────────────────────────────────────────────────────┘
```

**API:**
```python
dataset = cyxcloud.Dataset("cyx://QmABC/medical_xrays_processed")

# Get full lineage
lineage = dataset.lineage()
for step in lineage:
    print(f"{step.type}: {step.name}")
    print(f"  Date: {step.timestamp}")
    print(f"  Operator: {step.operator}")
    print(f"  Script: {step.script_cid}")

# Verify integrity
assert dataset.verify_lineage()  # Checks hash chain

# Export for compliance
dataset.export_lineage_report("lineage_report.pdf")

# Record new transformation
dataset.record_transform(
    name="augmentation_v1",
    script="cyx://QmXXX/augment.py",
    description="Applied random rotations and flips",
    operator="bob@research.org"
)
```

---

### 6. AI-Powered Smart Features

Automatic intelligence applied to stored data.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Smart Data Features                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🏷️ AUTO-TAGGING                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Uploaded: vacation_photos.zip                            │   │
│  │                                                          │   │
│  │ AI detected tags:                                        │   │
│  │ [beach] [sunset] [people] [outdoor] [summer]            │   │
│  │ [tropical] [water] [palm trees]                         │   │
│  │                                                          │   │
│  │ Suggested categories: Travel > Vacation > Beach          │   │
│  │                                                          │   │
│  │ [Accept] [Edit tags] [Disable auto-tag]                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  🔍 DUPLICATE DETECTION                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ⚠️ Similar file detected!                                │   │
│  │                                                          │   │
│  │ Uploading: dataset_v2.csv (1.2 GB)                      │   │
│  │ Existing:  dataset_final.csv (1.2 GB)                   │   │
│  │                                                          │   │
│  │ Similarity: 98.7%                                        │   │
│  │ Difference: 156 rows modified, 23 rows added            │   │
│  │                                                          │   │
│  │ [Upload as new version] [Replace] [Cancel]              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  🔎 SMART SEARCH                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Search: "medical images with lungs from 2024"           │   │
│  │                                                          │   │
│  │ Results:                                                 │   │
│  │ 📁 chest_xrays_2024/ - 12,345 images                    │   │
│  │ 📁 lung_ct_scans/ - 5,678 images                        │   │
│  │ 📁 covid_dataset_v3/ - 8,901 images                     │   │
│  │                                                          │   │
│  │ Also try: "chest radiographs", "pulmonary imaging"      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  📊 SCHEMA DETECTION (for tabular data)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Uploaded: sales_data.csv                                 │   │
│  │                                                          │   │
│  │ Detected schema:                                         │   │
│  │ ├── date (datetime) - parsed from "MM/DD/YYYY"          │   │
│  │ ├── product_id (string) - 1,234 unique values           │   │
│  │ ├── quantity (integer) - range [1, 999]                 │   │
│  │ ├── price (float) - range [$0.99, $9999.99]             │   │
│  │ ├── region (category) - [NA, EU, APAC, LATAM]           │   │
│  │ └── customer_id (string) - PII detected ⚠️              │   │
│  │                                                          │   │
│  │ Suggestions:                                             │   │
│  │ • Consider anonymizing customer_id                       │   │
│  │ • 2.3% missing values in 'region' column                │   │
│  │                                                          │   │
│  │ [Apply suggestions] [Ignore]                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  💡 RECOMMENDATIONS                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Based on your data, you might like:                      │   │
│  │                                                          │   │
│  │ 📊 Similar public datasets:                              │   │
│  │    • ImageNet-21k (14M images) - FREE                   │   │
│  │    • COCO 2024 (330k images) - FREE                     │   │
│  │                                                          │   │
│  │ 🛠️ Preprocessing scripts:                               │   │
│  │    • image_augmentation.py (★ 4.8)                      │   │
│  │    • noise_reduction.py (★ 4.5)                         │   │
│  │                                                          │   │
│  │ 📈 Pre-trained models:                                   │   │
│  │    • ResNet-50 (ImageNet) - Compatible                   │   │
│  │    • EfficientNet-B4 - Compatible                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 7. Enterprise Private Clusters

Dedicated infrastructure for organizations with strict requirements.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Enterprise Private Cluster                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  🏢 Acme Corporation - Private Cloud                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │  📍 Region: EU-West (Frankfurt)                         │   │
│  │     Data sovereignty: Germany/EU only                    │   │
│  │                                                          │   │
│  │  🖥️ Infrastructure:                                      │   │
│  │     Dedicated nodes: 25                                  │   │
│  │     Total storage: 500 TB                               │   │
│  │     Network: 10 Gbps dedicated                          │   │
│  │                                                          │   │
│  │  📊 Current usage:                                       │   │
│  │     Storage used: 234 TB (47%)                          │   │
│  │     Bandwidth: 2.3 TB/day                               │   │
│  │     Active users: 156                                    │   │
│  │                                                          │   │
│  │  🔒 Security:                                            │   │
│  │     ✅ VPN/Private network access only                  │   │
│  │     ✅ SSO integration (Okta, Azure AD)                 │   │
│  │     ✅ Audit logging enabled                            │   │
│  │     ✅ Data encrypted at rest (AES-256)                 │   │
│  │                                                          │   │
│  │  📜 Compliance:                                          │   │
│  │     ✅ GDPR certified                                   │   │
│  │     ✅ SOC 2 Type II                                    │   │
│  │     ✅ ISO 27001                                        │   │
│  │     ✅ HIPAA BAA available                              │   │
│  │                                                          │   │
│  │  💰 Billing:                                             │   │
│  │     Plan: Enterprise (annual)                           │   │
│  │     Monthly cost: $4,500                                │   │
│  │     SLA: 99.99% uptime guaranteed                       │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Network Isolation:                                              │
│                                                                  │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │   Public         │      │   Acme Private   │                │
│  │   CyxCloud       │ ═══X═│   Cluster        │                │
│  │   Network        │      │                  │                │
│  │                  │      │   🔒 Isolated    │                │
│  └──────────────────┘      └──────────────────┘                │
│         │                          │                            │
│         │                     VPN only                          │
│         │                          │                            │
│    Public users              Acme employees                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Enterprise features:**
```yaml
# Enterprise cluster configuration
cluster:
  name: acme-private
  type: dedicated

regions:
  - eu-west-1  # Primary
  - eu-west-2  # Disaster recovery

nodes:
  count: 25
  type: dedicated  # Not shared

network:
  type: private
  vpn_required: true
  allowed_ips:
    - 10.0.0.0/8
    - 192.168.1.0/24

security:
  encryption: aes-256-gcm
  key_management: customer-managed  # BYOK
  sso_provider: okta
  mfa_required: true
  audit_log: enabled

compliance:
  - gdpr
  - soc2
  - iso27001
  - hipaa

sla:
  uptime: 99.99%
  support: 24/7
  response_time: 1h (critical), 4h (high)
```

---

### 8. Real-Time Data Streams

Beyond static files - support for streaming data ingestion and consumption.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Real-Time Data Streams                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Use cases:                                                      │
│  • IoT sensor data → ML training pipeline                       │
│  • Live video feeds for real-time inference                     │
│  • Financial market data streaming                              │
│  • Log aggregation and analysis                                 │
│  • Social media firehose                                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │   Producers              Stream              Consumers   │   │
│  │                                                          │   │
│  │  ┌─────────┐         ┌──────────┐        ┌─────────┐   │   │
│  │  │ IoT     │────────►│          │───────►│ ML      │   │   │
│  │  │ Sensors │         │ CyxCloud │        │ Model   │   │   │
│  │  └─────────┘         │ Stream   │        └─────────┘   │   │
│  │                      │          │                       │   │
│  │  ┌─────────┐         │ Topics:  │        ┌─────────┐   │   │
│  │  │ Cameras │────────►│ -sensors │───────►│ Analytics│   │   │
│  │  │         │         │ -video   │        │ Dashboard│   │   │
│  │  └─────────┘         │ -logs    │        └─────────┘   │   │
│  │                      │          │                       │   │
│  │  ┌─────────┐         │          │        ┌─────────┐   │   │
│  │  │ App     │────────►│          │───────►│ Archive │   │   │
│  │  │ Logs    │         │          │        │ Storage │   │   │
│  │  └─────────┘         └──────────┘        └─────────┘   │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Python API:**
```python
import cyxcloud

# Create a stream
stream = cyxcloud.Stream.create(
    name="iot-sensors",
    retention="7d",  # Keep 7 days of data
    partitions=8
)

# Producer - send data
producer = stream.producer()
while True:
    data = read_sensor()
    producer.send({
        "sensor_id": "temp-001",
        "value": data.temperature,
        "timestamp": time.time()
    })

# Consumer - receive data
consumer = stream.consumer(group="ml-training")
async for record in consumer:
    model.update(record)  # Online learning

# Batch consumer - for training
batch_consumer = stream.batch_consumer(
    start_time="2024-12-01",
    end_time="2024-12-09",
    batch_size=1000
)
for batch in batch_consumer:
    train_batch(model, batch)

# Archive stream to storage (for later replay)
stream.archive_to("cyx://QmXXX/iot-archive/")
```

---

### 9. Data Quality & Validation

Automatic data quality checks and validation rules.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Data Quality Dashboard                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Dataset: customer_transactions.csv                              │
│  Last scan: 2024-12-09 10:30:00 UTC                             │
│                                                                  │
│  Overall Score: 87/100  [████████████████████░░░░]              │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  ✅ PASSED (7)                                                  │
│  ├── Schema validation: Matches expected schema                 │
│  ├── Row count: 1,234,567 (expected: >1M)                      │
│  ├── Column count: 15 (expected: 15)                           │
│  ├── Date format: Consistent ISO-8601                          │
│  ├── Encoding: UTF-8                                           │
│  ├── No duplicates: 0 duplicate rows                           │
│  └── Referential integrity: All foreign keys valid             │
│                                                                  │
│  ⚠️ WARNINGS (3)                                                │
│  ├── Missing values: 2.3% in 'region' column                   │
│  │   └── Recommendation: Impute or remove                      │
│  ├── Outliers: 156 values in 'amount' > 3σ                     │
│  │   └── Recommendation: Review for data entry errors          │
│  └── Skewed distribution: 'category' is 80% "electronics"      │
│      └── Recommendation: Consider stratified sampling          │
│                                                                  │
│  ❌ FAILED (1)                                                  │
│  └── PII detected: 'email' column contains email addresses     │
│      └── Action required: Anonymize before sharing             │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  [📥 Download Report] [🔧 Auto-Fix Issues] [📧 Send Alert]      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Validation rules:**
```python
import cyxcloud
from cyxcloud.quality import ValidationRule, Schema

# Define expected schema
schema = Schema({
    "customer_id": {"type": "string", "pattern": r"^CUST-\d{6}$"},
    "email": {"type": "string", "pii": True},
    "amount": {"type": "float", "min": 0, "max": 100000},
    "date": {"type": "datetime", "format": "ISO-8601"},
    "region": {"type": "category", "values": ["NA", "EU", "APAC"]}
})

# Define validation rules
rules = [
    ValidationRule("no_duplicates", columns=["customer_id", "date"]),
    ValidationRule("no_nulls", columns=["customer_id", "amount"]),
    ValidationRule("outlier_check", columns=["amount"], method="zscore", threshold=3),
    ValidationRule("freshness", max_age_days=7),
]

# Validate dataset
dataset = cyxcloud.Dataset("cyx://QmXXX/transactions")
report = dataset.validate(schema=schema, rules=rules)

print(f"Score: {report.score}/100")
print(f"Passed: {len(report.passed)}")
print(f"Warnings: {len(report.warnings)}")
print(f"Failed: {len(report.failed)}")

# Auto-fix issues
dataset.fix_issues(
    impute_missing="median",
    remove_outliers=True,
    anonymize_pii=True
)

# Set up continuous monitoring
dataset.enable_monitoring(
    check_interval="1h",
    alert_on_failure=True,
    alert_email="data-team@company.com"
)
```

---

### 10. ML Ops Integration

CyxCloud as the data backbone for the full ML lifecycle.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML Ops Integration                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      CyxCloud Storage Layer                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │  📊 Raw Data    🔄 Processed    📦 Features    🎯 Models │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│         │                │               │              │        │
│         │                │               │              │        │
│         ▼                ▼               ▼              ▼        │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐  │
│   │  Data    │    │  Data    │    │ Feature  │   │  Model   │  │
│   │ Ingestion│───►│Processing│───►│  Store   │──►│ Registry │  │
│   │          │    │  (ETL)   │    │          │   │          │  │
│   └──────────┘    └──────────┘    └──────────┘   └──────────┘  │
│                                          │              │        │
│                                          │              │        │
│                                          ▼              ▼        │
│                                    ┌──────────┐   ┌──────────┐  │
│                                    │Experiment│   │  Model   │  │
│                                    │ Tracking │   │ Serving  │  │
│                                    │(MLflow)  │   │          │  │
│                                    └──────────┘   └──────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Integrations:
├── MLflow - Experiment tracking
├── DVC - Data version control
├── Weights & Biases - Visualization
├── Kubeflow - Orchestration
├── Apache Airflow - Pipelines
└── CyxWiz Engine - Native support
```

**Feature Store:**
```python
import cyxcloud

# Register features
feature_store = cyxcloud.FeatureStore("cyx://QmXXX/features")

feature_store.register(
    name="customer_features",
    schema={
        "customer_id": "string",
        "total_purchases": "float",
        "avg_order_value": "float",
        "days_since_last_order": "int",
        "preferred_category": "string"
    },
    source="cyx://QmYYY/raw_transactions",
    transformation="cyx://QmZZZ/scripts/compute_features.py",
    update_frequency="daily"
)

# Use features in training
features = feature_store.get_features(
    feature_names=["total_purchases", "avg_order_value"],
    entity_ids=customer_ids,
    point_in_time="2024-12-01"  # Avoid data leakage
)
```

**Model Registry:**
```python
# Register trained model
model_registry = cyxcloud.ModelRegistry("cyx://QmXXX/models")

model_registry.register(
    name="churn_predictor",
    version="1.2.0",
    model_path="./model.pkl",
    metrics={
        "accuracy": 0.92,
        "f1": 0.89,
        "auc": 0.95
    },
    training_data="cyx://QmYYY/training_data_v3",
    features=["total_purchases", "avg_order_value", "days_since_last_order"],
    tags=["production-ready", "churn"]
)

# Load model for inference
model = model_registry.load("churn_predictor", version="latest")
predictions = model.predict(features)

# Model lineage
lineage = model_registry.get_lineage("churn_predictor:1.2.0")
print(f"Trained on: {lineage.training_data}")
print(f"Features from: {lineage.feature_store}")
print(f"Preprocessing: {lineage.transforms}")
```

---

## Open Questions

1. **Proof of Storage**: How to verify nodes actually store data? (Filecoin uses complex proofs)
2. **Incentive Balance**: How to balance storage vs compute rewards?
3. **Cross-chain Bridge**: Support Polygon/Ethereum in addition to Solana?
4. **Privacy**: How to enable ML on encrypted data? (Homomorphic encryption? TEEs?)
5. **Large Files**: Special handling for 100GB+ datasets?
6. **Regulatory**: How to handle GDPR "right to be forgotten" on immutable storage?

---

## Next Steps

1. Review this document with team
2. Finalize tech stack decisions
3. Create cyxcloud repository
4. Set up CI/CD pipeline
5. Begin Phase 1 implementation
6. Define testnet parameters

---

## References

- [IPFS Documentation](https://docs.ipfs.tech/)
- [Filecoin Spec](https://spec.filecoin.io/)
- [Storj Whitepaper](https://www.storj.io/whitepaper)
- [Reed-Solomon Erasure Coding](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction)
- [libp2p Specs](https://github.com/libp2p/specs)
- [Solana Cookbook](https://solanacookbook.com/)
