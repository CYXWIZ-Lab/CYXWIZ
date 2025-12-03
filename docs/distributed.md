# CyxWiz Distributed Data Architecture

This document describes the distributed data loading, transfer, and security architecture for the CyxWiz decentralized ML compute platform.

## Table of Contents

1. [Overview](#1-overview)
2. [Data Flow Architecture](#2-data-flow-architecture)
3. [Data Sources](#3-data-sources)
4. [Security Architecture](#4-security-architecture)
5. [Data Transfer Protocol](#5-data-transfer-protocol)
6. [Sharding Strategies](#6-sharding-strategies)
7. [Server Node Data Handling](#7-server-node-data-handling)
8. [Integration with Current Workflow](#8-integration-with-current-workflow)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Overview

### 1.1 Vision

CyxWiz is a decentralized ML compute platform where:
- **Engine** (Desktop Client): Creates models, submits training jobs
- **Server Nodes**: Execute training on distributed hardware
- **Central Server**: Orchestrates job assignment and node discovery

Data must flow securely between these components with:
- **Integrity**: Data arrives uncorrupted (FCS/checksums)
- **Confidentiality**: Sensitive data is encrypted
- **Authenticity**: Verify data source is trusted
- **Availability**: Resilient to node failures

### 1.2 Key Principles

| Principle | Description |
|-----------|-------------|
| **Data Locality** | Move computation to data when possible, not data to computation |
| **Minimal Transfer** | Only transfer what's needed (shards, not full datasets) |
| **Zero-Trust** | Verify everything, trust nothing by default |
| **End-to-End Encryption** | Data encrypted from source to destination |
| **Verifiable Integrity** | Every transfer verified with checksums/FCS |

---

## 2. Data Flow Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CyxWiz Distributed Data Flow                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         DATA SOURCES                                      │   │
│  │                                                                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │   │
│  │  │  Local  │  │  IPFS   │  │   S3/   │  │HuggingF.│  │  CyxWiz Node   │ │   │
│  │  │  Files  │  │         │  │  Cloud  │  │ Kaggle  │  │    Network     │ │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └───────┬────────┘ │   │
│  │       │            │            │            │                │          │   │
│  └───────┼────────────┼────────────┼────────────┼────────────────┼──────────┘   │
│          │            │            │            │                │              │
│          └────────────┴────────────┴────────────┴────────────────┘              │
│                                    │                                             │
│                                    ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      DATA TRANSFER LAYER                                  │   │
│  │                                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │  │                    Secure Transport (TLS 1.3)                        │ │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │ │   │
│  │  │  │ Encryption  │  │    FCS/     │  │   Chunk    │  │  Compress  │  │ │   │
│  │  │  │  (AES-256)  │  │  Checksum   │  │   Manager  │  │   (zstd)   │  │ │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │ │   │
│  │  └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                             │
│          ┌─────────────────────────┼─────────────────────────┐                  │
│          │                         │                         │                  │
│          ▼                         ▼                         ▼                  │
│  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐          │
│  │    Engine     │         │Central Server │         │  Server Node  │          │
│  │   (Client)    │◄───────▶│ (Orchestrator)│◄───────▶│   (Worker)    │          │
│  │               │         │               │         │               │          │
│  │ • Model design│         │ • Job routing │         │ • Training    │          │
│  │ • Job submit  │         │ • Node registry│        │ • Data cache  │          │
│  │ • Results view│         │ • Data routing│         │ • GPU compute │          │
│  └───────────────┘         └───────────────┘         └───────────────┘          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Modes

#### Mode 1: Direct Transfer (Engine → Server Node)

```
┌────────────┐                              ┌─────────────┐
│   Engine   │                              │ Server Node │
│            │                              │             │
│  Dataset   │ ─────── gRPC Stream ───────▶ │   Cache     │
│  (Local)   │      (encrypted chunks)      │             │
└────────────┘                              └─────────────┘

Use when:
• Small datasets (< 1GB)
• User owns the data
• Private/sensitive data
```

#### Mode 2: Reference Transfer (URI Only)

```
┌────────────┐          ┌─────────────┐          ┌─────────────┐
│   Engine   │          │   Central   │          │ Server Node │
│            │          │   Server    │          │             │
│  Submit:   │ ──────▶  │  Route URI  │ ──────▶  │  Download   │
│  uri://... │          │             │          │  from URI   │
└────────────┘          └─────────────┘          └─────────────┘

Use when:
• Large public datasets (ImageNet, COCO)
• Datasets on cloud storage (S3, GCS)
• IPFS-hosted datasets
• HuggingFace/Kaggle datasets
```

#### Mode 3: Distributed Sharding

```
                         ┌─────────────┐
                         │   Central   │
                         │   Server    │
                         │ (Scheduler) │
                         └──────┬──────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Server Node │     │ Server Node │     │ Server Node │
    │     A       │     │     B       │     │     C       │
    │             │     │             │     │             │
    │ Shard 0-33% │     │ Shard 33-66%│     │ Shard 66-100│
    │  (20K imgs) │     │  (20K imgs) │     │  (20K imgs) │
    └─────────────┘     └─────────────┘     └─────────────┘

Use when:
• Very large datasets (100GB+)
• Distributed training across multiple nodes
• Data parallelism strategies
```

---

## 3. Data Sources

### 3.1 Supported Sources

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCE REGISTRY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  URI Scheme          Provider              Security Level                   │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  file://             Local filesystem      ████████████ (Full control)      │
│  cyxwiz://           CyxWiz P2P network    ██████████░░ (E2E encrypted)     │
│  ipfs://             IPFS decentralized    ████████░░░░ (Content-addressed) │
│  s3://               AWS S3                ██████░░░░░░ (IAM + encryption)  │
│  gs://               Google Cloud Storage  ██████░░░░░░ (IAM + encryption)  │
│  az://               Azure Blob Storage    ██████░░░░░░ (IAM + encryption)  │
│  hf://               HuggingFace Hub       ████░░░░░░░░ (Public/Token)      │
│  kaggle://           Kaggle Datasets       ████░░░░░░░░ (API key)           │
│  http(s)://          Generic URL           ██░░░░░░░░░░ (TLS only)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 URI Protocol Specification

```
cyxwiz://<node_id>/<dataset_name>[@<version>][?params]

Examples:
  cyxwiz://node-abc123/mnist                     # Latest version
  cyxwiz://node-abc123/mnist@v3                  # Specific version
  cyxwiz://node-abc123/mnist@v3?shard=0-1000     # Specific samples
  cyxwiz://node-abc123/mnist?split=train         # Specific split

IPFS:
  ipfs://QmYwAPJzv5CZsnA.../mnist.parquet        # Content-addressed

Cloud:
  s3://my-bucket/datasets/training.parquet       # AWS S3
  gs://my-bucket/datasets/training.parquet       # Google Cloud
  az://container/datasets/training.parquet       # Azure Blob
```

### 3.3 Source Configuration

```cpp
// cyxwiz-engine/src/core/data_source.h

namespace cyxwiz {

enum class DataSourceType {
    Local,          // file://
    CyxWizNode,     // cyxwiz://
    IPFS,           // ipfs://
    S3,             // s3://
    GoogleCloud,    // gs://
    AzureBlob,      // az://
    HuggingFace,    // hf://
    Kaggle,         // kaggle://
    HTTP            // http:// or https://
};

struct DataSourceConfig {
    DataSourceType type;
    std::string uri;

    // Authentication
    std::string auth_token;         // API token
    std::string access_key;         // S3/Cloud access key
    std::string secret_key;         // S3/Cloud secret key
    std::string credentials_file;   // Path to credentials JSON

    // Transfer options
    bool enable_compression = true;
    std::string compression_codec = "zstd";  // zstd, lz4, snappy
    size_t chunk_size = 4 * 1024 * 1024;     // 4MB chunks
    int max_retries = 3;
    int timeout_seconds = 300;

    // Cache options
    bool enable_cache = true;
    std::string cache_dir;          // Local cache directory
    size_t cache_max_size = 10ULL * 1024 * 1024 * 1024;  // 10GB

    // Security options
    bool verify_checksum = true;
    bool require_encryption = false;
    std::string encryption_key;     // For encrypted sources
};

class DataSource {
public:
    virtual ~DataSource() = default;

    // Metadata
    virtual DataSourceType GetType() const = 0;
    virtual std::string GetURI() const = 0;
    virtual bool IsAvailable() const = 0;

    // Data access
    virtual std::shared_ptr<Dataset> Load(
        const std::string& uri,
        ProgressCallback progress = nullptr) = 0;

    virtual bool SupportsStreaming() const = 0;
    virtual std::unique_ptr<DataStream> OpenStream(
        const std::string& uri,
        size_t start_offset = 0) = 0;

    // Chunked access
    virtual std::vector<DataChunk> GetChunks(
        const std::string& uri,
        size_t chunk_size) = 0;

    virtual DataChunk FetchChunk(
        const std::string& uri,
        size_t chunk_index) = 0;
};

} // namespace cyxwiz
```

---

## 4. Security Architecture

### 4.1 Security Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY LAYER STACK                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer 7: Application Security                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • User authentication (JWT tokens)                                  │    │
│  │  • Access control lists (who can access what data)                   │    │
│  │  • Audit logging (track all data access)                             │    │
│  │  • Rate limiting (prevent abuse)                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Layer 6: Data Security                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • End-to-end encryption (AES-256-GCM)                               │    │
│  │  • Data classification (public/private/sensitive)                    │    │
│  │  • Data anonymization (PII removal)                                  │    │
│  │  • Encryption at rest (cached data)                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Layer 5: Transfer Security                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Frame Check Sequence (FCS) - CRC-32/CRC-64                        │    │
│  │  • SHA-256 checksums per chunk                                       │    │
│  │  • Merkle tree for dataset integrity                                 │    │
│  │  • Replay attack prevention (nonces)                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Layer 4: Transport Security                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • TLS 1.3 for all gRPC connections                                  │    │
│  │  • Certificate pinning                                               │    │
│  │  • mTLS (mutual TLS) for node-to-node                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Layer 3: Node Security                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Node identity verification (Ed25519 keys)                         │    │
│  │  • Reputation system (trust scores)                                  │    │
│  │  • Sandboxed execution (Docker/VM)                                   │    │
│  │  • Secure enclave support (SGX/TrustZone)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Frame Check Sequence (FCS) Implementation

FCS ensures data integrity at the frame/chunk level during transfer.

```cpp
// cyxwiz-protocol/include/fcs.h

#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <string>

namespace cyxwiz::security {

/**
 * CRC-32 implementation for Frame Check Sequence
 * Used for quick integrity checks during streaming transfer
 */
class CRC32 {
public:
    CRC32();

    // Process data incrementally
    void Update(const uint8_t* data, size_t length);
    void Update(const std::vector<uint8_t>& data);

    // Get final CRC value
    uint32_t Finalize();

    // One-shot calculation
    static uint32_t Calculate(const uint8_t* data, size_t length);
    static uint32_t Calculate(const std::vector<uint8_t>& data);

    // Verify data against expected CRC
    static bool Verify(const uint8_t* data, size_t length, uint32_t expected_crc);

private:
    uint32_t crc_;
    static const std::array<uint32_t, 256> table_;
    static std::array<uint32_t, 256> GenerateTable();
};

/**
 * SHA-256 for cryptographic integrity verification
 * Used for chunk-level verification after transfer
 */
class SHA256 {
public:
    static constexpr size_t DIGEST_SIZE = 32;
    using Digest = std::array<uint8_t, DIGEST_SIZE>;

    SHA256();

    void Update(const uint8_t* data, size_t length);
    void Update(const std::vector<uint8_t>& data);

    Digest Finalize();

    static Digest Calculate(const uint8_t* data, size_t length);
    static Digest Calculate(const std::vector<uint8_t>& data);
    static std::string DigestToHex(const Digest& digest);
    static bool Verify(const uint8_t* data, size_t length, const Digest& expected);

private:
    // Implementation uses OpenSSL or similar
    void* ctx_;  // EVP_MD_CTX*
};

/**
 * Merkle Tree for dataset-level integrity
 * Enables verification of individual chunks without full dataset
 */
class MerkleTree {
public:
    MerkleTree() = default;

    // Build tree from chunk hashes
    void Build(const std::vector<SHA256::Digest>& chunk_hashes);

    // Get root hash (represents entire dataset)
    SHA256::Digest GetRoot() const { return root_; }

    // Get proof for a specific chunk
    std::vector<SHA256::Digest> GetProof(size_t chunk_index) const;

    // Verify a chunk using proof
    static bool VerifyProof(
        const SHA256::Digest& chunk_hash,
        size_t chunk_index,
        const std::vector<SHA256::Digest>& proof,
        const SHA256::Digest& root);

    // Serialize/deserialize
    std::vector<uint8_t> Serialize() const;
    static MerkleTree Deserialize(const std::vector<uint8_t>& data);

private:
    SHA256::Digest root_;
    std::vector<std::vector<SHA256::Digest>> levels_;
};

/**
 * Data Frame with FCS
 * Wire format for data transfer
 */
struct DataFrame {
    // Header (24 bytes)
    uint32_t magic;           // 0x43595857 ("CYXW")
    uint32_t version;         // Protocol version
    uint64_t sequence;        // Frame sequence number (anti-replay)
    uint32_t payload_size;    // Size of payload
    uint32_t flags;           // Frame flags

    // Payload
    std::vector<uint8_t> payload;

    // Trailer (36 bytes)
    uint32_t fcs;             // CRC-32 of header + payload
    SHA256::Digest checksum;  // SHA-256 of payload

    // Serialize to wire format
    std::vector<uint8_t> Serialize() const;

    // Deserialize and verify
    static std::optional<DataFrame> Deserialize(
        const std::vector<uint8_t>& data,
        bool verify_fcs = true);

    // Frame flags
    enum Flags : uint32_t {
        FLAG_COMPRESSED     = 0x0001,
        FLAG_ENCRYPTED      = 0x0002,
        FLAG_LAST_FRAME     = 0x0004,
        FLAG_REQUIRES_ACK   = 0x0008,
        FLAG_RETRANSMIT     = 0x0010,
    };
};

} // namespace cyxwiz::security
```

### 4.3 Data Classification & Handling

```cpp
// cyxwiz-engine/src/core/data_classification.h

namespace cyxwiz::security {

/**
 * Data sensitivity levels
 */
enum class DataClassification {
    Public,         // Freely shareable (ImageNet, MNIST)
    Internal,       // Organization internal only
    Confidential,   // Sensitive business data
    Restricted,     // PII, PHI, regulated data
    TopSecret       // Highest security (encrypted at all times)
};

/**
 * Data handling policy based on classification
 */
struct DataPolicy {
    DataClassification classification;

    // Storage
    bool encrypt_at_rest;           // Encrypt cached data
    bool encrypt_in_transit;        // Always encrypted during transfer
    size_t max_cache_time_hours;    // Auto-delete after N hours

    // Transfer
    bool allow_third_party_nodes;   // Can untrusted nodes process?
    bool require_secure_enclave;    // Require SGX/TrustZone?
    int min_reputation_score;       // Minimum node reputation

    // Access
    bool require_mfa;               // Multi-factor auth required?
    bool log_all_access;            // Full audit trail
    std::vector<std::string> allowed_regions;  // Geographic restrictions

    // Cleanup
    bool secure_delete;             // Overwrite on delete
    bool auto_expire;               // Auto-delete after job completion

    // Default policies
    static DataPolicy ForClassification(DataClassification level);
};

/**
 * Sensitive data detector
 * Scans data for PII and other sensitive patterns
 */
class SensitiveDataDetector {
public:
    struct Detection {
        std::string field_name;
        std::string pattern_matched;
        size_t row_index;
        DataClassification suggested_level;
    };

    // Scan dataset for sensitive data
    std::vector<Detection> Scan(const Dataset& dataset);

    // Built-in patterns
    static constexpr const char* PATTERN_EMAIL = R"(\b[\w.+-]+@[\w-]+\.[\w.-]+\b)";
    static constexpr const char* PATTERN_PHONE = R"(\b\d{3}[-.]?\d{3}[-.]?\d{4}\b)";
    static constexpr const char* PATTERN_SSN = R"(\b\d{3}-\d{2}-\d{4}\b)";
    static constexpr const char* PATTERN_CREDIT_CARD = R"(\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b)";
    static constexpr const char* PATTERN_IP_ADDRESS = R"(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)";

    // Add custom patterns
    void AddPattern(const std::string& name, const std::string& regex,
                    DataClassification level);

private:
    std::vector<std::tuple<std::string, std::regex, DataClassification>> patterns_;
};

/**
 * Data anonymization utilities
 */
class DataAnonymizer {
public:
    // Anonymization strategies
    enum class Strategy {
        Mask,           // Replace with ***
        Hash,           // One-way hash
        Generalize,     // "John Smith" -> "Person"
        Suppress,       // Remove entirely
        Noise,          // Add random noise
        KAnonymity      // Ensure k-anonymity
    };

    // Anonymize specific columns
    void AnonymizeColumn(Dataset& dataset, const std::string& column,
                         Strategy strategy);

    // Anonymize based on detected sensitive data
    void AnonymizeDetected(Dataset& dataset,
                           const std::vector<SensitiveDataDetector::Detection>& detections);
};

} // namespace cyxwiz::security
```

### 4.4 Encryption

```cpp
// cyxwiz-protocol/include/encryption.h

namespace cyxwiz::security {

/**
 * AES-256-GCM encryption for data at rest and in transit
 */
class AES256GCM {
public:
    static constexpr size_t KEY_SIZE = 32;    // 256 bits
    static constexpr size_t NONCE_SIZE = 12;  // 96 bits
    static constexpr size_t TAG_SIZE = 16;    // 128 bits

    using Key = std::array<uint8_t, KEY_SIZE>;
    using Nonce = std::array<uint8_t, NONCE_SIZE>;
    using Tag = std::array<uint8_t, TAG_SIZE>;

    explicit AES256GCM(const Key& key);

    // Encrypt data
    struct EncryptedData {
        Nonce nonce;
        Tag tag;
        std::vector<uint8_t> ciphertext;
    };

    EncryptedData Encrypt(
        const uint8_t* plaintext, size_t length,
        const uint8_t* aad = nullptr, size_t aad_length = 0);

    // Decrypt data
    std::optional<std::vector<uint8_t>> Decrypt(
        const EncryptedData& encrypted,
        const uint8_t* aad = nullptr, size_t aad_length = 0);

    // Streaming encryption for large data
    class StreamEncryptor {
    public:
        StreamEncryptor(const Key& key, const Nonce& nonce);
        void Update(const uint8_t* data, size_t length,
                    std::vector<uint8_t>& output);
        Tag Finalize(std::vector<uint8_t>& output);
    };

    class StreamDecryptor {
    public:
        StreamDecryptor(const Key& key, const Nonce& nonce);
        void Update(const uint8_t* data, size_t length,
                    std::vector<uint8_t>& output);
        bool Finalize(const Tag& tag, std::vector<uint8_t>& output);
    };

private:
    Key key_;
};

/**
 * Key exchange for establishing session keys
 * Uses X25519 (Curve25519 ECDH)
 */
class KeyExchange {
public:
    static constexpr size_t PUBLIC_KEY_SIZE = 32;
    static constexpr size_t SECRET_KEY_SIZE = 32;
    static constexpr size_t SHARED_SECRET_SIZE = 32;

    using PublicKey = std::array<uint8_t, PUBLIC_KEY_SIZE>;
    using SecretKey = std::array<uint8_t, SECRET_KEY_SIZE>;
    using SharedSecret = std::array<uint8_t, SHARED_SECRET_SIZE>;

    // Generate key pair
    static std::pair<PublicKey, SecretKey> GenerateKeyPair();

    // Derive shared secret
    static SharedSecret DeriveSharedSecret(
        const SecretKey& my_secret,
        const PublicKey& their_public);

    // Derive AES key from shared secret using HKDF
    static AES256GCM::Key DeriveAESKey(
        const SharedSecret& shared_secret,
        const std::string& context = "cyxwiz-data-transfer");
};

/**
 * Node identity using Ed25519 signatures
 */
class NodeIdentity {
public:
    static constexpr size_t PUBLIC_KEY_SIZE = 32;
    static constexpr size_t SECRET_KEY_SIZE = 64;
    static constexpr size_t SIGNATURE_SIZE = 64;

    using PublicKey = std::array<uint8_t, PUBLIC_KEY_SIZE>;
    using SecretKey = std::array<uint8_t, SECRET_KEY_SIZE>;
    using Signature = std::array<uint8_t, SIGNATURE_SIZE>;

    // Generate identity key pair
    static std::pair<PublicKey, SecretKey> GenerateIdentity();

    // Sign data
    static Signature Sign(const SecretKey& secret_key,
                          const uint8_t* data, size_t length);

    // Verify signature
    static bool Verify(const PublicKey& public_key,
                       const uint8_t* data, size_t length,
                       const Signature& signature);

    // Node ID is hash of public key
    static std::string PublicKeyToNodeId(const PublicKey& public_key);
};

} // namespace cyxwiz::security
```

---

## 5. Data Transfer Protocol

### 5.1 Protocol Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA TRANSFER PROTOCOL STACK                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Application Layer                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DataTransferService (gRPC)                                          │    │
│  │  • InitiateTransfer    • StreamChunks    • ConfirmReceipt           │    │
│  │  • GetTransferStatus   • CancelTransfer  • RetryChunk               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Framing Layer                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┬─────────┐ │    │
│  │  │  Magic   │ Version  │ Sequence │  Size    │  Flags   │ Payload │ │    │
│  │  │ (4 bytes)│ (4 bytes)│ (8 bytes)│ (4 bytes)│ (4 bytes)│ (var)   │ │    │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘ │    │
│  │  ┌──────────┬──────────────────────────────────────────────────────┐ │    │
│  │  │   FCS    │                    SHA-256 Checksum                  │ │    │
│  │  │ (4 bytes)│                      (32 bytes)                      │ │    │
│  │  └──────────┴──────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Security Layer                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • AES-256-GCM encryption (optional, per data classification)       │    │
│  │  • X25519 key exchange for session keys                              │    │
│  │  • Ed25519 signatures for authentication                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Compression Layer                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • zstd (default) - best compression ratio                          │    │
│  │  • lz4 - fastest for real-time streaming                            │    │
│  │  • snappy - balanced                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Transport Layer                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  gRPC over TLS 1.3 (HTTP/2)                                          │    │
│  │  • Bidirectional streaming                                           │    │
│  │  • Flow control                                                      │    │
│  │  • Multiplexing                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 gRPC Service Definition

```protobuf
// cyxwiz-protocol/proto/data_transfer.proto

syntax = "proto3";
package cyxwiz.data;

import "google/protobuf/timestamp.proto";

// Data Transfer Service
service DataTransferService {
    // Initiate a new transfer
    rpc InitiateTransfer(TransferRequest) returns (TransferResponse);

    // Stream data chunks (bidirectional)
    rpc StreamData(stream DataChunk) returns (stream TransferAck);

    // Get transfer status
    rpc GetTransferStatus(TransferStatusRequest) returns (TransferStatus);

    // Cancel ongoing transfer
    rpc CancelTransfer(CancelRequest) returns (CancelResponse);

    // Request specific chunk (for retransmission)
    rpc RequestChunk(ChunkRequest) returns (DataChunk);

    // Verify data integrity after transfer
    rpc VerifyIntegrity(IntegrityRequest) returns (IntegrityResponse);
}

// Transfer request
message TransferRequest {
    string transfer_id = 1;           // Unique transfer ID
    string source_uri = 2;            // Source data URI
    string destination_node = 3;      // Target node ID

    // Data metadata
    DatasetMetadata metadata = 4;

    // Security options
    SecurityOptions security = 5;

    // Transfer options
    TransferOptions options = 6;
}

message DatasetMetadata {
    string name = 1;
    string type = 2;                  // CSV, Parquet, MNIST, etc.
    uint64 total_size = 3;            // Total bytes
    uint64 num_samples = 4;
    repeated uint64 shape = 5;        // Sample shape
    string dtype = 6;                 // Data type

    // Integrity
    bytes merkle_root = 7;            // Merkle tree root hash
    uint32 num_chunks = 8;
    uint32 chunk_size = 9;

    // Classification
    DataClassification classification = 10;
}

enum DataClassification {
    PUBLIC = 0;
    INTERNAL = 1;
    CONFIDENTIAL = 2;
    RESTRICTED = 3;
    TOP_SECRET = 4;
}

message SecurityOptions {
    bool encrypt_payload = 1;
    bytes encryption_key = 2;         // Encrypted with recipient's public key
    bytes sender_public_key = 3;
    bytes signature = 4;              // Signature of metadata
}

message TransferOptions {
    string compression = 1;           // zstd, lz4, snappy, none
    uint32 chunk_size = 2;            // Bytes per chunk
    uint32 max_retries = 3;
    uint32 timeout_seconds = 4;
    bool verify_chunks = 5;           // Verify each chunk on receive
    ShardingInfo sharding = 6;        // For partial transfers
}

message ShardingInfo {
    uint64 start_sample = 1;
    uint64 end_sample = 2;
    uint32 shard_index = 3;
    uint32 total_shards = 4;
}

// Data chunk
message DataChunk {
    string transfer_id = 1;
    uint32 chunk_index = 2;
    uint64 sequence = 3;              // Anti-replay sequence number

    bytes payload = 4;                // Actual data (possibly encrypted)

    // Integrity
    uint32 fcs = 5;                   // CRC-32 of payload
    bytes checksum = 6;               // SHA-256 of payload

    // Merkle proof for verification
    repeated bytes merkle_proof = 7;

    // Flags
    bool is_last = 8;
    bool is_retransmit = 9;
}

// Transfer acknowledgment
message TransferAck {
    string transfer_id = 1;
    uint32 chunk_index = 2;
    AckStatus status = 3;
    string error_message = 4;
}

enum AckStatus {
    ACK_OK = 0;
    ACK_CHECKSUM_MISMATCH = 1;
    ACK_FCS_ERROR = 2;
    ACK_DECRYPT_ERROR = 3;
    ACK_OUT_OF_ORDER = 4;
    ACK_DUPLICATE = 5;
}

// Transfer status
message TransferStatus {
    string transfer_id = 1;
    TransferState state = 2;
    uint32 chunks_received = 3;
    uint32 chunks_total = 4;
    uint64 bytes_transferred = 5;
    uint64 bytes_total = 6;
    double transfer_rate_mbps = 7;
    google.protobuf.Timestamp started_at = 8;
    google.protobuf.Timestamp eta = 9;
    repeated uint32 missing_chunks = 10;
}

enum TransferState {
    PENDING = 0;
    IN_PROGRESS = 1;
    VERIFYING = 2;
    COMPLETED = 3;
    FAILED = 4;
    CANCELLED = 5;
}

// Integrity verification
message IntegrityRequest {
    string transfer_id = 1;
    bool full_verification = 2;       // Verify all chunks vs quick check
}

message IntegrityResponse {
    bool valid = 1;
    bytes computed_merkle_root = 2;
    repeated uint32 corrupt_chunks = 3;
    string error_message = 4;
}

// Other messages
message TransferResponse {
    bool accepted = 1;
    string transfer_id = 2;
    string error_message = 3;
    bytes recipient_public_key = 4;   // For key exchange
}

message TransferStatusRequest {
    string transfer_id = 1;
}

message CancelRequest {
    string transfer_id = 1;
    string reason = 2;
}

message CancelResponse {
    bool success = 1;
    string message = 2;
}

message ChunkRequest {
    string transfer_id = 1;
    uint32 chunk_index = 2;
}
```

### 5.3 Transfer Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA TRANSFER SEQUENCE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Engine                    Central Server              Server Node         │
│      │                            │                           │              │
│      │  1. SubmitJob(dataset_uri) │                           │              │
│      │ ─────────────────────────► │                           │              │
│      │                            │                           │              │
│      │                            │  2. AssignNode()          │              │
│      │                            │ ─────────────────────────►│              │
│      │                            │                           │              │
│      │  3. GetNodeEndpoint()      │                           │              │
│      │ ◄───────────────────────── │                           │              │
│      │                            │                           │              │
│      │  4. KeyExchange (X25519)                               │              │
│      │ ◄─────────────────────────────────────────────────────►│              │
│      │                            │                           │              │
│      │  5. InitiateTransfer(metadata, merkle_root)            │              │
│      │ ─────────────────────────────────────────────────────► │              │
│      │                            │                           │              │
│      │  6. TransferResponse(accepted, recipient_pubkey)       │              │
│      │ ◄───────────────────────────────────────────────────── │              │
│      │                            │                           │              │
│      │  7. StreamData (chunks with FCS + checksum)            │              │
│      │ ═════════════════════════════════════════════════════► │              │
│      │                            │                           │              │
│      │  8. TransferAck (per chunk)                            │              │
│      │ ◄═════════════════════════════════════════════════════ │              │
│      │                            │                           │              │
│      │  ... repeat for all chunks ...                         │              │
│      │                            │                           │              │
│      │  9. VerifyIntegrity()                                  │              │
│      │ ─────────────────────────────────────────────────────► │              │
│      │                            │                           │              │
│      │  10. IntegrityResponse(merkle_root_match)              │              │
│      │ ◄───────────────────────────────────────────────────── │              │
│      │                            │                           │              │
│      │                            │  11. DataReady(job_id)    │              │
│      │                            │ ◄───────────────────────── │              │
│      │                            │                           │              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Chunk Verification Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CHUNK VERIFICATION PROCESS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Received Chunk                                                             │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                        │
│   │  1. Check FCS   │ ──── CRC-32 mismatch ────► Request Retransmit         │
│   │    (CRC-32)     │                                                        │
│   └────────┬────────┘                                                        │
│            │ OK                                                              │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │ 2. Verify SHA256│ ──── Hash mismatch ──────► Request Retransmit         │
│   │   Checksum      │                                                        │
│   └────────┬────────┘                                                        │
│            │ OK                                                              │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │ 3. Check Seq#   │ ──── Out of order ───────► Buffer / Reorder           │
│   │   (Anti-replay) │ ──── Duplicate ──────────► Discard                    │
│   └────────┬────────┘                                                        │
│            │ OK                                                              │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │ 4. Decrypt      │ ──── Decrypt failed ─────► Request Retransmit         │
│   │   (if encrypted)│                                                        │
│   └────────┬────────┘                                                        │
│            │ OK                                                              │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │ 5. Decompress   │ ──── Decompress failed ──► Request Retransmit         │
│   │   (if compressed│                                                        │
│   └────────┬────────┘                                                        │
│            │ OK                                                              │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │ 6. Verify Merkle│ ──── Proof invalid ──────► Request Retransmit         │
│   │   Proof         │       (data tampered)                                  │
│   └────────┬────────┘                                                        │
│            │ OK                                                              │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │ 7. Store Chunk  │                                                        │
│   │   & Send ACK    │                                                        │
│   └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Sharding Strategies

### 6.1 Sharding Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHARDING STRATEGIES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Strategy 1: SAMPLE SHARDING (Default)                                       │
│  ─────────────────────────────────────                                       │
│  Split dataset by sample index                                               │
│                                                                              │
│  Dataset: [S0, S1, S2, S3, S4, S5, S6, S7, S8, S9]                          │
│                      │                                                       │
│        ┌─────────────┼─────────────┐                                        │
│        ▼             ▼             ▼                                        │
│    ┌───────┐    ┌───────┐    ┌───────┐                                      │
│    │Node A │    │Node B │    │Node C │                                      │
│    │S0-S2  │    │S3-S5  │    │S6-S9  │                                      │
│    └───────┘    └───────┘    └───────┘                                      │
│                                                                              │
│  Best for: Homogeneous data, data parallelism                               │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Strategy 2: CLASS-BALANCED SHARDING                                         │
│  ────────────────────────────────────                                        │
│  Each shard has equal class distribution                                     │
│                                                                              │
│  Classes: [A A A B B B C C C] ──► Shuffle by class ──►                      │
│                                                                              │
│    ┌───────┐    ┌───────┐    ┌───────┐                                      │
│    │Node A │    │Node B │    │Node C │                                      │
│    │A, B, C│    │A, B, C│    │A, B, C│                                      │
│    └───────┘    └───────┘    └───────┘                                      │
│                                                                              │
│  Best for: Classification tasks, imbalanced datasets                        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Strategy 3: HASH-BASED SHARDING                                             │
│  ───────────────────────────────                                             │
│  Consistent hashing for reproducibility                                      │
│                                                                              │
│  shard_index = hash(sample_id) % num_shards                                 │
│                                                                              │
│  Best for: Deterministic distribution, cache-friendly                       │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Strategy 4: FEATURE SHARDING (Vertical)                                     │
│  ───────────────────────────────────────                                     │
│  Split by features, not samples                                              │
│                                                                              │
│  Features: [F1 F2 F3 F4 F5 F6]                                              │
│                                                                              │
│    ┌───────┐    ┌───────┐    ┌───────┐                                      │
│    │Node A │    │Node B │    │Node C │                                      │
│    │F1, F2 │    │F3, F4 │    │F5, F6 │                                      │
│    │(all S)│    │(all S)│    │(all S)│                                      │
│    └───────┘    └───────┘    └───────┘                                      │
│                                                                              │
│  Best for: Wide datasets, model parallelism                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Sharding Configuration

```cpp
// cyxwiz-engine/src/core/sharding.h

namespace cyxwiz {

enum class ShardingStrategy {
    Sample,         // Split by sample index (default)
    ClassBalanced,  // Maintain class distribution
    Hash,           // Consistent hash-based
    Feature,        // Vertical partitioning
    Custom          // User-defined
};

struct ShardingConfig {
    ShardingStrategy strategy = ShardingStrategy::Sample;
    int num_shards = 1;
    int seed = 42;                    // For reproducibility

    // For ClassBalanced
    bool maintain_class_ratio = true;

    // For Hash
    std::string hash_column;          // Column to hash on

    // For Feature
    std::vector<std::vector<std::string>> feature_groups;

    // For Custom
    std::function<int(size_t sample_index)> custom_sharding_fn;
};

class DataShardManager {
public:
    // Create shards from dataset
    std::vector<ShardInfo> CreateShards(
        const Dataset& dataset,
        const ShardingConfig& config);

    // Get samples for a specific shard
    std::vector<size_t> GetShardIndices(
        const Dataset& dataset,
        int shard_index,
        const ShardingConfig& config);

    // Validate shard assignment
    bool ValidateSharding(
        const std::vector<ShardInfo>& shards,
        size_t total_samples);

    // Rebalance shards if nodes change
    std::vector<ShardInfo> RebalanceShards(
        const std::vector<ShardInfo>& current_shards,
        const std::vector<std::string>& available_nodes);
};

struct ShardInfo {
    int shard_index;
    std::string assigned_node;
    std::vector<size_t> sample_indices;  // Or range
    size_t start_index;
    size_t end_index;
    size_t num_samples;
    SHA256::Digest checksum;             // Integrity
};

} // namespace cyxwiz
```

---

## 7. Server Node Data Handling

### 7.1 Data Lifecycle on Server Node

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   SERVER NODE DATA LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DATA RECEPTION                                                           │
│  ─────────────────                                                           │
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │   Receive   │────▶│   Verify    │────▶│   Decrypt   │                    │
│  │   Chunks    │     │  FCS + Hash │     │  (if enc.)  │                    │
│  └─────────────┘     └─────────────┘     └─────────────┘                    │
│         │                                       │                            │
│         │ Stream to disk                        │                            │
│         ▼                                       ▼                            │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │              SECURE TEMP STORAGE                     │                    │
│  │   /tmp/cyxwiz/{job_id}/data.parquet                 │                    │
│  │   (encrypted at rest if sensitive)                   │                    │
│  └─────────────────────────────────────────────────────┘                    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  2. DATA PROCESSING                                                          │
│  ──────────────────                                                          │
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │    Load     │────▶│   Batch     │────▶│   Train     │                    │
│  │  to Memory  │     │   Iterator  │     │   Model     │                    │
│  └─────────────┘     └─────────────┘     └─────────────┘                    │
│         │                                       │                            │
│         │ LRU Cache                             │ GPU Memory                 │
│         ▼                                       ▼                            │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │              MEMORY MANAGEMENT                       │                    │
│  │   • Lazy loading with sample cache                  │                    │
│  │   • Prefetch next batch                             │                    │
│  │   • GPU memory pinning                              │                    │
│  └─────────────────────────────────────────────────────┘                    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  3. DATA CLEANUP                                                             │
│  ───────────────                                                             │
│                                                                              │
│  On job completion or failure:                                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │              SECURE CLEANUP                          │                    │
│  │                                                      │                    │
│  │  if (data.classification >= CONFIDENTIAL):          │                    │
│  │      secure_erase(temp_storage)  # Overwrite        │                    │
│  │  else:                                               │                    │
│  │      if (cache_policy.allow_caching):               │                    │
│  │          move_to_cache(temp_storage)                │                    │
│  │      else:                                           │                    │
│  │          delete(temp_storage)                       │                    │
│  │                                                      │                    │
│  └─────────────────────────────────────────────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Server Node Storage Architecture

```cpp
// cyxwiz-server-node/src/storage/data_storage.h

namespace cyxwiz::server {

/**
 * Secure storage configuration
 */
struct StorageConfig {
    std::string base_path = "/var/cyxwiz/data";
    std::string temp_path = "/tmp/cyxwiz";

    // Size limits
    size_t max_storage_bytes = 100ULL * 1024 * 1024 * 1024;  // 100GB
    size_t max_temp_bytes = 50ULL * 1024 * 1024 * 1024;      // 50GB

    // Security
    bool encrypt_at_rest = true;
    bool secure_delete = true;     // Overwrite on delete
    std::string encryption_key;    // Derived from node identity

    // Caching
    bool enable_cache = true;
    size_t cache_max_bytes = 20ULL * 1024 * 1024 * 1024;  // 20GB
    int cache_ttl_hours = 24;
};

/**
 * Data storage manager for server node
 */
class DataStorageManager {
public:
    explicit DataStorageManager(const StorageConfig& config);
    ~DataStorageManager();

    // Job data management
    std::string AllocateJobStorage(const std::string& job_id,
                                    size_t estimated_size);

    bool WriteChunk(const std::string& job_id,
                    uint32_t chunk_index,
                    const std::vector<uint8_t>& data);

    bool FinalizeJobData(const std::string& job_id);

    // Load for processing
    std::shared_ptr<Dataset> LoadJobData(const std::string& job_id);

    // Cleanup
    void CleanupJobData(const std::string& job_id, bool secure = false);
    void CleanupExpiredCache();

    // Cache management
    bool IsCached(const std::string& dataset_uri);
    std::string GetCachePath(const std::string& dataset_uri);
    void CacheDataset(const std::string& dataset_uri,
                      const std::string& source_path);

    // Storage stats
    size_t GetUsedStorage() const;
    size_t GetAvailableStorage() const;
    size_t GetCacheSize() const;

private:
    StorageConfig config_;
    std::map<std::string, std::string> job_storage_paths_;
    std::mutex mutex_;

    // Encryption helpers
    std::vector<uint8_t> EncryptData(const std::vector<uint8_t>& plaintext);
    std::vector<uint8_t> DecryptData(const std::vector<uint8_t>& ciphertext);

    // Secure deletion
    void SecureDelete(const std::string& path);
};

/**
 * Download manager for external data sources
 */
class DataDownloadManager {
public:
    struct DownloadTask {
        std::string uri;
        std::string destination;
        size_t expected_size;
        SHA256::Digest expected_checksum;

        // Progress
        std::atomic<size_t> bytes_downloaded{0};
        std::atomic<DownloadState> state{DownloadState::Pending};
    };

    enum class DownloadState {
        Pending,
        Downloading,
        Verifying,
        Completed,
        Failed
    };

    // Download from external source
    std::future<bool> DownloadAsync(
        const std::string& uri,
        const std::string& destination,
        const SHA256::Digest& expected_checksum);

    // Download with progress callback
    bool Download(
        const std::string& uri,
        const std::string& destination,
        ProgressCallback progress = nullptr);

    // Verify downloaded data
    bool Verify(const std::string& path,
                const SHA256::Digest& expected_checksum);

private:
    // Source-specific downloaders
    bool DownloadHTTP(const std::string& url, const std::string& dest);
    bool DownloadS3(const std::string& uri, const std::string& dest);
    bool DownloadIPFS(const std::string& cid, const std::string& dest);
    bool DownloadGRPC(const std::string& node_uri, const std::string& dest);
};

} // namespace cyxwiz::server
```

### 7.3 Data Access Control on Server Node

```cpp
// cyxwiz-server-node/src/security/access_control.h

namespace cyxwiz::server::security {

/**
 * Access control for data on server node
 */
class DataAccessControl {
public:
    // Verify job has access to data
    bool CanAccessData(const std::string& job_id,
                       const std::string& data_uri);

    // Verify node is authorized to receive data
    bool CanReceiveData(const std::string& sender_node_id,
                        DataClassification classification);

    // Check if node meets security requirements
    bool MeetsSecurityRequirements(const DataPolicy& policy);

    // Get node's trust level
    int GetTrustLevel() const;

    // Audit logging
    void LogDataAccess(const std::string& job_id,
                       const std::string& data_uri,
                       const std::string& action);

    void LogDataTransfer(const std::string& source_node,
                         const std::string& data_uri,
                         size_t bytes,
                         bool success);
};

/**
 * Sandboxed execution environment
 */
class ExecutionSandbox {
public:
    enum class SandboxType {
        None,           // Direct execution (trusted)
        Docker,         // Docker container
        gVisor,         // gVisor sandbox
        Firecracker,    // Firecracker microVM
        SGX             // Intel SGX enclave
    };

    // Create sandbox for job
    std::unique_ptr<ExecutionSandbox> Create(
        SandboxType type,
        const std::string& job_id);

    // Mount data read-only inside sandbox
    bool MountData(const std::string& host_path,
                   const std::string& sandbox_path,
                   bool read_only = true);

    // Execute training inside sandbox
    int Execute(const std::string& command);

    // Cleanup sandbox
    void Destroy();

private:
    SandboxType type_;
    std::string container_id_;
};

} // namespace cyxwiz::server::security
```

---

## 8. Integration with Current Workflow

### 8.1 Integration Points

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED DATA INTEGRATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT COMPONENT              DISTRIBUTED EXTENSION                        │
│  ─────────────────              ──────────────────────                        │
│                                                                              │
│  DataRegistry                   + RemoteDatasetHandle                        │
│  ├── LoadDataset(path)          + LoadDataset(uri)                           │
│  ├── GetDataset(name)           + GetRemoteDataset(node, name)               │
│  └── ListDatasets()             + ListRemoteDatasets(node)                   │
│                                                                              │
│  DatasetPanel (GUI)             + Remote Datasets Tab                        │
│  ├── Load buttons               + Connect to Node button                     │
│  ├── Preview                    + Remote preview                             │
│  └── Split config               + Sharding config                            │
│                                                                              │
│  NodeEditor                     + Data Source Node                           │
│  ├── Input node                 + Remote Input node                          │
│  └── Dataset config             + URI config                                 │
│                                                                              │
│  TrainingManager                + DistributedTrainingManager                 │
│  ├── StartTraining()            + StartDistributedTraining()                 │
│  └── Local execution            + Remote execution with data transfer        │
│                                                                              │
│  JobManager                     + DataTransferJob                            │
│  ├── SubmitJob()                + PrepareDataTransfer()                      │
│  └── Job status                 + Transfer progress                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 DataRegistry Extensions

```cpp
// Extensions to data_registry.h

namespace cyxwiz {

/**
 * Remote dataset handle
 */
class RemoteDatasetHandle {
public:
    RemoteDatasetHandle(const std::string& node_id,
                        const std::string& dataset_name);

    bool IsValid() const;
    std::string GetNodeId() const { return node_id_; }
    std::string GetDatasetName() const { return dataset_name_; }
    std::string GetURI() const;

    // Remote operations
    DatasetInfo GetInfo() const;  // Fetches from remote
    DatasetPreview GetPreview(int max_samples = 5) const;

    // Transfer to local
    DatasetHandle TransferToLocal(
        const std::string& local_name,
        ProgressCallback progress = nullptr);

    // Shard access
    DatasetHandle GetShard(int shard_index, int num_shards);

private:
    std::string node_id_;
    std::string dataset_name_;
    mutable std::optional<DatasetInfo> cached_info_;
};

// Extended DataRegistry methods
class DataRegistry {
public:
    // ... existing methods ...

    // Remote dataset operations
    RemoteDatasetHandle ConnectToRemote(const std::string& uri);

    std::vector<RemoteDatasetHandle> ListRemoteDatasets(
        const std::string& node_id);

    DatasetHandle LoadFromURI(
        const std::string& uri,
        const std::string& local_name = "",
        ProgressCallback progress = nullptr);

    // Distributed training support
    std::vector<ShardInfo> CreateDistributedShards(
        const std::string& name,
        int num_shards,
        ShardingStrategy strategy = ShardingStrategy::Sample);

    bool TransferShard(
        const std::string& name,
        int shard_index,
        const std::string& target_node,
        ProgressCallback progress = nullptr);
};

} // namespace cyxwiz
```

### 8.3 GUI Integration

```cpp
// Extensions to dataset_panel.cpp

void DatasetPanel::RenderDistributedTab() {
    ImGui::Text("Remote Datasets");
    ImGui::Separator();

    // Node connection
    static char node_address[256] = "node-abc123.cyxwiz.network:50051";
    ImGui::InputText("Node Address", node_address, sizeof(node_address));

    if (ImGui::Button("Connect")) {
        ConnectToNode(node_address);
    }

    ImGui::SameLine();
    if (connected_node_.has_value()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "Connected");
    }

    ImGui::Spacing();
    ImGui::Separator();

    // Remote datasets list
    if (connected_node_.has_value()) {
        ImGui::Text("Available Datasets:");

        for (const auto& remote : remote_datasets_) {
            bool selected = (selected_remote_ == remote.GetDatasetName());

            if (ImGui::Selectable(remote.GetDatasetName().c_str(), selected)) {
                selected_remote_ = remote.GetDatasetName();
                FetchRemotePreview(remote);
            }
        }

        ImGui::Spacing();

        if (!selected_remote_.empty()) {
            // Transfer options
            ImGui::Text("Transfer Options:");

            static bool transfer_full = true;
            ImGui::Checkbox("Full Dataset", &transfer_full);

            if (!transfer_full) {
                static int shard_index = 0;
                static int num_shards = 4;
                ImGui::SliderInt("Shard", &shard_index, 0, num_shards - 1);
                ImGui::SliderInt("Total Shards", &num_shards, 2, 16);
            }

            if (ImGui::Button("Transfer to Local", ImVec2(-1, 0))) {
                TransferRemoteDataset(selected_remote_, !transfer_full);
            }

            // Transfer progress
            if (transfer_in_progress_) {
                ImGui::ProgressBar(transfer_progress_,
                    ImVec2(-1, 0), transfer_status_.c_str());
            }
        }
    }
}

void DatasetPanel::RenderShardingConfig() {
    ImGui::Text("Sharding Configuration");
    ImGui::Separator();

    static int sharding_strategy = 0;
    const char* strategies[] = {
        "Sample (Default)", "Class Balanced", "Hash-Based", "Feature"
    };
    ImGui::Combo("Strategy", &sharding_strategy, strategies, 4);

    static int num_shards = 4;
    ImGui::SliderInt("Number of Shards", &num_shards, 2, 32);

    if (ImGui::Button("Preview Sharding")) {
        PreviewSharding(static_cast<ShardingStrategy>(sharding_strategy),
                        num_shards);
    }

    // Show shard distribution
    if (!shard_preview_.empty()) {
        ImGui::Spacing();
        ImGui::Text("Shard Distribution:");

        for (const auto& shard : shard_preview_) {
            ImGui::Text("  Shard %d: %zu samples",
                shard.shard_index, shard.num_samples);
        }
    }
}
```

---

## 9. Implementation Roadmap

### Phase 5A: Core Infrastructure

```
Tasks:
├── [ ] Implement FCS/CRC-32 verification
├── [ ] Implement SHA-256 checksum system
├── [ ] Create DataFrame wire format
├── [ ] Add Merkle tree for dataset integrity
└── [ ] Implement basic encryption (AES-256-GCM)
```

### Phase 5B: Data Source Abstraction

```
Tasks:
├── [ ] Define DataSource interface
├── [ ] Implement LocalDataSource (file://)
├── [ ] Implement HTTPDataSource (https://)
├── [ ] Add URI parsing and routing
└── [ ] Create DataSourceFactory
```

### Phase 5C: gRPC Transfer Service

```
Tasks:
├── [ ] Define data_transfer.proto
├── [ ] Implement DataTransferService (Engine side)
├── [ ] Implement DataTransferService (Node side)
├── [ ] Add streaming with backpressure
└── [ ] Implement retry logic
```

### Phase 5D: Sharding

```
Tasks:
├── [ ] Implement ShardingConfig
├── [ ] Add sample sharding strategy
├── [ ] Add class-balanced sharding
├── [ ] Create DataShardManager
└── [ ] Test distributed training with shards
```

### Phase 5E: Security Hardening

```
Tasks:
├── [ ] Implement X25519 key exchange
├── [ ] Add Ed25519 node identity
├── [ ] Implement data classification
├── [ ] Add sensitive data detection
├── [ ] Implement secure cleanup
```

### Phase 5F: External Sources

```
Tasks:
├── [ ] Implement IPFS data source
├── [ ] Implement S3 data source
├── [ ] Add HuggingFace/Kaggle sources
├── [ ] Create download manager
└── [ ] Add caching layer
```

### Phase 5G: GUI Integration

```
Tasks:
├── [ ] Add Remote Datasets tab
├── [ ] Add transfer progress UI
├── [ ] Add sharding configuration UI
├── [ ] Add security settings panel
└── [ ] Add data source browser
```

---

## Appendix A: Security Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SECURITY IMPLEMENTATION CHECKLIST                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Transport Security                                                          │
│  ☐ TLS 1.3 for all gRPC connections                                        │
│  ☐ Certificate pinning for known nodes                                      │
│  ☐ mTLS for node-to-node communication                                      │
│                                                                              │
│  Data Integrity                                                              │
│  ☐ FCS (CRC-32) on every frame                                              │
│  ☐ SHA-256 checksum per chunk                                               │
│  ☐ Merkle tree for full dataset verification                                │
│  ☐ Sequence numbers for replay prevention                                   │
│                                                                              │
│  Encryption                                                                  │
│  ☐ AES-256-GCM for payload encryption                                       │
│  ☐ X25519 for key exchange                                                  │
│  ☐ Ed25519 for node identity/signatures                                     │
│  ☐ Encryption at rest for cached data                                       │
│                                                                              │
│  Access Control                                                              │
│  ☐ JWT tokens for user authentication                                       │
│  ☐ Node reputation system                                                   │
│  ☐ Data classification enforcement                                          │
│  ☐ Audit logging for all access                                             │
│                                                                              │
│  Data Protection                                                             │
│  ☐ Sensitive data detection                                                 │
│  ☐ PII anonymization options                                                │
│  ☐ Secure deletion (overwrite)                                              │
│  ☐ Geographic restrictions                                                  │
│                                                                              │
│  Sandboxing                                                                  │
│  ☐ Docker container isolation                                               │
│  ☐ Read-only data mounts                                                    │
│  ☐ Network isolation for sensitive jobs                                     │
│  ☐ SGX enclave support (optional)                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## References

- [TLS 1.3 Specification (RFC 8446)](https://tools.ietf.org/html/rfc8446)
- [gRPC Security Best Practices](https://grpc.io/docs/guides/auth/)
- [X25519 Key Exchange](https://cr.yp.to/ecdh.html)
- [Ed25519 Signatures](https://ed25519.cr.yp.to/)
- [AES-GCM (RFC 5116)](https://tools.ietf.org/html/rfc5116)
- [Merkle Trees](https://en.wikipedia.org/wiki/Merkle_tree)
- [CRC-32 Algorithm](https://en.wikipedia.org/wiki/Cyclic_redundancy_check)
- [IPFS Documentation](https://docs.ipfs.io/)
- [Docker Security](https://docs.docker.com/engine/security/)
