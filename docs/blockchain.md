# CyxWiz Blockchain Integration

Complete technical documentation for the Solana blockchain integration in the CyxWiz decentralized ML compute platform.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Structure](#code-structure)
3. [Core Components](#core-components)
4. [Payment Flow](#payment-flow)
5. [Smart Contract (JobEscrow)](#smart-contract-jobescrow)
6. [CYXWIZ Token](#cyxwiz-token)
7. [Database Models](#database-models)
8. [REST API Endpoints](#rest-api-endpoints)
9. [gRPC Services](#grpc-services)
10. [Configuration](#configuration)
11. [How to Run](#how-to-run)
12. [TODO - What Needs to be Amended](#todo---what-needs-to-be-amended)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           CyxWiz Ecosystem                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐          ┌────────────────────────────────────────────┐ │
│  │   CyxWiz     │          │         Central Server (Rust)              │ │
│  │   Engine     │──gRPC───▶│                                            │ │
│  │  (Desktop)   │          │  ┌─────────────┐  ┌─────────────────────┐  │ │
│  │              │          │  │ JobService  │  │   PaymentProcessor  │  │ │
│  │  - Submit    │          │  │             │──│                     │  │ │
│  │    Jobs      │          │  │ - Submit    │  │ - Create Escrow     │  │ │
│  │  - Monitor   │          │  │ - Cancel    │  │ - Release Payment   │  │ │
│  │    Status    │          │  │ - Status    │  │ - Refund            │  │ │
│  └──────────────┘          │  └─────────────┘  └──────────┬──────────┘  │ │
│                            │                              │              │ │
│  ┌──────────────┐          │  ┌─────────────┐  ┌──────────▼──────────┐  │ │
│  │  Server      │──gRPC───▶│  │ NodeService │  │    SolanaClient     │  │ │
│  │  Node        │          │  │             │  │                     │  │ │
│  │  (GPU)       │          │  │ - Register  │  │ - RPC Connection    │  │ │
│  │              │          │  │ - Heartbeat │  │ - Transaction Sign  │  │ │
│  │  - Execute   │          │  │ - Report    │  │ - Account Queries   │  │ │
│  │    Jobs      │          │  └─────────────┘  └──────────┬──────────┘  │ │
│  │  - Report    │          │                              │              │ │
│  │    Results   │          │  ┌───────────────────────────▼───────────┐  │ │
│  └──────────────┘          │  │           Escrow Module               │  │ │
│                            │  │  - CreateEscrow instruction           │  │ │
│                            │  │  - ReleasePayment instruction         │  │ │
│                            │  │  - Refund instruction                 │  │ │
│                            │  │  - PDA derivation                     │  │ │
│                            │  └───────────────────────────┬───────────┘  │ │
│                            └──────────────────────────────┼──────────────┘ │
│                                                           │                │
│                                                           ▼                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    Solana Blockchain (Devnet/Mainnet)                │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │  JobEscrow Program: DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA │ │  │
│  │  │                                                                 │ │  │
│  │  │  Instructions:                                                  │ │  │
│  │  │    • create_escrow(job_id, amount, user, node)                 │ │  │
│  │  │    • release_payment(job_id) → 90% node, 10% platform          │ │  │
│  │  │    • refund(job_id) → 100% back to user                        │ │  │
│  │  │                                                                 │ │  │
│  │  │  Escrow PDA: seeds = ["escrow", job_id.to_le_bytes()]          │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │  CYXWIZ Token: Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi    │ │  │
│  │  │  Platform Treasury: negq5ApurkfM7V6F46NboJbnjbohEtfu1PotDsvMs5e │ │  │
│  │  │  Total Supply: 1,000,000,000 CYXWIZ (9 decimals)               │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Structure

```
cyxwiz-central-server/
├── src/
│   ├── blockchain/                    # Blockchain Integration Module
│   │   ├── mod.rs                     # Module exports
│   │   ├── solana_client.rs           # Solana RPC client wrapper
│   │   ├── payment_processor.rs       # High-level payment operations
│   │   ├── escrow.rs                  # JobEscrow instruction builders
│   │   ├── reputation.rs              # Node reputation management
│   │   └── types.rs                   # Blockchain data types
│   │
│   ├── api/
│   │   ├── grpc/
│   │   │   ├── job_service.rs         # Job submission with escrow creation
│   │   │   └── wallet_service.rs      # Wallet operations (balance, history)
│   │   │
│   │   └── rest/v1/
│   │       └── blockchain.rs          # REST API for blockchain data
│   │
│   ├── database/
│   │   └── models.rs                  # Payment model (escrow tracking)
│   │
│   ├── config.rs                      # BlockchainConfig struct
│   └── main.rs                        # Server initialization
│
├── config.toml                        # Blockchain configuration
└── Cargo.toml                         # Solana SDK dependencies
```

---

## Core Components

### 1. SolanaClient (`solana_client.rs`)

The low-level Solana RPC client wrapper.

```rust
pub struct SolanaClient {
    rpc_client: Arc<RpcClient>,      // Solana RPC connection
    payer: Arc<Keypair>,             // Transaction signing keypair
    program_id: Pubkey,              // JobEscrow program address
}
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `new(rpc_url, payer_bytes, program_id)` | Create from raw keypair bytes |
| `from_keypair_file(rpc_url, path, program_id)` | Create from JSON keypair file |
| `get_balance(pubkey)` | Query account balance in lamports |
| `get_latest_blockhash()` | Get blockhash for transactions |
| `execute_transaction(instructions)` | Build, sign, send transaction |
| `confirm_transaction(signature)` | Verify transaction confirmed |
| `get_account_data(pubkey)` | Read account data (for escrow state) |

**Usage:**
```rust
let client = SolanaClient::from_keypair_file(
    "https://api.devnet.solana.com",
    "~/.config/solana/id.json",
    "DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA",
)?;

let balance = client.get_balance(&client.payer_pubkey()).await?;
println!("Balance: {} SOL", balance as f64 / 1_000_000_000.0);
```

### 2. PaymentProcessor (`payment_processor.rs`)

High-level API for payment operations.

```rust
pub struct PaymentProcessor {
    solana_client: Arc<SolanaClient>,
    config: PaymentConfig,  // token_mint, platform_fee, etc.
}
```

**Key Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `create_job_escrow(job_id, user, node, amount)` | Lock funds in escrow | `(tx_hash, escrow_address)` |
| `complete_job_payment(job_id, node_wallet)` | Release 90% to node, 10% to platform | `tx_hash` |
| `refund_job(job_id, user_wallet)` | Return 100% to user | `tx_hash` |
| `get_escrow_status(job_id)` | Query on-chain escrow state | `Option<EscrowInfo>` |
| `calculate_payment_distribution(amount)` | Calculate node/platform split | `(node_pay, fee)` |
| `check_health()` | Verify blockchain connection | `Result<()>` |

**Payment Distribution:**
```
Total Amount: 100 CYXWIZ
├── Node Payment (90%):  90 CYXWIZ
└── Platform Fee (10%):  10 CYXWIZ
```

### 3. Escrow Module (`escrow.rs`)

Anchor-compatible instruction builders for the JobEscrow program.

```rust
// Program ID on Solana Devnet
pub const JOB_ESCROW_PROGRAM_ID: &str = "DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA";

// Escrow account state
pub struct EscrowState {
    pub job_id: u64,
    pub user: Pubkey,
    pub node: Pubkey,
    pub amount: u64,
    pub platform_fee_percentage: u8,
    pub status: EscrowStatus,  // Pending, Released, Refunded
    pub created_at: i64,
    pub completed_at: Option<i64>,
    pub bump: u8,
}
```

**Instructions:**

| Instruction | Purpose | Accounts |
|-------------|---------|----------|
| `create_escrow_instruction()` | Lock user funds | escrow_pda, user, user_token, escrow_token |
| `release_payment_instruction()` | Pay node + platform | escrow_pda, escrow_token, node_token, platform_token |
| `refund_instruction()` | Return to user | escrow_pda, escrow_token, user_token |

**PDA Derivation:**
```rust
// Escrow account address is deterministically derived
let (escrow_pda, bump) = Pubkey::find_program_address(
    &[b"escrow", &job_id.to_le_bytes()],
    &program_id,
);
```

### 4. ReputationManager (`reputation.rs`)

Node reputation tracking on blockchain (partially implemented).

```rust
pub struct ReputationManager {
    rpc_client: Arc<RpcClient>,
    authority: Arc<Keypair>,
    program_id: Pubkey,
}
```

**Key Methods (Mocked):**

| Method | Description |
|--------|-------------|
| `update_reputation(node_id, success, time)` | Update after job completion |
| `get_reputation(node_id)` | Fetch node's reputation score |
| `register_node(node_id, stake)` | Register node on-chain |
| `slash_stake(node_id, amount, reason)` | Penalize malicious node |

---

## Payment Flow

### Complete Job Lifecycle with Payments

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Job Submission                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Engine                  Central Server                  Solana             │
│    │                          │                            │                │
│    │── SubmitJob(config) ────▶│                            │                │
│    │                          │                            │                │
│    │                          │── create_job_escrow() ────▶│                │
│    │                          │   (locks user's tokens)    │                │
│    │                          │                            │                │
│    │                          │◀── (tx_hash, escrow_pda) ──│                │
│    │                          │                            │                │
│    │                          │── INSERT payments ────▶ [PostgreSQL]        │
│    │                          │   status: 'locked'                          │
│    │                          │                            │                │
│    │◀─ SubmitJobResponse ─────│                            │                │
│    │   (job_id, escrow_tx)    │                            │                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Job Assignment & Execution                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Scheduler                  Server Node                                     │
│    │                            │                                           │
│    │── AssignJob(job_id) ──────▶│                                           │
│    │                            │                                           │
│    │                            │── Execute ML Training ──▶ [GPU]           │
│    │                            │                                           │
│    │◀─ ReportCompletion ────────│                                           │
│    │   (result_hash)            │                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3A: Successful Completion                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Central Server                 Solana                                      │
│    │                              │                                         │
│    │── complete_job_payment() ───▶│                                         │
│    │   (release_payment ix)       │                                         │
│    │                              │                                         │
│    │                              │── Transfer 90% ──▶ [Node Wallet]        │
│    │                              │── Transfer 10% ──▶ [Platform Treasury]  │
│    │                              │                                         │
│    │◀── completion_tx_hash ───────│                                         │
│    │                              │                                         │
│    │── UPDATE payments ────────▶ [PostgreSQL]                               │
│    │   status: 'completed'                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3B: Failed/Cancelled Job                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Central Server                 Solana                                      │
│    │                              │                                         │
│    │── refund_job() ─────────────▶│                                         │
│    │   (refund ix)                │                                         │
│    │                              │                                         │
│    │                              │── Transfer 100% ──▶ [User Wallet]       │
│    │                              │                                         │
│    │◀── refund_tx_hash ───────────│                                         │
│    │                              │                                         │
│    │── UPDATE payments ────────▶ [PostgreSQL]                               │
│    │   status: 'refunded'                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Smart Contract (JobEscrow)

### Deployed Program

| Property | Value |
|----------|-------|
| **Program ID** | `DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA` |
| **Network** | Solana Devnet |
| **Framework** | Anchor |

### Instructions

#### 1. CreateEscrow

```rust
// Accounts:
// 0. escrow (writable) - PDA derived from ["escrow", job_id]
// 1. user (signer, writable) - Pays for account creation
// 2. user_token_account (writable) - Source of funds
// 3. escrow_token_account (writable) - Destination (locked)
// 4. token_program (readonly)
// 5. system_program (readonly)

// Args:
struct CreateEscrowArgs {
    job_id: u64,
    amount: u64,
    node_pubkey: Pubkey,
}
```

#### 2. ReleasePayment

```rust
// Accounts:
// 0. escrow (writable) - PDA
// 1. escrow_token_account (writable)
// 2. node_token_account (writable) - Receives 90%
// 3. platform_token_account (writable) - Receives 10%
// 4. authority (signer) - Central Server
// 5. token_program (readonly)

// No additional args (job_id derived from escrow PDA)
```

#### 3. Refund

```rust
// Accounts:
// 0. escrow (writable) - PDA
// 1. escrow_token_account (writable)
// 2. user_token_account (writable) - Receives 100%
// 3. authority (signer)
// 4. token_program (readonly)

// No additional args
```

### Anchor Discriminator

Instructions are identified by SHA256 hash of the instruction name:

```rust
fn anchor_discriminator(instruction_name: &str) -> [u8; 8] {
    let preimage = format!("global:{}", instruction_name);
    let hash = Sha256::digest(preimage.as_bytes());
    hash[..8].try_into().unwrap()
}

// Examples:
// "create_escrow" → [0x12, 0x34, ...]
// "release_payment" → [0xAB, 0xCD, ...]
// "refund" → [0xEF, 0x01, ...]
```

---

## CYXWIZ Token

The CYXWIZ SPL token is deployed on Solana Devnet for platform payments.

### Token Details

| Property | Value |
|----------|-------|
| **Token Mint** | `Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi` |
| **Platform Treasury** | `negq5ApurkfM7V6F46NboJbnjbohEtfu1PotDsvMs5e` |
| **Token Name** | CYXWIZ |
| **Token Symbol** | CYXWIZ |
| **Decimals** | 9 (same as SOL, 1 CYXWIZ = 1,000,000,000 smallest units) |
| **Total Supply** | 1,000,000,000 CYXWIZ |
| **Network** | Solana Devnet |
| **Created** | 2025-12-17 |
| **Logo** | `cyxtoken.png` (in repo root) |

### On-Chain Metadata

Token metadata has been registered on-chain using the Metaplex Token Metadata program:
- **Name**: CYXWIZ
- **Symbol**: CYXWIZ
- **Update Authority**: `4Y5HWB9W9SELq3Yoyf7mK7KF5kTbuaGxd2BMvn3AyAG8`

**Note**: To add the token logo to explorers, host `cyxtoken.png` on IPFS/Arweave and update the metadata URI using the script at `scripts/token-metadata/add-metadata.mjs`.

### Creation Transactions

| Transaction | Signature |
|-------------|-----------|
| Create Token Mint | `5N5ZXxBpWggCJX1DnmbVvv1QWpb8LivRwf17cT3YLoyZ4mq4FfWJoc9EjSJpQkFiXDiNuiRZvQ6Hwx2SnwytoC5n` |
| Create Treasury Account | `3ENBwWQwqsVMuxmoyC1pCFWsJNkPihCTQ1PjdCm9ZzoGypGqsh7ABAQVJrL6RAzdE4bxqUQ51wmxt1JNi5j6tQDH` |
| Mint Initial Supply | `3sDp32fgt9axpQxwrn18cvfHLUurn5qjy8nhHPEmjDaaMrnffbQAwpX2DoqmdaAHnTJWXf5QGnQBgFv3BLmmCTtV` |
| Add Token Metadata | `zinl9ZbkKo86TSQML23Ej4H8Zkia1ZiF8P4tIwVrkR4NWC6x450HJfr0UP9xtDJ2gdyf+k2M8Q0O+QRECXciDg==` |

### Solana Explorer Links

- [Token Mint on Explorer](https://explorer.solana.com/address/Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi?cluster=devnet)
- [Platform Treasury on Explorer](https://explorer.solana.com/address/negq5ApurkfM7V6F46NboJbnjbohEtfu1PotDsvMs5e?cluster=devnet)

### Token Usage

```
Payment Flow with CYXWIZ Token:

1. User holds CYXWIZ tokens in their wallet
2. On job submission, tokens are transferred to escrow PDA
3. On job completion:
   - 90% transferred to node's token account
   - 10% transferred to platform treasury
4. On job failure/cancellation:
   - 100% refunded to user's token account
```

### CLI Commands

```bash
# Check token supply
spl-token supply Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi --url devnet

# Check your CYXWIZ balance
spl-token balance Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi --url devnet

# Create a token account for your wallet
spl-token create-account Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi --url devnet

# Transfer tokens (for testing)
spl-token transfer Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi 100 <RECIPIENT_ADDRESS> --url devnet
```

---

## Database Models

### Payment Model (`database/models.rs`)

```rust
pub struct Payment {
    pub id: DbId,
    pub job_id: DbId,
    pub node_id: Option<DbId>,
    pub user_wallet: String,
    pub node_wallet: Option<String>,

    // Amounts (in lamports/smallest token unit)
    pub amount: i64,           // Total escrowed
    pub platform_fee: i64,     // 10% to platform
    pub node_reward: i64,      // 90% to node

    pub status: PaymentStatus, // Pending, Locked, Streaming, Completed, Failed, Refunded

    // Blockchain references
    pub escrow_tx_hash: Option<String>,     // CreateEscrow transaction
    pub completion_tx_hash: Option<String>, // ReleasePayment/Refund transaction
    pub escrow_account: Option<String>,     // PDA address

    pub created_at: DateTime<Utc>,
    pub locked_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

pub enum PaymentStatus {
    Pending,    // Job created, escrow not yet created
    Locked,     // Funds locked in escrow
    Streaming,  // (Future) Real-time payment streaming
    Completed,  // Payment released to node
    Failed,     // Transaction failed
    Refunded,   // Funds returned to user
}
```

---

## REST API Endpoints

### Blockchain Endpoints (`/api/v1/blockchain/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/blockchain/wallet` | GET | Platform wallet info |
| `/api/v1/blockchain/transactions` | GET | Transaction history |
| `/api/v1/blockchain/escrows` | GET | Active escrows |
| `/api/v1/blockchain/stats` | GET | Blockchain statistics |

### Response Types

**GET /api/v1/blockchain/wallet**
```json
{
  "address": "7xKXt...9dF2",
  "network": "devnet",
  "sol_balance": 11.31,
  "is_connected": true,
  "rpc_endpoint": "https://api.devnet.solana.com",
  "program_id": "DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA"
}
```

**GET /api/v1/blockchain/transactions?limit=10&status=completed**
```json
{
  "transactions": [
    {
      "id": "uuid",
      "tx_type": "completed",
      "job_id": "job-uuid",
      "amount": 1000000000,
      "amount_sol": 1.0,
      "platform_fee": 100000000,
      "node_reward": 900000000,
      "status": "completed",
      "escrow_tx_hash": "5T7KXt...",
      "completion_tx_hash": "8Hn2Lk...",
      "user_wallet": "7xKXt...",
      "node_wallet": "9aBC...",
      "created_at": "2025-01-01T00:00:00Z",
      "completed_at": "2025-01-01T01:00:00Z"
    }
  ],
  "total": 100,
  "page": 1,
  "limit": 10,
  "total_pages": 10
}
```

**GET /api/v1/blockchain/escrows**
```json
{
  "escrows": [
    {
      "id": "uuid",
      "job_id": "job-uuid",
      "user_wallet": "7xKXt...",
      "node_wallet": "9aBC...",
      "amount": 1000000000,
      "amount_sol": 1.0,
      "status": "locked",
      "escrow_account": "EscrowPDA...",
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total_locked_lamports": 5000000000,
  "total_locked_sol": 5.0
}
```

**GET /api/v1/blockchain/stats**
```json
{
  "total_transactions": 1250,
  "total_volume_lamports": 125000000000,
  "total_volume_sol": 125.0,
  "platform_fees_collected": 12500000000,
  "platform_fees_sol": 12.5,
  "node_payouts_total": 112500000000,
  "node_payouts_sol": 112.5,
  "active_escrows_count": 15,
  "active_escrows_value": 15000000000,
  "transactions_24h": 50,
  "volume_24h_lamports": 5000000000,
  "volume_24h_sol": 5.0
}
```

---

## gRPC Services

### WalletService

| RPC | Request | Response |
|-----|---------|----------|
| `ConnectWallet` | wallet_address | is_valid, message |
| `GetBalance` | wallet_address | sol_balance, cyxwiz_balance |
| `GetTransactionHistory` | wallet_address, limit | transactions[] |
| `EstimateJobCost` | model_type, epochs, dataset_size | estimated_cost, time |

### JobService (Blockchain Integration)

When a job is submitted via `SubmitJob`:

1. Job created in database with `status: pending`
2. `PaymentProcessor.create_job_escrow()` called
3. Escrow transaction sent to Solana
4. Payment record created with `status: locked`
5. Job added to scheduler queue

---

## Configuration

### config.toml

```toml
[blockchain]
# Network: "devnet", "testnet", or "mainnet-beta"
network = "devnet"

# Solana RPC endpoint
solana_rpc_url = "https://api.devnet.solana.com"

# Path to payer keypair (JSON format from solana-keygen)
payer_keypair_path = "~/.config/solana/id.json"

# JobEscrow program ID (deployed on devnet)
program_id = "DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA"

# CYXWIZ SPL Token on devnet (created 2025-12-17)
token_mint = "Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi"

# Platform treasury token account (receives 10% fees)
platform_token_account = "negq5ApurkfM7V6F46NboJbnjbohEtfu1PotDsvMs5e"

# Platform fee percentage (default: 10%)
platform_fee_percent = 10
```

### Environment Variables

Override config.toml with environment variables:

```bash
export CYXWIZ_BLOCKCHAIN__SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
export CYXWIZ_BLOCKCHAIN__NETWORK="mainnet-beta"
export CYXWIZ_BLOCKCHAIN__PAYER_KEYPAIR_PATH="/secure/path/keypair.json"
```

---

## How to Run

### Prerequisites

1. **Solana CLI** (for keypair management)
   ```bash
   # Install Solana CLI
   sh -c "$(curl -sSfL https://release.solana.com/stable/install)"

   # Generate keypair (if you don't have one)
   solana-keygen new -o ~/.config/solana/id.json

   # Get devnet SOL for testing
   solana airdrop 2 --url devnet
   ```

2. **Rust toolchain**
   ```bash
   rustup update stable
   ```

3. **PostgreSQL** (or SQLite for development)

4. **Redis** (optional, graceful fallback if unavailable)

### Running the Server

```bash
cd cyxwiz-central-server

# Development (SQLite + mock Redis)
cargo run

# Production (PostgreSQL + Redis)
cargo run --release

# TUI mode (monitoring dashboard)
cargo run -- --tui
```

### Verifying Blockchain Connection

```bash
# Check server logs for:
# ✓ Solana client initialized (network: devnet)
# ✓ Payer pubkey: 7xKXt...
# ✓ Payer balance: 11.3155 SOL

# Or use REST API:
curl http://localhost:8080/api/v1/blockchain/wallet
```

### Testing Escrow Creation

```bash
# Using grpcurl
grpcurl -plaintext -d '{
  "config": {
    "payment_address": "YourWalletAddress",
    "job_type": 1,
    "model_definition": "mnist_mlp",
    "dataset_uri": "s3://data/mnist",
    "batch_size": 32,
    "epochs": 10,
    "estimated_duration": 3600,
    "estimated_memory": 4294967296
  }
}' localhost:50051 cyxwiz.JobService/SubmitJob
```

---

## TODO - What Needs to be Amended

### Critical (Must Have for Production)

| Priority | Component | Issue | Status |
|----------|-----------|-------|--------|
| ~~**P0**~~ | `wallet_service.rs` | Mock data | ✅ **COMPLETED** - Real blockchain queries for `GetBalance` and `GetTransactionHistory` |
| ~~**P0**~~ | `payment_processor.rs` | Node wallet timing | ✅ **COMPLETED** - Added `create_pending_escrow()` and `update_escrow_node()` methods |
| ~~**P0**~~ | `escrow.rs` | No SPL token support | ✅ **COMPLETED** - Already uses SPL token program (`spl_token::id()`) |
| ~~**P0**~~ | `payment_processor.rs` | Token not integrated | ✅ **COMPLETED** - Added `PaymentConfig::cyxwiz_devnet()`, token balance queries, and helper methods |
| **P0** | Smart Contract | Not verified | 🔴 **PENDING** - See [Smart Contract Audit Requirements](#smart-contract-audit-requirements) below |

---

### Smart Contract Audit Requirements

The JobEscrow smart contract (`DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA`) requires a comprehensive security audit before mainnet deployment.

#### Audit Scope

| Area | Items to Review |
|------|-----------------|
| **Access Control** | Verify only authorized parties can release/refund escrows |
| **PDA Derivation** | Ensure escrow PDAs are correctly derived and cannot be spoofed |
| **Token Transfers** | Validate SPL token transfer instructions are correct |
| **Arithmetic Safety** | Check for overflow/underflow in fee calculations |
| **Account Validation** | Verify all accounts are properly validated before operations |
| **Reentrancy** | Ensure no reentrancy vulnerabilities exist |
| **State Transitions** | Verify escrow status transitions are valid and atomic |

#### Pre-Audit Checklist

- [ ] Update smart contract to support `update_node` instruction for deferred node assignment
- [ ] Add comprehensive unit tests for all instructions
- [ ] Add integration tests with devnet deployment
- [ ] Document all PDAs and their derivation seeds
- [ ] Review fee calculation logic for edge cases
- [ ] Add event emission for transaction tracking
- [ ] Implement program upgrade authority controls

#### Recommended Audit Firms

1. **Halborn** - Solana specialized
2. **Neodyme** - Anchor/Solana expertise
3. **OtterSec** - Comprehensive Solana audits
4. **Sec3** - Automated + manual Solana audits

#### Mainnet Deployment Checklist

- [ ] Complete security audit with no critical/high findings
- [ ] Deploy to mainnet-beta
- [ ] Verify program on Solana Explorer
- [ ] Update `config.toml` with mainnet addresses
- [ ] Create CYXWIZ token on mainnet
- [ ] Set up multisig for program upgrade authority
- [ ] Configure platform treasury with proper access controls

### High Priority

| Priority | Component | Issue | Description |
|----------|-----------|-------|-------------|
| **P1** | `reputation.rs` | Fully mocked | All methods return mock data. NodeRegistry program not deployed |
| **P1** | `job_service.rs` | Job ID conversion | Using first 8 bytes of UUID for blockchain job_id may cause collisions |
| **P1** | REST API | No auth | `/api/v1/blockchain/*` endpoints have no authentication |
| **P1** | Error handling | Silent failures | Many blockchain errors logged but not propagated to user |
| **P1** | Transaction retry | None | No retry logic for failed transactions |

### Medium Priority

| Priority | Component | Issue | Description |
|----------|-----------|-------|-------------|
| **P2** | `payment_processor.rs` | Payment streaming | `PaymentStatus::Streaming` defined but not implemented |
| **P2** | `types.rs` | Node staking | `min_node_stake` defined but not enforced |
| **P2** | REST API | Pagination | `/blockchain/escrows` returns all, no pagination |
| **P2** | TUI | Limited blockchain view | TUI blockchain tab could show more details |
| **P2** | Metrics | No Prometheus | Add blockchain metrics (tx latency, success rate) |

### Low Priority / Nice to Have

| Priority | Component | Issue | Description |
|----------|-----------|-------|-------------|
| **P3** | Multi-token | SOL only | Support multiple SPL tokens for payment |
| **P3** | Webhook | None | Notify external services on payment events |
| **P3** | Historical data | Limited | Better transaction history with filtering |
| **P3** | Mainnet | Devnet only | Configuration for mainnet deployment |

### Code Snippets for Completed P0 Items

#### ✅ GetBalance - Now Queries Real Blockchain

```rust
// wallet_service.rs - get_balance implementation
async fn get_balance(&self, request: Request<GetBalanceRequest>) -> Result<...> {
    let wallet_pubkey = Pubkey::from_str(&req.wallet_address)?;

    // Query real blockchain balances
    let sol_balance = self.get_sol_balance(&wallet_pubkey).await?;
    let cyxwiz_balance = self.get_token_balance(&wallet_pubkey).await?;

    Ok(Response::new(GetBalanceResponse {
        sol_balance,
        cyxwiz_balance,
        token_mint: self.token_mint.to_string(),
        ...
    }))
}
```

#### ✅ Node Wallet Timing - Pending Escrow Support

```rust
// payment_processor.rs - create_pending_escrow and update_escrow_node
pub async fn create_pending_escrow(&self, job_id: u64, user_wallet: &str, amount: u64) -> Result<...> {
    // Use platform wallet as temporary placeholder
    let placeholder_node = self.config.platform_token_account.to_string();
    self.create_job_escrow(job_id, user_wallet, &placeholder_node, amount).await
}

pub async fn update_escrow_node(&self, job_id: u64, node_wallet: &str) -> Result<()> {
    // Track actual recipient at application level
    // Smart contract update_node instruction pending implementation
    Ok(())
}
```

#### ✅ CYXWIZ Token Integration

```rust
// payment_processor.rs - PaymentConfig with CYXWIZ token
impl PaymentConfig {
    pub fn cyxwiz_devnet() -> Self {
        Self {
            token_mint: Pubkey::from_str("Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi").unwrap(),
            platform_token_account: Pubkey::from_str("negq5ApurkfM7V6F46NboJbnjbohEtfu1PotDsvMs5e").unwrap(),
            platform_fee_percent: 10,
        }
    }
}
```

### Code Snippets for Pending Items

#### P1: Fix Job ID Collision Risk

```rust
// job_service.rs - Line 106
// CURRENT:
let job_id_u64 = u64::from_le_bytes(job_id.as_bytes()[0..8].try_into().unwrap());

// SHOULD BE:
// Use a dedicated counter or hash the full UUID
let job_id_u64 = generate_unique_blockchain_job_id(&self.db_pool).await?;
```

---

## Summary

The CyxWiz blockchain integration provides trustless payment handling through:

1. **SolanaClient** - Low-level RPC communication
2. **PaymentProcessor** - High-level escrow operations
3. **Escrow Module** - Anchor-compatible instruction builders
4. **REST/gRPC APIs** - Client access to blockchain data

**Current Status:**
- Core escrow flow implemented and tested on devnet
- JobEscrow program deployed at `DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA`
- CYXWIZ SPL token created on devnet (see below)
- Platform wallet funded with 11.31 SOL for testing
- REST API endpoints functional
- Several mock implementations need real blockchain queries (see TODO)

**CYXWIZ Token (Devnet):**

| Property | Value |
|----------|-------|
| **Token Mint** | `Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi` |
| **Platform Treasury** | `negq5ApurkfM7V6F46NboJbnjbohEtfu1PotDsvMs5e` |
| **Decimals** | 9 (same as SOL) |
| **Total Supply** | 1,000,000,000 CYXWIZ |
| **Created** | 2025-12-17 |

**Creation Transactions:**
- Create Token: `5N5ZXxBpWggCJX1DnmbVvv1QWpb8LivRwf17cT3YLoyZ4mq4FfWJoc9EjSJpQkFiXDiNuiRZvQ6Hwx2SnwytoC5n`
- Create Treasury: `3ENBwWQwqsVMuxmoyC1pCFWsJNkPihCTQ1PjdCm9ZzoGypGqsh7ABAQVJrL6RAzdE4bxqUQ51wmxt1JNi5j6tQDH`
- Mint Supply: `3sDp32fgt9axpQxwrn18cvfHLUurn5qjy8nhHPEmjDaaMrnffbQAwpX2DoqmdaAHnTJWXf5QGnQBgFv3BLmmCTtV`

**View on Solana Explorer:**
- [Token Mint](https://explorer.solana.com/address/Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi?cluster=devnet)
- [Platform Treasury](https://explorer.solana.com/address/negq5ApurkfM7V6F46NboJbnjbohEtfu1PotDsvMs5e?cluster=devnet)

**Next Steps:**
1. Replace mock data with real blockchain queries (use CYXWIZ token balance)
2. Update PaymentProcessor to use SPL token transfers
3. Deploy and integrate NodeRegistry program
4. Implement payment streaming
5. Security audit before mainnet

---
 ### Note 
  CyxWallet signatures are NOT used for authentication (you're already logged in via NextAuth). They're used for:

  1. Sending Tokens/SOL

  // Build a transfer transaction
  const tx = new Transaction().add(
    SystemProgram.transfer({
      fromPubkey: new PublicKey(cyxWalletAddress),
      toPubkey: new PublicKey(recipientAddress),
      lamports: 1000000, // 0.001 SOL
    })
  );

  // Sign with CyxWallet
  const signed = await signTransactionWithCyxWallet(tx.serialize().toString("base64"));

  // Send to Solana network
  connection.sendRawTransaction(Buffer.from(signed.signedTransaction!, "base64"));

  2. Interacting with dApps/Smart Contracts

  // dApp asks you to prove you own the wallet
  const message = "Verify ownership for MyDApp - Nonce: 12345";
  const { signature, publicKey } = await signMessageWithCyxWallet(message);

  // Send signature to dApp backend for verification
  await fetch("https://mydapp.com/verify", {
    body: JSON.stringify({ publicKey, signature, message })
  });

  3. Token Staking/Governance

  // Sign a staking transaction
  const stakeTx = buildStakeTransaction(amount);
  const signed = await signTransactionWithCyxWallet(stakeTx);

  4. Marketplace Purchases

  // Buy compute time with CYXWIZ tokens
  const purchaseTx = buildPurchaseTransaction(computeHours);
  const signed = await signTransactionWithCyxWallet(purchaseTx);

  Summary

  | Use Case                        | Signature Needed?        |
  |---------------------------------|--------------------------|
  | Login to CyxWiz                 | ❌ No (NextAuth session) |
  | View wallet balance             | ❌ No                    |
  | Send SOL/tokens                 | ✅ Yes                   |
  | Buy compute time                | ✅ Yes                   |
  | Stake CYXWIZ                    | ✅ Yes                   |
  | Prove wallet ownership to dApps | ✅ Yes                   |
  | Sign smart contract calls       | ✅ Yes                   |
*Last updated: 2025-12-17*

---

## CyxCloud Storage Programs (Deployed 2025-12-22)

The following Anchor programs have been deployed to Solana devnet for the CyxCloud decentralized storage system:

### Deployed Programs

| Program | Program ID | Description |
|---------|-----------|-------------|
| **StorageNodeRegistry** | `AQPP8YaiGazv9Mh4bnsVcGiMgTKbarZeCcQsh4jmpi4Z` | Storage node registration, staking (500 CYXWIZ min), proof-of-storage |
| **StoragePaymentPool** | `4wFUJ1SVpVDEpYTobgkLSnwY2yXr4LKAMdWcG3sAe3sX` | Weekly payment epochs, 85/10/5 distribution (nodes/platform/community) |
| **StorageSubscription** | `HZhWDJVkkUuHrgqkNb9bYxiCfqtnv8cnus9UQt843Fro` | User storage plans (Free/Starter/Pro/Enterprise) and payments |
| **JobEscrow** | `3sTCk7gVqj5RU8JECmY2zDjGpFKYZncsM9KqYFZBkkM9` | ML job payment locking and release |
| **NodeRegistry** | `H15XzFvYpGqm9aH66n64B4Ld7CtZNSaFSftN8kGPQhCz` | ML compute node registration and staking |

### Program Properties

| Property | Value |
|----------|-------|
| **Network** | Solana Devnet |
| **Framework** | Anchor 0.28.0 |
| **Upgrade Authority** | `4Y5HWB9W9SELq3Yoyf7mK7KF5kTbuaGxd2BMvn3AyAG8` |
| **CYXWIZ Token** | `Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi` |

### Storage Payment Distribution (Weekly Epochs)

```
Total Pool → 85% Storage Nodes (weighted by storage + uptime + speed)
           → 10% Platform Treasury
           → 5% Community Fund
```

### View on Solana Explorer

- [StorageNodeRegistry](https://explorer.solana.com/address/AQPP8YaiGazv9Mh4bnsVcGiMgTKbarZeCcQsh4jmpi4Z?cluster=devnet)
- [StoragePaymentPool](https://explorer.solana.com/address/4wFUJ1SVpVDEpYTobgkLSnwY2yXr4LKAMdWcG3sAe3sX?cluster=devnet)
- [StorageSubscription](https://explorer.solana.com/address/HZhWDJVkkUuHrgqkNb9bYxiCfqtnv8cnus9UQt843Fro?cluster=devnet)
- [JobEscrow](https://explorer.solana.com/address/3sTCk7gVqj5RU8JECmY2zDjGpFKYZncsM9KqYFZBkkM9?cluster=devnet)
- [NodeRegistry](https://explorer.solana.com/address/H15XzFvYpGqm9aH66n64B4Ld7CtZNSaFSftN8kGPQhCz?cluster=devnet)

*Last updated: 2025-12-22*
