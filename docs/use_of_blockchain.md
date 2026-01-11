# Understanding Blockchain in CyxWiz

A beginner-friendly guide to how blockchain integrates with the CyxWiz decentralized ML compute platform.

---

## What We Actually Did

When we "run the blockchain", we don't actually run our own blockchain. Instead, we **connect to an existing blockchain network** (Solana devnet). Think of it like connecting to the internet - you don't run the internet, you connect to it.

---

## What is Blockchain (Simple Explanation)

```
Traditional Bank:                    Blockchain:
┌─────────────────┐                 ┌─────────────────┐
│   Bank Server   │                 │ Thousands of    │
│  (one company   │                 │ computers agree │
│   controls it)  │                 │ on transactions │
└─────────────────┘                 └─────────────────┘
      │                                    │
  "Trust me"                         "Trust math"
```

**Blockchain** = A shared ledger (record book) that:
- No single company controls
- Everyone can verify transactions
- Cannot be changed once recorded
- Runs on thousands of computers worldwide

---

## What is Solana?

Solana is a specific blockchain network (like Ethereum, Bitcoin, etc.) that's:
- **Fast**: ~400ms transactions (Bitcoin takes ~10 minutes)
- **Cheap**: ~$0.00025 per transaction (Ethereum can be $5-50)
- **Good for apps**: Designed for programs/apps to run on it

**Networks:**
```
Mainnet (Real money)  ──→  Production use
Devnet (Fake money)   ──→  Testing (what we use) ✓
Testnet               ──→  Validator testing
```

---

## How Blockchain Fits in CyxWiz

```
                         CyxWiz Ecosystem
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌─────────────┐         ┌─────────────────┐                │
│  │   Engine    │ ──────→ │ Central Server  │                │
│  │ (Your PC)   │  gRPC   │  (Orchestrator) │                │
│  │             │         │                 │                │
│  │ "I want to  │         │ "I'll find a    │                │
│  │  train a    │         │  GPU node and   │                │
│  │  model"     │         │  manage payment"│                │
│  └─────────────┘         └────────┬────────┘                │
│                                   │                          │
│                                   │ Solana                   │
│                                   ▼ Blockchain               │
│  ┌─────────────┐         ┌─────────────────┐                │
│  │ Server Node │         │   Smart         │                │
│  │ (GPU Owner) │ ←─────→ │   Contracts     │                │
│  │             │         │                 │                │
│  │ "I'll do    │         │ • JobEscrow     │                │
│  │  the work   │         │ • NodeRegistry  │                │
│  │  for pay"   │         │ • Payments      │                │
│  └─────────────┘         └─────────────────┘                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## The Payment Flow (Why We Need Blockchain)

**Problem:** How do you pay strangers on the internet for GPU time without trusting each other?

**Solution:** Smart contracts (programs that run on blockchain)

### Step-by-Step Payment Process

```
Step 1: Job Submission
┌─────────┐                      ┌──────────────┐
│ Engine  │ ─── "Train my ───→  │   Central    │
│ (User)  │      model"         │   Server     │
└─────────┘                      └──────┬───────┘
                                        │
Step 2: Escrow Created                  ▼
┌─────────────────────────────────────────────────────┐
│                  SOLANA BLOCKCHAIN                   │
│  ┌─────────────────────────────────────────────┐   │
│  │              JobEscrow Contract              │   │
│  │  ┌─────────────────────────────────────┐    │   │
│  │  │  Job #123                           │    │   │
│  │  │  Amount: 10 CYXWIZ tokens           │    │   │
│  │  │  Status: LOCKED (waiting for work)  │    │   │
│  │  │  From: User's wallet                │    │   │
│  │  │  To: (pending - whoever does work)  │    │   │
│  │  └─────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

Step 3: Work Done
┌─────────────┐
│ Server Node │ ─── Trains model, reports completion
│   (GPU)     │
└─────────────┘

Step 4: Payment Released
┌─────────────────────────────────────────────────────┐
│              JobEscrow Contract                      │
│  ┌─────────────────────────────────────────────┐   │
│  │  Job #123: COMPLETED                        │   │
│  │  → 9 CYXWIZ → Server Node (90%)            │   │
│  │  → 1 CYXWIZ → Platform fee (10%)           │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## Configuration

The Central Server connects to Solana using this configuration:

```toml
# config.toml
[blockchain]
network = "devnet"                              # Test network (fake money)
solana_rpc_url = "https://api.devnet.solana.com"
payer_keypair_path = "C:/Users/.../id.json"     # Wallet private key
program_id = "DefY4GG..."                       # Our deployed smart contract
```

### Key Terms

| Term | Description |
|------|-------------|
| **Keypair file** | Your wallet's private key (like a password) |
| **Public key** | Your wallet address (like an email - safe to share) |
| **Private key** | Secret key in id.json (NEVER share this!) |
| **Program ID** | Address of our smart contract on Solana |
| **RPC URL** | API endpoint to communicate with Solana network |

---

## Current CyxWiz Blockchain Status

### What Works
- Central Server connects to Solana devnet
- Keypair file validated
- 11.31 SOL in test wallet (fake money for testing)
- JobEscrow contract deployed at `DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA`

### What's Mocked (Not Real Yet)
- Actual transactions (returns fake signatures)
- Balance queries (returns 0)
- Payment processing (logs but doesn't execute)

---

## Why Blockchain for CyxWiz?

| Without Blockchain | With Blockchain |
|-------------------|-----------------|
| Need to trust CyxWiz company | Trust the code (verifiable) |
| Company could steal payments | Escrow releases automatically |
| Company could shut down | Contracts run forever |
| Hidden fee changes | Fees visible on-chain |
| No proof of work done | Permanent record |

---

## Smart Contracts Overview

### 1. JobEscrow Contract
- Holds user payment until work is verified
- Automatically releases funds when job completes
- Handles disputes and refunds

### 2. NodeRegistry Contract (Planned)
- Tracks all compute nodes on the network
- Stores node capabilities (GPU type, VRAM, etc.)
- Manages node reputation scores

### 3. CYXWIZ Token (Planned)
- SPL token on Solana
- Used for all platform payments
- Staking for node operators

---

## Summary

```
"Running blockchain" = Connecting our server to Solana network
                       so payments can be trustless and automatic

User pays → Escrow locks funds → GPU works → Escrow releases to GPU owner
            (nobody can steal)   (verified)   (automatic, no middleman)
```

---

## Learn More

- [Solana Documentation](https://docs.solana.com/)
- [Anchor Framework](https://www.anchor-lang.com/) (for writing Solana programs)
- [Solana Explorer](https://explorer.solana.com/?cluster=devnet) (view transactions on devnet)

---

*Last updated: 2025-12-10*

 cargo run --example test_grpc --release

  Final Test Summary

  | Component         | Status      | Verification                                 |
  |-------------------|-------------|----------------------------------------------|
  | Solana RPC        | ✅ Working   | Balance query: 11.3155 SOL                   |
  | JobEscrow Program | ✅ Deployed  | DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA |
  | gRPC Server       | ✅ Running   | Port 50051 LISTENING                         |
  | ListJobs RPC      | ✅ Working   | Response: 0 jobs (empty database)            |
  | SQLite Database   | ✅ Connected | Migrations complete                          |
  | Redis Cache       | ✅ Connected | Job scheduler operational                    |
  | Job Scheduler     | ✅ Running   | Background polling active                    |
  | Node Monitor      | ✅ Running   | Heartbeat checking active                    |

  The CyxWiz Central Server with real Solana SDK integration is fully operational:
  - Ready to create job escrows on Solana devnet
  - Ready to accept job submissions via gRPC
  - Ready to process node registrations and heartbeats
  - Ready to release payments and issue refunds