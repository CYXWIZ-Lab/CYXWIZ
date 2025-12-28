# Blockchain Integration

CyxWiz uses blockchain technology to enable trustless payments and incentive alignment in the decentralized compute network.

## Overview

The blockchain integration provides:
- **CYXWIZ Token** - Native platform currency
- **Escrow Payments** - Secure job payments
- **Node Staking** - Reputation and priority
- **Reputation System** - Trust scoring
- **Governance** - Future DAO features

## Documentation Sections

| Section | Description |
|---------|-------------|
| [Token Economics](token-economics.md) | CYXWIZ token design |
| [Payment Flow](payment-flow.md) | Job payment lifecycle |
| [Smart Contracts](smart-contracts.md) | On-chain programs |
| [Solana Integration](solana-integration.md) | Technical implementation |
| [Wallet Guide](wallet-guide.md) | User wallet management |

## Blockchain Choice: Solana

CyxWiz uses **Solana** as the primary blockchain for:

| Feature | Solana Advantage |
|---------|------------------|
| **Speed** | 400ms finality |
| **Cost** | ~$0.00025 per transaction |
| **Throughput** | 65,000+ TPS |
| **Smart Contracts** | Rust-based programs |
| **Ecosystem** | Strong developer tools |

## CYXWIZ Token

### Token Specification

| Property | Value |
|----------|-------|
| **Name** | CyxWiz |
| **Symbol** | CYXWIZ |
| **Standard** | SPL Token (Solana) |
| **Decimals** | 9 |
| **Supply** | 1,000,000,000 (1B) |

### Token Utility

| Use Case | Description |
|----------|-------------|
| **Compute Payment** | Pay for training jobs |
| **Node Staking** | Stake for job priority |
| **Reputation Boost** | Higher stake = higher trust |
| **Governance** | Vote on protocol changes |
| **Model Marketplace** | Purchase/sell models |

## Payment Architecture

```
+------------------+                      +------------------+
|   CyxWiz Engine  |                      |   Server Node    |
|   (User/Client)  |                      |   (Worker)       |
+--------+---------+                      +--------+---------+
         |                                         |
         | 1. Submit Job + Payment                 |
         |                                         |
         v                                         |
+--------+---------+                               |
| Central Server   |                               |
|                  |                               |
| 2. Create Escrow |                               |
|    on Solana     |                               |
+--------+---------+                               |
         |                                         |
         | 3. Assign Job                           |
         +---------------------------------------->|
         |                                         |
         |                                         | 4. Execute Job
         |                                         |
         |<----------------------------------------+
         | 5. Report Completion                    |
         |                                         |
         | 6. Verify & Release Escrow              |
         |                                         |
         +---------------------------------------->|
           7. Payment Released                     |
                                                   v
                                           Node Wallet
                                           (90% of payment)
```

## Escrow System

### Escrow Account Structure

```rust
#[account]
pub struct JobEscrow {
    pub job_id: [u8; 32],          // Job identifier
    pub payer: Pubkey,             // Client wallet
    pub payee: Pubkey,             // Node wallet (set on assignment)
    pub amount: u64,               // Payment amount (lamports)
    pub token_mint: Pubkey,        // CYXWIZ token mint
    pub status: EscrowStatus,      // Created, Locked, Released, Refunded
    pub created_at: i64,           // Unix timestamp
    pub expires_at: i64,           // Expiration timestamp
    pub bump: u8,                  // PDA bump
}

#[derive(Clone, Copy, PartialEq, AnchorSerialize, AnchorDeserialize)]
pub enum EscrowStatus {
    Created,    // Funds deposited
    Locked,     // Job assigned to node
    Released,   // Payment sent to node
    Refunded,   // Returned to payer (cancelled/failed)
    Disputed,   // Under arbitration
}
```

### Escrow Operations

```rust
// Create escrow (when job submitted)
pub fn create_escrow(
    ctx: Context<CreateEscrow>,
    job_id: [u8; 32],
    amount: u64,
    expiration: i64,
) -> Result<()>;

// Lock escrow (when job assigned)
pub fn lock_escrow(
    ctx: Context<LockEscrow>,
    job_id: [u8; 32],
    node_wallet: Pubkey,
) -> Result<()>;

// Release escrow (when job completed)
pub fn release_escrow(
    ctx: Context<ReleaseEscrow>,
    job_id: [u8; 32],
) -> Result<()>;

// Refund escrow (if job cancelled/failed)
pub fn refund_escrow(
    ctx: Context<RefundEscrow>,
    job_id: [u8; 32],
) -> Result<()>;
```

## Payment Distribution

When a job completes successfully:

| Recipient | Percentage | Purpose |
|-----------|------------|---------|
| Node (Worker) | 90% | Compute compensation |
| Platform | 10% | Protocol maintenance |

### Future Distribution (with Staking)

| Recipient | Percentage | Purpose |
|-----------|------------|---------|
| Node (Worker) | 85% | Compute compensation |
| Stakers | 5% | Staking rewards |
| Platform | 5% | Protocol maintenance |
| DAO Treasury | 5% | Community fund |

## Reputation System

Nodes build reputation through:

1. **Job Completion** - Successfully finished jobs
2. **Accuracy** - Quality of results
3. **Uptime** - Consistent availability
4. **Speed** - Faster completion than estimated
5. **Stake Amount** - Skin in the game

### Reputation Score

```rust
pub struct NodeReputation {
    pub node_id: Pubkey,
    pub total_jobs: u64,
    pub successful_jobs: u64,
    pub failed_jobs: u64,
    pub total_compute_hours: u64,
    pub average_rating: f32,       // 0.0 - 5.0
    pub stake_amount: u64,
    pub score: f64,                // Calculated reputation (0.0 - 1.0)
}

// Score calculation
fn calculate_reputation(stats: &NodeReputation) -> f64 {
    let success_rate = stats.successful_jobs as f64 /
                       stats.total_jobs.max(1) as f64;
    let rating_factor = stats.average_rating as f64 / 5.0;
    let experience_factor = (stats.total_jobs as f64).log10() / 5.0;
    let stake_factor = (stats.stake_amount as f64).log10() / 12.0;

    // Weighted combination
    0.4 * success_rate +
    0.3 * rating_factor +
    0.15 * experience_factor +
    0.15 * stake_factor
}
```

## Staking

### Node Staking

Nodes can stake CYXWIZ tokens to:
- Increase reputation score
- Gain priority in job assignment
- Earn staking rewards

### Staking Tiers

| Tier | Stake Amount | Benefits |
|------|--------------|----------|
| **Bronze** | 1,000 CYXWIZ | Basic priority |
| **Silver** | 10,000 CYXWIZ | Medium priority, 1% bonus |
| **Gold** | 100,000 CYXWIZ | High priority, 3% bonus |
| **Platinum** | 1,000,000 CYXWIZ | Top priority, 5% bonus |

### Slashing

Stakes can be slashed for:
- Proven malicious behavior
- Repeated job failures
- Submitting invalid results

## Wallet Integration

### Supported Wallets

| Wallet | Support Level |
|--------|---------------|
| Phantom | Full |
| Solflare | Full |
| Backpack | Full |
| CLI Wallet | Full |

### Engine Wallet Panel

```
+------------------------------------------------------------------+
|  Wallet                                                   [x] [-] |
+------------------------------------------------------------------+
|                                                                   |
|  CONNECTED WALLET                                                 |
|  +-----------------------------------------------------------+   |
|  | Address: Abc123...xyz789                                   |   |
|  | Network: Devnet                                            |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  BALANCES                                                         |
|  +----------------------------+-------------------------------+   |
|  | SOL                        | 1.234 SOL                     |   |
|  | CYXWIZ                     | 15,000.00 CYXWIZ             |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  RECENT TRANSACTIONS                                              |
|  +-----------------------------------------------------------+   |
|  | Type      | Amount        | Status   | Time              |   |
|  +-----------------------------------------------------------+   |
|  | Payment   | -500 CYXWIZ   | Complete | 2h ago            |   |
|  | Received  | +450 CYXWIZ   | Complete | 1d ago            |   |
|  | Stake     | -1000 CYXWIZ  | Complete | 3d ago            |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [ Send ]  [ Receive ]  [ Stake ]  [ Unstake ]                    |
+------------------------------------------------------------------+
```

## Development Setup

### Local Testing

1. **Install Solana CLI**
   ```bash
   sh -c "$(curl -sSfL https://release.solana.com/v1.17.0/install)"
   ```

2. **Configure for Devnet**
   ```bash
   solana config set --url https://api.devnet.solana.com
   ```

3. **Create Keypair**
   ```bash
   solana-keygen new -o ~/.config/solana/id.json
   ```

4. **Request Airdrop (Devnet)**
   ```bash
   solana airdrop 2
   ```

### Central Server Configuration

```toml
[blockchain]
network = "devnet"
solana_rpc_url = "https://api.devnet.solana.com"
payer_keypair_path = "~/.config/solana/id.json"
program_id = "11111111111111111111111111111111"
```

## Security Considerations

### Private Key Management

- **Never** commit private keys to Git
- Use hardware wallets for mainnet
- Environment variables for CI/CD
- Encrypted storage for server keys

### Transaction Verification

All blockchain transactions are verified:
1. Signature validation
2. Account ownership checks
3. Amount verification
4. Timestamp validation

### Mainnet Checklist

- [ ] Audit smart contracts
- [ ] Multi-sig for admin keys
- [ ] Rate limiting
- [ ] Monitoring and alerts
- [ ] Backup procedures

## Network Configuration

| Network | RPC URL | Use |
|---------|---------|-----|
| Devnet | `https://api.devnet.solana.com` | Development |
| Testnet | `https://api.testnet.solana.com` | Testing |
| Mainnet | `https://api.mainnet-beta.solana.com` | Production |

---

**Next**: [Token Economics](token-economics.md) | [Payment Flow](payment-flow.md)
