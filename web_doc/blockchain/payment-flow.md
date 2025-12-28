# Payment Flow

Complete documentation of the CyxWiz payment system, from job submission to final settlement.

## Overview

CyxWiz uses Solana blockchain for trustless payments between clients (job submitters) and compute providers (server nodes).

```
+------------------+                                    +------------------+
|  CyxWiz Engine   |                                    |   Server Node    |
|     (Client)     |                                    |    (Provider)    |
+--------+---------+                                    +--------+---------+
         |                                                       |
         | 1. Submit Job + Create Escrow                         |
         |                                                       |
         v                                                       |
+--------+---------+                                             |
| Central Server   |                                             |
+--------+---------+                                             |
         |                                                       |
         | 2. Verify Escrow on Solana                            |
         |                                                       |
         | 3. Assign Job to Node                                 |
         +------------------------------------------------------>|
         |                                                       |
         |<------------------------------------------------------+
         | 4. Job Accepted                                       |
         |                                                       |
         |                         5. Execute Training           |
         |                                                       |
         |<------------------------------------------------------+
         | 6. Report Completion + Proof                          |
         |                                                       |
         | 7. Verify & Release Escrow                            |
         |                                                       |
         +------------------------------------------------------>|
         |                              8. Payment Received      |
         |                                                       |
```

## Payment Lifecycle

### Phase 1: Job Submission

**Client Actions:**

1. Estimate job cost based on:
   - Model complexity (parameters, layers)
   - Dataset size
   - Training duration (epochs)
   - Required hardware (GPU type, memory)

2. Create escrow transaction on Solana:
   ```rust
   // Pseudocode
   let escrow = create_escrow(
       job_id: "job-abc123",
       amount: 100_000_000,  // 0.1 SOL in lamports
       payer: client_wallet,
       expiration: now + 24_hours
   );
   ```

3. Submit job to Central Server with escrow transaction signature

**Central Server Actions:**

1. Receive job submission
2. Verify escrow transaction on Solana:
   - Check amount matches quoted price
   - Verify payer signature
   - Confirm escrow is funded
3. Queue job for assignment

### Phase 2: Job Assignment

**Central Server Actions:**

1. Find suitable node based on:
   - Hardware requirements (GPU, memory)
   - Node reputation score
   - Current load
   - Geographic location (optional)

2. Lock escrow to assigned node:
   ```rust
   lock_escrow(
       escrow_id,
       node_wallet: selected_node.wallet_address
   );
   ```

3. Send job assignment to node with:
   - Job configuration
   - Dataset location (IPFS/URL)
   - Authorization token

**Node Actions:**

1. Validate job requirements
2. Check available resources
3. Accept or reject job
4. Begin execution if accepted

### Phase 3: Job Execution

**Node Actions:**

1. Download dataset (if remote)
2. Initialize training environment
3. Execute training loop
4. Report progress periodically:
   ```protobuf
   message ProgressUpdate {
       string job_id = 1;
       double progress = 2;      // 0.0 to 1.0
       int32 current_epoch = 3;
       map<string, double> metrics = 4;
   }
   ```

**Central Server Actions:**

1. Relay progress updates to client
2. Monitor for timeouts
3. Handle cancellation requests

### Phase 4: Completion & Verification

**Node Actions:**

1. Generate completion proof:
   - Final model hash
   - Training metrics
   - Computation summary
   - Optional: cryptographic proof of work

2. Upload results:
   - Model weights to IPFS/storage
   - Training logs
   - Metric history

3. Submit completion report:
   ```protobuf
   message CompletionReport {
       string job_id = 1;
       string model_hash = 2;
       string model_uri = 3;
       map<string, double> final_metrics = 4;
       int64 compute_time_ms = 5;
       bytes proof_of_compute = 6;
   }
   ```

**Central Server Actions:**

1. Verify completion:
   - Check model hash matches uploaded model
   - Validate metrics are reasonable
   - Verify proof of compute (if enabled)

2. Quality checks:
   - Model file size
   - Training loss convergence
   - No errors in logs

### Phase 5: Payment Release

**Central Server Actions:**

1. Release escrow on successful completion:
   ```rust
   release_escrow(
       escrow_id,
       node_wallet: executing_node.wallet,
       platform_wallet: cyxwiz_treasury
   );
   ```

2. Payment distribution:
   - 90% to compute node
   - 10% to platform

**Solana Transaction:**

```rust
// Simplified escrow release
pub fn release_escrow(ctx: Context<ReleaseEscrow>) -> Result<()> {
    let escrow = &ctx.accounts.escrow;
    let node_share = escrow.amount * 90 / 100;
    let platform_share = escrow.amount - node_share;

    // Transfer to node
    transfer(
        CpiContext::new(ctx.accounts.token_program.to_account_info(), ...),
        node_share
    )?;

    // Transfer to platform
    transfer(
        CpiContext::new(ctx.accounts.token_program.to_account_info(), ...),
        platform_share
    )?;

    // Close escrow account
    escrow.status = EscrowStatus::Released;

    Ok(())
}
```

## Pricing Model

### Cost Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| **GPU Hours** | High | Primary cost driver |
| **Model Parameters** | Medium | Memory/compute requirements |
| **Dataset Size** | Low | Transfer/storage costs |
| **Priority** | Multiplier | 1x normal, 2x high, 3x critical |

### Price Calculation

```python
def calculate_job_price(job_config):
    # Base rate per GPU hour (in CYXWIZ tokens)
    base_rate = 1.0

    # GPU multiplier based on type
    gpu_multipliers = {
        'RTX_3060': 1.0,
        'RTX_3080': 1.5,
        'RTX_4090': 3.0,
        'A100': 5.0,
    }

    # Estimate GPU hours
    estimated_hours = estimate_training_time(
        model_params=job_config.model_parameters,
        dataset_size=job_config.dataset_size,
        epochs=job_config.epochs,
        batch_size=job_config.batch_size
    )

    # Calculate price
    gpu_mult = gpu_multipliers.get(job_config.gpu_type, 1.0)
    priority_mult = job_config.priority.multiplier

    price = base_rate * estimated_hours * gpu_mult * priority_mult

    # Add platform fee
    total = price * 1.1  # 10% platform fee

    return total
```

### Price Discovery

Future: Market-based pricing where nodes set their own rates:

```rust
struct NodePricing {
    base_rate_per_hour: f64,      // Base rate in CYXWIZ
    gpu_rates: HashMap<String, f64>,  // GPU-specific rates
    minimum_job_size: u64,        // Minimum payment
    maximum_job_duration: u64,    // Max hours
}
```

## Escrow Contract

### Account Structure

```rust
#[account]
pub struct JobEscrow {
    // Identifiers
    pub job_id: [u8; 32],
    pub bump: u8,

    // Parties
    pub payer: Pubkey,           // Client wallet
    pub payee: Pubkey,           // Node wallet (set on lock)
    pub platform: Pubkey,        // Platform treasury

    // Payment
    pub amount: u64,             // Total amount in lamports/tokens
    pub token_mint: Pubkey,      // CYXWIZ or SOL

    // Status
    pub status: EscrowStatus,
    pub created_at: i64,
    pub locked_at: i64,
    pub expires_at: i64,

    // Completion
    pub completed_at: i64,
    pub model_hash: [u8; 32],
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq)]
pub enum EscrowStatus {
    Created,     // Funds deposited
    Locked,      // Assigned to node
    Released,    // Payment completed
    Refunded,    // Returned to client
    Disputed,    // Under arbitration
    Expired,     // Past expiration
}
```

### Instructions

```rust
// Create escrow (client)
pub fn create_escrow(
    ctx: Context<CreateEscrow>,
    job_id: [u8; 32],
    amount: u64,
    expiration: i64,
) -> Result<()>;

// Lock escrow (central server)
pub fn lock_escrow(
    ctx: Context<LockEscrow>,
    job_id: [u8; 32],
    node_wallet: Pubkey,
) -> Result<()>;

// Release escrow (central server after completion)
pub fn release_escrow(
    ctx: Context<ReleaseEscrow>,
    job_id: [u8; 32],
) -> Result<()>;

// Refund escrow (on cancellation/failure)
pub fn refund_escrow(
    ctx: Context<RefundEscrow>,
    job_id: [u8; 32],
) -> Result<()>;

// Claim expired (client after expiration)
pub fn claim_expired(
    ctx: Context<ClaimExpired>,
    job_id: [u8; 32],
) -> Result<()>;
```

## Failure Handling

### Job Failure

If the job fails during execution:

1. Node reports failure with reason
2. Central Server verifies failure is legitimate
3. Escrow refunded to client (minus gas fees)
4. Node reputation affected if fault is theirs

### Node Disconnection

1. Central Server detects missed heartbeats
2. Job marked as abandoned
3. Escrow unlocked and job reassigned
4. Disconnected node's stake at risk (future)

### Timeout

1. Job exceeds estimated duration + buffer
2. Central Server requests progress update
3. If no response or insufficient progress:
   - Job terminated
   - Partial refund to client
   - Partial payment to node for completed work

### Dispute Resolution

Future: Decentralized arbitration system

1. Either party raises dispute
2. Evidence submitted (logs, metrics, model)
3. Arbitrators (staked token holders) vote
4. Funds distributed based on ruling

## Gas & Fees

### Transaction Costs

| Operation | Approximate Cost |
|-----------|------------------|
| Create Escrow | ~0.002 SOL |
| Lock Escrow | ~0.0001 SOL |
| Release Escrow | ~0.0001 SOL |
| Refund Escrow | ~0.0001 SOL |

### Fee Structure

| Fee | Percentage | Recipient |
|-----|------------|-----------|
| Platform Fee | 10% | CyxWiz Treasury |
| Future: Staker Rewards | 5% | Token Stakers |
| Future: DAO Treasury | 5% | Governance |

## Security Considerations

1. **Escrow Ownership**: Only Central Server can lock/release
2. **Expiration**: Clients can reclaim after timeout
3. **No Front-Running**: Job assignment happens off-chain
4. **Proof of Compute**: Future: cryptographic verification
5. **Stake Slashing**: Future: nodes lose stake for malicious behavior

---

**Next**: [Token Economics](token-economics.md) | [Smart Contracts](smart-contracts.md)
