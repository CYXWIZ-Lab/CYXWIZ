# Blockchain-Integrated P2P Training Workflow Architecture

## Executive Summary

This document presents a comprehensive redesign of the CyxWiz P2P training workflow with **blockchain as a core architectural component**, not an afterthought. The design leverages Solana blockchain for trust, verification, payments, and decentralized governance while maintaining the performance benefits of direct P2P communication between Engine and Server Nodes.

**Key Design Principles:**
1. **Hybrid Architecture**: On-chain for trust and payments, off-chain for high-throughput data transfer
2. **Zero-Knowledge Proof of Compute**: Cryptographic verification without revealing training data
3. **Economic Incentives**: Token-based reputation system and stake-based security
4. **Progressive Decentralization**: Gradual reduction of Central Server's role over time
5. **Gas Optimization**: Minimize on-chain transactions while maintaining security guarantees

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Blockchain Integration Points](#blockchain-integration-points)
3. [Smart Contract Architecture](#smart-contract-architecture)
4. [Complete Workflow with Blockchain](#complete-workflow-with-blockchain)
5. [Payment and Escrow Flow](#payment-and-escrow-flow)
6. [Trust and Verification Mechanisms](#trust-and-verification-mechanisms)
7. [Token Economics](#token-economics)
8. [Security Model](#security-model)
9. [Scalability Considerations](#scalability-considerations)
10. [Implementation Roadmap](#implementation-roadmap)
11. [Trade-offs and Design Decisions](#trade-offs-and-design-decisions)

---

## System Architecture Overview

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SOLANA BLOCKCHAIN                                │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ JobEscrow       │  │ NodeRegistry │  │ ReputationManager        │  │
│  │ Program (Smart  │  │ Program      │  │ Program                  │  │
│  │ Contract)       │  │              │  │                          │  │
│  │                 │  │              │  │                          │  │
│  │ - Create escrow │  │ - Register   │  │ - Update reputation      │  │
│  │ - Lock funds    │  │   nodes      │  │ - Slash stakes           │  │
│  │ - Release       │  │ - Verify     │  │ - Reward good behavior   │  │
│  │   payment       │  │   stakes     │  │ - Track job history      │  │
│  │ - Refund        │  │ - Heartbeat  │  │                          │  │
│  │ - Dispute       │  │   tracking   │  │                          │  │
│  └────────┬────────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│           │                  │                      │                   │
│           └──────────────────┴──────────────────────┘                   │
│                              │                                          │
│                    ┌─────────▼──────────┐                              │
│                    │ CYXWIZ Token (SPL) │                              │
│                    │ - Payments         │                              │
│                    │ - Staking          │                              │
│                    │ - Governance       │                              │
│                    └────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ On-chain transactions
                              │ (Payment, Escrow, Reputation)
                              │
┌─────────────────────────────▼─────────────────────────────────────────┐
│                      CENTRAL SERVER (Rust)                             │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Job Scheduler    │  │ Node         │  │ Blockchain Gateway   │    │
│  │                  │  │ Discovery    │  │                      │    │
│  │ - Match jobs to  │  │              │  │ - Monitor on-chain   │    │
│  │   nodes          │  │ - Filter by  │  │   events             │    │
│  │ - Optimize for   │  │   reputation │  │ - Submit txs         │    │
│  │   latency/cost   │  │ - Location   │  │ - Verify escrow      │    │
│  │ - Load balancing │  │   awareness  │  │ - Update reputation  │    │
│  └──────────────────┘  └──────────────┘  └──────────────────────┘    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Off-Chain Database (PostgreSQL/SQLite)                           │ │
│  │ - Job metadata, status, progress                                 │ │
│  │ - Node endpoints, capabilities, session tokens                   │ │
│  │ - Cache of on-chain reputation data                              │ │
│  │ - P2P connection credentials (JWT tokens)                        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────┬──────────────────────┬────────────────────────────┘
                    │                      │
                    │ gRPC (Job discovery) │ gRPC (Node registration)
                    │                      │
       ┌────────────▼─────────┐   ┌────────▼──────────┐
       │   Engine (Client)    │   │  Server Node      │
       │                      │   │  (Compute Worker) │
       │  ┌────────────────┐  │   │                   │
       │  │ Wallet Manager │  │   │  ┌─────────────┐  │
       │  │ - Sign txs     │  │   │  │ Stake Manager│ │
       │  │ - Approve      │  │   │  │ - Lock stake│  │
       │  │   escrow       │  │   │  │ - Earn      │  │
       │  │ - Check balance│  │   │  │   rewards   │  │
       │  └────────────────┘  │   │  └─────────────┘  │
       └──────────┬───────────┘   └─────────┬─────────┘
                  │                         │
                  │  Direct P2P Connection  │
                  │  (gRPC over TLS)        │
                  │                         │
                  │  - Training data        │
                  │  - Model weights        │
                  │  - Real-time metrics    │
                  │  - Checkpoints          │
                  │                         │
                  └─────────────────────────┘
```

### On-Chain vs Off-Chain Separation

| Data/Operation | On-Chain (Solana) | Off-Chain (Central Server + P2P) |
|----------------|-------------------|----------------------------------|
| **Payment escrow** | ✓ (Smart contract) | × |
| **Payment release** | ✓ (Verifiable on-chain) | × |
| **Node registration** | ✓ (Public registry) | × (Cache for performance) |
| **Node reputation** | ✓ (Immutable history) | × (Cache, updated periodically) |
| **Stake deposits** | ✓ (Locked in contract) | × |
| **Dispute resolution** | ✓ (DAO governance) | × (Initial claim filed off-chain) |
| **Job metadata** | × (Too expensive) | ✓ (Indexed, searchable) |
| **Job discovery** | × (Requires fast queries) | ✓ (Central Server matchmaking) |
| **Training data transfer** | × (Gas cost prohibitive) | ✓ (Direct P2P, high throughput) |
| **Real-time metrics** | × (Too frequent) | ✓ (Streamed P2P) |
| **Model weights** | × (Too large) | ✓ (P2P download, hash on-chain) |
| **Proof of compute** | ✓ (Merkle root/hash) | × (Full proof off-chain) |
| **Job completion event** | ✓ (Triggers payment) | × (Node signals first) |

**Design Rationale:**
- **On-chain**: Financial operations, trust anchors, immutable audit trails
- **Off-chain**: High-frequency data, large payloads, latency-sensitive operations

---

## Blockchain Integration Points

### 1. Job Lifecycle Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         JOB LIFECYCLE                                    │
│                                                                          │
│  [1. Job Submission] ──> [2. Escrow Creation] ──> [3. Node Assignment]  │
│         │                        │                         │             │
│         │ Off-chain              │ ON-CHAIN               │ Off-chain   │
│         │ (Central Server)       │ (Solana tx)            │ (Matching)  │
│         │                        │                         │             │
│         ▼                        ▼                         ▼             │
│  Engine sends job         User approves tx          Central Server      │
│  config + payment         Escrow smart contract     finds best node     │
│  amount to Central        locks CYXWIZ tokens       Returns endpoint +  │
│  Server via gRPC                                    JWT token           │
│                                                                          │
│  [4. P2P Training] ──> [5. Progress Updates] ──> [6. Completion]        │
│         │                        │                         │             │
│         │ Off-chain              │ Mixed                  │ ON-CHAIN    │
│         │ (Engine ↔ Node)        │ (Off-chain streaming,  │ (Payment)   │
│         │                        │  periodic on-chain)    │             │
│         ▼                        ▼                         ▼             │
│  Engine ↔ Node           Node streams metrics      Node submits proof   │
│  gRPC bidirectional      to Engine (real-time)     Smart contract       │
│  Training data +         Periodic checkpoints      verifies + releases  │
│  Model weights           reported to Central       payment: 90% node,   │
│  Real-time updates       Server (every epoch)      10% platform         │
│                          Central Server updates                         │
│                          on-chain reputation                             │
│                          (batched, every 10 jobs)                        │
│                                                                          │
│  [7. Reputation Update] ──> [8. Stake Return]                           │
│         │                            │                                   │
│         │ ON-CHAIN                  │ ON-CHAIN                          │
│         │ (Solana tx)                │ (Auto after unlock period)        │
│         ▼                            ▼                                   │
│  ReputationManager         If job successful:                           │
│  program updates node      - Release node's stake                       │
│  score based on:           - Add bonus to stake pool                    │
│  - Completion rate         If job failed:                               │
│  - Average time            - Slash stake (partial)                      │
│  - User feedback           - Penalty to reputation                      │
│                            Stake available for                           │
│                            withdrawal after cooldown                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Trust Establishment Points

```
┌──────────────────────────────────────────────────────────────┐
│              TRUST VERIFICATION CHECKPOINTS                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ▸ Node Registration (On-chain)                              │
│    └─> Node stakes minimum CYXWIZ tokens                     │
│    └─> Wallet address verified                               │
│    └─> Initial reputation score = 0 (unproven)               │
│                                                               │
│  ▸ Job Assignment (Off-chain + On-chain verification)        │
│    └─> Central Server checks on-chain stake                  │
│    └─> Verifies reputation score meets job requirements      │
│    └─> Issues JWT token signed by server                     │
│                                                               │
│  ▸ P2P Connection (Off-chain with cryptographic auth)        │
│    └─> Engine verifies JWT token                             │
│    └─> Node verifies token with Central Server               │
│    └─> TLS certificate pinning (node's public key)           │
│                                                               │
│  ▸ Training Progress (Periodic on-chain anchoring)           │
│    └─> Node sends progress to Engine (off-chain)             │
│    └─> Node sends checkpoints to Central Server              │
│    └─> Central Server verifies checkpoint hashes             │
│    └─> Merkle root posted on-chain (every N epochs)          │
│                                                               │
│  ▸ Job Completion (On-chain verification)                    │
│    └─> Node submits final weights hash                       │
│    └─> Node submits proof-of-compute                         │
│    └─> Smart contract verifies proof                         │
│    └─> Payment released only after verification              │
│                                                               │
│  ▸ Dispute Resolution (On-chain + DAO)                       │
│    └─> User/Node raises dispute on-chain                     │
│    └─> Evidence submitted (checkpoint hashes, logs)          │
│    └─> DAO votes on resolution                               │
│    └─> Smart contract executes verdict (refund/slash)        │
└──────────────────────────────────────────────────────────────┘
```

### 3. Payment Flow Integration

The payment system is **entirely on-chain** for trustlessness:

```
                    PAYMENT LIFECYCLE

┌─────────────────────────────────────────────────┐
│ 1. ESCROW CREATION (Before training starts)    │
├─────────────────────────────────────────────────┤
│                                                 │
│  User (Engine) ──> Solana Transaction          │
│                    │                            │
│                    ├─> JobEscrow.create()       │
│                    │   - job_id: UUID           │
│                    │   - amount: u64 (tokens)   │
│                    │   - user_wallet: Pubkey    │
│                    │   - timeout: 7 days        │
│                    │                            │
│                    └─> Result:                  │
│                        - escrow_account: PDA    │
│                        - tx_signature: String   │
│                                                 │
│  Smart Contract Actions:                        │
│  ✓ Transfer tokens from user → escrow PDA      │
│  ✓ Set escrow state = LOCKED                   │
│  ✓ Record job_id, timestamp, parties           │
│  ✓ Emit EscrowCreated event                    │
│                                                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ 2. PAYMENT RELEASE (After successful training)  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Server Node ──> Solana Transaction             │
│                  │                              │
│                  ├─> JobEscrow.complete()       │
│                  │   - job_id: UUID             │
│                  │   - proof_of_compute: Hash   │
│                  │   - final_weights_hash: Hash │
│                  │   - node_wallet: Pubkey      │
│                  │                              │
│                  └─> Smart Contract Verifies:   │
│                      ✓ Job exists and is active │
│                      ✓ Proof is valid (verify() │
│                         function call)          │
│                      ✓ Node is assigned to job  │
│                      ✓ Escrow is still locked   │
│                                                 │
│  Payment Distribution:                          │
│  ┌─────────────────────────────────────┐       │
│  │ Total Amount: 100 CYXWIZ            │       │
│  ├─────────────────────────────────────┤       │
│  │ Node Reward:  90 CYXWIZ (90%)       │       │
│  │ Platform Fee: 10 CYXWIZ (10%)       │       │
│  └─────────────────────────────────────┘       │
│                                                 │
│  Smart Contract Actions:                        │
│  ✓ Transfer 90 tokens → node_wallet            │
│  ✓ Transfer 10 tokens → platform_wallet        │
│  ✓ Set escrow state = COMPLETED                │
│  ✓ Update node reputation (+1 job)             │
│  ✓ Emit PaymentReleased event                  │
│                                                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ 3. REFUND (If job fails or is cancelled)       │
├─────────────────────────────────────────────────┤
│                                                 │
│  Trigger Conditions:                            │
│  • User cancels job before completion           │
│  • Node fails to complete within timeout        │
│  • Node goes offline (no heartbeat for 2 min)   │
│  • Dispute resolved in user's favor             │
│                                                 │
│  Central Server or User ──> Solana Transaction  │
│                             │                   │
│                             ├─> JobEscrow.      │
│                             │   refund()        │
│                             │   - job_id        │
│                             │   - reason_code   │
│                             │                   │
│                             └─> Smart Contract: │
│                                 ✓ Verify refund │
│                                   condition     │
│                                 ✓ Calculate     │
│                                   partial refund│
│                                   if needed     │
│                                                 │
│  Refund Calculation:                            │
│  • 0% progress = 100% refund                    │
│  • 50% progress = 50% refund (node gets 50%)    │
│  • 100% progress = 0% refund (job completed)    │
│                                                 │
│  Smart Contract Actions:                        │
│  ✓ Transfer refund amount → user_wallet        │
│  ✓ Transfer earned amount → node_wallet        │
│  ✓ Set escrow state = REFUNDED                 │
│  ✓ Update node reputation (penalty if fault)   │
│  ✓ Emit EscrowRefunded event                   │
│                                                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ 4. TIMEOUT AUTO-REFUND (Escrow safety)         │
├─────────────────────────────────────────────────┤
│                                                 │
│  If job_created_at + 7 days < now               │
│  AND escrow state == LOCKED                     │
│                                                 │
│  Anyone can call ──> JobEscrow.timeout_refund() │
│                                                 │
│  Smart Contract:                                │
│  ✓ Verify timeout condition                    │
│  ✓ Transfer all funds → user_wallet            │
│  ✓ Set escrow state = TIMEOUT_REFUNDED         │
│  ✓ Slash node's stake (penalty for timeout)    │
│  ✓ Emit TimeoutRefund event                    │
│                                                 │
│  This prevents funds from being locked forever  │
│  if node disappears or Central Server fails     │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Smart Contract Architecture

### Solana Program Structure

```rust
// File: cyxwiz-blockchain/programs/job-escrow/src/lib.rs

use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

declare_id!("CyxWizJobEscrowProgramIDHere111111111111111");

#[program]
pub mod job_escrow {
    use super::*;

    /// Create an escrow for a training job
    pub fn create_escrow(
        ctx: Context<CreateEscrow>,
        job_id: [u8; 16],  // UUID as bytes
        amount: u64,
        timeout_days: u8,
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        escrow.job_id = job_id;
        escrow.user = ctx.accounts.user.key();
        escrow.amount = amount;
        escrow.state = EscrowState::Locked;
        escrow.created_at = Clock::get()?.unix_timestamp;
        escrow.timeout_at = escrow.created_at + (timeout_days as i64 * 86400);
        escrow.bump = *ctx.bumps.get("escrow").unwrap();

        // Transfer tokens from user to escrow PDA
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.user_token.to_account_info(),
                    to: ctx.accounts.escrow_token.to_account_info(),
                    authority: ctx.accounts.user.to_account_info(),
                },
            ),
            amount,
        )?;

        emit!(EscrowCreated {
            job_id,
            user: escrow.user,
            amount,
            timestamp: escrow.created_at,
        });

        Ok(())
    }

    /// Complete job and release payment to node
    pub fn complete_payment(
        ctx: Context<CompletePayment>,
        job_id: [u8; 16],
        proof_of_compute: [u8; 32],  // SHA256 hash
        final_weights_hash: [u8; 32],
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;

        require!(escrow.job_id == job_id, EscrowError::JobIdMismatch);
        require!(escrow.state == EscrowState::Locked, EscrowError::InvalidState);

        // Verify proof of compute (simplified - real implementation would check merkle proof)
        require!(
            verify_proof_of_compute(proof_of_compute, final_weights_hash),
            EscrowError::InvalidProof
        );

        // Calculate distribution (90% node, 10% platform)
        let node_amount = (escrow.amount * 90) / 100;
        let platform_amount = escrow.amount - node_amount;

        let seeds = &[
            b"escrow",
            &escrow.job_id,
            &[escrow.bump],
        ];
        let signer = &[&seeds[..]];

        // Transfer to node
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.escrow_token.to_account_info(),
                    to: ctx.accounts.node_token.to_account_info(),
                    authority: escrow.to_account_info(),
                },
                signer,
            ),
            node_amount,
        )?;

        // Transfer to platform
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.escrow_token.to_account_info(),
                    to: ctx.accounts.platform_token.to_account_info(),
                    authority: escrow.to_account_info(),
                },
                signer,
            ),
            platform_amount,
        )?;

        escrow.state = EscrowState::Completed;
        escrow.completed_at = Clock::get()?.unix_timestamp;

        emit!(PaymentReleased {
            job_id,
            node: ctx.accounts.node.key(),
            node_amount,
            platform_amount,
            timestamp: escrow.completed_at,
        });

        Ok(())
    }

    /// Refund escrow to user (partial or full)
    pub fn refund(
        ctx: Context<Refund>,
        job_id: [u8; 16],
        progress_percentage: u8,  // 0-100
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;

        require!(escrow.job_id == job_id, EscrowError::JobIdMismatch);
        require!(escrow.state == EscrowState::Locked, EscrowError::InvalidState);

        // Calculate refund based on progress
        let refund_percentage = 100u64.saturating_sub(progress_percentage as u64);
        let refund_amount = (escrow.amount * refund_percentage) / 100;
        let node_amount = escrow.amount - refund_amount;

        let seeds = &[b"escrow", &escrow.job_id, &[escrow.bump]];
        let signer = &[&seeds[..]];

        // Refund to user
        if refund_amount > 0 {
            token::transfer(
                CpiContext::new_with_signer(
                    ctx.accounts.token_program.to_account_info(),
                    Transfer {
                        from: ctx.accounts.escrow_token.to_account_info(),
                        to: ctx.accounts.user_token.to_account_info(),
                        authority: escrow.to_account_info(),
                    },
                    signer,
                ),
                refund_amount,
            )?;
        }

        // Partial payment to node if work was done
        if node_amount > 0 {
            token::transfer(
                CpiContext::new_with_signer(
                    ctx.accounts.token_program.to_account_info(),
                    Transfer {
                        from: ctx.accounts.escrow_token.to_account_info(),
                        to: ctx.accounts.node_token.to_account_info(),
                        authority: escrow.to_account_info(),
                    },
                    signer,
                ),
                node_amount,
            )?;
        }

        escrow.state = EscrowState::Refunded;
        escrow.refunded_at = Clock::get()?.unix_timestamp;

        emit!(EscrowRefunded {
            job_id,
            user: escrow.user,
            refund_amount,
            node_amount,
            timestamp: escrow.refunded_at,
        });

        Ok(())
    }

    /// Timeout refund - anyone can trigger after timeout period
    pub fn timeout_refund(
        ctx: Context<TimeoutRefund>,
        job_id: [u8; 16],
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let clock = Clock::get()?;

        require!(escrow.job_id == job_id, EscrowError::JobIdMismatch);
        require!(escrow.state == EscrowState::Locked, EscrowError::InvalidState);
        require!(
            clock.unix_timestamp >= escrow.timeout_at,
            EscrowError::TimeoutNotReached
        );

        let seeds = &[b"escrow", &escrow.job_id, &[escrow.bump]];
        let signer = &[&seeds[..]];

        // Full refund to user
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.escrow_token.to_account_info(),
                    to: ctx.accounts.user_token.to_account_info(),
                    authority: escrow.to_account_info(),
                },
                signer,
            ),
            escrow.amount,
        )?;

        escrow.state = EscrowState::TimeoutRefunded;
        escrow.refunded_at = clock.unix_timestamp;

        emit!(TimeoutRefund {
            job_id,
            user: escrow.user,
            amount: escrow.amount,
            timestamp: escrow.refunded_at,
        });

        Ok(())
    }
}

// Account structures
#[account]
pub struct JobEscrow {
    pub job_id: [u8; 16],       // 16 bytes
    pub user: Pubkey,            // 32 bytes
    pub amount: u64,             // 8 bytes
    pub state: EscrowState,      // 1 byte
    pub created_at: i64,         // 8 bytes
    pub timeout_at: i64,         // 8 bytes
    pub completed_at: i64,       // 8 bytes
    pub refunded_at: i64,        // 8 bytes
    pub bump: u8,                // 1 byte
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum EscrowState {
    Locked,
    Completed,
    Refunded,
    TimeoutRefunded,
}

// Events
#[event]
pub struct EscrowCreated {
    pub job_id: [u8; 16],
    pub user: Pubkey,
    pub amount: u64,
    pub timestamp: i64,
}

#[event]
pub struct PaymentReleased {
    pub job_id: [u8; 16],
    pub node: Pubkey,
    pub node_amount: u64,
    pub platform_amount: u64,
    pub timestamp: i64,
}

#[event]
pub struct EscrowRefunded {
    pub job_id: [u8; 16],
    pub user: Pubkey,
    pub refund_amount: u64,
    pub node_amount: u64,
    pub timestamp: i64,
}

// Context structs
#[derive(Accounts)]
#[instruction(job_id: [u8; 16])]
pub struct CreateEscrow<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + std::mem::size_of::<JobEscrow>(),
        seeds = [b"escrow", &job_id],
        bump
    )]
    pub escrow: Account<'info, JobEscrow>,

    #[account(mut)]
    pub user: Signer<'info>,

    #[account(mut)]
    pub user_token: Account<'info, TokenAccount>,

    #[account(mut)]
    pub escrow_token: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

// ... (similar context structs for CompletePayment, Refund, TimeoutRefund)

// Helper functions
fn verify_proof_of_compute(proof: [u8; 32], weights_hash: [u8; 32]) -> bool {
    // TODO: Implement actual zero-knowledge proof verification
    // For now, just verify that proof is non-zero
    proof != [0u8; 32] && weights_hash != [0u8; 32]
}

// Errors
#[error_code]
pub enum EscrowError {
    #[msg("Job ID mismatch")]
    JobIdMismatch,
    #[msg("Invalid escrow state")]
    InvalidState,
    #[msg("Invalid proof of compute")]
    InvalidProof,
    #[msg("Timeout not reached")]
    TimeoutNotReached,
}
```

### Additional Smart Contracts

**NodeRegistry Program** (tracks registered nodes, stakes, reputation):
```rust
// cyxwiz-blockchain/programs/node-registry/src/lib.rs

#[program]
pub mod node_registry {
    // register_node() - Node stakes tokens to register
    // update_heartbeat() - Node proves liveness
    // slash_stake() - Penalize malicious nodes
    // withdraw_stake() - Withdraw after cooldown period
}

#[account]
pub struct NodeInfo {
    pub node_id: [u8; 32],
    pub owner: Pubkey,
    pub stake_amount: u64,
    pub reputation_score: u32,  // 0-1000
    pub total_jobs_completed: u64,
    pub total_jobs_failed: u64,
    pub last_heartbeat: i64,
    pub registered_at: i64,
    pub stake_locked_until: i64,
}
```

**ReputationManager Program** (updates scores, tracks history):
```rust
// cyxwiz-blockchain/programs/reputation-manager/src/lib.rs

#[program]
pub mod reputation_manager {
    // update_reputation() - Called after job completion
    // report_misbehavior() - User/Node reports issues
    // resolve_dispute() - DAO governance resolution
}

#[account]
pub struct ReputationScore {
    pub node_id: [u8; 32],
    pub score: u32,  // 0-1000
    pub completion_rate: u16,  // 0-10000 (basis points)
    pub average_time_ratio: u16,  // actual/estimated * 1000
    pub user_ratings_sum: u64,
    pub user_ratings_count: u32,
}
```

---

## Complete Workflow with Blockchain

### Phase 1: Job Submission with On-Chain Escrow

```
┌─────────────┐                  ┌─────────────────┐                ┌──────────────┐
│   Engine    │                  │ Central Server  │                │   Solana     │
│   (Client)  │                  │  (Coordinator)  │                │  Blockchain  │
└──────┬──────┘                  └────────┬────────┘                └──────┬───────┘
       │                                  │                                │
       │ 1. User designs model            │                                │
       │    Sets hyperparameters          │                                │
       │    Estimates cost: 100 CYXWIZ    │                                │
       │                                  │                                │
       │ 2. SubmitJobRequest              │                                │
       │    (job_config, no escrow yet)   │                                │
       │──────────────────────────────────>│                                │
       │                                  │                                │
       │                                  │ 3. Find suitable nodes         │
       │                                  │    - Query on-chain registry   │
       │                                  │    - Filter by stake > min     │
       │                                  │    - Check reputation > 700    │
       │                                  │    - Match hardware specs      │
       │                                  │                                │
       │                                  │ 4. Reserve best node           │
       │                                  │    (hold for 5 minutes)        │
       │                                  │                                │
       │ 5. NodeAssignmentReady           │                                │
       │    - node_endpoint               │                                │
       │    - estimated_cost: 100 CYXWIZ  │                                │
       │    - escrow_deadline: 5 min      │                                │
       │<──────────────────────────────────│                                │
       │                                  │                                │
       │ 6. User approves transaction     │                                │
       │    in Wallet UI (Phantom/etc)    │                                │
       │                                  │                                │
       │ 7. Solana Transaction:           │                                │
       │    JobEscrow.create_escrow()     │                                │
       │    - job_id                      │                                │
       │    - amount: 100 CYXWIZ          │                                │
       │    - timeout: 7 days             │                                │
       │──────────────────────────────────────────────────────────────────>│
       │                                  │                                │
       │                                  │                                │ Smart Contract:
       │                                  │                                │ ✓ Create escrow PDA
       │                                  │                                │ ✓ Transfer 100 tokens
       │                                  │                                │ ✓ State = LOCKED
       │                                  │                                │ ✓ Emit EscrowCreated
       │                                  │                                │
       │ 8. Transaction confirmed         │                                │
       │    tx_signature: "abc123..."     │                                │
       │<──────────────────────────────────────────────────────────────────│
       │                                  │                                │
       │ 9. ConfirmEscrow                 │                                │
       │    - job_id                      │                                │
       │    - escrow_tx: "abc123..."      │                                │
       │──────────────────────────────────>│                                │
       │                                  │                                │
       │                                  │ 10. Verify escrow on-chain     │
       │                                  │     (check tx exists, amount)  │
       │                                  │──────────────────────────────> │
       │                                  │                                │
       │                                  │ 11. Escrow verified            │
       │                                  │<────────────────────────────── │
       │                                  │                                │
       │                                  │ 12. Update job status          │
       │                                  │     Status: ESCROW_CONFIRMED   │
       │                                  │     Generate JWT auth_token    │
       │                                  │                                │
       │ 13. StartJobResponse             │                                │
       │     - auth_token (for P2P)       │                                │
       │     - node_endpoint              │                                │
       │<──────────────────────────────────│                                │
       │                                  │                                │
```

**Key Points:**
- Escrow is created **before** training starts
- Engine doesn't proceed until on-chain confirmation
- Central Server verifies escrow before issuing P2P credentials
- Job automatically refunds if escrow not confirmed within 5 minutes

### Phase 2: P2P Training with Periodic On-Chain Checkpoints

```
┌─────────────┐                  ┌─────────────────┐                ┌──────────────┐
│   Engine    │                  │  Server Node    │                │   Central    │
│             │                  │                 │                │   Server     │
└──────┬──────┘                  └────────┬────────┘                └──────┬───────┘
       │                                  │                                │
       │ 14. ConnectToNode(auth_token)    │                                │
       │─────────────────────────────────>│                                │
       │                                  │                                │
       │                                  │ 15. Verify token with Central  │
       │                                  │─────────────────────────────> │
       │                                  │                                │
       │                                  │ 16. Token valid, job_id match  │
       │                                  │<────────────────────────────── │
       │                                  │                                │
       │ 17. ConnectionAccepted           │                                │
       │<─────────────────────────────────│                                │
       │                                  │                                │
       │ 18. SendJob(config, dataset)     │                                │
       │─────────────────────────────────>│                                │
       │                                  │                                │
       │                                  │ 19. Validate job, start training│
       │                                  │                                │
       │ 20. JobAccepted                  │                                │
       │<─────────────────────────────────│                                │
       │                                  │                                │
       │                                  │ 21. NotifyJobAccepted          │
       │                                  │─────────────────────────────> │
       │                                  │                                │
       │                                  │                                │ Update DB:
       │                                  │                                │ Status: RUNNING
       │                                  │                                │
       │ 22. StreamTrainingMetrics        │                                │
       │     (bidirectional gRPC stream)  │                                │
       │<────────────────────────────────>│                                │
       │                                  │                                │
       │ ← Epoch 1/100, Loss: 0.543       │                                │
       │ ← GPU: 85%, ETA: 45min           │                                │
       │ ← Epoch 2/100, Loss: 0.421       │                                │
       │                                  │                                │
       │                                  │ Every 10 epochs:               │
       │                                  │ 23. ReportProgress             │
       │                                  │     - job_id                   │
       │                                  │     - current_epoch: 10        │
       │                                  │     - checkpoint_hash          │
       │                                  │─────────────────────────────> │
       │                                  │                                │
       │                                  │                                │ Store checkpoint hash
       │                                  │                                │ Update progress in DB
       │                                  │                                │
       │ ← Epoch 10/100 checkpoint        │ 24. ACK                        │
       │                                  │<────────────────────────────── │
       │                                  │                                │
       │ ... Training continues ...       │                                │
       │                                  │                                │
       │ User can pause/resume/stop ──────>│                                │
       │                                  │                                │
```

**Checkpoint Strategy:**
- **Real-time metrics**: Streamed P2P (off-chain) every 5 seconds
- **Progress checkpoints**: Sent to Central Server every 10 epochs
- **On-chain anchoring**: Merkle root of checkpoints posted every 50 epochs (gas optimization)

### Phase 3: Completion with On-Chain Payment Release

```
┌─────────────┐       ┌─────────────────┐       ┌─────────────────┐    ┌──────────────┐
│   Engine    │       │  Server Node    │       │ Central Server  │    │   Solana     │
└──────┬──────┘       └────────┬────────┘       └────────┬────────┘    └──────┬───────┘
       │                       │                         │                     │
       │                       │ Training completes      │                     │
       │                       │                         │                     │
       │ 25. TrainingComplete  │                         │                     │
       │    - final_weights    │                         │                     │
       │    - final_metrics    │                         │                     │
       │    - proof_of_compute │                         │                     │
       │<──────────────────────│                         │                     │
       │                       │                         │                     │
       │ 26. DownloadWeights   │                         │                     │
       │─────────────────────> │                         │                     │
       │                       │                         │                     │
       │ 27. Model file stream │                         │                     │
       │<──────────────────────│                         │                     │
       │                       │                         │                     │
       │ 28. Verify weights    │                         │                     │
       │     hash matches      │                         │                     │
       │                       │                         │                     │
       │                       │ 29. ReportCompletion    │                     │
       │                       │     - job_id            │                     │
       │                       │     - final_weights_hash│                     │
       │                       │     - proof_of_compute  │                     │
       │                       │─────────────────────────>│                     │
       │                       │                         │                     │
       │                       │                         │ 30. Validate proof  │
       │                       │                         │     - Verify hashes │
       │                       │                         │     - Check training│
       │                       │                         │       time reasonable│
       │                       │                         │                     │
       │                       │                         │ 31. Submit on-chain │
       │                       │                         │     JobEscrow.      │
       │                       │                         │     complete()      │
       │                       │                         │─────────────────────>│
       │                       │                         │                     │
       │                       │                         │                     │ Smart Contract:
       │                       │                         │                     │ ✓ Verify proof
       │                       │                         │                     │ ✓ Release payment
       │                       │                         │                     │   90 → node
       │                       │                         │                     │   10 → platform
       │                       │                         │                     │ ✓ State = COMPLETED
       │                       │                         │                     │ ✓ Emit PaymentReleased
       │                       │                         │                     │
       │                       │                         │ 32. Tx confirmed    │
       │                       │                         │<─────────────────────│
       │                       │                         │                     │
       │                       │ 33. PaymentConfirmation │                     │
       │                       │     - tx_hash           │                     │
       │                       │     - amount: 90 CYXWIZ │                     │
       │                       │<─────────────────────────│                     │
       │                       │                         │                     │
       │                       │                         │ 34. Update reputation│
       │                       │                         │     (batched)       │
       │                       │                         │─────────────────────>│
       │                       │                         │                     │
       │                       │                         │                     │ ReputationManager:
       │                       │                         │                     │ ✓ +10 score
       │                       │                         │                     │ ✓ +1 jobs_completed
       │                       │                         │                     │
       │ 35. GetJobStatus      │                         │                     │
       │───────────────────────────────────────────────> │                     │
       │                       │                         │                     │
       │ 36. JobStatus         │                         │                     │
       │     Status: COMPLETED │                         │                     │
       │     Payment: RELEASED │                         │                     │
       │<────────────────────────────────────────────────│                     │
       │                       │                         │                     │
```

**Payment Security:**
- Payment only released after on-chain proof verification
- Smart contract validates proof-of-compute hash
- Automatic distribution (no manual intervention)
- Transaction signature provides immutable payment proof

---

## Trust and Verification Mechanisms

### 1. Proof of Compute

The challenge: How does the network verify that a Server Node actually performed the training work without re-running the entire computation?

**Solution: Incremental Checkpoint Hashing + Zero-Knowledge Proofs**

```
┌────────────────────────────────────────────────────────────────┐
│                PROOF OF COMPUTE GENERATION                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Server Node during training:                                   │
│                                                                 │
│  For each epoch E (1 to N):                                     │
│    1. Train on dataset, compute loss and accuracy               │
│    2. Extract model weights: W_E                                │
│    3. Compute checkpoint hash:                                  │
│       H_E = SHA256(H_{E-1} || W_E || metrics_E || timestamp)    │
│                                                                 │
│  After final epoch N:                                           │
│    4. Build Merkle tree from all checkpoint hashes:             │
│       Merkle Tree: [H_1, H_2, ..., H_N]                         │
│    5. Compute Merkle root: ROOT                                 │
│    6. Generate zero-knowledge proof:                            │
│       PROOF = zk_prove(                                         │
│           statement: "I trained this model for N epochs",       │
│           witness: [W_1, ..., W_N, metrics_1, ..., metrics_N],  │
│           public: [ROOT, final_weights_hash]                    │
│       )                                                         │
│                                                                 │
│  Submit to blockchain:                                          │
│    - Merkle root: ROOT                                          │
│    - Final weights hash: SHA256(W_N)                            │
│    - Zero-knowledge proof: PROOF                                │
│    - Epoch count: N                                             │
│    - Total training time: T seconds                             │
│                                                                 │
├────────────────────────────────────────────────────────────────┤
│                    VERIFICATION (On-Chain)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Smart Contract verifies:                                       │
│    1. ✓ Zero-knowledge proof is valid (cryptographic check)    │
│    2. ✓ Final weights hash matches user-downloaded weights      │
│    3. ✓ Merkle root is well-formed                              │
│    4. ✓ Epoch count N ≥ required epochs                         │
│    5. ✓ Training time T within reasonable bounds:               │
│          estimated_time * 0.5 < T < estimated_time * 2.0        │
│    6. ✓ Node's reputation score ≥ minimum threshold             │
│                                                                 │
│  If all checks pass:                                            │
│    → Release payment                                            │
│    → Update node reputation (+10 score)                         │
│                                                                 │
│  If any check fails:                                            │
│    → Initiate dispute process                                   │
│    → Lock funds pending investigation                           │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- **Efficient**: Verifier doesn't re-run training (zk-SNARK proof is constant-size)
- **Trustless**: Cryptographically guaranteed correctness
- **Privacy-preserving**: Proof doesn't reveal training data or intermediate weights
- **Gas-optimized**: Single on-chain verification transaction

**Implementation Notes:**
- Use **Groth16** or **PLONK** for zk-SNARK construction
- Proof generation adds ~5-10% overhead to training time
- Proof verification is < 1ms on-chain (< 100,000 gas)

### 2. Node Reputation System

```
┌──────────────────────────────────────────────────────────────┐
│              REPUTATION SCORE CALCULATION                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Score = Σ (component_score * weight)                         │
│                                                               │
│  Components:                                                  │
│                                                               │
│  1. Completion Rate (40% weight)                              │
│     score = (jobs_completed / jobs_assigned) * 400            │
│     Example: 95/100 jobs = 380 points                         │
│                                                               │
│  2. Time Accuracy (25% weight)                                │
│     score = avg(1 - |actual_time - estimated_time| /          │
│                     estimated_time) * 250                     │
│     Example: avg 10% deviation = 225 points                   │
│                                                               │
│  3. User Ratings (20% weight)                                 │
│     score = avg(user_ratings) * 200 / 5                       │
│     Example: 4.5/5.0 avg rating = 180 points                  │
│                                                               │
│  4. Proof Quality (10% weight)                                │
│     score = (valid_proofs / total_proofs) * 100               │
│     Example: 100% valid proofs = 100 points                   │
│                                                               │
│  5. Uptime (5% weight)                                        │
│     score = (heartbeats_received / heartbeats_expected) * 50  │
│     Example: 98% uptime = 49 points                           │
│                                                               │
│  Total Score: 0-1000 points                                   │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Reputation Tiers:                                         ││
│  │                                                           ││
│  │  🥉 Bronze (0-299):   Can accept small jobs (<10 tokens) ││
│  │  🥈 Silver (300-599): Can accept medium jobs (<50 tokens)││
│  │  🥇 Gold (600-799):   Can accept large jobs (<200 tokens)││
│  │  💎 Diamond (800-1000): Can accept any size job          ││
│  │                         Priority in matchmaking           ││
│  │                         Lower platform fees (8% vs 10%)   ││
│  └──────────────────────────────────────────────────────────┘│
│                                                               │
│  Penalties:                                                   │
│  • Failed job: -50 points                                     │
│  • Timeout: -30 points                                        │
│  • Invalid proof: -100 points + stake slash (10%)             │
│  • Dispute resolved against node: -200 points + stake slash   │
│                                                               │
│  Recovery:                                                    │
│  • Successful job: +10 points                                 │
│  • Consistent uptime (30 days): +20 points                    │
│  • User 5-star rating: +5 points                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 3. Stake-Based Security

```
┌─────────────────────────────────────────────────────────────┐
│                NODE STAKING MECHANISM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Minimum Stake Requirements (based on job size):             │
│                                                              │
│  ┌────────────────────┬──────────────┬────────────────────┐ │
│  │ Job Payment Amount │ Min Stake    │ Slashing Risk      │ │
│  ├────────────────────┼──────────────┼────────────────────┤ │
│  │ < 10 CYXWIZ        │ 50 CYXWIZ    │ 10% on failure     │ │
│  │ 10-50 CYXWIZ       │ 200 CYXWIZ   │ 15% on failure     │ │
│  │ 50-200 CYXWIZ      │ 1000 CYXWIZ  │ 20% on failure     │ │
│  │ > 200 CYXWIZ       │ 5000 CYXWIZ  │ 25% on failure     │ │
│  └────────────────────┴──────────────┴────────────────────┘ │
│                                                              │
│  Stake Lock Period:                                          │
│  • Minimum lock: 30 days from registration                   │
│  • Extended lock: +7 days per active job                     │
│  • Withdrawal cooldown: 14 days after unlock                 │
│                                                              │
│  Slashing Conditions:                                        │
│  1. Job timeout (no progress for 2+ hours):                  │
│     → Slash 10% of stake                                     │
│                                                              │
│  2. Invalid proof-of-compute:                                │
│     → Slash 25% of stake                                     │
│     → Reputation penalty: -100 points                        │
│                                                              │
│  3. Dispute resolved against node (fraud):                   │
│     → Slash 50% of stake                                     │
│     → Permanent ban from network                             │
│                                                              │
│  4. Repeated failures (3+ jobs in 7 days):                   │
│     → Slash 30% of stake                                     │
│     → Temporary suspension (30 days)                         │
│                                                              │
│  Stake Rewards (if no slashing for 90 days):                 │
│  • Bonus APY: 5% annually                                    │
│  • Paid from platform fees pool                              │
│  • Compounds automatically                                   │
│                                                              │
│  Example Scenario:                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Node stakes: 1000 CYXWIZ                              │   │
│  │ Accepts job: 50 CYXWIZ payment                        │   │
│  │                                                       │   │
│  │ If successful:                                        │   │
│  │   → Earn 50 * 0.9 = 45 CYXWIZ (after 10% fee)        │   │
│  │   → Reputation +10                                    │   │
│  │   → Stake intact                                      │   │
│  │                                                       │   │
│  │ If failed (timeout):                                  │   │
│  │   → Slash 1000 * 0.1 = 100 CYXWIZ                     │   │
│  │   → Reputation -30                                    │   │
│  │   → User refunded 50 CYXWIZ                           │   │
│  │   → Slashed 100 goes to insurance pool               │   │
│  │                                                       │   │
│  │ Net loss for malicious behavior: -100 CYXWIZ          │   │
│  │ Net gain for honest behavior: +45 CYXWIZ              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Economic Security:**
- Cost of attack > potential gain (game-theoretically secure)
- Slashed funds go to insurance pool (protects users)
- Reputation system makes long-term honesty profitable

---

## Token Economics

### CYXWIZ Token Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    CYXWIZ TOKEN (SPL)                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Blockchain: Solana (SPL Token Standard)                      │
│  Total Supply: 1,000,000,000 CYXWIZ (fixed supply)           │
│  Decimals: 9                                                  │
│  Symbol: CYXWIZ                                               │
│                                                               │
│  Distribution:                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 30% - Community Rewards (vested over 5 years)           │ │
│  │ 25% - Node Operator Incentives (vested over 3 years)    │ │
│  │ 20% - Development Fund (team, vested over 4 years)      │ │
│  │ 15% - Ecosystem Growth (partnerships, grants)           │ │
│  │ 10% - Initial Liquidity (DEX pools)                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Utility:                                                     │
│  1. Payment for compute jobs                                  │
│  2. Node operator staking                                     │
│  3. Governance voting (DAO)                                   │
│  4. Access to premium features                                │
│  5. Reputation boosting                                       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Token Flow Diagram

```
                    CYXWIZ TOKEN ECONOMY

    ┌─────────────────────────────────────────────────────┐
    │              TOKEN CIRCULATION                       │
    └─────────────────────────────────────────────────────┘
                            │
                            │
         ┌──────────────────┴──────────────────┐
         │                                     │
         ▼                                     ▼
    ┌─────────┐                          ┌──────────┐
    │  Users  │                          │  Nodes   │
    │ (Engine)│                          │ (Miners) │
    └────┬────┘                          └────┬─────┘
         │                                     │
         │ 1. Pay for jobs                     │ 2. Earn from jobs
         │    (100 CYXWIZ)                     │    (90 CYXWIZ)
         │                                     │
         └────────────┬────────────────────────┘
                      │
                      │ 3. Platform fee (10 CYXWIZ)
                      │
                      ▼
              ┌────────────────┐
              │  Platform Pool │
              │  (Governance)  │
              └────────┬───────┘
                       │
                       │ Distribution:
                       ├─────> 40% → Staking rewards
                       ├─────> 30% → Development fund
                       ├─────> 20% → Liquidity mining
                       └─────> 10% → Burn (deflationary)

    ┌──────────────────────────────────────────────────┐
    │         STAKING AND GOVERNANCE                    │
    └──────────────────────────────────────────────────┘

    Nodes Stake ──> Lock tokens ──> Earn Rewards
         │              │               │
         │              │               └─> 5% APY + Job income
         │              │
         │              └─> Voting Power: 1 token = 1 vote
         │
         └─> Risk: Slashing for misbehavior

    Users Stake ──> Priority Matching ──> Lower fees
         │               │                    │
         │               │                    └─> 8% instead of 10%
         │               │
         │               └─> Skip queue for busy nodes
         │
         └─> Governance voting rights


    ┌──────────────────────────────────────────────────┐
    │           BURN MECHANISM (Deflationary)           │
    └──────────────────────────────────────────────────┘

    Sources of burn:
    • 10% of platform fees → burned quarterly
    • Failed job refunds (partial) → 5% burned
    • Slashed stakes → 20% burned, 80% to insurance

    Net effect:
    • Reduces circulating supply over time
    • Increases token value for long-term holders
    • Target: Reduce supply to 500M CYXWIZ over 10 years
```

### Pricing Model

```
┌────────────────────────────────────────────────────────────┐
│            JOB PRICING CALCULATION                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Base Price = Σ (resource_cost * duration)                  │
│                                                             │
│  Factors:                                                   │
│                                                             │
│  1. Compute Resource Cost:                                  │
│     • CPU-only: 0.5 CYXWIZ / hour                           │
│     • GPU (consumer, RTX 3070): 2 CYXWIZ / hour             │
│     • GPU (datacenter, A100): 10 CYXWIZ / hour              │
│                                                             │
│  2. Memory Cost:                                            │
│     • Per GB RAM: 0.1 CYXWIZ / hour                         │
│     • Per GB VRAM: 0.5 CYXWIZ / hour                        │
│                                                             │
│  3. Duration Multiplier:                                    │
│     • < 1 hour: 1.0x                                        │
│     • 1-6 hours: 0.9x (10% discount)                        │
│     • 6-24 hours: 0.8x (20% discount)                       │
│     • > 24 hours: 0.7x (30% discount)                       │
│                                                             │
│  4. Priority Multiplier:                                    │
│     • Low priority: 0.8x                                    │
│     • Normal: 1.0x                                          │
│     • High: 1.5x                                            │
│     • Critical: 2.0x                                        │
│                                                             │
│  5. Node Reputation Bonus:                                  │
│     • Bronze (0-299): +20% (higher risk)                    │
│     • Silver (300-599): +10%                                │
│     • Gold (600-799): 0%                                    │
│     • Diamond (800-1000): -10% (trusted, lower price)       │
│                                                             │
│  Example Calculation:                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Job: Train ResNet-50 on ImageNet                    │    │
│  │                                                     │    │
│  │ Resources:                                          │    │
│  │   • GPU: RTX 3070 (2 CYXWIZ/hr)                     │    │
│  │   • VRAM: 8 GB (4 CYXWIZ/hr)                        │    │
│  │   • RAM: 16 GB (1.6 CYXWIZ/hr)                      │    │
│  │                                                     │    │
│  │ Base cost: (2 + 4 + 1.6) = 7.6 CYXWIZ/hr            │    │
│  │ Duration: 12 hours                                  │    │
│  │ Duration discount: 0.8x                             │    │
│  │ Priority: Normal (1.0x)                             │    │
│  │ Node tier: Gold (0%)                                │    │
│  │                                                     │    │
│  │ Total: 7.6 * 12 * 0.8 * 1.0 = 72.96 CYXWIZ         │    │
│  │                                                     │    │
│  │ Rounded: 73 CYXWIZ                                  │    │
│  │                                                     │    │
│  │ User pays: 73 CYXWIZ (escrowed)                     │    │
│  │ Node earns: 65.7 CYXWIZ (90%)                       │    │
│  │ Platform: 7.3 CYXWIZ (10%)                          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Security Model

### Threat Model and Mitigations

```
┌──────────────────────────────────────────────────────────────┐
│                    SECURITY THREATS                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. MALICIOUS NODE (Fake Training)                            │
│     Attack: Node claims to train but doesn't, submits random  │
│            weights                                            │
│                                                               │
│     Mitigation:                                               │
│     ✓ Proof-of-compute verification (zk-SNARK)                │
│     ✓ Weights hash must match user-downloaded model           │
│     ✓ Training time must be reasonable (0.5x - 2x estimate)   │
│     ✓ Checkpoint hashes verified against Merkle tree          │
│     ✓ Stake slashing on invalid proof (25%)                   │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  2. MALICIOUS USER (Denial of Service)                        │
│     Attack: User submits many jobs, cancels them to waste     │
│            node resources                                     │
│                                                               │
│     Mitigation:                                               │
│     ✓ Escrow required before job starts (sunk cost)           │
│     ✓ Partial payment to node based on progress               │
│     ✓ Cancellation fee: 10% of escrow (non-refundable)        │
│     ✓ Rate limiting: Max 10 jobs per wallet per day           │
│     ✓ User reputation score (track cancellation rate)         │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  3. CENTRAL SERVER COMPROMISE                                 │
│     Attack: Attacker gains control of Central Server,         │
│            manipulates job assignments                        │
│                                                               │
│     Mitigation:                                               │
│     ✓ Critical state on blockchain (payments, reputation)     │
│     ✓ Central Server can't move funds (only users/contracts)  │
│     ✓ Job metadata signed by user's wallet (verifiable)       │
│     ✓ Nodes verify JWT tokens independently                   │
│     ✓ Fallback: Direct P2P discovery via DHT (future)         │
│     ✓ Multi-signature for admin operations                    │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  4. SYBIL ATTACK (Fake Nodes)                                 │
│     Attack: Attacker creates many fake nodes to game          │
│            reputation or pricing                              │
│                                                               │
│     Mitigation:                                               │
│     ✓ Minimum stake requirement per node (50-5000 CYXWIZ)     │
│     ✓ Stake lock period (30+ days)                            │
│     ✓ Reputation starts at 0 (must earn trust)                │
│     ✓ On-chain identity tied to wallet (traceable)            │
│     ✓ IP address limits: Max 1 node per IP                    │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  5. DATA POISONING (Malicious Training Data)                  │
│     Attack: User or node injects malicious data to corrupt    │
│            models or steal information                        │
│                                                               │
│     Mitigation:                                               │
│     ✓ Sandboxed execution environment (Docker)                │
│     ✓ Dataset hash verified before training                   │
│     ✓ Model weights inspected for anomalies                   │
│     ✓ User-provided datasets quarantined                      │
│     ✓ Reputation penalty for suspicious models                │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  6. MAN-IN-THE-MIDDLE (P2P Interception)                      │
│     Attack: Attacker intercepts Engine ↔ Node traffic         │
│                                                               │
│     Mitigation:                                               │
│     ✓ TLS 1.3 encryption for all P2P communication            │
│     ✓ Certificate pinning (node's public key in JWT)          │
│     ✓ Mutual authentication (Engine + Node verify tokens)     │
│     ✓ End-to-end encrypted weights transfer                   │
│     ✓ Checksum verification on every chunk                    │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  7. FRONT-RUNNING (MEV on Solana)                             │
│     Attack: Bots monitor mempool, front-run escrow txs        │
│            to steal node assignments                          │
│                                                               │
│     Mitigation:                                               │
│     ✓ Job assignments off-chain (Central Server)              │
│     ✓ Escrow creation doesn't reveal node assignment          │
│     ✓ JWT tokens are single-use, short-lived (5 min)          │
│     ✓ Node assignment committed on-chain AFTER escrow         │
│     ✓ Use Solana's Gulf Stream (no public mempool)            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Dispute Resolution Process

```
┌──────────────────────────────────────────────────────────────┐
│                DISPUTE RESOLUTION WORKFLOW                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Trigger: User or Node claims unfair treatment                │
│                                                               │
│  Step 1: Initiate Dispute (On-Chain)                          │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ User/Node submits dispute transaction:                │   │
│  │  - Dispute.create(job_id, claim, evidence_hash)       │   │
│  │  - Lock escrow funds (prevent withdrawal)             │   │
│  │  - Emit DisputeCreated event                          │   │
│  │  - State: PENDING_EVIDENCE                            │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  Step 2: Evidence Submission (48-hour window)                 │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Both parties submit evidence:                         │   │
│  │  • Training logs (hashed)                             │   │
│  │  • Checkpoint data                                    │   │
│  │  • Resource usage metrics                             │   │
│  │  • Communication transcripts                          │   │
│  │                                                       │   │
│  │ Evidence stored off-chain (IPFS), hash on-chain       │   │
│  │ State: EVIDENCE_SUBMITTED                             │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  Step 3: DAO Voting (7-day period)                            │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ CYXWIZ token holders vote:                            │   │
│  │  • Review evidence on IPFS                            │   │
│  │  • Cast vote: USER_FAVOR or NODE_FAVOR                │   │
│  │  • Voting weight: 1 CYXWIZ = 1 vote                   │   │
│  │  • Quorum: 5% of total supply                         │   │
│  │                                                       │   │
│  │ Incentive for voters:                                 │   │
│  │  • Correct vote → 0.1 CYXWIZ reward (from treasury)   │   │
│  │  • Incorrect vote → No penalty (encourage participation)│  │
│  │                                                       │   │
│  │ State: VOTING_ACTIVE                                  │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  Step 4: Verdict Execution (Automated)                        │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Smart contract executes based on vote outcome:        │   │
│  │                                                       │   │
│  │ If USER_FAVOR wins:                                   │   │
│  │   ✓ Full refund to user                               │   │
│  │   ✓ Slash node's stake (50%)                          │   │
│  │   ✓ Node reputation -200                              │   │
│  │   ✓ Slashed funds → 50% user, 50% treasury            │   │
│  │                                                       │   │
│  │ If NODE_FAVOR wins:                                   │   │
│  │   ✓ Release payment to node (90%)                     │   │
│  │   ✓ Platform fee (10%)                                │   │
│  │   ✓ Node reputation +5 (wrongful accusation cleared)  │   │
│  │   ✓ User reputation -10 (frivolous dispute)           │   │
│  │                                                       │   │
│  │ State: RESOLVED                                       │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  Step 5: Appeal (Optional, within 7 days)                     │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Losing party can appeal:                              │   │
│  │   • Requires 100 CYXWIZ deposit (forfeit if rejected) │   │
│  │   • Must provide new evidence                         │   │
│  │   • Re-vote with 2x quorum requirement                │   │
│  │   • Final decision is binding                         │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Scalability Considerations

### Gas Optimization Strategies

```
┌──────────────────────────────────────────────────────────────┐
│              GAS COST ANALYSIS & OPTIMIZATION                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Problem: Solana has low fees, but at scale they add up       │
│                                                               │
│  Typical Costs (as of 2025):                                  │
│  • Base transaction fee: ~0.000005 SOL (~$0.0005)             │
│  • Account creation: ~0.00203928 SOL (~$0.20)                 │
│  • Compute units: ~0.00001 SOL per 1M units (~$0.001)         │
│                                                               │
│  Per-Job Costs (Naive Approach):                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ 1. Create escrow: $0.20 (new PDA)                     │   │
│  │ 2. Complete payment: $0.001 (transfer tx)             │   │
│  │ 3. Update reputation: $0.001 (state update)           │   │
│  │ 4. Checkpoint updates (100x): $0.10 (100 * $0.001)    │   │
│  │                                                       │   │
│  │ Total per job: ~$0.31                                 │   │
│  │                                                       │   │
│  │ At 10,000 jobs/day: $3,100/day = $1.13M/year          │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  OPTIMIZATION 1: Batched Reputation Updates                   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Instead of: 1 tx per job                              │   │
│  │ Use: 1 tx per 100 jobs (batch update)                 │   │
│  │                                                       │   │
│  │ Central Server accumulates updates off-chain:         │   │
│  │   [(node_1, +10), (node_2, +10), (node_3, -30), ...] │   │
│  │                                                       │   │
│  │ Every hour, submit single batched transaction:        │   │
│  │   ReputationManager.batch_update(updates_array)       │   │
│  │                                                       │   │
│  │ Gas savings: 99% reduction ($0.10 → $0.001)           │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  OPTIMIZATION 2: Merkle Tree Checkpoints                      │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Instead of: Posting every checkpoint on-chain         │   │
│  │ Use: Post Merkle root every 50 epochs                 │   │
│  │                                                       │   │
│  │ Example (100-epoch job):                              │   │
│  │   • Checkpoint at epoch 1, 2, 3, ..., 100             │   │
│  │   • Build Merkle tree of all checkpoint hashes        │   │
│  │   • Post Merkle root at epoch 50 and 100 (2 txs)      │   │
│  │   • Store full tree off-chain (IPFS)                  │   │
│  │                                                       │   │
│  │ Gas savings: 98% reduction (100 txs → 2 txs)          │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  OPTIMIZATION 3: PDA Reuse                                    │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Instead of: Creating new escrow PDA for each job      │   │
│  │ Use: Reuse PDA after job completes                    │   │
│  │                                                       │   │
│  │ Smart contract design:                                │   │
│  │   • PDA seed: "escrow-{user_wallet}-{nonce}"          │   │
│  │   • Nonce increments per user                         │   │
│  │   • Reuse same PDA for subsequent jobs                │   │
│  │   • Close PDA after refund/completion (reclaim rent)  │   │
│  │                                                       │   │
│  │ Gas savings: 90% reduction on escrow creation          │   │
│  │               ($0.20 → $0.02)                         │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  OPTIMIZATION 4: Compression (State Compression API)          │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Use Solana's State Compression for large-scale data:  │   │
│  │   • Store reputation scores in compressed Merkle tree │   │
│  │   • Only store Merkle root on-chain (32 bytes)        │   │
│  │   • Full tree in off-chain indexer                    │   │
│  │                                                       │   │
│  │ Benefits:                                             │   │
│  │   • 1M nodes: $20K → $0.001 (99.999% reduction)       │   │
│  │   • Scales to billions of jobs                        │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  Optimized Per-Job Costs:                                     │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ 1. Create/reuse escrow: $0.02                         │   │
│  │ 2. Complete payment: $0.001                           │   │
│  │ 3. Batched reputation: $0.00001 (amortized)           │   │
│  │ 4. Merkle checkpoints: $0.002 (2 txs per job)         │   │
│  │                                                       │   │
│  │ Total per job: ~$0.023                                │   │
│  │                                                       │   │
│  │ At 10,000 jobs/day: $230/day = $84K/year              │   │
│  │                                                       │   │
│  │ 93% cost reduction vs. naive approach!                │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Horizontal Scaling Architecture

```
┌──────────────────────────────────────────────────────────────┐
│           MULTI-REGION DEPLOYMENT STRATEGY                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│        ┌─────────────────────────────────────┐               │
│        │    Solana Blockchain (Global)       │               │
│        │  - Single source of truth            │               │
│        │  - Escrow, payments, reputation      │               │
│        └─────────────┬───────────────────────┘               │
│                      │                                        │
│           ┌──────────┴──────────┐                             │
│           │                     │                             │
│           ▼                     ▼                             │
│  ┌─────────────────┐   ┌─────────────────┐                   │
│  │ Central Server  │   │ Central Server  │   ... (N replicas)│
│  │ (US East)       │   │ (EU West)       │                   │
│  └────────┬────────┘   └────────┬────────┘                   │
│           │                     │                             │
│           │  Shared State:      │                             │
│           │  • Redis cluster    │                             │
│           │  • PostgreSQL       │                             │
│           │    (multi-region)   │                             │
│           │                     │                             │
│           ├─────────────────────┤                             │
│           │                     │                             │
│    ┌──────▼───────┐     ┌──────▼───────┐                     │
│    │ Load Balancer│     │ Load Balancer│                     │
│    │ (GeoDNS)     │     │ (GeoDNS)     │                     │
│    └──────┬───────┘     └──────┬───────┘                     │
│           │                     │                             │
│           │ Route to nearest    │                             │
│           │ available node      │                             │
│           │                     │                             │
│    ┌──────▼──────┐       ┌─────▼──────┐                      │
│    │ Server Node │       │Server Node │                      │
│    │ Pool (1000s)│       │Pool (1000s)│                      │
│    └─────────────┘       └────────────┘                      │
│                                                               │
│  Benefits:                                                    │
│  ✓ Low latency: Users connect to nearest Central Server      │
│  ✓ High availability: 99.99% uptime (multi-region)           │
│  ✓ Load distribution: No single bottleneck                   │
│  ✓ Geo-aware matching: Nodes assigned to nearby users        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-3)

```
┌──────────────────────────────────────────────────────────────┐
│                   PHASE 1: BLOCKCHAIN FOUNDATION              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Month 1: Smart Contract Development                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: JobEscrow Program                            │  │
│  │   • Design escrow state machine                        │  │
│  │   • Implement create_escrow, complete, refund          │  │
│  │   • Write unit tests (Anchor framework)                │  │
│  │   • Deploy to Solana devnet                            │  │
│  │                                                        │  │
│  │ Week 3-4: NodeRegistry Program                         │  │
│  │   • Implement register_node, stake_tokens              │  │
│  │   • Heartbeat mechanism                                │  │
│  │   • Slash and withdraw functions                       │  │
│  │   • Integration tests with JobEscrow                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Month 2: Central Server Blockchain Integration               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1: Solana Client Wrapper                          │  │
│  │   • Replace mocked solana_client.rs with real SDK      │  │
│  │   • Implement transaction signing                      │  │
│  │   • Add retry logic and error handling                 │  │
│  │                                                        │  │
│  │ Week 2-3: Payment Processor Updates                    │  │
│  │   • Integrate with JobEscrow program                   │  │
│  │   • Escrow creation workflow                           │  │
│  │   • Payment release on completion                      │  │
│  │   • Refund logic                                       │  │
│  │                                                        │  │
│  │ Week 4: Event Listening                                │  │
│  │   • Subscribe to on-chain events                       │  │
│  │   • Update off-chain DB on EscrowCreated, etc.         │  │
│  │   • Reconciliation job (every 5 minutes)               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Month 3: Engine Wallet Integration                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: Wallet Connection UI                         │  │
│  │   • Phantom/Solflare wallet adapter (C++)              │  │
│  │   • Connect wallet button in Engine GUI                │  │
│  │   • Display CYXWIZ token balance                       │  │
│  │                                                        │  │
│  │ Week 3: Transaction Signing Flow                       │  │
│  │   • User approves escrow transaction in wallet         │  │
│  │   • Wait for confirmation                              │  │
│  │   • Send tx signature to Central Server                │  │
│  │                                                        │  │
│  │ Week 4: Testing and Refinement                         │  │
│  │   • End-to-end test: Job submission → Escrow → Payment │  │
│  │   • Handle edge cases (rejected txs, timeouts)         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Deliverables:                                                │
│  ✓ JobEscrow and NodeRegistry programs on devnet              │
│  ✓ Central Server can create/complete escrows                 │
│  ✓ Engine can connect wallet and approve transactions         │
│  ✓ Basic payment flow working end-to-end                      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 2: Proof of Compute (Months 4-6)

```
┌──────────────────────────────────────────────────────────────┐
│              PHASE 2: PROOF OF COMPUTE & REPUTATION           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Month 4: Zero-Knowledge Proof Infrastructure                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: Circuit Design                               │  │
│  │   • Define training computation circuit (circom)       │  │
│  │   • Constraints: epoch count, weights hash, metrics    │  │
│  │   • Compile to zk-SNARK (Groth16)                      │  │
│  │                                                        │  │
│  │ Week 3-4: Proof Generation (Server Node)               │  │
│  │   • Integrate zk-SNARK library (bellman or snarkjs)    │  │
│  │   • Generate proof after training completes            │  │
│  │   • Serialize proof for blockchain submission          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Month 5: Proof Verification                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: On-Chain Verifier                            │  │
│  │   • Deploy zk-SNARK verifier to Solana program         │  │
│  │   • Update JobEscrow.complete() to call verifier       │  │
│  │   • Gas optimization (verifier is compute-intensive)   │  │
│  │                                                        │  │
│  │ Week 3-4: Testing and Benchmarking                     │  │
│  │   • Test proof generation time (target: <10s overhead) │  │
│  │   • Test verification gas cost (target: <200K units)   │  │
│  │   • Security audit (informal, internal team)           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Month 6: Reputation System                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: ReputationManager Program                    │  │
│  │   • Implement score calculation algorithm              │  │
│  │   • Batched updates (100 jobs per tx)                  │  │
│  │   • Query reputation by node_id                        │  │
│  │                                                        │  │
│  │ Week 3: Central Server Integration                     │  │
│  │   • Job scheduler filters nodes by reputation          │  │
│  │   • Periodic batch updates to on-chain scores          │  │
│  │                                                        │  │
│  │ Week 4: Engine UI                                      │  │
│  │   • Display assigned node's reputation score           │  │
│  │   • Allow user to rate node after job completion       │  │
│  │   • Reputation leaderboard (top nodes)                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Deliverables:                                                │
│  ✓ Zero-knowledge proof generation and verification working   │
│  ✓ Payment only released with valid proof                     │
│  ✓ Reputation system tracking node performance                │
│  ✓ Job matching considers reputation scores                   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 3: Mainnet Launch (Months 7-9)

```
┌──────────────────────────────────────────────────────────────┐
│            PHASE 3: MAINNET DEPLOYMENT & SCALING              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Month 7: Security Audits                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: Smart Contract Audit                         │  │
│  │   • Hire external auditor (e.g., Kudelski, Trail of    │  │
│  │     Bits)                                              │  │
│  │   • Fix critical and high-severity issues              │  │
│  │   • Re-audit and publish report                        │  │
│  │                                                        │  │
│  │ Week 3-4: Infrastructure Security                      │  │
│  │   • Penetration testing (Central Server)               │  │
│  │   • DoS resilience testing                             │  │
│  │   • Cryptographic review (JWT, TLS, zk-SNARK)          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Month 8: Mainnet Deployment                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1: Token Launch                                   │  │
│  │   • Deploy CYXWIZ SPL token on Solana mainnet          │  │
│  │   • Initial liquidity: $500K on Raydium DEX            │  │
│  │   • Airdrop to early beta testers                      │  │
│  │                                                        │  │
│  │ Week 2: Smart Contracts to Mainnet                     │  │
│  │   • Deploy audited programs (JobEscrow, NodeRegistry)  │  │
│  │   • Multi-sig admin wallet (3-of-5)                    │  │
│  │   • Emergency pause functionality                      │  │
│  │                                                        │  │
│  │ Week 3-4: Phased Rollout                               │  │
│  │   • Invite 100 trusted nodes (early access)            │  │
│  │   • Process first 1000 real jobs                       │  │
│  │   • Monitor gas costs, performance, errors             │  │
│  │   • Gradual increase: 1K → 10K → 100K jobs/day         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Month 9: Optimization and Monitoring                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: Gas Optimization                             │  │
│  │   • Implement batched reputation updates               │  │
│  │   • Merkle tree checkpoints (reduce tx count)          │  │
│  │   • PDA reuse for escrow accounts                      │  │
│  │                                                        │  │
│  │ Week 3-4: Observability                                │  │
│  │   • Blockchain event indexer (e.g., The Graph)         │  │
│  │   • Grafana dashboards (txs, gas, errors)              │  │
│  │   • Alerting (Pagerduty for critical issues)           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Success Metrics:                                             │
│  ✓ 10,000+ jobs processed on mainnet                          │
│  ✓ 500+ active nodes registered                               │
│  ✓ <0.1% failure rate due to blockchain issues                │
│  ✓ Average gas cost < $0.05 per job                           │
│  ✓ 99.9% uptime for payment processing                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 4: Decentralization (Months 10-12)

```
┌──────────────────────────────────────────────────────────────┐
│          PHASE 4: DAO GOVERNANCE & FULL DECENTRALIZATION      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Month 10: DAO Foundation                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: Governance Token Distribution                │  │
│  │   • Distribute voting tokens to stakeholders:          │  │
│  │     - Node operators: 40% (weighted by stake)          │  │
│  │     - Token holders: 30%                               │  │
│  │     - Core team: 20% (vested)                          │  │
│  │     - Community: 10% (airdrops, bounties)              │  │
│  │                                                        │  │
│  │ Week 3-4: Governance Contracts                         │  │
│  │   • Deploy DAO voting contract                         │  │
│  │   • Proposal submission (min 1% of supply)             │  │
│  │   • Voting period: 7 days                              │  │
│  │   • Execution delay: 2 days (timelock)                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Month 11: Decentralized Dispute Resolution                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: Dispute Contract                             │  │
│  │   • Implement dispute creation, voting, execution      │  │
│  │   • Integrate with escrow (lock funds during dispute)  │  │
│  │   • Voter incentives (reward correct votes)            │  │
│  │                                                        │  │
│  │ Week 3-4: Testing with Real Cases                      │  │
│  │   • Simulate disputes on testnet                       │  │
│  │   • Community vote on sample cases                     │  │
│  │   • Refine voting UI and evidence presentation         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Month 12: Reduce Central Server Dependence                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Week 1-2: DHT-Based Node Discovery (Optional)          │  │
│  │   • Implement Kademlia DHT for P2P discovery           │  │
│  │   • Nodes publish capabilities to DHT                  │  │
│  │   • Engine queries DHT directly (fallback)             │  │
│  │   • Central Server still primary for performance       │  │
│  │                                                        │  │
│  │ Week 3-4: Parameter Governance                         │  │
│  │   • DAO controls key parameters:                       │  │
│  │     - Platform fee percentage (currently 10%)          │  │
│  │     - Minimum stake amounts                            │  │
│  │     - Slashing percentages                             │  │
│  │     - Proof-of-compute circuit upgrades                │  │
│  │   • First governance proposal: Adjust platform fee     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Vision (Year 2 and Beyond):                                  │
│  • Central Server becomes optional coordination layer         │
│  • Nodes can discover each other via DHT                      │
│  • All critical operations governed by DAO                    │
│  • Platform evolves through community proposals               │
│  • Multi-chain support (Polygon, Ethereum L2s)                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Trade-offs and Design Decisions

### Critical Design Choices

```
┌──────────────────────────────────────────────────────────────┐
│                BLOCKCHAIN INTEGRATION TRADE-OFFS              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. HYBRID vs FULLY ON-CHAIN ARCHITECTURE                     │
│                                                               │
│     Decision: HYBRID (On-chain for trust, off-chain for data) │
│                                                               │
│     ✓ Pros:                                                   │
│       • Low latency: P2P transfers don't wait for blockchain  │
│       • High throughput: 1000s of jobs/sec (vs 50 on-chain)   │
│       • Low cost: ~$0.02/job (vs $5+ fully on-chain)          │
│       • Flexible: Can optimize off-chain code without forks   │
│                                                               │
│     ✗ Cons:                                                   │
│       • Central Server dependency (single point of failure)   │
│       • Trust assumption: Users trust Central Server routing  │
│       • Complexity: Synchronizing on-chain and off-chain state│
│                                                               │
│     Mitigation:                                               │
│       • Critical state (payments, reputation) always on-chain │
│       • Central Server is open-source and auditable           │
│       • Fallback: DHT-based discovery if Central Server fails │
│       • Progressive decentralization over time                │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  2. SOLANA vs ETHEREUM (and L2s)                              │
│                                                               │
│     Decision: SOLANA (Primary), with Polygon bridge (Later)   │
│                                                               │
│     ✓ Pros:                                                   │
│       • Low fees: $0.0005 vs $5-50 on Ethereum                │
│       • High speed: 400ms finality vs 12s on Ethereum         │
│       • Scalable: 50K TPS vs 15 TPS on Ethereum               │
│       • Modern: Purpose-built for DeFi, not retrofitted       │
│                                                               │
│     ✗ Cons:                                                   │
│       • Smaller ecosystem: Fewer users, tools, auditors       │
│       • Network outages: History of downtime (improving)      │
│       • Rust learning curve: Fewer Solana developers          │
│       • Liquidity fragmentation: Cross-chain complexity       │
│                                                               │
│     Mitigation:                                               │
│       • Wormhole bridge to Ethereum for cross-chain liquidity │
│       • Multi-chain roadmap (Polygon in Phase 5)              │
│       • Failover: Cache critical state, resume after outage   │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  3. ZK-SNARK vs OPTIMISTIC VERIFICATION                       │
│                                                               │
│     Decision: ZK-SNARK (Groth16) for Proof of Compute         │
│                                                               │
│     ✓ Pros:                                                   │
│       • Trustless: Cryptographically guaranteed correctness   │
│       • Instant verification: <100ms on-chain                 │
│       • Privacy: Doesn't reveal training data or weights      │
│       • Compact: Constant-size proof (~200 bytes)             │
│                                                               │
│     ✗ Cons:                                                   │
│       • Setup complexity: Trusted setup ceremony required     │
│       • Proof generation overhead: +5-10% training time       │
│       • Circuit design: Requires zk expert knowledge          │
│       • Limited expressiveness: Can't prove arbitrary code    │
│                                                               │
│     Alternative (Optimistic Verification):                    │
│       • Assume node is honest, challenge if suspicious        │
│       • 7-day challenge period before payment release         │
│       • Cheaper (no proof generation), but slower payouts     │
│       • More vulnerable to griefing attacks                   │
│                                                               │
│     Justification:                                            │
│       • ML training is deterministic (good fit for zk)        │
│       • Instant payment benefits outweigh proof overhead      │
│       • Security is paramount for financial transactions      │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  4. FIXED PLATFORM FEE vs DYNAMIC PRICING                     │
│                                                               │
│     Decision: FIXED 10% (Initially), DAO-governed (Future)    │
│                                                               │
│     ✓ Pros:                                                   │
│       • Simple: Easy to understand and calculate              │
│       • Predictable: Users know exact costs upfront           │
│       • Fair: Same percentage for all job sizes               │
│                                                               │
│     ✗ Cons:                                                   │
│       • Inflexible: Can't adjust for market conditions        │
│       • Competitive pressure: Other platforms may undercut    │
│       • Size penalty: Small jobs pay proportionally more      │
│                                                               │
│     Future Evolution:                                         │
│       • DAO can vote to adjust fee (e.g., 8% for large jobs)  │
│       • Tiered pricing based on user/node reputation          │
│       • Promotional periods (5% fee to bootstrap network)     │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  5. IMMEDIATE PAYMENT vs VESTING SCHEDULE                     │
│                                                               │
│     Decision: IMMEDIATE (90% to node, 10% to platform)        │
│                                                               │
│     ✓ Pros:                                                   │
│       • Attracts nodes: Fast payouts are competitive edge     │
│       • Liquidity: Nodes can reinvest earnings immediately    │
│       • Simplicity: No complex vesting contracts              │
│                                                               │
│     ✗ Cons:                                                   │
│       • Dispute risk: Hard to claw back if fraud detected     │
│       • Sybil attack: Malicious node could cash out quickly   │
│                                                               │
│     Mitigation:                                               │
│       • Reputation system makes long-term honesty profitable  │
│       • Slashing stake punishes fraud (stake > job payment)   │
│       • Dispute resolution can slash stake if fraud proven    │
│       • Insurance pool covers user losses                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Security vs Decentralization vs Scalability

**The Blockchain Trilemma Applied to CyxWiz:**

```
               SECURITY
                   △
                  / \
                 /   \
                /     \
               /       \
              /  HYBRID \
             /  APPROACH \
            /             \
           /               \
          /                 \
         /                   \
  DECENTRALIZATION ◄──────────► SCALABILITY
```

**Our Position:**
- **High Security**: On-chain escrow, zk-SNARK proofs, stake slashing
- **Medium Decentralization**: Central Server for coordination, but non-custodial
- **High Scalability**: P2P data transfer, batched on-chain updates

**Long-term Goal:** Move toward higher decentralization (DHT discovery, multi-region Central Servers) while maintaining security and scalability.

---

## Conclusion

This blockchain-integrated P2P workflow architecture represents a **pragmatic balance** between decentralization ideals and real-world performance requirements. Key takeaways:

1. **Hybrid is the Future**: Fully on-chain ML training is impractical due to cost and latency. The future is hybrid architectures that use blockchain for trust anchors while keeping high-throughput operations off-chain.

2. **Proof of Compute is Essential**: Zero-knowledge proofs enable trustless verification without re-running computations. This is the killer feature that makes decentralized ML possible.

3. **Economic Incentives Drive Security**: Stake-based security and reputation systems create game-theoretic guarantees that honest behavior is profitable.

4. **Scalability Through Optimization**: Gas costs can be reduced by 90%+ through batching, Merkle trees, and state compression. These optimizations are crucial for viability at scale.

5. **Progressive Decentralization**: Start with a centralized coordinator for UX and performance, then gradually decentralize as the network matures and tooling improves.

**Next Steps:**
1. Review and approve this architecture with the core team
2. Begin Phase 1 implementation (Smart Contract Development)
3. Set up development environment for Solana and Anchor framework
4. Recruit blockchain developers and security auditors
5. Create detailed technical specifications for each smart contract

---

**Document Version:** 1.0
**Last Updated:** 2025-11-23
**Author:** CyxWiz Architecture Team
**Status:** Proposal - Pending Review
**Related Documents:**
- `P2P_WORKFLOW_DESIGN.md` - Base P2P architecture (non-blockchain)
- `CLAUDE.md` - Project technical guidelines
- `README.md` - Project overview and setup
