//! Blockchain module for CyxWiz Central Server
//!
//! Provides integration with Solana blockchain for:
//! - Payment escrow management
//! - Token transfers
//! - Transaction verification

pub mod escrow;
pub mod payment_processor;
pub mod solana_client;

// Re-export main types
pub use payment_processor::PaymentProcessor;
pub use solana_client::SolanaClient;

// Re-export Solana SDK types for convenience
pub use solana_sdk::pubkey::Pubkey;
pub use solana_sdk::signature::{Keypair, Signature};
pub use solana_sdk::instruction::{AccountMeta, Instruction};
pub use solana_sdk::transaction::Transaction;
