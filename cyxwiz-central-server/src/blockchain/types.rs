use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;

/// Result of creating an escrow account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscrowResult {
    pub escrow_account: Pubkey,
    pub tx_signature: String,
    pub job_id: u64,
    pub amount: u64,
    pub user: Pubkey,
    pub node: Pubkey,
}

/// Result of releasing payment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentResult {
    pub tx_signature: String,
    pub node_payment: u64,
    pub platform_fee: u64,
}

/// Node reputation data from blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeReputation {
    pub node_id: String,
    pub total_jobs: u64,
    pub successful_jobs: u64,
    pub failed_jobs: u64,
    pub success_rate: u8,
    pub avg_execution_time_ms: u64,
    pub slashes: u32,
}

/// Blockchain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainConfig {
    pub rpc_url: String,
    pub wallet_path: String,
    pub job_escrow_program_id: String,
    pub node_registry_program_id: String,
    pub cyxwiz_token_mint: String,
    pub platform_fee_percentage: u8,
    pub min_node_stake: u64,
}

impl Default for BlockchainConfig {
    fn default() -> Self {
        Self {
            rpc_url: "https://api.devnet.solana.com".to_string(),
            wallet_path: "~/.config/solana/id.json".to_string(),
            job_escrow_program_id: "JobEscrow111111111111111111111111111111111".to_string(),
            node_registry_program_id: "NodeRegistry11111111111111111111111111111".to_string(),
            cyxwiz_token_mint: "".to_string(),
            platform_fee_percentage: 10,
            min_node_stake: 100_000_000, // 100 CYXWIZ
        }
    }
}
