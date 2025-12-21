use anyhow::{anyhow, Result};
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    transaction::Transaction,
};
use std::str::FromStr;
use std::sync::Arc;
use tracing::{debug, info, warn, error};

use super::{BlockchainConfig, EscrowManager, ReputationManager};

/// Main blockchain client for interacting with Solana
pub struct BlockchainClient {
    rpc_client: Arc<RpcClient>,
    authority: Arc<Keypair>,
    config: BlockchainConfig,
    escrow_manager: EscrowManager,
    reputation_manager: ReputationManager,
}

impl BlockchainClient {
    /// Create a new blockchain client
    pub fn new(config: BlockchainConfig) -> Result<Self> {
        // Load wallet keypair
        let wallet_path = shellexpand::tilde(&config.wallet_path).to_string();
        let authority = Self::load_keypair(&wallet_path)?;

        // Create RPC client
        let rpc_client = RpcClient::new_with_commitment(
            config.rpc_url.clone(),
            CommitmentConfig::confirmed(),
        );

        // Parse program IDs
        let job_escrow_program = Pubkey::from_str(&config.job_escrow_program_id)?;
        let node_registry_program = Pubkey::from_str(&config.node_registry_program_id)?;
        let token_mint = Pubkey::from_str(&config.cyxwiz_token_mint)?;

        let rpc_client = Arc::new(rpc_client);
        let authority = Arc::new(authority);

        let escrow_manager = EscrowManager::new(
            Arc::clone(&rpc_client),
            Arc::clone(&authority),
            job_escrow_program,
            token_mint,
            config.platform_fee_percentage,
        );

        let reputation_manager = ReputationManager::new(
            Arc::clone(&rpc_client),
            Arc::clone(&authority),
            node_registry_program,
        );

        info!(
            "Blockchain client initialized with authority: {}",
            authority.pubkey()
        );

        Ok(Self {
            rpc_client,
            authority,
            config,
            escrow_manager,
            reputation_manager,
        })
    }

    /// Load keypair from file
    fn load_keypair(path: &str) -> Result<Keypair> {
        let file_content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read wallet file at {}: {}", path, e))?;

        // Parse JSON array of bytes
        let bytes: Vec<u8> = serde_json::from_str(&file_content)
            .map_err(|e| anyhow!("Failed to parse wallet JSON: {}", e))?;

        Keypair::from_bytes(&bytes)
            .map_err(|e| anyhow!("Failed to create keypair from bytes: {}", e))
    }

    /// Get the authority public key
    pub fn authority_pubkey(&self) -> Pubkey {
        self.authority.pubkey()
    }

    /// Check connection to Solana cluster
    pub async fn check_connection(&self) -> Result<()> {
        match self.rpc_client.get_version() {
            Ok(version) => {
                info!("Connected to Solana cluster version: {:?}", version);
                Ok(())
            }
            Err(e) => {
                error!("Failed to connect to Solana cluster: {}", e);
                Err(anyhow!("Solana connection failed: {}", e))
            }
        }
    }

    /// Get authority SOL balance
    pub fn get_balance(&self) -> Result<u64> {
        let balance = self.rpc_client.get_balance(&self.authority.pubkey())?;
        Ok(balance)
    }

    /// Access escrow manager
    pub fn escrow(&self) -> &EscrowManager {
        &self.escrow_manager
    }

    /// Access reputation manager
    pub fn reputation(&self) -> &ReputationManager {
        &self.reputation_manager
    }

    /// Get blockchain configuration
    pub fn config(&self) -> &BlockchainConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blockchain_config_default() {
        let config = BlockchainConfig::default();
        assert_eq!(config.platform_fee_percentage, 10);
        assert_eq!(config.min_node_stake, 100_000_000);
    }
}
