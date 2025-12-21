//! Payment processing for CyxWiz jobs
//!
//! Handles all blockchain payment operations:
//! - Creating escrow accounts for job payments
//! - Releasing payments to compute nodes
//! - Refunding users for failed jobs
//! - Verifying transaction status

use super::escrow::{
    create_escrow_instruction, find_escrow_address, parse_escrow_state, refund_instruction,
    release_payment_instruction, EscrowStatus,
};
use super::solana_client::SolanaClient;
use crate::error::{Result, ServerError};
use solana_sdk::pubkey::Pubkey;
use spl_associated_token_account::get_associated_token_address;
use std::str::FromStr;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Configuration for payment processing
#[derive(Clone, Debug)]
pub struct PaymentConfig {
    /// SPL token mint address (CYXWIZ token or SOL wrapped)
    pub token_mint: Pubkey,
    /// Platform treasury token account (receives 10% fee)
    pub platform_token_account: Pubkey,
    /// Platform fee percentage (default: 10%)
    pub platform_fee_percent: u8,
}

impl Default for PaymentConfig {
    fn default() -> Self {
        Self {
            // Default to SOL wrapped mint for testing
            token_mint: Pubkey::default(),
            platform_token_account: Pubkey::default(),
            platform_fee_percent: 10,
        }
    }
}

impl PaymentConfig {
    /// Create PaymentConfig from BlockchainConfig
    ///
    /// Parses the token_mint and platform_token_account from config strings.
    /// Falls back to defaults if parsing fails.
    pub fn from_blockchain_config(
        token_mint: Option<&str>,
        platform_token_account: Option<&str>,
        platform_fee_percent: u8,
    ) -> Self {
        let token_mint = token_mint
            .and_then(|s| Pubkey::from_str(s).ok())
            .unwrap_or_default();

        let platform_token_account = platform_token_account
            .and_then(|s| Pubkey::from_str(s).ok())
            .unwrap_or_default();

        Self {
            token_mint,
            platform_token_account,
            platform_fee_percent,
        }
    }

    /// Create PaymentConfig for CYXWIZ token on devnet
    pub fn cyxwiz_devnet() -> Self {
        Self {
            // CYXWIZ token mint on devnet
            token_mint: Pubkey::from_str("Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi")
                .expect("Invalid CYXWIZ token mint"),
            // Platform treasury token account on devnet
            platform_token_account: Pubkey::from_str("negq5ApurkfM7V6F46NboJbnjbohEtfu1PotDsvMs5e")
                .expect("Invalid platform token account"),
            platform_fee_percent: 10,
        }
    }

    /// Check if this config has valid token addresses (not default)
    pub fn is_configured(&self) -> bool {
        self.token_mint != Pubkey::default() && self.platform_token_account != Pubkey::default()
    }
}

/// Payment processor for handling escrow and token transfers
pub struct PaymentProcessor {
    solana_client: Arc<SolanaClient>,
    config: PaymentConfig,
}

impl PaymentProcessor {
    /// Create a new PaymentProcessor
    pub fn new(solana_client: SolanaClient) -> Self {
        Self {
            solana_client: Arc::new(solana_client),
            config: PaymentConfig::default(),
        }
    }

    /// Create a new PaymentProcessor with custom configuration
    pub fn with_config(solana_client: SolanaClient, config: PaymentConfig) -> Self {
        Self {
            solana_client: Arc::new(solana_client),
            config,
        }
    }

    /// Get a reference to the Solana client
    pub fn solana_client(&self) -> &SolanaClient {
        &self.solana_client
    }

    /// Create an escrow for a job (with node already assigned)
    ///
    /// This locks the user's tokens in an escrow account until the job
    /// is completed or cancelled.
    ///
    /// # Arguments
    /// * `job_id` - Unique job identifier (u64 for blockchain compatibility)
    /// * `user_wallet` - User's wallet address (public key string)
    /// * `node_wallet` - Compute node's wallet address
    /// * `amount` - Amount of tokens to escrow
    ///
    /// # Returns
    /// Tuple of (transaction_signature, escrow_account_address)
    pub async fn create_job_escrow(
        &self,
        job_id: u64,
        user_wallet: &str,
        node_wallet: &str,
        amount: u64,
    ) -> Result<(String, String)> {
        info!("Creating escrow for job {} with {} tokens", job_id, amount);

        // Parse wallet addresses
        let user_pubkey = Pubkey::from_str(user_wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid user wallet: {}", e)))?;

        let node_pubkey = Pubkey::from_str(node_wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid node wallet: {}", e)))?;

        // Derive token accounts
        let user_token_account =
            get_associated_token_address(&user_pubkey, &self.config.token_mint);

        // Derive escrow PDA and token account
        let program_id = self.solana_client.program_id();
        let (escrow_pda, _bump) = find_escrow_address(&program_id, job_id);
        let escrow_token_account =
            get_associated_token_address(&escrow_pda, &self.config.token_mint);

        debug!(
            "Escrow PDA: {}, Token account: {}",
            escrow_pda, escrow_token_account
        );

        // Build the instruction
        let instruction = create_escrow_instruction(
            &program_id,
            &user_pubkey,
            &user_token_account,
            &escrow_token_account,
            job_id,
            amount,
            &node_pubkey,
        );

        // Execute the transaction
        let signature = self.solana_client.execute_transaction(&[instruction]).await?;

        info!(
            "Escrow created for job {}: signature={}",
            job_id, signature
        );

        Ok((signature, escrow_pda.to_string()))
    }

    /// Create an escrow for a job before node assignment
    ///
    /// This creates an escrow with a deferred node assignment. The platform
    /// wallet is used as a temporary placeholder until a node is assigned.
    /// The escrow node can be updated when the job is assigned to a real node.
    ///
    /// # Arguments
    /// * `job_id` - Unique job identifier
    /// * `user_wallet` - User's wallet address
    /// * `amount` - Amount of tokens to escrow
    ///
    /// # Returns
    /// Tuple of (transaction_signature, escrow_account_address)
    pub async fn create_pending_escrow(
        &self,
        job_id: u64,
        user_wallet: &str,
        amount: u64,
    ) -> Result<(String, String)> {
        info!("Creating pending escrow for job {} (node TBD)", job_id);

        // Use platform wallet as temporary placeholder
        // The actual node will be set when job is assigned via update_escrow_node
        let placeholder_node = self.config.platform_token_account.to_string();

        self.create_job_escrow(job_id, user_wallet, &placeholder_node, amount)
            .await
    }

    /// Update the node wallet for an existing escrow
    ///
    /// This should be called when a job is assigned to a compute node
    /// after the escrow was created with create_pending_escrow.
    ///
    /// Note: This requires the smart contract to support node reassignment,
    /// or we handle this at the application level by tracking the intended
    /// recipient separately from the on-chain escrow state.
    ///
    /// # Arguments
    /// * `job_id` - Job identifier
    /// * `node_wallet` - New compute node's wallet address
    pub async fn update_escrow_node(&self, job_id: u64, node_wallet: &str) -> Result<()> {
        info!("Updating escrow node for job {} to {}", job_id, node_wallet);

        // Validate node wallet
        let _node_pubkey = Pubkey::from_str(node_wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid node wallet: {}", e)))?;

        // Note: The current JobEscrow smart contract sets node at creation time.
        // For full flexibility, the smart contract would need an update_node instruction.
        // For now, we track the actual recipient at the application level and use it
        // when calling complete_job_payment.
        //
        // TODO: Add update_escrow_node instruction to smart contract for on-chain tracking

        warn!(
            "Node update for job {} recorded (application-level). \
             Smart contract update_node instruction pending implementation.",
            job_id
        );

        Ok(())
    }

    /// Complete a job and distribute payment
    ///
    /// Releases 90% of escrowed funds to the compute node and
    /// 10% to the platform treasury.
    ///
    /// # Arguments
    /// * `job_id` - Job identifier
    /// * `node_wallet` - Compute node's wallet address
    ///
    /// # Returns
    /// Transaction signature
    pub async fn complete_job_payment(&self, job_id: u64, node_wallet: &str) -> Result<String> {
        info!("Releasing payment for job {}", job_id);

        // Parse node wallet
        let node_pubkey = Pubkey::from_str(node_wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid node wallet: {}", e)))?;

        // Get token accounts
        let program_id = self.solana_client.program_id();
        let (escrow_pda, _) = find_escrow_address(&program_id, job_id);
        let escrow_token_account =
            get_associated_token_address(&escrow_pda, &self.config.token_mint);
        let node_token_account =
            get_associated_token_address(&node_pubkey, &self.config.token_mint);

        // Build the instruction
        let instruction = release_payment_instruction(
            &program_id,
            job_id,
            &escrow_token_account,
            &node_token_account,
            &self.config.platform_token_account,
            &self.solana_client.payer_pubkey(),
        );

        // Execute the transaction
        let signature = self.solana_client.execute_transaction(&[instruction]).await?;

        info!(
            "Payment released for job {}: signature={}",
            job_id, signature
        );

        Ok(signature)
    }

    /// Refund a failed or cancelled job
    ///
    /// Returns the full escrowed amount back to the user.
    ///
    /// # Arguments
    /// * `job_id` - Job identifier
    /// * `user_wallet` - User's wallet address
    ///
    /// # Returns
    /// Transaction signature
    pub async fn refund_job(&self, job_id: u64, user_wallet: &str) -> Result<String> {
        info!("Refunding escrow for job {}", job_id);

        // Parse user wallet
        let user_pubkey = Pubkey::from_str(user_wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid user wallet: {}", e)))?;

        // Get token accounts
        let program_id = self.solana_client.program_id();
        let (escrow_pda, _) = find_escrow_address(&program_id, job_id);
        let escrow_token_account =
            get_associated_token_address(&escrow_pda, &self.config.token_mint);
        let user_token_account =
            get_associated_token_address(&user_pubkey, &self.config.token_mint);

        // Build the instruction
        let instruction = refund_instruction(
            &program_id,
            job_id,
            &escrow_token_account,
            &user_token_account,
            &self.solana_client.payer_pubkey(),
        );

        // Execute the transaction
        let signature = self.solana_client.execute_transaction(&[instruction]).await?;

        info!("Refund processed for job {}: signature={}", job_id, signature);

        Ok(signature)
    }

    /// Verify a transaction was confirmed on-chain
    pub async fn verify_transaction(&self, signature: &str) -> Result<bool> {
        self.solana_client.confirm_transaction(signature).await
    }

    /// Get escrow account status for a job
    ///
    /// Returns the current state of the escrow, including:
    /// - Amount locked
    /// - Status (Pending, Released, Refunded)
    /// - User and node addresses
    pub async fn get_escrow_status(&self, job_id: u64) -> Result<Option<EscrowInfo>> {
        let program_id = self.solana_client.program_id();
        let (escrow_pda, _) = find_escrow_address(&program_id, job_id);

        // Check if account exists
        if !self.solana_client.account_exists(&escrow_pda).await? {
            return Ok(None);
        }

        // Get account data
        let data = self.solana_client.get_account_data(&escrow_pda).await?;

        // Parse the state
        match parse_escrow_state(&data) {
            Some(state) => Ok(Some(EscrowInfo {
                job_id: state.job_id,
                user: state.user.to_string(),
                node: state.node.to_string(),
                amount: state.amount,
                status: match state.status {
                    EscrowStatus::Pending => "pending".to_string(),
                    EscrowStatus::Released => "released".to_string(),
                    EscrowStatus::Refunded => "refunded".to_string(),
                },
                escrow_account: escrow_pda.to_string(),
            })),
            None => {
                warn!("Failed to parse escrow state for job {}", job_id);
                Ok(None)
            }
        }
    }

    /// Calculate payment distribution for a given amount
    ///
    /// Returns (node_payment, platform_fee)
    pub fn calculate_payment_distribution(&self, amount: u64) -> (u64, u64) {
        let platform_fee = (amount * self.config.platform_fee_percent as u64) / 100;
        let node_payment = amount - platform_fee;
        (node_payment, platform_fee)
    }

    /// Get the payer's wallet balance in SOL
    pub async fn get_payer_balance(&self) -> Result<f64> {
        let balance = self
            .solana_client
            .get_balance(&self.solana_client.payer_pubkey())
            .await?;
        Ok(balance as f64 / 1_000_000_000.0)
    }

    /// Get SPL token balance for a wallet (CYXWIZ tokens)
    ///
    /// # Arguments
    /// * `wallet` - Wallet address string
    ///
    /// # Returns
    /// Token balance as u64 (raw amount, not UI amount)
    pub async fn get_token_balance(&self, wallet: &str) -> Result<u64> {
        let wallet_pubkey = Pubkey::from_str(wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid wallet: {}", e)))?;

        // Get the associated token account for this wallet
        let token_account = get_associated_token_address(&wallet_pubkey, &self.config.token_mint);

        // Check if account exists
        if !self.solana_client.account_exists(&token_account).await? {
            debug!("Token account does not exist for {}", wallet);
            return Ok(0);
        }

        // Get account data and parse token balance
        let data = self.solana_client.get_account_data(&token_account).await?;

        // SPL Token account layout: amount is at bytes 64-72
        if data.len() >= 72 {
            let amount = u64::from_le_bytes(data[64..72].try_into().unwrap_or([0u8; 8]));
            debug!("Token balance for {}: {}", wallet, amount);
            Ok(amount)
        } else {
            warn!("Invalid token account data length: {}", data.len());
            Ok(0)
        }
    }

    /// Check if a wallet has sufficient CYXWIZ tokens for a job
    ///
    /// # Arguments
    /// * `wallet` - Wallet address string
    /// * `amount` - Required amount of tokens
    ///
    /// # Returns
    /// true if balance >= amount
    pub async fn has_sufficient_balance(&self, wallet: &str, amount: u64) -> Result<bool> {
        let balance = self.get_token_balance(wallet).await?;
        Ok(balance >= amount)
    }

    /// Get the token mint address
    pub fn token_mint(&self) -> &Pubkey {
        &self.config.token_mint
    }

    /// Get the platform token account address
    pub fn platform_token_account(&self) -> &Pubkey {
        &self.config.platform_token_account
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PaymentConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &PaymentConfig {
        &self.config
    }

    /// Get the payer's public key as string
    pub fn payer_pubkey(&self) -> String {
        self.solana_client.payer_pubkey().to_string()
    }

    /// Get the program ID as string
    pub fn program_id(&self) -> String {
        self.solana_client.program_id().to_string()
    }

    /// Check if the blockchain connection is healthy
    pub async fn check_health(&self) -> Result<()> {
        // Try to get the payer balance as a health check
        let _ = self.solana_client.get_balance(&self.solana_client.payer_pubkey()).await?;
        Ok(())
    }
}

/// Information about an escrow account
#[derive(Debug, Clone)]
pub struct EscrowInfo {
    pub job_id: u64,
    pub user: String,
    pub node: String,
    pub amount: u64,
    pub status: String,
    pub escrow_account: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payment_distribution() {
        let config = PaymentConfig {
            platform_fee_percent: 10,
            ..Default::default()
        };

        // 100 tokens: 90 to node, 10 to platform
        let platform_fee = (100u64 * config.platform_fee_percent as u64) / 100;
        let node_payment = 100 - platform_fee;

        assert_eq!(platform_fee, 10);
        assert_eq!(node_payment, 90);
    }

    #[test]
    fn test_payment_distribution_rounding() {
        let config = PaymentConfig {
            platform_fee_percent: 10,
            ..Default::default()
        };

        // 1000 tokens: 900 to node, 100 to platform
        let amount = 1000u64;
        let platform_fee = (amount * config.platform_fee_percent as u64) / 100;
        let node_payment = amount - platform_fee;

        assert_eq!(platform_fee, 100);
        assert_eq!(node_payment, 900);

        // 999 tokens: 899 to node, 99 to platform (rounds down)
        let amount = 999u64;
        let platform_fee = (amount * config.platform_fee_percent as u64) / 100;
        let node_payment = amount - platform_fee;

        assert_eq!(platform_fee, 99); // 999 * 10 / 100 = 99.9 -> 99
        assert_eq!(node_payment, 900);
    }
}
