use super::solana_client::{
    SolanaClient, Pubkey, Signature, Instruction, AccountMeta, Transaction, pubkey, signature, system_program
};
use crate::database::models::{Payment, PaymentStatus};
use crate::error::{Result, ServerError};
use uuid::Uuid;

pub struct PaymentProcessor {
    solana_client: SolanaClient,
    platform_fee_percent: u8, // Default: 10%
}

impl PaymentProcessor {
    pub fn new(solana_client: SolanaClient) -> Self {
        Self {
            solana_client,
            platform_fee_percent: 10,
        }
    }

    /// Create an escrow for a job
    /// This locks the user's tokens in an escrow account
    pub async fn create_job_escrow(
        &self,
        job_id: Uuid,
        user_wallet: &str,
        amount: u64,
    ) -> Result<(String, String)> {
        // Parse user wallet address
        let user_pubkey = pubkey::from_str(user_wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid wallet address: {}", e)))?;

        // Derive escrow account address (PDA - Program Derived Address)
        let escrow_seed = format!("escrow-{}", job_id);
        let (escrow_account, _bump) = pubkey::find_program_address(
            &[escrow_seed.as_bytes()],
            &self.solana_client.program_id(),
        );

        // Build instruction to create escrow
        let instruction_data = self.build_create_escrow_instruction(job_id, amount)?;

        let instruction = Instruction {
            program_id: self.solana_client.program_id(),
            accounts: vec![
                AccountMeta::new(user_pubkey, true),  // User (signer, writable)
                AccountMeta::new(escrow_account.clone(), false), // Escrow account (writable)
                AccountMeta::new_readonly(system_program::id(), false),
            ],
            data: instruction_data,
        };

        // Create and send transaction
        let recent_blockhash = self.solana_client.get_latest_blockhash().await?;

        let transaction = Transaction::new_signed_with_payer(
            &[instruction],
            Some(&self.solana_client.payer_pubkey()),
            &[],
            recent_blockhash,
        );

        let signature = self.solana_client.send_transaction(transaction).await?;

        Ok((signature.to_string(), escrow_account.to_string()))
    }

    /// Complete a job and distribute payment
    /// 90% to node, 10% to platform
    pub async fn complete_job_payment(
        &self,
        job_id: Uuid,
        node_wallet: &str,
        amount: u64,
    ) -> Result<String> {
        // Calculate distribution
        let platform_fee = (amount * self.platform_fee_percent as u64) / 100;
        let node_reward = amount - platform_fee;

        // Parse node wallet
        let node_pubkey = pubkey::from_str(node_wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid node wallet: {}", e)))?;

        // Derive escrow account
        let escrow_seed = format!("escrow-{}", job_id);
        let (escrow_account, _bump) = pubkey::find_program_address(
            &[escrow_seed.as_bytes()],
            &self.solana_client.program_id(),
        );

        // Build instruction to complete payment
        let instruction_data = self.build_complete_payment_instruction(job_id, node_reward, platform_fee)?;

        let instruction = Instruction {
            program_id: self.solana_client.program_id(),
            accounts: vec![
                AccountMeta::new(escrow_account.clone(), false),           // Escrow account (writable)
                AccountMeta::new(node_pubkey, false),              // Node wallet (writable)
                AccountMeta::new(self.solana_client.payer_pubkey(), false), // Platform wallet (writable)
                AccountMeta::new_readonly(system_program::id(), false),
            ],
            data: instruction_data,
        };

        let recent_blockhash = self.solana_client.get_latest_blockhash().await?;

        let transaction = Transaction::new_signed_with_payer(
            &[instruction],
            Some(&self.solana_client.payer_pubkey()),
            &[],
            recent_blockhash,
        );

        let signature = self.solana_client.send_transaction(transaction).await?;

        Ok(signature.to_string())
    }

    /// Refund a failed job
    pub async fn refund_job(
        &self,
        job_id: Uuid,
        user_wallet: &str,
        amount: u64,
    ) -> Result<String> {
        let user_pubkey = pubkey::from_str(user_wallet)
            .map_err(|e| ServerError::Blockchain(format!("Invalid wallet address: {}", e)))?;

        let escrow_seed = format!("escrow-{}", job_id);
        let (escrow_account, _bump) = pubkey::find_program_address(
            &[escrow_seed.as_bytes()],
            &self.solana_client.program_id(),
        );

        // Build refund instruction
        let instruction_data = self.build_refund_instruction(job_id, amount)?;

        let instruction = Instruction {
            program_id: self.solana_client.program_id(),
            accounts: vec![
                AccountMeta::new(escrow_account.clone(), false),    // Escrow account (writable)
                AccountMeta::new(user_pubkey, false),       // User wallet (writable)
                AccountMeta::new_readonly(system_program::id(), false),
            ],
            data: instruction_data,
        };

        let recent_blockhash = self.solana_client.get_latest_blockhash().await?;

        let transaction = Transaction::new_signed_with_payer(
            &[instruction],
            Some(&self.solana_client.payer_pubkey()),
            &[],
            recent_blockhash,
        );

        let signature = self.solana_client.send_transaction(transaction).await?;

        Ok(signature.to_string())
    }

    /// Verify payment transaction on-chain
    pub async fn verify_transaction(&self, signature: &str) -> Result<bool> {
        let sig = signature::from_str(signature)
            .map_err(|e| ServerError::Blockchain(format!("Invalid signature: {}", e)))?;

        self.solana_client.confirm_transaction(&sig).await
    }

    // Helper methods to build instruction data
    // In a real implementation, these would match the smart contract's instruction format

    fn build_create_escrow_instruction(&self, job_id: Uuid, amount: u64) -> Result<Vec<u8>> {
        // Instruction format: [0] + job_id (16 bytes) + amount (8 bytes)
        let mut data = vec![0u8]; // Instruction discriminator for "CreateEscrow"
        data.extend_from_slice(job_id.as_bytes());
        data.extend_from_slice(&amount.to_le_bytes());
        Ok(data)
    }

    fn build_complete_payment_instruction(&self, job_id: Uuid, node_reward: u64, platform_fee: u64) -> Result<Vec<u8>> {
        // Instruction format: [1] + job_id + node_reward + platform_fee
        let mut data = vec![1u8]; // Instruction discriminator for "CompletePayment"
        data.extend_from_slice(job_id.as_bytes());
        data.extend_from_slice(&node_reward.to_le_bytes());
        data.extend_from_slice(&platform_fee.to_le_bytes());
        Ok(data)
    }

    fn build_refund_instruction(&self, job_id: Uuid, amount: u64) -> Result<Vec<u8>> {
        // Instruction format: [2] + job_id + amount
        let mut data = vec![2u8]; // Instruction discriminator for "Refund"
        data.extend_from_slice(job_id.as_bytes());
        data.extend_from_slice(&amount.to_le_bytes());
        Ok(data)
    }

    pub fn calculate_payment_distribution(&self, amount: u64) -> (u64, u64) {
        let platform_fee = (amount * self.platform_fee_percent as u64) / 100;
        let node_reward = amount - platform_fee;
        (node_reward, platform_fee)
    }
}
