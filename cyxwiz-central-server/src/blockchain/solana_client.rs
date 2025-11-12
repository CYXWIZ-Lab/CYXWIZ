// Temporarily mocked for SQLite quick test without Solana dependencies
use crate::error::{Result, ServerError};

// Mock types for Solana SDK
pub type Keypair = Vec<u8>;
pub type Pubkey = String;
pub type Signature = String;
pub type Hash = String;

// Mock instruction types
#[derive(Clone, Debug)]
pub struct Instruction {
    pub program_id: Pubkey,
    pub accounts: Vec<AccountMeta>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct AccountMeta {
    pub pubkey: Pubkey,
    pub is_signer: bool,
    pub is_writable: bool,
}

impl AccountMeta {
    pub fn new(pubkey: Pubkey, is_signer: bool) -> Self {
        Self { pubkey, is_signer, is_writable: true }
    }

    pub fn new_readonly(pubkey: Pubkey, is_signer: bool) -> Self {
        Self { pubkey, is_signer, is_writable: false }
    }
}

#[derive(Clone, Debug)]
pub struct Transaction {
    pub instructions: Vec<Instruction>,
    pub payer: Option<Pubkey>,
}

impl Transaction {
    pub fn new_signed_with_payer(
        _instructions: &[Instruction],
        _payer: Option<&Pubkey>,
        _signers: &[&Keypair],
        _recent_blockhash: Hash,
    ) -> Self {
        Self {
            instructions: Vec::new(),
            payer: None,
        }
    }
}

// Mock Pubkey implementation
pub mod pubkey {
    use super::Pubkey;

    // Helper function to parse pubkey from string (since we can't impl FromStr for String)
    pub fn from_str(s: &str) -> Result<Pubkey, String> {
        Ok(s.to_string())
    }

    pub fn find_program_address(seeds: &[&[u8]], _program_id: &Pubkey) -> (Pubkey, u8) {
        let seed_str = String::from_utf8_lossy(seeds[0]);
        (format!("pda_{}", seed_str), 255)
    }
}

// Mock Signature implementation
pub mod signature {
    use super::Signature;

    // Helper function to parse signature from string
    pub fn from_str(s: &str) -> Result<Signature, String> {
        Ok(s.to_string())
    }
}

// Mock system program
pub mod system_program {
    // Mock system program ID - return as a function to avoid const issues
    pub fn id() -> super::Pubkey {
        "11111111111111111111111111111111".to_string()
    }
}

#[derive(Clone)]
pub struct SolanaClient;

impl SolanaClient {
    pub fn new(_rpc_url: &str, _payer: Keypair, _program_id: &str) -> Result<Self> {
        Ok(Self)
    }

    pub fn from_keypair_file(_rpc_url: &str, _keypair_path: &str, _program_id: &str) -> Result<Self> {
        Err(ServerError::Blockchain("Solana disabled for quick test".to_string()))
    }

    pub fn payer_pubkey(&self) -> Pubkey {
        "mock_pubkey".to_string()
    }

    pub async fn get_balance(&self, _pubkey: &Pubkey) -> Result<u64> {
        Ok(0)
    }

    pub async fn send_transaction(&self, _transaction: Transaction) -> Result<Signature> {
        // Return mock signature for quick test
        Ok("mock_signature_12345".to_string())
    }

    pub async fn get_latest_blockhash(&self) -> Result<Hash> {
        Ok("mock_hash".to_string())
    }

    pub fn program_id(&self) -> Pubkey {
        "mock_program_id".to_string()
    }

    pub async fn confirm_transaction(&self, _signature: &Signature) -> Result<bool> {
        Ok(false)
    }

    pub async fn get_account_data(&self, _pubkey: &Pubkey) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
}
