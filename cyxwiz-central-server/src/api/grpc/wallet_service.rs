use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};

use crate::blockchain::SolanaClient;
use crate::pb::{
    wallet_service_server::WalletService,
    ConnectWalletRequest, ConnectWalletResponse,
    GetBalanceRequest, GetBalanceResponse,
    GetTransactionHistoryRequest, GetTransactionHistoryResponse,
    EstimateJobCostRequest, EstimateJobCostResponse,
    Transaction, TransactionType, TransactionStatus,
    StatusCode,
};

use std::sync::Arc;
use solana_sdk::pubkey::Pubkey;
use solana_client::rpc_client::RpcClient;
use spl_associated_token_account::get_associated_token_address;
use std::str::FromStr;

/// CYXWIZ Token mint address on devnet
pub const CYXWIZ_TOKEN_MINT: &str = "Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi";

pub struct WalletServiceImpl {
    blockchain_client: Option<Arc<SolanaClient>>,
    /// Direct RPC client for extended queries (transaction history, etc.)
    rpc_client: Option<RpcClient>,
    /// CYXWIZ token mint pubkey
    token_mint: Pubkey,
}

impl WalletServiceImpl {
    pub fn new(blockchain_client: Option<Arc<SolanaClient>>) -> Self {
        let token_mint = Pubkey::from_str(CYXWIZ_TOKEN_MINT).unwrap_or_default();

        // Create RPC client for extended queries
        let rpc_client = Some(RpcClient::new("https://api.devnet.solana.com".to_string()));

        Self {
            blockchain_client,
            rpc_client,
            token_mint,
        }
    }

    /// Create with custom token mint
    pub fn with_token_mint(blockchain_client: Option<Arc<SolanaClient>>, token_mint: &str) -> Self {
        let token_mint = Pubkey::from_str(token_mint).unwrap_or_default();
        let rpc_client = Some(RpcClient::new("https://api.devnet.solana.com".to_string()));

        Self {
            blockchain_client,
            rpc_client,
            token_mint,
        }
    }

    /// Get SPL token balance for a wallet
    async fn get_token_balance(&self, wallet: &Pubkey) -> Result<f64, String> {
        let rpc = self.rpc_client.as_ref().ok_or("RPC client not available")?;

        // Get associated token account for CYXWIZ
        let token_account = get_associated_token_address(wallet, &self.token_mint);

        // Try to get token account balance
        match rpc.get_token_account_balance(&token_account) {
            Ok(balance) => {
                // ui_amount is the human-readable balance with decimals applied
                Ok(balance.ui_amount.unwrap_or(0.0))
            }
            Err(e) => {
                // Account might not exist yet (no tokens received)
                debug!("Token account not found for {}: {}", wallet, e);
                Ok(0.0)
            }
        }
    }

    /// Get SOL balance for a wallet
    async fn get_sol_balance(&self, wallet: &Pubkey) -> Result<f64, String> {
        let rpc = self.rpc_client.as_ref().ok_or("RPC client not available")?;

        match rpc.get_balance(wallet) {
            Ok(lamports) => {
                // Convert lamports to SOL (1 SOL = 1_000_000_000 lamports)
                Ok(lamports as f64 / 1_000_000_000.0)
            }
            Err(e) => {
                error!("Failed to get SOL balance: {}", e);
                Err(format!("Failed to get balance: {}", e))
            }
        }
    }
}

#[tonic::async_trait]
impl WalletService for WalletServiceImpl {
    async fn connect_wallet(
        &self,
        request: Request<ConnectWalletRequest>,
    ) -> Result<Response<ConnectWalletResponse>, Status> {
        let req = request.into_inner();

        info!("Wallet connection request: {}", req.wallet_address);

        // Validate Solana address
        match Pubkey::from_str(&req.wallet_address) {
            Ok(pubkey) => {
                debug!("Valid Solana address: {}", pubkey);

                Ok(Response::new(ConnectWalletResponse {
                    status: StatusCode::StatusSuccess as i32,
                    message: "Wallet connected successfully".to_string(),
                    is_valid: true,
                    wallet_address: req.wallet_address,
                }))
            }
            Err(e) => {
                warn!("Invalid Solana address: {}", e);

                Ok(Response::new(ConnectWalletResponse {
                    status: StatusCode::StatusError as i32,
                    message: format!("Invalid Solana address: {}", e),
                    is_valid: false,
                    wallet_address: String::new(),
                }))
            }
        }
    }

    async fn get_balance(
        &self,
        request: Request<GetBalanceRequest>,
    ) -> Result<Response<GetBalanceResponse>, Status> {
        let req = request.into_inner();

        info!("Balance request for: {}", req.wallet_address);

        // Parse wallet address
        let wallet_pubkey = Pubkey::from_str(&req.wallet_address)
            .map_err(|e| Status::invalid_argument(format!("Invalid wallet address: {}", e)))?;

        // Check if RPC client is available
        if self.rpc_client.is_none() {
            return Ok(Response::new(GetBalanceResponse {
                status: StatusCode::StatusError as i32,
                message: "Blockchain integration is disabled".to_string(),
                sol_balance: 0.0,
                cyxwiz_balance: 0.0,
                token_mint: String::new(),
            }));
        }

        // Query real blockchain balances
        let sol_balance = self.get_sol_balance(&wallet_pubkey).await.unwrap_or_else(|e| {
            warn!("Failed to get SOL balance: {}", e);
            0.0
        });

        let cyxwiz_balance = self.get_token_balance(&wallet_pubkey).await.unwrap_or_else(|e| {
            warn!("Failed to get CYXWIZ balance: {}", e);
            0.0
        });

        info!(
            "Balance for {}: {} SOL, {} CYXWIZ",
            req.wallet_address, sol_balance, cyxwiz_balance
        );

        Ok(Response::new(GetBalanceResponse {
            status: StatusCode::StatusSuccess as i32,
            message: "Balance retrieved successfully".to_string(),
            sol_balance,
            cyxwiz_balance,
            token_mint: self.token_mint.to_string(),
        }))
    }

    async fn get_transaction_history(
        &self,
        request: Request<GetTransactionHistoryRequest>,
    ) -> Result<Response<GetTransactionHistoryResponse>, Status> {
        let req = request.into_inner();

        info!("Transaction history request for: {}", req.wallet_address);

        // Parse wallet address
        let wallet_pubkey = Pubkey::from_str(&req.wallet_address)
            .map_err(|e| Status::invalid_argument(format!("Invalid wallet address: {}", e)))?;

        let rpc = match &self.rpc_client {
            Some(client) => client,
            None => {
                return Ok(Response::new(GetTransactionHistoryResponse {
                    status: StatusCode::StatusError as i32,
                    message: "Blockchain integration is disabled".to_string(),
                    transactions: vec![],
                    total_count: 0,
                }));
            }
        };

        // Query real blockchain transactions using getSignaturesForAddress
        let limit = if req.limit > 0 { req.limit as usize } else { 20 };

        let signatures = match rpc.get_signatures_for_address(&wallet_pubkey) {
            Ok(sigs) => sigs.into_iter().take(limit).collect::<Vec<_>>(),
            Err(e) => {
                warn!("Failed to get transaction signatures: {}", e);
                return Ok(Response::new(GetTransactionHistoryResponse {
                    status: StatusCode::StatusSuccess as i32,
                    message: "No transaction history found".to_string(),
                    transactions: vec![],
                    total_count: 0,
                }));
            }
        };

        let mut transactions = Vec::new();

        for sig_info in &signatures {
            // Determine transaction type based on memo or other indicators
            // For now, classify as generic transfer
            let tx_type = if sig_info.memo.as_ref().map(|m| m.contains("Job")).unwrap_or(false) {
                TransactionType::JobPayment
            } else if sig_info.memo.as_ref().map(|m| m.contains("Reward")).unwrap_or(false) {
                TransactionType::RewardClaim
            } else if sig_info.memo.as_ref().map(|m| m.contains("Escrow")).unwrap_or(false) {
                TransactionType::Stake // Use Stake for escrow operations
            } else if sig_info.memo.as_ref().map(|m| m.contains("Refund")).unwrap_or(false) {
                TransactionType::Refund
            } else {
                TransactionType::Unknown // Generic transactions
            };

            let status = if sig_info.err.is_some() {
                TransactionStatus::Failed
            } else if sig_info.confirmation_status.as_ref().map(|s| {
                matches!(s, solana_transaction_status::TransactionConfirmationStatus::Finalized)
            }).unwrap_or(false) {
                TransactionStatus::Confirmed
            } else {
                TransactionStatus::Pending
            };

            transactions.push(Transaction {
                signature: sig_info.signature.clone(),
                r#type: tx_type as i32,
                status: status as i32,
                amount: 0.0, // Would need to fetch full transaction to get amount
                timestamp: sig_info.block_time.unwrap_or(0),
                description: sig_info.memo.clone().unwrap_or_else(|| "Transaction".to_string()),
                job_id: 0, // Would need to parse transaction for job ID
            });
        }

        let total_count = transactions.len() as i32;

        info!(
            "Retrieved {} transactions for {}",
            total_count, req.wallet_address
        );

        Ok(Response::new(GetTransactionHistoryResponse {
            status: StatusCode::StatusSuccess as i32,
            message: "Transaction history retrieved successfully".to_string(),
            transactions,
            total_count,
        }))
    }

    async fn estimate_job_cost(
        &self,
        request: Request<EstimateJobCostRequest>,
    ) -> Result<Response<EstimateJobCostResponse>, Status> {
        let req = request.into_inner();

        info!("Cost estimation request for model: {}", req.model_type);

        // Simple cost estimation algorithm
        let base_cost = match req.model_type.as_str() {
            "MNIST" | "SimpleNN" => 0.1,
            "ResNet-18" | "VGG-16" => 1.0,
            "ResNet-50" | "GPT-2" => 10.0,
            "GPT-3" | "BERT-Large" => 50.0,
            _ => 0.5, // Default
        };

        let dataset_multiplier = (req.dataset_size as f64 / 10000.0).max(0.1);
        let epoch_multiplier = req.epochs as f64;

        let tier_multiplier = match req.node_tier.as_str() {
            "basic" => 1.0,
            "premium" => 1.5,
            "enterprise" => 2.0,
            _ => 1.0,
        };

        let estimated_cost = base_cost * dataset_multiplier * epoch_multiplier * tier_multiplier;
        let estimated_time = estimated_cost * 10.0; // Rough estimate: 10 minutes per CYXWIZ token

        let breakdown = format!(
            "Base: {} CYXWIZ\nDataset: {}x\nEpochs: {}x\nTier: {}x\nTotal: {:.2} CYXWIZ",
            base_cost,
            dataset_multiplier,
            epoch_multiplier,
            tier_multiplier,
            estimated_cost
        );

        Ok(Response::new(EstimateJobCostResponse {
            status: StatusCode::StatusSuccess as i32,
            message: "Cost estimated successfully".to_string(),
            estimated_cost,
            estimated_time_minutes: estimated_time,
            breakdown,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connect_wallet_valid() {
        let service = WalletServiceImpl::new(None);

        let request = Request::new(ConnectWalletRequest {
            wallet_address: "11111111111111111111111111111111".to_string(),
        });

        let response = service.connect_wallet(request).await.unwrap();
        assert_eq!(response.get_ref().status, StatusCode::StatusSuccess as i32);
        assert!(response.get_ref().is_valid);
    }

    #[tokio::test]
    async fn test_connect_wallet_invalid() {
        let service = WalletServiceImpl::new(None);

        let request = Request::new(ConnectWalletRequest {
            wallet_address: "invalid-address".to_string(),
        });

        let response = service.connect_wallet(request).await.unwrap();
        assert_eq!(response.get_ref().status, StatusCode::StatusError as i32);
        assert!(!response.get_ref().is_valid);
    }

    #[tokio::test]
    async fn test_estimate_job_cost() {
        let service = WalletServiceImpl::new(None);

        let request = Request::new(EstimateJobCostRequest {
            model_type: "MNIST".to_string(),
            epochs: 10,
            dataset_size: 60000,
            node_tier: "basic".to_string(),
        });

        let response = service.estimate_job_cost(request).await.unwrap();
        assert_eq!(response.get_ref().status, StatusCode::StatusSuccess as i32);
        assert!(response.get_ref().estimated_cost > 0.0);
    }
}
