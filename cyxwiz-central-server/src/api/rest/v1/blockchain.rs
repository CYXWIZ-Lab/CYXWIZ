use axum::{
    extract::{Query, State},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::api::rest::v1::infrastructure::InfrastructureState;

// ==================== PRICE SERVICE ====================

#[derive(Clone, Debug)]
struct PriceCache {
    sol_price_usd: f64,
    cyxwiz_price_usd: f64,
    last_updated: Instant,
}

impl Default for PriceCache {
    fn default() -> Self {
        Self {
            sol_price_usd: 0.0,
            cyxwiz_price_usd: 0.0,
            last_updated: Instant::now() - Duration::from_secs(3600),
        }
    }
}

lazy_static::lazy_static! {
    static ref PRICE_CACHE: Arc<RwLock<PriceCache>> = Arc::new(RwLock::new(PriceCache::default()));
}

/// Get current prices with caching (1 minute TTL)
async fn get_prices() -> (f64, f64) {
    let cache = PRICE_CACHE.read().await;
    if cache.last_updated.elapsed() < Duration::from_secs(60) && cache.sol_price_usd > 0.0 {
        return (cache.sol_price_usd, cache.cyxwiz_price_usd);
    }
    drop(cache);

    let sol_price = fetch_sol_price().await.unwrap_or(0.0);
    // Mock CYXWIZ price for devnet - in production use Jupiter API
    let cyxwiz_price = 0.01;

    debug!("Fetched prices: SOL=${:.2}, CYXWIZ=${:.4}", sol_price, cyxwiz_price);

    let mut cache = PRICE_CACHE.write().await;
    cache.sol_price_usd = sol_price;
    cache.cyxwiz_price_usd = cyxwiz_price;
    cache.last_updated = Instant::now();

    (sol_price, cyxwiz_price)
}

/// Fetch SOL price from CoinGecko
async fn fetch_sol_price() -> Option<f64> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?;

    let response = client
        .get("https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd")
        .send()
        .await
        .ok()?;

    if response.status().is_success() {
        let json: serde_json::Value = response.json().await.ok()?;
        json.get("solana")?.get("usd")?.as_f64()
    } else {
        warn!("Failed to fetch SOL price: {}", response.status());
        None
    }
}

/// Price response
#[derive(Debug, Serialize)]
pub struct PriceInfo {
    pub sol_price_usd: f64,
    pub cyxwiz_price_usd: f64,
    pub last_updated: String,
}

// ==================== DATA TYPES ====================

/// Query parameters for transactions
#[derive(Debug, Deserialize)]
pub struct TransactionQueryParams {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
    pub status: Option<String>,
}

/// Wallet info response (with CYXWIZ and USD equivalents)
#[derive(Debug, Serialize)]
pub struct WalletInfo {
    pub address: String,
    pub network: String,
    pub sol_balance: f64,
    pub sol_balance_usd: f64,
    pub cyxwiz_balance: f64,
    pub cyxwiz_balance_usd: f64,
    pub token_mint: String,
    pub sol_price_usd: f64,
    pub cyxwiz_price_usd: f64,
    pub is_connected: bool,
    pub rpc_endpoint: String,
    pub program_id: String,
}

/// Transaction view
#[derive(Debug, Serialize)]
pub struct TransactionView {
    pub id: String,
    pub tx_type: String,
    pub job_id: String,
    pub amount: i64,
    pub amount_sol: f64,
    pub platform_fee: i64,
    pub node_reward: i64,
    pub status: String,
    pub escrow_tx_hash: Option<String>,
    pub completion_tx_hash: Option<String>,
    pub user_wallet: String,
    pub node_wallet: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

/// Active escrow view
#[derive(Debug, Serialize)]
pub struct EscrowView {
    pub id: String,
    pub job_id: String,
    pub user_wallet: String,
    pub node_wallet: Option<String>,
    pub amount: i64,
    pub amount_sol: f64,
    pub status: String,
    pub escrow_account: Option<String>,
    pub created_at: String,
}

/// Blockchain statistics (with USD equivalents)
#[derive(Debug, Serialize)]
pub struct BlockchainStats {
    pub total_transactions: i64,
    pub total_volume_lamports: i64,
    pub total_volume_sol: f64,
    pub total_volume_usd: f64,
    pub platform_fees_collected: i64,
    pub platform_fees_sol: f64,
    pub platform_fees_usd: f64,
    pub node_payouts_total: i64,
    pub node_payouts_sol: f64,
    pub node_payouts_usd: f64,
    pub active_escrows_count: i64,
    pub active_escrows_value: i64,
    pub active_escrows_sol: f64,
    pub active_escrows_usd: f64,
    pub transactions_24h: i64,
    pub volume_24h_lamports: i64,
    pub volume_24h_sol: f64,
    pub volume_24h_usd: f64,
    pub sol_price_usd: f64,
    pub cyxwiz_price_usd: f64,
}

/// Transactions response
#[derive(Debug, Serialize)]
pub struct TransactionsResponse {
    pub transactions: Vec<TransactionView>,
    pub total: i64,
    pub page: i32,
    pub limit: i64,
    pub total_pages: i32,
}

/// Escrows response
#[derive(Debug, Serialize)]
pub struct EscrowsResponse {
    pub escrows: Vec<EscrowView>,
    pub total_locked_lamports: i64,
    pub total_locked_sol: f64,
}

// ==================== ROUTES ====================

pub fn router(state: InfrastructureState) -> Router {
    Router::new()
        .route("/api/v1/blockchain/wallet", get(get_wallet_info))
        .route("/api/v1/blockchain/transactions", get(get_transactions))
        .route("/api/v1/blockchain/escrows", get(get_active_escrows))
        .route("/api/v1/blockchain/stats", get(get_blockchain_stats))
        .route("/api/v1/blockchain/prices", get(get_current_prices))
        .with_state(state)
}

// Helper to convert lamports to SOL
fn lamports_to_sol(lamports: i64) -> f64 {
    lamports as f64 / 1_000_000_000.0
}

// Helper to convert token units to UI amount (9 decimals)
fn tokens_to_ui_amount(amount: u64) -> f64 {
    amount as f64 / 1_000_000_000.0
}

// ==================== ENDPOINTS ====================

/// GET /api/v1/blockchain/prices - Current token prices
async fn get_current_prices() -> impl IntoResponse {
    let (sol_price, cyxwiz_price) = get_prices().await;

    Json(PriceInfo {
        sol_price_usd: sol_price,
        cyxwiz_price_usd: cyxwiz_price,
        last_updated: chrono::Utc::now().to_rfc3339(),
    })
}

/// GET /api/v1/blockchain/wallet - Platform wallet info with balances
async fn get_wallet_info(State(state): State<InfrastructureState>) -> impl IntoResponse {
    let (sol_price, cyxwiz_price) = get_prices().await;

    let (address, network, sol_balance, cyxwiz_balance, token_mint, is_connected, rpc_endpoint, program_id) =
        if let Some(ref processor) = state.payment_processor {
            // get_payer_balance() already returns SOL (not lamports)
            let sol_balance = processor.get_payer_balance().await.unwrap_or(0.0);

            // Get CYXWIZ token balance
            let cyxwiz_balance = match processor.get_token_balance(&processor.payer_pubkey()).await {
                Ok(amount) => tokens_to_ui_amount(amount),
                Err(e) => {
                    warn!("Failed to get CYXWIZ balance: {}", e);
                    0.0
                }
            };

            (
                processor.payer_pubkey(),
                "devnet".to_string(),
                sol_balance,
                cyxwiz_balance,
                processor.token_mint().to_string(),
                true,
                "https://api.devnet.solana.com".to_string(),
                processor.program_id(),
            )
        } else {
            (
                "Not configured".to_string(),
                "unknown".to_string(),
                0.0,
                0.0,
                "Not configured".to_string(),
                false,
                "Not configured".to_string(),
                "Not configured".to_string(),
            )
        };

    Json(WalletInfo {
        address,
        network,
        sol_balance,
        sol_balance_usd: sol_balance * sol_price,
        cyxwiz_balance,
        cyxwiz_balance_usd: cyxwiz_balance * cyxwiz_price,
        token_mint,
        sol_price_usd: sol_price,
        cyxwiz_price_usd: cyxwiz_price,
        is_connected,
        rpc_endpoint,
        program_id,
    })
}

/// GET /api/v1/blockchain/transactions - Transaction history
async fn get_transactions(
    State(state): State<InfrastructureState>,
    Query(params): Query<TransactionQueryParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(50).min(100);
    let offset = params.offset.unwrap_or(0);

    // Build query with optional status filter
    let mut query = String::from("SELECT * FROM payments");
    let mut count_query = String::from("SELECT COUNT(*) FROM payments");

    if let Some(ref status) = params.status {
        if status != "all" {
            query.push_str(&format!(" WHERE status = '{}'", status));
            count_query.push_str(&format!(" WHERE status = '{}'", status));
        }
    }

    query.push_str(" ORDER BY created_at DESC");
    query.push_str(&format!(" LIMIT {} OFFSET {}", limit, offset));

    // Get total count
    let total: i64 = sqlx::query_scalar(&count_query)
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    // Get payments
    let payments = sqlx::query_as::<_, crate::database::models::Payment>(&query)
        .fetch_all(&state.db_pool)
        .await
        .unwrap_or_default();

    let transactions: Vec<TransactionView> = payments.iter().map(|p| {
        TransactionView {
            id: p.id.to_string(),
            tx_type: format!("{:?}", p.status).to_lowercase(),
            job_id: p.job_id.to_string(),
            amount: p.amount,
            amount_sol: lamports_to_sol(p.amount),
            platform_fee: p.platform_fee,
            node_reward: p.node_reward,
            status: format!("{:?}", p.status).to_lowercase(),
            escrow_tx_hash: p.escrow_tx_hash.clone(),
            completion_tx_hash: p.completion_tx_hash.clone(),
            user_wallet: p.user_wallet.clone(),
            node_wallet: p.node_wallet.clone(),
            created_at: p.created_at.to_rfc3339(),
            completed_at: p.completed_at.map(|t| t.to_rfc3339()),
        }
    }).collect();

    let page = (offset / limit + 1) as i32;
    let total_pages = ((total as f64) / (limit as f64)).ceil() as i32;

    Json(TransactionsResponse {
        transactions,
        total,
        page,
        limit,
        total_pages,
    })
}

/// GET /api/v1/blockchain/escrows - Active escrows
async fn get_active_escrows(State(state): State<InfrastructureState>) -> impl IntoResponse {
    // Get payments with locked/pending status (active escrows)
    let escrows = sqlx::query_as::<_, crate::database::models::Payment>(
        "SELECT * FROM payments WHERE status IN ('pending', 'locked') ORDER BY created_at DESC"
    )
    .fetch_all(&state.db_pool)
    .await
    .unwrap_or_default();

    let escrow_views: Vec<EscrowView> = escrows.iter().map(|p| {
        EscrowView {
            id: p.id.to_string(),
            job_id: p.job_id.to_string(),
            user_wallet: p.user_wallet.clone(),
            node_wallet: p.node_wallet.clone(),
            amount: p.amount,
            amount_sol: lamports_to_sol(p.amount),
            status: format!("{:?}", p.status).to_lowercase(),
            escrow_account: p.escrow_account.clone(),
            created_at: p.created_at.to_rfc3339(),
        }
    }).collect();

    let total_locked: i64 = escrows.iter().map(|e| e.amount).sum();

    Json(EscrowsResponse {
        escrows: escrow_views,
        total_locked_lamports: total_locked,
        total_locked_sol: lamports_to_sol(total_locked),
    })
}

/// GET /api/v1/blockchain/stats - Blockchain statistics with USD values
async fn get_blockchain_stats(State(state): State<InfrastructureState>) -> impl IntoResponse {
    let (sol_price, cyxwiz_price) = get_prices().await;

    // Total transactions
    let total_transactions = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM payments")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    // Total volume (all completed payments)
    let total_volume = sqlx::query_scalar::<_, Option<i64>>(
        "SELECT SUM(amount) FROM payments WHERE status = 'completed'"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(None)
    .unwrap_or(0);

    // Platform fees collected
    let platform_fees = sqlx::query_scalar::<_, Option<i64>>(
        "SELECT SUM(platform_fee) FROM payments WHERE status = 'completed'"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(None)
    .unwrap_or(0);

    // Node payouts
    let node_payouts = sqlx::query_scalar::<_, Option<i64>>(
        "SELECT SUM(node_reward) FROM payments WHERE status = 'completed'"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(None)
    .unwrap_or(0);

    // Active escrows
    let active_escrows_count = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM payments WHERE status IN ('pending', 'locked')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    let active_escrows_value = sqlx::query_scalar::<_, Option<i64>>(
        "SELECT SUM(amount) FROM payments WHERE status IN ('pending', 'locked')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(None)
    .unwrap_or(0);

    // 24h metrics
    let transactions_24h = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM payments WHERE created_at > datetime('now', '-24 hours')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    let volume_24h = sqlx::query_scalar::<_, Option<i64>>(
        "SELECT SUM(amount) FROM payments WHERE status = 'completed' AND completed_at > datetime('now', '-24 hours')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(None)
    .unwrap_or(0);

    // Calculate SOL amounts
    let total_volume_sol = lamports_to_sol(total_volume);
    let platform_fees_sol = lamports_to_sol(platform_fees);
    let node_payouts_sol = lamports_to_sol(node_payouts);
    let active_escrows_sol = lamports_to_sol(active_escrows_value);
    let volume_24h_sol = lamports_to_sol(volume_24h);

    Json(BlockchainStats {
        total_transactions,
        total_volume_lamports: total_volume,
        total_volume_sol,
        total_volume_usd: total_volume_sol * sol_price,
        platform_fees_collected: platform_fees,
        platform_fees_sol,
        platform_fees_usd: platform_fees_sol * sol_price,
        node_payouts_total: node_payouts,
        node_payouts_sol,
        node_payouts_usd: node_payouts_sol * sol_price,
        active_escrows_count,
        active_escrows_value,
        active_escrows_sol,
        active_escrows_usd: active_escrows_sol * sol_price,
        transactions_24h,
        volume_24h_lamports: volume_24h,
        volume_24h_sol,
        volume_24h_usd: volume_24h_sol * sol_price,
        sol_price_usd: sol_price,
        cyxwiz_price_usd: cyxwiz_price,
    })
}
