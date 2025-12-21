//! MongoDB client for reading user data from the shared Website database.
//!
//! This module provides READ-ONLY access to the Website's MongoDB database
//! to look up user information for authentication and authorization.
//!
//! Connection is typically to MongoDB Atlas (mongodb+srv://).

use crate::config::MongoConfig;
use crate::error::{Result, ServerError};
use mongodb::{
    bson::{doc, oid::ObjectId},
    options::ClientOptions,
    Client, Collection, Database,
};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

/// MongoDB client for user lookups
pub struct MongoClient {
    client: Client,
    database: Database,
    user_collection: String,
}

// ============================================================================
// MongoDB User Schema (matches Website's user.ts)
// ============================================================================

/// CyxWallet - auto-generated embedded Solana wallet
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MongoCyxWallet {
    pub public_key: String,
    // Note: We do NOT read encryptedSecretKey for security
    #[serde(skip_deserializing)]
    pub encrypted_secret_key: Option<String>,
    pub created_at: Option<mongodb::bson::DateTime>,
    pub last_exported_at: Option<mongodb::bson::DateTime>,
}

/// Wallet balance for token economy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MongoWalletBalance {
    /// Spendable balance for compute purchases
    #[serde(default)]
    pub spot: f64,
    /// Claimable rewards (past lock period)
    #[serde(default)]
    pub earn: f64,
    /// Pending rewards (within 7-day lock)
    #[serde(default)]
    pub earn_locked: f64,
    /// Deposits awaiting allocation
    #[serde(default)]
    pub funding: f64,
    /// Trading margin
    #[serde(default)]
    pub futures: f64,
    /// Staked tokens
    #[serde(default)]
    pub stake: f64,
    pub stake_lock_until: Option<mongodb::bson::DateTime>,
}

/// Node operator reputation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MongoReputation {
    /// 0.0 - 1.0 composite score
    #[serde(default = "default_reputation_score")]
    pub score: f64,
    /// Percentage uptime (0-100)
    #[serde(default)]
    pub uptime: f64,
    /// Job success rate (0-100)
    #[serde(default)]
    pub success_rate: f64,
    /// Average response time in ms
    #[serde(default)]
    pub avg_response_time: f64,
    /// Total jobs completed
    #[serde(default)]
    pub jobs_completed: i64,
    /// Total jobs failed
    #[serde(default)]
    pub jobs_failed: i64,
    pub last_active: Option<mongodb::bson::DateTime>,
    /// Reputation tier
    #[serde(default = "default_tier")]
    pub tier: String,
}

fn default_reputation_score() -> f64 {
    0.5
}

fn default_tier() -> String {
    "starter".to_string()
}

impl Default for MongoReputation {
    fn default() -> Self {
        Self {
            score: 0.5,
            uptime: 0.0,
            success_rate: 0.0,
            avg_response_time: 0.0,
            jobs_completed: 0,
            jobs_failed: 0,
            last_active: None,
            tier: "starter".to_string(),
        }
    }
}

/// Verification status for Sybil protection
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MongoVerification {
    #[serde(default)]
    pub wallet_verified: bool,
    #[serde(default)]
    pub phone_verified: bool,
    #[serde(default)]
    pub email_verified: bool,
    /// KYC verification level (0-3)
    #[serde(default)]
    pub kyc_level: i32,
    pub verified_at: Option<mongodb::bson::DateTime>,
    /// Risk score (0-100, lower is better)
    #[serde(default = "default_risk_score")]
    pub risk_score: f64,
}

fn default_risk_score() -> f64 {
    50.0
}

/// User settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MongoSettings {
    #[serde(default)]
    pub email_notifications: bool,
    #[serde(default)]
    pub marketing_emails: bool,
    #[serde(default)]
    pub download_alerts: bool,
    #[serde(default)]
    pub comment_alerts: bool,
    #[serde(default)]
    pub two_factor_enabled: bool,
    /// Is this user a node operator?
    #[serde(default)]
    pub node_enabled: bool,
    /// Auto-accept job assignments
    #[serde(default = "default_true")]
    pub node_auto_accept: bool,
    /// Max concurrent jobs
    #[serde(default = "default_max_concurrent")]
    pub node_max_concurrent: i32,
}

fn default_true() -> bool {
    true
}

fn default_max_concurrent() -> i32 {
    1
}

/// Subscription info
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MongoSubscription {
    #[serde(default = "default_plan")]
    pub plan: String,
    #[serde(default = "default_status")]
    pub status: String,
    pub current_period_end: Option<mongodb::bson::DateTime>,
    #[serde(default = "default_compute_credits")]
    pub monthly_compute_credits: i64,
    #[serde(default)]
    pub compute_credits_used: i64,
    pub compute_credits_reset_at: Option<mongodb::bson::DateTime>,
}

fn default_plan() -> String {
    "free".to_string()
}

fn default_status() -> String {
    "active".to_string()
}

fn default_compute_credits() -> i64 {
    100
}

impl Default for MongoSubscription {
    fn default() -> Self {
        Self {
            plan: "free".to_string(),
            status: "active".to_string(),
            current_period_end: None,
            monthly_compute_credits: 100,
            compute_credits_used: 0,
            compute_credits_reset_at: None,
        }
    }
}

/// User document from MongoDB (matches Website's User model)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MongoUser {
    #[serde(rename = "_id")]
    pub id: ObjectId,
    pub email: String,
    pub username: String,
    pub name: Option<String>,
    pub image: Option<String>,

    /// External wallet address (Phantom/Solflare)
    pub wallet_address: Option<String>,
    pub wallet_provider: Option<String>,

    /// Embedded CyxWallet (auto-generated on signup)
    pub cyx_wallet: Option<MongoCyxWallet>,

    /// Token balances
    #[serde(default)]
    pub wallet: MongoWalletBalance,

    /// Node operator reputation
    #[serde(default)]
    pub reputation: MongoReputation,

    /// Verification status
    #[serde(default)]
    pub verification: MongoVerification,

    /// User settings
    #[serde(default)]
    pub settings: MongoSettings,

    /// Subscription info
    #[serde(default)]
    pub subscription: MongoSubscription,

    pub created_at: Option<mongodb::bson::DateTime>,
    pub updated_at: Option<mongodb::bson::DateTime>,
}

impl MongoUser {
    /// Get the user's primary wallet address (CyxWallet or external wallet)
    pub fn primary_wallet(&self) -> Option<String> {
        // Prefer CyxWallet if available
        if let Some(ref cyx_wallet) = self.cyx_wallet {
            return Some(cyx_wallet.public_key.clone());
        }
        // Fallback to external wallet
        self.wallet_address.clone()
    }

    /// Get spendable balance
    pub fn spendable_balance(&self) -> f64 {
        self.wallet.spot
    }

    /// Check if user is a verified node operator
    pub fn is_node_operator(&self) -> bool {
        self.settings.node_enabled
    }

    /// Get remaining compute credits
    pub fn remaining_compute_credits(&self) -> i64 {
        self.subscription.monthly_compute_credits - self.subscription.compute_credits_used
    }
}

// ============================================================================
// MongoClient Implementation
// ============================================================================

impl MongoClient {
    /// Create a new MongoDB client
    ///
    /// For Atlas, use connection string like:
    /// `mongodb+srv://user:password@cluster.mongodb.net/cyxwiz`
    pub async fn new(config: &MongoConfig) -> Result<Self> {
        info!("Connecting to MongoDB: {}", mask_connection_string(&config.url));

        let client_options = ClientOptions::parse(&config.url)
            .await
            .map_err(|e| ServerError::Internal(format!("MongoDB connection error: {}", e)))?;

        let client = Client::with_options(client_options)
            .map_err(|e| ServerError::Internal(format!("MongoDB client error: {}", e)))?;

        // Test connection
        let database = client.database(&config.database);
        database
            .run_command(doc! { "ping": 1 }, None)
            .await
            .map_err(|e| ServerError::Internal(format!("MongoDB ping failed: {}", e)))?;

        info!("MongoDB connected successfully to database: {}", config.database);

        Ok(Self {
            client,
            database,
            user_collection: config.user_collection.clone(),
        })
    }

    /// Create a mock client for testing (no actual connection)
    pub fn new_mock() -> Self {
        warn!("Creating mock MongoDB client - user lookups will fail");
        // This will panic if used, but allows server to start without MongoDB
        Self {
            client: Client::with_options(ClientOptions::default()).unwrap(),
            database: Client::with_options(ClientOptions::default())
                .unwrap()
                .database("mock"),
            user_collection: "users".to_string(),
        }
    }

    /// Get the users collection
    fn users(&self) -> Collection<MongoUser> {
        self.database.collection(&self.user_collection)
    }

    /// Look up user by MongoDB ObjectId (from JWT sub claim)
    pub async fn get_user_by_id(&self, user_id: &str) -> Result<Option<MongoUser>> {
        let oid = ObjectId::parse_str(user_id)
            .map_err(|e| ServerError::InvalidRequest(format!("Invalid user ID format: {}", e)))?;

        let user = self
            .users()
            .find_one(doc! { "_id": oid }, None)
            .await
            .map_err(|e| {
                error!("MongoDB query error: {}", e);
                ServerError::Internal(format!("Failed to query user: {}", e))
            })?;

        Ok(user)
    }

    /// Look up user by email
    pub async fn get_user_by_email(&self, email: &str) -> Result<Option<MongoUser>> {
        let user = self
            .users()
            .find_one(doc! { "email": email.to_lowercase() }, None)
            .await
            .map_err(|e| {
                error!("MongoDB query error: {}", e);
                ServerError::Internal(format!("Failed to query user by email: {}", e))
            })?;

        Ok(user)
    }

    /// Look up user by wallet address (either CyxWallet or external wallet)
    pub async fn get_user_by_wallet(&self, wallet_address: &str) -> Result<Option<MongoUser>> {
        // Search in both wallet fields
        let user = self
            .users()
            .find_one(
                doc! {
                    "$or": [
                        { "walletAddress": wallet_address },
                        { "cyxWallet.publicKey": wallet_address }
                    ]
                },
                None,
            )
            .await
            .map_err(|e| {
                error!("MongoDB query error: {}", e);
                ServerError::Internal(format!("Failed to query user by wallet: {}", e))
            })?;

        Ok(user)
    }

    /// Get user's spendable balance
    pub async fn get_user_balance(&self, user_id: &str) -> Result<f64> {
        let user = self.get_user_by_id(user_id).await?;
        Ok(user.map(|u| u.spendable_balance()).unwrap_or(0.0))
    }

    /// Check if user exists and is verified
    pub async fn verify_user(&self, user_id: &str) -> Result<bool> {
        let user = self.get_user_by_id(user_id).await?;
        Ok(user.map(|u| u.verification.email_verified).unwrap_or(false))
    }

    /// Check if user has sufficient balance for a job
    pub async fn has_sufficient_balance(&self, user_id: &str, required_amount: f64) -> Result<bool> {
        let balance = self.get_user_balance(user_id).await?;
        Ok(balance >= required_amount)
    }
}

/// Mask connection string for logging (hide password)
fn mask_connection_string(url: &str) -> String {
    // Simple masking - hide everything after :// until @
    if let Some(proto_end) = url.find("://") {
        if let Some(at_pos) = url.find('@') {
            let proto = &url[..proto_end + 3];
            let rest = &url[at_pos..];
            return format!("{}****{}", proto, rest);
        }
    }
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_connection_string() {
        let url = "mongodb+srv://user:password@cluster.mongodb.net/db";
        let masked = mask_connection_string(url);
        assert!(!masked.contains("password"));
        assert!(masked.contains("mongodb+srv://"));
        assert!(masked.contains("@cluster.mongodb.net"));
    }

    #[test]
    fn test_mongo_user_primary_wallet() {
        let user = MongoUser {
            id: ObjectId::new(),
            email: "test@example.com".to_string(),
            username: "test".to_string(),
            name: None,
            image: None,
            wallet_address: Some("external_wallet".to_string()),
            wallet_provider: None,
            cyx_wallet: Some(MongoCyxWallet {
                public_key: "cyx_wallet_public_key".to_string(),
                encrypted_secret_key: None,
                created_at: None,
                last_exported_at: None,
            }),
            wallet: MongoWalletBalance::default(),
            reputation: MongoReputation::default(),
            verification: MongoVerification::default(),
            settings: MongoSettings::default(),
            subscription: MongoSubscription::default(),
            created_at: None,
            updated_at: None,
        };

        // Should prefer CyxWallet
        assert_eq!(user.primary_wallet(), Some("cyx_wallet_public_key".to_string()));
    }
}
