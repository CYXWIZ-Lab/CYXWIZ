use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub mongo: MongoConfig,
    pub blockchain: BlockchainConfig,
    pub scheduler: SchedulerConfig,
    pub jwt: JwtConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub grpc_address: String,
    pub rest_address: String,
    pub max_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: usize,
}

/// MongoDB configuration for reading user data from Website's database.
/// Uses MongoDB Atlas connection string (mongodb+srv://).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MongoConfig {
    /// MongoDB connection URL (Atlas: mongodb+srv://user:pass@cluster.mongodb.net/db)
    pub url: String,
    /// Database name
    pub database: String,
    /// Users collection name (default: "users")
    #[serde(default = "default_user_collection")]
    pub user_collection: String,
}

fn default_user_collection() -> String {
    "users".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainConfig {
    /// Enable blockchain operations (set to false for local development)
    #[serde(default = "default_blockchain_enabled")]
    pub enabled: bool,
    pub solana_rpc_url: String,
    pub program_id: String,
    pub payer_keypair_path: String,
    pub network: String, // "devnet", "testnet", "mainnet-beta"
    /// SPL token mint address for CYXWIZ token (or wrapped SOL for testing)
    #[serde(default)]
    pub token_mint: Option<String>,
    /// Platform treasury token account for receiving fees
    #[serde(default)]
    pub platform_token_account: Option<String>,
    /// Platform fee percentage (default: 10%)
    #[serde(default = "default_platform_fee")]
    pub platform_fee_percent: u8,
}

fn default_blockchain_enabled() -> bool {
    true // Default to enabled for production safety
}

fn default_platform_fee() -> u8 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub job_poll_interval_ms: u64,
    pub node_heartbeat_timeout_ms: u64,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    /// Secret for user tokens (same as Website's AUTH_SECRET)
    pub secret: String,
    /// Secret for admin tokens (same as Admin App's ADMIN_AUTH_SECRET)
    #[serde(default)]
    pub admin_secret: Option<String>,
    /// Secret for P2P tokens (Central Server only)
    #[serde(default)]
    pub p2p_secret: Option<String>,
    /// P2P token expiration in seconds
    pub p2p_token_expiration_seconds: i64,
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, config::ConfigError> {
        let builder = config::Config::builder()
            .add_source(config::File::from(path.as_ref()))
            .add_source(config::Environment::with_prefix("CYXWIZ"))
            .build()?;

        builder.try_deserialize()
    }

    pub fn default() -> Self {
        Self {
            server: ServerConfig {
                grpc_address: "0.0.0.0:50051".to_string(),
                rest_address: "0.0.0.0:8080".to_string(),
                max_connections: 1000,
            },
            database: DatabaseConfig {
                url: "postgres://cyxwiz:cyxwiz@localhost/cyxwiz".to_string(),
                max_connections: 20,
                min_connections: 5,
            },
            redis: RedisConfig {
                url: "redis://localhost:6379".to_string(),
                pool_size: 10,
            },
            blockchain: BlockchainConfig {
                enabled: false, // Disabled by default for local development
                solana_rpc_url: "https://api.devnet.solana.com".to_string(),
                program_id: "DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA".to_string(),
                payer_keypair_path: "~/.config/solana/id.json".to_string(),
                network: "devnet".to_string(),
                token_mint: None, // Set to SPL token mint when available
                platform_token_account: None, // Set to platform treasury
                platform_fee_percent: 10,
            },
            scheduler: SchedulerConfig {
                job_poll_interval_ms: 1000,
                node_heartbeat_timeout_ms: 30000,
                max_retries: 3,
            },
            mongo: MongoConfig {
                url: "mongodb://localhost:27017/cyxwiz".to_string(),
                database: "cyxwiz".to_string(),
                user_collection: "users".to_string(),
            },
            jwt: JwtConfig {
                secret: "dev-secret-key-CHANGE-IN-PRODUCTION-use-long-random-string".to_string(),
                admin_secret: None,
                p2p_secret: None,
                p2p_token_expiration_seconds: 3600,
            },
        }
    }
}
