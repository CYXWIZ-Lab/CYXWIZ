use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub blockchain: BlockchainConfig,
    pub scheduler: SchedulerConfig,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainConfig {
    pub solana_rpc_url: String,
    pub program_id: String,
    pub payer_keypair_path: String,
    pub network: String, // "devnet", "testnet", "mainnet-beta"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub job_poll_interval_ms: u64,
    pub node_heartbeat_timeout_ms: u64,
    pub max_retries: u32,
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
                solana_rpc_url: "https://api.devnet.solana.com".to_string(),
                program_id: "CyxWiz11111111111111111111111111111111111".to_string(),
                payer_keypair_path: "~/.config/solana/id.json".to_string(),
                network: "devnet".to_string(),
            },
            scheduler: SchedulerConfig {
                job_poll_interval_ms: 1000,
                node_heartbeat_timeout_ms: 30000,
                max_retries: 3,
            },
        }
    }
}
