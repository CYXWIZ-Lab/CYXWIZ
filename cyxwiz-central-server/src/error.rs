use thiserror::Error;
use tonic::{Code, Status};

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Blockchain error: {0}")]
    Blockchain(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Job not found: {0}")]
    JobNotFound(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Internal server error: {0}")]
    Internal(String),

    #[error("Insufficient funds: required {required}, available {available}")]
    InsufficientFunds { required: u64, available: u64 },

    #[error("Node capacity exceeded")]
    CapacityExceeded,

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ServerError>;

impl From<ServerError> for Status {
    fn from(err: ServerError) -> Self {
        match err {
            ServerError::NodeNotFound(msg) | ServerError::JobNotFound(msg) | ServerError::NotFound(msg) => {
                Status::not_found(msg)
            }
            ServerError::InvalidRequest(msg) => Status::invalid_argument(msg),
            ServerError::InsufficientFunds { .. } => {
                Status::failed_precondition(err.to_string())
            }
            ServerError::CapacityExceeded => {
                Status::resource_exhausted("Node capacity exceeded")
            }
            ServerError::Database(e) => Status::internal(format!("Database error: {}", e)),
            ServerError::Redis(e) => Status::internal(format!("Cache error: {}", e)),
            ServerError::Blockchain(e) => Status::internal(format!("Blockchain error: {}", e)),
            _ => Status::internal(err.to_string()),
        }
    }
}
