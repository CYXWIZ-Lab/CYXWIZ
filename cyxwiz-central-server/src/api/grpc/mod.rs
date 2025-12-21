pub mod job_service;
pub mod node_service;
pub mod node_discovery_service;
pub mod job_status_service;
pub mod wallet_service;
// pub mod deployment_service;
// pub mod terminal_service;
// pub mod model_service;

pub use job_service::JobServiceImpl;
pub use node_service::NodeServiceImpl;
pub use node_discovery_service::NodeDiscoveryServiceImpl;
pub use job_status_service::JobStatusServiceImpl;
pub use wallet_service::WalletServiceImpl;
// pub use deployment_service::DeploymentServiceImpl;
// pub use terminal_service::TerminalServiceImpl;
// pub use model_service::ModelServiceImpl;

use std::sync::Arc;
use tonic::{metadata::MetadataMap, Status};
use crate::auth::{AuthContext, AuthInterceptor, JWTManager};
use crate::config::JwtConfig;
use crate::database::MongoClient;

/// Helper to authenticate gRPC requests using the AuthInterceptor
///
/// # Arguments
/// * `metadata` - gRPC request metadata containing Authorization header
/// * `jwt_manager` - JWT manager for token validation
/// * `mongo_client` - Optional MongoDB client for user lookups
/// * `jwt_config` - JWT configuration with secrets
///
/// # Returns
/// AuthContext with user information, or Status error
pub async fn authenticate_request(
    metadata: &MetadataMap,
    jwt_manager: &Arc<JWTManager>,
    mongo_client: Option<&Arc<MongoClient>>,
    jwt_config: &JwtConfig,
) -> Result<AuthContext, Status> {
    // Create a mock MongoClient if none available (will fail lookups gracefully)
    let mongo_arc = match mongo_client {
        Some(client) => Arc::clone(client),
        None => {
            // Return error if MongoDB is required for auth
            return Err(Status::unavailable(
                "Authentication service temporarily unavailable (MongoDB not connected)"
            ));
        }
    };

    let interceptor = AuthInterceptor::new(
        Arc::clone(jwt_manager),
        mongo_arc,
        jwt_config.clone(),
    );

    interceptor.authenticate(metadata).await
}

/// Helper to authenticate admin requests
pub async fn authenticate_admin_request(
    metadata: &MetadataMap,
    jwt_manager: &Arc<JWTManager>,
    mongo_client: Option<&Arc<MongoClient>>,
    jwt_config: &JwtConfig,
) -> Result<AuthContext, Status> {
    let mongo_arc = match mongo_client {
        Some(client) => Arc::clone(client),
        None => {
            return Err(Status::unavailable(
                "Authentication service temporarily unavailable (MongoDB not connected)"
            ));
        }
    };

    let interceptor = AuthInterceptor::new(
        Arc::clone(jwt_manager),
        mongo_arc,
        jwt_config.clone(),
    );

    interceptor.authenticate_admin(metadata).await
}
