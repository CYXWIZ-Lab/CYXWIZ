pub mod infrastructure;
pub mod nodes;
pub mod scheduler;
pub mod blockchain;

use axum::{middleware, Router};
use crate::api::rest::v1::infrastructure::InfrastructureState;
use crate::auth::middleware::rest::{require_admin, AuthState};

pub fn create_v1_router(state: InfrastructureState) -> Router {
    // Public routes (health checks, stats) - no auth required
    let public_routes = Router::new()
        .merge(infrastructure::router(state.clone()));

    // Check if we have auth components available
    let has_auth = state.jwt_manager.is_some() && state.mongo_client.is_some();

    if has_auth {
        // Create AuthState from InfrastructureState
        let auth_state = AuthState {
            jwt_manager: state.jwt_manager.clone().unwrap(),
            mongo_client: state.mongo_client.clone().unwrap(),
            jwt_config: state.jwt_config.clone(),
        };

        // Admin routes (require admin authentication)
        let admin_routes = Router::new()
            .merge(nodes::router(state.clone()))
            .merge(scheduler::router(state.clone()))
            .merge(blockchain::router(state))
            .layer(middleware::from_fn_with_state(auth_state, require_admin));

        Router::new()
            .merge(public_routes)
            .merge(admin_routes)
    } else {
        // No auth available - allow all routes (development mode)
        tracing::warn!("Auth components not available - REST routes unprotected");
        Router::new()
            .merge(public_routes)
            .merge(nodes::router(state.clone()))
            .merge(scheduler::router(state.clone()))
            .merge(blockchain::router(state))
    }
}
