pub mod dashboard;
pub mod v1;

use axum::Router;
use axum::http::Method;
use tower_http::cors::{Any, CorsLayer};

use crate::api::rest::v1::infrastructure::InfrastructureState;

pub use dashboard::create_router;

/// Create the full REST API router with CORS and all endpoints
pub fn create_rest_router(state: InfrastructureState) -> Router {
    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers(Any);

    // Combine legacy dashboard routes with new v1 API
    let legacy_router = dashboard::create_router(state.db_pool.clone());
    let v1_router = v1::create_v1_router(state);

    Router::new()
        .merge(legacy_router)
        .merge(v1_router)
        .layer(cors)
}
