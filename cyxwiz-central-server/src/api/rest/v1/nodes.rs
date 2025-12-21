use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::api::rest::v1::infrastructure::InfrastructureState;
use crate::database::queries;

/// Node filter parameters
#[derive(Debug, Deserialize)]
pub struct NodeFilterParams {
    pub status: Option<String>,
    pub search: Option<String>,
    pub has_gpu: Option<bool>,
    pub region: Option<String>,
    pub page: Option<i32>,
    pub limit: Option<i32>,
}

/// Metrics query parameters
#[derive(Debug, Deserialize)]
pub struct MetricsQueryParams {
    pub limit: Option<i64>,
}

/// Live node view
#[derive(Debug, Serialize)]
pub struct LiveNodeView {
    pub id: String,
    pub name: String,
    pub wallet: String,
    pub status: String,
    pub reputation: f64,
    pub cpu_cores: i32,
    pub ram_gb: i32,
    pub gpu_model: Option<String>,
    pub gpu_memory_gb: Option<i32>,
    pub has_cuda: bool,
    pub current_load: f64,
    pub jobs_completed: i64,
    pub jobs_failed: i64,
    pub uptime: f64,
    pub last_heartbeat: String,
    pub country: Option<String>,
    pub region: Option<String>,
    pub ip_address: String,
    pub port: i32,
}

/// Node statistics
#[derive(Debug, Serialize)]
pub struct NodeStats {
    pub total: i64,
    pub online: i64,
    pub offline: i64,
    pub busy: i64,
    pub maintenance: i64,
    pub total_gpu_nodes: i64,
    pub average_reputation: f64,
    pub average_load: f64,
}

/// Node metrics history entry
#[derive(Debug, Serialize)]
pub struct NodeMetricsView {
    pub node_id: String,
    pub cpu_usage: f64,
    pub ram_usage: f64,
    pub gpu_usage: Option<f64>,
    pub network_rx: i64,
    pub network_tx: i64,
    pub active_jobs: i32,
    pub timestamp: String,
}

/// Paginated nodes response
#[derive(Debug, Serialize)]
pub struct PaginatedNodes {
    pub nodes: Vec<LiveNodeView>,
    pub total: i64,
    pub page: i32,
    pub limit: i32,
    pub total_pages: i32,
}

pub fn router(state: InfrastructureState) -> Router {
    Router::new()
        .route("/api/v1/nodes/live", get(list_live_nodes))
        .route("/api/v1/nodes/stats", get(get_node_stats))
        .route("/api/v1/nodes/:id/metrics", get(get_node_metrics))
        .with_state(state)
}

/// GET /api/v1/nodes/live - List live nodes with filtering
async fn list_live_nodes(
    State(state): State<InfrastructureState>,
    Query(params): Query<NodeFilterParams>,
) -> impl IntoResponse {
    let page = params.page.unwrap_or(1).max(1);
    let limit = params.limit.unwrap_or(50).min(100).max(1);
    let offset = ((page - 1) * limit) as i64;

    // Build query with filters
    let mut base_query = String::from("SELECT * FROM nodes WHERE 1=1");
    let mut count_query = String::from("SELECT COUNT(*) FROM nodes WHERE 1=1");

    if let Some(ref status) = params.status {
        if status != "all" {
            base_query.push_str(&format!(" AND status = '{}'", status));
            count_query.push_str(&format!(" AND status = '{}'", status));
        }
    }

    if let Some(ref search) = params.search {
        if !search.is_empty() {
            let search_filter = format!(" AND (name LIKE '%{}%' OR wallet_address LIKE '%{}%')", search, search);
            base_query.push_str(&search_filter);
            count_query.push_str(&search_filter);
        }
    }

    if let Some(has_gpu) = params.has_gpu {
        if has_gpu {
            base_query.push_str(" AND gpu_model IS NOT NULL");
            count_query.push_str(" AND gpu_model IS NOT NULL");
        } else {
            base_query.push_str(" AND gpu_model IS NULL");
            count_query.push_str(" AND gpu_model IS NULL");
        }
    }

    if let Some(ref region) = params.region {
        if !region.is_empty() {
            base_query.push_str(&format!(" AND region = '{}'", region));
            count_query.push_str(&format!(" AND region = '{}'", region));
        }
    }

    base_query.push_str(" ORDER BY reputation_score DESC, last_heartbeat DESC");
    base_query.push_str(&format!(" LIMIT {} OFFSET {}", limit, offset));

    // Execute queries
    let total: i64 = sqlx::query_scalar(&count_query)
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    let nodes_result = sqlx::query_as::<_, crate::database::models::Node>(&base_query)
        .fetch_all(&state.db_pool)
        .await;

    let nodes: Vec<LiveNodeView> = match nodes_result {
        Ok(nodes) => nodes.iter().map(|n| LiveNodeView {
            id: n.id.to_string(),
            name: n.name.clone(),
            wallet: n.wallet_address.clone(),
            status: format!("{:?}", n.status).to_lowercase(),
            reputation: n.reputation_score,
            cpu_cores: n.cpu_cores,
            ram_gb: n.ram_gb,
            gpu_model: n.gpu_model.clone(),
            gpu_memory_gb: n.gpu_memory_gb,
            has_cuda: n.has_cuda,
            current_load: n.current_load,
            jobs_completed: n.total_jobs_completed,
            jobs_failed: n.total_jobs_failed,
            uptime: n.uptime_percentage,
            last_heartbeat: n.last_heartbeat.to_rfc3339(),
            country: n.country.clone(),
            region: n.region.clone(),
            ip_address: n.ip_address.clone(),
            port: n.port,
        }).collect(),
        Err(_) => vec![],
    };

    let total_pages = ((total as f64) / (limit as f64)).ceil() as i32;

    Json(PaginatedNodes {
        nodes,
        total,
        page,
        limit,
        total_pages,
    })
}

/// GET /api/v1/nodes/stats - Node statistics
async fn get_node_stats(State(state): State<InfrastructureState>) -> impl IntoResponse {
    let total = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    let online = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes WHERE status = 'online'")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    let offline = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes WHERE status = 'offline'")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    let busy = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes WHERE status = 'busy'")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    let maintenance = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes WHERE status = 'maintenance'")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    let total_gpu_nodes = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes WHERE gpu_model IS NOT NULL")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(0);

    let average_reputation = sqlx::query_scalar::<_, Option<f64>>("SELECT AVG(reputation_score) FROM nodes")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(None)
        .unwrap_or(0.0);

    let average_load = sqlx::query_scalar::<_, Option<f64>>("SELECT AVG(current_load) FROM nodes WHERE status = 'online'")
        .fetch_one(&state.db_pool)
        .await
        .unwrap_or(None)
        .unwrap_or(0.0);

    Json(NodeStats {
        total,
        online,
        offline,
        busy,
        maintenance,
        total_gpu_nodes,
        average_reputation,
        average_load,
    })
}

/// GET /api/v1/nodes/:id/metrics - Node metrics history
async fn get_node_metrics(
    State(state): State<InfrastructureState>,
    Path(node_id): Path<String>,
    Query(params): Query<MetricsQueryParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(100).min(1000);

    // Parse UUID
    let parsed_id = match uuid::Uuid::parse_str(&node_id) {
        Ok(id) => id,
        Err(_) => {
            return Json(serde_json::json!({
                "error": "Invalid node ID format"
            })).into_response();
        }
    };

    match queries::get_node_metrics_history(&state.db_pool, parsed_id, limit).await {
        Ok(metrics) => {
            let views: Vec<NodeMetricsView> = metrics.iter().map(|m| NodeMetricsView {
                node_id: m.node_id.to_string(),
                cpu_usage: m.cpu_usage_percent,
                ram_usage: m.ram_usage_percent,
                gpu_usage: m.gpu_usage_percent,
                network_rx: m.network_rx_bytes,
                network_tx: m.network_tx_bytes,
                active_jobs: m.active_jobs,
                timestamp: m.timestamp.to_rfc3339(),
            }).collect();

            Json(serde_json::json!({
                "node_id": node_id,
                "metrics": views,
                "count": views.len()
            })).into_response()
        }
        Err(e) => {
            Json(serde_json::json!({
                "error": format!("Failed to fetch metrics: {}", e)
            })).into_response()
        }
    }
}
