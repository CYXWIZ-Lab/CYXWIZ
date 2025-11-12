use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use crate::database::DbPool;
use uuid::Uuid;

use crate::database::queries;

#[derive(Clone)]
pub struct AppState {
    pub db_pool: DbPool,
}

#[derive(Serialize)]
pub struct NetworkStats {
    pub total_nodes: i64,
    pub online_nodes: i64,
    pub total_jobs: i64,
    pub pending_jobs: i64,
    pub running_jobs: i64,
    pub completed_jobs: i64,
    pub total_compute_hours: f64,
}

#[derive(Serialize)]
pub struct NodeSummary {
    pub id: String,
    pub name: String,
    pub status: String,
    pub reputation: f64,
    pub jobs_completed: i64,
    pub current_load: f64,
}

#[derive(Serialize)]
pub struct JobSummary {
    pub id: String,
    pub user: String,
    pub status: String,
    pub job_type: String,
    pub created_at: String,
    pub assigned_node: Option<String>,
}

#[derive(Deserialize)]
pub struct ListQuery {
    pub page: Option<i32>,
    pub limit: Option<i32>,
}

pub fn create_router(db_pool: DbPool) -> Router {
    let state = AppState { db_pool };

    Router::new()
        .route("/api/health", get(health_check))
        .route("/api/stats", get(get_network_stats))
        .route("/api/nodes", get(list_nodes))
        .route("/api/nodes/:id", get(get_node))
        .route("/api/jobs", get(list_jobs))
        .route("/api/jobs/:id", get(get_job))
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "service": "cyxwiz-central-server"
    }))
}

async fn get_network_stats(State(state): State<AppState>) -> impl IntoResponse {
    // TODO: Implement actual queries for stats
    // For now, return mock data
    Json(NetworkStats {
        total_nodes: 100,
        online_nodes: 85,
        total_jobs: 1523,
        pending_jobs: 12,
        running_jobs: 8,
        completed_jobs: 1503,
        total_compute_hours: 45234.5,
    })
}

async fn list_nodes(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    match queries::list_available_nodes(&state.db_pool).await {
        Ok(nodes) => {
            let summaries: Vec<NodeSummary> = nodes
                .iter()
                .map(|n| NodeSummary {
                    id: n.id.to_string(),
                    name: n.name.clone(),
                    status: format!("{:?}", n.status),
                    reputation: n.reputation_score,
                    jobs_completed: n.total_jobs_completed,
                    current_load: n.current_load,
                })
                .collect();

            Json(summaries).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_node(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    let node_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "Invalid node ID" })),
            )
                .into_response()
        }
    };

    match queries::get_node_by_id(&state.db_pool, node_id).await {
        Ok(node) => Json(serde_json::json!({
            "id": node.id,
            "name": node.name,
            "wallet": node.wallet_address,
            "status": format!("{:?}", node.status),
            "reputation": node.reputation_score,
            "stake": node.stake_amount,
            "cpu_cores": node.cpu_cores,
            "ram_gb": node.ram_gb,
            "gpu_model": node.gpu_model,
            "gpu_memory_gb": node.gpu_memory_gb,
            "has_cuda": node.has_cuda,
            "jobs_completed": node.total_jobs_completed,
            "jobs_failed": node.total_jobs_failed,
            "uptime": node.uptime_percentage,
            "current_load": node.current_load,
            "registered_at": node.registered_at,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50).min(100) as i64;

    match queries::list_pending_jobs(&state.db_pool, limit).await {
        Ok(jobs) => {
            let summaries: Vec<JobSummary> = jobs
                .iter()
                .map(|j| JobSummary {
                    id: j.id.to_string(),
                    user: j.user_wallet.clone(),
                    status: format!("{:?}", j.status),
                    job_type: j.job_type.clone(),
                    created_at: j.created_at.to_rfc3339(),
                    assigned_node: j.assigned_node_id.map(|id| id.to_string()),
                })
                .collect();

            Json(summaries).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_job(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    let job_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "Invalid job ID" })),
            )
                .into_response()
        }
    };

    match queries::get_job_by_id(&state.db_pool, job_id).await {
        Ok(job) => Json(serde_json::json!({
            "id": job.id,
            "user": job.user_wallet,
            "status": format!("{:?}", job.status),
            "job_type": job.job_type,
            "required_gpu": job.required_gpu,
            "required_ram_gb": job.required_ram_gb,
            "estimated_duration_seconds": job.estimated_duration_seconds,
            "estimated_cost": job.estimated_cost,
            "actual_cost": job.actual_cost,
            "assigned_node_id": job.assigned_node_id,
            "retry_count": job.retry_count,
            "result_hash": job.result_hash,
            "error_message": job.error_message,
            "metadata": job.metadata,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}
