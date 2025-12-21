use axum::{
    extract::{Query, State},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::api::rest::v1::infrastructure::InfrastructureState;

/// Query parameters for queue
#[derive(Debug, Deserialize)]
pub struct QueueQueryParams {
    pub limit: Option<i64>,
}

/// Scheduler status response (matches frontend SchedulerStatus type)
#[derive(Debug, Serialize)]
pub struct SchedulerStatus {
    pub running: bool,
    pub jobs_in_queue: i64,
    pub active_assignments: i64,
    pub uptime_seconds: i64,
}

/// Queued job view (matches frontend QueuedJob type)
#[derive(Debug, Serialize)]
pub struct QueuedJobView {
    pub id: String,
    pub job_type: String,
    pub status: String,
    pub priority: i32,
    pub user_wallet: String,
    pub estimated_cost: f64,
    pub created_at: String,
    pub required_gpu: bool,
    pub required_ram_gb: i32,
}

/// Job assignment view
#[derive(Debug, Serialize)]
pub struct JobAssignmentView {
    pub job_id: String,
    pub node_id: String,
    pub node_name: String,
    pub assigned_at: String,
    pub status: String,
    pub job_type: String,
}

/// Scheduler throughput metrics (matches frontend ThroughputStats type)
#[derive(Debug, Serialize)]
pub struct SchedulerThroughput {
    pub jobs_per_minute: f64,
    pub jobs_per_hour: f64,
    pub average_assignment_time_seconds: f64,
    pub total_assigned_today: i64,
    pub total_completed_today: i64,
}

/// Queue status response (matches frontend expected shape)
#[derive(Debug, Serialize)]
pub struct QueueStatus {
    pub jobs: Vec<QueuedJobView>,
    pub total: i64,
}

/// Assignments response (matches frontend expected shape)
#[derive(Debug, Serialize)]
pub struct AssignmentsResponse {
    pub assignments: Vec<JobAssignmentView>,
    pub total: i64,
}

pub fn router(state: InfrastructureState) -> Router {
    Router::new()
        .route("/api/v1/scheduler/status", get(get_scheduler_status))
        .route("/api/v1/scheduler/queue", get(get_job_queue))
        .route("/api/v1/scheduler/assignments", get(get_job_assignments))
        .route("/api/v1/scheduler/throughput", get(get_scheduler_throughput))
        .with_state(state)
}

/// GET /api/v1/scheduler/status - Scheduler status
async fn get_scheduler_status(State(state): State<InfrastructureState>) -> impl IntoResponse {
    let running = state.scheduler
        .as_ref()
        .map(|s| s.is_running())
        .unwrap_or(false);

    // Get queue size (pending jobs)
    let jobs_in_queue = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM jobs WHERE status = 'pending'"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    // Get active assignments (running jobs)
    let active_assignments = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM jobs WHERE status IN ('assigned', 'running')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    // Calculate uptime
    let uptime_seconds = (chrono::Utc::now() - state.start_time).num_seconds();

    Json(SchedulerStatus {
        running,
        jobs_in_queue,
        active_assignments,
        uptime_seconds,
    })
}

/// GET /api/v1/scheduler/queue - Job queue status
async fn get_job_queue(
    State(state): State<InfrastructureState>,
    Query(params): Query<QueueQueryParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(50).min(100);

    // Get queued jobs
    let db_jobs = sqlx::query_as::<_, crate::database::models::Job>(
        "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT $1"
    )
    .bind(limit)
    .fetch_all(&state.db_pool)
    .await
    .unwrap_or_default();

    let jobs: Vec<QueuedJobView> = db_jobs.iter().map(|j| {
        // Convert JobStatus enum to string
        let status_str = format!("{:?}", j.status).to_lowercase();
        // Priority based on retry_count (higher retry = higher priority)
        let priority = j.retry_count;
        // Convert estimated_cost from i64 (lamports) to f64 (SOL)
        let estimated_cost_sol = j.estimated_cost as f64 / 1_000_000_000.0;

        QueuedJobView {
            id: j.id.to_string(),
            job_type: j.job_type.clone(),
            status: status_str,
            priority,
            user_wallet: j.user_wallet.clone(),
            estimated_cost: estimated_cost_sol,
            created_at: j.created_at.to_rfc3339(),
            required_gpu: j.required_gpu,
            required_ram_gb: j.required_ram_gb,
        }
    }).collect();

    // Get total count
    let total = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM jobs WHERE status = 'pending'"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    Json(QueueStatus {
        jobs,
        total,
    })
}

/// GET /api/v1/scheduler/assignments - Recent job assignments
async fn get_job_assignments(State(state): State<InfrastructureState>) -> impl IntoResponse {
    // Get running jobs with their assigned nodes
    let assignments_query = r#"
        SELECT
            j.id as job_id,
            j.assigned_node_id,
            j.status,
            j.job_type,
            j.started_at,
            n.name as node_name
        FROM jobs j
        LEFT JOIN nodes n ON j.assigned_node_id = n.id
        WHERE j.status IN ('assigned', 'running')
        ORDER BY j.started_at DESC
        LIMIT 50
    "#;

    #[derive(sqlx::FromRow)]
    struct AssignmentRow {
        job_id: String,
        assigned_node_id: Option<String>,
        status: String,
        #[allow(dead_code)]
        job_type: String,
        started_at: Option<chrono::DateTime<chrono::Utc>>,
        node_name: Option<String>,
    }

    let rows = sqlx::query_as::<_, AssignmentRow>(assignments_query)
        .fetch_all(&state.db_pool)
        .await
        .unwrap_or_default();

    let assignments: Vec<JobAssignmentView> = rows.iter().map(|r| JobAssignmentView {
        job_id: r.job_id.clone(),
        node_id: r.assigned_node_id.clone().unwrap_or_default(),
        node_name: r.node_name.clone().unwrap_or_else(|| "Unknown".to_string()),
        assigned_at: r.started_at.map(|t| t.to_rfc3339()).unwrap_or_default(),
        status: r.status.clone(),
        job_type: r.job_type.clone(),
    }).collect();

    let total = assignments.len() as i64;

    Json(AssignmentsResponse {
        assignments,
        total,
    })
}

/// GET /api/v1/scheduler/throughput - Scheduler throughput metrics
async fn get_scheduler_throughput(State(state): State<InfrastructureState>) -> impl IntoResponse {
    // Jobs completed today (since midnight)
    let total_completed_today = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM jobs WHERE status = 'completed' AND completed_at > datetime('now', 'start of day')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    // Jobs assigned today
    let total_assigned_today = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM jobs WHERE started_at > datetime('now', 'start of day')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    // Jobs completed in last hour for rate calculation
    let jobs_last_hour = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM jobs WHERE status = 'completed' AND completed_at > datetime('now', '-1 hour')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    let jobs_per_hour = jobs_last_hour as f64;
    let jobs_per_minute = jobs_per_hour / 60.0;

    // Average assignment time (from creation to start)
    let avg_assignment_time = sqlx::query_scalar::<_, Option<f64>>(
        "SELECT AVG(CAST((julianday(started_at) - julianday(created_at)) * 86400 AS REAL)) FROM jobs WHERE started_at IS NOT NULL AND started_at > datetime('now', '-24 hours')"
    )
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(None)
    .unwrap_or(0.0);

    Json(SchedulerThroughput {
        jobs_per_minute,
        jobs_per_hour,
        average_assignment_time_seconds: avg_assignment_time,
        total_assigned_today,
        total_completed_today,
    })
}
