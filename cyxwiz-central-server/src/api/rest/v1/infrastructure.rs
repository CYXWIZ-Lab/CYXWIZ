use axum::{
    extract::State,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::auth::JWTManager;
use crate::blockchain::PaymentProcessor;
use crate::cache::RedisCache;
use crate::config::JwtConfig;
use crate::database::{DbPool, MongoClient};
use crate::scheduler::JobScheduler;

/// Shared state for infrastructure endpoints
#[derive(Clone)]
pub struct InfrastructureState {
    pub db_pool: DbPool,
    pub redis_cache: Option<Arc<RwLock<RedisCache>>>,
    pub payment_processor: Option<Arc<PaymentProcessor>>,
    pub scheduler: Option<Arc<JobScheduler>>,
    pub start_time: DateTime<Utc>,
    // Authentication components
    pub mongo_client: Option<Arc<MongoClient>>,
    pub jwt_manager: Option<Arc<JWTManager>>,
    pub jwt_config: JwtConfig,
}

/// Service health status
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Down,
}

/// Individual service health info
#[derive(Debug, Serialize)]
pub struct ServiceHealth {
    pub status: ServiceStatus,
    pub latency_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Enhanced health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: i64,
    pub timestamp: DateTime<Utc>,
    pub services: ServicesHealth,
}

/// Health status of all services (simple strings for frontend)
#[derive(Debug, Serialize)]
pub struct ServicesHealth {
    pub database: String,
    pub redis: String,
    pub blockchain: String,
    pub scheduler: String,
}

/// Node statistics
#[derive(Debug, Serialize)]
pub struct NodeStatsResponse {
    pub total: i64,
    pub online: i64,
    pub offline: i64,
    pub busy: i64,
}

/// Job statistics
#[derive(Debug, Serialize)]
pub struct JobStatsResponse {
    pub total: i64,
    pub pending: i64,
    pub running: i64,
    pub completed: i64,
    pub failed: i64,
}

/// Compute statistics
#[derive(Debug, Serialize)]
pub struct ComputeStatsResponse {
    pub total_hours: f64,
    pub average_utilization: f64,
}

/// Payment statistics
#[derive(Debug, Serialize)]
pub struct PaymentStatsResponse {
    pub total_transactions: i64,
    pub total_volume_sol: f64,
    pub pending_escrows: i64,
}

/// Infrastructure statistics response (nested structure for frontend)
#[derive(Debug, Serialize)]
pub struct InfrastructureStats {
    pub nodes: NodeStatsResponse,
    pub jobs: JobStatsResponse,
    pub compute: ComputeStatsResponse,
    pub payments: PaymentStatsResponse,
}

pub fn router(state: InfrastructureState) -> Router {
    Router::new()
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/infrastructure/stats", get(get_infrastructure_stats))
        .with_state(state)
}

/// GET /api/v1/health - Enhanced health check
async fn health_check(State(state): State<InfrastructureState>) -> impl IntoResponse {
    let now = Utc::now();
    let uptime = (now - state.start_time).num_seconds();

    // Check database health
    let db_health = check_database_health(&state.db_pool).await;

    // Check Redis health
    let redis_health = check_redis_health(&state.redis_cache).await;

    // Check blockchain health
    let blockchain_health = check_blockchain_health(&state.payment_processor).await;

    // Check scheduler health
    let scheduler_health = check_scheduler_health(&state.scheduler).await;

    // Determine overall status
    let overall_status = determine_overall_status(&db_health, &redis_health, &blockchain_health, &scheduler_health);

    // Convert ServiceStatus to simple strings for frontend
    let status_to_string = |status: &ServiceStatus| -> String {
        match status {
            ServiceStatus::Healthy => "healthy".to_string(),
            ServiceStatus::Degraded => "degraded".to_string(),
            ServiceStatus::Down => "disconnected".to_string(),
        }
    };

    Json(HealthResponse {
        status: status_to_string(&overall_status),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: uptime,
        timestamp: now,
        services: ServicesHealth {
            database: status_to_string(&db_health.status),
            redis: status_to_string(&redis_health.status),
            blockchain: status_to_string(&blockchain_health.status),
            scheduler: status_to_string(&scheduler_health.status),
        },
    })
}

/// GET /api/v1/infrastructure/stats - Infrastructure statistics
async fn get_infrastructure_stats(State(state): State<InfrastructureState>) -> impl IntoResponse {
    // Get node counts by status
    let node_stats = get_node_counts(&state.db_pool).await;

    // Get job counts by status
    let job_stats = get_job_counts(&state.db_pool).await;

    // Get payment stats
    let payment_stats = get_payment_stats(&state.db_pool).await;

    // Calculate average utilization (online nodes / total nodes)
    let avg_utilization = if node_stats.total > 0 {
        node_stats.busy as f64 / node_stats.total as f64
    } else {
        0.0
    };

    // Convert payment volume from lamports to SOL
    let volume_sol = payment_stats.volume as f64 / 1_000_000_000.0;

    Json(InfrastructureStats {
        nodes: NodeStatsResponse {
            total: node_stats.total,
            online: node_stats.online,
            offline: node_stats.offline,
            busy: node_stats.busy,
        },
        jobs: JobStatsResponse {
            total: job_stats.total,
            pending: job_stats.pending,
            running: job_stats.running,
            completed: job_stats.completed,
            failed: job_stats.failed,
        },
        compute: ComputeStatsResponse {
            total_hours: job_stats.compute_hours,
            average_utilization: avg_utilization,
        },
        payments: PaymentStatsResponse {
            total_transactions: payment_stats.count,
            total_volume_sol: volume_sol,
            pending_escrows: payment_stats.pending_escrows,
        },
    })
}

// Helper functions

async fn check_database_health(pool: &DbPool) -> ServiceHealth {
    let start = std::time::Instant::now();

    match sqlx::query("SELECT 1").execute(pool).await {
        Ok(_) => {
            let latency = start.elapsed().as_millis() as i64;
            ServiceHealth {
                status: if latency < 100 { ServiceStatus::Healthy } else { ServiceStatus::Degraded },
                latency_ms: Some(latency),
                message: None,
            }
        }
        Err(e) => ServiceHealth {
            status: ServiceStatus::Down,
            latency_ms: None,
            message: Some(format!("Database error: {}", e)),
        },
    }
}

async fn check_redis_health(redis: &Option<Arc<RwLock<RedisCache>>>) -> ServiceHealth {
    match redis {
        Some(cache) => {
            let start = std::time::Instant::now();
            let cache_guard = cache.read().await;

            // Try to ping Redis
            match cache_guard.ping().await {
                Ok(_) => {
                    let latency = start.elapsed().as_millis() as i64;
                    ServiceHealth {
                        status: if latency < 50 { ServiceStatus::Healthy } else { ServiceStatus::Degraded },
                        latency_ms: Some(latency),
                        message: None,
                    }
                }
                Err(e) => ServiceHealth {
                    status: ServiceStatus::Down,
                    latency_ms: None,
                    message: Some(format!("Redis error: {}", e)),
                },
            }
        }
        None => ServiceHealth {
            status: ServiceStatus::Down,
            latency_ms: None,
            message: Some("Redis not configured".to_string()),
        },
    }
}

async fn check_blockchain_health(processor: &Option<Arc<PaymentProcessor>>) -> ServiceHealth {
    match processor {
        Some(proc) => {
            let start = std::time::Instant::now();

            // Try to get balance as a health check
            match proc.check_health().await {
                Ok(_) => {
                    let latency = start.elapsed().as_millis() as i64;
                    ServiceHealth {
                        status: if latency < 500 { ServiceStatus::Healthy } else { ServiceStatus::Degraded },
                        latency_ms: Some(latency),
                        message: None,
                    }
                }
                Err(e) => ServiceHealth {
                    status: ServiceStatus::Degraded,
                    latency_ms: None,
                    message: Some(format!("Blockchain warning: {}", e)),
                },
            }
        }
        None => ServiceHealth {
            status: ServiceStatus::Down,
            latency_ms: None,
            message: Some("Blockchain not configured".to_string()),
        },
    }
}

async fn check_scheduler_health(scheduler: &Option<Arc<JobScheduler>>) -> ServiceHealth {
    match scheduler {
        Some(sched) => {
            let is_running = sched.is_running();
            ServiceHealth {
                status: if is_running { ServiceStatus::Healthy } else { ServiceStatus::Degraded },
                latency_ms: None,
                message: if is_running { None } else { Some("Scheduler paused".to_string()) },
            }
        }
        None => ServiceHealth {
            status: ServiceStatus::Down,
            latency_ms: None,
            message: Some("Scheduler not configured".to_string()),
        },
    }
}

fn determine_overall_status(
    db: &ServiceHealth,
    redis: &ServiceHealth,
    blockchain: &ServiceHealth,
    scheduler: &ServiceHealth,
) -> ServiceStatus {
    // Database is critical
    if matches!(db.status, ServiceStatus::Down) {
        return ServiceStatus::Down;
    }

    // Check for any degraded services
    let degraded_count = [db, redis, blockchain, scheduler]
        .iter()
        .filter(|s| matches!(s.status, ServiceStatus::Degraded))
        .count();

    let down_count = [redis, blockchain, scheduler]
        .iter()
        .filter(|s| matches!(s.status, ServiceStatus::Down))
        .count();

    if down_count >= 2 || degraded_count >= 3 {
        ServiceStatus::Degraded
    } else {
        ServiceStatus::Healthy
    }
}

// Database count helpers

struct NodeCounts {
    total: i64,
    online: i64,
    offline: i64,
    busy: i64,
}

async fn get_node_counts(pool: &DbPool) -> NodeCounts {
    let total = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    let online = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes WHERE status = 'online'")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    let offline = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes WHERE status = 'offline'")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    let busy = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM nodes WHERE status = 'busy'")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    NodeCounts { total, online, offline, busy }
}

struct JobCounts {
    total: i64,
    pending: i64,
    running: i64,
    completed: i64,
    failed: i64,
    compute_hours: f64,
}

async fn get_job_counts(pool: &DbPool) -> JobCounts {
    let total = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM jobs")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    let pending = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM jobs WHERE status = 'pending'")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    let running = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    let completed = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM jobs WHERE status = 'completed'")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    let failed = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM jobs WHERE status = 'failed'")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    // Calculate total compute hours from completed jobs
    let compute_hours = sqlx::query_scalar::<_, Option<f64>>(
        "SELECT SUM(CAST(estimated_duration_seconds AS REAL) / 3600.0) FROM jobs WHERE status = 'completed'"
    )
    .fetch_one(pool)
    .await
    .unwrap_or(None)
    .unwrap_or(0.0);

    JobCounts { total, pending, running, completed, failed, compute_hours }
}

struct PaymentCounts {
    count: i64,
    volume: i64,
    pending_escrows: i64,
}

async fn get_payment_stats(pool: &DbPool) -> PaymentCounts {
    let count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM payments")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    let volume = sqlx::query_scalar::<_, Option<i64>>("SELECT SUM(amount) FROM payments WHERE status = 'completed'")
        .fetch_one(pool)
        .await
        .unwrap_or(None)
        .unwrap_or(0);

    let pending_escrows = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM payments WHERE status = 'escrowed'")
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    PaymentCounts { count, volume, pending_escrows }
}
