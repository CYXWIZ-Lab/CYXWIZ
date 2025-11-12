use crate::database::queries;
use crate::error::Result;
use crate::tui::app::{App, LogLevel};
use std::time::Instant;

pub async fn update_app_data(app: &mut App) -> Result<()> {
    let start = Instant::now();

    // Update uptime
    app.update_uptime();

    // Fetch nodes
    match queries::list_available_nodes(&app.db_pool).await {
        Ok(nodes) => {
            let total = nodes.len();
            let online = nodes.iter().filter(|n| matches!(n.status, crate::database::models::NodeStatus::Online)).count();

            app.stats.total_nodes = total;
            app.stats.online_nodes = online;
            app.nodes = nodes;
            app.stats.db_healthy = true;
        }
        Err(e) => {
            app.add_log(LogLevel::Error, format!("Failed to fetch nodes: {}", e));
            app.stats.db_healthy = false;
        }
    }

    // Fetch jobs
    match queries::list_pending_jobs(&app.db_pool, 100).await {
        Ok(jobs) => {
            let pending = jobs.iter().filter(|j| matches!(j.status, crate::database::models::JobStatus::Pending)).count();
            let running = jobs.iter().filter(|j| matches!(j.status, crate::database::models::JobStatus::Running)).count();

            app.stats.pending_jobs = pending;
            app.stats.active_jobs = running;
            app.jobs = jobs;
        }
        Err(e) => {
            app.add_log(LogLevel::Error, format!("Failed to fetch jobs: {}", e));
        }
    }

    // Check Redis health (commented out for TUI-only mode without Redis)
    // let redis_start = Instant::now();
    // let redis_result = app.cache.write().await.get_queue_length().await;
    // match redis_result {
    //     Ok(_) => {
    //         app.stats.redis_healthy = true;
    //         app.stats.redis_latency_ms = redis_start.elapsed().as_millis() as u64;
    //     }
    //     Err(e) => {
    //         app.add_log(LogLevel::Error, format!("Redis error: {}", e));
    //         app.stats.redis_healthy = false;
    //     }
    // }
    // For TUI-only mode, just mark Redis as unavailable
    app.stats.redis_healthy = false;
    app.stats.redis_latency_ms = 0;

    // Update database latency
    app.stats.db_latency_ms = start.elapsed().as_millis() as u64;

    // Assume Solana is healthy (would need actual check)
    app.stats.solana_healthy = true;
    app.stats.solana_latency_ms = 45;

    app.last_update = chrono::Utc::now();

    Ok(())
}
