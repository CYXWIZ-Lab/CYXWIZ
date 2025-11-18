use super::models::*;
use crate::database::DbPool;
use crate::error::{Result, ServerError};
use chrono::{DateTime, Utc};
use sqlx::Row;
use uuid::Uuid;

// Node queries
pub async fn create_node(pool: &DbPool, node: &Node) -> Result<Node> {
    let row = sqlx::query_as::<_, Node>(
        r#"
        INSERT INTO nodes (
            id, wallet_address, name, status, reputation_score, stake_amount,
            cpu_cores, ram_gb, gpu_model, gpu_memory_gb, has_cuda, has_opencl,
            total_jobs_completed, total_jobs_failed, uptime_percentage, current_load,
            country, region, ip_address, port, last_heartbeat, registered_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
            $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23
        )
        RETURNING *
        "#,
    )
    .bind(&node.id)
    .bind(&node.wallet_address)
    .bind(&node.name)
    .bind(&node.status)
    .bind(node.reputation_score)
    .bind(node.stake_amount)
    .bind(node.cpu_cores)
    .bind(node.ram_gb)
    .bind(&node.gpu_model)
    .bind(node.gpu_memory_gb)
    .bind(node.has_cuda)
    .bind(node.has_opencl)
    .bind(node.total_jobs_completed)
    .bind(node.total_jobs_failed)
    .bind(node.uptime_percentage)
    .bind(node.current_load)
    .bind(&node.country)
    .bind(&node.region)
    .bind(&node.ip_address)
    .bind(node.port)
    .bind(node.last_heartbeat)
    .bind(node.registered_at)
    .bind(node.updated_at)
    .fetch_one(pool)
    .await?;

    Ok(row)
}

pub async fn get_node_by_id(pool: &DbPool, node_id: DbId) -> Result<Node> {
    #[cfg(feature = "sqlite-compat")]
    let node_id_str = node_id.clone();

    #[cfg(not(feature = "sqlite-compat"))]
    let node_id_str = node_id.to_string();

    let node = sqlx::query_as::<_, Node>("SELECT * FROM nodes WHERE id = $1")
        .bind(node_id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ServerError::NodeNotFound(node_id_str))?;

    Ok(node)
}

pub async fn get_node_by_wallet(pool: &DbPool, wallet: &str) -> Result<Option<Node>> {
    let node = sqlx::query_as::<_, Node>("SELECT * FROM nodes WHERE wallet_address = $1")
        .bind(wallet)
        .fetch_optional(pool)
        .await?;

    Ok(node)
}

pub async fn list_available_nodes(pool: &DbPool) -> Result<Vec<Node>> {
    let nodes = sqlx::query_as::<_, Node>(
        "SELECT * FROM nodes WHERE status = 'online' AND current_load < 0.9 ORDER BY reputation_score DESC"
    )
    .fetch_all(pool)
    .await?;

    Ok(nodes)
}

pub async fn get_all_online_nodes(pool: &DbPool) -> Result<Vec<Node>> {
    let nodes = sqlx::query_as::<_, Node>(
        "SELECT * FROM nodes WHERE status = 'online'"
    )
    .fetch_all(pool)
    .await?;

    Ok(nodes)
}

pub async fn update_node_status(pool: &DbPool, node_id: DbId, status: NodeStatus) -> Result<()> {
    sqlx::query("UPDATE nodes SET status = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2")
        .bind(status)
        .bind(node_id)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn update_node_heartbeat(pool: &DbPool, node_id: DbId) -> Result<()> {
    let now = chrono::Utc::now();
    sqlx::query("UPDATE nodes SET last_heartbeat = $1, updated_at = $2 WHERE id = $3")
        .bind(now)
        .bind(now)
        .bind(node_id)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn update_node_load(pool: &DbPool, node_id: DbId, load: f64) -> Result<()> {
    let now = chrono::Utc::now();
    sqlx::query("UPDATE nodes SET current_load = $1, updated_at = $2 WHERE id = $3")
        .bind(load)
        .bind(now)
        .bind(node_id)
        .execute(pool)
        .await?;

    Ok(())
}

// Job queries
pub async fn create_job(pool: &DbPool, job: &Job) -> Result<Job> {
    let row = sqlx::query_as::<_, Job>(
        r#"
        INSERT INTO jobs (
            id, user_wallet, status, job_type,
            required_gpu, required_gpu_memory_gb, required_ram_gb, estimated_duration_seconds,
            estimated_cost, actual_cost, assigned_node_id, retry_count,
            result_hash, error_message, metadata,
            created_at, started_at, completed_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
        )
        RETURNING *
        "#,
    )
    .bind(&job.id)
    .bind(&job.user_wallet)
    .bind(&job.status)
    .bind(&job.job_type)
    .bind(job.required_gpu)
    .bind(job.required_gpu_memory_gb)
    .bind(job.required_ram_gb)
    .bind(job.estimated_duration_seconds)
    .bind(job.estimated_cost)
    .bind(job.actual_cost)
    .bind(&job.assigned_node_id)
    .bind(job.retry_count)
    .bind(&job.result_hash)
    .bind(&job.error_message)
    .bind(&job.metadata)
    .bind(job.created_at)
    .bind(job.started_at)
    .bind(job.completed_at)
    .bind(job.updated_at)
    .fetch_one(pool)
    .await?;

    Ok(row)
}

pub async fn get_job_by_id(pool: &DbPool, job_id: DbId) -> Result<Job> {
    #[cfg(feature = "sqlite-compat")]
    let job_id_str = job_id.clone();

    #[cfg(not(feature = "sqlite-compat"))]
    let job_id_str = job_id.to_string();

    let job = sqlx::query_as::<_, Job>("SELECT * FROM jobs WHERE id = $1")
        .bind(&job_id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ServerError::JobNotFound(job_id_str))?;

    Ok(job)
}

pub async fn list_pending_jobs(pool: &DbPool, limit: i64) -> Result<Vec<Job>> {
    let jobs = sqlx::query_as::<_, Job>(
        "SELECT * FROM jobs WHERE status = 'pending' AND retry_count < 3 ORDER BY created_at ASC LIMIT $1"
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(jobs)
}

pub async fn assign_job_to_node(pool: &DbPool, job_id: DbId, node_id: DbId) -> Result<()> {
    sqlx::query(
        "UPDATE jobs SET status = 'assigned', assigned_node_id = $1, updated_at = NOW() WHERE id = $2"
    )
    .bind(node_id)
    .bind(job_id)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn start_job(pool: &DbPool, job_id: DbId) -> Result<()> {
    #[cfg(feature = "sqlite-compat")]
    let query = "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = $1";

    #[cfg(not(feature = "sqlite-compat"))]
    let query = "UPDATE jobs SET status = 'running', started_at = NOW(), updated_at = NOW() WHERE id = $1";

    sqlx::query(query)
        .bind(job_id)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn complete_job(pool: &DbPool, job_id: DbId, result_hash: &str, actual_cost: i64) -> Result<()> {
    #[cfg(feature = "sqlite-compat")]
    let query = "UPDATE jobs SET status = 'completed', result_hash = $2, actual_cost = $3, completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = $1";

    #[cfg(not(feature = "sqlite-compat"))]
    let query = "UPDATE jobs SET status = 'completed', result_hash = $2, actual_cost = $3, completed_at = NOW(), updated_at = NOW() WHERE id = $1";

    sqlx::query(query)
        .bind(job_id)
        .bind(result_hash)
        .bind(actual_cost)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn fail_job(pool: &DbPool, job_id: DbId, error_msg: &str) -> Result<()> {
    #[cfg(feature = "sqlite-compat")]
    let query = "UPDATE jobs SET status = 'failed', error_message = $2, retry_count = retry_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = $1";

    #[cfg(not(feature = "sqlite-compat"))]
    let query = "UPDATE jobs SET status = 'failed', error_message = $2, retry_count = retry_count + 1, updated_at = NOW() WHERE id = $1";

    sqlx::query(query)
        .bind(job_id)
        .bind(error_msg)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn update_job_status(pool: &DbPool, job_id: DbId, status: &str) -> Result<()> {
    #[cfg(feature = "sqlite-compat")]
    let query = "UPDATE jobs SET status = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $1";

    #[cfg(not(feature = "sqlite-compat"))]
    let query = "UPDATE jobs SET status = $2, updated_at = NOW() WHERE id = $1";

    sqlx::query(query)
        .bind(job_id)
        .bind(status)
        .execute(pool)
        .await?;

    Ok(())
}

// Payment queries
pub async fn create_payment(pool: &DbPool, payment: &Payment) -> Result<Payment> {
    let row = sqlx::query_as::<_, Payment>(
        r#"
        INSERT INTO payments (
            id, job_id, node_id, user_wallet, node_wallet,
            amount, platform_fee, node_reward, status,
            escrow_tx_hash, completion_tx_hash, escrow_account,
            created_at, locked_at, completed_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
        )
        RETURNING *
        "#,
    )
    .bind(&payment.id)
    .bind(&payment.job_id)
    .bind(&payment.node_id)
    .bind(&payment.user_wallet)
    .bind(&payment.node_wallet)
    .bind(payment.amount)
    .bind(payment.platform_fee)
    .bind(payment.node_reward)
    .bind(&payment.status)
    .bind(&payment.escrow_tx_hash)
    .bind(&payment.completion_tx_hash)
    .bind(&payment.escrow_account)
    .bind(payment.created_at)
    .bind(payment.locked_at)
    .bind(payment.completed_at)
    .fetch_one(pool)
    .await?;

    Ok(row)
}

pub async fn update_payment_status(
    pool: &DbPool,
    payment_id: Uuid,
    status: PaymentStatus,
    tx_hash: Option<&str>,
) -> Result<()> {
    sqlx::query(
        "UPDATE payments SET status = $1, completion_tx_hash = $2 WHERE id = $3"
    )
    .bind(status)
    .bind(tx_hash)
    .bind(payment_id)
    .execute(pool)
    .await?;

    Ok(())
}

// Metrics queries
pub async fn insert_node_metrics(pool: &DbPool, metrics: &NodeMetrics) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO node_metrics (
            id, node_id, cpu_usage_percent, ram_usage_percent,
            gpu_usage_percent, gpu_memory_usage_percent,
            network_rx_bytes, network_tx_bytes, active_jobs, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        "#,
    )
    .bind(&metrics.id)
    .bind(&metrics.node_id)
    .bind(metrics.cpu_usage_percent)
    .bind(metrics.ram_usage_percent)
    .bind(metrics.gpu_usage_percent)
    .bind(metrics.gpu_memory_usage_percent)
    .bind(metrics.network_rx_bytes)
    .bind(metrics.network_tx_bytes)
    .bind(metrics.active_jobs)
    .bind(metrics.timestamp)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn get_node_metrics_history(
    pool: &DbPool,
    node_id: Uuid,
    limit: i64,
) -> Result<Vec<NodeMetrics>> {
    let metrics = sqlx::query_as::<_, NodeMetrics>(
        "SELECT * FROM node_metrics WHERE node_id = $1 ORDER BY timestamp DESC LIMIT $2"
    )
    .bind(node_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(metrics)
}

// ============================================================================
// DEPLOYMENT QUERIES
// ============================================================================

// Model queries
// pub async fn create_model(pool: &DbPool, model: &super::models::Model) -> Result<super::models::Model> {
//     let row = sqlx::query_as::<_, super::models::Model>(
//         r#"
//         INSERT INTO models (
//             id, name, description, owner_user_id,
//             format, source, source_url, size_bytes,
//             min_vram_bytes, min_ram_bytes, min_cpu_cores, required_device_type, gpu_preference,
//             is_public, price_per_download, download_count, rating, rating_count, tags,
//             storage_path, checksum_sha256, metadata,
//             created_at, updated_at
//         ) VALUES (
//             $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
//             $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
//         )
//         RETURNING *
//         "#,
//     )
//     .bind(&model.id)
//     .bind(&model.name)
//     .bind(&model.description)
//     .bind(&model.owner_user_id)
//     .bind(&model.format)
//     .bind(&model.source)
//     .bind(&model.source_url)
//     .bind(model.size_bytes)
//     .bind(model.min_vram_bytes)
//     .bind(model.min_ram_bytes)
//     .bind(model.min_cpu_cores)
//     .bind(&model.required_device_type)
//     .bind(&model.gpu_preference)
//     .bind(model.is_public)
//     .bind(model.price_per_download)
//     .bind(model.download_count)
//     .bind(model.rating)
//     .bind(model.rating_count)
//     .bind(&model.tags)
//     .bind(&model.storage_path)
//     .bind(&model.checksum_sha256)
//     .bind(&model.metadata)
//     .bind(model.created_at)
//     .bind(model.updated_at)
//     .fetch_one(pool)
//     .await?;
//
//     Ok(row)
// }

// pub async fn get_model_by_id(pool: &DbPool, model_id: Uuid) -> Result<super::models::Model> {
//     let model = sqlx::query_as::<_, super::models::Model>("SELECT * FROM models WHERE id = $1")
//         .bind(model_id)
//         .fetch_optional(pool)
//         .await?
//         .ok_or_else(|| ServerError::NotFound(format!("Model {} not found", model_id)))?;
//
//     Ok(model)
// }
//
// pub async fn list_models(
//     pool: &DbPool,
//     search_query: Option<&str>,
//     tags: &[String],
//     public_only: bool,
//     limit: i64,
//     offset: i64,
// ) -> Result<Vec<super::models::Model>> {
//     let mut query = String::from("SELECT * FROM models WHERE 1=1");
//
//     if public_only {
//         query.push_str(" AND is_public = TRUE");
//     }
//
//     if let Some(search) = search_query {
//         if !search.is_empty() {
//             query.push_str(&format!(" AND (name ILIKE '%{}%' OR description ILIKE '%{}%')", search, search));
//         }
//     }
//
//     if !tags.is_empty() {
//         query.push_str(" AND tags && $1");
//     }
//
//     query.push_str(" ORDER BY rating DESC, download_count DESC LIMIT $2 OFFSET $3");
//
//     let models = if tags.is_empty() {
//         sqlx::query_as::<_, super::models::Model>(&query)
//             .bind(limit)
//             .bind(offset)
//             .fetch_all(pool)
//             .await?
//     } else {
//         sqlx::query_as::<_, super::models::Model>(&query)
//             .bind(tags)
//             .bind(limit)
//             .bind(offset)
//             .fetch_all(pool)
//             .await?
//     };
//
//     Ok(models)
// }
//
// pub async fn delete_model(pool: &DbPool, model_id: Uuid, owner_id: &str) -> Result<()> {
//     let result = sqlx::query("DELETE FROM models WHERE id = $1 AND owner_user_id = $2")
//         .bind(model_id)
//         .bind(owner_id)
//         .execute(pool)
//         .await?;
//
//     if result.rows_affected() == 0 {
//         return Err(ServerError::NotFound(format!("Model {} not found or not owned by user", model_id)));
//     }
//
//     Ok(())
// }

// Deployment queries
pub async fn create_deployment(pool: &DbPool, deployment: &super::models::Deployment) -> Result<super::models::Deployment> {
    let row = sqlx::query_as::<_, super::models::Deployment>(
        r#"
        INSERT INTO deployments (
            id, user_id, model_id,
            type, status, status_message,
            assigned_node_id, max_price_per_hour, actual_hourly_rate, preferred_region,
            environment_vars, runtime_params, port, enable_terminal,
            endpoint_url, terminal_endpoint,
            payment_escrow_address, payment_escrow_tx_hash, total_cost,
            uptime_seconds, total_requests, avg_latency_ms,
            created_at, started_at, stopped_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
            $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26
        )
        RETURNING *
        "#,
    )
    .bind(&deployment.id)
    .bind(&deployment.user_id)
    .bind(&deployment.model_id)
    .bind(&deployment.deployment_type)
    .bind(&deployment.status)
    .bind(&deployment.status_message)
    .bind(&deployment.assigned_node_id)
    .bind(deployment.max_price_per_hour)
    .bind(deployment.actual_hourly_rate)
    .bind(&deployment.preferred_region)
    .bind(&deployment.environment_vars)
    .bind(&deployment.runtime_params)
    .bind(deployment.port)
    .bind(deployment.enable_terminal)
    .bind(&deployment.endpoint_url)
    .bind(&deployment.terminal_endpoint)
    .bind(&deployment.payment_escrow_address)
    .bind(&deployment.payment_escrow_tx_hash)
    .bind(deployment.total_cost)
    .bind(deployment.uptime_seconds)
    .bind(deployment.total_requests)
    .bind(deployment.avg_latency_ms)
    .bind(deployment.created_at)
    .bind(deployment.started_at)
    .bind(deployment.stopped_at)
    .bind(deployment.updated_at)
    .fetch_one(pool)
    .await?;

    Ok(row)
}

pub async fn get_deployment_by_id(pool: &DbPool, deployment_id: Uuid) -> Result<super::models::Deployment> {
    let deployment = sqlx::query_as::<_, super::models::Deployment>("SELECT * FROM deployments WHERE id = $1")
        .bind(deployment_id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ServerError::NotFound(format!("Deployment {} not found", deployment_id)))?;

    Ok(deployment)
}

pub async fn list_deployments(
    pool: &DbPool,
    user_id: &str,
    type_filter: Option<super::models::DeploymentType>,
    status_filter: Option<super::models::DeploymentStatus>,
    limit: i64,
    offset: i64,
) -> Result<Vec<super::models::Deployment>> {
    let mut query = String::from("SELECT * FROM deployments WHERE user_id = $1");
    let mut param_count = 1;

    if type_filter.is_some() {
        param_count += 1;
        query.push_str(&format!(" AND type = ${}", param_count));
    }

    if status_filter.is_some() {
        param_count += 1;
        query.push_str(&format!(" AND status = ${}", param_count));
    }

    param_count += 1;
    let limit_param = param_count;
    param_count += 1;
    let offset_param = param_count;

    query.push_str(&format!(" ORDER BY created_at DESC LIMIT ${} OFFSET ${}", limit_param, offset_param));

    let mut q = sqlx::query_as::<_, super::models::Deployment>(&query).bind(user_id);

    if let Some(t) = type_filter {
        q = q.bind(t);
    }

    if let Some(s) = status_filter {
        q = q.bind(s);
    }

    let deployments = q.bind(limit).bind(offset).fetch_all(pool).await?;

    Ok(deployments)
}

pub async fn update_deployment_status(
    pool: &DbPool,
    deployment_id: Uuid,
    status: super::models::DeploymentStatus,
    message: Option<&str>,
) -> Result<()> {
    sqlx::query("UPDATE deployments SET status = $1, status_message = $2, updated_at = NOW() WHERE id = $3")
        .bind(status)
        .bind(message)
        .bind(deployment_id)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn update_deployment_node(
    pool: &DbPool,
    deployment_id: Uuid,
    node_id: Uuid,
    endpoint_url: &str,
    terminal_endpoint: Option<&str>,
) -> Result<()> {
    sqlx::query(
        "UPDATE deployments SET assigned_node_id = $1, endpoint_url = $2, terminal_endpoint = $3, updated_at = NOW() WHERE id = $4"
    )
    .bind(node_id)
    .bind(endpoint_url)
    .bind(terminal_endpoint)
    .bind(deployment_id)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn stop_deployment(pool: &DbPool, deployment_id: Uuid) -> Result<()> {
    sqlx::query("UPDATE deployments SET status = 'stopped', stopped_at = NOW(), updated_at = NOW() WHERE id = $1")
        .bind(deployment_id)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn delete_deployment(pool: &DbPool, deployment_id: Uuid, user_id: &str) -> Result<()> {
    let result = sqlx::query("DELETE FROM deployments WHERE id = $1 AND user_id = $2")
        .bind(deployment_id)
        .bind(user_id)
        .execute(pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(ServerError::NotFound(format!("Deployment {} not found or not owned by user", deployment_id)));
    }

    Ok(())
}

// Terminal session queries
pub async fn create_terminal_session(
    pool: &DbPool,
    session: &super::models::TerminalSession,
) -> Result<super::models::TerminalSession> {
    let row = sqlx::query_as::<_, super::models::TerminalSession>(
        r#"
        INSERT INTO terminal_sessions (
            id, deployment_id, user_id, status, rows, cols,
            last_activity, data_sent_bytes, data_received_bytes,
            created_at, closed_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        )
        RETURNING *
        "#,
    )
    .bind(&session.id)
    .bind(&session.deployment_id)
    .bind(&session.user_id)
    .bind(&session.status)
    .bind(session.rows)
    .bind(session.cols)
    .bind(session.last_activity)
    .bind(session.data_sent_bytes)
    .bind(session.data_received_bytes)
    .bind(session.created_at)
    .bind(session.closed_at)
    .fetch_one(pool)
    .await?;

    Ok(row)
}

pub async fn get_terminal_session_by_id(pool: &DbPool, session_id: Uuid) -> Result<super::models::TerminalSession> {
    let session = sqlx::query_as::<_, super::models::TerminalSession>(
        "SELECT * FROM terminal_sessions WHERE id = $1"
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ServerError::NotFound(format!("Terminal session {} not found", session_id)))?;

    Ok(session)
}

pub async fn update_terminal_session_activity(
    pool: &DbPool,
    session_id: Uuid,
    data_sent: i64,
    data_received: i64,
) -> Result<()> {
    sqlx::query(
        "UPDATE terminal_sessions SET data_sent_bytes = data_sent_bytes + $1, data_received_bytes = data_received_bytes + $2, last_activity = NOW() WHERE id = $3"
    )
    .bind(data_sent)
    .bind(data_received)
    .bind(session_id)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn close_terminal_session(pool: &DbPool, session_id: Uuid) -> Result<()> {
    sqlx::query("UPDATE terminal_sessions SET status = 'closed', closed_at = NOW() WHERE id = $1")
        .bind(session_id)
        .execute(pool)
        .await?;

    Ok(())
}

// Deployment metrics queries
pub async fn create_deployment_metric(
    pool: &DbPool,
    metric: &super::models::DeploymentMetric,
) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO deployment_metrics (
            id, deployment_id, cpu_usage_percent, gpu_usage_percent,
            memory_usage_bytes, vram_usage_bytes, request_count,
            avg_latency_ms, throughput_rps, error_count, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        "#,
    )
    .bind(&metric.id)
    .bind(&metric.deployment_id)
    .bind(metric.cpu_usage_percent)
    .bind(metric.gpu_usage_percent)
    .bind(metric.memory_usage_bytes)
    .bind(metric.vram_usage_bytes)
    .bind(metric.request_count)
    .bind(metric.avg_latency_ms)
    .bind(metric.throughput_rps)
    .bind(metric.error_count)
    .bind(metric.timestamp)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn get_deployment_metrics_history(
    pool: &DbPool,
    deployment_id: Uuid,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    limit: i64,
) -> Result<Vec<super::models::DeploymentMetric>> {
    let mut query = String::from("SELECT * FROM deployment_metrics WHERE deployment_id = $1");
    let mut param_count = 1;

    if start_time.is_some() {
        param_count += 1;
        query.push_str(&format!(" AND timestamp >= ${}", param_count));
    }

    if end_time.is_some() {
        param_count += 1;
        query.push_str(&format!(" AND timestamp <= ${}", param_count));
    }

    param_count += 1;
    query.push_str(&format!(" ORDER BY timestamp DESC LIMIT ${}", param_count));

    let mut q = sqlx::query_as::<_, super::models::DeploymentMetric>(&query).bind(deployment_id);

    if let Some(start) = start_time {
        q = q.bind(start);
    }

    if let Some(end) = end_time {
        q = q.bind(end);
    }

    let metrics = q.bind(limit).fetch_all(pool).await?;

    Ok(metrics)
}

// Model download tracking (commented out for TUI-only build)
// pub async fn record_model_download(
//     pool: &DbPool,
//     model_id: Uuid,
//     user_id: &str,
//     payment_amount: i64,
//     payment_tx_hash: Option<&str>,
// ) -> Result<()> {
//     sqlx::query(
//         "INSERT INTO model_downloads (id, model_id, user_id, payment_amount, payment_tx_hash, downloaded_at) VALUES ($1, $2, $3, $4, $5, NOW())"
//     )
//     .bind(Uuid::new_v4())
//     .bind(model_id)
//     .bind(user_id)
//     .bind(payment_amount)
//     .bind(payment_tx_hash)
//     .execute(pool)
//     .await?;
//
//     Ok(())
// }

// ================================================================================
// Model queries
// ================================================================================

pub async fn create_model(pool: &DbPool, model: &super::models::Model) -> Result<super::models::Model> {
    // TODO: Implement actual database insertion
    // For now, return a stub to allow compilation
    Ok(model.clone())
}

pub async fn get_model_by_id(pool: &DbPool, model_id: DbId) -> Result<super::models::Model> {
    // TODO: Implement actual database query
    // For now, return a stub to allow compilation
    use crate::database::models::{Model, ModelFormat, ModelSource};
    use chrono::Utc;

    Ok(Model {
        id: model_id,
        name: "stub_model".to_string(),
        description: Some("Stub model for compilation".to_string()),
        owner_user_id: "stub_owner".to_string(),
        format: ModelFormat::Onnx,
        source: ModelSource::CyxwizHub,
        source_url: None,
        size_bytes: 0,
        min_vram_bytes: 0,
        min_ram_bytes: 0,
        min_cpu_cores: 0,
        required_device_type: None,
        gpu_preference: None,
        is_public: false,
        price_per_download: 0,
        download_count: 0,
        rating: 0.0,
        rating_count: 0,
        tags: vec![],
        storage_path: String::new(),
        checksum_sha256: String::new(),
        metadata: serde_json::json!({}),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    })
}

pub async fn list_models(
    pool: &DbPool,
    user_id: Option<&str>,
    page_size: i64,
    page_token: Option<&str>,
    public_only: bool,
) -> Result<(Vec<super::models::Model>, Option<String>)> {
    // TODO: Implement actual database query with pagination
    // For now, return empty list to allow compilation
    Ok((vec![], None))
}

pub async fn delete_model(pool: &DbPool, model_id: Uuid, user_id: &str) -> Result<()> {
    // TODO: Implement actual database deletion
    // For now, return success to allow compilation
    Ok(())
}

// Get node by IP:port endpoint
pub async fn get_node_by_endpoint(pool: &DbPool, ip_address: &str, port: i32) -> Result<Option<Node>> {
    let node = sqlx::query_as::<_, Node>(
        "SELECT * FROM nodes WHERE ip_address = $1 AND port = $2"
    )
    .bind(ip_address)
    .bind(port)
    .fetch_optional(pool)
    .await?;

    Ok(node)
}

// Update node endpoint (IP and port)
pub async fn update_node_endpoint(pool: &DbPool, node_id: &DbId, ip_address: &str, port: i32) -> Result<()> {
    sqlx::query(
        "UPDATE nodes SET ip_address = $1, port = $2, updated_at = NOW() WHERE id = $3"
    )
    .bind(ip_address)
    .bind(port)
    .bind(node_id)
    .execute(pool)
    .await?;

    Ok(())
}
