use crate::cache::RedisCache;
use crate::config::SchedulerConfig;
use crate::database::{
    models::{Job, Node},
    queries,
};
use crate::error::Result;
use crate::scheduler::matcher::JobMatcher;
use crate::database::DbPool;
use crate::pb::{node_service_client::NodeServiceClient, AssignJobRequest, JobConfig, JobType, JobPriority, DeviceType};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};
use uuid::Uuid;

pub struct JobScheduler {
    db_pool: DbPool,
    cache: Arc<RwLock<RedisCache>>,
    config: SchedulerConfig,
}

impl JobScheduler {
    pub fn new(db_pool: DbPool, cache: RedisCache, config: SchedulerConfig) -> Self {
        Self {
            db_pool,
            cache: Arc::new(RwLock::new(cache)),
            config,
        }
    }

    /// Start the scheduler loop
    pub async fn run(self: Arc<Self>) {
        info!("Job scheduler started");

        let mut ticker = interval(Duration::from_millis(self.config.job_poll_interval_ms));

        loop {
            ticker.tick().await;

            if let Err(e) = self.process_pending_jobs().await {
                error!("Error processing jobs: {}", e);
            }
        }
    }

    /// Process pending jobs from the queue
    async fn process_pending_jobs(&self) -> Result<()> {
        // Get pending jobs from database
        let pending_jobs = queries::list_pending_jobs(&self.db_pool, 10).await?;

        if pending_jobs.is_empty() {
            return Ok(());
        }

        info!("Found {} pending jobs to process", pending_jobs.len());

        // Get available nodes
        let available_nodes = queries::list_available_nodes(&self.db_pool).await?;

        if available_nodes.is_empty() {
            warn!("No available nodes for job assignment");
            return Ok(());
        }

        info!("Found {} available nodes", available_nodes.len());

        // Match jobs to nodes
        for job in pending_jobs {
            if let Err(e) = self.assign_job_to_best_node(&job, &available_nodes).await {
                error!("Failed to assign job {}: {}", job.id, e);
            }
        }

        Ok(())
    }

    /// Assign a job to the best matching node
    async fn assign_job_to_best_node(&self, job: &Job, available_nodes: &[Node]) -> Result<()> {
        // Find best node using matcher
        let best_node = JobMatcher::find_best_node(job, available_nodes);

        match best_node {
            Some(node) => {
                info!(
                    "Assigning job {} to node {} (reputation: {:.2}, load: {:.2})",
                    job.id, node.id, node.reputation_score, node.current_load
                );

                // Update database
                queries::assign_job_to_node(&self.db_pool, job.id, node.id).await?;

                // Cache assignment
                let mut cache = self.cache.write().await;
                cache.cache_job_assignment(job.id, node.id, 3600).await?;

                // Notify node via gRPC about new job assignment
                if let Err(e) = self.send_job_to_node(node, job).await {
                    error!("Failed to send job {} to node {}: {}", job.id, node.id, e);
                    // TODO: Implement update_job_status to mark job as failed/pending
                    // For now, job remains in "assigned" state and will be retried
                    return Err(e);
                }

                info!("Job {} successfully sent to node {}", job.id, node.id);

                Ok(())
            }
            None => {
                warn!("No suitable node found for job {}", job.id);
                Ok(())
            }
        }
    }

    /// Handle job completion
    pub async fn handle_job_completion(
        &self,
        job_id: Uuid,
        result_hash: &str,
        actual_cost: i64,
    ) -> Result<()> {
        info!("Job {} completed with result hash {}", job_id, result_hash);

        // Update job status in database
        queries::complete_job(&self.db_pool, job_id, result_hash, actual_cost).await?;

        // Get job details
        let job = queries::get_job_by_id(&self.db_pool, job_id).await?;

        // Update node metrics
        if let Some(node_id) = job.assigned_node_id {
            // Increment completed jobs count
            // Update node load (decrement)
            queries::update_node_load(&self.db_pool, node_id, 0.0).await?;

            // TODO: Trigger payment distribution via blockchain
        }

        Ok(())
    }

    /// Handle job failure
    pub async fn handle_job_failure(&self, job_id: Uuid, error_message: &str) -> Result<()> {
        error!("Job {} failed: {}", job_id, error_message);

        // Mark job as failed
        queries::fail_job(&self.db_pool, job_id, error_message).await?;

        // Get job details
        let job = queries::get_job_by_id(&self.db_pool, job_id).await?;

        // If retry count is below max, re-queue the job
        if job.retry_count < self.config.max_retries as i32 {
            info!("Re-queuing job {} (retry {}/{})", job_id, job.retry_count, self.config.max_retries);

            // Add back to pending queue
            let mut cache = self.cache.write().await;
            cache.push_job(job_id).await?;
        } else {
            warn!("Job {} exceeded max retries, giving up", job_id);

            // TODO: Trigger refund via blockchain
        }

        // Update node reputation (penalize for failure)
        if let Some(node_id) = job.assigned_node_id {
            // TODO: Decrement node reputation
        }

        Ok(())
    }

    /// Monitor node heartbeats and mark offline nodes
    pub async fn monitor_node_heartbeats(&self) -> Result<()> {
        // Get all online nodes
        let nodes = queries::list_available_nodes(&self.db_pool).await?;

        let timeout_threshold = chrono::Utc::now()
            - chrono::Duration::milliseconds(self.config.node_heartbeat_timeout_ms as i64);

        for node in nodes {
            if node.last_heartbeat < timeout_threshold {
                warn!("Node {} missed heartbeat, marking offline", node.id);

                // Mark node as offline
                queries::update_node_status(&self.db_pool, node.id, crate::database::models::NodeStatus::Offline).await?;

                // Re-assign any jobs that were assigned to this node
                // TODO: Implement job reassignment
            }
        }

        Ok(())
    }

    // Send job assignment to Server Node via gRPC
    async fn send_job_to_node(&self, node: &Node, job: &Job) -> Result<()> {
        // For local testing, use hardcoded endpoint
        // TODO: Store node address in database during registration
        let node_endpoint = format!("http://127.0.0.1:50054");

        info!("Connecting to node {} at {}", node.id, node_endpoint);

        // Create gRPC client
        let mut client = NodeServiceClient::connect(node_endpoint)
            .await
            .map_err(|e| {
                error!("Failed to connect to node {}: {}", node.id, e);
                crate::error::ServerError::Internal(format!("gRPC connection failed: {}", e))
            })?;

        // Build JobConfig from database Job model
        let job_config = self.build_job_config(job)?;

        // Create AssignJob request
        let request = AssignJobRequest {
            node_id: node.id.to_string(),
            job: Some(job_config),
            authorization_token: String::new(), // TODO: Implement auth tokens
        };

        info!("Sending AssignJob RPC for job {} to node {}", job.id, node.id);

        // Call AssignJob RPC
        let response = client.assign_job(request)
            .await
            .map_err(|e| {
                error!("AssignJob RPC failed for job {}: {}", job.id, e);
                crate::error::ServerError::Internal(format!("AssignJob RPC failed: {}", e))
            })?;

        let assign_response = response.into_inner();

        // Check if node accepted the job
        if assign_response.accepted {
            info!("Node {} accepted job {}", node.id, job.id);
            Ok(())
        } else {
            let error_msg = assign_response.error
                .map(|e| e.message)
                .unwrap_or_else(|| "Unknown error".to_string());
            warn!("Node {} rejected job {}: {}", node.id, job.id, error_msg);
            Err(crate::error::ServerError::Internal(format!("Node rejected job: {}", error_msg)))
        }
    }

    // Convert database Job model to protobuf JobConfig
    fn build_job_config(&self, job: &Job) -> Result<JobConfig> {
        // Parse metadata JSON for additional job parameters
        let metadata = &job.metadata;

        // Extract job-specific parameters from metadata
        let model_definition = metadata.get("model_definition")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let dataset_uri = metadata.get("dataset_uri")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let batch_size = metadata.get("batch_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(32) as i32;

        let epochs = metadata.get("epochs")
            .and_then(|v| v.as_i64())
            .unwrap_or(10) as i32;

        // Convert hyperparameters from JSON to proto map
        let mut hyperparameters = HashMap::new();
        if let Some(hyper_obj) = metadata.get("hyperparameters").and_then(|v| v.as_object()) {
            for (key, value) in hyper_obj {
                if let Some(val_str) = value.as_str() {
                    hyperparameters.insert(key.clone(), val_str.to_string());
                } else {
                    hyperparameters.insert(key.clone(), value.to_string());
                }
            }
        }

        // Determine job type (using integer values from proto)
        let job_type = match job.job_type.as_str() {
            "training" => 1,     // JOB_TYPE_TRAINING
            "inference" => 2,    // JOB_TYPE_INFERENCE
            "evaluation" => 3,   // JOB_TYPE_EVALUATION
            "preprocessing" => 4, // JOB_TYPE_PREPROCESSING
            _ => 0,              // JOB_TYPE_UNKNOWN
        };

        // Determine device type (using integer values from proto)
        let required_device = if job.required_gpu {
            2  // DEVICE_CUDA
        } else {
            1  // DEVICE_CPU
        };

        Ok(JobConfig {
            job_id: job.id.to_string(),
            job_type,
            priority: 1, // PRIORITY_NORMAL - TODO: Add priority field to Job model
            model_definition,
            hyperparameters,
            dataset_uri,
            batch_size,
            epochs,
            required_device,
            estimated_memory: (job.required_ram_gb as i64) * 1024 * 1024 * 1024, // Convert GB to bytes
            estimated_duration: job.estimated_duration_seconds as i64,
            payment_amount: job.estimated_cost as f64 / 1_000_000.0, // Convert to CYXWIZ tokens
            payment_address: job.user_wallet.clone(),
            escrow_tx_hash: String::new(), // TODO: Get from payments table
        })
    }
}
