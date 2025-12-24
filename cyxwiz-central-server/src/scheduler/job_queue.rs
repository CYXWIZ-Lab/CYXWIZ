use crate::blockchain::PaymentProcessor;
use crate::cache::RedisCache;
use crate::config::SchedulerConfig;
use crate::database::{
    models::{Job, Node, DbId},
    queries,
};
use crate::error::Result;
use crate::scheduler::matcher::JobMatcher;
use crate::database::DbPool;
use crate::pb::{node_service_client::NodeServiceClient, AssignJobRequest, JobConfig};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

pub struct JobScheduler {
    db_pool: DbPool,
    cache: Arc<RwLock<RedisCache>>,
    config: SchedulerConfig,
    payment_processor: Option<Arc<PaymentProcessor>>,
}

impl JobScheduler {
    pub fn new(db_pool: DbPool, cache: RedisCache, config: SchedulerConfig) -> Self {
        Self {
            db_pool,
            cache: Arc::new(RwLock::new(cache)),
            config,
            payment_processor: None,
        }
    }

    /// Create scheduler with payment processor for blockchain integration
    pub fn with_payment_processor(
        db_pool: DbPool,
        cache: RedisCache,
        config: SchedulerConfig,
        payment_processor: Arc<PaymentProcessor>,
    ) -> Self {
        Self {
            db_pool,
            cache: Arc::new(RwLock::new(cache)),
            config,
            payment_processor: Some(payment_processor),
        }
    }

    /// Check if the scheduler is running
    /// Returns true if the scheduler has been initialized
    pub fn is_running(&self) -> bool {
        // The scheduler is considered running if it has been created
        // The run() method is started in a separate task
        true
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

        info!("========================================");
        info!("[P2P WORKFLOW] Central Server: Found {} pending jobs to process", pending_jobs.len());

        // Get available nodes
        let available_nodes = queries::list_available_nodes(&self.db_pool).await?;

        if available_nodes.is_empty() {
            warn!("[P2P WORKFLOW] No available nodes for job assignment - jobs will wait");
            return Ok(());
        }

        info!("[P2P WORKFLOW] Found {} available nodes for matching", available_nodes.len());
        for node in &available_nodes {
            info!("  Node {}: GPU={:?}, RAM={}GB, Load={:.1}%",
                  node.id, node.gpu_model, node.ram_gb, node.current_load * 100.0);
        }

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
        info!("[P2P WORKFLOW] Matching job {} (GPU: {}, VRAM: {:?}GB, RAM: {}GB)",
              job.id, job.required_gpu, job.required_gpu_memory_gb, job.required_ram_gb);

        // Find best node using matcher
        let best_node = JobMatcher::find_best_node(job, available_nodes);

        match best_node {
            Some(node) => {
                info!("========================================");
                info!("[P2P WORKFLOW] STEP 2: Assigning job to node!");
                info!("  Job ID: {}", job.id);
                info!("  Node ID: {}", node.id);
                info!("  Node GPU: {:?}", node.gpu_model);
                info!("  Node Reputation: {:.2}", node.reputation_score);
                info!("  Node Load: {:.1}%", node.current_load * 100.0);
                info!("========================================");

                // Update database
                queries::assign_job_to_node(&self.db_pool, job.id.clone(), node.id.clone()).await?;
                info!("[P2P WORKFLOW] Database updated: job {} assigned to node {}", job.id, node.id);

                // Cache assignment
                let mut cache = self.cache.write().await;
                cache.cache_job_assignment(job.id.clone(), node.id.clone(), 3600).await?;

                // Notify node via gRPC about new job assignment
                info!("[P2P WORKFLOW] Sending job config to Server Node via gRPC...");
                if let Err(e) = self.send_job_to_node(node, job).await {
                    error!("[P2P WORKFLOW] Failed to send job {} to node {}: {}", job.id, node.id, e);
                    // TODO: Implement update_job_status to mark job as failed/pending
                    // For now, job remains in "assigned" state and will be retried
                    return Err(e);
                }

                info!("[P2P WORKFLOW] Job {} successfully sent to node {}", job.id, node.id);
                info!("[P2P WORKFLOW] Engine will poll GetJobStatus and see NodeAssignment");

                Ok(())
            }
            None => {
                // Log why each node was rejected
                for node in available_nodes {
                    if job.required_gpu && node.gpu_model.is_none() {
                        warn!(
                            "Job {} requires GPU but node {} has no GPU registered (gpu_model: None)",
                            job.id, node.id
                        );
                    } else if let (Some(required_vram), Some(available_vram)) = (job.required_gpu_memory_gb, node.gpu_memory_gb) {
                        if required_vram > available_vram {
                            warn!(
                                "Job {} requires {} GB VRAM but node {} only has {} GB",
                                job.id, required_vram, node.id, available_vram
                            );
                        }
                    } else if job.required_ram_gb > node.ram_gb {
                        warn!(
                            "Job {} requires {} GB RAM but node {} only has {} GB",
                            job.id, job.required_ram_gb, node.id, node.ram_gb
                        );
                    }
                }
                warn!("No suitable node found for job {} (requires_gpu: {}, gpu_mem: {:?}, ram: {})",
                    job.id, job.required_gpu, job.required_gpu_memory_gb, job.required_ram_gb);
                Ok(())
            }
        }
    }

    /// Handle job completion
    pub async fn handle_job_completion(
        &self,
        job_id: DbId,
        result_hash: &str,
        actual_cost: i64,
    ) -> Result<()> {
        info!("========================================");
        info!("[P2P WORKFLOW] STEP 6: Job completed! Processing payment...");
        info!("  Job ID: {}", job_id);
        info!("  Result Hash: {}", result_hash);
        info!("  Actual Cost: {} lamports", actual_cost);
        info!("========================================");

        // Update job status in database
        queries::complete_job(&self.db_pool, job_id.clone(), result_hash, actual_cost).await?;

        // Get job details
        let job = queries::get_job_by_id(&self.db_pool, job_id.clone()).await?;

        // Update node metrics and trigger payment
        if let Some(ref node_id) = job.assigned_node_id {
            // Update node load (decrement)
            queries::update_node_load(&self.db_pool, node_id.clone(), 0.0).await?;

            // Trigger payment distribution via blockchain
            if let Some(ref payment_processor) = self.payment_processor {
                // Get node wallet address from database
                match queries::get_node_by_id(&self.db_pool, node_id.clone()).await {
                    Ok(node) => {
                        // Convert job_id to u64 for blockchain
                        let job_id_u64 = job_id_to_u64(&job_id);

                        info!(
                            "Releasing payment for job {} to node wallet {}",
                            job_id, node.wallet_address
                        );

                        match payment_processor
                            .complete_job_payment(job_id_u64, &node.wallet_address)
                            .await
                        {
                            Ok(signature) => {
                                info!(
                                    "Payment released for job {}: signature={}",
                                    job_id, signature
                                );
                                // TODO: Store payment signature in database
                            }
                            Err(e) => {
                                error!("Failed to release payment for job {}: {}", job_id, e);
                                // Payment failure doesn't fail the job completion
                                // Manual intervention may be needed
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to get node {} for payment: {}", node_id, e);
                    }
                }
            } else {
                warn!("Payment processor not available - skipping payment for job {}", job_id);
            }
        }

        Ok(())
    }

    /// Handle job failure
    pub async fn handle_job_failure(&self, job_id: DbId, error_message: &str) -> Result<()> {
        error!("Job {} failed: {}", job_id, error_message);

        // Mark job as failed
        queries::fail_job(&self.db_pool, job_id.clone(), error_message).await?;

        // Get job details
        let job = queries::get_job_by_id(&self.db_pool, job_id.clone()).await?;

        // If retry count is below max, re-queue the job
        if job.retry_count < self.config.max_retries as i32 {
            info!("Re-queuing job {} (retry {}/{})", job_id, job.retry_count, self.config.max_retries);

            // Add back to pending queue
            let mut cache = self.cache.write().await;
            cache.push_job(job_id).await?;
        } else {
            warn!("Job {} exceeded max retries, triggering refund", job_id);

            // Trigger refund via blockchain
            if let Some(ref payment_processor) = self.payment_processor {
                let job_id_u64 = job_id_to_u64(&job_id);

                info!("Refunding escrow for job {} to user wallet {}", job_id, job.user_wallet);

                match payment_processor.refund_job(job_id_u64, &job.user_wallet).await {
                    Ok(signature) => {
                        info!("Refund processed for job {}: signature={}", job_id, signature);
                        // TODO: Store refund signature in database
                    }
                    Err(e) => {
                        error!("Failed to process refund for job {}: {}", job_id, e);
                        // Refund failure requires manual intervention
                    }
                }
            } else {
                warn!("Payment processor not available - cannot refund job {}", job_id);
            }
        }

        // Update node reputation (penalize for failure)
        if let Some(ref node_id) = job.assigned_node_id {
            // Decrement node reputation
            if let Err(e) = queries::update_node_reputation(&self.db_pool, node_id.clone(), -0.1).await {
                warn!("Failed to update reputation for node {}: {}", node_id, e);
            }
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
        // Node registers with P2P port (50052), but NodeService is on P2P port + 3 (50055)
        // TODO: Add node_service_port field to database schema for cleaner solution
        let node_service_port = node.port + 3;
        let node_endpoint = format!("http://{}:{}", node.ip_address, node_service_port);

        info!("Connecting to node {} at {} (NodeService)", node.id, node_endpoint);

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
            "training" | "1" => 1,     // JOB_TYPE_TRAINING
            "inference" | "2" => 2,    // JOB_TYPE_INFERENCE
            "evaluation" | "3" => 3,   // JOB_TYPE_EVALUATION
            "preprocessing" | "4" => 4, // JOB_TYPE_PREPROCESSING
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
            reservation_duration_minutes: (job.estimated_duration_seconds / 60) as i32, // Convert seconds to minutes
        })
    }
}

/// Convert DbId to u64 for blockchain operations
/// Uses first 8 bytes of UUID to create a unique identifier
fn job_id_to_u64(job_id: &DbId) -> u64 {
    #[cfg(feature = "sqlite-compat")]
    {
        // DbId is String - try to parse as UUID
        use std::str::FromStr;
        if let Ok(uuid) = uuid::Uuid::from_str(job_id) {
            let bytes = uuid.as_bytes();
            u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0u8; 8]))
        } else {
            // Fallback: hash the string
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            job_id.hash(&mut hasher);
            hasher.finish()
        }
    }

    #[cfg(not(feature = "sqlite-compat"))]
    {
        // DbId is already Uuid - use directly
        let bytes = job_id.as_bytes();
        u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0u8; 8]))
    }
}
