use crate::cache::RedisCache;
use crate::config::SchedulerConfig;
use crate::database::{
    models::{Job, Node},
    queries,
};
use crate::error::Result;
use crate::scheduler::matcher::JobMatcher;
use crate::database::DbPool;
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

                // TODO: Notify node via gRPC about new job assignment

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
        if job.retry_count < self.config.max_retries {
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
}
