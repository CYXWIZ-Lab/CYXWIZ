use crate::database::{queries, DbPool};
use crate::scheduler::JobScheduler;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};
use uuid::Uuid;

pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}

use crate::pb::{
    job_status_service_server::JobStatusService,
    UpdateJobStatusRequest, UpdateJobStatusResponse,
    ReportJobResultRequest, ReportJobResultResponse,
    StatusCode,
};

pub struct JobStatusServiceImpl {
    db_pool: DbPool,
    scheduler: Arc<JobScheduler>,
}

impl JobStatusServiceImpl {
    pub fn new(db_pool: DbPool, scheduler: Arc<JobScheduler>) -> Self {
        Self { db_pool, scheduler }
    }
}

#[tonic::async_trait]
impl JobStatusService for JobStatusServiceImpl {
    /// Handle job progress updates from Server Nodes
    async fn update_job_status(
        &self,
        request: Request<UpdateJobStatusRequest>,
    ) -> std::result::Result<Response<UpdateJobStatusResponse>, Status> {
        let req = request.into_inner();

        info!(
            "Received status update for job {} from node {}: {:.1}% complete",
            req.job_id, req.node_id, req.progress * 100.0
        );

        // Parse job ID
        let job_id_uuid = Uuid::parse_str(&req.job_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid job ID: {}", e)))?;

        #[cfg(feature = "sqlite-compat")]
        let job_id = req.job_id.clone();

        #[cfg(not(feature = "sqlite-compat"))]
        let job_id = job_id_uuid;

        // Retrieve job from database to verify it exists
        let job = match queries::get_job_by_id(&self.db_pool, job_id.clone()).await {
            Ok(job) => job,
            Err(e) => {
                error!("Failed to retrieve job {}: {}", req.job_id, e);
                return Err(Status::not_found(format!("Job not found: {}", req.job_id)));
            }
        };

        // Verify the node ID matches the assigned node
        if let Some(assigned_node_id) = &job.assigned_node_id {
            #[cfg(feature = "sqlite-compat")]
            let node_matches = assigned_node_id == &req.node_id;

            #[cfg(not(feature = "sqlite-compat"))]
            let node_matches = {
                match Uuid::parse_str(&req.node_id) {
                    Ok(node_uuid) => assigned_node_id == &node_uuid,
                    Err(_) => false,
                }
            };

            if !node_matches {
                warn!(
                    "Node ID mismatch for job {}: expected {}, got {}",
                    req.job_id, assigned_node_id, req.node_id
                );
                // For now, allow mismatches (for testing compatibility)
                // TODO: Enforce strict node ID matching in production
            }
        }

        // Update job progress and status in database
        let status_str = match req.status {
            x if x == StatusCode::StatusInProgress as i32 => "running",
            x if x == StatusCode::StatusSuccess as i32 => "completed",
            x if x == StatusCode::StatusFailed as i32 => "failed",
            _ => "assigned", // Unknown status, keep current
        };

        // Update job status in database
        if let Err(e) = queries::update_job_status(&self.db_pool, job_id.clone(), status_str).await {
            error!("Failed to update job {} status to {}: {}", req.job_id, status_str, e);
            return Err(Status::internal(format!("Failed to update job status: {}", e)));
        }

        // TODO: Update job progress, current_epoch, and metrics in database
        // This requires adding these fields to the jobs table first
        // For now, just log the progress
        info!(
            "Job {} progress: epoch {}/{} - metrics: {:?}",
            job_id,
            req.current_epoch,
            "?", // We don't have total epochs in the request
            req.metrics
        );

        // TODO: Store metrics in a separate metrics table or in job metadata JSON

        // Determine if job should continue
        let should_continue = true; // TODO: Implement cancellation logic

        Ok(Response::new(UpdateJobStatusResponse {
            status: StatusCode::StatusSuccess as i32,
            should_continue,
            error: None,
        }))
    }

    /// Handle final job results from Server Nodes
    async fn report_job_result(
        &self,
        request: Request<ReportJobResultRequest>,
    ) -> std::result::Result<Response<ReportJobResultResponse>, Status> {
        let req = request.into_inner();

        info!(
            "Received job result for job {} from node {}: status={}",
            req.job_id,
            req.node_id,
            if req.final_status == StatusCode::StatusSuccess as i32 {
                "SUCCESS"
            } else {
                "FAILED"
            }
        );

        // Parse job ID
        let job_id_uuid = Uuid::parse_str(&req.job_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid job ID: {}", e)))?;

        #[cfg(feature = "sqlite-compat")]
        let job_id = req.job_id.clone();

        #[cfg(not(feature = "sqlite-compat"))]
        let job_id = job_id_uuid;

        // Retrieve job to verify it exists
        let job = match queries::get_job_by_id(&self.db_pool, job_id.clone()).await {
            Ok(job) => job,
            Err(e) => {
                error!("Failed to retrieve job {}: {}", req.job_id, e);
                return Err(Status::not_found(format!("Job not found: {}", req.job_id)));
            }
        };

        // Determine final status
        let final_status = if req.final_status == StatusCode::StatusSuccess as i32 {
            "completed"
        } else {
            "failed"
        };

        // Log final metrics
        if !req.final_metrics.is_empty() {
            info!("Job {} final metrics: {:?}", req.job_id, req.final_metrics);
        }

        // Log model output if present
        if !req.model_weights_uri.is_empty() {
            info!(
                "Job {} produced model: {} (size: {} bytes, hash: {})",
                req.job_id, req.model_weights_uri, req.model_size, req.model_weights_hash
            );
        }

        // Log compute time
        info!(
            "Job {} total compute time: {:.2}s",
            req.job_id,
            req.total_compute_time as f64 / 1000.0
        );

        // Log error if job failed
        if !req.error_message.is_empty() {
            error!("Job {} failed with error: {}", req.job_id, req.error_message);
        }

        // Update job status in database
        match queries::update_job_status(&self.db_pool, job_id.clone(), final_status).await {
            Ok(_) => {
                info!("Updated job {} status to: {}", req.job_id, final_status);
            }
            Err(e) => {
                error!("Failed to update job {} status: {}", req.job_id, e);
                return Err(Status::internal(format!("Database error: {}", e)));
            }
        }

        // TODO: Store final results in database:
        // - model_weights_uri
        // - final_metrics
        // - total_compute_time
        // - model_weights_hash for verification

        // Trigger payment release for successful jobs or handle failure
        if final_status == "completed" {
            // Calculate actual cost based on compute time
            // For now, use a simple formula: compute_time_ms * rate_per_ms
            let actual_cost = (req.total_compute_time as i64) / 1000; // Simple: 1 unit per second

            if let Err(e) = self.scheduler.handle_job_completion(
                job_id.clone(),
                &req.model_weights_hash,
                actual_cost,
            ).await {
                error!("Failed to handle job completion payment for {}: {}", req.job_id, e);
                // Don't fail the RPC - job is complete, payment issue is logged
            }
        } else {
            // Job failed - let scheduler handle retry logic and potential refund
            if let Err(e) = self.scheduler.handle_job_failure(
                job_id.clone(),
                &req.error_message,
            ).await {
                error!("Failed to handle job failure for {}: {}", req.job_id, e);
                // Don't fail the RPC - failure is logged
            }
        }

        Ok(Response::new(ReportJobResultResponse {
            status: StatusCode::StatusSuccess as i32,
            error: None,
        }))
    }
}
