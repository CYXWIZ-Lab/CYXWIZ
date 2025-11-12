use crate::blockchain::PaymentProcessor;
use crate::database::{
    models::{Job, JobStatus as DbJobStatus, Payment, PaymentStatus},
    queries,
};
use crate::error::ServerError;
use crate::scheduler::{JobMatcher, JobScheduler};
use chrono::Utc;
use crate::database::DbPool;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{error, info};
use uuid::Uuid;


use crate::pb::{
    job_service_server::JobService, CancelJobRequest, CancelJobResponse, GetJobStatusRequest,
    GetJobStatusResponse, ListJobsRequest, ListJobsResponse, SubmitJobRequest,
    SubmitJobResponse, StatusCode,
};

pub struct JobServiceImpl {
    db_pool: DbPool,
    scheduler: Arc<JobScheduler>,
    payment_processor: Arc<PaymentProcessor>,
}

impl crate::pb::job_service_server::JobServiceImpl {
    pub fn new(
        db_pool: DbPool,
        scheduler: Arc<JobScheduler>,
        payment_processor: Arc<PaymentProcessor>,
    ) -> Self {
        Self {
            db_pool,
            scheduler,
            payment_processor,
        }
    }
}

#[tonic::async_trait]
#[tonic::async_trait]
impl crate::pb::job_service_server::JobService for JobServiceImpl {
    type StreamJobUpdatesStream = tokio_stream::wrappers::ReceiverStream<std::result::Result<pb::JobUpdateStream, tonic::Status>>;

    async fn submit_job(
        &self,
        request: Request<SubmitJobRequest>,
    ) -> std::result::Result<Response<SubmitJobResponse>, Status> {
        let req = request.into_inner();
        let config = req.config.ok_or_else(|| Status::invalid_argument("Job config is required"))?;

        info!("Received job submission from user: {}", config.payment_address);

        // Generate job ID
        let job_id = Uuid::new_v4();

        // Estimate cost based on requirements
        let estimated_cost = JobMatcher::estimate_cost(
            config.required_device != 1, // Assume GPU if not CPU
            Some((config.estimated_memory / (1024 * 1024 * 1024)) as i32), // Convert bytes to GB
            8, // Default RAM requirement
            config.estimated_duration as i32,
        );

        // Create job in database
        let job = Job {
            id: job_id,
            user_wallet: config.payment_address.clone(),
            status: DbJobStatus::Pending,
            job_type: format!("{:?}", config.job_type),
            required_gpu: config.required_device != 1,
            required_gpu_memory_gb: Some((config.estimated_memory / (1024 * 1024 * 1024)) as i32),
            required_ram_gb: 8, // Default
            estimated_duration_seconds: config.estimated_duration as i32,
            estimated_cost,
            actual_cost: None,
            assigned_node_id: None,
            retry_count: 0,
            result_hash: None,
            error_message: None,
            metadata: serde_json::json!({
                "model_definition": config.model_definition,
                "dataset_uri": config.dataset_uri,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
            }),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            updated_at: Utc::now(),
        };

        match queries::create_job(&self.db_pool, &job).await {
            Ok(_) => {
                info!("Job {} created successfully", job_id);

                // Create payment escrow on blockchain
                let escrow_result = self
                    .payment_processor
                    .create_job_escrow(job_id, &config.payment_address, estimated_cost as u64)
                    .await;

                match escrow_result {
                    Ok((tx_hash, escrow_account)) => {
                        info!("Escrow created: tx={}, account={}", tx_hash, escrow_account);

                        // Create payment record
                        let (node_reward, platform_fee) =
                            self.payment_processor.calculate_payment_distribution(estimated_cost as u64);

                        let payment = Payment {
                            id: Uuid::new_v4(),
                            job_id,
                            node_id: None,
                            user_wallet: config.payment_address.clone(),
                            node_wallet: None,
                            amount: estimated_cost,
                            platform_fee: platform_fee as i64,
                            node_reward: node_reward as i64,
                            status: PaymentStatus::Locked,
                            escrow_tx_hash: Some(tx_hash.clone()),
                            completion_tx_hash: None,
                            escrow_account: Some(escrow_account),
                            created_at: Utc::now(),
                            locked_at: Some(Utc::now()),
                            completed_at: None,
                        };

                        if let Err(e) = queries::create_payment(&self.db_pool, &payment).await {
                            error!("Failed to create payment record: {}", e);
                        }

                        // Job will be picked up by scheduler

                        Ok(Response::new(SubmitJobResponse {
                            job_id: job_id.to_string(),
                            status: StatusCode::StatusSuccess as i32,
                            assigned_node_id: String::new(),
                            error: None,
                            estimated_start_time: 0,
                        }))
                    }
                    Err(e) => {
                        error!("Failed to create escrow: {}", e);

                        // Delete job from database
                        // TODO: Add delete_job query

                        Ok(Response::new(SubmitJobResponse {
                            job_id: String::new(),
                            status: StatusCode::StatusError as i32,
                            assigned_node_id: String::new(),
                            error: Some(pb::Error {
                                code: 1,
                                message: format!("Failed to create payment escrow: {}", e),
                                details: String::new(),
                            }),
                            estimated_start_time: 0,
                        }))
                    }
                }
            }
            Err(e) => {
                error!("Failed to create job: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }

    async fn get_job_status(
        &self,
        request: Request<GetJobStatusRequest>,
    ) -> std::result::Result<Response<GetJobStatusResponse>, Status> {
        let req = request.into_inner();
        let job_id = Uuid::parse_str(&req.job_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid job ID: {}", e)))?;

        match queries::get_job_by_id(&self.db_pool, job_id).await {
            Ok(job) => {
                let status_code = match job.status {
                    DbJobStatus::Pending => StatusCode::StatusPending,
                    DbJobStatus::Assigned | DbJobStatus::Running => StatusCode::StatusInProgress,
                    DbJobStatus::Completed => StatusCode::StatusCompleted,
                    DbJobStatus::Failed => StatusCode::StatusFailed,
                    DbJobStatus::Cancelled => StatusCode::StatusCancelled,
                };

                let progress = if job.status == DbJobStatus::Completed {
                    1.0
                } else if job.status == DbJobStatus::Running {
                    0.5 // TODO: Get actual progress from metrics
                } else {
                    0.0
                };

                Ok(Response::new(GetJobStatusResponse {
                    status: Some(pb::JobStatus {
                        job_id: job.id.to_string(),
                        status: status_code as i32,
                        progress,
                        current_node_id: job.assigned_node_id.map(|id| id.to_string()).unwrap_or_default(),
                        start_time: job.started_at.map(|t| t.timestamp()).unwrap_or(0),
                        end_time: job.completed_at.map(|t| t.timestamp()).unwrap_or(0),
                        error: job.error_message.map(|msg| pb::Error {
                            code: 1,
                            message: msg,
                            details: String::new(),
                        }),
                        metrics: std::collections::HashMap::new(),
                        current_epoch: 0,
                    }),
                    error: None,
                }))
            }
            Err(ServerError::JobNotFound(msg)) => Err(Status::not_found(msg)),
            Err(e) => Err(Status::internal(format!("Database error: {}", e))),
        }
    }

    async fn cancel_job(
        &self,
        request: Request<CancelJobRequest>,
    ) -> std::result::Result<Response<CancelJobResponse>, Status> {
        let req = request.into_inner();
        let job_id = Uuid::parse_str(&req.job_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid job ID: {}", e)))?;

        info!("Cancelling job {}: {}", job_id, req.reason);

        match queries::get_job_by_id(&self.db_pool, job_id).await {
            Ok(job) => {
                // Can only cancel pending or assigned jobs
                if matches!(job.status, DbJobStatus::Completed | DbJobStatus::Failed | DbJobStatus::Cancelled) {
                    return Err(Status::failed_precondition("Job cannot be cancelled"));
                }

                // Update job status to cancelled
                // TODO: Add cancel_job query

                // Issue refund
                let refund_result = self
                    .payment_processor
                    .refund_job(job_id, &job.user_wallet, job.estimated_cost as u64)
                    .await;

                match refund_result {
                    Ok(tx_hash) => {
                        info!("Refund issued: {}", tx_hash);

                        Ok(Response::new(CancelJobResponse {
                            status: StatusCode::StatusCancelled as i32,
                            refund_issued: true,
                            error: None,
                        }))
                    }
                    Err(e) => {
                        error!("Failed to issue refund: {}", e);

                        Ok(Response::new(CancelJobResponse {
                            status: StatusCode::StatusError as i32,
                            refund_issued: false,
                            error: Some(pb::Error {
                                code: 1,
                                message: format!("Failed to issue refund: {}", e),
                                details: String::new(),
                            }),
                        }))
                    }
                }
            }
            Err(ServerError::JobNotFound(msg)) => Err(Status::not_found(msg)),
            Err(e) => Err(Status::internal(format!("Database error: {}", e))),
        }
    }

    async fn stream_job_updates(
        &self,
        _request: Request<GetJobStatusRequest>,
    ) -> std::result::Result<Response<Self::StreamJobUpdatesStream>, Status> {
        // TODO: Implement streaming
        Err(Status::unimplemented("Streaming not implemented yet"))
    }

    async fn list_jobs(
        &self,
        _request: Request<ListJobsRequest>,
    ) -> std::result::Result<Response<ListJobsResponse>, Status> {
        // TODO: Implement job listing
        Err(Status::unimplemented("Job listing not implemented yet"))
    }
}
