use crate::auth::JWTManager;
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
    SubmitJobResponse, StatusCode, Error, JobStatus, JobUpdateStream, NodeAssignment, DeleteJobRequest, DeleteJobResponse,
};

pub struct JobServiceImpl {
    db_pool: DbPool,
    scheduler: Arc<JobScheduler>,
    payment_processor: Arc<PaymentProcessor>,
    jwt_manager: Arc<JWTManager>,
}

impl JobServiceImpl {
    pub fn new(
        db_pool: DbPool,
        scheduler: Arc<JobScheduler>,
        payment_processor: Arc<PaymentProcessor>,
        jwt_manager: Arc<JWTManager>,
    ) -> Self {
        Self {
            db_pool,
            scheduler,
            payment_processor,
            jwt_manager,
        }
    }
}

#[tonic::async_trait]
impl JobService for JobServiceImpl {
    type StreamJobUpdatesStream = tokio_stream::wrappers::ReceiverStream<std::result::Result<JobUpdateStream, tonic::Status>>;

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
            id: job_id.to_string(),
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
                let job_id_u64 = u64::from_le_bytes(job_id.as_bytes()[0..8].try_into().unwrap());
                let escrow_result = self
                    .payment_processor
                    .create_job_escrow(job_id_u64, &config.payment_address, &config.payment_address, estimated_cost as u64)
                    .await;

                match escrow_result {
                    Ok((tx_hash, escrow_account)) => {
                        info!("Escrow created: tx={}, account={}", tx_hash, escrow_account);

                        // Create payment record
                        let (node_reward, platform_fee) =
                            self.payment_processor.calculate_payment_distribution(estimated_cost as u64);

                        let payment = Payment {
                            id: Uuid::new_v4().to_string(),
                            job_id: job_id.to_string(),
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

                        Ok(Response::new(SubmitJobResponse {
                            job_id: job_id.to_string(),
                            status: StatusCode::StatusSuccess as i32,
                            node_assignment: None,
                            assigned_node_id: String::new(),
                            error: None,
                            estimated_start_time: 0,
                        }))
                    }
                    Err(e) => {
                        error!("Failed to create escrow: {}", e);
                        Ok(Response::new(SubmitJobResponse {
                            job_id: String::new(),
                            status: StatusCode::StatusError as i32,
                            node_assignment: None,
                            assigned_node_id: String::new(),
                            error: Some(Error {
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

        match queries::get_job_by_id(&self.db_pool, job_id.to_string()).await {
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

                // Generate NodeAssignment if job has been assigned to a node
                let node_assignment = if let Some(ref node_id_str) = job.assigned_node_id {
                    // Parse node_id string to Uuid for JWT generation
                    let node_id_uuid = match Uuid::parse_str(node_id_str) {
                        Ok(id) => id,
                        Err(e) => {
                            error!("Invalid node ID format in database: {}", e);
                            return Err(Status::internal("Invalid node ID in database"));
                        }
                    };

                    // Get node details from database
                    match queries::get_node_by_id(&self.db_pool, node_id_str.clone()).await {
                        Ok(node) => {
                            // Generate JWT token for P2P authentication
                            match self.jwt_manager.generate_p2p_token(
                                &job.user_wallet,
                                job_id,
                                node_id_uuid,
                                3600,  // 1 hour expiration
                            ) {
                                Ok(token) => {
                                    let node_endpoint = format!("{}:{}", node.ip_address, node.port);
                                    info!("Generated P2P token for job {} -> node {} ({})",
                                          job_id, node_id_str, node_endpoint);

                                    Some(NodeAssignment {
                                        node_id: node_id_str.clone(),
                                        node_endpoint,
                                        auth_token: token,
                                        token_expires_at: Utc::now().timestamp() + 3600,
                                        node_public_key: String::new(),  // TODO: Add TLS support
                                    })
                                }
                                Err(e) => {
                                    error!("Failed to generate JWT for job {}: {}", job_id, e);
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to get node details for node {}: {}", node_id_str, e);
                            None
                        }
                    }
                } else {
                    None
                };

                Ok(Response::new(GetJobStatusResponse {
                    status: Some(JobStatus {
                        job_id: job.id.to_string(),
                        status: status_code as i32,
                        progress,
                        current_node_id: job.assigned_node_id.map(|id| id.to_string()).unwrap_or_default(),
                        start_time: job.started_at.map(|t| t.timestamp()).unwrap_or(0),
                        end_time: job.completed_at.map(|t| t.timestamp()).unwrap_or(0),
                        error: job.error_message.map(|msg| Error {
                            code: 1,
                            message: msg,
                            details: String::new(),
                        }),
                        metrics: std::collections::HashMap::new(),
                        current_epoch: 0,
                    }),
                    node_assignment,  // NEW: P2P connection info
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

        match queries::get_job_by_id(&self.db_pool, job_id.to_string()).await {
            Ok(job) => {
                // Can only cancel pending or assigned jobs
                if matches!(job.status, DbJobStatus::Completed | DbJobStatus::Failed | DbJobStatus::Cancelled) {
                    return Err(Status::failed_precondition("Job cannot be cancelled"));
                }

                // Update job status to cancelled
                // TODO: Add cancel_job query to update DB status

                // Issue refund on blockchain
                let job_id_u64 = u64::from_le_bytes(job_id.as_bytes()[0..8].try_into().unwrap());
                let refund_result = self
                    .payment_processor
                    .refund_job(job_id_u64, &job.user_wallet)
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
                            error: Some(Error {
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
        request: Request<ListJobsRequest>,
    ) -> std::result::Result<Response<ListJobsResponse>, Status> {
        let req = request.into_inner();

        let page_size = if req.page_size > 0 { req.page_size as i64 } else { 100 };
        let offset = 0; // TODO: Implement pagination with page_token

        match queries::list_all_jobs(&self.db_pool, page_size, offset).await {
            Ok(db_jobs) => {
                let mut jobs = Vec::new();

                for job in db_jobs {
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
                        0.5
                    } else {
                        0.0
                    };

                    jobs.push(JobStatus {
                        job_id: job.id.to_string(),
                        status: status_code as i32,
                        progress,
                        current_node_id: job.assigned_node_id.map(|id| id.to_string()).unwrap_or_default(),
                        start_time: job.started_at.map(|t| t.timestamp()).unwrap_or(0),
                        end_time: job.completed_at.map(|t| t.timestamp()).unwrap_or(0),
                        error: job.error_message.map(|msg| Error {
                            code: 1,
                            message: msg,
                            details: String::new(),
                        }),
                        metrics: std::collections::HashMap::new(),
                        current_epoch: 0,
                    });
                }

                let job_count = jobs.len() as i32;
                info!("Listed {} jobs", job_count);

                Ok(Response::new(ListJobsResponse {
                    jobs,
                    next_page_token: String::new(),
                    total_count: job_count,
                }))
            }
            Err(e) => {
                error!("Failed to list jobs: {}", e);
                Err(Status::internal(format!("Failed to list jobs: {}", e)))
            }
        }
    }

    async fn delete_job(
        &self,
        request: Request<DeleteJobRequest>,
    ) -> std::result::Result<Response<DeleteJobResponse>, Status> {
        let req = request.into_inner();
        let job_id = Uuid::parse_str(&req.job_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid job ID: {}", e)))?;

        info!("Deleting job {}", job_id);

        // Check if job exists and is in a deletable state
        match queries::get_job_by_id(&self.db_pool, job_id.to_string()).await {
            Ok(job) => {
                // Only allow deletion of completed, failed, or cancelled jobs
                if matches!(job.status, DbJobStatus::Pending | DbJobStatus::Assigned | DbJobStatus::Running) {
                    return Ok(Response::new(DeleteJobResponse {
                        status: StatusCode::StatusError as i32,
                        error: Some(Error {
                            code: 2,
                            message: "Cannot delete active job. Cancel it first.".to_string(),
                            details: String::new(),
                        }),
                    }));
                }

                // Delete job from database
                match queries::delete_job(&self.db_pool, job_id.to_string()).await {
                    Ok(_) => {
                        info!("Job {} deleted successfully", job_id);
                        Ok(Response::new(DeleteJobResponse {
                            status: StatusCode::StatusSuccess as i32,
                            error: None,
                        }))
                    }
                    Err(e) => {
                        error!("Failed to delete job {}: {}", job_id, e);
                        Ok(Response::new(DeleteJobResponse {
                            status: StatusCode::StatusError as i32,
                            error: Some(Error {
                                code: 1,
                                message: format!("Failed to delete job: {}", e),
                                details: String::new(),
                            }),
                        }))
                    }
                }
            }
            Err(ServerError::JobNotFound(_)) => {
                Ok(Response::new(DeleteJobResponse {
                    status: StatusCode::StatusSuccess as i32,  // Already deleted
                    error: None,
                }))
            }
            Err(e) => Err(Status::internal(format!("Database error: {}", e))),
        }
    }
}
