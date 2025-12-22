use crate::auth::JWTManager;
use crate::blockchain::PaymentProcessor;
use crate::database::DbPool;
use crate::scheduler::JobScheduler;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::pb::{
    job_service_server::JobService, CancelJobRequest, CancelJobResponse, GetJobStatusRequest,
    GetJobStatusResponse, ListJobsRequest, ListJobsResponse, SubmitJobRequest,
    SubmitJobResponse, StatusCode, Error, JobStatus, JobUpdateStream, DeleteJobRequest, DeleteJobResponse,
};

/// JobService - DEPRECATED
///
/// Jobs now go via P2P directly between Engine and Server Node.
/// The Central Server only handles:
/// - Node reservations (ReservationService)
/// - Payment escrow (PaymentProcessor)
///
/// This service is kept for backward compatibility but returns
/// "not implemented" for all operations.
pub struct JobServiceImpl {
    #[allow(dead_code)]
    db_pool: DbPool,
    #[allow(dead_code)]
    scheduler: Arc<JobScheduler>,
    #[allow(dead_code)]
    payment_processor: Arc<PaymentProcessor>,
    #[allow(dead_code)]
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

    /// DEPRECATED: Jobs now go via P2P. Use ReservationService.ReserveNode instead.
    async fn submit_job(
        &self,
        _request: Request<SubmitJobRequest>,
    ) -> std::result::Result<Response<SubmitJobResponse>, Status> {
        info!("submit_job called - DEPRECATED: Use ReservationService.ReserveNode for P2P training");
        Ok(Response::new(SubmitJobResponse {
            job_id: String::new(),
            status: StatusCode::StatusError as i32,
            node_assignment: None,
            assigned_node_id: String::new(),
            error: Some(Error {
                code: 100,
                message: "DEPRECATED: Job submission via Central Server is no longer supported. Use ReservationService.ReserveNode to reserve a node, then connect via P2P.".to_string(),
                details: String::new(),
            }),
            estimated_start_time: 0,
        }))
    }

    /// DEPRECATED: Job status is tracked via P2P connection to Server Node.
    async fn get_job_status(
        &self,
        request: Request<GetJobStatusRequest>,
    ) -> std::result::Result<Response<GetJobStatusResponse>, Status> {
        let req = request.into_inner();
        info!("get_job_status called for {} - DEPRECATED: Track job status via P2P", req.job_id);
        Ok(Response::new(GetJobStatusResponse {
            status: Some(JobStatus {
                job_id: req.job_id,
                status: StatusCode::StatusError as i32,
                progress: 0.0,
                current_node_id: String::new(),
                start_time: 0,
                end_time: 0,
                error: Some(Error {
                    code: 100,
                    message: "DEPRECATED: Job status tracking via Central Server is no longer supported. Track status via P2P connection to Server Node.".to_string(),
                    details: String::new(),
                }),
                metrics: std::collections::HashMap::new(),
                current_epoch: 0,
            }),
            node_assignment: None,
            error: None,
        }))
    }

    /// DEPRECATED: Cancel jobs via P2P connection to Server Node.
    async fn cancel_job(
        &self,
        request: Request<CancelJobRequest>,
    ) -> std::result::Result<Response<CancelJobResponse>, Status> {
        let req = request.into_inner();
        info!("cancel_job called for {} - DEPRECATED: Cancel via P2P", req.job_id);
        Ok(Response::new(CancelJobResponse {
            status: StatusCode::StatusError as i32,
            refund_issued: false,
            error: Some(Error {
                code: 100,
                message: "DEPRECATED: Job cancellation via Central Server is no longer supported. Cancel via P2P connection to Server Node.".to_string(),
                details: String::new(),
            }),
        }))
    }

    async fn stream_job_updates(
        &self,
        _request: Request<GetJobStatusRequest>,
    ) -> std::result::Result<Response<Self::StreamJobUpdatesStream>, Status> {
        Err(Status::unimplemented("DEPRECATED: Job streaming via Central Server is no longer supported. Use P2P connection."))
    }

    /// DEPRECATED: Job history is no longer tracked by Central Server.
    async fn list_jobs(
        &self,
        _request: Request<ListJobsRequest>,
    ) -> std::result::Result<Response<ListJobsResponse>, Status> {
        info!("list_jobs called - DEPRECATED: Job history not tracked");
        Ok(Response::new(ListJobsResponse {
            jobs: vec![],
            next_page_token: String::new(),
            total_count: 0,
        }))
    }

    /// DEPRECATED: Jobs are not stored in database.
    async fn delete_job(
        &self,
        request: Request<DeleteJobRequest>,
    ) -> std::result::Result<Response<DeleteJobResponse>, Status> {
        let req = request.into_inner();
        info!("delete_job called for {} - DEPRECATED: Jobs not stored", req.job_id);
        Ok(Response::new(DeleteJobResponse {
            status: StatusCode::StatusSuccess as i32,
            error: None,
        }))
    }
}
