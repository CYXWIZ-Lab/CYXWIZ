use crate::database::{
    models::{Node, NodeStatus as DbNodeStatus},
    queries,
};
use crate::error::ServerError;
use crate::scheduler::JobScheduler;
use chrono::Utc;
use crate::database::DbPool;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};
use uuid::Uuid;

pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}

use crate::pb::{
    node_service_server::NodeService, AssignJobRequest, AssignJobResponse, HeartbeatRequest,
    HeartbeatResponse, RegisterNodeRequest, RegisterNodeResponse, ReportCompletionRequest,
    ReportCompletionResponse, ReportProgressRequest, ReportProgressResponse, StatusCode,
    GetNodeMetricsRequest, GetNodeMetricsResponse,
};

pub struct NodeServiceImpl {
    db_pool: DbPool,
    scheduler: Arc<JobScheduler>,
}

impl crate::pb::node_service_server::NodeServiceImpl {
    pub fn new(db_pool: DbPool, scheduler: Arc<JobScheduler>) -> Self {
        Self { db_pool, scheduler }
    }
}

#[tonic::async_trait]
#[tonic::async_trait]
impl crate::pb::node_service_server::NodeService for NodeServiceImpl {
    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> std::result::Result<Response<RegisterNodeResponse>, Status> {
        let req = request.into_inner();
        let node_info = req.info.ok_or_else(|| Status::invalid_argument("Node info is required"))?;

        info!("Registering node: {}", node_info.name);

        // Check if node already exists (by wallet address)
        match queries::get_node_by_wallet(&self.db_pool, &node_info.wallet_address).await {
            Ok(Some(existing_node)) => {
                warn!("Node with wallet {} already registered", node_info.wallet_address);

                return Ok(Response::new(RegisterNodeResponse {
                    status: StatusCode::StatusSuccess as i32,
                    node_id: existing_node.id.to_string(),
                    session_token: format!("session_{}", existing_node.id),
                    error: None,
                }));
            }
            Ok(None) => {
                // Continue with registration
            }
            Err(e) => {
                error!("Database error: {}", e);
                return Err(Status::internal(format!("Database error: {}", e)));
            }
        }

        // Extract device capabilities
        let devices = &node_info.devices;
        let has_cuda = devices.iter().any(|d| d.device_type == pb::DeviceType::DeviceCuda as i32);
        let has_opencl = devices.iter().any(|d| d.device_type == pb::DeviceType::DeviceOpencl as i32);

        let gpu_device = devices.iter().find(|d| {
            d.device_type == pb::DeviceType::DeviceCuda as i32
                || d.device_type == pb::DeviceType::DeviceOpencl as i32
        });

        let node_id = Uuid::new_v4();
        let node = Node {
            id: node_id,
            wallet_address: node_info.wallet_address.clone(),
            name: node_info.name.clone(),
            status: DbNodeStatus::Online,
            reputation_score: 0.5, // Initial reputation
            stake_amount: (node_info.staked_amount * 1_000_000_000.0) as i64, // Convert to lamports

            cpu_cores: node_info.cpu_cores,
            ram_gb: (node_info.ram_total / (1024 * 1024 * 1024)) as i32,
            gpu_model: gpu_device.as_ref().map(|d| d.device_name.clone()),
            gpu_memory_gb: gpu_device
                .as_ref()
                .map(|d| (d.memory_total / (1024 * 1024 * 1024)) as i32),
            has_cuda,
            has_opencl,

            total_jobs_completed: node_info.total_jobs_completed,
            total_jobs_failed: 0,
            uptime_percentage: node_info.uptime_percentage,
            current_load: 0.0,

            country: None, // TODO: Extract from region
            region: Some(node_info.region.clone()),

            last_heartbeat: Utc::now(),
            registered_at: Utc::now(),
            updated_at: Utc::now(),
        };

        match queries::create_node(&self.db_pool, &node).await {
            Ok(_) => {
                info!("Node {} registered successfully", node_id);

                Ok(Response::new(RegisterNodeResponse {
                    status: StatusCode::StatusSuccess as i32,
                    node_id: node_id.to_string(),
                    session_token: format!("session_{}", node_id),
                    error: None,
                }))
            }
            Err(e) => {
                error!("Failed to register node: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> std::result::Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid node ID: {}", e)))?;

        // Update heartbeat timestamp
        match queries::update_node_heartbeat(&self.db_pool, node_id).await {
            Ok(_) => {
                // Update node status if provided
                if let Some(current_status) = req.current_status {
                    // Calculate current load based on active jobs
                    let load = if req.active_jobs.is_empty() {
                        0.0
                    } else {
                        req.active_jobs.len() as f64 / 4.0 // Assume max 4 concurrent jobs
                    };

                    if let Err(e) = queries::update_node_load(&self.db_pool, node_id, load).await {
                        warn!("Failed to update node load: {}", e);
                    }
                }

                Ok(Response::new(HeartbeatResponse {
                    status: StatusCode::StatusSuccess as i32,
                    keep_alive: true,
                    jobs_to_cancel: vec![],
                    error: None,
                }))
            }
            Err(ServerError::NodeNotFound(msg)) => Err(Status::not_found(msg)),
            Err(e) => Err(Status::internal(format!("Database error: {}", e))),
        }
    }

    async fn assign_job(
        &self,
        _request: Request<AssignJobRequest>,
    ) -> std::result::Result<Response<AssignJobResponse>, Status> {
        // This is called by Central Server to Node, not implemented here
        Err(Status::unimplemented("AssignJob is called by server, not implemented in this service"))
    }

    async fn report_progress(
        &self,
        request: Request<ReportProgressRequest>,
    ) -> std::result::Result<Response<ReportProgressResponse>, Status> {
        let req = request.into_inner();
        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid node ID: {}", e)))?;
        let job_id = Uuid::parse_str(&req.job_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid job ID: {}", e)))?;

        info!("Progress report from node {} for job {}", node_id, job_id);

        // Update job status
        if let Some(status) = req.status {
            if status.status == StatusCode::StatusInProgress as i32 {
                if let Err(e) = queries::start_job(&self.db_pool, job_id).await {
                    error!("Failed to update job status: {}", e);
                }
            }
        }

        // Store metrics (TODO: implement metrics storage)

        Ok(Response::new(ReportProgressResponse {
            status: StatusCode::StatusSuccess as i32,
            continue_job: true,
            error: None,
        }))
    }

    async fn report_completion(
        &self,
        request: Request<ReportCompletionRequest>,
    ) -> std::result::Result<Response<ReportCompletionResponse>, Status> {
        let req = request.into_inner();
        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid node ID: {}", e)))?;
        let job_id = Uuid::parse_str(&req.job_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid job ID: {}", e)))?;

        let result = req.result.ok_or_else(|| Status::invalid_argument("Job result is required"))?;

        info!("Job {} completed by node {}", job_id, node_id);

        // Update job status in database
        let actual_cost = result.total_compute_time * 10_000; // Simple cost calculation
        match self
            .scheduler
            .handle_job_completion(job_id, &result.model_weights_hash, actual_cost)
            .await
        {
            Ok(_) => {
                // Payment distribution is handled in scheduler

                // TODO: Get actual payment tx hash from blockchain
                let payment_tx_hash = format!("tx_{}", job_id);

                Ok(Response::new(ReportCompletionResponse {
                    status: StatusCode::StatusSuccess as i32,
                    payment_released: true,
                    payment_tx_hash,
                    error: None,
                }))
            }
            Err(e) => {
                error!("Failed to complete job: {}", e);
                Err(Status::internal(format!("Failed to complete job: {}", e)))
            }
        }
    }

    async fn get_node_metrics(
        &self,
        request: Request<GetNodeMetricsRequest>,
    ) -> std::result::Result<Response<GetNodeMetricsResponse>, Status> {
        let req = request.into_inner();
        let node_id = Uuid::parse_str(&req.node_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid node ID: {}", e)))?;

        match queries::get_node_metrics_history(&self.db_pool, node_id, 100).await {
            Ok(metrics) => {
                let metric_points: Vec<pb::MetricPoint> = metrics
                    .iter()
                    .map(|m| pb::MetricPoint {
                        timestamp: m.timestamp.timestamp(),
                        cpu_usage: m.cpu_usage_percent,
                        gpu_usage: m.gpu_usage_percent.unwrap_or(0.0),
                        memory_usage: m.ram_usage_percent,
                        network_throughput: (m.network_rx_bytes + m.network_tx_bytes) as f64,
                        power_consumption: 0.0, // TODO: Add power consumption tracking
                    })
                    .collect();

                Ok(Response::new(GetNodeMetricsResponse {
                    metrics: metric_points,
                    error: None,
                }))
            }
            Err(e) => Err(Status::internal(format!("Failed to get metrics: {}", e))),
        }
    }
}
