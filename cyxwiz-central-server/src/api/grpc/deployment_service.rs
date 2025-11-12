use crate::database::{
    models::{Deployment, DeploymentStatus, DeploymentType, Model},
    queries, DbPool,
};
use crate::error::ServerError;
use crate::scheduler::JobMatcher;
use chrono::Utc;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};
use uuid::Uuid;

// Import generated proto types
pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}

use pb::{
    deployment_service_server::DeploymentService, CreateDeploymentRequest, CreateDeploymentResponse,
    DeleteDeploymentRequest, DeleteDeploymentResponse, GetDeploymentMetricsRequest,
    GetDeploymentMetricsResponse, GetDeploymentRequest, GetDeploymentResponse,
    ListDeploymentsRequest, ListDeploymentsResponse, StopDeploymentRequest, StopDeploymentResponse,
    StatusCode, Deployment as PbDeployment, DeploymentStatus as PbDeploymentStatus,
    DeploymentType as PbDeploymentType,
};

pub struct DeploymentServiceImpl {
    db_pool: DbPool,
}

impl DeploymentServiceImpl {
    pub fn new(db_pool: DbPool) -> Self {
        Self { db_pool }
    }

    /// Convert database deployment model to protobuf
    fn deployment_to_pb(&self, deployment: &Deployment, model: &Model) -> PbDeployment {
        // Convert deployment type
        let deployment_type = match deployment.deployment_type {
            DeploymentType::Local => PbDeploymentType::DeploymentTypeLocal as i32,
            DeploymentType::Network => PbDeploymentType::DeploymentTypeNetwork as i32,
        };

        // Convert deployment status
        let status = match deployment.status {
            DeploymentStatus::Pending => PbDeploymentStatus::DeploymentStatusPending as i32,
            DeploymentStatus::Provisioning => PbDeploymentStatus::DeploymentStatusProvisioning as i32,
            DeploymentStatus::Loading => PbDeploymentStatus::DeploymentStatusLoading as i32,
            DeploymentStatus::Ready => PbDeploymentStatus::DeploymentStatusReady as i32,
            DeploymentStatus::Running => PbDeploymentStatus::DeploymentStatusRunning as i32,
            DeploymentStatus::Stopped => PbDeploymentStatus::DeploymentStatusStopped as i32,
            DeploymentStatus::Failed => PbDeploymentStatus::DeploymentStatusFailed as i32,
            DeploymentStatus::Terminated => PbDeploymentStatus::DeploymentStatusTerminated as i32,
        };

        PbDeployment {
            deployment_id: deployment.id.to_string(),
            config: None, // Config is only needed on creation
            status,
            assigned_node: None, // TODO: Fetch node info if assigned
            status_message: deployment.status_message.clone().unwrap_or_default(),
            created_at: deployment.created_at.timestamp(),
            started_at: deployment.started_at.map(|t| t.timestamp()).unwrap_or(0),
            stopped_at: deployment.stopped_at.map(|t| t.timestamp()).unwrap_or(0),
            endpoint_url: deployment.endpoint_url.clone().unwrap_or_default(),
            terminal_endpoint: deployment.terminal_endpoint.clone().unwrap_or_default(),
            payment_escrow_address: deployment.payment_escrow_address.clone().unwrap_or_default(),
            hourly_rate: deployment.actual_hourly_rate.unwrap_or(0) as f64 / 1_000_000_000.0,
            total_cost: deployment.total_cost as f64 / 1_000_000_000.0,
            payment_tx_hash: deployment.payment_escrow_tx_hash.clone().unwrap_or_default(),
            uptime_seconds: deployment.uptime_seconds,
            total_requests: deployment.total_requests,
            avg_latency_ms: deployment.avg_latency_ms,
            error: None,
        }
    }
}

#[tonic::async_trait]
impl DeploymentService for DeploymentServiceImpl {
    async fn create_deployment(
        &self,
        request: Request<CreateDeploymentRequest>,
    ) -> Result<Response<CreateDeploymentResponse>, Status> {
        let req = request.into_inner();
        let config = req
            .config
            .ok_or_else(|| Status::invalid_argument("Deployment config is required"))?;

        info!(
            "Creating deployment for user: {}, type: {:?}",
            config.user_id,
            config.r#type()
        );

        // Parse deployment type
        let deployment_type = match config.r#type() {
            PbDeploymentType::DeploymentTypeLocal => DeploymentType::Local,
            PbDeploymentType::DeploymentTypeNetwork => DeploymentType::Network,
            _ => {
                return Err(Status::invalid_argument("Invalid deployment type"));
            }
        };

        // Get model information
        let model_info = config
            .model
            .ok_or_else(|| Status::invalid_argument("Model info is required"))?;

        let model_id = Uuid::parse_str(&model_info.model_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid model ID: {}", e)))?;

        // Verify model exists
        let model = match queries::get_model_by_id(&self.db_pool, model_id).await {
            Ok(m) => m,
            Err(e) => {
                error!("Model not found: {}", e);
                return Err(Status::not_found(format!("Model {} not found", model_id)));
            }
        };

        // For network deployments, find a suitable node
        let (assigned_node_id, endpoint_url, terminal_endpoint) = if matches!(deployment_type, DeploymentType::Network) {
            // Get available nodes
            let available_nodes = queries::list_available_nodes(&self.db_pool)
                .await
                .map_err(|e| Status::internal(format!("Failed to list nodes: {}", e)))?;

            if available_nodes.is_empty() {
                return Err(Status::unavailable("No compute nodes available"));
            }

            // Create a dummy job for matching (using model requirements)
            let dummy_job = crate::database::models::Job {
                id: Uuid::new_v4(),
                user_wallet: config.user_id.clone(),
                status: crate::database::models::JobStatus::Pending,
                job_type: "deployment".to_string(),
                required_gpu: model.min_vram_bytes > 0,
                required_gpu_memory_gb: Some((model.min_vram_bytes / (1024 * 1024 * 1024)) as i32),
                required_ram_gb: (model.min_ram_bytes / (1024 * 1024 * 1024)) as i32,
                estimated_duration_seconds: 3600, // 1 hour default
                estimated_cost: 0,
                actual_cost: None,
                assigned_node_id: None,
                retry_count: 0,
                result_hash: None,
                error_message: None,
                metadata: serde_json::json!({}),
                created_at: Utc::now(),
                started_at: None,
                completed_at: None,
                updated_at: Utc::now(),
            };

            // Find best matching node
            let best_node = JobMatcher::find_best_node(&dummy_job, &available_nodes)
                .ok_or_else(|| Status::unavailable("No suitable node found for deployment"))?;

            info!("Selected node {} for deployment", best_node.id);

            // Generate endpoints (in real implementation, these would be actual URLs)
            let endpoint = format!("http://{}:8080/inference", best_node.wallet_address);
            let terminal = if config.enable_terminal {
                Some(format!("ws://{}:8080/terminal", best_node.wallet_address))
            } else {
                None
            };

            (Some(best_node.id), Some(endpoint), terminal)
        } else {
            // Local deployment - no node assignment
            (None, Some("http://localhost:8080/inference".to_string()), None)
        };

        // Create deployment record
        let deployment_id = Uuid::new_v4();
        let deployment = Deployment {
            id: deployment_id,
            user_id: config.user_id.clone(),
            model_id,
            deployment_type,
            status: DeploymentStatus::Pending,
            status_message: Some("Deployment created, pending provisioning".to_string()),
            assigned_node_id,
            max_price_per_hour: config.max_price_per_hour.map(|p| (p * 1_000_000_000.0) as i64),
            actual_hourly_rate: None, // Will be set when provisioned
            preferred_region: if config.preferred_region.is_empty() {
                None
            } else {
                Some(config.preferred_region.clone())
            },
            environment_vars: serde_json::to_value(&config.environment_vars).unwrap_or(serde_json::json!({})),
            runtime_params: serde_json::to_value(&config.runtime_params).unwrap_or(serde_json::json!({})),
            port: if config.port > 0 { Some(config.port) } else { None },
            enable_terminal: config.enable_terminal,
            endpoint_url,
            terminal_endpoint,
            payment_escrow_address: None, // TODO: Create escrow for network deployments
            payment_escrow_tx_hash: None,
            total_cost: 0,
            uptime_seconds: 0,
            total_requests: 0,
            avg_latency_ms: 0.0,
            created_at: Utc::now(),
            started_at: None,
            stopped_at: None,
            updated_at: Utc::now(),
        };

        match queries::create_deployment(&self.db_pool, &deployment).await {
            Ok(created_deployment) => {
                info!("Deployment {} created successfully", deployment_id);

                let pb_deployment = self.deployment_to_pb(&created_deployment, &model);

                Ok(Response::new(CreateDeploymentResponse {
                    status: StatusCode::StatusSuccess as i32,
                    deployment: Some(pb_deployment),
                    error: None,
                }))
            }
            Err(e) => {
                error!("Failed to create deployment: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }

    async fn get_deployment(
        &self,
        request: Request<GetDeploymentRequest>,
    ) -> Result<Response<GetDeploymentResponse>, Status> {
        let req = request.into_inner();
        let deployment_id = Uuid::parse_str(&req.deployment_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid deployment ID: {}", e)))?;

        match queries::get_deployment_by_id(&self.db_pool, deployment_id).await {
            Ok(deployment) => {
                // Verify user owns deployment
                if deployment.user_id != req.user_id {
                    return Err(Status::permission_denied("Access denied"));
                }

                // Get model info
                let model = queries::get_model_by_id(&self.db_pool, deployment.model_id)
                    .await
                    .map_err(|e| Status::internal(format!("Failed to get model: {}", e)))?;

                let pb_deployment = self.deployment_to_pb(&deployment, &model);

                Ok(Response::new(GetDeploymentResponse {
                    deployment: Some(pb_deployment),
                    error: None,
                }))
            }
            Err(ServerError::NotFound(msg)) => Err(Status::not_found(msg)),
            Err(e) => Err(Status::internal(format!("Database error: {}", e))),
        }
    }

    async fn list_deployments(
        &self,
        request: Request<ListDeploymentsRequest>,
    ) -> Result<Response<ListDeploymentsResponse>, Status> {
        let req = request.into_inner();

        let page_size = if req.page_size > 0 {
            req.page_size.min(100) as i64
        } else {
            20
        };

        // Parse page token as offset
        let offset: i64 = req
            .page_token
            .parse()
            .unwrap_or(0);

        // Parse filters
        let type_filter = if req.type_filter() != PbDeploymentType::DeploymentTypeUnknown {
            Some(match req.type_filter() {
                PbDeploymentType::DeploymentTypeLocal => DeploymentType::Local,
                PbDeploymentType::DeploymentTypeNetwork => DeploymentType::Network,
                _ => return Err(Status::invalid_argument("Invalid type filter")),
            })
        } else {
            None
        };

        let status_filter = if req.status_filter() != PbDeploymentStatus::DeploymentStatusUnknown {
            Some(match req.status_filter() {
                PbDeploymentStatus::DeploymentStatusPending => DeploymentStatus::Pending,
                PbDeploymentStatus::DeploymentStatusProvisioning => DeploymentStatus::Provisioning,
                PbDeploymentStatus::DeploymentStatusLoading => DeploymentStatus::Loading,
                PbDeploymentStatus::DeploymentStatusReady => DeploymentStatus::Ready,
                PbDeploymentStatus::DeploymentStatusRunning => DeploymentStatus::Running,
                PbDeploymentStatus::DeploymentStatusStopped => DeploymentStatus::Stopped,
                PbDeploymentStatus::DeploymentStatusFailed => DeploymentStatus::Failed,
                PbDeploymentStatus::DeploymentStatusTerminated => DeploymentStatus::Terminated,
                _ => return Err(Status::invalid_argument("Invalid status filter")),
            })
        } else {
            None
        };

        match queries::list_deployments(
            &self.db_pool,
            &req.user_id,
            type_filter,
            status_filter,
            page_size,
            offset,
        )
        .await
        {
            Ok(deployments) => {
                let mut pb_deployments = Vec::new();

                // Fetch model info for each deployment
                for deployment in &deployments {
                    if let Ok(model) = queries::get_model_by_id(&self.db_pool, deployment.model_id).await {
                        pb_deployments.push(self.deployment_to_pb(deployment, &model));
                    }
                }

                let next_page_token = if deployments.len() == page_size as usize {
                    (offset + page_size).to_string()
                } else {
                    String::new()
                };

                Ok(Response::new(ListDeploymentsResponse {
                    deployments: pb_deployments,
                    next_page_token,
                    total_count: 0, // TODO: Get actual count
                    error: None,
                }))
            }
            Err(e) => Err(Status::internal(format!("Database error: {}", e))),
        }
    }

    async fn stop_deployment(
        &self,
        request: Request<StopDeploymentRequest>,
    ) -> Result<Response<StopDeploymentResponse>, Status> {
        let req = request.into_inner();
        let deployment_id = Uuid::parse_str(&req.deployment_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid deployment ID: {}", e)))?;

        info!("Stopping deployment {}", deployment_id);

        // Get deployment
        let deployment = match queries::get_deployment_by_id(&self.db_pool, deployment_id).await {
            Ok(d) => d,
            Err(ServerError::NotFound(msg)) => return Err(Status::not_found(msg)),
            Err(e) => return Err(Status::internal(format!("Database error: {}", e))),
        };

        // Verify user owns deployment
        if deployment.user_id != req.user_id {
            return Err(Status::permission_denied("Access denied"));
        }

        // Check if deployment can be stopped
        if !deployment.can_be_stopped() {
            return Err(Status::failed_precondition(format!(
                "Deployment in status {:?} cannot be stopped",
                deployment.status
            )));
        }

        // Stop the deployment
        match queries::stop_deployment(&self.db_pool, deployment_id).await {
            Ok(_) => {
                info!("Deployment {} stopped successfully", deployment_id);

                // Fetch updated deployment
                let updated_deployment = queries::get_deployment_by_id(&self.db_pool, deployment_id)
                    .await
                    .map_err(|e| Status::internal(format!("Failed to fetch updated deployment: {}", e)))?;

                let model = queries::get_model_by_id(&self.db_pool, updated_deployment.model_id)
                    .await
                    .map_err(|e| Status::internal(format!("Failed to get model: {}", e)))?;

                let pb_deployment = self.deployment_to_pb(&updated_deployment, &model);

                Ok(Response::new(StopDeploymentResponse {
                    status: StatusCode::StatusSuccess as i32,
                    deployment: Some(pb_deployment),
                    error: None,
                }))
            }
            Err(e) => {
                error!("Failed to stop deployment: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }

    async fn delete_deployment(
        &self,
        request: Request<DeleteDeploymentRequest>,
    ) -> Result<Response<DeleteDeploymentResponse>, Status> {
        let req = request.into_inner();
        let deployment_id = Uuid::parse_str(&req.deployment_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid deployment ID: {}", e)))?;

        info!("Deleting deployment {}", deployment_id);

        // Get deployment first to check if it can be deleted
        let deployment = match queries::get_deployment_by_id(&self.db_pool, deployment_id).await {
            Ok(d) => d,
            Err(ServerError::NotFound(msg)) => return Err(Status::not_found(msg)),
            Err(e) => return Err(Status::internal(format!("Database error: {}", e))),
        };

        // Verify user owns deployment
        if deployment.user_id != req.user_id {
            return Err(Status::permission_denied("Access denied"));
        }

        // Check if deployment can be deleted (must be stopped/failed/terminated)
        if !deployment.can_be_deleted() {
            return Err(Status::failed_precondition(format!(
                "Deployment in status {:?} cannot be deleted. Stop it first.",
                deployment.status
            )));
        }

        // Delete the deployment
        match queries::delete_deployment(&self.db_pool, deployment_id, &req.user_id).await {
            Ok(_) => {
                info!("Deployment {} deleted successfully", deployment_id);

                Ok(Response::new(DeleteDeploymentResponse {
                    status: StatusCode::StatusSuccess as i32,
                    error: None,
                }))
            }
            Err(ServerError::NotFound(msg)) => Err(Status::not_found(msg)),
            Err(e) => {
                error!("Failed to delete deployment: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }

    async fn get_deployment_metrics(
        &self,
        request: Request<GetDeploymentMetricsRequest>,
    ) -> Result<Response<GetDeploymentMetricsResponse>, Status> {
        let req = request.into_inner();
        let deployment_id = Uuid::parse_str(&req.deployment_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid deployment ID: {}", e)))?;

        // Parse time range
        let start_time = if req.start_time > 0 {
            Some(chrono::DateTime::from_timestamp(req.start_time, 0).unwrap_or(Utc::now()))
        } else {
            None
        };

        let end_time = if req.end_time > 0 {
            Some(chrono::DateTime::from_timestamp(req.end_time, 0).unwrap_or(Utc::now()))
        } else {
            None
        };

        match queries::get_deployment_metrics_history(
            &self.db_pool,
            deployment_id,
            start_time,
            end_time,
            100, // Limit to 100 data points
        )
        .await
        {
            Ok(metrics) => {
                let pb_metrics: Vec<pb::DeploymentMetricPoint> = metrics
                    .iter()
                    .map(|m| pb::DeploymentMetricPoint {
                        timestamp: m.timestamp.timestamp(),
                        cpu_usage: m.cpu_usage_percent,
                        gpu_usage: m.gpu_usage_percent.unwrap_or(0.0),
                        memory_usage: m.memory_usage_bytes as f64,
                        request_count: m.request_count,
                        avg_latency_ms: m.avg_latency_ms,
                        throughput_rps: m.throughput_rps,
                    })
                    .collect();

                Ok(Response::new(GetDeploymentMetricsResponse {
                    metrics: pb_metrics,
                    error: None,
                }))
            }
            Err(e) => Err(Status::internal(format!("Failed to get metrics: {}", e))),
        }
    }
}
