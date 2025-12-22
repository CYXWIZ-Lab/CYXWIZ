use crate::database::{
    models::{Node, NodeDevice, NodeStatus as DbNodeStatus},
    queries, MongoClient,
};
use crate::error::ServerError;
use crate::scheduler::JobScheduler;
use crate::auth::JWTManager;
use crate::config::JwtConfig;
use chrono::Utc;
use crate::database::DbPool;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::pb::{
    node_service_server::NodeService, AssignJobRequest, AssignJobResponse, HeartbeatRequest,
    HeartbeatResponse, RegisterNodeRequest, RegisterNodeResponse, ReportCompletionRequest,
    ReportCompletionResponse, ReportProgressRequest, ReportProgressResponse, StatusCode,
    GetNodeMetricsRequest, GetNodeMetricsResponse, MetricPoint, DeviceType,
    JobAcceptedRequest, JobAcceptedResponse,
};

pub struct NodeServiceImpl {
    db_pool: DbPool,
    scheduler: Arc<JobScheduler>,
    jwt_manager: Arc<JWTManager>,
    mongo_client: Option<Arc<MongoClient>>,
    jwt_config: JwtConfig,
}

impl NodeServiceImpl {
    pub fn new(
        db_pool: DbPool,
        scheduler: Arc<JobScheduler>,
        jwt_manager: Arc<JWTManager>,
        mongo_client: Option<Arc<MongoClient>>,
        jwt_config: JwtConfig,
    ) -> Self {
        Self { db_pool, scheduler, jwt_manager, mongo_client, jwt_config }
    }

    /// Helper to parse a UUID string into DbId type
    /// For sqlite-compat: validates UUID format but returns String
    /// For postgres: returns Uuid directly
    #[allow(dead_code)]
    fn parse_db_id(s: &str) -> std::result::Result<crate::database::models::DbId, Status> {
        #[cfg(feature = "sqlite-compat")]
        {
            // Validate UUID format but keep as string
            Uuid::parse_str(s)
                .map_err(|e| Status::invalid_argument(format!("Invalid ID: {}", e)))?;
            Ok(s.to_string())
        }

        #[cfg(not(feature = "sqlite-compat"))]
        {
            Uuid::parse_str(s)
                .map_err(|e| Status::invalid_argument(format!("Invalid ID: {}", e)))
        }
    }

    /// Store or update devices for a node
    /// This clears existing devices and inserts new ones from the registration request
    /// Also updates the nodes table gpu_model/gpu_memory_gb fields for job matching
    async fn store_node_devices(&self, node_id: &str, devices: &[crate::pb::DeviceCaps]) {
        use crate::database::models::DbId;

        if devices.is_empty() {
            info!("No devices to store for node {}", node_id);
            return;
        }

        info!("Storing {} devices for node {}", devices.len(), node_id);

        // Parse node ID
        #[cfg(feature = "sqlite-compat")]
        let db_node_id: DbId = node_id.to_string();

        #[cfg(not(feature = "sqlite-compat"))]
        let db_node_id: DbId = match Uuid::parse_str(node_id) {
            Ok(id) => id,
            Err(e) => {
                error!("Failed to parse node ID {}: {}", node_id, e);
                return;
            }
        };

        // Delete existing devices for this node
        match queries::delete_node_devices(&self.db_pool, &db_node_id).await {
            Ok(count) => {
                if count > 0 {
                    info!("Deleted {} old devices for node {}", count, node_id);
                }
            }
            Err(e) => {
                warn!("Failed to delete old devices for node {}: {}", node_id, e);
                // Continue anyway - we'll try to insert new ones
            }
        }

        // Track GPU info for updating nodes table
        let mut has_cuda = false;
        let mut has_opencl = false;
        let mut best_gpu_name: Option<String> = None;
        let mut best_gpu_memory_gb: Option<i32> = None;

        // Insert new devices
        for (idx, device) in devices.iter().enumerate() {
            let device_type = match device.device_type {
                x if x == DeviceType::DeviceCuda as i32 => "cuda",
                x if x == DeviceType::DeviceOpencl as i32 => "opencl",
                x if x == DeviceType::DeviceCpu as i32 => "cpu",
                _ => "unknown",
            };

            // Skip unknown device types
            if device_type == "unknown" {
                warn!("Skipping device {} with unknown type {}", idx, device.device_type);
                continue;
            }

            #[cfg(feature = "sqlite-compat")]
            let device_id: DbId = Uuid::new_v4().to_string();

            #[cfg(not(feature = "sqlite-compat"))]
            let device_id: DbId = Uuid::new_v4();

            // For GPU devices, use vram_total/vram_available instead of memory_total/memory_available
            // because memory_total is system RAM, while vram_total is the actual GPU VRAM
            let is_gpu = device_type == "cuda" || device_type == "opencl";
            let (mem_total, mem_available) = if is_gpu && device.vram_total > 0 {
                (device.vram_total, device.vram_available)
            } else {
                (device.memory_total, device.memory_available)
            };

            // Track GPU capabilities for nodes table update
            if device_type == "cuda" {
                has_cuda = true;
            }
            if device_type == "opencl" {
                has_opencl = true;
            }
            if is_gpu {
                let gpu_mem_gb = (mem_total / (1024 * 1024 * 1024)) as i32;
                // Keep track of best GPU (one with most VRAM)
                if best_gpu_memory_gb.is_none() || gpu_mem_gb > best_gpu_memory_gb.unwrap_or(0) {
                    best_gpu_name = Some(device.device_name.clone());
                    best_gpu_memory_gb = Some(gpu_mem_gb);
                }
            }

            let node_device = NodeDevice {
                id: device_id,
                node_id: db_node_id.clone(),
                device_type: device_type.to_string(),
                device_index: idx as i32,
                device_name: device.device_name.clone(),
                is_enabled: true,  // All devices sent are enabled
                vram_allocated_mb: (mem_available / (1024 * 1024)) as i64,
                cores_allocated: device.compute_units,
                memory_total_bytes: mem_total,
                memory_available_bytes: mem_available,
                compute_units: device.compute_units,
                supports_fp64: device.supports_fp64,
                supports_fp16: device.supports_fp16,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };

            match queries::create_node_device(&self.db_pool, &node_device).await {
                Ok(_) => {
                    info!("Stored device: {} {} ({}MB VRAM, {} compute units)",
                          device_type, device.device_name,
                          node_device.vram_allocated_mb, device.compute_units);
                }
                Err(e) => {
                    error!("Failed to store device {} for node {}: {}", idx, node_id, e);
                }
            }
        }

        // Update nodes table with GPU info for job matching
        // This is critical - the job scheduler checks nodes.gpu_model, not node_devices table
        if has_cuda || has_opencl || best_gpu_name.is_some() {
            info!("Updating node {} GPU info: model={:?}, memory={}GB, cuda={}, opencl={}",
                  node_id, best_gpu_name, best_gpu_memory_gb.unwrap_or(0), has_cuda, has_opencl);

            if let Err(e) = queries::update_node_gpu_info(
                &self.db_pool,
                &db_node_id,
                best_gpu_name.as_deref(),
                best_gpu_memory_gb,
                has_cuda,
                has_opencl,
            ).await {
                error!("Failed to update node GPU info for {}: {}", node_id, e);
            }
        }
    }
}

#[tonic::async_trait]
impl NodeService for NodeServiceImpl {
    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> std::result::Result<Response<RegisterNodeResponse>, Status> {
        // Try to authenticate from gRPC metadata first
        let metadata = request.metadata();
        let mut authenticated_user_id: Option<String> = None;
        let mut authenticated_wallet: Option<String> = None;

        // Try metadata-based authentication (Authorization: Bearer <token>)
        if let Some(auth_header) = metadata.get("authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if auth_str.starts_with("Bearer ") {
                    let token = &auth_str[7..];
                    match self.jwt_manager.verify_user_token(token) {
                        Ok(claims) => {
                            authenticated_user_id = Some(claims.sub.clone());

                            // Look up wallet from MongoDB if available
                            if let Some(ref mongo) = self.mongo_client {
                                if let Ok(Some(user)) = mongo.get_user_by_id(&claims.sub).await {
                                    authenticated_wallet = user.primary_wallet();
                                    info!("Found user wallet from MongoDB: {:?}", authenticated_wallet);
                                }
                            }
                            // Fallback to wallet from token
                            if authenticated_wallet.is_none() {
                                authenticated_wallet = claims.wallet_address;
                            }
                        }
                        Err(e) => {
                            warn!("Metadata token validation failed: {}", e);
                        }
                    }
                }
            }
        }

        let req = request.into_inner();
        let node_info = req.info.ok_or_else(|| Status::invalid_argument("Node info is required"))?;

        // Fallback: try body-based authentication token
        if authenticated_user_id.is_none() {
            let auth_token = &req.authentication_token;
            if auth_token.is_empty() {
                warn!("Node {} attempting registration without authentication token", node_info.name);
                // For now, allow unauthenticated registrations but log warning
                // In production, you may want to reject unauthenticated registrations
            } else {
                match self.jwt_manager.verify_user_token(auth_token) {
                    Ok(claims) => {
                        info!("Node {} authenticated successfully for user {}",
                              node_info.name, claims.sub);
                        authenticated_user_id = Some(claims.sub.clone());

                        // Look up wallet from MongoDB if available
                        if let Some(ref mongo) = self.mongo_client {
                            if let Ok(Some(user)) = mongo.get_user_by_id(&claims.sub).await {
                                authenticated_wallet = user.primary_wallet();
                            }
                        }
                        // Fallback to wallet from token
                        if authenticated_wallet.is_none() {
                            authenticated_wallet = claims.wallet_address;
                        }

                        // If token contains wallet, verify it matches node's wallet
                        if let Some(ref token_wallet) = authenticated_wallet {
                            if !node_info.wallet_address.is_empty()
                               && token_wallet != &node_info.wallet_address {
                                warn!("Wallet mismatch: token has {}, node claims {}",
                                      token_wallet, node_info.wallet_address);
                                // Allow for now, but log the discrepancy
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Node {} has invalid authentication token: {}",
                              node_info.name, e);
                        // For now, continue with registration but mark as unauthenticated
                        // In production, you may want to reject invalid tokens:
                        // return Err(Status::unauthenticated("Invalid authentication token"));
                    }
                }
            }
        }

        info!("Registering node: {} from {}:{}", node_info.name, node_info.ip_address, node_info.port);

        // Check if node already exists (by wallet address or IP:port combination)
        if !node_info.wallet_address.is_empty() {
            // Primary check: wallet address (for authenticated nodes)
            match queries::get_node_by_wallet(&self.db_pool, &node_info.wallet_address).await {
                Ok(Some(existing_node)) => {
                    info!("Node with wallet {} already registered, updating IP to {}:{} and setting online",
                          node_info.wallet_address, node_info.ip_address, node_info.port);

                    let node_id = existing_node.id.clone();

                    // Update IP address and port for existing node
                    if let Err(e) = queries::update_node_endpoint(&self.db_pool, &node_id,
                                                                   &node_info.ip_address, node_info.port).await {
                        warn!("Failed to update node endpoint: {}", e);
                    }

                    // Set node status to online (critical for job assignment!)
                    if let Err(e) = queries::update_node_status(&self.db_pool, node_id.clone(), DbNodeStatus::Online).await {
                        warn!("Failed to update node status: {}", e);
                    }

                    // Update heartbeat timestamp
                    if let Err(e) = queries::update_node_heartbeat(&self.db_pool, node_id.clone()).await {
                        warn!("Failed to update node heartbeat: {}", e);
                    }

                    // Store devices from registration request (re-registration updates devices)
                    let node_id_str = node_id.to_string();
                    self.store_node_devices(&node_id_str, &node_info.devices).await;

                    return Ok(Response::new(RegisterNodeResponse {
                        status: StatusCode::StatusSuccess as i32,
                        node_id: node_id_str.clone(),
                        session_token: format!("session_{}", node_id_str),
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
        } else {
            // Fallback check: IP:port combination (for unauthenticated nodes)
            match queries::get_node_by_endpoint(&self.db_pool, &node_info.ip_address, node_info.port).await {
                Ok(Some(existing_node)) => {
                    info!("Node at {}:{} already registered, setting status to online", node_info.ip_address, node_info.port);

                    let node_id = existing_node.id.clone();

                    // Set node status to online (critical for job assignment!)
                    if let Err(e) = queries::update_node_status(&self.db_pool, node_id.clone(), DbNodeStatus::Online).await {
                        warn!("Failed to update node status: {}", e);
                    }

                    // Update heartbeat timestamp
                    if let Err(e) = queries::update_node_heartbeat(&self.db_pool, node_id.clone()).await {
                        warn!("Failed to update node heartbeat: {}", e);
                    }

                    // Store devices from registration request (re-registration updates devices)
                    let node_id_str = node_id.to_string();
                    self.store_node_devices(&node_id_str, &node_info.devices).await;

                    return Ok(Response::new(RegisterNodeResponse {
                        status: StatusCode::StatusSuccess as i32,
                        node_id: node_id_str.clone(),
                        session_token: format!("session_{}", node_id_str),
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
        }

        // Extract device capabilities
        let devices = &node_info.devices;
        let has_cuda = devices.iter().any(|d| d.device_type == DeviceType::DeviceCuda as i32);
        let has_opencl = devices.iter().any(|d| d.device_type == DeviceType::DeviceOpencl as i32);

        let gpu_device = devices.iter().find(|d| {
            d.device_type == DeviceType::DeviceCuda as i32
                || d.device_type == DeviceType::DeviceOpencl as i32
        });

        // Create node ID - conditional on database type
        #[cfg(feature = "sqlite-compat")]
        let node_id: crate::database::models::DbId = Uuid::new_v4().to_string();

        #[cfg(not(feature = "sqlite-compat"))]
        let node_id: crate::database::models::DbId = Uuid::new_v4();

        // Use authenticated wallet if available, otherwise use node's claimed wallet
        let final_wallet = authenticated_wallet
            .clone()
            .unwrap_or_else(|| node_info.wallet_address.clone());

        let node = Node {
            id: node_id.clone(),
            wallet_address: final_wallet,
            name: node_info.name.clone(),
            status: DbNodeStatus::Online,
            reputation_score: 75.0, // Initial reputation (Normal tier)
            stake_amount: (node_info.staked_amount * 1_000_000_000.0) as i64, // Convert to lamports

            // Reputation tracking (for ban system)
            strike_count: 0,
            banned_until: None,
            total_bans: 0,
            last_strike_at: None,

            // User association - set from authenticated JWT
            user_id: authenticated_user_id.clone(),
            // Device ID - will be set from client's hardware fingerprint
            device_id: None,

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

            ip_address: node_info.ip_address.clone(),
            port: node_info.port,

            last_heartbeat: Utc::now(),
            registered_at: Utc::now(),
            updated_at: Utc::now(),
        };

        match queries::create_node(&self.db_pool, &node).await {
            Ok(_) => {
                info!("Node {} registered successfully", node_id);

                // Store devices from registration request
                let node_id_str = node_id.to_string();
                self.store_node_devices(&node_id_str, &node_info.devices).await;

                Ok(Response::new(RegisterNodeResponse {
                    status: StatusCode::StatusSuccess as i32,
                    node_id: node_id_str,
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
        let node_id = Self::parse_db_id(&req.node_id)?;

        // Update heartbeat timestamp
        match queries::update_node_heartbeat(&self.db_pool, node_id.clone()).await {
            Ok(_) => {
                // Update node status if provided
                if let Some(_current_status) = req.current_status {
                    // Calculate current load based on active jobs
                    let load = if req.active_jobs.is_empty() {
                        0.0
                    } else {
                        req.active_jobs.len() as f64 / 4.0 // Assume max 4 concurrent jobs
                    };

                    if let Err(e) = queries::update_node_load(&self.db_pool, node_id.clone(), load).await {
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
        let node_id = Self::parse_db_id(&req.node_id)?;
        let job_id = Self::parse_db_id(&req.job_id)?;

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
        let node_id = Self::parse_db_id(&req.node_id)?;
        let job_id = Self::parse_db_id(&req.job_id)?;

        let result = req.result.ok_or_else(|| Status::invalid_argument("Job result is required"))?;

        info!("Job {} completed by node {}", job_id, node_id);

        // Update job status in database
        let actual_cost = result.total_compute_time * 10_000; // Simple cost calculation
        match self
            .scheduler
            .handle_job_completion(job_id.clone(), &result.model_weights_hash, actual_cost)
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
        // Note: get_node_metrics_history requires Uuid directly for now
        let node_id_uuid = Uuid::parse_str(&req.node_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid node ID: {}", e)))?;

        match queries::get_node_metrics_history(&self.db_pool, node_id_uuid, 100).await {
            Ok(metrics) => {
                let metric_points: Vec<MetricPoint> = metrics
                    .iter()
                    .map(|m| MetricPoint {
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

    async fn notify_job_accepted(
        &self,
        request: Request<JobAcceptedRequest>,
    ) -> std::result::Result<Response<JobAcceptedResponse>, Status> {
        let req = request.into_inner();
        let node_id = Self::parse_db_id(&req.node_id)?;
        let job_id = Self::parse_db_id(&req.job_id)?;

        info!(
            "Job {} accepted by node {} via P2P from {} (endpoint: {})",
            job_id, node_id, req.engine_address, req.node_endpoint
        );

        // Mark job as running
        if let Err(e) = queries::start_job(&self.db_pool, job_id).await {
            error!("Failed to update job status to running: {}", e);
        }

        Ok(Response::new(JobAcceptedResponse {
            status: StatusCode::StatusSuccess as i32,
            acknowledged: true,
            error: None,
        }))
    }
}
