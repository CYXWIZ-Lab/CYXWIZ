//! Job Reservation Service Implementation
//!
//! Handles time-based node reservations for training jobs.
//! Users reserve a node for a duration (e.g., 1 hour), pay upfront via escrow,
//! and get direct P2P access to the node for that time period.

use crate::auth::JWTManager;
use crate::blockchain::SolanaClient;
use crate::config::JwtConfig;
use crate::database::{queries, DbPool, MongoClient};
use crate::database::models::{NodeStatus, DbId};
use crate::pb::{
    job_reservation_service_server::JobReservationService,
    ReserveNodeRequest, ReserveNodeResponse,
    ConfirmJobCompleteRequest, ConfirmJobCompleteResponse,
    ExtendReservationRequest, ExtendReservationResponse,
    ReleaseReservationRequest, ReleaseReservationResponse,
    GetReservationRequest, GetReservationResponse,
    ListReservationsRequest, ListReservationsResponse,
    EngineHeartbeatRequest, EngineHeartbeatResponse,
    ReservationInfo, ReservationStatus, StatusCode, Error,
};
use chrono::{Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{info, warn, error};
use uuid::Uuid;

/// Active reservation tracking (in-memory, would be Redis in production)
#[derive(Debug, Clone)]
pub struct ActiveReservation {
    pub reservation_id: String,
    pub user_id: String,
    pub user_wallet: String,
    pub node_id: String,
    pub node_wallet: String,
    pub node_endpoint: String,
    pub status: i32,  // ReservationStatus
    pub start_time: i64,
    pub end_time: i64,
    pub escrow_account: String,
    pub escrow_amount: i64,
    pub price_per_hour: f64,
    pub current_job_id: Option<String>,
    pub jobs_completed: i32,
    // Heartbeat tracking
    pub last_heartbeat: i64,         // Unix timestamp of last Engine heartbeat
    pub training_active: bool,       // Whether training is currently running
    pub job_progress: f64,           // 0.0 to 1.0
    pub current_metrics: HashMap<String, f64>,  // Current training metrics
}

/// Type alias for the active reservations map
pub type ActiveReservationsMap = Arc<RwLock<HashMap<String, ActiveReservation>>>;

pub struct ReservationServiceImpl {
    db_pool: DbPool,
    jwt_manager: Arc<JWTManager>,
    jwt_config: JwtConfig,
    mongo_client: Option<Arc<MongoClient>>,
    solana_client: Option<Arc<SolanaClient>>,
    /// Active reservations (in production, use Redis)
    active_reservations: ActiveReservationsMap,
}

impl ReservationServiceImpl {
    pub fn new(
        db_pool: DbPool,
        jwt_manager: Arc<JWTManager>,
        jwt_config: JwtConfig,
        mongo_client: Option<Arc<MongoClient>>,
        solana_client: Option<Arc<SolanaClient>>,
    ) -> Self {
        Self {
            db_pool,
            jwt_manager,
            jwt_config,
            mongo_client,
            solana_client,
            active_reservations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new ReservationServiceImpl with a shared active_reservations map
    /// This allows the SessionMonitor to access the same map
    pub fn with_shared_reservations(
        db_pool: DbPool,
        jwt_manager: Arc<JWTManager>,
        jwt_config: JwtConfig,
        mongo_client: Option<Arc<MongoClient>>,
        solana_client: Option<Arc<SolanaClient>>,
        active_reservations: ActiveReservationsMap,
    ) -> Self {
        Self {
            db_pool,
            jwt_manager,
            jwt_config,
            mongo_client,
            solana_client,
            active_reservations,
        }
    }

    /// Helper to parse a UUID string into DbId type
    fn parse_db_id(s: &str) -> Result<DbId, Status> {
        #[cfg(feature = "sqlite-compat")]
        {
            uuid::Uuid::parse_str(s)
                .map_err(|e| Status::invalid_argument(format!("Invalid ID: {}", e)))?;
            Ok(s.to_string())
        }

        #[cfg(not(feature = "sqlite-compat"))]
        {
            uuid::Uuid::parse_str(s)
                .map_err(|e| Status::invalid_argument(format!("Invalid ID: {}", e)))
        }
    }

    /// Generate a P2P authentication token for direct Engine → Node communication
    fn generate_p2p_token(
        &self,
        reservation_id: &str,
        user_id: &str,
        node_id: &str,
        duration_seconds: i64,
    ) -> Result<String, Status> {
        // Parse UUIDs for the JWT manager
        let job_uuid = uuid::Uuid::parse_str(reservation_id)
            .map_err(|e| Status::internal(format!("Invalid reservation ID: {}", e)))?;
        let node_uuid = uuid::Uuid::parse_str(node_id)
            .map_err(|e| Status::internal(format!("Invalid node ID: {}", e)))?;

        // Use JWT manager to create P2P token
        self.jwt_manager
            .generate_p2p_token(user_id, job_uuid, node_uuid, duration_seconds)
            .map_err(|e| Status::internal(format!("Failed to create P2P token: {}", e)))
    }

    /// Calculate cost for reservation duration
    fn calculate_cost(&self, price_per_hour: f64, duration_minutes: i32) -> i64 {
        let hours = duration_minutes as f64 / 60.0;
        let cost = price_per_hour * hours;
        // Convert to lamports (assuming 1 token = 1e9 lamports)
        (cost * 1_000_000_000.0) as i64
    }

    /// Get a reference to active reservations for external monitoring
    pub fn get_active_reservations(&self) -> Arc<RwLock<HashMap<String, ActiveReservation>>> {
        Arc::clone(&self.active_reservations)
    }

    /// Get database pool reference
    pub fn get_db_pool(&self) -> DbPool {
        self.db_pool.clone()
    }
}

#[tonic::async_trait]
impl JobReservationService for ReservationServiceImpl {
    async fn reserve_node(
        &self,
        request: Request<ReserveNodeRequest>,
    ) -> Result<Response<ReserveNodeResponse>, Status> {
        let req = request.into_inner();

        info!(
            "ReserveNode request: node_id={}, user_wallet={}, duration={}min",
            req.node_id, req.user_wallet, req.duration_minutes
        );

        // Validate inputs
        if req.node_id.is_empty() {
            return Ok(Response::new(ReserveNodeResponse {
                status: StatusCode::StatusFailed as i32,
                error: Some(Error {
                    code: 400,
                    message: "Node ID is required".to_string(),
                    ..Default::default()
                }),
                ..Default::default()
            }));
        }

        if req.user_wallet.is_empty() {
            return Ok(Response::new(ReserveNodeResponse {
                status: StatusCode::StatusFailed as i32,
                error: Some(Error {
                    code: 400,
                    message: "User wallet is required".to_string(),
                    ..Default::default()
                }),
                ..Default::default()
            }));
        }

        if req.duration_minutes < 10 {
            return Ok(Response::new(ReserveNodeResponse {
                status: StatusCode::StatusFailed as i32,
                error: Some(Error {
                    code: 400,
                    message: "Minimum reservation is 10 minutes".to_string(),
                    ..Default::default()
                }),
                ..Default::default()
            }));
        }

        if req.duration_minutes > 24 * 60 {
            return Ok(Response::new(ReserveNodeResponse {
                status: StatusCode::StatusFailed as i32,
                error: Some(Error {
                    code: 400,
                    message: "Maximum reservation is 24 hours (1440 minutes)".to_string(),
                    ..Default::default()
                }),
                ..Default::default()
            }));
        }

        // Get node from database
        let node_id = Self::parse_db_id(&req.node_id)?;
        let node = match queries::get_node(&self.db_pool, node_id.clone()).await {
            Ok(Some(n)) => n,
            Ok(None) => {
                warn!("Node not found: {}", req.node_id);
                return Ok(Response::new(ReserveNodeResponse {
                    status: StatusCode::StatusFailed as i32,
                    error: Some(Error {
                        code: 404,
                        message: "Node not found".to_string(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }));
            }
            Err(e) => {
                error!("Database error: {}", e);
                return Err(Status::internal(format!("Database error: {}", e)));
            }
        };

        // Check node status
        if node.status != NodeStatus::Online {
            warn!("Node {} is not online (status: {:?})", req.node_id, node.status);
            return Ok(Response::new(ReserveNodeResponse {
                status: StatusCode::StatusFailed as i32,
                error: Some(Error {
                    code: 409,
                    message: format!("Node is not available (status: {:?})", node.status),
                    ..Default::default()
                }),
                ..Default::default()
            }));
        }

        // Check if node is already reserved
        {
            let reservations = self.active_reservations.read().await;
            for (_, res) in reservations.iter() {
                if res.node_id == req.node_id && res.status == ReservationStatus::ReservationActive as i32 {
                    warn!("Node {} is already reserved", req.node_id);
                    return Ok(Response::new(ReserveNodeResponse {
                        status: StatusCode::StatusFailed as i32,
                        error: Some(Error {
                            code: 409,
                            message: "Node is already reserved".to_string(),
                            ..Default::default()
                        }),
                        ..Default::default()
                    }));
                }
            }
        }

        // Generate reservation ID
        let reservation_id = Uuid::new_v4().to_string();
        let job_id = reservation_id.clone(); // Use same ID for first job

        // Calculate times
        let now = Utc::now();
        let start_time = now.timestamp();
        let end_time = (now + Duration::minutes(req.duration_minutes as i64)).timestamp();
        let p2p_token_expires = end_time + 300; // Token valid 5 min after reservation ends

        // Get node pricing (default if not set)
        let price_per_hour = 0.10; // Default price, would come from node.pricing

        // Calculate escrow amount
        let escrow_amount = self.calculate_cost(price_per_hour, req.duration_minutes);

        // Create escrow on blockchain (if Solana client available)
        let escrow_account = if let Some(ref _solana) = self.solana_client {
            // TODO: Call solana_client.create_escrow(...)
            // For now, generate a placeholder PDA
            format!("escrow_{}", reservation_id)
        } else {
            warn!("Solana client not available, using placeholder escrow");
            format!("escrow_{}", reservation_id)
        };

        // Mark node as BUSY in database
        if let Err(e) = queries::update_node_status(&self.db_pool, node_id.clone(), NodeStatus::Busy).await {
            error!("Failed to update node status: {}", e);
            return Err(Status::internal(format!("Failed to reserve node: {}", e)));
        }
        info!("Node {} marked as BUSY", req.node_id);

        // Generate P2P authentication token
        let p2p_auth_token = self.generate_p2p_token(
            &reservation_id,
            &req.user_wallet,
            &req.node_id,
            p2p_token_expires,
        )?;

        // Build node endpoint
        let node_endpoint = format!("{}:{}", node.ip_address, node.port);

        // Store active reservation
        let reservation = ActiveReservation {
            reservation_id: reservation_id.clone(),
            user_id: req.user_wallet.clone(),
            user_wallet: req.user_wallet.clone(),
            node_id: req.node_id.clone(),
            node_wallet: node.wallet_address.clone(),
            node_endpoint: node_endpoint.clone(),
            status: ReservationStatus::ReservationPending as i32,
            start_time,
            end_time,
            escrow_account: escrow_account.clone(),
            escrow_amount,
            price_per_hour,
            current_job_id: Some(job_id.clone()),
            jobs_completed: 0,
            // Initialize heartbeat tracking
            last_heartbeat: start_time,  // Set initial heartbeat to start time
            training_active: false,
            job_progress: 0.0,
            current_metrics: HashMap::new(),
        };

        {
            let mut reservations = self.active_reservations.write().await;
            reservations.insert(reservation_id.clone(), reservation);
        }

        info!(
            "Reservation created: id={}, node={}, duration={}min, escrow={}",
            reservation_id, req.node_id, req.duration_minutes, escrow_amount
        );

        Ok(Response::new(ReserveNodeResponse {
            status: StatusCode::StatusSuccess as i32,
            reservation_id: reservation_id.clone(),
            job_id,
            node_endpoint,
            p2p_auth_token,
            p2p_token_expires,
            escrow_account,
            escrow_amount,
            reservation_start: start_time,
            reservation_expires: end_time,
            price_per_hour,
            error: None,
        }))
    }

    async fn confirm_job_complete(
        &self,
        request: Request<ConfirmJobCompleteRequest>,
    ) -> Result<Response<ConfirmJobCompleteResponse>, Status> {
        let req = request.into_inner();

        info!(
            "ConfirmJobComplete: reservation_id={}, job_id={}, success={}",
            req.reservation_id, req.job_id, req.success
        );

        // Find the reservation
        let reservation = {
            let reservations = self.active_reservations.read().await;
            reservations.get(&req.reservation_id).cloned()
        };

        let reservation = match reservation {
            Some(r) => r,
            None => {
                warn!("Reservation not found: {}", req.reservation_id);
                return Ok(Response::new(ConfirmJobCompleteResponse {
                    status: StatusCode::StatusFailed as i32,
                    error: Some(Error {
                        code: 404,
                        message: "Reservation not found".to_string(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }));
            }
        };

        // Update reservation status
        {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&req.reservation_id) {
                res.status = ReservationStatus::ReservationCompleting as i32;
                res.jobs_completed += 1;
            }
        }

        // TODO: Wait for node confirmation, then release payment
        // For now, simulate payment release
        let payment_released = req.success;
        let payment_tx_hash = if payment_released {
            format!("tx_{}", Uuid::new_v4())
        } else {
            String::new()
        };

        // Mark reservation as completed and release node
        {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&req.reservation_id) {
                res.status = ReservationStatus::ReservationCompleted as i32;
            }
        }

        // Mark node as Online (free) in database
        let node_id = Self::parse_db_id(&reservation.node_id)?;
        if let Err(e) = queries::update_node_status(&self.db_pool, node_id, NodeStatus::Online).await {
            error!("Failed to update node status: {}", e);
        } else {
            info!("Node {} marked as ONLINE (released)", reservation.node_id);
        }

        info!(
            "Job completed: reservation_id={}, payment_released={}",
            req.reservation_id, payment_released
        );

        Ok(Response::new(ConfirmJobCompleteResponse {
            status: StatusCode::StatusSuccess as i32,
            payment_released,
            payment_tx_hash,
            error: None,
        }))
    }

    async fn extend_reservation(
        &self,
        request: Request<ExtendReservationRequest>,
    ) -> Result<Response<ExtendReservationResponse>, Status> {
        let req = request.into_inner();

        info!(
            "ExtendReservation: reservation_id={}, additional_minutes={}",
            req.reservation_id, req.additional_minutes
        );

        // Find the reservation
        let reservation = {
            let reservations = self.active_reservations.read().await;
            reservations.get(&req.reservation_id).cloned()
        };

        let reservation = match reservation {
            Some(r) => r,
            None => {
                return Ok(Response::new(ExtendReservationResponse {
                    status: StatusCode::StatusFailed as i32,
                    error: Some(Error {
                        code: 404,
                        message: "Reservation not found".to_string(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }));
            }
        };

        // Calculate additional escrow
        let additional_escrow = self.calculate_cost(reservation.price_per_hour, req.additional_minutes);

        // Update reservation end time
        let new_expires = reservation.end_time + (req.additional_minutes as i64 * 60);

        {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&req.reservation_id) {
                res.end_time = new_expires;
                res.escrow_amount += additional_escrow;
            }
        }

        info!(
            "Reservation extended: id={}, new_expires={}, additional_escrow={}",
            req.reservation_id, new_expires, additional_escrow
        );

        Ok(Response::new(ExtendReservationResponse {
            status: StatusCode::StatusSuccess as i32,
            new_expires,
            additional_escrow,
            escrow_tx_hash: format!("ext_tx_{}", Uuid::new_v4()),
            error: None,
        }))
    }

    async fn release_reservation(
        &self,
        request: Request<ReleaseReservationRequest>,
    ) -> Result<Response<ReleaseReservationResponse>, Status> {
        let req = request.into_inner();

        info!(
            "ReleaseReservation: reservation_id={}, reason={}",
            req.reservation_id, req.reason
        );

        // Find the reservation
        let reservation = {
            let reservations = self.active_reservations.read().await;
            reservations.get(&req.reservation_id).cloned()
        };

        let reservation = match reservation {
            Some(r) => r,
            None => {
                return Ok(Response::new(ReleaseReservationResponse {
                    status: StatusCode::StatusFailed as i32,
                    error: Some(Error {
                        code: 404,
                        message: "Reservation not found".to_string(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }));
            }
        };

        let now = Utc::now().timestamp();
        let time_used_seconds = now - reservation.start_time;
        let total_time_seconds = reservation.end_time - reservation.start_time;

        // Calculate proportional payment
        let usage_ratio = time_used_seconds as f64 / total_time_seconds as f64;
        let payment_released = (reservation.escrow_amount as f64 * usage_ratio) as i64;
        let refund_amount = reservation.escrow_amount - payment_released;

        // Update reservation status
        {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&req.reservation_id) {
                res.status = ReservationStatus::ReservationCancelled as i32;
            }
        }

        // Mark node as Online in database
        let node_id = Self::parse_db_id(&reservation.node_id)?;
        if let Err(e) = queries::update_node_status(&self.db_pool, node_id, NodeStatus::Online).await {
            error!("Failed to update node status: {}", e);
        } else {
            info!("Node {} marked as ONLINE (released early)", reservation.node_id);
        }

        info!(
            "Reservation released early: id={}, time_used={}s, payment={}, refund={}",
            req.reservation_id, time_used_seconds, payment_released, refund_amount
        );

        Ok(Response::new(ReleaseReservationResponse {
            status: StatusCode::StatusSuccess as i32,
            time_used_seconds,
            payment_released,
            refund_amount,
            payment_tx_hash: format!("pay_tx_{}", Uuid::new_v4()),
            refund_tx_hash: format!("ref_tx_{}", Uuid::new_v4()),
            error: None,
        }))
    }

    async fn get_reservation(
        &self,
        request: Request<GetReservationRequest>,
    ) -> Result<Response<GetReservationResponse>, Status> {
        let req = request.into_inner();

        let reservation = {
            let reservations = self.active_reservations.read().await;
            reservations.get(&req.reservation_id).cloned()
        };

        match reservation {
            Some(r) => {
                let now = Utc::now().timestamp();
                let time_remaining = std::cmp::max(0, r.end_time - now);

                Ok(Response::new(GetReservationResponse {
                    status: StatusCode::StatusSuccess as i32,
                    reservation: Some(ReservationInfo {
                        reservation_id: r.reservation_id,
                        user_id: r.user_id,
                        user_wallet: r.user_wallet,
                        node_id: r.node_id,
                        node_wallet: r.node_wallet,
                        node_endpoint: r.node_endpoint,
                        status: r.status,
                        start_time: r.start_time,
                        end_time: r.end_time,
                        time_remaining_seconds: time_remaining,
                        escrow_account: r.escrow_account,
                        escrow_amount: r.escrow_amount,
                        price_per_hour: r.price_per_hour,
                        current_job_id: r.current_job_id.unwrap_or_default(),
                        job_progress: 0.0,
                        current_metrics: HashMap::new(),
                        jobs_completed: r.jobs_completed,
                    }),
                    error: None,
                }))
            }
            None => {
                Ok(Response::new(GetReservationResponse {
                    status: StatusCode::StatusFailed as i32,
                    reservation: None,
                    error: Some(Error {
                        code: 404,
                        message: "Reservation not found".to_string(),
                        ..Default::default()
                    }),
                }))
            }
        }
    }

    async fn list_reservations(
        &self,
        request: Request<ListReservationsRequest>,
    ) -> Result<Response<ListReservationsResponse>, Status> {
        let req = request.into_inner();

        let now = Utc::now().timestamp();
        let reservations = self.active_reservations.read().await;

        let mut result: Vec<ReservationInfo> = reservations
            .values()
            .filter(|r| {
                if !req.user_id.is_empty() && r.user_id != req.user_id {
                    return false;
                }
                if req.active_only {
                    let is_active = r.status == ReservationStatus::ReservationActive as i32
                        || r.status == ReservationStatus::ReservationPending as i32;
                    if !is_active {
                        return false;
                    }
                }
                true
            })
            .map(|r| {
                let time_remaining = std::cmp::max(0, r.end_time - now);
                ReservationInfo {
                    reservation_id: r.reservation_id.clone(),
                    user_id: r.user_id.clone(),
                    user_wallet: r.user_wallet.clone(),
                    node_id: r.node_id.clone(),
                    node_wallet: r.node_wallet.clone(),
                    node_endpoint: r.node_endpoint.clone(),
                    status: r.status,
                    start_time: r.start_time,
                    end_time: r.end_time,
                    time_remaining_seconds: time_remaining,
                    escrow_account: r.escrow_account.clone(),
                    escrow_amount: r.escrow_amount,
                    price_per_hour: r.price_per_hour,
                    current_job_id: r.current_job_id.clone().unwrap_or_default(),
                    job_progress: 0.0,
                    current_metrics: HashMap::new(),
                    jobs_completed: r.jobs_completed,
                }
            })
            .collect();

        let total_count = result.len() as i32;

        // Apply pagination
        if req.page_size > 0 {
            result.truncate(req.page_size as usize);
        }

        Ok(Response::new(ListReservationsResponse {
            reservations: result,
            next_page_token: String::new(),
            total_count,
        }))
    }

    async fn engine_heartbeat(
        &self,
        request: Request<EngineHeartbeatRequest>,
    ) -> Result<Response<EngineHeartbeatResponse>, Status> {
        let req = request.into_inner();
        let now = Utc::now().timestamp();

        // Find and update the reservation
        let reservation = {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&req.reservation_id) {
                // Update heartbeat timestamp
                res.last_heartbeat = now;
                res.training_active = req.training_active;
                res.job_progress = req.job_progress;
                res.current_metrics = req.current_metrics.clone();

                // Update status to ACTIVE if it was PENDING
                if res.status == ReservationStatus::ReservationPending as i32 {
                    res.status = ReservationStatus::ReservationActive as i32;
                    info!(
                        "Reservation {} activated (first heartbeat received)",
                        req.reservation_id
                    );
                }

                // Check if reservation has expired
                if now > res.end_time {
                    res.status = ReservationStatus::ReservationExpired as i32;
                    warn!(
                        "Reservation {} has expired during heartbeat",
                        req.reservation_id
                    );
                    return Ok(Response::new(EngineHeartbeatResponse {
                        status: StatusCode::StatusFailed as i32,
                        time_remaining_seconds: 0,
                        should_extend: false,
                        message: "Reservation has expired".to_string(),
                        error: Some(Error {
                            code: 410,
                            message: "Reservation expired".to_string(),
                            ..Default::default()
                        }),
                    }));
                }

                Some(res.clone())
            } else {
                None
            }
        };

        match reservation {
            Some(res) => {
                let time_remaining = std::cmp::max(0, res.end_time - now);
                // Suggest extension if less than 5 minutes remaining
                let should_extend = time_remaining < 300 && res.training_active;

                if should_extend {
                    info!(
                        "Reservation {} has {}s remaining, suggesting extension",
                        req.reservation_id, time_remaining
                    );
                }

                Ok(Response::new(EngineHeartbeatResponse {
                    status: StatusCode::StatusSuccess as i32,
                    time_remaining_seconds: time_remaining,
                    should_extend,
                    message: String::new(),
                    error: None,
                }))
            }
            None => {
                warn!("Heartbeat for unknown reservation: {}", req.reservation_id);
                Ok(Response::new(EngineHeartbeatResponse {
                    status: StatusCode::StatusFailed as i32,
                    time_remaining_seconds: 0,
                    should_extend: false,
                    message: "Reservation not found".to_string(),
                    error: Some(Error {
                        code: 404,
                        message: "Reservation not found".to_string(),
                        ..Default::default()
                    }),
                }))
            }
        }
    }
}
