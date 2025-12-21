use crate::database::{
    models::{TerminalSession, TerminalSessionStatus},
    queries, DbPool,
};
use crate::error::ServerError;
use chrono::Utc;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};
use uuid::Uuid;

// Import generated proto types
pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}

use pb::{
    terminal_service_server::TerminalService, CloseTerminalSessionRequest,
    CloseTerminalSessionResponse, CreateTerminalSessionRequest, CreateTerminalSessionResponse,
    SimpleResponse, StatusCode, TerminalData, TerminalResize,
};

type TerminalStream = Pin<Box<dyn Stream<Item = Result<TerminalData, Status>> + Send>>;

/// Session manager to handle active terminal connections
struct SessionManager {
    // Map of session_id -> sender channel
    sessions: HashMap<String, mpsc::Sender<Result<TerminalData, Status>>>,
}

impl SessionManager {
    fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    fn add_session(&mut self, session_id: String, sender: mpsc::Sender<Result<TerminalData, Status>>) {
        self.sessions.insert(session_id, sender);
    }

    fn remove_session(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    async fn send_to_session(&self, session_id: &str, data: TerminalData) -> Result<(), String> {
        if let Some(sender) = self.sessions.get(session_id) {
            sender
                .send(Ok(data))
                .await
                .map_err(|e| format!("Failed to send data: {}", e))?;
            Ok(())
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }
}

pub struct TerminalServiceImpl {
    db_pool: DbPool,
    session_manager: Arc<RwLock<SessionManager>>,
}

impl TerminalServiceImpl {
    pub fn new(db_pool: DbPool) -> Self {
        Self {
            db_pool,
            session_manager: Arc::new(RwLock::new(SessionManager::new())),
        }
    }
}

#[tonic::async_trait]
impl TerminalService for TerminalServiceImpl {
    async fn create_session(
        &self,
        request: Request<CreateTerminalSessionRequest>,
    ) -> Result<Response<CreateTerminalSessionResponse>, Status> {
        let req = request.into_inner();

        let deployment_id = Uuid::parse_str(&req.deployment_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid deployment ID: {}", e)))?;

        info!(
            "Creating terminal session for deployment {} by user {}",
            deployment_id, req.user_id
        );

        // Verify deployment exists and is running
        let deployment = match queries::get_deployment_by_id(&self.db_pool, deployment_id).await {
            Ok(d) => d,
            Err(ServerError::NotFound(msg)) => return Err(Status::not_found(msg)),
            Err(e) => return Err(Status::internal(format!("Database error: {}", e))),
        };

        // Verify user owns deployment
        if deployment.user_id != req.user_id {
            return Err(Status::permission_denied("Access denied"));
        }

        // Check if deployment is running and supports terminal
        if !deployment.is_running() {
            return Err(Status::failed_precondition("Deployment is not running"));
        }

        if !deployment.enable_terminal {
            return Err(Status::failed_precondition(
                "Terminal access not enabled for this deployment",
            ));
        }

        // Create terminal session
        let session_id = Uuid::new_v4();
        let session = TerminalSession {
            id: session_id,
            deployment_id,
            user_id: req.user_id.clone(),
            status: TerminalSessionStatus::Active,
            rows: req.rows,
            cols: req.cols,
            last_activity: Utc::now(),
            data_sent_bytes: 0,
            data_received_bytes: 0,
            created_at: Utc::now(),
            closed_at: None,
        };

        match queries::create_terminal_session(&self.db_pool, &session).await {
            Ok(created_session) => {
                info!("Terminal session {} created successfully", session_id);

                Ok(Response::new(CreateTerminalSessionResponse {
                    status: StatusCode::StatusSuccess as i32,
                    session: Some(pb::TerminalSession {
                        session_id: created_session.id.to_string(),
                        deployment_id: created_session.deployment_id.to_string(),
                        user_id: created_session.user_id,
                        status: pb::TerminalSessionStatus::TerminalSessionActive as i32,
                        created_at: created_session.created_at.timestamp(),
                        last_activity: created_session.last_activity.timestamp(),
                        rows: created_session.rows,
                        cols: created_session.cols,
                    }),
                    error: None,
                }))
            }
            Err(e) => {
                error!("Failed to create terminal session: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }

    type StreamTerminalStream = TerminalStream;

    async fn stream_terminal(
        &self,
        request: Request<tonic::Streaming<TerminalData>>,
    ) -> Result<Response<Self::StreamTerminalStream>, Status> {
        let mut in_stream = request.into_inner();

        // Create channels for bidirectional communication
        let (tx, rx) = mpsc::channel::<Result<TerminalData, Status>>(100);

        let db_pool = self.db_pool.clone();
        let session_manager = Arc::clone(&self.session_manager);

        // Spawn task to handle incoming data from client
        tokio::spawn(async move {
            let mut session_id: Option<String> = None;

            while let Some(result) = in_stream.next().await {
                match result {
                    Ok(data) => {
                        // First message should contain session_id
                        if session_id.is_none() {
                            if data.session_id.is_empty() {
                                error!("First message must contain session_id");
                                break;
                            }
                            session_id = Some(data.session_id.clone());

                            // Register session in manager
                            let mut manager = session_manager.write().await;
                            manager.add_session(data.session_id.clone(), tx.clone());

                            info!("Terminal session {} connected", data.session_id);
                        }

                        let sid = session_id.as_ref().unwrap();

                        // Parse session ID
                        let session_uuid = match Uuid::parse_str(sid) {
                            Ok(id) => id,
                            Err(e) => {
                                error!("Invalid session ID: {}", e);
                                break;
                            }
                        };

                        // Update activity tracking
                        if let Err(e) = queries::update_terminal_session_activity(
                            &db_pool,
                            session_uuid,
                            data.data.len() as i64,
                            0,
                        )
                        .await
                        {
                            warn!("Failed to update session activity: {}", e);
                        }

                        // In a real implementation, forward data to the actual terminal on the node
                        // For now, just echo back the data as a demo
                        let echo_data = TerminalData {
                            session_id: sid.clone(),
                            deployment_id: data.deployment_id,
                            data: data.data,
                            is_eof: data.is_eof,
                            sequence_number: data.sequence_number,
                        };

                        if let Err(e) = tx.send(Ok(echo_data)).await {
                            error!("Failed to send data back to client: {}", e);
                            break;
                        }

                        // Check for EOF
                        if data.is_eof {
                            info!("Terminal session {} received EOF", sid);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Error receiving terminal data: {}", e);
                        break;
                    }
                }
            }

            // Cleanup on disconnect
            if let Some(sid) = session_id {
                info!("Terminal session {} disconnected", sid);
                let mut manager = session_manager.write().await;
                manager.remove_session(&sid);

                // Close session in database
                if let Ok(session_uuid) = Uuid::parse_str(&sid) {
                    if let Err(e) = queries::close_terminal_session(&db_pool, session_uuid).await {
                        error!("Failed to close terminal session: {}", e);
                    }
                }
            }
        });

        // Return the receiver as a stream
        let out_stream = ReceiverStream::new(rx);
        Ok(Response::new(
            Box::pin(out_stream) as Self::StreamTerminalStream
        ))
    }

    async fn resize_terminal(
        &self,
        request: Request<TerminalResize>,
    ) -> Result<Response<SimpleResponse>, Status> {
        let req = request.into_inner();

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid session ID: {}", e)))?;

        info!(
            "Resizing terminal session {} to {}x{}",
            session_id, req.rows, req.cols
        );

        // Verify session exists
        let _session = match queries::get_terminal_session_by_id(&self.db_pool, session_id).await {
            Ok(s) => s,
            Err(ServerError::NotFound(msg)) => return Err(Status::not_found(msg)),
            Err(e) => return Err(Status::internal(format!("Database error: {}", e))),
        };

        // In a real implementation, send resize event to the actual terminal
        // For now, just acknowledge

        Ok(Response::new(SimpleResponse {
            status: StatusCode::StatusSuccess as i32,
            error: None,
        }))
    }

    async fn close_session(
        &self,
        request: Request<CloseTerminalSessionRequest>,
    ) -> Result<Response<CloseTerminalSessionResponse>, Status> {
        let req = request.into_inner();

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid session ID: {}", e)))?;

        info!("Closing terminal session {}", session_id);

        // Close the session
        match queries::close_terminal_session(&self.db_pool, session_id).await {
            Ok(_) => {
                // Remove from session manager
                let mut manager = self.session_manager.write().await;
                manager.remove_session(&session_id.to_string());

                info!("Terminal session {} closed successfully", session_id);

                Ok(Response::new(CloseTerminalSessionResponse {
                    status: StatusCode::StatusSuccess as i32,
                    error: None,
                }))
            }
            Err(ServerError::NotFound(msg)) => Err(Status::not_found(msg)),
            Err(e) => {
                error!("Failed to close terminal session: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }
}
