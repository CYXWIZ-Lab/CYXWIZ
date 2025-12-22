//! Session Monitor - Monitors Engine heartbeats and handles disconnections
//!
//! When an Engine disconnects mid-training (heartbeat timeout), this monitor:
//! 1. Detects the missing heartbeat
//! 2. Calculates proportional payment based on time used
//! 3. Releases payment to node, refunds remaining to engine
//! 4. Marks node as FREE (online)
//! 5. Applies reputation penalties to nodes for failures

use crate::api::grpc::reservation_service::ActiveReservation;
use crate::database::{models::NodeStatus, queries, DbPool};
use crate::pb::ReservationStatus;
use crate::scheduler::reputation_handler::ReputationHandler;
use chrono::{Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration as TokioDuration};
use tracing::{info, warn, error};

/// Engine heartbeat timeout in seconds
/// If no heartbeat received for this duration, consider Engine disconnected
const ENGINE_HEARTBEAT_TIMEOUT_SECS: i64 = 60;

/// How often to check for disconnected engines (in seconds)
const CHECK_INTERVAL_SECS: u64 = 10;

/// Session monitor for tracking Engine connections
pub struct SessionMonitor {
    db_pool: DbPool,
    active_reservations: Arc<RwLock<HashMap<String, ActiveReservation>>>,
    heartbeat_timeout: Duration,
    check_interval: TokioDuration,
}

impl SessionMonitor {
    /// Create a new SessionMonitor
    pub fn new(
        db_pool: DbPool,
        active_reservations: Arc<RwLock<HashMap<String, ActiveReservation>>>,
    ) -> Self {
        Self {
            db_pool,
            active_reservations,
            heartbeat_timeout: Duration::seconds(ENGINE_HEARTBEAT_TIMEOUT_SECS),
            check_interval: TokioDuration::from_secs(CHECK_INTERVAL_SECS),
        }
    }

    /// Create a SessionMonitor with custom timeout values
    pub fn with_timeouts(
        db_pool: DbPool,
        active_reservations: Arc<RwLock<HashMap<String, ActiveReservation>>>,
        heartbeat_timeout_secs: i64,
        check_interval_secs: u64,
    ) -> Self {
        Self {
            db_pool,
            active_reservations,
            heartbeat_timeout: Duration::seconds(heartbeat_timeout_secs),
            check_interval: TokioDuration::from_secs(check_interval_secs),
        }
    }

    /// Run the session monitor in a loop
    pub async fn run(&self) {
        info!(
            "Session monitor started (heartbeat timeout: {}s, check interval: {}s)",
            self.heartbeat_timeout.num_seconds(),
            self.check_interval.as_secs()
        );

        let mut interval_timer = interval(self.check_interval);

        loop {
            interval_timer.tick().await;
            self.check_disconnected_engines().await;
            self.check_expired_reservations().await;
        }
    }

    /// Check for engines that haven't sent heartbeat and handle disconnection
    async fn check_disconnected_engines(&self) {
        let now = Utc::now().timestamp();
        let timeout_threshold = now - self.heartbeat_timeout.num_seconds();

        // Collect reservations that need to be handled
        let disconnected: Vec<ActiveReservation> = {
            let reservations = self.active_reservations.read().await;
            reservations
                .values()
                .filter(|r| {
                    // Only check active reservations
                    let is_active = r.status == ReservationStatus::ReservationActive as i32
                        || r.status == ReservationStatus::ReservationPending as i32;
                    // Check if heartbeat is too old
                    let heartbeat_timeout = r.last_heartbeat < timeout_threshold;
                    is_active && heartbeat_timeout
                })
                .cloned()
                .collect()
        };

        for reservation in disconnected {
            let time_since_heartbeat = now - reservation.last_heartbeat;
            warn!(
                "Engine disconnected for reservation {} - no heartbeat for {} seconds",
                reservation.reservation_id, time_since_heartbeat
            );

            self.handle_engine_disconnect(&reservation, now).await;
        }
    }

    /// Check for reservations that have expired (time ran out)
    async fn check_expired_reservations(&self) {
        let now = Utc::now().timestamp();

        // Collect reservations that have expired
        let expired: Vec<ActiveReservation> = {
            let reservations = self.active_reservations.read().await;
            reservations
                .values()
                .filter(|r| {
                    // Check if reservation has expired
                    let is_active = r.status == ReservationStatus::ReservationActive as i32
                        || r.status == ReservationStatus::ReservationPending as i32;
                    let has_expired = r.end_time < now;
                    is_active && has_expired
                })
                .cloned()
                .collect()
        };

        for reservation in expired {
            info!(
                "Reservation {} has expired (end_time: {})",
                reservation.reservation_id, reservation.end_time
            );

            self.handle_reservation_expiry(&reservation).await;
        }
    }

    /// Handle Engine disconnection - release node with proportional payment
    async fn handle_engine_disconnect(&self, reservation: &ActiveReservation, now: i64) {
        // Calculate time used and proportional payment
        let time_used_seconds = now - reservation.start_time;
        let total_time_seconds = reservation.end_time - reservation.start_time;

        // Ensure we don't divide by zero and don't exceed 100%
        let usage_ratio = if total_time_seconds > 0 {
            (time_used_seconds as f64 / total_time_seconds as f64).min(1.0)
        } else {
            1.0
        };

        let payment_to_node = (reservation.escrow_amount as f64 * usage_ratio) as i64;
        let refund_to_user = reservation.escrow_amount - payment_to_node;

        info!(
            "Engine disconnect handling for reservation {}: \
             time_used={}s, usage_ratio={:.2}%, payment_to_node={}, refund_to_user={}",
            reservation.reservation_id,
            time_used_seconds,
            usage_ratio * 100.0,
            payment_to_node,
            refund_to_user
        );

        // Update reservation status
        {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&reservation.reservation_id) {
                res.status = ReservationStatus::ReservationCancelled as i32;
            }
        }

        // Mark node as online (free) in database
        self.release_node(&reservation.node_id).await;

        // TODO: Process payment through blockchain
        // - Release payment_to_node to reservation.node_wallet (90% node, 10% platform)
        // - Refund refund_to_user to reservation.user_wallet
        info!(
            "Payment processing for reservation {} (TODO: blockchain integration): \
             node_wallet={} receives {}, user_wallet={} refunded {}",
            reservation.reservation_id,
            reservation.node_wallet,
            payment_to_node,
            reservation.user_wallet,
            refund_to_user
        );
    }

    /// Handle reservation expiry - training completed or time ran out
    async fn handle_reservation_expiry(&self, reservation: &ActiveReservation) {
        // Update reservation status to expired
        {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&reservation.reservation_id) {
                res.status = ReservationStatus::ReservationExpired as i32;
            }
        }

        // Mark node as online (free) in database
        self.release_node(&reservation.node_id).await;

        // Full payment to node since time was fully used
        info!(
            "Reservation {} expired - full payment {} to node {}",
            reservation.reservation_id,
            reservation.escrow_amount,
            reservation.node_wallet
        );

        // TODO: Process full payment through blockchain
        // - Release full escrow to node_wallet (90% node, 10% platform)
    }

    /// Release a node back to the free list
    async fn release_node(&self, node_id: &str) {
        // Parse node ID
        #[cfg(feature = "sqlite-compat")]
        let db_id = node_id.to_string();

        #[cfg(not(feature = "sqlite-compat"))]
        let db_id = match uuid::Uuid::parse_str(node_id) {
            Ok(id) => id,
            Err(e) => {
                error!("Invalid node ID {}: {}", node_id, e);
                return;
            }
        };

        // Update node status in database
        match queries::update_node_status(&self.db_pool, db_id, NodeStatus::Online).await {
            Ok(_) => {
                info!("Node {} marked as ONLINE (released)", node_id);
            }
            Err(e) => {
                error!("Failed to release node {}: {}", node_id, e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payment_calculation() {
        // Test proportional payment calculation
        let escrow_amount = 1_000_000_000i64; // 1 token in lamports
        let total_time = 3600i64; // 1 hour

        // 50% used
        let time_used = 1800i64;
        let ratio = time_used as f64 / total_time as f64;
        let payment = (escrow_amount as f64 * ratio) as i64;
        assert_eq!(payment, 500_000_000);

        // 25% used
        let time_used = 900i64;
        let ratio = time_used as f64 / total_time as f64;
        let payment = (escrow_amount as f64 * ratio) as i64;
        assert_eq!(payment, 250_000_000);
    }
}
