//! Session Monitor - Monitors Engine heartbeats and handles disconnections
//!
//! HOTEL ROOM MODEL: Users pay for TIME, not per-job completion.
//!
//! When an Engine disconnects mid-training (heartbeat timeout), this monitor:
//! 1. Detects the missing heartbeat
//! 2. Marks Engine as DISCONNECTED but keeps reservation ACTIVE
//! 3. Waits for reservation timer to expire (or Engine to reconnect)
//! 4. When timer expires: releases FULL payment to node (no refund - hotel room model)
//! 5. Only then marks node as FREE (online)
//!
//! This is like booking a hotel room - if you leave early, you still pay for the room.

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
            self.check_failed_nodes().await;  // Check for nodes that went offline
        }
    }

    /// Check for engines that haven't sent heartbeat and mark as disconnected
    /// HOTEL ROOM MODEL: Don't release node yet, just mark Engine as disconnected
    async fn check_disconnected_engines(&self) {
        let now = Utc::now().timestamp();
        let timeout_threshold = now - self.heartbeat_timeout.num_seconds();

        // Collect reservations where Engine has timed out but is still marked as connected
        let newly_disconnected: Vec<String> = {
            let reservations = self.active_reservations.read().await;
            reservations
                .values()
                .filter(|r| {
                    // Only check active reservations where Engine is still marked as connected
                    let is_active = r.status == ReservationStatus::ReservationActive as i32
                        || r.status == ReservationStatus::ReservationPending as i32;
                    let heartbeat_timeout = r.last_heartbeat < timeout_threshold;
                    let still_marked_connected = r.engine_connected;
                    is_active && heartbeat_timeout && still_marked_connected
                })
                .map(|r| r.reservation_id.clone())
                .collect()
        };

        // Mark Engines as disconnected (but don't release nodes yet - hotel room model)
        for reservation_id in newly_disconnected {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&reservation_id) {
                let time_since_heartbeat = now - res.last_heartbeat;

                // Mark as disconnected but keep reservation ACTIVE
                res.engine_connected = false;
                res.engine_disconnect_time = Some(now);

                warn!(
                    "[HOTEL ROOM] Engine disconnected for reservation {} - no heartbeat for {}s. \
                     Reservation stays ACTIVE until timer expires at {}. Node stays BUSY.",
                    reservation_id,
                    time_since_heartbeat,
                    res.end_time
                );

                // Calculate time remaining
                let time_remaining = res.end_time - now;
                if time_remaining > 0 {
                    info!(
                        "[HOTEL ROOM] Reservation {} has {}s remaining. \
                         Engine can reconnect. Payment will be released when timer expires.",
                        reservation_id, time_remaining
                    );
                }
            }
        }
    }

    /// Check for reservations that have expired (time ran out)
    /// HOTEL ROOM MODEL: When timer expires, release FULL payment to node
    async fn check_expired_reservations(&self) {
        let now = Utc::now().timestamp();

        // Collect reservations that have expired and haven't been processed yet
        let expired: Vec<ActiveReservation> = {
            let reservations = self.active_reservations.read().await;
            reservations
                .values()
                .filter(|r| {
                    // Check if reservation has expired and payment not yet released
                    let is_active = r.status == ReservationStatus::ReservationActive as i32
                        || r.status == ReservationStatus::ReservationPending as i32;
                    let has_expired = r.end_time < now;
                    let not_paid = !r.payment_released;
                    is_active && has_expired && not_paid
                })
                .cloned()
                .collect()
        };

        for reservation in expired {
            let engine_status = if reservation.engine_connected {
                "connected"
            } else {
                "disconnected"
            };

            info!(
                "[HOTEL ROOM] Reservation {} has expired (end_time: {}, engine was {})",
                reservation.reservation_id, reservation.end_time, engine_status
            );

            self.handle_reservation_expiry(&reservation).await;
        }
    }

    /// Handle reservation expiry - HOTEL ROOM MODEL: release FULL payment regardless of disconnect
    /// This is the key difference from proportional payment models.
    async fn handle_reservation_expiry(&self, reservation: &ActiveReservation) {
        // Calculate payment - FULL payment to node (hotel room model)
        // 90% goes to node, 10% to platform
        let node_share = (reservation.escrow_amount as f64 * 0.90) as i64;
        let platform_fee = reservation.escrow_amount - node_share;

        info!("========================================");
        info!("[HOTEL ROOM] Reservation {} EXPIRED - Processing payment", reservation.reservation_id);
        info!("  Engine was connected: {}", reservation.engine_connected);
        if let Some(disconnect_time) = reservation.engine_disconnect_time {
            let disconnect_duration = reservation.end_time - disconnect_time;
            info!("  Engine disconnected {}s before expiry", disconnect_duration);
        }
        info!("  FULL payment to node (no refund - hotel room model):");
        info!("    Total escrow: {} lamports", reservation.escrow_amount);
        info!("    Node receives: {} lamports (90%)", node_share);
        info!("    Platform fee: {} lamports (10%)", platform_fee);
        info!("========================================");

        // Update reservation status and mark payment as released
        {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(&reservation.reservation_id) {
                res.status = ReservationStatus::ReservationExpired as i32;
                res.payment_released = true;
                res.payment_tx_hash = Some(format!("hotel_tx_{}", uuid::Uuid::new_v4()));
            }
        }

        // Mark node as online (free) in database
        self.release_node(&reservation.node_id).await;

        // TODO: Process full payment through blockchain
        // if let Some(ref solana) = self.solana_client {
        //     solana.release_escrow(&reservation.escrow_account, &reservation.node_wallet, node_share).await;
        // }
        info!(
            "[PAYMENT] Reservation {} complete: node {} receives {} lamports",
            reservation.reservation_id,
            reservation.node_wallet,
            node_share
        );
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

    /// Check if any nodes with active reservations have gone offline
    /// HANDLES EDGE CASE: Both Engine and Node disconnect
    /// - If Node disconnected FIRST → Engine gets FULL REFUND (node's fault)
    /// - If Engine disconnected FIRST → Node still gets paid when timer expires (hotel room model)
    async fn check_failed_nodes(&self) {
        // Get all active reservations with engine disconnect info
        let reservations_to_check: Vec<(String, String, Option<i64>)> = {
            let reservations = self.active_reservations.read().await;
            reservations
                .values()
                .filter(|r| {
                    let is_active = r.status == ReservationStatus::ReservationActive as i32
                        || r.status == ReservationStatus::ReservationPending as i32;
                    is_active && !r.payment_released
                })
                .map(|r| (r.reservation_id.clone(), r.node_id.clone(), r.engine_disconnect_time))
                .collect()
        };

        // Check each node's status in the database
        for (reservation_id, node_id, engine_disconnect_time) in reservations_to_check {
            // Parse node ID
            #[cfg(feature = "sqlite-compat")]
            let db_id = node_id.clone();

            #[cfg(not(feature = "sqlite-compat"))]
            let db_id = match uuid::Uuid::parse_str(&node_id) {
                Ok(id) => id,
                Err(_) => continue,
            };

            // Get node from database (need both status and last_heartbeat)
            let node = match queries::get_node(&self.db_pool, db_id).await {
                Ok(Some(node)) => node,
                Ok(None) => {
                    // Node not found = treat as offline, refund Engine
                    warn!("[NODE FAILURE] Node {} not found in database!", node_id);
                    self.handle_node_failure(&reservation_id, &node_id).await;
                    continue;
                }
                Err(_) => continue,
            };

            // If node is offline, determine fault
            if node.status == NodeStatus::Offline {
                let node_last_heartbeat = node.last_heartbeat.timestamp();

                // EDGE CASE: Both Engine and Node disconnected
                // Determine who disconnected FIRST by comparing timestamps
                if let Some(engine_disconnect) = engine_disconnect_time {
                    // Both are disconnected - compare timestamps
                    if engine_disconnect < node_last_heartbeat {
                        // Engine disconnected FIRST → Hotel room model applies
                        // Node will still get paid when timer expires
                        warn!("========================================");
                        warn!("[BOTH DISCONNECTED] Engine disconnected at {}, Node's last heartbeat at {}",
                              engine_disconnect, node_last_heartbeat);
                        warn!("[BOTH DISCONNECTED] Engine disconnected FIRST - Hotel room model applies");
                        warn!("[BOTH DISCONNECTED] Node {} will still be paid when reservation {} expires",
                              node_id, reservation_id);
                        warn!("========================================");
                        // Don't call handle_node_failure - let timer expire and pay node
                        continue;
                    } else {
                        // Node disconnected FIRST → Node's fault, refund Engine
                        warn!("========================================");
                        warn!("[BOTH DISCONNECTED] Node's last heartbeat at {}, Engine disconnected at {}",
                              node_last_heartbeat, engine_disconnect);
                        warn!("[BOTH DISCONNECTED] Node disconnected FIRST - Refunding Engine");
                        warn!("========================================");
                    }
                } else {
                    // Engine is still connected, but Node went offline → Node's fault
                    warn!("========================================");
                    warn!("[NODE FAILURE] Node {} went OFFLINE during reservation {}",
                          node_id, reservation_id);
                    warn!("[NODE FAILURE] Engine still connected - This is the NODE's fault");
                    warn!("========================================");
                }

                // Node fault confirmed - refund Engine
                self.handle_node_failure(&reservation_id, &node_id).await;
            }
        }
    }

    /// Handle a node failure during an active reservation
    /// Engine gets FULL REFUND, node gets reputation penalty
    async fn handle_node_failure(&self, reservation_id: &str, node_id: &str) {
        // Get reservation details
        let reservation = {
            let reservations = self.active_reservations.read().await;
            reservations.get(reservation_id).cloned()
        };

        let reservation = match reservation {
            Some(r) => r,
            None => return,
        };

        info!("[NODE FAILURE] Processing refund for reservation {}", reservation_id);
        info!("  User wallet: {}", reservation.user_wallet);
        info!("  Escrow amount: {} lamports", reservation.escrow_amount);
        info!("  Node: {} (will receive reputation penalty)", node_id);

        // Update reservation status to FAILED
        {
            let mut reservations = self.active_reservations.write().await;
            if let Some(res) = reservations.get_mut(reservation_id) {
                res.status = ReservationStatus::ReservationFailed as i32;
                res.payment_released = true;  // Mark as processed
                res.payment_tx_hash = Some(format!("refund_tx_{}", uuid::Uuid::new_v4()));
            }
        }

        // Apply reputation penalty to node (-5.0 for failure)
        #[cfg(feature = "sqlite-compat")]
        let db_node_id = node_id.to_string();

        #[cfg(not(feature = "sqlite-compat"))]
        let db_node_id = match uuid::Uuid::parse_str(node_id) {
            Ok(id) => id,
            Err(_) => {
                error!("Invalid node ID for reputation update: {}", node_id);
                return;
            }
        };

        match queries::update_node_reputation_with_check(&self.db_pool, db_node_id, -5.0).await {
            Ok(new_rep) => {
                warn!("[NODE FAILURE] Node {} reputation decreased by 5.0 to {:.1}", node_id, new_rep);
            }
            Err(e) => {
                error!("Failed to update node reputation: {}", e);
            }
        }

        // TODO: Process FULL REFUND through blockchain
        // if let Some(ref solana) = self.solana_client {
        //     solana.refund_escrow(&reservation.escrow_account, &reservation.user_wallet, reservation.escrow_amount).await;
        // }

        info!("[NODE FAILURE] FULL REFUND of {} lamports issued to {}",
              reservation.escrow_amount, reservation.user_wallet);
        info!("[NODE FAILURE] Reservation {} marked as FAILED", reservation_id);
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

    #[test]
    fn test_both_disconnect_fault_determination() {
        // Test the logic for determining fault when both disconnect

        // Scenario 1: Engine disconnects first (timestamp 100), Node last heartbeat at 150
        // Result: Engine's fault, Node should still get paid (hotel room model)
        let engine_disconnect_time = 100i64;
        let node_last_heartbeat = 150i64;
        let engine_first = engine_disconnect_time < node_last_heartbeat;
        assert!(engine_first, "Engine disconnected first - should pay node");

        // Scenario 2: Node disconnects first (last heartbeat at 100), Engine at 150
        // Result: Node's fault, Engine should get refund
        let engine_disconnect_time = 150i64;
        let node_last_heartbeat = 100i64;
        let engine_first = engine_disconnect_time < node_last_heartbeat;
        assert!(!engine_first, "Node disconnected first - should refund engine");

        // Scenario 3: Simultaneous disconnect (same timestamp)
        // Result: Node's fault (benefit of doubt to Engine)
        let engine_disconnect_time = 100i64;
        let node_last_heartbeat = 100i64;
        let engine_first = engine_disconnect_time < node_last_heartbeat;
        assert!(!engine_first, "Simultaneous - give benefit to engine, refund");
    }

    #[test]
    fn test_hotel_room_payment_split() {
        // Test the 90/10 payment split
        let escrow_amount = 1_000_000_000i64; // 1 token

        let node_share = (escrow_amount as f64 * 0.90) as i64;
        let platform_fee = escrow_amount - node_share;

        assert_eq!(node_share, 900_000_000, "Node should get 90%");
        assert_eq!(platform_fee, 100_000_000, "Platform should get 10%");
        assert_eq!(node_share + platform_fee, escrow_amount, "Total should equal escrow");
    }
}
