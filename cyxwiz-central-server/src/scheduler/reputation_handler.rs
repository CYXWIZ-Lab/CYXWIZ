//! Reputation Handler - Manages node reputation, strikes, and bans
//!
//! This handler processes reputation changes and automatically applies
//! strikes and bans according to the reputation tier system.

use crate::database::{models::*, queries, DbPool};
use chrono::{Duration, Utc};
use tracing::{info, warn, error};

/// Reputation handler for processing reputation changes and strikes
pub struct ReputationHandler {
    db_pool: DbPool,
}

impl ReputationHandler {
    pub fn new(db_pool: DbPool) -> Self {
        Self { db_pool }
    }

    /// Apply a reputation change to a node and check for strikes/bans
    /// Returns true if the node was banned as a result
    pub async fn apply_reputation_change(
        &self,
        node_id: DbId,
        delta: f64,
        reason: &str,
    ) -> Result<bool, crate::error::ServerError> {
        // Get current node state
        let node = queries::get_node(&self.db_pool, node_id.clone()).await?
            .ok_or_else(|| crate::error::ServerError::Internal("Node not found".to_string()))?;

        let old_score = node.reputation_score;
        let was_above_normal = old_score >= REPUTATION_NORMAL_THRESHOLD;
        let was_above_probation = old_score >= REPUTATION_PROBATION_THRESHOLD;

        // Apply the reputation change
        let new_score = queries::update_node_reputation_with_check(
            &self.db_pool,
            node_id.clone(),
            delta,
        ).await?;

        let is_above_normal = new_score >= REPUTATION_NORMAL_THRESHOLD;
        let is_above_probation = new_score >= REPUTATION_PROBATION_THRESHOLD;

        info!(
            "Reputation change for node {}: {} -> {} (delta: {}, reason: {})",
            node_id, old_score, new_score, delta, reason
        );

        // Check if the node crossed the probation threshold (dropped below 50)
        if was_above_normal && !is_above_normal {
            warn!(
                "Node {} dropped below normal threshold ({} -> {}). Recording strike.",
                node_id, old_score, new_score
            );

            // Record a strike
            let strike_count = queries::record_node_strike(&self.db_pool, node_id.clone()).await?;

            // Check if this is the 3rd strike
            if strike_count >= 3 {
                // Ban the node!
                let node = queries::get_node(&self.db_pool, node_id.clone()).await?
                    .ok_or_else(|| crate::error::ServerError::Internal("Node not found".to_string()))?;

                let ban_duration = node.calculate_ban_duration();
                let ban_until = Utc::now() + ban_duration;

                warn!(
                    "Node {} reached {} strikes! Banning for {} hours until {}",
                    node_id, strike_count, ban_duration.num_hours(), ban_until
                );

                queries::ban_node(&self.db_pool, node_id.clone(), ban_until).await?;

                return Ok(true); // Node was banned
            } else {
                info!(
                    "Node {} now has {} strikes (ban threshold: 3)",
                    node_id, strike_count
                );
            }
        }

        // Check if the node dropped below probation threshold (into banned territory)
        if was_above_probation && !is_above_probation {
            warn!(
                "Node {} dropped below probation threshold ({} -> {}). Node effectively banned by low reputation.",
                node_id, old_score, new_score
            );
            // No strike recorded, but node is effectively banned until reputation recovers
        }

        Ok(false) // Node was not banned
    }

    /// Apply positive reputation for job completion
    pub async fn job_completed_success(
        &self,
        node_id: DbId,
        had_good_metrics: bool,
    ) -> Result<(), crate::error::ServerError> {
        let delta = if had_good_metrics {
            REPUTATION_JOB_SUCCESS + REPUTATION_GOOD_METRICS_BONUS
        } else {
            REPUTATION_JOB_SUCCESS
        };

        self.apply_reputation_change(node_id, delta, "Job completed successfully").await?;
        Ok(())
    }

    /// Apply positive reputation for free work completion (higher reward for recovery)
    pub async fn free_work_completed(
        &self,
        node_id: DbId,
    ) -> Result<(), crate::error::ServerError> {
        self.apply_reputation_change(
            node_id,
            REPUTATION_FREE_WORK_SUCCESS,
            "Free work completed for reputation recovery",
        ).await?;
        Ok(())
    }

    /// Apply negative reputation for node disconnect/failure
    pub async fn node_disconnected(
        &self,
        node_id: DbId,
    ) -> Result<bool, crate::error::ServerError> {
        self.apply_reputation_change(
            node_id,
            REPUTATION_NODE_DISCONNECT,
            "Node disconnected or failed during job",
        ).await
    }

    /// Apply negative reputation for heartbeat timeout
    pub async fn heartbeat_timeout(
        &self,
        node_id: DbId,
    ) -> Result<bool, crate::error::ServerError> {
        self.apply_reputation_change(
            node_id,
            REPUTATION_HEARTBEAT_TIMEOUT,
            "Heartbeat timeout",
        ).await
    }

    /// Apply negative reputation for user complaint
    pub async fn user_complaint(
        &self,
        node_id: DbId,
        reason: &str,
    ) -> Result<bool, crate::error::ServerError> {
        let full_reason = format!("User complaint: {}", reason);
        self.apply_reputation_change(node_id, REPUTATION_USER_COMPLAINT, &full_reason).await
    }

    /// Check for expired bans and unban nodes
    pub async fn process_expired_bans(&self) -> Result<i32, crate::error::ServerError> {
        let expired_nodes = queries::get_expired_bans(&self.db_pool).await?;
        let count = expired_nodes.len() as i32;

        for node in expired_nodes {
            info!(
                "Unbanning node {} (ban expired at {:?})",
                node.id, node.banned_until
            );

            queries::unban_node(&self.db_pool, node.id.clone()).await?;

            info!(
                "Node {} unbanned. Reputation reset to probation level (25). Strike counter reset.",
                node.id
            );
        }

        if count > 0 {
            info!("Processed {} expired bans", count);
        }

        Ok(count)
    }

    /// Get summary of node reputation tiers
    pub async fn get_reputation_summary(&self) -> Result<ReputationSummary, crate::error::ServerError> {
        let all_nodes = queries::get_all_nodes(&self.db_pool).await?;
        let banned_nodes = queries::get_banned_nodes(&self.db_pool).await?;
        let probation_nodes = queries::get_probation_nodes(&self.db_pool).await?;

        let premium_count = all_nodes.iter()
            .filter(|n| n.reputation_score >= REPUTATION_PREMIUM_THRESHOLD && !n.is_banned())
            .count();

        let normal_count = all_nodes.iter()
            .filter(|n| n.reputation_score >= REPUTATION_NORMAL_THRESHOLD
                && n.reputation_score < REPUTATION_PREMIUM_THRESHOLD
                && !n.is_banned())
            .count();

        Ok(ReputationSummary {
            total_nodes: all_nodes.len(),
            premium_count,
            normal_count,
            probation_count: probation_nodes.len(),
            banned_count: banned_nodes.len(),
        })
    }
}

/// Summary of node reputation tiers
#[derive(Debug, Clone)]
pub struct ReputationSummary {
    pub total_nodes: usize,
    pub premium_count: usize,
    pub normal_count: usize,
    pub probation_count: usize,
    pub banned_count: usize,
}

/// Background task to periodically check for expired bans
pub struct BanExpirationChecker {
    db_pool: DbPool,
}

impl BanExpirationChecker {
    pub fn new(db_pool: DbPool) -> Self {
        Self { db_pool }
    }

    /// Run the ban expiration checker in a loop
    pub async fn run(&self) {
        use tokio::time::{interval, Duration};

        info!("Ban expiration checker started (check interval: 60s)");

        let mut interval_timer = interval(Duration::from_secs(60));

        loop {
            interval_timer.tick().await;

            let handler = ReputationHandler::new(self.db_pool.clone());
            match handler.process_expired_bans().await {
                Ok(count) => {
                    if count > 0 {
                        info!("Processed {} expired bans", count);
                    }
                }
                Err(e) => {
                    error!("Failed to process expired bans: {}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ban_duration_calculation() {
        // Create a mock node with different total_bans values
        let mut node = Node {
            id: "test".to_string(),
            wallet_address: "wallet".to_string(),
            name: "test".to_string(),
            status: NodeStatus::Online,
            reputation_score: 75.0,
            stake_amount: 0,
            strike_count: 0,
            banned_until: None,
            total_bans: 0,
            last_strike_at: None,
            user_id: None,
            device_id: None,
            cpu_cores: 4,
            ram_gb: 8,
            gpu_model: None,
            gpu_memory_gb: None,
            has_cuda: false,
            has_opencl: false,
            total_jobs_completed: 0,
            total_jobs_failed: 0,
            uptime_percentage: 100.0,
            current_load: 0.0,
            country: None,
            region: None,
            ip_address: "127.0.0.1".to_string(),
            port: 50051,
            last_heartbeat: Utc::now(),
            registered_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // First ban: 24 hours
        node.total_bans = 0;
        assert_eq!(node.calculate_ban_duration(), Duration::hours(24));

        // Second ban: 48 hours
        node.total_bans = 1;
        assert_eq!(node.calculate_ban_duration(), Duration::hours(48));

        // Third+ ban: 1 week (168 hours)
        node.total_bans = 2;
        assert_eq!(node.calculate_ban_duration(), Duration::hours(168));

        node.total_bans = 5;
        assert_eq!(node.calculate_ban_duration(), Duration::hours(168));
    }

    #[test]
    fn test_reputation_thresholds() {
        assert!(REPUTATION_PREMIUM_THRESHOLD > REPUTATION_NORMAL_THRESHOLD);
        assert!(REPUTATION_NORMAL_THRESHOLD > REPUTATION_PROBATION_THRESHOLD);
        assert!(REPUTATION_PROBATION_THRESHOLD > 0.0);
    }
}
