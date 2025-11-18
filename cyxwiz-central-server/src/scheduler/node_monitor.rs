use crate::database::{models::NodeStatus, queries, DbPool};
use chrono::{Duration, Utc};
use tokio::time::{interval, Duration as TokioDuration};
use tracing::{info, warn};
use uuid::Uuid;

/// Monitors node heartbeats and marks nodes as offline if they haven't sent heartbeat recently
pub struct NodeMonitor {
    db_pool: DbPool,
    heartbeat_timeout: Duration,
    check_interval: TokioDuration,
}

impl NodeMonitor {
    /// Create a new NodeMonitor
    ///
    /// # Arguments
    /// * `db_pool` - Database connection pool
    /// * `heartbeat_timeout_secs` - Seconds without heartbeat before marking node offline (default: 30)
    /// * `check_interval_secs` - Seconds between checks (default: 10)
    pub fn new(db_pool: DbPool, heartbeat_timeout_secs: i64, check_interval_secs: u64) -> Self {
        Self {
            db_pool,
            heartbeat_timeout: Duration::seconds(heartbeat_timeout_secs),
            check_interval: TokioDuration::from_secs(check_interval_secs),
        }
    }

    /// Run the node monitor in a loop
    pub async fn run(&self) {
        info!("Node monitor started (timeout: {}s, check interval: {}s)",
              self.heartbeat_timeout.num_seconds(),
              self.check_interval.as_secs());

        let mut interval_timer = interval(self.check_interval);

        loop {
            interval_timer.tick().await;
            self.check_disconnected_nodes().await;
        }
    }

    /// Check for nodes that haven't sent heartbeat and mark them as offline
    async fn check_disconnected_nodes(&self) {
        let timeout_threshold = Utc::now() - self.heartbeat_timeout;

        match queries::get_all_online_nodes(&self.db_pool).await {
            Ok(nodes) => {
                for node in nodes {
                    if node.last_heartbeat < timeout_threshold {
                        // Node missed heartbeat, mark as offline
                        let time_since_heartbeat = Utc::now() - node.last_heartbeat;

                        warn!("Node {} ({}) disconnected - no heartbeat for {} seconds",
                             node.id,
                             node.name,
                             time_since_heartbeat.num_seconds());

                        if let Err(e) = queries::update_node_status(&self.db_pool, node.id, NodeStatus::Offline).await {
                            warn!("Failed to mark node {} as offline: {}", node.id, e);
                        } else {
                            info!("Node {} ({}) marked as OFFLINE", node.id, node.name);
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to check node heartbeats: {}", e);
            }
        }
    }
}
