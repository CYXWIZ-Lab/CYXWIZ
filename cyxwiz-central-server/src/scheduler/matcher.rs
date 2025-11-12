use crate::database::models::{Job, Node};
use uuid::Uuid;

/// Intelligent job-to-node matching based on multiple factors:
/// - Node capabilities vs job requirements
/// - Node reputation score
/// - Current node load
/// - Node uptime
pub struct JobMatcher;

impl JobMatcher {
    pub fn find_best_node<'a>(job: &Job, available_nodes: &'a [Node]) -> Option<&'a Node> {
        if available_nodes.is_empty() {
            return None;
        }

        // Filter nodes that can handle the job
        let capable_nodes: Vec<&Node> = available_nodes
            .iter()
            .filter(|node| node.can_handle_job(job) && node.is_available())
            .collect();

        if capable_nodes.is_empty() {
            return None;
        }

        // Calculate match scores for each node
        let mut scored_nodes: Vec<(&Node, f64)> = capable_nodes
            .iter()
            .map(|node| (*node, node.calculate_match_score(job)))
            .collect();

        // Sort by score descending
        scored_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return the best match
        scored_nodes.first().map(|(node, _score)| *node)
    }

    /// Find multiple nodes for a distributed job
    pub fn find_nodes_for_distributed_job<'a>(
        job: &Job,
        available_nodes: &'a [Node],
        num_nodes: usize,
    ) -> Vec<&'a Node> {
        let capable_nodes: Vec<&Node> = available_nodes
            .iter()
            .filter(|node| node.can_handle_job(job) && node.is_available())
            .collect();

        if capable_nodes.is_empty() {
            return vec![];
        }

        let mut scored_nodes: Vec<(&Node, f64)> = capable_nodes
            .iter()
            .map(|node| (*node, node.calculate_match_score(job)))
            .collect();

        scored_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored_nodes
            .into_iter()
            .take(num_nodes)
            .map(|(node, _)| node)
            .collect()
    }

    /// Estimate job cost based on requirements and duration
    pub fn estimate_cost(
        required_gpu: bool,
        required_gpu_memory_gb: Option<i32>,
        required_ram_gb: i32,
        estimated_duration_seconds: i32,
    ) -> i64 {
        // Base rates (in CYXWIZ token smallest units)
        // Assuming 1 CYXWIZ = 1_000_000_000 smallest units (like lamports)
        const BASE_CPU_RATE_PER_SECOND: i64 = 10_000; // 0.00001 CYXWIZ per second
        const GPU_MULTIPLIER: f64 = 10.0; // GPU is 10x more expensive
        const RAM_COST_PER_GB_PER_SECOND: i64 = 1_000;

        let mut cost_per_second = BASE_CPU_RATE_PER_SECOND;

        if required_gpu {
            cost_per_second = (cost_per_second as f64 * GPU_MULTIPLIER) as i64;

            // Additional cost for GPU memory
            if let Some(vram_gb) = required_gpu_memory_gb {
                cost_per_second += vram_gb as i64 * RAM_COST_PER_GB_PER_SECOND * 2;
            }
        }

        // RAM cost
        cost_per_second += required_ram_gb as i64 * RAM_COST_PER_GB_PER_SECOND;

        // Total cost
        cost_per_second * estimated_duration_seconds as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::models::{NodeStatus};
    use chrono::Utc;

    fn create_test_node(
        id: &str,
        gpu: bool,
        gpu_mem: i32,
        ram: i32,
        reputation: f64,
        load: f64,
    ) -> Node {
        Node {
            id: Uuid::parse_str(id).unwrap(),
            wallet_address: format!("wallet_{}", id),
            name: format!("Node {}", id),
            status: NodeStatus::Online,
            reputation_score: reputation,
            stake_amount: 1000_000_000,
            cpu_cores: 16,
            ram_gb: ram,
            gpu_model: if gpu { Some("RTX 4090".to_string()) } else { None },
            gpu_memory_gb: if gpu { Some(gpu_mem) } else { None },
            has_cuda: gpu,
            has_opencl: false,
            total_jobs_completed: 100,
            total_jobs_failed: 5,
            uptime_percentage: 99.5,
            current_load: load,
            country: Some("US".to_string()),
            region: Some("us-west".to_string()),
            last_heartbeat: Utc::now(),
            registered_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn create_test_job(gpu_required: bool, gpu_mem: i32, ram: i32) -> Job {
        Job {
            id: Uuid::new_v4(),
            user_wallet: "user_wallet".to_string(),
            status: crate::database::models::JobStatus::Pending,
            job_type: "training".to_string(),
            required_gpu: gpu_required,
            required_gpu_memory_gb: if gpu_required { Some(gpu_mem) } else { None },
            required_ram_gb: ram,
            estimated_duration_seconds: 3600,
            estimated_cost: 1_000_000,
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
        }
    }

    #[test]
    fn test_find_best_node_with_gpu() {
        let job = create_test_job(true, 16, 32);
        let nodes = vec![
            create_test_node("00000000-0000-0000-0000-000000000001", true, 24, 64, 0.9, 0.3),
            create_test_node("00000000-0000-0000-0000-000000000002", true, 16, 32, 0.8, 0.5),
            create_test_node("00000000-0000-0000-0000-000000000003", false, 0, 32, 0.95, 0.2),
        ];

        let best = JobMatcher::find_best_node(&job, &nodes);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id.to_string(), "00000000-0000-0000-0000-000000000001");
    }

    #[test]
    fn test_estimate_cost() {
        let cost_cpu = JobMatcher::estimate_cost(false, None, 16, 3600);
        assert!(cost_cpu > 0);

        let cost_gpu = JobMatcher::estimate_cost(true, Some(16), 32, 3600);
        assert!(cost_gpu > cost_cpu);
    }
}
