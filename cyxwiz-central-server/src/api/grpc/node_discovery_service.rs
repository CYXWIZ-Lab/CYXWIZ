use crate::database::{queries, DbPool};
use crate::pb::{
    node_discovery_service_server::NodeDiscoveryService,
    FindNodesRequest, FindNodesResponse, ListNodesRequest, ListNodesResponse,
    GetNodeInfoRequest, GetNodeInfoResponse, NodeInfo, DeviceCaps, DeviceType,
    NodePricing, Version,
};
use tonic::{Request, Response, Status};
use tracing::{info, warn, error};

pub struct NodeDiscoveryServiceImpl {
    db_pool: DbPool,
}

impl NodeDiscoveryServiceImpl {
    pub fn new(db_pool: DbPool) -> Self {
        Self { db_pool }
    }

    /// Helper to parse a UUID string into DbId type
    #[allow(dead_code)]
    fn parse_db_id(s: &str) -> Result<crate::database::models::DbId, Status> {
        #[cfg(feature = "sqlite-compat")]
        {
            // Validate UUID format but keep as string
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

    fn db_device_to_proto(&self, d: &crate::database::models::NodeDevice) -> DeviceCaps {
        let device_type = match d.device_type.as_str() {
            "cuda" => DeviceType::DeviceCuda as i32,
            "opencl" => DeviceType::DeviceOpencl as i32,
            "cpu" => DeviceType::DeviceCpu as i32,
            _ => DeviceType::DeviceUnknown as i32,
        };
        DeviceCaps {
            device_type,
            device_name: d.device_name.clone(),
            memory_total: d.memory_total_bytes,
            memory_available: d.memory_available_bytes,
            compute_units: d.compute_units,
            supports_fp64: d.supports_fp64,
            supports_fp16: d.supports_fp16,
            gpu_model: d.device_name.clone(),
            vram_total: d.memory_total_bytes,
            vram_available: d.memory_available_bytes,
            driver_version: String::new(),
            cuda_version: String::new(),
            pcie_generation: 0,
            pcie_lanes: 0,
            compute_capability: 0.0,
        }
    }

    fn convert_db_node_to_proto(&self, node: &crate::database::models::Node, devices: Vec<DeviceCaps>) -> NodeInfo {
        // Determine primary device type from devices
        let has_cuda = devices.iter().any(|d| d.device_type == DeviceType::DeviceCuda as i32);
        let has_opencl = devices.iter().any(|d| d.device_type == DeviceType::DeviceOpencl as i32);

        NodeInfo {
            node_id: node.id.to_string(),
            name: node.name.clone(),
            version: Some(Version {
                major: 0,
                minor: 1,
                patch: 0,
                build: String::new(),
            }),
            devices,
            cpu_cores: node.cpu_cores,
            ram_total: (node.ram_gb as i64) * 1024 * 1024 * 1024,
            ram_available: (node.ram_gb as i64) * 1024 * 1024 * 1024 / 2, // Estimate 50% available
            ip_address: node.ip_address.clone(),
            port: node.port,
            region: node.region.clone().unwrap_or_default(),
            compute_score: self.calculate_compute_score(&node, has_cuda, has_opencl),
            // Proto expects 0.0-1.0, DB stores 0-100 - convert for Engine display
            reputation_score: node.reputation_score / 100.0,
            total_jobs_completed: node.total_jobs_completed,
            total_compute_hours: 0, // TODO: Track this
            average_rating: (node.reputation_score / 100.0) * 5.0, // Convert to 5-star rating (0-5 stars)
            staked_amount: (node.stake_amount as f64) / 1_000_000_000.0, // Convert lamports to SOL
            wallet_address: node.wallet_address.clone(),
            is_online: node.status == crate::database::models::NodeStatus::Online,
            last_heartbeat: node.last_heartbeat.timestamp(),
            uptime_percentage: node.uptime_percentage,
            supported_formats: vec!["ONNX".to_string(), "GGUF".to_string()],
            max_model_size: 10 * 1024 * 1024 * 1024, // 10GB default
            supports_terminal_access: false,
            available_runtimes: vec![], // Runtime availability
            // Pricing: Hourly model like cloud providers (AWS, GCP, Azure)
            // Base rate scales with GPU capability
            pricing: Some(NodePricing {
                billing_model: 0, // BILLING_HOURLY
                // Price in CYX tokens per hour (competitive with cloud providers)
                // AWS g4dn: $0.52/hr, GCP T4: $0.35/hr, Azure NC6: $0.90/hr
                // CyxWiz decentralized: $0.25/hr base, scales with GPU
                price_per_hour: if has_cuda {
                    0.25 + (node.gpu_memory_gb.unwrap_or(0) as f64 * 0.02) // $0.25 base + $0.02 per GB VRAM
                } else if has_opencl {
                    0.15 + (node.gpu_memory_gb.unwrap_or(0) as f64 * 0.01) // $0.15 base + $0.01 per GB VRAM
                } else {
                    0.05 + (node.cpu_cores as f64 * 0.005) // CPU-only: $0.05 base + $0.005 per core
                },
                price_per_epoch: 0.01,
                price_per_job_base: 0.0,
                price_per_inference: 0.001, // $0.001 per inference request
                minimum_charge: 0.05, // Minimum $0.05 charge
                minimum_duration_minutes: 10, // Minimum 10 minutes
                discount_1h_plus: 0.10,  // 10% discount for >1 hour
                discount_24h_plus: 0.20, // 20% discount for >24 hours
                discount_bulk: 0.05,
                // USD equivalent (assuming 1 CYX = 1 USD for simplicity, should be fetched from oracle)
                usd_equivalent: if has_cuda {
                    0.25 + (node.gpu_memory_gb.unwrap_or(0) as f64 * 0.02)
                } else if has_opencl {
                    0.15 + (node.gpu_memory_gb.unwrap_or(0) as f64 * 0.01)
                } else {
                    0.05 + (node.cpu_cores as f64 * 0.005)
                },
                price_updated_at: chrono::Utc::now().timestamp(),
                accepts_cyxwiz_token: true,
                accepts_sol: true,
                accepts_usdc: true, // Accept USDC stablecoin
                free_tier_available: false, // No free tier by default - real pricing
                free_tier_minutes: 0,
            }),
        }
    }

    fn calculate_compute_score(&self, node: &crate::database::models::Node, has_cuda: bool, has_opencl: bool) -> f64 {
        let mut score = 0.0;

        // Base score from CPU
        score += node.cpu_cores as f64 * 10.0;

        // RAM bonus
        score += node.ram_gb as f64 * 5.0;

        // GPU bonus
        if has_cuda {
            score += 500.0; // CUDA is highly valued
            if let Some(gpu_mem) = node.gpu_memory_gb {
                score += gpu_mem as f64 * 20.0;
            }
        } else if has_opencl {
            score += 200.0; // OpenCL is good
            if let Some(gpu_mem) = node.gpu_memory_gb {
                score += gpu_mem as f64 * 15.0;
            }
        }

        // Reputation multiplier (DB stores 0-100, convert to 0.0-1.0 for calculation)
        // A node with 75% reputation gets multiplier of 0.5 + 0.75 * 0.5 = 0.875
        let rep_normalized = node.reputation_score / 100.0;
        score *= 0.5 + rep_normalized * 0.5;

        score
    }
}

#[tonic::async_trait]
impl NodeDiscoveryService for NodeDiscoveryServiceImpl {
    async fn list_nodes(
        &self,
        request: Request<ListNodesRequest>,
    ) -> Result<Response<ListNodesResponse>, Status> {
        let req = request.into_inner();
        info!("ListNodes request: online_only={}, page_size={}", req.online_only, req.page_size);

        // Get nodes from database
        let nodes = match queries::get_all_nodes(&self.db_pool).await {
            Ok(nodes) => nodes,
            Err(e) => {
                error!("Failed to get nodes: {}", e);
                return Err(Status::internal(format!("Database error: {}", e)));
            }
        };

        let mut node_infos = Vec::new();
        let mut online_count = 0i32;
        let mut total_compute_power = 0.0f64;
        let total_node_count = nodes.len() as i32;

        for node in nodes {
            // Filter by online status if requested
            if req.online_only && node.status != crate::database::models::NodeStatus::Online {
                continue;
            }

            if node.status == crate::database::models::NodeStatus::Online {
                online_count += 1;
            }

            // Get devices for this node
            let devices: Vec<DeviceCaps> = match queries::get_node_devices(&self.db_pool, &node.id).await {
                Ok(devs) => {
                    info!("Node {} has {} devices", node.id, devs.len());
                    devs.iter().map(|d| {
                        info!("  Device: {} type={} memory_total={}B memory_avail={}B",
                              d.device_name, d.device_type, d.memory_total_bytes, d.memory_available_bytes);
                        self.db_device_to_proto(d)
                    }).collect()
                }
                Err(e) => {
                    warn!("Failed to get devices for node {}: {}", node.id, e);
                    vec![]
                }
            };

            let node_info = self.convert_db_node_to_proto(&node, devices);
            total_compute_power += node_info.compute_score;
            node_infos.push(node_info);

            // Limit results
            if req.page_size > 0 && node_infos.len() >= req.page_size as usize {
                break;
            }
        }

        info!("Returning {} nodes ({} online)", node_infos.len(), online_count);

        Ok(Response::new(ListNodesResponse {
            nodes: node_infos,
            total_count: total_node_count,
            online_count,
            network_total_compute: total_compute_power,
            network_avg_price: 0.10, // Default average price
            next_page_token: String::new(),
        }))
    }

    async fn find_nodes(
        &self,
        request: Request<FindNodesRequest>,
    ) -> Result<Response<FindNodesResponse>, Status> {
        let req = request.into_inner();
        info!("FindNodes request: device={:?}, min_vram={}, max_price={}",
              req.required_device, req.min_vram, req.max_price_per_hour);

        // Get all online nodes
        let nodes = match queries::get_all_nodes(&self.db_pool).await {
            Ok(nodes) => nodes,
            Err(e) => {
                error!("Failed to get nodes: {}", e);
                return Err(Status::internal(format!("Database error: {}", e)));
            }
        };

        let mut matching_nodes = Vec::new();

        for node in nodes {
            // Only online nodes
            if node.status != crate::database::models::NodeStatus::Online {
                continue;
            }

            // Get devices
            let devices: Vec<DeviceCaps> = match queries::get_node_devices(&self.db_pool, &node.id).await {
                Ok(devs) => devs.iter().map(|d| self.db_device_to_proto(d)).collect(),
                Err(_) => vec![],
            };

            // Filter by device type
            if req.required_device != DeviceType::DeviceUnknown as i32 {
                let has_required = devices.iter().any(|d| d.device_type == req.required_device);
                if !has_required {
                    continue;
                }
            }

            // Filter by VRAM
            if req.min_vram > 0 {
                let max_vram = devices.iter().map(|d| d.vram_total).max().unwrap_or(0);
                if max_vram < req.min_vram {
                    continue;
                }
            }

            // Filter by reputation
            if req.min_reputation > 0.0 && node.reputation_score < req.min_reputation {
                continue;
            }

            // Filter by region
            if !req.preferred_region.is_empty() {
                match &node.region {
                    Some(region) => {
                        let region_lower = region.as_str().to_lowercase();
                        let preferred_lower = req.preferred_region.as_str().to_lowercase();
                        if !region_lower.contains(&preferred_lower) {
                            continue;
                        }
                    }
                    None => continue,
                }
            }

            let node_info = self.convert_db_node_to_proto(&node, devices);
            matching_nodes.push(node_info);
        }

        // Sort results
        match req.sort_by {
            1 => matching_nodes.sort_by(|a, b| b.compute_score.partial_cmp(&a.compute_score).unwrap_or(std::cmp::Ordering::Equal)), // SORT_BY_PERFORMANCE
            2 => matching_nodes.sort_by(|a, b| b.reputation_score.partial_cmp(&a.reputation_score).unwrap_or(std::cmp::Ordering::Equal)), // SORT_BY_REPUTATION
            3 => matching_nodes.sort_by(|a, b| b.uptime_percentage.partial_cmp(&a.uptime_percentage).unwrap_or(std::cmp::Ordering::Equal)), // SORT_BY_AVAILABILITY (higher uptime = better)
            _ => matching_nodes.sort_by(|a, b| {
                let price_a = a.pricing.as_ref().map(|p| p.price_per_hour).unwrap_or(f64::MAX);
                let price_b = b.pricing.as_ref().map(|p| p.price_per_hour).unwrap_or(f64::MAX);
                price_a.partial_cmp(&price_b).unwrap_or(std::cmp::Ordering::Equal)
            }), // SORT_BY_PRICE (default)
        }

        let total_matching = matching_nodes.len() as i32;

        // Calculate price statistics
        let prices: Vec<f64> = matching_nodes.iter()
            .filter_map(|n| n.pricing.as_ref().map(|p| p.price_per_hour))
            .collect();
        let avg_price = if prices.is_empty() { 0.0 } else { prices.iter().sum::<f64>() / prices.len() as f64 };
        let min_price = prices.iter().cloned().fold(f64::MAX, f64::min);
        let max_price = prices.iter().cloned().fold(0.0f64, f64::max);

        // Limit results
        if req.max_results > 0 {
            matching_nodes.truncate(req.max_results as usize);
        }

        info!("Found {} matching nodes", matching_nodes.len());

        Ok(Response::new(FindNodesResponse {
            nodes: matching_nodes,
            total_matching,
            avg_price_per_hour: avg_price,
            min_price_per_hour: if min_price == f64::MAX { 0.0 } else { min_price },
            max_price_per_hour: max_price,
            error: None,
        }))
    }

    async fn get_node_info(
        &self,
        request: Request<GetNodeInfoRequest>,
    ) -> Result<Response<GetNodeInfoResponse>, Status> {
        let req = request.into_inner();
        info!("GetNodeInfo request: node_id={}", req.node_id);

        let node_id = Self::parse_db_id(&req.node_id)?;

        let node = match queries::get_node(&self.db_pool, node_id).await {
            Ok(Some(n)) => n,
            Ok(None) => return Err(Status::not_found("Node not found")),
            Err(e) => return Err(Status::internal(format!("Database error: {}", e))),
        };

        // Get devices
        let devices: Vec<DeviceCaps> = match queries::get_node_devices(&self.db_pool, &node.id).await {
            Ok(devs) => devs.iter().map(|d| self.db_device_to_proto(d)).collect(),
            Err(_) => vec![],
        };

        let node_info = self.convert_db_node_to_proto(&node, devices);

        Ok(Response::new(GetNodeInfoResponse {
            info: Some(node_info),
            error: None,
        }))
    }
}
