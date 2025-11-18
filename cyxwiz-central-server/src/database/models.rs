use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

// Conditional type aliases for database compatibility
// SQLite (tests) uses TEXT for IDs, PostgreSQL (production) uses UUID
#[cfg(feature = "sqlite-compat")]
pub type DbId = String;

#[cfg(not(feature = "sqlite-compat"))]
pub type DbId = Uuid;

// Node status enum
// Note: Using TEXT representation for SQLite compatibility
// PostgreSQL will also accept TEXT values instead of ENUM type
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(rename_all = "lowercase")]
pub enum NodeStatus {
    Online,
    Offline,
    Busy,
    Maintenance,
}

// Job status enum
// Note: Using TEXT representation for SQLite compatibility
// PostgreSQL will also accept TEXT values instead of ENUM type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Assigned,
    Running,
    Completed,
    Failed,
    Cancelled,
}

// Payment status enum
// Note: Using TEXT representation for SQLite compatibility
// PostgreSQL will also accept TEXT values instead of ENUM type
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(rename_all = "lowercase")]
pub enum PaymentStatus {
    Pending,
    Locked,
    Streaming,
    Completed,
    Failed,
    Refunded,
}

// Node model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Node {
    pub id: DbId,
    pub wallet_address: String,
    pub name: String,
    pub status: NodeStatus,
    pub reputation_score: f64,
    pub stake_amount: i64, // in lamports (smallest unit of SOL)

    // Capabilities
    pub cpu_cores: i32,
    pub ram_gb: i32,
    pub gpu_model: Option<String>,
    pub gpu_memory_gb: Option<i32>,
    pub has_cuda: bool,
    pub has_opencl: bool,

    // Metrics
    pub total_jobs_completed: i64,
    pub total_jobs_failed: i64,
    pub uptime_percentage: f64,
    pub current_load: f64, // 0.0 to 1.0

    // Location
    pub country: Option<String>,
    pub region: Option<String>,

    pub last_heartbeat: DateTime<Utc>,
    pub registered_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Job model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Job {
    pub id: DbId,
    pub user_wallet: String,
    pub status: JobStatus,
    pub job_type: String, // "training", "inference", etc.

    // Requirements
    pub required_gpu: bool,
    pub required_gpu_memory_gb: Option<i32>,
    pub required_ram_gb: i32,
    pub estimated_duration_seconds: i32,

    // Cost
    pub estimated_cost: i64, // in CYXWIZ tokens (smallest unit)
    pub actual_cost: Option<i64>,

    // Assignment
    pub assigned_node_id: Option<DbId>,
    pub retry_count: i32,

    // Results
    pub result_hash: Option<String>,
    pub error_message: Option<String>,

    // Metadata
    pub metadata: serde_json::Value, // Store additional job parameters as JSON

    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
}

// Payment model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Payment {
    pub id: DbId,
    pub job_id: DbId,
    pub node_id: Option<DbId>,
    pub user_wallet: String,
    pub node_wallet: Option<String>,

    pub amount: i64, // in CYXWIZ tokens (smallest unit)
    pub platform_fee: i64,
    pub node_reward: i64,

    pub status: PaymentStatus,

    // Blockchain
    pub escrow_tx_hash: Option<String>,
    pub completion_tx_hash: Option<String>,
    pub escrow_account: Option<String>,

    pub created_at: DateTime<Utc>,
    pub locked_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

// Node metrics (time-series data)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct NodeMetrics {
    pub id: DbId,
    pub node_id: DbId,

    pub cpu_usage_percent: f64,
    pub ram_usage_percent: f64,
    pub gpu_usage_percent: Option<f64>,
    pub gpu_memory_usage_percent: Option<f64>,

    pub network_rx_bytes: i64,
    pub network_tx_bytes: i64,

    pub active_jobs: i32,

    pub timestamp: DateTime<Utc>,
}

// Helper functions for Node
impl Node {
    pub fn is_available(&self) -> bool {
        matches!(self.status, NodeStatus::Online) && self.current_load < 0.9
    }

    pub fn can_handle_job(&self, job: &Job) -> bool {
        if job.required_gpu && self.gpu_model.is_none() {
            return false;
        }

        if let (Some(required_vram), Some(available_vram)) =
            (job.required_gpu_memory_gb, self.gpu_memory_gb) {
            if required_vram > available_vram {
                return false;
            }
        }

        if job.required_ram_gb > self.ram_gb {
            return false;
        }

        true
    }

    pub fn calculate_match_score(&self, job: &Job) -> f64 {
        if !self.can_handle_job(job) {
            return 0.0;
        }

        let mut score = 0.0;

        // Reputation weight (40%)
        score += self.reputation_score * 0.4;

        // Availability weight (30%)
        let availability = 1.0 - self.current_load;
        score += availability * 0.3;

        // Capability match weight (20%)
        let capability_match = if job.required_gpu {
            if self.gpu_model.is_some() { 1.0 } else { 0.0 }
        } else {
            0.8 // Slight penalty if node has GPU but job doesn't need it
        };
        score += capability_match * 0.2;

        // Uptime weight (10%)
        score += (self.uptime_percentage / 100.0) * 0.1;

        score
    }
}

// Helper functions for Job
impl Job {
    pub fn can_be_assigned(&self) -> bool {
        matches!(self.status, JobStatus::Pending) && self.retry_count < 3
    }

    pub fn can_retry(&self) -> bool {
        matches!(self.status, JobStatus::Failed) && self.retry_count < 3
    }
}

// ============================================================================
// DEPLOYMENT SYSTEM MODELS
// ============================================================================

// Deployment type enum
// Note: Using TEXT representation for SQLite compatibility
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(rename_all = "lowercase")]
pub enum DeploymentType {
    Local,
    Network,
}

// Deployment status enum
// Note: Using TEXT representation for SQLite compatibility
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(rename_all = "lowercase")]
pub enum DeploymentStatus {
    Pending,
    Provisioning,
    Loading,
    Ready,
    Running,
    Stopped,
    Failed,
    Terminated,
}

// Model format enum
// Note: Using TEXT representation for SQLite compatibility
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(rename_all = "lowercase")]
pub enum ModelFormat {
    Onnx,
    Gguf,
    Pytorch,
    Tensorflow,
    Safetensors,
    Tflite,
    Torchscript,
}

// Model source enum
// Note: Using TEXT representation for SQLite compatibility
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(rename_all = "lowercase")]
pub enum ModelSource {
    Local,
    Huggingface,
    CyxwizHub,
    Url,
}

// Terminal session status enum
// Note: Using TEXT representation for SQLite compatibility
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(rename_all = "lowercase")]
pub enum TerminalSessionStatus {
    Active,
    Closed,
    Error,
}

// Model registry entry
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Model {
    pub id: DbId,
    pub name: String,
    pub description: Option<String>,
    pub owner_user_id: String, // Wallet address

    // Model details
    pub format: ModelFormat,
    pub source: ModelSource,
    pub source_url: Option<String>,
    pub size_bytes: i64,

    // Requirements
    pub min_vram_bytes: i64,
    pub min_ram_bytes: i64,
    pub min_cpu_cores: i32,
    pub required_device_type: Option<String>,
    pub gpu_preference: Option<String>,

    // Marketplace
    pub is_public: bool,
    pub price_per_download: i64,
    pub download_count: i64,
    pub rating: f64,
    pub rating_count: i32,
    pub tags: Vec<String>,

    // Storage
    pub storage_path: String,
    pub checksum_sha256: String,

    // Metadata
    pub metadata: serde_json::Value,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Deployment instance
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Deployment {
    pub id: DbId,
    pub user_id: String, // Wallet address
    pub model_id: DbId,

    // Configuration
    #[sqlx(rename = "type")]
    pub deployment_type: DeploymentType,
    pub status: DeploymentStatus,
    pub status_message: Option<String>,

    // Network deployment specific
    pub assigned_node_id: Option<DbId>,
    pub max_price_per_hour: Option<i64>,
    pub actual_hourly_rate: Option<i64>,
    pub preferred_region: Option<String>,

    // Runtime configuration
    pub environment_vars: serde_json::Value,
    pub runtime_params: serde_json::Value,
    pub port: Option<i32>,
    pub enable_terminal: bool,

    // Endpoints
    pub endpoint_url: Option<String>,
    pub terminal_endpoint: Option<String>,

    // Payment (network deployments)
    pub payment_escrow_address: Option<String>,
    pub payment_escrow_tx_hash: Option<String>,
    pub total_cost: i64,

    // Metrics
    pub uptime_seconds: i64,
    pub total_requests: i64,
    pub avg_latency_ms: f64,

    // Timestamps
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
}

// Terminal session
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TerminalSession {
    pub id: DbId,
    pub deployment_id: DbId,
    pub user_id: String, // Wallet address

    pub status: TerminalSessionStatus,

    // Terminal configuration
    pub rows: i32,
    pub cols: i32,

    // Activity tracking
    pub last_activity: DateTime<Utc>,
    pub data_sent_bytes: i64,
    pub data_received_bytes: i64,

    pub created_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
}

// Deployment metrics (time-series data)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct DeploymentMetric {
    pub id: DbId,
    pub deployment_id: DbId,

    // Resource usage
    pub cpu_usage_percent: f64,
    pub gpu_usage_percent: Option<f64>,
    pub memory_usage_bytes: i64,
    pub vram_usage_bytes: Option<i64>,

    // Request metrics
    pub request_count: i64,
    pub avg_latency_ms: f64,
    pub throughput_rps: f64, // Requests per second

    // Error tracking
    pub error_count: i64,

    pub timestamp: DateTime<Utc>,
}

// Model download record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ModelDownload {
    pub id: DbId,
    pub model_id: DbId,
    pub user_id: String, // Wallet address

    pub payment_amount: i64,
    pub payment_tx_hash: Option<String>,

    pub downloaded_at: DateTime<Utc>,
}

// Model rating
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ModelRating {
    pub id: DbId,
    pub model_id: DbId,
    pub user_id: String, // Wallet address

    pub rating: i32, // 1-5
    pub review: Option<String>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Helper functions for Deployment
impl Deployment {
    pub fn is_running(&self) -> bool {
        matches!(
            self.status,
            DeploymentStatus::Running | DeploymentStatus::Ready
        )
    }

    pub fn can_be_stopped(&self) -> bool {
        matches!(
            self.status,
            DeploymentStatus::Running
                | DeploymentStatus::Ready
                | DeploymentStatus::Loading
                | DeploymentStatus::Provisioning
        )
    }

    pub fn can_be_deleted(&self) -> bool {
        matches!(
            self.status,
            DeploymentStatus::Stopped | DeploymentStatus::Failed | DeploymentStatus::Terminated
        )
    }
}

// Helper functions for Model
impl Model {
    pub fn can_be_downloaded(&self, user_id: &str) -> bool {
        self.is_public || self.owner_user_id == user_id
    }

    pub fn requires_payment(&self, user_id: &str) -> bool {
        self.price_per_download > 0 && self.owner_user_id != user_id
    }
}
