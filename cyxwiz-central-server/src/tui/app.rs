use crate::cache::RedisCache;
use crate::database::models::{Job, Node, Payment};
use crate::error::Result;
use chrono::{DateTime, Utc};
use crate::database::DbPool;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Dashboard,
    Nodes,
    Jobs,
    Blockchain,
    Logs,
    Settings,
}

impl View {
    pub fn next(&self) -> Self {
        match self {
            View::Dashboard => View::Nodes,
            View::Nodes => View::Jobs,
            View::Jobs => View::Blockchain,
            View::Blockchain => View::Logs,
            View::Logs => View::Settings,
            View::Settings => View::Dashboard,
        }
    }

    pub fn previous(&self) -> Self {
        match self {
            View::Dashboard => View::Settings,
            View::Nodes => View::Dashboard,
            View::Jobs => View::Nodes,
            View::Blockchain => View::Jobs,
            View::Logs => View::Blockchain,
            View::Settings => View::Logs,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            View::Dashboard => "Dashboard",
            View::Nodes => "Nodes",
            View::Jobs => "Jobs",
            View::Blockchain => "Blockchain",
            View::Logs => "Logs",
            View::Settings => "Settings",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub total_nodes: usize,
    pub online_nodes: usize,
    pub active_jobs: usize,
    pub pending_jobs: usize,
    pub completed_jobs_24h: usize,
    pub total_compute_hours: f64,
    pub db_healthy: bool,
    pub db_latency_ms: u64,
    pub redis_healthy: bool,
    pub redis_latency_ms: u64,
    pub solana_healthy: bool,
    pub solana_latency_ms: u64,
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            online_nodes: 0,
            active_jobs: 0,
            pending_jobs: 0,
            completed_jobs_24h: 0,
            total_compute_hours: 0.0,
            db_healthy: false,
            db_latency_ms: 0,
            redis_healthy: false,
            redis_latency_ms: 0,
            solana_healthy: false,
            solana_latency_ms: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
    Success,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Success => "SUCCESS",
        }
    }
}

pub struct App {
    pub db_pool: DbPool,
    pub cache: Arc<RwLock<RedisCache>>,
    pub current_view: View,
    pub should_quit: bool,
    pub stats: NetworkStats,
    pub nodes: Vec<Node>,
    pub jobs: Vec<Job>,
    pub payments: Vec<Payment>,
    pub logs: Vec<LogEntry>,
    pub selected_node_index: usize,
    pub selected_job_index: usize,
    pub uptime_seconds: u64,
    pub started_at: DateTime<Utc>,
    pub last_update: DateTime<Utc>,

    // For graphs
    pub job_throughput_history: Vec<(DateTime<Utc>, u64)>, // (timestamp, jobs_per_minute)
}

impl App {
    pub fn new(db_pool: DbPool, cache: Arc<RwLock<RedisCache>>) -> Self {
        let now = Utc::now();
        Self {
            db_pool,
            cache,
            current_view: View::Dashboard,
            should_quit: false,
            stats: NetworkStats::default(),
            nodes: Vec::new(),
            jobs: Vec::new(),
            payments: Vec::new(),
            logs: Vec::new(),
            selected_node_index: 0,
            selected_job_index: 0,
            uptime_seconds: 0,
            started_at: now,
            last_update: now,
            job_throughput_history: Vec::new(),
        }
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    pub fn next_view(&mut self) {
        self.current_view = self.current_view.next();
    }

    pub fn previous_view(&mut self) {
        self.current_view = self.current_view.previous();
    }

    pub fn select_next_node(&mut self) {
        if !self.nodes.is_empty() {
            self.selected_node_index = (self.selected_node_index + 1) % self.nodes.len();
        }
    }

    pub fn select_previous_node(&mut self) {
        if !self.nodes.is_empty() {
            self.selected_node_index = if self.selected_node_index == 0 {
                self.nodes.len() - 1
            } else {
                self.selected_node_index - 1
            };
        }
    }

    pub fn select_next_job(&mut self) {
        if !self.jobs.is_empty() {
            self.selected_job_index = (self.selected_job_index + 1) % self.jobs.len();
        }
    }

    pub fn select_previous_job(&mut self) {
        if !self.jobs.is_empty() {
            self.selected_job_index = if self.selected_job_index == 0 {
                self.jobs.len() - 1
            } else {
                self.selected_job_index - 1
            };
        }
    }

    pub fn add_log(&mut self, level: LogLevel, message: String) {
        self.logs.push(LogEntry {
            timestamp: Utc::now(),
            level,
            message,
        });

        // Keep only last 100 logs
        if self.logs.len() > 100 {
            self.logs.remove(0);
        }
    }

    pub fn get_selected_node(&self) -> Option<&Node> {
        self.nodes.get(self.selected_node_index)
    }

    pub fn get_selected_job(&self) -> Option<&Job> {
        self.jobs.get(self.selected_job_index)
    }

    pub fn update_uptime(&mut self) {
        self.uptime_seconds = (Utc::now() - self.started_at).num_seconds() as u64;
    }

    pub fn format_uptime(&self) -> String {
        let days = self.uptime_seconds / 86400;
        let hours = (self.uptime_seconds % 86400) / 3600;
        let minutes = (self.uptime_seconds % 3600) / 60;

        if days > 0 {
            format!("{}d {}h {}m", days, hours, minutes)
        } else if hours > 0 {
            format!("{}h {}m", hours, minutes)
        } else {
            format!("{}m", minutes)
        }
    }
}
