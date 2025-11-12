use crate::config::RedisConfig;
use crate::error::Result;
use redis::{aio::ConnectionManager, AsyncCommands, Client};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone)]
pub struct RedisCache {
    manager: Option<ConnectionManager>,
    mock_mode: bool,
}

impl RedisCache {
    pub async fn new(config: &RedisConfig) -> Result<Self> {
        let client = Client::open(config.url.as_str())?;
        let manager = ConnectionManager::new(client).await?;
        Ok(Self {
            manager: Some(manager),
            mock_mode: false,
        })
    }

    /// Create a mock cache for testing/demo without Redis
    pub fn new_mock() -> Self {
        Self {
            manager: None,
            mock_mode: true,
        }
    }

    fn get_manager(&mut self) -> Result<&mut ConnectionManager> {
        if self.mock_mode {
            return Err(crate::error::ServerError::Internal("Mock cache mode - Redis not available".to_string()));
        }
        self.manager.as_mut().ok_or_else(||
            crate::error::ServerError::Internal("Redis connection not initialized".to_string())
        )
    }

    // Job queue operations
    pub async fn push_job(&mut self, job_id: Uuid) -> Result<()> {
        let manager = self.get_manager()?;
        manager.rpush("job_queue", job_id.to_string()).await?;
        Ok(())
    }

    pub async fn pop_job(&mut self) -> Result<Option<Uuid>> {
        let manager = self.get_manager()?;
        let result: Option<String> = manager.lpop("job_queue", None).await?;
        Ok(result.and_then(|s| Uuid::parse_str(&s).ok()))
    }

    pub async fn get_queue_length(&mut self) -> Result<usize> {
        let manager = self.get_manager()?;
        let len: usize = manager.llen("job_queue").await?;
        Ok(len)
    }

    // Node status cache
    pub async fn cache_node_status(&mut self, node_id: Uuid, status: &str, ttl_secs: u64) -> Result<()> {
        let key = format!("node:{}:status", node_id);
        let manager = self.get_manager()?;
        manager.set_ex(&key, status, ttl_secs).await?;
        Ok(())
    }

    pub async fn get_node_status(&mut self, node_id: Uuid) -> Result<Option<String>> {
        let key = format!("node:{}:status", node_id);
        let manager = self.get_manager()?;
        let result: Option<String> = manager.get(&key).await?;
        Ok(result)
    }

    // Job assignment cache
    pub async fn cache_job_assignment(&mut self, job_id: Uuid, node_id: Uuid, ttl_secs: u64) -> Result<()> {
        let key = format!("job:{}:assigned_node", job_id);
        let manager = self.get_manager()?;
        manager.set_ex(&key, node_id.to_string(), ttl_secs).await?;
        Ok(())
    }

    pub async fn get_job_assignment(&mut self, job_id: Uuid) -> Result<Option<Uuid>> {
        let key = format!("job:{}:assigned_node", job_id);
        let manager = self.get_manager()?;
        let result: Option<String> = manager.get(&key).await?;
        Ok(result.and_then(|s| Uuid::parse_str(&s).ok()))
    }

    // Rate limiting
    pub async fn check_rate_limit(&mut self, key: &str, max_requests: usize, window_secs: i64) -> Result<bool> {
        let rate_key = format!("rate:{}", key);
        let manager = self.get_manager()?;
        let count: usize = manager.incr(&rate_key, 1).await?;

        if count == 1 {
            manager.expire(&rate_key, window_secs).await?;
        }

        Ok(count <= max_requests)
    }

    // Generic key-value operations
    pub async fn set<T: Serialize>(&mut self, key: &str, value: &T, ttl_secs: Option<u64>) -> Result<()> {
        let serialized = serde_json::to_string(value)?;
        let manager = self.get_manager()?;

        if let Some(ttl) = ttl_secs {
            manager.set_ex(key, serialized, ttl).await?;
        } else {
            manager.set(key, serialized).await?;
        }

        Ok(())
    }

    pub async fn get<T: for<'de> Deserialize<'de>>(&mut self, key: &str) -> Result<Option<T>> {
        let manager = self.get_manager()?;
        let result: Option<String> = manager.get(key).await?;

        match result {
            Some(s) => {
                let deserialized = serde_json::from_str(&s)?;
                Ok(Some(deserialized))
            }
            None => Ok(None),
        }
    }

    pub async fn delete(&mut self, key: &str) -> Result<()> {
        let manager = self.get_manager()?;
        manager.del(key).await?;
        Ok(())
    }

    pub async fn exists(&mut self, key: &str) -> Result<bool> {
        let manager = self.get_manager()?;
        let result: bool = manager.exists(key).await?;
        Ok(result)
    }
}
