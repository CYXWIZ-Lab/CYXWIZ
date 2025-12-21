pub mod models;
pub mod mongo;
pub mod queries;

pub use mongo::{MongoClient, MongoUser};

use crate::config::DatabaseConfig;
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use std::time::Duration;

pub type DbPool = SqlitePool;

pub async fn create_pool(config: &DatabaseConfig) -> Result<DbPool, sqlx::Error> {
    SqlitePoolOptions::new()
        .max_connections(config.max_connections)
        .min_connections(config.min_connections)
        .acquire_timeout(Duration::from_secs(30))
        .connect(&config.url)
        .await
}

pub async fn run_migrations(pool: &DbPool) -> Result<(), sqlx::Error> {
    sqlx::migrate!("./migrations")
        .run(pool)
        .await
        .map_err(|e| sqlx::Error::Migrate(Box::new(e)))
}
