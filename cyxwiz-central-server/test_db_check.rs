use sqlx::SqlitePool;
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::main]
async fn main() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let db_url = format!("sqlite://{}?mode=rwc", db_path.display());

    let pool = SqlitePool::connect(&db_url).await.unwrap();
   
    // Run migrations
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();

    // Insert a job
    let job_id = Uuid::new_v4();
    let job_id_str = job_id.to_string();
    println!("Inserting job with ID: {}", job_id_str);

    sqlx::query(
        "INSERT INTO jobs (
            id, user_wallet, status, job_type,
            required_gpu, required_gpu_memory_gb, required_ram_gb, estimated_duration_seconds,
            estimated_cost, actual_cost, assigned_node_id, retry_count,
            result_hash, error_message, metadata,
            created_at, started_at, completed_at, updated_at
         )
         VALUES (?, 'test_wallet', 'pending', 'training', 0, NULL, 8, 3600, 1000, NULL, NULL, 0, NULL, NULL, '{}', datetime('now'), NULL, NULL, datetime('now'))"
    )
    .bind(&job_id_str)
    .bind("pending")
    .execute(&pool)
    .await.unwrap();

    println!("Job inserted successfully");

    // Try to query it back using UUID
    println!("\nTrying to query with Uuid::parse_str...");
    let result = sqlx::query("SELECT * FROM jobs WHERE id = $1")
        .bind(job_id)
        .fetch_optional(&pool)
        .await;

    match result {
        Ok(Some(_)) => println!("✅ Found job using $1 placeholder with Uuid"),
        Ok(None) => println!("❌ Job NOT found using $1 placeholder with Uuid"),
        Err(e) => println!("❌ Error: {}", e),
    }

    // Try with ? placeholder
    println!("\nTrying to query with ? placeholder and Uuid...");
    let result = sqlx::query("SELECT * FROM jobs WHERE id = ?")
        .bind(job_id)
        .fetch_optional(&pool)
        .await;

    match result {
        Ok(Some(_)) => println!("✅ Found job using ? placeholder with Uuid"),
        Ok(None) => println!("❌ Job NOT found using ? placeholder with Uuid"),
        Err(e) => println!("❌ Error: {}", e),
    }

    // Try with string
    println!("\nTrying to query with ? placeholder and string...");
    let result = sqlx::query("SELECT * FROM jobs WHERE id = ?")
        .bind(&job_id_str)
        .fetch_optional(&pool)
        .await;

    match result {
        Ok(Some(_)) => println!("✅ Found job using ? placeholder with string"),
        Ok(None) => println!("❌ Job NOT found using ? placeholder with string"),
        Err(e) => println!("❌ Error: {}", e),
    }

    // List all jobs
    println!("\nListing all jobs in database...");
    let rows: Vec<(String,)> = sqlx::query_as("SELECT id FROM jobs")
        .fetch_all(&pool)
        .await.unwrap();
    
    println!("Found {} jobs:", rows.len());
    for (id,) in rows {
        println!("  - {}", id);
    }
}
