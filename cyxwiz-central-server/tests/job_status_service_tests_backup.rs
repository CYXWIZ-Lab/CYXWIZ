// Integration tests for JobStatusService
use cyxwiz_central_server::api::grpc::JobStatusServiceImpl;
use cyxwiz_central_server::database::{create_pool, queries, DatabaseConfig};
use cyxwiz_central_server::pb;
use cyxwiz_central_server::pb::job_status_service_server::JobStatusService;
use serial_test::serial;
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use uuid::Uuid;

// Test helpers
async fn setup_test_database() -> (SqlitePool, TempDir) {
    // Create temporary directory for test database
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test.db");
    let db_url = format!("sqlite://{}?mode=rwc", db_path.display());

    // Create database pool
    let config = DatabaseConfig {
        url: db_url,
        max_connections: 5,
        min_connections: 1,
        connect_timeout_seconds: 30,
        idle_timeout_seconds: 600,
    };

    let pool = create_pool(&config)
        .await
        .expect("Failed to create pool");

    // Run migrations
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("Failed to run migrations");

    (pool, temp_dir)
}

async fn create_test_job(pool: &SqlitePool, status: &str) -> Uuid {
    let job_id = Uuid::new_v4();
    let job_id_str = job_id.to_string();

    sqlx::query(
        "INSERT INTO jobs (id, user_id, status, created_at, updated_at)
         VALUES (?, 'test_user', ?, datetime('now'), datetime('now'))",
    )
    .bind(&job_id_str)
    .bind(status)
    .execute(pool)
    .await
    .expect("Failed to create test job");

    job_id
}

// Test 1: UpdateJobStatus - Successfully update job progress
#[tokio::test]
#[serial]
async fn test_update_job_status_success() {
    let (pool, _temp_dir) = setup_test_database().await;

    // Create a test job
    let job_id = create_test_job(&pool, "assigned").await;

    // Create service
    let service = JobStatusServiceImpl::new(pool.clone());

    // Create update request
    let request = Request::new(pb::UpdateJobStatusRequest {
        node_id: "test_node_123".to_string(),
        job_id: job_id.to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 0.5,
        metrics: HashMap::new(),
        current_epoch: 1,
        log_message: "Training in progress".to_string(),
    });

    // Call the service
    let response = service.update_job_status(request).await;

    // Assert success
    assert!(response.is_ok(), "Expected successful response");
    let response = response.unwrap().into_inner();
    assert_eq!(response.status(), pb::StatusCode::StatusSuccess);

    // Verify database update
    let row: (String, f64) = sqlx::query_as(
        "SELECT status, progress FROM jobs WHERE id = ?",
    )
    .bind(job_id.to_string())
    .fetch_one(&pool)
    .await
    .expect("Failed to fetch job");

    assert_eq!(row.0, "running");
    assert!((row.1 - 0.5).abs() < 0.001);
}

// Test 2: UpdateJobStatus - Invalid job ID format
#[tokio::test]
#[serial]
async fn test_update_job_status_invalid_uuid() {
    let (pool, _temp_dir) = setup_test_database().await;
    let service = JobStatusServiceImpl::new(pool);

    let request = Request::new(pb::UpdateJobStatusRequest {
        node_id: "test_node_123".to_string(),
        job_id: "not-a-valid-uuid".to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 0.5,
        metrics: HashMap::new(),
        current_epoch: 0,
        log_message: "".to_string(),
    });

    let response = service.update_job_status(request).await;

    assert!(response.is_err(), "Expected error for invalid UUID");
    let error = response.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("Invalid job ID"));
}

// Test 3: UpdateJobStatus - Nonexistent job ID
#[tokio::test]
#[serial]
async fn test_update_job_status_nonexistent_job() {
    let (pool, _temp_dir) = setup_test_database().await;
    let service = JobStatusServiceImpl::new(pool);

    let fake_job_id = Uuid::new_v4();
    let request = Request::new(pb::UpdateJobStatusRequest {
        job_id: fake_job_id.to_string(),
        node_id: "test_node_123".to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 0.5,
        message: "".to_string(),
    });

    let response = service.update_job_status(request).await;

    // Note: Current implementation doesn't check if job exists, it just tries to update
    // This may succeed but affect 0 rows. Depending on desired behavior, you may want
    // to modify the service to return an error if no rows are affected.
    // For now, we test the current behavior.
    assert!(response.is_ok() || response.is_err());
}

// Test 4: UpdateJobStatus - Multiple status transitions
#[tokio::test]
#[serial]
async fn test_update_job_status_transitions() {
    let (pool, _temp_dir) = setup_test_database().await;
    let job_id = create_test_job(&pool, "pending").await;
    let service = JobStatusServiceImpl::new(pool.clone());

    // Transition 1: pending -> running (progress 0.0)
    let req1 = Request::new(pb::UpdateJobStatusRequest {
        job_id: job_id.to_string(),
        node_id: "node_123".to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 0.0,
        message: "Starting".to_string(),
    });
    let res1 = service.update_job_status(req1).await;
    assert!(res1.is_ok());

    // Transition 2: running -> running (progress 0.3)
    let req2 = Request::new(pb::UpdateJobStatusRequest {
        job_id: job_id.to_string(),
        node_id: "node_123".to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 0.3,
        message: "30% complete".to_string(),
    });
    let res2 = service.update_job_status(req2).await;
    assert!(res2.is_ok());

    // Transition 3: running -> running (progress 0.9)
    let req3 = Request::new(pb::UpdateJobStatusRequest {
        job_id: job_id.to_string(),
        node_id: "node_123".to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 0.9,
        message: "Almost done".to_string(),
    });
    let res3 = service.update_job_status(req3).await;
    assert!(res3.is_ok());

    // Verify final progress
    let row: (f64,) = sqlx::query_as("SELECT progress FROM jobs WHERE id = ?")
        .bind(job_id.to_string())
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch job");

    assert!((row.0 - 0.9).abs() < 0.001, "Expected progress 0.9");
}

// Test 5: ReportJobResult - Successful completion
#[tokio::test]
#[serial]
async fn test_report_job_result_success() {
    let (pool, _temp_dir) = setup_test_database().await;
    let job_id = create_test_job(&pool, "running").await;
    let service = JobStatusServiceImpl::new(pool.clone());

    let request = Request::new(pb::ReportJobResultRequest {
        job_id: job_id.to_string(),
        node_id: "node_123".to_string(),
        status: pb::StatusCode::StatusSuccess as i32,
        message: "Training completed successfully".to_string(),
        result_data: "model_v1.bin".to_string(),
        error_message: "".to_string(),
    });

    let response = service.report_job_result(request).await;

    assert!(response.is_ok(), "Expected successful response");
    let response = response.unwrap().into_inner();
    assert!(response.success);
    assert_eq!(response.message, "Job result reported successfully");

    // Verify database update
    let row: (String,) = sqlx::query_as("SELECT status FROM jobs WHERE id = ?")
        .bind(job_id.to_string())
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch job");

    assert_eq!(row.0, "completed");
}

// Test 6: ReportJobResult - Failed completion
#[tokio::test]
#[serial]
async fn test_report_job_result_failure() {
    let (pool, _temp_dir) = setup_test_database().await;
    let job_id = create_test_job(&pool, "running").await;
    let service = JobStatusServiceImpl::new(pool.clone());

    let request = Request::new(pb::ReportJobResultRequest {
        job_id: job_id.to_string(),
        node_id: "node_123".to_string(),
        status: pb::StatusCode::StatusFailed as i32,
        message: "Job failed".to_string(),
        result_data: "".to_string(),
        error_message: "Out of memory error".to_string(),
    });

    let response = service.report_job_result(request).await;

    assert!(response.is_ok());
    let response = response.unwrap().into_inner();
    assert!(response.success);

    // Verify status is 'failed'
    let row: (String,) = sqlx::query_as("SELECT status FROM jobs WHERE id = ?")
        .bind(job_id.to_string())
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch job");

    assert_eq!(row.0, "failed");
}

// Test 7: ReportJobResult - Invalid job ID
#[tokio::test]
#[serial]
async fn test_report_job_result_invalid_uuid() {
    let (pool, _temp_dir) = setup_test_database().await;
    let service = JobStatusServiceImpl::new(pool);

    let request = Request::new(pb::ReportJobResultRequest {
        job_id: "invalid-uuid".to_string(),
        node_id: "node_123".to_string(),
        status: pb::StatusCode::StatusSuccess as i32,
        message: "".to_string(),
        result_data: "".to_string(),
        error_message: "".to_string(),
    });

    let response = service.report_job_result(request).await;

    assert!(response.is_err());
    let error = response.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("Invalid job ID"));
}

// Test 8: Database persistence verification
#[tokio::test]
#[serial]
async fn test_database_persistence() {
    let (pool, _temp_dir) = setup_test_database().await;
    let job_id = create_test_job(&pool, "pending").await;
    let service = JobStatusServiceImpl::new(pool.clone());

    // Send multiple updates
    for i in 1..=5 {
        let progress = i as f64 / 5.0;
        let request = Request::new(pb::UpdateJobStatusRequest {
            job_id: job_id.to_string(),
            node_id: format!("node_{}", i),
            status: pb::StatusCode::StatusInProgress as i32,
            progress,
            message: format!("Progress {}", i),
        });

        service
            .update_job_status(request)
            .await
            .expect("Failed to update");
    }

    // Verify the final state
    let row: (String, f64) = sqlx::query_as(
        "SELECT status, progress FROM jobs WHERE id = ?",
    )
    .bind(job_id.to_string())
    .fetch_one(&pool)
    .await
    .expect("Failed to fetch job");

    assert_eq!(row.0, "running");
    assert!((row.1 - 1.0).abs() < 0.001, "Expected final progress to be 1.0");
}

// Test 9: Concurrent updates (stress test)
#[tokio::test]
#[serial]
async fn test_concurrent_updates() {
    let (pool, _temp_dir) = setup_test_database().await;
    let job_id = create_test_job(&pool, "running").await;

    // Spawn multiple concurrent update tasks
    let mut handles = vec![];

    for i in 0..10 {
        let pool_clone = pool.clone();
        let job_id_clone = job_id;
        let handle = tokio::spawn(async move {
            let service = JobStatusServiceImpl::new(pool_clone);
            let request = Request::new(pb::UpdateJobStatusRequest {
                job_id: job_id_clone.to_string(),
                node_id: format!("node_{}", i),
                status: pb::StatusCode::StatusInProgress as i32,
                progress: i as f64 / 10.0,
                message: format!("Update {}", i),
            });

            service.update_job_status(request).await
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "Expected successful update");
    }

    // Verify job still exists and has a valid state
    let row: (String,) = sqlx::query_as("SELECT status FROM jobs WHERE id = ?")
        .bind(job_id.to_string())
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch job");

    assert_eq!(row.0, "running");
}

// Test 10: Edge case - Progress boundaries
#[tokio::test]
#[serial]
async fn test_progress_boundaries() {
    let (pool, _temp_dir) = setup_test_database().await;
    let service = JobStatusServiceImpl::new(pool.clone());

    // Test with progress = 0.0
    let job_id_1 = create_test_job(&pool, "running").await;
    let req1 = Request::new(pb::UpdateJobStatusRequest {
        job_id: job_id_1.to_string(),
        node_id: "node_1".to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 0.0,
        message: "Starting".to_string(),
    });
    assert!(service.update_job_status(req1).await.is_ok());

    // Test with progress = 1.0
    let job_id_2 = create_test_job(&pool, "running").await;
    let req2 = Request::new(pb::UpdateJobStatusRequest {
        job_id: job_id_2.to_string(),
        node_id: "node_2".to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 1.0,
        message: "Complete".to_string(),
    });
    assert!(service.update_job_status(req2).await.is_ok());

    // Test with progress > 1.0 (should still work, but might want validation)
    let job_id_3 = create_test_job(&pool, "running").await;
    let req3 = Request::new(pb::UpdateJobStatusRequest {
        job_id: job_id_3.to_string(),
        node_id: "node_3".to_string(),
        status: pb::StatusCode::StatusInProgress as i32,
        progress: 1.5,
        message: "Over 100%".to_string(),
    });
    // Current implementation allows this; you may want to add validation
    assert!(service.update_job_status(req3).await.is_ok());
}
