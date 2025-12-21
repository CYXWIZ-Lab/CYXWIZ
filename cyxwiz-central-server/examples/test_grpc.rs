//! Simple gRPC test client for CyxWiz Central Server
//!
//! Run with: cargo run --example test_grpc

use tonic::Request;

pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}

use pb::job_service_client::JobServiceClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Connecting to gRPC server at http://127.0.0.1:50051...");

    let mut client = JobServiceClient::connect("http://127.0.0.1:50051").await?;

    println!("✓ Connected successfully!");

    // Test ListJobs
    println!("\nCalling ListJobs...");
    let request = Request::new(pb::ListJobsRequest {
        page_size: 10,
        page_token: String::new(),
        user_id: String::new(),
        filter_type: 0,
        filter_status: 0,
    });

    let response = client.list_jobs(request).await?;
    let jobs = response.into_inner();

    println!("✓ ListJobs response:");
    println!("  Total count: {}", jobs.total_count);
    println!("  Jobs returned: {}", jobs.jobs.len());

    for job in &jobs.jobs {
        println!("  - Job ID: {}", job.job_id);
        println!("    Status: {:?}", job.status);
        println!("    Progress: {:.1}%", job.progress * 100.0);
    }

    println!("\n✓ gRPC test completed successfully!");
    Ok(())
}
