# Phase 5.3: Job Execution Integration - Implementation Plan

**Goal**: Connect Central Server job assignment with Server Node job execution

## Architecture Overview

```
┌──────────────────────┐                    ┌───────────────────────┐
│  Central Server      │                    │   Server Node         │
│  (Rust)              │                    │   (C++)               │
│                      │                    │                       │
│  Port 50051 (gRPC)   │                    │  Port 50052 (Deploy)  │
│  ┌────────────────┐  │                    │  Port 50053 (Term)    │
│  │ NodeService    │◄─┼────Register────────┤  NodeClient           │
│  │ - RegisterNode │  │                    │                       │
│  │ - Heartbeat    │◄─┼────Heartbeat───────┤                       │
│  └────────────────┘  │                    │                       │
│                      │                    │  ┌─────────────────┐  │
│  ┌────────────────┐  │                    │  │ NodeService     │  │
│  │ Job Scheduler  │  │                    │  │ - AssignJob ◄───┼──┤
│  │                ├──┼─────AssignJob──────┼─►│                 │  │
│  └────────────────┘  │                    │  └─────────────────┘  │
│                      │                    │         │             │
│                      │                    │         ▼             │
│                      │                    │  ┌─────────────────┐  │
│  ┌────────────────┐  │                    │  │ JobExecutor     │  │
│  │ Job Database   │◄─┼───ReportProgress───┤◄─┤ - ExecuteAsync  │  │
│  └────────────────┘  │                    │  │ - Callbacks     │  │
│                      │                    │  └─────────────────┘  │
└──────────────────────┘                    └───────────────────────┘
```

## Implementation Tasks

### Task 1: Create NodeServiceImpl on Server Node ✅

**File**: `cyxwiz-server-node/src/node_service.h`
**File**: `cyxwiz-server-node/src/node_service.cpp`

**Purpose**: Implement the AssignJob RPC endpoint that:
1. Receives job assignments from Central Server
2. Validates the job configuration
3. Passes job to JobExecutor
4. Returns acceptance status

**Key Methods**:
```cpp
class NodeServiceImpl final : public protocol::NodeService::Service {
public:
    explicit NodeServiceImpl(JobExecutor* job_executor, const std::string& node_id);

    grpc::Status AssignJob(
        grpc::ServerContext* context,
        const protocol::AssignJobRequest* request,
        protocol::AssignJobResponse* response) override;

private:
    JobExecutor* job_executor_;
    std::string node_id_;
};
```

### Task 2: Integrate JobExecutor into Main Loop

**File**: `cyxwiz-server-node/src/main.cpp`

**Changes**:
1. Create JobExecutor instance
2. Create NodeServiceImpl with JobExecutor reference
3. Add NodeService to existing gRPC server (port 50052 or 50051?)
4. Set up progress and completion callbacks

**Code Outline**:
```cpp
// Create JobExecutor
auto job_executor = std::make_unique<JobExecutor>(node_id, device);

// Set up callbacks
job_executor->SetProgressCallback([&](const std::string& job_id, double progress, const TrainingMetrics& metrics) {
    // Report progress to Central Server via NodeClient
    spdlog::info("Job {} progress: {:.1f}%", job_id, progress * 100);
});

job_executor->SetCompletionCallback([&](const std::string& job_id, bool success, const std::string& error_msg) {
    // Report completion to Central Server
    spdlog::info("Job {} completed: {}", job_id, success ? "SUCCESS" : "FAILED");
});

// Create NodeServiceImpl
auto node_service = std::make_unique<NodeServiceImpl>(job_executor.get(), node_id);

// Add to gRPC server builder
grpc::ServerBuilder builder;
builder.AddListeningPort("0.0.0.0:50054", grpc::InsecureServerCredentials());  // Separate port for NodeService?
builder.RegisterService(node_service.get());
auto server = builder.BuildAndStart();
```

### Task 3: Implement Progress Reporting

**File**: `cyxwiz-server-node/src/node_client.h`
**File**: `cyxwiz-server-node/src/node_client.cpp`

**Add Methods**:
```cpp
class NodeClient {
public:
    // Existing methods...

    // New methods for job reporting
    bool ReportJobProgress(const std::string& job_id, const protocol::JobStatus& status);
    bool ReportJobCompletion(const std::string& job_id, const protocol::JobResult& result);

private:
    // Keep track of active jobs for heartbeat
    std::vector<std::string> active_jobs_;
    std::mutex jobs_mutex_;
};
```

### Task 4: Central Server Job Assignment

**File**: `cyxwiz-central-server/src/scheduler/job_queue.rs`

**Add Method**:
```rust
impl JobQueue {
    // Existing methods...

    /// Actually call AssignJob RPC on the selected node
    pub async fn send_job_to_node(&self, job_id: Uuid, node_id: Uuid) -> Result<()> {
        // Get node connection info from database
        let node_info = database::queries::get_node_by_id(&self.db_pool, node_id).await?;

        // Create gRPC client to Server Node
        let node_address = format!("http://{}:{}", node_info.ip_address, node_info.port);
        let mut client = NodeServiceClient::connect(node_address).await?;

        // Get job config from database
        let job = database::queries::get_job(&self.db_pool, job_id).await?;

        // Build AssignJobRequest
        let request = AssignJobRequest {
            node_id: node_id.to_string(),
            job: Some(job.into()),  // Convert DB Job to protocol JobConfig
            authorization_token: generate_job_token(job_id, node_id),
        };

        // Call AssignJob RPC
        let response = client.assign_job(request).await?;

        if response.into_inner().accepted {
            info!("Job {} assigned to node {} successfully", job_id, node_id);
            Ok(())
        } else {
            Err(anyhow!("Node rejected job assignment"))
        }
    }
}
```

### Task 5: Testing

**Create**: `cyxwiz-server-node/test_job_submit.py`

**Purpose**: Python script to test end-to-end job submission

```python
import grpc
from cyxwiz.protocol import job_pb2, job_pb2_grpc

def submit_test_job():
    # Connect to Central Server
    channel = grpc.insecure_channel('localhost:50051')
    stub = job_pb2_grpc.JobServiceStub(channel)

    # Create job config
    job_config = job_pb2.JobConfig(
        job_type=job_pb2.JOB_TYPE_TRAINING,
        priority=job_pb2.PRIORITY_NORMAL,
        model_definition='{"layers": [{"type": "dense", "units": 10}]}',
        hyperparameters={
            'learning_rate': '0.01',
            'optimizer': 'adam',
            'batch_size': '32',
            'epochs': '10'
        },
        dataset_uri='mock://random_data',
        batch_size=32,
        epochs=10
    )

    # Submit job
    request = job_pb2.SubmitJobRequest(config=job_config)
    response = stub.SubmitJob(request)

    print(f"Job submitted: {response.job_id}")
    print(f"Assigned to node: {response.assigned_node_id}")

    return response.job_id

if __name__ == '__main__':
    job_id = submit_test_job()
    print(f"Job {job_id} submitted successfully!")
```

## Implementation Order

1. **Create NodeServiceImpl** (Task 1)
   - Write node_service.h
   - Write node_service.cpp with AssignJob handler
   - Add to CMakeLists.txt

2. **Integrate JobExecutor** (Task 2)
   - Modify main.cpp to create JobExecutor
   - Set up callbacks
   - Start NodeService gRPC server

3. **Add Progress Reporting** (Task 3)
   - Extend NodeClient with ReportJobProgress
   - Implement in job_executor callbacks

4. **Extend Central Server** (Task 4)
   - Add send_job_to_node method
   - Call from assign_job_to_node in job_queue.rs

5. **Test End-to-End** (Task 5)
   - Write test script
   - Submit job
   - Verify execution
   - Check progress reports

## Port Assignments

Current:
- Central Server: 50051 (gRPC)
- Central Server: 8080 (REST)
- Server Node: 50052 (DeploymentService)
- Server Node: 50053 (TerminalService)

Proposed:
- **Server Node: 50051 (NodeService - for AssignJob)** ← Conflict! Central Server uses this

Alternative:
- **Server Node: 50054 (NodeService - for AssignJob)**

OR better yet, reuse existing server on 50052:
- **Server Node: 50052 (Multi-service: DeploymentService + NodeService)**

## Expected Flow

1. User submits job to Central Server (port 50051)
2. Central Server's job_queue finds suitable node
3. Central Server calls `AssignJob` RPC on Server Node (port 50052)
4. Server Node's NodeServiceImpl receives job
5. NodeServiceImpl passes job to JobExecutor
6. JobExecutor starts async training
7. JobExecutor calls progress callback
8. NodeClient reports progress to Central Server
9. JobExecutor completes job
10. JobExecutor calls completion callback
11. NodeClient reports completion to Central Server

## Success Criteria

- [x] Server Node can receive AssignJob RPC
- [ ] JobExecutor executes training job
- [ ] Progress updates sent to Central Server
- [ ] Job completion reported
- [ ] Central Server updates job status in database
- [ ] End-to-end test passes

## Files to Create/Modify

### New Files:
1. `cyxwiz-server-node/src/node_service.h`
2. `cyxwiz-server-node/src/node_service.cpp`
3. `cyxwiz-server-node/test_job_submit.py`

### Modified Files:
1. `cyxwiz-server-node/src/main.cpp`
2. `cyxwiz-server-node/src/node_client.h`
3. `cyxwiz-server-node/src/node_client.cpp`
4. `cyxwiz-server-node/CMakeLists.txt`
5. `cyxwiz-central-server/src/scheduler/job_queue.rs`

## Next Phase Preview (Phase 5.4)

Once Phase 5.3 is complete, Phase 5.4 will focus on:
- Job lifecycle management (queuing, cancellation)
- Multiple concurrent jobs
- Resource allocation
- Failure handling and retry logic
