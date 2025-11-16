# Phase 5: Job Execution & Scheduling

## Overview
Implement the core distributed ML training workflow - job submission, scheduling, execution, and result retrieval.

## Architecture

```
Engine (User) → Central Server → Server Node
     │              │                 │
     │   Submit Job │                 │
     │─────────────>│                 │
     │              │   Assign Job    │
     │              │────────────────>│
     │              │                 │
     │              │   (Training)    │
     │              │                 │
     │              │<────────────────│
     │              │   Report Progress
     │<─────────────│                 │
     │   Stream Updates               │
```

## Components

### 1. Central Server - Job Scheduler (Rust)
**File**: `cyxwiz-central-server/src/scheduler/job_matcher.rs` (already exists with basic structure)

**Features to Implement:**
- [x] Job queue management (basic structure exists)
- [ ] Node selection algorithm based on:
  - Hardware requirements (GPU/CPU, memory, compute)
  - Node availability and load
  - Cost estimation
- [ ] Job assignment logic
- [ ] Job lifecycle management (pending → assigned → running → completed/failed)
- [ ] Job timeout and retry logic

**Methods Needed:**
```rust
pub struct JobScheduler {
    db_pool: DbPool,
    redis_cache: RedisCache,
    config: SchedulerConfig,
}

impl JobScheduler {
    // Already exists
    pub fn new(...) -> Self { ... }

    // TODO: Implement
    pub async fn enqueue_job(&self, job: Job) -> Result<()>;
    pub async fn find_suitable_node(&self, job: &Job) -> Result<Option<Uuid>>;
    pub async fn assign_job_to_node(&self, job_id: Uuid, node_id: Uuid) -> Result<()>;
    pub async fn handle_job_completion(&self, job_id: Uuid, success: bool) -> Result<()>;
    pub async fn handle_node_failure(&self, node_id: Uuid) -> Result<()>;
}
```

### 2. Central Server - Job Service (Rust)
**File**: `cyxwiz-central-server/src/api/grpc/job_service.rs` (exists with TODO markers)

**Endpoints:**
- [x] `SubmitJob` - Submit training job (basic impl exists, needs scheduler integration)
- [x] `GetJobStatus` - Query job status (implemented)
- [x] `CancelJob` - Cancel job (implemented)
- [ ] `StreamJobUpdates` - Real-time progress streaming (marked TODO)
- [ ] `ListJobs` - List user's jobs (marked TODO)

**TODO Locations:**
- Line 282: `StreamJobUpdates` implementation
- Line 289: `ListJobs` implementation
- Line 136-143: Integrate scheduler for job assignment after payment escrow

### 3. Server Node - Job Executor (C++)
**File**: `cyxwiz-server-node/src/job_executor.cpp` (marked TODO in main.cpp)

**Features to Implement:**
```cpp
class JobExecutor {
public:
    JobExecutor(const std::string& node_id, cyxwiz::Device* device);

    // Core execution
    bool ExecuteJob(const Job& job);
    void CancelJob(const std::string& job_id);

    // Progress reporting
    void ReportProgress(const std::string& job_id, float progress);
    void ReportMetrics(const std::string& job_id, const Metrics& metrics);

    // Result handling
    void SaveResults(const std::string& job_id, const TrainingResults& results);

private:
    std::string node_id_;
    cyxwiz::Device* device_;
    std::unordered_map<std::string, std::thread> active_jobs_;

    // Training loop
    void RunTraining(const JobConfig& config);
    void LoadDataset(const std::string& dataset_uri);
    cyxwiz::Model* BuildModel(const std::string& model_definition);
};
```

### 4. Server Node - Job RPC Handler (C++)
**File**: `cyxwiz-server-node/src/node_server.cpp` (marked TODO)

**gRPC Service Implementation:**
```cpp
class NodeServiceImpl final : public cyxwiz::protocol::NodeService::Service {
    Status ExecuteJob(ServerContext* context,
                     const ExecuteJobRequest* request,
                     ExecuteJobResponse* response) override {
        // 1. Validate job request
        // 2. Queue job for execution
        // 3. Return job acceptance status
    }

    Status GetJobProgress(ServerContext* context,
                         const JobProgressRequest* request,
                         JobProgressResponse* response) override {
        // Return current training metrics
    }

    Status CancelJob(ServerContext* context,
                    const CancelJobRequest* request,
                    CancelJobResponse* response) override {
        // Stop training and cleanup
    }
};
```

### 5. Protocol Extensions (Protobuf)
**File**: Check `cyxwiz-protocol/proto/job.proto` for existing messages

**Verify These Messages Exist:**
- `JobConfig` - Training configuration (model, dataset, hyperparameters)
- `JobStatus` - Current job state and metrics
- `ExecuteJobRequest/Response` - Node receives job assignment
- `JobProgressUpdate` - Streaming progress data

**May Need to Add:**
```protobuf
message TrainingMetrics {
    int32 current_epoch = 1;
    int32 total_epochs = 2;
    float loss = 3;
    float accuracy = 4;
    float learning_rate = 5;
    int64 samples_processed = 6;
    int64 time_elapsed_ms = 7;
}

message JobResult {
    string job_id = 1;
    bool success = 2;
    string error_message = 3;
    TrainingMetrics final_metrics = 4;
    string model_checkpoint_uri = 5;
    map<string, float> evaluation_metrics = 6;
}
```

### 6. Database Queries (Rust)
**File**: `cyxwiz-central-server/src/database/queries.rs`

**Add These Functions:**
```rust
// Job assignment
pub async fn assign_job(pool: &DbPool, job_id: Uuid, node_id: Uuid) -> Result<()>;
pub async fn update_job_status(pool: &DbPool, job_id: Uuid, status: JobStatus) -> Result<()>;
pub async fn get_pending_jobs(pool: &DbPool) -> Result<Vec<Job>>;
pub async fn get_available_nodes(pool: &DbPool) -> Result<Vec<Node>>;

// Progress tracking
pub async fn update_job_progress(pool: &DbPool, job_id: Uuid, progress: f32, metrics: &str) -> Result<()>;
pub async fn record_job_completion(pool: &DbPool, job_id: Uuid, success: bool, result_hash: &str) -> Result<()>;

// Node load management
pub async fn get_node_current_jobs(pool: &DbPool, node_id: Uuid) -> Result<Vec<Job>>;
pub async fn update_node_load(pool: &DbPool, node_id: Uuid, cpu_usage: f32, gpu_usage: f32, memory_usage: f32) -> Result<()>;
```

## Implementation Phases

### Phase 5.1: Job Scheduling Infrastructure (Central Server)
**Priority**: HIGH
**Estimated Time**: 4-6 hours

**Tasks:**
1. Implement `JobScheduler::find_suitable_node()` with node matching algorithm
2. Implement `JobScheduler::assign_job_to_node()`
3. Add database queries for job assignment
4. Test scheduler with mock jobs and nodes

**Acceptance Criteria:**
- Scheduler can select appropriate node based on job requirements
- Jobs are assigned to nodes in database
- Scheduler handles no available nodes gracefully

### Phase 5.2: Server Node Job Execution (C++)
**Priority**: HIGH
**Estimated Time**: 6-8 hours

**Tasks:**
1. Create `JobExecutor` class with training loop
2. Integrate `cyxwiz-backend` Model API for training
3. Implement dataset loading from URI
4. Add progress reporting callback
5. Test local job execution without network

**Acceptance Criteria:**
- Server Node can execute a simple training job
- Progress is reported periodically
- Results are saved locally
- Errors are handled gracefully

### Phase 5.3: Node-Server Communication (gRPC)
**Priority**: HIGH
**Estimated Time**: 4-6 hours

**Tasks:**
1. Implement `ExecuteJob` RPC in Central Server
2. Implement job reception in Server Node
3. Add progress streaming from Node to Server
4. Test end-to-end job execution

**Acceptance Criteria:**
- Central Server can send job to Server Node via gRPC
- Server Node acknowledges job receipt
- Progress updates stream back to Central Server
- Job completion is recorded

### Phase 5.4: Job Lifecycle Management
**Priority**: MEDIUM
**Estimated Time**: 3-4 hours

**Tasks:**
1. Implement job timeout handling
2. Add job cancellation support
3. Implement automatic retry on failure
4. Add job history and logging

**Acceptance Criteria:**
- Jobs timeout after configured duration
- Users can cancel running jobs
- Failed jobs retry automatically (configurable)
- Job history is preserved in database

### Phase 5.5: Integration Testing
**Priority**: HIGH
**Estimated Time**: 2-3 hours

**Tasks:**
1. Create end-to-end test scenario
2. Test multiple concurrent jobs
3. Test node failure handling
4. Performance benchmarking

**Acceptance Criteria:**
- Multiple jobs can run concurrently on different nodes
- System handles node disconnection gracefully
- Job results are accurate and reproducible

## Testing Strategy

### Unit Tests
```rust
// Central Server - Scheduler
#[tokio::test]
async fn test_find_suitable_node() {
    // Test node selection algorithm
}

#[tokio::test]
async fn test_job_assignment() {
    // Test job assignment logic
}
```

```cpp
// Server Node - Executor
TEST_CASE("JobExecutor can run simple training") {
    // Test basic training execution
}

TEST_CASE("JobExecutor reports progress") {
    // Test progress reporting
}
```

### Integration Tests
1. **Full Workflow Test**:
   - Start Central Server
   - Start 2 Server Nodes (different capabilities)
   - Submit 3 jobs with different requirements
   - Verify correct node assignment
   - Verify all jobs complete successfully

2. **Failure Recovery Test**:
   - Start Central Server and 1 Server Node
   - Submit job
   - Kill Server Node mid-execution
   - Verify job is rescheduled to another node

3. **Load Balancing Test**:
   - Start Central Server and 3 Server Nodes
   - Submit 10 jobs rapidly
   - Verify even distribution across nodes
   - Verify no node is overloaded

## Success Metrics

- [ ] Central Server can schedule jobs to appropriate nodes
- [ ] Server Nodes can execute ML training jobs using cyxwiz-backend
- [ ] Progress is streamed in real-time from Node → Server → (eventually) Engine
- [ ] Jobs complete successfully and results are stored
- [ ] System handles node failures and job retries
- [ ] Multiple concurrent jobs work without conflicts
- [ ] End-to-end latency < 2 seconds for job submission to execution start

## Dependencies

**Rust Crates** (already in Cargo.toml):
- `tokio` - Async runtime
- `sqlx` - Database access
- `tonic` - gRPC
- `serde_json` - Job metadata serialization

**C++ Libraries** (already available):
- `cyxwiz-backend` - ML training
- `grpc` - Network communication
- `spdlog` - Logging

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Job execution hangs | HIGH | Implement timeout mechanism |
| Node crashes during training | HIGH | Auto-retry with different node |
| Dataset loading fails | MEDIUM | Validate URI before assignment |
| Memory overflow on node | MEDIUM | Check available memory before assignment |
| Concurrent job conflicts | MEDIUM | Job queue with proper locking |

## Next Steps After Phase 5

- **Phase 6**: Engine GUI integration (job submission UI, progress visualization)
- **Phase 7**: Blockchain payments integration
- **Phase 8**: Model marketplace and result verification

## Getting Started

1. Review existing code:
   - `cyxwiz-central-server/src/scheduler/job_matcher.rs` (basic structure)
   - `cyxwiz-central-server/src/api/grpc/job_service.rs` (SubmitJob impl)
   - `cyxwiz-protocol/proto/job.proto` (job messages)

2. Start with Phase 5.1 (Job Scheduling Infrastructure)

3. Create feature branch:
   ```bash
   git checkout -b phase5-job-execution
   ```

Ready to begin implementation!
