# CyxWiz P2P Training Workflow Design

## Overview

This document describes the **peer-to-peer (P2P) training workflow** where the CyxWiz Engine communicates directly with Server Nodes (miners) while the Central Server acts as a coordinator and payment processor.

## Architecture Principles

1. **Central Server = Coordinator** - Handles discovery, matchmaking, escrow, and rewards
2. **Engine ↔ Server Node = P2P** - Direct communication for training data and real-time metrics
3. **Trust but Verify** - Central Server tracks job lifecycle but doesn't proxy training traffic

## Complete Workflow

### Phase 1: Job Submission & Node Discovery

```
┌─────────────┐                  ┌─────────────────┐
│   Engine    │                  │ Central Server  │
└──────┬──────┘                  └────────┬────────┘
       │                                  │
       │ 1. SubmitJob(JobConfig)          │
       │─────────────────────────────────>│
       │                                  │
       │                                  │ 2. Create job in DB
       │                                  │    Status: PENDING
       │                                  │    Create payment escrow
       │                                  │
       │                                  │ 3. Find suitable nodes
       │                                  │    - Match hardware requirements
       │                                  │    - Check node availability
       │                                  │    - Consider proximity/latency
       │                                  │    - Check reputation score
       │                                  │
       │ 4. SubmitJobResponse             │
       │    - job_id                      │
       │    - node_endpoint               │
       │    - auth_token                  │
       │<─────────────────────────────────│
       │                                  │
```

**New RPC needed:**
- Response includes `node_endpoint` (IP:port) and `auth_token` for Engine→Node connection

### Phase 2: Direct Engine → Server Node Connection

```
┌─────────────┐                  ┌─────────────────┐
│   Engine    │                  │  Server Node    │
└──────┬──────┘                  └────────┬────────┘
       │                                  │
       │ 5. ConnectToNode(auth_token)     │
       │─────────────────────────────────>│
       │                                  │
       │                                  │ 6. Verify auth token
       │                                  │    with Central Server
       │                                  │
       │ 7. ConnectionAccepted            │
       │<─────────────────────────────────│
       │                                  │
       │ 8. SendJob(JobConfig, dataset)   │
       │─────────────────────────────────>│
       │                                  │
       │                                  │ 9. Validate job
       │                                  │    Accept or Reject
       │                                  │
       │ 10. JobAccepted                  │
       │<─────────────────────────────────│
       │                                  │
```

**New services needed:**
- `JobExecutionService` on Server Node (accepts jobs from Engine)
- Token verification endpoint on Central Server

### Phase 3: Server Node Reports Job Acceptance

```
┌─────────────────┐              ┌─────────────────┐
│  Server Node    │              │ Central Server  │
└────────┬────────┘              └────────┬────────┘
         │                                │
         │ 11. NotifyJobAccepted          │
         │     - node_id                  │
         │     - job_id                   │
         │     - engine_address           │
         │────────────────────────────────>│
         │                                │
         │                                │ 12. Update job status
         │                                │     Status: RUNNING
         │                                │     assigned_node_id: node_id
         │                                │     started_at: NOW
         │                                │
         │ 13. ACK                        │
         │<────────────────────────────────│
         │                                │
```

**Database updates:**
- Job status: PENDING → RUNNING
- Set `assigned_node_id`, `started_at`

### Phase 4: Training with Real-Time Streaming

```
┌─────────────┐                  ┌─────────────────┐
│   Engine    │                  │  Server Node    │
└──────┬──────┘                  └────────┬────────┘
       │                                  │
       │ 14. StreamTrainingMetrics        │
       │     (bidirectional stream)       │
       │<────────────────────────────────>│
       │                                  │
       │                                  │ Training starts
       │                                  │
       │ ← Epoch 1/10, Loss: 0.543        │
       │ ← Epoch 2/10, Loss: 0.421        │
       │ ← Epoch 3/10, Loss: 0.389        │
       │ ← Model weights (checkpoint)     │
       │ ← GPU usage: 85%, ETA: 45min     │
       │                                  │
       │   User can pause/resume ────────>│
       │                                  │
```

**Real-time metrics streamed:**
- Current epoch / total epochs
- Loss, accuracy, other metrics
- Time remaining estimate
- GPU/CPU usage
- Intermediate model weights (optional checkpoints)

**Benefits:**
- Engine shows live training visualization (graphs, logs)
- User can monitor progress in real-time
- Early stopping if user sees poor convergence

### Phase 5: Periodic Progress Reports to Central Server

```
┌─────────────────┐              ┌─────────────────┐
│  Server Node    │              │ Central Server  │
└────────┬────────┘              └────────┬────────┘
         │                                │
         │ Every 30s or per epoch:        │
         │                                │
         │ 15. ReportProgress             │
         │     - job_id                   │
         │     - progress: 0.35 (35%)     │
         │     - current_epoch: 3         │
         │     - metrics: {loss: 0.389}   │
         │────────────────────────────────>│
         │                                │
         │                                │ Update job in DB
         │                                │
         │ 16. ACK                        │
         │<────────────────────────────────│
         │                                │
```

**Purpose:**
- Central Server tracks job isn't stalled
- Detect if Server Node crashes (no reports for 2 minutes → timeout)
- Provide fallback status API for Engine (if direct connection drops)

### Phase 6: Training Completion

```
┌─────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Engine    │       │  Server Node    │       │ Central Server  │
└──────┬──────┘       └────────┬────────┘       └────────┬────────┘
       │                       │                         │
       │                       │ Training completes      │
       │                       │                         │
       │ 17. JobComplete       │                         │
       │    - final_weights    │                         │
       │    - final_metrics    │                         │
       │    - training_time    │                         │
       │<──────────────────────│                         │
       │                       │                         │
       │ 18. DownloadWeights   │                         │
       │─────────────────────>│                         │
       │                       │                         │
       │ 19. Model file        │                         │
       │<──────────────────────│                         │
       │                       │                         │
       │                       │ 20. ReportCompletion    │
       │                       │     - job_id            │
       │                       │     - result_hash       │
       │                       │     - proof_of_compute  │
       │                       │─────────────────────────>│
       │                       │                         │
       │                       │                         │ 21. Verify completion
       │                       │                         │     Update job: COMPLETED
       │                       │                         │     Release escrow payment
       │                       │                         │     90% → Server Node
       │                       │                         │     10% → Platform fee
       │                       │                         │
       │                       │ 22. PaymentConfirmation │
       │                       │    - tx_hash            │
       │                       │    - amount             │
       │                       │<─────────────────────────│
       │                       │                         │
       │ 23. Engine polls      │                         │
       │     GetJobStatus      │                         │
       │─────────────────────────────────────────────────>│
       │                       │                         │
       │ 24. JobStatus         │                         │
       │     Status: COMPLETED │                         │
       │<─────────────────────────────────────────────────│
       │                       │                         │
```

### Phase 7: Failure Handling

**Scenario A: Server Node Crashes During Training**

```
Server Node → Central Server: (No heartbeat/progress for 2 minutes)
Central Server: Marks job as FAILED, issues refund to user
Central Server → Engine: JobStatus = FAILED (via polling)
```

**Scenario B: Engine Disconnects**

```
Engine disconnects, but Server Node keeps training
Server Node → Central Server: Reports completion normally
Central Server: Job marked COMPLETED, payment released
Engine reconnects later → Can still download final weights
```

**Scenario C: User Cancels Job**

```
Engine → Central Server: CancelJob(job_id)
Central Server → Server Node: Send cancel signal via heartbeat response
Server Node: Stops training, cleans up
Central Server: Partial refund based on progress (e.g., 50% done = 50% refund)
```

## Protocol Buffer Updates Needed

### 1. Update `job.proto` - Add node discovery response

```protobuf
message SubmitJobResponse {
  string job_id = 1;
  StatusCode status = 2;

  // NEW: Node assignment for P2P connection
  NodeAssignment node_assignment = 3;

  Error error = 4;
  int64 estimated_start_time = 5;
}

message NodeAssignment {
  string node_id = 1;
  string node_endpoint = 2;  // e.g., "192.168.1.100:50052"
  string auth_token = 3;      // JWT token for Engine→Node auth
  int64 token_expires_at = 4; // Unix timestamp
}
```

### 2. Add new `execution.proto` - Engine ↔ Server Node

```protobuf
syntax = "proto3";

package cyxwiz.protocol;

import "common.proto";
import "job.proto";

// Service running on Server Node (port 50052)
service JobExecutionService {
  // Engine establishes connection with auth token
  rpc ConnectToNode(ConnectRequest) returns (ConnectResponse);

  // Engine sends job details and dataset
  rpc SendJob(SendJobRequest) returns (SendJobResponse);

  // Bidirectional streaming for real-time training updates
  rpc StreamTrainingMetrics(stream TrainingCommand) returns (stream TrainingUpdate);

  // Engine downloads final model weights
  rpc DownloadWeights(DownloadRequest) returns (stream WeightsChunk);
}

message ConnectRequest {
  string auth_token = 1;
  string job_id = 2;
  string engine_version = 3;
}

message ConnectResponse {
  StatusCode status = 1;
  string node_id = 2;
  NodeCapabilities capabilities = 3;
  Error error = 4;
}

message SendJobRequest {
  string job_id = 1;
  JobConfig config = 2;
  bytes initial_dataset = 3;  // Or dataset URI
}

message SendJobResponse {
  StatusCode status = 1;
  bool accepted = 2;
  string estimated_start_time = 3;
  Error error = 4;
}

// Engine → Server Node commands
message TrainingCommand {
  oneof command {
    bool pause = 1;
    bool resume = 2;
    bool stop = 3;
    bool request_checkpoint = 4;
  }
}

// Server Node → Engine updates
message TrainingUpdate {
  string job_id = 1;

  oneof update {
    TrainingProgress progress = 2;
    TrainingComplete complete = 3;
    TrainingError error = 4;
  }
}

message TrainingProgress {
  int32 current_epoch = 1;
  int32 total_epochs = 2;
  double progress_percentage = 3;  // 0.0 to 1.0

  map<string, double> metrics = 4;  // loss, accuracy, etc.

  double gpu_usage = 5;
  double cpu_usage = 6;
  double memory_usage = 7;

  int64 estimated_time_remaining = 8;  // seconds
  int64 elapsed_time = 9;              // seconds

  bytes checkpoint_weights = 10;  // Optional intermediate weights
}

message TrainingComplete {
  string result_hash = 1;
  map<string, double> final_metrics = 2;
  int64 total_training_time = 3;
  string weights_location = 4;  // Where to download final weights
}

message TrainingError {
  string error_message = 1;
  string stack_trace = 2;
}

message DownloadRequest {
  string job_id = 1;
}

message WeightsChunk {
  bytes data = 1;
  int64 offset = 2;
  int64 total_size = 3;
}
```

### 3. Update `node.proto` - Add job acceptance notification

```protobuf
// Add to NodeService

rpc NotifyJobAccepted(JobAcceptedRequest) returns (JobAcceptedResponse);

message JobAcceptedRequest {
  string node_id = 1;
  string job_id = 2;
  string engine_address = 3;  // Where Engine connected from
  int64 accepted_at = 4;
}

message JobAcceptedResponse {
  StatusCode status = 1;
  Error error = 2;
}
```

## Implementation Phases

### Phase 1: Protocol & Database (Week 1)
- ✅ Add `execution.proto` with `JobExecutionService`
- ✅ Update `job.proto` with `NodeAssignment`
- ✅ Update `node.proto` with `NotifyJobAccepted`
- ✅ Generate C++ and Rust bindings
- ✅ Add `auth_token` field to jobs table

### Phase 2: Central Server Updates (Week 1-2)
- ✅ Modify `SubmitJob` to include node assignment in response
- ✅ Implement JWT token generation for Engine→Node auth
- ✅ Add `NotifyJobAccepted` RPC handler
- ✅ Update job status transitions: PENDING → RUNNING → COMPLETED
- ✅ Add token verification endpoint for Server Nodes

### Phase 3: Server Node Implementation (Week 2-3)
- ✅ Implement `JobExecutionService` gRPC server (port 50052)
- ✅ Add auth token verification
- ✅ Implement `ConnectToNode` and `SendJob` RPCs
- ✅ Implement `StreamTrainingMetrics` bidirectional stream
- ✅ Integrate with job executor to send real-time updates
- ✅ Implement `DownloadWeights` streaming RPC
- ✅ Add `NotifyJobAccepted` call to Central Server

### Phase 4: Engine Implementation (Week 3-4)
- ✅ Add `JobExecutionServiceClient` for P2P connections
- ✅ Update job submission flow to receive node assignment
- ✅ Implement direct connection to Server Node
- ✅ Create real-time training visualization panel (ImPlot)
- ✅ Display live metrics (loss curves, progress bars, ETA)
- ✅ Implement pause/resume/stop controls
- ✅ Add model weights download on completion

### Phase 5: Testing & Security (Week 4-5)
- ✅ Test full P2P workflow end-to-end
- ✅ Test failure scenarios (node crash, engine disconnect)
- ✅ Implement TLS for Engine↔Node connections
- ✅ Add proof-of-compute verification
- ✅ Stress test with multiple concurrent jobs
- ✅ Latency and bandwidth optimization

## Security Considerations

1. **Authentication Token**:
   - JWT token signed by Central Server
   - Contains: job_id, node_id, expiration time
   - Server Node verifies with Central Server on first connection
   - Short-lived (5 minute expiration)

2. **TLS Encryption**:
   - All Engine↔Node communication over TLS
   - Server Node generates self-signed cert (or uses Let's Encrypt)
   - Central Server provides node's public key to Engine

3. **Proof of Compute**:
   - Server Node generates proof (hash of intermediate checkpoints)
   - Central Server can verify training actually occurred
   - Prevents nodes from claiming rewards without doing work

4. **Rate Limiting**:
   - Limit concurrent jobs per Engine wallet
   - Prevent spam job submissions
   - Detect malicious nodes

## Performance Optimizations

1. **Checkpoint Streaming**:
   - Only send checkpoints every N epochs (configurable)
   - Compress checkpoint data (gzip)

2. **Metrics Batching**:
   - Send metrics every 5 seconds, not every batch
   - Reduces network overhead

3. **Node Selection**:
   - Prioritize geographically close nodes (lower latency)
   - Consider node load balancing

4. **Fallback to Central Server**:
   - If Engine↔Node connection fails, Central Server proxies updates
   - Graceful degradation

## Benefits of P2P Design

✅ **Scalability**: Central Server doesn't bottleneck on training traffic
✅ **Low Latency**: Direct Engine↔Node connection for real-time updates
✅ **Better UX**: Live training visualization, immediate feedback
✅ **Bandwidth Efficiency**: Large model weights don't go through Central Server
✅ **Flexibility**: Engine and Node can negotiate custom protocols
✅ **Privacy**: Training data only shared between Engine and Node

## Migration from Current Design

The current centralized design can coexist during transition:

1. **Keep existing `SubmitJob`** - Works as-is
2. **Add optional `node_assignment` field** - Backward compatible
3. **Engines can detect** if Central Server supports P2P (check for `node_assignment`)
4. **Fallback mode**: If Engine doesn't support P2P, Central Server uses old flow

---

**Status**: This design replaces the centralized training flow with a P2P architecture where Engine and Server Node communicate directly while Central Server coordinates discovery, payments, and verification.

**Next Steps**: Review this design, then begin implementation starting with Protocol updates (Phase 1).
