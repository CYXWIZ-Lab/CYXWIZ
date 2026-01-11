# Updated P2P Training Flow Architecture

## Overview

This document describes the complete P2P distributed training flow for CyxWiz, covering the entire lifecycle from node discovery to payment settlement, including all failure scenarios.

**Key Principle**: Central Server is stateless for job execution - it only orchestrates node assignment, escrow management, and payment settlement. The actual training happens directly between Engine and Server Node via P2P.

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: DISCOVERY                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Engine                          Central Server                                 │
│     │                                   │                                        │
│     │  1. Connect()                     │                                        │
│     ├──────────────────────────────────>│                                        │
│     │                                   │                                        │
│     │  2. ListFreeNodes()               │                                        │
│     ├──────────────────────────────────>│                                        │
│     │                                   │  Query nodes WHERE status='online'     │
│     │                                   │  AND current_load < max_load           │
│     │  NodeList [                       │  AND reputation >= 50                  │
│     │    {id, hw_specs, price/hr,       │                                        │
│     │     reputation, availability}     │                                        │
│     │  ]                                │                                        │
│     │<──────────────────────────────────┤                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2: JOB SUBMISSION                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Engine                          Central Server              Solana Blockchain  │
│     │                                   │                           │            │
│     │  User builds model in NodeGraph   │                           │            │
│     │  User selects node from list      │                           │            │
│     │  User sets: epochs, batch_size,   │                           │            │
│     │            duration (e.g., 1hr)   │                           │            │
│     │                                   │                           │            │
│     │  3. ReserveNode(                  │                           │            │
│     │       node_id,                    │                           │            │
│     │       my_wallet,                  │                           │            │
│     │       duration_hours,             │                           │            │
│     │       job_config                  │                           │            │
│     │     )                             │                           │            │
│     ├──────────────────────────────────>│                           │            │
│     │                                   │                           │            │
│     │                                   │  4. Calculate cost:       │            │
│     │                                   │     cost = price/hr ×     │            │
│     │                                   │            duration       │            │
│     │                                   │                           │            │
│     │                                   │  5. Create escrow         │            │
│     │                                   ├──────────────────────────>│            │
│     │                                   │     Lock user tokens      │            │
│     │                                   │<──────────────────────────┤            │
│     │                                   │                           │            │
│     │                                   │  6. Mark node as BUSY     │            │
│     │                                   │     Remove from free list │            │
│     │                                   │     Store: job_id,        │            │
│     │                                   │            start_time,    │            │
│     │                                   │            end_time,      │            │
│     │                                   │            escrow_account │            │
│     │                                   │                           │            │
│     │  7. ReservationConfirmed(         │                           │            │
│     │       job_id,                     │                           │            │
│     │       node_endpoint,              │                           │            │
│     │       p2p_auth_token,             │                           │            │
│     │       escrow_account              │                           │            │
│     │     )                             │                           │            │
│     │<──────────────────────────────────┤                           │            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3: P2P CONNECTION                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Engine                                              Server Node                │
│     │                                                      │                     │
│     │  8. P2P Connect(                                     │                     │
│     │       node_endpoint,                                 │                     │
│     │       job_id,                                        │                     │
│     │       p2p_auth_token                                 │                     │
│     │     )                                                │                     │
│     ├─────────────────────────────────────────────────────>│                     │
│     │                                                      │                     │
│     │  9. SendJobConfig(                                   │                     │
│     │       model_definition,                              │                     │
│     │       hyperparameters,                               │                     │
│     │       epochs, batch_size                             │                     │
│     │     )                                                │                     │
│     ├─────────────────────────────────────────────────────>│                     │
│     │                                                      │                     │
│     │  10. StartTrainingStream()                           │                     │
│     │      [Bidirectional gRPC stream]                     │                     │
│     │<════════════════════════════════════════════════════>│                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 4: DATA TRANSFER                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Engine                       File Server                Server Node            │
│     │                              │                           │                 │
│     │  User's dataset is on Engine │                           │                 │
│     │                              │                           │                 │
│     │  11. Upload dataset chunks   │                           │                 │
│     ├─────────────────────────────>│                           │                 │
│     │                              │                           │                 │
│     │  12. DatasetReady(uri)       │                           │                 │
│     ├─────────────────────────────────────────────────────────>│                 │
│     │                              │                           │                 │
│     │                              │  13. Download dataset     │                 │
│     │                              │<──────────────────────────┤                 │
│     │                              │      chunks as needed     │                 │
│     │                              ├─────────────────────────>│                 │
│                                                                                  │
│   Alternative: Lazy Loading (Server Node pulls batches on-demand)                │
│     │                                                      │                     │
│     │<─────── RequestBatch(epoch, batch_idx) ──────────────┤                     │
│     ├─────────── BatchData(images, labels) ───────────────>│                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 5: TRAINING                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Engine (Dashboard)                                      Server Node            │
│     │                                                          │                 │
│     │                                   [Training Loop]        │                 │
│     │                                   for epoch in epochs:   │                 │
│     │                                     for batch in data:   │                 │
│     │                                       forward()          │                 │
│     │                                       backward()         │                 │
│     │                                       optimizer.step()   │                 │
│     │                                                          │                 │
│     │  14. TrainingProgress(                                   │                 │
│     │        epoch, batch,                                     │                 │
│     │        loss, accuracy,                                   │                 │
│     │        gpu_utilization                                   │                 │
│     │      )                                                   │                 │
│     │<─────────────────────────────────────────────────────────┤                 │
│     │                                                          │                 │
│     │  [Dashboard displays real-time metrics]                  │                 │
│     │                                                          │                 │
│     │  15. User sends: Pause/Resume/Cancel                     │                 │
│     ├─────────────────────────────────────────────────────────>│                 │
│     │                                                          │                 │
│     │      Pause: Save checkpoint, pause training loop         │                 │
│     │      Resume: Load checkpoint, continue training          │                 │
│     │      Cancel: Stop training, cleanup resources            │                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 6: JOB COMPLETION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Engine              Central Server              Server Node       Blockchain   │
│     │                       │                          │                │        │
│     │                       │  16. TrainingComplete(   │                │        │
│     │                       │        job_id,           │                │        │
│     │                       │        model_hash,       │                │        │
│     │                       │        final_metrics     │                │        │
│     │                       │      )                   │                │        │
│     │                       │<─────────────────────────┤                │        │
│     │                       │                          │                │        │
│     │  17. JobComplete(     │                          │                │        │
│     │        job_id,        │                          │                │        │
│     │        success=true   │                          │                │        │
│     │      )                │                          │                │        │
│     ├──────────────────────>│                          │                │        │
│     │                       │                          │                │        │
│     │                       │  18. Verify both confirmations            │        │
│     │                       │      (Engine + Node agree)                │        │
│     │                       │                          │                │        │
│     │                       │  19. Release payment                      │        │
│     │                       ├──────────────────────────────────────────>│        │
│     │                       │      90% to node_wallet  │                │        │
│     │                       │      10% to platform     │                │        │
│     │                       │<──────────────────────────────────────────┤        │
│     │                       │                          │                │        │
│     │                       │  20. Mark node as FREE   │                │        │
│     │                       │      Add back to free list                │        │
│     │                       │      Increment node reputation (+0.1)     │        │
│     │                       │                          │                │        │
│     │  21. PaymentComplete( │                          │                │        │
│     │        tx_hash        │                          │                │        │
│     │      )                │                          │                │        │
│     │<──────────────────────┤                          │                │        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Failure Scenarios & Handling

### Scenario 1: Engine Disconnects Mid-Training

```
Engine                    Central Server                    Server Node
  │                             │                                │
  │  [Engine crashes or         │                                │
  │   user closes app]          │                                │
  ╳                             │                                │
                                │  22. Heartbeat timeout         │
                                │      (no ping from Engine      │
                                │       for 60 seconds)          │
                                │                                │
                                │  23. SendStopCommand()         │
                                ├───────────────────────────────>│
                                │                                │
                                │  24. Calculate proportional    │
                                │      payment:                  │
                                │      used_time = now -         │
                                │                  start_time    │
                                │      used_cost = (used_time /  │
                                │                   total_time)  │
                                │                  × total_cost  │
                                │                                │
                                │  25. Release used_cost to node │
                                │      (90% node, 10% platform)  │
                                │                                │
                                │  26. Refund remaining to       │
                                │      engine wallet             │
                                │                                │
                                │  27. Mark node as FREE         │
```

### Scenario 2: Server Node Fails/Disconnects

```
Engine                    Central Server                    Server Node
  │                             │                                │
  │                             │  28. Heartbeat missed          │
  │                             │      (no heartbeat for 30s)    │
  │                             │                                ╳
  │                             │                                │
  │                             │  29. Mark node as OFFLINE      │
  │                             │      Decrement reputation      │
  │                             │      (-0.5 per failure)        │
  │                             │                                │
  │                             │  30. If reputation < 50:       │
  │                             │      Move to FREE_WORK_LIST    │
  │                             │      (must do free jobs to     │
  │                             │       rebuild reputation)      │
  │                             │                                │
  │  31. NodeFailed(            │                                │
  │        job_id,              │                                │
  │        reason               │                                │
  │      )                      │                                │
  │<────────────────────────────┤                                │
  │                             │                                │
  │                             │  32. FULL REFUND to engine     │
  │                             │      (nothing deducted -       │
  │                             │       node's fault)            │
  │                             │                                │
  │  33. Suggest alternative    │                                │
  │      nodes to retry         │                                │
  │<────────────────────────────┤                                │
```

### Scenario 3: User Controls Training (Pause/Resume/Cancel)

**IMPORTANT**: During the reserved time frame, training control is DIRECT P2P between Engine and Node.
The Central Server is NOT involved in pause/resume/cancel commands - it only tracks the reservation time.

```
Engine                                                    Server Node
  │                                                            │
  │  [User has reserved time, training is in progress]         │
  │                                                            │
  │  ═══════════════════════════════════════════════════════   │
  │        DIRECT P2P COMMUNICATION (No Central Server)        │
  │  ═══════════════════════════════════════════════════════   │
  │                                                            │
  │  PAUSE TRAINING:                                           │
  │  ────────────────                                          │
  │  34. PauseTraining(job_id)                                 │
  ├───────────────────────────────────────────────────────────>│
  │                                                            │
  │                                     Save checkpoint        │
  │                                     Pause training loop    │
  │                                     Keep P2P connection    │
  │                                                            │
  │  35. PauseConfirmed(checkpoint_info)                       │
  │<───────────────────────────────────────────────────────────┤
  │                                                            │
  │  [Node remains BUSY, waiting for Resume command]           │
  │  [User can take break, review metrics, etc.]               │
  │                                                            │
  │  RESUME TRAINING:                                          │
  │  ────────────────                                          │
  │  36. ResumeTraining(job_id)                                │
  ├───────────────────────────────────────────────────────────>│
  │                                                            │
  │                                     Load checkpoint        │
  │                                     Resume training loop   │
  │                                                            │
  │  37. ResumeConfirmed(current_epoch)                        │
  │<───────────────────────────────────────────────────────────┤
  │                                                            │
  │  [Training continues with progress updates...]             │
  │                                                            │
  │  CANCEL TRAINING (within reserved time):                   │
  │  ───────────────────────────────────────                   │
  │  38. CancelTraining(job_id, reason)                        │
  ├───────────────────────────────────────────────────────────>│
  │                                                            │
  │                                     Stop training          │
  │                                     Cleanup resources      │
  │                                     Reset node state       │
  │                                     Close P2P stream       │
  │                                                            │
  │  39. CancelConfirmed(                                      │
  │        epochs_completed,                                   │
  │        partial_model_available                             │
  │      )                                                     │
  │<───────────────────────────────────────────────────────────┤
  │                                                            │
  │  [Node resets to IDLE state within same reservation]       │
  │  [User can start NEW training with remaining time]         │
  │                                                            │
```

**Key Points:**
1. **No Central Server involvement** - Pause/Resume/Cancel are P2P commands
2. **Reservation time continues** - User still has their reserved time slot
3. **Node stays BUSY** - The node is still reserved for this user
4. **User can restart** - After cancel, user can start a new training job with remaining time
5. **Checkpoint support** - Pause saves checkpoint, Resume loads it
6. **No payment changes** - User already paid for the time slot, not per-job

**After Cancel - User Options:**
- Start a new training job with different parameters
- Start training on a different model
- Let reservation expire (no refund - time was reserved)
- Continue with remaining reservation time

---

## Node Reputation System

### Reputation Score: 0-100

| Score Range | Status | Actions |
|-------------|--------|---------|
| 80-100 | Premium | Priority in node list, higher rates allowed |
| 50-79 | Normal | Standard free list |
| 25-49 | Probation | FREE_WORK_LIST only (no payment) |
| 0-24 | Banned | Cannot accept any jobs |

### Temporary Ban System

**Strike System**: Track how many times a node falls below the probation threshold (< 50 reputation).

| Strikes | Consequence |
|---------|-------------|
| 1st drop below 50 | Warning, enter FREE_WORK_LIST |
| 2nd drop below 50 | Warning, must complete 5 free jobs before paid work |
| 3rd drop below 50 | **24-hour ban** from the platform |

**Ban Mechanics**:
- When a node gets banned, it cannot:
  - Accept any jobs (free or paid)
  - Appear in any node lists
  - Send heartbeats (connection refused)
- After 24 hours:
  - Ban is automatically lifted
  - Node must re-register
  - Reputation starts at 25 (Probation level)
  - Strike counter resets to 0
- If node gets banned again after recovery:
  - 48-hour ban on 4th offense
  - 1-week ban on 5th offense
  - Permanent review on 6th offense (manual approval required)

```
[Node Reputation Flow]

 100 ─────────────────────────── Premium
  │
  80 ─────────────────────────── Normal
  │
  50 ─────────────────────────── Probation threshold
  │    ↓ Strike 1: Warning
  │    ↓ Strike 2: 5 free jobs required
  │    ↓ Strike 3: 24-hour BAN
  25 ───────────────────────────
  │
   0 ─────────────────────────── Banned (permanent if stays here)
```

### Reputation Changes

| Event | Change |
|-------|--------|
| Job completed successfully | +1.0 |
| Job completed with good metrics | +0.5 bonus |
| Free work job completed | +2.0 (faster recovery) |
| Node disconnect/failure | -5.0 |
| Heartbeat timeout | -2.0 |
| User complaint | -10.0 (manual review) |

### Free Work List Rules
- Nodes with reputation < 50 enter FREE_WORK_LIST
- Must complete jobs without payment to rebuild reputation
- Each free job completed adds +2.0 reputation
- Once reputation >= 50, node moves back to normal FREE_LIST
- Platform still takes 10% of what would have been the payment (as service fee from user)

---

## State Machines

### Node States
```
                 ┌─────────────────┐
                 │    OFFLINE      │
                 │ (not connected) │
                 └────────┬────────┘
                          │ Register + Heartbeat
                          ▼
                 ┌─────────────────┐
        ┌───────│     ONLINE      │◄──────────┐
        │       │   (free list)   │           │
        │       └────────┬────────┘           │
        │                │ ReserveNode        │
        │                ▼                    │
        │       ┌─────────────────┐           │
        │       │      BUSY       │           │
        │       │ (assigned job)  │───────────┤ Job complete
        │       └────────┬────────┘           │
        │                │ Heartbeat fail     │
        │                ▼                    │
        │       ┌─────────────────┐           │
        │       │   OFFLINE       │           │
        │       │ (rep penalty)   │───────────┘ Re-register
        │       └────────┬────────┘
        │                │ Reputation < 50
        │                ▼
        │       ┌─────────────────┐
        └──────►│   FREE_WORK     │
                │ (unpaid jobs)   │───────────► Rep >= 50 → ONLINE
                └─────────────────┘
```

### Job States (Engine-side)
```
IDLE → SUBMITTING → RESERVED → CONNECTING → TRAINING ──────► COMPLETED
                       │            │           │
                       │            │           ├──► PAUSED ──► TRAINING (resume)
                       │            │           │       │
                       │            │           │       ▼
                       │            │           │   CANCELLED (user cancel while paused)
                       │            │           │
                       │            │           ▼
                       │            │       CANCELLED (user cancel)
                       │            │           │
                       │            │           ▼
                       │            │       IDLE (can start new training with remaining time)
                       │            ▼
                       │        FAILED (connection error)
                       ▼
                   FAILED (escrow error)
```

### Node States (During Reservation)
```
[Reservation starts]
       │
       ▼
   IDLE (waiting for job)
       │
       │ Engine sends job config
       ▼
   TRAINING ◄──────────────┐
       │                   │
       │ Pause command     │ Resume command
       ▼                   │
   PAUSED ─────────────────┘
       │
       │ Cancel command
       ▼
   IDLE (reset, wait for new job)
       │
       │ Reservation expires
       ▼
   FREE (available for new reservation)
```

---

## Data Structures

### Central Server - In-Memory State (Redis)

```rust
// Active job sessions (not persisted to DB)
struct ActiveSession {
    job_id: String,
    engine_wallet: String,
    node_id: String,
    node_wallet: String,
    start_time: DateTime<Utc>,
    reserved_until: DateTime<Utc>,
    escrow_account: String,
    escrow_amount: u64,
    last_engine_heartbeat: DateTime<Utc>,
    last_node_heartbeat: DateTime<Utc>,
    status: SessionStatus,
}

enum SessionStatus {
    Reserved,      // Escrow created, waiting for P2P connection
    Active,        // Training in progress
    Completing,    // Waiting for both confirmations
    Completed,     // Payment released
    Cancelled,     // User cancelled
    Failed,        // Node failed
}

// Node availability (cached from DB)
struct NodeCache {
    free_nodes: Vec<NodeId>,        // reputation >= 50
    free_work_nodes: Vec<NodeId>,   // reputation < 50
    busy_nodes: HashMap<NodeId, JobId>,
    offline_nodes: Vec<NodeId>,
    banned_nodes: Vec<NodeId>,      // Currently banned
}

// Node reputation tracking (persisted in DB)
struct NodeReputation {
    node_id: String,
    score: f64,                     // 0-100
    strike_count: i32,              // Number of times dropped below 50
    banned_until: Option<DateTime<Utc>>,  // If banned, when does it expire
    total_bans: i32,                // Lifetime ban count (for escalation)
    last_strike_at: Option<DateTime<Utc>>,
}
```

### Engine - Local State

```cpp
struct JobSession {
    std::string job_id;
    std::string node_id;
    std::string node_endpoint;
    std::string p2p_auth_token;
    std::string escrow_account;

    JobState state;
    std::chrono::time_point start_time;
    std::chrono::time_point reserved_until;

    // P2P connection
    std::shared_ptr<P2PClient> p2p_client;

    // Training progress
    int current_epoch;
    float current_loss;
    float current_accuracy;
};
```

---

## API Contracts

### Central Server gRPC Services

```protobuf
service NodeDiscoveryService {
    // Get list of available nodes (already implemented)
    rpc ListFreeNodes(ListNodesRequest) returns (ListNodesResponse);
    rpc FindNodes(FindNodesRequest) returns (FindNodesResponse);
}

service JobReservationService {
    // NEW: Reserve a specific node for training
    rpc ReserveNode(ReserveNodeRequest) returns (ReserveNodeResponse);

    // NEW: Confirm job completion (from Engine)
    rpc ConfirmJobComplete(ConfirmJobCompleteRequest) returns (ConfirmJobCompleteResponse);

    // NEW: Cancel job (from Engine)
    rpc CancelJob(CancelJobRequest) returns (CancelJobResponse);

    // NEW: Extend reservation time
    rpc ExtendReservation(ExtendReservationRequest) returns (ExtendReservationResponse);
}

service NodeStatusService {
    // Heartbeat from nodes (already implemented)
    rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);

    // NEW: Report job completion (from Node)
    rpc ReportJobComplete(ReportJobCompleteRequest) returns (ReportJobCompleteResponse);
}

// ============================================================================
// P2P SERVICE (Direct Engine <-> Node communication, NO Central Server)
// ============================================================================
// This service runs on the Server Node and is called directly by the Engine
// during the reserved time slot. Central Server is NOT involved.

service P2PTrainingService {
    // Bidirectional stream for training (already implemented as StreamTrainingMetrics)
    rpc StreamTraining(stream TrainingCommand) returns (stream TrainingProgress);

    // Training control commands (P2P direct)
    rpc PauseTraining(PauseRequest) returns (PauseResponse);
    rpc ResumeTraining(ResumeRequest) returns (ResumeResponse);
    rpc CancelTraining(CancelRequest) returns (CancelResponse);

    // Start new training job (within same reservation)
    rpc StartNewJob(StartJobRequest) returns (StartJobResponse);
}
```

### New Message Types

```protobuf
message ReserveNodeRequest {
    string node_id = 1;              // Selected node from free list
    string user_wallet = 2;          // User's Solana wallet for payment
    int32 duration_hours = 3;        // Reservation duration
    JobConfig job_config = 4;        // Model, hyperparameters, etc.
}

message ReserveNodeResponse {
    StatusCode status = 1;
    string job_id = 2;               // Unique job identifier
    string node_endpoint = 3;        // IP:port for P2P connection
    string p2p_auth_token = 4;       // JWT for P2P authentication
    string escrow_account = 5;       // Solana escrow PDA
    int64 escrow_amount = 6;         // Amount locked
    int64 reservation_expires = 7;   // Unix timestamp
    Error error = 10;
}

message ConfirmJobCompleteRequest {
    string job_id = 1;
    string model_hash = 2;           // Hash of trained model weights
    map<string, double> metrics = 3; // Final training metrics
}

message ReportJobCompleteRequest {
    string job_id = 1;
    string node_id = 2;
    string model_hash = 3;
    int64 total_epochs_completed = 4;
    int64 training_time_seconds = 5;
    map<string, double> final_metrics = 6;
}

// ============================================================================
// P2P Message Types (Direct Engine <-> Node)
// ============================================================================

message PauseRequest {
    string job_id = 1;
}

message PauseResponse {
    bool success = 1;
    string checkpoint_path = 2;      // Path to saved checkpoint
    int32 current_epoch = 3;
    int32 current_batch = 4;
    string message = 5;
}

message ResumeRequest {
    string job_id = 1;
    string checkpoint_path = 2;      // Optional: resume from specific checkpoint
}

message ResumeResponse {
    bool success = 1;
    int32 resumed_epoch = 2;
    int32 resumed_batch = 3;
    string message = 4;
}

message CancelRequest {
    string job_id = 1;
    string reason = 2;               // Optional reason for logging
    bool save_partial_model = 3;     // Whether to save current weights
}

message CancelResponse {
    bool success = 1;
    int32 epochs_completed = 2;
    bool partial_model_saved = 3;
    string partial_model_path = 4;   // Path if saved
    string message = 5;
}

message StartJobRequest {
    string job_id = 1;               // New job ID (or reuse reservation ID)
    JobConfig job_config = 2;        // New model/hyperparameters
}

message StartJobResponse {
    bool accepted = 1;
    string message = 2;
}
```

---

## Implementation Tasks

### Phase 1: Core Reservation Flow
1. [ ] Add `ReserveNode` RPC to Central Server
2. [ ] Implement node locking (mark as BUSY, remove from free list)
3. [ ] Create escrow on reservation (not job submission)
4. [ ] Return P2P connection details to Engine

### Phase 2: Session Management
5. [ ] Add `ActiveSession` tracking in Redis
6. [ ] Implement Engine heartbeat monitoring
7. [ ] Implement session timeout handling
8. [ ] Add proportional payment calculation

### Phase 3: Completion Flow
9. [ ] Add dual confirmation (Engine + Node must both confirm)
10. [ ] Implement payment release on confirmation
11. [ ] Add node reputation update on completion

### Phase 4: Failure Handling
12. [ ] Implement Engine disconnect detection
13. [ ] Implement Node failure handling
14. [ ] Add full refund mechanism for node failures
15. [ ] Add proportional refund for user cancellation

### Phase 5: Reputation & Ban System
16. [ ] Add reputation score to Node model
17. [ ] Implement FREE_WORK_LIST for low-reputation nodes
18. [ ] Add reputation recovery through free work
19. [ ] Implement reputation-based node filtering
20. [ ] Add strike tracking (increment when rep drops below 50)
21. [ ] Implement 24-hour ban on 3rd strike
22. [ ] Add ban expiration check and auto-unban
23. [ ] Implement escalating ban durations (48h, 1 week, permanent)
24. [ ] Add banned_nodes list to NodeCache

### Phase 6: UI Integration
25. [ ] Update Engine UI to show node selection from list
26. [ ] Add reservation confirmation dialog
27. [ ] Add real-time training dashboard
28. [ ] Add job control buttons (Pause/Resume/Cancel)

### Phase 7: P2P Training Controls
29. [ ] Implement PauseTraining RPC on Server Node
30. [ ] Implement ResumeTraining RPC on Server Node
31. [ ] Implement CancelTraining RPC on Server Node
32. [ ] Add checkpoint save/load for pause/resume
33. [ ] Add StartNewJob RPC for restarting within reservation
34. [ ] Update Engine UI with Pause/Resume/Cancel buttons

---

## Files to Modify

### Central Server (Rust)
| File | Changes |
|------|---------|
| `src/api/grpc/mod.rs` | Add JobReservationService |
| `src/api/grpc/reservation_service.rs` | NEW: Implement ReserveNode, ConfirmJobComplete |
| `src/scheduler/session_manager.rs` | NEW: ActiveSession tracking |
| `src/scheduler/reputation.rs` | NEW: Reputation calculations |
| `src/database/models.rs` | Add reputation_score, node status fields |
| `src/database/queries.rs` | Add reputation update queries |
| `src/blockchain/payment_processor.rs` | Add proportional payment release |
| `proto/reservation.proto` | NEW: Reservation service definitions |

### Engine (C++)
| File | Changes |
|------|---------|
| `src/network/reservation_client.h/cpp` | NEW: ReserveNode client |
| `src/network/job_manager.cpp` | Update to use reservation flow |
| `src/gui/panels/node_selection_panel.cpp` | NEW: Node selection UI |
| `src/gui/panels/training_dashboard.cpp` | Update for real-time metrics |

### Server Node (C++)
| File | Changes |
|------|---------|
| `src/node_client.cpp` | Add ReportJobComplete |
| `src/job_executor.cpp` | Add completion reporting |

---

## Authentication (Already Implemented)

The system already has JWT authentication in place:
- Engine: `AuthClient` handles login, returns JWT token
- Central Server: `JWTManager` validates tokens, `GrpcAuthInterceptor` middleware
- P2P: JWT token passed in `NodeAssignment.auth_token` for P2P connection

No additional auth work needed.

---

## Questions to Resolve

1. **Duration Extension**: Should we allow extending reservation while training? (Yes, add ExtendReservation RPC)

2. **Partial Results**: If Engine disconnects, should we save partial model weights? (Yes, checkpoint to file server)

3. **Multi-Node Training**: Future support for distributed training across multiple nodes? (Defer to v2)

4. **Minimum Duration**: What's the minimum reservation time? (Suggest: 10 minutes)

5. **Maximum Duration**: What's the maximum single reservation? (Suggest: 24 hours, can extend)
