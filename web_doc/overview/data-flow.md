# Data Flow

This document describes how data flows through the CyxWiz platform during various operations.

## Job Lifecycle

### 1. Job Creation

```
User Action: Creates model in Node Editor
                    |
                    v
    +--------------------------------+
    |        CyxWiz Engine           |
    |                                |
    |  1. Validate graph topology    |
    |  2. Generate model definition  |
    |  3. Package with dataset URI   |
    |  4. Calculate payment estimate |
    +--------------------------------+
                    |
                    v
         SubmitJobRequest
         {
           job_type: TRAINING,
           model_definition: {...},
           dataset_uri: "ipfs://...",
           hyperparameters: {...},
           payment_amount: 100.0,
           required_device: GPU
         }
```

### 2. Job Acceptance

```
                    |
                    v
    +--------------------------------+
    |      Central Server            |
    |                                |
    |  1. Validate request           |
    |  2. Check user balance         |
    |  3. Create Solana escrow       |
    |  4. Insert into job queue      |
    |  5. Return job_id              |
    +--------------------------------+
                    |
                    v
         SubmitJobResponse
         {
           job_id: "uuid-...",
           status: QUEUED,
           estimated_start: 1702000000
         }
```

### 3. Job Scheduling

```
    +--------------------------------+
    |        Job Scheduler           |
    |        (Background Task)       |
    |                                |
    |  Loop every 5 seconds:         |
    |  1. Get pending jobs           |
    |  2. Get available nodes        |
    |  3. Match by requirements      |
    |  4. Select best node           |
    |  5. Send assignment            |
    +--------------------------------+
                    |
                    v
    +--------------------------------+
    |        Node Matcher            |
    |                                |
    |  Score = f(                    |
    |    device_match,               |
    |    memory_available,           |
    |    reputation,                 |
    |    location,                   |
    |    queue_length                |
    |  )                             |
    +--------------------------------+
```

### 4. Job Assignment

```
                    |
                    v
         AssignJobRequest
         --> Server Node
                    |
                    v
    +--------------------------------+
    |        Server Node             |
    |                                |
    |  1. Validate authorization     |
    |  2. Check resource availability|
    |  3. Accept or reject           |
    |  4. Download dataset           |
    |  5. Initialize training        |
    +--------------------------------+
                    |
                    v
         AssignJobResponse
         {
           accepted: true,
           estimated_start: "now"
         }
```

### 5. Training Execution

```
    +--------------------------------+
    |        Job Executor            |
    |                                |
    |  For each epoch:               |
    |                                |
    |  1. Load batch from DataLoader |
    |  2. Forward pass               |
    |  3. Compute loss               |
    |  4. Backward pass              |
    |  5. Optimizer step             |
    |  6. Report progress            |
    +--------------------------------+
                    |
              Every N batches
                    |
                    v
         ReportProgressRequest
         {
           job_id,
           progress: 0.35,
           metrics: {
             loss: 0.342,
             accuracy: 0.876
           },
           current_epoch: 7
         }
```

### 6. Progress Streaming

```
    Central Server                    Engine
         |                              |
         |   StreamJobUpdates(job_id)   |
         |<-----------------------------|
         |                              |
         |   JobUpdateStream            |
         |----------------------------->|
         |   { progress, metrics }      |
         |                              |
         |   JobUpdateStream            |
         |----------------------------->|
         |   { progress, metrics }      |
         |                              |
        ...                            ...
```

### 7. Job Completion

```
    +--------------------------------+
    |        Server Node             |
    |                                |
    |  1. Save model weights         |
    |  2. Upload to storage          |
    |  3. Calculate final metrics    |
    |  4. Sign completion proof      |
    +--------------------------------+
                    |
                    v
         ReportCompletionRequest
         {
           job_id,
           final_status: SUCCESS,
           model_weights_uri: "ipfs://...",
           model_weights_hash: "sha256:...",
           final_metrics: {...},
           total_compute_time: 3600000,
           signature: "..."
         }
```

### 8. Payment Release

```
    +--------------------------------+
    |        Central Server          |
    |                                |
    |  1. Verify completion          |
    |  2. Validate model hash        |
    |  3. Release escrow             |
    |  4. Transfer to node wallet    |
    |  5. Update job status          |
    |  6. Notify engine              |
    +--------------------------------+
                    |
                    v
    +--------------------------------+
    |      Solana Blockchain         |
    |                                |
    |  Transaction:                  |
    |  - From: Escrow Account        |
    |  - To: Node Wallet             |
    |  - Amount: 90 CYXWIZ (90%)     |
    |                                |
    |  Transaction:                  |
    |  - From: Escrow Account        |
    |  - To: Platform Fee Account    |
    |  - Amount: 10 CYXWIZ (10%)     |
    +--------------------------------+
```

## Real-Time Updates

### Training Dashboard Updates

```
Training Loop (Python)              C++ Dashboard            ImPlot Rendering
        |                                 |                        |
        | cyxwiz.add_loss_point(e, l)     |                        |
        |-------------------------------->|                        |
        |                                 |                        |
        |                    Lock mutex    |                        |
        |                    Append data   |                        |
        |                    Unlock mutex  |                        |
        |                                 |                        |
        |                                 |   ImGui Render Loop    |
        |                                 |----------------------->|
        |                                 |                        |
        |                                 |   Lock mutex           |
        |                                 |   Read data            |
        |                                 |   Unlock mutex         |
        |                                 |                        |
        |                                 |   ImPlot::PlotLine()   |
        |                                 |                        |
```

### Node Heartbeat

```
Server Node                    Central Server                  Database
     |                              |                              |
     |   HeartbeatRequest           |                              |
     |----------------------------->|                              |
     |   { node_id, status }        |                              |
     |                              |                              |
     |                              |   UPDATE nodes               |
     |                              |   SET last_seen = NOW()      |
     |                              |----------------------------->|
     |                              |                              |
     |   HeartbeatResponse          |                              |
     |<-----------------------------|                              |
     |   { keep_alive: true }       |                              |
     |                              |                              |
    10s                            10s                            10s
     |                              |                              |
     |   HeartbeatRequest           |                              |
     |----------------------------->|                              |
    ...                            ...                            ...
```

## File Transfers

### Dataset Upload

```
User selects file in Engine
            |
            v
    +------------------+
    |  Asset Browser   |
    |                  |
    |  1. Read file    |
    |  2. Calculate    |
    |     checksum     |
    |  3. Compress     |
    |     (optional)   |
    +------------------+
            |
            v
    +------------------+
    |  IPFS Upload     |
    |                  |
    |  Returns: CID    |
    |  ipfs://Qm...    |
    +------------------+
            |
            v
    +------------------+
    |  Job Submission  |
    |                  |
    |  dataset_uri:    |
    |  "ipfs://Qm..."  |
    +------------------+
```

### Model Download

```
Training completes
            |
            v
    +------------------+
    |  Server Node     |
    |                  |
    |  1. Serialize    |
    |     model        |
    |  2. Upload to    |
    |     storage      |
    +------------------+
            |
            v
    Returns: model_weights_uri
            |
            v
    +------------------+
    |  Engine          |
    |                  |
    |  1. Get URI from |
    |     completion   |
    |  2. Download     |
    |     model        |
    |  3. Verify hash  |
    |  4. Load into    |
    |     memory       |
    +------------------+
```

## Metrics Collection

### Hardware Metrics

```
+----------------------------------+
|        Metrics Collector         |
|        (Server Node)             |
|                                  |
|  Collect every 5 seconds:        |
|                                  |
|  CPU:                            |
|  - Usage per core                |
|  - Temperature                   |
|  - Frequency                     |
|                                  |
|  GPU:                            |
|  - Utilization                   |
|  - Memory used/total             |
|  - Temperature                   |
|  - Power draw                    |
|                                  |
|  Memory:                         |
|  - RAM used/total                |
|  - Swap used/total               |
|                                  |
|  Network:                        |
|  - Bytes sent/received           |
|  - Active connections            |
+----------------------------------+
            |
            v
    Store in time-series buffer
            |
            v
    Include in heartbeat
            |
            v
    Central Server stores
    in metrics table
```

### Training Metrics

```
+----------------------------------+
|        Training Loop             |
|                                  |
|  Each Batch:                     |
|  - batch_loss                    |
|  - batch_accuracy                |
|  - batch_time_ms                 |
|  - learning_rate                 |
|                                  |
|  Each Epoch:                     |
|  - epoch_loss (avg)              |
|  - epoch_accuracy                |
|  - validation_loss               |
|  - validation_accuracy           |
|  - epoch_time_ms                 |
|                                  |
|  Training End:                   |
|  - total_time_ms                 |
|  - final_loss                    |
|  - final_accuracy                |
|  - best_epoch                    |
+----------------------------------+
```

## Caching Strategy

### Redis Cache Layers

```
Layer 1: Request Cache (TTL: 60s)
+----------------------------------+
|  Key: "job:{job_id}:status"      |
|  Value: JobStatus JSON           |
|  Purpose: Reduce DB queries      |
+----------------------------------+

Layer 2: Node Cache (TTL: 30s)
+----------------------------------+
|  Key: "node:{node_id}:info"      |
|  Value: NodeInfo JSON            |
|  Purpose: Fast node lookups      |
+----------------------------------+

Layer 3: Session Cache (TTL: 1h)
+----------------------------------+
|  Key: "session:{token}"          |
|  Value: Session data             |
|  Purpose: Auth validation        |
+----------------------------------+

Layer 4: Metrics Buffer
+----------------------------------+
|  Key: "metrics:{node_id}:latest" |
|  Value: Latest metrics           |
|  Purpose: Dashboard updates      |
+----------------------------------+
```

### Cache Invalidation

```
Event: Job Status Change
            |
            v
    +------------------+
    |  Database        |
    |  UPDATE          |
    +------------------+
            |
            v
    +------------------+
    |  Redis           |
    |  DEL key         |
    |  or              |
    |  SET with new    |
    |  value           |
    +------------------+
            |
            v
    Next read fetches
    from DB and
    repopulates cache
```

## Error Handling

### Network Errors

```
Request fails
     |
     v
+------------------+
|  Retry Logic     |
|                  |
|  Attempt 1: 1s   |
|  Attempt 2: 2s   |
|  Attempt 3: 4s   |
|  Attempt 4: 8s   |
|  Attempt 5: 16s  |
+------------------+
     |
     v
If all fail:
- Log error
- Update status
- Notify user
- Queue for later
```

### Training Errors

```
Exception in training
         |
         v
+------------------+
|  Job Executor    |
|                  |
|  1. Catch error  |
|  2. Save state   |
|  3. Report fail  |
+------------------+
         |
         v
ReportCompletionRequest
{
  final_status: FAILED,
  error_message: "...",
  last_checkpoint_uri: "..."
}
         |
         v
+------------------+
|  Central Server  |
|                  |
|  1. Update job   |
|  2. Release      |
|     partial      |
|     payment?     |
|  3. Notify user  |
+------------------+
```

---

**Next**: [Security Model](security.md) | [Architecture](architecture.md)
