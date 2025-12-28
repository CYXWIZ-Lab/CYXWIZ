# Dataset Streaming Service - Implementation

**Status: Implemented** (December 2024)

## Quick Start

### How to Use Remote Dataset Streaming

1. **Load a dataset in the Engine**
   - Use the Dataset Panel to load MNIST, CIFAR-10, or a custom dataset
   - Or load via Python: `dataset = cyxwiz.load_mnist("path/to/mnist")`

2. **Submit a job with `remote://` URI**
   - Open Server Connection dialog
   - The default Dataset URI is now `remote://engine`
   - Click "Submit Job"

3. **Automatic Registration**
   - When you submit with `remote://`, the Engine automatically:
     - Gets the first loaded dataset from DataRegistry
     - Registers it for streaming to Server Node
   - Server Node will request batches on-demand during training

### Supported Dataset URI Formats

| URI Format | Description | Status |
|------------|-------------|--------|
| `remote://engine` | Lazy-load from Engine via P2P | ✓ Implemented |
| `mock://mnist` | Generates fake MNIST-like data | ✓ Implemented |
| `mock://cifar10` | Generates fake CIFAR-10-like data | ✓ Implemented |
| `file://path/to/data` | Local file on Server Node | Partial |
| `ipfs://Qm...` | IPFS hash reference | Planned |

## Overview

Implement a **lazy-loading dataset streaming service** where the Engine acts as a data server and the Server Node requests batches on-demand during training - like reading from disk, but over the network.

## Key Design Decisions

Based on user requirements:
- **Extend existing P2P connection** (reuse JWT security, no extra port)
- **Raw bytes (float32)** serialization (simple, fast)
- **Implement now** with simulated training (infrastructure ready for real training)

## Key Design Principle

**Reverse the data flow**: Instead of Engine pushing dataset to Server Node, the Server Node **pulls batches on-demand** from Engine during training.

```
Traditional (Current):     Engine ──push dataset──> Server Node
Lazy Loading (New):        Engine <──request batch── Server Node
```

## Architecture (Extended P2P Bidirectional Stream)

The existing `StreamTrainingMetrics` bidirectional stream is extended to handle **both** training updates AND dataset requests:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              ENGINE                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     P2PClient (Existing)                         │    │
│  │                                                                  │    │
│  │  StreamTrainingMetrics() ←── bidirectional stream                │    │
│  │      │                                                           │    │
│  │      ├── Receives: TrainingUpdate, DatasetRequest               │    │
│  │      └── Sends:    TrainingCommand, BatchResponse               │    │
│  │                                                                  │    │
│  │  DatasetProvider (NEW):                                          │    │
│  │      OnDatasetRequest(req) → BatchResponse from DataRegistry     │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ▲                                          │
│                              │  Existing P2P connection (:50052)        │
│                              │  JWT authenticated                       │
│                              ▼                                          │
┌─────────────────────────────────────────────────────────────────────────┐
│                           SERVER NODE                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              JobExecutionService (Existing)                      │    │
│  │                                                                  │    │
│  │  StreamTrainingMetrics() ←── bidirectional stream                │    │
│  │      │                                                           │    │
│  │      ├── Sends:    TrainingUpdate, DatasetRequest               │    │
│  │      └── Receives: TrainingCommand, BatchResponse               │    │
│  │                                                                  │    │
│  │  RemoteDataLoader (NEW):                                         │    │
│  │      GetNextBatch() → sends DatasetRequest, waits for BatchData │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits of Extended P2P:**
- Reuses existing JWT authentication
- No additional port needed
- Single connection for all P2P communication
- Already has bidirectional streaming infrastructure

## Data Flow During Training

```
Phase 1: Job Submission
─────────────────────────────────────────────────────────────────────────
Engine                    Central Server              Server Node
   │                           │                           │
   │  SubmitJob(config)        │                           │
   │  dataset_uri="remote://   │                           │
   │    engine:50060/job123"   │                           │
   ├──────────────────────────>│                           │
   │                           │                           │
   │  NodeAssignment           │                           │
   │<──────────────────────────┤                           │

Phase 2: P2P Connection + Dataset Info
─────────────────────────────────────────────────────────────────────────
Engine                                            Server Node
   │                                                   │
   │<────────── ConnectToNode(jwt) ────────────────────┤
   │                                                   │
   │<────────── GetDatasetInfo(job_id) ────────────────┤
   │                                                   │
   │  DatasetInfoResponse:                             │
   │    train: {num_samples: 50000}                    │
   │    val:   {num_samples: 10000}                    │
   │    test:  {available: false}  ← stays on Engine   │
   │    shape: [1, 28, 28]                             │
   │───────────────────────────────────────────────────>

Phase 3: Training with Lazy Loading
─────────────────────────────────────────────────────────────────────────
Engine                                            Server Node
   │                                                   │
   │                               [Start training]    │
   │                               for epoch in epochs:│
   │                                                   │
   │<── GetBatch(TRAIN, [0,1,2...31]) ─────────────────┤ Batch 0
   │────── BatchResponse(images, labels) ──────────────>
   │                               model.train(batch)  │
   │                                                   │
   │<── GetBatch(TRAIN, [32,33...63]) ─────────────────┤ Batch 1
   │────── BatchResponse(images, labels) ──────────────>
   │                               model.train(batch)  │
   │                               ...                 │
   │                                                   │
   │<── GetBatch(VAL, [0,1,2...31]) ────────────────────┤ Validation
   │────── BatchResponse ───────────────────────────────>
   │                               eval(batch)         │

Phase 4: Testing (Local on Engine)
─────────────────────────────────────────────────────────────────────────
Engine                                            Server Node
   │                                                   │
   │<────────── TrainingComplete ──────────────────────┤
   │<────────── DownloadWeights ───────────────────────┤
   │                                                   │
   │  [Load model locally]                             │
   │  [Test on local test set - FREE, PRIVATE]         │
```

## Protocol Design (execution.proto extension)

Extend the existing `TrainingCommand` and `TrainingUpdate` messages to include dataset operations:

```protobuf
// Modify execution.proto - Extend existing messages

// ============================================================================
// Dataset Split Enum
// ============================================================================

enum DatasetSplit {
  SPLIT_TRAIN = 0;
  SPLIT_VALIDATION = 1;
  SPLIT_TEST = 2;
}

// ============================================================================
// Dataset Streaming Messages (for lazy-loading from Engine)
// ============================================================================

// Server Node → Engine: Request dataset metadata
message DatasetInfoRequest {
  string job_id = 1;
  string auth_token = 2;  // JWT for verification
}

// Engine → Server Node: Dataset metadata response
message DatasetInfoResponse {
  StatusCode status = 1;

  // Per-split information
  SplitInfo train = 2;
  SplitInfo validation = 3;
  SplitInfo test = 4;

  // Common metadata
  repeated int32 sample_shape = 10;   // [1, 28, 28] for MNIST
  repeated int32 label_shape = 11;    // [10] for one-hot, [1] for class idx
  string dtype = 12;                   // "float32"
  int32 num_classes = 13;              // 10 for MNIST
  repeated string class_names = 14;    // ["0", "1", ..., "9"]

  Error error = 20;
}

// Information about a dataset split
message SplitInfo {
  int64 num_samples = 1;      // 50000 for MNIST train
  bool available = 2;         // false if kept private (e.g., test)
  bool shuffle_enabled = 3;   // Server can request shuffled
}

// Server Node → Engine: Request a batch of samples
message BatchRequest {
  string job_id = 1;
  string auth_token = 2;
  DatasetSplit split = 3;             // TRAIN, VALIDATION, or TEST
  repeated int64 sample_indices = 4;  // Which samples to get [0, 5, 23, 100, ...]
  int32 request_id = 5;               // For matching responses to requests
}

// Engine → Server Node: Batch data response
message BatchResponse {
  StatusCode status = 1;
  int32 request_id = 2;               // Matches the request

  bytes images = 3;                   // Flattened tensor data (float32)
  bytes labels = 4;                   // Flattened labels
  repeated int64 batch_shape = 5;     // [batch_size, C, H, W]
  repeated int64 label_shape = 6;     // [batch_size, num_classes] or [batch_size]

  Error error = 10;
}

// ============================================================================
// Extended TrainingCommand (Engine → Server Node)
// Now includes batch responses for dataset streaming
// ============================================================================

message TrainingCommand {
  oneof command {
    // Training control commands
    bool pause = 1;
    bool resume = 2;
    bool stop = 3;
    bool request_checkpoint = 4;
    UpdateHyperparameters update_params = 5;

    // Dataset streaming responses (Engine sends batch data to Server Node)
    BatchResponse batch_response = 10;
    DatasetInfoResponse dataset_info_response = 11;
  }
}

// ============================================================================
// Extended TrainingUpdate (Server Node → Engine)
// Now includes dataset requests
// ============================================================================

message TrainingUpdate {
  string job_id = 1;
  int64 timestamp = 2;

  oneof update {
    // Existing updates
    TrainingProgress progress = 3;
    TrainingCheckpoint checkpoint = 4;
    TrainingComplete complete = 5;
    TrainingError error = 6;
    LogMessage log = 7;

    // NEW: Dataset requests (Server Node requests data from Engine)
    DatasetInfoRequest dataset_info_request = 10;
    BatchRequest batch_request = 11;
  }
}
```

## Implementation Components

### 1. Engine: DatasetProvider

**File**: `cyxwiz-engine/src/network/dataset_provider.h/.cpp`

```cpp
class DatasetProvider {
public:
    DatasetProvider(DataRegistry* registry);

    // Register dataset for a job (called before job submission)
    void RegisterDataset(const std::string& job_id,
                         const std::string& dataset_name,
                         const DatasetConfig& config);

    void UnregisterDataset(const std::string& job_id);

    // Handle incoming requests from Server Node
    DatasetInfoResponse HandleDatasetInfoRequest(const DatasetInfoRequest& request);
    BatchResponse HandleBatchRequest(const BatchRequest& request);

private:
    DataRegistry* registry_;
    struct RegisteredDataset {
        std::string dataset_name;
        DatasetConfig config;
        int64_t train_samples;
        int64_t val_samples;
        int64_t test_samples;
        std::vector<int32_t> sample_shape;
        int32_t num_classes;
    };
    std::map<std::string, RegisteredDataset> datasets_;  // job_id -> dataset
    std::mutex mutex_;
};
```

### 2. Engine: Integration with P2PClient

**File**: `cyxwiz-engine/src/network/p2p_client.cpp` (modify)

In the `StreamTrainingMetrics` handler, add logic to handle dataset requests:
- When receiving `dataset_info_request`, call `DatasetProvider::HandleDatasetInfoRequest`
- When receiving `batch_request`, call `DatasetProvider::HandleBatchRequest`
- Send responses back through the bidirectional stream

### 3. Server Node: RemoteDataLoader

**File**: `cyxwiz-server-node/src/remote_data_loader.h/.cpp`

```cpp
class RemoteDataLoader {
public:
    RemoteDataLoader(std::shared_ptr<StreamWriter> stream_writer,
                     std::shared_ptr<ResponseQueue> response_queue,
                     const std::string& job_id,
                     DatasetSplit split,
                     int batch_size,
                     bool shuffle = true);

    // Initialize by requesting dataset info
    bool Initialize();

    // Iterator interface
    bool HasNextBatch();
    Batch GetNextBatch();
    void Reset();  // Start new epoch

    // Info
    int64_t NumSamples() const { return num_samples_; }
    int64_t NumBatches() const;
    int CurrentEpoch() const { return current_epoch_; }

private:
    std::shared_ptr<StreamWriter> stream_writer_;
    std::shared_ptr<ResponseQueue> response_queue_;
    std::string job_id_;
    DatasetSplit split_;
    int batch_size_;
    bool shuffle_;

    int64_t num_samples_;
    std::vector<int64_t> indices_;
    int current_batch_idx_;
    int current_epoch_;
    int next_request_id_;
};
```

### 4. Server Node: Integration with JobExecutor

**File**: `cyxwiz-server-node/src/job_executor.cpp` (modify)

```cpp
bool JobExecutor::LoadDataset(const std::string& dataset_uri, ...) {
    // Existing handlers...
    if (dataset_uri.find("mock://") == 0) { ... }
    if (dataset_uri.find("file://") == 0) { ... }

    // NEW: Remote dataset from Engine (lazy loading)
    if (dataset_uri.find("remote://") == 0) {
        // Create remote loaders that request data through P2P stream
        train_loader_ = std::make_unique<RemoteDataLoader>(
            stream_writer_, response_queue_, job_id_,
            SPLIT_TRAIN, config.batch_size(), /*shuffle=*/true);

        val_loader_ = std::make_unique<RemoteDataLoader>(
            stream_writer_, response_queue_, job_id_,
            SPLIT_VALIDATION, config.batch_size(), /*shuffle=*/false);

        // Initialize to get dataset info
        if (!train_loader_->Initialize() || !val_loader_->Initialize()) {
            return false;
        }

        return true;
    }

    return false;
}
```

## Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `cyxwiz-engine/src/network/dataset_provider.h` | DatasetProvider interface |
| `cyxwiz-engine/src/network/dataset_provider.cpp` | DatasetProvider implementation |
| `cyxwiz-server-node/src/remote_data_loader.h` | RemoteDataLoader interface |
| `cyxwiz-server-node/src/remote_data_loader.cpp` | RemoteDataLoader implementation |

### Modify Files
| File | Changes |
|------|---------|
| `cyxwiz-protocol/proto/execution.proto` | Add dataset streaming messages |
| `cyxwiz-engine/src/network/p2p_client.cpp` | Handle dataset requests in stream |
| `cyxwiz-engine/src/network/p2p_client.h` | Add DatasetProvider member |
| `cyxwiz-engine/CMakeLists.txt` | Add new source files |
| `cyxwiz-server-node/src/job_executor.cpp` | Add remote:// URI handler |
| `cyxwiz-server-node/src/job_executor.h` | Add RemoteDataLoader members |
| `cyxwiz-server-node/src/job_execution_service.cpp` | Route dataset responses to RemoteDataLoader |
| `cyxwiz-server-node/CMakeLists.txt` | Add new source files |

## Implementation Order

1. **Protocol**
   - Add messages to execution.proto
   - Rebuild protocol library

2. **Engine DatasetProvider**
   - Create DatasetProvider class
   - Implement HandleDatasetInfoRequest
   - Implement HandleBatchRequest
   - Add to CMakeLists

3. **Engine P2PClient Integration**
   - Add DatasetProvider member
   - Handle dataset requests in stream handler
   - Send responses through stream

4. **Server Node RemoteDataLoader**
   - Create RemoteDataLoader class
   - Implement batch requesting via stream
   - Implement response waiting
   - Add iterator interface

5. **Server Node JobExecutor Integration**
   - Add remote:// URI handler
   - Use RemoteDataLoader in training loop
   - Route responses from stream to loaders

6. **Testing**
   - Test with MNIST dataset
   - Verify batch streaming works
   - Check memory usage

## Key Features

### Memory Efficient
- Only current batch in memory at a time
- No full dataset download
- Works with datasets larger than RAM

### Shuffle Support
- Server Node decides shuffle order
- Requests indices in shuffled order
- Different shuffle each epoch

### Test Set Privacy
- Test split stays on Engine
- Server Node only gets train + val
- User tests locally (free)

## Questions Resolved

| Question | Answer |
|----------|--------|
| Where is training? | Server Node |
| Where is validation? | Server Node (lazy loaded from Engine) |
| Where is testing? | Engine (local, free, private) |
| What data is streamed? | Train + Validation batches on-demand |
| Who controls shuffle? | Server Node (requests indices) |
| Memory usage? | Only current batch in memory |

## Implementation Summary

### Files Created

| File | Description |
|------|-------------|
| `cyxwiz-engine/src/network/dataset_provider.h` | DatasetProvider class that handles batch requests |
| `cyxwiz-engine/src/network/dataset_provider.cpp` | Implementation of HandleDatasetInfoRequest and HandleBatchRequest |
| `cyxwiz-server-node/src/remote_data_loader.h` | RemoteDataLoader class for lazy-loading from Engine |
| `cyxwiz-server-node/src/remote_data_loader.cpp` | Iterator interface with GetNextBatch, Reset, etc. |

### Files Modified

| File | Changes |
|------|---------|
| `cyxwiz-protocol/proto/execution.proto` | Added DatasetSplit, BatchRequest, BatchResponse, DatasetInfoRequest, DatasetInfoResponse, SplitInfo messages. Extended TrainingCommand and TrainingUpdate with dataset fields. |
| `cyxwiz-engine/src/network/p2p_client.h` | Added DatasetProvider member, RegisterDatasetForJob/UnregisterDatasetForJob methods |
| `cyxwiz-engine/src/network/p2p_client.cpp` | Added HandleDatasetRequest, routes dataset_info_request and batch_request from stream |
| `cyxwiz-engine/CMakeLists.txt` | Added dataset_provider.cpp/.h |
| `cyxwiz-server-node/src/job_execution_service.h` | Added RemoteDataLoader forward declaration, added train_loader/val_loader to JobSession |
| `cyxwiz-server-node/src/job_execution_service.cpp` | Create RemoteDataLoaders for remote:// URIs, handle batch_response and dataset_info_response in command thread, use loaders in simulated training |
| `cyxwiz-server-node/CMakeLists.txt` | Added remote_data_loader.cpp |

### How It Works

1. **Job Submission**: Engine submits job with `dataset_uri="remote://..."`
2. **Stream Setup**: Server Node creates RemoteDataLoaders linked to the bidirectional stream
3. **Dataset Info**: Server Node sends DatasetInfoRequest, Engine responds with metadata
4. **Training Loop**: For each batch:
   - Server Node sends BatchRequest with sample indices
   - Engine fetches data from DataRegistry and sends BatchResponse
   - Server Node receives batch and continues training
5. **Epoch End**: Server Node resets and reshuffles indices

### Testing

To test the dataset streaming:

1. **Start all services**
   ```bash
   # Terminal 1: Central Server
   cd cyxwiz-central-server && cargo run --release

   # Terminal 2: Server Node Daemon
   ./build/bin/Release/cyxwiz-server-daemon.exe

   # Terminal 3: Engine
   ./build/bin/Release/cyxwiz-engine.exe
   ```

2. **Load a dataset in the Engine**
   - Open Dataset Panel
   - Click "Add Dataset" → Select MNIST or load custom data
   - Wait for dataset to load

3. **Submit a job**
   - Open Server Connection dialog (Network menu)
   - Connect to Central Server (localhost:50051)
   - Ensure Dataset URI is `remote://engine`
   - Click "Submit Job"

4. **Watch the logs**
   - Engine logs: `[P2P WORKFLOW] STEP 3c: Registered dataset for lazy streaming`
   - Server Node logs: `Setting up RemoteDataLoader for lazy data streaming`
   - Engine logs: `Received BatchRequest for job ..., N indices`
   - Server Node logs: `Processed batch X (Y samples)`

### Current Implementation Status

| Component | File | Status |
|-----------|------|--------|
| Protocol messages | `execution.proto` | ✓ Complete |
| DatasetProvider | `dataset_provider.cpp` | ✓ Complete |
| P2P batch handling | `p2p_client.cpp` | ✓ Complete |
| RemoteDataLoader | `remote_data_loader.cpp` | ✓ Complete |
| Job submission with remote:// | `connection_dialog.cpp` | ✓ Complete |
| Training loop with batches | `job_execution_service.cpp` | ✓ Simulated |
| Real ArrayFire training | `job_executor.cpp` | ⏳ TODO |

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No datasets loaded" warning | No dataset in DataRegistry | Load a dataset in Engine first |
| "Job not found" error | Job ID not in active_jobs_ | Ensure job was properly submitted |
| Timeout waiting for batch | Network or processing delay | Increase timeout in RemoteDataLoader |
| Empty batch response | Dataset indices out of range | Check dataset size matches expected |
