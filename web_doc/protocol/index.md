# Protocol Reference

This document provides comprehensive documentation for the gRPC protocol definitions used for communication between CyxWiz components.

## Overview

All network communication in CyxWiz uses gRPC with Protocol Buffers:

```
Engine <--gRPC--> Central Server <--gRPC--> Server Node
```

## Protocol Files

| File | Purpose |
|------|---------|
| `common.proto` | Shared types and enums |
| `job.proto` | Job submission and management |
| `node.proto` | Node registration and services |
| `compute.proto` | Direct compute operations |
| `wallet.proto` | Wallet and payment operations |
| `deployment.proto` | Model deployment |
| `execution.proto` | Job execution details |
| `daemon.proto` | Daemon mode IPC |

## Common Types

### common.proto

```protobuf
syntax = "proto3";
package cyxwiz.protocol;

// Version information
message Version {
  int32 major = 1;
  int32 minor = 2;
  int32 patch = 3;
}

// Status codes
enum StatusCode {
  STATUS_UNKNOWN = 0;
  STATUS_SUCCESS = 1;
  STATUS_PENDING = 2;
  STATUS_RUNNING = 3;
  STATUS_COMPLETED = 4;
  STATUS_FAILED = 5;
  STATUS_CANCELLED = 6;
  STATUS_QUEUED = 7;
}

// Device types
enum DeviceType {
  DEVICE_UNKNOWN = 0;
  DEVICE_CPU = 1;
  DEVICE_CUDA = 2;
  DEVICE_OPENCL = 3;
  DEVICE_METAL = 4;
}

// Device capabilities
message DeviceCapabilities {
  DeviceType device_type = 1;
  string device_name = 2;
  int64 total_memory = 3;
  int64 available_memory = 4;
  int32 compute_units = 5;
  string driver_version = 6;
  double compute_score = 7;
}

// Error information
message Error {
  int32 code = 1;
  string message = 2;
  string details = 3;
}

// Tensor shape and data
message TensorInfo {
  repeated int32 shape = 1;
  string dtype = 2;        // float32, float64, int32, etc.
  int64 size_bytes = 3;
}
```

## Job Service

### job.proto

The Job Service handles job submission from clients (Engine) to the Central Server.

```protobuf
// Job types
enum JobType {
  JOB_TYPE_UNKNOWN = 0;
  JOB_TYPE_TRAINING = 1;
  JOB_TYPE_INFERENCE = 2;
  JOB_TYPE_EVALUATION = 3;
  JOB_TYPE_PREPROCESSING = 4;
}

// Job priority
enum JobPriority {
  PRIORITY_LOW = 0;
  PRIORITY_NORMAL = 1;
  PRIORITY_HIGH = 2;
  PRIORITY_CRITICAL = 3;
}

// Job configuration
message JobConfig {
  string job_id = 1;
  JobType job_type = 2;
  JobPriority priority = 3;
  string model_definition = 4;      // JSON or script
  map<string, string> hyperparameters = 5;
  string dataset_uri = 6;           // IPFS hash or URL
  int32 batch_size = 7;
  int32 epochs = 8;
  DeviceType required_device = 9;
  int64 estimated_memory = 10;
  int64 estimated_duration = 11;
  double payment_amount = 12;
  string payment_address = 13;
  string escrow_tx_hash = 14;
}

// Job status
message JobStatus {
  string job_id = 1;
  StatusCode status = 2;
  double progress = 3;              // 0.0 to 1.0
  string current_node_id = 4;
  int64 start_time = 5;
  int64 end_time = 6;
  Error error = 7;
  map<string, double> metrics = 8;
  int32 current_epoch = 9;
}

// Job result
message JobResult {
  string job_id = 1;
  StatusCode status = 2;
  string model_weights_uri = 3;
  string model_weights_hash = 4;
  int64 model_size = 5;
  map<string, double> final_metrics = 6;
  int64 total_compute_time = 7;
  double energy_consumed = 8;
  string proof_of_compute = 9;
  repeated string signatures = 10;
}

// P2P node assignment
message NodeAssignment {
  string node_id = 1;
  string node_endpoint = 2;
  string auth_token = 3;
  int64 token_expires_at = 4;
  string node_public_key = 5;
}

// Service definition
service JobService {
  rpc SubmitJob(SubmitJobRequest) returns (SubmitJobResponse);
  rpc GetJobStatus(GetJobStatusRequest) returns (GetJobStatusResponse);
  rpc CancelJob(CancelJobRequest) returns (CancelJobResponse);
  rpc StreamJobUpdates(GetJobStatusRequest) returns (stream JobUpdateStream);
  rpc ListJobs(ListJobsRequest) returns (ListJobsResponse);
}
```

### Request/Response Messages

```protobuf
message SubmitJobRequest {
  JobConfig config = 1;
  bytes initial_data = 2;
}

message SubmitJobResponse {
  string job_id = 1;
  StatusCode status = 2;
  NodeAssignment node_assignment = 3;
  string assigned_node_id = 4;
  Error error = 5;
  int64 estimated_start_time = 6;
}

message GetJobStatusRequest {
  string job_id = 1;
}

message GetJobStatusResponse {
  JobStatus status = 1;
  NodeAssignment node_assignment = 2;
  Error error = 3;
}

message CancelJobRequest {
  string job_id = 1;
  string reason = 2;
}

message CancelJobResponse {
  StatusCode status = 1;
  bool refund_issued = 2;
  Error error = 3;
}

message JobUpdateStream {
  string job_id = 1;
  JobStatus status = 2;
  map<string, double> live_metrics = 3;
  string log_message = 4;
  bytes visualization_data = 5;
}

message ListJobsRequest {
  string user_id = 1;
  int32 page_size = 2;
  string page_token = 3;
  JobType filter_type = 4;
  StatusCode filter_status = 5;
}

message ListJobsResponse {
  repeated JobStatus jobs = 1;
  string next_page_token = 2;
  int32 total_count = 3;
}
```

## Node Service

### node.proto

The Node Service handles node registration and management.

```protobuf
// Node information
message NodeInfo {
  string node_id = 1;
  string name = 2;
  Version version = 3;
  repeated DeviceCapabilities devices = 4;
  int32 cpu_cores = 5;
  int64 ram_total = 6;
  int64 ram_available = 7;
  string ip_address = 8;
  int32 port = 9;
  string region = 10;
  double compute_score = 11;
  double reputation_score = 12;
  int64 total_jobs_completed = 13;
  int64 total_compute_hours = 14;
  double average_rating = 15;
  double staked_amount = 16;
  string wallet_address = 17;
  bool is_online = 18;
  int64 last_heartbeat = 19;
  double uptime_percentage = 20;
  repeated string supported_formats = 21;
  int64 max_model_size = 22;
  bool supports_terminal_access = 23;
  repeated string available_runtimes = 24;
}

// Service definition
service NodeService {
  rpc RegisterNode(RegisterNodeRequest) returns (RegisterNodeResponse);
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
  rpc AssignJob(AssignJobRequest) returns (AssignJobResponse);
  rpc ReportProgress(ReportProgressRequest) returns (ReportProgressResponse);
  rpc ReportCompletion(ReportCompletionRequest) returns (ReportCompletionResponse);
  rpc GetNodeMetrics(GetNodeMetricsRequest) returns (GetNodeMetricsResponse);
  rpc NotifyJobAccepted(JobAcceptedRequest) returns (JobAcceptedResponse);
}

// Registration
message RegisterNodeRequest {
  NodeInfo info = 1;
  string authentication_token = 2;
  bytes public_key = 3;
}

message RegisterNodeResponse {
  StatusCode status = 1;
  string node_id = 2;
  string session_token = 3;
  Error error = 4;
}

// Heartbeat
message HeartbeatRequest {
  string node_id = 1;
  NodeInfo current_status = 2;
  repeated string active_jobs = 3;
}

message HeartbeatResponse {
  StatusCode status = 1;
  bool keep_alive = 2;
  repeated string jobs_to_cancel = 3;
  Error error = 4;
}

// Job assignment
message AssignJobRequest {
  string node_id = 1;
  JobConfig job = 2;
  string authorization_token = 3;
}

message AssignJobResponse {
  StatusCode status = 1;
  string job_id = 2;
  bool accepted = 3;
  Error error = 4;
  string estimated_start_time = 5;
}
```

## Job Status Service

### Reporting from Node to Server

```protobuf
service JobStatusService {
  rpc UpdateJobStatus(UpdateJobStatusRequest) returns (UpdateJobStatusResponse);
  rpc ReportJobResult(ReportJobResultRequest) returns (ReportJobResultResponse);
}

message UpdateJobStatusRequest {
  string node_id = 1;
  string job_id = 2;
  StatusCode status = 3;
  double progress = 4;
  map<string, double> metrics = 5;
  int32 current_epoch = 6;
  string log_message = 7;
}

message UpdateJobStatusResponse {
  StatusCode status = 1;
  bool should_continue = 2;
  Error error = 3;
}

message ReportJobResultRequest {
  string node_id = 1;
  string job_id = 2;
  StatusCode final_status = 3;
  map<string, double> final_metrics = 4;
  string model_weights_uri = 5;
  string model_weights_hash = 6;
  int64 model_size = 7;
  int64 total_compute_time = 8;
  string error_message = 9;
}

message ReportJobResultResponse {
  StatusCode status = 1;
  Error error = 2;
}
```

## Wallet Service

### wallet.proto

```protobuf
service WalletService {
  rpc GetBalance(GetBalanceRequest) returns (GetBalanceResponse);
  rpc CreateEscrow(CreateEscrowRequest) returns (CreateEscrowResponse);
  rpc ReleaseEscrow(ReleaseEscrowRequest) returns (ReleaseEscrowResponse);
  rpc GetTransactionHistory(GetTransactionHistoryRequest) returns (GetTransactionHistoryResponse);
}

message GetBalanceRequest {
  string wallet_address = 1;
}

message GetBalanceResponse {
  double sol_balance = 1;
  double cyxwiz_balance = 2;
  Error error = 3;
}

message CreateEscrowRequest {
  string job_id = 1;
  string payer_address = 2;
  double amount = 3;
  int64 expiration_time = 4;
}

message CreateEscrowResponse {
  string escrow_id = 1;
  string tx_signature = 2;
  Error error = 3;
}
```

## Usage Examples

### C++ Client

```cpp
#include <grpcpp/grpcpp.h>
#include "job.grpc.pb.h"

using namespace cyxwiz::protocol;

// Create channel and stub
auto channel = grpc::CreateChannel("localhost:50051",
    grpc::InsecureChannelCredentials());
auto stub = JobService::NewStub(channel);

// Submit job
SubmitJobRequest request;
request.mutable_config()->set_job_type(JOB_TYPE_TRAINING);
request.mutable_config()->set_dataset_uri("ipfs://...");

SubmitJobResponse response;
grpc::ClientContext context;
grpc::Status status = stub->SubmitJob(&context, request, &response);

if (status.ok()) {
    std::cout << "Job ID: " << response.job_id() << std::endl;
}
```

### Rust Server

```rust
use tonic::{Request, Response, Status};
use crate::pb::*;

pub struct JobServiceImpl {
    db: Pool<Postgres>,
}

#[tonic::async_trait]
impl JobService for JobServiceImpl {
    async fn submit_job(
        &self,
        request: Request<SubmitJobRequest>,
    ) -> Result<Response<SubmitJobResponse>, Status> {
        let req = request.into_inner();

        // Validate and process
        let job_id = uuid::Uuid::new_v4().to_string();

        // Store in database
        // ...

        Ok(Response::new(SubmitJobResponse {
            job_id,
            status: StatusCode::Queued as i32,
            ..Default::default()
        }))
    }
}
```

### Python Client

```python
import grpc
from cyxwiz_protocol import job_pb2, job_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = job_pb2_grpc.JobServiceStub(channel)

# Submit job
request = job_pb2.SubmitJobRequest()
request.config.job_type = job_pb2.JOB_TYPE_TRAINING
request.config.dataset_uri = "ipfs://..."

response = stub.SubmitJob(request)
print(f"Job ID: {response.job_id}")
```

## Streaming

### Server Streaming (Job Updates)

```protobuf
rpc StreamJobUpdates(GetJobStatusRequest) returns (stream JobUpdateStream);
```

```cpp
// C++ client
grpc::ClientContext context;
auto reader = stub->StreamJobUpdates(&context, request);

JobUpdateStream update;
while (reader->Read(&update)) {
    std::cout << "Progress: " << update.status().progress() << std::endl;
}
```

```rust
// Rust server
async fn stream_job_updates(
    &self,
    request: Request<GetJobStatusRequest>,
) -> Result<Response<Self::StreamJobUpdatesStream>, Status> {
    let job_id = request.into_inner().job_id;

    let (tx, rx) = tokio::sync::mpsc::channel(10);

    tokio::spawn(async move {
        // Send updates periodically
        loop {
            let update = get_job_update(&job_id).await;
            if tx.send(Ok(update)).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    Ok(Response::new(ReceiverStream::new(rx)))
}
```

## Error Handling

### Error Codes

| Code | Meaning |
|------|---------|
| 0 | Unknown error |
| 1 | Invalid argument |
| 2 | Not found |
| 3 | Already exists |
| 4 | Permission denied |
| 5 | Resource exhausted |
| 6 | Internal error |
| 7 | Unavailable |
| 8 | Timeout |

### Error Response

```protobuf
message Error {
  int32 code = 1;
  string message = 2;
  string details = 3;  // JSON for structured details
}
```

---

**Next**: [Job Service Details](job-service.md) | [Node Service Details](node-service.md)
