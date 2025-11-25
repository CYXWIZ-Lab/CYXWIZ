import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_TYPE_UNKNOWN: _ClassVar[JobType]
    JOB_TYPE_TRAINING: _ClassVar[JobType]
    JOB_TYPE_INFERENCE: _ClassVar[JobType]
    JOB_TYPE_EVALUATION: _ClassVar[JobType]
    JOB_TYPE_PREPROCESSING: _ClassVar[JobType]

class JobPriority(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PRIORITY_LOW: _ClassVar[JobPriority]
    PRIORITY_NORMAL: _ClassVar[JobPriority]
    PRIORITY_HIGH: _ClassVar[JobPriority]
    PRIORITY_CRITICAL: _ClassVar[JobPriority]
JOB_TYPE_UNKNOWN: JobType
JOB_TYPE_TRAINING: JobType
JOB_TYPE_INFERENCE: JobType
JOB_TYPE_EVALUATION: JobType
JOB_TYPE_PREPROCESSING: JobType
PRIORITY_LOW: JobPriority
PRIORITY_NORMAL: JobPriority
PRIORITY_HIGH: JobPriority
PRIORITY_CRITICAL: JobPriority

class JobConfig(_message.Message):
    __slots__ = ("job_id", "job_type", "priority", "model_definition", "hyperparameters", "dataset_uri", "batch_size", "epochs", "required_device", "estimated_memory", "estimated_duration", "payment_amount", "payment_address", "escrow_tx_hash")
    class HyperparametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    MODEL_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    HYPERPARAMETERS_FIELD_NUMBER: _ClassVar[int]
    DATASET_URI_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    EPOCHS_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_DEVICE_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_MEMORY_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_DURATION_FIELD_NUMBER: _ClassVar[int]
    PAYMENT_AMOUNT_FIELD_NUMBER: _ClassVar[int]
    PAYMENT_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ESCROW_TX_HASH_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    job_type: JobType
    priority: JobPriority
    model_definition: str
    hyperparameters: _containers.ScalarMap[str, str]
    dataset_uri: str
    batch_size: int
    epochs: int
    required_device: _common_pb2.DeviceType
    estimated_memory: int
    estimated_duration: int
    payment_amount: float
    payment_address: str
    escrow_tx_hash: str
    def __init__(self, job_id: _Optional[str] = ..., job_type: _Optional[_Union[JobType, str]] = ..., priority: _Optional[_Union[JobPriority, str]] = ..., model_definition: _Optional[str] = ..., hyperparameters: _Optional[_Mapping[str, str]] = ..., dataset_uri: _Optional[str] = ..., batch_size: _Optional[int] = ..., epochs: _Optional[int] = ..., required_device: _Optional[_Union[_common_pb2.DeviceType, str]] = ..., estimated_memory: _Optional[int] = ..., estimated_duration: _Optional[int] = ..., payment_amount: _Optional[float] = ..., payment_address: _Optional[str] = ..., escrow_tx_hash: _Optional[str] = ...) -> None: ...

class JobStatus(_message.Message):
    __slots__ = ("job_id", "status", "progress", "current_node_id", "start_time", "end_time", "error", "metrics", "current_epoch")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: _common_pb2.StatusCode
    progress: float
    current_node_id: str
    start_time: int
    end_time: int
    error: _common_pb2.Error
    metrics: _containers.ScalarMap[str, float]
    current_epoch: int
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., progress: _Optional[float] = ..., current_node_id: _Optional[str] = ..., start_time: _Optional[int] = ..., end_time: _Optional[int] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ..., metrics: _Optional[_Mapping[str, float]] = ..., current_epoch: _Optional[int] = ...) -> None: ...

class JobResult(_message.Message):
    __slots__ = ("job_id", "status", "model_weights_uri", "model_weights_hash", "model_size", "final_metrics", "total_compute_time", "energy_consumed", "proof_of_compute", "signatures")
    class FinalMetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_WEIGHTS_URI_FIELD_NUMBER: _ClassVar[int]
    MODEL_WEIGHTS_HASH_FIELD_NUMBER: _ClassVar[int]
    MODEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    FINAL_METRICS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COMPUTE_TIME_FIELD_NUMBER: _ClassVar[int]
    ENERGY_CONSUMED_FIELD_NUMBER: _ClassVar[int]
    PROOF_OF_COMPUTE_FIELD_NUMBER: _ClassVar[int]
    SIGNATURES_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: _common_pb2.StatusCode
    model_weights_uri: str
    model_weights_hash: str
    model_size: int
    final_metrics: _containers.ScalarMap[str, float]
    total_compute_time: int
    energy_consumed: float
    proof_of_compute: str
    signatures: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., model_weights_uri: _Optional[str] = ..., model_weights_hash: _Optional[str] = ..., model_size: _Optional[int] = ..., final_metrics: _Optional[_Mapping[str, float]] = ..., total_compute_time: _Optional[int] = ..., energy_consumed: _Optional[float] = ..., proof_of_compute: _Optional[str] = ..., signatures: _Optional[_Iterable[str]] = ...) -> None: ...

class SubmitJobRequest(_message.Message):
    __slots__ = ("config", "initial_data")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    INITIAL_DATA_FIELD_NUMBER: _ClassVar[int]
    config: JobConfig
    initial_data: bytes
    def __init__(self, config: _Optional[_Union[JobConfig, _Mapping]] = ..., initial_data: _Optional[bytes] = ...) -> None: ...

class NodeAssignment(_message.Message):
    __slots__ = ("node_id", "node_endpoint", "auth_token", "token_expires_at", "node_public_key")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    AUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOKEN_EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    NODE_PUBLIC_KEY_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    node_endpoint: str
    auth_token: str
    token_expires_at: int
    node_public_key: str
    def __init__(self, node_id: _Optional[str] = ..., node_endpoint: _Optional[str] = ..., auth_token: _Optional[str] = ..., token_expires_at: _Optional[int] = ..., node_public_key: _Optional[str] = ...) -> None: ...

class SubmitJobResponse(_message.Message):
    __slots__ = ("job_id", "status", "node_assignment", "assigned_node_id", "error", "estimated_start_time")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NODE_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_START_TIME_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: _common_pb2.StatusCode
    node_assignment: NodeAssignment
    assigned_node_id: str
    error: _common_pb2.Error
    estimated_start_time: int
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., node_assignment: _Optional[_Union[NodeAssignment, _Mapping]] = ..., assigned_node_id: _Optional[str] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ..., estimated_start_time: _Optional[int] = ...) -> None: ...

class GetJobStatusRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class GetJobStatusResponse(_message.Message):
    __slots__ = ("status", "node_assignment", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NODE_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: JobStatus
    node_assignment: NodeAssignment
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[JobStatus, _Mapping]] = ..., node_assignment: _Optional[_Union[NodeAssignment, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class CancelJobRequest(_message.Message):
    __slots__ = ("job_id", "reason")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    reason: str
    def __init__(self, job_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class CancelJobResponse(_message.Message):
    __slots__ = ("status", "refund_issued", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REFUND_ISSUED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    refund_issued: bool
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., refund_issued: bool = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class JobUpdateStream(_message.Message):
    __slots__ = ("job_id", "status", "live_metrics", "log_message", "visualization_data")
    class LiveMetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LIVE_METRICS_FIELD_NUMBER: _ClassVar[int]
    LOG_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    VISUALIZATION_DATA_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: JobStatus
    live_metrics: _containers.ScalarMap[str, float]
    log_message: str
    visualization_data: bytes
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[JobStatus, _Mapping]] = ..., live_metrics: _Optional[_Mapping[str, float]] = ..., log_message: _Optional[str] = ..., visualization_data: _Optional[bytes] = ...) -> None: ...

class ListJobsRequest(_message.Message):
    __slots__ = ("user_id", "page_size", "page_token", "filter_type", "filter_status")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    FILTER_TYPE_FIELD_NUMBER: _ClassVar[int]
    FILTER_STATUS_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    page_size: int
    page_token: str
    filter_type: JobType
    filter_status: _common_pb2.StatusCode
    def __init__(self, user_id: _Optional[str] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., filter_type: _Optional[_Union[JobType, str]] = ..., filter_status: _Optional[_Union[_common_pb2.StatusCode, str]] = ...) -> None: ...

class ListJobsResponse(_message.Message):
    __slots__ = ("jobs", "next_page_token", "total_count")
    JOBS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[JobStatus]
    next_page_token: str
    total_count: int
    def __init__(self, jobs: _Optional[_Iterable[_Union[JobStatus, _Mapping]]] = ..., next_page_token: _Optional[str] = ..., total_count: _Optional[int] = ...) -> None: ...
