import common_pb2 as _common_pb2
import job_pb2 as _job_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NodeInfo(_message.Message):
    __slots__ = ("node_id", "name", "version", "devices", "cpu_cores", "ram_total", "ram_available", "ip_address", "port", "region", "compute_score", "reputation_score", "total_jobs_completed", "total_compute_hours", "average_rating", "staked_amount", "wallet_address", "is_online", "last_heartbeat", "uptime_percentage", "supported_formats", "max_model_size", "supports_terminal_access", "available_runtimes")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICES_FIELD_NUMBER: _ClassVar[int]
    CPU_CORES_FIELD_NUMBER: _ClassVar[int]
    RAM_TOTAL_FIELD_NUMBER: _ClassVar[int]
    RAM_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    IP_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_SCORE_FIELD_NUMBER: _ClassVar[int]
    REPUTATION_SCORE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_JOBS_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COMPUTE_HOURS_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_RATING_FIELD_NUMBER: _ClassVar[int]
    STAKED_AMOUNT_FIELD_NUMBER: _ClassVar[int]
    WALLET_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    IS_ONLINE_FIELD_NUMBER: _ClassVar[int]
    LAST_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    UPTIME_PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_FORMATS_FIELD_NUMBER: _ClassVar[int]
    MAX_MODEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_TERMINAL_ACCESS_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_RUNTIMES_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    name: str
    version: _common_pb2.Version
    devices: _containers.RepeatedCompositeFieldContainer[_common_pb2.DeviceCapabilities]
    cpu_cores: int
    ram_total: int
    ram_available: int
    ip_address: str
    port: int
    region: str
    compute_score: float
    reputation_score: float
    total_jobs_completed: int
    total_compute_hours: int
    average_rating: float
    staked_amount: float
    wallet_address: str
    is_online: bool
    last_heartbeat: int
    uptime_percentage: float
    supported_formats: _containers.RepeatedScalarFieldContainer[str]
    max_model_size: int
    supports_terminal_access: bool
    available_runtimes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, node_id: _Optional[str] = ..., name: _Optional[str] = ..., version: _Optional[_Union[_common_pb2.Version, _Mapping]] = ..., devices: _Optional[_Iterable[_Union[_common_pb2.DeviceCapabilities, _Mapping]]] = ..., cpu_cores: _Optional[int] = ..., ram_total: _Optional[int] = ..., ram_available: _Optional[int] = ..., ip_address: _Optional[str] = ..., port: _Optional[int] = ..., region: _Optional[str] = ..., compute_score: _Optional[float] = ..., reputation_score: _Optional[float] = ..., total_jobs_completed: _Optional[int] = ..., total_compute_hours: _Optional[int] = ..., average_rating: _Optional[float] = ..., staked_amount: _Optional[float] = ..., wallet_address: _Optional[str] = ..., is_online: bool = ..., last_heartbeat: _Optional[int] = ..., uptime_percentage: _Optional[float] = ..., supported_formats: _Optional[_Iterable[str]] = ..., max_model_size: _Optional[int] = ..., supports_terminal_access: bool = ..., available_runtimes: _Optional[_Iterable[str]] = ...) -> None: ...

class RegisterNodeRequest(_message.Message):
    __slots__ = ("info", "authentication_token", "public_key")
    INFO_FIELD_NUMBER: _ClassVar[int]
    AUTHENTICATION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_KEY_FIELD_NUMBER: _ClassVar[int]
    info: NodeInfo
    authentication_token: str
    public_key: bytes
    def __init__(self, info: _Optional[_Union[NodeInfo, _Mapping]] = ..., authentication_token: _Optional[str] = ..., public_key: _Optional[bytes] = ...) -> None: ...

class RegisterNodeResponse(_message.Message):
    __slots__ = ("status", "node_id", "session_token", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    node_id: str
    session_token: str
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., node_id: _Optional[str] = ..., session_token: _Optional[str] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("node_id", "current_status", "active_jobs")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STATUS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_JOBS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    current_status: NodeInfo
    active_jobs: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, node_id: _Optional[str] = ..., current_status: _Optional[_Union[NodeInfo, _Mapping]] = ..., active_jobs: _Optional[_Iterable[str]] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("status", "keep_alive", "jobs_to_cancel", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    KEEP_ALIVE_FIELD_NUMBER: _ClassVar[int]
    JOBS_TO_CANCEL_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    keep_alive: bool
    jobs_to_cancel: _containers.RepeatedScalarFieldContainer[str]
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., keep_alive: bool = ..., jobs_to_cancel: _Optional[_Iterable[str]] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class AssignJobRequest(_message.Message):
    __slots__ = ("node_id", "job", "authorization_token")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    AUTHORIZATION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    job: _job_pb2.JobConfig
    authorization_token: str
    def __init__(self, node_id: _Optional[str] = ..., job: _Optional[_Union[_job_pb2.JobConfig, _Mapping]] = ..., authorization_token: _Optional[str] = ...) -> None: ...

class AssignJobResponse(_message.Message):
    __slots__ = ("status", "job_id", "accepted", "error", "estimated_start_time")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_START_TIME_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    job_id: str
    accepted: bool
    error: _common_pb2.Error
    estimated_start_time: str
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., job_id: _Optional[str] = ..., accepted: bool = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ..., estimated_start_time: _Optional[str] = ...) -> None: ...

class ReportProgressRequest(_message.Message):
    __slots__ = ("node_id", "job_id", "status", "current_metrics")
    class CurrentMetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_METRICS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    job_id: str
    status: _job_pb2.JobStatus
    current_metrics: _containers.ScalarMap[str, float]
    def __init__(self, node_id: _Optional[str] = ..., job_id: _Optional[str] = ..., status: _Optional[_Union[_job_pb2.JobStatus, _Mapping]] = ..., current_metrics: _Optional[_Mapping[str, float]] = ...) -> None: ...

class ReportProgressResponse(_message.Message):
    __slots__ = ("status", "continue_job", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CONTINUE_JOB_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    continue_job: bool
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., continue_job: bool = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class ReportCompletionRequest(_message.Message):
    __slots__ = ("node_id", "job_id", "result", "signature")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    job_id: str
    result: _job_pb2.JobResult
    signature: bytes
    def __init__(self, node_id: _Optional[str] = ..., job_id: _Optional[str] = ..., result: _Optional[_Union[_job_pb2.JobResult, _Mapping]] = ..., signature: _Optional[bytes] = ...) -> None: ...

class ReportCompletionResponse(_message.Message):
    __slots__ = ("status", "payment_released", "payment_tx_hash", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PAYMENT_RELEASED_FIELD_NUMBER: _ClassVar[int]
    PAYMENT_TX_HASH_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    payment_released: bool
    payment_tx_hash: str
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., payment_released: bool = ..., payment_tx_hash: _Optional[str] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class GetNodeMetricsRequest(_message.Message):
    __slots__ = ("node_id", "start_time", "end_time")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    start_time: int
    end_time: int
    def __init__(self, node_id: _Optional[str] = ..., start_time: _Optional[int] = ..., end_time: _Optional[int] = ...) -> None: ...

class GetNodeMetricsResponse(_message.Message):
    __slots__ = ("metrics", "error")
    METRICS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    metrics: _containers.RepeatedCompositeFieldContainer[MetricPoint]
    error: _common_pb2.Error
    def __init__(self, metrics: _Optional[_Iterable[_Union[MetricPoint, _Mapping]]] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class MetricPoint(_message.Message):
    __slots__ = ("timestamp", "cpu_usage", "gpu_usage", "memory_usage", "network_throughput", "power_consumption")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CPU_USAGE_FIELD_NUMBER: _ClassVar[int]
    GPU_USAGE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_FIELD_NUMBER: _ClassVar[int]
    NETWORK_THROUGHPUT_FIELD_NUMBER: _ClassVar[int]
    POWER_CONSUMPTION_FIELD_NUMBER: _ClassVar[int]
    timestamp: int
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    network_throughput: float
    power_consumption: float
    def __init__(self, timestamp: _Optional[int] = ..., cpu_usage: _Optional[float] = ..., gpu_usage: _Optional[float] = ..., memory_usage: _Optional[float] = ..., network_throughput: _Optional[float] = ..., power_consumption: _Optional[float] = ...) -> None: ...

class JobAcceptedRequest(_message.Message):
    __slots__ = ("node_id", "job_id", "engine_address", "accepted_at", "node_endpoint")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    ENGINE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_AT_FIELD_NUMBER: _ClassVar[int]
    NODE_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    job_id: str
    engine_address: str
    accepted_at: int
    node_endpoint: str
    def __init__(self, node_id: _Optional[str] = ..., job_id: _Optional[str] = ..., engine_address: _Optional[str] = ..., accepted_at: _Optional[int] = ..., node_endpoint: _Optional[str] = ...) -> None: ...

class JobAcceptedResponse(_message.Message):
    __slots__ = ("status", "acknowledged", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    acknowledged: bool
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., acknowledged: bool = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class FindNodesRequest(_message.Message):
    __slots__ = ("required_device", "min_memory", "min_reputation", "preferred_region", "max_results")
    REQUIRED_DEVICE_FIELD_NUMBER: _ClassVar[int]
    MIN_MEMORY_FIELD_NUMBER: _ClassVar[int]
    MIN_REPUTATION_FIELD_NUMBER: _ClassVar[int]
    PREFERRED_REGION_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    required_device: _common_pb2.DeviceType
    min_memory: int
    min_reputation: float
    preferred_region: str
    max_results: int
    def __init__(self, required_device: _Optional[_Union[_common_pb2.DeviceType, str]] = ..., min_memory: _Optional[int] = ..., min_reputation: _Optional[float] = ..., preferred_region: _Optional[str] = ..., max_results: _Optional[int] = ...) -> None: ...

class FindNodesResponse(_message.Message):
    __slots__ = ("nodes", "error")
    NODES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeInfo]
    error: _common_pb2.Error
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeInfo, _Mapping]]] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class ListNodesRequest(_message.Message):
    __slots__ = ("page_size", "page_token", "online_only")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    ONLINE_ONLY_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    online_only: bool
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., online_only: bool = ...) -> None: ...

class ListNodesResponse(_message.Message):
    __slots__ = ("nodes", "next_page_token", "total_count")
    NODES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeInfo]
    next_page_token: str
    total_count: int
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ..., total_count: _Optional[int] = ...) -> None: ...

class GetNodeInfoRequest(_message.Message):
    __slots__ = ("node_id",)
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    def __init__(self, node_id: _Optional[str] = ...) -> None: ...

class GetNodeInfoResponse(_message.Message):
    __slots__ = ("info", "error")
    INFO_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    info: NodeInfo
    error: _common_pb2.Error
    def __init__(self, info: _Optional[_Union[NodeInfo, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class UpdateJobStatusRequest(_message.Message):
    __slots__ = ("node_id", "job_id", "status", "progress", "metrics", "current_epoch", "log_message")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    LOG_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    job_id: str
    status: _common_pb2.StatusCode
    progress: float
    metrics: _containers.ScalarMap[str, float]
    current_epoch: int
    log_message: str
    def __init__(self, node_id: _Optional[str] = ..., job_id: _Optional[str] = ..., status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., progress: _Optional[float] = ..., metrics: _Optional[_Mapping[str, float]] = ..., current_epoch: _Optional[int] = ..., log_message: _Optional[str] = ...) -> None: ...

class UpdateJobStatusResponse(_message.Message):
    __slots__ = ("status", "should_continue", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SHOULD_CONTINUE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    should_continue: bool
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., should_continue: bool = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class ReportJobResultRequest(_message.Message):
    __slots__ = ("node_id", "job_id", "final_status", "final_metrics", "model_weights_uri", "model_weights_hash", "model_size", "total_compute_time", "error_message")
    class FinalMetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    FINAL_STATUS_FIELD_NUMBER: _ClassVar[int]
    FINAL_METRICS_FIELD_NUMBER: _ClassVar[int]
    MODEL_WEIGHTS_URI_FIELD_NUMBER: _ClassVar[int]
    MODEL_WEIGHTS_HASH_FIELD_NUMBER: _ClassVar[int]
    MODEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COMPUTE_TIME_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    job_id: str
    final_status: _common_pb2.StatusCode
    final_metrics: _containers.ScalarMap[str, float]
    model_weights_uri: str
    model_weights_hash: str
    model_size: int
    total_compute_time: int
    error_message: str
    def __init__(self, node_id: _Optional[str] = ..., job_id: _Optional[str] = ..., final_status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., final_metrics: _Optional[_Mapping[str, float]] = ..., model_weights_uri: _Optional[str] = ..., model_weights_hash: _Optional[str] = ..., model_size: _Optional[int] = ..., total_compute_time: _Optional[int] = ..., error_message: _Optional[str] = ...) -> None: ...

class ReportJobResultResponse(_message.Message):
    __slots__ = ("status", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.StatusCode
    error: _common_pb2.Error
    def __init__(self, status: _Optional[_Union[_common_pb2.StatusCode, str]] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...
