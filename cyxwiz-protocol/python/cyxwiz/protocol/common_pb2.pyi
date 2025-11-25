from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StatusCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATUS_UNKNOWN: _ClassVar[StatusCode]
    STATUS_SUCCESS: _ClassVar[StatusCode]
    STATUS_ERROR: _ClassVar[StatusCode]
    STATUS_PENDING: _ClassVar[StatusCode]
    STATUS_IN_PROGRESS: _ClassVar[StatusCode]
    STATUS_COMPLETED: _ClassVar[StatusCode]
    STATUS_FAILED: _ClassVar[StatusCode]
    STATUS_CANCELLED: _ClassVar[StatusCode]

class DeviceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEVICE_UNKNOWN: _ClassVar[DeviceType]
    DEVICE_CPU: _ClassVar[DeviceType]
    DEVICE_CUDA: _ClassVar[DeviceType]
    DEVICE_OPENCL: _ClassVar[DeviceType]
    DEVICE_METAL: _ClassVar[DeviceType]
    DEVICE_VULKAN: _ClassVar[DeviceType]

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DATA_TYPE_UNKNOWN: _ClassVar[DataType]
    DATA_TYPE_FLOAT32: _ClassVar[DataType]
    DATA_TYPE_FLOAT64: _ClassVar[DataType]
    DATA_TYPE_INT32: _ClassVar[DataType]
    DATA_TYPE_INT64: _ClassVar[DataType]
    DATA_TYPE_UINT8: _ClassVar[DataType]
STATUS_UNKNOWN: StatusCode
STATUS_SUCCESS: StatusCode
STATUS_ERROR: StatusCode
STATUS_PENDING: StatusCode
STATUS_IN_PROGRESS: StatusCode
STATUS_COMPLETED: StatusCode
STATUS_FAILED: StatusCode
STATUS_CANCELLED: StatusCode
DEVICE_UNKNOWN: DeviceType
DEVICE_CPU: DeviceType
DEVICE_CUDA: DeviceType
DEVICE_OPENCL: DeviceType
DEVICE_METAL: DeviceType
DEVICE_VULKAN: DeviceType
DATA_TYPE_UNKNOWN: DataType
DATA_TYPE_FLOAT32: DataType
DATA_TYPE_FLOAT64: DataType
DATA_TYPE_INT32: DataType
DATA_TYPE_INT64: DataType
DATA_TYPE_UINT8: DataType

class Error(_message.Message):
    __slots__ = ("code", "message", "details")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    code: int
    message: str
    details: str
    def __init__(self, code: _Optional[int] = ..., message: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class TensorInfo(_message.Message):
    __slots__ = ("shape", "dtype", "name")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: DataType
    name: str
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[_Union[DataType, str]] = ..., name: _Optional[str] = ...) -> None: ...

class DeviceCapabilities(_message.Message):
    __slots__ = ("device_type", "device_name", "memory_total", "memory_available", "compute_units", "supports_fp64", "supports_fp16", "gpu_model", "vram_total", "vram_available", "driver_version", "cuda_version", "pcie_generation", "pcie_lanes", "compute_capability")
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_FIELD_NUMBER: _ClassVar[int]
    MEMORY_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_UNITS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_FP64_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_FP16_FIELD_NUMBER: _ClassVar[int]
    GPU_MODEL_FIELD_NUMBER: _ClassVar[int]
    VRAM_TOTAL_FIELD_NUMBER: _ClassVar[int]
    VRAM_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    CUDA_VERSION_FIELD_NUMBER: _ClassVar[int]
    PCIE_GENERATION_FIELD_NUMBER: _ClassVar[int]
    PCIE_LANES_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    device_type: DeviceType
    device_name: str
    memory_total: int
    memory_available: int
    compute_units: int
    supports_fp64: bool
    supports_fp16: bool
    gpu_model: str
    vram_total: int
    vram_available: int
    driver_version: str
    cuda_version: str
    pcie_generation: int
    pcie_lanes: int
    compute_capability: float
    def __init__(self, device_type: _Optional[_Union[DeviceType, str]] = ..., device_name: _Optional[str] = ..., memory_total: _Optional[int] = ..., memory_available: _Optional[int] = ..., compute_units: _Optional[int] = ..., supports_fp64: bool = ..., supports_fp16: bool = ..., gpu_model: _Optional[str] = ..., vram_total: _Optional[int] = ..., vram_available: _Optional[int] = ..., driver_version: _Optional[str] = ..., cuda_version: _Optional[str] = ..., pcie_generation: _Optional[int] = ..., pcie_lanes: _Optional[int] = ..., compute_capability: _Optional[float] = ...) -> None: ...

class Version(_message.Message):
    __slots__ = ("major", "minor", "patch", "build")
    MAJOR_FIELD_NUMBER: _ClassVar[int]
    MINOR_FIELD_NUMBER: _ClassVar[int]
    PATCH_FIELD_NUMBER: _ClassVar[int]
    BUILD_FIELD_NUMBER: _ClassVar[int]
    major: int
    minor: int
    patch: int
    build: str
    def __init__(self, major: _Optional[int] = ..., minor: _Optional[int] = ..., patch: _Optional[int] = ..., build: _Optional[str] = ...) -> None: ...
