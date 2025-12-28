# Device Management

Device management in pycyxwiz for GPU/CPU selection and hardware information.

## Overview

pycyxwiz supports multiple compute backends:

| Backend | Description | Use Case |
|---------|-------------|----------|
| CPU | System CPU | Fallback, debugging |
| CUDA | NVIDIA GPU | Training (best performance) |
| OpenCL | AMD/Intel GPU | Cross-platform GPU |
| Metal | Apple Silicon | macOS acceleration |

## Checking Availability

```python
import pycyxwiz as cx

# Check specific backends
print("CUDA:", cx.cuda_available())      # NVIDIA GPUs
print("OpenCL:", cx.opencl_available())  # AMD/Intel GPUs
print("Metal:", cx.metal_available())    # Apple Metal
```

## Device Enumeration

### Get All Devices

```python
devices = cx.get_available_devices()

for device in devices:
    print(f"Device {device.device_id}: {device.name}")
    print(f"  Type: {device.type}")
    print(f"  Memory: {device.memory_total // (1024**3)} GB")
    print(f"  Compute Units: {device.compute_units}")
    print(f"  FP64 Support: {device.supports_fp64}")
    print()
```

### DeviceInfo Structure

| Property | Type | Description |
|----------|------|-------------|
| `device_id` | int | Device index |
| `name` | str | Device name (e.g., "NVIDIA RTX 3080") |
| `type` | DeviceType | CPU, CUDA, OPENCL, METAL |
| `memory_total` | int | Total memory in bytes |
| `memory_available` | int | Available memory in bytes |
| `compute_units` | int | Number of compute units/cores |
| `supports_fp64` | bool | Double precision support |
| `supports_fp16` | bool | Half precision support |

## Device Selection

### Get Device by Type

```python
# Get first CUDA device
cuda_device = cx.get_device(cx.DeviceType.CUDA, 0)

# Get second OpenCL device
opencl_device = cx.get_device(cx.DeviceType.OPENCL, 1)

# Get CPU device
cpu_device = cx.get_device(cx.DeviceType.CPU, 0)
```

### Set Active Device

```python
# By device object
device = cx.get_device(cx.DeviceType.CUDA, 0)
cx.set_device(device)

# Or get and set in one step
device = cx.Device(cx.DeviceType.CUDA, 0)
device.set_active()
```

### Check Current Device

```python
current = cx.Device.get_current_device()
print(f"Current device: {current.get_info().name}")
```

## Device Class

```python
class Device:
    def __init__(self, type: DeviceType, device_id: int = 0)

    # Methods
    def get_type(self) -> DeviceType
    def get_device_id(self) -> int
    def get_info(self) -> DeviceInfo
    def set_active(self) -> None
    def is_active(self) -> bool

    # Static methods
    @staticmethod
    def get_available_devices() -> List[DeviceInfo]

    @staticmethod
    def get_current_device() -> Device
```

### Usage

```python
# Create device
device = cx.Device(cx.DeviceType.CUDA, 0)

# Get information
info = device.get_info()
print(f"Name: {info.name}")
print(f"Memory: {info.memory_total} bytes")

# Activate
device.set_active()
print(f"Is active: {device.is_active()}")

# All tensors created now use this device
tensor = cx.Tensor.random([1000, 1000])
```

## DeviceType Enum

```python
cx.DeviceType.CPU     # CPU backend
cx.DeviceType.CUDA    # NVIDIA CUDA
cx.DeviceType.OPENCL  # OpenCL (AMD, Intel, etc.)
cx.DeviceType.METAL   # Apple Metal (macOS)
cx.DeviceType.VULKAN  # Vulkan (experimental)
```

## Multi-GPU Setup

### Enumerate GPUs

```python
devices = cx.get_available_devices()
cuda_devices = [d for d in devices if d.type == cx.DeviceType.CUDA]

print(f"Found {len(cuda_devices)} CUDA devices:")
for d in cuda_devices:
    print(f"  [{d.device_id}] {d.name}")
```

### Select Specific GPU

```python
# Use GPU 1 (second GPU)
device = cx.Device(cx.DeviceType.CUDA, 1)
device.set_active()
```

### Memory Management

```python
info = cx.Device.get_current_device().get_info()

total_gb = info.memory_total / (1024**3)
available_gb = info.memory_available / (1024**3)
used_gb = total_gb - available_gb

print(f"GPU Memory: {used_gb:.1f} / {total_gb:.1f} GB used")
```

## Best Practices

### Auto-Select Best Device

```python
def get_best_device():
    """Select best available compute device."""
    if cx.cuda_available():
        return cx.Device(cx.DeviceType.CUDA, 0)
    elif cx.opencl_available():
        return cx.Device(cx.DeviceType.OPENCL, 0)
    elif cx.metal_available():
        return cx.Device(cx.DeviceType.METAL, 0)
    else:
        return cx.Device(cx.DeviceType.CPU, 0)

device = get_best_device()
device.set_active()
print(f"Using: {device.get_info().name}")
```

### GPU with Most Memory

```python
def get_gpu_with_most_memory():
    """Select GPU with the most available memory."""
    devices = cx.get_available_devices()
    gpus = [d for d in devices if d.type in (cx.DeviceType.CUDA, cx.DeviceType.OPENCL)]

    if not gpus:
        return None

    best = max(gpus, key=lambda d: d.memory_available)
    return cx.Device(best.type, best.device_id)
```

### Fallback Pattern

```python
try:
    device = cx.Device(cx.DeviceType.CUDA, 0)
    device.set_active()
    print("Using CUDA")
except Exception as e:
    print(f"CUDA not available: {e}")
    device = cx.Device(cx.DeviceType.CPU, 0)
    device.set_active()
    print("Falling back to CPU")
```

## Performance Tips

### Check FP16 Support

```python
info = cx.Device.get_current_device().get_info()
if info.supports_fp16:
    print("FP16 supported - can use mixed precision training")
```

### Monitor Memory

```python
def check_memory():
    info = cx.Device.get_current_device().get_info()
    available = info.memory_available / (1024**3)
    if available < 1.0:
        print(f"Warning: Low GPU memory ({available:.2f} GB)")
        return False
    return True

# Before large allocation
if check_memory():
    large_tensor = cx.Tensor.random([10000, 10000])
```

### Device Info Caching

```python
# Cache device info to avoid repeated queries
_device_info_cache = {}

def get_device_info(device_id=0):
    if device_id not in _device_info_cache:
        device = cx.Device(cx.DeviceType.CUDA, device_id)
        _device_info_cache[device_id] = device.get_info()
    return _device_info_cache[device_id]
```

## Example: Device Selection Script

```python
import pycyxwiz as cx

def setup_device():
    """Setup and display device information."""
    print("=" * 50)
    print("CyxWiz Device Configuration")
    print("=" * 50)

    # List all devices
    devices = cx.get_available_devices()
    print(f"\nFound {len(devices)} device(s):\n")

    for d in devices:
        print(f"[{d.device_id}] {d.name}")
        print(f"    Type: {d.type.name}")
        print(f"    Memory: {d.memory_total // (1024**3)} GB")
        print(f"    Compute Units: {d.compute_units}")
        print()

    # Select best device
    if cx.cuda_available():
        device = cx.Device(cx.DeviceType.CUDA, 0)
        print("Selected: CUDA device 0")
    elif cx.opencl_available():
        device = cx.Device(cx.DeviceType.OPENCL, 0)
        print("Selected: OpenCL device 0")
    else:
        device = cx.Device(cx.DeviceType.CPU, 0)
        print("Selected: CPU (no GPU available)")

    device.set_active()

    # Verify
    current = cx.Device.get_current_device()
    info = current.get_info()
    print(f"\nActive device: {info.name}")
    print("=" * 50)

    return device

# Run setup
device = setup_device()
```

---

**Next**: [Linear Algebra](linalg.md) | [Back to Index](index.md)
