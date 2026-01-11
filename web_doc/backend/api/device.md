# Device API Reference

Device management and GPU acceleration in cyxwiz-backend, providing abstraction over CUDA, OpenCL, and CPU backends via ArrayFire.

## Device Types

```cpp
namespace cyxwiz {

enum class DeviceType {
    CPU,        // CPU backend
    CUDA,       // NVIDIA CUDA
    OpenCL,     // OpenCL (AMD, Intel, etc.)
    Metal,      // Apple Metal (macOS/iOS)
    Auto        // Automatic selection (best available)
};

enum class Backend {
    CPU,
    CUDA,
    OpenCL,
    Metal
};

} // namespace cyxwiz
```

## Device Class

```cpp
namespace cyxwiz {

class CYXWIZ_API Device {
public:
    // Get device information
    static DeviceInfo GetInfo(int device_id = -1);
    static std::vector<DeviceInfo> GetAllDevices();

    // Device selection
    static void SetDevice(int device_id);
    static void SetDevice(DeviceType type, int device_index = 0);
    static int GetDevice();
    static DeviceType GetDeviceType();

    // Backend management
    static void SetBackend(Backend backend);
    static Backend GetBackend();
    static std::vector<Backend> GetAvailableBackends();
    static bool IsBackendAvailable(Backend backend);

    // Memory management
    static size_t GetMemoryUsed();
    static size_t GetMemoryTotal();
    static size_t GetMemoryAvailable();
    static void FreeMemory();
    static void GarbageCollect();

    // Synchronization
    static void Sync();
    static void SyncAll();

    // Device capabilities
    static bool SupportsDoublesPrecision();
    static bool SupportsHalfPrecision();
    static int GetComputeCapability();  // CUDA only
    static int GetMaxThreads();
    static int GetMaxBlockSize();

    // Multi-GPU
    static int GetDeviceCount();
    static int GetDeviceCount(DeviceType type);
    static void EnablePeerAccess(int device_from, int device_to);
    static bool CanAccessPeer(int device_from, int device_to);

private:
    Device() = delete;  // Static class
};

} // namespace cyxwiz
```

## DeviceInfo Structure

```cpp
struct DeviceInfo {
    int device_id;
    std::string name;
    DeviceType type;
    Backend backend;

    // Memory
    size_t total_memory;
    size_t available_memory;
    size_t used_memory;

    // Compute capabilities
    int compute_units;           // CUDA cores / OpenCL compute units
    int max_work_group_size;
    int max_threads_per_block;
    int warp_size;               // 32 for NVIDIA, 64 for AMD

    // Version info
    std::string driver_version;
    int compute_capability_major;  // CUDA only
    int compute_capability_minor;  // CUDA only

    // Features
    bool supports_double;
    bool supports_half;
    bool supports_unified_memory;

    // Performance estimate
    double compute_score;  // TFLOPS estimate
};
```

## Usage Examples

### Device Query

```cpp
#include <cyxwiz/device.h>

using namespace cyxwiz;

// Get all available devices
auto devices = Device::GetAllDevices();
for (const auto& dev : devices) {
    std::cout << "Device " << dev.device_id << ": " << dev.name << std::endl;
    std::cout << "  Type: " << (dev.type == DeviceType::CUDA ? "CUDA" : "Other") << std::endl;
    std::cout << "  Memory: " << dev.total_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "  Compute Units: " << dev.compute_units << std::endl;
}

// Get current device info
DeviceInfo info = Device::GetInfo();
std::cout << "Current device: " << info.name << std::endl;
```

### Device Selection

```cpp
// Select by device ID
Device::SetDevice(0);  // First GPU

// Select by type
Device::SetDevice(DeviceType::CUDA, 0);   // First CUDA device
Device::SetDevice(DeviceType::OpenCL, 1); // Second OpenCL device
Device::SetDevice(DeviceType::CPU);       // CPU backend

// Auto-select best available
Device::SetDevice(DeviceType::Auto);

// Check what's available
if (Device::IsBackendAvailable(Backend::CUDA)) {
    Device::SetBackend(Backend::CUDA);
}
```

### Memory Management

```cpp
// Check memory usage
size_t used = Device::GetMemoryUsed();
size_t total = Device::GetMemoryTotal();
size_t available = Device::GetMemoryAvailable();

std::cout << "Memory: " << used / (1024*1024) << " / "
          << total / (1024*1024) << " MB" << std::endl;

// Force garbage collection
Device::GarbageCollect();

// Free cached memory
Device::FreeMemory();
```

### Synchronization

```cpp
// Synchronize current device
Device::Sync();

// Synchronize all devices (multi-GPU)
Device::SyncAll();

// Use for timing
auto start = std::chrono::high_resolution_clock::now();
tensor.MatMul(other);
Device::Sync();  // Wait for GPU operation to complete
auto end = std::chrono::high_resolution_clock::now();
```

### Multi-GPU

```cpp
// Get device count
int num_gpus = Device::GetDeviceCount(DeviceType::CUDA);
std::cout << "Found " << num_gpus << " CUDA devices" << std::endl;

// Enable peer-to-peer access between GPUs
if (num_gpus >= 2 && Device::CanAccessPeer(0, 1)) {
    Device::EnablePeerAccess(0, 1);
    Device::EnablePeerAccess(1, 0);
}

// Use different GPUs for different operations
Device::SetDevice(0);
Tensor a = Randn({1000, 1000});

Device::SetDevice(1);
Tensor b = Randn({1000, 1000});

// Copy between devices
Tensor a_on_1 = a.ToDevice(DeviceType::CUDA);  // Copy to current device
```

## Device Context Manager

```cpp
namespace cyxwiz {

class DeviceScope {
public:
    DeviceScope(int device_id);
    DeviceScope(DeviceType type, int index = 0);
    ~DeviceScope();

    // Disable copy
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_device_;
};

} // namespace cyxwiz
```

### Usage

```cpp
// Temporary device switch
{
    DeviceScope scope(1);  // Switch to device 1
    Tensor t = Randn({1000, 1000});  // Created on device 1
    // Operations here run on device 1
}  // Automatically switches back to previous device
```

## Backend Initialization

```cpp
namespace cyxwiz {

// Initialize backend (called automatically, but can be explicit)
void InitializeBackend(Backend backend = Backend::CUDA);

// Finalize (cleanup)
void FinalizeBackend();

// Check initialization status
bool IsInitialized();

// Get backend info string
std::string GetBackendInfo();

}
```

### Usage

```cpp
// Explicit initialization (optional)
cyxwiz::InitializeBackend(Backend::CUDA);

// Get info
std::cout << cyxwiz::GetBackendInfo() << std::endl;
// Output: "ArrayFire v3.8.0 (CUDA, 64-bit Linux, build ...)"
```

## Device Memory Allocation

```cpp
namespace cyxwiz {

class DeviceMemory {
public:
    // Allocate on current device
    static void* Allocate(size_t bytes);

    // Allocate on specific device
    static void* Allocate(size_t bytes, int device_id);

    // Free memory
    static void Free(void* ptr);

    // Memory transfer
    static void CopyToDevice(void* dst, const void* src, size_t bytes);
    static void CopyToHost(void* dst, const void* src, size_t bytes);
    static void CopyDeviceToDevice(void* dst, const void* src,
                                   size_t bytes,
                                   int dst_device, int src_device);

    // Pinned (page-locked) memory for faster transfers
    static void* AllocatePinned(size_t bytes);
    static void FreePinned(void* ptr);
};

}
```

## Stream Management (Advanced)

```cpp
namespace cyxwiz {

class Stream {
public:
    Stream();
    Stream(int device_id);
    ~Stream();

    // Get native handle
    void* NativeHandle();

    // Synchronization
    void Synchronize();
    bool Query();  // Check if complete

    // Set as current
    void SetCurrent();

    // Static
    static Stream& GetDefault();
};

}
```

### Usage

```cpp
// Create multiple streams for concurrent operations
Stream stream1;
Stream stream2;

stream1.SetCurrent();
Tensor a = Randn({1000, 1000});
Tensor c = a.MatMul(a);  // Runs on stream1

stream2.SetCurrent();
Tensor b = Randn({1000, 1000});
Tensor d = b.MatMul(b);  // Runs on stream2 concurrently

// Wait for both
stream1.Synchronize();
stream2.Synchronize();
```

## Python Bindings

```python
import pycyxwiz as cyx

# Device info
devices = cyx.device.get_all_devices()
for dev in devices:
    print(f"Device {dev.id}: {dev.name}")
    print(f"  Memory: {dev.total_memory // (1024**2)} MB")

# Device selection
cyx.device.set_device(0)
cyx.device.set_device('cuda', 0)
cyx.device.set_device('cpu')

# Memory
used = cyx.device.memory_used()
total = cyx.device.memory_total()
print(f"Memory: {used / total * 100:.1f}% used")

# Context manager
with cyx.device.DeviceScope(1):
    # Operations here run on device 1
    tensor = cyx.randn([1000, 1000])

# Check capabilities
if cyx.device.is_backend_available('cuda'):
    cyx.device.set_backend('cuda')
```

## Performance Tips

### Device Selection

1. **Use CUDA for NVIDIA GPUs**: Best performance for training
2. **Use OpenCL for AMD/Intel**: Cross-platform alternative
3. **Fall back to CPU**: For small tensors or debugging

### Memory Management

1. **Monitor memory usage**: Prevent OOM errors
2. **Clear cache periodically**: `Device::GarbageCollect()`
3. **Use pinned memory**: For large CPU-GPU transfers
4. **Pre-allocate tensors**: Avoid allocation during training

### Multi-GPU

1. **Enable peer access**: Faster GPU-to-GPU transfers
2. **Balance workload**: Distribute evenly across GPUs
3. **Minimize transfers**: Keep data on same GPU when possible

## Error Handling

```cpp
try {
    Device::SetDevice(DeviceType::CUDA, 0);
    Tensor t = Randn({10000, 10000});  // Might fail on small GPU
} catch (const DeviceException& e) {
    std::cerr << "Device error: " << e.what() << std::endl;
    // Fall back to CPU
    Device::SetDevice(DeviceType::CPU);
}
```

---

**Next**: [Tensor API](tensor.md) | [Layer API](layers.md)
