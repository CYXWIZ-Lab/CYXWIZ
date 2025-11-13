#include "node_client.h"
#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "iphlpapi.lib")
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

namespace cyxwiz {
namespace servernode {

// ============================================================================
// HardwareDetector Implementation
// ============================================================================

protocol::NodeInfo HardwareDetector::DetectHardwareInfo(const std::string& node_id) {
    protocol::NodeInfo info;

    // Basic identification
    info.set_node_id(node_id);
    info.set_name("CyxWiz-Node-" + node_id.substr(0, 8));

    auto* version = info.mutable_version();
    // Parse version string (e.g., "0.1.0") or use defaults
    std::string version_str = cyxwiz::GetVersionString();
    int major = 0, minor = 1, patch = 0;
    sscanf(version_str.c_str(), "%d.%d.%d", &major, &minor, &patch);
    version->set_major(major);
    version->set_minor(minor);
    version->set_patch(patch);
    version->set_build(version_str);

    // Hardware
    info.set_cpu_cores(GetCPUCores());
    info.set_ram_total(GetTotalRAM());
    info.set_ram_available(GetAvailableRAM());

    // Detect GPU/compute devices
    DetectDevices(&info);

    // Network
    info.set_ip_address(GetLocalIPAddress());
    info.set_port(50052);  // Deployment service port
    info.set_region("unknown");  // TODO: Detect geographic region

    // Performance (initialize with defaults)
    info.set_compute_score(0.0);
    info.set_reputation_score(0.5);  // Start with neutral reputation
    info.set_total_jobs_completed(0);
    info.set_total_compute_hours(0);
    info.set_average_rating(0.0);

    // Staking (not implemented yet)
    info.set_staked_amount(0.0);
    info.set_wallet_address("");  // TODO: Get from config or generate

    // Status
    info.set_is_online(true);
    info.set_last_heartbeat(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );
    info.set_uptime_percentage(0.0);

    // Deployment capabilities
    info.add_supported_formats("ONNX");
    info.add_supported_formats("GGUF");
    info.add_supported_formats("PyTorch");
    info.set_max_model_size(10LL * 1024 * 1024 * 1024);  // 10 GB
    info.set_supports_terminal_access(true);
    info.add_available_runtimes("onnxruntime");
    info.add_available_runtimes("llama.cpp");
    info.add_available_runtimes("pytorch");

    return info;
}

int HardwareDetector::GetCPUCores() {
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return static_cast<int>(sysinfo.dwNumberOfProcessors);
#else
    return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#endif
}

int64_t HardwareDetector::GetTotalRAM() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return static_cast<int64_t>(memInfo.ullTotalPhys);
#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    return static_cast<int64_t>(memInfo.totalram) * memInfo.mem_unit;
#endif
}

int64_t HardwareDetector::GetAvailableRAM() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return static_cast<int64_t>(memInfo.ullAvailPhys);
#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    return static_cast<int64_t>(memInfo.freeram) * memInfo.mem_unit;
#endif
}

void HardwareDetector::DetectDevices(protocol::NodeInfo* node_info) {
    // Get available devices from the backend
    auto devices = cyxwiz::Device::GetAvailableDevices();

    spdlog::info("Detected {} device(s)", devices.size());

    for (const auto& device_info : devices) {
        // Add a new HardwareCapabilities message for each device
        auto* hw_cap = node_info->add_devices();

        // Map cyxwiz::DeviceType to protocol::DeviceType
        switch (device_info.type) {
            case cyxwiz::DeviceType::CPU:
                hw_cap->set_device_type(protocol::DEVICE_CPU);
                break;
            case cyxwiz::DeviceType::CUDA:
                hw_cap->set_device_type(protocol::DEVICE_CUDA);
                break;
            case cyxwiz::DeviceType::OPENCL:
                hw_cap->set_device_type(protocol::DEVICE_OPENCL);
                break;
            case cyxwiz::DeviceType::METAL:
                hw_cap->set_device_type(protocol::DEVICE_METAL);
                break;
            case cyxwiz::DeviceType::VULKAN:
                hw_cap->set_device_type(protocol::DEVICE_VULKAN);
                break;
            default:
                hw_cap->set_device_type(protocol::DEVICE_UNKNOWN);
                break;
        }

        // Set basic device info
        hw_cap->set_device_name(device_info.name);
        hw_cap->set_memory_total(static_cast<int64_t>(device_info.memory_total));
        hw_cap->set_memory_available(static_cast<int64_t>(device_info.memory_available));
        hw_cap->set_compute_units(device_info.compute_units);
        hw_cap->set_supports_fp64(device_info.supports_fp64);
        hw_cap->set_supports_fp16(device_info.supports_fp16);

        // Set extended GPU info (for GPU devices)
        if (device_info.type == cyxwiz::DeviceType::CUDA ||
            device_info.type == cyxwiz::DeviceType::OPENCL) {
            hw_cap->set_gpu_model(device_info.name);
            hw_cap->set_vram_total(static_cast<int64_t>(device_info.memory_total));
            hw_cap->set_vram_available(static_cast<int64_t>(device_info.memory_available));

            // TODO: Extract more detailed GPU info (driver version, CUDA version, PCIe info, compute capability)
            // This would require direct ArrayFire/CUDA API calls for full details
        }

        spdlog::info("  [{}] {} - {:.2f} GB total, {:.2f} GB available",
                     device_info.device_id,
                     device_info.name,
                     device_info.memory_total / (1024.0 * 1024.0 * 1024.0),
                     device_info.memory_available / (1024.0 * 1024.0 * 1024.0));
    }
}

std::string HardwareDetector::GetLocalIPAddress() {
#ifdef _WIN32
    // Windows implementation
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == SOCKET_ERROR) {
        return "127.0.0.1";
    }

    struct addrinfo hints, *result;
    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;  // IPv4
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    if (getaddrinfo(hostname, NULL, &hints, &result) != 0) {
        return "127.0.0.1";
    }

    struct sockaddr_in* addr = (struct sockaddr_in*)result->ai_addr;
    char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr->sin_addr, ip_str, INET_ADDRSTRLEN);

    std::string ip(ip_str);
    freeaddrinfo(result);
    return ip;
#else
    // Linux/Unix implementation
    struct ifaddrs *ifAddrStruct = nullptr;
    struct ifaddrs *ifa = nullptr;
    void *tmpAddrPtr = nullptr;

    getifaddrs(&ifAddrStruct);

    for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) {
            continue;
        }

        if (ifa->ifa_addr->sa_family == AF_INET) {
            // IPv4 Address
            tmpAddrPtr = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char addressBuffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);

            // Skip loopback
            std::string addr(addressBuffer);
            if (addr != "127.0.0.1" && ifa->ifa_name[0] != 'l') {  // Skip "lo" interface
                if (ifAddrStruct != nullptr) freeifaddrs(ifAddrStruct);
                return addr;
            }
        }
    }

    if (ifAddrStruct != nullptr) freeifaddrs(ifAddrStruct);
    return "127.0.0.1";
#endif
}

// ============================================================================
// NodeClient Implementation
// ============================================================================

NodeClient::NodeClient(const std::string& central_server_address, const std::string& node_id)
    : central_server_address_(central_server_address)
    , node_id_(node_id)
    , is_registered_(false)
    , heartbeat_interval_seconds_(10)
{
    spdlog::info("NodeClient created for Central Server: {}", central_server_address);

    // Create gRPC channel
    channel_ = grpc::CreateChannel(central_server_address, grpc::InsecureChannelCredentials());
    stub_ = protocol::NodeService::NewStub(channel_);
}

NodeClient::~NodeClient() {
    StopHeartbeat();
}

bool NodeClient::Register() {
    spdlog::info("Registering node {} with Central Server...", node_id_);

    // Detect hardware capabilities
    auto node_info = HardwareDetector::DetectHardwareInfo(node_id_);

    // Create registration request
    protocol::RegisterNodeRequest request;
    *request.mutable_info() = node_info;
    request.set_authentication_token("");  // TODO: Implement authentication
    request.set_public_key("");            // TODO: Implement crypto

    // Send registration request
    protocol::RegisterNodeResponse response;
    grpc::ClientContext context;

    // Set timeout
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(10);
    context.set_deadline(deadline);

    grpc::Status status = stub_->RegisterNode(&context, request, &response);

    if (status.ok()) {
        if (response.status() == protocol::STATUS_SUCCESS) {
            node_id_ = response.node_id();
            session_token_ = response.session_token();
            is_registered_ = true;

            spdlog::info("Node registered successfully!");
            spdlog::info("  Node ID: {}", node_id_);
            spdlog::info("  Session Token: {}", session_token_);
            return true;
        } else {
            spdlog::error("Registration failed: {}",
                         response.has_error() ? response.error().message() : "Unknown error");
            return false;
        }
    } else {
        spdlog::error("gRPC error during registration: {} (code: {})",
                     status.error_message(), static_cast<int>(status.error_code()));
        return false;
    }
}

bool NodeClient::StartHeartbeat(int interval_seconds) {
    if (!is_registered_) {
        spdlog::error("Cannot start heartbeat: node not registered");
        return false;
    }

    heartbeat_interval_seconds_ = interval_seconds;
    should_stop_heartbeat_ = false;

    heartbeat_thread_ = std::thread([this]() { HeartbeatLoop(); });

    spdlog::info("Heartbeat started (interval: {}s)", interval_seconds);
    return true;
}

void NodeClient::StopHeartbeat() {
    should_stop_heartbeat_ = true;
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
    spdlog::info("Heartbeat stopped");
}

bool NodeClient::SendHeartbeat() {
    if (!is_registered_) {
        return false;
    }

    protocol::HeartbeatRequest request;
    request.set_node_id(node_id_);

    // Add current status
    auto* current_status = request.mutable_current_status();
    current_status->set_node_id(node_id_);
    current_status->set_is_online(true);
    current_status->set_ram_available(HardwareDetector::GetAvailableRAM());
    current_status->set_last_heartbeat(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );

    // Add active jobs
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        for (const auto& job_id : active_jobs_) {
            request.add_active_jobs(job_id);
        }
    }

    protocol::HeartbeatResponse response;
    grpc::ClientContext context;

    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
    context.set_deadline(deadline);

    grpc::Status status = stub_->Heartbeat(&context, request, &response);

    if (status.ok() && response.status() == protocol::STATUS_SUCCESS) {
        spdlog::debug("Heartbeat sent successfully");
        return response.keep_alive();
    } else {
        spdlog::warn("Heartbeat failed: {}", status.error_message());
        return false;
    }
}

void NodeClient::SetActiveJobs(const std::vector<std::string>& job_ids) {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    active_jobs_ = job_ids;
}

void NodeClient::HeartbeatLoop() {
    spdlog::debug("Heartbeat loop started");

    while (!should_stop_heartbeat_) {
        if (!SendHeartbeat()) {
            spdlog::warn("Heartbeat failed, will retry...");
        }

        // Sleep for interval
        for (int i = 0; i < heartbeat_interval_seconds_; ++i) {
            if (should_stop_heartbeat_) break;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    spdlog::debug("Heartbeat loop stopped");
}

} // namespace servernode
} // namespace cyxwiz
