// docker_manager.h - Docker container management for model sandboxing
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace cyxwiz::servernode::security {

// Container configuration for model deployment
struct ContainerConfig {
    std::string image = "cyxwiz/model-runner:latest";  // Base image
    std::string model_path;                             // Path to model file
    std::string name;                                   // Container name
    size_t memory_limit_mb = 4096;                      // RAM limit
    size_t gpu_memory_limit_mb = 0;                     // VRAM limit (0 = no limit)
    float cpu_quota = 1.0f;                             // CPU fraction (0.0-1.0)
    bool network_enabled = false;                       // Allow network access
    int port = 0;                                       // Exposed port (0 = auto)
    std::vector<std::string> environment;              // Environment variables
    std::vector<std::string> volumes;                  // Volume mounts
    bool gpu_enabled = false;                          // Enable GPU passthrough
};

// Container resource statistics
struct ContainerStats {
    float cpu_percent = 0.0f;
    size_t memory_used_mb = 0;
    size_t memory_limit_mb = 0;
    size_t network_rx_bytes = 0;
    size_t network_tx_bytes = 0;
    std::string status;                                // running, paused, stopped
};

// Container information
struct ContainerInfo {
    std::string id;
    std::string name;
    std::string image;
    std::string status;
    std::string created;
    int port = 0;
    ContainerStats stats;
};

class DockerManager {
public:
    DockerManager();
    ~DockerManager();

    // Check Docker availability
    bool IsAvailable() const { return docker_available_; }
    std::string GetDockerVersion() const { return docker_version_; }
    bool RefreshAvailability();

    // Image management
    bool PullImage(const std::string& image);
    bool ImageExists(const std::string& image);
    std::vector<std::string> ListImages();

    // Container lifecycle
    std::string CreateContainer(const std::string& model_id, const ContainerConfig& config);
    bool StartContainer(const std::string& container_id);
    bool StopContainer(const std::string& container_id, int timeout_seconds = 10);
    bool KillContainer(const std::string& container_id);
    bool RemoveContainer(const std::string& container_id, bool force = false);

    // Container info
    bool IsRunning(const std::string& container_id);
    std::optional<ContainerInfo> GetContainerInfo(const std::string& container_id);
    std::vector<ContainerInfo> ListContainers(bool all = false);
    std::string GetContainerLogs(const std::string& container_id, int lines = 100);

    // Resource monitoring
    std::optional<ContainerStats> GetStats(const std::string& container_id);

    // Execute command in container
    struct ExecResult {
        int exit_code;
        std::string stdout_output;
        std::string stderr_output;
    };
    ExecResult ExecInContainer(const std::string& container_id,
                                const std::vector<std::string>& command);

    // Health check
    bool IsContainerHealthy(const std::string& container_id);

    // GPU support
    bool HasNvidiaDocker() const { return nvidia_docker_available_; }

    // Cleanup
    void CleanupStoppedContainers();
    void RemoveUnusedImages();

private:
    bool docker_available_ = false;
    bool nvidia_docker_available_ = false;
    std::string docker_version_;

    // Execute Docker CLI command
    struct CommandResult {
        int exit_code;
        std::string output;
        std::string error;
    };
    CommandResult ExecuteDockerCommand(const std::vector<std::string>& args);

    // Parse Docker JSON output
    std::vector<ContainerInfo> ParseContainerList(const std::string& json);
    ContainerStats ParseContainerStats(const std::string& json);

    // Build docker run arguments
    std::vector<std::string> BuildRunArgs(const std::string& name,
                                           const ContainerConfig& config);
};

// Global Docker manager singleton
class DockerManagerSingleton {
public:
    static DockerManager& Instance();

private:
    DockerManagerSingleton() = default;
};

} // namespace cyxwiz::servernode::security
