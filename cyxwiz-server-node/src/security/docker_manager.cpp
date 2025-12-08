// docker_manager.cpp - Docker container management implementation
#include "security/docker_manager.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <array>
#include <cstdio>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#define popen _popen
#define pclose _pclose
#else
#include <cstdlib>
#endif

namespace cyxwiz::servernode::security {

DockerManager::DockerManager() {
    RefreshAvailability();
}

DockerManager::~DockerManager() = default;

bool DockerManager::RefreshAvailability() {
    // Check if Docker is available
    auto result = ExecuteDockerCommand({"version", "--format", "{{.Server.Version}}"});
    docker_available_ = (result.exit_code == 0 && !result.output.empty());

    if (docker_available_) {
        docker_version_ = result.output;
        // Remove trailing newline
        while (!docker_version_.empty() && (docker_version_.back() == '\n' || docker_version_.back() == '\r')) {
            docker_version_.pop_back();
        }
        spdlog::info("DockerManager: Docker {} available", docker_version_);

        // Check for NVIDIA Docker support
        auto nvidia_result = ExecuteDockerCommand({"info", "--format", "{{.Runtimes}}"});
        nvidia_docker_available_ = (nvidia_result.exit_code == 0 &&
                                    nvidia_result.output.find("nvidia") != std::string::npos);
        if (nvidia_docker_available_) {
            spdlog::info("DockerManager: NVIDIA GPU support available");
        }
    } else {
        spdlog::warn("DockerManager: Docker not available - sandboxing disabled");
    }

    return docker_available_;
}

DockerManager::CommandResult DockerManager::ExecuteDockerCommand(const std::vector<std::string>& args) {
    CommandResult result;
    result.exit_code = -1;

    // Build command line
    std::ostringstream cmd;
    cmd << "docker";
    for (const auto& arg : args) {
        cmd << " ";
        // Escape arguments containing spaces
        if (arg.find(' ') != std::string::npos) {
            cmd << "\"" << arg << "\"";
        } else {
            cmd << arg;
        }
    }
    cmd << " 2>&1";

    std::string command = cmd.str();
    spdlog::debug("DockerManager: Executing: {}", command);

    // Execute command
    std::array<char, 4096> buffer;
    std::string output;

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        spdlog::error("DockerManager: Failed to execute command");
        return result;
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    result.exit_code = pclose(pipe);
#ifndef _WIN32
    if (WIFEXITED(result.exit_code)) {
        result.exit_code = WEXITSTATUS(result.exit_code);
    }
#endif

    result.output = output;
    return result;
}

bool DockerManager::PullImage(const std::string& image) {
    spdlog::info("DockerManager: Pulling image {}", image);
    auto result = ExecuteDockerCommand({"pull", image});
    return result.exit_code == 0;
}

bool DockerManager::ImageExists(const std::string& image) {
    auto result = ExecuteDockerCommand({"image", "inspect", image});
    return result.exit_code == 0;
}

std::vector<std::string> DockerManager::ListImages() {
    std::vector<std::string> images;

    auto result = ExecuteDockerCommand({"images", "--format", "{{.Repository}}:{{.Tag}}"});
    if (result.exit_code != 0) {
        return images;
    }

    std::istringstream iss(result.output);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line != "<none>:<none>") {
            images.push_back(line);
        }
    }

    return images;
}

std::vector<std::string> DockerManager::BuildRunArgs(const std::string& name,
                                                       const ContainerConfig& config) {
    std::vector<std::string> args = {"run", "-d"};

    // Container name
    if (!name.empty()) {
        args.push_back("--name");
        args.push_back(name);
    }

    // Memory limit
    if (config.memory_limit_mb > 0) {
        args.push_back("-m");
        args.push_back(std::to_string(config.memory_limit_mb) + "m");
    }

    // CPU quota
    if (config.cpu_quota > 0 && config.cpu_quota < 1.0f) {
        args.push_back("--cpus");
        args.push_back(std::to_string(config.cpu_quota));
    }

    // Network
    if (!config.network_enabled) {
        args.push_back("--network");
        args.push_back("none");
    }

    // Port mapping
    if (config.port > 0) {
        args.push_back("-p");
        args.push_back(std::to_string(config.port) + ":8080");
    }

    // Environment variables
    for (const auto& env : config.environment) {
        args.push_back("-e");
        args.push_back(env);
    }

    // Volume mounts
    for (const auto& vol : config.volumes) {
        args.push_back("-v");
        args.push_back(vol);
    }

    // GPU support
    if (config.gpu_enabled && nvidia_docker_available_) {
        args.push_back("--gpus");
        args.push_back("all");
    }

    // Model path as volume
    if (!config.model_path.empty()) {
        args.push_back("-v");
        args.push_back(config.model_path + ":/model:ro");
    }

    // Security options
    args.push_back("--security-opt");
    args.push_back("no-new-privileges:true");

    // Read-only root filesystem (model files should be in /model)
    args.push_back("--read-only");

    // Temp filesystem for runtime
    args.push_back("--tmpfs");
    args.push_back("/tmp:rw,noexec,nosuid,size=512m");

    // Image
    args.push_back(config.image);

    return args;
}

std::string DockerManager::CreateContainer(const std::string& model_id,
                                            const ContainerConfig& config) {
    if (!docker_available_) {
        spdlog::error("DockerManager: Docker not available");
        return "";
    }

    // Check if image exists, pull if not
    if (!ImageExists(config.image)) {
        spdlog::info("DockerManager: Image {} not found, pulling...", config.image);
        if (!PullImage(config.image)) {
            spdlog::error("DockerManager: Failed to pull image {}", config.image);
            return "";
        }
    }

    // Generate container name
    std::string container_name = "cyxwiz-model-" + model_id;

    // Build and execute run command
    auto args = BuildRunArgs(container_name, config);
    auto result = ExecuteDockerCommand(args);

    if (result.exit_code != 0) {
        spdlog::error("DockerManager: Failed to create container: {}", result.output);
        return "";
    }

    // Container ID is in output
    std::string container_id = result.output;
    while (!container_id.empty() && (container_id.back() == '\n' || container_id.back() == '\r')) {
        container_id.pop_back();
    }

    spdlog::info("DockerManager: Created container {} for model {}", container_id.substr(0, 12), model_id);
    return container_id;
}

bool DockerManager::StartContainer(const std::string& container_id) {
    auto result = ExecuteDockerCommand({"start", container_id});
    if (result.exit_code == 0) {
        spdlog::info("DockerManager: Started container {}", container_id.substr(0, 12));
    }
    return result.exit_code == 0;
}

bool DockerManager::StopContainer(const std::string& container_id, int timeout_seconds) {
    auto result = ExecuteDockerCommand({"stop", "-t", std::to_string(timeout_seconds), container_id});
    if (result.exit_code == 0) {
        spdlog::info("DockerManager: Stopped container {}", container_id.substr(0, 12));
    }
    return result.exit_code == 0;
}

bool DockerManager::KillContainer(const std::string& container_id) {
    auto result = ExecuteDockerCommand({"kill", container_id});
    return result.exit_code == 0;
}

bool DockerManager::RemoveContainer(const std::string& container_id, bool force) {
    std::vector<std::string> args = {"rm"};
    if (force) args.push_back("-f");
    args.push_back(container_id);

    auto result = ExecuteDockerCommand(args);
    if (result.exit_code == 0) {
        spdlog::info("DockerManager: Removed container {}", container_id.substr(0, 12));
    }
    return result.exit_code == 0;
}

bool DockerManager::IsRunning(const std::string& container_id) {
    auto result = ExecuteDockerCommand({"inspect", "-f", "{{.State.Running}}", container_id});
    return result.exit_code == 0 && result.output.find("true") != std::string::npos;
}

std::optional<ContainerInfo> DockerManager::GetContainerInfo(const std::string& container_id) {
    auto result = ExecuteDockerCommand({
        "inspect", "-f",
        "{{.Id}}|{{.Name}}|{{.Config.Image}}|{{.State.Status}}|{{.Created}}", container_id
    });

    if (result.exit_code != 0) {
        return std::nullopt;
    }

    ContainerInfo info;
    std::istringstream iss(result.output);
    std::string field;
    std::vector<std::string> fields;

    while (std::getline(iss, field, '|')) {
        fields.push_back(field);
    }

    if (fields.size() >= 5) {
        info.id = fields[0];
        info.name = fields[1];
        if (!info.name.empty() && info.name[0] == '/') {
            info.name = info.name.substr(1);
        }
        info.image = fields[2];
        info.status = fields[3];
        info.created = fields[4];
    }

    return info;
}

std::vector<ContainerInfo> DockerManager::ListContainers(bool all) {
    std::vector<ContainerInfo> containers;

    std::vector<std::string> args = {"ps", "--format",
        "{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}"};
    if (all) args.push_back("-a");

    auto result = ExecuteDockerCommand(args);
    if (result.exit_code != 0) {
        return containers;
    }

    std::istringstream iss(result.output);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;

        ContainerInfo info;
        std::vector<std::string> fields;
        std::istringstream line_stream(line);
        std::string field;

        while (std::getline(line_stream, field, '|')) {
            fields.push_back(field);
        }

        if (fields.size() >= 4) {
            info.id = fields[0];
            info.name = fields[1];
            info.image = fields[2];
            info.status = fields[3];
            containers.push_back(info);
        }
    }

    return containers;
}

std::string DockerManager::GetContainerLogs(const std::string& container_id, int lines) {
    auto result = ExecuteDockerCommand({"logs", "--tail", std::to_string(lines), container_id});
    return result.output;
}

std::optional<ContainerStats> DockerManager::GetStats(const std::string& container_id) {
    auto result = ExecuteDockerCommand({
        "stats", "--no-stream", "--format",
        "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}", container_id
    });

    if (result.exit_code != 0) {
        return std::nullopt;
    }

    ContainerStats stats;

    // Parse output like: "0.50%|100MiB / 512MiB|1.2kB / 0B"
    std::string line = result.output;
    size_t pos1 = line.find('|');
    size_t pos2 = line.find('|', pos1 + 1);

    if (pos1 != std::string::npos) {
        std::string cpu = line.substr(0, pos1);
        // Remove % sign
        cpu.erase(std::remove(cpu.begin(), cpu.end(), '%'), cpu.end());
        try {
            stats.cpu_percent = std::stof(cpu);
        } catch (...) {}
    }

    if (pos1 != std::string::npos && pos2 != std::string::npos) {
        std::string mem = line.substr(pos1 + 1, pos2 - pos1 - 1);
        // Parse memory like "100MiB / 512MiB"
        size_t slash = mem.find('/');
        if (slash != std::string::npos) {
            std::string used = mem.substr(0, slash);
            std::string limit = mem.substr(slash + 1);
            // Simple parse - just look for number
            try {
                stats.memory_used_mb = static_cast<size_t>(std::stof(used));
                stats.memory_limit_mb = static_cast<size_t>(std::stof(limit));
            } catch (...) {}
        }
    }

    stats.status = IsRunning(container_id) ? "running" : "stopped";

    return stats;
}

DockerManager::ExecResult DockerManager::ExecInContainer(const std::string& container_id,
                                                          const std::vector<std::string>& command) {
    ExecResult exec_result;
    exec_result.exit_code = -1;

    std::vector<std::string> args = {"exec", container_id};
    args.insert(args.end(), command.begin(), command.end());

    auto result = ExecuteDockerCommand(args);
    exec_result.exit_code = result.exit_code;
    exec_result.stdout_output = result.output;
    exec_result.stderr_output = result.error;

    return exec_result;
}

bool DockerManager::IsContainerHealthy(const std::string& container_id) {
    auto result = ExecuteDockerCommand({"inspect", "-f", "{{.State.Health.Status}}", container_id});
    return result.exit_code == 0 &&
           (result.output.find("healthy") != std::string::npos ||
            result.output.find("starting") != std::string::npos);
}

void DockerManager::CleanupStoppedContainers() {
    auto result = ExecuteDockerCommand({"container", "prune", "-f"});
    if (result.exit_code == 0) {
        spdlog::info("DockerManager: Cleaned up stopped containers");
    }
}

void DockerManager::RemoveUnusedImages() {
    auto result = ExecuteDockerCommand({"image", "prune", "-f"});
    if (result.exit_code == 0) {
        spdlog::info("DockerManager: Cleaned up unused images");
    }
}

// Singleton
DockerManager& DockerManagerSingleton::Instance() {
    static DockerManager instance;
    return instance;
}

} // namespace cyxwiz::servernode::security
