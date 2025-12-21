// daemon_main.cpp - Entry point for cyxwiz-server-daemon
// Headless service that runs training, deployment, and provides IPC for GUI

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <iostream>
#include <memory>
#include <thread>
#include <csignal>
#include <atomic>
#include <cstring>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <shlobj.h>
#endif

#include "deployment_handler.h"
#include "deployment_manager.h"
#include "terminal_handler.h"
#include "node_client.h"
#include "job_executor.h"
#include "node_service.h"
#include "job_execution_service.h"
#include "core/backend_manager.h"
#include "core/metrics_collector.h"
#include "core/state_manager.h"
#include "core/config_manager.h"
#include "ipc/daemon_service.h"
#include "security/tls_config.h"
#include "security/audit_logger.h"
#include "http/openai_api_server.h"
#include "inference_handler.h"

// Global flag for shutdown
std::atomic<bool> g_shutdown{false};

void SignalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("Received shutdown signal");
        g_shutdown = true;
    }
}

void PrintUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nOptions:\n"
              << "  --ipc-address=ADDR   IPC address for GUI connection (default: localhost:50054)\n"
              << "  --central-server=ADDR Central server address (default: localhost:50051)\n"
              << "  --http-port=PORT     HTTP REST API port (default: 8082)\n"
              << "  --inference-addr=ADDR gRPC InferenceService address (default: 0.0.0.0:50057)\n"
              << "  --config=PATH        Path to config file (default: ~/.cyxwiz/daemon.yaml)\n"
              << "  --tls                Enable TLS for gRPC servers\n"
              << "  --tls-cert=PATH      Path to TLS certificate file\n"
              << "  --tls-key=PATH       Path to TLS private key file\n"
              << "  --tls-ca=PATH        Path to CA certificate (enables mutual TLS)\n"
              << "  --tls-auto           Auto-generate self-signed certificate if none exists\n"
              << "  --help               Show this help message\n"
              << "\nThe daemon provides:\n"
              << "  - gRPC IPC service for GUI/TUI client connections\n"
              << "  - P2P service for Engine connections\n"
              << "  - Model deployment with OpenAI-compatible API\n"
              << "  - HTTP REST API for inference at http://localhost:8082/v1/predict\n"
              << "  - Job execution for distributed training\n"
              << std::endl;
}

struct DaemonConfig {
    std::string ipc_address = "localhost:50054";
    std::string p2p_address = "0.0.0.0:50052";
    std::string terminal_address = "0.0.0.0:50053";
    std::string node_service_address = "0.0.0.0:50055";
    std::string deployment_address = "0.0.0.0:50056";
    std::string inference_address = "0.0.0.0:50057";
    std::string central_server = "localhost:50051";
    std::string config_path;
    int http_port = 8082;  // HTTP REST API port

    // P2P Authentication (must match Central Server's jwt.secret or jwt.p2p_secret)
    // Default matches Central Server's config.toml for development
    std::string p2p_secret = "your-super-secret-jwt-key-change-in-production";

    // TLS settings
    bool enable_tls = false;
    std::string tls_cert_path;
    std::string tls_key_path;
    std::string tls_ca_path;
    bool tls_auto = false;
};

// Node ID persistence helpers
std::string GetNodeIdConfigPath() {
#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
        std::string config_dir = std::string(path) + "\\CyxWiz";
        CreateDirectoryA(config_dir.c_str(), nullptr);
        return config_dir + "\\node_registration.json";
    }
    return "node_registration.json";
#else
    const char* home = getenv("HOME");
    if (home) {
        std::string config_dir = std::string(home) + "/.config/cyxwiz";
        mkdir(config_dir.c_str(), 0755);
        return config_dir + "/node_registration.json";
    }
    return "node_registration.json";
#endif
}

std::string LoadPersistedNodeId() {
    try {
        std::string config_path = GetNodeIdConfigPath();
        std::ifstream file(config_path);
        if (!file.is_open()) {
            return "";  // No persisted node_id
        }

        nlohmann::json j;
        file >> j;

        if (j.contains("central_server_node_id") && j["central_server_node_id"].is_string()) {
            std::string node_id = j["central_server_node_id"].get<std::string>();
            spdlog::info("Loaded persisted Central Server node_id: {}", node_id);
            return node_id;
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to load persisted node_id: {}", e.what());
    }
    return "";
}

void SaveNodeId(const std::string& central_server_node_id, const std::string& local_node_id) {
    try {
        std::string config_path = GetNodeIdConfigPath();

        nlohmann::json j;
        j["central_server_node_id"] = central_server_node_id;
        j["local_node_id"] = local_node_id;
        j["last_updated"] = std::time(nullptr);

        std::ofstream file(config_path);
        if (file.is_open()) {
            file << j.dump(2);
            spdlog::info("Saved node_id to {}", config_path);
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to save node_id: {}", e.what());
    }
}

// Global callback for node_id persistence (called by DaemonServiceImpl)
std::function<void(const std::string&, const std::string&)> g_save_node_id_callback;

DaemonConfig ParseArgs(int argc, char** argv) {
    DaemonConfig config;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--ipc-address=", 14) == 0) {
            config.ipc_address = argv[i] + 14;
        } else if (std::strncmp(argv[i], "--central-server=", 17) == 0) {
            config.central_server = argv[i] + 17;
        } else if (std::strncmp(argv[i], "--http-port=", 12) == 0) {
            config.http_port = std::atoi(argv[i] + 12);
        } else if (std::strncmp(argv[i], "--inference-addr=", 17) == 0) {
            config.inference_address = argv[i] + 17;
        } else if (std::strncmp(argv[i], "--config=", 9) == 0) {
            config.config_path = argv[i] + 9;
        } else if (std::strncmp(argv[i], "--p2p-secret=", 13) == 0) {
            config.p2p_secret = argv[i] + 13;
        } else if (std::strcmp(argv[i], "--tls") == 0) {
            config.enable_tls = true;
        } else if (std::strncmp(argv[i], "--tls-cert=", 11) == 0) {
            config.tls_cert_path = argv[i] + 11;
        } else if (std::strncmp(argv[i], "--tls-key=", 10) == 0) {
            config.tls_key_path = argv[i] + 10;
        } else if (std::strncmp(argv[i], "--tls-ca=", 9) == 0) {
            config.tls_ca_path = argv[i] + 9;
        } else if (std::strcmp(argv[i], "--tls-auto") == 0) {
            config.tls_auto = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            PrintUsage(argv[0]);
            std::exit(0);
        }
    }

    return config;
}

int main(int argc, char** argv) {
    spdlog::info("CyxWiz Server Daemon v{}", cyxwiz::GetVersionString());
    spdlog::info("========================================");

    // Parse arguments
    DaemonConfig daemon_config = ParseArgs(argc, argv);

    // Install signal handlers
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // Initialize backend
    if (!cyxwiz::Initialize()) {
        spdlog::error("Failed to initialize backend");
        return 1;
    }

    // Load or generate node ID
    // First try to load persisted Central Server node_id
    std::string persisted_node_id = LoadPersistedNodeId();
    std::string local_node_id = "node_" + std::to_string(std::time(nullptr));

    // Use persisted ID if available, otherwise use local ID
    // Note: The NodeClient will use this as initial ID, but may receive a new UUID from Central Server
    std::string node_id = persisted_node_id.empty() ? local_node_id : persisted_node_id;

    if (!persisted_node_id.empty()) {
        spdlog::info("Using persisted Central Server node_id: {}", node_id);
    } else {
        spdlog::info("Generated new local node_id: {}", node_id);
    }

    // Set up save callback
    g_save_node_id_callback = [local_node_id](const std::string& central_id, const std::string& /*unused*/) {
        SaveNodeId(central_id, local_node_id);
    };

    spdlog::info("Node ID: {}", node_id);
    spdlog::info("IPC service: {}", daemon_config.ipc_address);
    spdlog::info("P2P service: {}", daemon_config.p2p_address);
    spdlog::info("Central server: {}", daemon_config.central_server);

    // Initialize TLS if enabled
    std::shared_ptr<grpc::ServerCredentials> server_credentials;
    auto& tls_manager = cyxwiz::servernode::security::TLSManager::Instance();

    if (daemon_config.enable_tls || daemon_config.tls_auto) {
        spdlog::info("TLS enabled, initializing...");

        // Auto-generate self-signed cert if requested and no certs provided
        if (daemon_config.tls_auto &&
            (daemon_config.tls_cert_path.empty() || daemon_config.tls_key_path.empty())) {

            std::string default_cert = "cyxwiz-daemon.crt";
            std::string default_key = "cyxwiz-daemon.key";

            // Check if certs already exist
            std::ifstream cert_test(default_cert);
            if (!cert_test.good()) {
                spdlog::info("Auto-generating self-signed TLS certificate...");
                if (cyxwiz::servernode::security::TLSConfig::GenerateSelfSigned(
                        default_cert, default_key, "cyxwiz-daemon", 365)) {
                    spdlog::info("Self-signed certificate generated");
                    daemon_config.tls_cert_path = default_cert;
                    daemon_config.tls_key_path = default_key;
                } else {
                    spdlog::error("Failed to generate self-signed certificate");
                    spdlog::warn("Falling back to insecure mode");
                    daemon_config.enable_tls = false;
                }
            } else {
                spdlog::info("Using existing certificate: {}", default_cert);
                daemon_config.tls_cert_path = default_cert;
                daemon_config.tls_key_path = default_key;
            }
        }

        // Load TLS config if we have cert paths
        if (!daemon_config.tls_cert_path.empty() && !daemon_config.tls_key_path.empty()) {
            if (tls_manager.Initialize(daemon_config.tls_cert_path,
                                       daemon_config.tls_key_path,
                                       daemon_config.tls_ca_path)) {
                server_credentials = tls_manager.GetConfig().GetServerCredentials();
                spdlog::info("TLS initialized successfully");

                // Log audit event
                cyxwiz::servernode::security::AuditLogger::Instance().Log(
                    cyxwiz::servernode::security::AuditEvent::TLS_ENABLED,
                    "system", "daemon", true, "TLS enabled for gRPC servers");
            } else {
                spdlog::error("Failed to initialize TLS - falling back to insecure mode");
                daemon_config.enable_tls = false;

                cyxwiz::servernode::security::AuditLogger::Instance().Log(
                    cyxwiz::servernode::security::AuditEvent::INVALID_CERTIFICATE,
                    "system", "daemon", false, "TLS initialization failed");
            }
        }
    }

    // Use insecure credentials if TLS not configured
    if (!server_credentials) {
        server_credentials = grpc::InsecureServerCredentials();
        spdlog::warn("Running gRPC servers WITHOUT TLS encryption");
    }

    try {
        // Initialize core services
        auto& backend = cyxwiz::servernode::core::BackendManager::Instance();

        // Initialize config manager
        auto config_manager = std::make_unique<cyxwiz::servernode::core::ConfigManager>();
        if (!daemon_config.config_path.empty()) {
            config_manager->Load(daemon_config.config_path);
        }

        // Initialize metrics collector
        auto metrics_collector = std::make_unique<cyxwiz::servernode::core::MetricsCollector>();
        metrics_collector->StartCollection();

        // Initialize state manager
        auto state_manager = std::make_unique<cyxwiz::servernode::core::StateManager>();

        // Get current device for job execution
        cyxwiz::Device* device = cyxwiz::Device::GetCurrentDevice();

        // Create JobExecutor
        auto job_executor = std::make_shared<cyxwiz::servernode::JobExecutor>(node_id, device);

        // Set up progress callback
        job_executor->SetProgressCallback([&state_manager](
            const std::string& job_id,
            double progress,
            const cyxwiz::servernode::TrainingMetrics& metrics)
        {
            spdlog::debug("Job {} progress: {:.1f}%", job_id, progress * 100.0);
            // Update state manager
            if (state_manager) {
                cyxwiz::servernode::core::JobState job;
                job.id = job_id;
                job.progress = static_cast<float>(progress);
                job.current_epoch = metrics.current_epoch;
                job.total_epochs = metrics.total_epochs;
                job.loss = static_cast<float>(metrics.loss);
                job.accuracy = static_cast<float>(metrics.accuracy);
                state_manager->UpdateJob(job);
            }
        });

        // Set up completion callback
        job_executor->SetCompletionCallback([&state_manager](
            const std::string& job_id,
            bool success,
            const std::string& error_msg)
        {
            if (success) {
                spdlog::info("Job {} completed successfully", job_id);
            } else {
                spdlog::error("Job {} failed: {}", job_id, error_msg);
            }
            if (state_manager) {
                state_manager->RemoveJob(job_id);
            }
        });

        // Create deployment manager
        auto deployment_manager = std::make_shared<cyxwiz::servernode::DeploymentManager>(node_id);

        // Create NodeServiceImpl
        auto node_service = std::make_unique<cyxwiz::servernode::NodeServiceImpl>(
            job_executor.get(),
            node_id
        );

        // Create gRPC server for NodeService
        grpc::ServerBuilder node_service_builder;
        node_service_builder.AddListeningPort(
            daemon_config.node_service_address,
            server_credentials
        );
        node_service_builder.RegisterService(node_service.get());
        auto node_grpc_server = node_service_builder.BuildAndStart();

        if (!node_grpc_server) {
            spdlog::error("Failed to start NodeService gRPC server");
            cyxwiz::Shutdown();
            return 1;
        }
        spdlog::info("NodeService started on {}", daemon_config.node_service_address);

        // Create P2P JobExecutionService
        auto p2p_service = std::make_unique<cyxwiz::server_node::JobExecutionServiceImpl>();
        p2p_service->Initialize(job_executor, daemon_config.central_server,
                              node_id, daemon_config.p2p_secret);

        if (!p2p_service->StartServer(daemon_config.p2p_address)) {
            spdlog::error("Failed to start P2P JobExecutionService");
            node_grpc_server->Shutdown();
            cyxwiz::Shutdown();
            return 1;
        }
        spdlog::info("P2P JobExecutionService started on {}", daemon_config.p2p_address);

        // Create deployment handler
        cyxwiz::servernode::DeploymentHandler deployment_handler(
            daemon_config.deployment_address,
            deployment_manager
        );

        if (!deployment_handler.Start()) {
            spdlog::error("Failed to start deployment handler");
            p2p_service->StopServer();
            node_grpc_server->Shutdown();
            cyxwiz::Shutdown();
            return 1;
        }

        // Create terminal handler
        cyxwiz::servernode::TerminalHandler terminal_handler(daemon_config.terminal_address);

        if (!terminal_handler.Start()) {
            spdlog::error("Failed to start terminal handler");
            deployment_handler.Stop();
            p2p_service->StopServer();
            node_grpc_server->Shutdown();
            cyxwiz::Shutdown();
            return 1;
        }

        // Create and start HTTP REST API server
        auto http_server = std::make_unique<cyxwiz::servernode::OpenAIAPIServer>(
            daemon_config.http_port,
            deployment_manager.get()
        );

        if (!http_server->Start()) {
            spdlog::error("Failed to start HTTP REST API server");
            terminal_handler.Stop();
            deployment_handler.Stop();
            p2p_service->StopServer();
            node_grpc_server->Shutdown();
            cyxwiz::Shutdown();
            return 1;
        }

        // Create and start gRPC InferenceService
        auto inference_server = std::make_unique<cyxwiz::servernode::InferenceServer>(
            daemon_config.inference_address,
            deployment_manager.get()
        );

        if (!inference_server->Start()) {
            spdlog::error("Failed to start InferenceService");
            http_server->Stop();
            terminal_handler.Stop();
            deployment_handler.Stop();
            p2p_service->StopServer();
            node_grpc_server->Shutdown();
            cyxwiz::Shutdown();
            return 1;
        }

        // Create NodeClient for Central Server communication (but don't auto-connect)
        // Connection happens when user applies allocations via GUI
        auto node_client = std::make_unique<cyxwiz::servernode::NodeClient>(
            daemon_config.central_server,
            node_id
        );
        spdlog::info("Central Server configured at {} (not connected - waiting for user allocation)",
                     daemon_config.central_server);

        // Create and start IPC daemon service (for GUI connections)
        auto daemon_service = std::make_unique<cyxwiz::servernode::ipc::DaemonServiceImpl>(
            node_id,
            job_executor.get(),
            deployment_manager.get(),
            node_client.get(),
            metrics_collector.get(),
            state_manager.get(),
            config_manager.get()
        );

        // Set shutdown callback
        daemon_service->SetShutdownCallback([](bool graceful) {
            spdlog::info("Shutdown requested via IPC (graceful={})", graceful);
            g_shutdown = true;
        });

        if (!daemon_service->Start(daemon_config.ipc_address)) {
            spdlog::error("Failed to start IPC daemon service");
            terminal_handler.Stop();
            deployment_handler.Stop();
            p2p_service->StopServer();
            node_grpc_server->Shutdown();
            cyxwiz::Shutdown();
            return 1;
        }

        spdlog::info("========================================");
        spdlog::info("Server Daemon is ready!");
        spdlog::info("  IPC service (GUI):    {}", daemon_config.ipc_address);
        spdlog::info("  P2P service (Engine): {}", daemon_config.p2p_address);
        spdlog::info("  Node service:         {}", daemon_config.node_service_address);
        spdlog::info("  Deployment endpoint:  {}", daemon_config.deployment_address);
        spdlog::info("  Terminal endpoint:    {}", daemon_config.terminal_address);
        spdlog::info("  HTTP REST API:        http://0.0.0.0:{}", daemon_config.http_port);
        spdlog::info("  Inference gRPC:       {}", daemon_config.inference_address);
        spdlog::info("  TLS encryption:       {}", tls_manager.IsEnabled() ? "ENABLED" : "DISABLED");
        spdlog::info("========================================");
        spdlog::info("Press Ctrl+C to shutdown");

        // Main loop - wait for shutdown signal
        while (!g_shutdown) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        spdlog::info("Shutting down gracefully...");

        // Stop services in reverse order
        spdlog::info("Stopping IPC daemon service...");
        daemon_service->Stop();

        spdlog::info("Stopping HTTP REST API server...");
        http_server->Stop();

        spdlog::info("Stopping InferenceService...");
        inference_server->Stop();

        spdlog::info("Stopping metrics collector...");
        metrics_collector->StopCollection();

        spdlog::info("Stopping P2P JobExecutionService...");
        p2p_service->StopServer();

        spdlog::info("Stopping NodeService gRPC server...");
        node_grpc_server->Shutdown();

        spdlog::info("Stopping terminal and deployment handlers...");
        terminal_handler.Stop();
        deployment_handler.Stop();

        spdlog::info("Server Daemon shutdown complete");

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        cyxwiz::Shutdown();
        return 1;
    }

    cyxwiz::Shutdown();
    return 0;
}
