// tls_config.h - TLS configuration for gRPC servers and clients
#pragma once

#include <string>
#include <memory>
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>

namespace cyxwiz::servernode::security {

class TLSConfig {
public:
    TLSConfig() = default;
    ~TLSConfig() = default;

    // Load certificates from files
    // cert_path: Path to PEM certificate file
    // key_path: Path to PEM private key file
    // ca_path: Optional path to CA certificate for client verification
    bool LoadFromFiles(const std::string& cert_path,
                       const std::string& key_path,
                       const std::string& ca_path = "");

    // Load certificates from PEM strings directly
    bool LoadFromPEM(const std::string& cert_pem,
                     const std::string& key_pem,
                     const std::string& ca_pem = "");

    // Generate self-signed certificate for development
    // Generates cert and key files at specified paths
    static bool GenerateSelfSigned(const std::string& cert_path,
                                   const std::string& key_path,
                                   const std::string& common_name,
                                   int validity_days = 365);

    // Get gRPC server credentials
    // If mutual TLS is enabled (CA cert loaded), clients must present valid certs
    std::shared_ptr<grpc::ServerCredentials> GetServerCredentials() const;

    // Get gRPC client credentials
    // target_name: Expected server name for verification (empty = skip verification)
    std::shared_ptr<grpc::ChannelCredentials> GetClientCredentials(
        const std::string& target_name = "") const;

    // Check if TLS is properly configured
    bool IsLoaded() const { return loaded_; }

    // Check if mutual TLS is enabled
    bool IsMutualTLS() const { return !ca_pem_.empty(); }

    // Get certificate info
    std::string GetCertSubject() const;
    std::string GetCertExpiry() const;

    // Static helper to read file contents
    static std::string ReadFile(const std::string& path);

private:
    std::string cert_pem_;
    std::string key_pem_;
    std::string ca_pem_;
    bool loaded_ = false;

    // Parse certificate to extract info
    bool ParseCertificateInfo();
    std::string cert_subject_;
    std::string cert_expiry_;
};

// Global TLS configuration singleton
class TLSManager {
public:
    static TLSManager& Instance();

    // Initialize TLS from config file paths
    bool Initialize(const std::string& cert_path,
                    const std::string& key_path,
                    const std::string& ca_path = "");

    // Initialize with auto-generated self-signed cert (for development)
    bool InitializeSelfSigned(const std::string& data_dir,
                              const std::string& common_name = "localhost");

    // Check if TLS is enabled and ready
    bool IsEnabled() const { return config_.IsLoaded(); }

    // Get the TLS configuration
    TLSConfig& GetConfig() { return config_; }
    const TLSConfig& GetConfig() const { return config_; }

    // Convenience methods for getting credentials
    std::shared_ptr<grpc::ServerCredentials> GetServerCredentials() const;
    std::shared_ptr<grpc::ChannelCredentials> GetClientCredentials(
        const std::string& target_name = "") const;

    // Get insecure credentials (fallback when TLS disabled)
    static std::shared_ptr<grpc::ServerCredentials> GetInsecureServerCredentials();
    static std::shared_ptr<grpc::ChannelCredentials> GetInsecureClientCredentials();

private:
    TLSManager() = default;
    TLSConfig config_;
};

} // namespace cyxwiz::servernode::security
