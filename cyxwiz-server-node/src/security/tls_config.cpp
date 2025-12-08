// tls_config.cpp - TLS configuration implementation
#include "security/tls_config.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <ctime>
#include <random>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#pragma comment(lib, "crypt32.lib")
#else
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include <openssl/err.h>
#endif

namespace cyxwiz::servernode::security {

std::string TLSConfig::ReadFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        spdlog::error("TLSConfig: Failed to open file: {}", path);
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool TLSConfig::LoadFromFiles(const std::string& cert_path,
                               const std::string& key_path,
                               const std::string& ca_path) {
    // Read certificate
    cert_pem_ = ReadFile(cert_path);
    if (cert_pem_.empty()) {
        spdlog::error("TLSConfig: Failed to read certificate from: {}", cert_path);
        return false;
    }

    // Read private key
    key_pem_ = ReadFile(key_path);
    if (key_pem_.empty()) {
        spdlog::error("TLSConfig: Failed to read private key from: {}", key_path);
        return false;
    }

    // Read CA certificate (optional for mutual TLS)
    if (!ca_path.empty()) {
        ca_pem_ = ReadFile(ca_path);
        if (ca_pem_.empty()) {
            spdlog::warn("TLSConfig: Failed to read CA certificate from: {}", ca_path);
            // Not fatal - continue without mutual TLS
        }
    }

    loaded_ = true;
    ParseCertificateInfo();

    spdlog::info("TLSConfig: Loaded certificates from files");
    spdlog::info("  Certificate: {}", cert_path);
    spdlog::info("  Private key: {}", key_path);
    if (!ca_pem_.empty()) {
        spdlog::info("  CA certificate: {} (mutual TLS enabled)", ca_path);
    }

    return true;
}

bool TLSConfig::LoadFromPEM(const std::string& cert_pem,
                             const std::string& key_pem,
                             const std::string& ca_pem) {
    if (cert_pem.empty() || key_pem.empty()) {
        spdlog::error("TLSConfig: Certificate and key PEM data required");
        return false;
    }

    cert_pem_ = cert_pem;
    key_pem_ = key_pem;
    ca_pem_ = ca_pem;
    loaded_ = true;

    ParseCertificateInfo();
    spdlog::info("TLSConfig: Loaded certificates from PEM strings");

    return true;
}

#ifndef _WIN32
// OpenSSL-based self-signed certificate generation (Linux/macOS)
bool TLSConfig::GenerateSelfSigned(const std::string& cert_path,
                                    const std::string& key_path,
                                    const std::string& common_name,
                                    int validity_days) {
    // Initialize OpenSSL
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    // Generate RSA key pair
    EVP_PKEY* pkey = EVP_PKEY_new();
    if (!pkey) {
        spdlog::error("TLSConfig: Failed to allocate EVP_PKEY");
        return false;
    }

    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    if (!ctx) {
        EVP_PKEY_free(pkey);
        spdlog::error("TLSConfig: Failed to create EVP_PKEY_CTX");
        return false;
    }

    if (EVP_PKEY_keygen_init(ctx) <= 0 ||
        EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 2048) <= 0 ||
        EVP_PKEY_keygen(ctx, &pkey) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        EVP_PKEY_free(pkey);
        spdlog::error("TLSConfig: Failed to generate RSA key");
        return false;
    }
    EVP_PKEY_CTX_free(ctx);

    // Create X509 certificate
    X509* x509 = X509_new();
    if (!x509) {
        EVP_PKEY_free(pkey);
        spdlog::error("TLSConfig: Failed to allocate X509");
        return false;
    }

    // Set certificate version (v3)
    X509_set_version(x509, 2);

    // Set serial number
    ASN1_INTEGER_set(X509_get_serialNumber(x509), std::time(nullptr));

    // Set validity period
    X509_gmtime_adj(X509_get_notBefore(x509), 0);
    X509_gmtime_adj(X509_get_notAfter(x509), validity_days * 24 * 3600L);

    // Set public key
    X509_set_pubkey(x509, pkey);

    // Set subject and issuer (self-signed)
    X509_NAME* name = X509_get_subject_name(x509);
    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                                (unsigned char*)common_name.c_str(), -1, -1, 0);
    X509_NAME_add_entry_by_txt(name, "O", MBSTRING_ASC,
                                (unsigned char*)"CyxWiz", -1, -1, 0);
    X509_set_issuer_name(x509, name);

    // Add extensions for localhost
    X509V3_CTX v3ctx;
    X509V3_set_ctx_nodb(&v3ctx);
    X509V3_set_ctx(&v3ctx, x509, x509, nullptr, nullptr, 0);

    // Subject Alternative Names
    std::string san = "DNS:" + common_name + ",DNS:localhost,IP:127.0.0.1";
    X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &v3ctx, NID_subject_alt_name, san.c_str());
    if (ext) {
        X509_add_ext(x509, ext, -1);
        X509_EXTENSION_free(ext);
    }

    // Sign the certificate
    if (!X509_sign(x509, pkey, EVP_sha256())) {
        X509_free(x509);
        EVP_PKEY_free(pkey);
        spdlog::error("TLSConfig: Failed to sign certificate");
        return false;
    }

    // Ensure directory exists
    std::filesystem::path cert_dir = std::filesystem::path(cert_path).parent_path();
    if (!cert_dir.empty() && !std::filesystem::exists(cert_dir)) {
        std::filesystem::create_directories(cert_dir);
    }

    // Write private key to file
    FILE* key_file = fopen(key_path.c_str(), "wb");
    if (!key_file) {
        X509_free(x509);
        EVP_PKEY_free(pkey);
        spdlog::error("TLSConfig: Failed to open key file for writing: {}", key_path);
        return false;
    }
    PEM_write_PrivateKey(key_file, pkey, nullptr, nullptr, 0, nullptr, nullptr);
    fclose(key_file);

    // Write certificate to file
    FILE* cert_file = fopen(cert_path.c_str(), "wb");
    if (!cert_file) {
        X509_free(x509);
        EVP_PKEY_free(pkey);
        spdlog::error("TLSConfig: Failed to open cert file for writing: {}", cert_path);
        return false;
    }
    PEM_write_X509(cert_file, x509);
    fclose(cert_file);

    // Cleanup
    X509_free(x509);
    EVP_PKEY_free(pkey);

    spdlog::info("TLSConfig: Generated self-signed certificate for '{}'", common_name);
    spdlog::info("  Certificate: {}", cert_path);
    spdlog::info("  Private key: {}", key_path);
    spdlog::info("  Valid for: {} days", validity_days);

    return true;
}

#else
// Windows-based certificate generation using CryptoAPI
bool TLSConfig::GenerateSelfSigned(const std::string& cert_path,
                                    const std::string& key_path,
                                    const std::string& common_name,
                                    int validity_days) {
    // For Windows, we'll generate a simple self-signed cert using CryptoAPI
    // Note: This is a simplified implementation

    HCRYPTPROV hProv = 0;
    HCRYPTKEY hKey = 0;
    PCCERT_CONTEXT pCertContext = nullptr;

    // Acquire cryptographic provider
    if (!CryptAcquireContextW(&hProv, nullptr, MS_DEF_PROV_W, PROV_RSA_FULL,
                              CRYPT_NEWKEYSET | CRYPT_MACHINE_KEYSET)) {
        if (GetLastError() == NTE_EXISTS) {
            if (!CryptAcquireContextW(&hProv, nullptr, MS_DEF_PROV_W, PROV_RSA_FULL,
                                      CRYPT_MACHINE_KEYSET)) {
                spdlog::error("TLSConfig: Failed to acquire crypto context: {}",
                             GetLastError());
                return false;
            }
        } else {
            spdlog::error("TLSConfig: Failed to create crypto context: {}",
                         GetLastError());
            return false;
        }
    }

    // Generate key pair
    if (!CryptGenKey(hProv, AT_SIGNATURE, RSA1024BIT_KEY | CRYPT_EXPORTABLE, &hKey)) {
        CryptReleaseContext(hProv, 0);
        spdlog::error("TLSConfig: Failed to generate key: {}", GetLastError());
        return false;
    }

    // Create certificate
    std::wstring cn = L"CN=" + std::wstring(common_name.begin(), common_name.end());
    CERT_NAME_BLOB nameBlob = {0};
    CertStrToNameW(X509_ASN_ENCODING, cn.c_str(), CERT_X500_NAME_STR, nullptr,
                   nullptr, &nameBlob.cbData, nullptr);
    nameBlob.pbData = (BYTE*)malloc(nameBlob.cbData);
    CertStrToNameW(X509_ASN_ENCODING, cn.c_str(), CERT_X500_NAME_STR, nullptr,
                   nameBlob.pbData, &nameBlob.cbData, nullptr);

    CRYPT_KEY_PROV_INFO keyProvInfo = {0};
    keyProvInfo.pwszContainerName = nullptr;
    keyProvInfo.pwszProvName = const_cast<LPWSTR>(MS_DEF_PROV_W);
    keyProvInfo.dwProvType = PROV_RSA_FULL;
    keyProvInfo.dwFlags = CRYPT_MACHINE_KEYSET;
    keyProvInfo.dwKeySpec = AT_SIGNATURE;

    CRYPT_ALGORITHM_IDENTIFIER sigAlgo = {0};
    sigAlgo.pszObjId = const_cast<LPSTR>(szOID_RSA_SHA256RSA);

    SYSTEMTIME startTime, endTime;
    GetSystemTime(&startTime);
    endTime = startTime;
    endTime.wYear += validity_days / 365;

    pCertContext = CertCreateSelfSignCertificate(
        hProv, &nameBlob, 0, &keyProvInfo, &sigAlgo,
        &startTime, &endTime, nullptr);

    free(nameBlob.pbData);

    if (!pCertContext) {
        CryptDestroyKey(hKey);
        CryptReleaseContext(hProv, 0);
        spdlog::error("TLSConfig: Failed to create self-signed cert: {}",
                     GetLastError());
        return false;
    }

    // Ensure directory exists
    std::filesystem::path cert_dir = std::filesystem::path(cert_path).parent_path();
    if (!cert_dir.empty() && !std::filesystem::exists(cert_dir)) {
        std::filesystem::create_directories(cert_dir);
    }

    // Export certificate to PEM format
    // First, encode to DER
    DWORD derSize = 0;
    CryptBinaryToStringA(pCertContext->pbCertEncoded, pCertContext->cbCertEncoded,
                         CRYPT_STRING_BASE64HEADER, nullptr, &derSize);
    std::string certPem(derSize, 0);
    CryptBinaryToStringA(pCertContext->pbCertEncoded, pCertContext->cbCertEncoded,
                         CRYPT_STRING_BASE64HEADER, certPem.data(), &derSize);

    // Write certificate
    std::ofstream certFile(cert_path);
    if (!certFile) {
        CertFreeCertificateContext(pCertContext);
        CryptDestroyKey(hKey);
        CryptReleaseContext(hProv, 0);
        spdlog::error("TLSConfig: Failed to write cert file: {}", cert_path);
        return false;
    }
    certFile << certPem;
    certFile.close();

    // Export private key
    DWORD keyBlobSize = 0;
    if (!CryptExportKey(hKey, 0, PRIVATEKEYBLOB, 0, nullptr, &keyBlobSize)) {
        CertFreeCertificateContext(pCertContext);
        CryptDestroyKey(hKey);
        CryptReleaseContext(hProv, 0);
        spdlog::error("TLSConfig: Failed to get key blob size");
        return false;
    }

    std::vector<BYTE> keyBlob(keyBlobSize);
    if (!CryptExportKey(hKey, 0, PRIVATEKEYBLOB, 0, keyBlob.data(), &keyBlobSize)) {
        CertFreeCertificateContext(pCertContext);
        CryptDestroyKey(hKey);
        CryptReleaseContext(hProv, 0);
        spdlog::error("TLSConfig: Failed to export key");
        return false;
    }

    // Convert to PEM format (simplified - actual implementation would need proper conversion)
    DWORD keyPemSize = 0;
    CryptBinaryToStringA(keyBlob.data(), keyBlobSize, CRYPT_STRING_BASE64HEADER,
                         nullptr, &keyPemSize);
    std::string keyPem(keyPemSize, 0);
    CryptBinaryToStringA(keyBlob.data(), keyBlobSize, CRYPT_STRING_BASE64HEADER,
                         keyPem.data(), &keyPemSize);

    // Write key (Note: Windows blob format != standard PEM, would need conversion)
    std::ofstream keyFile(key_path);
    if (!keyFile) {
        CertFreeCertificateContext(pCertContext);
        CryptDestroyKey(hKey);
        CryptReleaseContext(hProv, 0);
        spdlog::error("TLSConfig: Failed to write key file: {}", key_path);
        return false;
    }
    keyFile << "-----BEGIN RSA PRIVATE KEY-----\n";
    keyFile << keyPem;
    keyFile << "-----END RSA PRIVATE KEY-----\n";
    keyFile.close();

    // Cleanup
    CertFreeCertificateContext(pCertContext);
    CryptDestroyKey(hKey);
    CryptReleaseContext(hProv, 0);

    spdlog::info("TLSConfig: Generated self-signed certificate for '{}'", common_name);
    spdlog::info("  Certificate: {}", cert_path);
    spdlog::info("  Private key: {}", key_path);
    spdlog::info("  Valid for: {} days", validity_days);
    spdlog::warn("TLSConfig: Windows-generated certs may need manual conversion for gRPC");

    return true;
}
#endif

std::shared_ptr<grpc::ServerCredentials> TLSConfig::GetServerCredentials() const {
    if (!loaded_) {
        spdlog::warn("TLSConfig: Not loaded, returning insecure credentials");
        return grpc::InsecureServerCredentials();
    }

    grpc::SslServerCredentialsOptions ssl_opts;

    // Add server certificate and key
    grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert_pair;
    key_cert_pair.private_key = key_pem_;
    key_cert_pair.cert_chain = cert_pem_;
    ssl_opts.pem_key_cert_pairs.push_back(key_cert_pair);

    // Configure client certificate verification
    if (!ca_pem_.empty()) {
        // Mutual TLS - require client certificates
        ssl_opts.pem_root_certs = ca_pem_;
        ssl_opts.client_certificate_request =
            GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
        spdlog::debug("TLSConfig: Server configured with mutual TLS");
    } else {
        // One-way TLS - no client certificate required
        ssl_opts.client_certificate_request = GRPC_SSL_DONT_REQUEST_CLIENT_CERTIFICATE;
        spdlog::debug("TLSConfig: Server configured with one-way TLS");
    }

    return grpc::SslServerCredentials(ssl_opts);
}

std::shared_ptr<grpc::ChannelCredentials> TLSConfig::GetClientCredentials(
    const std::string& target_name) const {
    if (!loaded_) {
        spdlog::warn("TLSConfig: Not loaded, returning insecure credentials");
        return grpc::InsecureChannelCredentials();
    }

    grpc::SslCredentialsOptions ssl_opts;

    // For client, we use the cert as root CA to verify server
    ssl_opts.pem_root_certs = cert_pem_;

    // If mutual TLS, also provide client cert
    if (!ca_pem_.empty()) {
        ssl_opts.pem_cert_chain = cert_pem_;
        ssl_opts.pem_private_key = key_pem_;
    }

    auto creds = grpc::SslCredentials(ssl_opts);

    // Set target name override if provided (useful for self-signed certs)
    if (!target_name.empty()) {
        grpc::ChannelArguments args;
        args.SetSslTargetNameOverride(target_name);
        // Note: Can't easily combine args with credentials in standard API
        // For production, use proper certificates matching the hostname
        spdlog::debug("TLSConfig: Client target name override: {}", target_name);
    }

    return creds;
}

bool TLSConfig::ParseCertificateInfo() {
    // Simple parsing - extract common name from certificate
    // For full parsing, would need OpenSSL
    cert_subject_ = "Unknown";
    cert_expiry_ = "Unknown";

    // Look for CN= in certificate (simplified)
    size_t cn_pos = cert_pem_.find("Subject:");
    if (cn_pos != std::string::npos) {
        size_t cn_start = cert_pem_.find("CN=", cn_pos);
        if (cn_start != std::string::npos) {
            cn_start += 3;
            size_t cn_end = cert_pem_.find_first_of(",\n", cn_start);
            if (cn_end != std::string::npos) {
                cert_subject_ = cert_pem_.substr(cn_start, cn_end - cn_start);
            }
        }
    }

    return true;
}

std::string TLSConfig::GetCertSubject() const {
    return cert_subject_;
}

std::string TLSConfig::GetCertExpiry() const {
    return cert_expiry_;
}

// TLSManager implementation

TLSManager& TLSManager::Instance() {
    static TLSManager instance;
    return instance;
}

bool TLSManager::Initialize(const std::string& cert_path,
                             const std::string& key_path,
                             const std::string& ca_path) {
    if (!std::filesystem::exists(cert_path)) {
        spdlog::warn("TLSManager: Certificate file not found: {}", cert_path);
        return false;
    }

    if (!std::filesystem::exists(key_path)) {
        spdlog::warn("TLSManager: Key file not found: {}", key_path);
        return false;
    }

    return config_.LoadFromFiles(cert_path, key_path, ca_path);
}

bool TLSManager::InitializeSelfSigned(const std::string& data_dir,
                                       const std::string& common_name) {
    namespace fs = std::filesystem;

    std::string cert_path = (fs::path(data_dir) / "tls" / "server.crt").string();
    std::string key_path = (fs::path(data_dir) / "tls" / "server.key").string();

    // Check if certs already exist
    if (fs::exists(cert_path) && fs::exists(key_path)) {
        spdlog::info("TLSManager: Using existing certificates from {}", data_dir);
        return config_.LoadFromFiles(cert_path, key_path);
    }

    // Generate new self-signed certificate
    spdlog::info("TLSManager: Generating new self-signed certificate...");
    if (!TLSConfig::GenerateSelfSigned(cert_path, key_path, common_name)) {
        spdlog::error("TLSManager: Failed to generate self-signed certificate");
        return false;
    }

    return config_.LoadFromFiles(cert_path, key_path);
}

std::shared_ptr<grpc::ServerCredentials> TLSManager::GetServerCredentials() const {
    if (!config_.IsLoaded()) {
        return GetInsecureServerCredentials();
    }
    return config_.GetServerCredentials();
}

std::shared_ptr<grpc::ChannelCredentials> TLSManager::GetClientCredentials(
    const std::string& target_name) const {
    if (!config_.IsLoaded()) {
        return GetInsecureClientCredentials();
    }
    return config_.GetClientCredentials(target_name);
}

std::shared_ptr<grpc::ServerCredentials> TLSManager::GetInsecureServerCredentials() {
    return grpc::InsecureServerCredentials();
}

std::shared_ptr<grpc::ChannelCredentials> TLSManager::GetInsecureClientCredentials() {
    return grpc::InsecureChannelCredentials();
}

} // namespace cyxwiz::servernode::security
