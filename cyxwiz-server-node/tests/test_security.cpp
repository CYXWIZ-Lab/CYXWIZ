// test_security.cpp - Unit tests for security components
// Tests API key management, rate limiting, audit logging, TLS config, and wallet

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>

#include "../src/api/api_key_manager.h"
#include "../src/api/rate_limiter.h"
#include "../src/security/audit_logger.h"
#include "../src/security/tls_config.h"
#include "../src/security/wallet_manager.h"
#include "../src/security/docker_manager.h"

using namespace cyxwiz::servernode::api;
using namespace cyxwiz::servernode::security;

// ========== API Key Manager Tests ==========

TEST_CASE("API Key Manager - Key Generation", "[security][api-key]") {
    // Use temp file for tests
    std::string test_storage = "test_api_keys.json";
    APIKeyManager manager(test_storage);

    SECTION("Generated key has correct format") {
        auto result = manager.CreateKey("test-key");

        REQUIRE(result.success);
        REQUIRE_FALSE(result.key_id.empty());
        REQUIRE_FALSE(result.full_key.empty());

        // Key format: cyx_sk_live_[32 chars]
        REQUIRE(result.full_key.substr(0, 12) == "cyx_sk_live_");
        REQUIRE(result.full_key.length() == 44);  // 12 prefix + 32 random
    }

    SECTION("Generated keys are unique") {
        auto result1 = manager.CreateKey("key1");
        auto result2 = manager.CreateKey("key2");

        REQUIRE(result1.full_key != result2.full_key);
        REQUIRE(result1.key_id != result2.key_id);
    }

    SECTION("Key validation works") {
        auto result = manager.CreateKey("validate-test");
        REQUIRE(result.success);

        // Valid key should pass
        REQUIRE(manager.ValidateKey(result.full_key));

        // Invalid key should fail
        REQUIRE_FALSE(manager.ValidateKey("cyx_sk_live_invalidkey123"));
        REQUIRE_FALSE(manager.ValidateKey(""));
        REQUIRE_FALSE(manager.ValidateKey("some_random_string"));
    }

    // Cleanup
    std::filesystem::remove(test_storage);
}

TEST_CASE("API Key Manager - Key Revocation", "[security][api-key]") {
    std::string test_storage = "test_api_keys_revoke.json";
    APIKeyManager manager(test_storage);

    SECTION("Revoked keys are rejected") {
        auto result = manager.CreateKey("to-revoke");
        REQUIRE(result.success);

        // Key should work initially
        REQUIRE(manager.ValidateKey(result.full_key));

        // Revoke the key
        REQUIRE(manager.RevokeKey(result.key_id));

        // Key should now fail validation
        REQUIRE_FALSE(manager.ValidateKey(result.full_key));
    }

    SECTION("Revoking non-existent key returns false") {
        REQUIRE_FALSE(manager.RevokeKey("non-existent-id"));
    }

    std::filesystem::remove(test_storage);
}

TEST_CASE("API Key Manager - Model Access Control", "[security][api-key]") {
    std::string test_storage = "test_api_keys_models.json";
    APIKeyManager manager(test_storage);

    SECTION("Key with model restrictions") {
        std::vector<std::string> allowed_models = {"model-a", "model-b"};
        auto result = manager.CreateKey("restricted-key", 60, allowed_models);
        REQUIRE(result.success);

        // Allowed models should pass
        REQUIRE(manager.ValidateKeyForModel(result.full_key, "model-a"));
        REQUIRE(manager.ValidateKeyForModel(result.full_key, "model-b"));

        // Disallowed model should fail
        REQUIRE_FALSE(manager.ValidateKeyForModel(result.full_key, "model-c"));
    }

    SECTION("Key without restrictions accesses any model") {
        auto result = manager.CreateKey("unrestricted-key");
        REQUIRE(result.success);

        REQUIRE(manager.ValidateKeyForModel(result.full_key, "any-model"));
        REQUIRE(manager.ValidateKeyForModel(result.full_key, "another-model"));
    }

    std::filesystem::remove(test_storage);
}

TEST_CASE("API Key Manager - Persistence", "[security][api-key]") {
    std::string test_storage = "test_api_keys_persist.json";

    std::string saved_key;
    std::string saved_key_id;

    // Create key with first manager
    {
        APIKeyManager manager1(test_storage);
        auto result = manager1.CreateKey("persist-test");
        REQUIRE(result.success);
        saved_key = result.full_key;
        saved_key_id = result.key_id;
    }

    // Verify with new manager instance (loads from file)
    {
        APIKeyManager manager2(test_storage);
        manager2.Load();  // Must explicitly load keys from file
        REQUIRE(manager2.ValidateKey(saved_key));

        auto key_info = manager2.GetKey(saved_key_id);
        REQUIRE(key_info != nullptr);
        REQUIRE(key_info->name == "persist-test");
    }

    std::filesystem::remove(test_storage);
}

TEST_CASE("API Key Manager - SHA256 Hashing", "[security][api-key]") {
    SECTION("Hashing produces consistent results") {
        std::string input = "test_input_string";
        std::string hash1 = APIKeyManager::SHA256Hash(input);
        std::string hash2 = APIKeyManager::SHA256Hash(input);

        REQUIRE(hash1 == hash2);
        REQUIRE(hash1.length() == 64);  // 256 bits = 64 hex chars
    }

    SECTION("Different inputs produce different hashes") {
        std::string hash1 = APIKeyManager::SHA256Hash("input1");
        std::string hash2 = APIKeyManager::SHA256Hash("input2");

        REQUIRE(hash1 != hash2);
    }
}

// ========== Rate Limiter Tests ==========

TEST_CASE("Rate Limiter - Basic Limiting", "[security][rate-limit]") {
    RateLimiter limiter;

    SECTION("Requests under limit pass") {
        const int rpm_limit = 10;

        for (int i = 0; i < rpm_limit; i++) {
            REQUIRE(limiter.Allow("test-key", rpm_limit));
        }
    }

    SECTION("Requests over limit are blocked") {
        const int rpm_limit = 5;

        // Use up all requests
        for (int i = 0; i < rpm_limit; i++) {
            REQUIRE(limiter.Allow("over-limit-key", rpm_limit));
        }

        // 6th request should be blocked
        REQUIRE_FALSE(limiter.Allow("over-limit-key", rpm_limit));
    }

    SECTION("Different keys have separate limits") {
        const int rpm_limit = 3;

        // Key A uses all requests
        for (int i = 0; i < rpm_limit; i++) {
            limiter.Allow("key-a", rpm_limit);
        }

        // Key B should still have its full quota
        REQUIRE(limiter.Allow("key-b", rpm_limit));
    }
}

TEST_CASE("Rate Limiter - Rate Limit Result", "[security][rate-limit]") {
    RateLimiter limiter;

    SECTION("Check returns remaining quota") {
        const int rpm_limit = 10;

        auto result1 = limiter.Check("quota-key", rpm_limit);
        REQUIRE(result1.remaining == rpm_limit);
        REQUIRE(result1.limit == rpm_limit);

        // Use some requests
        limiter.Allow("quota-key", rpm_limit);
        limiter.Allow("quota-key", rpm_limit);
        limiter.Allow("quota-key", rpm_limit);

        auto result2 = limiter.Check("quota-key", rpm_limit);
        REQUIRE(result2.remaining == rpm_limit - 3);
    }

    SECTION("Reset time is provided when limited") {
        const int rpm_limit = 2;

        limiter.Allow("reset-key", rpm_limit);
        limiter.Allow("reset-key", rpm_limit);

        auto result = limiter.Check("reset-key", rpm_limit);
        REQUIRE(result.remaining == 0);
        REQUIRE(result.retry_after > 0);
        REQUIRE(result.retry_after <= 60);  // At most 60 seconds
    }
}

TEST_CASE("Rate Limiter - Global Singleton", "[security][rate-limit]") {
    auto& limiter1 = GlobalRateLimiter::Instance();
    auto& limiter2 = GlobalRateLimiter::Instance();

    REQUIRE(&limiter1 == &limiter2);
}

// ========== Audit Logger Tests ==========

TEST_CASE("Audit Logger - Event Logging", "[security][audit]") {
    auto& logger = AuditLogger::Instance();

    SECTION("Log events are recorded") {
        size_t count_before = logger.GetEntryCount();
        logger.Log(AuditEvent::API_KEY_CREATED, "system", "key-123", true, "Test key created");

        auto entries = logger.GetRecentEntries(10);
        REQUIRE(entries.size() >= 1);
        REQUIRE(logger.GetEntryCount() > count_before);

        auto& entry = entries.back();
        REQUIRE(entry.event == AuditEvent::API_KEY_CREATED);
        REQUIRE(entry.actor == "system");
        REQUIRE(entry.resource == "key-123");
        REQUIRE(entry.success == true);
        REQUIRE(entry.timestamp > 0);
    }

    SECTION("Failed events are recorded") {
        size_t count_before = logger.GetEntryCount();
        logger.Log(AuditEvent::API_KEY_REJECTED, "unknown", "invalid-key", false, "Invalid API key");

        auto entries = logger.GetRecentEntries(1);  // Get only the most recent entry
        REQUIRE(entries.size() == 1);
        REQUIRE(logger.GetEntryCount() > count_before);

        auto& entry = entries[0];  // Most recent entry
        REQUIRE(entry.event == AuditEvent::API_KEY_REJECTED);
        REQUIRE(entry.success == false);
    }

    SECTION("Query by time range") {
        int64_t before_log = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        logger.Log(AuditEvent::MODEL_DEPLOYED, "admin", "model-xyz", true);

        auto entries = logger.GetEntriesSince(before_log, 10);
        REQUIRE(entries.size() >= 1);

        // All entries should be after our timestamp
        for (const auto& entry : entries) {
            REQUIRE(entry.timestamp >= before_log);
        }
    }

    SECTION("Entry limit is respected") {
        // Log many events
        for (int i = 0; i < 20; i++) {
            logger.Log(AuditEvent::API_KEY_USED, "user", "key-" + std::to_string(i), true);
        }

        auto entries = logger.GetRecentEntries(5);
        REQUIRE(entries.size() == 5);
    }
}

TEST_CASE("Audit Logger - Event Types", "[security][audit]") {
    auto& logger = AuditLogger::Instance();

    SECTION("All event types can be logged") {
        size_t count_before = logger.GetEntryCount();

        logger.Log(AuditEvent::API_KEY_CREATED, "a", "r", true);
        logger.Log(AuditEvent::API_KEY_REVOKED, "a", "r", true);
        logger.Log(AuditEvent::API_KEY_USED, "a", "r", true);
        logger.Log(AuditEvent::API_KEY_REJECTED, "a", "r", false);
        logger.Log(AuditEvent::MODEL_ACCESS_GRANTED, "a", "r", true);
        logger.Log(AuditEvent::MODEL_ACCESS_DENIED, "a", "r", false);
        logger.Log(AuditEvent::RATE_LIMIT_EXCEEDED, "a", "r", false);
        logger.Log(AuditEvent::MODEL_DEPLOYED, "a", "r", true);
        logger.Log(AuditEvent::MODEL_UNDEPLOYED, "a", "r", true);
        logger.Log(AuditEvent::WALLET_CONNECTED, "a", "r", true);
        logger.Log(AuditEvent::WALLET_DISCONNECTED, "a", "r", true);
        logger.Log(AuditEvent::TRANSACTION_INITIATED, "a", "r", true);
        logger.Log(AuditEvent::TRANSACTION_COMPLETED, "a", "r", true);
        logger.Log(AuditEvent::TLS_ENABLED, "a", "r", true);
        logger.Log(AuditEvent::TLS_DISABLED, "a", "r", false);
        logger.Log(AuditEvent::INVALID_CERTIFICATE, "a", "r", false);

        REQUIRE(logger.GetEntryCount() >= count_before + 16);
    }
}

TEST_CASE("Audit Logger - GetAuditEventName", "[security][audit]") {
    SECTION("Event names are correct") {
        REQUIRE(std::string(GetAuditEventName(AuditEvent::API_KEY_CREATED)) == "API_KEY_CREATED");
        REQUIRE(std::string(GetAuditEventName(AuditEvent::RATE_LIMIT_EXCEEDED)) == "RATE_LIMIT_EXCEEDED");
        REQUIRE(std::string(GetAuditEventName(AuditEvent::WALLET_CONNECTED)) == "WALLET_CONNECTED");
    }
}

// ========== TLS Config Tests ==========

TEST_CASE("TLS Config - Certificate Loading", "[security][tls]") {
    TLSConfig config;

    SECTION("Loading non-existent files fails") {
        REQUIRE_FALSE(config.LoadFromFiles("nonexistent.crt", "nonexistent.key"));
        REQUIRE_FALSE(config.IsLoaded());
    }

    SECTION("IsLoaded returns false initially") {
        TLSConfig fresh_config;
        REQUIRE_FALSE(fresh_config.IsLoaded());
    }
}

TEST_CASE("TLS Config - Self-Signed Generation", "[security][tls]") {
    std::string cert_path = "test_cert.crt";
    std::string key_path = "test_cert.key";

    // Cleanup any existing files
    std::filesystem::remove(cert_path);
    std::filesystem::remove(key_path);

    SECTION("Generate self-signed certificate") {
        bool generated = TLSConfig::GenerateSelfSigned(cert_path, key_path, "test-server", 30);

        // Generation may fail on systems without OpenSSL
        if (generated) {
            REQUIRE(std::filesystem::exists(cert_path));
            REQUIRE(std::filesystem::exists(key_path));

            // Certificate file should contain PEM header
            std::ifstream cert_file(cert_path);
            std::string first_line;
            std::getline(cert_file, first_line);
            REQUIRE(first_line.find("BEGIN CERTIFICATE") != std::string::npos);

            // Key file should contain PEM header
            std::ifstream key_file(key_path);
            std::getline(key_file, first_line);
            REQUIRE(first_line.find("BEGIN") != std::string::npos);
        }
    }

    // Cleanup
    std::filesystem::remove(cert_path);
    std::filesystem::remove(key_path);
}

TEST_CASE("TLS Manager - Singleton", "[security][tls]") {
    auto& manager1 = TLSManager::Instance();
    auto& manager2 = TLSManager::Instance();

    REQUIRE(&manager1 == &manager2);
}

TEST_CASE("TLS Manager - Insecure Fallback", "[security][tls]") {
    auto server_creds = TLSManager::GetInsecureServerCredentials();
    auto client_creds = TLSManager::GetInsecureClientCredentials();

    REQUIRE(server_creds != nullptr);
    REQUIRE(client_creds != nullptr);
}

// ========== Wallet Manager Tests ==========

TEST_CASE("Wallet Manager - Address Validation", "[security][wallet]") {
    SECTION("Valid Solana addresses") {
        // Valid base58 addresses (32-44 chars)
        REQUIRE(WalletManager::IsValidSolanaAddress("11111111111111111111111111111111"));  // 32 chars
        REQUIRE(WalletManager::IsValidSolanaAddress("CYXWiz1111111111111111111111111111111111111"));  // 43 chars
        REQUIRE(WalletManager::IsValidSolanaAddress("7EYnhQoR9YM3N7UoaKRoA44Uy8JeaZV3qyouov87awMs"));  // 44 chars
    }

    SECTION("Invalid addresses") {
        // Too short
        REQUIRE_FALSE(WalletManager::IsValidSolanaAddress("111111111111111"));

        // Invalid characters (0, O, I, l not in base58)
        REQUIRE_FALSE(WalletManager::IsValidSolanaAddress("0OOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"));  // Contains 0 and O

        // Empty
        REQUIRE_FALSE(WalletManager::IsValidSolanaAddress(""));

        // Too long
        REQUIRE_FALSE(WalletManager::IsValidSolanaAddress(
            "1111111111111111111111111111111111111111111111111111111111111"));
    }
}

TEST_CASE("Wallet Manager - Connection State", "[security][wallet]") {
    WalletManager wallet(SolanaNetwork::Devnet);

    SECTION("Initially disconnected") {
        REQUIRE_FALSE(wallet.IsConnected());
        REQUIRE(wallet.GetAddress().empty());
    }

    SECTION("Invalid address connection fails") {
        REQUIRE_FALSE(wallet.ConnectExternalWallet("invalid"));
        REQUIRE_FALSE(wallet.IsConnected());
    }

    SECTION("Disconnect clears state") {
        // Note: This test may fail if no network, but should handle gracefully
        wallet.ConnectExternalWallet("11111111111111111111111111111111");
        wallet.Disconnect();
        REQUIRE_FALSE(wallet.IsConnected());
        REQUIRE(wallet.GetAddress().empty());
    }
}

TEST_CASE("Wallet Manager - Network Configuration", "[security][wallet]") {
    SECTION("Mainnet configuration") {
        WalletManager wallet(SolanaNetwork::Mainnet);
        REQUIRE(wallet.GetNetwork() == SolanaNetwork::Mainnet);
        REQUIRE(wallet.GetRPCUrl().find("mainnet") != std::string::npos);
    }

    SECTION("Devnet configuration") {
        WalletManager wallet(SolanaNetwork::Devnet);
        REQUIRE(wallet.GetNetwork() == SolanaNetwork::Devnet);
        REQUIRE(wallet.GetRPCUrl().find("devnet") != std::string::npos);
    }

    SECTION("Testnet configuration") {
        WalletManager wallet(SolanaNetwork::Testnet);
        REQUIRE(wallet.GetNetwork() == SolanaNetwork::Testnet);
        REQUIRE(wallet.GetRPCUrl().find("testnet") != std::string::npos);
    }

    SECTION("Custom RPC URL") {
        WalletManager wallet("https://custom-rpc.example.com");
        REQUIRE(wallet.GetRPCUrl() == "https://custom-rpc.example.com");
    }
}

TEST_CASE("Wallet Manager - Earnings Tracking", "[security][wallet]") {
    WalletManager wallet(SolanaNetwork::Devnet);

    SECTION("Initial earnings are zero") {
        REQUIRE(wallet.GetTotalEarnings() == 0.0);
        REQUIRE(wallet.GetPendingEarnings() == 0.0);
    }

    SECTION("Recording earnings updates totals") {
        wallet.RecordEarning(10.5, "job-001");
        wallet.RecordEarning(5.25, "job-002");

        REQUIRE(wallet.GetTotalEarnings() == 15.75);
        REQUIRE(wallet.GetPendingEarnings() == 15.75);
    }
}

TEST_CASE("Wallet Manager - State Persistence", "[security][wallet]") {
    std::string state_path = "test_wallet_state.json";

    // Create and save state
    {
        WalletManager wallet1(SolanaNetwork::Devnet);
        wallet1.ConnectExternalWallet("11111111111111111111111111111111");
        wallet1.RecordEarning(100.0, "test-job");
        wallet1.SaveState(state_path);
    }

    // Load state in new instance
    {
        WalletManager wallet2(SolanaNetwork::Devnet);
        REQUIRE(wallet2.LoadState(state_path));
        REQUIRE(wallet2.GetTotalEarnings() == 100.0);
    }

    std::filesystem::remove(state_path);
}

TEST_CASE("Wallet Manager - CYXWIZ Mint Address", "[security][wallet]") {
    std::string mint = WalletManager::GetCYXWIZMintAddress();
    REQUIRE_FALSE(mint.empty());
    REQUIRE(WalletManager::IsValidSolanaAddress(mint));
}

// ========== Docker Manager Tests ==========

TEST_CASE("Docker Manager - Availability Check", "[security][docker]") {
    DockerManager docker;

    // Docker may or may not be available on test machine
    if (docker.IsAvailable()) {
        REQUIRE_FALSE(docker.GetDockerVersion().empty());
    }
}

TEST_CASE("Docker Manager - Container Config", "[security][docker]") {
    ContainerConfig config;

    SECTION("Default configuration") {
        REQUIRE(config.memory_limit_mb == 4096);
        REQUIRE(config.cpu_quota == 1.0f);
        REQUIRE_FALSE(config.network_enabled);
        REQUIRE_FALSE(config.gpu_enabled);
    }

    SECTION("Custom configuration") {
        config.memory_limit_mb = 8192;
        config.cpu_quota = 0.5f;
        config.network_enabled = true;
        config.gpu_enabled = true;
        config.port = 8080;

        REQUIRE(config.memory_limit_mb == 8192);
        REQUIRE(config.cpu_quota == 0.5f);
        REQUIRE(config.network_enabled);
        REQUIRE(config.gpu_enabled);
        REQUIRE(config.port == 8080);
    }
}

TEST_CASE("Docker Manager - Singleton", "[security][docker]") {
    auto& docker1 = DockerManagerSingleton::Instance();
    auto& docker2 = DockerManagerSingleton::Instance();

    REQUIRE(&docker1 == &docker2);
}

// Main function to run tests
int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
