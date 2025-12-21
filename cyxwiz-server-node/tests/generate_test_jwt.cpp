/**
 * Test JWT Generator - Creates valid P2P JWT tokens for testing
 *
 * Usage:
 *   generate_test_jwt <secret> <job_id> <node_id> [user_id] [expiry_seconds]
 *
 * Example:
 *   generate_test_jwt "your-super-secret-jwt-key" "job_001" "node_123"
 *   generate_test_jwt "your-super-secret-jwt-key" "job_001" "node_123" "user_456" 3600
 */

#include <iostream>
#include <string>
#include <chrono>
#include <jwt-cpp/jwt.h>
#include <jwt-cpp/traits/nlohmann-json/traits.h>

using jwt_traits = jwt::traits::nlohmann_json;

std::string GenerateP2PToken(
    const std::string& secret,
    const std::string& job_id,
    const std::string& node_id,
    const std::string& user_id = "test_user",
    int expiry_seconds = 3600) {

    auto now = std::chrono::system_clock::now();
    auto exp = now + std::chrono::seconds(expiry_seconds);

    auto token = jwt::create<jwt_traits>()
        .set_type("JWT")
        .set_issuer("CyxWiz-Central-Server")
        .set_subject(user_id)
        .set_issued_at(now)
        .set_expires_at(exp)
        .set_payload_claim("job_id", jwt::basic_claim<jwt_traits>(job_id))
        .set_payload_claim("node_id", jwt::basic_claim<jwt_traits>(node_id))
        .sign(jwt::algorithm::hs256{secret});

    return token;
}

int main(int argc, char* argv[]) {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       P2P JWT Test Token Generator                        ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;

    if (argc < 4) {
        std::cerr << "\nUsage: " << argv[0] << " <secret> <job_id> <node_id> [user_id] [expiry_seconds]\n" << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " \"your-super-secret-jwt-key\" \"job_001\" \"node_123\"" << std::endl;
        std::cerr << "  " << argv[0] << " \"your-super-secret-jwt-key\" \"job_001\" \"node_123\" \"user@example.com\" 7200\n" << std::endl;
        return 1;
    }

    std::string secret = argv[1];
    std::string job_id = argv[2];
    std::string node_id = argv[3];
    std::string user_id = argc > 4 ? argv[4] : "test_user";
    int expiry_seconds = argc > 5 ? std::stoi(argv[5]) : 3600;

    std::cout << "\nParameters:" << std::endl;
    std::cout << "  Secret: " << secret.substr(0, 10) << "..." << std::endl;
    std::cout << "  Job ID: " << job_id << std::endl;
    std::cout << "  Node ID: " << node_id << std::endl;
    std::cout << "  User ID: " << user_id << std::endl;
    std::cout << "  Expiry: " << expiry_seconds << " seconds" << std::endl;

    try {
        std::string token = GenerateP2PToken(secret, job_id, node_id, user_id, expiry_seconds);

        std::cout << "\n═══════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "Generated JWT Token:" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════\n" << std::endl;
        std::cout << token << std::endl;
        std::cout << "\n═══════════════════════════════════════════════════════════════" << std::endl;

        // Verify it works
        auto decoded = jwt::decode<jwt_traits>(token);
        auto verifier = jwt::verify<jwt_traits>(jwt::default_clock{})
            .allow_algorithm(jwt::algorithm::hs256{secret})
            .with_issuer("CyxWiz-Central-Server");

        verifier.verify(decoded);

        std::cout << "\n[OK] Token verified successfully!" << std::endl;
        std::cout << "  Subject: " << decoded.get_subject() << std::endl;
        std::cout << "  Issuer: " << decoded.get_issuer() << std::endl;
        std::cout << "  Job ID: " << decoded.get_payload_claim("job_id").as_string() << std::endl;
        std::cout << "  Node ID: " << decoded.get_payload_claim("node_id").as_string() << std::endl;

        auto exp = decoded.get_expires_at();
        auto now = std::chrono::system_clock::now();
        auto remaining = std::chrono::duration_cast<std::chrono::seconds>(exp - now).count();
        std::cout << "  Expires in: " << remaining << " seconds" << std::endl;

        std::cout << "\nTo use with mock_engine_client:" << std::endl;
        std::cout << "  mock_engine_client localhost:50052 " << job_id << " \"" << token << "\"" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to generate token: " << e.what() << std::endl;
        return 1;
    }
}
