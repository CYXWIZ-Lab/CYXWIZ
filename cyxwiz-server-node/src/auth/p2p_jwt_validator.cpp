#include "p2p_jwt_validator.h"
#include <jwt-cpp/jwt.h>
#include <jwt-cpp/traits/nlohmann-json/traits.h>
#include <spdlog/spdlog.h>
#include <chrono>

namespace cyxwiz {

// Use nlohmann-json traits (already a dependency in vcpkg)
using jwt_traits = jwt::traits::nlohmann_json;

P2PJwtValidator::P2PJwtValidator(const std::string& secret)
    : secret_(secret) {
    if (secret_.empty()) {
        spdlog::warn("P2PJwtValidator initialized with empty secret - all tokens will be rejected");
    }
}

std::optional<P2PAuthClaims> P2PJwtValidator::ValidateToken(const std::string& token) {
    if (token.empty()) {
        spdlog::warn("P2P JWT: Empty token");
        return std::nullopt;
    }

    if (secret_.empty()) {
        spdlog::error("P2P JWT: No secret configured - cannot validate");
        return std::nullopt;
    }

    try {
        // Decode the JWT using nlohmann-json traits
        auto decoded = jwt::decode<jwt_traits>(token);

        // Create verifier with HS256 algorithm and expected issuer
        // jwt::verify takes only json_traits as template parameter
        auto verifier = jwt::verify<jwt_traits>(jwt::default_clock{})
            .allow_algorithm(jwt::algorithm::hs256{secret_})
            .with_issuer("CyxWiz-Central-Server");

        // Verify signature and claims
        verifier.verify(decoded);

        // Check expiration (jwt-cpp does this automatically, but let's be explicit)
        auto exp_claim = decoded.get_expires_at();
        auto now = std::chrono::system_clock::now();
        if (exp_claim < now) {
            auto exp_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                exp_claim.time_since_epoch()).count();
            spdlog::warn("P2P JWT: Token expired at {}", exp_seconds);
            return std::nullopt;
        }

        // Extract claims
        P2PAuthClaims claims;
        claims.sub = decoded.get_subject();
        claims.iss = decoded.get_issuer();

        // Get custom claims
        if (decoded.has_payload_claim("job_id")) {
            claims.job_id = decoded.get_payload_claim("job_id").as_string();
        } else {
            spdlog::warn("P2P JWT: Missing job_id claim");
            return std::nullopt;
        }

        if (decoded.has_payload_claim("node_id")) {
            claims.node_id = decoded.get_payload_claim("node_id").as_string();
        } else {
            spdlog::warn("P2P JWT: Missing node_id claim");
            return std::nullopt;
        }

        // Get timestamps
        claims.exp = std::chrono::duration_cast<std::chrono::seconds>(
            decoded.get_expires_at().time_since_epoch()).count();
        claims.iat = std::chrono::duration_cast<std::chrono::seconds>(
            decoded.get_issued_at().time_since_epoch()).count();

        spdlog::debug("P2P JWT validated successfully: sub={}, job_id={}, node_id={}",
                      claims.sub, claims.job_id, claims.node_id);

        return claims;

    } catch (const jwt::error::token_verification_exception& e) {
        spdlog::warn("P2P JWT verification failed: {}", e.what());
        return std::nullopt;
    } catch (const jwt::error::claim_not_present_exception& e) {
        spdlog::warn("P2P JWT missing required claim: {}", e.what());
        return std::nullopt;
    } catch (const std::exception& e) {
        spdlog::error("P2P JWT decode error: {}", e.what());
        return std::nullopt;
    }
}

bool P2PJwtValidator::ValidateForJob(const std::string& token,
                                      const std::string& expected_job_id,
                                      const std::string& expected_node_id) {
    auto claims = ValidateToken(token);
    if (!claims) {
        return false;
    }

    // Verify job_id matches what we expect
    if (claims->job_id != expected_job_id) {
        spdlog::warn("P2P JWT job_id mismatch: got '{}', expected '{}'",
                     claims->job_id, expected_job_id);
        return false;
    }

    // Verify node_id matches our node's ID
    if (claims->node_id != expected_node_id) {
        spdlog::warn("P2P JWT node_id mismatch: got '{}', expected '{}'",
                     claims->node_id, expected_node_id);
        return false;
    }

    spdlog::info("P2P JWT validated for job {} from user {}",
                 expected_job_id, claims->sub);
    return true;
}

} // namespace cyxwiz
