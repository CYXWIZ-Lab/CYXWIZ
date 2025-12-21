#pragma once
/**
 * @file p2p_jwt_validator.h
 * @brief P2P JWT validator for Engine → Server Node connections
 *
 * This validator verifies JWT tokens issued by the Central Server for P2P
 * direct connections between Engine and Server Node. Each token authorizes
 * a specific Engine to execute a specific job on a specific node.
 *
 * Token Structure (from Central Server auth/jwt.rs):
 * - sub: Engine wallet address or user ID
 * - job_id: Job UUID this token authorizes
 * - node_id: Assigned Server Node UUID
 * - exp: Expiration timestamp (Unix)
 * - iat: Issued at timestamp (Unix)
 * - iss: "CyxWiz-Central-Server"
 */

#include <string>
#include <optional>
#include <cstdint>

namespace cyxwiz {

/**
 * @brief P2P authentication claims extracted from JWT
 */
struct P2PAuthClaims {
    std::string sub;       // Engine wallet/user ID
    std::string job_id;    // Job UUID
    std::string node_id;   // Server Node UUID
    int64_t exp;           // Expiration (Unix timestamp)
    int64_t iat;           // Issued at (Unix timestamp)
    std::string iss;       // Issuer (should be "CyxWiz-Central-Server")
};

/**
 * @brief Validates P2P JWT tokens from Central Server
 *
 * Uses HS256 (HMAC-SHA256) algorithm for verification, matching the
 * Central Server's JWT signing configuration.
 */
class P2PJwtValidator {
public:
    /**
     * @brief Construct a validator with the shared secret
     * @param secret The secret key matching Central Server's jwt.secret or jwt.p2p_secret
     */
    explicit P2PJwtValidator(const std::string& secret);

    /**
     * @brief Validate token and extract claims
     * @param token The JWT token string
     * @return Claims if valid, nullopt if invalid/expired
     */
    std::optional<P2PAuthClaims> ValidateToken(const std::string& token);

    /**
     * @brief Validate token for a specific job on this node
     * @param token The JWT token string
     * @param expected_job_id The job ID this connection should be for
     * @param expected_node_id This node's UUID
     * @return true if token is valid AND matches job/node, false otherwise
     */
    bool ValidateForJob(const std::string& token,
                        const std::string& expected_job_id,
                        const std::string& expected_node_id);

private:
    std::string secret_;
};

} // namespace cyxwiz
