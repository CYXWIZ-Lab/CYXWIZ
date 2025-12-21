use jsonwebtoken::{encode, decode, Header, Validation, Algorithm, EncodingKey, DecodingKey};
use serde::{Serialize, Deserialize};
use chrono::Utc;
use uuid::Uuid;
use crate::error::{Result, ServerError};

/// JWT claims for user authentication
/// Used to authenticate Server Node registration and other user operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAuthClaims {
    /// Subject - User ID
    pub sub: String,
    /// User email (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Wallet address for payments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wallet_address: Option<String>,
    /// Expiration timestamp (Unix time)
    pub exp: i64,
    /// Issued at timestamp (Unix time)
    pub iat: i64,
    /// Issuer
    #[serde(default)]
    pub iss: String,
}

/// JWT claims for admin authentication
/// Used for Admin App → Central Server REST API calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminAuthClaims {
    /// Subject - Admin user ID
    pub sub: String,
    /// Admin email
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Admin role (super_admin, admin, moderator, support)
    #[serde(default)]
    pub role: String,
    /// Admin permissions
    #[serde(default)]
    pub permissions: Vec<String>,
    /// Expiration timestamp (Unix time)
    pub exp: i64,
    /// Issued at timestamp (Unix time)
    pub iat: i64,
}

/// JWT claims for P2P authentication
/// Used to authenticate Engine→Server Node connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PAuthClaims {
    /// Subject - Engine wallet address or ID
    pub sub: String,
    /// Job ID this token is valid for
    pub job_id: String,
    /// Assigned Server Node ID
    pub node_id: String,
    /// Expiration timestamp (Unix time)
    pub exp: i64,
    /// Issued at timestamp (Unix time)
    pub iat: i64,
    /// Issuer - always "CyxWiz-Central-Server"
    pub iss: String,
}

/// JWT token manager for P2P authentication
pub struct JWTManager {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    secret: String,
}

impl JWTManager {
    /// Create new JWT manager with secret key
    pub fn new(secret: &str) -> Self {
        Self {
            encoding_key: EncodingKey::from_secret(secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
            secret: secret.to_string(),
        }
    }

    /// Generate P2P authentication token
    ///
    /// # Arguments
    /// * `engine_id` - Engine wallet address or identifier
    /// * `job_id` - Job UUID
    /// * `node_id` - Assigned Server Node UUID
    /// * `expires_in_seconds` - Token lifetime (typically 3600 = 1 hour)
    ///
    /// # Returns
    /// JWT token string
    pub fn generate_p2p_token(
        &self,
        engine_id: &str,
        job_id: Uuid,
        node_id: Uuid,
        expires_in_seconds: i64,
    ) -> Result<String> {
        let now = Utc::now().timestamp();
        let claims = P2PAuthClaims {
            sub: engine_id.to_string(),
            job_id: job_id.to_string(),
            node_id: node_id.to_string(),
            exp: now + expires_in_seconds,
            iat: now,
            iss: "CyxWiz-Central-Server".to_string(),
        };

        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| ServerError::AuthError(format!("Failed to encode JWT: {}", e)))
    }

    /// Verify and decode P2P authentication token
    ///
    /// # Arguments
    /// * `token` - JWT token string
    ///
    /// # Returns
    /// Decoded claims if valid
    pub fn verify_p2p_token(&self, token: &str) -> Result<P2PAuthClaims> {
        let validation = Validation::new(Algorithm::HS256);
        let token_data = decode::<P2PAuthClaims>(token, &self.decoding_key, &validation)
            .map_err(|e| ServerError::AuthError(format!("Invalid JWT token: {}", e)))?;

        Ok(token_data.claims)
    }

    /// Check if token is expired
    pub fn is_token_expired(&self, claims: &P2PAuthClaims) -> bool {
        let now = Utc::now().timestamp();
        claims.exp < now
    }

    /// Get remaining token lifetime in seconds
    pub fn get_token_lifetime(&self, claims: &P2PAuthClaims) -> i64 {
        let now = Utc::now().timestamp();
        (claims.exp - now).max(0)
    }

    /// Verify and decode user authentication token
    ///
    /// # Arguments
    /// * `token` - JWT token string from user login
    ///
    /// # Returns
    /// Decoded claims if valid, error otherwise
    pub fn verify_user_token(&self, token: &str) -> Result<UserAuthClaims> {
        let mut validation = Validation::new(Algorithm::HS256);
        // Don't require specific issuer, be flexible
        validation.validate_aud = false;

        let token_data = decode::<UserAuthClaims>(token, &self.decoding_key, &validation)
            .map_err(|e| ServerError::AuthError(format!("Invalid user token: {}", e)))?;

        // Check if token is expired
        let now = Utc::now().timestamp();
        if token_data.claims.exp < now {
            return Err(ServerError::AuthError("Token has expired".to_string()));
        }

        Ok(token_data.claims)
    }

    /// Generate a user authentication token
    ///
    /// # Arguments
    /// * `user_id` - User's ID
    /// * `email` - Optional email address
    /// * `wallet_address` - Optional wallet address
    /// * `expires_in_seconds` - Token lifetime
    ///
    /// # Returns
    /// JWT token string
    pub fn generate_user_token(
        &self,
        user_id: &str,
        email: Option<&str>,
        wallet_address: Option<&str>,
        expires_in_seconds: i64,
    ) -> Result<String> {
        let now = Utc::now().timestamp();
        let claims = UserAuthClaims {
            sub: user_id.to_string(),
            email: email.map(|e| e.to_string()),
            wallet_address: wallet_address.map(|w| w.to_string()),
            exp: now + expires_in_seconds,
            iat: now,
            iss: "CyxWiz-Central-Server".to_string(),
        };

        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| ServerError::AuthError(format!("Failed to encode user JWT: {}", e)))
    }

    /// Verify and decode admin authentication token
    /// Admin tokens are signed with a different secret (ADMIN_AUTH_SECRET)
    ///
    /// # Arguments
    /// * `token` - JWT token string from admin login
    /// * `admin_secret` - The admin-specific JWT secret
    ///
    /// # Returns
    /// Decoded admin claims if valid
    pub fn verify_admin_token(token: &str, admin_secret: &str) -> Result<AdminAuthClaims> {
        let decoding_key = DecodingKey::from_secret(admin_secret.as_bytes());
        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_aud = false;

        let token_data = decode::<AdminAuthClaims>(token, &decoding_key, &validation)
            .map_err(|e| ServerError::AuthError(format!("Invalid admin token: {}", e)))?;

        // Check if token is expired
        let now = Utc::now().timestamp();
        if token_data.claims.exp < now {
            return Err(ServerError::AuthError("Admin token has expired".to_string()));
        }

        Ok(token_data.claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_verify_token() {
        let manager = JWTManager::new("test-secret-key");
        let job_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();

        let token = manager
            .generate_p2p_token("engine_wallet_123", job_id, node_id, 3600)
            .unwrap();

        assert!(!token.is_empty());

        let claims = manager.verify_p2p_token(&token).unwrap();
        assert_eq!(claims.sub, "engine_wallet_123");
        assert_eq!(claims.job_id, job_id.to_string());
        assert_eq!(claims.node_id, node_id.to_string());
        assert_eq!(claims.iss, "CyxWiz-Central-Server");
        assert!(!manager.is_token_expired(&claims));
    }

    #[test]
    fn test_invalid_token() {
        let manager = JWTManager::new("test-secret-key");
        let result = manager.verify_p2p_token("invalid.token.here");
        assert!(result.is_err());
    }

    #[test]
    fn test_expired_token() {
        let manager = JWTManager::new("test-secret-key");
        let job_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();

        // Generate token that expires immediately
        let token = manager
            .generate_p2p_token("engine_wallet_123", job_id, node_id, -10)
            .unwrap();

        let claims = manager.verify_p2p_token(&token).unwrap();
        assert!(manager.is_token_expired(&claims));
    }

    #[test]
    fn test_token_lifetime() {
        let manager = JWTManager::new("test-secret-key");
        let job_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();

        let token = manager
            .generate_p2p_token("engine_wallet_123", job_id, node_id, 3600)
            .unwrap();

        let claims = manager.verify_p2p_token(&token).unwrap();
        let lifetime = manager.get_token_lifetime(&claims);

        // Should be approximately 3600 seconds (allow for test execution time)
        assert!(lifetime >= 3595 && lifetime <= 3600);
    }
}
