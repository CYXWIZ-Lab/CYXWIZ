//! Authentication middleware for gRPC and REST API requests.
//!
//! This module provides authentication and authorization for:
//! - User requests (Engine, Server Node) via Website-issued JWT
//! - Admin requests (Admin App) via Admin-issued JWT
//! - P2P connections (Engine → Server Node) via Central Server-issued JWT

use crate::auth::{AdminAuthClaims, JWTManager, UserAuthClaims};
use crate::config::JwtConfig;
use crate::database::{MongoClient, MongoUser};
use crate::error::{Result, ServerError};
use std::sync::Arc;
use tonic::metadata::MetadataMap;
use tonic::Status;
use tracing::{debug, warn};

/// Authenticated user/admin context extracted from JWT
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// User ID (MongoDB ObjectId string)
    pub user_id: String,
    /// User's primary wallet address (CyxWallet or external)
    pub wallet_address: Option<String>,
    /// User's email
    pub email: Option<String>,
    /// Whether this is an admin request
    pub is_admin: bool,
    /// Admin role if applicable (super_admin, admin, moderator, support)
    pub admin_role: Option<String>,
    /// Admin permissions if applicable
    pub admin_permissions: Vec<String>,
    /// The original user document from MongoDB (for additional data access)
    pub user: Option<MongoUser>,
}

impl AuthContext {
    /// Create AuthContext from user claims and MongoDB user
    pub fn from_user(claims: &UserAuthClaims, user: Option<MongoUser>) -> Self {
        let wallet_address = user
            .as_ref()
            .and_then(|u| u.primary_wallet())
            .or_else(|| claims.wallet_address.clone());

        Self {
            user_id: claims.sub.clone(),
            wallet_address,
            email: claims.email.clone(),
            is_admin: false,
            admin_role: None,
            admin_permissions: vec![],
            user,
        }
    }

    /// Create AuthContext from admin claims
    pub fn from_admin(claims: &AdminAuthClaims) -> Self {
        Self {
            user_id: claims.sub.clone(),
            wallet_address: None,
            email: claims.email.clone(),
            is_admin: true,
            admin_role: Some(claims.role.clone()),
            admin_permissions: claims.permissions.clone(),
            user: None,
        }
    }

    /// Check if admin has a specific permission
    pub fn has_permission(&self, permission: &str) -> bool {
        if !self.is_admin {
            return false;
        }
        // super_admin has all permissions
        if self.admin_role.as_deref() == Some("super_admin") {
            return true;
        }
        self.admin_permissions.iter().any(|p| p == permission)
    }

    /// Check if user can perform node operations
    pub fn can_manage_nodes(&self) -> bool {
        if self.is_admin {
            return self.has_permission("manage:nodes") || self.has_permission("admin:*");
        }
        // Regular users can only manage their own nodes (checked elsewhere)
        true
    }
}

/// Authentication interceptor for gRPC requests
pub struct AuthInterceptor {
    jwt_manager: Arc<JWTManager>,
    mongo_client: Arc<MongoClient>,
    jwt_config: JwtConfig,
}

impl AuthInterceptor {
    /// Create a new auth interceptor
    pub fn new(
        jwt_manager: Arc<JWTManager>,
        mongo_client: Arc<MongoClient>,
        jwt_config: JwtConfig,
    ) -> Self {
        Self {
            jwt_manager,
            mongo_client,
            jwt_config,
        }
    }

    /// Extract and validate JWT from gRPC metadata
    /// Returns AuthContext with user information
    pub async fn authenticate(&self, metadata: &MetadataMap) -> std::result::Result<AuthContext, Status> {
        let token = self.extract_token(metadata)?;

        // Try to validate as user token first
        match self.jwt_manager.verify_user_token(&token) {
            Ok(claims) => {
                debug!("User token validated for user_id: {}", claims.sub);

                // Look up user in MongoDB for additional data
                let user = match self.mongo_client.get_user_by_id(&claims.sub).await {
                    Ok(u) => u,
                    Err(e) => {
                        warn!("Failed to look up user in MongoDB: {}", e);
                        None
                    }
                };

                Ok(AuthContext::from_user(&claims, user))
            }
            Err(e) => {
                debug!("User token validation failed: {}", e);
                Err(Status::unauthenticated("Invalid or expired token"))
            }
        }
    }

    /// Extract and validate admin JWT from gRPC metadata
    /// Uses the separate admin secret for validation
    pub async fn authenticate_admin(&self, metadata: &MetadataMap) -> std::result::Result<AuthContext, Status> {
        let token = self.extract_token(metadata)?;

        // Get admin secret
        let admin_secret = self.jwt_config.admin_secret.as_ref().ok_or_else(|| {
            warn!("Admin authentication attempted but admin_secret not configured");
            Status::internal("Admin authentication not configured")
        })?;

        match JWTManager::verify_admin_token(&token, admin_secret) {
            Ok(claims) => {
                debug!("Admin token validated for admin_id: {}, role: {}", claims.sub, claims.role);
                Ok(AuthContext::from_admin(&claims))
            }
            Err(e) => {
                debug!("Admin token validation failed: {}", e);
                Err(Status::unauthenticated("Invalid or expired admin token"))
            }
        }
    }

    /// Authenticate either user or admin (tries both)
    /// Useful for endpoints that accept both types
    pub async fn authenticate_any(&self, metadata: &MetadataMap) -> std::result::Result<AuthContext, Status> {
        // Try admin auth first if admin secret is configured
        if self.jwt_config.admin_secret.is_some() {
            if let Ok(ctx) = self.authenticate_admin(metadata).await {
                return Ok(ctx);
            }
        }

        // Fall back to user auth
        self.authenticate(metadata).await
    }

    /// Extract bearer token from gRPC metadata
    fn extract_token(&self, metadata: &MetadataMap) -> std::result::Result<String, Status> {
        let auth_header = metadata
            .get("authorization")
            .ok_or_else(|| Status::unauthenticated("Missing authorization header"))?;

        let auth_str = auth_header
            .to_str()
            .map_err(|_| Status::unauthenticated("Invalid authorization header"))?;

        if !auth_str.starts_with("Bearer ") {
            return Err(Status::unauthenticated("Invalid authorization scheme, expected Bearer"));
        }

        Ok(auth_str[7..].to_string())
    }
}

/// REST API authentication helpers for Axum middleware
pub mod rest {
    use super::*;
    use axum::{
        body::Body,
        extract::State,
        http::{Request, StatusCode},
        middleware::Next,
        response::Response,
    };

    /// Shared state for auth middleware
    #[derive(Clone)]
    pub struct AuthState {
        pub jwt_manager: Arc<JWTManager>,
        pub mongo_client: Arc<MongoClient>,
        pub jwt_config: JwtConfig,
    }

    /// Axum middleware that requires authentication for requests
    pub async fn require_auth(
        State(state): State<AuthState>,
        mut request: Request<Body>,
        next: Next,
    ) -> std::result::Result<Response, StatusCode> {
        let token = extract_bearer_token(&request).map_err(|_| StatusCode::UNAUTHORIZED)?;

        // Validate user token
        let claims = state
            .jwt_manager
            .verify_user_token(&token)
            .map_err(|_| StatusCode::UNAUTHORIZED)?;

        // Look up user in MongoDB
        let user = state
            .mongo_client
            .get_user_by_id(&claims.sub)
            .await
            .ok()
            .flatten();

        let auth_context = AuthContext::from_user(&claims, user);

        // Insert auth context into request extensions
        request.extensions_mut().insert(auth_context);

        Ok(next.run(request).await)
    }

    /// Axum middleware that requires admin authentication
    pub async fn require_admin(
        State(state): State<AuthState>,
        mut request: Request<Body>,
        next: Next,
    ) -> std::result::Result<Response, StatusCode> {
        let token = extract_bearer_token(&request).map_err(|_| StatusCode::UNAUTHORIZED)?;

        // Get admin secret
        let admin_secret = state
            .jwt_config
            .admin_secret
            .as_ref()
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

        // Validate admin token
        let claims = JWTManager::verify_admin_token(&token, admin_secret)
            .map_err(|_| StatusCode::UNAUTHORIZED)?;

        let auth_context = AuthContext::from_admin(&claims);

        // Insert auth context into request extensions
        request.extensions_mut().insert(auth_context);

        Ok(next.run(request).await)
    }

    /// Extract bearer token from HTTP request
    fn extract_bearer_token(request: &Request<Body>) -> std::result::Result<String, ServerError> {
        let auth_header = request
            .headers()
            .get("authorization")
            .ok_or_else(|| ServerError::AuthError("Missing authorization header".to_string()))?;

        let auth_str = auth_header
            .to_str()
            .map_err(|_| ServerError::AuthError("Invalid authorization header".to_string()))?;

        if !auth_str.starts_with("Bearer ") {
            return Err(ServerError::AuthError("Invalid authorization scheme".to_string()));
        }

        Ok(auth_str[7..].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_context_from_user() {
        let claims = UserAuthClaims {
            sub: "user123".to_string(),
            email: Some("test@example.com".to_string()),
            wallet_address: Some("wallet123".to_string()),
            exp: 9999999999,
            iat: 1000000000,
            iss: "test".to_string(),
        };

        let ctx = AuthContext::from_user(&claims, None);
        assert_eq!(ctx.user_id, "user123");
        assert_eq!(ctx.email, Some("test@example.com".to_string()));
        assert_eq!(ctx.wallet_address, Some("wallet123".to_string()));
        assert!(!ctx.is_admin);
    }

    #[test]
    fn test_auth_context_from_admin() {
        let claims = AdminAuthClaims {
            sub: "admin123".to_string(),
            email: Some("admin@example.com".to_string()),
            role: "admin".to_string(),
            permissions: vec!["manage:nodes".to_string(), "view:jobs".to_string()],
            exp: 9999999999,
            iat: 1000000000,
        };

        let ctx = AuthContext::from_admin(&claims);
        assert_eq!(ctx.user_id, "admin123");
        assert!(ctx.is_admin);
        assert_eq!(ctx.admin_role, Some("admin".to_string()));
        assert!(ctx.has_permission("manage:nodes"));
        assert!(!ctx.has_permission("delete:users"));
    }

    #[test]
    fn test_super_admin_has_all_permissions() {
        let claims = AdminAuthClaims {
            sub: "superadmin".to_string(),
            email: None,
            role: "super_admin".to_string(),
            permissions: vec![],
            exp: 9999999999,
            iat: 1000000000,
        };

        let ctx = AuthContext::from_admin(&claims);
        assert!(ctx.has_permission("anything"));
        assert!(ctx.has_permission("delete:users"));
        assert!(ctx.has_permission("manage:nodes"));
    }
}

/// gRPC Interceptor for authentication
pub mod grpc {
    use super::*;
    use tonic::{Request, Status};
    use tracing::info;

    /// Create an auth interceptor closure that validates JWT tokens
    pub fn create_auth_interceptor(
        jwt_manager: Arc<JWTManager>,
    ) -> impl Fn(Request<()>) -> std::result::Result<Request<()>, Status> + Clone + Send + Sync + 'static
    {
        move |request: Request<()>| {
            let metadata = request.metadata();

            let auth_header = match metadata.get("authorization") {
                Some(header) => header,
                None => {
                    warn!("gRPC request missing authorization header");
                    return Err(Status::unauthenticated("Missing authorization header"));
                }
            };

            let auth_str = match auth_header.to_str() {
                Ok(s) => s,
                Err(_) => {
                    return Err(Status::unauthenticated("Invalid authorization header encoding"));
                }
            };

            if !auth_str.starts_with("Bearer ") {
                return Err(Status::unauthenticated("Invalid authorization scheme, expected Bearer"));
            }

            let token = &auth_str[7..];

            match jwt_manager.verify_user_token(token) {
                Ok(claims) => {
                    debug!("gRPC request authenticated for user: {}", claims.sub);
                    Ok(request)
                }
                Err(e) => {
                    warn!("JWT validation failed: {}", e);
                    Err(Status::unauthenticated("Invalid or expired token"))
                }
            }
        }
    }
}
