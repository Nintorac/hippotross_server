//! Shared error types for Claude-compatible API responses.
//!
//! This module provides error types that match Anthropic's API error format
//! for compatibility with Claude API clients.

use salvo::prelude::*;
use serde::{Deserialize, Serialize};

/// Top-level error response wrapper.
///
/// Matches Claude API error format:
/// ```json
/// {
///   "type": "error",
///   "error": {
///     "type": "invalid_request_error",
///     "message": "..."
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiErrorResponse {
    /// Always "error" for error responses
    #[serde(rename = "type")]
    pub error_type: &'static str,
    /// The error details
    pub error: ApiErrorDetail,
}

/// Detailed error information.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiErrorDetail {
    /// The category of error
    #[serde(rename = "type")]
    pub kind: ApiErrorKind,
    /// Human-readable error message
    pub message: String,
    /// Optional parameter that caused the error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

/// Error categories matching Claude API error types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApiErrorKind {
    /// Invalid request parameters or format
    InvalidRequestError,
    /// Authentication failed
    AuthenticationError,
    /// Insufficient permissions
    PermissionError,
    /// Resource not found
    NotFoundError,
    /// Rate limit exceeded
    RateLimitError,
    /// Internal server error
    ApiError,
    /// Server overloaded
    OverloadedError,
}

impl ApiErrorResponse {
    /// Create a new error response.
    pub fn new(kind: ApiErrorKind, message: impl Into<String>) -> Self {
        Self {
            error_type: "error",
            error: ApiErrorDetail {
                kind,
                message: message.into(),
                param: None,
            },
        }
    }

    /// Create an invalid request error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::InvalidRequestError, message)
    }

    /// Create an authentication error.
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::AuthenticationError, message)
    }

    /// Create a permission error.
    pub fn permission(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::PermissionError, message)
    }

    /// Create a not found error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::NotFoundError, message)
    }

    /// Create a rate limit error.
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::RateLimitError, message)
    }

    /// Create an internal API error.
    pub fn api_error(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::ApiError, message)
    }

    /// Create an overloaded error.
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::OverloadedError, message)
    }

    /// Add parameter information to the error.
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.error.param = Some(param.into());
        self
    }

    /// Get the appropriate HTTP status code for this error.
    pub fn status_code(&self) -> StatusCode {
        match self.error.kind {
            ApiErrorKind::InvalidRequestError => StatusCode::BAD_REQUEST,
            ApiErrorKind::AuthenticationError => StatusCode::UNAUTHORIZED,
            ApiErrorKind::PermissionError => StatusCode::FORBIDDEN,
            ApiErrorKind::NotFoundError => StatusCode::NOT_FOUND,
            ApiErrorKind::RateLimitError => StatusCode::TOO_MANY_REQUESTS,
            ApiErrorKind::OverloadedError => StatusCode::SERVICE_UNAVAILABLE,
            ApiErrorKind::ApiError => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Implement Salvo's Writer trait for automatic response rendering.
#[async_trait]
impl Writer for ApiErrorResponse {
    async fn write(self, _req: &mut Request, _depot: &mut Depot, res: &mut Response) {
        res.status_code(self.status_code());
        res.render(Json(self));
    }
}

/// Implement Scribe for OpenAPI documentation.
impl EndpointOutRegister for ApiErrorResponse {
    fn register(
        _components: &mut salvo::oapi::Components,
        _operation: &mut salvo::oapi::Operation,
    ) {
        // Error responses are documented via ToSchema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_request_error() {
        let err = ApiErrorResponse::invalid_request("messages cannot be empty");
        assert_eq!(err.error.kind, ApiErrorKind::InvalidRequestError);
        assert_eq!(err.error.message, "messages cannot be empty");
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_error_with_param() {
        let err = ApiErrorResponse::invalid_request("invalid value").with_param("temperature");
        assert_eq!(err.error.param, Some("temperature".to_string()));
    }

    #[test]
    fn test_error_serialization() {
        let err = ApiErrorResponse::invalid_request("test error");
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"type\":\"invalid_request_error\""));
        assert!(json.contains("\"message\":\"test error\""));
    }

    #[test]
    fn test_status_codes() {
        assert_eq!(
            ApiErrorResponse::authentication("").status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ApiErrorResponse::permission("").status_code(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            ApiErrorResponse::not_found("").status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ApiErrorResponse::rate_limit("").status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ApiErrorResponse::api_error("").status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            ApiErrorResponse::overloaded("").status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
