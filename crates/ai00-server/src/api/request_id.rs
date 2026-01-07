//! Request ID middleware for distributed tracing.
//!
//! Extracts X-Request-ID header from incoming requests or generates a new UUID.
//! The request ID is added to the depot for logging and returned in response headers.

use salvo::prelude::*;

/// Header name for request ID.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Request ID extracted or generated for this request.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

impl RequestId {
    /// Generate a new request ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Get the request ID string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Handler that extracts or generates request ID and adds to response.
#[handler]
pub async fn request_id_handler(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    // Extract from header or generate new
    let request_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| RequestId(s.to_string()))
        .unwrap_or_else(RequestId::new);

    // Add to depot for downstream handlers and logging
    depot.insert("request_id", request_id.clone());

    // Add to response headers
    if let Ok(value) = request_id.as_str().parse() {
        res.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_generation() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();

        // Should be unique
        assert_ne!(id1.as_str(), id2.as_str());

        // Should be valid UUID format
        assert!(uuid::Uuid::parse_str(id1.as_str()).is_ok());
    }

    #[test]
    fn test_request_id_display() {
        let id = RequestId("test-id-123".to_string());
        assert_eq!(format!("{}", id), "test-id-123");
    }
}
