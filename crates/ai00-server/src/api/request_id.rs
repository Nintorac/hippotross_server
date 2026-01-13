//! Request ID middleware for distributed tracing.
//!
//! Implements two-field ID strategy:
//! - `trace_id`: Extracted from x-request-id header for cross-service correlation
//! - `request_id`: Generated fresh (UUID7) for this service's request handling
//!
//! The request context accumulates metrics throughout the request lifecycle
//! and emits a canonical log line on completion.

use salvo::prelude::*;

use crate::logging::RequestContext;

/// Header name for incoming trace ID (cross-service correlation).
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Header name for this service's span ID.
pub const SPAN_ID_HEADER: &str = "x-span-id";

/// Handler that creates request context with trace_id and request_id.
///
/// - Extracts `x-request-id` header as `trace_id` (for cross-service correlation)
/// - Generates fresh UUID7 as `request_id` (this service's span ID)
/// - Stores `RequestContext` in depot for downstream handlers
/// - Adds both IDs to response headers
#[handler]
pub async fn request_id_handler(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    // Extract trace_id from incoming header (cross-service correlation)
    let trace_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Create request context with trace_id (request_id is generated as UUID7)
    let context = RequestContext::new(trace_id.clone());

    // Add request_id to response headers
    if let Ok(value) = context.request_id.parse() {
        res.headers_mut().insert(SPAN_ID_HEADER, value);
    }

    // Return trace_id in x-request-id if present, otherwise use our request_id
    let response_trace = trace_id.as_ref().unwrap_or(&context.request_id);
    if let Ok(value) = response_trace.parse() {
        res.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    // Store context in depot for downstream handlers
    depot.insert("request_context", context);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_context_generation() {
        let ctx1 = RequestContext::new(None);
        let ctx2 = RequestContext::new(None);

        // Request IDs should be unique
        assert_ne!(ctx1.request_id, ctx2.request_id);

        // Should be valid UUID7 format
        assert!(uuid::Uuid::parse_str(&ctx1.request_id).is_ok());
    }

    #[test]
    fn test_request_context_with_trace_id() {
        let trace_id = "external-trace-123".to_string();
        let ctx = RequestContext::new(Some(trace_id.clone()));

        // trace_id should be preserved
        assert_eq!(ctx.trace_id, Some(trace_id));

        // request_id should still be generated
        assert!(uuid::Uuid::parse_str(&ctx.request_id).is_ok());
    }
}
