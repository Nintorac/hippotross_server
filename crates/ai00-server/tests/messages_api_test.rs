//! Integration tests for Claude-compatible Messages API.

mod common;

use ai00_server::api::messages::{
    ContentBlock, MessageContent, MessageParam, MessageRole, MessagesRequest, MessagesResponse,
    StopReason,
};
use ai00_server::api::error::{ApiErrorKind, ApiErrorResponse};
use rstest::rstest;
use serde_json::json;

/// Test that the request types deserialize correctly.
#[test]
fn test_messages_request_deserialization() {
    let json = json!({
        "model": "rwkv",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100
    });

    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(request.model, "rwkv");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.max_tokens, 100);
}

/// Test message content variants.
#[test]
fn test_message_content_variants() {
    // Simple text content
    let json = json!({"role": "user", "content": "Hello"});
    let msg: MessageParam = serde_json::from_value(json).unwrap();
    assert_eq!(msg.content.to_text(), "Hello");

    // Array content with text block
    let json = json!({
        "role": "user",
        "content": [{"type": "text", "text": "Hello from array"}]
    });
    let msg: MessageParam = serde_json::from_value(json).unwrap();
    assert_eq!(msg.content.to_text(), "Hello from array");
}

/// Test response serialization.
#[test]
fn test_messages_response_serialization() {
    let response = MessagesResponse::new(
        "rwkv-model".to_string(),
        vec![ContentBlock::Text {
            text: "Hello!".to_string(),
        }],
        Default::default(),
    );

    let json = serde_json::to_value(&response).unwrap();
    assert!(json["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "assistant");
    assert_eq!(json["content"][0]["type"], "text");
    assert_eq!(json["content"][0]["text"], "Hello!");
}

/// Test stop reason serialization.
#[rstest]
#[case(StopReason::EndTurn, "end_turn")]
#[case(StopReason::MaxTokens, "max_tokens")]
#[case(StopReason::StopSequence, "stop_sequence")]
#[case(StopReason::ToolUse, "tool_use")]
fn test_stop_reason_serialization(#[case] reason: StopReason, #[case] expected: &str) {
    let json = serde_json::to_value(reason).unwrap();
    assert_eq!(json, expected);
}

/// Test error response format.
#[test]
fn test_error_response_format() {
    let error = ApiErrorResponse::invalid_request("test error").with_param("test_param");
    let json = serde_json::to_value(&error).unwrap();

    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["message"], "test error");
    assert_eq!(json["error"]["param"], "test_param");
}

/// Test various error kinds map to correct types.
#[rstest]
#[case(ApiErrorKind::InvalidRequestError, "invalid_request_error")]
#[case(ApiErrorKind::AuthenticationError, "authentication_error")]
#[case(ApiErrorKind::PermissionError, "permission_error")]
#[case(ApiErrorKind::NotFoundError, "not_found_error")]
#[case(ApiErrorKind::RateLimitError, "rate_limit_error")]
#[case(ApiErrorKind::ApiError, "api_error")]
#[case(ApiErrorKind::OverloadedError, "overloaded_error")]
fn test_error_kind_serialization(#[case] kind: ApiErrorKind, #[case] expected: &str) {
    let json = serde_json::to_value(kind).unwrap();
    assert_eq!(json, expected);
}

/// Test request with all optional fields.
#[test]
fn test_full_request_deserialization() {
    let json = json!({
        "model": "rwkv-7-g1",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "system": "You are a helpful assistant.",
        "max_tokens": 1000,
        "stream": true,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "stop_sequences": ["\n\n", "END"]
    });

    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(request.model, "rwkv-7-g1");
    assert_eq!(request.system, Some("You are a helpful assistant.".to_string()));
    assert!(request.stream);
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.top_k, Some(40));
    assert_eq!(request.stop_sequences, Some(vec!["\n\n".to_string(), "END".to_string()]));
}
