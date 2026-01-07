//! Integration tests for Claude-compatible Messages API.

mod common;

use ai00_server::api::messages::{
    generate_tool_system_prompt, validate_tool_name, ContentBlock, MessageContent, MessageParam,
    MessageRole, MessagesRequest, MessagesResponse, StopReason, Tool, ToolChoice, ToolChoiceSimple,
    ToolChoiceSpecific,
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

// =============================================================================
// Tool Definition Tests
// =============================================================================

/// Test tool name validation regex.
#[rstest]
#[case("get_weather", true)]
#[case("search_web", true)]
#[case("my-tool-name", true)]
#[case("Tool_123", true)]
#[case("a", true)]  // Minimum length 1
#[case("", false)]  // Empty not allowed
#[case("tool name", false)]  // Spaces not allowed
#[case("tool.name", false)]  // Dots not allowed
#[case("tool@name", false)]  // Special chars not allowed
fn test_tool_name_validation(#[case] name: &str, #[case] expected: bool) {
    assert_eq!(validate_tool_name(name), expected, "Failed for: {}", name);
}

/// Test tool name length limits separately (can't use .repeat() in rstest attributes).
#[test]
fn test_tool_name_length_limits() {
    // 64 chars - should be valid (max length)
    let max_valid = "a".repeat(64);
    assert!(validate_tool_name(&max_valid), "64-char name should be valid");

    // 65 chars - should be invalid (too long)
    let too_long = "a".repeat(65);
    assert!(!validate_tool_name(&too_long), "65-char name should be invalid");
}

/// Test Tool struct deserialization.
#[test]
fn test_tool_deserialization() {
    let json = json!({
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state"
                }
            },
            "required": ["location"]
        }
    });

    let tool: Tool = serde_json::from_value(json).unwrap();
    assert_eq!(tool.name, "get_weather");
    assert_eq!(tool.description, Some("Get the current weather for a location".to_string()));
    assert!(tool.input_schema.is_object());
    assert!(tool.validate().is_ok());
}

/// Test Tool validation.
#[test]
fn test_tool_validation() {
    // Invalid name
    let tool = Tool {
        name: "invalid name".to_string(),
        description: None,
        input_schema: json!({"type": "object"}),
        cache_control: None,
    };
    assert!(tool.validate().is_err());

    // Null schema
    let tool = Tool {
        name: "valid_name".to_string(),
        description: None,
        input_schema: serde_json::Value::Null,
        cache_control: None,
    };
    assert!(tool.validate().is_err());

    // Valid tool
    let tool = Tool {
        name: "valid_name".to_string(),
        description: Some("A tool".to_string()),
        input_schema: json!({"type": "object"}),
        cache_control: None,
    };
    assert!(tool.validate().is_ok());
}

/// Test ToolChoice deserialization - simple string variants.
#[rstest]
#[case("auto", ToolChoiceSimple::Auto)]
#[case("none", ToolChoiceSimple::None)]
#[case("any", ToolChoiceSimple::Any)]
fn test_tool_choice_simple_deserialization(#[case] input: &str, #[case] expected: ToolChoiceSimple) {
    let json = json!(input);
    let choice: ToolChoice = serde_json::from_value(json).unwrap();
    match choice {
        ToolChoice::Simple(simple) => assert_eq!(simple, expected),
        _ => panic!("Expected simple choice"),
    }
}

/// Test ToolChoice deserialization - specific tool.
#[test]
fn test_tool_choice_specific_deserialization() {
    let json = json!({
        "type": "tool",
        "name": "get_weather"
    });

    let choice: ToolChoice = serde_json::from_value(json).unwrap();
    match choice {
        ToolChoice::Specific(specific) => {
            assert_eq!(specific.choice_type, "tool");
            assert_eq!(specific.name, "get_weather");
        }
        _ => panic!("Expected specific choice"),
    }
}

/// Test request with tools.
#[test]
fn test_request_with_tools() {
    let json = json!({
        "model": "rwkv",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "max_tokens": 100,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        ],
        "tool_choice": "auto"
    });

    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(request.tools.is_some());
    let tools = request.tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "get_weather");

    assert!(request.tool_choice.is_some());
}

/// Test ToolChoiceSpecific constructor.
#[test]
fn test_tool_choice_specific_new() {
    let choice = ToolChoiceSpecific::new("my_tool");
    assert_eq!(choice.choice_type, "tool");
    assert_eq!(choice.name, "my_tool");
    assert!(choice.disable_parallel_tool_use.is_none());
}

// =============================================================================
// Tool Prompt Injection Tests (Hermes/Qwen Format)
// =============================================================================

/// Test generate_tool_system_prompt with empty tools.
#[test]
fn test_generate_tool_system_prompt_empty() {
    let result = generate_tool_system_prompt(&[]);
    assert!(result.is_empty());
}

/// Test generate_tool_system_prompt with a single tool.
#[test]
fn test_generate_tool_system_prompt_single_tool() {
    let tools = vec![Tool {
        name: "get_weather".to_string(),
        description: Some("Get the current weather for a location".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }),
        cache_control: None,
    }];

    let result = generate_tool_system_prompt(&tools);

    // Should use Hermes/Qwen format with <tools> tag
    assert!(result.contains("<tools>"));
    assert!(result.contains("</tools>"));
    assert!(result.contains("<tool_call>"));
    assert!(result.contains("</tool_call>"));

    // Should contain the tool definition as JSON
    assert!(result.contains("get_weather"));
    assert!(result.contains("Get the current weather for a location"));
    // JSON is compact (no spaces after colons)
    assert!(result.contains("\"type\":\"function\""));
}

/// Test generate_tool_system_prompt with multiple tools.
#[test]
fn test_generate_tool_system_prompt_multiple_tools() {
    let tools = vec![
        Tool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            input_schema: json!({"type": "object"}),
            cache_control: None,
        },
        Tool {
            name: "search_web".to_string(),
            description: Some("Search the web".to_string()),
            input_schema: json!({"type": "object"}),
            cache_control: None,
        },
    ];

    let result = generate_tool_system_prompt(&tools);

    // Should contain both tools
    assert!(result.contains("get_weather"));
    assert!(result.contains("search_web"));
    assert!(result.contains("Get weather"));
    assert!(result.contains("Search the web"));
}

/// Test Tool::to_hermes_json formatting.
#[test]
fn test_tool_to_hermes_json() {
    let tool = Tool {
        name: "calculate".to_string(),
        description: Some("Perform arithmetic calculations".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }),
        cache_control: None,
    };

    let json = tool.to_hermes_json();

    // Should have Hermes function format
    assert_eq!(json["type"], "function");
    assert_eq!(json["function"]["name"], "calculate");
    assert_eq!(json["function"]["description"], "Perform arithmetic calculations");
    assert!(json["function"]["parameters"]["properties"]["expression"].is_object());
}
