//! Integration tests for Claude-compatible Messages API.

mod common;

use ai00_server::api::error::{ApiErrorKind, ApiErrorResponse};
use ai00_server::api::messages::{
    emit_error, generate_thinking_signature, generate_tool_system_prompt, validate_tool_name,
    ContentBlock, MessageContent, MessageParam, MessageRole, MessagesRequest, MessagesResponse,
    StopReason, StreamErrorEvent, ThinkingConfig, ThinkingExtractor, ThinkingStreamParser,
    ThinkingStreamState, Tool, ToolChoice, ToolChoiceSimple, ToolChoiceSpecific,
};
use ai00_server::config::PromptsConfig;
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
    assert_eq!(
        request.system,
        Some("You are a helpful assistant.".to_string())
    );
    assert!(request.stream);
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.top_k, Some(40));
    assert_eq!(
        request.stop_sequences,
        Some(vec!["\n\n".to_string(), "END".to_string()])
    );
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
#[case("a", true)] // Minimum length 1
#[case("", false)] // Empty not allowed
#[case("tool name", false)] // Spaces not allowed
#[case("tool.name", false)] // Dots not allowed
#[case("tool@name", false)] // Special chars not allowed
fn test_tool_name_validation(#[case] name: &str, #[case] expected: bool) {
    assert_eq!(validate_tool_name(name), expected, "Failed for: {}", name);
}

/// Test tool name length limits separately (can't use .repeat() in rstest attributes).
#[test]
fn test_tool_name_length_limits() {
    // 64 chars - should be valid (max length)
    let max_valid = "a".repeat(64);
    assert!(
        validate_tool_name(&max_valid),
        "64-char name should be valid"
    );

    // 65 chars - should be invalid (too long)
    let too_long = "a".repeat(65);
    assert!(
        !validate_tool_name(&too_long),
        "65-char name should be invalid"
    );
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
    assert_eq!(
        tool.description,
        Some("Get the current weather for a location".to_string())
    );
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
fn test_tool_choice_simple_deserialization(
    #[case] input: &str,
    #[case] expected: ToolChoiceSimple,
) {
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
    let result = generate_tool_system_prompt(&[], None, None);
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

    let result = generate_tool_system_prompt(&tools, None, None);

    // Should use ai00 XML format
    assert!(result.contains("<ai00:available_tools>"));
    assert!(result.contains("</ai00:available_tools>"));
    assert!(result.contains("<ai00:function_calls>"));
    assert!(result.contains("<invoke name="));
    assert!(result.contains("<parameter name="));

    // Should contain the tool definition with Anthropic-style JSON
    assert!(result.contains("<tool name=\"get_weather\">"));
    assert!(result.contains("Get the current weather for a location"));
    // JSON is pretty-printed with proper indentation
    assert!(result.contains("\"name\": \"get_weather\""));
    assert!(result.contains("\"input_schema\""));
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

    let result = generate_tool_system_prompt(&tools, None, None);

    // Should contain both tools
    assert!(result.contains("get_weather"));
    assert!(result.contains("search_web"));
    assert!(result.contains("Get weather"));
    assert!(result.contains("Search the web"));
}

/// Test generate_tool_system_prompt ignores custom header/footer (deprecated).
#[test]
fn test_generate_tool_system_prompt_custom_prompts() {
    let tools = vec![Tool {
        name: "my_tool".to_string(),
        description: Some("A custom tool".to_string()),
        input_schema: json!({"type": "object"}),
        cache_control: None,
    }];

    let custom_header = "[TOOLS]\n";
    let custom_footer = "\n[/TOOLS]\nUse tools wisely.";

    let result = generate_tool_system_prompt(&tools, Some(custom_header), Some(custom_footer));

    // Custom header/footer are now ignored (deprecated)
    // Should use ai00 format regardless
    assert!(result.contains("<ai00:available_tools>"));
    assert!(result.contains("</ai00:available_tools>"));
    assert!(result.contains("my_tool"));
    // Should NOT use the custom values
    assert!(!result.contains("[TOOLS]"));
    assert!(!result.contains("Use tools wisely"));
}

/// Test generate_tool_system_prompt uses ai00 XML format.
#[test]
fn test_generate_tool_system_prompt_defaults() {
    let tools = vec![Tool {
        name: "test_tool".to_string(),
        description: Some("Test".to_string()),
        input_schema: json!({"type": "object"}),
        cache_control: None,
    }];

    let result = generate_tool_system_prompt(&tools, None, None);

    // Should have ai00 XML format
    assert!(result.contains("<ai00:available_tools>"));
    assert!(result.contains("</ai00:available_tools>"));
    assert!(result.contains("<tool name=\"test_tool\">"));
    // Should have tool calling instructions
    assert!(result.contains("<ai00:function_calls>"));
}

/// Test PromptsConfig default values.
/// See docs/ai00_chat_format.md for format details.
#[test]
fn test_prompts_config_defaults() {
    let config = PromptsConfig::default();

    // Check role defaults (ai00 v1 format - lowercase for XML tags)
    assert_eq!(config.role_user, "user");
    assert_eq!(config.role_assistant, "assistant");
    assert_eq!(config.role_system, "system");

    // Check assistant prefix defaults (ai00 XML format)
    assert_eq!(config.assistant_prefix, "<ai00:assistant>\n");
    assert_eq!(
        config.assistant_prefix_thinking,
        "<ai00:assistant>\n<think>\n"
    );

    // Check thinking suffix defaults
    assert_eq!(config.thinking_suffix_short, " think a bit");
    assert_eq!(config.thinking_suffix_standard, " think");
    assert_eq!(config.thinking_suffix_extended, " think a lot");

    // Check default stop sequences (ai00 format)
    assert_eq!(config.default_stop_sequences, vec!["</ai00:assistant>"]);

    // Check tool header and footer contain expected content
    assert!(config.tool_header.contains("# Tools"));
    assert!(config.tool_footer.contains("NEVER use <tool_use>"));
}

// =============================================================================
// Tool Result Tests
// =============================================================================

/// Test tool_result content block parsing with string content.
#[test]
fn test_tool_result_string_content() {
    let json = json!({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01abc123",
                "content": "The weather in NYC is 72°F and sunny."
            }
        ]
    });

    let msg: MessageParam = serde_json::from_value(json).unwrap();
    let text = msg.content.to_text();

    // Should use ai00 format
    assert!(text.contains("<ai00:function_results>"));
    assert!(text.contains("</ai00:function_results>"));
    assert!(text.contains("<result name=\"toolu_01abc123\">"));
    assert!(text.contains("72°F"));
}

/// Test tool_result content block parsing with array content.
#[test]
fn test_tool_result_array_content() {
    let json = json!({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_02xyz789",
                "content": [
                    {"type": "text", "text": "Search results:"},
                    {"type": "text", "text": "1. First result"}
                ]
            }
        ]
    });

    let msg: MessageParam = serde_json::from_value(json).unwrap();
    let text = msg.content.to_text();

    // Should use ai00 format
    assert!(text.contains("<ai00:function_results>"));
    assert!(text.contains("<result name=\"toolu_02xyz789\">"));
    assert!(text.contains("Search results:"));
    assert!(text.contains("First result"));
}

/// Test tool_result with is_error flag.
#[test]
fn test_tool_result_with_error() {
    let json = json!({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_03err456",
                "content": "Error: API rate limit exceeded",
                "is_error": true
            }
        ]
    });

    let msg: MessageParam = serde_json::from_value(json).unwrap();
    let text = msg.content.to_text();

    // Should use ai00 format with is_error attribute
    assert!(text.contains("<ai00:function_results>"));
    assert!(text.contains("is_error=\"true\""));
    assert!(text.contains("rate limit"));
}

/// Test tool_use in assistant message formats as ai00:function_calls.
#[test]
fn test_tool_use_in_assistant_message() {
    let json = json!({
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_04call789",
                "name": "get_weather",
                "input": {"location": "San Francisco"}
            }
        ]
    });

    let msg: MessageParam = serde_json::from_value(json).unwrap();
    let text = msg.content.to_text();

    // Should use ai00 format
    assert!(text.contains("<ai00:function_calls>"));
    assert!(text.contains("</ai00:function_calls>"));
    assert!(text.contains("<invoke name=\"get_weather\">"));
    assert!(text.contains("San Francisco"));
}

/// Test mixed content with text and tool_result.
#[test]
fn test_mixed_text_and_tool_result() {
    let json = json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "Here's the result:"},
            {
                "type": "tool_result",
                "tool_use_id": "toolu_05mix123",
                "content": "42"
            }
        ]
    });

    let msg: MessageParam = serde_json::from_value(json).unwrap();
    let text = msg.content.to_text();

    assert!(text.contains("Here's the result:"));
    // Should use ai00 format
    assert!(text.contains("<ai00:function_results>"));
    assert!(text.contains("42"));
}

/// Test generate_tool_system_prompt formatting.
#[test]
fn test_generate_tool_system_prompt() {
    use ai00_server::api::messages::generate_tool_system_prompt;

    let tools = vec![Tool {
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
    }];

    let result = generate_tool_system_prompt(&tools, None, None);

    // Should have ai00 XML format
    assert!(result.contains("<ai00:available_tools>"));
    assert!(result.contains("</ai00:available_tools>"));
    assert!(result.contains("<tool name=\"calculate\">"));
    assert!(result.contains("</tool>"));
    // Should have Anthropic-style JSON fields
    assert!(result.contains("\"name\": \"calculate\""));
    assert!(result.contains("\"description\": \"Perform arithmetic calculations\""));
    assert!(result.contains("\"input_schema\""));
    // Should have tool calling instructions
    assert!(result.contains("<ai00:function_calls>"));
    assert!(result.contains("<invoke name="));
    assert!(result.contains("<parameter name="));
}

// =============================================================================
// ToolParser Integration Tests
// =============================================================================

use ai00_server::api::messages::ToolParser;

/// Test ToolParser detects tool_call blocks.
#[test]
fn test_tool_parser_detects_tool_call() {
    let mut parser = ToolParser::new();

    let output = r#"Let me check the weather.
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>"#;

    let result = parser.feed(output);
    let final_result = parser.finalize();

    // Should have detected one tool use
    assert!(parser.has_tool_use());
    assert_eq!(parser.tool_count(), 1);

    // Collect all tool uses
    let mut tools: Vec<_> = result.tool_uses;
    tools.extend(final_result.tool_uses);
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(tools[0].input["location"], "NYC");

    // Should have text content
    let text = result.text.unwrap_or_default() + &final_result.text.unwrap_or_default();
    assert!(text.contains("Let me check the weather"));
}

/// Test ToolParser with multiple tool calls.
#[test]
fn test_tool_parser_multiple_tool_calls() {
    let mut parser = ToolParser::new();

    let output = r#"<tool_call>
{"name": "search", "arguments": {"query": "rust async"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"expr": "2+2"}}
</tool_call>"#;

    let result = parser.feed(output);
    let final_result = parser.finalize();

    // Should have detected two tool uses
    assert_eq!(parser.tool_count(), 2);

    let mut tools: Vec<_> = result.tool_uses;
    tools.extend(final_result.tool_uses);
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].name, "search");
    assert_eq!(tools[1].name, "calculate");
}

/// Test full conversation flow with tool_use and tool_result.
#[test]
fn test_conversation_flow_with_tools() {
    // First turn: user asks a question
    let user_msg = json!({
        "role": "user",
        "content": "What's the weather in Paris?"
    });
    let user: MessageParam = serde_json::from_value(user_msg).unwrap();
    assert_eq!(user.content.to_text(), "What's the weather in Paris?");

    // Second turn: assistant responds with tool_use
    let assistant_msg = json!({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll check the weather for you."
            },
            {
                "type": "tool_use",
                "id": "toolu_01weather123",
                "name": "get_weather",
                "input": {"location": "Paris, France"}
            }
        ]
    });
    let assistant: MessageParam = serde_json::from_value(assistant_msg).unwrap();
    let assistant_text = assistant.content.to_text();
    assert!(assistant_text.contains("I'll check the weather"));
    // Should use ai00 format
    assert!(assistant_text.contains("<ai00:function_calls>"));
    assert!(assistant_text.contains("<invoke name=\"get_weather\">"));

    // Third turn: user provides tool_result
    let user_result_msg = json!({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01weather123",
                "content": "Weather in Paris: 18°C, Partly cloudy"
            }
        ]
    });
    let user_result: MessageParam = serde_json::from_value(user_result_msg).unwrap();
    let result_text = user_result.content.to_text();
    // Should use ai00 format
    assert!(result_text.contains("<ai00:function_results>"));
    assert!(result_text.contains("<result name=\"toolu_01weather123\">"));
    assert!(result_text.contains("18°C"));
}

/// Test StopReason::ToolUse is correctly serialized.
#[test]
fn test_stop_reason_tool_use_serialization() {
    let reason = StopReason::ToolUse;
    let json = serde_json::to_value(reason).unwrap();
    assert_eq!(json, "tool_use");
}

/// Test response with tool_use content blocks.
#[test]
fn test_response_with_tool_use_content() {
    let content = vec![
        ContentBlock::Text {
            text: "Let me search for that.".to_string(),
        },
        ContentBlock::ToolUse {
            id: "toolu_abc123".to_string(),
            name: "web_search".to_string(),
            input: json!({"query": "Rust programming"}),
        },
    ];

    let response = MessagesResponse::new("test-model".to_string(), content, Default::default())
        .with_stop_reason(StopReason::ToolUse);

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["stop_reason"], "tool_use");
    assert_eq!(json["content"].as_array().unwrap().len(), 2);
    assert_eq!(json["content"][0]["type"], "text");
    assert_eq!(json["content"][1]["type"], "tool_use");
    assert_eq!(json["content"][1]["name"], "web_search");
}

// =============================================================================
// ThinkingConfig Tests
// =============================================================================

/// Test ThinkingConfig enabled deserialization.
#[test]
fn test_thinking_config_enabled_deserialization() {
    let json = json!({
        "type": "enabled",
        "budget_tokens": 10000
    });

    let config: ThinkingConfig = serde_json::from_value(json).unwrap();
    assert!(config.is_enabled());
    assert_eq!(config.budget_tokens(), Some(10000));
}

/// Test ThinkingConfig disabled deserialization.
#[test]
fn test_thinking_config_disabled_deserialization() {
    let json = json!({
        "type": "disabled"
    });

    let config: ThinkingConfig = serde_json::from_value(json).unwrap();
    assert!(!config.is_enabled());
    assert_eq!(config.budget_tokens(), None);
}

/// Test ThinkingConfig validation.
#[test]
fn test_thinking_config_validation() {
    // Valid configuration
    let config = ThinkingConfig::Enabled {
        budget_tokens: 5000,
    };
    assert!(config.validate(10000).is_ok());

    // budget_tokens too low
    let config = ThinkingConfig::Enabled { budget_tokens: 500 };
    assert!(config.validate(10000).is_err());

    // budget_tokens >= max_tokens
    let config = ThinkingConfig::Enabled {
        budget_tokens: 10000,
    };
    assert!(config.validate(10000).is_err());

    // Disabled is always valid
    let config = ThinkingConfig::Disabled;
    assert!(config.validate(100).is_ok());
}

/// Test ThinkingConfig thinking tiers.
#[rstest]
#[case(1024, 1)]
#[case(4095, 1)]
#[case(4096, 2)]
#[case(16383, 2)]
#[case(16384, 3)]
#[case(65535, 3)]
#[case(65536, 4)]
#[case(100000, 4)]
fn test_thinking_config_tiers(#[case] budget: usize, #[case] expected_tier: u8) {
    let config = ThinkingConfig::Enabled {
        budget_tokens: budget,
    };
    assert_eq!(config.thinking_tier(), Some(expected_tier));
}

/// Test ThinkingConfig disabled has no tier.
#[test]
fn test_thinking_config_disabled_tier() {
    let config = ThinkingConfig::Disabled;
    assert_eq!(config.thinking_tier(), None);
}

/// Test request with thinking configuration.
#[test]
fn test_request_with_thinking_config() {
    let json = json!({
        "model": "rwkv",
        "messages": [{"role": "user", "content": "Solve this math problem"}],
        "max_tokens": 20000,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 10000
        }
    });

    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(request.thinking.is_some());
    let thinking = request.thinking.unwrap();
    assert!(thinking.is_enabled());
    assert_eq!(thinking.budget_tokens(), Some(10000));
}

/// Test ContentBlock::Thinking serialization.
#[test]
fn test_thinking_content_block_serialization() {
    let block = ContentBlock::Thinking {
        thinking: "Let me reason through this...".to_string(),
        signature: "sig_abc123def456".to_string(),
    };

    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "thinking");
    assert_eq!(json["thinking"], "Let me reason through this...");
    assert_eq!(json["signature"], "sig_abc123def456");
}

/// Test ContentBlock::Thinking deserialization.
#[test]
fn test_thinking_content_block_deserialization() {
    let json = json!({
        "type": "thinking",
        "thinking": "Step 1: analyze the problem...",
        "signature": "sig_1234567890abcdef"
    });

    let block: ContentBlock = serde_json::from_value(json).unwrap();
    match block {
        ContentBlock::Thinking {
            thinking,
            signature,
        } => {
            assert_eq!(thinking, "Step 1: analyze the problem...");
            assert_eq!(signature, "sig_1234567890abcdef");
        }
        _ => panic!("Expected Thinking content block"),
    }
}

/// Test response with thinking content block.
#[test]
fn test_response_with_thinking_block() {
    let response = MessagesResponse::new(
        "rwkv-7-g1".to_string(),
        vec![
            ContentBlock::Thinking {
                thinking: "Let me think about this...".to_string(),
                signature: "sig_test12345678".to_string(),
            },
            ContentBlock::Text {
                text: "The answer is 42.".to_string(),
            },
        ],
        Default::default(),
    );

    let json = serde_json::to_value(&response).unwrap();

    // Check thinking block
    assert_eq!(json["content"][0]["type"], "thinking");
    assert_eq!(json["content"][0]["thinking"], "Let me think about this...");
    assert_eq!(json["content"][0]["signature"], "sig_test12345678");

    // Check text block
    assert_eq!(json["content"][1]["type"], "text");
    assert_eq!(json["content"][1]["text"], "The answer is 42.");
}

/// Test that thinking in MessageContent extracts thinking text.
#[test]
fn test_message_content_with_thinking_to_text() {
    let content = MessageContent::Blocks(vec![
        ContentBlock::Thinking {
            thinking: "My reasoning process...".to_string(),
            signature: "sig_ignored".to_string(),
        },
        ContentBlock::Text {
            text: "Final answer".to_string(),
        },
    ]);

    let text = content.to_text();
    // Thinking content should be included
    assert!(text.contains("My reasoning process..."));
    assert!(text.contains("Final answer"));
}

// ThinkingExtractor Integration Tests

/// Test ThinkingExtractor extracts thinking from model output.
#[test]
fn test_thinking_extractor_integration() {
    let extractor = ThinkingExtractor::new();
    let model_output = "<think>Step 1: Analyze the problem\nStep 2: Consider options\nStep 3: Choose best approach</think>\n\nBased on my analysis, the answer is 42.";

    let result = extractor.extract(model_output);

    assert!(result.has_thinking);
    assert!(result.thinking.as_ref().unwrap().contains("Step 1"));
    assert!(result.thinking.as_ref().unwrap().contains("Step 3"));
    assert_eq!(result.response, "Based on my analysis, the answer is 42.");
}

/// Test ThinkingExtractor handles missing thinking tags.
#[test]
fn test_thinking_extractor_no_thinking() {
    let extractor = ThinkingExtractor::new();
    let model_output = "This is a direct response without thinking.";

    let result = extractor.extract(model_output);

    assert!(!result.has_thinking);
    assert!(result.thinking.is_none());
    assert_eq!(result.response, model_output);
}

/// Test signature generation produces consistent results.
#[test]
fn test_thinking_signature_consistency() {
    let thinking = "Some extended reasoning process";

    let sig1 = generate_thinking_signature(thinking);
    let sig2 = generate_thinking_signature(thinking);

    // Same input should produce same signature
    assert_eq!(sig1, sig2);

    // Signature format check
    assert!(sig1.starts_with("sig_"));
    assert_eq!(sig1.len(), 20); // sig_ (4) + 16 hex chars

    // Different input should produce different signature
    let sig3 = generate_thinking_signature("Different thinking");
    assert_ne!(sig1, sig3);
}

// ThinkingStreamParser Integration Tests

/// Test ThinkingStreamParser handles complete thinking flow.
#[test]
fn test_thinking_stream_parser_full_flow() {
    let mut parser = ThinkingStreamParser::new();

    // Start in thinking state
    assert_eq!(parser.state(), ThinkingStreamState::InsideThinking);

    // Feed thinking content
    let r1 = parser.feed("Let me think");
    assert!(r1.thinking.is_some());
    assert!(!r1.thinking_complete);

    // Feed more thinking
    let r2 = parser.feed(" about this problem...");
    assert!(r2.thinking.is_some());

    // Feed end tag and response
    let r3 = parser.feed("</think>\n\nThe answer is 42.");
    assert!(r3.thinking_complete);
    assert_eq!(parser.state(), ThinkingStreamState::AfterThinking);

    // Verify accumulated content
    assert!(parser.thinking_content().contains("Let me think"));
    assert!(parser.thinking_content().contains("about this problem"));
}

/// Test ThinkingStreamParser handles split end tag.
#[test]
fn test_thinking_stream_parser_split_tag() {
    let mut parser = ThinkingStreamParser::new();

    // Feed content ending with partial tag
    parser.feed("thinking</th");
    assert_eq!(parser.state(), ThinkingStreamState::InsideThinking);

    // Complete the tag
    let result = parser.feed("ink>response");
    assert!(result.thinking_complete);
    assert_eq!(parser.state(), ThinkingStreamState::AfterThinking);
}

/// Test ThinkingStreamParser finalize without end tag.
#[test]
fn test_thinking_stream_parser_finalize_incomplete() {
    let mut parser = ThinkingStreamParser::new();

    parser.feed("Still thinking without closing tag");
    let result = parser.finalize();

    // Should mark thinking as complete even without end tag
    assert!(result.thinking_complete);
}

// Streaming Error Tests

/// Test streaming error event without partial content.
#[test]
fn test_stream_error_event_simple() {
    let event = emit_error("api_error", "Generation failed", None);

    // Should be named "error"
    // Note: SseEvent internals aren't easily inspectable, so we verify via JSON in the text
    let text = format!("{:?}", event);
    assert!(text.contains("error"));
}

/// Test streaming error event with partial content.
#[test]
fn test_stream_error_event_with_partial() {
    let partial = vec![ContentBlock::Text {
        text: "Partial response before error...".to_string(),
    }];

    let _event = emit_error(
        "overloaded_error",
        "Server overloaded",
        Some(partial.clone()),
    );

    // Verify StreamErrorEvent serialization
    let error_data = StreamErrorEvent {
        event_type: "error",
        error: ai00_server::api::messages::StreamErrorData {
            error_type: "overloaded_error".to_string(),
            message: "Server overloaded".to_string(),
            partial_content: Some(partial),
        },
    };

    let json = serde_json::to_value(&error_data).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["type"], "overloaded_error");
    assert_eq!(json["error"]["message"], "Server overloaded");
    assert!(json["error"]["partial_content"].is_array());
    assert_eq!(json["error"]["partial_content"][0]["type"], "text");
}

// =============================================================================
// BNF Schema Tests
// =============================================================================

/// Test request with bnf_schema serializes correctly.
#[test]
fn test_bnf_schema_serialization() {
    let request = MessagesRequest {
        model: "test".into(),
        messages: vec![MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".into()),
        }],
        system: None,
        max_tokens: 100,
        stream: false,
        stop_sequences: None,
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        metadata: None,
        bnf_schema: Some("start ::= \"hello\"".into()),
        bnf_validation: None,
    };
    let json = serde_json::to_value(&request).unwrap();
    assert_eq!(json["bnf_schema"], "start ::= \"hello\"");
}

/// Test request without bnf_schema deserializes correctly (optional field).
#[test]
fn test_bnf_schema_optional() {
    let json = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100
    });
    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(request.bnf_schema.is_none());
}

/// Test bnf_schema is not serialized when None.
#[test]
fn test_bnf_schema_skips_serialization_when_none() {
    let request = MessagesRequest {
        model: "test".into(),
        messages: vec![MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".into()),
        }],
        system: None,
        max_tokens: 100,
        stream: false,
        stop_sequences: None,
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        metadata: None,
        bnf_schema: None,
        bnf_validation: None,
    };
    let json = serde_json::to_value(&request).unwrap();
    assert!(json.get("bnf_schema").is_none());
}

/// Test bnf_schema deserialization with complex grammar.
#[test]
fn test_bnf_schema_complex_grammar() {
    let grammar = r#"start ::= json_object
json_object ::= "{" key_value_list "}"
key_value_list ::= key_value ("," key_value)*
key_value ::= string ":" value
string ::= "\"" [a-zA-Z]+ "\""
value ::= string | number
number ::= [0-9]+"#;

    let json = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "Generate JSON"}],
        "max_tokens": 100,
        "bnf_schema": grammar
    });

    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(request.bnf_schema.as_deref(), Some(grammar));
}

// =============================================================================
// BNF Validation Level Tests
// =============================================================================

use ai00_server::api::messages::BnfValidationLevel;

/// Test BnfValidationLevel enum serialization.
#[rstest]
#[case(BnfValidationLevel::None, "none")]
#[case(BnfValidationLevel::Structural, "structural")]
#[case(BnfValidationLevel::SchemaAware, "schema_aware")]
fn test_bnf_validation_level_serialization(
    #[case] level: BnfValidationLevel,
    #[case] expected: &str,
) {
    let json = serde_json::to_value(&level).unwrap();
    assert_eq!(json, expected);
}

/// Test BnfValidationLevel enum deserialization.
#[rstest]
#[case("none", BnfValidationLevel::None)]
#[case("structural", BnfValidationLevel::Structural)]
#[case("schema_aware", BnfValidationLevel::SchemaAware)]
fn test_bnf_validation_level_deserialization(
    #[case] input: &str,
    #[case] expected: BnfValidationLevel,
) {
    let level: BnfValidationLevel = serde_json::from_value(json!(input)).unwrap();
    assert_eq!(level, expected);
}

/// Test BnfValidationLevel is_enabled method.
#[test]
fn test_bnf_validation_level_is_enabled() {
    assert!(!BnfValidationLevel::None.is_enabled());
    assert!(BnfValidationLevel::Structural.is_enabled());
    assert!(BnfValidationLevel::SchemaAware.is_enabled());
}

/// Test BnfValidationLevel default is None.
#[test]
fn test_bnf_validation_level_default() {
    let level = BnfValidationLevel::default();
    assert_eq!(level, BnfValidationLevel::None);
}

/// Test bnf_validation parameter in request serialization.
#[test]
fn test_bnf_validation_request_serialization() {
    let request = MessagesRequest {
        model: "test".into(),
        messages: vec![MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".into()),
        }],
        system: None,
        max_tokens: 100,
        stream: false,
        stop_sequences: None,
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        metadata: None,
        bnf_schema: None,
        bnf_validation: Some(BnfValidationLevel::Structural),
    };
    let json = serde_json::to_value(&request).unwrap();
    assert_eq!(json["bnf_validation"], "structural");
}

/// Test bnf_validation parameter deserialization.
#[test]
fn test_bnf_validation_request_deserialization() {
    let json = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "bnf_validation": "schema_aware"
    });
    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(
        request.bnf_validation,
        Some(BnfValidationLevel::SchemaAware)
    );
}

/// Test bnf_validation is optional (None when not provided).
#[test]
fn test_bnf_validation_optional() {
    let json = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100
    });
    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(request.bnf_validation.is_none());
}

/// Test bnf_validation is not serialized when None.
#[test]
fn test_bnf_validation_skips_serialization_when_none() {
    let request = MessagesRequest {
        model: "test".into(),
        messages: vec![MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".into()),
        }],
        system: None,
        max_tokens: 100,
        stream: false,
        stop_sequences: None,
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        metadata: None,
        bnf_schema: None,
        bnf_validation: None,
    };
    let json = serde_json::to_value(&request).unwrap();
    assert!(json.get("bnf_validation").is_none());
}

/// Test both bnf_schema and bnf_validation can be set.
#[test]
fn test_bnf_schema_and_validation_together() {
    let json = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "bnf_schema": "start ::= \"hello\"",
        "bnf_validation": "structural"
    });
    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(request.bnf_schema.as_deref(), Some("start ::= \"hello\""));
    assert_eq!(request.bnf_validation, Some(BnfValidationLevel::Structural));
}

// =============================================================================
// BNF Grammar Generation Integration Tests
// =============================================================================

use ai00_server::api::messages::bnf_generator::generate_bnf_schema;
use ai00_server::api::messages::bnf_grammars::{
    build_structural_grammar, GRAMMAR_JSON_PRIMITIVES, GRAMMAR_UNIFIED,
};

/// Test that generate_bnf_schema returns None for None level.
#[test]
fn test_integration_generate_bnf_schema_none_level() {
    let stop_seqs = vec!["\n\n".to_string()];

    let result = generate_bnf_schema(None, false, BnfValidationLevel::None, &stop_seqs);
    assert!(result.is_none());

    let result = generate_bnf_schema(None, true, BnfValidationLevel::None, &stop_seqs);
    assert!(result.is_none());
}

/// Test Structural level generates unified grammar with all features.
/// The unified grammar always includes both thinking and tool support (both optional).
#[test]
fn test_integration_structural_unified_grammar() {
    let stop_seqs = vec!["\n\n".to_string()];

    // All parameter combinations produce the same unified grammar
    let result = generate_bnf_schema(None, false, BnfValidationLevel::Structural, &stop_seqs);
    assert!(result.is_some());

    let grammar = result.unwrap();
    // Unified grammar always contains both thinking and function call tags
    assert!(grammar.contains("<think>"));
    assert!(grammar.contains("</think>"));
    assert!(grammar.contains("<ai00:function_calls>"));
    assert!(grammar.contains("</ai00:function_calls>"));
    // Should contain JSON primitives
    assert!(grammar.contains("json_object"));
    // Should use complement regex
    assert!(grammar.contains("#ex'"));
}

/// Test Structural level produces same grammar regardless of parameters.
#[test]
fn test_integration_structural_all_params_same_grammar() {
    let tools = vec![Tool {
        name: "test_tool".to_string(),
        description: Some("Test description".to_string()),
        input_schema: json!({"type": "object"}),
        cache_control: None,
    }];
    let stop_seqs = vec!["\n\n".to_string()];

    // All combinations should produce identical grammar
    let g1 = generate_bnf_schema(None, false, BnfValidationLevel::Structural, &stop_seqs).unwrap();
    let g2 = generate_bnf_schema(None, true, BnfValidationLevel::Structural, &stop_seqs).unwrap();
    let g3 = generate_bnf_schema(
        Some(&tools),
        false,
        BnfValidationLevel::Structural,
        &stop_seqs,
    )
    .unwrap();
    let g4 = generate_bnf_schema(
        Some(&tools),
        true,
        BnfValidationLevel::Structural,
        &stop_seqs,
    )
    .unwrap();

    assert_eq!(g1, g2, "thinking param should not affect unified grammar");
    assert_eq!(g2, g3, "tools param should not affect unified grammar");
    assert_eq!(g3, g4, "both params should not affect unified grammar");
}

/// Test SchemaAware level generates tool-specific grammars.
#[test]
fn test_integration_schema_aware_with_tools() {
    let tools = vec![Tool {
        name: "get_weather".to_string(),
        description: Some("Get weather for a location".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }),
        cache_control: None,
    }];
    let stop_seqs = vec!["\n\n".to_string()];

    let result = generate_bnf_schema(
        Some(&tools),
        false,
        BnfValidationLevel::SchemaAware,
        &stop_seqs,
    );
    assert!(result.is_some());

    let grammar = result.unwrap();

    // Should have tool-specific rules
    assert!(grammar.contains("get_weather_call"));
    assert!(grammar.contains("get_weather_input"));

    // Should have enum values from schema
    assert!(grammar.contains("celsius") || grammar.contains("fahrenheit"));

    // SchemaAware also includes thinking (unified approach)
    assert!(grammar.contains("<think>"));
}

/// Test SchemaAware with multiple tools generates dispatch rule.
#[test]
fn test_integration_schema_aware_multiple_tools() {
    let tools = vec![
        Tool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }),
            cache_control: None,
        },
        Tool {
            name: "search".to_string(),
            description: Some("Search the web".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }),
            cache_control: None,
        },
    ];
    let stop_seqs = vec!["\n\n".to_string()];

    let result = generate_bnf_schema(
        Some(&tools),
        false,
        BnfValidationLevel::SchemaAware,
        &stop_seqs,
    );
    assert!(result.is_some());

    let grammar = result.unwrap();

    // Should have both tool-specific rules
    assert!(grammar.contains("get_weather_call"));
    assert!(grammar.contains("search_call"));
}

/// Test SchemaAware always includes thinking (unified grammar).
#[test]
fn test_integration_schema_aware_always_has_thinking() {
    let tools = vec![Tool {
        name: "test_tool".to_string(),
        description: Some("Test".to_string()),
        input_schema: json!({"type": "object"}),
        cache_control: None,
    }];
    let stop_seqs = vec!["\n\n".to_string()];

    // Even with thinking=false, unified grammar includes thinking
    let result = generate_bnf_schema(
        Some(&tools),
        false,
        BnfValidationLevel::SchemaAware,
        &stop_seqs,
    );
    assert!(result.is_some());

    let grammar = result.unwrap();

    // Should have thinking support (unified grammar)
    assert!(grammar.contains("thinking"));
    assert!(grammar.contains("<think>"));

    // Should also have tool rules
    assert!(grammar.contains("test_tool_call"));
}

/// Test grammar constants are valid KBNF.
#[test]
fn test_integration_grammar_constants_structure() {
    // JSON primitives constant
    assert!(GRAMMAR_JSON_PRIMITIVES.contains("::="));
    assert!(GRAMMAR_JSON_PRIMITIVES.contains(";"));
    assert!(GRAMMAR_JSON_PRIMITIVES.contains("json_object"));

    // Unified grammar constant
    assert!(GRAMMAR_UNIFIED.contains("::="));
    assert!(GRAMMAR_UNIFIED.contains("<think>"));
    assert!(GRAMMAR_UNIFIED.contains("<ai00:function_calls>")); // ai00 XML format
    assert!(GRAMMAR_UNIFIED.contains("#ex'")); // Complement regex
}

/// Test build_structural_grammar helper produces unified grammar.
#[test]
fn test_integration_build_structural_grammar() {
    let stop_seqs = vec!["\n\n".to_string()];

    // All combinations produce unified grammar with both features
    for (thinking, tools) in [(false, false), (true, false), (false, true), (true, true)] {
        let grammar = build_structural_grammar(thinking, tools, &stop_seqs);

        // Unified grammar always has start rule
        assert!(
            grammar.contains("start::="),
            "Missing start for ({}, {})",
            thinking,
            tools
        );
        assert!(
            grammar.contains("json_object"),
            "Missing json_object for ({}, {})",
            thinking,
            tools
        );

        // Unified grammar always has both (both are optional in the grammar)
        assert!(
            grammar.contains("<think>"),
            "Missing <think> for ({}, {})",
            thinking,
            tools
        );
        assert!(
            grammar.contains("<ai00:function_calls>"),
            "Missing <ai00:function_calls> for ({}, {})",
            thinking,
            tools
        );

        // Should have terminator
        assert!(
            grammar.contains("terminator::="),
            "Missing terminator for ({}, {})",
            thinking,
            tools
        );
    }
}

/// Test grammar generation with complex nested schema.
#[test]
fn test_integration_schema_aware_nested_schema() {
    let tools = vec![Tool {
        name: "create_user".to_string(),
        description: Some("Create a user".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "email"]
                },
                "notify": {"type": "boolean"}
            },
            "required": ["user"]
        }),
        cache_control: None,
    }];
    let stop_seqs = vec!["\n\n".to_string()];

    let result = generate_bnf_schema(
        Some(&tools),
        false,
        BnfValidationLevel::SchemaAware,
        &stop_seqs,
    );
    assert!(result.is_some());

    let grammar = result.unwrap();

    // Should handle nested object
    assert!(grammar.contains("create_user_input"));
    // Should have property references
    assert!(grammar.contains("user") || grammar.contains("name") || grammar.contains("email"));
}

/// Test SchemaAware fallback when no tools provided.
/// With unified grammar, it always returns the structural grammar (thinking and tools are optional).
#[test]
fn test_integration_schema_aware_no_tools_fallback() {
    let stop_seqs = vec!["\n\n".to_string()];

    // No tools - should fall back to unified structural grammar
    let result = generate_bnf_schema(None, true, BnfValidationLevel::SchemaAware, &stop_seqs);
    assert!(result.is_some());

    let grammar = result.unwrap();
    // Unified grammar always has both
    assert!(grammar.contains("<think>"));
    assert!(grammar.contains("<ai00:function_calls>")); // ai00 XML format

    // No tools, no thinking - still returns unified grammar
    let result = generate_bnf_schema(None, false, BnfValidationLevel::SchemaAware, &stop_seqs);
    assert!(result.is_some()); // Changed: now always returns unified grammar
}

// =============================================================================
// User Grammar Wrapping Tests (ninchat-4pt)
// =============================================================================

use ai00_server::api::messages::bnf_grammars::wrap_grammar_with_thinking;

/// Test wrap_grammar_with_thinking adds thinking block support.
#[test]
fn test_wrap_grammar_with_thinking_basic() {
    let user_grammar = r#"start ::= greeting;
greeting ::= "Hello" | "Hi";"#;

    let wrapped = wrap_grammar_with_thinking(user_grammar);

    // Should have new start rule with thinking block (uses ? for optional in KBNF)
    assert!(wrapped.contains("start::=thinking_block? user_start"));
    // User's start should be renamed to user_start
    assert!(wrapped.contains("user_start::= greeting"));
    // User's other rules should be preserved
    assert!(wrapped.contains("greeting ::="));
    // Should have thinking block definitions
    assert!(wrapped.contains("thinking_block"));
    assert!(wrapped.contains("<think>"));
    assert!(wrapped.contains("</think>"));
}

/// Test wrap_grammar_with_thinking preserves user's non-start rules.
#[test]
fn test_wrap_grammar_preserves_rules() {
    let user_grammar = r#"start ::= json_response;
json_response ::= "{" key_value "}";
key_value ::= string ":" string;
string ::= '"' content '"';
content ::= #"[^\"]*";"#;

    let wrapped = wrap_grammar_with_thinking(user_grammar);

    // All user rules should be preserved
    assert!(wrapped.contains("json_response ::="));
    assert!(wrapped.contains("key_value ::="));
    assert!(wrapped.contains("string ::="));
    assert!(wrapped.contains("content ::="));
}

/// Test wrap_grammar_with_thinking with complex start rule.
#[test]
fn test_wrap_grammar_complex_start() {
    let user_grammar = r#"start ::= text | json_object | list;
text ::= #"[a-zA-Z ]+";
json_object ::= "{" "}";
list ::= "[" "]";"#;

    let wrapped = wrap_grammar_with_thinking(user_grammar);

    // Start rule should be renamed (note: replacement removes space before ::=)
    assert!(wrapped.contains("user_start::= text | json_object | list"));
    // New start rule should reference user_start (uses ? for optional in KBNF)
    assert!(wrapped.contains("thinking_block? user_start"));
}

/// Test that bnf_schema + thinking no longer causes an error.
#[test]
fn test_bnf_schema_with_thinking_allowed() {
    // This combination should now be accepted (grammar is wrapped automatically)
    let json = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "bnf_schema": "start ::= \"hello\" | \"world\";",
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1000
        }
    });

    // Should deserialize without error
    let request: MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(request.bnf_schema.is_some());
    assert!(request.thinking.is_some());
}

// =============================================================================
// Handler Integration Tests (Tool Parsing Flow)
// =============================================================================
//
// These tests verify the full tool parsing flow as it would happen in the
// handler, simulating the token stream and parsing.

/// Test that has_tools evaluates correctly with tools in request.
#[test]
fn test_has_tools_detection() {
    // Request without tools
    let request_no_tools = MessagesRequest {
        model: "test".into(),
        messages: vec![MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".into()),
        }],
        system: None,
        max_tokens: 100,
        stream: false,
        stop_sequences: None,
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        metadata: None,
        bnf_schema: None,
        bnf_validation: None,
    };

    let has_tools = request_no_tools
        .tools
        .as_ref()
        .map(|t| !t.is_empty())
        .unwrap_or(false);
    assert!(!has_tools, "has_tools should be false when tools is None");

    // Request with empty tools array
    let request_empty_tools = MessagesRequest {
        tools: Some(vec![]),
        ..request_no_tools.clone()
    };

    let has_tools = request_empty_tools
        .tools
        .as_ref()
        .map(|t| !t.is_empty())
        .unwrap_or(false);
    assert!(!has_tools, "has_tools should be false when tools is empty");

    // Request with tools
    let request_with_tools = MessagesRequest {
        tools: Some(vec![Tool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            input_schema: json!({"type": "object"}),
            cache_control: None,
        }]),
        ..request_no_tools
    };

    let has_tools = request_with_tools
        .tools
        .as_ref()
        .map(|t| !t.is_empty())
        .unwrap_or(false);
    assert!(has_tools, "has_tools should be true when tools are present");
}

/// Simulate the handler's tool parsing logic from respond_one.
/// This tests the exact flow used in the handler.
fn simulate_respond_one_tool_parsing(text: &str, has_tools: bool) -> Vec<ContentBlock> {
    if has_tools {
        // Parse the output for tool_call blocks
        let mut parser = ToolParser::new();
        let result = parser.feed(text);
        let final_result = parser.finalize();

        let mut content_blocks: Vec<ContentBlock> = Vec::new();

        // Add text content if any
        let text_content = result.text.unwrap_or_default() + &final_result.text.unwrap_or_default();
        let trimmed_text = text_content.trim();
        if !trimmed_text.is_empty() {
            content_blocks.push(ContentBlock::Text {
                text: trimmed_text.to_string(),
            });
        }

        // Add tool_use blocks
        let mut all_tools: Vec<_> = result.tool_uses;
        all_tools.extend(final_result.tool_uses);

        for tool_use in all_tools.iter() {
            content_blocks.push(ContentBlock::ToolUse {
                id: tool_use.id.clone(),
                name: tool_use.name.clone(),
                input: tool_use.input.clone(),
            });
        }

        content_blocks
    } else {
        // Simple text response
        vec![ContentBlock::Text {
            text: text.trim().to_string(),
        }]
    }
}

/// Test handler tool parsing with a simple tool_call.
#[test]
fn test_handler_tool_parsing_simple() {
    let model_output = r#"I'll check the weather for you.
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>"#;

    let blocks = simulate_respond_one_tool_parsing(model_output, true);

    assert_eq!(blocks.len(), 2, "Should have text and tool_use blocks");

    // First block should be text
    match &blocks[0] {
        ContentBlock::Text { text } => {
            assert!(
                text.contains("check the weather"),
                "Text should contain message"
            );
        }
        _ => panic!("First block should be text"),
    }

    // Second block should be tool_use
    match &blocks[1] {
        ContentBlock::ToolUse { id, name, input } => {
            assert!(id.starts_with("toolu_"), "ID should have tool prefix");
            assert_eq!(name, "get_weather");
            assert_eq!(input["location"], "NYC");
        }
        _ => panic!("Second block should be tool_use"),
    }
}

/// Test handler tool parsing when has_tools is false (should not parse).
#[test]
fn test_handler_tool_parsing_disabled() {
    let model_output = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>"#;

    // When has_tools is false, should return raw text
    let blocks = simulate_respond_one_tool_parsing(model_output, false);

    assert_eq!(blocks.len(), 1, "Should have only text block");
    match &blocks[0] {
        ContentBlock::Text { text } => {
            assert!(
                text.contains("<tool_call>"),
                "Raw text should contain tool_call tag"
            );
        }
        _ => panic!("Should be text block"),
    }
}

/// Test handler tool parsing with multiple tool calls.
#[test]
fn test_handler_tool_parsing_multiple() {
    let model_output = r#"Let me search and calculate.
<tool_call>
{"name": "search", "arguments": {"query": "rust async"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"expr": "2+2"}}
</tool_call>"#;

    let blocks = simulate_respond_one_tool_parsing(model_output, true);

    assert_eq!(blocks.len(), 3, "Should have text and 2 tool_use blocks");

    match &blocks[1] {
        ContentBlock::ToolUse { name, .. } => assert_eq!(name, "search"),
        _ => panic!("Should be tool_use"),
    }

    match &blocks[2] {
        ContentBlock::ToolUse { name, .. } => assert_eq!(name, "calculate"),
        _ => panic!("Should be tool_use"),
    }
}

/// Test handler tool parsing with tool_call but NO text before it.
#[test]
fn test_handler_tool_parsing_no_text() {
    let model_output = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>"#;

    let blocks = simulate_respond_one_tool_parsing(model_output, true);

    // Should only have tool_use block (no empty text block)
    assert_eq!(blocks.len(), 1, "Should have only tool_use block");
    match &blocks[0] {
        ContentBlock::ToolUse { name, .. } => assert_eq!(name, "get_weather"),
        _ => panic!("Should be tool_use"),
    }
}

/// Test handler tool parsing with compact JSON (no newlines in JSON).
#[test]
fn test_handler_tool_parsing_compact_json() {
    let model_output =
        r#"<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>"#;

    let blocks = simulate_respond_one_tool_parsing(model_output, true);

    assert_eq!(blocks.len(), 1, "Should have tool_use block");
    match &blocks[0] {
        ContentBlock::ToolUse { name, input, .. } => {
            assert_eq!(name, "get_weather");
            assert_eq!(input["location"], "NYC");
        }
        _ => panic!("Should be tool_use"),
    }
}

/// Test handler tool parsing with nested JSON arguments.
#[test]
fn test_handler_tool_parsing_nested_json() {
    let model_output = r#"<tool_call>
{"name": "create_user", "arguments": {"user": {"name": "John", "email": "john@test.com"}, "notify": true}}
</tool_call>"#;

    let blocks = simulate_respond_one_tool_parsing(model_output, true);

    assert_eq!(blocks.len(), 1);
    match &blocks[0] {
        ContentBlock::ToolUse { name, input, .. } => {
            assert_eq!(name, "create_user");
            assert_eq!(input["user"]["name"], "John");
            assert_eq!(input["notify"], true);
        }
        _ => panic!("Should be tool_use"),
    }
}

/// Test handler tool parsing with empty arguments.
#[test]
fn test_handler_tool_parsing_empty_args() {
    let model_output = r#"<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call>"#;

    let blocks = simulate_respond_one_tool_parsing(model_output, true);

    assert_eq!(blocks.len(), 1);
    match &blocks[0] {
        ContentBlock::ToolUse { name, input, .. } => {
            assert_eq!(name, "get_current_time");
            assert!(input.as_object().unwrap().is_empty());
        }
        _ => panic!("Should be tool_use"),
    }
}

/// Test handler tool parsing with text after tool call.
#[test]
fn test_handler_tool_parsing_text_after() {
    let model_output = r#"<tool_call>
{"name": "search", "arguments": {"q": "test"}}
</tool_call>
Here are the results."#;

    let blocks = simulate_respond_one_tool_parsing(model_output, true);

    // Should have text (after) and tool_use
    // Note: With current parser, text after might be captured
    assert!(blocks.len() >= 1, "Should have at least tool_use block");

    // Find the tool_use block
    let tool_use = blocks
        .iter()
        .find(|b| matches!(b, ContentBlock::ToolUse { .. }));
    assert!(tool_use.is_some(), "Should have tool_use block");
}

/// Test that ToolParser correctly handles streaming tokens.
#[test]
fn test_tool_parser_streaming_simulation() {
    let mut parser = ToolParser::new();
    let mut all_tools = Vec::new();
    let mut all_text = String::new();

    // Simulate streaming tokens that form a tool_call
    let tokens = [
        "Let me ",
        "check. ",
        "<tool",
        "_call>",
        "{\"name\": ",
        "\"get_weather\", ",
        "\"arguments\": ",
        "{\"location\": \"NYC\"}}",
        "</tool_call>",
    ];

    for token in tokens {
        let result = parser.feed(token);
        if let Some(text) = result.text {
            all_text.push_str(&text);
        }
        all_tools.extend(result.tool_uses);
    }

    let final_result = parser.finalize();
    if let Some(text) = final_result.text {
        all_text.push_str(&text);
    }
    all_tools.extend(final_result.tool_uses);

    assert_eq!(all_tools.len(), 1, "Should have parsed one tool");
    assert_eq!(all_tools[0].name, "get_weather");
    assert!(
        all_text.contains("Let me check"),
        "Should have text content"
    );
}

/// Test response JSON structure matches Claude API format.
#[test]
fn test_response_structure_with_tool_use() {
    let content = vec![
        ContentBlock::Text {
            text: "Let me search for that.".to_string(),
        },
        ContentBlock::ToolUse {
            id: "toolu_000000000001".to_string(),
            name: "web_search".to_string(),
            input: json!({"query": "Rust programming"}),
        },
    ];

    let response = MessagesResponse::new("rwkv".to_string(), content, Default::default())
        .with_stop_reason(StopReason::ToolUse);

    let json = serde_json::to_value(&response).unwrap();

    // Verify structure matches Claude API
    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "assistant");
    assert_eq!(json["stop_reason"], "tool_use");

    // Verify content array
    let content = json["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);

    // Text block
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "Let me search for that.");

    // Tool use block
    assert_eq!(content[1]["type"], "tool_use");
    assert_eq!(content[1]["id"], "toolu_000000000001");
    assert_eq!(content[1]["name"], "web_search");
    assert_eq!(content[1]["input"]["query"], "Rust programming");
}

/// Test parsing model output that uses "input" instead of "arguments".
/// The model might use "input" (Claude format) instead of "arguments" (OpenAI format).
#[test]
fn test_tool_parser_input_vs_arguments() {
    let mut parser = ToolParser::new();

    // Our parser expects "arguments" field (Hermes/Qwen format)
    let with_arguments = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>"#;

    let result = parser.feed(with_arguments);
    let final_result = parser.finalize();
    let tools: Vec<_> = result
        .tool_uses
        .into_iter()
        .chain(final_result.tool_uses)
        .collect();

    assert_eq!(tools.len(), 1, "Should parse with 'arguments' field");
    assert_eq!(tools[0].input["location"], "NYC");
}

/// Test that model output with <tool_use> tags is NOT parsed.
/// (We only support <tool_call> format per Hermes/Qwen spec)
#[test]
fn test_tool_use_tags_not_parsed() {
    let mut parser = ToolParser::new();

    // Model might output <tool_use> if it was trained on Claude format
    let with_tool_use = r#"<tool_use>
{"name": "get_weather", "input": {"location": "NYC"}}
</tool_use>"#;

    let result = parser.feed(with_tool_use);
    let final_result = parser.finalize();
    let tools: Vec<_> = result
        .tool_uses
        .into_iter()
        .chain(final_result.tool_uses)
        .collect();

    // Should NOT parse - we only support <tool_call>
    assert_eq!(tools.len(), 0, "Should NOT parse <tool_use> tags");

    // The <tool_use> content should be in the text output
    let text = result.text.unwrap_or_default() + &final_result.text.unwrap_or_default();
    assert!(
        text.contains("<tool_use>"),
        "tool_use should appear in text"
    );
}
