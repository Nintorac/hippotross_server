//! Demo test to show the generated prompt format.

use ai00_server::api::messages::{
    ContentBlock, MessageContent, MessageParam, MessageRole, ThinkingConfig, Tool,
    ToolResultContent,
};
use ai00_server::config::PromptsConfig;
use serde_json::json;

// Re-implement build_prompt here since it's not public
fn build_prompt(
    system: Option<&str>,
    messages: &[MessageParam],
    tools: Option<&[Tool]>,
    thinking: Option<&ThinkingConfig>,
    prompts: &PromptsConfig,
) -> String {
    let mut prompt = String::new();

    // Add system prompt with XML turn markers
    if let Some(sys) = system {
        prompt.push_str(&format!("<ai00:{}>\n", prompts.role_system));
        prompt.push_str(sys);

        // Inject tool definitions into system prompt if provided
        if let Some(tools) = tools {
            if !tools.is_empty() {
                prompt.push_str(&generate_tool_system_prompt(tools));
            }
        }

        prompt.push_str(&format!("\n</ai00:{}>\n\n", prompts.role_system));
    } else if let Some(tools) = tools {
        if !tools.is_empty() {
            prompt.push_str(&format!("<ai00:{}>\n", prompts.role_system));
            prompt.push_str(&generate_tool_system_prompt(tools));
            prompt.push_str(&format!("\n</ai00:{}>\n\n", prompts.role_system));
        }
    }

    // Format conversation messages with XML turn markers
    let msg_count = messages.len();
    for (i, msg) in messages.iter().enumerate() {
        let role = match msg.role {
            MessageRole::User => &prompts.role_user,
            MessageRole::Assistant => &prompts.role_assistant,
        };
        let content = msg.content.to_text();

        // For the last user message when thinking is enabled, append think suffix
        let is_last_user = i == msg_count - 1 && msg.role == MessageRole::User;
        let think_suffix = if is_last_user {
            get_thinking_suffix(thinking, prompts)
        } else {
            ""
        };

        prompt.push_str(&format!(
            "<ai00:{}>\n{}{}\n</ai00:{}>\n\n",
            role, content, think_suffix, role
        ));
    }

    // Add assistant prefix for generation
    if thinking.map(|t| t.is_enabled()).unwrap_or(false) {
        prompt.push_str(&prompts.assistant_prefix_thinking);
    } else {
        prompt.push_str(&prompts.assistant_prefix);
    }

    prompt.trim_end().to_string()
}

fn get_thinking_suffix<'a>(
    thinking: Option<&ThinkingConfig>,
    prompts: &'a PromptsConfig,
) -> &'a str {
    match thinking {
        Some(ThinkingConfig::Enabled { budget_tokens }) => match *budget_tokens {
            0..=4095 => &prompts.thinking_suffix_short,
            4096..=16383 => &prompts.thinking_suffix_standard,
            _ => &prompts.thinking_suffix_extended,
        },
        _ => "",
    }
}

fn generate_tool_system_prompt(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut result = String::from("\n\n<ai00:available_tools>\n");

    for tool in tools {
        result.push_str(&format!("  <tool name=\"{}\">\n", tool.name));

        let tool_json = json!({
            "name": tool.name,
            "description": tool.description.clone().unwrap_or_default(),
            "input_schema": tool.input_schema
        });

        if let Ok(pretty) = serde_json::to_string_pretty(&tool_json) {
            for line in pretty.lines() {
                result.push_str("    ");
                result.push_str(line);
                result.push('\n');
            }
        }

        result.push_str("  </tool>\n");
    }

    result.push_str("</ai00:available_tools>\n\n");
    result.push_str(
        "To call a function, use this format:\n\
         <ai00:function_calls>\n  \
         <invoke name=\"function_name\">\n    \
         <parameter name=\"param\">value</parameter>\n  \
         </invoke>\n\
         </ai00:function_calls>",
    );

    result
}

#[test]
fn demo_prompt_output() {
    let prompts = PromptsConfig::default();

    let tools = vec![Tool {
        name: "get_weather".to_string(),
        description: Some("Get weather for a location".to_string()),
        input_schema: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        cache_control: None,
    }];

    let messages = vec![
        MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Hi there!".into()),
        },
        MessageParam {
            role: MessageRole::Assistant,
            content: MessageContent::Text("Hello! How can I help you today?".into()),
        },
        MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("What's the weather in NYC?".into()),
        },
        MessageParam {
            role: MessageRole::Assistant,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Let me check that for you.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_01abc".into(),
                    name: "get_weather".into(),
                    input: json!({"location": "NYC"}),
                },
            ]),
        },
        MessageParam {
            role: MessageRole::User,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_01abc".into(),
                content: ToolResultContent::Text("72°F, Sunny".into()),
                is_error: false,
            }]),
        },
        MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Thanks! What about London?".into()),
        },
    ];

    let thinking = ThinkingConfig::Enabled { budget_tokens: 5000 };

    let prompt = build_prompt(
        Some("You are a helpful weather assistant."),
        &messages,
        Some(&tools),
        Some(&thinking),
        &prompts,
    );

    println!("\n\n=== GENERATED PROMPT ===\n{}\n=== END PROMPT ===\n\n", prompt);

    // Write to file for inspection
    std::fs::write("/tmp/ai00_prompt_output.txt", &prompt).unwrap();
    println!("Written to /tmp/ai00_prompt_output.txt");

    // Verify new format elements
    assert!(prompt.contains("<ai00:system>"));
    assert!(prompt.contains("</ai00:system>"));
    assert!(prompt.contains("<ai00:user>"));
    assert!(prompt.contains("</ai00:user>"));
    assert!(prompt.contains("<ai00:assistant>"));
    assert!(prompt.contains("<ai00:available_tools>"));
    assert!(prompt.contains("<tool name=\"get_weather\">"));
}
