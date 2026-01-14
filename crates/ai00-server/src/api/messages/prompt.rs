//! Prompt building utilities for RWKV models.
//!
//! This module contains functions for building prompts from messages,
//! used by both the HTTP server and CLI tools like make-binidx.

use super::types::{generate_tool_system_prompt, MessageParam, MessageRole, ThinkingConfig, Tool};
use crate::config::PromptsConfig;

/// Build RWKV prompt from messages using ai00 chat format.
///
/// Format (ai00 v1):
/// ```text
/// <ai00:system>
/// SYSTEM_PROMPT
/// </ai00:system>
///
/// <ai00:user>
/// USER_MESSAGE
/// </ai00:user>
///
/// <ai00:assistant>
/// ASSISTANT_MESSAGE
/// </ai00:assistant>
///
/// <ai00:user>
/// USER_MESSAGE
/// </ai00:user>
///
/// <ai00:assistant>
/// ```
///
/// For thinking mode:
/// ```text
/// <ai00:user>
/// USER_MESSAGE think
/// </ai00:user>
///
/// <ai00:assistant>
/// <think>
/// ```
pub fn build_prompt(
    system: Option<&str>,
    messages: &[MessageParam],
    tools: Option<&[Tool]>,
    thinking: Option<&ThinkingConfig>,
    prompts: &PromptsConfig,
) -> String {
    build_prompt_inner(system, messages, tools, thinking, prompts, true)
}

/// Build RWKV prompt without the trailing assistant prefix.
///
/// Use this for training data where we don't want the incomplete
/// assistant turn at the end.
pub fn build_training_prompt(
    system: Option<&str>,
    messages: &[MessageParam],
    tools: Option<&[Tool]>,
    thinking: Option<&ThinkingConfig>,
    prompts: &PromptsConfig,
) -> String {
    build_prompt_inner(system, messages, tools, thinking, prompts, false)
}

fn build_prompt_inner(
    system: Option<&str>,
    messages: &[MessageParam],
    tools: Option<&[Tool]>,
    thinking: Option<&ThinkingConfig>,
    prompts: &PromptsConfig,
    include_assistant_prefix: bool,
) -> String {
    let mut prompt = String::new();

    // Add system prompt with XML turn markers
    // Newlines are fully preserved within turns (no filtering needed)
    if let Some(sys) = system {
        prompt.push_str(&format!("<ai00:{}>\n", prompts.role_system));
        prompt.push_str(sys);

        // Inject tool definitions into system prompt if provided
        if let Some(tools) = tools {
            if !tools.is_empty() {
                prompt.push_str(&generate_tool_system_prompt(
                    tools,
                    Some(&prompts.tool_header),
                    Some(&prompts.tool_footer),
                ));
            }
        }

        prompt.push_str(&format!("\n</ai00:{}>\n\n", prompts.role_system));
    } else if let Some(tools) = tools {
        // If no system prompt but tools provided, create one for tools
        if !tools.is_empty() {
            prompt.push_str(&format!("<ai00:{}>\n", prompts.role_system));
            prompt.push_str(&generate_tool_system_prompt(
                tools,
                Some(&prompts.tool_header),
                Some(&prompts.tool_footer),
            ));
            prompt.push_str(&format!("\n</ai00:{}>\n\n", prompts.role_system));
        }
    }

    // Format conversation messages with XML turn markers
    // Track current open turn to merge consecutive same-role messages
    let msg_count = messages.len();
    let mut current_turn: Option<MessageRole> = None;
    let mut turn_has_tool_use = false; // Track if current turn has pending tool_use

    // Helper to check if next non-tool-result message has same role
    let next_regular_msg_role = |from_idx: usize| -> Option<MessageRole> {
        messages[from_idx + 1..]
            .iter()
            .find(|m| !(m.role == MessageRole::User && m.content.is_tool_result_only()))
            .map(|m| m.role.clone())
    };

    for (i, msg) in messages.iter().enumerate() {
        let content = msg.content.to_text();

        // Tool result messages are injected without turn wrappers
        // They appear immediately after </ai00:function_calls> in the assistant turn
        if msg.role == MessageRole::User && msg.content.is_tool_result_only() {
            // Just output the <ai00:function_results> block directly (no turn wrapper)
            prompt.push_str(&content);
            prompt.push('\n');
            // Don't close the assistant turn yet - there may be more content
            continue;
        }

        // Get role string for this message
        let role_str = match msg.role {
            MessageRole::User => &prompts.role_user,
            MessageRole::Assistant => &prompts.role_assistant,
        };

        // For the last user message when thinking is enabled, append think suffix
        let is_last_user = i == msg_count - 1 && msg.role == MessageRole::User;
        let think_suffix = if is_last_user {
            get_thinking_suffix(thinking, prompts)
        } else {
            ""
        };

        // Check if this message contains tool_use
        let has_tool_use =
            msg.role == MessageRole::Assistant && content.contains("<ai00:function_calls>");

        // Check if next message has same role (to decide whether to close turn)
        let next_same_role = next_regular_msg_role(i) == Some(msg.role.clone());

        // Determine if we need to close current turn and/or start new one
        match current_turn {
            Some(current_role) if current_role == msg.role => {
                // Same role as current turn - append content (merge consecutive)
                prompt.push_str("\n");
                prompt.push_str(&content);
                prompt.push_str(think_suffix);

                if has_tool_use {
                    turn_has_tool_use = true;
                }

                // Decide whether to close the turn
                if !next_same_role && !turn_has_tool_use {
                    // Next message is different role and no pending tool_use - close turn
                    prompt.push_str(&format!("\n</ai00:{}>\n\n", role_str));
                    current_turn = None;
                    turn_has_tool_use = false;
                } else {
                    // Keep turn open (next is same role or has pending tool_use)
                    prompt.push('\n');
                }
            }
            Some(current_role) => {
                // Different role - close current turn first
                let prev_role_str = match current_role {
                    MessageRole::User => &prompts.role_user,
                    MessageRole::Assistant => &prompts.role_assistant,
                };
                prompt.push_str(&format!("</ai00:{}>\n\n", prev_role_str));
                turn_has_tool_use = false;

                // Start new turn
                prompt.push_str(&format!("<ai00:{}>\n{}{}", role_str, content, think_suffix));
                current_turn = Some(msg.role.clone());

                if has_tool_use {
                    turn_has_tool_use = true;
                }

                // Decide whether to close the turn
                if !next_same_role && !turn_has_tool_use {
                    prompt.push_str(&format!("\n</ai00:{}>\n\n", role_str));
                    current_turn = None;
                    turn_has_tool_use = false;
                } else {
                    prompt.push('\n');
                }
            }
            None => {
                // No current turn - start new one
                prompt.push_str(&format!("<ai00:{}>\n{}{}", role_str, content, think_suffix));
                current_turn = Some(msg.role.clone());

                if has_tool_use {
                    turn_has_tool_use = true;
                }

                // Decide whether to close the turn
                if !next_same_role && !turn_has_tool_use {
                    prompt.push_str(&format!("\n</ai00:{}>\n\n", role_str));
                    current_turn = None;
                    turn_has_tool_use = false;
                } else {
                    prompt.push('\n');
                }
            }
        }
    }

    // Close any remaining open turn
    if let Some(role) = current_turn {
        let role_str = match role {
            MessageRole::User => &prompts.role_user,
            MessageRole::Assistant => &prompts.role_assistant,
        };
        prompt.push_str(&format!("</ai00:{}>\n\n", role_str));
    }

    // Add assistant prefix for generation (opens the assistant turn)
    // Skip for training data where we want complete turns only
    if include_assistant_prefix {
        if thinking.map(|t| t.is_enabled()).unwrap_or(false) {
            // Thinking mode: use configurable thinking prefix
            prompt.push_str(&prompts.assistant_prefix_thinking);
        } else {
            prompt.push_str(&prompts.assistant_prefix);
        }
    }

    // RWKV requires no trailing whitespace or tokenizer may produce non-English output
    // See: https://huggingface.co/BlinkDL/rwkv7-g1
    prompt.trim_end().to_string()
}

/// Get the thinking suffix to append to user message based on budget.
pub fn get_thinking_suffix<'a>(
    thinking: Option<&ThinkingConfig>,
    prompts: &'a PromptsConfig,
) -> &'a str {
    match thinking {
        Some(ThinkingConfig::Enabled { budget_tokens }) => {
            // Map budget to thinking intensity
            match *budget_tokens {
                0..=4095 => &prompts.thinking_suffix_short,     // Tier 1: shorter thinking
                4096..=16383 => &prompts.thinking_suffix_standard, // Tier 2: standard thinking
                _ => &prompts.thinking_suffix_extended,         // Tier 3+: extended thinking
            }
        }
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_simple() {
        use super::super::types::{MessageContent, MessageParam, MessageRole};

        let prompts = PromptsConfig::default();
        let messages = vec![MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
        }];

        let prompt = build_prompt(Some("You are helpful."), &messages, None, None, &prompts);

        // Verify XML turn format
        assert!(prompt.contains("<ai00:system>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("</ai00:system>"));
        assert!(prompt.contains("<ai00:user>"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("</ai00:user>"));
        assert!(prompt.contains("<ai00:assistant>"));
    }

    #[test]
    fn test_build_prompt_preserves_newlines() {
        use super::super::types::{MessageContent, MessageParam, MessageRole};

        let prompts = PromptsConfig::default();
        let messages = vec![MessageParam {
            role: MessageRole::User,
            content: MessageContent::Text("Line 1\n\nLine 2\n\n\nLine 3".to_string()),
        }];

        let prompt = build_prompt(None, &messages, None, None, &prompts);

        // Verify newlines are preserved (no filtering)
        assert!(prompt.contains("Line 1\n\nLine 2\n\n\nLine 3"));
    }

    #[test]
    fn test_build_prompt_multi_turn() {
        use super::super::types::{MessageContent, MessageParam, MessageRole};

        let prompts = PromptsConfig::default();
        let messages = vec![
            MessageParam {
                role: MessageRole::User,
                content: MessageContent::Text("Hello".to_string()),
            },
            MessageParam {
                role: MessageRole::Assistant,
                content: MessageContent::Text("Hi there!".to_string()),
            },
            MessageParam {
                role: MessageRole::User,
                content: MessageContent::Text("How are you?".to_string()),
            },
        ];

        let prompt = build_prompt(None, &messages, None, None, &prompts);

        // Verify turn order is preserved
        let user1_pos = prompt.find("<ai00:user>\nHello").unwrap();
        let asst_pos = prompt.find("<ai00:assistant>\nHi there!").unwrap();
        let user2_pos = prompt.find("<ai00:user>\nHow are you?").unwrap();
        let final_asst_pos = prompt.rfind("<ai00:assistant>").unwrap();

        assert!(user1_pos < asst_pos);
        assert!(asst_pos < user2_pos);
        assert!(user2_pos < final_asst_pos);
    }

    #[test]
    fn test_build_prompt_tool_result_injection() {
        use super::super::types::{
            ContentBlock, MessageContent, MessageParam, MessageRole, ToolResultContent,
        };

        let prompts = PromptsConfig::default();
        let messages = vec![
            // User asks about weather
            MessageParam {
                role: MessageRole::User,
                content: MessageContent::Text("What's the weather in Tokyo?".to_string()),
            },
            // Assistant calls tool
            MessageParam {
                role: MessageRole::Assistant,
                content: MessageContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "I'll check that for you.".to_string(),
                    },
                    ContentBlock::ToolUse {
                        id: "toolu_001".to_string(),
                        name: "get_weather".to_string(),
                        input: serde_json::json!({"city": "Tokyo"}),
                    },
                ]),
            },
            // Tool result (should NOT be wrapped in <ai00:user>)
            MessageParam {
                role: MessageRole::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_001".to_string(),
                    content: ToolResultContent::Text(r#"{"temp": 22, "condition": "sunny"}"#.to_string()),
                    is_error: false,
                }]),
            },
            // Assistant continues after receiving tool result (same turn!)
            MessageParam {
                role: MessageRole::Assistant,
                content: MessageContent::Text("It's 22°C and sunny in Tokyo!".to_string()),
            },
        ];

        let prompt = build_prompt(None, &messages, None, None, &prompts);

        // Tool result should be injected WITHOUT <ai00:user> wrapper
        assert!(!prompt.contains("<ai00:user>\n<ai00:function_results>"));

        // Function results should appear directly after function_calls
        let function_calls_end = prompt.find("</ai00:function_calls>").unwrap();
        let function_results_start = prompt.find("<ai00:function_results>").unwrap();

        // function_results should come after function_calls (with only whitespace between)
        assert!(function_results_start > function_calls_end);

        // The text between should only be whitespace/newlines
        let between = &prompt[function_calls_end + "</ai00:function_calls>".len()..function_results_start];
        assert!(between.trim().is_empty(), "Expected only whitespace between function_calls and function_results, got: {:?}", between);

        // Continuation should be in the SAME assistant turn, not a new one
        // Count how many times <ai00:assistant> appears
        let assistant_opens: Vec<_> = prompt.match_indices("<ai00:assistant>").collect();
        let assistant_closes: Vec<_> = prompt.match_indices("</ai00:assistant>").collect();

        // Should have exactly 1 assistant turn (open at start of tool call, close after continuation)
        // Plus the final prefix for generation
        assert_eq!(assistant_opens.len(), 2, "Expected 2 <ai00:assistant> (1 turn + 1 prefix), got {}", assistant_opens.len());
        assert_eq!(assistant_closes.len(), 1, "Expected 1 </ai00:assistant>, got {}", assistant_closes.len());

        // The continuation text should be inside the assistant turn
        assert!(prompt.contains("It's 22°C and sunny in Tokyo!\n</ai00:assistant>"));
    }

    #[test]
    fn test_no_consecutive_same_role_turns() {
        use super::super::types::{ContentBlock, MessageContent, MessageParam, MessageRole};

        let prompts = PromptsConfig::default();

        // Simulate Toucan-style data where assistant has separate text and tool_use messages
        let messages = vec![
            MessageParam {
                role: MessageRole::User,
                content: MessageContent::Text("What's the weather?".to_string()),
            },
            // First assistant message: text only
            MessageParam {
                role: MessageRole::Assistant,
                content: MessageContent::Text("I'll check that for you.".to_string()),
            },
            // Second assistant message: tool_use only (consecutive!)
            MessageParam {
                role: MessageRole::Assistant,
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "toolu_001".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "Tokyo"}),
                }]),
            },
        ];

        // Use build_training_prompt to avoid the inference prefix at the end
        let prompt = build_training_prompt(None, &messages, None, None, &prompts);

        // Should NOT have consecutive </ai00:assistant>\n\n<ai00:assistant>
        assert!(
            !prompt.contains("</ai00:assistant>\n\n<ai00:assistant>"),
            "Should not have consecutive assistant turns. Got:\n{}",
            prompt
        );

        // Should merge into single assistant turn with both text and function_calls
        assert!(prompt.contains("I'll check that for you."));
        assert!(prompt.contains("<ai00:function_calls>"));

        // The text should appear before function_calls in the same turn
        let text_pos = prompt.find("I'll check that for you.").unwrap();
        let func_pos = prompt.find("<ai00:function_calls>").unwrap();
        assert!(text_pos < func_pos, "Text should appear before function_calls");
    }

    #[test]
    fn test_no_consecutive_user_turns() {
        use super::super::types::{MessageContent, MessageParam, MessageRole};

        let prompts = PromptsConfig::default();

        // Two consecutive user messages (shouldn't happen in normal flow, but test anyway)
        let messages = vec![
            MessageParam {
                role: MessageRole::User,
                content: MessageContent::Text("First question".to_string()),
            },
            MessageParam {
                role: MessageRole::User,
                content: MessageContent::Text("Second question".to_string()),
            },
        ];

        // Use build_training_prompt to avoid the inference prefix at the end
        let prompt = build_training_prompt(None, &messages, None, None, &prompts);

        // Should NOT have consecutive </ai00:user>\n\n<ai00:user>
        assert!(
            !prompt.contains("</ai00:user>\n\n<ai00:user>"),
            "Should not have consecutive user turns. Got:\n{}",
            prompt
        );
    }
}
