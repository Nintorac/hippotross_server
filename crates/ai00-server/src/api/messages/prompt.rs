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

    // Add assistant prefix for generation (opens the assistant turn)
    if thinking.map(|t| t.is_enabled()).unwrap_or(false) {
        // Thinking mode: use configurable thinking prefix
        prompt.push_str(&prompts.assistant_prefix_thinking);
    } else {
        prompt.push_str(&prompts.assistant_prefix);
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
}
