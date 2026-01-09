//! Prompt building utilities for RWKV models.
//!
//! This module contains functions for building prompts from messages,
//! used by both the HTTP server and CLI tools like make-binidx.

use super::types::{generate_tool_system_prompt, MessageParam, MessageRole, ThinkingConfig, Tool};
use crate::config::PromptsConfig;

/// Collapse multiple consecutive newlines into a single newline.
///
/// RWKV uses `\n\n` as a "chat round separator" in pretrain data, so we must
/// avoid having `\n\n` within message content to prevent premature turn endings.
///
/// See: https://huggingface.co/BlinkDL/rwkv7-g1
pub fn normalize_newlines(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_newline = false;

    for c in text.chars() {
        if c == '\n' {
            if !prev_was_newline {
                result.push('\n');
            }
            prev_was_newline = true;
        } else {
            result.push(c);
            prev_was_newline = false;
        }
    }
    result
}

/// Build RWKV prompt from messages.
///
/// Format for RWKV7-G1 (https://huggingface.co/BlinkDL/rwkv7-g1):
/// ```text
/// System: SYSTEM_PROMPT
///
/// User: USER_MESSAGE
///
/// Assistant: ASSISTANT_MESSAGE
///
/// User: USER_MESSAGE
///
/// Assistant:
/// ```
///
/// For thinking mode (RWKV 20250922+ models):
/// ```text
/// User: USER_PROMPT think
///
/// Assistant: <think
/// ```
/// With variants "think a bit" (shorter) and "think a lot" (longer).
pub fn build_prompt(
    system: Option<&str>,
    messages: &[MessageParam],
    tools: Option<&[Tool]>,
    thinking: Option<&ThinkingConfig>,
    prompts: &PromptsConfig,
) -> String {
    let mut prompt = String::new();

    // Add system prompt first (from top-level param, not message role)
    // Normalize newlines to prevent \n\n from being interpreted as turn separator
    if let Some(sys) = system {
        let normalized_sys = normalize_newlines(sys);
        prompt.push_str(&format!("{}: {}", prompts.role_system, normalized_sys));

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

        prompt.push_str("\n\n");
    } else if let Some(tools) = tools {
        // If no system prompt but tools provided, create one for tools
        if !tools.is_empty() {
            prompt.push_str(&format!("{}:", prompts.role_system));
            prompt.push_str(&generate_tool_system_prompt(
                tools,
                Some(&prompts.tool_header),
                Some(&prompts.tool_footer),
            ));
            prompt.push_str("\n\n");
        }
    }

    // Format conversation messages
    // Normalize newlines in content to prevent \n\n from being interpreted as turn separator
    let msg_count = messages.len();
    for (i, msg) in messages.iter().enumerate() {
        let role = match msg.role {
            MessageRole::User => &prompts.role_user,
            MessageRole::Assistant => &prompts.role_assistant,
        };
        let content = normalize_newlines(&msg.content.to_text());

        // For the last user message when thinking is enabled, append think suffix
        let is_last_user = i == msg_count - 1 && msg.role == MessageRole::User;
        let think_suffix = if is_last_user {
            get_thinking_suffix(thinking, prompts)
        } else {
            ""
        };

        prompt.push_str(&format!("{}: {}{}\n\n", role, content, think_suffix));
    }

    // Add assistant prefix for generation
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
    fn test_normalize_newlines() {
        assert_eq!(normalize_newlines("a\n\nb"), "a\nb");
        assert_eq!(normalize_newlines("a\n\n\nb"), "a\nb");
        assert_eq!(normalize_newlines("a\nb"), "a\nb");
        assert_eq!(normalize_newlines("abc"), "abc");
    }

    #[test]
    fn test_build_prompt_simple() {
        use super::super::types::{MessageContent, MessageParam, MessageRole};

        let prompts = PromptsConfig::default();
        let messages = vec![
            MessageParam {
                role: MessageRole::User,
                content: MessageContent::Text("Hello".to_string()),
            },
        ];

        let prompt = build_prompt(Some("You are helpful."), &messages, None, None, &prompts);

        assert!(prompt.contains("System: You are helpful."));
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.ends_with("Assistant:"));
    }
}
