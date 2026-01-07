//! Types for Claude-compatible Messages API.
//!
//! These types match the Anthropic Messages API format for compatibility
//! with Claude API clients (e.g., LibreChat with `defaultParamsEndpoint: 'anthropic'`).

use lazy_static::lazy_static;
use regex::Regex;
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};

lazy_static! {
    /// Regex for validating tool names: 1-64 chars, alphanumeric plus underscore/hyphen.
    static ref TOOL_NAME_REGEX: Regex = Regex::new(r"^[a-zA-Z0-9_-]{1,64}$").unwrap();
}

/// Validate that a tool name matches the required pattern.
pub fn validate_tool_name(name: &str) -> bool {
    TOOL_NAME_REGEX.is_match(name)
}

/// Message role - only "user" or "assistant" allowed.
/// Note: Claude API does NOT support "system" role in messages array.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    #[default]
    User,
    Assistant,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "User"),
            MessageRole::Assistant => write!(f, "Assistant"),
        }
    }
}

/// Content block types in messages and responses.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },

    /// Tool use request from the model
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Unique tool call ID (e.g., "toolu_01A09q...")
        id: String,
        /// Tool name
        name: String,
        /// Tool arguments as JSON
        input: serde_json::Value,
    },

    /// Tool result from the client
    #[serde(rename = "tool_result")]
    ToolResult {
        /// ID of the tool_use this is responding to
        tool_use_id: String,
        /// Result content
        #[serde(default)]
        content: ToolResultContent,
        /// Whether the tool execution failed
        #[serde(default)]
        is_error: bool,
    },

    /// Thinking/reasoning trace
    #[serde(rename = "thinking")]
    Thinking {
        /// The reasoning content
        thinking: String,
        /// Placeholder signature (hash-based, not Anthropic-compatible)
        signature: String,
    },
}

/// Tool result content - can be string or array of content blocks.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ToolResultContent {
    #[default]
    Empty,
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Message content - can be simple string or array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text string (shorthand)
    Text(String),
    /// Array of content blocks
    Blocks(Vec<ContentBlock>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl MessageContent {
    /// Extract text content from message, concatenating text blocks.
    pub fn to_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MessageParam {
    /// Message role (user or assistant)
    pub role: MessageRole,
    /// Message content
    pub content: MessageContent,
}

/// Stop reason for Claude Messages API.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural completion
    EndTurn,
    /// Token limit reached
    MaxTokens,
    /// Custom stop sequence triggered
    StopSequence,
    /// Model invoked a tool
    ToolUse,
    /// API response still in progress
    #[default]
    #[serde(untagged)]
    Null,
}

/// Map internal FinishReason to Claude StopReason.
impl From<ai00_core::FinishReason> for StopReason {
    fn from(reason: ai00_core::FinishReason) -> Self {
        match reason {
            ai00_core::FinishReason::Stop => StopReason::EndTurn,
            ai00_core::FinishReason::Length => StopReason::MaxTokens,
            ai00_core::FinishReason::ContentFilter => StopReason::EndTurn,
            ai00_core::FinishReason::Null => StopReason::Null,
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema)]
pub struct Usage {
    /// Tokens in the input/prompt
    pub input_tokens: usize,
    /// Tokens in the output/completion
    pub output_tokens: usize,
    /// Cache tokens (always 0 for RWKV - no caching)
    #[serde(default)]
    pub cache_creation_input_tokens: usize,
    /// Cache read tokens (always 0 for RWKV)
    #[serde(default)]
    pub cache_read_input_tokens: usize,
}

impl From<ai00_core::TokenCounter> for Usage {
    fn from(counter: ai00_core::TokenCounter) -> Self {
        Self {
            input_tokens: counter.prompt,
            output_tokens: counter.completion,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }
    }
}

/// Tool definition for function calling.
///
/// Matches Claude API tool schema for compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Tool {
    /// Tool name (must match ^[a-zA-Z0-9_-]{1,64}$)
    pub name: String,

    /// Human-readable description of what the tool does
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for the tool's input parameters
    pub input_schema: serde_json::Value,

    /// Cache control settings (forward compatibility - ignored)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
}

impl Tool {
    /// Validate the tool definition.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !validate_tool_name(&self.name) {
            return Err("Tool name must be 1-64 alphanumeric characters, underscores, or hyphens");
        }
        if self.input_schema.is_null() {
            return Err("Tool input_schema is required");
        }
        Ok(())
    }
}

/// How the model should choose which tool to use.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Simple string choice: "auto", "none", or "any"
    Simple(ToolChoiceSimple),
    /// Specific tool choice
    Specific(ToolChoiceSpecific),
}

impl Default for ToolChoice {
    fn default() -> Self {
        ToolChoice::Simple(ToolChoiceSimple::Auto)
    }
}

/// Simple tool choice values.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceSimple {
    /// Model automatically decides whether to use tools
    #[default]
    Auto,
    /// Model will not use any tools
    None,
    /// Model must use one of the provided tools
    Any,
}

/// Specific tool choice - forces the model to use a particular tool.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ToolChoiceSpecific {
    /// Always "tool" for specific tool choice
    #[serde(rename = "type")]
    pub choice_type: String,
    /// Name of the tool to use
    pub name: String,
    /// Whether to disable parallel tool use (forward compatibility)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}

impl ToolChoiceSpecific {
    /// Create a new specific tool choice.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            choice_type: "tool".to_string(),
            name: name.into(),
            disable_parallel_tool_use: None,
        }
    }
}

/// Messages API request.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MessagesRequest {
    /// Model identifier
    pub model: String,

    /// Conversation messages (roles: "user" | "assistant" only)
    pub messages: Vec<MessageParam>,

    /// System prompt (top-level, NOT a message role)
    #[serde(default)]
    pub system: Option<String>,

    /// Maximum tokens to generate (required)
    pub max_tokens: usize,

    /// Enable streaming response
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,

    /// Sampling temperature (0.0 - 1.0)
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Top-p sampling
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Top-k sampling
    #[serde(default)]
    pub top_k: Option<usize>,

    /// Tools available for the model to use
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// How the model should choose which tool to use
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Metadata for request tracking (forward compatibility - ignored)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Messages API response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MessagesResponse {
    /// Unique message ID (e.g., "msg_01XFDUDYJgAACzvnptvVoYEL")
    pub id: String,

    /// Always "message" for Messages API
    #[serde(rename = "type")]
    pub object: String,

    /// Always "assistant"
    pub role: String,

    /// Model identifier
    pub model: String,

    /// Response content blocks
    pub content: Vec<ContentBlock>,

    /// Why generation stopped
    pub stop_reason: StopReason,

    /// Which stop sequence triggered (if stop_reason == StopSequence)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,

    /// Token usage statistics
    pub usage: Usage,
}

impl MessagesResponse {
    /// Create a new response with the given content.
    pub fn new(model: String, content: Vec<ContentBlock>, usage: Usage) -> Self {
        Self {
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            object: "message".to_string(),
            role: "assistant".to_string(),
            model,
            content,
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage,
        }
    }

    /// Set the stop reason.
    pub fn with_stop_reason(mut self, reason: StopReason) -> Self {
        self.stop_reason = reason;
        self
    }

    /// Set the stop sequence that triggered the stop.
    pub fn with_stop_sequence(mut self, sequence: String) -> Self {
        self.stop_sequence = Some(sequence);
        self
    }
}
