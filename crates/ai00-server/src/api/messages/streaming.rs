//! Claude-style SSE streaming for Messages API.
//!
//! Implements the Claude streaming format with named event types:
//! - message_start
//! - content_block_start
//! - content_block_delta
//! - content_block_stop
//! - message_delta
//! - message_stop
//! - ping (keep-alive)

use salvo::sse::SseEvent;
use serde::{Deserialize, Serialize};

use super::types::*;

/// message_start event - includes full message object with empty content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub message: MessageStartData,
}

/// Partial message data for message_start event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStartData {
    pub id: String,
    #[serde(rename = "type")]
    pub object: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// content_block_start event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: usize,
    pub content_block: ContentBlock,
}

/// content_block_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: usize,
    pub delta: ContentDelta,
}

/// Delta content types for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },

    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },

    #[serde(rename = "signature_delta")]
    Signature { signature: String },

    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

/// content_block_stop event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: usize,
}

/// message_delta event - final stop reason and usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub delta: MessageDeltaData,
    pub usage: OutputUsage,
}

/// Delta data for message_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaData {
    pub stop_reason: StopReason,
    pub stop_sequence: Option<String>,
}

/// Output-only usage for message_delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputUsage {
    pub output_tokens: usize,
}

/// message_stop event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
}

/// ping event for keep-alive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
}

/// Create a message_start SSE event.
pub fn emit_message_start(id: String, model: String, input_tokens: usize) -> SseEvent {
    let event = MessageStartEvent {
        event_type: "message_start",
        message: MessageStartData {
            id,
            object: "message",
            role: "assistant",
            model,
            content: vec![],
            stop_reason: None,
            stop_sequence: None,
            usage: Usage {
                input_tokens,
                output_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        },
    };
    SseEvent::default()
        .name("message_start")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a content_block_start SSE event for text.
pub fn emit_content_block_start_text(index: usize) -> SseEvent {
    let event = ContentBlockStartEvent {
        event_type: "content_block_start",
        index,
        content_block: ContentBlock::Text {
            text: String::new(),
        },
    };
    SseEvent::default()
        .name("content_block_start")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a content_block_start SSE event for tool_use.
pub fn emit_content_block_start_tool_use(index: usize, id: String, name: String) -> SseEvent {
    let event = ContentBlockStartEvent {
        event_type: "content_block_start",
        index,
        content_block: ContentBlock::ToolUse {
            id,
            name,
            input: serde_json::Value::Object(Default::default()),
        },
    };
    SseEvent::default()
        .name("content_block_start")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a content_block_delta SSE event for text.
pub fn emit_text_delta(index: usize, text: String) -> SseEvent {
    let event = ContentBlockDeltaEvent {
        event_type: "content_block_delta",
        index,
        delta: ContentDelta::Text { text },
    };
    SseEvent::default()
        .name("content_block_delta")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a content_block_delta SSE event for tool input JSON.
pub fn emit_input_json_delta(index: usize, partial_json: String) -> SseEvent {
    let event = ContentBlockDeltaEvent {
        event_type: "content_block_delta",
        index,
        delta: ContentDelta::InputJson { partial_json },
    };
    SseEvent::default()
        .name("content_block_delta")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a content_block_start SSE event for thinking.
pub fn emit_content_block_start_thinking(index: usize) -> SseEvent {
    let event = ContentBlockStartEvent {
        event_type: "content_block_start",
        index,
        content_block: ContentBlock::Thinking {
            thinking: String::new(),
            signature: String::new(),
        },
    };
    SseEvent::default()
        .name("content_block_start")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a content_block_delta SSE event for thinking.
pub fn emit_thinking_delta(index: usize, thinking: String) -> SseEvent {
    let event = ContentBlockDeltaEvent {
        event_type: "content_block_delta",
        index,
        delta: ContentDelta::Thinking { thinking },
    };
    SseEvent::default()
        .name("content_block_delta")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a content_block_delta SSE event for thinking signature.
pub fn emit_signature_delta(index: usize, signature: String) -> SseEvent {
    let event = ContentBlockDeltaEvent {
        event_type: "content_block_delta",
        index,
        delta: ContentDelta::Signature { signature },
    };
    SseEvent::default()
        .name("content_block_delta")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a content_block_stop SSE event.
pub fn emit_content_block_stop(index: usize) -> SseEvent {
    let event = ContentBlockStopEvent {
        event_type: "content_block_stop",
        index,
    };
    SseEvent::default()
        .name("content_block_stop")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a message_delta SSE event.
pub fn emit_message_delta(stop_reason: StopReason, output_tokens: usize) -> SseEvent {
    let event = MessageDeltaEvent {
        event_type: "message_delta",
        delta: MessageDeltaData {
            stop_reason,
            stop_sequence: None,
        },
        usage: OutputUsage { output_tokens },
    };
    SseEvent::default()
        .name("message_delta")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a message_stop SSE event.
pub fn emit_message_stop() -> SseEvent {
    let event = MessageStopEvent {
        event_type: "message_stop",
    };
    SseEvent::default()
        .name("message_stop")
        .text(serde_json::to_string(&event).unwrap())
}

/// Create a ping SSE event for keep-alive.
pub fn emit_ping() -> SseEvent {
    let event = PingEvent { event_type: "ping" };
    SseEvent::default()
        .name("ping")
        .text(serde_json::to_string(&event).unwrap())
}

/// error event - reports streaming error with optional partial content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamErrorEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub error: StreamErrorData,
}

/// Error data for streaming error event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamErrorData {
    /// Error type (e.g., "api_error", "overloaded_error")
    #[serde(rename = "type")]
    pub error_type: String,
    /// Human-readable error message
    pub message: String,
    /// Partial content blocks accumulated before error (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_content: Option<Vec<ContentBlock>>,
}

/// Create an error SSE event with optional partial content.
pub fn emit_error(
    error_type: &str,
    message: &str,
    partial_content: Option<Vec<ContentBlock>>,
) -> SseEvent {
    let event = StreamErrorEvent {
        event_type: "error",
        error: StreamErrorData {
            error_type: error_type.to_string(),
            message: message.to_string(),
            partial_content,
        },
    };
    SseEvent::default()
        .name("error")
        .text(serde_json::to_string(&event).unwrap())
}
