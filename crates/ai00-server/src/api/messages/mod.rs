//! Claude-compatible Messages API.
//!
//! This module provides a `/v1/messages` endpoint compatible with
//! Anthropic's Claude Messages API format.

mod handler;
mod streaming;
mod thinking_extractor;
mod tool_parser;
mod types;

pub use handler::messages_handler;
pub use thinking_extractor::{generate_thinking_signature, ThinkingExtractor, ThinkingResult};
pub use tool_parser::{ParseResult, ParsedToolUse, ToolParser};
pub use types::*;
