//! Claude-compatible Messages API.
//!
//! This module provides a `/v1/messages` endpoint compatible with
//! Anthropic's Claude Messages API format.

pub mod bnf_generator;
pub mod bnf_grammars;
mod handler;
pub mod prompt;
mod streaming;
mod thinking_extractor;
mod tool_parser;
mod types;

pub use handler::messages_handler;
pub use streaming::{emit_error, StreamErrorData, StreamErrorEvent};
pub use thinking_extractor::{
    generate_thinking_signature, ThinkingExtractor, ThinkingResult, ThinkingStreamParser,
    ThinkingStreamResult, ThinkingStreamState,
};
pub use tool_parser::{ParseResult, ParsedToolUse, ToolParser};
pub use types::*;
