//! Claude-compatible Messages API.
//!
//! This module provides a `/v1/messages` endpoint compatible with
//! Anthropic's Claude Messages API format.

mod handler;
mod streaming;
mod types;

pub use handler::messages_handler;
pub use types::*;
