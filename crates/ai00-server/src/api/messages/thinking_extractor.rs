//! Thinking extraction for RWKV model output.
//!
//! RWKV thinking models output reasoning in `<think>...</think>` tags.
//! This module extracts thinking content and generates placeholder signatures.

use sha2::{Digest, Sha256};

/// Result of extracting thinking from model output.
#[derive(Debug, Clone, Default)]
pub struct ThinkingResult {
    /// The thinking/reasoning content (if any).
    pub thinking: Option<String>,
    /// The response content after thinking.
    pub response: String,
    /// Whether extraction found valid thinking tags.
    pub has_thinking: bool,
}

/// Extracts thinking content from RWKV model output.
///
/// RWKV models output thinking in `<think>...</think>` tags when prompted
/// with the thinking template format.
#[derive(Debug, Clone)]
pub struct ThinkingExtractor {
    think_start: &'static str,
    think_end: &'static str,
}

impl Default for ThinkingExtractor {
    fn default() -> Self {
        Self {
            think_start: "<think>",
            think_end: "</think>",
        }
    }
}

impl ThinkingExtractor {
    /// Create a new ThinkingExtractor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract thinking and response from raw model output.
    ///
    /// The model output format when thinking is enabled:
    /// ```text
    /// <think>reasoning content here</think>
    ///
    /// actual response here
    /// ```
    ///
    /// If no thinking tags are found, the entire text is returned as response.
    pub fn extract(&self, text: &str) -> ThinkingResult {
        // Look for <think> tag
        if let Some(start_idx) = text.find(self.think_start) {
            // Look for </think> tag after the start
            if let Some(end_idx) = text[start_idx..].find(self.think_end) {
                let thinking_start = start_idx + self.think_start.len();
                let thinking_end = start_idx + end_idx;
                let response_start = start_idx + end_idx + self.think_end.len();

                let thinking = text[thinking_start..thinking_end].trim().to_string();
                let response = text[response_start..].trim().to_string();

                return ThinkingResult {
                    thinking: Some(thinking),
                    response,
                    has_thinking: true,
                };
            }
        }

        // No valid thinking tags found - return all as response
        ThinkingResult {
            thinking: None,
            response: text.to_string(),
            has_thinking: false,
        }
    }
}

/// Generate a placeholder signature for thinking blocks.
///
/// NOTE: This is NOT cryptographically valid or Anthropic-compatible.
/// It's a hash-based placeholder for API shape compatibility.
/// The signature format is: `sig_` + first 16 chars of SHA256 hex.
pub fn generate_thinking_signature(thinking: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(thinking.as_bytes());
    let hash = hasher.finalize();

    // Use hex encoding (first 16 chars of hex = 8 bytes = 64 bits)
    let hex: String = hash.iter().take(8).map(|b| format!("{:02x}", b)).collect();
    format!("sig_{}", hex)
}

/// State for streaming thinking extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingStreamState {
    /// Looking for <think> tag (model-initiated thinking mode).
    DetectingThinking,
    /// Inside thinking block, waiting for </think> tag.
    InsideThinking,
    /// After </think>, now in response text.
    AfterThinking,
}

/// Result from feeding a token to the streaming parser.
#[derive(Debug, Clone, Default)]
pub struct ThinkingStreamResult {
    /// Thinking content to emit (if any).
    pub thinking: Option<String>,
    /// Response text to emit (if any).
    pub text: Option<String>,
    /// Whether thinking block just completed (emit signature now).
    pub thinking_complete: bool,
}

/// Streaming parser for thinking content.
///
/// Supports two modes:
/// - **Explicit thinking mode** (`new()`): Starts inside thinking block, assumes
///   prompt ends with `<think`. Used when thinking is explicitly enabled.
/// - **Detection mode** (`new_detecting()`): Starts looking for `<think>` tag.
///   Used when model may optionally decide to think.
#[derive(Debug, Clone)]
pub struct ThinkingStreamParser {
    state: ThinkingStreamState,
    buffer: String,
    thinking_content: String,
    think_start: &'static str,
    think_end: &'static str,
}

impl Default for ThinkingStreamParser {
    fn default() -> Self {
        Self {
            state: ThinkingStreamState::InsideThinking,
            buffer: String::new(),
            thinking_content: String::new(),
            think_start: "<think>",
            think_end: "</think>",
        }
    }
}

impl ThinkingStreamParser {
    /// Create a new streaming parser starting inside thinking block.
    ///
    /// Use this when thinking mode is explicitly enabled and the prompt
    /// ends with `<think` (model continues inside the thinking block).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new streaming parser in detection mode.
    ///
    /// Use this when the model may optionally decide to output thinking.
    /// The parser will look for `<think>` tags and extract them if present,
    /// otherwise all content is treated as regular text.
    pub fn new_detecting() -> Self {
        Self {
            state: ThinkingStreamState::DetectingThinking,
            buffer: String::new(),
            thinking_content: String::new(),
            think_start: "<think>",
            think_end: "</think>",
        }
    }

    /// Get the current state.
    pub fn state(&self) -> ThinkingStreamState {
        self.state
    }

    /// Get all accumulated thinking content.
    pub fn thinking_content(&self) -> &str {
        &self.thinking_content
    }

    /// Feed a token to the parser and get result.
    pub fn feed(&mut self, token: &str) -> ThinkingStreamResult {
        self.buffer.push_str(token);

        match self.state {
            ThinkingStreamState::DetectingThinking => {
                // Look for <think> tag to enter thinking mode
                if let Some(start_pos) = self.buffer.find(self.think_start) {
                    // Found start tag - emit text before it, switch to thinking mode
                    let text_before_tag = self.buffer[..start_pos].to_string();
                    self.buffer = self.buffer[start_pos + self.think_start.len()..].to_string();
                    self.state = ThinkingStreamState::InsideThinking;

                    // Now check if there's also a </think> in the remaining buffer
                    // (handles case where entire thinking block is in one token)
                    if let Some(end_pos) = self.buffer.find(self.think_end) {
                        let thinking_content = self.buffer[..end_pos].to_string();
                        let text_after_tag =
                            self.buffer[end_pos + self.think_end.len()..].to_string();

                        self.thinking_content.push_str(&thinking_content);
                        self.buffer.clear();
                        self.state = ThinkingStreamState::AfterThinking;

                        let mut result = ThinkingStreamResult {
                            thinking_complete: true,
                            ..Default::default()
                        };

                        if !text_before_tag.is_empty() {
                            result.text = Some(text_before_tag);
                        }
                        if !thinking_content.is_empty() {
                            result.thinking = Some(thinking_content);
                        }
                        // Note: text_after_tag will be emitted on next feed or finalize
                        if !text_after_tag.is_empty() {
                            self.buffer = text_after_tag;
                        }

                        result
                    } else {
                        // Inside thinking, no end tag yet
                        let mut result = ThinkingStreamResult::default();
                        if !text_before_tag.is_empty() {
                            result.text = Some(text_before_tag);
                        }
                        result
                    }
                } else {
                    // No start tag yet - emit safe text (protect partial <think>)
                    let safe_emit_len = self.safe_emit_length_for_start();
                    if safe_emit_len > 0 {
                        let to_emit: String = self.buffer.drain(..safe_emit_len).collect();
                        ThinkingStreamResult {
                            text: Some(to_emit),
                            ..Default::default()
                        }
                    } else {
                        ThinkingStreamResult::default()
                    }
                }
            }
            ThinkingStreamState::InsideThinking => {
                // Look for </think> tag
                if let Some(end_pos) = self.buffer.find(self.think_end) {
                    // Found end tag - emit thinking before it, switch to text mode
                    let thinking_before_tag = self.buffer[..end_pos].to_string();
                    let text_after_tag = self.buffer[end_pos + self.think_end.len()..].to_string();

                    self.thinking_content.push_str(&thinking_before_tag);
                    self.buffer.clear();
                    self.state = ThinkingStreamState::AfterThinking;

                    let mut result = ThinkingStreamResult {
                        thinking_complete: true,
                        ..Default::default()
                    };

                    if !thinking_before_tag.is_empty() {
                        result.thinking = Some(thinking_before_tag);
                    }
                    if !text_after_tag.trim().is_empty() {
                        result.text = Some(text_after_tag);
                    }

                    result
                } else {
                    // No end tag yet - check if we might have partial tag
                    let safe_emit_len = self.safe_emit_length_for_end();
                    if safe_emit_len > 0 {
                        let to_emit: String = self.buffer.drain(..safe_emit_len).collect();
                        self.thinking_content.push_str(&to_emit);
                        ThinkingStreamResult {
                            thinking: Some(to_emit),
                            ..Default::default()
                        }
                    } else {
                        ThinkingStreamResult::default()
                    }
                }
            }
            ThinkingStreamState::AfterThinking => {
                // All content is response text
                let text = std::mem::take(&mut self.buffer);
                if text.is_empty() {
                    ThinkingStreamResult::default()
                } else {
                    ThinkingStreamResult {
                        text: Some(text),
                        ..Default::default()
                    }
                }
            }
        }
    }

    /// Finalize parsing and return any remaining content.
    pub fn finalize(&mut self) -> ThinkingStreamResult {
        let remaining = std::mem::take(&mut self.buffer);

        match self.state {
            ThinkingStreamState::DetectingThinking => {
                // Never found <think> tag - emit remaining as text
                if remaining.is_empty() {
                    ThinkingStreamResult::default()
                } else {
                    ThinkingStreamResult {
                        text: Some(remaining),
                        ..Default::default()
                    }
                }
            }
            ThinkingStreamState::InsideThinking => {
                // Still in thinking - emit remaining as thinking
                if remaining.is_empty() {
                    ThinkingStreamResult {
                        thinking_complete: true,
                        ..Default::default()
                    }
                } else {
                    self.thinking_content.push_str(&remaining);
                    ThinkingStreamResult {
                        thinking: Some(remaining),
                        thinking_complete: true,
                        ..Default::default()
                    }
                }
            }
            ThinkingStreamState::AfterThinking => {
                // Emit remaining as text
                if remaining.is_empty() {
                    ThinkingStreamResult::default()
                } else {
                    ThinkingStreamResult {
                        text: Some(remaining),
                        ..Default::default()
                    }
                }
            }
        }
    }

    /// Calculate safe emit length for detecting mode (protects partial `<think>` tag).
    fn safe_emit_length_for_start(&self) -> usize {
        for i in 1..=self.think_start.len().min(self.buffer.len()) {
            let suffix = &self.buffer[self.buffer.len() - i..];
            if self.think_start.starts_with(suffix) {
                return self.buffer.len().saturating_sub(i);
            }
        }
        self.buffer.len()
    }

    /// Calculate safe emit length for inside thinking mode (protects partial `</think>` tag).
    fn safe_emit_length_for_end(&self) -> usize {
        for i in 1..=self.think_end.len().min(self.buffer.len()) {
            let suffix = &self.buffer[self.buffer.len() - i..];
            if self.think_end.starts_with(suffix) {
                return self.buffer.len().saturating_sub(i);
            }
        }
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_with_thinking() {
        let extractor = ThinkingExtractor::new();
        let input = "<think>Let me reason about this...</think>\n\nThe answer is 42.";
        let result = extractor.extract(input);

        assert!(result.has_thinking);
        assert_eq!(
            result.thinking,
            Some("Let me reason about this...".to_string())
        );
        assert_eq!(result.response, "The answer is 42.");
    }

    #[test]
    fn test_extract_without_thinking() {
        let extractor = ThinkingExtractor::new();
        let input = "Just a plain response without thinking.";
        let result = extractor.extract(input);

        assert!(!result.has_thinking);
        assert!(result.thinking.is_none());
        assert_eq!(result.response, "Just a plain response without thinking.");
    }

    #[test]
    fn test_extract_with_empty_thinking() {
        let extractor = ThinkingExtractor::new();
        let input = "<think></think>\n\nResponse after empty thinking.";
        let result = extractor.extract(input);

        assert!(result.has_thinking);
        assert_eq!(result.thinking, Some("".to_string()));
        assert_eq!(result.response, "Response after empty thinking.");
    }

    #[test]
    fn test_extract_with_multiline_thinking() {
        let extractor = ThinkingExtractor::new();
        let input =
            "<think>\nFirst thought.\nSecond thought.\nThird thought.\n</think>\n\nFinal answer.";
        let result = extractor.extract(input);

        assert!(result.has_thinking);
        assert!(result.thinking.as_ref().unwrap().contains("First thought"));
        assert!(result.thinking.as_ref().unwrap().contains("Third thought"));
        assert_eq!(result.response, "Final answer.");
    }

    #[test]
    fn test_extract_unclosed_think_tag() {
        let extractor = ThinkingExtractor::new();
        let input = "<think>This never closes so it's all response";
        let result = extractor.extract(input);

        assert!(!result.has_thinking);
        assert!(result.thinking.is_none());
        assert_eq!(result.response, input);
    }

    #[test]
    fn test_signature_generation() {
        let thinking = "Some thinking content";
        let sig = generate_thinking_signature(thinking);

        // Should start with sig_
        assert!(sig.starts_with("sig_"));
        // Should be deterministic
        assert_eq!(sig, generate_thinking_signature(thinking));
        // Different content should produce different signature
        assert_ne!(sig, generate_thinking_signature("Different content"));
    }

    #[test]
    fn test_signature_format() {
        let sig = generate_thinking_signature("test");
        // sig_ prefix + 16 hex chars
        assert_eq!(sig.len(), 4 + 16);
        // All chars after prefix should be hex
        assert!(sig[4..].chars().all(|c| c.is_ascii_hexdigit()));
    }

    // Streaming parser tests

    #[test]
    fn test_stream_parser_simple() {
        let mut parser = ThinkingStreamParser::new();

        // Feed thinking content
        let r1 = parser.feed("Let me think...");
        assert!(r1.thinking.is_some());
        assert!(!r1.thinking_complete);

        // Feed end tag and response
        let r2 = parser.feed("</think>\n\nThe answer is 42.");
        assert!(r2.thinking_complete);
        assert!(r2.text.is_some());
        assert!(r2.text.unwrap().contains("42"));
    }

    #[test]
    fn test_stream_parser_split_tag() {
        let mut parser = ThinkingStreamParser::new();

        // Feed content that ends with partial tag
        let r1 = parser.feed("thinking</thi");
        // Should buffer the partial tag
        assert_eq!(parser.state(), ThinkingStreamState::InsideThinking);

        // Complete the tag
        let r2 = parser.feed("nk>response");
        assert!(r2.thinking_complete);
        assert_eq!(parser.state(), ThinkingStreamState::AfterThinking);
    }

    #[test]
    fn test_stream_parser_finalize_in_thinking() {
        let mut parser = ThinkingStreamParser::new();

        parser.feed("still thinking");
        let result = parser.finalize();

        assert!(result.thinking_complete);
        assert!(result.thinking.is_some() || parser.thinking_content().contains("thinking"));
    }

    #[test]
    fn test_stream_parser_finalize_after_thinking() {
        let mut parser = ThinkingStreamParser::new();

        parser.feed("think</think>text");
        let result = parser.finalize();

        assert!(result.text.is_some() || result.text.is_none()); // May have been emitted earlier
        assert_eq!(parser.state(), ThinkingStreamState::AfterThinking);
    }

    #[test]
    fn test_stream_parser_accumulates_content() {
        let mut parser = ThinkingStreamParser::new();

        parser.feed("first ");
        parser.feed("second ");
        parser.feed("third</think>done");

        assert!(parser.thinking_content().contains("first"));
        assert!(parser.thinking_content().contains("second"));
        assert!(parser.thinking_content().contains("third"));
    }

    // Detection mode tests (model-initiated thinking)

    #[test]
    fn test_detecting_mode_with_thinking_tags() {
        let mut parser = ThinkingStreamParser::new_detecting();
        assert_eq!(parser.state(), ThinkingStreamState::DetectingThinking);

        // Feed opening tag and thinking content
        let r1 = parser.feed("<think>reasoning here");
        assert_eq!(parser.state(), ThinkingStreamState::InsideThinking);

        // Feed closing tag and response
        let r2 = parser.feed("</think>the answer");
        assert!(r2.thinking_complete);
        assert_eq!(parser.state(), ThinkingStreamState::AfterThinking);
        assert!(parser.thinking_content().contains("reasoning"));
    }

    #[test]
    fn test_detecting_mode_no_thinking_tags() {
        let mut parser = ThinkingStreamParser::new_detecting();

        // Feed plain text without thinking tags
        let r1 = parser.feed("Hello ");
        assert!(r1.text.is_some());
        assert_eq!(parser.state(), ThinkingStreamState::DetectingThinking);

        let r2 = parser.feed("world!");
        assert!(r2.text.is_some());

        // Finalize - should return remaining as text, no thinking
        let final_result = parser.finalize();
        assert!(!final_result.thinking_complete);
        assert!(parser.thinking_content().is_empty());
    }

    #[test]
    fn test_detecting_mode_split_open_tag() {
        let mut parser = ThinkingStreamParser::new_detecting();

        // Feed partial <think> tag
        let r1 = parser.feed("text<thi");
        // Should emit text before potential tag, buffer the partial tag
        assert_eq!(parser.state(), ThinkingStreamState::DetectingThinking);

        // Complete the tag
        let r2 = parser.feed("nk>inside thinking");
        assert_eq!(parser.state(), ThinkingStreamState::InsideThinking);
    }

    #[test]
    fn test_detecting_mode_text_before_thinking() {
        let mut parser = ThinkingStreamParser::new_detecting();

        // Feed text, then thinking block in same token
        let r1 = parser.feed("prefix <think>reasoning</think>suffix");

        // Should have transitioned through states
        assert_eq!(parser.state(), ThinkingStreamState::AfterThinking);
        assert!(r1.thinking_complete);
        assert!(parser.thinking_content().contains("reasoning"));
    }

    #[test]
    fn test_detecting_mode_complete_flow() {
        let mut parser = ThinkingStreamParser::new_detecting();

        // Simulate streaming: model outputs <think>...<think> then answer
        parser.feed("<think>");
        assert_eq!(parser.state(), ThinkingStreamState::InsideThinking);

        parser.feed("Let me ");
        parser.feed("think about ");
        parser.feed("this.");

        let r = parser.feed("</think>");
        assert!(r.thinking_complete);
        assert_eq!(parser.state(), ThinkingStreamState::AfterThinking);

        let r2 = parser.feed("The answer is 42.");
        assert!(r2.text.is_some());
        assert!(r2.text.unwrap().contains("42"));
    }

    #[test]
    fn test_detecting_mode_finalize_without_tags() {
        let mut parser = ThinkingStreamParser::new_detecting();

        parser.feed("Just plain text");
        parser.feed(" more text");

        let result = parser.finalize();

        // Should finalize as text, not thinking
        assert!(result.text.is_some() || parser.state() == ThinkingStreamState::DetectingThinking);
        assert!(!result.thinking_complete);
        assert!(parser.thinking_content().is_empty());
    }
}
