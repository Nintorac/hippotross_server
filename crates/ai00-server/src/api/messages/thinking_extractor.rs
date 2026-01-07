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
        let input = "<think>\nFirst thought.\nSecond thought.\nThird thought.\n</think>\n\nFinal answer.";
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
}
