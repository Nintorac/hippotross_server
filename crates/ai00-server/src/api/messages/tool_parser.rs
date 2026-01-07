//! Incremental streaming parser for tool_call tags in model output.
//!
//! Parses Hermes/Qwen-style `<tool_call>...</tool_call>` blocks containing
//! JSON function calls.

use serde::Deserialize;
use serde_json::Value;

/// A parsed tool call from the model output.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallJson {
    /// Function name
    pub name: String,
    /// Function arguments as JSON
    #[serde(default)]
    pub arguments: Value,
}

/// A fully parsed tool use with generated ID.
#[derive(Debug, Clone)]
pub struct ParsedToolUse {
    /// Generated tool use ID (e.g., "toolu_01abc...")
    pub id: String,
    /// Tool name
    pub name: String,
    /// Tool input as JSON
    pub input: Value,
}

/// State machine for parsing tool_call tags incrementally.
#[derive(Debug, Default)]
pub struct ToolParser {
    /// Current parser state
    state: ParserState,
    /// Buffer for accumulating tag names
    tag_buffer: String,
    /// Buffer for accumulating JSON content inside tool_call
    json_buffer: String,
    /// Completed tool calls
    completed_tools: Vec<ParsedToolUse>,
    /// Text content outside tool calls
    text_buffer: String,
    /// Index for generating tool use IDs
    tool_index: usize,
}

/// Parser state machine states.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum ParserState {
    /// Normal text content
    #[default]
    Text,
    /// Saw '<', might be start of tag
    MaybeTagStart,
    /// Inside opening tag name
    OpenTagName,
    /// Inside closing tag (saw '</')
    CloseTagName,
    /// Inside <tool_call> block, accumulating JSON
    InToolCall,
}

/// Result of feeding a token to the parser.
#[derive(Debug, Default)]
pub struct ParseResult {
    /// Text content to emit (may be empty)
    pub text: Option<String>,
    /// Completed tool uses to emit
    pub tool_uses: Vec<ParsedToolUse>,
    /// Whether we're currently inside a tool_call block
    pub in_tool_block: bool,
}

impl ToolParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a token to the parser and get parse results.
    pub fn feed(&mut self, token: &str) -> ParseResult {
        for ch in token.chars() {
            self.process_char(ch);
        }

        let mut result = ParseResult::default();

        // Emit accumulated text if not in tool block
        if !self.text_buffer.is_empty() && self.state == ParserState::Text {
            result.text = Some(std::mem::take(&mut self.text_buffer));
        }

        // Emit completed tools
        result.tool_uses = std::mem::take(&mut self.completed_tools);
        result.in_tool_block = self.state == ParserState::InToolCall;

        result
    }

    /// Process a single character through the state machine.
    fn process_char(&mut self, ch: char) {
        match self.state {
            ParserState::Text => {
                if ch == '<' {
                    self.state = ParserState::MaybeTagStart;
                } else {
                    self.text_buffer.push(ch);
                }
            }

            ParserState::MaybeTagStart => {
                if ch == '/' {
                    self.state = ParserState::CloseTagName;
                    self.tag_buffer.clear();
                } else if ch.is_alphabetic() || ch == '_' {
                    self.state = ParserState::OpenTagName;
                    self.tag_buffer.clear();
                    self.tag_buffer.push(ch);
                } else {
                    // Not a valid tag, emit the '<' and continue
                    self.text_buffer.push('<');
                    self.text_buffer.push(ch);
                    self.state = ParserState::Text;
                }
            }

            ParserState::OpenTagName => {
                if ch == '>' {
                    if self.tag_buffer == "tool_call" {
                        // Enter tool call mode
                        self.state = ParserState::InToolCall;
                        self.json_buffer.clear();
                    } else {
                        // Unknown tag, emit as text
                        self.text_buffer.push('<');
                        self.text_buffer.push_str(&self.tag_buffer);
                        self.text_buffer.push('>');
                        self.state = ParserState::Text;
                    }
                    self.tag_buffer.clear();
                } else if ch.is_alphanumeric() || ch == '_' {
                    self.tag_buffer.push(ch);
                } else {
                    // Invalid tag character, treat as text
                    self.text_buffer.push('<');
                    self.text_buffer.push_str(&self.tag_buffer);
                    self.text_buffer.push(ch);
                    self.tag_buffer.clear();
                    self.state = ParserState::Text;
                }
            }

            ParserState::CloseTagName => {
                if ch == '>' {
                    if self.tag_buffer == "tool_call" {
                        // End of tool call - parse the JSON
                        self.complete_tool_call();
                        self.state = ParserState::Text;
                    } else {
                        // Unknown closing tag, emit as text
                        self.text_buffer.push_str("</");
                        self.text_buffer.push_str(&self.tag_buffer);
                        self.text_buffer.push('>');
                        self.state = ParserState::Text;
                    }
                    self.tag_buffer.clear();
                } else if ch.is_alphanumeric() || ch == '_' {
                    self.tag_buffer.push(ch);
                } else {
                    // Invalid closing tag
                    self.text_buffer.push_str("</");
                    self.text_buffer.push_str(&self.tag_buffer);
                    self.text_buffer.push(ch);
                    self.tag_buffer.clear();
                    self.state = ParserState::Text;
                }
            }

            ParserState::InToolCall => {
                if ch == '<' {
                    self.state = ParserState::MaybeTagStart;
                } else {
                    self.json_buffer.push(ch);
                }
            }
        }
    }

    /// Parse accumulated JSON and create a tool use.
    fn complete_tool_call(&mut self) {
        let json_str = self.json_buffer.trim();
        if json_str.is_empty() {
            self.json_buffer.clear();
            return;
        }

        // Try to parse the JSON
        if let Ok(call) = serde_json::from_str::<ToolCallJson>(json_str) {
            let id = format!("toolu_{:012x}", self.tool_index);
            self.tool_index += 1;

            self.completed_tools.push(ParsedToolUse {
                id,
                name: call.name,
                input: call.arguments,
            });
        }

        self.json_buffer.clear();
    }

    /// Finalize parsing and return any remaining content.
    pub fn finalize(&mut self) -> ParseResult {
        let mut result = ParseResult::default();

        // Emit any remaining text
        if !self.text_buffer.is_empty() {
            result.text = Some(std::mem::take(&mut self.text_buffer));
        }

        // Emit any completed tools
        result.tool_uses = std::mem::take(&mut self.completed_tools);

        result
    }

    /// Check if the parser has detected any tool use in the stream.
    pub fn has_tool_use(&self) -> bool {
        self.tool_index > 0
    }

    /// Get the total number of completed tool uses.
    pub fn tool_count(&self) -> usize {
        self.tool_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_text() {
        let mut parser = ToolParser::new();
        let result = parser.feed("Hello, world!");
        assert_eq!(result.text, Some("Hello, world!".to_string()));
        assert!(result.tool_uses.is_empty());
        assert!(!result.in_tool_block);
    }

    #[test]
    fn test_simple_tool_call() {
        let mut parser = ToolParser::new();

        let result = parser.feed(
            r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>"#,
        );

        assert_eq!(result.tool_uses.len(), 1);
        let tool = &result.tool_uses[0];
        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.input["location"], "NYC");
        assert!(tool.id.starts_with("toolu_"));
    }

    #[test]
    fn test_text_before_tool() {
        let mut parser = ToolParser::new();

        let result1 = parser.feed("Let me check the weather. ");
        assert_eq!(result1.text, Some("Let me check the weather. ".to_string()));

        let result2 = parser.feed(
            r#"<tool_call>
{"name": "weather", "arguments": {}}
</tool_call>"#,
        );
        assert_eq!(result2.tool_uses.len(), 1);
        assert_eq!(result2.tool_uses[0].name, "weather");
    }

    #[test]
    fn test_multiple_tools() {
        let mut parser = ToolParser::new();

        let result = parser.feed(
            r#"<tool_call>
{"name": "tool1", "arguments": {}}
</tool_call>
<tool_call>
{"name": "tool2", "arguments": {"x": 1}}
</tool_call>"#,
        );

        assert_eq!(result.tool_uses.len(), 2);
        assert_eq!(result.tool_uses[0].name, "tool1");
        assert_eq!(result.tool_uses[1].name, "tool2");
        assert_eq!(result.tool_uses[1].input["x"], 1);
    }

    #[test]
    fn test_partial_streaming() {
        let mut parser = ToolParser::new();
        let mut all_tools = Vec::new();

        // Simulate streaming tokens
        let tokens = [
            "<tool",
            "_call>",
            r#"{"name": "#,
            r#""test", "#,
            r#""arguments": {}}"#,
            "</tool_call>",
        ];

        for token in tokens {
            let result = parser.feed(token);
            all_tools.extend(result.tool_uses);
        }

        let final_result = parser.finalize();
        all_tools.extend(final_result.tool_uses);

        assert_eq!(all_tools.len(), 1);
        assert_eq!(all_tools[0].name, "test");
    }

    #[test]
    fn test_has_tool_use() {
        let mut parser = ToolParser::new();
        assert!(!parser.has_tool_use());

        parser.feed(r#"<tool_call>{"name": "x", "arguments": {}}</tool_call>"#);
        assert!(parser.has_tool_use());
        assert_eq!(parser.tool_count(), 1);
    }

    #[test]
    fn test_text_mixed_with_tools() {
        let mut parser = ToolParser::new();

        let result1 = parser.feed("First I'll search. ");
        assert_eq!(result1.text, Some("First I'll search. ".to_string()));

        let result2 = parser.feed(r#"<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>"#);
        assert_eq!(result2.tool_uses.len(), 1);

        let result3 = parser.feed(" Then calculate. ");
        assert_eq!(result3.text, Some(" Then calculate. ".to_string()));

        let result4 = parser.feed(r#"<tool_call>{"name": "calc", "arguments": {}}</tool_call>"#);
        assert_eq!(result4.tool_uses.len(), 1);

        assert_eq!(parser.tool_count(), 2);
    }
}
