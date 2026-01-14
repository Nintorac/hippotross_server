//! Incremental streaming parsers for tool call tags in model output.
//!
//! Contains two parsers:
//! - `ToolParser`: Legacy parser for Hermes/Qwen-style `<tool_call>` blocks
//! - `Ai00FunctionCallsParser`: Parser for ai00 v1 `<ai00:function_calls>` format

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

// =============================================================================
// ai00 v1 Function Calls Parser
// =============================================================================

/// State machine for parsing ai00:function_calls tags incrementally.
///
/// Parses the ai00 v1 tool calling format:
/// ```xml
/// <ai00:function_calls>
///   <invoke name="tool_name">
///     <parameter name="param1">value1</parameter>
///     <parameter name="param2">value2</parameter>
///   </invoke>
/// </ai00:function_calls>
/// ```
#[derive(Debug, Default)]
pub struct Ai00FunctionCallsParser {
    /// Current parser state
    state: Ai00ParserState,
    /// Buffer for accumulating tag content
    tag_buffer: String,
    /// Buffer for current invoke name
    current_invoke_name: String,
    /// Buffer for current parameter name
    current_param_name: String,
    /// Buffer for current parameter value
    current_param_value: String,
    /// Parameters collected for current invoke
    current_params: serde_json::Map<String, Value>,
    /// Completed tool calls
    completed_tools: Vec<ParsedToolUse>,
    /// Text content outside function_calls blocks
    text_buffer: String,
    /// Index for generating tool use IDs
    tool_index: usize,
    /// Depth tracker for nested tags
    in_function_calls: bool,
}

/// Parser state machine states for ai00 format.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
enum Ai00ParserState {
    /// Normal text content
    #[default]
    Text,
    /// Saw '<', might be start of tag
    MaybeTagStart,
    /// Inside opening tag name (collecting tag name)
    InOpenTag,
    /// Inside closing tag (saw '</')
    InCloseTag,
    /// Inside attribute name (after tag name, before '=')
    InAttrName,
    /// Inside attribute value (after '="')
    InAttrValue,
    /// Inside parameter value (between > and </parameter>)
    InParamValue,
}

impl Ai00FunctionCallsParser {
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

        // Emit accumulated text if not in function_calls block
        if !self.text_buffer.is_empty() && !self.in_function_calls {
            result.text = Some(std::mem::take(&mut self.text_buffer));
        }

        // Emit completed tools
        result.tool_uses = std::mem::take(&mut self.completed_tools);
        result.in_tool_block = self.in_function_calls;

        result
    }

    /// Process a single character through the state machine.
    fn process_char(&mut self, ch: char) {
        match &self.state {
            Ai00ParserState::Text => {
                if ch == '<' {
                    self.state = Ai00ParserState::MaybeTagStart;
                    self.tag_buffer.clear();
                } else if !self.in_function_calls {
                    self.text_buffer.push(ch);
                }
            }

            Ai00ParserState::MaybeTagStart => {
                if ch == '/' {
                    self.state = Ai00ParserState::InCloseTag;
                    self.tag_buffer.clear();
                } else if ch.is_alphabetic() || ch == '_' || ch == ':' {
                    self.state = Ai00ParserState::InOpenTag;
                    self.tag_buffer.clear();
                    self.tag_buffer.push(ch);
                } else {
                    // Not a valid tag start
                    if !self.in_function_calls {
                        self.text_buffer.push('<');
                        self.text_buffer.push(ch);
                    }
                    self.state = Ai00ParserState::Text;
                }
            }

            Ai00ParserState::InOpenTag => {
                if ch == '>' {
                    self.handle_open_tag_complete();
                    self.state = if self.tag_buffer == "parameter" {
                        Ai00ParserState::InParamValue
                    } else {
                        Ai00ParserState::Text
                    };
                } else if ch.is_whitespace() {
                    // May have attributes
                    if self.tag_buffer == "invoke" || self.tag_buffer == "parameter" {
                        self.state = Ai00ParserState::InAttrName;
                    }
                } else if ch.is_alphanumeric() || ch == '_' || ch == ':' || ch == '-' {
                    self.tag_buffer.push(ch);
                } else {
                    // Invalid character, treat as text
                    if !self.in_function_calls {
                        self.text_buffer.push('<');
                        self.text_buffer.push_str(&self.tag_buffer);
                        self.text_buffer.push(ch);
                    }
                    self.tag_buffer.clear();
                    self.state = Ai00ParserState::Text;
                }
            }

            Ai00ParserState::InCloseTag => {
                if ch == '>' {
                    self.handle_close_tag();
                    self.tag_buffer.clear();
                    self.state = Ai00ParserState::Text;
                } else if ch.is_alphanumeric() || ch == '_' || ch == ':' || ch == '-' {
                    self.tag_buffer.push(ch);
                } else {
                    // Invalid closing tag
                    if !self.in_function_calls {
                        self.text_buffer.push_str("</");
                        self.text_buffer.push_str(&self.tag_buffer);
                        self.text_buffer.push(ch);
                    }
                    self.tag_buffer.clear();
                    self.state = Ai00ParserState::Text;
                }
            }

            Ai00ParserState::InAttrName => {
                if ch == '=' {
                    // Attribute name complete, wait for value
                } else if ch == '"' {
                    self.state = Ai00ParserState::InAttrValue;
                } else if ch == '>' {
                    self.handle_open_tag_complete();
                    self.state = if self.tag_buffer == "parameter" {
                        Ai00ParserState::InParamValue
                    } else {
                        Ai00ParserState::Text
                    };
                } else if ch.is_whitespace() {
                    // Skip whitespace
                } else if ch.is_alphanumeric() || ch == '_' {
                    // Part of attribute name (we only care about "name" attr)
                }
            }

            Ai00ParserState::InAttrValue => {
                if ch == '"' {
                    // Attribute value complete
                    if self.tag_buffer == "invoke" {
                        // Store invoke name
                        self.current_invoke_name = std::mem::take(&mut self.current_param_name);
                        self.current_params.clear();
                    } else if self.tag_buffer == "parameter" {
                        // Store parameter name
                        // current_param_name already set
                    }
                    self.state = Ai00ParserState::InAttrName;
                } else {
                    // Accumulate attribute value
                    if self.tag_buffer == "invoke" {
                        self.current_param_name.push(ch);
                    } else if self.tag_buffer == "parameter" {
                        self.current_param_name.push(ch);
                    }
                }
            }

            Ai00ParserState::InParamValue => {
                if ch == '<' {
                    // Might be closing tag
                    self.state = Ai00ParserState::MaybeTagStart;
                } else {
                    self.current_param_value.push(ch);
                }
            }
        }
    }

    /// Handle completion of an opening tag.
    fn handle_open_tag_complete(&mut self) {
        match self.tag_buffer.as_str() {
            "ai00:function_calls" => {
                self.in_function_calls = true;
            }
            "invoke" => {
                // invoke name should be set from attribute
            }
            "parameter" => {
                self.current_param_value.clear();
            }
            _ => {}
        }
    }

    /// Handle a closing tag.
    fn handle_close_tag(&mut self) {
        match self.tag_buffer.as_str() {
            "ai00:function_calls" => {
                self.in_function_calls = false;
            }
            "invoke" => {
                // Complete this invoke as a tool call
                if !self.current_invoke_name.is_empty() {
                    let id = format!("toolu_{:012x}", self.tool_index);
                    self.tool_index += 1;

                    self.completed_tools.push(ParsedToolUse {
                        id,
                        name: std::mem::take(&mut self.current_invoke_name),
                        input: Value::Object(std::mem::take(&mut self.current_params)),
                    });
                }
            }
            "parameter" => {
                // Add parameter to current params
                if !self.current_param_name.is_empty() {
                    let name = std::mem::take(&mut self.current_param_name);
                    let value = std::mem::take(&mut self.current_param_value).trim().to_string();

                    // Try to parse value as JSON, otherwise use as string
                    let json_value = if let Ok(v) = serde_json::from_str::<Value>(&value) {
                        v
                    } else {
                        Value::String(value)
                    };

                    self.current_params.insert(name, json_value);
                }
            }
            _ => {}
        }
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

    /// Check if we're currently inside a function_calls block.
    pub fn in_function_calls(&self) -> bool {
        self.in_function_calls
    }
}

#[cfg(test)]
mod ai00_parser_tests {
    use super::*;

    #[test]
    fn test_ai00_plain_text() {
        let mut parser = Ai00FunctionCallsParser::new();
        let result = parser.feed("Hello, world!");
        assert_eq!(result.text, Some("Hello, world!".to_string()));
        assert!(result.tool_uses.is_empty());
        assert!(!result.in_tool_block);
    }

    #[test]
    fn test_ai00_simple_function_call() {
        let mut parser = Ai00FunctionCallsParser::new();

        let result = parser.feed(
            r#"<ai00:function_calls>
  <invoke name="get_weather">
    <parameter name="city">Tokyo</parameter>
  </invoke>
</ai00:function_calls>"#,
        );

        assert_eq!(result.tool_uses.len(), 1);
        let tool = &result.tool_uses[0];
        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.input["city"], "Tokyo");
        assert!(tool.id.starts_with("toolu_"));
    }

    #[test]
    fn test_ai00_multiple_parameters() {
        let mut parser = Ai00FunctionCallsParser::new();

        let result = parser.feed(
            r#"<ai00:function_calls>
  <invoke name="get_weather">
    <parameter name="city">Tokyo</parameter>
    <parameter name="unit">celsius</parameter>
  </invoke>
</ai00:function_calls>"#,
        );

        assert_eq!(result.tool_uses.len(), 1);
        let tool = &result.tool_uses[0];
        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.input["city"], "Tokyo");
        assert_eq!(tool.input["unit"], "celsius");
    }

    #[test]
    fn test_ai00_multiple_invokes() {
        let mut parser = Ai00FunctionCallsParser::new();

        let result = parser.feed(
            r#"<ai00:function_calls>
  <invoke name="search">
    <parameter name="query">weather Tokyo</parameter>
  </invoke>
  <invoke name="translate">
    <parameter name="text">hello</parameter>
    <parameter name="to">ja</parameter>
  </invoke>
</ai00:function_calls>"#,
        );

        assert_eq!(result.tool_uses.len(), 2);
        assert_eq!(result.tool_uses[0].name, "search");
        assert_eq!(result.tool_uses[0].input["query"], "weather Tokyo");
        assert_eq!(result.tool_uses[1].name, "translate");
        assert_eq!(result.tool_uses[1].input["text"], "hello");
        assert_eq!(result.tool_uses[1].input["to"], "ja");
    }

    #[test]
    fn test_ai00_text_before_function_call() {
        let mut parser = Ai00FunctionCallsParser::new();

        let result1 = parser.feed("Let me check that for you.\n");
        assert_eq!(result1.text, Some("Let me check that for you.\n".to_string()));

        let result2 = parser.feed(
            r#"<ai00:function_calls>
  <invoke name="check">
    <parameter name="item">status</parameter>
  </invoke>
</ai00:function_calls>"#,
        );
        assert_eq!(result2.tool_uses.len(), 1);
        assert_eq!(result2.tool_uses[0].name, "check");
    }

    #[test]
    fn test_ai00_streaming_partial() {
        let mut parser = Ai00FunctionCallsParser::new();
        let mut all_tools = Vec::new();

        // Simulate streaming tokens
        let tokens = [
            "<ai00:function",
            "_calls>\n  <invoke name=\"",
            "test\">",
            "\n    <parameter name=\"arg\">",
            "value</",
            "parameter>\n  </invoke>",
            "\n</ai00:function_calls>",
        ];

        for token in tokens {
            let result = parser.feed(token);
            all_tools.extend(result.tool_uses);
        }

        let final_result = parser.finalize();
        all_tools.extend(final_result.tool_uses);

        assert_eq!(all_tools.len(), 1);
        assert_eq!(all_tools[0].name, "test");
        assert_eq!(all_tools[0].input["arg"], "value");
    }

    #[test]
    fn test_ai00_numeric_parameter() {
        let mut parser = Ai00FunctionCallsParser::new();

        let result = parser.feed(
            r#"<ai00:function_calls>
  <invoke name="calc">
    <parameter name="x">42</parameter>
    <parameter name="y">3.14</parameter>
  </invoke>
</ai00:function_calls>"#,
        );

        assert_eq!(result.tool_uses.len(), 1);
        let tool = &result.tool_uses[0];
        // Numbers should be parsed as JSON numbers
        assert_eq!(tool.input["x"], 42);
        assert_eq!(tool.input["y"], 3.14);
    }

    #[test]
    fn test_ai00_json_parameter() {
        let mut parser = Ai00FunctionCallsParser::new();

        let result = parser.feed(
            r#"<ai00:function_calls>
  <invoke name="process">
    <parameter name="data">{"key": "value", "num": 123}</parameter>
  </invoke>
</ai00:function_calls>"#,
        );

        assert_eq!(result.tool_uses.len(), 1);
        let tool = &result.tool_uses[0];
        assert_eq!(tool.input["data"]["key"], "value");
        assert_eq!(tool.input["data"]["num"], 123);
    }

    #[test]
    fn test_ai00_has_tool_use() {
        let mut parser = Ai00FunctionCallsParser::new();
        assert!(!parser.has_tool_use());

        parser.feed(
            r#"<ai00:function_calls>
  <invoke name="x">
    <parameter name="a">b</parameter>
  </invoke>
</ai00:function_calls>"#,
        );
        assert!(parser.has_tool_use());
        assert_eq!(parser.tool_count(), 1);
    }
}
