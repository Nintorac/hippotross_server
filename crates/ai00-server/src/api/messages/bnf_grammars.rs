//! Stage 1 structural BNF grammar constants for constrained decoding.
//!
//! These grammars enforce correct syntax for thinking tags, tool calls, and JSON
//! without requiring schema-specific validation (that's Stage 2).
//!
//! Grammar format: KBNF (Koishi's BNF) - see spec Part 8.2 for syntax reference.

/// Generic JSON grammar primitives.
///
/// Provides basic JSON value parsing:
/// - Objects: `{ "key": value, ... }`
/// - Arrays: `[ value, ... ]`
/// - Strings: `"..."` with escape handling
/// - Numbers: integers, decimals, scientific notation
/// - Booleans: `true`, `false`
/// - Null: `null`
///
/// These rules are designed to be composed with other grammars.
pub const GRAMMAR_JSON_PRIMITIVES: &str = r#"
json_object::='{' ws members ws '}';
members::=pair (',' ws pair)*;
pair::=string ws ':' ws json_value;
json_value::=string | number | json_object | json_array | 'true' | 'false' | 'null';
json_array::='[' ws elements ws ']';
elements::=json_value (',' ws json_value)*;
string::='"' string_content '"';
string_content::=#'[^"\\\\]*(\\\\.[^"\\\\]*)*';
number::=#'-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?';
ws::=#'[ \\t\\n\\r]*';
"#;

/// Thinking-only grammar for extended thinking mode.
///
/// Enforces the structure:
/// - Optional `<think>...</think>` block at the start
/// - Followed by a response
///
/// The thinking content allows any characters except the closing tag,
/// enabling free-form reasoning inside the thinking block.
pub const GRAMMAR_THINKING_ONLY: &str = r#"
start::=thinking_response | plain_response;
thinking_response::='<think>' thinking_content '</think>' ws response;
thinking_content::=#'[^<]*';
plain_response::=response;
response::=#'[^\\x00]*';
"#;

/// Tool call structure grammar without schema validation.
///
/// Enforces the structure:
/// - Optional text before/after tool calls
/// - Tool calls wrapped in `<tool_use>...</tool_use>` tags
/// - Tool JSON must have `name` and `input` fields
///
/// Note: This grammar validates structure only, not the actual tool names
/// or input schemas. Use SchemaAware level for full validation.
pub const GRAMMAR_TOOLS_ONLY: &str = r#"
start::=text_or_tools;
text_or_tools::=text? tool_sequence?;
tool_sequence::=tool_use text? tool_sequence?;
tool_use::='<tool_use>' ws tool_json ws '</tool_use>';
tool_json::='{' ws '"name"' ws ':' ws string ws ',' ws '"input"' ws ':' ws json_object ws '}';
text::=#'[^<]*';
"#;

/// Combined grammar for thinking blocks with tool calls.
///
/// Enforces the structure:
/// - Optional `<think>...</think>` block at the start
/// - Followed by optional text and tool calls
///
/// This grammar is used when both extended thinking and tools are enabled.
pub const GRAMMAR_THINKING_PLUS_TOOLS: &str = r#"
start::=thinking_block? text_or_tools;
thinking_block::='<think>' thinking_content '</think>' ws;
thinking_content::=#'[^<]*';
text_or_tools::=text? tool_sequence?;
tool_sequence::=tool_use text? tool_sequence?;
tool_use::='<tool_use>' ws tool_json ws '</tool_use>';
tool_json::='{' ws '"name"' ws ':' ws string ws ',' ws '"input"' ws ':' ws json_object ws '}';
text::=#'[^<]*';
"#;

/// Thinking wrapper for user-provided grammars.
///
/// When the user provides a custom `bnf_schema` but thinking is also enabled,
/// this wrapper adds thinking tag support around their grammar.
///
/// Usage: Prepend to user's grammar where their start rule becomes `user_start`.
pub const GRAMMAR_THINKING_WRAPPER: &str = r#"
start::=thinking_block? user_start;
thinking_block::='<think>' thinking_content '</think>' ws;
thinking_content::=#'[^<]*';
ws::=#'[ \\t\\n\\r]*';
"#;

/// Build a complete structural grammar based on features enabled.
///
/// Combines the appropriate grammar constants based on what's needed:
/// - Base JSON primitives are always included
/// - Thinking grammar when thinking is enabled
/// - Tools grammar when tools are present
/// - Combined grammar when both are enabled
pub fn build_structural_grammar(thinking_enabled: bool, tools_present: bool) -> String {
    let mut grammar = String::new();

    // Always include JSON primitives as base
    grammar.push_str(GRAMMAR_JSON_PRIMITIVES);
    grammar.push('\n');

    // Add feature-specific grammar
    match (thinking_enabled, tools_present) {
        (true, true) => grammar.push_str(GRAMMAR_THINKING_PLUS_TOOLS),
        (true, false) => grammar.push_str(GRAMMAR_THINKING_ONLY),
        (false, true) => grammar.push_str(GRAMMAR_TOOLS_ONLY),
        (false, false) => {
            // No special structure needed - just allow any text
            grammar.push_str("start::=#'[^\\x00]*';\n");
        }
    }

    grammar
}

/// Wrap a user-provided grammar with thinking support.
///
/// Renames the user's `start` rule to `user_start` and prepends
/// the thinking wrapper to allow optional thinking blocks.
///
/// # Arguments
/// * `user_grammar` - The user's custom BNF grammar
///
/// # Returns
/// A new grammar with thinking tag support prepended
pub fn wrap_grammar_with_thinking(user_grammar: &str) -> String {
    let mut wrapped = String::new();

    // Add thinking wrapper (defines new start rule)
    wrapped.push_str(GRAMMAR_THINKING_WRAPPER);
    wrapped.push('\n');

    // Rename user's start rule to user_start
    // This is a simple text replacement - production code might need
    // a proper parser for complex grammars
    let renamed = user_grammar
        .replace("start::=", "user_start::=")
        .replace("start ::=", "user_start::=");
    wrapped.push_str(&renamed);

    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_json_primitives_not_empty() {
        assert!(!GRAMMAR_JSON_PRIMITIVES.is_empty());
        assert!(GRAMMAR_JSON_PRIMITIVES.contains("json_object"));
        assert!(GRAMMAR_JSON_PRIMITIVES.contains("json_value"));
        assert!(GRAMMAR_JSON_PRIMITIVES.contains("string"));
        assert!(GRAMMAR_JSON_PRIMITIVES.contains("number"));
    }

    #[test]
    fn test_grammar_thinking_only_structure() {
        assert!(GRAMMAR_THINKING_ONLY.contains("start::="));
        assert!(GRAMMAR_THINKING_ONLY.contains("<think>"));
        assert!(GRAMMAR_THINKING_ONLY.contains("</think>"));
        assert!(GRAMMAR_THINKING_ONLY.contains("thinking_content"));
    }

    #[test]
    fn test_grammar_tools_only_structure() {
        assert!(GRAMMAR_TOOLS_ONLY.contains("start::="));
        assert!(GRAMMAR_TOOLS_ONLY.contains("<tool_use>"));
        assert!(GRAMMAR_TOOLS_ONLY.contains("</tool_use>"));
        assert!(GRAMMAR_TOOLS_ONLY.contains("tool_json"));
        // Check for JSON field names in the grammar
        assert!(GRAMMAR_TOOLS_ONLY.contains(r#"'"name"'"#));
        assert!(GRAMMAR_TOOLS_ONLY.contains(r#"'"input"'"#));
    }

    #[test]
    fn test_grammar_thinking_plus_tools_structure() {
        assert!(GRAMMAR_THINKING_PLUS_TOOLS.contains("start::="));
        assert!(GRAMMAR_THINKING_PLUS_TOOLS.contains("<think>"));
        assert!(GRAMMAR_THINKING_PLUS_TOOLS.contains("</think>"));
        assert!(GRAMMAR_THINKING_PLUS_TOOLS.contains("<tool_use>"));
        assert!(GRAMMAR_THINKING_PLUS_TOOLS.contains("</tool_use>"));
        assert!(GRAMMAR_THINKING_PLUS_TOOLS.contains("thinking_block"));
    }

    #[test]
    fn test_build_structural_grammar_thinking_only() {
        let grammar = build_structural_grammar(true, false);
        assert!(grammar.contains("json_object")); // From primitives
        assert!(grammar.contains("<think>")); // From thinking
        assert!(!grammar.contains("<tool_use>")); // No tools
    }

    #[test]
    fn test_build_structural_grammar_tools_only() {
        let grammar = build_structural_grammar(false, true);
        assert!(grammar.contains("json_object")); // From primitives
        assert!(grammar.contains("<tool_use>")); // From tools
        assert!(!grammar.contains("<think>")); // No thinking
    }

    #[test]
    fn test_build_structural_grammar_both() {
        let grammar = build_structural_grammar(true, true);
        assert!(grammar.contains("json_object")); // From primitives
        assert!(grammar.contains("<think>")); // From combined
        assert!(grammar.contains("<tool_use>")); // From combined
    }

    #[test]
    fn test_build_structural_grammar_neither() {
        let grammar = build_structural_grammar(false, false);
        assert!(grammar.contains("json_object")); // From primitives
        assert!(grammar.contains("start::=")); // Has start rule
        assert!(!grammar.contains("<think>"));
        assert!(!grammar.contains("<tool_use>"));
    }

    #[test]
    fn test_wrap_grammar_with_thinking() {
        let user_grammar = r#"start::=greeting;
greeting::='Hello' | 'Hi';"#;

        let wrapped = wrap_grammar_with_thinking(user_grammar);

        // Should have new start rule from wrapper
        assert!(wrapped.contains("start::=thinking_block? user_start"));
        // User's start should be renamed
        assert!(wrapped.contains("user_start::=greeting"));
        // User's other rules preserved
        assert!(wrapped.contains("greeting::="));
    }

    #[test]
    fn test_grammar_constants_have_valid_kbnf_syntax() {
        // Basic syntax checks - ensure grammars follow KBNF patterns
        let grammars = [
            GRAMMAR_JSON_PRIMITIVES,
            GRAMMAR_THINKING_ONLY,
            GRAMMAR_TOOLS_ONLY,
            GRAMMAR_THINKING_PLUS_TOOLS,
            GRAMMAR_THINKING_WRAPPER,
        ];

        for grammar in grammars {
            // Each grammar should have at least one rule definition
            assert!(
                grammar.contains("::="),
                "Grammar must contain rule definitions"
            );

            // Rules should end with semicolons
            assert!(grammar.contains(';'), "Grammar rules must end with semicolons");

            // Check balanced quotes in literals
            let double_quote_count = grammar.chars().filter(|c| *c == '"').count();
            assert!(
                double_quote_count % 2 == 0,
                "Unbalanced double quotes in grammar"
            );
        }
    }
}
