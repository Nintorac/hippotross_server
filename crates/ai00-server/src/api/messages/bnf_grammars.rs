//! Unified structural BNF grammar for constrained decoding.
//!
//! This module provides a single unified grammar that handles all combinations
//! of thinking blocks, tool calls, and plain text responses. The grammar always
//! allows optional thinking, preventing conflicts when models spontaneously decide
//! to "think" regardless of configuration.
//!
//! Grammar format: KBNF (Koishi's BNF) - see spec Part 8.2 for syntax reference.
//!
//! ## Key Design Decisions
//!
//! - `#ex'pattern'` = KBNF complement regex (matches text NOT containing pattern)
//! - `</ai00:function_calls>` acts as implicit terminator (grammar completion)
//! - Single function_calls block per turn (system injects response, model continues)
//! - Only text-only responses require explicit terminator
//! - Allows `<` in text before function calls (e.g., "2 < 3")
//!
//! ## ai00 XML Tool Format
//!
//! Tool calls use XML-based format instead of JSON:
//! ```xml
//! <ai00:function_calls>
//!   <invoke name="get_weather">
//!     <parameter name="city">Tokyo</parameter>
//!   </invoke>
//! </ai00:function_calls>
//! ```

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

/// Unified structural grammar for all response types.
///
/// This grammar handles all combinations:
/// - Plain text responses (requires terminator)
/// - Thinking + text responses (requires terminator)
/// - Function calls (terminates at `</ai00:function_calls>`)
/// - Thinking + function calls (terminates at `</ai00:function_calls>`)
///
/// The grammar uses KBNF's complement regex (`#ex'pattern'`) to match text
/// that does not contain specific tags, allowing `<` characters in regular text.
///
/// ## Structure
///
/// ```text
/// start ::= thinking? content
/// thinking ::= '<think>' ... '</think>'
/// content ::= function_calls | text_response
/// ```
///
/// Function calls terminate the grammar (allowing result injection).
/// Text-only responses require an explicit terminator (e.g., `</ai00:assistant>`).
///
/// ## Output Format
///
/// ```xml
/// <ai00:function_calls>
///   <invoke name="tool_name">
///     <parameter name="param1">value1</parameter>
///   </invoke>
/// </ai00:function_calls>
/// ```
pub const GRAMMAR_UNIFIED: &str = r#"
start::=thinking? content;
thinking::='<think>' #ex'</think>' '</think>' ws;
content::=function_calls | text_response;
function_calls::=#ex'<ai00:function_calls>' '<ai00:function_calls>\n' invokes '</ai00:function_calls>';
invokes::=invoke*;
invoke::='  <invoke name="' tool_name '">\n' params '  </invoke>\n';
params::=param*;
param::='    <parameter name="' param_name '">' param_value '</parameter>\n';
tool_name::=#'[a-zA-Z0-9_-]+';
param_name::=#'[a-zA-Z0-9_]+';
param_value::=#ex'</parameter>';
text_response::=#ex'<ai00:function_calls>' #'[\\s\\S]*' terminator;
"#;

/// Thinking wrapper for user-provided grammars.
///
/// When the user provides a custom `bnf_schema` but thinking is also enabled,
/// this wrapper adds thinking tag support around their grammar.
///
/// Usage: Prepend to user's grammar where their start rule becomes `user_start`.
pub const GRAMMAR_THINKING_WRAPPER: &str = r#"
start::=thinking_block? user_start;
thinking_block::='<think>' #ex'</think>' '</think>' ws;
ws::=#'[ \\t\\n\\r]*';
"#;

/// Build the terminator rule from stop sequences.
///
/// Generates a rule like: `terminator::='\n\n' | '</s>' | '\n';`
pub fn build_terminator_rule(stop_sequences: &[String]) -> String {
    if stop_sequences.is_empty() {
        // Default to double newline if no stop sequences
        return "terminator::='\\n\\n';\n".to_string();
    }

    let alternatives: Vec<String> = stop_sequences
        .iter()
        .map(|s| {
            // Escape special characters for KBNF
            let escaped = s
                .replace('\\', "\\\\")
                .replace('\'', "\\'")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            format!("'{}'", escaped)
        })
        .collect();

    format!("terminator::={};\n", alternatives.join(" | "))
}

/// Build a complete structural grammar for constrained decoding.
///
/// Uses the unified grammar that always allows optional thinking blocks
/// and handles both tool calls and text responses. The grammar parameters
/// are kept for API compatibility but the unified grammar handles all cases.
///
/// ## Arguments
///
/// * `_thinking_enabled` - Ignored; thinking is always optional in unified grammar
/// * `_tools_present` - Ignored; tool calls are always allowed in unified grammar
/// * `stop_sequences` - Stop sequences used to build the terminator rule
///
/// ## Returns
///
/// A complete KBNF grammar string ready for the BNF sampler.
pub fn build_structural_grammar(
    _thinking_enabled: bool,
    _tools_present: bool,
    stop_sequences: &[String],
) -> String {
    let mut grammar = String::new();

    // Always include JSON primitives as base
    grammar.push_str(GRAMMAR_JSON_PRIMITIVES);
    grammar.push('\n');

    // Use unified grammar for all cases
    grammar.push_str(GRAMMAR_UNIFIED);
    grammar.push('\n');

    // Add terminator rule based on stop sequences
    grammar.push_str(&build_terminator_rule(stop_sequences));

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
    fn test_grammar_unified_structure() {
        // Unified grammar should have start rule
        assert!(GRAMMAR_UNIFIED.contains("start::="));

        // Should support thinking (optional)
        assert!(GRAMMAR_UNIFIED.contains("thinking"));
        assert!(GRAMMAR_UNIFIED.contains("<think>"));
        assert!(GRAMMAR_UNIFIED.contains("</think>"));

        // Should support function calls (ai00 XML format)
        assert!(GRAMMAR_UNIFIED.contains("function_calls"));
        assert!(GRAMMAR_UNIFIED.contains("<ai00:function_calls>"));
        assert!(GRAMMAR_UNIFIED.contains("</ai00:function_calls>"));

        // Should have invoke structure
        assert!(GRAMMAR_UNIFIED.contains("invoke"));
        assert!(GRAMMAR_UNIFIED.contains("<invoke name="));
        assert!(GRAMMAR_UNIFIED.contains("</invoke>"));

        // Should have parameter structure
        assert!(GRAMMAR_UNIFIED.contains("param"));
        assert!(GRAMMAR_UNIFIED.contains("<parameter name="));
        assert!(GRAMMAR_UNIFIED.contains("</parameter>"));

        // Should have text response with terminator
        assert!(GRAMMAR_UNIFIED.contains("text_response"));
        assert!(GRAMMAR_UNIFIED.contains("terminator"));

        // Should use complement regex for flexible matching
        assert!(GRAMMAR_UNIFIED.contains("#ex'"));
    }

    #[test]
    fn test_grammar_unified_uses_complement_regex() {
        // The unified grammar should use #ex for matching text before tags
        // This allows < characters in text (e.g., "2 < 3") and in parameter values
        assert!(GRAMMAR_UNIFIED.contains("#ex'</think>'"));
        assert!(GRAMMAR_UNIFIED.contains("#ex'<ai00:function_calls>'"));
        // param_value uses #ex'</parameter>' to allow < in values
        assert!(GRAMMAR_UNIFIED.contains("param_value::=#ex'</parameter>'"));
    }

    #[test]
    fn test_build_structural_grammar_unified() {
        // All parameter combinations should produce the same unified grammar
        let stop_seqs = vec!["</ai00:assistant>".to_string()];

        let grammar_tt = build_structural_grammar(true, true, &stop_seqs);
        let grammar_tf = build_structural_grammar(true, false, &stop_seqs);
        let grammar_ft = build_structural_grammar(false, true, &stop_seqs);
        let grammar_ff = build_structural_grammar(false, false, &stop_seqs);

        // All should contain unified grammar elements
        for grammar in [&grammar_tt, &grammar_tf, &grammar_ft, &grammar_ff] {
            assert!(grammar.contains("json_object")); // From primitives
            assert!(grammar.contains("<think>")); // Always in unified
            assert!(grammar.contains("<ai00:function_calls>")); // ai00 format
            assert!(grammar.contains("terminator::=")); // From stop sequences
        }
    }

    #[test]
    fn test_build_structural_grammar_includes_terminator() {
        let stop_seqs = vec!["\n\n".to_string(), "</s>".to_string()];
        let grammar = build_structural_grammar(false, false, &stop_seqs);

        assert!(grammar.contains("terminator::="));
        assert!(grammar.contains("'\\n\\n'"));
        assert!(grammar.contains("'</s>'"));
    }

    #[test]
    fn test_build_terminator_rule_empty() {
        let rule = build_terminator_rule(&[]);
        assert_eq!(rule, "terminator::='\\n\\n';\n");
    }

    #[test]
    fn test_build_terminator_rule_multiple() {
        let stop_seqs = vec!["\n\n".to_string(), "\n".to_string()];
        let rule = build_terminator_rule(&stop_seqs);
        assert!(rule.contains("'\\n\\n'"));
        assert!(rule.contains("'\\n'"));
        assert!(rule.contains(" | "));
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
        // Should use complement regex in thinking block
        assert!(wrapped.contains("#ex'</think>'"));
    }

    #[test]
    fn test_grammar_constants_have_valid_kbnf_syntax() {
        // Basic syntax checks - ensure grammars follow KBNF patterns
        let grammars = [
            GRAMMAR_JSON_PRIMITIVES,
            GRAMMAR_UNIFIED,
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
