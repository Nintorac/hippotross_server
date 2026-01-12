//! JSON Schema to KBNF grammar conversion for Stage 2 schema-aware validation.
//!
//! This module provides functions to convert JSON Schema definitions into
//! KBNF grammar rules, enabling schema-specific constrained decoding.

use serde_json::Value;
use std::collections::HashSet;

use super::types::Tool;

/// Context for generating unique rule names during recursive schema conversion.
#[derive(Debug, Default)]
pub struct GeneratorContext {
    /// Counter for generating unique rule names
    rule_counter: usize,
    /// Accumulated grammar rules
    rules: Vec<String>,
}

impl GeneratorContext {
    /// Create a new generator context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a unique rule name with the given prefix.
    pub fn unique_rule(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.rule_counter);
        self.rule_counter += 1;
        name
    }

    /// Add a rule to the accumulated rules.
    pub fn add_rule(&mut self, rule: String) {
        self.rules.push(rule);
    }

    /// Get all accumulated rules as a single grammar string.
    pub fn into_grammar(self) -> String {
        self.rules.join("\n")
    }
}

/// Convert a JSON Schema to KBNF grammar rules.
///
/// Supports:
/// - `type: "object"` with `properties` and `required`
/// - `type: "string"` with optional `enum`
/// - `type: "number"` and `type: "integer"`
/// - `type: "boolean"`
/// - `type: "array"` with optional `items`
/// - `anyOf` and `oneOf` (converted to alternation)
///
/// # Arguments
/// * `schema` - The JSON Schema value
/// * `rule_name` - The name for the generated rule
/// * `ctx` - Generator context for tracking rules and unique names
///
/// # Returns
/// The name of the generated rule (may be same as input or a reference to base rules)
pub fn json_schema_to_kbnf(schema: &Value, rule_name: &str, ctx: &mut GeneratorContext) -> String {
    // Handle anyOf/oneOf first
    if let Some(any_of) = schema.get("anyOf").and_then(|v| v.as_array()) {
        return handle_any_of(any_of, rule_name, ctx);
    }
    if let Some(one_of) = schema.get("oneOf").and_then(|v| v.as_array()) {
        return handle_any_of(one_of, rule_name, ctx); // Same handling
    }

    match schema.get("type").and_then(|t| t.as_str()) {
        Some("object") => handle_object(schema, rule_name, ctx),
        Some("string") => handle_string(schema, rule_name, ctx),
        Some("number") | Some("integer") => {
            ctx.add_rule(format!("{}::=number;", rule_name));
            rule_name.to_string()
        }
        Some("boolean") => {
            ctx.add_rule(format!("{}::='true' | 'false';", rule_name));
            rule_name.to_string()
        }
        Some("array") => handle_array(schema, rule_name, ctx),
        Some("null") => {
            ctx.add_rule(format!("{}::='null';", rule_name));
            rule_name.to_string()
        }
        _ => {
            // Unknown or missing type - fallback to generic JSON value
            ctx.add_rule(format!("{}::=json_value;", rule_name));
            rule_name.to_string()
        }
    }
}

/// Handle anyOf/oneOf by converting to alternation.
fn handle_any_of(variants: &[Value], rule_name: &str, ctx: &mut GeneratorContext) -> String {
    let mut variant_rules = Vec::new();

    for variant in variants.iter() {
        let variant_name = ctx.unique_rule(&format!("{}_var", rule_name));
        json_schema_to_kbnf(variant, &variant_name, ctx);
        variant_rules.push(variant_name);
    }

    ctx.add_rule(format!("{}::={};", rule_name, variant_rules.join(" | ")));
    rule_name.to_string()
}

/// Handle object type with properties and required fields.
fn handle_object(schema: &Value, rule_name: &str, ctx: &mut GeneratorContext) -> String {
    let props = schema.get("properties").and_then(|p| p.as_object());
    let required: HashSet<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let props = match props {
        Some(p) if !p.is_empty() => p,
        _ => {
            // Empty object or no properties - allow any JSON object
            ctx.add_rule(format!("{}::=json_object;", rule_name));
            return rule_name.to_string();
        }
    };

    // Separate required and optional properties
    let mut required_props: Vec<(&String, &Value)> = Vec::new();
    let mut optional_props: Vec<(&String, &Value)> = Vec::new();

    for (key, value) in props {
        if required.contains(key.as_str()) {
            required_props.push((key, value));
        } else {
            optional_props.push((key, value));
        }
    }

    // Generate rules for each property value
    let mut property_parts = Vec::new();

    // Required properties first
    for (i, (key, prop_schema)) in required_props.iter().enumerate() {
        let value_rule = ctx.unique_rule(&format!("{}_prop_{}", rule_name, key));
        json_schema_to_kbnf(prop_schema, &value_rule, ctx);

        let comma = if i > 0 { "',' ws " } else { "" };
        property_parts.push(format!(
            "{}'\"{}\"' ws ':' ws {}",
            comma, key, value_rule
        ));
    }

    // Optional properties (wrapped in (...)?)
    for (key, prop_schema) in optional_props.iter() {
        let value_rule = ctx.unique_rule(&format!("{}_prop_{}", rule_name, key));
        json_schema_to_kbnf(prop_schema, &value_rule, ctx);

        // Optional property needs comma handling
        let comma_prefix = if !property_parts.is_empty() || !required_props.is_empty() {
            "',' ws "
        } else {
            ""
        };
        property_parts.push(format!(
            "({}'\"{}\"' ws ':' ws {})?",
            comma_prefix, key, value_rule
        ));
    }

    // Build the object rule
    let members = property_parts.join(" ws ");
    ctx.add_rule(format!(
        "{}::='{{' ws {} ws '}}';",
        rule_name, members
    ));

    rule_name.to_string()
}

/// Handle string type with optional enum constraint.
fn handle_string(schema: &Value, rule_name: &str, ctx: &mut GeneratorContext) -> String {
    if let Some(enum_vals) = schema.get("enum").and_then(|e| e.as_array()) {
        // Enum: generate alternation of literal strings
        let vals: Vec<String> = enum_vals
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| format!("'\"{}\"'", escape_kbnf_string(s)))
            .collect();

        if vals.is_empty() {
            ctx.add_rule(format!("{}::=string;", rule_name));
        } else {
            ctx.add_rule(format!("{}::={};", rule_name, vals.join(" | ")));
        }
    } else {
        ctx.add_rule(format!("{}::=string;", rule_name));
    }

    rule_name.to_string()
}

/// Handle array type with optional items schema.
fn handle_array(schema: &Value, rule_name: &str, ctx: &mut GeneratorContext) -> String {
    if let Some(items_schema) = schema.get("items") {
        // Generate rule for array items
        let items_rule = ctx.unique_rule(&format!("{}_items", rule_name));
        json_schema_to_kbnf(items_schema, &items_rule, ctx);

        // Array with typed items
        let elements_rule = ctx.unique_rule(&format!("{}_elements", rule_name));
        ctx.add_rule(format!(
            "{}::={} (',' ws {})*;",
            elements_rule, items_rule, items_rule
        ));
        ctx.add_rule(format!(
            "{}::='[' ws {}? ws ']';",
            rule_name, elements_rule
        ));
    } else {
        // No items schema - allow any JSON array
        ctx.add_rule(format!("{}::=json_array;", rule_name));
    }

    rule_name.to_string()
}

/// Escape special characters in a string for KBNF literal.
fn escape_kbnf_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Convenience function to convert a JSON Schema to a complete KBNF grammar.
///
/// This wraps json_schema_to_kbnf with context management and includes
/// the base JSON primitives needed for the generated rules.
pub fn schema_to_grammar(schema: &Value, start_rule: &str) -> String {
    use super::bnf_grammars::GRAMMAR_JSON_PRIMITIVES;

    let mut ctx = GeneratorContext::new();
    json_schema_to_kbnf(schema, start_rule, &mut ctx);

    let mut grammar = String::new();
    grammar.push_str(GRAMMAR_JSON_PRIMITIVES);
    grammar.push('\n');
    grammar.push_str(&ctx.into_grammar());
    grammar
}

/// Generate tool name alternatives from tool definitions.
///
/// Creates a grammar rule that matches any of the provided tool names.
///
/// # Example
/// ```text
/// Input: [Tool{name: "get_weather"}, Tool{name: "search"}]
/// Output: tool_name ::= "get_weather" | "search";
/// ```
pub fn generate_tool_name_grammar(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let names: Vec<String> = tools
        .iter()
        .map(|t| format!("'{}'", t.name))
        .collect();

    format!("tool_name::={};", names.join(" | "))
}

/// Generate input grammar for each tool based on its input_schema.
///
/// Creates per-tool rules for:
/// 1. Tool call structure: `{tool_name}_call` - matches `{"name": "tool_name", "input": ...}`
/// 2. Tool input schema: `{tool_name}_input` - matches the tool's input JSON Schema
/// 3. Dispatch rule: `tool_call` - alternation of all tool call rules
///
/// # Returns
/// A tuple of (grammar_rules, context) where grammar_rules is the string of all rules
/// and context contains the accumulated state for potential further use.
pub fn generate_tool_grammars(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut ctx = GeneratorContext::new();
    let mut tool_calls = Vec::new();

    for tool in tools {
        let call_rule = format!("{}_call", tool.name);
        let input_rule = format!("{}_input", tool.name);

        // Generate input schema rule using json_schema_to_kbnf
        json_schema_to_kbnf(&tool.input_schema, &input_rule, &mut ctx);

        // Tool call rule: {"name": "tool_name", "arguments": ...}
        ctx.add_rule(format!(
            r#"{}::='{{' ws '"name"' ws ':' ws '"{}"' ws ',' ws '"arguments"' ws ':' ws {} ws '}}';"#,
            call_rule, tool.name, input_rule
        ));

        tool_calls.push(call_rule);
    }

    // Dispatch rule - alternation of all tool calls
    ctx.add_rule(format!("tool_call::={};", tool_calls.join(" | ")));

    ctx.into_grammar()
}

/// Generate a complete schema-aware grammar for tools.
///
/// Combines:
/// - JSON primitives from Stage 1
/// - Unified structure rules (thinking always optional)
/// - Per-tool call and input rules with schema validation
///
/// The grammar always allows optional thinking blocks since models may
/// spontaneously decide to "think" regardless of configuration.
///
/// Note: Unlike the structural grammar, SchemaAware validates tool names
/// and argument schemas against the provided tool definitions.
pub fn generate_schema_aware_grammar(tools: &[Tool]) -> String {
    use super::bnf_grammars::GRAMMAR_JSON_PRIMITIVES;

    let mut grammar = String::new();

    // Base JSON primitives
    grammar.push_str(GRAMMAR_JSON_PRIMITIVES);
    grammar.push('\n');

    // Unified structure with thinking always optional
    // Uses complement regex to allow < in text before tool calls
    // Note: `tool_call` is defined by generate_tool_grammars as a dispatch rule
    grammar.push_str(r#"
start::=thinking? content;
thinking::='<think>' #ex'</think>' '</think>' ws;
content::=tool_call_block | text_response;
tool_call_block::=#ex'<tool_call>' '<tool_call>' ws tool_call ws '</tool_call>';
text_response::=#'.*' terminator;
"#);
    grammar.push('\n');

    // Tool-specific rules (validates tool names and schemas)
    // This defines `tool_call::=tool1_call | tool2_call | ...`
    grammar.push_str(&generate_tool_grammars(tools));

    grammar
}

/// Main entry point for BNF schema generation based on request parameters.
///
/// This is the function that should be called from the handler to generate
/// the appropriate grammar based on the validation level and request features.
///
/// # Arguments
/// * `tools` - Optional slice of tool definitions
/// * `thinking_enabled` - Whether extended thinking is enabled (kept for API compat, ignored)
/// * `validation_level` - The BNF validation level from the request
/// * `stop_sequences` - Stop sequences used to build the terminator rule
///
/// # Returns
/// `Some(grammar)` if a grammar should be applied, `None` if no constraints
///
/// # Validation Level Behavior
/// - `None`: Returns `None` (no grammar constraints)
/// - `Structural`: Uses unified grammar (thinking always optional)
/// - `SchemaAware`: Generates grammar with tool schema validation
pub fn generate_bnf_schema(
    tools: Option<&[Tool]>,
    _thinking_enabled: bool,
    validation_level: super::types::BnfValidationLevel,
    stop_sequences: &[String],
) -> Option<String> {
    use super::bnf_grammars::build_structural_grammar;
    use super::types::BnfValidationLevel;

    match validation_level {
        BnfValidationLevel::None => None,

        BnfValidationLevel::Structural => {
            // Always use unified grammar - thinking is always optional
            // Parameters are ignored since unified grammar handles all cases
            Some(build_structural_grammar(false, false, stop_sequences))
        }

        BnfValidationLevel::SchemaAware => {
            let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);

            if !has_tools {
                // No tools to validate - fall back to structural grammar
                return Some(build_structural_grammar(false, false, stop_sequences));
            }

            // Generate full schema-aware grammar with terminator
            let mut grammar = generate_schema_aware_grammar(tools.unwrap());
            grammar.push_str(&super::bnf_grammars::build_terminator_rule(stop_sequences));
            Some(grammar)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_string() {
        let schema = json!({"type": "string"});
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "test_str", &mut ctx);
        let grammar = ctx.into_grammar();
        assert!(grammar.contains("test_str::=string;"));
    }

    #[test]
    fn test_string_enum() {
        let schema = json!({
            "type": "string",
            "enum": ["red", "green", "blue"]
        });
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "color", &mut ctx);
        let grammar = ctx.into_grammar();
        assert!(grammar.contains("color::="));
        assert!(grammar.contains(r#""red""#));
        assert!(grammar.contains(r#""green""#));
        assert!(grammar.contains(r#""blue""#));
    }

    #[test]
    fn test_number() {
        let schema = json!({"type": "number"});
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "num", &mut ctx);
        let grammar = ctx.into_grammar();
        assert!(grammar.contains("num::=number;"));
    }

    #[test]
    fn test_integer() {
        let schema = json!({"type": "integer"});
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "int", &mut ctx);
        let grammar = ctx.into_grammar();
        assert!(grammar.contains("int::=number;"));
    }

    #[test]
    fn test_boolean() {
        let schema = json!({"type": "boolean"});
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "flag", &mut ctx);
        let grammar = ctx.into_grammar();
        assert!(grammar.contains("flag::='true' | 'false';"));
    }

    #[test]
    fn test_null() {
        let schema = json!({"type": "null"});
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "nil", &mut ctx);
        let grammar = ctx.into_grammar();
        assert!(grammar.contains("nil::='null';"));
    }

    #[test]
    fn test_simple_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "person", &mut ctx);
        let grammar = ctx.into_grammar();

        // Should have person rule with object syntax
        assert!(grammar.contains("person::='{'"));
        // Should have property rules
        assert!(grammar.contains(r#""name""#));
        // Age should be optional (wrapped in (...)?)
        assert!(grammar.contains("?"));
    }

    #[test]
    fn test_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"}
                    },
                    "required": ["email"]
                }
            },
            "required": ["user"]
        });
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "data", &mut ctx);
        let grammar = ctx.into_grammar();

        // Should have rules for both levels
        assert!(grammar.contains("data::="));
        assert!(grammar.contains(r#""user""#));
        assert!(grammar.contains(r#""email""#));
    }

    #[test]
    fn test_array_simple() {
        let schema = json!({"type": "array"});
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "arr", &mut ctx);
        let grammar = ctx.into_grammar();
        assert!(grammar.contains("arr::=json_array;"));
    }

    #[test]
    fn test_array_with_items() {
        let schema = json!({
            "type": "array",
            "items": {"type": "string"}
        });
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "tags", &mut ctx);
        let grammar = ctx.into_grammar();

        // Should have array rule with items
        assert!(grammar.contains("tags::='['"));
        assert!(grammar.contains("']'"));
    }

    #[test]
    fn test_any_of() {
        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "number"}
            ]
        });
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "str_or_num", &mut ctx);
        let grammar = ctx.into_grammar();

        // Should have alternation
        assert!(grammar.contains("str_or_num::="));
        assert!(grammar.contains(" | "));
    }

    #[test]
    fn test_one_of() {
        let schema = json!({
            "oneOf": [
                {"type": "boolean"},
                {"type": "null"}
            ]
        });
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "bool_or_null", &mut ctx);
        let grammar = ctx.into_grammar();

        assert!(grammar.contains("bool_or_null::="));
        assert!(grammar.contains(" | "));
    }

    #[test]
    fn test_unknown_type_fallback() {
        let schema = json!({"description": "any value"});
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "any", &mut ctx);
        let grammar = ctx.into_grammar();
        assert!(grammar.contains("any::=json_value;"));
    }

    #[test]
    fn test_schema_to_grammar_includes_primitives() {
        let schema = json!({"type": "string"});
        let grammar = schema_to_grammar(&schema, "start");

        // Should include primitives from bnf_grammars
        assert!(grammar.contains("json_object::="));
        assert!(grammar.contains("json_value::="));
        assert!(grammar.contains("string::="));
    }

    #[test]
    fn test_escape_kbnf_string() {
        assert_eq!(escape_kbnf_string("hello"), "hello");
        assert_eq!(escape_kbnf_string(r#"say "hi""#), r#"say \"hi\""#);
        assert_eq!(escape_kbnf_string(r"path\to\file"), r"path\\to\\file");
    }

    #[test]
    fn test_generator_context_unique_rules() {
        let mut ctx = GeneratorContext::new();
        let r1 = ctx.unique_rule("prop");
        let r2 = ctx.unique_rule("prop");
        let r3 = ctx.unique_rule("other");

        assert_ne!(r1, r2);
        assert_ne!(r2, r3);
        assert!(r1.starts_with("prop_"));
        assert!(r3.starts_with("other_"));
    }

    #[test]
    fn test_complex_tool_schema() {
        // Simulate a get_weather tool schema
        let schema = json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        });
        let mut ctx = GeneratorContext::new();
        json_schema_to_kbnf(&schema, "get_weather_input", &mut ctx);
        let grammar = ctx.into_grammar();

        // Required field
        assert!(grammar.contains(r#""location""#));
        // Enum field
        assert!(grammar.contains(r#""celsius""#));
        assert!(grammar.contains(r#""fahrenheit""#));
    }

    // --- Tool Grammar Generation Tests ---

    fn make_tool(name: &str, schema: Value) -> Tool {
        Tool {
            name: name.to_string(),
            description: Some(format!("Test tool: {}", name)),
            input_schema: schema,
            cache_control: None,
        }
    }

    #[test]
    fn test_generate_tool_name_grammar_empty() {
        let grammar = generate_tool_name_grammar(&[]);
        assert!(grammar.is_empty());
    }

    #[test]
    fn test_generate_tool_name_grammar_single() {
        let tools = vec![make_tool("get_weather", json!({"type": "object"}))];
        let grammar = generate_tool_name_grammar(&tools);
        assert_eq!(grammar, r#"tool_name::='get_weather';"#);
    }

    #[test]
    fn test_generate_tool_name_grammar_multiple() {
        let tools = vec![
            make_tool("get_weather", json!({"type": "object"})),
            make_tool("search", json!({"type": "object"})),
            make_tool("calculate", json!({"type": "object"})),
        ];
        let grammar = generate_tool_name_grammar(&tools);
        assert!(grammar.contains("tool_name::="));
        assert!(grammar.contains(r#"'get_weather'"#));
        assert!(grammar.contains(r#"'search'"#));
        assert!(grammar.contains(r#"'calculate'"#));
        assert!(grammar.contains(" | "));
    }

    #[test]
    fn test_generate_tool_grammars_empty() {
        let grammar = generate_tool_grammars(&[]);
        assert!(grammar.is_empty());
    }

    #[test]
    fn test_generate_tool_grammars_single() {
        let tools = vec![make_tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        )];
        let grammar = generate_tool_grammars(&tools);

        // Should have tool call rule
        assert!(grammar.contains("get_weather_call::="));
        // Should reference input rule
        assert!(grammar.contains("get_weather_input"));
        // Should have dispatch rule
        assert!(grammar.contains("tool_call::=get_weather_call;"));
        // Should have location property
        assert!(grammar.contains(r#""location""#));
    }

    #[test]
    fn test_generate_tool_grammars_multiple() {
        let tools = vec![
            make_tool(
                "get_weather",
                json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }),
            ),
            make_tool(
                "search",
                json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }),
            ),
        ];
        let grammar = generate_tool_grammars(&tools);

        // Both tools should have call rules
        assert!(grammar.contains("get_weather_call::="));
        assert!(grammar.contains("search_call::="));

        // Dispatch rule should have alternation
        assert!(grammar.contains("tool_call::="));
        assert!(grammar.contains("get_weather_call"));
        assert!(grammar.contains("search_call"));
        assert!(grammar.contains(" | "));
    }

    #[test]
    fn test_generate_tool_grammars_with_enum() {
        let tools = vec![make_tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }),
        )];
        let grammar = generate_tool_grammars(&tools);

        // Should have enum values
        assert!(grammar.contains(r#""celsius""#));
        assert!(grammar.contains(r#""fahrenheit""#));
    }

    #[test]
    fn test_generate_schema_aware_grammar_unified_structure() {
        let tools = vec![make_tool(
            "search",
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }),
        )];
        let grammar = generate_schema_aware_grammar(&tools);

        // Should have base primitives
        assert!(grammar.contains("json_object::="));

        // Should have unified start rule with optional thinking
        assert!(grammar.contains("start::=thinking? content"));

        // Should always have thinking support (unified grammar)
        assert!(grammar.contains("thinking"));
        assert!(grammar.contains("<think>"));
        assert!(grammar.contains("</think>"));

        // Should have tool structure
        assert!(grammar.contains("tool_call"));
        assert!(grammar.contains("<tool_call>"));
        assert!(grammar.contains("</tool_call>"));

        // Should use complement regex
        assert!(grammar.contains("#ex'"));
    }

    #[test]
    fn test_generate_schema_aware_grammar_complete_example() {
        // Replicate the example from spec 8.4.4
        let tools = vec![
            make_tool(
                "get_weather",
                json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }),
            ),
            make_tool(
                "search",
                json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }),
            ),
        ];
        let grammar = generate_schema_aware_grammar(&tools);

        // All expected components
        assert!(grammar.contains("start::="));
        assert!(grammar.contains("thinking"));
        assert!(grammar.contains("content"));
        assert!(grammar.contains("<tool_call>"));
        assert!(grammar.contains("</tool_call>"));

        // Tool dispatch (schema-aware generates per-tool rules)
        assert!(grammar.contains("get_weather_call"));
        assert!(grammar.contains("search_call"));

        // Tool-specific rules
        assert!(grammar.contains("get_weather_input"));
        assert!(grammar.contains("search_input"));
    }

    // --- generate_bnf_schema Entry Point Tests ---

    #[test]
    fn test_generate_bnf_schema_none_level() {
        use super::super::types::BnfValidationLevel;

        let tools = vec![make_tool("search", json!({"type": "object"}))];
        let stop_seqs = vec!["\n\n".to_string()];

        // None level should always return None, regardless of tools/thinking
        assert!(generate_bnf_schema(Some(&tools), false, BnfValidationLevel::None, &stop_seqs).is_none());
        assert!(generate_bnf_schema(Some(&tools), true, BnfValidationLevel::None, &stop_seqs).is_none());
        assert!(generate_bnf_schema(None, true, BnfValidationLevel::None, &stop_seqs).is_none());
        assert!(generate_bnf_schema(None, false, BnfValidationLevel::None, &stop_seqs).is_none());
    }

    #[test]
    fn test_generate_bnf_schema_structural_unified() {
        use super::super::types::BnfValidationLevel;

        let stop_seqs = vec!["\n\n".to_string()];

        // Structural now always returns unified grammar (thinking always optional)
        let result = generate_bnf_schema(None, false, BnfValidationLevel::Structural, &stop_seqs);
        assert!(result.is_some());

        let grammar = result.unwrap();
        // Unified grammar always has both thinking and tools
        assert!(grammar.contains("<think>"));
        assert!(grammar.contains("<tool_call>"));
        assert!(grammar.contains("terminator::="));
    }

    #[test]
    fn test_generate_bnf_schema_structural_all_params_produce_same_grammar() {
        use super::super::types::BnfValidationLevel;

        let tools = vec![make_tool("search", json!({"type": "object"}))];
        let stop_seqs = vec!["\n\n".to_string()];

        // All parameter combinations should produce the same unified grammar
        let g1 = generate_bnf_schema(None, false, BnfValidationLevel::Structural, &stop_seqs).unwrap();
        let g2 = generate_bnf_schema(None, true, BnfValidationLevel::Structural, &stop_seqs).unwrap();
        let g3 = generate_bnf_schema(Some(&tools), false, BnfValidationLevel::Structural, &stop_seqs).unwrap();
        let g4 = generate_bnf_schema(Some(&tools), true, BnfValidationLevel::Structural, &stop_seqs).unwrap();

        // All should be identical (unified grammar ignores params)
        assert_eq!(g1, g2);
        assert_eq!(g2, g3);
        assert_eq!(g3, g4);
    }

    #[test]
    fn test_generate_bnf_schema_schema_aware_no_tools() {
        use super::super::types::BnfValidationLevel;

        let stop_seqs = vec!["\n\n".to_string()];

        // SchemaAware without tools falls back to unified structural grammar
        let result = generate_bnf_schema(None, true, BnfValidationLevel::SchemaAware, &stop_seqs);
        assert!(result.is_some());

        let grammar = result.unwrap();
        // Unified grammar has both (but no tool-specific rules)
        assert!(grammar.contains("<think>"));
        assert!(grammar.contains("<tool_call>"));
        // No tool-specific rules since no tools provided (check for pattern like "get_weather_call")
        // The grammar should not have tool-specific dispatch rules
        assert!(!grammar.contains("_input"), "Should not have tool-specific input rules");
    }

    #[test]
    fn test_generate_bnf_schema_schema_aware_with_tools() {
        use super::super::types::BnfValidationLevel;

        let tools = vec![make_tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }),
        )];
        let stop_seqs = vec!["\n\n".to_string()];

        let result = generate_bnf_schema(Some(&tools), false, BnfValidationLevel::SchemaAware, &stop_seqs);
        assert!(result.is_some());

        let grammar = result.unwrap();
        // Should have tool-specific rules
        assert!(grammar.contains("get_weather_call"));
        assert!(grammar.contains("get_weather_input"));
        // Should also have thinking (unified grammar)
        assert!(grammar.contains("<think>"));
    }
}
