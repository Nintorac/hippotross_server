# BNF Grammar for Constrained Decoding

This document describes the BNF (Backus-Naur Form) grammar system used for constrained decoding in ai00_server.

## Overview

BNF validation constrains model output to follow specific structural patterns. This prevents malformed responses and ensures consistent formatting for tool calls and thinking blocks.

## Validation Levels

The API supports three validation levels via the `bnf_validation` parameter:

### `none`

No grammar constraints. Model output is unconstrained.

```json
{
  "bnf_validation": "none"
}
```

### `structural`

Enforces correct syntax for thinking blocks and tool calls without validating tool schemas.

```json
{
  "bnf_validation": "structural"
}
```

The structural grammar validates:
- `<think>...</think>` blocks are properly closed
- `<tool_call>...</tool_call>` blocks are properly closed
- Tool JSON has `name` and `arguments` fields
- Response terminates with stop sequences

### `schema_aware`

Full validation including tool names and argument schemas.

```json
{
  "bnf_validation": "schema_aware"
}
```

SchemaAware additionally validates:
- Tool names match provided tool definitions
- Tool arguments conform to JSON schemas

## Unified Grammar

The grammar uses a unified structure that handles all combinations of thinking and tools:

```ebnf
start         ::= thinking? content

thinking      ::= '<think>' #ex'</think>' '</think>' ws

content       ::= tool_call | text_response

tool_call     ::= #ex'<tool_call>' '<tool_call>' ws tool_json ws '</tool_call>'

text_response ::= #'.*' terminator

tool_json     ::= '{' ws '"name"' ws ':' ws string ws ',' ws '"arguments"' ws ':' ws json_object ws '}'
```

### Key Design Decisions

1. **Thinking is always optional**: The grammar allows `<think>` blocks regardless of configuration. This prevents conflicts when models spontaneously decide to "think".

2. **Tool call terminates**: A `</tool_call>` tag acts as an implicit terminator, allowing the system to inject tool responses.

3. **Text requires terminator**: Text-only responses require an explicit terminator (e.g., `\n\n`).

4. **Complement regex**: The `#ex'pattern'` syntax matches any text NOT containing the pattern, allowing `<` characters in regular text (e.g., "2 < 3").

## Valid Output Patterns

### Plain Text
```
Hello, how can I help you?

```

### Thinking + Text
```
<think>Let me consider this question...</think>
The answer is 42.

```

### Tool Call (Single)
```
<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>
```

### Thinking + Tool Call
```
<think>I should check the weather first.</think>
<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>
```

### Text Before Tool Call
```
Let me search for that information.
<tool_call>{"name": "web_search", "arguments": {"query": "AI news"}}</tool_call>
```

## API Configuration

### Auto-Enable Behavior

When `bnf_validation` is not specified, the system auto-enables `structural` validation when tools or thinking are present in the request.

### Example Request

```json
{
  "model": "rwkv-world",
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"}
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {"type": "string"}
        },
        "required": ["location"]
      }
    }
  ],
  "bnf_validation": "structural",
  "stop": ["\n\n"]
}
```

## Custom Grammars

You can provide custom grammars via the `bnf_schema` parameter:

```json
{
  "bnf_schema": "start::='yes' | 'no';",
  "bnf_validation": "none"
}
```

When thinking is enabled with a custom grammar, the system automatically wraps it to allow optional thinking blocks.

## Troubleshooting

### Model output is truncated

Check that your stop sequences are properly configured. The grammar requires a terminator for text responses.

### Tool calls fail to parse

Ensure tool JSON has both `name` and `arguments` fields:
```json
{"name": "tool_name", "arguments": {"key": "value"}}
```

### Grammar compilation errors

If using custom grammars, ensure they follow KBNF syntax:
- Rules end with semicolons: `rule::=pattern;`
- Regex patterns use `#'...'` syntax
- String literals use single quotes: `'literal'`

## KBNF Reference

The grammar uses KBNF (Koishi's BNF) syntax:

| Syntax | Meaning |
|--------|---------|
| `::=` | Rule definition |
| `\|` | Alternation |
| `?` | Optional (0 or 1) |
| `*` | Repetition (0 or more) |
| `+` | Repetition (1 or more) |
| `'...'` | String literal |
| `#'...'` | Regex pattern |
| `#ex'...'` | Complement regex (matches text NOT containing pattern) |

For more details, see the [KBNF documentation](https://github.com/Dan-wanna-M/kbnf).
