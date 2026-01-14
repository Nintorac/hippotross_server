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

Enforces correct syntax for thinking blocks and function calls without validating tool schemas.

```json
{
  "bnf_validation": "structural"
}
```

The structural grammar validates:
- `<think>...</think>` blocks are properly closed
- `<ai00:function_calls>...</ai00:function_calls>` blocks are properly closed
- `<invoke>` and `<parameter>` structure is correct
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
start           ::= thinking? content
thinking        ::= '<think>' #ex'</think>' '</think>' ws

content         ::= function_calls | text_response

function_calls  ::= #ex'<ai00:function_calls>' '<ai00:function_calls>\n' invokes '</ai00:function_calls>'
invokes         ::= invoke*
invoke          ::= '  <invoke name="' tool_name '">\n' params '  </invoke>\n'
params          ::= param*
param           ::= '    <parameter name="' param_name '">' param_value '</parameter>\n'

tool_name       ::= #'[a-zA-Z0-9_-]+'
param_name      ::= #'[a-zA-Z0-9_]+'
param_value     ::= #ex'</parameter>'

text_response   ::= #ex'<ai00:function_calls>' #'[\s\S]*' terminator
```

### Key Design Decisions

1. **Thinking is always optional**: The grammar allows `<think>` blocks regardless of configuration. This prevents conflicts when models spontaneously decide to "think".

2. **Function calls terminate**: The grammar naturally completes at `</ai00:function_calls>`, triggering `stop_reason: tool_use`.

3. **Text requires terminator**: Text-only responses require an explicit terminator (e.g., `</ai00:assistant>`).

4. **Complement regex**: The `#ex'pattern'` syntax matches any text NOT containing the pattern, allowing `<` characters in regular text and parameter values.

## Valid Output Patterns

### Plain Text
```
Hello, how can I help you?
</ai00:assistant>
```

### Thinking + Text
```
<think>Let me consider this question...</think>
The answer is 42.
</ai00:assistant>
```

### Function Call (Single)
```xml
<ai00:function_calls>
  <invoke name="get_weather">
    <parameter name="location">Tokyo</parameter>
  </invoke>
</ai00:function_calls>
```

### Thinking + Function Call
```xml
<think>I should check the weather first.</think>
<ai00:function_calls>
  <invoke name="get_weather">
    <parameter name="location">Tokyo</parameter>
  </invoke>
</ai00:function_calls>
```

### Text Before Function Call
```xml
Let me search for that information.
<ai00:function_calls>
  <invoke name="web_search">
    <parameter name="query">AI news</parameter>
  </invoke>
</ai00:function_calls>
```

### Multiple Parallel Function Calls
```xml
<ai00:function_calls>
  <invoke name="get_weather">
    <parameter name="city">Tokyo</parameter>
  </invoke>
  <invoke name="get_weather">
    <parameter name="city">London</parameter>
  </invoke>
</ai00:function_calls>
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
  "stop": ["</ai00:assistant>"]
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

Check that your stop sequences are properly configured. The default stop sequence is `</ai00:assistant>`.

### Function calls fail to parse

Ensure the output follows the XML structure:
```xml
<ai00:function_calls>
  <invoke name="tool_name">
    <parameter name="key">value</parameter>
  </invoke>
</ai00:function_calls>
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
