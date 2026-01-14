# ai00 Chat Format v1

This document describes the ai00 v1 chat prompt format used by ai00_server.

## Format Overview

ai00 uses XML-namespaced turn markers with the `ai00:` prefix:

```xml
<ai00:system>
You are a helpful assistant.
</ai00:system>

<ai00:user>
Hello!
</ai00:user>

<ai00:assistant>
Hi there! How can I help you today?
</ai00:assistant>
```

## Turn Separators

| Role | Opening Tag | Closing Tag |
|------|-------------|-------------|
| System | `<ai00:system>` | `</ai00:system>` |
| User | `<ai00:user>` | `</ai00:user>` |
| Assistant | `<ai00:assistant>` | `</ai00:assistant>` |

Each turn is wrapped in its corresponding tags. Turns are separated by double newlines (`\n\n`).

## Newline Handling

**Important change from v0**: Newlines within message content are now preserved as-is. There is no normalization of `\n\n` sequences - the XML tags provide unambiguous turn boundaries.

## Thinking Mode

For models that support extended thinking, use `<think>` tags:

```xml
<ai00:assistant>
<think>
Let me reason through this step by step...
</think>
Based on my analysis, the answer is 42.
</ai00:assistant>
```

### Assistant Prefixes

| Mode | Prefix |
|------|--------|
| Normal | `<ai00:assistant>\n` |
| Thinking | `<ai00:assistant>\n<think>\n` |

## Tool Calling

### Tool Definitions

Tools are defined in the system prompt using `<ai00:available_tools>`:

```xml
<ai00:system>
You are a helpful assistant.

<ai00:available_tools>
  <tool name="get_weather">
    {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "city": { "type": "string" }
        },
        "required": ["city"]
      }
    }
  </tool>
</ai00:available_tools>
</ai00:system>
```

### Tool Calls (Model Output)

When the model wants to call a tool, it outputs:

```xml
<ai00:function_calls>
  <invoke name="get_weather">
    <parameter name="city">Tokyo</parameter>
  </invoke>
</ai00:function_calls>
```

Multiple tools can be called in parallel:

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

### Tool Results (Injected)

After tool execution, results are injected:

```xml
<ai00:function_results>
  <result name="toolu_01abc123">
    {
      "temperature": 22,
      "condition": "sunny"
    }
  </result>
</ai00:function_results>
```

### Complete Tool Call Flow

```xml
<ai00:user>
What's the weather in Tokyo?
</ai00:user>

<ai00:assistant>
I'll check that for you.

<ai00:function_calls>
  <invoke name="get_weather">
    <parameter name="city">Tokyo</parameter>
  </invoke>
</ai00:function_calls>
<ai00:function_results>
  <result name="toolu_01abc123">
    {"temperature": 22, "condition": "sunny"}
  </result>
</ai00:function_results>

It's 22°C and sunny in Tokyo!
</ai00:assistant>
```

## Stop Sequences

The default stop sequence is `</ai00:assistant>`. This marks the end of the assistant's turn.

For tool calling, the BNF grammar naturally terminates at `</ai00:function_calls>`, triggering `stop_reason: tool_use` in the API response.

## Configuration

Format settings are configurable in `Config.toml` under the `[prompts]` section:

```toml
[prompts]
role_user = "user"
role_assistant = "assistant"
role_system = "system"
assistant_prefix = "<ai00:assistant>\n"
assistant_prefix_thinking = "<ai00:assistant>\n<think>\n"
default_stop_sequences = ["</ai00:assistant>"]
```

## Migration from v0

| v0 Format | v1 Format |
|-----------|-----------|
| `User: content` | `<ai00:user>\ncontent\n</ai00:user>` |
| `Assistant: content` | `<ai00:assistant>\ncontent\n</ai00:assistant>` |
| `System: content` | `<ai00:system>\ncontent\n</ai00:system>` |
| `<tool_call>{"name":...}</tool_call>` | `<ai00:function_calls><invoke name="...">...</invoke></ai00:function_calls>` |

## Recommended Sampling Parameters

From the RWKV7-G1 documentation:

| Use Case | Temperature | Top-P | Presence | Frequency | Decay |
|----------|-------------|-------|----------|-----------|-------|
| Math | 0.3 | 0.3 | 0 | 0 | 0.996 |
| Chat | 1.0 | 0.3 | 0.5 | 0.5 | 0.996 |
| Creative | 0.6 | 0.6-0.8 | 1-2 | 0-0.2 | 0.99 |
