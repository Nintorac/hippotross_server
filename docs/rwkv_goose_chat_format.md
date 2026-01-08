# RWKV Goose Chat Format

This document describes the chat prompt format for RWKV7-G1 models (the "Goose" series).

## Source

- [RWKV7-G1 HuggingFace](https://huggingface.co/BlinkDL/rwkv7-g1)

## Chat Format Structure

Format: `System: ...\n\nUser: ...\n\nAssistant: ...\n\nUser: ...\n\nAssistant:`

Each turn is separated by `\n\n` (double newline). This is critical - RWKV uses `\n\n` as a "chat round separator" in its pretrain data.

### Roles

| Role | Format |
|------|--------|
| System | `System: content` |
| User | `User: content` |
| Assistant | `Assistant: content` |

## Newline Normalization

**Important**: Replace all `\n\n` within message content with `\n` (single newline).

If message content contains `\n\n`, it will be interpreted as a turn separator and cause premature turn endings. The `normalize_newlines()` function in `handler.rs` handles this automatically.

## Thinking Mode

For models dated 20250922 and newer, thinking mode is supported.

Format: `User: USER_PROMPT think\n\nAssistant: <think`

Variants:
- ` think a bit` - shorter thinking (tier 1: 1024-4095 tokens)
- ` think` - standard thinking (tier 2: 4096-16383 tokens)
- ` think a lot` - extended thinking (tier 3+: 16384+ tokens)

**Important**: Do NOT close the `<think` bracket - the model continues from there.

## Assistant Prefix

| Mode | Prefix |
|------|--------|
| Normal | `Assistant:` |
| Thinking | `Assistant: <think` |

Note: No trailing space after `Assistant:` - the model generates the space.

## Trailing Whitespace

**Critical**: There should not be any whitespace at the end of the prompt.

From the docs: "There should not be any space at the end of your input (strip it) or you will upset the tokenizer and see non-English response."

## Token 0 Prefix

The RWKV docs mention prepending Token ID 0 (`<|rwkv_tokenizer_end_of_text|>`) before prompts due to state initialization issues.

However, ai00_server handles state initialization through its state management system (`state.init()`), so this is typically not needed. The Token 0 requirement applies to raw model usage without proper state initialization.

## Recommended Sampling Parameters

From the official docs:

| Use Case | Temperature | Top-P | Presence | Frequency | Decay |
|----------|-------------|-------|----------|-----------|-------|
| Math | 0.3 | 0.3 | 0 | 0 | 0.996 |
| Chat | 1.0 | 0.3 | 0.5 | 0.5 | 0.996 |
| Creative | 0.6 | 0.6-0.8 | 1-2 | 0-0.2 | 0.99 |

## Configuration

All format settings are configurable in `Config.toml` under the `[prompts]` section. See the comments there for default values.
