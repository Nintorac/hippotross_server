# Logging Strategy

This document describes the structured logging implementation for ai00_server, following wide format logging patterns with canonical log lines.

## Overview

### Philosophy

ai00_server uses **wide format logging** where each significant operation emits a single comprehensive log entry containing all relevant context. This contrasts with narrow logging where information is scattered across multiple related entries.

Benefits:
- **Searchability**: Find everything about a request with one query
- **Correlation**: All context in one place eliminates cross-referencing
- **Observability**: Log aggregation tools can directly index structured fields
- **Debugging**: Complete picture without reconstruction from fragments

### Canonical Log Lines

The most important wide event is the **canonical log line** emitted at the end of each API request. It captures:
- Request identification (request_id, trace_id)
- Request metadata (model, stream, max_tokens)
- Feature flags (has_tools, has_thinking)
- Accumulated metrics (prompt_tokens, output_tokens, duration_ms)
- Final state (finish_reason)

## Library Choice

We use the [`tracing`](https://docs.rs/tracing) crate instead of `log` or `slog` for several reasons:

| Feature | tracing | log | slog |
|---------|---------|-----|------|
| Async-native | Yes | No | Partial |
| Structured fields | Built-in | Via extensions | Yes |
| Spans (context) | Yes | No | No |
| Ecosystem | Rich (tokio/axum/tower) | Limited | Moderate |
| JSON output | Native | Via formatter | Yes |
| Compile-time filtering | Yes | Yes | Yes |

The Rust async ecosystem standardized on `tracing`, making it the natural choice for Salvo-based services.

## Event Categories

### Server Lifecycle Events

Events marking server state transitions:

| Event | Level | When | Fields |
|-------|-------|------|--------|
| `server_startup` | INFO | Process start | binary, version |
| `config_loaded` | INFO | Config parsed | config_path |
| `plugin_loaded` | INFO/WARN | Plugin load attempt | plugin_name, success |
| `server_binding` | INFO | Before listen | address, tls, acme |
| `server_shutdown` | INFO | Graceful stop | signal |

### Model Operations Events

Events related to model loading, management, and persistence:

| Event | Level | When | Fields |
|-------|-------|------|--------|
| `model_load` | INFO | Load begins | path, tokenizer_path, batch_size, chunk_size |
| `model_metadata` | INFO | After parsing | version, layers, embed_size, hidden_size, vocab_size, heads |
| `model_format` | INFO | Format detection | format |
| `gpu_context` | INFO | GPU init | adapter_info |
| `state_loaded` | INFO | State file loaded | path, name, state_id, is_default |
| `model_unload` | INFO | Model released | - |
| `model_save` | INFO | Save complete | output_path |

### Request Lifecycle Events

The canonical log line emitted at request completion:

| Event | Level | When | Fields |
|-------|-------|------|--------|
| `request_complete` | INFO | Response done | canonical, request_id, trace_id, model, stream, max_tokens, has_tools, has_thinking, message_count, prompt_tokens, output_tokens, duration_ms, finish_reason |

The `canonical=true` field marks this as the authoritative log line for the request.

### Inference Engine Events

Per-batch inference events for detailed performance analysis:

| Event | Level | When | Fields |
|-------|-------|------|--------|
| `inference_batch` | INFO | Batch done | request_id, trace_id, batch, prompt_token_count, cache_hit_tokens, output_token_count, queue_wait_ms, cache_fetch_us, prefill_ms, decode_ms, total_ms, finish_reason |

#### Timing Breakdown

The `inference_batch` event includes granular timing metrics:

| Field | Description |
|-------|-------------|
| `queue_wait_ms` | Time waiting for an available inference slot (retry loop) |
| `cache_fetch_us` | Time for cache lookup + GPU state load in **microseconds** (`checkout()` + `load()`) |
| `prefill_ms` | Time processing prompt tokens (0 on full cache hit) |
| `decode_ms` | Time for token generation loop |
| `total_ms` | Total time in `process()` (prefill + decode) |

```
┌──────────────┐    ┌───────────┐    ┌─────────────────┐    ┌──────────────┐
│ recv request │───▶│queue_wait │───▶│  cache_fetch    │───▶│   process()  │
│              │    │   _ms     │    │     _us         │    │prefill+decode│
└──────────────┘    └───────────┘    └─────────────────┘    └──────────────┘
```

### Error Events

Error and warning events with context:

| Event | Level | When | Fields |
|-------|-------|------|--------|
| `request_validation_failed` | WARN | Invalid request | request_id, error |
| `model_load_failed` | ERROR | Load failure | path, error |
| `state_load_failed` | WARN | State load error | path, error |
| `slot_update_failed` | ERROR | Inference slot error | batch, error |
| `token_decode_failed` | WARN | Tokenization issue | request_id, token_id, error |
| `jwt_encode_failed` | WARN | Auth error | error |
| `path_validation_failed` | ERROR | Invalid file path | path, error |
| `directory_read_failed` | ERROR | Directory access | path, error |
| `unzip_failed` | ERROR | Unzip error | source, dest, error |

## ID Strategy

Two identifiers enable request correlation:

### trace_id

- **Source**: Extracted from `x-request-id` HTTP header
- **Purpose**: Cross-service correlation (tracing across service boundaries)
- **Type**: Optional - may not be present if caller doesn't provide it
- **Propagation**: Passed through to downstream services

### request_id

- **Source**: Generated as UUID7 by this service
- **Purpose**: Unique identifier for this specific operation
- **Type**: Always present - generated fresh for every request
- **Format**: UUID7 (time-sortable UUID)

This follows OpenTelemetry conventions: `trace_id` is global across services, `request_id` (equivalent to span_id) is local to this service.

Example correlation scenario:
```
Frontend → API Gateway (x-request-id: abc123) → ai00_server
                                                 ├─ trace_id: abc123
                                                 └─ request_id: 01945abc-def0-7123-...
```

## Configuration

### Log Level Filtering (RUST_LOG)

Control log verbosity via the `RUST_LOG` environment variable:

```bash
# Default level
RUST_LOG=ai00_server=info,ai00_core=info

# Enable debug for core inference
RUST_LOG=ai00_core=debug

# Verbose everything
RUST_LOG=debug

# Quiet (only errors)
RUST_LOG=error

# Mixed levels
RUST_LOG=ai00_server=debug,ai00_core=info
```

### Output Format (LOG_PRETTY)

Toggle between JSON (production) and human-readable (development) output:

```bash
# JSON output (default)
# unset LOG_PRETTY

# Human-readable output
LOG_PRETTY=1
```

## Debug Logging

### Raw Model I/O

Enable debug-level logging to capture raw model inputs and outputs:

```bash
RUST_LOG=ai00_core=debug
```

This emits two additional events:

| Event | Level | When | Fields |
|-------|-------|------|--------|
| `model_input` | DEBUG | Before inference | request_id, trace_id, raw_prompt, token_count |
| `model_output` | DEBUG | After inference | request_id, trace_id, raw_output, token_count |

**Security Warning**: Debug logging may contain sensitive user data including:
- Full user prompts
- Model responses
- Personal information in conversations

Only enable debug logging in development or controlled debugging scenarios. Never enable in production with untrusted users.

## Example Output

### Production (JSON mode, default)

Server startup:
```json
{"timestamp":"2026-01-13T10:15:32.123Z","level":"INFO","target":"ai00_server","event":"server_startup","binary":"ai00_server","version":"0.8.3"}
```

Model loaded:
```json
{"timestamp":"2026-01-13T10:15:33.456Z","level":"INFO","target":"ai00_core","event":"model_load","path":"/models/rwkv.st","tokenizer_path":"/models/rwkv_vocab.json","batch_size":4,"chunk_size":256}
```

Request canonical log:
```json
{"timestamp":"2026-01-13T10:15:45.789Z","level":"INFO","target":"ai00_server::api::messages","event":"request_complete","canonical":true,"request_id":"01945abc-def0-7123-4567-890abcdef012","trace_id":"ext-corr-id-from-caller","model":"rwkv-7b","stream":true,"max_tokens":4096,"has_tools":true,"has_thinking":false,"message_count":3,"prompt_tokens":1247,"output_tokens":892,"duration_ms":4521,"finish_reason":"end_turn"}
```

Error event:
```json
{"timestamp":"2026-01-13T10:15:50.123Z","level":"ERROR","target":"ai00_server","event":"model_load_failed","path":"/models/missing.st","error":"file not found"}
```

### Development (Pretty mode, LOG_PRETTY=1)

```
  2026-01-13T10:15:32.123Z  INFO ai00_server
    event: server_startup
    binary: ai00_server
    version: 0.8.3

  2026-01-13T10:15:45.789Z  INFO ai00_server::api::messages
    event: request_complete
    canonical: true
    request_id: 01945abc-def0-7123-4567-890abcdef012
    trace_id: ext-corr-id-from-caller
    model: rwkv-7b
    stream: true
    max_tokens: 4096
    has_tools: true
    has_thinking: false
    message_count: 3
    prompt_tokens: 1247
    output_tokens: 892
    duration_ms: 4521
    finish_reason: end_turn
```

## Querying Logs

### Finding Requests

By request_id (this service's span):
```bash
jq 'select(.request_id == "01945abc-...")' logs.jsonl
```

By trace_id (cross-service correlation):
```bash
jq 'select(.trace_id == "ext-corr-id")' logs.jsonl
```

### Performance Analysis

Slow requests (>5 seconds):
```bash
jq 'select(.event == "request_complete" and .duration_ms > 5000)' logs.jsonl
```

Token-heavy requests:
```bash
jq 'select(.event == "request_complete" and .prompt_tokens > 10000)' logs.jsonl
```

### Error Investigation

All errors for a request:
```bash
jq 'select(.request_id == "01945abc-..." and .level == "ERROR")' logs.jsonl
```

Specific error types:
```bash
jq 'select(.event == "model_load_failed")' logs.jsonl
```

## Manual Testing

### Test 1: Baseline Request

Verify basic timing metrics:

```bash
curl -s http://localhost:65530/api/oai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rwkv", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10}'
docker logs ai00-server --tail 3 | grep inference_batch
```

Expected: `cache_hit_tokens=0`, `prefill_ms>0`, `cache_fetch_us` in single digits.

### Test 2: Cache Hit

Verify prefill drops to 0 on cache hit:

```bash
# Use a long prompt (>128 tokens) for reliable caching
PROMPT="You are an expert assistant. Here is background context about AI history dating back to ancient times... [long prompt]. What year was Dartmouth?"

# Request 1 - cache miss
curl -s localhost:65530/api/oai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"rwkv\", \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}], \"max_tokens\": 20}"

sleep 1

# Request 2 - cache hit (same prompt)
curl -s localhost:65530/api/oai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"rwkv\", \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}], \"max_tokens\": 20}"

docker logs ai00-server --tail 4 | grep inference_batch
```

Expected:
- Request 1: `cache_hit_tokens=0`, `prefill_ms=~700-800`
- Request 2: `cache_hit_tokens=211`, `prefill_ms=0`

### Test 3: Queue Pressure

Verify queue_wait_ms increases under load (more requests than slots):

```bash
# Send 12 parallel requests with 6 slots available
for i in {1..12}; do
  curl -s localhost:65530/api/oai/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"rwkv\", \"messages\": [{\"role\": \"user\", \"content\": \"Count to $i\"}], \"max_tokens\": 30}" &
done
wait
docker logs ai00-server --tail 12 | grep inference_batch
```

Expected: First 6 requests show `queue_wait_ms=0-1`, second 6 show `queue_wait_ms>1000` (waiting for slots).

## Implementation Details

### Source Files

- `ai00_server/crates/ai00-server/src/logging.rs` - Event types and helpers
- `ai00_server/crates/ai00-server/src/main.rs` - Logger initialization
- `ai00_server/crates/ai00-server/src/api/request_id.rs` - RequestContext management

### Adding New Events

1. Add a helper function in the appropriate submodule of `logging.rs`:
   ```rust
   pub fn my_event(field1: &str, field2: usize) {
       tracing::info!(
           event = "my_event",
           field1 = %field1,
           field2 = field2,
           "Description of event"
       );
   }
   ```

2. Call the helper at the appropriate location:
   ```rust
   use crate::logging;
   logging::my_module::my_event(&value1, value2);
   ```

### Field Formatting

- Use `%field` for Display formatting (most strings/numbers)
- Use `?field` for Debug formatting (Option types, complex structures)
- Use bare `field` for primitive types that implement Value trait
