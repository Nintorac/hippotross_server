# make-binidx

CLI tool to convert `/v1/messages` API requests (JSONL) to RWKV binidx format for fine-tuning.

## Overview

This tool processes JSONL files containing `MessagesRequest` objects and converts them to the binidx format used by RWKV trainers. It uses the **exact same code path** as the ai00_server to build prompts, ensuring training data is perfectly aligned with inference behavior.

## Features

- **Streaming I/O**: Processes data line-by-line for memory efficiency with large datasets
- **Stdin support**: Auto-detects piped input when `--input` is omitted
- **Text-only mode**: Output formatted prompts for inspection/debugging
- **Exact alignment**: Uses the same prompt-building code as the inference server

## Installation

Build from the workspace root:

```bash
cargo build -p make-binidx --release
```

## Usage

### Generate binidx files

```bash
# From file
make-binidx \
  --input data.jsonl \
  --output training_data \
  --tokenizer assets/tokenizer/rwkv_vocab_v20230424.json \
  --prompts-config assets/configs/Config.toml

# From stdin (auto-detected when piped)
cat data.jsonl | make-binidx \
  --output training_data \
  --tokenizer assets/tokenizer/rwkv_vocab_v20230424.json \
  --prompts-config assets/configs/Config.toml
```

Output:
- `training_data.bin` - Token data (u16 little-endian)
- `training_data.idx` - Document index for random access

### Inspect prompts (text-only mode)

```bash
# View formatted prompts without tokenizing
make-binidx \
  --input data.jsonl \
  --prompts-config assets/configs/Config.toml \
  --text-only

# Custom separator
make-binidx \
  --input data.jsonl \
  --prompts-config assets/configs/Config.toml \
  --text-only \
  --separator "==="
```

### Options

| Option | Required | Description |
|--------|----------|-------------|
| `-i, --input <FILE>` | No* | Input JSONL file. If omitted, reads from stdin (must be piped). |
| `-o, --output <FILE>` | Yes** | Output basename (creates .bin and .idx files) |
| `-t, --tokenizer <FILE>` | Yes** | Path to tokenizer JSON file |
| `-p, --prompts-config <FILE>` | Yes | Path to prompts config TOML file |
| `--ctx-len <N>` | No | Context length for stats (default: 4096) |
| `--text-only` | No | Output prompts as text instead of binidx |
| `--separator <STR>` | No | Separator for text-only mode (default: "---") |

\* Required unless piped from stdin
\*\* Required unless `--text-only` is set

## Input Format

Standard `/v1/messages` request format, one per line:

```jsonl
{"model":"rwkv","system":"You are helpful.","messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}],"max_tokens":1024}
{"model":"rwkv","messages":[{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4."}],"max_tokens":1024}
```

Supports:
- System prompts
- Multi-turn conversations
- Tool use (`tool_use` and `tool_result` content blocks)
- Thinking mode (extended thinking content)
- Mixed content blocks

See `test_data/sample_requests.jsonl` for comprehensive examples.

## Output Format (binidx)

The binidx format consists of two files:

### `.bin` file
Raw token data as concatenated little-endian `u16` values. Each document is followed by token 0 (EOS marker).

### `.idx` file
Binary index for random access:
```
[magic: u64]     = 0x584449 ("IDX")
[version: u64]   = 1
[dtype: u64]     = 3 (uint16)
[num_docs: u64]
[sizes: u32[]]   = token count per document (including EOS)
[offsets: u64[]] = cumulative token offsets
[total: u64]     = total token count
```

## Training Integration

After generating binidx files, use with RWKV trainer:

```bash
# The tool outputs hints for trainer args
make-binidx --input data.jsonl --output train ...

# Example output:
# For RWKV trainer:
#   --my_exit_tokens 150000
#   --magic_prime <largest 3n+2 prime less than 35>
```

## Testing

```bash
# Run all tests
cargo test -p make-binidx

# Run with output
cargo test -p make-binidx -- --nocapture
```

## License

Same as ai00_server (MIT/Apache-2.0).
