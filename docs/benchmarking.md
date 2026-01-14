# Benchmarking

This document describes the benchmarking tools for measuring ai00_server performance.

## Overview

The benchmark suite measures inference performance across three scenarios:

| Benchmark | Description | Use Case |
|-----------|-------------|----------|
| **Sequential** | Requests run one after another | Single-user latency, baseline performance |
| **Parallel** | All requests run simultaneously | Maximum throughput, batch processing |
| **Cached** | Same prompt with cache warming | Cache efficiency, multi-user same context |

## Quick Start

```bash
# Run sequential benchmark (25 prompts)
./ai00_server/benchmarks/run-benchmark.sh

# Run parallel benchmark (25 concurrent requests)
./ai00_server/benchmarks/run-benchmark-parallel.sh

# Run cached benchmark (warm cache, then 25 parallel)
./ai00_server/benchmarks/run-benchmark-cached.sh

# View results
open http://localhost:8888/
```

## Benchmark Scripts

### Sequential Benchmark

`run-benchmark.sh` - Runs prompts one at a time to measure single-request latency.

```bash
./ai00_server/benchmarks/run-benchmark.sh [OPTIONS]

Options:
  --max-tokens N    Maximum tokens to generate (default: 8192)
  --model NAME      Model name (auto-detected if not set)
  --api-url URL     API endpoint URL
```

**What it measures:**
- Time to first token (TTFT) - prefill latency
- Decode speed (tokens/second)
- Total request latency

**Best for:** Measuring baseline performance without contention.

### Parallel Benchmark

`run-benchmark-parallel.sh` - Runs all prompts simultaneously.

```bash
./ai00_server/benchmarks/run-benchmark-parallel.sh [OPTIONS]

Options:
  --max-tokens N    Maximum tokens to generate (default: 8192)
  --model NAME      Model name (auto-detected if not set)
  --api-url URL     API endpoint URL
```

**What it measures:**
- Total wall-clock time for all requests
- Throughput under load
- Batching efficiency

**Best for:** Understanding how the server handles concurrent load.

### Cached Benchmark

`run-benchmark-cached.sh` - Warms cache with a single request, then runs parallel requests with the same prompt.

```bash
./ai00_server/benchmarks/run-benchmark-cached.sh [OPTIONS]

Options:
  --max-tokens N    Maximum tokens to generate (default: 8192)
  --parallel N      Number of parallel requests (default: 25)
  --prompt TEXT     Custom prompt to use
```

**What it measures:**
- Cache warm time
- Prefill speed with cache hits
- Cache efficiency under parallel load

**Best for:** Multi-user scenarios with shared context (e.g., system prompts).

## Benchmark Prompts

Prompts are stored in `ai00_server/benchmarks/prompts.txt`. The default set includes 25 prompts of varying complexity:

| Category | Count | Description |
|----------|-------|-------------|
| Short Q&A | 5 | Single-sentence answers |
| Medium explanations | 5 | Paragraph-length responses |
| Detailed explanations | 5 | Multi-paragraph technical content |
| Creative writing | 5 | Stories, poems, haikus |
| Analytical tasks | 5 | Compare/contrast, pros/cons |

To customize prompts, edit `prompts.txt`. Lines starting with `#` are comments.

## Viewing Results

### Dashboard

Results are served at `http://localhost:8888/` via nginx. The dashboard shows:

- **Index page**: List of all benchmark runs with summary stats
- **Report page**: Detailed view with charts for each run

### Report Contents

Each report includes:

**Summary Cards:**
- Request count
- Average prefill speed (tok/s)
- Average decode speed (tok/s)
- Time to first token (TTFT)
- Total inference time

**Charts:**
- Prompt size vs prefill duration (scatter)
- Completion size vs decode duration (scatter)
- Throughput by request (bar)
- Time breakdown by request (stacked bar)

**Request Table:**
- Per-request metrics: prompt tokens, completion tokens, prefill time, decode time, throughput

### Log Files

Raw logs are stored in `html/logs/` as JSONL files:

```
html/logs/benchmark-20260114-142914.jsonl
```

Each file contains:
1. **Metadata row** (first line): Benchmark summary
2. **Server logs**: Full ai00_server JSON logs

**Metadata schema:**
```json
{
  "event": "benchmark_metadata",
  "timestamp": "20260114-142914",
  "type": "sequential|parallel|cached",
  "model_name": "rwkv7-g1c-2.9b-...",
  "total_requests": 25,
  "max_tokens": 8192,
  "wall_clock_s": 441.6,
  "avg_prefill_toks": 59.0,
  "avg_decode_toks": 27.9
}
```

## Key Metrics

### Prefill Speed (tok/s)

Tokens processed per second during the prefill (prompt processing) phase:

```
prefill_toks = prompt_tokens / (prefill_ms / 1000)
```

Higher is better. Affected by:
- Prompt length
- Cache hits (dramatic improvement)
- GPU memory bandwidth

### Decode Speed (tok/s)

Tokens generated per second during the decode (generation) phase:

```
decode_toks = output_tokens / (decode_ms / 1000)
```

Higher is better. Affected by:
- Batch size (concurrent requests reduce per-request speed)
- Model size
- GPU compute capacity

### Time to First Token (TTFT)

Time from request start to first token generation. Approximately equals prefill time.

Lower is better. Critical for interactive applications.

## Expected Results

Typical results for RWKV 2.9B on AMD integrated GPU (Radeon 780M):

| Benchmark | Prefill | Decode | Notes |
|-----------|---------|--------|-------|
| Sequential | ~55 tok/s | ~28 tok/s | Baseline |
| Parallel (25) | ~55 tok/s | ~6 tok/s | Decode limited by batching |
| Cached (25) | ~185 tok/s | ~8 tok/s | 3x prefill from cache hits |

## Server Configuration

The benchmark scripts automatically:

1. Restart ai00_server to clear caches
2. Wait for model to load
3. Run the benchmark
4. Save logs with metadata

To ensure clean measurements, each run starts fresh. If you want to measure warm performance, use the cached benchmark.

### Environment Variables

```bash
# Disable pretty logging for JSON output (required for report parsing)
# LOG_PRETTY=1  # Comment out or remove this line

# Set log level
RUST_LOG=ai00_server=debug,ai00_core=debug
```

## Docker Setup

The benchmark reports are served by nginx:

```yaml
# docker-compose.yml
benchmark-reports:
  image: nginx:alpine
  ports:
    - "8888:80"
  volumes:
    - ./html:/usr/share/nginx/html:ro
    - ./html/nginx.conf:/etc/nginx/conf.d/default.conf:ro
```

Start with:
```bash
docker compose up -d benchmark-reports
```

## Troubleshooting

### Reports show "unknown" type

Check that metadata is on a single line in the JSONL:
```bash
head -1 html/logs/benchmark-*.jsonl
```

Should show compact JSON, not pretty-printed.

### Logs not parsing

Ensure `LOG_PRETTY` is unset in docker-compose.yml:
```yaml
environment:
  # - LOG_PRETTY=1  # Must be commented out
  - RUST_LOG=ai00_server=debug,ai00_core=debug
```

### Server not restarting

The scripts use podman. If using docker directly, modify the `restart_server()` function in each script.

## Extending

### Custom Prompts

Edit `ai00_server/benchmarks/prompts.txt`:
```
# My custom prompts
What is the meaning of life?
Write a sonnet about AI.
```

### Custom Benchmark

Create a new script based on the existing ones. Key functions:
- `restart_server()` - Clear caches
- `detect_model()` - Get model name
- `run_request()` - Execute single request
- Save logs with metadata to `html/logs/`
