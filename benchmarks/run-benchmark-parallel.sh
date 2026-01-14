#!/usr/bin/env bash
# RWKV AI00 Server Parallel Benchmark Runner
# Restarts server to clear caches, runs ALL inference requests in parallel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROMPTS_FILE="$SCRIPT_DIR/prompts.txt"
LOGS_DIR="$PROJECT_DIR/html/logs"

# Configuration
API_URL="${API_URL:-http://localhost:65530/api/oai/v1/chat/completions}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MODEL="${MODEL:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_deps() {
    local missing=()
    for cmd in curl jq podman; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        exit 1
    fi
}

restart_server() {
    log_info "Restarting ai00-server to clear caches..."
    cd "$PROJECT_DIR"
    podman rm -f ai00-server 2>/dev/null || true
    docker compose up -d --no-deps ai00-server 2>/dev/null

    log_info "Waiting for server to be ready..."
    local max_wait=120
    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        if curl -s "${API_URL%/chat/completions}/models" | jq -e '.data[0].id' &>/dev/null; then
            log_success "Server is ready"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    echo ""
    log_error "Server did not become ready within ${max_wait}s"
    exit 1
}

detect_model() {
    if [[ -n "$MODEL" ]]; then
        echo "$MODEL"
        return
    fi
    local model
    model=$(curl -s "${API_URL%/chat/completions}/models" | jq -r '.data[0].id')
    if [[ -z "$model" || "$model" == "null" ]]; then
        log_error "Could not detect model name"
        exit 1
    fi
    echo "$model"
}

run_request() {
    local prompt="$1"
    local model="$2"
    local request_num="$3"
    local output_file="$4"

    local payload
    payload=$(jq -n \
        --arg model "$model" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$MAX_TOKENS" \
        '{
            model: $model,
            messages: [{role: "user", content: $prompt}],
            max_tokens: $max_tokens
        }')

    local start end elapsed
    start=$(date +%s.%N)

    local response
    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "$payload")

    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    local preview
    preview=$(echo "$response" | jq -r '.choices[0].message.content // "ERROR"' 2>/dev/null | head -c 60 | tr '\n' ' ')

    printf "%d %.2f %s\n" "$request_num" "$elapsed" "$preview" >> "$output_file"
}

run_benchmark() {
    check_deps

    log_info "RWKV AI00 Server PARALLEL Benchmark"
    echo "==========================================="

    restart_server

    local model
    model=$(detect_model)
    log_info "Model: $model"
    log_info "Max tokens: $MAX_TOKENS"
    echo ""

    if [[ ! -f "$PROMPTS_FILE" ]]; then
        log_error "Prompts file not found: $PROMPTS_FILE"
        exit 1
    fi

    local prompts=()
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        prompts+=("$line")
    done < "$PROMPTS_FILE"

    log_info "Launching ${#prompts[@]} parallel requests..."
    echo ""

    local results_file
    results_file=$(mktemp)

    local start_time
    start_time=$(date +%s.%N)

    local pids=()
    local i=1
    for prompt in "${prompts[@]}"; do
        run_request "$prompt" "$model" "$i" "$results_file" &
        pids+=($!)
        i=$((i + 1))
    done

    local total=${#pids[@]}
    local completed=0

    log_info "Waiting for $total requests to complete..."
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
        completed=$((completed + 1))
        printf "\r  Progress: %d/%d " "$completed" "$total"
    done
    echo ""

    local end_time wall_clock
    end_time=$(date +%s.%N)
    wall_clock=$(echo "$end_time - $start_time" | bc)

    echo ""
    log_success "All requests complete in ${wall_clock}s (wall clock)"
    echo ""

    log_info "Results:"
    sort -n "$results_file" | while read -r num elapsed preview; do
        printf "  [%2d] %6.2fs | %s...\n" "$num" "$elapsed" "$preview"
    done

    rm -f "$results_file"
    echo ""

    # Save logs with metadata
    mkdir -p "$LOGS_DIR"
    local timestamp
    timestamp=$(date +%Y%m%d-%H%M%S)
    local log_file="$LOGS_DIR/benchmark-${timestamp}.jsonl"

    log_info "Saving logs to $log_file..."

    local logs
    logs=$(docker logs ai00-server 2>&1)

    local total_requests
    total_requests=$(echo "$logs" | grep -c '"event":"inference_batch"' || echo "0")

    local prefill_sum=0 decode_sum=0 count=0
    while IFS= read -r line; do
        local prompt_tokens prefill_ms output_tokens decode_ms
        prompt_tokens=$(echo "$line" | jq -r '.fields.prompt_tokens // 0')
        prefill_ms=$(echo "$line" | jq -r '.fields.prefill_ms // 0')
        output_tokens=$(echo "$line" | jq -r '.fields.output_tokens // 0')
        decode_ms=$(echo "$line" | jq -r '.fields.decode_ms // 0')

        if [[ "$prefill_ms" != "0" && "$prefill_ms" != "null" ]]; then
            local prefill_toks=$(echo "scale=2; $prompt_tokens / ($prefill_ms / 1000)" | bc)
            prefill_sum=$(echo "$prefill_sum + $prefill_toks" | bc)
        fi
        if [[ "$decode_ms" != "0" && "$decode_ms" != "null" ]]; then
            local decode_toks=$(echo "scale=2; $output_tokens / ($decode_ms / 1000)" | bc)
            decode_sum=$(echo "$decode_sum + $decode_toks" | bc)
        fi
        count=$((count + 1))
    done < <(echo "$logs" | grep '"event":"inference_batch"')

    local avg_prefill avg_decode
    avg_prefill=$(echo "scale=2; $prefill_sum / $count" | bc 2>/dev/null || echo "0")
    avg_decode=$(echo "scale=2; $decode_sum / $count" | bc 2>/dev/null || echo "0")

    jq -cn \
        --arg timestamp "$timestamp" \
        --arg type "parallel" \
        --arg model_name "$model" \
        --argjson total_requests "$total_requests" \
        --argjson max_tokens "$MAX_TOKENS" \
        --argjson wall_clock_s "$(echo "$wall_clock" | bc)" \
        --argjson avg_prefill_toks "$avg_prefill" \
        --argjson avg_decode_toks "$avg_decode" \
        '{
            event: "benchmark_metadata",
            timestamp: $timestamp,
            type: $type,
            model_name: $model_name,
            total_requests: $total_requests,
            max_tokens: $max_tokens,
            wall_clock_s: $wall_clock_s,
            avg_prefill_toks: $avg_prefill_toks,
            avg_decode_toks: $avg_decode_toks
        }' > "$log_file"

    echo "$logs" | grep '^\{' >> "$log_file"

    log_success "Saved: $log_file"
    log_info "View at: http://localhost:8888/"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --api-url) API_URL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Runs all benchmark prompts in PARALLEL to test concurrent performance."
            echo ""
            echo "Options:"
            echo "  --max-tokens N    Maximum tokens to generate (default: 8192)"
            echo "  --model NAME      Model name (auto-detected if not set)"
            echo "  --api-url URL     API endpoint URL"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

run_benchmark
