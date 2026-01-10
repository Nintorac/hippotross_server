#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb",
# ]
# ///
"""
Convert Toucan-1.5M dataset to MessagesRequest JSONL format.

Streams from HuggingFace parquet files and outputs to stdout.

Usage:
    # Stream from HuggingFace
    ./toucan_to_messages.py > output.jsonl
    ./toucan_to_messages.py --limit 100 > sample.jsonl

    # Stream directly to make-binidx
    ./toucan_to_messages.py --limit 1000 | make-binidx -o train -t tokenizer.json -p Config.toml

    # Show sample conversion
    ./toucan_to_messages.py --sample

    # Use different dataset subset
    ./toucan_to_messages.py --url "hf://datasets/Agent-Ark/Toucan-1.5M/Qwen3-32B/*.parquet"
"""

import ast
import json
import sys
from typing import Iterator

import duckdb

# HuggingFace parquet URL for the SFT subset
DATASET_URL = "hf://datasets/Agent-Ark/Toucan-1.5M/SFT/*.parquet"


def parse_tool_call(content: str) -> dict:
    """Parse tool_call content which is a Python dict-like string."""
    # Content is like: {'name': 'tool_name', 'arguments': '{"arg": "value"}'}
    try:
        parsed = ast.literal_eval(content)
        name = parsed["name"]
        # Arguments is a JSON string inside
        args_str = parsed.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        return {"name": name, "input": args}
    except Exception as e:
        print(f"Warning: Failed to parse tool_call: {e}", file=sys.stderr)
        return {"name": "unknown", "input": {}}


def convert_tools(tools_json: str) -> list[dict]:
    """Convert OpenAI-style tools to Anthropic-style."""
    tools = json.loads(tools_json)
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            result.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object"}),
            })
    return result


def convert_messages(messages_json: str) -> list[dict]:
    """
    Convert Toucan message format to Anthropic MessagesRequest format.

    Toucan roles: user, assistant, tool_call, tool_response
    Anthropic roles: user, assistant (with content blocks)
    """
    messages = json.loads(messages_json)
    result = []
    tool_call_id_counter = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            result.append({"role": "user", "content": content})
            i += 1

        elif role == "assistant":
            result.append({"role": "assistant", "content": content})
            i += 1

        elif role == "tool_call":
            # Collect consecutive tool_calls into one assistant message
            tool_uses = []
            while i < len(messages) and messages[i]["role"] == "tool_call":
                tool_call_id_counter += 1
                tool_id = f"toolu_{tool_call_id_counter:04d}"
                parsed = parse_tool_call(messages[i]["content"])
                tool_uses.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": parsed["name"],
                    "input": parsed["input"],
                })
                i += 1

            result.append({
                "role": "assistant",
                "content": tool_uses,
            })

            # Collect consecutive tool_responses into one user message
            tool_results = []
            response_idx = 0
            while i < len(messages) and messages[i]["role"] == "tool_response":
                # Match response to tool_use by position
                if response_idx < len(tool_uses):
                    tool_id = tool_uses[response_idx]["id"]
                else:
                    tool_id = f"toolu_{tool_call_id_counter + response_idx + 1:04d}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": messages[i]["content"],
                })
                response_idx += 1
                i += 1

            if tool_results:
                result.append({
                    "role": "user",
                    "content": tool_results,
                })

        else:
            # Unknown role, skip
            print(f"Warning: Unknown role '{role}'", file=sys.stderr)
            i += 1

    return result


def stream_rows(
    con: duckdb.DuckDBPyConnection, limit: int | None = None, url: str = DATASET_URL
) -> Iterator[tuple]:
    """Stream rows from the dataset."""
    query = f"SELECT messages, tools FROM '{url}'"
    if limit:
        query += f" LIMIT {limit}"

    # Use fetchmany for streaming
    result = con.execute(query)
    while True:
        batch = result.fetchmany(100)
        if not batch:
            break
        yield from batch


def convert_row(messages_json: str, tools_json: str) -> dict:
    """Convert a single row to MessagesRequest format."""
    messages = convert_messages(messages_json)
    tools = convert_tools(tools_json)

    return {
        "model": "rwkv",
        "messages": messages,
        "tools": tools if tools else None,
        "max_tokens": 4096,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Toucan-1.5M to MessagesRequest JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire SFT subset
  %(prog)s > toucan_sft.jsonl

  # Convert first 1000 rows
  %(prog)s --limit 1000 > sample.jsonl

  # Use Qwen3-32B subset
  %(prog)s --url "hf://datasets/Agent-Ark/Toucan-1.5M/Qwen3-32B/*.parquet"

  # Pipe directly to make-binidx
  %(prog)s --limit 1000 | make-binidx -o train -t tokenizer.json -p Config.toml
        """,
    )
    parser.add_argument("--limit", type=int, help="Limit number of rows to process")
    parser.add_argument("--sample", action="store_true", help="Print sample conversion and exit")
    parser.add_argument(
        "--url",
        default=DATASET_URL,
        help=f"HuggingFace parquet URL (default: {DATASET_URL})",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    con = duckdb.connect()

    if args.sample:
        # Just show a sample conversion
        row = con.execute(f"SELECT messages, tools FROM '{args.url}' LIMIT 1").fetchone()
        result = convert_row(row[0], row[1])
        print(json.dumps(result, indent=2))
        return

    if not args.quiet:
        print(f"Streaming from: {args.url}", file=sys.stderr)

    # Stream and convert
    count = 0
    errors = 0
    for messages_json, tools_json in stream_rows(con, args.limit, args.url):
        try:
            result = convert_row(messages_json, tools_json)
            # Remove None values
            result = {k: v for k, v in result.items() if v is not None}
            print(json.dumps(result))
            count += 1

            # Progress indicator
            if not args.quiet and count % 1000 == 0:
                print(f"  Converted {count} rows...", file=sys.stderr)

        except Exception as e:
            errors += 1
            if not args.quiet:
                print(f"Warning: Failed to convert row {count + errors}: {e}", file=sys.stderr)

    if not args.quiet:
        print(f"Done: {count} converted, {errors} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
