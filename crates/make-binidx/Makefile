# make-binidx Makefile
#
# Targets for generating training data from various datasets.

SHELL := /bin/bash

# Paths
TOKENIZER := ../../assets/tokenizer/rwkv_vocab_v20230424.json
CONFIG := ../../assets/configs/Config.toml
OUTPUT_DIR := /workspace/data

# Git info for output naming
GIT_COMMIT := $(shell git rev-parse --short HEAD)
GIT_STATUS := $(shell git diff --quiet && git diff --cached --quiet && echo clean || echo dirty)

# Default max tokens (context length)
MAX_TOKENS ?= 4096

.PHONY: help toucan-sft

help:
	@echo "make-binidx training data generation"
	@echo ""
	@echo "Targets:"
	@echo "  toucan-sft    Generate binidx from Toucan-1.5M SFT dataset"
	@echo ""
	@echo "Options:"
	@echo "  MAX_TOKENS=N  Filter rows exceeding N tokens (default: 4096)"
	@echo "  LIMIT=N       Limit number of rows to process"
	@echo ""
	@echo "Examples:"
	@echo "  make toucan-sft"
	@echo "  make toucan-sft MAX_TOKENS=2048"
	@echo "  make toucan-sft LIMIT=10000"

# Ensure output directory exists
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Build the make-binidx binary
target/release/make-binidx:
	cargo build --release -p make-binidx

# Generate Toucan SFT binidx
toucan-sft: $(OUTPUT_DIR) target/release/make-binidx
	@echo "Generating Toucan SFT binidx..."
	@echo "  Commit: $(GIT_COMMIT)"
	@echo "  Status: $(GIT_STATUS)"
	@echo "  Output: $(OUTPUT_DIR)/toucan_sft_$(GIT_COMMIT)_$(GIT_STATUS)/data.{bin,idx}"
	mkdir -p $(OUTPUT_DIR)/toucan_sft_$(GIT_COMMIT)_$(GIT_STATUS)
	./toucan_to_messages.py $(if $(LIMIT),--limit $(LIMIT),) | \
		../../target/release/make-binidx \
			-o $(OUTPUT_DIR)/toucan_sft_$(GIT_COMMIT)_$(GIT_STATUS)/data \
			-t $(TOKENIZER) \
			-p $(CONFIG) \
			$(if $(MAX_TOKENS),--max-tokens $(MAX_TOKENS),)

