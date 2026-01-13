//! Structured logging types for wide format logging.
//!
//! This module defines event structures for canonical log lines following the
//! wide format logging pattern. Each category of events captures complete context
//! in a single structured log entry.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Get current timestamp in milliseconds since Unix epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Server lifecycle events
pub mod lifecycle {
    /// Emitted at server startup.
    pub fn server_startup(binary: &str, version: &str) {
        tracing::info!(
            event = "server_startup",
            binary = %binary,
            version = %version,
            "Server starting"
        );
    }

    /// Emitted when configuration is loaded.
    pub fn config_loaded(config_path: &str) {
        tracing::info!(
            event = "config_loaded",
            config_path = %config_path,
            "Configuration loaded"
        );
    }

    /// Emitted when a plugin is loaded.
    pub fn plugin_loaded(plugin_name: &str, success: bool) {
        if success {
            tracing::info!(
                event = "plugin_loaded",
                plugin_name = %plugin_name,
                success = true,
                "Plugin loaded successfully"
            );
        } else {
            tracing::warn!(
                event = "plugin_loaded",
                plugin_name = %plugin_name,
                success = false,
                "Plugin failed to load"
            );
        }
    }

    /// Emitted when server binds to address.
    pub fn server_binding(address: &str, tls: bool, acme: bool) {
        tracing::info!(
            event = "server_binding",
            address = %address,
            tls = tls,
            acme = acme,
            "Server binding"
        );
    }

    /// Emitted on server shutdown.
    pub fn server_shutdown(signal: &str) {
        tracing::info!(
            event = "server_shutdown",
            signal = %signal,
            "Server shutting down"
        );
    }
}

/// Model operation events
pub mod model {
    /// Emitted when model loading begins.
    pub fn model_load(path: &str, tokenizer_path: &str, batch_size: usize, chunk_size: usize) {
        tracing::info!(
            event = "model_load",
            path = %path,
            tokenizer_path = %tokenizer_path,
            batch_size = batch_size,
            chunk_size = chunk_size,
            "Loading model"
        );
    }

    /// Emitted with model metadata after parsing.
    pub fn model_metadata(
        version: &str,
        layers: usize,
        embed_size: usize,
        hidden_size: usize,
        vocab_size: usize,
        heads: usize,
    ) {
        tracing::info!(
            event = "model_metadata",
            version = %version,
            layers = layers,
            embed_size = embed_size,
            hidden_size = hidden_size,
            vocab_size = vocab_size,
            heads = heads,
            "Model metadata"
        );
    }

    /// Emitted when model format is detected.
    pub fn model_format(format: &str) {
        tracing::info!(
            event = "model_format",
            format = %format,
            "Model format detected"
        );
    }

    /// Emitted with GPU adapter info.
    pub fn gpu_context(adapter_info: &str) {
        tracing::info!(
            event = "gpu_context",
            adapter_info = %adapter_info,
            "GPU context created"
        );
    }

    /// Emitted when initial state is loaded.
    pub fn state_loaded(path: &str, name: &str, state_id: &str, is_default: bool) {
        tracing::info!(
            event = "state_loaded",
            path = %path,
            name = %name,
            state_id = %state_id,
            is_default = is_default,
            "State loaded"
        );
    }

    /// Emitted when model is unloaded.
    pub fn model_unload() {
        tracing::info!(event = "model_unload", "Model unloaded");
    }

    /// Emitted when model is saved.
    pub fn model_save(output_path: &str) {
        tracing::info!(
            event = "model_save",
            output_path = %output_path,
            "Model saved"
        );
    }
}

/// Request context for accumulating metrics throughout request lifecycle.
#[derive(Debug)]
pub struct RequestContext {
    /// UUID7, generated fresh for this service's request handling.
    pub request_id: String,
    /// From x-request-id header, for cross-service correlation.
    pub trace_id: Option<String>,
    /// Request start time for duration calculation.
    start_time: Instant,
    /// Requested model name.
    pub model: String,
    /// Whether streaming is enabled.
    pub stream: bool,
    /// Maximum tokens limit.
    pub max_tokens: usize,
    /// Whether tool calling is enabled.
    pub has_tools: bool,
    /// Whether thinking mode is enabled.
    pub has_thinking: bool,
    /// Number of input messages.
    pub message_count: usize,
    /// Number of prompt tokens (accumulated).
    pub prompt_tokens: usize,
    /// Number of output tokens (accumulated).
    pub output_tokens: usize,
    /// Finish reason.
    pub finish_reason: String,
}

impl RequestContext {
    /// Create a new request context with generated request_id.
    pub fn new(trace_id: Option<String>) -> Self {
        Self {
            request_id: uuid::Uuid::now_v7().to_string(),
            trace_id,
            start_time: Instant::now(),
            model: String::new(),
            stream: false,
            max_tokens: 0,
            has_tools: false,
            has_thinking: false,
            message_count: 0,
            prompt_tokens: 0,
            output_tokens: 0,
            finish_reason: String::new(),
        }
    }

    /// Record prompt token count.
    pub fn record_prompt_tokens(&mut self, count: usize) {
        self.prompt_tokens = count;
    }

    /// Record output token count.
    pub fn record_output_tokens(&mut self, count: usize) {
        self.output_tokens = count;
    }

    /// Set finish reason.
    pub fn set_finish_reason(&mut self, reason: &str) {
        self.finish_reason = reason.to_string();
    }

    /// Get duration since request start.
    pub fn duration_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Emit the canonical log line for this request.
    pub fn emit_canonical_log(&self) {
        let duration_ms = self.duration_ms();
        let timestamp_ms = crate::logging::now_ms();

        tracing::info!(
            event = "request_complete",
            canonical = true,
            timestamp_ms = timestamp_ms,
            request_id = %self.request_id,
            trace_id = ?self.trace_id,
            model = %self.model,
            stream = self.stream,
            max_tokens = self.max_tokens,
            has_tools = self.has_tools,
            has_thinking = self.has_thinking,
            message_count = self.message_count,
            prompt_tokens = self.prompt_tokens,
            output_tokens = self.output_tokens,
            duration_ms = duration_ms,
            finish_reason = %self.finish_reason,
            "Request completed"
        );
    }

    /// Convert to a StreamLogContext for passing to stream handlers.
    pub fn to_stream_log_context(self) -> StreamLogContext {
        StreamLogContext {
            request_id: self.request_id,
            trace_id: self.trace_id,
            model: self.model,
            max_tokens: self.max_tokens,
            has_tools: self.has_tools,
            has_thinking: self.has_thinking,
            message_count: self.message_count,
        }
    }
}

/// Context passed to stream handlers for logging at stream completion.
#[derive(Debug, Clone)]
pub struct StreamLogContext {
    pub request_id: String,
    pub trace_id: Option<String>,
    pub model: String,
    pub max_tokens: usize,
    pub has_tools: bool,
    pub has_thinking: bool,
    pub message_count: usize,
}

impl StreamLogContext {
    /// Emit canonical log with actual metrics from TokenCounter.
    pub fn emit_with_counter(&self, counter: &ai00_core::TokenCounter, finish_reason: &str) {
        let timestamp_ms = crate::logging::now_ms();

        tracing::info!(
            event = "request_complete",
            canonical = true,
            timestamp_ms = timestamp_ms,
            request_id = %self.request_id,
            trace_id = ?self.trace_id,
            model = %self.model,
            stream = true,
            max_tokens = self.max_tokens,
            has_tools = self.has_tools,
            has_thinking = self.has_thinking,
            message_count = self.message_count,
            prompt_tokens = counter.prompt,
            output_tokens = counter.completion,
            duration_ms = counter.duration.as_millis() as u64,
            finish_reason = %finish_reason,
            "Request completed"
        );
    }
}

/// Inference batch events
pub mod inference {
    /// Emitted after each inference batch completes.
    pub fn batch_complete(
        request_id: &str,
        trace_id: Option<&str>,
        batch: usize,
        prompt_token_count: usize,
        cache_hit_tokens: usize,
        output_token_count: usize,
        prefill_ms: u64,
        decode_ms: u64,
        total_ms: u64,
        finish_reason: &str,
    ) {
        tracing::info!(
            event = "inference_batch",
            request_id = %request_id,
            trace_id = ?trace_id,
            batch = batch,
            prompt_token_count = prompt_token_count,
            cache_hit_tokens = cache_hit_tokens,
            output_token_count = output_token_count,
            prefill_ms = prefill_ms,
            decode_ms = decode_ms,
            total_ms = total_ms,
            finish_reason = %finish_reason,
            "Inference batch complete"
        );
    }
}

/// Debug-level payload logging (enabled via RUST_LOG=ai00_core=debug)
pub mod debug {
    /// Log combined model input and output after inference completes.
    /// This provides a single event with both the prompt and generated text for easier debugging.
    pub fn model_io(
        request_id: &str,
        trace_id: Option<&str>,
        input_prompt: &str,
        input_tokens: usize,
        output_text: &str,
        output_tokens: usize,
    ) {
        tracing::debug!(
            event = "model_io",
            request_id = %request_id,
            trace_id = ?trace_id,
            input_prompt = %input_prompt,
            input_tokens = input_tokens,
            output_text = %output_text,
            output_tokens = output_tokens,
            "Model I/O complete"
        );
    }
}

/// Error events
pub mod errors {
    /// Request validation failed.
    pub fn request_validation(request_id: &str, error: &str) {
        tracing::warn!(
            event = "request_validation_failed",
            request_id = %request_id,
            error = %error,
            "Request validation failed"
        );
    }

    /// Model load failed.
    pub fn model_load_failed(path: &str, error: &str) {
        tracing::error!(
            event = "model_load_failed",
            path = %path,
            error = %error,
            "Model load failed"
        );
    }

    /// State load failed.
    pub fn state_load_failed(path: &str, error: &str) {
        tracing::warn!(
            event = "state_load_failed",
            path = %path,
            error = %error,
            "State load failed"
        );
    }

    /// Slot update failed.
    pub fn slot_update_failed(batch: usize, error: &str) {
        tracing::error!(
            event = "slot_update_failed",
            batch = batch,
            error = %error,
            "Slot update failed"
        );
    }

    /// Token decode failed.
    pub fn token_decode_failed(request_id: &str, token_id: u32, error: &str) {
        tracing::warn!(
            event = "token_decode_failed",
            request_id = %request_id,
            token_id = token_id,
            error = %error,
            "Token decode failed"
        );
    }

    /// JWT encode failed.
    pub fn jwt_encode_failed(error: &str) {
        tracing::warn!(
            event = "jwt_encode_failed",
            error = %error,
            "JWT encoding failed"
        );
    }

    /// File path validation failed.
    pub fn path_validation_failed(path: &str, error: &str) {
        tracing::error!(
            event = "path_validation_failed",
            path = %path,
            error = %error,
            "Path validation failed"
        );
    }

    /// Directory read failed.
    pub fn directory_read_failed(path: &str, error: &str) {
        tracing::error!(
            event = "directory_read_failed",
            path = %path,
            error = %error,
            "Directory read failed"
        );
    }

    /// Unzip operation failed.
    pub fn unzip_failed(source: &str, dest: &str, error: &str) {
        tracing::error!(
            event = "unzip_failed",
            source = %source,
            dest = %dest,
            error = %error,
            "Unzip failed"
        );
    }
}
