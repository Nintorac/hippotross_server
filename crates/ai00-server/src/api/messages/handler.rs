//! Request handlers for Claude-compatible Messages API.

use std::sync::Arc;

use ai00_core::{GenerateRequest, ThreadRequest, Token, MAX_TOKENS};
use futures_util::StreamExt;
use salvo::{oapi::extract::JsonBody, prelude::*, sse::SseEvent};
use tokio::sync::RwLock;

use super::bnf_generator::generate_bnf_schema;
use super::bnf_grammars::wrap_grammar_with_thinking;
use super::streaming::*;
use super::thinking_extractor::{
    generate_thinking_signature, ThinkingExtractor, ThinkingStreamParser,
};
use super::tool_parser::ToolParser;
use super::types::{
    generate_tool_system_prompt, BnfValidationLevel, ContentBlock, MessageParam, MessageRole,
    MessagesRequest, MessagesResponse, StopReason, ThinkingConfig, Tool,
};
use crate::{
    api::{error::ApiErrorResponse, request_info},
    config::{Config, PromptsConfig},
    types::ThreadSender,
    SLEEP,
};

use ai00_core::sampler::nucleus::{NucleusParams, NucleusSampler};

/// Collapse multiple consecutive newlines into a single newline.
///
/// RWKV uses `\n\n` as a "chat round separator" in pretrain data, so we must
/// avoid having `\n\n` within message content to prevent premature turn endings.
///
/// See: https://huggingface.co/BlinkDL/rwkv7-g1
fn normalize_newlines(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_newline = false;

    for c in text.chars() {
        if c == '\n' {
            if !prev_was_newline {
                result.push('\n');
            }
            prev_was_newline = true;
        } else {
            result.push(c);
            prev_was_newline = false;
        }
    }
    result
}

/// Build RWKV prompt from messages.
///
/// Format for RWKV7-G1 (https://huggingface.co/BlinkDL/rwkv7-g1):
/// ```text
/// System: SYSTEM_PROMPT
///
/// User: USER_MESSAGE
///
/// Assistant: ASSISTANT_MESSAGE
///
/// User: USER_MESSAGE
///
/// Assistant:
/// ```
///
/// For thinking mode (RWKV 20250922+ models):
/// ```text
/// User: USER_PROMPT think
///
/// Assistant: <think
/// ```
/// With variants "think a bit" (shorter) and "think a lot" (longer).
fn build_prompt(
    system: Option<&str>,
    messages: &[MessageParam],
    tools: Option<&[Tool]>,
    thinking: Option<&ThinkingConfig>,
    prompts: &PromptsConfig,
) -> String {
    let mut prompt = String::new();

    // Add system prompt first (from top-level param, not message role)
    // Normalize newlines to prevent \n\n from being interpreted as turn separator
    if let Some(sys) = system {
        let normalized_sys = normalize_newlines(sys);
        prompt.push_str(&format!("{}: {}", prompts.role_system, normalized_sys));

        // Inject tool definitions into system prompt if provided
        if let Some(tools) = tools {
            if !tools.is_empty() {
                prompt.push_str(&generate_tool_system_prompt(
                    tools,
                    Some(&prompts.tool_header),
                    Some(&prompts.tool_footer),
                ));
            }
        }

        prompt.push_str("\n\n");
    } else if let Some(tools) = tools {
        // If no system prompt but tools provided, create one for tools
        if !tools.is_empty() {
            prompt.push_str(&format!("{}:", prompts.role_system));
            prompt.push_str(&generate_tool_system_prompt(
                tools,
                Some(&prompts.tool_header),
                Some(&prompts.tool_footer),
            ));
            prompt.push_str("\n\n");
        }
    }

    // Format conversation messages
    // Normalize newlines in content to prevent \n\n from being interpreted as turn separator
    let msg_count = messages.len();
    for (i, msg) in messages.iter().enumerate() {
        let role = match msg.role {
            MessageRole::User => &prompts.role_user,
            MessageRole::Assistant => &prompts.role_assistant,
        };
        let content = normalize_newlines(&msg.content.to_text());

        // For the last user message when thinking is enabled, append think suffix
        let is_last_user = i == msg_count - 1 && msg.role == MessageRole::User;
        let think_suffix = if is_last_user {
            get_thinking_suffix(thinking, prompts)
        } else {
            ""
        };

        prompt.push_str(&format!("{}: {}{}\n\n", role, content, think_suffix));
    }

    // Add assistant prefix for generation
    if thinking.map(|t| t.is_enabled()).unwrap_or(false) {
        // Thinking mode: use configurable thinking prefix
        prompt.push_str(&prompts.assistant_prefix_thinking);
    } else {
        prompt.push_str(&prompts.assistant_prefix);
    }

    // RWKV requires no trailing whitespace or tokenizer may produce non-English output
    // See: https://huggingface.co/BlinkDL/rwkv7-g1
    prompt.trim_end().to_string()
}

/// Get the thinking suffix to append to user message based on budget.
fn get_thinking_suffix<'a>(thinking: Option<&ThinkingConfig>, prompts: &'a PromptsConfig) -> &'a str {
    match thinking {
        Some(ThinkingConfig::Enabled { budget_tokens }) => {
            // Map budget to thinking intensity
            match *budget_tokens {
                0..=4095 => &prompts.thinking_suffix_short,      // Tier 1: shorter thinking
                4096..=16383 => &prompts.thinking_suffix_standard,  // Tier 2: standard thinking
                _ => &prompts.thinking_suffix_extended,          // Tier 3+: extended thinking
            }
        }
        _ => "",
    }
}

/// Determine the effective BNF validation level and schema.
///
/// Logic:
/// 1. If `bnf_validation` is explicitly set, use that level
/// 2. If `bnf_validation` is None and tools/thinking present, auto-enable Structural
/// 3. If raw `bnf_schema` is provided, use that (only when validation is None)
///
/// Returns (effective_level, schema_to_use).
fn resolve_bnf_config(req: &MessagesRequest) -> (BnfValidationLevel, Option<String>) {
    let has_tools = req.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
    let has_thinking = req
        .thinking
        .as_ref()
        .map(|t| t.is_enabled())
        .unwrap_or(false);

    // Determine effective validation level
    let effective_level = match req.bnf_validation {
        // Explicitly set - use that
        Some(level) => level,
        // Not set - auto-enable Structural if tools/thinking present
        None => {
            if has_tools || has_thinking {
                // Auto-enable Structural validation for structured output
                BnfValidationLevel::Structural
            } else {
                BnfValidationLevel::None
            }
        }
    };

    // Determine schema to use
    let schema = match effective_level {
        BnfValidationLevel::None => {
            // Use raw bnf_schema if provided
            // If thinking is enabled with raw schema, wrap it with thinking support
            match (&req.bnf_schema, has_thinking) {
                (Some(user_schema), true) => {
                    // Wrap user's grammar to allow thinking block first
                    Some(wrap_grammar_with_thinking(user_schema))
                }
                (Some(user_schema), false) => {
                    // Use raw schema as-is
                    Some(user_schema.clone())
                }
                (None, _) => None,
            }
        }
        BnfValidationLevel::Structural | BnfValidationLevel::SchemaAware => {
            // Generate grammar based on validation level
            generate_bnf_schema(req.tools.as_deref(), has_thinking, effective_level)
        }
    };

    (effective_level, schema)
}

/// Convert MessagesRequest to GenerateRequest.
fn to_generate_request(req: &MessagesRequest, prompts: &PromptsConfig) -> GenerateRequest {
    let prompt = build_prompt(
        req.system.as_deref(),
        &req.messages,
        req.tools.as_deref(),
        req.thinking.as_ref(),
        prompts,
    );

    // Extract model text from previous assistant messages
    let model_text = req
        .messages
        .iter()
        .filter(|m| m.role == MessageRole::Assistant)
        .map(|m| m.content.to_text())
        .collect::<Vec<_>>()
        .join("\n\n");

    let max_tokens = req.max_tokens.min(MAX_TOKENS);

    let stop = req
        .stop_sequences
        .clone()
        .unwrap_or_else(|| prompts.default_stop_sequences.clone());

    // Build sampler from request parameters
    let temperature = req.temperature.unwrap_or(1.0);
    let top_p = req.top_p.unwrap_or(0.5);
    let top_k = req.top_k.unwrap_or(128);

    let sampler = Arc::new(RwLock::new(NucleusSampler::new(NucleusParams {
        top_p,
        top_k,
        temperature,
        ..Default::default()
    })));

    // Resolve BNF validation level and get effective schema
    let (_effective_level, bnf_schema) = resolve_bnf_config(req);

    GenerateRequest {
        prompt,
        model_text,
        max_tokens,
        stop,
        sampler,
        bnf_schema,
        ..Default::default()
    }
}

/// Validate the messages request.
fn validate_request(req: &MessagesRequest) -> Result<(), ApiErrorResponse> {
    // Validate model is provided
    if req.model.is_empty() {
        return Err(
            ApiErrorResponse::invalid_request("model is required")
                .with_param("model"),
        );
    }

    // Validate messages array
    if req.messages.is_empty() {
        return Err(ApiErrorResponse::invalid_request("messages cannot be empty"));
    }

    // First message must be from user (Claude API requirement)
    if req.messages.first().map(|m| m.role) != Some(MessageRole::User) {
        return Err(
            ApiErrorResponse::invalid_request("first message must have role 'user'")
                .with_param("messages.0.role"),
        );
    }

    // Validate max_tokens
    if req.max_tokens == 0 {
        return Err(
            ApiErrorResponse::invalid_request("max_tokens must be greater than 0")
                .with_param("max_tokens"),
        );
    }

    // Validate temperature range
    if let Some(temp) = req.temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err(
                ApiErrorResponse::invalid_request("temperature must be between 0.0 and 2.0")
                    .with_param("temperature"),
            );
        }
    }

    // Validate top_p range
    if let Some(top_p) = req.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(
                ApiErrorResponse::invalid_request("top_p must be between 0.0 and 1.0")
                    .with_param("top_p"),
            );
        }
    }

    // Validate top_k if provided
    if let Some(top_k) = req.top_k {
        if top_k == 0 {
            return Err(
                ApiErrorResponse::invalid_request("top_k must be greater than 0")
                    .with_param("top_k"),
            );
        }
    }

    // Validate stop_sequences if provided
    if let Some(ref stop_seqs) = req.stop_sequences {
        if stop_seqs.len() > 8 {
            return Err(
                ApiErrorResponse::invalid_request("stop_sequences cannot have more than 8 items")
                    .with_param("stop_sequences"),
            );
        }
        for (i, seq) in stop_seqs.iter().enumerate() {
            if seq.is_empty() {
                return Err(
                    ApiErrorResponse::invalid_request("stop_sequences cannot contain empty strings")
                        .with_param(format!("stop_sequences.{}", i)),
                );
            }
        }
    }

    // Validate thinking configuration if provided
    if let Some(ref thinking) = req.thinking {
        if let Err(msg) = thinking.validate(req.max_tokens) {
            return Err(
                ApiErrorResponse::invalid_request(msg).with_param("thinking.budget_tokens"),
            );
        }
    }

    // Validate bnf_schema if provided (raw grammar mode)
    if let Some(ref schema) = req.bnf_schema {
        if schema.trim().is_empty() {
            return Err(
                ApiErrorResponse::invalid_request("bnf_schema cannot be empty")
                    .with_param("bnf_schema"),
            );
        }
        // Note: bnf_schema + thinking is now supported via wrap_grammar_with_thinking()
        // which automatically prepends thinking block support to user grammars
    }

    // Validate bnf_validation if provided
    if let Some(ref level) = req.bnf_validation {
        if level.is_enabled() {
            // Structural/SchemaAware require tools or thinking to be useful
            // (otherwise there's nothing to constrain structurally)
            let has_tools = req.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
            let has_thinking = req
                .thinking
                .as_ref()
                .map(|t| t.is_enabled())
                .unwrap_or(false);

            if !has_tools && !has_thinking && req.bnf_schema.is_none() {
                // Warn but don't error - user might know what they're doing
                // In the future, SchemaAware could be used for JSON mode without tools
            }
        }
    }

    Ok(())
}

/// Handle non-streaming messages request.
async fn respond_one(
    depot: &mut Depot,
    request: MessagesRequest,
    res: &mut Response,
) -> Result<(), ApiErrorResponse> {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let config = depot.obtain::<Config>().unwrap();
    let prompts = &config.prompts;

    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let gen_request = Box::new(to_generate_request(&request, prompts));
    let _ = sender.send(ThreadRequest::Generate {
        request: gen_request,
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let mut token_counter = ai00_core::TokenCounter::default();
    let mut finish_reason = ai00_core::FinishReason::Null;
    let mut text = String::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Start => {}
            Token::Content(token) => {
                text += &token;
            }
            Token::Stop(reason, counter) => {
                finish_reason = reason;
                token_counter = counter;
                break;
            }
            Token::Done => break,
            _ => {}
        }
    }

    // Check if thinking is enabled
    let thinking_enabled = request
        .thinking
        .as_ref()
        .map(|t| t.is_enabled())
        .unwrap_or(false);

    // Check if tools are enabled
    let has_tools = request
        .tools
        .as_ref()
        .map(|t| !t.is_empty())
        .unwrap_or(false);

    // Extract thinking if enabled (do this before tool parsing)
    let (thinking_block, text_for_parsing) = if thinking_enabled {
        let extractor = ThinkingExtractor::new();
        let result = extractor.extract(&text);

        let thinking_block = result.thinking.map(|thinking| {
            let signature = generate_thinking_signature(&thinking);
            ContentBlock::Thinking { thinking, signature }
        });

        (thinking_block, result.response)
    } else {
        (None, text)
    };

    let (content, stop_reason) = if has_tools {
        // Parse the output for tool_call blocks
        let mut parser = ToolParser::new();
        let result = parser.feed(&text_for_parsing);
        let final_result = parser.finalize();

        let mut content_blocks: Vec<ContentBlock> = Vec::new();

        // Add thinking block first (if any)
        if let Some(block) = thinking_block {
            content_blocks.push(block);
        }

        // Add text content if any
        let text_content = result
            .text
            .unwrap_or_default()
            + &final_result.text.unwrap_or_default();
        let trimmed_text = text_content.trim();
        if !trimmed_text.is_empty() {
            content_blocks.push(ContentBlock::Text {
                text: trimmed_text.to_string(),
            });
        }

        // Add tool_use blocks
        let mut all_tools: Vec<_> = result.tool_uses;
        all_tools.extend(final_result.tool_uses);

        for tool_use in all_tools.iter() {
            content_blocks.push(ContentBlock::ToolUse {
                id: tool_use.id.clone(),
                name: tool_use.name.clone(),
                input: tool_use.input.clone(),
            });
        }

        // Determine stop reason
        let stop_reason = if !all_tools.is_empty() {
            StopReason::ToolUse
        } else {
            finish_reason.into()
        };

        (content_blocks, stop_reason)
    } else {
        // Simple text response (possibly with thinking)
        let mut content_blocks: Vec<ContentBlock> = Vec::new();

        // Add thinking block first (if any)
        if let Some(block) = thinking_block {
            content_blocks.push(block);
        }

        // Add text content
        let trimmed = text_for_parsing.trim();
        if !trimmed.is_empty() {
            content_blocks.push(ContentBlock::Text {
                text: trimmed.to_string(),
            });
        }

        (content_blocks, finish_reason.into())
    };

    let response =
        MessagesResponse::new(model_name, content, token_counter.into()).with_stop_reason(stop_reason);

    res.render(Json(response));
    Ok(())
}

/// Handle streaming messages request with Claude-style SSE events.
async fn respond_stream(depot: &mut Depot, request: MessagesRequest, res: &mut Response) {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let config = depot.obtain::<Config>().unwrap();
    let prompts = &config.prompts;

    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let gen_request = Box::new(to_generate_request(&request, prompts));
    let _ = sender.send(ThreadRequest::Generate {
        request: gen_request,
        tokenizer: info.tokenizer.clone(),
        sender: token_sender,
    });

    // Generate message ID
    let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    // Estimate input tokens (rough approximation)
    let input_tokens = request
        .messages
        .iter()
        .map(|m| m.content.to_text().len() / 4)
        .sum::<usize>()
        + request.system.as_ref().map(|s| s.len() / 4).unwrap_or(0);

    // Check if tools and thinking are enabled
    let has_tools = request
        .tools
        .as_ref()
        .map(|t| !t.is_empty())
        .unwrap_or(false);
    let has_thinking = request
        .thinking
        .as_ref()
        .map(|t| t.is_enabled())
        .unwrap_or(false);

    match (has_thinking, has_tools) {
        (true, false) => {
            // Thinking-aware streaming
            respond_stream_with_thinking(res, token_receiver, message_id, model_name, input_tokens)
                .await;
        }
        (false, true) => {
            // Tool-aware streaming with ToolParser
            respond_stream_with_tools(res, token_receiver, message_id, model_name, input_tokens)
                .await;
        }
        (true, true) => {
            // Both thinking and tools: use tool-aware streaming
            // (thinking extraction in tool mode is handled during finalization)
            respond_stream_with_tools(res, token_receiver, message_id, model_name, input_tokens)
                .await;
        }
        (false, false) => {
            // Simple streaming without parsing
            respond_stream_simple(res, token_receiver, message_id, model_name, input_tokens).await;
        }
    }
}

/// Simple streaming handler without tool parsing.
async fn respond_stream_simple(
    res: &mut Response,
    token_receiver: flume::Receiver<Token>,
    message_id: String,
    model_name: String,
    input_tokens: usize,
) {
    let mut output_tokens = 0usize;
    let mut start_token = true;

    let stream =
        token_receiver
            .into_stream()
            .map(move |token| -> Result<SseEvent, std::convert::Infallible> {
                match token {
                    Token::Start => Ok(emit_message_start(
                        message_id.clone(),
                        model_name.clone(),
                        input_tokens,
                    )),
                    Token::Content(text) => {
                        output_tokens += 1;

                        if start_token {
                            start_token = false;
                            let trimmed = text.trim_start().to_string();
                            if trimmed.is_empty() {
                                return Ok(emit_content_block_start_text(0));
                            }
                            return Ok(emit_text_delta(0, trimmed));
                        }

                        if text.is_empty() {
                            Ok(emit_ping())
                        } else {
                            Ok(emit_text_delta(0, text))
                        }
                    }
                    Token::Stop(reason, _counter) => {
                        let stop_reason: StopReason = reason.into();
                        Ok(emit_message_delta(stop_reason, output_tokens))
                    }
                    Token::Done => Ok(emit_message_stop()),
                    _ => Ok(emit_ping()),
                }
            });

    salvo::sse::stream(res, stream);
}

/// Streaming handler with thinking parsing.
/// Detects <think>...</think> blocks and emits thinking_delta/signature_delta events.
async fn respond_stream_with_thinking(
    res: &mut Response,
    token_receiver: flume::Receiver<Token>,
    message_id: String,
    model_name: String,
    input_tokens: usize,
) {
    use std::cell::RefCell;

    // Shared state for the streaming handler
    struct StreamState {
        parser: ThinkingStreamParser,
        output_tokens: usize,
        thinking_block_index: usize,
        text_block_index: usize,
        thinking_block_started: bool,
        text_block_started: bool,
        message_started: bool,
    }

    let state = RefCell::new(StreamState {
        parser: ThinkingStreamParser::new(),
        output_tokens: 0,
        thinking_block_index: 0,
        text_block_index: 1, // Text block comes after thinking
        thinking_block_started: false,
        text_block_started: false,
        message_started: false,
    });

    let stream = token_receiver.into_stream().flat_map(move |token| {
        let mut events: Vec<Result<SseEvent, std::convert::Infallible>> = Vec::new();
        let mut state = state.borrow_mut();

        match token {
            Token::Start => {
                state.message_started = true;
                events.push(Ok(emit_message_start(
                    message_id.clone(),
                    model_name.clone(),
                    input_tokens,
                )));
            }
            Token::Content(text) => {
                state.output_tokens += 1;

                // Feed token to parser
                let result = state.parser.feed(&text);

                // Emit thinking content if any
                if let Some(thinking_text) = result.thinking {
                    // Start thinking block if needed
                    if !state.thinking_block_started {
                        events.push(Ok(emit_content_block_start_thinking(
                            state.thinking_block_index,
                        )));
                        state.thinking_block_started = true;
                    }
                    events.push(Ok(emit_thinking_delta(
                        state.thinking_block_index,
                        thinking_text,
                    )));
                }

                // Check if thinking just completed
                if result.thinking_complete && state.thinking_block_started {
                    // Emit signature
                    let signature = generate_thinking_signature(state.parser.thinking_content());
                    events.push(Ok(emit_signature_delta(
                        state.thinking_block_index,
                        signature,
                    )));
                    // Close thinking block
                    events.push(Ok(emit_content_block_stop(state.thinking_block_index)));
                }

                // Emit text content if any
                if let Some(text_content) = result.text {
                    if !text_content.is_empty() {
                        // Start text block if needed
                        if !state.text_block_started {
                            events.push(Ok(emit_content_block_start_text(state.text_block_index)));
                            state.text_block_started = true;
                        }
                        events.push(Ok(emit_text_delta(state.text_block_index, text_content)));
                    }
                }
            }
            Token::Stop(reason, _counter) => {
                // Finalize parser
                let final_result = state.parser.finalize();

                // Emit any remaining thinking
                if let Some(thinking_text) = final_result.thinking {
                    if !state.thinking_block_started {
                        events.push(Ok(emit_content_block_start_thinking(
                            state.thinking_block_index,
                        )));
                        state.thinking_block_started = true;
                    }
                    events.push(Ok(emit_thinking_delta(
                        state.thinking_block_index,
                        thinking_text,
                    )));
                }

                // Close thinking block if still open
                if final_result.thinking_complete && state.thinking_block_started {
                    let signature = generate_thinking_signature(state.parser.thinking_content());
                    events.push(Ok(emit_signature_delta(
                        state.thinking_block_index,
                        signature,
                    )));
                    events.push(Ok(emit_content_block_stop(state.thinking_block_index)));
                }

                // Emit any remaining text
                if let Some(text_content) = final_result.text {
                    if !text_content.is_empty() {
                        if !state.text_block_started {
                            events.push(Ok(emit_content_block_start_text(state.text_block_index)));
                            state.text_block_started = true;
                        }
                        events.push(Ok(emit_text_delta(state.text_block_index, text_content)));
                    }
                }

                // Close text block if open
                if state.text_block_started {
                    events.push(Ok(emit_content_block_stop(state.text_block_index)));
                }

                // Emit message delta
                let stop_reason: StopReason = reason.into();
                events.push(Ok(emit_message_delta(stop_reason, state.output_tokens)));
            }
            Token::Done => {
                events.push(Ok(emit_message_stop()));
            }
            _ => {
                events.push(Ok(emit_ping()));
            }
        }

        futures_util::stream::iter(events)
    });

    salvo::sse::stream(res, stream);
}

/// Streaming handler with tool parsing.
/// Detects <tool_call> blocks and emits tool_use content blocks.
async fn respond_stream_with_tools(
    res: &mut Response,
    token_receiver: flume::Receiver<Token>,
    message_id: String,
    model_name: String,
    input_tokens: usize,
) {
    use std::cell::RefCell;

    // Shared state for the streaming handler
    struct StreamState {
        parser: ToolParser,
        output_tokens: usize,
        content_block_index: usize,
        text_block_started: bool,
        message_started: bool,
    }

    let state = RefCell::new(StreamState {
        parser: ToolParser::new(),
        output_tokens: 0,
        content_block_index: 0,
        text_block_started: false,
        message_started: false,
    });

    let stream = token_receiver.into_stream().flat_map(move |token| {
        let mut events: Vec<Result<SseEvent, std::convert::Infallible>> = Vec::new();
        let mut state = state.borrow_mut();

        match token {
            Token::Start => {
                state.message_started = true;
                events.push(Ok(emit_message_start(
                    message_id.clone(),
                    model_name.clone(),
                    input_tokens,
                )));
            }
            Token::Content(text) => {
                state.output_tokens += 1;

                // Feed token to parser
                let result = state.parser.feed(&text);

                // Emit text content if any
                if let Some(text_content) = result.text {
                    if !text_content.is_empty() {
                        // Start text block if needed
                        if !state.text_block_started {
                            events.push(Ok(emit_content_block_start_text(
                                state.content_block_index,
                            )));
                            state.text_block_started = true;
                        }
                        events.push(Ok(emit_text_delta(
                            state.content_block_index,
                            text_content,
                        )));
                    }
                }

                // Emit completed tool uses
                for tool_use in result.tool_uses {
                    // Close text block if open
                    if state.text_block_started {
                        events.push(Ok(emit_content_block_stop(state.content_block_index)));
                        state.content_block_index += 1;
                        state.text_block_started = false;
                    }

                    // Emit tool_use block
                    events.push(Ok(emit_content_block_start_tool_use(
                        state.content_block_index,
                        tool_use.id,
                        tool_use.name,
                    )));

                    // Emit the input JSON as a single delta
                    let input_json = serde_json::to_string(&tool_use.input).unwrap_or_default();
                    events.push(Ok(emit_input_json_delta(
                        state.content_block_index,
                        input_json,
                    )));

                    // Close tool_use block
                    events.push(Ok(emit_content_block_stop(state.content_block_index)));
                    state.content_block_index += 1;
                }
            }
            Token::Stop(reason, _counter) => {
                // Finalize parser
                let final_result = state.parser.finalize();

                // Emit any remaining text
                if let Some(text_content) = final_result.text {
                    if !text_content.is_empty() {
                        if !state.text_block_started {
                            events.push(Ok(emit_content_block_start_text(
                                state.content_block_index,
                            )));
                            state.text_block_started = true;
                        }
                        events.push(Ok(emit_text_delta(
                            state.content_block_index,
                            text_content,
                        )));
                    }
                }

                // Emit any remaining tool uses
                for tool_use in final_result.tool_uses {
                    if state.text_block_started {
                        events.push(Ok(emit_content_block_stop(state.content_block_index)));
                        state.content_block_index += 1;
                        state.text_block_started = false;
                    }

                    events.push(Ok(emit_content_block_start_tool_use(
                        state.content_block_index,
                        tool_use.id,
                        tool_use.name,
                    )));
                    let input_json = serde_json::to_string(&tool_use.input).unwrap_or_default();
                    events.push(Ok(emit_input_json_delta(
                        state.content_block_index,
                        input_json,
                    )));
                    events.push(Ok(emit_content_block_stop(state.content_block_index)));
                    state.content_block_index += 1;
                }

                // Close any open text block
                if state.text_block_started {
                    events.push(Ok(emit_content_block_stop(state.content_block_index)));
                }

                // Determine stop reason
                let stop_reason = if state.parser.has_tool_use() {
                    StopReason::ToolUse
                } else {
                    reason.into()
                };

                events.push(Ok(emit_message_delta(stop_reason, state.output_tokens)));
            }
            Token::Done => {
                events.push(Ok(emit_message_stop()));
            }
            _ => {
                events.push(Ok(emit_ping()));
            }
        }

        futures_util::stream::iter(events)
    });

    salvo::sse::stream(res, stream);
}

/// Generate messages completion (Claude-compatible).
///
/// This endpoint provides Claude Messages API compatibility for RWKV models.
#[endpoint(
    tags("messages"),
    responses(
        (status_code = 200, description = "Successful completion", body = MessagesResponse),
        (status_code = 400, description = "Invalid request", body = ApiErrorResponse),
        (status_code = 500, description = "Server error", body = ApiErrorResponse),
    )
)]
pub async fn messages_handler(
    depot: &mut Depot,
    req: JsonBody<MessagesRequest>,
    res: &mut Response,
) {
    let request = req.0;

    // Validate request
    if let Err(err) = validate_request(&request) {
        res.status_code(err.status_code());
        res.render(Json(err));
        return;
    }

    match request.stream {
        true => respond_stream(depot, request, res).await,
        false => {
            if let Err(err) = respond_one(depot, request, res).await {
                res.status_code(err.status_code());
                res.render(Json(err));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_newlines_collapses_double() {
        assert_eq!(normalize_newlines("hello\n\nworld"), "hello\nworld");
    }

    #[test]
    fn test_normalize_newlines_collapses_triple() {
        assert_eq!(normalize_newlines("hello\n\n\nworld"), "hello\nworld");
    }

    #[test]
    fn test_normalize_newlines_preserves_single() {
        assert_eq!(normalize_newlines("hello\nworld"), "hello\nworld");
    }

    #[test]
    fn test_normalize_newlines_multiple_groups() {
        assert_eq!(
            normalize_newlines("a\n\nb\n\n\nc\nd"),
            "a\nb\nc\nd"
        );
    }

    #[test]
    fn test_normalize_newlines_empty() {
        assert_eq!(normalize_newlines(""), "");
    }

    #[test]
    fn test_normalize_newlines_no_newlines() {
        assert_eq!(normalize_newlines("hello world"), "hello world");
    }

    #[test]
    fn test_normalize_newlines_only_newlines() {
        assert_eq!(normalize_newlines("\n\n\n\n"), "\n");
    }
}
