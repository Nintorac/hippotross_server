//! Request handlers for Claude-compatible Messages API.

use std::sync::Arc;

use ai00_core::{GenerateRequest, ThreadRequest, Token, MAX_TOKENS};
use futures_util::StreamExt;
use salvo::{oapi::extract::JsonBody, prelude::*, sse::SseEvent};
use tokio::sync::RwLock;

use super::streaming::*;
use super::tool_parser::ToolParser;
use super::types::*;
use crate::{
    api::{error::ApiErrorResponse, request_info},
    types::ThreadSender,
    SLEEP,
};

use ai00_core::sampler::nucleus::{NucleusParams, NucleusSampler};

/// Build RWKV prompt from messages.
fn build_prompt(
    system: Option<&str>,
    messages: &[MessageParam],
    tools: Option<&[Tool]>,
) -> String {
    let mut prompt = String::new();

    // Add system prompt first (from top-level param, not message role)
    if let Some(sys) = system {
        prompt.push_str(&format!("System: {}", sys));

        // Inject tool definitions into system prompt if provided
        if let Some(tools) = tools {
            if !tools.is_empty() {
                prompt.push_str(&generate_tool_system_prompt(tools));
            }
        }

        prompt.push_str("\n\n");
    } else if let Some(tools) = tools {
        // If no system prompt but tools provided, create one for tools
        if !tools.is_empty() {
            prompt.push_str("System:");
            prompt.push_str(&generate_tool_system_prompt(tools));
            prompt.push_str("\n\n");
        }
    }

    // Format conversation
    for msg in messages {
        let role = match msg.role {
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
        };
        let content = msg.content.to_text();
        prompt.push_str(&format!("{}: {}\n\n", role, content));
    }

    // Add assistant prefix for generation
    prompt.push_str("Assistant:");
    prompt
}

/// Convert MessagesRequest to GenerateRequest.
fn to_generate_request(req: &MessagesRequest) -> GenerateRequest {
    let prompt = build_prompt(
        req.system.as_deref(),
        &req.messages,
        req.tools.as_deref(),
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
        .unwrap_or_else(|| vec!["\n\nUser:".to_string()]);

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

    GenerateRequest {
        prompt,
        model_text,
        max_tokens,
        stop,
        sampler,
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

    Ok(())
}

/// Handle non-streaming messages request.
async fn respond_one(
    depot: &mut Depot,
    request: MessagesRequest,
    res: &mut Response,
) -> Result<(), ApiErrorResponse> {
    let sender = depot.obtain::<ThreadSender>().unwrap();

    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let gen_request = Box::new(to_generate_request(&request));
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

    // Check if tools are enabled and parse for tool_use
    let has_tools = request.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);

    let (content, stop_reason) = if has_tools {
        // Parse the output for tool_call blocks
        let mut parser = ToolParser::new();
        let result = parser.feed(&text);
        let final_result = parser.finalize();

        let mut content_blocks: Vec<ContentBlock> = Vec::new();

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
        // Simple text response
        let content = vec![ContentBlock::Text {
            text: text.trim().to_string(),
        }];
        (content, finish_reason.into())
    };

    let response =
        MessagesResponse::new(model_name, content, token_counter.into()).with_stop_reason(stop_reason);

    res.render(Json(response));
    Ok(())
}

/// Handle streaming messages request with Claude-style SSE events.
async fn respond_stream(depot: &mut Depot, request: MessagesRequest, res: &mut Response) {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let gen_request = Box::new(to_generate_request(&request));
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

    // Check if tools are enabled
    let has_tools = request.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);

    if has_tools {
        // Use tool-aware streaming with ToolParser
        respond_stream_with_tools(
            res,
            token_receiver,
            message_id,
            model_name,
            input_tokens,
        )
        .await;
    } else {
        // Simple streaming without tool parsing
        respond_stream_simple(res, token_receiver, message_id, model_name, input_tokens).await;
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
