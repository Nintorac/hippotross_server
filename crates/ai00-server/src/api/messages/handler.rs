//! Request handlers for Claude-compatible Messages API.

use std::sync::Arc;

use ai00_core::{GenerateRequest, ThreadRequest, Token, MAX_TOKENS};
use futures_util::StreamExt;
use salvo::{oapi::extract::JsonBody, prelude::*, sse::SseEvent};
use tokio::sync::RwLock;

use super::streaming::*;
use super::types::*;
use crate::{
    api::{error::ApiErrorResponse, request_info},
    types::ThreadSender,
    SLEEP,
};

use ai00_core::sampler::nucleus::{NucleusParams, NucleusSampler};

/// Build RWKV prompt from messages.
fn build_prompt(system: Option<&str>, messages: &[MessageParam]) -> String {
    let mut prompt = String::new();

    // Add system prompt first (from top-level param, not message role)
    if let Some(sys) = system {
        prompt.push_str(&format!("System: {}\n\n", sys));
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
    let prompt = build_prompt(req.system.as_deref(), &req.messages);

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

    let content = vec![ContentBlock::Text {
        text: text.trim().to_string(),
    }];

    let response = MessagesResponse::new(model_name, content, token_counter.into())
        .with_stop_reason(finish_reason.into());

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

    let mut output_tokens = 0usize;
    let mut start_token = true;
    let mut started = false;

    let stream = token_receiver.into_stream().map(move |token| -> Result<SseEvent, std::convert::Infallible> {
        match token {
            Token::Start => {
                started = true;
                // Emit message_start event
                Ok(emit_message_start(message_id.clone(), model_name.clone(), input_tokens))
            }
            Token::Content(text) => {
                output_tokens += 1;

                // On first content, emit content_block_start if we haven't
                if start_token {
                    start_token = false;
                    let trimmed = text.trim_start().to_string();
                    if trimmed.is_empty() {
                        return Ok(emit_content_block_start_text(0));
                    }
                    // For first non-empty token, we'd ideally emit both start and delta
                    // but SSE is one event at a time, so just emit the delta
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
            Token::Done => {
                Ok(emit_message_stop())
            }
            _ => Ok(emit_ping()),
        }
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
