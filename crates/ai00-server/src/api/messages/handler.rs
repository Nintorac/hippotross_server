//! Request handlers for Claude-compatible Messages API.

use std::sync::Arc;

use ai00_core::{GenerateRequest, ThreadRequest, Token, MAX_TOKENS};
use futures_util::StreamExt;
use salvo::{oapi::extract::JsonBody, prelude::*};
use tokio::sync::RwLock;

use super::types::*;
use crate::{
    api::{error::ApiErrorResponse, request_info},
    types::ThreadSender,
    SLEEP,
};

use ai00_core::sampler::nucleus::{NucleusParams, NucleusSampler};

/// Special token required at start of prompt for RWKV state initialization.
const RWKV_EOT: &str = "";  // Token ID = 0, but we use empty string as tokenizer handles it

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
    if req.messages.is_empty() {
        return Err(ApiErrorResponse::invalid_request("messages cannot be empty"));
    }

    if req.max_tokens == 0 {
        return Err(
            ApiErrorResponse::invalid_request("max_tokens must be greater than 0")
                .with_param("max_tokens"),
        );
    }

    if let Some(temp) = req.temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err(
                ApiErrorResponse::invalid_request("temperature must be between 0.0 and 2.0")
                    .with_param("temperature"),
            );
        }
    }

    if let Some(top_p) = req.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(
                ApiErrorResponse::invalid_request("top_p must be between 0.0 and 1.0")
                    .with_param("top_p"),
            );
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

    // For now, only support non-streaming
    if request.stream {
        let err = ApiErrorResponse::invalid_request(
            "Streaming not yet implemented. Set stream: false",
        );
        res.status_code(err.status_code());
        res.render(Json(err));
        return;
    }

    if let Err(err) = respond_one(depot, request, res).await {
        res.status_code(err.status_code());
        res.render(Json(err));
    }
}
