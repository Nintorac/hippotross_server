use salvo::{
    oapi::{ToResponse, ToSchema},
    prelude::*,
};
use serde::Serialize;

use crate::{api::request_info, types::ThreadSender, SLEEP};

/// Model capabilities for Claude API compatibility.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelCapabilities {
    /// Whether the model supports tool/function calling
    pub tool_use: bool,
    /// Whether the model supports extended thinking traces
    pub extended_thinking: bool,
    /// Whether the model supports vision/image inputs
    pub vision: bool,
    /// Maximum context window size in tokens
    pub max_context_tokens: usize,
    /// Maximum output tokens per request
    pub max_output_tokens: usize,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            tool_use: true,           // Supported via Hermes format
            extended_thinking: true,  // Supported via <think> tags
            vision: false,            // Not yet supported
            max_context_tokens: 32768,
            max_output_tokens: 4096,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
struct ModelChoice {
    object: String,
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    created: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owned_by: Option<String>,
    capabilities: ModelCapabilities,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct ModelResponse {
    data: Vec<ModelChoice>,
}

/// Model name and id of the current choice.
#[endpoint(responses((status_code = 200, body = ModelResponse)))]
pub async fn models(depot: &mut Depot) -> Json<ModelResponse> {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.to_owned(), SLEEP).await;
    let model_name = info
        .reload
        .model_path
        .file_stem()
        .map(|stem| stem.to_string_lossy())
        .unwrap_or_default();

    // Get model file creation time if available
    let created = std::fs::metadata(&info.reload.model_path)
        .ok()
        .and_then(|m| m.created().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());

    Json(ModelResponse {
        data: vec![ModelChoice {
            object: "model".into(),
            id: model_name.into(),
            created,
            owned_by: Some("rwkv".into()),
            capabilities: ModelCapabilities::default(),
        }],
    })
}
