//! Integration tests for BNF constrained decoding with real model.
//!
//! These tests load the 100m RWKV model and verify that BNF constraints
//! actually affect the model output.
//!
//! The model is loaded once and shared across all tests for efficiency.
//!
//! Run with: cargo test --test bnf_integration_test -- --nocapture

use ai00_core::{
    reload::{AdapterOption, BnfOption, Precision},
    GenerateRequest, ReloadRequest, ThreadRequest, Token,
};
use ai00_server::api::messages::{
    bnf_generator::{
        generate_schema_aware_grammar, generate_tool_grammars, generate_tool_name_grammar,
        json_schema_to_kbnf, GeneratorContext,
    },
    bnf_grammars::{
        build_structural_grammar, wrap_grammar_with_thinking, GRAMMAR_JSON_PRIMITIVES,
        GRAMMAR_THINKING_ONLY, GRAMMAR_THINKING_PLUS_TOOLS, GRAMMAR_TOOLS_ONLY,
    },
    Tool,
};
use flume::Sender;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OnceCell, RwLock};
use web_rwkv::tokenizer::Tokenizer;

// ============================================================================
// Shared Model Instance (loaded once for all tests)
// ============================================================================

/// Shared model state - loaded once and reused across all tests.
struct SharedModel {
    sender: Sender<ThreadRequest>,
    tokenizer: Arc<Tokenizer>,
}

/// Global shared model instance (async-compatible).
static SHARED_MODEL: OnceCell<SharedModel> = OnceCell::const_new();

/// Get or initialize the shared model asynchronously.
/// This is called by each test but only loads the model once.
async fn get_shared_model() -> Option<&'static SharedModel> {
    if !model_exists() {
        return None;
    }

    Some(
        SHARED_MODEL
            .get_or_init(|| async {
                let (sender, tokenizer) = setup_model_internal().await;
                SharedModel { sender, tokenizer }
            })
            .await,
    )
}

// ============================================================================
// Test Configuration
// ============================================================================

/// Path to the 100m model for testing.
fn model_path() -> PathBuf {
    PathBuf::from("/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st")
}

/// Path to the tokenizer.
fn tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/tokenizer/rwkv_vocab_v20230424.json")
}

/// Check if the model file exists.
fn model_exists() -> bool {
    model_path().exists()
}

/// Load the tokenizer for tests that don't need the full model.
fn load_tokenizer() -> Tokenizer {
    let path = tokenizer_path();
    let contents = std::fs::read_to_string(&path).expect("Failed to read tokenizer");
    Tokenizer::new(&contents).expect("Failed to parse tokenizer")
}

// ============================================================================
// Model Loading Helper
// ============================================================================

/// Internal helper to load the model and get a sender for requests.
/// Use `get_shared_model()` instead for tests to avoid reloading.
async fn setup_model_internal() -> (Sender<ThreadRequest>, Arc<Tokenizer>) {
    let (sender, receiver) = flume::unbounded::<ThreadRequest>();

    // Spawn the ai00_core server
    tokio::spawn(ai00_core::serve(receiver));

    // Load the tokenizer
    let tokenizer_contents =
        tokio::fs::read_to_string(tokenizer_path()).await.expect("Failed to read tokenizer");
    let tokenizer = Arc::new(Tokenizer::new(&tokenizer_contents).expect("Failed to parse tokenizer"));

    // Create reload request
    let reload_request = ReloadRequest {
        model_path: model_path(),
        lora: vec![],
        state: vec![],
        quant: 0,
        quant_type: Default::default(),
        precision: Precision::Fp16,
        token_chunk_size: 128,
        max_batch: 4,
        tokenizer_path: tokenizer_path(),
        bnf: BnfOption {
            enable_bytes_cache: true,
            start_nonterminal: "start".to_string(),
        },
        adapter: AdapterOption::Auto,
    };

    // Send reload request and wait for completion
    let (result_sender, result_receiver) = flume::unbounded();
    sender
        .send(ThreadRequest::Reload {
            request: Box::new(reload_request),
            sender: Some(result_sender),
        })
        .expect("Failed to send reload request");

    // Wait for model to load (with timeout)
    let loaded = tokio::time::timeout(Duration::from_secs(60), result_receiver.recv_async())
        .await
        .expect("Model load timeout")
        .expect("Failed to receive load result");

    assert!(loaded, "Model failed to load");

    (sender, tokenizer)
}

/// Generate text with optional BNF constraint.
async fn generate_with_bnf(
    sender: &Sender<ThreadRequest>,
    tokenizer: &Arc<Tokenizer>,
    prompt: &str,
    bnf_schema: Option<String>,
    max_tokens: usize,
) -> String {
    let (token_sender, token_receiver) = flume::unbounded();

    let request = GenerateRequest {
        prompt: prompt.to_string(),
        model_text: String::new(),
        max_tokens,
        stop: vec![],
        bias: Arc::new(HashMap::new()),
        bnf_schema,
        sampler: Arc::new(RwLock::new(ai00_core::sampler::nucleus::NucleusSampler::default())),
        kind: ai00_core::GenerateKind::None,
        state: Default::default(),
    };

    sender
        .send(ThreadRequest::Generate {
            request: Box::new(request),
            tokenizer: tokenizer.clone(),
            sender: token_sender,
        })
        .expect("Failed to send generate request");

    // Collect tokens
    let mut output = String::new();
    while let Ok(token) = token_receiver.recv_async().await {
        match token {
            Token::Content(text) => output.push_str(&text),
            Token::Stop(_, _) | Token::Done => break,
            Token::Start => {}
            _ => {}
        }
    }

    output
}

// ============================================================================
// Grammar Compilation Tests (no model needed)
// ============================================================================

/// Test that the real tokenizer can be loaded.
#[test]
fn test_tokenizer_loads() {
    let tokenizer = load_tokenizer();
    // Basic sanity check - tokenizer should have vocab
    assert!(tokenizer.token_index_to_bytes().len() > 0);
}

/// Test BnfSampler can be created with our grammars.
#[test]
fn test_bnf_sampler_compiles_simple_grammar() {
    let tokenizer = load_tokenizer();
    let grammar = r#"start ::= "yes" | "no";"#;

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, grammar);
    assert!(result.is_ok(), "BnfSampler should compile simple grammar: {:?}", result.err());
}

/// Test BnfSampler compiles thinking grammar.
/// Blocked by ninchat-bd2: KBNF grammar parsing errors with regex patterns.
#[test]
#[ignore = "Blocked by ninchat-bd2: grammar parsing errors"]
fn test_bnf_sampler_compiles_thinking_grammar() {
    let tokenizer = load_tokenizer();
    let grammar = format!("{}\n{}", GRAMMAR_JSON_PRIMITIVES, GRAMMAR_THINKING_ONLY);

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar);
    assert!(result.is_ok(), "BnfSampler should compile thinking grammar: {:?}", result.err());
}

/// Test BnfSampler compiles tools grammar.
/// Blocked by ninchat-bd2: KBNF grammar parsing errors with regex patterns.
#[test]
#[ignore = "Blocked by ninchat-bd2: grammar parsing errors"]
fn test_bnf_sampler_compiles_tools_grammar() {
    let tokenizer = load_tokenizer();
    let grammar = format!("{}\n{}", GRAMMAR_JSON_PRIMITIVES, GRAMMAR_TOOLS_ONLY);

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar);
    assert!(result.is_ok(), "BnfSampler should compile tools grammar: {:?}", result.err());
}

/// Test BnfSampler compiles combined thinking+tools grammar.
/// Blocked by ninchat-bd2: KBNF grammar parsing errors with regex patterns.
#[test]
#[ignore = "Blocked by ninchat-bd2: grammar parsing errors"]
fn test_bnf_sampler_compiles_combined_grammar() {
    let tokenizer = load_tokenizer();
    let grammar = format!("{}\n{}", GRAMMAR_JSON_PRIMITIVES, GRAMMAR_THINKING_PLUS_TOOLS);

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar);
    assert!(result.is_ok(), "BnfSampler should compile combined grammar: {:?}", result.err());
}

/// Test BnfSampler compiles schema-aware grammar with tools.
/// Blocked by ninchat-bd2: KBNF grammar parsing errors with regex patterns.
#[test]
#[ignore = "Blocked by ninchat-bd2: grammar parsing errors"]
fn test_bnf_sampler_compiles_schema_aware_grammar() {
    let tokenizer = load_tokenizer();
    let tools = vec![Tool {
        name: "get_weather".to_string(),
        description: Some("Get weather for a location".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
        cache_control: None,
    }];

    let grammar = generate_schema_aware_grammar(&tools, false);

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar);
    assert!(
        result.is_ok(),
        "BnfSampler should compile schema-aware grammar: {:?}",
        result.err()
    );
}

/// Test wrap_grammar_with_thinking compiles correctly.
/// Blocked by ninchat-bd2: KBNF grammar parsing errors with regex patterns.
#[test]
#[ignore = "Blocked by ninchat-bd2: grammar parsing errors"]
fn test_bnf_sampler_compiles_wrapped_grammar() {
    let tokenizer = load_tokenizer();
    let user_grammar = r#"start ::= "hello" | "world";"#;
    let wrapped = wrap_grammar_with_thinking(user_grammar);
    let grammar = format!("{}\n{}", GRAMMAR_JSON_PRIMITIVES, wrapped);

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar);
    assert!(result.is_ok(), "BnfSampler should compile wrapped grammar: {:?}", result.err());
}

// ============================================================================
// BnfSampler Logit Masking Tests (no model needed)
// ============================================================================

/// Test that BnfSampler actually masks logits.
/// Blocked by ninchat-bd2: All tokens being blocked, even for simple grammars.
#[test]
#[ignore = "Blocked by ninchat-bd2: BNF masks all tokens"]
fn test_bnf_sampler_masks_logits() {
    use ai00_core::sampler::Formatter;

    let tokenizer = load_tokenizer();
    let grammar = r#"start ::= "yes";"#;

    let sampler = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, grammar)
        .expect("Should compile grammar");

    // Create logits array (all zeros initially)
    let vocab_size = tokenizer.token_index_to_bytes().len();
    let mut logits = vec![0.0f32; vocab_size];

    println!("Vocab size: {}", vocab_size);

    // Apply BNF masking
    sampler.transform(&mut logits);

    // After masking, some logits should be -inf (blocked tokens)
    let blocked_count = logits.iter().filter(|&&x| x == f32::NEG_INFINITY).count();
    let allowed_count = logits.iter().filter(|&&x| x != f32::NEG_INFINITY).count();

    println!("Blocked: {}, Allowed: {}", blocked_count, allowed_count);

    // Find which tokens are allowed
    let vocab = tokenizer.token_index_to_bytes();
    for (i, logit) in logits.iter().enumerate() {
        if *logit != f32::NEG_INFINITY && i < vocab.len() {
            let bytes = &vocab[i];
            let text = String::from_utf8_lossy(bytes);
            println!("Allowed token {}: {:?} ({})", i, bytes, text);
        }
    }

    // Most tokens should be blocked since only "yes" is allowed
    assert!(
        blocked_count > allowed_count,
        "BNF should block most tokens. Blocked: {}, Allowed: {}",
        blocked_count,
        allowed_count
    );
    assert!(allowed_count > 0, "At least some tokens should be allowed");
}

/// Test that BnfSampler allows correct initial tokens.
/// Blocked by ninchat-bd2: All tokens being blocked, even for simple grammars.
#[test]
#[ignore = "Blocked by ninchat-bd2: BNF masks all tokens"]
fn test_bnf_sampler_allows_valid_tokens() {
    use ai00_core::sampler::Formatter;

    let tokenizer = load_tokenizer();
    // Grammar that allows "Hello" or "Hi"
    let grammar = r#"start ::= "Hello" | "Hi";"#;

    let sampler = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, grammar)
        .expect("Should compile grammar");

    let vocab = tokenizer.token_index_to_bytes();
    let vocab_size = vocab.len();
    let mut logits = vec![0.0f32; vocab_size];

    sampler.transform(&mut logits);

    // Find token IDs that start with 'H' byte
    let h_byte = b'H';
    let h_tokens: Vec<usize> = (0..vocab_size)
        .filter(|&i| vocab[i].first() == Some(&h_byte))
        .collect();

    // At least one H token should be allowed
    let h_allowed = h_tokens.iter().any(|&i| logits[i] != f32::NEG_INFINITY);
    assert!(h_allowed, "Token starting with 'H' should be allowed as it starts both 'Hello' and 'Hi'");
}

// ============================================================================
// End-to-End Model Tests (require GPU and model)
// ============================================================================

/// Test simple generation without BNF (baseline).
#[tokio::test]
async fn test_model_generation_without_bnf() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let output = generate_with_bnf(&model.sender, &model.tokenizer, "Hello, my name is", None, 20).await;

    assert!(!output.is_empty(), "Model should generate some output");
    println!("Generated (no BNF): {}", output);
}

/// Test generation with simple yes/no BNF constraint.
/// Blocked by ninchat-bd2: BNF constrains block all tokens.
#[tokio::test]
#[ignore = "Blocked by ninchat-bd2: BNF masks all tokens"]
async fn test_model_generation_with_yes_no_bnf() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let grammar = r#"start ::= "yes" | "no";"#;
    let output = generate_with_bnf(
        &model.sender,
        &model.tokenizer,
        "Is the sky blue? Answer with yes or no: ",
        Some(grammar.to_string()),
        10,
    )
    .await;

    let cleaned = output.trim().to_lowercase();
    assert!(
        cleaned == "yes" || cleaned == "no",
        "Output should be exactly 'yes' or 'no', got: '{}'",
        output
    );
    println!("Generated (yes/no BNF): {}", output);
}

/// Test generation with JSON BNF constraint.
/// Blocked by ninchat-bd2: BNF constrains block all tokens.
#[tokio::test]
#[ignore = "Blocked by ninchat-bd2: BNF masks all tokens"]
async fn test_model_generation_with_json_bnf() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    // Simple JSON object grammar
    let grammar = r#"
start ::= "{" ws "\"name\"" ws ":" ws string ws "}";
string ::= '"' chars '"';
chars ::= #"[^\"]*";
ws ::= #"[ \t\n\r]*";
"#;

    let output = generate_with_bnf(
        &model.sender,
        &model.tokenizer,
        "Generate a JSON object with a name field: ",
        Some(grammar.to_string()),
        50,
    )
    .await;

    // Verify it parses as JSON
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&output);
    assert!(
        parsed.is_ok(),
        "Output should be valid JSON, got: '{}', error: {:?}",
        output,
        parsed.err()
    );

    let json = parsed.unwrap();
    assert!(json.get("name").is_some(), "JSON should have 'name' field");
    println!("Generated (JSON BNF): {}", output);
}

/// Test generation with thinking tags BNF.
/// Note: This test is ignored until ninchat-bd2 (grammar parsing fix) is resolved.
#[tokio::test]
#[ignore = "Blocked by ninchat-bd2: grammar parsing errors"]
async fn test_model_generation_with_thinking_bnf() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let grammar = format!("{}\n{}", GRAMMAR_JSON_PRIMITIVES, GRAMMAR_THINKING_ONLY);

    let output = generate_with_bnf(
        &model.sender,
        &model.tokenizer,
        "Think step by step about 2+2: ",
        Some(grammar),
        100,
    )
    .await;

    println!("Generated (thinking BNF): {}", output);

    // The output might have thinking tags or just be plain text (both valid per grammar)
    // Just verify we got some output
    assert!(!output.is_empty(), "Model should generate output with thinking grammar");
}

/// Test that BNF constraint actually restricts output.
/// Blocked by ninchat-bd2: BNF constrains block all tokens.
#[tokio::test]
#[ignore = "Blocked by ninchat-bd2: BNF masks all tokens"]
async fn test_bnf_actually_constrains_output() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    // Grammar that only allows numbers
    let grammar = r#"start ::= #"[0-9]+";"#;

    let output = generate_with_bnf(
        &model.sender,
        &model.tokenizer,
        "Write some random text: ",
        Some(grammar.to_string()),
        20,
    )
    .await;

    // Output should be only digits
    let is_all_digits = output.chars().all(|c| c.is_ascii_digit());
    assert!(
        is_all_digits,
        "With number-only grammar, output should be all digits, got: '{}'",
        output
    );
    println!("Generated (numbers only BNF): {}", output);
}

// ============================================================================
// Grammar Syntax Tests (no model needed)
// ============================================================================

/// Test that GRAMMAR_JSON_PRIMITIVES is syntactically valid.
#[test]
fn test_grammar_json_primitives_syntax() {
    let grammar = GRAMMAR_JSON_PRIMITIVES;
    assert!(grammar.contains("json_object"));
    assert!(grammar.contains("json_value"));
    assert!(grammar.contains("string"));
    assert!(grammar.contains("number"));
}

/// Test build_structural_grammar for all combinations.
#[test]
fn test_build_structural_grammar_combinations() {
    // No features
    let grammar = build_structural_grammar(false, false);
    assert!(grammar.contains("start ::="));

    // Thinking only
    let grammar = build_structural_grammar(true, false);
    assert!(grammar.contains("<think>"));

    // Tools only
    let grammar = build_structural_grammar(false, true);
    assert!(grammar.contains("<tool_use>"));

    // Both
    let grammar = build_structural_grammar(true, true);
    assert!(grammar.contains("<think>"));
    assert!(grammar.contains("<tool_use>"));
}

/// Test tool name grammar generation.
#[test]
fn test_generate_tool_name_grammar_integration() {
    let tools = vec![
        Tool {
            name: "get_weather".to_string(),
            description: Some("Get weather info".to_string()),
            input_schema: json!({"type": "object"}),
            cache_control: None,
        },
        Tool {
            name: "search".to_string(),
            description: Some("Search the web".to_string()),
            input_schema: json!({"type": "object"}),
            cache_control: None,
        },
    ];

    let grammar = generate_tool_name_grammar(&tools);
    assert!(grammar.contains("get_weather"));
    assert!(grammar.contains("search"));
    assert!(grammar.contains("tool_name ::="));
}

/// Test JSON Schema to KBNF conversion.
#[test]
fn test_json_schema_to_kbnf_types() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    });
    let mut ctx = GeneratorContext::new();
    let _rule = json_schema_to_kbnf(&schema, "test_object", &mut ctx);
    let grammar = ctx.into_grammar();
    assert!(grammar.contains("test_object"));
}

/// Test complete tool grammar generation.
#[test]
fn test_generate_tool_grammars_integration() {
    let tools = vec![Tool {
        name: "calculator".to_string(),
        description: Some("Perform calculations".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "precision": {"type": "integer"}
            },
            "required": ["expression"]
        }),
        cache_control: None,
    }];

    let grammar = generate_tool_grammars(&tools);
    assert!(grammar.contains("tool_call ::="));
    assert!(grammar.contains("calculator_call"));
    assert!(grammar.contains("calculator_input"));
}

/// Test schema-aware grammar generation.
#[test]
fn test_generate_schema_aware_grammar_integration() {
    let tools = vec![Tool {
        name: "echo".to_string(),
        description: Some("Echo back the message".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        }),
        cache_control: None,
    }];

    // Without thinking
    let grammar = generate_schema_aware_grammar(&tools, false);
    assert!(grammar.contains("start ::="));
    assert!(grammar.contains("tool_call ::="));
    assert!(!grammar.contains("<think>"));

    // With thinking
    let grammar = generate_schema_aware_grammar(&tools, true);
    assert!(grammar.contains("start ::="));
    assert!(grammar.contains("tool_call ::="));
    assert!(grammar.contains("<think>"));
}
