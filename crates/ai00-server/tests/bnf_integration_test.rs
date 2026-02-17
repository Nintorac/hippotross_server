//! Integration tests for BNF constrained decoding with real model.
//!
//! These tests load the 100m RWKV model and verify that BNF constraints
//! actually affect the model output.
//!
//! The model is loaded once and shared across all tests for efficiency.
//!
//! Run with: cargo test --test bnf_integration_test -- --nocapture

use ai00_core::{
    reload::{AdapterOption, Backend, BnfOption, Precision},
    GenerateRequest, ReloadRequest, ThreadRequest, Token,
};
use ai00_server::api::messages::{
    bnf_generator::{
        generate_schema_aware_grammar, generate_tool_grammars, generate_tool_name_grammar,
        json_schema_to_kbnf, GeneratorContext,
    },
    bnf_grammars::{
        build_structural_grammar, wrap_grammar_with_thinking, GRAMMAR_JSON_PRIMITIVES,
        GRAMMAR_UNIFIED,
    },
    Tool,
};
use flume::Sender;
use lazy_static::lazy_static;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::{OnceCell, RwLock};
use web_rwkv::tokenizer::Tokenizer;

// ============================================================================
// Global Runtime (persists across all tests)
// ============================================================================

lazy_static! {
    /// Global runtime for the serve task.
    /// This runtime persists across all tests so the serve task doesn't die
    /// when individual test runtimes are dropped.
    static ref GLOBAL_RUNTIME: Runtime = Runtime::new().expect("Failed to create global runtime");
}

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

/// Path to the model for testing.
/// Can be overridden with BNF_TEST_MODEL env var.
/// Default: 0.1B model for fast CI. Use 2.9B+ for tool calling tests.
fn model_path() -> PathBuf {
    std::env::var("BNF_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st"))
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

    // Spawn the ai00_core server on the GLOBAL_RUNTIME so it persists across tests.
    // Each #[tokio::test] creates its own runtime that gets dropped when the test ends.
    // By spawning on GLOBAL_RUNTIME, the serve task survives across all tests.
    GLOBAL_RUNTIME.spawn(ai00_core::serve(receiver));

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
        backend: Backend::WebGpu,
    };

    // Send reload request and wait for completion
    let (result_sender, result_receiver) = flume::unbounded();
    sender
        .send(ThreadRequest::Reload {
            request: Box::new(reload_request),
            sender: Some(result_sender),
        })
        .expect("Failed to send reload request");

    // Wait for model to load (with timeout - larger models need more time)
    let loaded = tokio::time::timeout(Duration::from_secs(300), result_receiver.recv_async())
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
        request_id: None,
        trace_id: None,
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
    let grammar = r#"start::='yes' | 'no';"#;

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, grammar);
    assert!(result.is_ok(), "BnfSampler should compile simple grammar: {:?}", result.err());
}

/// Test BnfSampler compiles the unified grammar.
#[test]
fn test_bnf_sampler_compiles_unified_grammar() {
    let tokenizer = load_tokenizer();
    let stop_seqs = vec!["\n\n".to_string()];
    let grammar = build_structural_grammar(false, false, &stop_seqs);

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar);
    assert!(result.is_ok(), "BnfSampler should compile unified grammar: {:?}", result.err());
}

/// Test BnfSampler compiles unified grammar with all stop sequence variants.
#[test]
fn test_bnf_sampler_compiles_unified_grammar_with_stop_sequences() {
    let tokenizer = load_tokenizer();
    let stop_seqs = vec!["\n\n".to_string(), "</s>".to_string()];
    let grammar = build_structural_grammar(true, true, &stop_seqs);

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar);
    assert!(result.is_ok(), "BnfSampler should compile unified grammar with stop sequences: {:?}", result.err());
}

/// Test GRAMMAR_UNIFIED constant compiles directly.
#[test]
fn test_bnf_sampler_compiles_grammar_unified_constant() {
    let tokenizer = load_tokenizer();
    // Add terminator rule manually since GRAMMAR_UNIFIED doesn't include it
    let grammar = format!("{}\n{}\nterminator::='\\n\\n';", GRAMMAR_JSON_PRIMITIVES, GRAMMAR_UNIFIED);

    let result = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar);
    assert!(result.is_ok(), "BnfSampler should compile GRAMMAR_UNIFIED: {:?}", result.err());
}

/// Test BnfSampler compiles schema-aware grammar with tools.
#[test]
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

    // generate_schema_aware_grammar now always includes thinking (unified grammar)
    let mut grammar = generate_schema_aware_grammar(&tools);
    grammar.push_str("\nterminator::='\\n\\n';");

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
fn test_bnf_sampler_compiles_wrapped_grammar() {
    let tokenizer = load_tokenizer();
    let user_grammar = r#"start::='hello' | 'world';"#;
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
fn test_bnf_sampler_masks_logits() {
    use ai00_core::sampler::Formatter;

    let tokenizer = load_tokenizer();
    let grammar = r#"start::='yes';"#;

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
fn test_bnf_sampler_allows_valid_tokens() {
    use ai00_core::sampler::Formatter;

    let tokenizer = load_tokenizer();
    // Grammar that allows "Hello" or "Hi"
    let grammar = r#"start::='Hello' | 'Hi';"#;

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
// Grammar Text-Only Output Tests (no model needed)
// ============================================================================

/// Test that the unified grammar allows text-only output (no tool calls required).
/// This verifies the grammar doesn't force tool use - plain text tokens should be valid.
#[test]
fn test_tools_grammar_allows_text_only_output() {
    use ai00_core::sampler::Formatter;

    let tokenizer = load_tokenizer();
    let stop_seqs = vec!["\n\n".to_string()];
    let grammar = build_structural_grammar(false, true, &stop_seqs);

    println!("Grammar:\n{}", grammar);

    let sampler = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar)
        .expect("Should compile tools grammar");

    let vocab = tokenizer.token_index_to_bytes();
    let vocab_size = vocab.len();
    let mut logits = vec![0.0f32; vocab_size];

    // Apply BNF masking to get initial allowed tokens
    sampler.transform(&mut logits);

    // Count and show ALL allowed tokens (first 20)
    let all_allowed: Vec<(usize, String)> = (0..vocab_size)
        .filter(|&i| logits[i] != f32::NEG_INFINITY && !vocab[i].is_empty())
        .take(20)
        .map(|i| (i, String::from_utf8_lossy(&vocab[i]).to_string()))
        .collect();

    let total_allowed = (0..vocab_size)
        .filter(|&i| logits[i] != f32::NEG_INFINITY)
        .count();

    println!("Total allowed tokens: {}", total_allowed);
    println!("First 20 allowed tokens: {:?}", all_allowed);

    // Find tokens that are plain text (not starting with '<')
    let text_tokens_allowed: Vec<(usize, String)> = (0..vocab_size)
        .filter(|&i| {
            logits[i] != f32::NEG_INFINITY
                && !vocab[i].is_empty()
                && vocab[i][0] != b'<'
        })
        .take(10)
        .map(|i| (i, String::from_utf8_lossy(&vocab[i]).to_string()))
        .collect();

    println!("Text tokens (not '<') allowed at start: {:?}", text_tokens_allowed);

    // Check specifically for "Hello" token (33155 from the other test)
    let hello_token = 33155usize;
    if hello_token < vocab_size {
        let hello_allowed = logits[hello_token] != f32::NEG_INFINITY;
        println!("Token 33155 ('Hello') allowed: {}, logit: {}", hello_allowed, logits[hello_token]);
    }

    // Find if '<' tokens are allowed (for tool_call)
    let angle_bracket_tokens: Vec<(usize, String)> = (0..vocab_size)
        .filter(|&i| logits[i] != f32::NEG_INFINITY && !vocab[i].is_empty() && vocab[i][0] == b'<')
        .take(10)
        .map(|i| (i, String::from_utf8_lossy(&vocab[i]).to_string()))
        .collect();

    println!("'<' tokens allowed: {:?}", angle_bracket_tokens);

    // The grammar should allow BOTH text tokens AND '<' tokens
    // text? tool_sequence? means both are optional
    // If this fails, the grammar is forcing tool use
    assert!(
        !text_tokens_allowed.is_empty() || total_allowed > angle_bracket_tokens.len(),
        "Grammar should allow plain text tokens (not starting with '<'). This is required for text-only responses. Total allowed: {}, '<' tokens: {}",
        total_allowed, angle_bracket_tokens.len()
    );
}

/// Test that text tokens like "Hello" can be fed to the sampler with tools grammar.
#[test]
fn test_tools_grammar_accepts_hello_world() {
    use ai00_core::sampler::Formatter;

    let tokenizer = load_tokenizer();
    let stop_seqs = vec!["\n\n".to_string()];
    let grammar = build_structural_grammar(false, true, &stop_seqs);

    let mut sampler = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar)
        .expect("Should compile tools grammar");

    // Tokenize "Hello" - a simple text without tool calls
    let text = "Hello";
    let tokens = tokenizer.encode(text.as_bytes()).expect("Should tokenize 'Hello'");

    println!("Tokens for '{}': {:?}", text, tokens);

    // Feed each token to the sampler
    let mut all_accepted = true;
    let vocab = tokenizer.token_index_to_bytes();
    for (i, &token) in tokens.iter().enumerate() {
        let token_id = token as usize;
        let finished = sampler.update(token as u32);
        println!("Token {}: {} (id={}) -> finished={}", i,
            String::from_utf8_lossy(&vocab[token_id]),
            token, finished);

        if finished && i < tokens.len() - 1 {
            println!("Sampler finished early at token {}", i);
            all_accepted = false;
            break;
        }
    }

    assert!(
        all_accepted,
        "Grammar should accept 'Hello' as valid text-only output"
    );
}

/// Test that multiple tokens can be generated without early termination.
/// This is the key test for the "model only samples one token" bug.
///
/// The solution: require a terminator (from stop sequences) to complete the grammar.
/// This prevents early "finished" signals.
#[test]
fn test_tools_grammar_allows_multi_token_text() {
    use ai00_core::sampler::Formatter;

    let tokenizer = load_tokenizer();
    let stop_seqs = vec!["\n\n".to_string()];
    let grammar = build_structural_grammar(false, true, &stop_seqs);

    let mut sampler = ai00_core::sampler::bnf::BnfSampler::new(&tokenizer, &grammar)
        .expect("Should compile tools grammar");

    // Tokenize a longer sentence
    let text = "The capital of Italy is Rome.";
    let tokens = tokenizer.encode(text.as_bytes()).expect("Should tokenize text");

    println!("Tokens for '{}': {:?}", text, tokens);
    assert!(tokens.len() > 1, "Test requires multiple tokens");

    let vocab = tokenizer.token_index_to_bytes();
    let mut finished_count = 0;

    for (i, &token) in tokens.iter().enumerate() {
        let token_id = token as usize;

        // Check that the token is allowed BEFORE we update
        let mut logits = vec![0.0f32; vocab.len()];
        sampler.transform(&mut logits);
        let token_allowed = logits[token_id] != f32::NEG_INFINITY;

        let finished = sampler.update(token as u32);
        println!("Token {}: {:?} (id={}) -> allowed={}, finished={}",
            i, String::from_utf8_lossy(&vocab[token_id]), token, token_allowed, finished);

        if finished {
            finished_count += 1;
        }

        // Key assertion: each token should be allowed
        assert!(
            token_allowed,
            "Token {} ({:?}) should be allowed by grammar at position {}",
            token, String::from_utf8_lossy(&vocab[token_id]), i
        );
    }

    println!("Finished signals during generation: {}", finished_count);

    // The grammar shouldn't signal finished prematurely for text-only content
    // (finished_count might be > 0 at the end, but not at every token)
    assert!(
        finished_count <= tokens.len() / 2,
        "Grammar should not signal 'finished' after every token. Got {} finished signals for {} tokens.",
        finished_count, tokens.len()
    );
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
async fn test_model_generation_with_yes_no_bnf() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let grammar = r#"start::='yes' | 'no';"#;
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
async fn test_model_generation_with_json_bnf() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    // Simple JSON object grammar (kbnf syntax)
    let grammar = r#"
start::='{' ws '"name"' ws ':' ws string ws '}';
string::='"' chars '"';
chars::=#'[^"]*';
ws::=#'[ \\t\\n\\r]*';
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
#[tokio::test]
async fn test_model_generation_with_unified_bnf() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let stop_seqs = vec!["\n\n".to_string()];
    let grammar = build_structural_grammar(true, true, &stop_seqs);

    let output = generate_with_bnf(
        &model.sender,
        &model.tokenizer,
        "Think step by step about 2+2: ",
        Some(grammar),
        100,
    )
    .await;

    println!("Generated (unified BNF): {}", output);

    // The output might have thinking tags or just be plain text (both valid per grammar)
    // Just verify we got some output
    assert!(!output.is_empty(), "Model should generate output with unified grammar");
}

/// Test that BNF constraint actually restricts output.
/// Blocked by ninchat-bd2: BNF constrains block all tokens.
#[tokio::test]
async fn test_bnf_actually_constrains_output() {
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    // Grammar that only allows numbers
    let grammar = r#"start::=#'[0-9]+';"#;

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

/// Test build_structural_grammar produces unified grammar for all combinations.
/// With the unified grammar, all parameter combinations produce the same output
/// (thinking and tools are always optional in the grammar).
#[test]
fn test_build_structural_grammar_unified_for_all_combinations() {
    let stop_seqs = vec!["\n\n".to_string()];

    // All combinations produce the same unified grammar
    for (thinking, tools) in [(false, false), (true, false), (false, true), (true, true)] {
        let grammar = build_structural_grammar(thinking, tools, &stop_seqs);

        // Unified grammar always has start rule
        assert!(grammar.contains("start::="), "Missing start rule for ({}, {})", thinking, tools);

        // Unified grammar always includes thinking (optional)
        assert!(grammar.contains("<think>"), "Missing <think> for ({}, {})", thinking, tools);

        // Unified grammar always includes function calls (ai00 XML format)
        assert!(grammar.contains("<ai00:function_calls>"), "Missing <ai00:function_calls> for ({}, {})", thinking, tools);

        // Unified grammar always has terminator
        assert!(grammar.contains("terminator::="), "Missing terminator for ({}, {})", thinking, tools);

        // Unified grammar uses complement regex
        assert!(grammar.contains("#ex'"), "Missing complement regex for ({}, {})", thinking, tools);
    }
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
    assert!(grammar.contains("tool_name::="));
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
    assert!(grammar.contains("tool_call::="));
    assert!(grammar.contains("calculator_call"));
    assert!(grammar.contains("calculator_input"));
}

/// Test schema-aware grammar generation with unified grammar.
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

    // Unified grammar always includes thinking (optional)
    let grammar = generate_schema_aware_grammar(&tools);
    assert!(grammar.contains("start::="));
    assert!(grammar.contains("<think>"));  // Always present in unified grammar
    assert!(grammar.contains("<ai00:function_calls>"));  // ai00 XML format
    assert!(grammar.contains("echo_call"));  // Tool-specific rule
    assert!(grammar.contains("echo_input"));  // Tool-specific input rule
    // Uses complement regex
    assert!(grammar.contains("#ex'"));
}

// ============================================================================
// Tool Calling Integration Tests (require GPU and model)
// ============================================================================
//
// These tests require a model that supports function calling (2.9B+).
// Run with: BNF_TEST_MODEL=/workspace/models/rwkv7-g1c-2.9b-20251231-ctx8192.st cargo test --test bnf_integration_test tool_call -- --nocapture

use ai00_server::api::messages::{generate_tool_system_prompt, ToolParser};

/// Check if we're using a model that supports tool calling.
fn model_supports_tool_calling() -> bool {
    let path = model_path();
    let path_str = path.to_string_lossy();
    // 0.1B model doesn't support tool calling
    !path_str.contains("0.1b")
}

/// Build a prompt that requests the model to use a tool.
fn build_tool_prompt(user_message: &str, tools: &[Tool]) -> String {
    let tool_system = generate_tool_system_prompt(tools, None, None);
    format!(
        "System: You are a helpful assistant.{}\n\nUser: {}\n\nA:",
        tool_system, user_message
    )
}

/// Test that model generates tool_call tags when prompted with tools.
#[tokio::test]
async fn test_model_generates_tool_call_tags() {
    if !model_supports_tool_calling() {
        eprintln!("Model {:?} doesn't support tool calling, skipping. Set BNF_TEST_MODEL to use a larger model.", model_path());
        return;
    }
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let tools = vec![Tool {
        name: "get_weather".to_string(),
        description: Some("Get the current weather for a location".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country"
                }
            },
            "required": ["location"]
        }),
        cache_control: None,
    }];

    let prompt = build_tool_prompt("What is the weather in Tokyo?", &tools);
    println!("Prompt:\n{}", prompt);

    let output = generate_with_bnf(&model.sender, &model.tokenizer, &prompt, None, 100).await;
    println!("Model output:\n{}", output);

    // Check that the output contains tool_call tags
    let has_tool_call = output.contains("<tool_call>");
    let has_tool_use = output.contains("<tool_use>");

    println!("Has <tool_call>: {}", has_tool_call);
    println!("Has <tool_use>: {}", has_tool_use);

    assert!(
        has_tool_call,
        "Model should generate <tool_call> tags. Got: {}",
        output
    );
    assert!(
        !has_tool_use,
        "Model should NOT use <tool_use> tags (should use <tool_call>). Got: {}",
        output
    );
}

/// Test that ToolParser correctly parses model's tool_call output.
#[tokio::test]
async fn test_tool_parser_parses_model_output() {
    if !model_supports_tool_calling() {
        eprintln!("Model {:?} doesn't support tool calling, skipping.", model_path());
        return;
    }
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let tools = vec![Tool {
        name: "get_weather".to_string(),
        description: Some("Get the current weather for a location".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
        cache_control: None,
    }];

    let prompt = build_tool_prompt("What's the weather in Paris?", &tools);
    let output = generate_with_bnf(&model.sender, &model.tokenizer, &prompt, None, 100).await;

    println!("Model output:\n{}", output);

    // Parse with ToolParser
    let mut parser = ToolParser::new();
    let result = parser.feed(&output);
    let final_result = parser.finalize();

    let mut all_tools: Vec<_> = result.tool_uses;
    all_tools.extend(final_result.tool_uses);

    println!("Parsed {} tool calls", all_tools.len());
    for tool in &all_tools {
        println!("  - {}: {:?}", tool.name, tool.input);
    }

    assert!(
        parser.has_tool_use(),
        "ToolParser should detect tool calls in model output. Output was: {}",
        output
    );

    assert!(
        !all_tools.is_empty(),
        "Should have at least one parsed tool call"
    );

    // Verify the tool name matches
    assert_eq!(
        all_tools[0].name, "get_weather",
        "Tool name should be 'get_weather'"
    );
}

/// Test tool calling with multiple tools available.
#[tokio::test]
async fn test_model_tool_call_with_multiple_tools() {
    if !model_supports_tool_calling() {
        eprintln!("Model {:?} doesn't support tool calling, skipping.", model_path());
        return;
    }
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let tools = vec![
        Tool {
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
        },
        Tool {
            name: "search".to_string(),
            description: Some("Search the web for information".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
            cache_control: None,
        },
    ];

    let prompt = build_tool_prompt("Search for the latest news about AI", &tools);
    let output = generate_with_bnf(&model.sender, &model.tokenizer, &prompt, None, 100).await;

    println!("Model output:\n{}", output);

    let mut parser = ToolParser::new();
    parser.feed(&output);
    parser.finalize();

    assert!(
        parser.has_tool_use(),
        "Model should call a tool when prompted. Output: {}",
        output
    );
}

/// Test that the tool prompt format is being followed.
#[tokio::test]
async fn test_tool_call_json_format() {
    if !model_supports_tool_calling() {
        eprintln!("Model {:?} doesn't support tool calling, skipping.", model_path());
        return;
    }
    let Some(model) = get_shared_model().await else {
        eprintln!("Model not found at {:?}, skipping test", model_path());
        return;
    };

    let tools = vec![Tool {
        name: "calculate".to_string(),
        description: Some("Perform a calculation".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }),
        cache_control: None,
    }];

    let prompt = build_tool_prompt("Calculate 2 + 2", &tools);
    let output = generate_with_bnf(&model.sender, &model.tokenizer, &prompt, None, 100).await;

    println!("Model output:\n{}", output);

    // Parse with ToolParser
    let mut parser = ToolParser::new();
    let result = parser.feed(&output);
    let final_result = parser.finalize();

    let mut all_tools: Vec<_> = result.tool_uses;
    all_tools.extend(final_result.tool_uses);

    if !all_tools.is_empty() {
        let tool = &all_tools[0];
        println!("Parsed tool: {} with input: {}", tool.name, tool.input);

        // Verify the JSON has the expected structure
        assert_eq!(tool.name, "calculate");
        assert!(
            tool.input.is_object(),
            "Tool input should be a JSON object"
        );
    } else {
        // If no tools parsed, check if the output has the right format at all
        if output.contains("<tool_call>") {
            panic!(
                "Output contains <tool_call> but parser didn't extract it. Output: {}",
                output
            );
        } else if output.contains("<tool_use>") {
            panic!(
                "Model used <tool_use> instead of <tool_call>. Prompt should prevent this. Output: {}",
                output
            );
        } else {
            panic!(
                "Model did not generate any tool call tags. Output: {}",
                output
            );
        }
    }
}
