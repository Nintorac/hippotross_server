//! Integration tests for make-binidx CLI.
//!
//! These tests verify streaming functionality from both file and stdin sources.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::process::{Command, Stdio};

use tempfile::TempDir;

/// Path to assets directory (relative to workspace root).
fn assets_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets")
}

/// Get path to the compiled binary.
fn binary_path() -> std::path::PathBuf {
    // The binary is built in the workspace target directory
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target")
        .join("debug")
        .join("make-binidx")
}

/// Create a minimal test JSONL file.
fn create_test_jsonl(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("test.jsonl");
    let mut file = File::create(&path).unwrap();
    writeln!(
        file,
        r#"{{"model":"rwkv","system":"You are helpful.","messages":[{{"role":"user","content":"Hi"}},{{"role":"assistant","content":"Hello!"}}],"max_tokens":100}}"#
    )
    .unwrap();
    writeln!(
        file,
        r#"{{"model":"rwkv","messages":[{{"role":"user","content":"Test"}}],"max_tokens":100}}"#
    )
    .unwrap();
    path
}

/// Create a minimal prompts config for testing.
/// Uses default PromptsConfig values from ai00-server.
fn create_test_config(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("config.toml");
    // Minimal config - prompts section uses defaults:
    // role_user = "User", role_assistant = "Assistant", role_system = "System"
    // assistant_prefix = "Assistant:"
    let config = r#"
[prompts]
# Uses defaults - no overrides needed
"#;
    fs::write(&path, config).unwrap();
    path
}

#[test]
fn test_text_only_from_file() {
    let temp_dir = TempDir::new().unwrap();
    let jsonl_path = create_test_jsonl(&temp_dir);
    let config_path = create_test_config(&temp_dir);

    let output = Command::new(binary_path())
        .args([
            "--input",
            jsonl_path.to_str().unwrap(),
            "--prompts-config",
            config_path.to_str().unwrap(),
            "--text-only",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "Command failed: {}", stderr);
    assert!(
        stderr.contains("Streaming from"),
        "Should show streaming source"
    );
    assert!(
        stderr.contains("Processed 2 prompts"),
        "Should process 2 prompts"
    );

    // Check output contains expected content (uses ai00 XML format)
    assert!(stdout.contains("<ai00:system>"), "Missing system opening tag");
    assert!(stdout.contains("You are helpful."), "Missing system content");
    assert!(stdout.contains("</ai00:system>"), "Missing system closing tag");
    assert!(stdout.contains("<ai00:user>"), "Missing user opening tag");
    assert!(stdout.contains("Hi"), "Missing user message");
    assert!(stdout.contains("</ai00:user>"), "Missing user closing tag");
    assert!(stdout.contains("<ai00:assistant>"), "Missing assistant opening tag");
    assert!(stdout.contains("Hello!"), "Missing assistant response");
    assert!(stdout.contains("</ai00:assistant>"), "Missing assistant closing tag");
    assert!(stdout.contains("---"), "Missing separator between prompts");
    // Training prompts should NOT have trailing assistant prefix
    assert!(
        !stdout.ends_with("<ai00:assistant>\n"),
        "Should not have trailing assistant prefix"
    );
}

#[test]
fn test_text_only_from_stdin() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = create_test_config(&temp_dir);

    let jsonl_data = r#"{"model":"rwkv","messages":[{"role":"user","content":"Stdin test"}],"max_tokens":100}"#;

    let mut child = Command::new(binary_path())
        .args([
            "--prompts-config",
            config_path.to_str().unwrap(),
            "--text-only",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn command");

    // Write to stdin
    child
        .stdin
        .take()
        .unwrap()
        .write_all(jsonl_data.as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("Failed to wait on child");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "Command failed: {}", stderr);
    assert!(
        stderr.contains("Streaming from stdin"),
        "Should indicate stdin source"
    );
    assert!(stdout.contains("<ai00:user>"), "Missing user opening tag");
    assert!(stdout.contains("Stdin test"), "Missing stdin content");
    assert!(stdout.contains("</ai00:user>"), "Missing user closing tag");
}

#[test]
fn test_binidx_from_file() {
    let temp_dir = TempDir::new().unwrap();
    let jsonl_path = create_test_jsonl(&temp_dir);
    let config_path = create_test_config(&temp_dir);
    let output_path = temp_dir.path().join("output");
    let tokenizer_path = assets_dir().join("tokenizer/rwkv_vocab_v20230424.json");

    // Skip test if tokenizer not available
    if !tokenizer_path.exists() {
        eprintln!("Skipping test: tokenizer not found at {:?}", tokenizer_path);
        return;
    }

    let output = Command::new(binary_path())
        .args([
            "--input",
            jsonl_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--prompts-config",
            config_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "Command failed: {}", stderr);
    assert!(stderr.contains("Documents:    2"), "Should have 2 documents");

    // Verify output files exist
    let bin_path = output_path.with_extension("bin");
    let idx_path = output_path.with_extension("idx");
    assert!(bin_path.exists(), "Missing .bin file");
    assert!(idx_path.exists(), "Missing .idx file");

    // Verify .bin has content
    let bin_size = fs::metadata(&bin_path).unwrap().len();
    assert!(bin_size > 0, "Empty .bin file");

    // Verify .idx has valid header (Megatron MMapIndexedDataset format)
    let mut idx_file = File::open(&idx_path).unwrap();
    let mut magic_bytes = [0u8; 9];
    idx_file.read_exact(&mut magic_bytes).unwrap();
    assert_eq!(&magic_bytes, b"MMIDIDX\x00\x00", "Invalid idx magic number");
}

#[test]
fn test_binidx_from_stdin() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = create_test_config(&temp_dir);
    let output_path = temp_dir.path().join("output");
    let tokenizer_path = assets_dir().join("tokenizer/rwkv_vocab_v20230424.json");

    // Skip test if tokenizer not available
    if !tokenizer_path.exists() {
        eprintln!("Skipping test: tokenizer not found at {:?}", tokenizer_path);
        return;
    }

    let jsonl_data = r#"{"model":"rwkv","messages":[{"role":"user","content":"Binidx stdin test"}],"max_tokens":100}"#;

    let mut child = Command::new(binary_path())
        .args([
            "--output",
            output_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--prompts-config",
            config_path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn command");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(jsonl_data.as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("Failed to wait on child");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "Command failed: {}", stderr);
    assert!(
        stderr.contains("Streaming from stdin"),
        "Should indicate stdin"
    );
    assert!(stderr.contains("Documents:    1"), "Should have 1 document");

    // Verify output files
    assert!(
        output_path.with_extension("bin").exists(),
        "Missing .bin file"
    );
    assert!(
        output_path.with_extension("idx").exists(),
        "Missing .idx file"
    );
}

#[test]
fn test_custom_separator() {
    let temp_dir = TempDir::new().unwrap();
    let jsonl_path = create_test_jsonl(&temp_dir);
    let config_path = create_test_config(&temp_dir);

    let output = Command::new(binary_path())
        .args([
            "--input",
            jsonl_path.to_str().unwrap(),
            "--prompts-config",
            config_path.to_str().unwrap(),
            "--text-only",
            "--separator",
            "===BREAK===",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("===BREAK==="),
        "Should use custom separator"
    );
    assert!(!stdout.contains("---"), "Should not use default separator");
}

#[test]
fn test_empty_lines_skipped() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = create_test_config(&temp_dir);

    // JSONL with empty lines
    let jsonl_path = temp_dir.path().join("sparse.jsonl");
    let mut file = File::create(&jsonl_path).unwrap();
    writeln!(file).unwrap(); // Empty line
    writeln!(
        file,
        r#"{{"model":"rwkv","messages":[{{"role":"user","content":"One"}}],"max_tokens":100}}"#
    )
    .unwrap();
    writeln!(file).unwrap(); // Empty line
    writeln!(file, "   ").unwrap(); // Whitespace only
    writeln!(
        file,
        r#"{{"model":"rwkv","messages":[{{"role":"user","content":"Two"}}],"max_tokens":100}}"#
    )
    .unwrap();
    writeln!(file).unwrap(); // Trailing empty line

    let output = Command::new(binary_path())
        .args([
            "--input",
            jsonl_path.to_str().unwrap(),
            "--prompts-config",
            config_path.to_str().unwrap(),
            "--text-only",
        ])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "Command failed: {}", stderr);
    assert!(
        stderr.contains("Processed 2 prompts"),
        "Should process only 2 non-empty lines"
    );
}

#[test]
fn test_missing_required_args_binidx_mode() {
    let temp_dir = TempDir::new().unwrap();
    let jsonl_path = create_test_jsonl(&temp_dir);
    let config_path = create_test_config(&temp_dir);

    // Missing --output
    let output = Command::new(binary_path())
        .args([
            "--input",
            jsonl_path.to_str().unwrap(),
            "--prompts-config",
            config_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success(), "Should fail without --output");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--output is required"),
        "Should mention missing --output"
    );
}

#[test]
fn test_missing_tokenizer_binidx_mode() {
    let temp_dir = TempDir::new().unwrap();
    let jsonl_path = create_test_jsonl(&temp_dir);
    let config_path = create_test_config(&temp_dir);
    let output_path = temp_dir.path().join("output");

    // Missing --tokenizer
    let output = Command::new(binary_path())
        .args([
            "--input",
            jsonl_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
            "--prompts-config",
            config_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success(), "Should fail without --tokenizer");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--tokenizer is required"),
        "Should mention missing --tokenizer"
    );
}
