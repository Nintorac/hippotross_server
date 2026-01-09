//! make-binidx: Convert JSONL message requests to RWKV binidx format.
//!
//! This tool reads JSONL files containing `/v1/messages` requests and converts
//! them to the binidx format used for RWKV model fine-tuning.

mod binidx;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use ai00_server::api::messages::prompt::build_prompt;
use ai00_server::api::messages::MessagesRequest;
use ai00_server::config::Config;
use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use web_rwkv::tokenizer::Tokenizer;

use binidx::BinidxWriter;

/// Convert JSONL message requests to RWKV binidx format.
#[derive(Parser, Debug)]
#[command(name = "make-binidx")]
#[command(about = "Convert JSONL message requests to RWKV binidx format")]
#[command(version)]
struct Args {
    /// Input JSONL file containing MessagesRequest objects
    #[arg(short, long)]
    input: PathBuf,

    /// Output file basename (creates .bin and .idx files)
    /// Required unless --text-only is set
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to tokenizer JSON file
    /// Required unless --text-only is set
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,

    /// Path to prompts config TOML file
    #[arg(short, long)]
    prompts_config: PathBuf,

    /// Context length for chunking (default: 4096)
    #[arg(long, default_value = "4096")]
    ctx_len: usize,

    /// Number of times to repeat and shuffle the data (default: 1)
    #[arg(long, default_value = "1")]
    repeat: usize,

    /// Output formatted prompts to stdout instead of generating binidx
    #[arg(long)]
    text_only: bool,

    /// Separator between prompts in text-only mode (default: "---")
    #[arg(long, default_value = "---")]
    separator: String,
}

/// Read and parse JSONL file into MessagesRequest objects.
fn read_jsonl(path: &PathBuf) -> Result<Vec<MessagesRequest>> {
    let file = File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
    let reader = BufReader::new(file);

    let mut requests = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        let request: MessagesRequest = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse line {} as MessagesRequest", line_num + 1))?;

        requests.push(request);
    }

    Ok(requests)
}

/// Load config and extract PromptsConfig.
fn load_prompts_config(path: &PathBuf) -> Result<Config> {
    let contents =
        std::fs::read_to_string(path).with_context(|| format!("Failed to read {:?}", path))?;

    let config: Config =
        toml::from_str(&contents).with_context(|| format!("Failed to parse {:?}", path))?;

    Ok(config)
}

/// Load the RWKV tokenizer from JSON file.
fn load_tokenizer(path: &PathBuf) -> Result<Tokenizer> {
    let contents =
        std::fs::read_to_string(path).with_context(|| format!("Failed to read {:?}", path))?;

    Tokenizer::new(&contents).with_context(|| format!("Failed to parse tokenizer {:?}", path))
}

/// Run text-only mode: print formatted prompts to stdout.
fn run_text_only(args: &Args) -> Result<()> {
    eprintln!("Loading config from {:?}...", args.prompts_config);
    let config = load_prompts_config(&args.prompts_config)?;

    eprintln!("Reading JSONL from {:?}...", args.input);
    let requests = read_jsonl(&args.input)?;
    eprintln!("Loaded {} requests", requests.len());

    for (i, req) in requests.iter().enumerate() {
        // Build prompt using exact server code path
        let prompt = build_prompt(
            req.system.as_deref(),
            &req.messages,
            req.tools.as_deref(),
            req.thinking.as_ref(),
            &config.prompts,
        );

        // Print separator between prompts (not before first)
        if i > 0 {
            println!("{}", args.separator);
        }

        println!("{}", prompt);
    }

    eprintln!("\nProcessed {} prompts", requests.len());
    Ok(())
}

/// Run binidx generation mode.
fn run_binidx(args: &Args) -> Result<()> {
    let output_path = args.output.as_ref().unwrap();
    let tokenizer_path = args.tokenizer.as_ref().unwrap();

    eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = load_tokenizer(tokenizer_path)?;

    eprintln!("Loading config from {:?}...", args.prompts_config);
    let config = load_prompts_config(&args.prompts_config)?;

    eprintln!("Reading JSONL from {:?}...", args.input);
    let requests = read_jsonl(&args.input)?;
    eprintln!("Loaded {} requests", requests.len());

    eprintln!("Creating binidx files at {:?}...", output_path);
    let mut writer = BinidxWriter::new(output_path)?;

    // Progress bar
    let pb = ProgressBar::new(requests.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut total_prompt_tokens = 0u64;

    for req in &requests {
        // Build prompt using exact server code path
        let prompt = build_prompt(
            req.system.as_deref(),
            &req.messages,
            req.tools.as_deref(),
            req.thinking.as_ref(),
            &config.prompts,
        );

        // Tokenize using same approach as server:
        // Token 0 prefix + encoded prompt
        let mut tokens = vec![0u32];
        tokens.extend(
            tokenizer
                .encode(prompt.as_bytes())
                .with_context(|| "Failed to tokenize prompt")?,
        );

        total_prompt_tokens += tokens.len() as u64;

        // Write to binidx (adds EOS token)
        writer.add_document(&tokens)?;

        pb.inc(1);
    }

    pb.finish_with_message("done");

    // Finish and get stats
    let stats = writer.finish(output_path)?;

    eprintln!("\n=== Statistics ===");
    eprintln!("Documents:    {}", stats.num_documents);
    eprintln!("Total tokens: {} (including EOS markers)", stats.total_tokens);
    eprintln!("Prompt tokens: {} (before EOS)", total_prompt_tokens);
    eprintln!(
        "Output files: {:?}.bin, {:?}.idx",
        output_path, output_path
    );

    // Print magic_prime calculation hint for RWKV trainer
    let data_len = stats.total_tokens;
    let ctx_len = args.ctx_len as u64;
    if data_len > ctx_len {
        let target = data_len / ctx_len - 1;
        eprintln!("\nFor RWKV trainer:");
        eprintln!("  --my_exit_tokens {}", stats.total_tokens);
        eprintln!(
            "  --magic_prime <largest 3n+2 prime less than {}>",
            target
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate arguments
    if !args.text_only {
        if args.output.is_none() {
            anyhow::bail!("--output is required unless --text-only is set");
        }
        if args.tokenizer.is_none() {
            anyhow::bail!("--tokenizer is required unless --text-only is set");
        }
    }

    if args.text_only {
        run_text_only(&args)
    } else {
        run_binidx(&args)
    }
}
