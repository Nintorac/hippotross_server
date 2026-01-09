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
fn run_binidx(_args: &Args) -> Result<()> {
    // TODO: Implement binidx generation
    anyhow::bail!("Binidx generation not yet implemented")
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
