//! make-binidx: Convert JSONL message requests to RWKV binidx format.
//!
//! This tool reads JSONL files containing `/v1/messages` requests and converts
//! them to the binidx format used for RWKV model fine-tuning.

use std::path::PathBuf;

use anyhow::Result;
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

    println!("make-binidx v{}", env!("CARGO_PKG_VERSION"));
    println!("Input: {:?}", args.input);

    if args.text_only {
        println!("Mode: text-only (separator: {:?})", args.separator);
    } else {
        println!("Output: {:?}", args.output.as_ref().unwrap());
        println!("Tokenizer: {:?}", args.tokenizer.as_ref().unwrap());
        println!("Context length: {}", args.ctx_len);
        println!("Repeat: {}", args.repeat);
    }

    // TODO: Implement actual processing
    println!("\n[Skeleton - implementation pending]");

    Ok(())
}
