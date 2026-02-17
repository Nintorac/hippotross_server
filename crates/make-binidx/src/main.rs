//! make-binidx: Convert JSONL message requests to RWKV binidx format.
//!
//! This tool reads JSONL files containing `/v1/messages` requests and converts
//! them to the binidx format used for RWKV model fine-tuning.
//!
//! Supports streaming from files or stdin for processing large datasets.

mod binidx;

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

use ai00_server::api::messages::prompt::build_training_prompt;
use ai00_server::api::messages::MessagesRequest;
use ai00_server::config::Config;
use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use web_rwkv::tokenizer::Tokenizer;

use binidx::BinidxWriter;

/// Convert JSONL message requests to RWKV binidx format.
///
/// Reads MessagesRequest objects from JSONL input (file or stdin) and converts
/// them to binidx format for RWKV fine-tuning. Processes data in streaming mode
/// to handle files larger than memory.
///
/// Examples:
///   # From file
///   make-binidx -i data.jsonl -o output -t tokenizer.json -p config.toml
///
///   # From stdin (auto-detected)
///   cat data.jsonl | make-binidx -o output -t tokenizer.json -p config.toml
///
///   # Text-only mode for debugging
///   make-binidx -i data.jsonl -p config.toml --text-only
#[derive(Parser, Debug)]
#[command(name = "make-binidx")]
#[command(about = "Convert JSONL message requests to RWKV binidx format")]
#[command(version)]
struct Args {
    /// Input JSONL file containing MessagesRequest objects.
    /// If omitted, reads from stdin (auto-detected when piped).
    #[arg(short, long)]
    input: Option<PathBuf>,

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

    /// Skip rows with more than N tokens (optional filter)
    #[arg(long)]
    max_tokens: Option<usize>,

    /// Output formatted prompts to stdout instead of generating binidx
    #[arg(long)]
    text_only: bool,

    /// Separator between prompts in text-only mode (default: "---")
    #[arg(long, default_value = "---")]
    separator: String,
}

/// Input source for JSONL data.
enum InputSource {
    File(PathBuf),
    Stdin,
}

/// Create a buffered reader from the input source.
fn create_reader(source: &InputSource) -> Result<Box<dyn BufRead>> {
    match source {
        InputSource::File(path) => {
            let file = File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
            Ok(Box::new(BufReader::new(file)))
        }
        InputSource::Stdin => Ok(Box::new(BufReader::new(io::stdin().lock()))),
    }
}

/// Determine the input source based on args and stdin state.
fn get_input_source(args: &Args) -> Result<InputSource> {
    match &args.input {
        Some(path) => Ok(InputSource::File(path.clone())),
        None => {
            // Check if stdin is piped (not a terminal) using atty
            if atty::is(atty::Stream::Stdin) {
                anyhow::bail!(
                    "No input specified and stdin is a terminal.\n\
                     Use --input <file> or pipe data to stdin."
                );
            }
            Ok(InputSource::Stdin)
        }
    }
}

/// Parse a single JSONL line into a MessagesRequest.
fn parse_line(line: &str, line_num: usize) -> Result<Option<MessagesRequest>> {
    let line = line.trim();

    // Skip empty lines
    if line.is_empty() {
        return Ok(None);
    }

    let request: MessagesRequest = serde_json::from_str(line)
        .with_context(|| format!("Failed to parse line {} as MessagesRequest", line_num + 1))?;

    Ok(Some(request))
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

/// Run text-only mode: print formatted prompts to stdout (streaming).
fn run_text_only(args: &Args) -> Result<()> {
    eprintln!("Loading config from {:?}...", args.prompts_config);
    let config = load_prompts_config(&args.prompts_config)?;

    let source = get_input_source(args)?;
    match &source {
        InputSource::File(path) => eprintln!("Streaming from {:?}...", path),
        InputSource::Stdin => eprintln!("Streaming from stdin..."),
    }

    let reader = create_reader(&source)?;
    let mut count = 0usize;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;

        let Some(req) = parse_line(&line, line_num)? else {
            continue;
        };

        // Build training prompt (no trailing assistant prefix)
        let prompt = build_training_prompt(
            req.system.as_deref(),
            &req.messages,
            req.tools.as_deref(),
            req.thinking.as_ref(),
            &config.prompts,
        );

        // Print separator between prompts (not before first)
        if count > 0 {
            println!("{}", args.separator);
        }

        println!("{}", prompt);
        count += 1;
    }

    eprintln!("\nProcessed {} prompts", count);
    Ok(())
}

/// Run binidx generation mode (streaming).
fn run_binidx(args: &Args) -> Result<()> {
    let output_path = args.output.as_ref().unwrap();
    let tokenizer_path = args.tokenizer.as_ref().unwrap();

    eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = load_tokenizer(tokenizer_path)?;

    eprintln!("Loading config from {:?}...", args.prompts_config);
    let config = load_prompts_config(&args.prompts_config)?;

    let source = get_input_source(args)?;
    match &source {
        InputSource::File(path) => eprintln!("Streaming from {:?}...", path),
        InputSource::Stdin => eprintln!("Streaming from stdin..."),
    }

    eprintln!("Creating binidx files at {:?}...", output_path);
    let mut writer = BinidxWriter::new(output_path)?;

    // Progress spinner (unknown total when streaming)
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {pos} documents processed")
            .unwrap(),
    );

    let reader = create_reader(&source)?;
    let mut total_prompt_tokens = 0u64;
    let mut doc_count = 0u64;
    let mut skipped_count = 0u64;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;

        let Some(req) = parse_line(&line, line_num)? else {
            continue;
        };

        // Build training prompt (no trailing assistant prefix)
        let prompt = build_training_prompt(
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

        // Skip if exceeds max_tokens filter
        if let Some(max) = args.max_tokens {
            if tokens.len() > max {
                skipped_count += 1;
                continue;
            }
        }

        total_prompt_tokens += tokens.len() as u64;

        // Write to binidx immediately (adds EOS token)
        writer.add_document(&tokens)?;

        doc_count += 1;
        pb.set_position(doc_count);
    }

    pb.finish_with_message("done");

    // Finish and get stats
    let stats = writer.finish(output_path)?;

    eprintln!("\n=== Statistics ===");
    eprintln!("Documents:    {}", stats.num_documents);
    if skipped_count > 0 {
        eprintln!(
            "Skipped:      {} (exceeded --max-tokens {})",
            skipped_count,
            args.max_tokens.unwrap()
        );
    }
    eprintln!(
        "Total tokens: {} (including EOS markers)",
        stats.total_tokens
    );
    eprintln!("Prompt tokens: {} (before EOS)", total_prompt_tokens);
    eprintln!("Output files: {:?}.bin, {:?}.idx", output_path, output_path);

    // Print magic_prime calculation hint for RWKV trainer
    let data_len = stats.total_tokens;
    let ctx_len = args.ctx_len as u64;
    if data_len > ctx_len {
        let target = data_len / ctx_len - 1;
        eprintln!("\nFor RWKV trainer:");
        eprintln!("  --my_exit_tokens {}", stats.total_tokens);
        eprintln!("  --magic_prime <largest 3n+2 prime less than {}>", target);
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

    // Note: --input is validated in get_input_source() which checks for
    // piped stdin when no input file is provided.

    if args.text_only {
        run_text_only(&args)
    } else {
        run_binidx(&args)
    }
}
