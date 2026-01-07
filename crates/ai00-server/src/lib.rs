//! ai00-server library crate.
//!
//! This module exports the API types and handlers for testing.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Result};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};

pub mod api;
pub mod config;
pub mod types;

/// Sleep duration between retry attempts.
pub const SLEEP: Duration = Duration::from_millis(500);

/// Build a path from a permitted base directory and a name.
///
/// Returns an error if the resulting path would escape the permitted directory.
pub fn build_path(path: impl AsRef<Path>, name: impl AsRef<Path>) -> Result<PathBuf> {
    let permitted = path.as_ref();
    let name = name.as_ref();
    if name.ancestors().any(|p| p.ends_with(Path::new(".."))) {
        bail!("cannot have \"..\" in names");
    }
    let path = match name.is_absolute() || name.starts_with(permitted) {
        true => name.into(),
        false => permitted.join(name),
    };
    match path.starts_with(permitted) {
        true => Ok(path),
        false => bail!("path not permitted"),
    }
}

/// Check if a path is within any of the permitted directories.
pub fn check_path_permitted(path: impl AsRef<Path>, permitted: &[&str]) -> Result<()> {
    let current_path = std::env::current_dir()?;
    for sub in permitted {
        let permitted = current_path.join(sub).canonicalize()?;
        let path = path.as_ref().canonicalize()?;
        if path.starts_with(permitted) {
            return Ok(());
        }
    }
    bail!("path not permitted")
}

/// Load a configuration file from the given path.
pub async fn load_config(path: impl AsRef<Path>) -> Result<config::Config> {
    let file = File::open(path).await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(toml::from_str(&contents)?)
}

/// Text embedding model wrapper.
#[cfg(feature = "embed")]
pub struct TextEmbed {
    pub tokenizer: tokenizers::Tokenizer,
    pub model: fastembed::TextEmbedding,
    pub info: fastembed::ModelInfo<fastembed::EmbeddingModel>,
}
