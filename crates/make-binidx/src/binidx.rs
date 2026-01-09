//! binidx file format writer.
//!
//! The binidx format consists of two files:
//! - `.bin`: Raw token data as little-endian u16
//! - `.idx`: Index with document boundaries for random access
//!
//! This format is used by RWKV trainers for efficient data loading.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

/// Magic number for the .idx file header.
const IDX_MAGIC: u64 = 0x584449; // "IDX" in ASCII

/// Version of the index format.
const IDX_VERSION: u64 = 1;

/// Data type code for uint16 tokens.
const DTYPE_UINT16: u64 = 3;

/// Writer for binidx format files.
pub struct BinidxWriter {
    bin_writer: BufWriter<File>,
    sizes: Vec<u32>,
    total_tokens: u64,
}

impl BinidxWriter {
    /// Create a new binidx writer.
    ///
    /// Creates `{output_path}.bin` for token data.
    /// The `.idx` file is written when `finish()` is called.
    pub fn new(output_path: &Path) -> Result<Self> {
        let bin_path = output_path.with_extension("bin");
        let bin_file =
            File::create(&bin_path).with_context(|| format!("Failed to create {:?}", bin_path))?;

        Ok(Self {
            bin_writer: BufWriter::new(bin_file),
            sizes: Vec::new(),
            total_tokens: 0,
        })
    }

    /// Add a document's tokens to the dataset.
    ///
    /// Tokens are written as little-endian u16. A token 0 (EOS) is automatically
    /// appended after the document.
    pub fn add_document(&mut self, tokens: &[u32]) -> Result<()> {
        // Write tokens as u16 (RWKV vocab is 65536, fits in u16)
        for &token in tokens {
            let token_u16 = token as u16;
            self.bin_writer.write_all(&token_u16.to_le_bytes())?;
        }

        // Append EOS token (0)
        self.bin_writer.write_all(&0u16.to_le_bytes())?;

        // Track document size (including EOS)
        let doc_size = tokens.len() as u32 + 1;
        self.sizes.push(doc_size);
        self.total_tokens += doc_size as u64;

        Ok(())
    }

    /// Finish writing and create the .idx file.
    ///
    /// Returns statistics about the written data.
    pub fn finish(mut self, output_path: &Path) -> Result<BinidxStats> {
        // Flush the bin file
        self.bin_writer.flush()?;

        // Write the idx file
        let idx_path = output_path.with_extension("idx");
        let mut idx_file =
            File::create(&idx_path).with_context(|| format!("Failed to create {:?}", idx_path))?;

        // Write header
        idx_file.write_all(&IDX_MAGIC.to_le_bytes())?;
        idx_file.write_all(&IDX_VERSION.to_le_bytes())?;
        idx_file.write_all(&DTYPE_UINT16.to_le_bytes())?;

        // Write number of documents
        let num_docs = self.sizes.len() as u64;
        idx_file.write_all(&num_docs.to_le_bytes())?;

        // Write document sizes
        for size in &self.sizes {
            idx_file.write_all(&size.to_le_bytes())?;
        }

        // Write document offsets (cumulative sum)
        let mut offset: u64 = 0;
        for size in &self.sizes {
            idx_file.write_all(&offset.to_le_bytes())?;
            offset += *size as u64;
        }

        // Write total token count
        idx_file.write_all(&self.total_tokens.to_le_bytes())?;

        Ok(BinidxStats {
            num_documents: self.sizes.len(),
            total_tokens: self.total_tokens,
        })
    }
}

/// Statistics about a written binidx dataset.
#[derive(Debug, Clone)]
pub struct BinidxStats {
    pub num_documents: usize,
    pub total_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::TempDir;

    #[test]
    fn test_write_binidx() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let output_path = temp_dir.path().join("test");

        let mut writer = BinidxWriter::new(&output_path)?;

        // Add two documents
        writer.add_document(&[1, 2, 3])?; // Will become [1, 2, 3, 0]
        writer.add_document(&[10, 20])?; // Will become [10, 20, 0]

        let stats = writer.finish(&output_path)?;

        assert_eq!(stats.num_documents, 2);
        assert_eq!(stats.total_tokens, 7); // 4 + 3

        // Verify .bin file
        let mut bin_data = Vec::new();
        File::open(output_path.with_extension("bin"))?.read_to_end(&mut bin_data)?;

        // Each token is 2 bytes (u16 little-endian)
        assert_eq!(bin_data.len(), 14); // 7 tokens * 2 bytes

        // Check first document: [1, 2, 3, 0]
        assert_eq!(&bin_data[0..2], &1u16.to_le_bytes());
        assert_eq!(&bin_data[2..4], &2u16.to_le_bytes());
        assert_eq!(&bin_data[4..6], &3u16.to_le_bytes());
        assert_eq!(&bin_data[6..8], &0u16.to_le_bytes()); // EOS

        // Check second document: [10, 20, 0]
        assert_eq!(&bin_data[8..10], &10u16.to_le_bytes());
        assert_eq!(&bin_data[10..12], &20u16.to_le_bytes());
        assert_eq!(&bin_data[12..14], &0u16.to_le_bytes()); // EOS

        // Verify .idx file exists and has correct header
        let mut idx_data = Vec::new();
        File::open(output_path.with_extension("idx"))?.read_to_end(&mut idx_data)?;

        // Check header
        assert_eq!(&idx_data[0..8], &IDX_MAGIC.to_le_bytes());
        assert_eq!(&idx_data[8..16], &IDX_VERSION.to_le_bytes());
        assert_eq!(&idx_data[16..24], &DTYPE_UINT16.to_le_bytes());
        assert_eq!(&idx_data[24..32], &2u64.to_le_bytes()); // num_docs

        Ok(())
    }
}
