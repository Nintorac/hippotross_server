//! binidx file format writer (Megatron MMapIndexedDataset format).
//!
//! The binidx format consists of two files:
//! - `.bin`: Raw token data as little-endian u16
//! - `.idx`: Index with document boundaries for random access
//!
//! Format specification (from Megatron-LM):
//! - Magic: b"MMIDIDX\x00\x00" (9 bytes)
//! - Version: u64 LE (value 1)
//! - Dtype: u8 (8 = uint16)
//! - Element count: u64 LE (number of sequences)
//! - Doc count: u64 LE (number of documents)
//! - Sizes: i32[] LE (length of each sequence in tokens)
//! - Pointers: i64[] LE (byte offset of each sequence in .bin)
//! - Doc_idx: i64[] LE (indices into sizes array, one per doc + sentinel)
//!
//! This format is used by RWKV trainers for efficient data loading.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

/// Magic header for Megatron MMapIndexedDataset format.
const IDX_MAGIC: &[u8; 9] = b"MMIDIDX\x00\x00";

/// Version of the index format.
const IDX_VERSION: u64 = 1;

/// Data type code for uint16 tokens (Megatron dtype mapping: 8 = np.uint16).
const DTYPE_UINT16: u8 = 8;

/// Writer for binidx format files.
pub struct BinidxWriter {
    bin_writer: BufWriter<File>,
    sizes: Vec<i32>,
    pointers: Vec<i64>,
    current_byte_offset: i64,
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
            pointers: Vec::new(),
            current_byte_offset: 0,
            total_tokens: 0,
        })
    }

    /// Add a document's tokens to the dataset.
    ///
    /// Tokens are written as little-endian u16. A token 0 (EOS) is automatically
    /// appended after the document.
    pub fn add_document(&mut self, tokens: &[u32]) -> Result<()> {
        // Record byte offset before writing this document
        self.pointers.push(self.current_byte_offset);

        // Write tokens as u16 (RWKV vocab is 65536, fits in u16)
        for &token in tokens {
            let token_u16 = token as u16;
            self.bin_writer.write_all(&token_u16.to_le_bytes())?;
        }

        // Append EOS token (0)
        self.bin_writer.write_all(&0u16.to_le_bytes())?;

        // Track document size (including EOS)
        let doc_size = (tokens.len() + 1) as i32;
        self.sizes.push(doc_size);
        self.total_tokens += doc_size as u64;

        // Update byte offset (each token is 2 bytes for u16)
        self.current_byte_offset += doc_size as i64 * 2;

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

        // Write header: magic (9 bytes) + version (8 bytes) + dtype (1 byte)
        idx_file.write_all(IDX_MAGIC)?;
        idx_file.write_all(&IDX_VERSION.to_le_bytes())?;
        idx_file.write_all(&[DTYPE_UINT16])?;

        // Write element count (number of sequences)
        let num_seqs = self.sizes.len() as u64;
        idx_file.write_all(&num_seqs.to_le_bytes())?;

        // Write doc count (num_seqs + 1 for the sentinel at the end)
        // In Megatron format: doc_idx has one entry per document plus a sentinel
        // For our use case, each sequence IS a document, so doc_count = num_seqs + 1
        let num_docs = num_seqs + 1;
        idx_file.write_all(&num_docs.to_le_bytes())?;

        // Write sizes array (i32 for each sequence)
        for size in &self.sizes {
            idx_file.write_all(&size.to_le_bytes())?;
        }

        // Write pointers array (i64 byte offsets into .bin file)
        for ptr in &self.pointers {
            idx_file.write_all(&ptr.to_le_bytes())?;
        }

        // Write doc_idx array (i64 indices into sizes array)
        // Each document maps to its sequence index, plus a sentinel at the end
        for i in 0..=self.sizes.len() {
            idx_file.write_all(&(i as i64).to_le_bytes())?;
        }

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
        writer.add_document(&[1, 2, 3])?; // Will become [1, 2, 3, 0] (size=4)
        writer.add_document(&[10, 20])?; // Will become [10, 20, 0] (size=3)

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

        // Verify .idx file structure (Megatron MMapIndexedDataset format)
        let mut idx_data = Vec::new();
        File::open(output_path.with_extension("idx"))?.read_to_end(&mut idx_data)?;

        // Header: magic (9 bytes) + version (8 bytes) + dtype (1 byte) = 18 bytes
        assert_eq!(&idx_data[0..9], IDX_MAGIC);
        assert_eq!(&idx_data[9..17], &IDX_VERSION.to_le_bytes());
        assert_eq!(idx_data[17], DTYPE_UINT16);

        // Element count: 2 sequences
        let elem_count = u64::from_le_bytes(idx_data[18..26].try_into().unwrap());
        assert_eq!(elem_count, 2);

        // Doc count: 3 (2 docs + 1 sentinel)
        let doc_count = u64::from_le_bytes(idx_data[26..34].try_into().unwrap());
        assert_eq!(doc_count, 3);

        // Sizes array: [4, 3] (i32 each)
        let size0 = i32::from_le_bytes(idx_data[34..38].try_into().unwrap());
        let size1 = i32::from_le_bytes(idx_data[38..42].try_into().unwrap());
        assert_eq!(size0, 4);
        assert_eq!(size1, 3);

        // Pointers array: [0, 8] (byte offsets, i64 each)
        let ptr0 = i64::from_le_bytes(idx_data[42..50].try_into().unwrap());
        let ptr1 = i64::from_le_bytes(idx_data[50..58].try_into().unwrap());
        assert_eq!(ptr0, 0); // First doc at byte 0
        assert_eq!(ptr1, 8); // Second doc at byte 8 (4 tokens * 2 bytes)

        // Doc_idx array: [0, 1, 2] (indices into sizes, i64 each)
        let doc0 = i64::from_le_bytes(idx_data[58..66].try_into().unwrap());
        let doc1 = i64::from_le_bytes(idx_data[66..74].try_into().unwrap());
        let doc2 = i64::from_le_bytes(idx_data[74..82].try_into().unwrap());
        assert_eq!(doc0, 0);
        assert_eq!(doc1, 1);
        assert_eq!(doc2, 2); // Sentinel

        // Total size check
        // Header: 18, elem_count: 8, doc_count: 8, sizes: 8, pointers: 16, doc_idx: 24 = 82
        assert_eq!(idx_data.len(), 82);

        Ok(())
    }

    #[test]
    fn test_format_matches_sample() -> Result<()> {
        // Test that our format matches the expected Megatron format structure
        let temp_dir = TempDir::new()?;
        let output_path = temp_dir.path().join("test");

        let mut writer = BinidxWriter::new(&output_path)?;

        // Add 7 documents with varying sizes to match sample structure
        writer.add_document(&[1, 2, 3, 4, 5, 6, 7, 8])?; // 9 tokens (8 + EOS)
        writer.add_document(&[1, 2, 3, 4, 5])?; // 6 tokens
        writer.add_document(&vec![1; 19])?; // 20 tokens
        writer.add_document(&vec![1; 21])?; // 22 tokens
        writer.add_document(&vec![1; 26])?; // 27 tokens
        writer.add_document(&vec![1; 25])?; // 26 tokens
        writer.add_document(&vec![1; 49])?; // 50 tokens

        let stats = writer.finish(&output_path)?;
        assert_eq!(stats.num_documents, 7);
        assert_eq!(stats.total_tokens, 160);

        // Verify idx file size matches expected for 7 elements, 8 docs
        // 18 (header) + 8 + 8 + 7*4 + 7*8 + 8*8 = 18 + 16 + 28 + 56 + 64 = 182
        let idx_data = std::fs::read(output_path.with_extension("idx"))?;
        assert_eq!(idx_data.len(), 182);

        // Verify bin file size
        let bin_data = std::fs::read(output_path.with_extension("bin"))?;
        assert_eq!(bin_data.len(), 320); // 160 tokens * 2 bytes

        Ok(())
    }
}
