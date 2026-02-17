use anyhow::Result;
use kbnf::{
    engine_like::AcceptTokenError, AcceptTokenResult, Engine, EngineLike, Token, Vocabulary,
};
use web_rwkv::tokenizer::Tokenizer;

use super::Formatter;

#[derive(Debug)]
pub struct BnfSampler(Engine);

impl BnfSampler {
    pub fn new(tokenizer: &Tokenizer, schema: &str) -> Result<Self> {
        let tokens = tokenizer
            .token_index_to_bytes()
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k as u32, Token(v.clone().into_boxed_slice())))
            .collect();
        let strings = tokenizer
            .token_index_to_bytes()
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k as u32, String::from_utf8_lossy(v).to_string()))
            .collect();
        let vocab = Vocabulary::new(tokens, strings)?;
        let mut engine = Engine::new(schema, vocab)?;
        engine.compute_allowed_token_ids();
        Ok(Self(engine))
    }
}

impl Formatter for BnfSampler {
    fn transform(&self, output: &mut [f32]) {
        let output = &mut output[..self.0.vocab().vocab_size()];
        self.0.mask_logits(output).expect("bnf transform error")
    }

    fn update(&mut self, token: u32) -> bool {
        // Grammar "Finished" means "satisfied, could end here" - not "must stop".
        // Let stop strings and EOS token control when to actually stop.
        // Only halt on actual errors (invalid token rejected by grammar).
        let halt = match self.0.try_accept_new_token(token) {
            Ok(AcceptTokenResult::Finished) => false, // Satisfied but can continue
            Ok(AcceptTokenResult::Ongoing) => false,
            Err(AcceptTokenError::Finished) => false, // Also don't halt
            Err(_) => true,                           // Invalid token - halt
        };
        self.0.compute_allowed_token_ids();
        halt
    }
}
