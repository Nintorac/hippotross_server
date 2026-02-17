//! Mock implementations for testing.
//!
//! Provides mock ThreadSender that responds with predetermined tokens.

#![allow(dead_code)]

use ai00_core::{FinishReason, ThreadRequest, Token, TokenCounter};
use flume::Sender;
use std::time::Duration;

/// Create a mock ThreadSender that responds with predetermined text.
pub fn create_mock_sender(text_response: &str) -> Sender<ThreadRequest> {
    let (tx, rx) = flume::unbounded::<ThreadRequest>();
    let response = text_response.to_string();

    tokio::spawn(async move {
        while let Ok(request) = rx.recv_async().await {
            match request {
                ThreadRequest::Generate { sender, .. } => {
                    let _ = sender.send(Token::Start);
                    let _ = sender.send(Token::Content(response.clone()));
                    let _ = sender.send(Token::Stop(
                        FinishReason::Stop,
                        TokenCounter {
                            prompt: 10,
                            completion: response.len() / 4,
                            total: 10 + response.len() / 4,
                            duration: Duration::from_millis(100),
                        },
                    ));
                    let _ = sender.send(Token::Done);
                }
                ThreadRequest::Info(info_sender) => {
                    // For info requests, we'd need a RuntimeInfo
                    // Skip for now as tests may not need this
                    drop(info_sender);
                }
                _ => {}
            }
        }
    });

    tx
}

/// Create a mock sender that streams tokens one at a time.
pub fn create_streaming_mock_sender(tokens: Vec<&str>) -> Sender<ThreadRequest> {
    let (tx, rx) = flume::unbounded::<ThreadRequest>();
    let tokens: Vec<String> = tokens.into_iter().map(String::from).collect();

    tokio::spawn(async move {
        while let Ok(request) = rx.recv_async().await {
            if let ThreadRequest::Generate { sender, .. } = request {
                let _ = sender.send(Token::Start);
                for token in &tokens {
                    let _ = sender.send(Token::Content(token.clone()));
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                let _ = sender.send(Token::Stop(
                    FinishReason::Stop,
                    TokenCounter {
                        prompt: 10,
                        completion: tokens.len(),
                        total: 10 + tokens.len(),
                        duration: Duration::from_millis(tokens.len() as u64 * 10),
                    },
                ));
                let _ = sender.send(Token::Done);
            }
        }
    });

    tx
}

/// Create a mock sender that returns a length-limited response.
pub fn create_length_limited_mock_sender(text: &str) -> Sender<ThreadRequest> {
    let (tx, rx) = flume::unbounded::<ThreadRequest>();
    let response = text.to_string();

    tokio::spawn(async move {
        while let Ok(request) = rx.recv_async().await {
            if let ThreadRequest::Generate { sender, .. } = request {
                let _ = sender.send(Token::Start);
                let _ = sender.send(Token::Content(response.clone()));
                let _ = sender.send(Token::Stop(
                    FinishReason::Length,
                    TokenCounter {
                        prompt: 10,
                        completion: response.len() / 4,
                        total: 10 + response.len() / 4,
                        duration: Duration::from_millis(100),
                    },
                ));
                let _ = sender.send(Token::Done);
            }
        }
    });

    tx
}
