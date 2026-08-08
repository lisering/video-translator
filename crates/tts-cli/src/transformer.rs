//! Transformer 构建块 — Qwen3-TTS 共享组件
//!
//! 包含 KVCache、RoPE/MRoPE、Attention (GQA + QK norm)、MLP (SwiGLU)、DecoderLayer。
//! 参考 TrevorS/qwen3-tts-rs 的 `src/models/transformer.rs`。

pub mod attention;
pub mod kv_cache;
pub mod layer;
pub mod mlp;
pub mod qlinear;
pub mod rope;

// Re-export for backward compatibility (crate::transformer::*)
pub use attention::Attention;
pub use kv_cache::{AnyKVCache, KVCache};
pub use layer::DecoderLayer;
pub use mlp::Mlp;
pub use qlinear::QLinear;
pub use rope::{MRoPE, RoPEType, RotaryEmbedding};
