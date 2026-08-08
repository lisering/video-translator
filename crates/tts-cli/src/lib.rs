//! # vt-tts — 纯 Rust 声音克隆 TTS 引擎
//!
//! 参考和借鉴以下开源项目的架构设计，从零实现：
//! - **Qwen3-TTS** (TrevorS/qwen3-tts-rs) — 三阶段流水线架构、ECAPA-TDNN 说话人编码
//! - **QORA-TTS** (qora-protocol) — 纯 Rust 张量运算、SIMD 优化、系统感知
//! - **neutts-rs** (eugenehp/neutts-rs) — GGUF 骨干网络、NeuCodec 解码器
//! - **IndexTTS-Rust** (8b-is) — ONNX 推理、BigVGAN 声码器
//! - **rs-voice-toolkit** (soddygo) — 异步 trait 设计、多引擎抽象
//!
//! ## 架构
//!
//! 三阶段 TTS 流水线（参考 Qwen3-TTS）：
//!
//! ```text
//! 文本 → [TextTokenizer] → token IDs
//!                              │
//!                              ▼
//!                    ┌─────────────────┐     参考音频 ──→ [SpeakerEncoder] ──→ 说话人嵌入
//!                    │   TalkerModel   │ ←── (speaker_embedding)
//!                    │  (Transformer)  │
//!                    └────────┬────────┘
//!                             │ 语义 token 序列
//!                             ▼
//!                    ┌─────────────────┐
//!                    │  CodePredictor   │
//!                    │ (自回归解码器)    │
//!                    └────────┬────────┘
//!                             │ 声学 token (16 codebooks)
//!                             ▼
//!                    ┌─────────────────┐
//!                    │  AudioDecoder    │
//!                    │ (ConvNeXt + ISTFT)│
//!                    └────────┬────────┘
//!                             │
//!                             ▼
//!                         音频波形 (24kHz)
//! ```

pub mod audio;
pub mod config;
pub mod decoder;
pub mod engine;
pub mod model_config;
pub mod speaker;
pub mod talker;
pub mod tokenizer;
pub mod transformer;

pub use audio::AudioBuffer;
pub use config::TtsEngineConfig;
pub use engine::{SynthesisOptions, SynthesisResult, TtsEngine, VoiceClonePrompt};
