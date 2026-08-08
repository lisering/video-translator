//! MOSS-TTS 借鉴模块 — 纯 Rust 实现的 TTS 推理核心逻辑
//!
//! 从 MOSS-TTS 项目移植的纯 Rust 模块，
//! 消除对 Python subprocess 的依赖，实现原生 Rust TTS 推理。
//!
//! # 模块概览
//! - [`constants`][]: Token ID 常量（从 config.json 加载，含 hardcoded fallback）
//! - [`sampling`][]: 采样函数（top_k/top_p/softmax/multinomial/repetition_penalty）
//! - [`delay_state`][]: Delay Pattern 状态机（自回归生成核心逻辑）
//! - [`embedding`][]: 多通道 Embedding 查找（33 个表求和）
//! - [`lm_heads`][]: 33 个预测头的 LM head 投影（预堆叠 + BLAS matmul）
//! - [`prompt`][]: 生成 prompt 构建 + 输出解析
//!
//! # 架构
//! 借鉴 MOSS-TTS 的 llama_cpp 后端架构：
//! 1. Token 常量 + 采样 → 纯计算，无外部依赖
//! 2. Embedding 查找 → 预加载 .npy 权重，Rayon 并行求和
//! 3. LM head 投影 → 预堆叠权重，单次 BLAS matmul
//! 4. Delay 状态机 → 纯 NumPy 逻辑的 Rust 移植
//! 5. Prompt 构建 → BPE tokenizer + delay pattern 编码
//!
//! # 数据流
//! ```text
//! text → build_generation_prompt() → input_ids (S, 33)
//!      → EmbeddingLookup → hidden_state (hidden_dim,)
//!      → [backbone: llama.cpp / Candle / Python subprocess]
//!      → backbone hidden_state
//!      → LMHeads → text_logits + audio_logits (32, vocab)
//!      → delay_state.step() → next_input_ids (33,)
//!      → 循环直到 is_stopping
//!      → parse_generation_output() → text + audio_codes
//! ```

pub mod audio_tokenizer;
pub mod batch_eval;
pub mod constants;
pub mod delay_state;
pub mod embedding;
pub mod lm_heads;
pub mod prompt;
pub mod sampling;
pub mod streaming_speaker;
pub mod tts_control;

pub use audio_tokenizer::{
    estimate_duration_secs, estimate_frames, DecodeRequest, DecodeResult, EncodeRequest,
    EncodeResult, OnnxAudioTokenizer, OnnxAudioTokenizerConfig,
};
pub use batch_eval::{
    builtin_eval_texts, BatchEvalSummary, EvalParams, EvalText, EvalTextSet, EvalTimer,
    TtsEvalResult,
};
pub use constants::{MossTtsConstants, N_VQ, SAMPLE_RATE};
pub use delay_state::{
    apply_de_delay_pattern, apply_delay_pattern, extract_audio_segments, init_delay_state, step,
    DelayState, SamplingConfig,
};
pub use embedding::EmbeddingLookup;
pub use lm_heads::LmHeads;
pub use prompt::{build_generation_prompt, parse_generation_output, MossTtsTokenizer};
pub use sampling::{apply_repetition_penalty, apply_top_k, apply_top_p, sample_token, softmax};
pub use streaming_speaker::{
    Speaker, SpeakerManager, StreamState, StreamingAudioChunk, StreamingCallback, StreamingRequest,
    StreamingTts,
};
pub use tts_control::{
    estimate_audio_duration, estimate_tokens_for_duration, to_moss_language_tag, tokens_per_char,
    TtsControlParams,
};
