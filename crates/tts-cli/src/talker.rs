//! Talker 模型 + 核心引擎实现
//!
//! 基于 Candle ML 框架实现 Qwen3-TTS 三阶段流水线：
//! 1. **TalkerModel** (Transformer): 文本 → 语义 token 序列
//! 2. **CodePredictor** (自回归解码器): 语义 token → 声学 token (16 codebooks)
//! 3. **AudioDecoder**: codec tokens → 音频波形
//!
//! 参考 TrevorS/qwen3-tts-rs 的实现。

pub mod candle_engine;
pub mod code_predictor;
pub mod model;
pub mod sampling;
pub mod tokens;
pub mod types;
pub mod weights;

// Re-export for backward compatibility (crate::talker::* / vt_tts::talker::*)
pub use candle_engine::CandleTtsEngine;
pub use code_predictor::CodePredictor;
pub use model::TalkerModel;
pub use sampling::{
    argmax, argmax_on_device, is_ngram_banned, parse_quantize, sample_top_k, sample_top_k_gpu,
    update_ngram_table,
};
pub use tokens::{codec_tokens, special_tokens, tts_tokens};
pub use types::{Language, Speaker};
pub use weights::{
    compute_dtype_for_device, convert_weights_dtype, create_device, load_safetensors,
};
