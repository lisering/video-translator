//! Video Translator 核心库 (`vt-core`)
//!
//! 提供视频翻译配音工具的核心数据模型、配置管理、错误处理和音视频处理基础设施。
//!
//! # 模块概览
//! - [`error`][]: 统一错误类型 [`AppError`] 与 [`AppResult<T>`]
//! - [`config`][]: 应用配置管理（TOML 加载、默认值合并）
//! - [`models`][]: 核心数据模型（`Segment`、`SegmentStatus` 状态机）
//! - [`media`][]: 音视频处理（音频提取、视频合成、媒体探测）
//! - [`asr`][]: 语音识别（Whisper 集成、VAD 预处理、模型管理）
//! - [`translate`][]: 翻译模块（DeepLX 在线 + 本地降级两级路由、术语表管理、BLEU 评估）
//! - [`tts`][]: 语音合成模块（KokoroEngine + macOS `say` 降级、批量合成、音频缓存）
//! - [`pipeline`][]: 流水线引擎（ASR → 翻译 → TTS 三阶段异步并行编排）
//! - [`model_manager`][]: 统一模型管理（翻译、ASR、TTS 三类模型的下载、缓存、校验）

pub mod asr;
pub mod audio_post_process;
pub mod batch;
pub mod cfm_decoder;
pub mod checkpoint;
pub mod cloning;
pub mod config;
pub mod config_validation;
pub mod diarization;
pub mod dpo;
pub mod error;
pub mod g2pw;
pub mod gender_detect;
pub mod golden_master;
pub mod vibe_testing;
pub mod meanflow_design;
pub mod media;
pub mod model_manager;
pub mod models;
pub mod moss_tts;
pub mod multi_ref;
pub mod pipeline;
pub mod sentence_split;
pub mod sola;
pub mod speaker_embedding;
pub mod speed_rate;
pub mod streaming_audio;
pub mod subtitle_postprocess;
pub mod text_bucketing;
pub mod text_normalize;
pub mod translate;
pub mod translation_extras;
pub mod tts;
pub mod tts_cache;
pub mod voice_extractor;
pub mod voice_manager;
