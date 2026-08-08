//! 流水线引擎模块
//!
//! 将 ASR（语音识别）、翻译、TTS（语音合成）三阶段串联为异步并行流水线，
//! 实现阶段间并行处理：ASR 处理第 N 段时，翻译第 N-1 段，TTS 第 N-2 段。
//!
//! # 架构
//! ```text
//! Video → AudioExtractor → WAV → Split → Chunks
//!                                        │
//!                    ┌───────────────────┘
//!                    ▼
//!              ┌─ Channel 1 ─┐    ┌─ Channel 2 ─┐    ┌─ Channel 3 ─┐
//!  Chunks ──▶ │   ASR Stage  │ ─▶│ Translation │ ─▶│   TTS Stage  │ ─▶ Output
//!              └──────────────┘    └──────────────┘    └──────────────┘
//! ```
//!
//! # 背压控制
//! 使用 `tokio::sync::mpsc` 有界通道，当下游阶段处理速度慢于上游时，
//! 上游发送操作会自动阻塞，防止内存无限增长。
//!
//! # 错误恢复
//! 单个 Segment 的 ASR/翻译/TTS 失败不会中断整个流水线，
//! 失败的 Segment 会被跳过并记录日志，已完成的 Segment 不受影响。
//!
//! # 示例
//! ```no_run
//! use vt_core::pipeline::{Pipeline, PipelineBuilder};
//! use vt_core::config::Config;
//! use vt_core::error::AppResult;
//!
//! # async fn run() -> AppResult<()> {
//! let pipeline = PipelineBuilder::default()
//!     // .asr_engine(...)
//!     // .translation_provider(...)
//!     // .tts_engine(...)
//!     // .audio_extractor(...)
//!     .build()?;
//!
//! let config = Config::default();
//! let segments = pipeline.process_video(std::path::Path::new("input.mp4"), &config).await?;
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::asr::{detect_speech_segments, read_wav_mono, AsrEngine, VadConfig};
use crate::audio_post_process::AudioPostProcessor;
use crate::cloning::CloningIntegration;
use crate::config::{Config, TranslationMode};
use crate::error::{AppError, AppResult};
use crate::media::AudioExtractor;
use crate::models::segment::Segment;
use crate::speed_rate::{SpeedRateConfig, SpeedRateProcessor};
use crate::text_normalize::{clean_asr_output, normalize_text, NormalizationConfig};
use crate::translate::TranslationProvider;
use crate::tts::TtsEngine;
use crate::tts_cache::{TranslationCache, TtsCache};
use crate::voice_extractor::VoiceExtractor;

// ─── 内部数据结构 ─────────────────────────────────────────

/// 音频片段信息（内部使用）
///
/// 表示分割后的一个音频片段，包含文件路径和时间范围。
struct AudioChunkInfo {
    /// 片段索引（从 0 开始）
    index: usize,
    /// WAV 文件路径
    path: PathBuf,
    /// 起始时间（秒，相对于原始音频）
    start_time: f64,
    /// 结束时间（秒，相对于原始音频）
    end_time: f64,
}

/// 音频片段数据（内部使用）
///
/// 包含原始采样数据和时间范围，用于写入 WAV 文件。
struct AudioChunkData {
    /// 起始时间（秒）
    start_time: f64,
    /// 结束时间（秒）
    end_time: f64,
    /// 采样数据
    samples: Vec<f32>,
}

// ─── 进度追踪器 ───────────────────────────────────────────

/// 流水线进度追踪器
///
/// 使用原子计数器追踪 ASR、翻译、TTS 三阶段的实时进度，线程安全。
/// 可被多个 tokio 任务共享，通过 `Arc` 克隆传递。
///
/// # 进度计算
/// 整体进度分两大阶段：
/// - **ASR 阶段**（5%–50%）：按 `asr_completed / total_chunks` 线性增长
/// - **TTS 阶段**（50%–95%）：ASR 完成后按 `tts_completed / total_segments` 线性增长
/// - 音频提取占 0%–5%，视频合成占 95%–100%
///
/// # 示例
/// ```no_run
/// use std::sync::Arc;
/// use vt_core::pipeline::ProgressTracker;
///
/// let tracker = Arc::new(ProgressTracker::new());
/// // 传入 pipeline.process_video_with_progress(path, config, &tracker)
/// ```
#[derive(Debug)]
pub struct ProgressTracker {
    /// 总 chunk 数（音频分割后已知）
    total_chunks: AtomicUsize,
    /// ASR 已完成的 chunk 数
    asr_completed: AtomicUsize,
    /// ASR 阶段是否全部完成
    asr_done: AtomicBool,
    /// ASR 产生的总 segment 数（ASR 完成后设置）
    total_segments: AtomicUsize,
    /// 翻译已完成的 segment 数
    translation_completed: AtomicUsize,
    /// TTS 已完成的 segment 数
    tts_completed: AtomicUsize,
    /// 输出已收集的 segment 数
    output_count: AtomicUsize,
    /// 流水线开始时间
    start: Instant,
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressTracker {
    /// 创建空的进度追踪器
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_chunks: AtomicUsize::new(0),
            asr_completed: AtomicUsize::new(0),
            asr_done: AtomicBool::new(false),
            total_segments: AtomicUsize::new(0),
            translation_completed: AtomicUsize::new(0),
            tts_completed: AtomicUsize::new(0),
            output_count: AtomicUsize::new(0),
            start: Instant::now(),
        }
    }

    /// 设置总 chunk 数
    pub fn set_total_chunks(&self, n: usize) {
        self.total_chunks.store(n, Ordering::Relaxed);
    }

    /// ASR 完成一个 chunk
    pub fn inc_asr(&self) {
        self.asr_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// 标记 ASR 阶段完成，并设置总 segment 数
    pub fn finish_asr(&self, total_segments: usize) {
        self.total_segments.store(total_segments, Ordering::Relaxed);
        self.asr_done.store(true, Ordering::Relaxed);
    }

    /// 翻译完成一个 segment
    pub fn inc_translation(&self) {
        self.translation_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// TTS 完成一个 segment
    pub fn inc_tts(&self) {
        self.tts_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// 输出收集到一个 segment
    pub fn inc_output(&self) {
        self.output_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 获取 ASR 已完成的 chunk 数
    pub fn asr_completed(&self) -> usize {
        self.asr_completed.load(Ordering::Relaxed)
    }

    /// 获取总 chunk 数
    pub fn total_chunks(&self) -> usize {
        self.total_chunks.load(Ordering::Relaxed)
    }

    /// 获取总 segment 数
    pub fn total_segments(&self) -> usize {
        self.total_segments.load(Ordering::Relaxed)
    }

    /// 获取翻译已完成的 segment 数
    pub fn translation_completed(&self) -> usize {
        self.translation_completed.load(Ordering::Relaxed)
    }

    /// 获取 TTS 已完成的 segment 数
    pub fn tts_completed(&self) -> usize {
        self.tts_completed.load(Ordering::Relaxed)
    }

    /// ASR 阶段是否完成
    pub fn is_asr_done(&self) -> bool {
        self.asr_done.load(Ordering::Relaxed)
    }

    /// 计算整体进度百分比（0.0 – 1.0）
    ///
    /// - 音频提取前：0.0
    /// - ASR 阶段：0.05 – 0.50
    /// - TTS 阶段：0.50 – 0.95
    /// - 全部完成：0.95（留 5% 给视频合成）
    pub fn overall_progress(&self) -> f64 {
        let total_chunks = self.total_chunks.load(Ordering::Relaxed);
        if total_chunks == 0 {
            return 0.0;
        }

        let asr_done = self.asr_done.load(Ordering::Relaxed);
        let asr_completed = self.asr_completed.load(Ordering::Relaxed);

        if !asr_done {
            // ASR 阶段：5% → 50%
            let ratio = asr_completed as f64 / total_chunks as f64;
            return 0.05 + 0.45 * ratio;
        }

        // ASR 完成，追踪 TTS
        let total_segments = self.total_segments.load(Ordering::Relaxed);
        if total_segments == 0 {
            return 0.95; // 无 segment，几乎完成
        }

        let tts_completed = self.tts_completed.load(Ordering::Relaxed);
        let ratio = tts_completed as f64 / total_segments as f64;
        0.50 + 0.45 * ratio
    }

    /// 已用时间（秒）
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// 预估剩余时间（秒）
    ///
    /// 基于当前进度和已用时间线性外推。
    pub fn eta_secs(&self) -> Option<f64> {
        let progress = self.overall_progress();
        if progress <= 0.05 {
            return None;
        }
        let elapsed = self.elapsed_secs();
        // 预估总时间 = 已用时间 / 当前进度（0.95 为流水线部分上限）
        let pipeline_ratio = 0.95; // 流水线占总进度的 95%
        let estimated_total = elapsed / (progress / pipeline_ratio);
        let remaining = estimated_total - elapsed;
        if remaining > 0.0 {
            Some(remaining)
        } else {
            Some(0.0)
        }
    }
}

// ─── 进度事件 ─────────────────────────────────────────────

/// 流水线进度事件
///
/// 用于 UI 集成和进度报告。本 Session 仅通过 `tracing` 日志输出，
/// 未来可通过回调或通道发送进度事件。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressEvent {
    /// 音频提取完成
    AudioExtracted {
        /// 音频时长（秒）
        duration_secs: f64,
    },
    /// 音频分割完成
    AudioSplit {
        /// 片段数量
        chunk_count: usize,
    },
    /// ASR 完成一个片段
    AsrCompleted {
        /// 片段索引
        chunk_index: usize,
        /// 识别出的 Segment 数量
        segment_count: usize,
    },
    /// 翻译完成一个 Segment
    TranslationCompleted {
        /// Segment ID
        segment_id: String,
    },
    /// TTS 完成一个 Segment
    TtsCompleted {
        /// Segment ID
        segment_id: String,
    },
    /// 流水线完成
    PipelineCompleted {
        /// 总 Segment 数量
        total_segments: usize,
    },
}

// ─── Pipeline 结构体 ──────────────────────────────────────

/// 流水线引擎
///
/// 将 ASR、翻译、TTS 串联为异步并行流水线。
///
/// # 并行策略
/// 三个阶段分别运行在独立的 tokio 任务中，通过有界通道传递 Segment：
/// - **阶段 1（ASR）**：接收音频片段，调用 `AsrEngine` 生成带 `source_text` 的 Segment
/// - **阶段 2（翻译）**：接收有 `source_text` 的 Segment，调用 `TranslationProvider` 填充 `target_text`
/// - **阶段 3（TTS）**：接收有 `target_text` 的 Segment，调用 `TtsEngine` 合成音频并填充 `tts_audio_path`
///
/// # 背压控制
/// 通道使用有界容量（由 `PipelineConfig.channel_capacity` 配置），
/// 当下游处理速度慢时，上游自动阻塞，防止内存溢出。
pub struct Pipeline {
    /// ASR 引擎
    asr: Arc<dyn AsrEngine + Send + Sync>,
    /// 翻译提供者
    translator: Arc<dyn TranslationProvider + Send + Sync>,
    /// TTS 引擎
    tts: Arc<dyn TtsEngine + Send + Sync>,
    /// 音频提取器
    extractor: Arc<dyn AudioExtractor + Send + Sync>,
    /// 声音克隆集成辅助器（可选）
    ///
    /// 如果设置，TTS 阶段会优先尝试声音克隆合成，
    /// 失败时自动降级到标准 TTS 引擎。
    cloning: Option<Arc<CloningIntegration>>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("asr", &"<dyn AsrEngine>")
            .field("translator", &"<dyn TranslationProvider>")
            .field("tts", &"<dyn TtsEngine>")
            .field("extractor", &"<dyn AudioExtractor>")
            .field("cloning", &self.cloning.as_ref().map(|c| c.engine_name()))
            .finish()
    }
}

impl Pipeline {
    /// 创建流水线构建器
    ///
    /// # 示例
    /// ```no_run
    /// use vt_core::pipeline::Pipeline;
    ///
    /// let builder = Pipeline::builder();
    /// ```
    #[must_use]
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::default()
    }

    /// 处理视频：提取音频 → 分割 → ASR → 翻译 → TTS → 返回 Segment 列表
    ///
    /// 完整流程：
    /// 1. 使用 `AudioExtractor` 从视频中提取 16kHz mono WAV 音频
    /// 2. 按 VAD 或固定时长分割音频为多个片段
    /// 3. 启动三个并行任务（ASR、翻译、TTS），通过有界通道传递 Segment
    /// 4. 收集所有完成的 Segment，按时间排序后返回
    ///
    /// # 参数
    /// - `video_path`: 输入视频文件路径
    /// - `config`: 应用配置（包含 `PipelineConfig`、`TtsConfig` 等）
    ///
    /// # 返回
    /// 所有成功完成的 Segment 列表，按 `start` 时间升序排列。
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 视频文件不存在
    /// - [`AppError::FFmpeg`][]: 音频提取失败
    /// - [`AppError::AudioDecodeError`][]: WAV 读取或分割失败
    /// - [`AppError::PipelineError`][]: 任务异常终止
    ///
    /// # 部分失败
    /// 单个 Segment 的 ASR/翻译/TTS 失败不会中断流水线，
    /// 失败的 Segment 会被跳过并记录日志。
    pub async fn process_video(
        &self,
        video_path: &Path,
        config: &Config,
    ) -> AppResult<Vec<Segment>> {
        let tracker = Arc::new(ProgressTracker::new());
        self.process_video_with_progress(video_path, config, &tracker)
            .await
    }

    /// 处理视频（带进度追踪）：提取音频 → 分割 → ASR → 翻译 → TTS → 返回 Segment 列表
    ///
    /// 与 [`process_video`](Self::process_video) 相同，但接受外部 [`ProgressTracker`]，
    /// 调用方可实时读取各阶段进度和 ETA。
    ///
    /// # 参数
    /// - `video_path`: 输入视频文件路径
    /// - `config`: 应用配置
    /// - `tracker`: 进度追踪器（`Arc` 包裹），各阶段会实时更新其计数器
    ///
    /// # 示例
    /// ```no_run
    /// use std::sync::Arc;
    /// use vt_core::pipeline::{Pipeline, PipelineBuilder, ProgressTracker};
    /// use vt_core::config::Config;
    /// # use vt_core::error::AppResult;
    ///
    /// # async fn run() -> AppResult<()> {
    /// let pipeline = PipelineBuilder::default().build()?;
    /// let tracker = Arc::new(ProgressTracker::new());
    /// let config = Config::default();
    /// let segments = pipeline
    ///     .process_video_with_progress(std::path::Path::new("input.mp4"), &config, &tracker)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn process_video_with_progress(
        &self,
        video_path: &Path,
        config: &Config,
        tracker: &Arc<ProgressTracker>,
    ) -> AppResult<Vec<Segment>> {
        let pipeline_config = &config.pipeline;

        // ── 步骤 1：提取音频 ──────────────────────────────
        tracing::info!("Pipeline: extracting audio from {:?}", video_path);
        let temp_dir = tempfile::TempDir::new()?;
        let wav_path = temp_dir.path().join("full_audio.wav");
        self.extractor.extract_audio(video_path, &wav_path)?;

        // ── 步骤 2：读取 WAV 并分割 ────────────────────────
        let (samples, sample_rate) = read_wav_mono(&wav_path)?;
        let duration_secs = samples.len() as f64 / sample_rate as f64;
        tracing::info!(
            "Pipeline: audio loaded: {:.1}s, {} samples, {}Hz",
            duration_secs,
            samples.len(),
            sample_rate
        );

        if samples.is_empty() {
            tracing::warn!("Pipeline: audio is empty, no segments to process");
            return Ok(Vec::new());
        }

        let chunks = split_audio(&samples, sample_rate, pipeline_config)?;
        let chunk_infos = write_chunks(&chunks, temp_dir.path(), sample_rate)?;
        let total_chunks = chunk_infos.len();
        tracker.set_total_chunks(total_chunks);
        tracing::info!("Pipeline: audio split into {} chunks", total_chunks);

        if chunk_infos.is_empty() {
            tracing::warn!("Pipeline: no chunks to process after splitting");
            return Ok(Vec::new());
        }

        // ── 步骤 3：创建有界通道 ──────────────────────────
        let cap = pipeline_config.channel_capacity;
        let (tx_asr, rx_asr) = tokio::sync::mpsc::channel::<AudioChunkInfo>(cap);
        let (tx_translate, rx_translate) = tokio::sync::mpsc::channel::<Segment>(cap);
        let (tx_tts, rx_tts) = tokio::sync::mpsc::channel::<Segment>(cap);
        let (tx_output, mut rx_output) = tokio::sync::mpsc::channel::<Segment>(cap);

        // ── 步骤 4：启动 ASR 阶段 ─────────────────────────
        let asr = self.asr.clone();
        let asr_tracker = tracker.clone();
        let asr_handle = tokio::spawn(async move {
            let mut rx = rx_asr;
            let mut total_segments = 0usize;
            while let Some(chunk) = rx.recv().await {
                let asr = asr.clone();
                let chunk_path = chunk.path.clone();
                let chunk_start = chunk.start_time;
                let chunk_end = chunk.end_time;
                let chunk_index = chunk.index;

                let result = tokio::task::spawn_blocking(move || asr.transcribe(&chunk_path)).await;

                match result {
                    Ok(Ok(segments)) => {
                        tracing::info!(
                            "ASR: chunk {} ({:.1}s-{:.1}s) → {} segments",
                            chunk_index,
                            chunk_start,
                            chunk_end,
                            segments.len()
                        );
                        total_segments += segments.len();
                        for mut seg in segments {
                            // 调整时间戳为绝对时间（加上 chunk 偏移）
                            seg.start += chunk_start;
                            seg.end += chunk_start;

                            // P4: 文本归一化 — 清洗 ASR 输出 + 归一化
                            let norm_config = NormalizationConfig::default();
                            let cleaned = clean_asr_output(&seg.source_text);
                            seg.source_text = normalize_text(&cleaned, &norm_config);

                            // 状态转换：Pending → Transcribing
                            if let Err(e) = seg.start_transcribing() {
                                tracing::warn!(
                                    "ASR: failed to start transcribing segment {}: {}",
                                    seg.id,
                                    e
                                );
                                continue;
                            }

                            if tx_translate.send(seg).await.is_err() {
                                tracing::warn!("ASR: translation channel closed, stopping");
                                break;
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::error!("ASR: failed for chunk {}: {}", chunk_index, e);
                    }
                    Err(e) => {
                        tracing::error!("ASR: task panicked for chunk {}: {}", chunk_index, e);
                    }
                }
                // ASR 完成一个 chunk，更新进度
                asr_tracker.inc_asr();
            }
            tracing::info!("ASR stage: completed, total segments: {}", total_segments);
            total_segments
        });

        // ── 步骤 5：启动翻译阶段 ───────────────────────────
        // 使用上下文感知翻译：维护前序段落滑动窗口，提升代词消解和术语一致性
        let translator = self.translator.clone();
        let source_lang = config.asr.language.clone();
        let post_correction_enabled = config.translation.post_correction_enabled;
        let translate_tracker = tracker.clone();
        let context_window_size = 3; // 保留最近 3 条已翻译段落作为上下文
        let translation_mode = config.translation.translation_mode;

        // 翻译缓存（可选）
        let translation_cache_enabled = config.cache.translation_cache_enabled;
        let translation_cache: Option<TranslationCache> = if translation_cache_enabled {
            match TranslationCache::default() {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize translation cache: {}, cache disabled",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };
        // 翻译后端名称（用于缓存 key）
        let translation_backend_name = if config.translation.prefer_online {
            "deeplx"
        } else {
            "local"
        };
        let translation_model_name = config
            .translation
            .model_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "default".to_string());

        let translate_handle = tokio::spawn(async move {
            let mut rx = rx_translate;
            // 上下文窗口：跨 spawn_blocking 共享，需 Arc+Mutex
            let ctx = std::sync::Arc::new(std::sync::Mutex::new(
                crate::translate::TranslationContext::new(context_window_size),
            ));

            // SRT 模式：收集所有 segments，分批 SRT 翻译
            if translation_mode == TranslationMode::Srt {
                tracing::info!("Translation stage: SRT batch mode, collecting all segments...");
                let mut all_segments: Vec<Segment> = Vec::new();
                while let Some(seg) = rx.recv().await {
                    all_segments.push(seg);
                }

                if all_segments.is_empty() {
                    tracing::warn!("Translation stage: no segments received");
                    return;
                }

                tracing::info!(
                    "Translation stage: collected {} segments for SRT batch translation",
                    all_segments.len()
                );

                // 分批处理（每批最多 50 段，避免 LLM 上下文过长）
                let batch_size = 50;
                for chunk in all_segments.chunks_mut(batch_size) {
                    // 步骤1：检查缓存，分离命中和未命中的 segments
                    let mut to_translate: Vec<Segment> = Vec::new();
                    let mut cached_segs: Vec<Segment> = Vec::new();

                    for seg in chunk.iter_mut() {
                        let cache_key = if translation_cache.is_some() {
                            Some(TranslationCache::cache_key(
                                translation_backend_name,
                                &source_lang,
                                "zh",
                                &translation_model_name,
                                &seg.source_text,
                            ))
                        } else {
                            None
                        };

                        if let (Some(ref cache), Some(ref key)) = (&translation_cache, &cache_key) {
                            if let Some(cached_text) = cache.get(key) {
                                tracing::info!("Translation cache hit for segment {}", seg.id);
                                seg.target_text = Some(cached_text);
                                cached_segs.push(seg.clone());
                                continue;
                            }
                        }

                        to_translate.push(seg.clone());
                    }

                    // 步骤2：发送缓存命中的 segments 到 TTS
                    for mut seg in cached_segs {
                        let target_text = seg.target_text.take().unwrap_or_default();
                        if let Err(e) = seg.finish_transcribing(target_text) {
                            tracing::warn!(
                                "Failed to finish transcribing segment {}: {}",
                                seg.id,
                                e
                            );
                        }
                        translate_tracker.inc_translation();
                        if tx_tts.send(seg).await.is_err() {
                            tracing::warn!("Translation: TTS channel closed, stopping");
                            return;
                        }
                    }

                    // 步骤3：批量 SRT 翻译未缓存的 segments
                    if !to_translate.is_empty() {
                        let translate_count = to_translate.len();
                        let translator = translator.clone();
                        let src = source_lang.clone();
                        let cache_clone = translation_cache.clone();
                        let backend_name = translation_backend_name.to_string();
                        let model_name = translation_model_name.clone();

                        let result = tokio::task::spawn_blocking(move || {
                            let mut segments = to_translate;
                            translator.translate_srt(&mut segments, &src, "zh")?;

                            // 术语后校正
                            if post_correction_enabled {
                                crate::translate::apply_post_correction(&mut segments);
                            }

                            // 写入翻译缓存 + 状态转换
                            for seg in &mut segments {
                                // 写入缓存
                                if let Some(ref cache) = cache_clone {
                                    let key = TranslationCache::cache_key(
                                        &backend_name,
                                        &src,
                                        "zh",
                                        &model_name,
                                        &seg.source_text,
                                    );
                                    if let Some(ref text) = seg.target_text {
                                        if let Err(e) = cache.put(text, &key) {
                                            tracing::warn!(
                                                "Failed to write translation cache: {}",
                                                e
                                            );
                                        }
                                    }
                                }

                                // 状态转换
                                let target_text = seg.target_text.take().unwrap_or_default();
                                seg.finish_transcribing(target_text)?;
                            }

                            Ok::<Vec<Segment>, AppError>(segments)
                        })
                        .await;

                        match result {
                            Ok(Ok(translated)) => {
                                tracing::info!(
                                    "Translation: SRT batch completed for {} segments",
                                    translate_count
                                );
                                for seg in translated {
                                    translate_tracker.inc_translation();
                                    if tx_tts.send(seg).await.is_err() {
                                        tracing::warn!("Translation: TTS channel closed, stopping");
                                        return;
                                    }
                                }
                            }
                            Ok(Err(e)) => {
                                tracing::error!("Translation: SRT batch failed: {}", e);
                            }
                            Err(e) => {
                                tracing::error!("Translation: SRT batch panicked: {}", e);
                            }
                        }
                    }
                }

                tracing::info!("Translation stage: completed (SRT mode)");
                return;
            }

            // 逐段翻译模式（默认）
            while let Some(mut seg) = rx.recv().await {
                let translator = translator.clone();
                let seg_id = seg.id.clone();
                let src = source_lang.clone();
                let ctx = ctx.clone();

                // ── 翻译缓存检查 ──
                let cache_key = if translation_cache.is_some() {
                    Some(TranslationCache::cache_key(
                        translation_backend_name,
                        &source_lang,
                        "zh",
                        &translation_model_name,
                        &seg.source_text,
                    ))
                } else {
                    None
                };

                if let (Some(ref cache), Some(ref key)) = (&translation_cache, &cache_key) {
                    if let Some(cached_text) = cache.get(key) {
                        tracing::info!("Translation cache hit for segment {}", seg_id);
                        seg.target_text = Some(cached_text);
                        let target_text = seg.target_text.take().unwrap_or_default();
                        if let Err(e) = seg.finish_transcribing(target_text) {
                            tracing::warn!(
                                "Failed to finish transcribing segment {}: {}",
                                seg_id,
                                e
                            );
                        }
                        translate_tracker.inc_translation();
                        if tx_tts.send(seg).await.is_err() {
                            tracing::warn!("Translation: TTS channel closed, stopping");
                            break;
                        }
                        continue;
                    }
                }

                let cache_key_for_write = cache_key.clone();
                let translation_cache_for_write = translation_cache.clone();

                let result = tokio::task::spawn_blocking(move || {
                    // 使用上下文感知翻译（若后端支持则注入对话历史）
                    let source_text = seg.source_text.clone();
                    let mut seg = seg;

                    translator.translate_segment_with_context(
                        &mut seg,
                        &src,
                        "zh",
                        &ctx.lock().expect("ctx lock"),
                    )?;

                    // 术语后校正
                    if post_correction_enabled {
                        let mut segments = std::slice::from_mut(&mut seg);
                        crate::translate::apply_post_correction(&mut segments);
                    }

                    // 将本段 (source, target) 加入上下文窗口供后续段落参考
                    let target_text = seg.target_text.clone().unwrap_or_default();
                    if let Ok(mut ctx_guard) = ctx.lock() {
                        ctx_guard.push(source_text, target_text);
                    }

                    // 写入翻译缓存
                    if let (Some(cache), Some(key)) =
                        (translation_cache_for_write.as_ref(), &cache_key_for_write)
                    {
                        if let Some(ref text) = seg.target_text {
                            if let Err(e) = cache.put(text, key) {
                                tracing::warn!("Failed to write translation cache: {}", e);
                            }
                        }
                    }

                    // 状态转换：Transcribing → Translated
                    let target_text = seg.target_text.take().unwrap_or_default();
                    seg.finish_transcribing(target_text)?;
                    Ok::<Segment, AppError>(seg)
                })
                .await;

                match result {
                    Ok(Ok(seg)) => {
                        tracing::info!("Translation: completed for segment {}", seg_id);
                        translate_tracker.inc_translation();
                        if tx_tts.send(seg).await.is_err() {
                            tracing::warn!("Translation: TTS channel closed, stopping");
                            break;
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Translation: failed for segment {}: {}", seg_id, e);
                    }
                    Err(e) => {
                        tracing::error!("Translation: task panicked for segment {}: {}", seg_id, e);
                    }
                }
            }
            tracing::info!("Translation stage: completed");
        });

        // ── 步骤 6：启动 TTS 阶段（含声音克隆自动提取+降级）────
        let tts = self.tts.clone();
        let tts_config = config.tts.clone();
        let tts_tracker = tracker.clone();
        let cloning_integration = self.cloning.clone();
        let cloning_config = config.cloning.clone();
        // 完整音频 WAV 路径（用于自动提取参考音频）
        let full_wav_path = wav_path.to_path_buf();
        // 自动提取的参考音频状态（懒加载：首次克隆时提取）
        let auto_ref_state: Arc<std::sync::Mutex<Option<(PathBuf, String)>>> =
            Arc::new(std::sync::Mutex::new(None));

        // TTS 配音缓存 + 静音移除配置
        let tts_cache_enabled = config.cache.tts_cache_enabled;
        let tts_remove_silence = config.cache.tts_remove_silence;
        let tts_cache: Option<TtsCache> = if tts_cache_enabled {
            match TtsCache::default() {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!("Failed to initialize TTS cache: {}, cache disabled", e);
                    None
                }
            }
        } else {
            None
        };
        let tts_engine_name = config.tts.engine.clone();
        let tts_voice = config.tts.voice.clone();
        let tts_speed = config.tts.speed;
        let tts_volume = config.tts.volume;
        let tts_pitch = config.tts.pitch;
        let cloning_engine_name = config.cloning.engine.clone();

        // 逐段声画对齐 SpeedRate（可选）
        let speed_rate_processor: Option<SpeedRateProcessor> = if config.speed_rate.enabled {
            Some(SpeedRateProcessor::new(SpeedRateConfig {
                mode: config.speed_rate.mode,
                max_audio_speed: config.speed_rate.max_audio_speed,
                max_video_slow: config.speed_rate.max_video_slow,
                ..Default::default()
            }))
        } else {
            None
        };

        let tts_handle = tokio::spawn(async move {
            let mut rx = rx_tts;
            // 缓冲区：在首次提取参考音频前，收集多个 segment 以选择更长的参考片段
            let mut pending_segments: Vec<Segment> = Vec::new();
            'tts_loop: while let Some(seg) = rx.recv().await {
                // 如果参考音频尚未提取，收集多个 segment 用于选择更长的参考
                let ref_already_extracted = {
                    let state = auto_ref_state
                        .lock()
                        .expect("auto_ref_state mutex poisoned");
                    state.is_some()
                };

                let segments_to_process: Vec<Segment> = if ref_already_extracted {
                    vec![seg]
                } else {
                    // 首次提取：收集当前 segment + 尝试非阻塞接收更多 segment
                    pending_segments.push(seg);
                    while pending_segments.len() < 8 {
                        match rx.try_recv() {
                            Ok(more_seg) => pending_segments.push(more_seg),
                            Err(_) => break,
                        }
                    }
                    tracing::info!(
                        "Collected {} segments for reference audio extraction",
                        pending_segments.len()
                    );
                    std::mem::take(&mut pending_segments)
                };

                // 处理所有 segments（可能是 1 个或多个）
                // 保存所有 segments 的副本用于参考音频提取（仅首次提取时使用）
                let all_segments_for_ref = segments_to_process.clone();
                for mut seg in segments_to_process {
                    let tts = tts.clone();
                    let tts_config = tts_config.clone();
                    let seg_id = seg.id.clone();
                    let cloning_integration = cloning_integration.clone();
                    let cloning_config = cloning_config.clone();
                    let full_wav_path = full_wav_path.clone();
                    let auto_ref_state = auto_ref_state.clone();
                    let all_segments_for_ref = all_segments_for_ref.clone();
                    let speed_rate_processor = speed_rate_processor.clone();

                    // ── TTS 缓存检查 ──
                    let target_text = seg.target_text.clone().unwrap_or_default();
                    let tts_cache_key = if tts_cache.is_some() {
                        let engine = if cloning_integration.is_some() {
                            &cloning_engine_name
                        } else {
                            &tts_engine_name
                        };
                        let voice = if cloning_integration.is_some() {
                            "cloned"
                        } else {
                            &tts_voice
                        };
                        Some(TtsCache::cache_key(
                            &target_text,
                            voice,
                            tts_speed,
                            tts_volume,
                            tts_pitch,
                            engine,
                        ))
                    } else {
                        None
                    };

                    if let (Some(ref cache), Some(ref key)) = (&tts_cache, &tts_cache_key) {
                        if let Some(cached_path) = cache.get(key) {
                            tracing::info!("TTS cache hit for segment {}", seg_id);
                            // 状态转换：Translated → Synthesizing → Completed
                            if let Err(e) = seg.start_synthesizing() {
                                tracing::warn!(
                                    "Failed to start synthesizing for segment {}: {}",
                                    seg_id,
                                    e
                                );
                            }
                            if let Err(e) =
                                seg.finish_synthesizing(cached_path.to_string_lossy().to_string())
                            {
                                tracing::warn!(
                                    "Failed to finish synthesizing for segment {}: {}",
                                    seg_id,
                                    e
                                );
                            }
                            tts_tracker.inc_tts();
                            if tx_output.send(seg).await.is_err() {
                                tracing::warn!("TTS: output channel closed, stopping");
                                break 'tts_loop;
                            }
                            continue;
                        }
                    }

                    let tts_cache_key_for_write = tts_cache_key.clone();
                    let tts_cache_for_write = tts_cache.clone();
                    let tts_remove_silence = tts_remove_silence;

                    let result = tokio::task::spawn_blocking(move || {
                    let log_id = seg.id.clone();

                    // 尝试声音克隆（如果已配置）
                    if let Some(ref cloning) = cloning_integration {
                        // 获取或自动提取参考音频
                        let ref_audio = {
                            // 先检查是否有手动提供的参考音频
                            let ref_dir = PathBuf::from(&cloning_config.reference_audio_dir);
                            let manual_ref = ref_dir.join("reference.wav");

                            if manual_ref.exists() {
                                tracing::debug!(
                                    "Using manually provided reference audio: {:?}",
                                    manual_ref
                                );
                                Some(manual_ref)
                            } else {
                                // 自动提取：从视频音频中截取参考片段
                                let mut state = auto_ref_state
                                    .lock()
                                    .expect("auto_ref_state mutex poisoned");
                                if let Some((ref_path, _)) = state.as_ref() {
                                    // 已提取过，复用
                                    tracing::debug!(
                                        "Using auto-extracted reference audio: {:?}",
                                        ref_path
                                    );
                                    Some(ref_path.clone())
                                } else {
                                    // 首次提取：使用所有已收集的 segments 选择最佳参考
                                    let segments_for_ref = all_segments_for_ref.clone();

                                    let ref_output = ref_dir.join("auto_reference.wav");
                                    // 使用 VoiceExtractor 进行增强的参考音频提取
                                    let voice_extractor = VoiceExtractor::new(
                                        cloning_config.voice_extractor.clone(),
                                    );
                                    match voice_extractor.extract_reference_audio(
                                        &full_wav_path,
                                        &segments_for_ref,
                                        &ref_output,
                                    ) {
                                        Ok(Some(ref_audio)) => {
                                            let ref_path = ref_audio.path.clone();
                                            let prompt_text = ref_audio.prompt_text.clone();
                                            tracing::info!(
                                                "Auto-extracted reference audio for cloning: \
                                                {:?} ({:.1}s, prompt: \"{}\")",
                                                ref_path,
                                                ref_audio.duration_secs,
                                                if prompt_text.len() > 50 {
                                                    format!("{}...", prompt_text.chars().take(50).collect::<String>())
                                                } else {
                                                    prompt_text.clone()
                                                }
                                            );

                                            // 更新引擎的 prompt_text
                                            cloning.set_prompt_text(&prompt_text);

                                            // P1: 预热说话人 — 提前将参考音频发送给 TTS 服务端缓存
                                            // 后续 TTS 请求携带 speaker_id 即可跳过 prompt 创建步骤
                                            if let Err(e) = cloning.prewarm_speaker(
                                                "speaker_0",
                                                &ref_path,
                                                Some(&prompt_text),
                                            ) {
                                                tracing::warn!(
                                                    "Speaker prewarm failed (non-fatal, will use lazy init): {}",
                                                    e
                                                );
                                            }

                                            *state = Some((ref_path.clone(), prompt_text));
                                            Some(ref_path)
                                        }
                                        Ok(None) => {
                                            tracing::debug!(
                                                "Auto-extract: no suitable segment yet, \
                                                skipping voice cloning for segment {}",
                                                log_id
                                            );
                                            None
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "Auto-extract reference audio failed: {}, \
                                                skipping voice cloning for segment {}",
                                                e,
                                                log_id
                                            );
                                            None
                                        }
                                    }
                                }
                            }
                        };

                        if let Some(ref_audio) = ref_audio {
                            // P2: 句子级 TTS — 长文本按句拆分，逐句合成后交叉淡入淡出拼接
                            let target_text = seg.target_text.as_deref().unwrap_or("");
                            let cloned_path = if target_text.chars().count() > 80 {
                                // 长文本：句子级合成
                                match cloning.synthesize_with_sentence_split(
                                    target_text,
                                    &ref_audio,
                                    80,
                                    50,
                                ) {
                                    Ok(path) => Some(path),
                                    Err(e) => {
                                        tracing::warn!(
                                            "Sentence-level cloning failed for segment {}: {}, \
                                            falling back to standard cloning",
                                            log_id,
                                            e
                                        );
                                        // 回退到整体合成
                                        match cloning.try_synthesize(&seg, &ref_audio) {
                                            Ok(p) => p,
                                            Err(_) => None,
                                        }
                                    }
                                }
                            } else {
                                // 短文本：直接整体合成
                                match cloning.try_synthesize(&seg, &ref_audio) {
                                    Ok(p) => p,
                                    Err(_) => None,
                                }
                            };

                            match cloned_path {
                                Some(cloned_path) => {
                                    tracing::info!(
                                        "Voice cloning succeeded for segment {} → {:?}",
                                        log_id,
                                        cloned_path
                                    );
                                    let mut seg = seg;
                                    // 状态转换：Translated → Synthesizing → Completed
                                    if let Err(e) = seg.start_synthesizing() {
                                        tracing::warn!("Failed to start synthesizing for segment {}: {}", log_id, e);
                                    }
                                    if let Err(e) = seg.finish_synthesizing(cloned_path.to_string_lossy().to_string()) {
                                        tracing::warn!("Failed to finish synthesizing for segment {}: {}", log_id, e);
                                    }

                                    // ── TTS 后处理：静音移除 + 缓存写入 ──
                                    if let Some(ref audio_path) = seg.tts_audio_path {
                                        let audio_path = std::path::Path::new(audio_path);
                                        // 静音移除
                                        if tts_remove_silence {
                                            if let Err(e) = AudioPostProcessor::remove_silence(audio_path) {
                                                tracing::warn!("Silence removal failed for segment {}: {}", log_id, e);
                                            }
                                        }
                                        // 逐段声画对齐
                                        if let Some(ref processor) = speed_rate_processor {
                                            let source_dur = seg.end - seg.start;
                                            if let Err(e) = processor.process_audio(audio_path, source_dur) {
                                                tracing::warn!("SpeedRate failed for segment {}: {}", log_id, e);
                                            }
                                        }
                                        // 缓存写入
                                        if let (Some(cache), Some(key)) = (tts_cache_for_write.as_ref(), &tts_cache_key_for_write) {
                                            if let Err(e) = cache.put(audio_path, key) {
                                                tracing::warn!("Failed to write TTS cache for segment {}: {}", log_id, e);
                                            }
                                        }
                                    }

                                    return Ok(seg);
                                }
                                None => {
                                    tracing::warn!(
                                        "Voice cloning returned None for segment {}, \
                                        falling back to standard TTS",
                                        log_id
                                    );
                                }
                            }
                        }
                    }

                    // 标准 TTS 合成（降级路径）
                    let mut segments = vec![seg];
                    tts.synthesize_segments(&mut segments, &tts_config)?;
                    let seg = segments.remove(0);

                    // ── TTS 后处理：静音移除 + 缓存写入 ──
                    if let Some(ref audio_path) = seg.tts_audio_path {
                        let audio_path = std::path::Path::new(audio_path);
                        // 静音移除
                        if tts_remove_silence {
                            if let Err(e) = AudioPostProcessor::remove_silence(audio_path) {
                                tracing::warn!("Silence removal failed for segment {}: {}", log_id, e);
                            }
                        }
                        // 逐段声画对齐
                        if let Some(ref processor) = speed_rate_processor {
                            let source_dur = seg.end - seg.start;
                            if let Err(e) = processor.process_audio(audio_path, source_dur) {
                                tracing::warn!("SpeedRate failed for segment {}: {}", log_id, e);
                            }
                        }
                        // 缓存写入
                        if let (Some(cache), Some(key)) = (tts_cache_for_write.as_ref(), &tts_cache_key_for_write) {
                            if let Err(e) = cache.put(audio_path, key) {
                                tracing::warn!("Failed to write TTS cache for segment {}: {}", log_id, e);
                            }
                        }
                    }

                    Ok::<Segment, AppError>(seg)
                })
                .await;

                    match result {
                        Ok(Ok(seg)) => {
                            tracing::info!("TTS: completed for segment {}", seg_id);
                            tts_tracker.inc_tts();
                            if tx_output.send(seg).await.is_err() {
                                tracing::warn!("TTS: output channel closed, stopping");
                                break 'tts_loop;
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::error!("TTS: failed for segment {}: {}", seg_id, e);
                        }
                        Err(e) => {
                            tracing::error!("TTS: task panicked for segment {}: {}", seg_id, e);
                        }
                    }
                } // end for seg in segments_to_process
            }
            tracing::info!("TTS stage: completed");
        });

        // ── 步骤 7：启动输出收集任务 ─────────────────────
        // 独立任务收集输出，避免主任务在发送 chunk 时因 tx_output 满而死锁
        let output_handle = tokio::spawn(async move {
            let mut results = Vec::new();
            while let Some(seg) = rx_output.recv().await {
                results.push(seg);
            }
            results
        });

        // ── 步骤 8：输入 chunk 到 ASR 通道 ─────────────────
        for chunk in chunk_infos {
            if tx_asr.send(chunk).await.is_err() {
                tracing::error!("Pipeline: failed to send chunk to ASR stage");
                break;
            }
        }
        drop(tx_asr); // 通知 ASR 阶段：无更多 chunk

        // ── 步骤 9：等待各阶段完成 ─────────────────────────
        // 顺序等待确保通道正确关闭：
        // ASR 结束 → drop tx_translate → 翻译结束 → drop tx_tts → TTS 结束 → drop tx_output
        let asr_total_segments = asr_handle
            .await
            .map_err(|e| AppError::PipelineError(format!("ASR task join error: {e}")))?;
        // ASR 完成，设置总 segment 数供进度计算
        tracker.finish_asr(asr_total_segments);
        translate_handle
            .await
            .map_err(|e| AppError::PipelineError(format!("Translation task join error: {e}")))?;
        tts_handle
            .await
            .map_err(|e| AppError::PipelineError(format!("TTS task join error: {e}")))?;
        // tx_output 在 TTS 任务闭包中，任务结束后自动 drop，output_handle 将完成

        // ── 步骤 10：收集结果并排序 ───────────────────────
        let mut results = output_handle.await.map_err(|e| {
            AppError::PipelineError(format!("Output collection task join error: {e}"))
        })?;

        results.sort_by(|a, b| a.start.total_cmp(&b.start));

        tracing::info!(
            "Pipeline: completed with {} segments (from {} chunks)",
            results.len(),
            total_chunks
        );

        Ok(results)
    }
}

// ─── PipelineBuilder ──────────────────────────────────────

/// 流水线构建器
///
/// 使用构建器模式组装 `Pipeline`，各引擎通过 trait object 注入。
///
/// # 示例
/// ```no_run
/// use vt_core::pipeline::PipelineBuilder;
///
/// let pipeline = PipelineBuilder::default()
///     // .asr_engine(engine)
///     // .translation_provider(provider)
///     // .tts_engine(tts)
///     // .audio_extractor(extractor)
///     .build()?;
/// # Ok::<(), vt_core::error::AppError>(())
/// ```
#[derive(Default)]
pub struct PipelineBuilder {
    /// ASR 引擎
    asr: Option<Arc<dyn AsrEngine + Send + Sync>>,
    /// 翻译提供者
    translator: Option<Arc<dyn TranslationProvider + Send + Sync>>,
    /// TTS 引擎
    tts: Option<Arc<dyn TtsEngine + Send + Sync>>,
    /// 音频提取器
    extractor: Option<Arc<dyn AudioExtractor + Send + Sync>>,
    /// 声音克隆集成辅助器（可选）
    cloning: Option<Arc<CloningIntegration>>,
}

impl PipelineBuilder {
    /// 设置 ASR 引擎
    #[must_use]
    pub fn asr_engine(mut self, engine: impl AsrEngine + 'static) -> Self {
        self.asr = Some(Arc::new(engine));
        self
    }

    /// 设置翻译提供者
    #[must_use]
    pub fn translation_provider(mut self, provider: impl TranslationProvider + 'static) -> Self {
        self.translator = Some(Arc::new(provider));
        self
    }

    /// 设置 TTS 引擎
    #[must_use]
    pub fn tts_engine(mut self, engine: impl TtsEngine + 'static) -> Self {
        self.tts = Some(Arc::new(engine));
        self
    }

    /// 设置音频提取器
    #[must_use]
    pub fn audio_extractor(mut self, extractor: impl AudioExtractor + 'static) -> Self {
        self.extractor = Some(Arc::new(extractor));
        self
    }

    /// 设置声音克隆集成辅助器（可选）
    ///
    /// 启用后，TTS 阶段会优先尝试声音克隆合成，
    /// 失败时自动降级到标准 TTS 引擎。
    ///
    /// # 参数
    /// - `integration`: 声音克隆集成辅助器（`CloningIntegration` 实例）
    #[must_use]
    pub fn cloning_integration(mut self, integration: CloningIntegration) -> Self {
        self.cloning = Some(Arc::new(integration));
        self
    }

    /// 构建流水线
    ///
    /// # 错误
    /// - [`AppError::Config`][]: 未设置某个引擎
    pub fn build(self) -> AppResult<Pipeline> {
        Ok(Pipeline {
            asr: self
                .asr
                .ok_or_else(|| AppError::Config("ASR engine not set".to_string()))?,
            translator: self
                .translator
                .ok_or_else(|| AppError::Config("Translation provider not set".to_string()))?,
            tts: self
                .tts
                .ok_or_else(|| AppError::Config("TTS engine not set".to_string()))?,
            extractor: self
                .extractor
                .ok_or_else(|| AppError::Config("Audio extractor not set".to_string()))?,
            cloning: self.cloning,
        })
    }
}

// ─── 音频分割辅助函数 ─────────────────────────────────────

/// 分割音频采样数据为多个片段
///
/// 根据 `PipelineConfig` 选择 VAD 分割或固定时长分割。
fn split_audio(
    samples: &[f32],
    sample_rate: u32,
    config: &crate::config::PipelineConfig,
) -> AppResult<Vec<AudioChunkData>> {
    if config.enable_vad_split {
        let chunks = split_by_vad(samples, sample_rate);
        if chunks.is_empty() {
            tracing::info!("VAD detected no speech, falling back to fixed duration split");
            split_by_duration(samples, sample_rate, config.segment_duration_secs)
        } else {
            Ok(chunks)
        }
    } else {
        split_by_duration(samples, sample_rate, config.segment_duration_secs)
    }
}

/// 使用 VAD 分割音频
///
/// 调用 `detect_speech_segments` 检测语音段，提取每个语音段的采样数据。
fn split_by_vad(samples: &[f32], sample_rate: u32) -> Vec<AudioChunkData> {
    let vad_config = VadConfig::default();
    let speech_segments = detect_speech_segments(samples, sample_rate, &vad_config);

    speech_segments
        .iter()
        .filter_map(|seg| {
            let start_sample = ((seg.start_ms as f64 / 1000.0) * sample_rate as f64) as usize;
            let end_sample = ((seg.end_ms as f64 / 1000.0) * sample_rate as f64) as usize;
            let start_sample = start_sample.min(samples.len());
            let end_sample = end_sample.min(samples.len());

            if start_sample >= end_sample {
                return None;
            }

            Some(AudioChunkData {
                start_time: seg.start_ms as f64 / 1000.0,
                end_time: seg.end_ms as f64 / 1000.0,
                samples: samples[start_sample..end_sample].to_vec(),
            })
        })
        .collect()
}

/// 按固定时长分割音频
fn split_by_duration(
    samples: &[f32],
    sample_rate: u32,
    duration_secs: f64,
) -> AppResult<Vec<AudioChunkData>> {
    if duration_secs <= 0.0 {
        return Err(AppError::Config(format!(
            "segment_duration_secs must be > 0, got {duration_secs}"
        )));
    }

    let chunk_size = (duration_secs * sample_rate as f64) as usize;
    if chunk_size == 0 {
        return Err(AppError::Config(format!(
            "segment_duration_secs ({duration_secs}s) too small for sample rate {sample_rate}Hz"
        )));
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let end = (start + chunk_size).min(samples.len());
        let start_time = start as f64 / sample_rate as f64;
        let end_time = end as f64 / sample_rate as f64;
        chunks.push(AudioChunkData {
            start_time,
            end_time,
            samples: samples[start..end].to_vec(),
        });
        start = end;
    }

    Ok(chunks)
}

/// 将音频片段写入 WAV 文件
///
/// 每个 chunk 写入一个 16kHz mono 16-bit PCM WAV 文件。
fn write_chunks(
    chunks: &[AudioChunkData],
    output_dir: &Path,
    sample_rate: u32,
) -> AppResult<Vec<AudioChunkInfo>> {
    let mut infos = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.iter().enumerate() {
        if chunk.samples.is_empty() {
            tracing::warn!("Skipping empty chunk {}", index);
            continue;
        }

        let path = output_dir.join(format!("chunk_{index:04}.wav"));
        write_wav_chunk(&path, &chunk.samples, sample_rate)?;

        infos.push(AudioChunkInfo {
            index,
            path,
            start_time: chunk.start_time,
            end_time: chunk.end_time,
        });
    }
    Ok(infos)
}

/// 将采样数据写入 16kHz mono 16-bit PCM WAV 文件
fn write_wav_chunk(path: &Path, samples: &[f32], sample_rate: u32) -> AppResult<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| {
        AppError::AudioDecodeError(format!("Failed to create WAV writer {path:?}: {e}"))
    })?;

    for sample in samples {
        let i16_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer
            .write_sample(i16_sample)
            .map_err(|e| AppError::AudioDecodeError(format!("Failed to write WAV sample: {e}")))?;
    }

    writer
        .finalize()
        .map_err(|e| AppError::AudioDecodeError(format!("Failed to finalize WAV: {e}")))?;

    Ok(())
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 `PipelineBuilder::default()` 所有字段为 `None`。
    #[test]
    fn test_pipeline_builder_default() {
        let builder = PipelineBuilder::default();
        assert!(builder.asr.is_none());
        assert!(builder.translator.is_none());
        assert!(builder.tts.is_none());
        assert!(builder.extractor.is_none());
    }

    /// 验证 `PipelineBuilder::build()` 在缺少引擎时返回错误。
    #[test]
    fn test_pipeline_build_missing_engines() {
        let result = PipelineBuilder::default().build();
        assert!(result.is_err());
    }

    /// 验证固定时长分割逻辑。
    #[test]
    fn test_split_by_duration() {
        let samples = vec![0.5; 16000 * 10]; // 10 秒
        let chunks = split_by_duration(&samples, 16000, 3.0).expect("split_by_duration failed");

        assert_eq!(chunks.len(), 4); // 3+3+3+1 秒
        assert!((chunks[0].start_time - 0.0).abs() < 0.01);
        assert!((chunks[0].end_time - 3.0).abs() < 0.01);
        assert!((chunks[1].start_time - 3.0).abs() < 0.01);
        assert!((chunks[3].end_time - 10.0).abs() < 0.01);
    }

    /// 验证固定时长分割对空采样返回空列表。
    #[test]
    fn test_split_by_duration_empty() {
        let samples: Vec<f32> = Vec::new();
        let chunks = split_by_duration(&samples, 16000, 3.0).expect("split_by_duration failed");
        assert!(chunks.is_empty());
    }

    /// 验证 `split_by_duration` 对非法时长返回错误。
    #[test]
    fn test_split_by_duration_invalid() {
        let samples = vec![0.5; 100];
        assert!(split_by_duration(&samples, 16000, 0.0).is_err());
        assert!(split_by_duration(&samples, 16000, -1.0).is_err());
    }

    /// 验证 WAV chunk 写入。
    #[test]
    fn test_write_wav_chunk() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_chunk.wav");
        let samples = vec![0.5; 1600]; // 0.1 秒

        write_wav_chunk(&path, &samples, 16000).expect("write_wav_chunk failed");

        assert!(path.exists());

        // 验证 WAV 格式
        let reader = hound::WavReader::open(&path).expect("Failed to open WAV");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
    }
}
