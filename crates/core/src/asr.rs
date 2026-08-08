//! 语音识别 (ASR) 模块
//!
//! 基于 `whisper-rs`（whisper.cpp Rust 绑定）实现音频转带时间戳字幕的转换。
//!
//! # 功能概览
//! - [`AsrEngine`] trait：定义语音转录的标准接口
//! - [`WhisperEngine`]：基于 Whisper 的具体实现，支持 Metal GPU 加速
//! - [`VadConfig`] / [`detect_speech_segments`]：基于能量阈值的 VAD
//! - [`ModelManager`]：模型文件下载与缓存管理
//! - [`read_wav_mono`]：读取 16kHz mono WAV 文件为 f32 采样数据

use std::path::{Path, PathBuf};

use hound::WavReader;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::error::{AppError, AppResult};
use crate::models::segment::Segment;

// ─── VAD ─────────────────────────────────────────────────

/// VAD 配置参数
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// 每帧大小（毫秒）
    pub frame_size_ms: u32,
    /// RMS 能量阈值
    pub energy_threshold: f32,
    /// 最短语音段时长（毫秒）
    pub min_speech_duration_ms: u32,
    /// 最短静音时长（毫秒）
    pub min_silence_duration_ms: u32,
    /// 语音段前后填充（毫秒）
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_size_ms: 30,
            energy_threshold: 0.03,
            min_speech_duration_ms: 500,
            min_silence_duration_ms: 700,
            speech_pad_ms: 200,
        }
    }
}

/// VAD 检测到的语音片段（毫秒）
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSegment {
    /// 起始时间（毫秒）
    pub start_ms: i64,
    /// 结束时间（毫秒）
    pub end_ms: i64,
}

/// 检测音频中的语音片段（跳过静音段）
///
/// 基于 RMS 能量阈值的 VAD 算法，时间复杂度 O(n)。
pub fn detect_speech_segments(
    samples: &[f32],
    sample_rate: u32,
    config: &VadConfig,
) -> Vec<SpeechSegment> {
    if samples.is_empty() {
        return Vec::new();
    }

    let frame_size = ((sample_rate as u64 * config.frame_size_ms as u64) / 1000) as usize;
    if frame_size == 0 {
        return Vec::new();
    }

    let n_frames = samples.len() / frame_size;
    if n_frames == 0 {
        return Vec::new();
    }

    // 计算每帧 RMS 能量
    let energies: Vec<f32> = (0..n_frames)
        .map(|i| {
            let start = i * frame_size;
            let end = (start + frame_size).min(samples.len());
            let frame = &samples[start..end];
            let sum_sq: f32 = frame.iter().map(|s| s * s).sum();
            (sum_sq / frame.len() as f32).sqrt()
        })
        .collect();

    // 标记语音/静音帧
    let is_speech: Vec<bool> = energies
        .iter()
        .map(|&e| e > config.energy_threshold)
        .collect();

    // 合并连续语音帧为段
    let mut segments: Vec<SpeechSegment> = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut silence_count = 0usize;

    let silence_threshold_frames =
        (config.min_silence_duration_ms as u64 / config.frame_size_ms as u64) as usize;

    for (i, &speech) in is_speech.iter().enumerate() {
        if speech {
            if current_start.is_none() {
                current_start = Some(i);
            }
            silence_count = 0;
        } else if current_start.is_some() {
            silence_count += 1;
            if silence_count >= silence_threshold_frames {
                let start_frame = current_start.unwrap();
                let end_frame = i.saturating_sub(silence_count);
                let start_ms = (start_frame as u64 * config.frame_size_ms as u64) as i64;
                let end_ms = (end_frame as u64 * config.frame_size_ms as u64) as i64;
                let duration = end_ms - start_ms;
                if duration >= config.min_speech_duration_ms as i64 {
                    segments.push(make_padded_segment(start_ms, end_ms, config));
                }
                current_start = None;
                silence_count = 0;
            }
        }
    }

    // 处理最后一个未结束的语音段
    if let Some(start_frame) = current_start {
        let start_ms = (start_frame as u64 * config.frame_size_ms as u64) as i64;
        let end_ms = (n_frames as u64 * config.frame_size_ms as u64) as i64;
        let duration = end_ms - start_ms;
        if duration >= config.min_speech_duration_ms as i64 {
            segments.push(make_padded_segment(start_ms, end_ms, config));
        }
    }

    segments
}

fn make_padded_segment(start_ms: i64, end_ms: i64, config: &VadConfig) -> SpeechSegment {
    let pad = config.speech_pad_ms as i64;
    SpeechSegment {
        start_ms: (start_ms - pad).max(0),
        end_ms: end_ms + pad,
    }
}

// ─── WAV 读取 ────────────────────────────────────────────

/// 读取 16kHz mono PCM WAV 文件为 f32 采样数据
///
/// # 错误
/// - [`AppError::FileNotFound`][]: 文件不存在
/// - [`AppError::AudioDecodeError`][]: WAV 格式不符
pub fn read_wav_mono(path: &Path) -> AppResult<(Vec<f32>, u32)> {
    if !path.exists() {
        return Err(AppError::FileNotFound(path.to_path_buf()));
    }

    let mut reader = WavReader::open(path)
        .map_err(|e| AppError::AudioDecodeError(format!("Failed to open WAV: {e}")))?;

    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        return Err(AppError::AudioDecodeError(format!(
            "Expected 16kHz sample rate, got {}Hz",
            spec.sample_rate
        )));
    }

    if spec.channels != 1 {
        return Err(AppError::AudioDecodeError(format!(
            "Expected mono audio (1 channel), got {} channels",
            spec.channels
        )));
    }

    if spec.bits_per_sample != 16 {
        return Err(AppError::AudioDecodeError(format!(
            "Expected 16-bit PCM, got {}-bit",
            spec.bits_per_sample
        )));
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| {
            s.map(|v| v as f32 / 32768.0)
                .map_err(|e| AppError::AudioDecodeError(format!("Sample read error: {e}")))
        })
        .collect::<AppResult<Vec<_>>>()?;

    tracing::debug!(
        "Read WAV: {} samples, {}Hz, {} channels",
        samples.len(),
        spec.sample_rate,
        spec.channels
    );

    Ok((samples, spec.sample_rate))
}

// ─── 模型管理 ────────────────────────────────────────────

/// 魔搭 (ModelScope) 文件下载 API 基础 URL
const MODELSCOPE_API_BASE: &str = "https://modelscope.cn/api/v1/models";

/// 魔搭上 Whisper GGUF 模型仓库（large-v3-turbo Q5 量化版）
const WHISPER_GGML_REPO: &str = "Whisper/whisper-large-v3-turbo-gguf";

/// 模型名称到魔搭文件名的映射
///
/// 支持的模型名称:
/// - `whisper-tiny` → `ggml-tiny.bin` (77MB)
/// - `whisper-small` → `ggml-small.bin` (487MB)
/// - `whisper-large-v3-turbo` → `ggml-large-v3-turbo.bin` (1.6GB)
/// - `whisper-large-v3-turbo-q5` → `ggml-large-v3-turbo-q5_0.bin` (~900MB)
/// - `whisper-large-v3` → `ggml-large-v3-turbo.bin` (1.6GB, 同 turbo)
fn resolve_model_filename(model_name: &str) -> &str {
    match model_name {
        "whisper-tiny" | "whisper-tiny.en" => "ggml-tiny.bin",
        "whisper-base" | "whisper-base.en" => "ggml-tiny.bin", // 无 base，回退 tiny
        "whisper-small" | "whisper-small.en" => "ggml-small.bin",
        "whisper-medium" | "whisper-medium.en" => "ggml-small.bin", // 无 medium，回退 small
        "whisper-large-v3" | "whisper-large-v3-turbo" => "ggml-large-v3-turbo.bin",
        "whisper-large-v3-turbo-q5" | "whisper-large-v3-turbo-q5_0" => {
            "ggml-large-v3-turbo-q5_0.bin"
        }
        // 如果传入的已经是 ggml 文件名，直接使用
        name if name.starts_with("ggml-") => name,
        _ => "ggml-large-v3-turbo-q5_0.bin", // 默认使用 large-v3-turbo Q5
    }
}

/// Whisper 模型文件管理器
///
/// 负责模型文件的下载、缓存和路径查找。
/// 所有模型从魔搭 (ModelScope) 下载，国内可直接访问。
/// 默认缓存目录为 `~/.cache/video-translator/models/`。
#[derive(Debug)]
pub struct ModelManager {
    cache_dir: PathBuf,
}

impl ModelManager {
    /// 创建默认的模型管理器
    ///
    /// # 错误
    /// - [`AppError::Config`][]: `HOME` 环境变量未设置
    /// - [`AppError::Io`][]: 目录创建失败
    pub fn new() -> AppResult<Self> {
        let home = std::env::var("HOME")
            .map_err(|_| AppError::Config("HOME environment variable not set".to_string()))?;
        let cache_dir = PathBuf::from(home)
            .join(".cache")
            .join("video-translator")
            .join("models");
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    /// 使用指定的缓存目录创建模型管理器
    pub fn with_cache_dir(cache_dir: impl AsRef<Path>) -> AppResult<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    /// 获取缓存目录路径
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// 获取指定模型名称的缓存路径
    #[must_use]
    pub fn model_path(&self, model_name: &str) -> PathBuf {
        let filename = resolve_model_filename(model_name);
        self.cache_dir.join(filename)
    }

    /// 确保模型文件存在，不存在则自动从魔搭下载
    ///
    /// # 错误
    /// - [`AppError::ModelDownloadError`][]: 下载失败
    pub fn ensure_model(&self, model_name: &str) -> AppResult<PathBuf> {
        let model_path = self.model_path(model_name);
        if model_path.exists() {
            tracing::debug!("Model already cached at {:?}", model_path);
            return Ok(model_path);
        }
        self.download_model(model_name, &model_path)?;
        Ok(model_path)
    }

    fn download_model(&self, model_name: &str, dest: &Path) -> AppResult<()> {
        let filename = resolve_model_filename(model_name);
        let url = format!(
            "{MODELSCOPE_API_BASE}/{WHISPER_GGML_REPO}/repo?Revision=master&FilePath={filename}"
        );
        tracing::info!("Downloading model from ModelScope: {url}");
        tracing::info!("Saving to {dest:?}");

        let response = ureq::get(&url)
            .call()
            .map_err(|e| AppError::ModelDownloadError(format!("HTTP request failed: {e}")))?;

        let mut file = std::fs::File::create(dest).map_err(|e| {
            AppError::ModelDownloadError(format!("Failed to create file {dest:?}: {e}"))
        })?;

        let mut reader = response.into_reader();
        std::io::copy(&mut reader, &mut file).map_err(|e| {
            AppError::ModelDownloadError(format!("Failed to write model data: {e}"))
        })?;

        tracing::info!("Model downloaded successfully to {dest:?}");
        Ok(())
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            cache_dir: PathBuf::from("/tmp/video-translator/models"),
        })
    }
}

// ─── AsrEngine Trait ─────────────────────────────────────

/// 语音识别引擎接口
///
/// # 线程安全
/// 实现者必须满足 `Send + Sync`，以支持异步流水线中的并行处理。
pub trait AsrEngine: Send + Sync {
    /// 将音频文件转录为带时间戳的片段列表
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 音频文件不存在
    /// - [`AppError::AudioDecodeError`][]: WAV 解码失败
    /// - [`AppError::TranscriptionError`][]: Whisper 推理失败
    fn transcribe(&self, audio_path: &Path) -> AppResult<Vec<Segment>>;
}

// ─── WhisperConfig ───────────────────────────────────────

/// Whisper 引擎配置
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// 模型文件路径。为空则使用 `ModelManager` 自动下载默认模型。
    pub model_path: PathBuf,
    /// 源语言代码
    pub language: String,
    /// 是否启用 Metal GPU 加速
    pub use_metal: bool,
    /// 是否启用 VAD 预处理
    pub use_vad: bool,
    /// VAD 配置参数
    pub vad_config: VadConfig,
    /// 推理线程数
    pub n_threads: i32,
    /// 初始提示词（可选）
    ///
    /// 用于引导模型识别特定领域的术语。
    /// 例如 IT 类视频可设为 `"The following is a technical video about software engineering."`
    /// 帮助模型正确识别 API、GPU、Docker 等技术术语。
    pub initial_prompt: Option<String>,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            language: "en".to_string(),
            use_metal: true,
            use_vad: true,
            vad_config: VadConfig::default(),
            n_threads: 4,
            initial_prompt: None,
        }
    }
}

impl WhisperConfig {
    /// 设置模型文件路径
    #[must_use]
    pub fn with_model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = path.as_ref().to_path_buf();
        self
    }

    /// 设置源语言
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// 设置是否启用 Metal 加速
    #[must_use]
    pub fn with_metal(mut self, use_metal: bool) -> Self {
        self.use_metal = use_metal;
        self
    }

    /// 设置是否启用 VAD 预处理
    #[must_use]
    pub fn with_vad(mut self, use_vad: bool) -> Self {
        self.use_vad = use_vad;
        self
    }

    /// 设置推理线程数
    #[must_use]
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// 设置初始提示词
    ///
    /// 用于引导模型识别特定领域的术语。
    /// 例如 IT 类视频可设为 `"The following is a technical video about software engineering."`
    #[must_use]
    pub fn with_initial_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.initial_prompt = Some(prompt.into());
        self
    }
}

// ─── WhisperEngine ───────────────────────────────────────

/// 基于 whisper.cpp 的语音识别引擎
///
/// 通过 `whisper-rs` 绑定加载 GGML 模型，支持 Metal GPU 加速和 VAD 预处理。
#[derive(Debug)]
pub struct WhisperEngine {
    ctx: WhisperContext,
    language: String,
    use_vad: bool,
    vad_config: VadConfig,
    n_threads: i32,
    initial_prompt: Option<String>,
}

impl WhisperEngine {
    /// 从模型路径创建引擎（使用默认配置）
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 模型文件不存在
    /// - [`AppError::ModelLoadError`][]: 模型加载失败
    pub fn from_model_path(model_path: impl AsRef<Path>) -> AppResult<Self> {
        let config = WhisperConfig::default().with_model_path(model_path);
        Self::new(config)
    }

    /// 从配置创建引擎
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 模型文件不存在
    /// - [`AppError::ModelLoadError`][]: 模型加载失败
    pub fn new(config: WhisperConfig) -> AppResult<Self> {
        let model_path = if config.model_path.as_os_str().is_empty() {
            let manager = ModelManager::new()?;
            manager.ensure_model("whisper-large-v3-turbo-q5")?
        } else {
            config.model_path.clone()
        };

        if !model_path.exists() {
            return Err(AppError::FileNotFound(model_path));
        }

        tracing::info!("Loading Whisper model from {:?}", model_path);

        let ctx_params = WhisperContextParameters {
            use_gpu: config.use_metal,
            ..Default::default()
        };

        let ctx = WhisperContext::new_with_params(&model_path, ctx_params)
            .map_err(|e| AppError::ModelLoadError(format!("Failed to load model: {e}")))?;

        tracing::info!("Whisper model loaded successfully");

        Ok(Self {
            ctx,
            language: config.language,
            use_vad: config.use_vad,
            vad_config: config.vad_config,
            n_threads: config.n_threads,
            initial_prompt: config.initial_prompt,
        })
    }

    fn transcribe_segment(
        &self,
        sub_samples: &[f32],
        offset_ms: i64,
        segment_counter: &mut usize,
    ) -> AppResult<Vec<Segment>> {
        if sub_samples.is_empty() {
            return Ok(Vec::new());
        }

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| AppError::TranscriptionError(format!("Failed to create state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.n_threads);
        params.set_language(Some(&self.language));
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_nst(true);
        // 温度设为 0.0：翻译场景需要确定性输出，保证相同输入产生相同结果
        params.set_temperature(0.0);
        // 设置初始提示词：引导模型识别特定领域术语（如 IT、医学等）
        if let Some(ref prompt) = self.initial_prompt {
            params.set_initial_prompt(prompt);
        }

        state
            .full(params, sub_samples)
            .map_err(|e| AppError::TranscriptionError(format!("Whisper inference failed: {e}")))?;

        let n_segments = state.full_n_segments();
        let mut segments = Vec::with_capacity(n_segments as usize);

        for i in 0..n_segments {
            let seg = state.get_segment(i).ok_or_else(|| {
                AppError::TranscriptionError(format!("Failed to get segment {i}"))
            })?;

            let text = seg
                .to_str_lossy()
                .map_err(|e| {
                    AppError::TranscriptionError(format!("Failed to get segment text: {e}"))
                })?
                .trim()
                .to_string();

            if text.is_empty() {
                continue;
            }

            let start_sec = offset_ms as f64 / 1000.0 + seg.start_timestamp() as f64 / 100.0;
            let end_sec = offset_ms as f64 / 1000.0 + seg.end_timestamp() as f64 / 100.0;

            *segment_counter += 1;
            segments.push(Segment::new(
                format!("seg-{segment_counter:04}"),
                start_sec,
                end_sec,
                text,
            ));
        }

        Ok(segments)
    }
}

impl AsrEngine for WhisperEngine {
    #[tracing::instrument(skip(self), fields(audio = ?audio_path))]
    fn transcribe(&self, audio_path: &Path) -> AppResult<Vec<Segment>> {
        let (samples, sample_rate) = read_wav_mono(audio_path)?;

        if samples.is_empty() {
            tracing::warn!("Audio file contains no samples: {:?}", audio_path);
            return Ok(Vec::new());
        }

        let speech_segments: Vec<SpeechSegment> = if self.use_vad {
            let segs = detect_speech_segments(&samples, sample_rate, &self.vad_config);
            tracing::info!(
                "VAD detected {} speech segment(s) from {} samples",
                segs.len(),
                samples.len()
            );
            segs
        } else {
            let total_ms = (samples.len() as f64 / sample_rate as f64 * 1000.0) as i64;
            vec![SpeechSegment {
                start_ms: 0,
                end_ms: total_ms,
            }]
        };

        if speech_segments.is_empty() {
            tracing::info!("No speech detected in audio");
            return Ok(Vec::new());
        }

        let mut all_segments = Vec::new();
        let mut segment_counter = 0usize;

        for speech in &speech_segments {
            let start_sample = ((speech.start_ms as f64 / 1000.0) * sample_rate as f64) as usize;
            let end_sample = ((speech.end_ms as f64 / 1000.0) * sample_rate as f64) as usize;
            let start_sample = start_sample.min(samples.len());
            let end_sample = end_sample.min(samples.len());

            if start_sample >= end_sample {
                tracing::warn!(
                    "Skipping invalid speech segment: start={start_sample} >= end={end_sample}"
                );
                continue;
            }

            let sub_samples = &samples[start_sample..end_sample];

            tracing::debug!(
                "Transcribing segment: {}ms-{}ms ({} samples)",
                speech.start_ms,
                speech.end_ms,
                sub_samples.len()
            );

            let segs =
                self.transcribe_segment(sub_samples, speech.start_ms, &mut segment_counter)?;
            all_segments.extend(segs);
        }

        all_segments.sort_by(|a, b| a.start.total_cmp(&b.start));

        // 字幕后处理：合并短句、修复重叠、标点重分配
        let pre_count = all_segments.len();
        let post_proc =
            crate::subtitle_postprocess::SubtitlePostProcessor::with_language(&self.language);
        post_proc.process(&mut all_segments);
        if all_segments.len() != pre_count {
            tracing::info!(
                "Subtitle post-processing: {} → {} segments",
                pre_count,
                all_segments.len()
            );
        }
        tracing::info!("Transcription complete: {} segments", all_segments.len());
        Ok(all_segments)
    }
}
