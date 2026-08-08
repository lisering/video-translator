//! 配置管理模块
//!
//! 提供 [`Config`] 及其子配置 [`AsrConfig`]、[`TtsConfig`] 的定义，
//! 支持从 TOML 文件加载配置并与默认值合并。

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};
use crate::model_manager::ModelSource;
use crate::speed_rate::SpeedRateMode;
use crate::translation_extras::SubtitleType;

// ─── 默认值函数 ───────────────────────────────────────────

fn default_asr_model() -> String {
    "whisper-large-v3-turbo".to_string()
}

fn default_asr_use_metal() -> bool {
    true
}

fn default_asr_language() -> String {
    "en".to_string()
}

fn default_tts_speed() -> f32 {
    1.0
}

fn default_tts_voice() -> String {
    "Tingting".to_string()
}

fn default_tts_model_variant() -> String {
    "v1.1-onnx".to_string()
}

fn default_tts_fallback_to_say() -> bool {
    true
}

fn default_tts_engine() -> String {
    "say".to_string()
}

fn default_tts_device() -> String {
    "cpu".to_string()
}

fn default_tts_pitch() -> f32 {
    1.0
}

fn default_tts_voice_id() -> String {
    "tingting".to_string()
}

fn default_tts_sample_rate() -> u32 {
    24000
}

fn default_tts_volume() -> f32 {
    1.0
}

fn default_tts_auto_voice_selection() -> bool {
    false
}

/// 默认固定随机种子（确保音色一致性）
fn default_tts_seed() -> Option<u64> {
    Some(42)
}

/// 默认生成温度（降低随机性，0.3 = 更稳定）
fn default_tts_temperature() -> f64 {
    0.3
}

/// 默认音色稳定性（0.8 = 高稳定性）
fn default_tts_stability() -> f64 {
    0.8
}

/// 默认高频衰减量（dB，-3.0 = 衰减 3dB 以减少齿音）
fn default_tts_eq_high_shelf_db() -> f64 {
    -3.0
}

/// 默认交叉淡入淡出时长（毫秒）
fn default_tts_crossfade_duration_ms() -> u64 {
    50
}

/// 默认是否启用强制术语表
fn default_translation_force_glossary() -> bool {
    true
}

/// 默认是否启用翻译后术语校正
fn default_translation_post_correction_enabled() -> bool {
    true
}

/// 翻译模式
///
/// 控制翻译阶段的处理方式：
/// - `Segment`: 逐段翻译（默认），通过流水线通道并行处理
/// - `Srt`: SRT 批量翻译，将所有段落组成 SRT 格式一次性发送给 LLM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranslationMode {
    /// 逐段翻译（默认）
    #[default]
    Segment,
    /// SRT 批量翻译
    Srt,
}

/// 默认翻译模式
fn default_translation_mode() -> TranslationMode {
    TranslationMode::Segment
}

fn default_tts_cache_dir() -> String {
    "~/.cache/video-translator/tts_cache".to_string()
}

fn default_tts_parallel_tasks() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn default_output_dir() -> String {
    "./output".to_string()
}

fn default_max_concurrent_tasks() -> usize {
    4
}

fn default_translation_batch_size() -> usize {
    10
}

fn default_segment_duration_secs() -> f64 {
    30.0
}

fn default_channel_capacity() -> usize {
    100
}

fn default_enable_vad_split() -> bool {
    true
}

// ─── 新增模块默认值 ────────────────────────────────────────

fn default_diarization_enabled() -> bool {
    false
}

fn default_diarization_engine() -> String {
    "speakrs".to_string()
}

fn default_diarization_use_coreml() -> bool {
    true
}

fn default_cloning_enabled() -> bool {
    true
}

fn default_cloning_engine() -> String {
    "indextts".to_string()
}

fn default_cloning_reference_dir() -> String {
    "./references".to_string()
}

fn default_cloning_auto_extract() -> bool {
    true
}

/// 默认 GPT-SoVITS API 端点
fn default_gpt_sovits_api_url() -> String {
    "http://127.0.0.1:9880".to_string()
}

/// 默认 GPT-SoVITS 请求超时（秒）
fn default_gpt_sovits_timeout_secs() -> u64 {
    60
}

/// 默认参考音频提示文本语言
fn default_gpt_sovits_prompt_lang() -> String {
    "en".to_string()
}

/// 默认目标合成语言
fn default_gpt_sovits_text_lang() -> String {
    "zh".to_string()
}

/// 默认文本分割方法
fn default_gpt_sovits_text_split_method() -> String {
    "cut5".to_string()
}

/// 默认 Top-K 采样
fn default_gpt_sovits_top_k() -> u32 {
    5
}

/// 默认 Top-P 采样
fn default_gpt_sovits_top_p() -> f32 {
    1.0
}

/// 默认采样温度
fn default_gpt_sovits_temperature() -> f32 {
    1.0
}

/// 默认重复惩罚
fn default_gpt_sovits_repetition_penalty() -> f32 {
    1.35
}

// ─── 参考音频提取配置默认值 ──────────────────────────────

/// 默认是否启用 ffmpeg 人声增强
fn default_voice_extractor_enable_enhancement() -> bool {
    true
}

/// 默认是否启用静音修剪
fn default_voice_extractor_enable_silence_trim() -> bool {
    true
}

/// 默认是否启用音量归一化
fn default_voice_extractor_enable_normalization() -> bool {
    true
}

/// 默认静音检测阈值（dB）
fn default_voice_extractor_silence_threshold_db() -> f64 {
    -40.0
}

/// 默认目标 RMS 电平（dBFS）
fn default_voice_extractor_target_rms_db() -> f64 {
    -20.0
}

/// 默认最小参考音频时长（秒）
fn default_voice_extractor_min_duration() -> f64 {
    3.0
}

/// 默认最大参考音频时长（秒）
fn default_voice_extractor_max_duration() -> f64 {
    10.0
}

/// 默认理想参考音频时长（秒）
fn default_voice_extractor_ideal_duration() -> f64 {
    5.0
}

// ─── 子进程克隆引擎默认值 ────────────────────────────────

/// 默认子进程克隆超时（秒）
fn default_clone_timeout_secs() -> u64 {
    120
}

fn default_batch_max_concurrent() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(3)
}

fn default_batch_memory_threshold() -> f64 {
    80.0
}

fn default_batch_enable_priority() -> bool {
    true
}

fn default_checkpoint_enabled() -> bool {
    true
}

fn default_checkpoint_dir() -> String {
    "~/.cache/video-translator/checkpoints".to_string()
}

fn default_checkpoint_retention_days() -> u32 {
    7
}

fn default_performance_enable_profiling() -> bool {
    false
}

fn default_performance_flamegraph_output() -> String {
    "./flamegraph.svg".to_string()
}

// ─── 新增功能默认值 ────────────────────────────────────────

/// 默认是否启用 TTS 配音缓存
fn default_tts_cache_enabled() -> bool {
    true
}

/// 默认是否启用翻译缓存
fn default_translation_cache_enabled() -> bool {
    true
}

/// 默认是否启用 TTS 静音移除
fn default_tts_remove_silence() -> bool {
    true
}

/// 默认字幕类型
fn default_subtitle_type() -> SubtitleType {
    SubtitleType::None
}

/// 默认是否启用声画对齐 SpeedRate
fn default_speed_rate_enabled() -> bool {
    false
}

/// 默认 SpeedRate 模式
fn default_speed_rate_mode() -> SpeedRateMode {
    SpeedRateMode::Hybrid
}

/// 默认最大音频加速倍率
fn default_speed_rate_max_audio() -> f64 {
    1.3
}

/// 默认最大视频慢放倍率
fn default_speed_rate_max_video() -> f64 {
    2.0
}

/// 默认背景音量（0.2 = 背景音降低到 20%）
fn default_bgm_volume() -> f32 {
    0.2
}

// ─── 音频同步配置 ─────────────────────────────────────────

/// 音频同步模式
///
/// 控制 TTS 音频与原视频时间轴的对齐策略。
///
/// # 模式说明
/// - **Trim（截断）**：TTS 音频超出时间槽时截断尾部，不足时补静音。简单可靠，可能丢失译文尾部内容。
/// - **SpeedUp（加速）**：使用 ffmpeg `atempo` 滤镜加速 TTS 音频以适应时间槽。保留内容完整，但高倍率加速时听感不自然。
/// - **VideoSlow（视频慢放）**：不修改 TTS 音频，通过 `setpts` 滤镜慢放视频以适应较长音频。音质完美，但视频画面会变慢。
/// - **Hybrid（兼顾）**：先加速 TTS 音频至上限（`max_speed_ratio`），剩余部分通过视频慢放补偿。平衡音质与画面流畅度。
///
/// # 选择建议
/// - 教学视频、讲座：`Hybrid`（推荐）
/// - 快节奏短视频、新闻：`SpeedUp`
/// - 配音质量优先、画面节奏不重要：`VideoSlow`
/// - 简单快速、可接受内容截断：`Trim`
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioSyncMode {
    /// 截断模式：音频长了就剪，短了就补静音
    Trim,
    /// 加速模式：使用 ffmpeg atempo 加速音频以适应时间槽
    #[default]
    SpeedUp,
    /// 视频慢放模式：不修改音频，慢放视频以适应较长音频
    VideoSlow,
    /// 兼顾模式：先加速音频至上限，剩余通过视频慢放补偿
    Hybrid,
}

/// 默认最大加速比（Hybrid 模式下 TTS 加速上限）
fn default_sync_max_speed_ratio() -> f32 {
    1.3
}

/// 音频同步配置
///
/// 控制 TTS 音频与原视频时间轴的对齐策略，可通过 `[audio_sync]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSyncConfig {
    /// 同步模式
    #[serde(default)]
    pub mode: AudioSyncMode,

    /// 加速上限倍率（仅 `SpeedUp` 和 `Hybrid` 模式生效）
    ///
    /// - `SpeedUp` 模式：TTS 音频最多加速到此倍率，超出部分截断
    /// - `Hybrid` 模式：TTS 音频最多加速到此倍率，剩余部分通过视频慢放补偿
    ///
    /// 范围：1.0–2.0，推荐 1.3（听感自然且补偿量小）
    #[serde(default = "default_sync_max_speed_ratio")]
    pub max_speed_ratio: f32,
}

impl Default for AudioSyncConfig {
    fn default() -> Self {
        Self {
            mode: AudioSyncMode::default(),
            max_speed_ratio: default_sync_max_speed_ratio(),
        }
    }
}

// ─── 流水线配置 ─────────────────────────────────────────

/// 流水线配置
///
/// 控制 ASR → 翻译 → TTS 三阶段流水线的音频分割策略、通道缓冲和并行度。
/// 可通过 TOML 文件中的 `[pipeline]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// 音频分割时长（秒），用于固定时长分割模式
    #[serde(default = "default_segment_duration_secs")]
    pub segment_duration_secs: f64,

    /// 通道缓冲区大小（背压控制，防止生产者过快导致内存溢出）
    #[serde(default = "default_channel_capacity")]
    pub channel_capacity: usize,

    /// 是否使用 VAD 分割而非固定时长
    #[serde(default = "default_enable_vad_split")]
    pub enable_vad_split: bool,
}

impl Default for PipelineConfig {
    /// 返回默认流水线配置
    ///
    /// - 分割时长: 30 秒
    /// - 通道容量: 100
    /// - VAD 分割: 启用
    fn default() -> Self {
        Self {
            segment_duration_secs: default_segment_duration_secs(),
            channel_capacity: default_channel_capacity(),
            enable_vad_split: default_enable_vad_split(),
        }
    }
}

// ─── ASR 配置 ─────────────────────────────────────────────

/// ASR（自动语音识别）配置
///
/// 控制转录引擎的模型选择、硬件加速和源语言设置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrConfig {
    /// Whisper 模型名称（如 `whisper-large-v3`、`whisper-medium`）
    #[serde(default = "default_asr_model")]
    pub model: String,
    /// 是否启用 Metal 硬件加速（macOS M 系列芯片）
    #[serde(default = "default_asr_use_metal")]
    pub use_metal: bool,
    /// 源视频语言代码（如 `en` 表示英文）
    #[serde(default = "default_asr_language")]
    pub language: String,
}

impl Default for AsrConfig {
    /// 返回默认 ASR 配置
    ///
    /// - 模型: `whisper-large-v3-turbo`
    /// - Metal 加速: 启用
    /// - 语言: 英文 (`en`)
    fn default() -> Self {
        Self {
            model: default_asr_model(),
            use_metal: default_asr_use_metal(),
            language: default_asr_language(),
        }
    }
}

// ─── TTS 配置 ─────────────────────────────────────────────

/// TTS（文本转语音）配置
///
/// 控制语音合成的引擎选择、语速、音色、音调、音量、采样率和并行任务数。
/// 支持通过 TOML 文件中的 `[tts]` 段配置。
///
/// # 字段说明
/// - `engine`: TTS 引擎类型（`"say"` | `"kokoro"`）
/// - `voice_id`: 音色 ID（通过 `VoiceManager::list_voices()` 查看可用音色）
/// - `speed`: 语速倍率（0.5–2.0，默认 1.0）
/// - `pitch`: 音调倍率（0.8–1.2，默认 1.0）
/// - `volume`: 音量倍率（0.0–2.0，默认 1.0）
/// - `sample_rate`: 采样率（16000 / 24000 / 48000，默认 24000）
/// - `fallback_to_say`: 新引擎加载失败时是否回退到 macOS `say`
/// - `auto_voice_selection`: 是否根据视频说话人性别自动选择音色
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    /// TTS 引擎名称（`"say"` | `"kokoro"`）
    #[serde(default = "default_tts_engine")]
    pub engine: String,
    /// 语音合成语速倍率（1.0 为正常语速，范围 0.5–2.0）
    #[serde(default = "default_tts_speed")]
    pub speed: f32,
    /// 音调倍率（1.0 为不变，范围 0.8–1.2，< 1.0 降低音调，> 1.0 升高音调）
    #[serde(default = "default_tts_pitch")]
    pub pitch: f32,
    /// 音量倍率（1.0 为不变，范围 0.0–2.0）
    #[serde(default = "default_tts_volume")]
    pub volume: f32,
    /// 音色 ID（通过 `VoiceManager::list_voices()` 查看可用音色）
    ///
    /// 内置音色包括：`tingting`（女）、`meijia`（女）、`sinji`（女）、
    /// `zhiming`（男）、`weiqiang`（男）、`haoze`（男）
    #[serde(default = "default_tts_voice_id")]
    pub voice_id: String,
    /// 语音音色名称（macOS `say` 命令语音，如 `Tingting`、`Meijia`）
    ///
    /// 向后兼容字段：当 `voice_id` 未设置时使用此字段。
    #[serde(default = "default_tts_voice")]
    pub voice: String,
    /// 音频采样率（16000 / 24000 / 48000，默认 24000）
    #[serde(default = "default_tts_sample_rate")]
    pub sample_rate: u32,
    /// 推理设备（`"cpu"` 或 `"metal"`，当前仅影响 Kokoro 引擎）
    #[serde(default = "default_tts_device")]
    pub device: String,
    /// 缓存目录路径（存储已合成的音频文件）
    #[serde(default = "default_tts_cache_dir")]
    pub cache_dir: String,
    /// 并行合成任务数（默认等于 CPU 核心数）
    #[serde(default = "default_tts_parallel_tasks")]
    pub parallel_tasks: usize,
    /// Kokoro 模型变体名称（如 `v1.1-onnx`）
    #[serde(default = "default_tts_model_variant")]
    pub model_variant: String,
    /// Kokoro ONNX 模型文件路径（为 `None` 时从缓存加载）
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    /// 当 Kokoro 引擎加载失败时是否回退到 macOS `say` 命令
    #[serde(default = "default_tts_fallback_to_say")]
    pub fallback_to_say: bool,
    /// 是否根据视频说话人性别自动选择音色
    #[serde(default = "default_tts_auto_voice_selection")]
    pub auto_voice_selection: bool,

    /// 固定随机种子（确保多次合成音色一致，默认 42）
    ///
    /// Kokoro 引擎使用此参数控制合成随机性。
    /// `say` 引擎不受此参数影响（本身为确定性合成）。
    #[serde(default = "default_tts_seed")]
    pub seed: Option<u64>,

    /// 生成温度（0.0-1.0，默认 0.3）
    ///
    /// 较低的温度减少生成随机性，使音色更稳定。
    /// Kokoro 引擎使用此参数；`say` 引擎不受影响。
    #[serde(default = "default_tts_temperature")]
    pub temperature: f64,

    /// 音色稳定性（0.0-1.0，默认 0.8）
    ///
    /// 较高的值使音色在多句合成间保持一致。
    /// Kokoro 引擎使用此参数；`say` 引擎不受影响。
    #[serde(default = "default_tts_stability")]
    pub stability: f64,

    /// 高频均衡器衰减量（dB，默认 -3.0）
    ///
    /// 衰减 6kHz 以上的高频以减少齿音（sibilance）。
    /// 设为 0.0 可禁用高频衰减。
    #[serde(default = "default_tts_eq_high_shelf_db")]
    pub eq_high_shelf_db: f64,

    /// 交叉淡入淡出时长（毫秒，默认 50）
    ///
    /// 相邻音频片段衔接处的淡入淡出时长，消除拼接感。
    #[serde(default = "default_tts_crossfade_duration_ms")]
    pub crossfade_duration_ms: u64,
}

impl Default for TtsConfig {
    /// 返回默认 TTS 配置
    ///
    /// - 引擎: `say`（macOS 离线神经语音）
    /// - 语速: 1.0（正常语速）
    /// - 音调: 1.0（不变）
    /// - 音量: 1.0（不变）
    /// - 音色 ID: `tingting`（标准普通话女声）
    /// - 采样率: 24000 Hz
    /// - 缓存目录: `~/.cache/video-translator/tts_cache`
    /// - 并行任务数: CPU 核心数
    /// - 回退到 say: 启用
    fn default() -> Self {
        Self {
            engine: default_tts_engine(),
            speed: default_tts_speed(),
            pitch: default_tts_pitch(),
            volume: default_tts_volume(),
            voice_id: default_tts_voice_id(),
            voice: default_tts_voice(),
            sample_rate: default_tts_sample_rate(),
            device: default_tts_device(),
            cache_dir: default_tts_cache_dir(),
            parallel_tasks: default_tts_parallel_tasks(),
            model_variant: default_tts_model_variant(),
            model_path: None,
            fallback_to_say: default_tts_fallback_to_say(),
            auto_voice_selection: default_tts_auto_voice_selection(),
            seed: default_tts_seed(),
            temperature: default_tts_temperature(),
            stability: default_tts_stability(),
            eq_high_shelf_db: default_tts_eq_high_shelf_db(),
            crossfade_duration_ms: default_tts_crossfade_duration_ms(),
        }
    }
}

// ─── 翻译配置 ─────────────────────────────────────────────

/// 默认推理设备
fn default_translation_device() -> String {
    "metal".to_string()
}

/// 默认最大 token 数
fn default_translation_max_tokens() -> usize {
    256
}

/// 默认采样温度
fn default_translation_temperature() -> f32 {
    0.3
}

/// 默认 DeepLX 服务端点
fn default_dlx_endpoint() -> String {
    "http://localhost:1188".to_string()
}

/// 默认 DeepLX 请求超时（秒）
fn default_dlx_timeout_secs() -> u64 {
    10
}

/// 默认 DeepLX 最大重试次数
fn default_dlx_max_retries() -> usize {
    3
}

/// 默认是否优先使用在线翻译
fn default_prefer_online() -> bool {
    true
}

/// 默认是否在在线翻译失败时降级到本地
fn default_fallback_on_error() -> bool {
    true
}

/// 默认健康检查间隔（秒）
fn default_health_check_interval_secs() -> u64 {
    300
}

/// 翻译模块配置
///
/// 控制 DeepLX 在线翻译 + 本地降级引擎的完整配置。
///
/// # 两级降级架构
/// 1. **DeepLX（优先）**：通过 HTTP 调用自部署的 DeepLX 服务
/// 2. **本地降级**：通过 `llama-server` 子进程加载 Hy-MT2 GGUF 模型
///
/// 可通过 TOML 文件中的 `[translation]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// 术语表文件路径（支持 JSON 或 CSV 格式）
    #[serde(default)]
    pub glossary_path: Option<String>,

    /// 批量翻译大小（每批处理的最大文本数）
    #[serde(default = "default_translation_batch_size")]
    pub batch_size: usize,

    /// 本地模型文件路径
    ///
    /// 若为 `None`，则使用 `model_source` 从缓存目录加载。
    /// 优先级：`model_path` > `model_source` 缓存路径。
    #[serde(default)]
    pub model_path: Option<PathBuf>,

    /// 推理设备
    ///
    /// - `"metal"`: 使用 Apple Metal GPU 加速（macOS M 系列芯片）
    /// - `"cpu"`: 使用 CPU 推理
    #[serde(default = "default_translation_device")]
    pub device: String,

    /// 生成的最大 token 数
    #[serde(default = "default_translation_max_tokens")]
    pub max_tokens: usize,

    /// 采样温度
    #[serde(default = "default_translation_temperature")]
    pub temperature: f32,

    /// 模型来源
    #[serde(default)]
    pub model_source: ModelSource,

    // ── DeepLX 在线翻译配置 ──
    /// DeepLX 服务端点 URL
    #[serde(default = "default_dlx_endpoint")]
    pub dlx_endpoint: String,

    /// DeepLX 请求超时时间（秒）
    #[serde(default = "default_dlx_timeout_secs")]
    pub dlx_timeout_secs: u64,

    /// DeepLX 请求最大重试次数（仅对 429 和 5xx 重试）
    #[serde(default = "default_dlx_max_retries")]
    pub dlx_max_retries: usize,

    // ── 路由配置 ──
    /// 是否优先使用在线翻译（DeepLX）
    #[serde(default = "default_prefer_online")]
    pub prefer_online: bool,

    /// 在线翻译失败时是否自动降级到本地模型
    #[serde(default = "default_fallback_on_error")]
    pub fallback_on_error: bool,

    /// DeepLX 健康检查间隔（秒）
    #[serde(default = "default_health_check_interval_secs")]
    pub health_check_interval_secs: u64,

    /// 是否启用强制术语表（翻译前对编程术语进行占位符替换）
    ///
    /// 启用后，`println`、`format!` 等编程术语会在翻译前被替换为占位符，
    /// 翻译完成后再还原为正确的中文术语，避免翻译模型拆分或误译。
    #[serde(default = "default_translation_force_glossary")]
    pub force_glossary: bool,

    /// 是否启用翻译后术语校正
    ///
    /// 启用后，在翻译完成后对 `target_text` 进行正则替换，
    /// 将常见错误术语（如"打印行"→"打印并换行"）修正为正确术语。
    #[serde(default = "default_translation_post_correction_enabled")]
    pub post_correction_enabled: bool,

    /// 翻译模式：逐段翻译或 SRT 批量翻译
    ///
    /// - `"segment"`: 逐段翻译（默认），通过流水线通道并行处理，支持上下文感知
    /// - `"srt"`: SRT 批量翻译，将所有段落组成 SRT 格式一次性发送给 LLM，
    ///   LLM 看到完整上下文，翻译后按 SRT 结构解析回 segments
    ///
    /// SRT 模式适合短视频（<30 段），长视频建议使用 segment 模式以利用流水线并行。
    #[serde(default = "default_translation_mode")]
    pub translation_mode: TranslationMode,
}

impl Default for TranslationConfig {
    /// 返回默认翻译配置
    ///
    /// - 批量大小: 10
    /// - 设备: CPU
    /// - 最大 token 数: 256
    /// - 采样温度: 0.3
    /// - 模型来源: ModelScope（Qwen2.5-3B-Instruct GGUF）
    /// - 其余字段为 `None`
    fn default() -> Self {
        Self {
            glossary_path: None,
            batch_size: default_translation_batch_size(),
            model_path: None,
            device: default_translation_device(),
            max_tokens: default_translation_max_tokens(),
            temperature: default_translation_temperature(),
            model_source: ModelSource::default(),
            dlx_endpoint: default_dlx_endpoint(),
            dlx_timeout_secs: default_dlx_timeout_secs(),
            dlx_max_retries: default_dlx_max_retries(),
            prefer_online: default_prefer_online(),
            fallback_on_error: default_fallback_on_error(),
            health_check_interval_secs: default_health_check_interval_secs(),
            force_glossary: default_translation_force_glossary(),
            post_correction_enabled: default_translation_post_correction_enabled(),
            translation_mode: default_translation_mode(),
        }
    }
}

// ─── 说话人分离配置 ───────────────────────────────────────

/// 说话人分离配置
///
/// 控制说话人分离引擎的启用状态、后端选择和硬件加速。
/// 可通过 TOML 文件中的 `[diarization]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationConfig {
    /// 是否启用说话人分离
    #[serde(default = "default_diarization_enabled")]
    pub enabled: bool,

    /// 引擎名称（`speakrs`、`polyvoice`、`pyannote-rs`）
    #[serde(default = "default_diarization_engine")]
    pub engine: String,

    /// 是否使用 CoreML 加速（Apple Silicon）
    #[serde(default = "default_diarization_use_coreml")]
    pub use_coreml: bool,
}

impl Default for DiarizationConfig {
    /// 返回默认说话人分离配置
    ///
    /// - 启用: 否（默认关闭，不影响核心流程）
    /// - 引擎: `speakrs`
    /// - CoreML: 启用
    fn default() -> Self {
        Self {
            enabled: default_diarization_enabled(),
            engine: default_diarization_engine(),
            use_coreml: default_diarization_use_coreml(),
        }
    }
}

// ─── 声音克隆配置 ─────────────────────────────────────────

/// 声音克隆配置
///
/// 控制声音克隆引擎的启用状态、后端选择、参考音频目录和自动提取行为。
/// 支持 GPT-SoVITS API v2 作为后端引擎。
/// 可通过 TOML 文件中的 `[cloning]` 段配置。
///
/// # GPT-SoVITS 配置示例
/// ```toml
/// [cloning]
/// enabled = true
/// engine = "gpt-sovits"
/// api_url = "http://127.0.0.1:9880"
/// prompt_text = "Hello, this is a reference audio."
/// prompt_lang = "en"
/// text_lang = "zh"
/// gpt_model = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
/// sovits_model = "GPT_SoVITS/pretrained_models/s2G488k.pth"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningConfig {
    /// 是否启用声音克隆
    #[serde(default = "default_cloning_enabled")]
    pub enabled: bool,

    /// 引擎名称（`gpt-sovits`、`indextts`、`neutts`、`vibevoice`）
    #[serde(default = "default_cloning_engine")]
    pub engine: String,

    /// 参考音频目录路径
    #[serde(default = "default_cloning_reference_dir")]
    pub reference_audio_dir: String,

    /// 是否自动从视频中提取说话人参考音频
    #[serde(default = "default_cloning_auto_extract")]
    pub auto_extract_speaker: bool,

    // ── GPT-SoVITS API v2 配置 ──
    /// GPT-SoVITS API v2 服务端点 URL
    ///
    /// 默认: `http://127.0.0.1:9880`
    /// 需要先启动 GPT-SoVITS API 服务: `python api_v2.py -a 127.0.0.1 -p 9880`
    #[serde(default = "default_gpt_sovits_api_url")]
    pub api_url: String,

    /// GPT-SoVITS 请求超时时间（秒）
    ///
    /// 克隆合成可能耗时较长，默认 60 秒。
    #[serde(default = "default_gpt_sovits_timeout_secs")]
    pub timeout_secs: u64,

    /// 参考音频的提示文本（参考音频对应的文字内容）
    ///
    /// 零样本克隆的关键参数：需要提供参考音频中说的原文。
    /// 例如参考音频说的是 "Hello, welcome to this video"，
    /// 则 prompt_text 应为该英文原文。
    #[serde(default)]
    pub prompt_text: Option<String>,

    /// 参考音频提示文本的语言（`en`、`zh`、`ja`、`ko` 等）
    #[serde(default = "default_gpt_sovits_prompt_lang")]
    pub prompt_lang: String,

    /// 目标合成文本的语言（`zh`、`en`、`ja`、`ko` 等）
    ///
    /// 视频翻译场景下通常为 `zh`（中文配音）
    #[serde(default = "default_gpt_sovits_text_lang")]
    pub text_lang: String,

    /// 文本分割方法
    ///
    /// - `cut0`: 不分割
    /// - `cut1`: 按标点分割
    /// - `cut5`: 按句子分割（默认，推荐）
    #[serde(default = "default_gpt_sovits_text_split_method")]
    pub text_split_method: String,

    /// Top-K 采样参数（默认 5）
    #[serde(default = "default_gpt_sovits_top_k")]
    pub top_k: u32,

    /// Top-P 采样参数（默认 1.0）
    #[serde(default = "default_gpt_sovits_top_p")]
    pub top_p: f32,

    /// 采样温度（默认 1.0，较低值更稳定）
    #[serde(default = "default_gpt_sovits_temperature")]
    pub temperature: f32,

    /// 重复惩罚（默认 1.35，防止 GPT 模型重复生成）
    #[serde(default = "default_gpt_sovits_repetition_penalty")]
    pub repetition_penalty: f32,

    /// GPT 模型权重路径（服务器端路径）
    ///
    /// 如果设置，引擎初始化时会调用 `/set_gpt_weights` 切换模型。
    /// 路径是 GPT-SoVITS 服务器上的文件路径，不是本地路径。
    #[serde(default)]
    pub gpt_model: Option<String>,

    /// SoVITS 模型权重路径（服务器端路径）
    ///
    /// 如果设置，引擎初始化时会调用 `/set_sovits_weights` 切换模型。
    /// 路径是 GPT-SoVITS 服务器上的文件路径，不是本地路径。
    #[serde(default)]
    pub sovits_model: Option<String>,

    // ── 子进程克隆引擎配置 ──
    /// 子进程克隆引擎命令路径
    ///
    /// 用于 `subprocess`、`indextts`、`qwen3-tts` 引擎类型。
    /// 指向外部 TTS CLI 工具的可执行文件路径。
    ///
    /// # 示例
    /// - IndexTTS-Rust: `/path/to/indextts`
    /// - qwen3_tts_rs: `/path/to/voice_clone`
    #[serde(default)]
    pub clone_command: Option<String>,

    /// 子进程克隆引擎模型路径
    ///
    /// 指向 TTS 模型文件或目录的路径（取决于引擎要求）。
    #[serde(default)]
    pub clone_model_path: Option<String>,

    /// 子进程克隆引擎参数模板
    ///
    /// 使用占位符替换的方式传递参数。支持的占位符：
    /// - `{text}`: 要合成的目标文本
    /// - `{ref_audio}`: 参考音频文件路径
    /// - `{output}`: 输出音频文件路径
    /// - `{model}`: 模型路径（`clone_model_path` 的值）
    /// - `{prompt_text}`: 参考音频对应的提示文本
    ///
    /// # IndexTTS-Rust 示例
    /// ```toml
    /// clone_args = ["synthesize", "--text", "{text}", "--voice", "{ref_audio}", "--output", "{output}"]
    /// ```
    ///
    /// # qwen3_tts_rs 示例
    /// ```toml
    /// clone_args = ["{model}", "{ref_audio}", "{text}", "chinese", "{output}"]
    /// ```
    #[serde(default)]
    pub clone_args: Vec<String>,

    /// 子进程克隆引擎超时时间（秒）
    ///
    /// 克隆合成可能耗时较长，默认 120 秒。
    #[serde(default = "default_clone_timeout_secs")]
    pub clone_timeout_secs: u64,

    // ── 参考音频提取配置 ──
    /// 参考音频提取增强配置
    ///
    /// 控制自动提取参考音频时的音频增强行为（人声增强、静音修剪、音量归一化）。
    #[serde(default)]
    pub voice_extractor: VoiceExtractorConfig,
}

impl Default for CloningConfig {
    /// 返回默认声音克隆配置
    ///
    /// - 启用: 是（默认启用声音克隆，克隆失败时自动降级到标准 TTS）
    /// - 引擎: `indextts`
    /// - 参考音频目录: `./references`
    /// - 自动提取: 启用
    /// - API URL: `http://127.0.0.1:9880`
    /// - 超时: 60 秒
    /// - prompt_lang: `en`
    /// - text_lang: `zh`
    fn default() -> Self {
        Self {
            enabled: default_cloning_enabled(),
            engine: default_cloning_engine(),
            reference_audio_dir: default_cloning_reference_dir(),
            auto_extract_speaker: default_cloning_auto_extract(),
            api_url: default_gpt_sovits_api_url(),
            timeout_secs: default_gpt_sovits_timeout_secs(),
            prompt_text: None,
            prompt_lang: default_gpt_sovits_prompt_lang(),
            text_lang: default_gpt_sovits_text_lang(),
            text_split_method: default_gpt_sovits_text_split_method(),
            top_k: default_gpt_sovits_top_k(),
            top_p: default_gpt_sovits_top_p(),
            temperature: default_gpt_sovits_temperature(),
            repetition_penalty: default_gpt_sovits_repetition_penalty(),
            gpt_model: None,
            sovits_model: None,
            clone_command: None,
            clone_model_path: None,
            clone_args: Vec::new(),
            clone_timeout_secs: default_clone_timeout_secs(),
            voice_extractor: VoiceExtractorConfig::default(),
        }
    }
}

// ─── 参考音频提取配置 ───────────────────────────────────

/// 参考音频提取配置
///
/// 控制从视频音频中自动提取参考音频时的音频增强行为。
/// 嵌套在 [`CloningConfig`] 中，通过 `[cloning.voice_extractor]` 段配置。
///
/// # 功能
/// - **人声增强**：使用 ffmpeg 滤波器去除低频噪声和高频杂音
/// - **静音修剪**：自动检测并裁剪音频首尾的静音段
/// - **音量归一化**：将音频 RMS 电平调整到目标值
///
/// # 配置示例
/// ```toml
/// [cloning.voice_extractor]
/// enable_enhancement = true
/// enable_silence_trim = true
/// enable_normalization = true
/// silence_threshold_db = -40.0
/// target_rms_db = -20.0
/// min_duration_secs = 3.0
/// max_duration_secs = 10.0
/// ideal_duration_secs = 5.0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceExtractorConfig {
    /// 是否启用 ffmpeg 人声增强（高通+低通滤波）
    #[serde(default = "default_voice_extractor_enable_enhancement")]
    pub enable_enhancement: bool,

    /// 是否启用静音修剪（裁剪首尾静音段）
    #[serde(default = "default_voice_extractor_enable_silence_trim")]
    pub enable_silence_trim: bool,

    /// 是否启用音量归一化（RMS 归一化）
    #[serde(default = "default_voice_extractor_enable_normalization")]
    pub enable_normalization: bool,

    /// 静音检测阈值（dB，-40 表示 -40dB 以下为静音）
    #[serde(default = "default_voice_extractor_silence_threshold_db")]
    pub silence_threshold_db: f64,

    /// 目标 RMS 电平（dBFS，-20 表示 -20dBFS）
    #[serde(default = "default_voice_extractor_target_rms_db")]
    pub target_rms_db: f64,

    /// 最小参考音频时长（秒）
    #[serde(default = "default_voice_extractor_min_duration")]
    pub min_duration_secs: f64,

    /// 最大参考音频时长（秒）
    #[serde(default = "default_voice_extractor_max_duration")]
    pub max_duration_secs: f64,

    /// 理想参考音频时长（秒，选择片段时优先接近此值）
    #[serde(default = "default_voice_extractor_ideal_duration")]
    pub ideal_duration_secs: f64,
}

impl Default for VoiceExtractorConfig {
    /// 返回默认参考音频提取配置
    ///
    /// - 人声增强: 启用
    /// - 静音修剪: 启用
    /// - 音量归一化: 启用
    /// - 静音阈值: -40dB
    /// - 目标 RMS: -20dBFS
    /// - 时长范围: 3–10 秒，理想 5 秒
    fn default() -> Self {
        Self {
            enable_enhancement: default_voice_extractor_enable_enhancement(),
            enable_silence_trim: default_voice_extractor_enable_silence_trim(),
            enable_normalization: default_voice_extractor_enable_normalization(),
            silence_threshold_db: default_voice_extractor_silence_threshold_db(),
            target_rms_db: default_voice_extractor_target_rms_db(),
            min_duration_secs: default_voice_extractor_min_duration(),
            max_duration_secs: default_voice_extractor_max_duration(),
            ideal_duration_secs: default_voice_extractor_ideal_duration(),
        }
    }
}

// ─── 批量处理配置 ─────────────────────────────────────────

/// 批量处理配置
///
/// 控制批量任务队列的最大并发数、内存阈值和优先级支持。
/// 可通过 TOML 文件中的 `[batch]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// 最大并行任务数（默认 = CPU 核心数 - 1）
    #[serde(default = "default_batch_max_concurrent")]
    pub max_concurrent: usize,

    /// 内存使用阈值百分比，超过时自动降低并发数
    #[serde(default = "default_batch_memory_threshold")]
    pub memory_threshold: f64,

    /// 是否启用任务优先级调度
    #[serde(default = "default_batch_enable_priority")]
    pub enable_priority: bool,
}

impl Default for BatchConfig {
    /// 返回默认批量处理配置
    ///
    /// - 最大并发: CPU 核心数 - 1
    /// - 内存阈值: 80%
    /// - 优先级: 启用
    fn default() -> Self {
        Self {
            max_concurrent: default_batch_max_concurrent(),
            memory_threshold: default_batch_memory_threshold(),
            enable_priority: default_batch_enable_priority(),
        }
    }
}

// ─── 检查点配置 ───────────────────────────────────────────

/// 检查点配置
///
/// 控制断点续传功能的启用状态、存储目录和保留策略。
/// 可通过 TOML 文件中的 `[checkpoint]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// 是否启用断点续传
    #[serde(default = "default_checkpoint_enabled")]
    pub enabled: bool,

    /// 检查点文件存储目录
    #[serde(default = "default_checkpoint_dir")]
    pub dir: String,

    /// 检查点保留天数（超过后自动清理）
    #[serde(default = "default_checkpoint_retention_days")]
    pub retention_days: u32,
}

impl Default for CheckpointConfig {
    /// 返回默认检查点配置
    ///
    /// - 启用: 是
    /// - 目录: `~/.cache/video-translator/checkpoints`
    /// - 保留天数: 7
    fn default() -> Self {
        Self {
            enabled: default_checkpoint_enabled(),
            dir: default_checkpoint_dir(),
            retention_days: default_checkpoint_retention_days(),
        }
    }
}

// ─── 性能调优配置 ─────────────────────────────────────────

/// 性能调优配置
///
/// 控制性能分析工具的启用状态和输出路径。
/// 可通过 TOML 文件中的 `[performance]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// 是否启用性能分析
    #[serde(default = "default_performance_enable_profiling")]
    pub enable_profiling: bool,

    /// 火焰图输出路径
    #[serde(default = "default_performance_flamegraph_output")]
    pub flamegraph_output: String,
}

impl Default for PerformanceConfig {
    /// 返回默认性能调优配置
    ///
    /// - 性能分析: 关闭
    /// - 火焰图输出: `./flamegraph.svg`
    fn default() -> Self {
        Self {
            enable_profiling: default_performance_enable_profiling(),
            flamegraph_output: default_performance_flamegraph_output(),
        }
    }
}

// ─── 字幕后处理配置 (TOML 映射) ────────────────────────────

/// 字幕后处理配置（TOML 映射结构）
///
/// 控制 ASR 输出字幕的后处理行为。
/// 通过 `[subtitle_postprocess]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitlePostProcessConfigField {
    /// 最短字幕时长（秒），短于此值考虑合并
    #[serde(default = "default_subtitle_min_duration")]
    pub min_duration: f64,

    /// 最长字幕时长（秒）
    #[serde(default = "default_subtitle_max_duration")]
    pub max_duration: f64,

    /// 合并时前后间隙阈值（秒）
    #[serde(default = "default_subtitle_merge_gap")]
    pub merge_gap_threshold: f64,

    /// 是否启用标点碎片重分配
    #[serde(default = "default_subtitle_fragment_redistribution")]
    pub enable_fragment_redistribution: bool,
}

fn default_subtitle_min_duration() -> f64 {
    1.0
}
fn default_subtitle_max_duration() -> f64 {
    10.0
}
fn default_subtitle_merge_gap() -> f64 {
    2.0
}
fn default_subtitle_fragment_redistribution() -> bool {
    true
}

impl Default for SubtitlePostProcessConfigField {
    fn default() -> Self {
        Self {
            min_duration: default_subtitle_min_duration(),
            max_duration: default_subtitle_max_duration(),
            merge_gap_threshold: default_subtitle_merge_gap(),
            enable_fragment_redistribution: default_subtitle_fragment_redistribution(),
        }
    }
}

// ─── 声画对齐 SpeedRate 配置 (TOML 映射) ───────────────────

/// 声画对齐 SpeedRate 配置（TOML 映射结构）
///
/// 控制逐段声画对齐的加速/慢放策略。
/// 通过 `[speed_rate]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedRateConfigField {
    /// 是否启用逐段声画对齐
    #[serde(default = "default_speed_rate_enabled")]
    pub enabled: bool,

    /// 对齐模式
    #[serde(default = "default_speed_rate_mode")]
    pub mode: SpeedRateMode,

    /// 最大音频加速倍率
    #[serde(default = "default_speed_rate_max_audio")]
    pub max_audio_speed: f64,

    /// 最大视频慢放倍率
    #[serde(default = "default_speed_rate_max_video")]
    pub max_video_slow: f64,
}

impl Default for SpeedRateConfigField {
    fn default() -> Self {
        Self {
            enabled: default_speed_rate_enabled(),
            mode: default_speed_rate_mode(),
            max_audio_speed: default_speed_rate_max_audio(),
            max_video_slow: default_speed_rate_max_video(),
        }
    }
}

// ─── 缓存配置 ─────────────────────────────────────────────

/// TTS 配音缓存 + 翻译缓存配置
///
/// 通过 `[cache]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// 是否启用 TTS 配音缓存
    #[serde(default = "default_tts_cache_enabled")]
    pub tts_cache_enabled: bool,

    /// 是否启用翻译缓存
    #[serde(default = "default_translation_cache_enabled")]
    pub translation_cache_enabled: bool,

    /// 是否启用 TTS 静音移除
    #[serde(default = "default_tts_remove_silence")]
    pub tts_remove_silence: bool,

    /// 缓存保留天数（自动清理过期缓存）
    #[serde(default = "default_cache_retention_days")]
    pub retention_days: u32,
}

fn default_cache_retention_days() -> u32 {
    7
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            tts_cache_enabled: default_tts_cache_enabled(),
            translation_cache_enabled: default_translation_cache_enabled(),
            tts_remove_silence: default_tts_remove_silence(),
            retention_days: default_cache_retention_days(),
        }
    }
}

// ─── 字幕输出配置 ─────────────────────────────────────────

/// 字幕输出配置
///
/// 控制是否生成 SRT 字幕文件及字幕类型。
/// 通过 `[subtitle]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleConfig {
    /// 字幕类型（none / hard / soft / hard_bilingual / soft_bilingual）
    #[serde(default = "default_subtitle_type")]
    pub subtitle_type: SubtitleType,

    /// SRT 文件输出目录（None = 与输出视频同目录）
    #[serde(default)]
    pub output_dir: Option<String>,
}

impl Default for SubtitleConfig {
    fn default() -> Self {
        Self {
            subtitle_type: default_subtitle_type(),
            output_dir: None,
        }
    }
}

// ─── 背景音乐配置 ─────────────────────────────────────────

/// 背景音乐配置
///
/// 控制是否在配音中混合背景音乐。
/// 通过 `[background_music]` 段配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundMusicConfig {
    /// 背景音乐文件路径（None = 不混合背景音乐）
    #[serde(default)]
    pub path: Option<String>,

    /// 背景音量（0.0-1.0，0.2 = 20% 音量）
    #[serde(default = "default_bgm_volume")]
    pub volume: f32,

    /// 是否循环背景音乐以匹配配音长度
    #[serde(default = "default_bgm_loop")]
    pub loop_bgm: bool,
}

fn default_bgm_loop() -> bool {
    true
}

impl Default for BackgroundMusicConfig {
    fn default() -> Self {
        Self {
            path: None,
            volume: default_bgm_volume(),
            loop_bgm: default_bgm_loop(),
        }
    }
}

// ─── 应用顶层配置 ─────────────────────────────────────────

/// 应用顶层配置
///
/// 聚合 ASR、TTS 子配置以及输出目录和并发任务数等全局参数。
/// 可通过 [`Config::from_file`] 从 TOML 文件加载，缺失字段自动使用默认值。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// ASR（语音识别）配置
    #[serde(default)]
    pub asr: AsrConfig,

    /// TTS（语音合成）配置
    #[serde(default)]
    pub tts: TtsConfig,

    /// 翻译模块配置
    #[serde(default)]
    pub translation: TranslationConfig,

    /// 输出目录路径
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// 最大并发任务数
    #[serde(default = "default_max_concurrent_tasks")]
    pub max_concurrent_tasks: usize,

    /// 流水线配置
    #[serde(default)]
    pub pipeline: PipelineConfig,

    /// 说话人分离配置
    #[serde(default)]
    pub diarization: DiarizationConfig,

    /// 声音克隆配置
    #[serde(default)]
    pub cloning: CloningConfig,

    /// 批量处理配置
    #[serde(default)]
    pub batch: BatchConfig,

    /// 检查点配置
    #[serde(default)]
    pub checkpoint: CheckpointConfig,

    /// 性能调优配置
    #[serde(default)]
    pub performance: PerformanceConfig,

    /// 音频同步配置
    #[serde(default)]
    pub audio_sync: AudioSyncConfig,

    /// 字幕后处理配置
    #[serde(default)]
    pub subtitle_postprocess: SubtitlePostProcessConfigField,

    /// 声画对齐 SpeedRate 配置
    #[serde(default)]
    pub speed_rate: SpeedRateConfigField,

    /// TTS 配音缓存 + 翻译缓存配置
    #[serde(default)]
    pub cache: CacheConfig,

    /// 字幕输出配置
    #[serde(default)]
    pub subtitle: SubtitleConfig,

    /// 背景音乐配置
    #[serde(default)]
    pub background_music: BackgroundMusicConfig,
}

impl Default for Config {
    /// 返回默认配置
    ///
    /// 所有子配置使用各自的 `Default` 实现。
    fn default() -> Self {
        Self {
            asr: AsrConfig::default(),
            tts: TtsConfig::default(),
            translation: TranslationConfig::default(),
            output_dir: default_output_dir(),
            max_concurrent_tasks: default_max_concurrent_tasks(),
            pipeline: PipelineConfig::default(),
            diarization: DiarizationConfig::default(),
            cloning: CloningConfig::default(),
            batch: BatchConfig::default(),
            checkpoint: CheckpointConfig::default(),
            performance: PerformanceConfig::default(),
            audio_sync: AudioSyncConfig::default(),
            subtitle_postprocess: SubtitlePostProcessConfigField::default(),
            speed_rate: SpeedRateConfigField::default(),
            cache: CacheConfig::default(),
            subtitle: SubtitleConfig::default(),
            background_music: BackgroundMusicConfig::default(),
        }
    }
}

impl Config {
    /// 从 TOML 文件加载配置
    ///
    /// 读取指定路径的 TOML 文件并反序列化为 [`Config`]。
    /// 缺失的字段会自动使用对应的默认值填充（合并逻辑）。
    ///
    /// # 参数
    /// - `path`: TOML 配置文件路径
    ///
    /// # 错误
    /// - [`AppError::Io`][]: 文件读取失败（如文件不存在）
    /// - [`AppError::TomlDe`][]: TOML 反序列化失败
    ///
    /// # 示例
    /// ```no_run
    /// use vt_core::config::Config;
    /// use vt_core::error::AppResult;
    ///
    /// fn load() -> AppResult<Config> {
    ///     Config::from_file("config.toml")
    /// }
    /// ```
    #[tracing::instrument(skip(path), fields(path = ?path.as_ref()))]
    pub fn from_file<P: AsRef<Path>>(path: P) -> AppResult<Self> {
        let path_ref = path.as_ref();
        let content = std::fs::read_to_string(path_ref).map_err(|e| {
            AppError::Config(format!("Failed to read config file {:?}: {e}", path_ref))
        })?;
        let config: Config = toml::from_str(&content)?;

        // P6: 配置验证 — 加载后自动验证
        let warnings = crate::config_validation::validate_config(
            &config,
            crate::config_validation::ValidationLevel::Standard,
        );
        for w in &warnings {
            match w.severity {
                crate::config_validation::Severity::Error => {
                    tracing::error!("Config validation: {}", w);
                }
                crate::config_validation::Severity::Warning => {
                    tracing::warn!("Config validation: {}", w);
                }
                crate::config_validation::Severity::Info => {
                    tracing::info!("Config validation: {}", w);
                }
            }
        }

        // P6: 未知字段检测
        let unknown = crate::config_validation::detect_unknown_fields(&content);
        for w in &unknown {
            tracing::warn!("Config validation: {}", w);
        }

        Ok(config)
    }
}
