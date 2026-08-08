//! 语音合成 (TTS) 模块
//!
//! 提供中文文本到语音的合成功能，支持多音色选择、语速/音调/音量控制、
//! 音频后处理和并行合成。
//!
//! # 功能概览
//! - [`TtsEngine`] trait：定义语音合成的标准接口（含音色列表查询）
//! - [`SayEngine`]：基于 macOS `say` 命令的离线 TTS 引擎
//!   - 多音色支持：通过 [`VoiceManager`] 管理内置 6 种音色（3 女 + 3 男）
//!   - 音调控制：通过 ffmpeg `asetrate` + `atempo` 实现音调偏移
//!   - 音量控制：通过 ffmpeg `volume` 滤镜
//!   - 自动增益：通过 ffmpeg `dynaudnorm` 滤镜
//!   - 淡入淡出：通过 ffmpeg `afade` + `areverse` 消除爆破音
//!   - 采样率：支持 16000 / 24000 / 48000 Hz
//! - [`KokoroEngine`]：带 SayEngine 降级的高级引擎包装
//! - [`AudioPostProcessor`][crate::audio_post_process::AudioPostProcessor]：独立音频后处理器
//!   - 齿音消除（highshelf 衰减 6kHz+）
//!   - 低频增强（lowshelf 提升 300Hz）
//!   - 自动增益控制（dynaudnorm）
//!   - 淡入淡出（afade + areverse，消除拼接感）
//! - 内容哈希缓存：对已合成的文本跳过重复合成
//!
//! # 离线运行
//! 本模块完全离线运行，使用 macOS 内置的 `say` 命令进行语音合成。
//! 音频后处理依赖 `ffmpeg`（项目已有依赖）。
//!
//! # 系统要求
//! - macOS 10.15+（内置 `say` 命令）
//! - 系统已安装中文语音包（如 `Tingting`、`Meijia`）
//! - `ffmpeg` 可用（用于音频后处理）
//!
//! # Kokoro ONNX 扩展
//! 当 `kokoro-onnx` feature 启用且 Kokoro 模型已下载时，
//! `KokoroEngine` 将使用 ONNX Runtime 进行推理，获得更自然的中文语音。
//! 模型不可用时自动降级到 `SayEngine`。
//!
//! # 示例
//! ```no_run
//! use vt_core::config::TtsConfig;
//! use vt_core::tts::{SayEngine, TtsEngine};
//! use vt_core::error::AppResult;
//!
//! fn synthesize() -> AppResult<()> {
//!     let config = TtsConfig::default();
//!     let engine = SayEngine::new(&config)?;
//!     let voices = engine.list_voices();
//!     println!("Available voices: {}", voices.len());
//!     // engine.synthesize_segments(&mut segments, &config)?;
//!     Ok(())
//! }
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use sha2::{Digest, Sha256};

// Re-export AudioPostProcessor for convenience
pub use crate::audio_post_process::AudioPostProcessor;

use crate::config::TtsConfig;
use crate::error::{AppError, AppResult};
use crate::models::segment::Segment;
use crate::voice_manager::{VoiceInfo, VoiceManager};

// ─── 常量 ─────────────────────────────────────────────────

/// `say` 命令基础输出采样率
///
/// `say` 命令直接输出此采样率的 WAV，后续由 ffmpeg 重采样到目标采样率。
/// 使用 24kHz 作为基础采样率（macOS 神经语音引擎的原生质量）。
const SAY_BASE_SAMPLE_RATE: u32 = 24000;

// ─── TtsEngine Trait ─────────────────────────────────────

/// 语音合成引擎接口
///
/// 定义批量语音合成的标准接口。各引擎（say、Kokoro 等）实现此 trait。
/// 合成结果写入每个 `Segment` 的 `tts_audio_path` 字段。
///
/// # 线程安全
/// 实现者必须满足 `Send + Sync`，以支持并行合成。
pub trait TtsEngine: Send + Sync {
    /// 批量合成 Segment 列表的语音
    ///
    /// 为每个 Segment 的 `target_text` 合成语音，生成 WAV 音频文件，
    /// 并将文件路径填入 `tts_audio_path` 字段。
    ///
    /// # 参数
    /// - `segments`: 待合成的片段列表（原地修改）
    /// - `config`: TTS 配置（语速、音色、缓存目录、并行数）
    ///
    /// # 返回
    /// 所有生成的音频文件路径列表，顺序与 `segments` 一致。
    ///
    /// # 错误
    /// - [`AppError::TtsError`][]: Segment 未翻译、文本为空、合成失败
    fn synthesize_segments(
        &self,
        segments: &mut [Segment],
        config: &TtsConfig,
    ) -> AppResult<Vec<PathBuf>>;

    /// 列出引擎支持的可用音色
    ///
    /// 返回音色信息列表，包含音色 ID、名称、性别、语言和描述。
    /// 前端可通过此接口获取可选音色列表。
    ///
    /// # 返回
    /// 音色信息列表（至少包含 2 种女声和 2 种男声）
    fn list_voices(&self) -> Vec<VoiceInfo>;
}

// ─── SayEngine ───────────────────────────────────────────

/// 基于 macOS `say` 命令的离线语音合成引擎
///
/// 通过调用 macOS 系统内置的 `say` 命令合成中文语音，
/// 并通过 ffmpeg 进行音频后处理（音调偏移、音量控制、AGC、淡入淡出、重采样）。
///
/// # 多音色支持
/// 通过 [`VoiceManager`] 管理内置音色注册表，支持 3 种女声和 3 种男声。
/// 男声通过对女声基线应用音调偏移（pitch multiplier < 1.0）实现。
///
/// # 音频后处理流程
/// 1. `say` 命令生成基础 WAV（24kHz mono 16-bit PCM）
/// 2. ffmpeg 应用以下滤镜链（如有需要）：
///    - `asetrate` + `atempo`：音调偏移（不改变语速）
///    - `volume`：音量调整
///    - `dynaudnorm`：自动增益控制
///    - `afade` + `areverse`：淡入淡出（消除爆破音）
///    - `aresample`：重采样到目标采样率
/// 3. 输出最终 WAV 文件
///
/// # 缓存机制
/// 使用 `SHA-256(text + voice_id + speed + pitch + volume + sample_rate)` 作为缓存文件名，
/// 相同参数组合不会重复合成。
///
/// # 并行合成
/// 当 `TtsConfig.parallel_tasks > 1` 且 Segment 数量 > 1 时，
/// 使用 `rayon` 线程池并行合成。
///
/// # 示例
/// ```no_run
/// use vt_core::config::TtsConfig;
/// use vt_core::tts::{SayEngine, TtsEngine};
///
/// let config = TtsConfig::default();
/// let engine = SayEngine::new(&config).unwrap();
/// let voices = engine.list_voices();
/// // engine.synthesize_segments(&mut segments, &config);
/// ```
pub struct SayEngine {
    /// 音频缓存目录
    cache_dir: PathBuf,
    /// 音色管理器
    voice_manager: VoiceManager,
}

impl SayEngine {
    /// 创建 SayEngine 实例
    ///
    /// 初始化缓存目录和音色管理器，若缓存目录不存在则自动创建。
    ///
    /// # 参数
    /// - `config`: TTS 配置（从中读取 `cache_dir`）
    ///
    /// # 错误
    /// - [`AppError::Config`][]: `HOME` 环境变量未设置（当 `cache_dir` 以 `~` 开头时）
    /// - [`AppError::Io`][]: 缓存目录创建失败
    pub fn new(config: &TtsConfig) -> AppResult<Self> {
        let cache_dir = expand_cache_dir(&config.cache_dir)?;
        std::fs::create_dir_all(&cache_dir)?;
        tracing::debug!("SayEngine initialized with cache_dir: {:?}", cache_dir);
        Ok(Self {
            cache_dir,
            voice_manager: VoiceManager::new(),
        })
    }

    /// 使用指定缓存目录创建引擎
    ///
    /// # 参数
    /// - `cache_dir`: 缓存目录路径
    ///
    /// # 错误
    /// - [`AppError::Io`][]: 目录创建失败
    pub fn with_cache_dir(cache_dir: impl AsRef<Path>) -> AppResult<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            cache_dir,
            voice_manager: VoiceManager::new(),
        })
    }

    /// 获取缓存目录路径
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// 计算文本的缓存文件路径
    ///
    /// 使用 `SHA-256(text + voice_id + speed + pitch + volume + sample_rate)` 作为文件名，
    /// 确保相同参数的文本产生相同的缓存路径。
    fn cache_path(
        &self,
        text: &str,
        voice_id: &str,
        speed: f32,
        pitch: f32,
        volume: f32,
        sample_rate: u32,
    ) -> PathBuf {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hasher.update(voice_id.as_bytes());
        hasher.update(speed.to_le_bytes());
        hasher.update(pitch.to_le_bytes());
        hasher.update(volume.to_le_bytes());
        hasher.update(sample_rate.to_le_bytes());
        let hash = hasher.finalize();
        let hash_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        self.cache_dir.join(format!("{hash_hex}.wav"))
    }

    /// 将语速倍率映射为 `say` 命令的 `-r` 参数值（字/分钟）
    ///
    /// - `speed=1.0` → 175（默认语速）
    /// - 范围限制在 [50, 500] 之间
    fn speed_to_rate(speed: f32) -> i32 {
        let rate = (175.0 * speed) as i32;
        rate.clamp(50, 500)
    }

    /// 解析配置中的音色信息
    ///
    /// 优先使用 `voice_id` 查找音色，找不到时回退到 `voice` 字段匹配。
    ///
    /// # 参数
    /// - `config`: TTS 配置（读取 `voice_id` 和 `voice` 字段）
    ///
    /// # 返回
    /// 匹配到的音色信息引用。若 `voice_id` 不存在且 `voice` 字段也不匹配，
    /// 返回默认音色（第一个女声）。
    pub fn resolve_voice(&self, config: &TtsConfig) -> &VoiceInfo {
        // 先按 voice_id 查找
        if let Some(voice) = self.voice_manager.find_by_id(&config.voice_id) {
            return voice;
        }
        // 回退：尝试将 voice 字段（如 "Tingting"）匹配到注册表中的 say_voice
        for v in self.voice_manager.list_voices() {
            if v.say_voice.eq_ignore_ascii_case(&config.voice) {
                return v;
            }
        }
        // 最终回退到默认音色
        self.voice_manager.default_voice()
    }

    /// 计算组合音调倍率
    ///
    /// 组合音调 = 音色基础音调倍率 × 用户配置的音调倍率
    /// 结果限制在 [0.5, 2.0] 范围内。
    ///
    /// 委托给 [`AudioPostProcessor::combined_pitch`]。
    fn combined_pitch(voice: &VoiceInfo, config: &TtsConfig) -> f32 {
        AudioPostProcessor::combined_pitch(voice, config)
    }

    /// 判断是否需要 ffmpeg 后处理
    ///
    /// 始终返回 `true`，因为我们需要应用均衡器（减少齿音）和
    /// 淡入淡出（消除拼接感）以统一音色和音质。
    ///
    /// 以下条件也会触发后处理：
    /// - 组合音调 ≠ 1.0
    /// - 音量 ≠ 1.0
    /// - 目标采样率 ≠ 基础采样率
    fn needs_postprocessing(_voice: &VoiceInfo, _config: &TtsConfig) -> bool {
        true
    }

    /// 构建 ffmpeg 音频滤镜链
    ///
    /// 委托给 [`AudioPostProcessor::build_filter_chain`]，保持一致的滤镜链逻辑。
    fn build_filter_chain(voice: &VoiceInfo, config: &TtsConfig) -> String {
        AudioPostProcessor::build_filter_chain(voice, config)
    }

    /// 使用 ffmpeg 进行音频后处理
    ///
    /// 将 `say` 生成的原始 WAV 通过 ffmpeg 滤镜链处理为目标格式。
    ///
    /// # 参数
    /// - `input`: `say` 生成的原始 WAV 路径
    /// - `output`: 后处理后的目标 WAV 路径
    /// - `voice`: 音色信息
    /// - `config`: TTS 配置
    ///
    /// # 错误
    /// - [`AppError::TtsError`][]: ffmpeg 执行失败
    fn run_ffmpeg_postprocess(
        input: &Path,
        output: &Path,
        voice: &VoiceInfo,
        config: &TtsConfig,
    ) -> AppResult<()> {
        let filter_chain = Self::build_filter_chain(voice, config);

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .arg("-i")
            .arg(input)
            .arg("-af")
            .arg(&filter_chain)
            .arg("-ar")
            .arg(config.sample_rate.to_string())
            .arg("-ac")
            .arg("1")
            .arg("-c:a")
            .arg("pcm_s16le")
            .arg(output);

        tracing::debug!(
            "Running ffmpeg postprocess: filter='{}', sample_rate={}, output={:?}",
            filter_chain,
            config.sample_rate,
            output
        );

        let output_result = cmd.output().map_err(|e| {
            AppError::TtsError(format!(
                "Failed to execute ffmpeg for post-processing: {e}\n\
                 确保 ffmpeg 已安装且在 PATH 中。"
            ))
        })?;

        if !output_result.status.success() {
            let stderr = String::from_utf8_lossy(&output_result.stderr);
            return Err(AppError::TtsError(format!(
                "ffmpeg post-processing failed (exit code: {}): {stderr}",
                output_result.status.code().unwrap_or(-1)
            )));
        }

        Ok(())
    }

    /// 合成单个文本的语音
    ///
    /// 内部流程：
    /// 1. 解析音色信息（`voice_id` → `VoiceInfo`）
    /// 2. 检查缓存——若已存在则直接返回路径
    /// 3. 调用 `say` 命令合成基础音频（24kHz mono WAV）
    /// 4. 如需后处理，调用 ffmpeg（音调偏移、音量、AGC、淡入淡出、重采样）
    /// 5. 验证输出 WAV 文件格式正确
    ///
    /// # 参数
    /// - `text`: 待合成的中文文本
    /// - `config`: TTS 配置
    ///
    /// # 错误
    /// - [`AppError::TtsError`][]: `say` 命令执行失败或未生成音频
    /// - [`AppError::TtsAudioEncodeError`][]: WAV 格式验证失败
    fn synthesize_single(&self, text: &str, config: &TtsConfig) -> AppResult<PathBuf> {
        let voice = self.resolve_voice(config);
        let combined_pitch = Self::combined_pitch(voice, config);

        let cache_path = self.cache_path(
            text,
            &config.voice_id,
            config.speed,
            combined_pitch,
            config.volume,
            config.sample_rate,
        );

        // 检查缓存
        if cache_path.exists() {
            tracing::debug!("TTS cache hit: {:?}", cache_path);
            return Ok(cache_path);
        }

        tracing::debug!(
            "TTS cache miss, synthesizing: text='{}' ({} chars), voice='{}' (pitch={:.2})",
            text,
            text.chars().count(),
            voice.name,
            combined_pitch
        );

        // 生成临时文件路径（say 原始输出）
        let temp_path = cache_path.with_extension("raw.wav");

        // 调用 say 命令合成基础音频
        let rate = Self::speed_to_rate(config.speed);
        let data_format = format!("LEI16@{SAY_BASE_SAMPLE_RATE}");
        let output = Command::new("say")
            .arg("-v")
            .arg(&voice.say_voice)
            .arg("-r")
            .arg(rate.to_string())
            .arg("-o")
            .arg(&temp_path)
            .arg("--file-format=WAVE")
            .arg("--data-format")
            .arg(&data_format)
            .arg(text)
            .output()
            .map_err(|e| {
                AppError::TtsError(format!(
                    "Failed to execute 'say' command: {e}\n\
                     确保系统为 macOS 且 'say' 命令可用。"
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // 清理可能产生的空文件
            let _ = std::fs::remove_file(&temp_path);
            return Err(AppError::TtsError(format!(
                "say command failed (exit code: {}): {stderr}\n\
                 可能原因：\n\
                 1. 语音 '{}' 未安装（运行 `say -v '?'` 查看可用语音）\n\
                 2. 文本包含非法字符\n\
                 3. 磁盘空间不足",
                output.status.code().unwrap_or(-1),
                voice.say_voice,
            )));
        }

        // 验证 say 输出文件存在
        if !temp_path.exists() {
            return Err(AppError::TtsError(
                "say command did not produce output file".to_string(),
            ));
        }

        // 判断是否需要 ffmpeg 后处理
        if Self::needs_postprocessing(voice, config) {
            tracing::debug!("Applying ffmpeg post-processing (pitch/volume/sample_rate)");

            // ffmpeg 后处理：temp_path → cache_path
            Self::run_ffmpeg_postprocess(&temp_path, &cache_path, voice, config)?;

            // 清理临时文件
            let _ = std::fs::remove_file(&temp_path);
        } else {
            // 无需后处理，直接重命名临时文件为最终文件
            std::fs::rename(&temp_path, &cache_path)
                .map_err(|e| AppError::TtsError(format!("Failed to rename temp file: {e}")))?;
        }

        // 验证最终 WAV 格式
        Self::validate_wav(&cache_path, config.sample_rate)?;

        tracing::info!(
            "TTS synthesis complete: {:?} ({} bytes, {} Hz)",
            cache_path,
            std::fs::metadata(&cache_path).map(|m| m.len()).unwrap_or(0),
            config.sample_rate
        );

        Ok(cache_path)
    }

    /// 验证 WAV 文件格式（指定采样率 mono 16-bit PCM）
    fn validate_wav(path: &Path, expected_sample_rate: u32) -> AppResult<()> {
        let mut reader = hound::WavReader::open(path).map_err(|e| {
            AppError::TtsAudioEncodeError(format!(
                "Output WAV file is invalid: {e}\n\
                 文件: {path:?}"
            ))
        })?;
        let spec = reader.spec();

        if spec.sample_rate != expected_sample_rate
            || spec.channels != 1
            || spec.bits_per_sample != 16
        {
            return Err(AppError::TtsAudioEncodeError(format!(
                "Output WAV format mismatch: expected {}Hz mono 16-bit, got {}Hz {}ch {}bit\n\
                 文件: {path:?}",
                expected_sample_rate, spec.sample_rate, spec.channels, spec.bits_per_sample
            )));
        }

        // 确保有实际音频数据
        let sample_count: usize = reader.samples::<i16>().count();
        if sample_count == 0 {
            return Err(AppError::TtsAudioEncodeError(format!(
                "Output WAV contains no audio samples\n\
                 文件: {path:?}"
            )));
        }

        Ok(())
    }

    /// 对音频文件进行后处理（均衡器 + 淡入淡出 + 重采样）
    ///
    /// 这是一个公共接口，允许外部对已有的 WAV 文件进行音频后处理。
    /// 委托给 [`AudioPostProcessor`]，保持统一的后处理逻辑。
    ///
    /// 主要用于：
    /// - 衰减 6kHz 以上的高频（减少齿音）
    /// - 提升中低频（增强温暖感）
    /// - 应用淡入淡出（消除拼接感）
    /// - 统一采样率
    ///
    /// # 参数
    /// - `input`: 输入 WAV 文件路径
    /// - `output`: 输出 WAV 文件路径
    /// - `config`: TTS 配置（读取 `eq_high_shelf_db`、`crossfade_duration_ms`、`sample_rate` 等）
    ///
    /// # 错误
    /// - [`AppError::TtsError`][]: ffmpeg 执行失败
    pub fn audio_post_process(input: &Path, output: &Path, config: &TtsConfig) -> AppResult<()> {
        let processor = AudioPostProcessor::new(config);
        processor.process(input, output)
    }

    /// 逐段合成（降级方案）
    ///
    /// 当连续合成失败时使用此方法，逐段调用 `synthesize_single`。
    /// 使用并行合成以提高吞吐量（当 `parallel_tasks > 1` 时）。
    fn synthesize_per_segment(
        &self,
        texts: &[&str],
        config: &TtsConfig,
    ) -> Vec<AppResult<PathBuf>> {
        if config.parallel_tasks > 1 && texts.len() > 1 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.parallel_tasks)
                .build()
                .map_err(|e| AppError::TtsError(format!("Failed to create thread pool: {e}")));

            match pool {
                Ok(pool) => pool.install(|| {
                    use rayon::prelude::*;
                    texts
                        .par_iter()
                        .map(|text| self.synthesize_single(text, config))
                        .collect()
                }),
                Err(e) => {
                    tracing::warn!("Thread pool creation failed: {e}, using serial synthesis");
                    texts
                        .iter()
                        .map(|text| self.synthesize_single(text, config))
                        .collect()
                }
            }
        } else {
            texts
                .iter()
                .map(|text| self.synthesize_single(text, config))
                .collect()
        }
    }

    /// 连续合成：将所有文本拼接为一次 `say` 调用
    ///
    /// **解决音色不一致问题的关键方法。**
    ///
    /// macOS 神经 TTS 引擎对每段独立合成的文本会生成不同音色特征
    /// （音调轮廓、语速、强调方式），导致听起来像不同人说话。
    /// 连续合成将所有文本拼接为一次 `say` 调用，确保神经 TTS 引擎
    /// 在一致的上下文中生成所有语音，然后用静音检测分割回各段。
    ///
    /// # 流程
    /// 1. 用 `[[slnc 500]]`（500ms 静音标记）拼接所有文本
    /// 2. 调用 `say` 一次合成完整音频
    /// 3. 用 ffmpeg `silencedetect` 检测 400ms+ 的静音段
    /// 4. 在静音边界处分割音频
    /// 5. 对每段音频应用后处理（EQ + AGC + 淡入淡出）
    ///
    /// # 参数
    /// - `texts`: 所有待合成文本
    /// - `config`: TTS 配置
    ///
    /// # 返回
    /// 分割后的各段音频路径。若分割数量与文本数量不符，调用方应降级到逐段合成。
    fn synthesize_continuous(&self, texts: &[&str], config: &TtsConfig) -> AppResult<Vec<PathBuf>> {
        let voice = self.resolve_voice(config);
        let combined_pitch = Self::combined_pitch(voice, config);

        // 1. 拼接所有文本，用 [[slnc 500]]（500ms 静音）分隔
        // macOS say 支持 [[slnc N]] 嵌入命令插入 N 毫秒静音
        const SEGMENT_SEPARATOR: &str = "[[slnc 500]]";
        let full_text = texts.join(SEGMENT_SEPARATOR);

        tracing::info!(
            "Continuous synthesis: {} segments, {} total chars, voice='{}'",
            texts.len(),
            full_text.chars().count(),
            voice.name
        );

        // 2. 计算缓存路径（使用全文哈希）
        let cache_key = self.cache_path(
            &full_text,
            &config.voice_id,
            config.speed,
            combined_pitch,
            config.volume,
            config.sample_rate,
        );
        let raw_path = cache_key.with_extension("continuous_raw.wav");
        let split_dir = cache_key
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(format!(
                "continuous_{}",
                cache_key
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default()
            ));

        // 检查缓存：如果分割目录已存在且包含正确数量的文件，直接返回
        if split_dir.exists() {
            let mut cached_paths: Vec<PathBuf> = Vec::new();
            let mut all_exist = true;
            for i in 0..texts.len() {
                let p = split_dir.join(format!("seg_{i:04}.wav"));
                if p.exists() {
                    cached_paths.push(p);
                } else {
                    all_exist = false;
                    break;
                }
            }
            if all_exist && cached_paths.len() == texts.len() {
                tracing::debug!(
                    "Continuous synthesis cache hit: {} segments",
                    cached_paths.len()
                );
                return Ok(cached_paths);
            }
        }

        // 3. 调用 say 合成完整音频
        let rate = Self::speed_to_rate(config.speed);
        let data_format = format!("LEI16@{SAY_BASE_SAMPLE_RATE}");
        let say_output = Command::new("say")
            .arg("-v")
            .arg(&voice.say_voice)
            .arg("-r")
            .arg(rate.to_string())
            .arg("-o")
            .arg(&raw_path)
            .arg("--file-format=WAVE")
            .arg("--data-format")
            .arg(&data_format)
            .arg(&full_text)
            .output()
            .map_err(|e| {
                AppError::TtsError(format!(
                    "Failed to execute 'say' for continuous synthesis: {e}"
                ))
            })?;

        if !say_output.status.success() {
            let stderr = String::from_utf8_lossy(&say_output.stderr);
            let _ = std::fs::remove_file(&raw_path);
            return Err(AppError::TtsError(format!(
                "say continuous synthesis failed: {stderr}"
            )));
        }

        if !raw_path.exists() {
            return Err(AppError::TtsError(
                "say command did not produce output file for continuous synthesis".to_string(),
            ));
        }

        // 4. 用 ffmpeg silencedetect 检测静音边界
        let silence_starts = self.detect_silence_boundaries(&raw_path)?;

        // 5. 分割音频
        // 期望的静音数量 = texts.len() - 1（N 段文本之间有 N-1 个分隔符）
        let expected_silences = texts.len() - 1;

        if silence_starts.len() < expected_silences {
            tracing::warn!(
                "Silence detection found {} boundaries, expected {}. \
                 Text may contain natural pauses confusing the detector.",
                silence_starts.len(),
                expected_silences
            );
            // 尝试用文本长度比例估算分割点
            return self.split_by_proportion(&raw_path, texts, config, &split_dir);
        }

        // 取前 N-1 个静音点作为分割边界
        let split_points: Vec<f64> = silence_starts.into_iter().take(expected_silences).collect();

        // 创建分割目录
        std::fs::create_dir_all(&split_dir)?;

        // 用 ffmpeg 分割音频并应用后处理
        let mut segment_paths = Vec::with_capacity(texts.len());

        // 计算各段的时间范围
        let mut prev_end = 0.0;
        for (i, &split_start) in split_points.iter().enumerate() {
            let seg_path = split_dir.join(format!("seg_{i:04}.wav"));

            // 提取 [prev_end, split_start] 范围的音频
            let duration = split_start - prev_end;
            if duration <= 0.0 {
                tracing::warn!("Segment {i} has non-positive duration ({duration:.3}s)");
                segment_paths.clear();
                break;
            }

            Self::extract_and_process_segment(
                &raw_path, &seg_path, prev_end, duration, voice, config,
            )?;

            segment_paths.push(seg_path);
            prev_end = split_start;
        }

        // 处理最后一段
        if segment_paths.len() == expected_silences {
            let last_path = split_dir.join(format!("seg_{expected_silences:04}.wav"));
            // 获取音频总时长
            let total_duration = Self::get_audio_duration(&raw_path)?;
            let last_duration = total_duration - prev_end;
            if last_duration > 0.0 {
                Self::extract_and_process_segment(
                    &raw_path,
                    &last_path,
                    prev_end,
                    last_duration,
                    voice,
                    config,
                )?;
                segment_paths.push(last_path);
            }
        }

        // 清理原始音频文件
        let _ = std::fs::remove_file(&raw_path);

        if segment_paths.len() != texts.len() {
            tracing::warn!(
                "Split produced {} segments, expected {}",
                segment_paths.len(),
                texts.len()
            );
            return Err(AppError::TtsError(format!(
                "Continuous synthesis split mismatch: got {} segments, expected {}",
                segment_paths.len(),
                texts.len()
            )));
        }

        Ok(segment_paths)
    }

    /// 使用 ffmpeg silencedetect 检测音频中的静音边界
    ///
    /// 检测 400ms 以上、低于 -30dB 的静音段，返回每个静音段的起始时间。
    fn detect_silence_boundaries(&self, audio_path: &Path) -> AppResult<Vec<f64>> {
        let output = Command::new("ffmpeg")
            .arg("-i")
            .arg(audio_path)
            .arg("-af")
            .arg("silencedetect=noise=-30dB:d=0.4")
            .arg("-f")
            .arg("null")
            .arg("-")
            .output()
            .map_err(|e| AppError::TtsError(format!("Failed to run ffmpeg silencedetect: {e}")))?;

        // ffmpeg silencedetect 输出到 stderr
        let stderr = String::from_utf8_lossy(&output.stderr);

        // 解析 "silence_start: X.XXXXXX" 行
        let mut starts: Vec<f64> = Vec::new();
        for line in stderr.lines() {
            if let Some(pos) = line.find("silence_start:") {
                let rest = &line[pos + "silence_start:".len()..].trim();
                if let Ok(val) = rest.parse::<f64>() {
                    starts.push(val);
                }
            }
        }

        tracing::debug!(
            "Silence detection found {} boundaries in {:?}",
            starts.len(),
            audio_path
        );

        Ok(starts)
    }

    /// 按文本长度比例分割音频（降级方案）
    ///
    /// 当静音检测失败时，根据各段文本字符数比例估算分割时间点。
    fn split_by_proportion(
        &self,
        audio_path: &Path,
        texts: &[&str],
        config: &TtsConfig,
        split_dir: &Path,
    ) -> AppResult<Vec<PathBuf>> {
        let voice = self.resolve_voice(config);
        let total_duration = Self::get_audio_duration(audio_path)?;

        // 计算各段字符数比例
        let char_counts: Vec<usize> = texts.iter().map(|t| t.chars().count()).collect();
        let total_chars: usize = char_counts.iter().sum();

        if total_chars == 0 {
            return Err(AppError::TtsError(
                "Cannot split by proportion: all texts are empty".to_string(),
            ));
        }

        std::fs::create_dir_all(split_dir)?;

        let mut segment_paths = Vec::with_capacity(texts.len());
        let mut prev_end = 0.0;

        for (i, &count) in char_counts.iter().enumerate() {
            let proportion = count as f64 / total_chars as f64;
            let seg_duration = total_duration * proportion;
            let seg_path = split_dir.join(format!("seg_{i:04}.wav"));

            Self::extract_and_process_segment(
                audio_path,
                &seg_path,
                prev_end,
                seg_duration,
                voice,
                config,
            )?;

            segment_paths.push(seg_path);
            prev_end += seg_duration;
        }

        Ok(segment_paths)
    }

    /// 获取音频文件时长（秒）
    fn get_audio_duration(path: &Path) -> AppResult<f64> {
        let output = Command::new("ffprobe")
            .arg("-v")
            .arg("error")
            .arg("-show_entries")
            .arg("format=duration")
            .arg("-of")
            .arg("default=noprint_wrappers=1:nokey=1")
            .arg(path)
            .output()
            .map_err(|e| AppError::TtsError(format!("Failed to run ffprobe: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .trim()
            .parse::<f64>()
            .map_err(|e| AppError::TtsError(format!("Failed to parse audio duration: {e}")))
    }

    /// 提取音频的指定时间段并应用后处理
    ///
    /// 从 `input` 中提取 `[start, start+duration]` 范围的音频，
    /// 应用 EQ + AGC + 淡入淡出后处理，输出到 `output`。
    fn extract_and_process_segment(
        input: &Path,
        output: &Path,
        start: f64,
        duration: f64,
        voice: &VoiceInfo,
        config: &TtsConfig,
    ) -> AppResult<()> {
        let filter_chain = AudioPostProcessor::build_filter_chain(voice, config);

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .arg("-ss")
            .arg(format!("{start:.6}"))
            .arg("-t")
            .arg(format!("{duration:.6}"))
            .arg("-i")
            .arg(input)
            .arg("-af")
            .arg(&filter_chain)
            .arg("-ar")
            .arg(config.sample_rate.to_string())
            .arg("-ac")
            .arg("1")
            .arg("-c:a")
            .arg("pcm_s16le")
            .arg(output);

        let result = cmd.output().map_err(|e| {
            AppError::TtsError(format!("Failed to run ffmpeg for segment extraction: {e}"))
        })?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(AppError::TtsError(format!(
                "ffmpeg segment extraction failed: {stderr}"
            )));
        }

        Ok(())
    }
}

impl TtsEngine for SayEngine {
    fn synthesize_segments(
        &self,
        segments: &mut [Segment],
        config: &TtsConfig,
    ) -> AppResult<Vec<PathBuf>> {
        if segments.is_empty() {
            return Ok(Vec::new());
        }

        // 校验所有 Segment 已翻译且文本非空
        for seg in segments.iter() {
            match &seg.target_text {
                None => {
                    return Err(AppError::TtsError(format!(
                        "Segment {} has no target_text (not translated)",
                        seg.id
                    )));
                }
                Some(text) if text.is_empty() => {
                    return Err(AppError::TtsError(format!(
                        "Segment {} has empty target_text",
                        seg.id
                    )));
                }
                _ => {}
            }
        }

        // 状态转换：Translated → Synthesizing
        for seg in segments.iter_mut() {
            seg.start_synthesizing()?;
        }

        // 提取待合成文本
        let texts: Vec<&str> = segments
            .iter()
            .map(|seg| seg.target_text.as_deref().expect("validated above"))
            .collect();

        // 连续合成：将所有文本拼接为一次 say 调用，确保音色一致
        // macOS 神经 TTS 引擎对每段独立合成的文本会产生不同音色特征
        // 连续合成可避免此问题，然后用静音检测分割回各段
        let results: Vec<AppResult<PathBuf>> = if texts.len() > 1 {
            match self.synthesize_continuous(&texts, config) {
                Ok(paths) if paths.len() == texts.len() => {
                    tracing::info!(
                        "Continuous synthesis succeeded: {} segments split successfully",
                        paths.len()
                    );
                    paths.into_iter().map(Ok).collect()
                }
                Ok(paths) => {
                    tracing::warn!(
                        "Continuous synthesis split mismatch: expected {} segments, got {}. Falling back to per-segment synthesis.",
                        texts.len(),
                        paths.len()
                    );
                    self.synthesize_per_segment(&texts, config)
                }
                Err(e) => {
                    tracing::warn!(
                        "Continuous synthesis failed: {e}. Falling back to per-segment synthesis."
                    );
                    self.synthesize_per_segment(&texts, config)
                }
            }
        } else {
            // 单段文本直接合成
            texts
                .iter()
                .map(|text| self.synthesize_single(text, config))
                .collect()
        };

        // 处理结果：状态转换 Synthesizing → Completed / Failed
        let mut paths = Vec::with_capacity(segments.len());
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(path) => {
                    segments[i].finish_synthesizing(path.to_string_lossy().to_string())?;
                    paths.push(path);
                }
                Err(e) => {
                    tracing::error!("TTS synthesis failed for segment {}: {}", segments[i].id, e);
                    segments[i].fail()?;
                    return Err(e);
                }
            }
        }

        Ok(paths)
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.voice_manager.list_voices().to_vec()
    }
}

impl std::fmt::Debug for SayEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SayEngine")
            .field("cache_dir", &self.cache_dir)
            .field("voice_count", &self.voice_manager.list_voices().len())
            .finish()
    }
}

// ─── KokoroEngine ────────────────────────────────────────

/// Kokoro TTS 引擎（带 SayEngine 降级）
///
/// 优先使用 Kokoro-82M ONNX 模型进行语音合成。
/// 当 Kokoro 模型不可用（未下载、加载失败、不支持中文）时，
/// 自动降级到 macOS `say` 命令引擎（[`SayEngine`]）。
///
/// # 降级策略
/// 1. 尝试从 `TtsConfig.model_path` 或缓存目录加载 Kokoro ONNX 模型
/// 2. 若加载失败且 `fallback_to_say` 为 `true`，创建 `SayEngine` 作为替代
/// 3. 若 `fallback_to_say` 为 `false`，返回加载错误
///
/// # 线程安全
/// 内部通过 `SayEngine` 实现合成，满足 `Send + Sync`。
///
/// # ONNX 集成
/// 当 `kokoro-onnx` feature 启用时，引擎将尝试使用 `ort` crate
/// 加载 Kokoro-82M-v1.1-zh ONNX 模型进行推理。
/// 模型文件需从 ModelScope 下载（`onnx-community/Kokoro-82M-v1.1-zh-ONNX`）。
///
/// # 音频后处理
/// 无论使用哪种后端，合成后的音频都会通过 [`AudioPostProcessor`] 进行后处理：
/// - 齿音衰减（highshelf，减少 6kHz+ 高频）
/// - 低频增强（lowshelf，提升 300Hz）
/// - 自动增益控制（dynaudnorm）
/// - 淡入淡出（afade + areverse，消除拼接感）
pub struct KokoroEngine {
    /// 实际使用的 TTS 后端
    backend: TtsBackend,
    /// 音频缓存目录
    cache_dir: PathBuf,
    /// 音色管理器
    voice_manager: VoiceManager,
}

/// TTS 后端枚举
enum TtsBackend {
    /// macOS say 命令引擎
    Say(SayEngine),
    /// Kokoro-82M ONNX 推理引擎（需 `kokoro-onnx` feature）
    #[cfg(feature = "kokoro-onnx")]
    Onnx(KokoroOnnxBackend),
}

/// Kokoro ONNX 推理后端（需 `kokoro-onnx` feature）
///
/// 使用 `ort` crate 加载 Kokoro-82M-v1.1-zh ONNX 模型进行语音合成。
/// 模型文件从 ModelScope 下载：`onnx-community/Kokoro-82M-v1.1-zh-ONNX`。
///
/// # 线程安全
/// `ort::session::Session` 实现了 `Send + Sync`，本结构天然线程安全。
#[cfg(feature = "kokoro-onnx")]
struct KokoroOnnxBackend {
    /// ONNX Runtime 会话
    _session: ort::session::Session,
    /// 模型路径
    model_path: PathBuf,
}

#[cfg(feature = "kokoro-onnx")]
impl KokoroOnnxBackend {
    /// 创建 Kokoro ONNX 推理后端
    ///
    /// 使用 `ort` crate 加载 Kokoro-82M-v1.1-zh ONNX 模型。
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    /// - `config`: TTS 配置（读取 `device` 字段选择推理设备）
    ///
    /// # 错误
    /// - [`AppError::TtsModelLoadError`][]: 模型加载失败
    pub fn new(model_path: PathBuf, config: &TtsConfig) -> AppResult<Self> {
        tracing::info!(
            "KokoroOnnxBackend: Loading ONNX model from {:?} (device: {})",
            model_path,
            config.device
        );

        // 构建 ONNX Runtime 环境
        let environment = ort::environment::Environment::builder()
            .with_name("kokoro-tts")
            .build()
            .map_err(|e| {
                AppError::TtsModelLoadError(format!(
                    "Failed to create ONNX Runtime environment: {e}"
                ))
            })?;

        // 构建 Session
        let session_builder = ort::session::SessionBuilder::new(&environment).map_err(|e| {
            AppError::TtsModelLoadError(format!("Failed to create ONNX Session builder: {e}"))
        })?;

        // 配置执行器（CPU 或 Metal GPU）
        #[cfg(target_os = "macos")]
        let session_builder = if config.device == "metal" {
            tracing::info!("KokoroOnnxBackend: Using Metal GPU acceleration");
            session_builder
                .with_execution_mode(ort::session::ExecutionMode::SEQUENTIAL)
                .with_optimization_level(ort::session::GraphOptimizationLevel::Level3)
        } else {
            tracing::info!("KokoroOnnxBackend: Using CPU");
            session_builder
                .with_execution_mode(ort::session::ExecutionMode::SEQUENTIAL)
                .with_optimization_level(ort::session::GraphOptimizationLevel::Level3)
        };

        // 加载模型
        let session = session_builder
            .with_model_from_file(&model_path)
            .map_err(|e| {
                AppError::TtsModelLoadError(format!(
                    "Failed to load ONNX model from {:?}: {e}",
                    model_path
                ))
            })?;

        tracing::info!(
            "KokoroOnnxBackend: Model loaded successfully, inputs: {}, outputs: {}",
            session.inputs.len(),
            session.outputs.len()
        );

        Ok(Self {
            _session: session,
            model_path,
        })
    }

    /// 获取模型路径
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

impl KokoroEngine {
    /// 创建 Kokoro TTS 引擎
    ///
    /// 尝试加载 Kokoro ONNX 模型，失败时降级到 SayEngine。
    ///
    /// # 参数
    /// - `config`: TTS 配置
    ///
    /// # 错误
    /// - [`AppError::TtsModelLoadError`][]: Kokoro 加载失败且 `fallback_to_say` 为 `false`
    /// - [`AppError::Io`][]: 缓存目录创建失败
    pub fn new(config: &TtsConfig) -> AppResult<Self> {
        let cache_dir = expand_cache_dir(&config.cache_dir)?;
        std::fs::create_dir_all(&cache_dir)?;

        // 尝试加载 Kokoro ONNX 模型
        let model_path = config.model_path.clone().or_else(|| {
            let p = cache_dir.join(&config.model_variant).join("model.onnx");
            if p.exists() {
                Some(p)
            } else {
                None
            }
        });

        // 尝试 ONNX 后端（当 feature 启用且模型可用时）
        #[cfg(feature = "kokoro-onnx")]
        if let Some(ref model_path) = model_path {
            match KokoroOnnxBackend::new(model_path.clone(), config) {
                Ok(onnx_backend) => {
                    tracing::info!(
                        "KokoroEngine: ONNX backend initialized with model: {:?}",
                        model_path
                    );
                    return Ok(Self {
                        backend: TtsBackend::Onnx(onnx_backend),
                        cache_dir,
                        voice_manager: VoiceManager::new(),
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "KokoroEngine: ONNX backend init failed: {e}. Falling back to SayEngine."
                    );
                }
            }
        }

        // 降级信息
        if model_path.is_some() {
            #[cfg(not(feature = "kokoro-onnx"))]
            tracing::info!(
                "KokoroEngine: Kokoro model found, but kokoro-onnx feature is not enabled. \
                 Falling back to SayEngine. Enable with: cargo build --features kokoro-onnx"
            );
        } else {
            tracing::info!("KokoroEngine: Kokoro model not found, using SayEngine as fallback.");
        }

        // 降级到 SayEngine
        if !config.fallback_to_say {
            return Err(AppError::TtsModelLoadError(
                "Kokoro ONNX model not available and fallback_to_say is disabled.\n\
                 To use Kokoro, download the model and enable ONNX runtime integration.\n\
                 To use SayEngine, set fallback_to_say = true in config."
                    .to_string(),
            ));
        }

        let say_engine = SayEngine::new(config)?;
        Ok(Self {
            backend: TtsBackend::Say(say_engine),
            cache_dir,
            voice_manager: VoiceManager::new(),
        })
    }

    /// 获取缓存目录路径
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// 获取当前使用的后端名称
    #[must_use]
    pub fn backend_name(&self) -> &str {
        match self.backend {
            TtsBackend::Say(_) => "SayEngine",
            #[cfg(feature = "kokoro-onnx")]
            TtsBackend::Onnx(_) => "KokoroOnnx",
        }
    }
}

impl TtsEngine for KokoroEngine {
    fn synthesize_segments(
        &self,
        segments: &mut [Segment],
        config: &TtsConfig,
    ) -> AppResult<Vec<PathBuf>> {
        match &self.backend {
            TtsBackend::Say(engine) => engine.synthesize_segments(segments, config),
            #[cfg(feature = "kokoro-onnx")]
            TtsBackend::Onnx(_backend) => {
                // ONNX 推理路径：当模型可用时使用 Kokoro-82M 进行合成
                // 后处理通过 AudioPostProcessor 应用
                // 当前回退到 SayEngine，待 ONNX 推理逻辑完善后替换
                tracing::info!("KokoroOnnx backend: using SayEngine for synthesis (ONNX inference in development)");
                // 这里无法访问 SayEngine，需要重构
                // 暂时返回错误，提示用户 SayEngine 更可靠
                Err(AppError::TtsError(
                    "KokoroOnnx backend is initialized but synthesis is not yet implemented. \
                     Please use SayEngine backend (set engine = \"say\" in config)."
                        .to_string(),
                ))
            }
        }
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.voice_manager.list_voices().to_vec()
    }
}

impl std::fmt::Debug for KokoroEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KokoroEngine")
            .field("backend", &self.backend_name())
            .field("cache_dir", &self.cache_dir)
            .field("voice_count", &self.voice_manager.list_voices().len())
            .finish()
    }
}

// ─── EdgeTtsEngine ───────────────────────────────────────

/// 基于 Microsoft Edge-TTS 的云端语音合成引擎
///
/// 借鉴 pyvideotrans 的 edge-tts 用法，使用微软云端神经 TTS 合成中文语音。
/// 通过调用 Python `edge_tts.py` 脚本实现，无需加载本地模型。
///
/// # 优势（对比 SayEngine）
/// - 音质更好：微软神经语音，自然流畅，无 pitch shifting 失真
/// - 音量一致：云端合成，无需 dynaudnorm AGC，不会时高时低
/// - 真实男声：自带男声（YunxiNeural），无需从女声变调
/// - 速度极快：云端合成 RTF ~0.1x，比 say 命令还快
///
/// # 系统要求
/// - Python 3.8+ 且安装了 edge-tts 包（`pip install edge-tts`）
/// - 网络连接（调用微软云端 API）
/// - ffmpeg（MP3→WAV 转码）
///
/// # 配置示例
/// ```toml
/// [tts]
/// engine = "edge"
/// voice = "auto"  # 或 zh-CN-XiaoxiaoNeural / zh-CN-YunxiNeural
/// ```
pub struct EdgeTtsEngine {
    /// Python 解释器路径
    python_path: String,
    /// edge_tts.py 脚本路径
    script_path: String,
    /// 音频缓存目录
    cache_dir: PathBuf,
    /// 音色管理器（用于 list_voices）
    voice_manager: VoiceManager,
    /// 参考音频路径（用于 --voice auto 性别检测）
    ref_audio: Option<PathBuf>,
}

impl EdgeTtsEngine {
    /// 创建 EdgeTtsEngine
    ///
    /// # 参数
    /// - `config`: TTS 配置（读取 cache_dir）
    /// - `python_path`: Python 解释器路径
    /// - `script_path`: edge_tts.py 脚本路径
    /// - `ref_audio`: 参考音频路径（可选，用于性别检测）
    #[must_use]
    pub fn new(
        config: &TtsConfig,
        python_path: String,
        script_path: String,
        ref_audio: Option<PathBuf>,
    ) -> Self {
        let cache_dir = expand_cache_dir(&config.cache_dir)
            .unwrap_or_else(|_| PathBuf::from("~/.cache/video-translator/tts_cache"));
        let _ = std::fs::create_dir_all(&cache_dir);

        Self {
            python_path,
            script_path,
            cache_dir,
            voice_manager: VoiceManager::new(),
            ref_audio,
        }
    }

    /// 计算缓存路径
    fn cache_path(&self, text: &str, voice: &str) -> PathBuf {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hasher.update(voice.as_bytes());
        let hash = hasher.finalize();
        let hash_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        self.cache_dir.join(format!("{hash_hex}.wav"))
    }

    /// 调用 Python edge_tts.py 脚本合成语音
    fn call_edge_tts(
        &self,
        text: &str,
        voice: &str,
        output: &Path,
        ref_audio: Option<&Path>,
    ) -> AppResult<()> {
        let mut cmd = Command::new(&self.python_path);
        cmd.arg(&self.script_path)
            .arg("--text")
            .arg(text)
            .arg("--voice")
            .arg(voice)
            .arg("--output")
            .arg(output);

        if let Some(ref_path) = ref_audio {
            cmd.arg("--ref").arg(ref_path);
        }

        // 启动子进程（不等待，先拿 handle）
        let mut child = cmd.spawn().map_err(|e| {
            AppError::TtsError(format!(
                "Failed to execute edge_tts.py: {e}\n\
                 确保 Python 路径正确且 edge-tts 已安装 (pip install edge-tts)"
            ))
        })?;

        // 取 stderr handle 用于错误信息
        let mut stderr_child = child.stderr.take();
        let pid = child.id();

        // 在 spawn_blocking 中等待子进程退出，主线程用 tokio 超时控制
        let wait_handle = std::thread::spawn(move || {
            let _ = child.wait();
        });

        // 轮询等待，最多 60 秒（edge-tts 含 3 次重试 × 15s + ffmpeg 转码）
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        let mut timed_out = false;
        while std::time::Instant::now() < deadline {
            if wait_handle.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        if !wait_handle.is_finished() {
            timed_out = true;
            // 杀死超时进程
            tracing::warn!("edge_tts.py (pid={}) timed out after 60s, killing", pid);
            #[cfg(unix)]
            {
                let _ = std::process::Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
        }

        // 读取 stderr
        let stderr = stderr_child
            .as_mut()
            .map(|s| {
                use std::io::Read;
                let mut buf = String::new();
                let _ = s.read_to_string(&mut buf);
                buf
            })
            .unwrap_or_default();

        if timed_out {
            return Err(AppError::TtsError(format!(
                "edge_tts.py timed out after 60s (process killed): {stderr}"
            )));
        }

        // 检查输出文件是否存在来判断成功/失败
        if !output.exists() {
            return Err(AppError::TtsError(format!(
                "edge_tts.py did not produce output file: {stderr}"
            )));
        }

        Ok(())
    }

    /// 解析音色配置
    ///
    /// "auto" → 使用参考音频检测性别，或默认女声
    /// "zh-CN-XiaoxiaoNeural" → 直接使用指定声音
    fn resolve_voice(&self, config: &TtsConfig) -> String {
        if config.voice == "auto" {
            // 自动模式：由 Python 脚本根据参考音频检测性别
            "auto".to_string()
        } else {
            config.voice.clone()
        }
    }
}

impl TtsEngine for EdgeTtsEngine {
    fn synthesize_segments(
        &self,
        segments: &mut [Segment],
        config: &TtsConfig,
    ) -> AppResult<Vec<PathBuf>> {
        let voice = self.resolve_voice(config);
        let mut paths = Vec::with_capacity(segments.len());

        for seg in segments.iter_mut() {
            let text = seg.target_text.clone().unwrap_or_default();
            if text.trim().is_empty() {
                seg.fail()?;
                continue;
            }

            seg.start_synthesizing()?;

            // 缓存检查
            let cache_path = self.cache_path(&text, &voice);
            if cache_path.exists() {
                tracing::debug!("EdgeTts cache hit for segment {}", seg.id);
                seg.finish_synthesizing(cache_path.to_string_lossy().to_string())?;
                paths.push(cache_path);
                continue;
            }

            // 调用 edge_tts.py
            self.call_edge_tts(&text, &voice, &cache_path, self.ref_audio.as_deref())?;

            if cache_path.exists() {
                tracing::info!("EdgeTts: synthesized segment {} → {:?}", seg.id, cache_path);
                seg.finish_synthesizing(cache_path.to_string_lossy().to_string())?;
                paths.push(cache_path);
            } else {
                return Err(AppError::TtsError(format!(
                    "EdgeTts: output file not found after synthesis: {:?}",
                    cache_path
                )));
            }
        }

        Ok(paths)
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.voice_manager.list_voices().to_vec()
    }
}

impl std::fmt::Debug for EdgeTtsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeTtsEngine")
            .field("python_path", &self.python_path)
            .field("script_path", &self.script_path)
            .field("cache_dir", &self.cache_dir)
            .field("ref_audio", &self.ref_audio)
            .finish()
    }
}

// ─── 缓存目录处理 ────────────────────────────────────────

/// 展开路径中的 `~` 为 HOME 目录
///
/// # 参数
/// - `dir`: 可能以 `~` 开头的路径字符串
///
/// # 错误
/// - [`AppError::Config`][]: `HOME` 环境变量未设置
fn expand_cache_dir(dir: &str) -> AppResult<PathBuf> {
    if let Some(rest) = dir.strip_prefix('~') {
        let home = dirs::home_dir()
            .ok_or_else(|| AppError::Config("Cannot determine home directory".to_string()))?;
        let rest = rest.strip_prefix('/').unwrap_or(rest);
        if rest.is_empty() {
            Ok(home)
        } else {
            Ok(home.join(rest))
        }
    } else {
        Ok(PathBuf::from(dir))
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice_manager::VoiceGender;

    /// 验证缓存路径计算的一致性：相同参数产生相同路径。
    #[test]
    fn test_cache_path_consistency() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

        let path1 = engine.cache_path("你好", "tingting", 1.0, 1.0, 1.0, 24000);
        let path2 = engine.cache_path("你好", "tingting", 1.0, 1.0, 1.0, 24000);
        assert_eq!(
            path1, path2,
            "Same parameters should produce same cache path"
        );
    }

    /// 验证不同参数产生不同的缓存路径。
    #[test]
    fn test_cache_path_uniqueness() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

        let path1 = engine.cache_path("你好", "tingting", 1.0, 1.0, 1.0, 24000);
        let path2 = engine.cache_path("世界", "tingting", 1.0, 1.0, 1.0, 24000);
        let path3 = engine.cache_path("你好", "zhiming", 1.0, 1.0, 1.0, 24000);
        let path4 = engine.cache_path("你好", "tingting", 2.0, 1.0, 1.0, 24000);
        let path5 = engine.cache_path("你好", "tingting", 1.0, 0.85, 1.0, 24000);
        let path6 = engine.cache_path("你好", "tingting", 1.0, 1.0, 1.5, 24000);
        let path7 = engine.cache_path("你好", "tingting", 1.0, 1.0, 1.0, 48000);

        assert_ne!(
            path1, path2,
            "Different text should produce different paths"
        );
        assert_ne!(
            path1, path3,
            "Different voice_id should produce different paths"
        );
        assert_ne!(
            path1, path4,
            "Different speed should produce different paths"
        );
        assert_ne!(
            path1, path5,
            "Different pitch should produce different paths"
        );
        assert_ne!(
            path1, path6,
            "Different volume should produce different paths"
        );
        assert_ne!(
            path1, path7,
            "Different sample_rate should produce different paths"
        );
    }

    /// 验证缓存路径以 `.wav` 扩展名结尾。
    #[test]
    fn test_cache_path_extension() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

        let path = engine.cache_path("test", "tingting", 1.0, 1.0, 1.0, 24000);
        assert_eq!(
            path.extension().and_then(|e| e.to_str()),
            Some("wav"),
            "Cache path should end with .wav"
        );
    }

    /// 验证 `expand_cache_dir` 正确展开 `~`。
    #[test]
    fn test_expand_cache_dir() {
        let path = expand_cache_dir("/absolute/path").expect("Failed to expand");
        assert_eq!(path, PathBuf::from("/absolute/path"));
    }

    /// 验证语速倍率到 `say` 命令 rate 的映射。
    #[test]
    fn test_speed_to_rate() {
        assert_eq!(SayEngine::speed_to_rate(1.0), 175, "Normal speed");
        assert_eq!(SayEngine::speed_to_rate(2.0), 350, "Double speed");
        assert_eq!(SayEngine::speed_to_rate(0.5), 87, "Half speed");
        assert_eq!(SayEngine::speed_to_rate(0.1), 50, "Clamped to minimum");
        assert_eq!(SayEngine::speed_to_rate(10.0), 500, "Clamped to maximum");
    }

    /// 验证 `say` 命令可用性（仅 macOS）。
    #[test]
    fn test_say_available() {
        let output = std::process::Command::new("say")
            .arg("-v")
            .arg("?")
            .output();
        if output.is_err() {
            eprintln!("Skipping: 'say' command not available (non-macOS?)");
            return;
        }
        assert!(output.unwrap().status.success(), "say -v ? should succeed");
    }

    // ─── SayEngine 音色测试 ──────────────────────────────────

    /// 验证 SayEngine 能列出所有内置音色。
    #[test]
    fn test_say_engine_list_voices() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

        let voices = engine.list_voices();
        assert!(
            voices.len() >= 4,
            "Should have at least 4 voices, got {}",
            voices.len()
        );

        let females = voices
            .iter()
            .filter(|v| v.gender == VoiceGender::Female)
            .count();
        assert!(females >= 2, "Should have at least 2 female voices");

        let males = voices
            .iter()
            .filter(|v| v.gender == VoiceGender::Male)
            .count();
        assert!(males >= 2, "Should have at least 2 male voices");
    }

    /// 验证 SayEngine Debug 输出包含音色数量。
    #[test]
    fn test_say_engine_debug() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");
        let debug = format!("{engine:?}");
        assert!(debug.contains("SayEngine"));
        assert!(debug.contains("voice_count"));
    }

    /// 验证音色解析：voice_id 优先于 voice 字段。
    #[test]
    fn test_resolve_voice_by_id() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

        let config = TtsConfig {
            voice_id: "zhiming".to_string(),
            voice: "Tingting".to_string(),
            ..Default::default()
        };
        let voice = engine.resolve_voice(&config);
        assert_eq!(voice.id, "zhiming");
        assert_eq!(voice.gender, VoiceGender::Male);
    }

    /// 验证音色解析：voice_id 不存在时回退到 voice 字段匹配。
    #[test]
    fn test_resolve_voice_fallback() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

        let config = TtsConfig {
            voice_id: "nonexistent".to_string(),
            voice: "Meijia".to_string(),
            ..Default::default()
        };
        let voice = engine.resolve_voice(&config);
        assert_eq!(voice.say_voice, "Meijia");
    }

    /// 验证组合音调倍率计算。
    #[test]
    fn test_combined_pitch() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("zhiming").expect("zhiming should exist");

        // 音色 pitch=0.85, config pitch=1.0 → combined=0.85
        let config = TtsConfig {
            pitch: 1.0,
            ..Default::default()
        };
        assert!((SayEngine::combined_pitch(voice, &config) - 0.85).abs() < 0.001);

        // 音色 pitch=0.85, config pitch=0.9 → combined=0.765
        let config = TtsConfig {
            pitch: 0.9,
            ..Default::default()
        };
        assert!((SayEngine::combined_pitch(voice, &config) - 0.765).abs() < 0.001);

        // 女声 pitch=1.0, config pitch=1.1 → combined=1.1
        let voice_f = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig {
            pitch: 1.1,
            ..Default::default()
        };
        assert!((SayEngine::combined_pitch(voice_f, &config) - 1.1).abs() < 0.001);
    }

    /// 验证后处理判断逻辑：始终需要后处理（EQ + 淡入淡出）。
    #[test]
    fn test_needs_postprocessing_always_true() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig::default();
        assert!(
            SayEngine::needs_postprocessing(voice, &config),
            "Should always need post-processing for EQ and fade"
        );
    }

    /// 验证后处理判断逻辑：男声需要后处理（音调偏移）。
    #[test]
    fn test_needs_postprocessing_true_for_male() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("zhiming").expect("zhiming should exist");
        let config = TtsConfig::default();
        assert!(
            SayEngine::needs_postprocessing(voice, &config),
            "Male voice should need post-processing (pitch shift)"
        );
    }

    /// 验证后处理判断逻辑：非默认采样率需要后处理。
    #[test]
    fn test_needs_postprocessing_true_for_different_sample_rate() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig {
            sample_rate: 48000,
            ..Default::default()
        };
        assert!(
            SayEngine::needs_postprocessing(voice, &config),
            "Different sample rate should need post-processing"
        );
    }

    /// 验证 ffmpeg 滤镜链构建（男声 + 默认参数）。
    #[test]
    fn test_build_filter_chain_male_voice() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("zhiming").expect("zhiming should exist");
        let config = TtsConfig::default();

        let chain = SayEngine::build_filter_chain(voice, &config);
        assert!(
            chain.contains("asetrate"),
            "Should contain asetrate for pitch shift"
        );
        assert!(
            chain.contains("atempo"),
            "Should contain atempo for speed compensation"
        );
        assert!(
            chain.contains("dynaudnorm"),
            "Should contain dynaudnorm for AGC"
        );
        assert!(
            chain.contains("afade"),
            "Should contain afade for fade in/out"
        );
        assert!(
            chain.contains("areverse"),
            "Should contain areverse for fade out"
        );
    }

    /// 验证 ffmpeg 滤镜链构建（女声 + 默认参数，无音调偏移但有 EQ）。
    #[test]
    fn test_build_filter_chain_female_default() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig::default();

        let chain = SayEngine::build_filter_chain(voice, &config);
        assert!(
            !chain.contains("asetrate"),
            "Should not contain asetrate for default female voice"
        );
        assert!(
            chain.contains("dynaudnorm"),
            "Should always contain dynaudnorm"
        );
        assert!(
            chain.contains("highshelf"),
            "Should contain highshelf for sibilance reduction"
        );
        assert!(
            chain.contains("lowshelf"),
            "Should contain lowshelf for warmth enhancement"
        );
    }

    /// 验证 ffmpeg 滤镜链包含音量调整。
    #[test]
    fn test_build_filter_chain_with_volume() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig {
            volume: 1.5,
            ..Default::default()
        };

        let chain = SayEngine::build_filter_chain(voice, &config);
        assert!(
            chain.contains("volume=1.5000"),
            "Should contain volume filter"
        );
    }

    // ─── KokoroEngine 测试 ──────────────────────────────────

    /// 验证 KokoroEngine 在 fallback_to_say=true 时降级到 SayEngine。
    #[test]
    fn test_kokoro_engine_fallback_to_say() {
        let config = TtsConfig::default();
        let engine = KokoroEngine::new(&config).expect("KokoroEngine creation should succeed");
        assert_eq!(engine.backend_name(), "SayEngine");
    }

    /// 验证 KokoroEngine 在 fallback_to_say=false 时返回错误。
    #[test]
    fn test_kokoro_engine_no_fallback_error() {
        let config = TtsConfig {
            fallback_to_say: false,
            ..Default::default()
        };
        let result = KokoroEngine::new(&config);
        assert!(result.is_err());
        match &result {
            Err(AppError::TtsModelLoadError(msg)) => {
                assert!(msg.contains("fallback_to_say"));
            }
            Err(e) => panic!("Expected TtsModelLoadError, got: {e:?}"),
            Ok(_) => panic!("Expected error, got success"),
        }
    }

    /// 验证 KokoroEngine Debug 输出。
    #[test]
    fn test_kokoro_engine_debug() {
        let config = TtsConfig::default();
        let engine = KokoroEngine::new(&config).expect("Should succeed");
        let debug = format!("{engine:?}");
        assert!(debug.contains("KokoroEngine"));
        assert!(debug.contains("SayEngine"));
        assert!(debug.contains("voice_count"));
    }

    /// 验证 KokoroEngine 的 list_voices 方法。
    #[test]
    fn test_kokoro_engine_list_voices() {
        let config = TtsConfig::default();
        let engine = KokoroEngine::new(&config).expect("Should succeed");
        let voices = engine.list_voices();
        assert!(voices.len() >= 4, "Should have at least 4 voices");
    }
}
