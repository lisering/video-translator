//! 参考音频自动提取模块
//!
//! 从视频音频中自动识别并提取主要说话人的干净语音片段（3–10 秒），
//! 作为声音克隆的参考音频。
//!
//! # 功能概览
//! - [`VoiceExtractor`][]：参考音频提取器，整合人声增强、静音修剪和音量归一化
//! - [`ReferenceAudio`][]：提取结果，包含路径、时长、说话人 ID、采样率和提示文本
//! - [`VoiceExtractorConfig`][]：提取配置（见 [`config`](crate::config) 模块）
//!
//! # 工作流程
//! 1. **片段选择**：根据 ASR Segment 列表，选择时长最接近理想值（默认 5 秒）且文本非空的片段
//! 2. **音频截取**：从完整 WAV 中按时间戳截取对应音频段
//! 3. **人声增强**（可选）：使用 ffmpeg 高通+低通滤波去除低频噪声和高频杂音
//! 4. **静音修剪**（可选）：检测并裁剪音频首尾的静音段
//! 5. **音量归一化**（可选）：将音频 RMS 电平调整到目标值（默认 -20dBFS）
//!
//! # 性能要求
//! - 提取 + 增强 ≤ 1 秒（不含 ffmpeg 增强时）
//! - ffmpeg 增强增加约 0.5 秒
//!
//! # 示例
//! ```no_run
//! use vt_core::voice_extractor::VoiceExtractor;
//! use vt_core::config::VoiceExtractorConfig;
//! use vt_core::models::segment::Segment;
//! use std::path::Path;
//!
//! let extractor = VoiceExtractor::new(VoiceExtractorConfig::default());
//! let segments = vec![
//!     Segment::new("seg-1".into(), 2.0, 7.0, "Hello world".into()),
//! ];
//! let result = extractor.extract_reference_audio(
//!     Path::new("full_audio.wav"),
//!     &segments,
//!     Path::new("reference.wav"),
//! ).expect("extraction failed");
//! ```

use std::path::{Path, PathBuf};

use crate::asr::read_wav_mono;
use crate::config::VoiceExtractorConfig;
use crate::error::{AppError, AppResult};
use crate::models::segment::Segment;

// ─── 参考音频结果 ─────────────────────────────────────────

/// 提取的参考音频
///
/// 包含提取的参考音频文件路径和元数据。
#[derive(Debug, Clone)]
pub struct ReferenceAudio {
    /// 参考音频文件路径
    pub path: PathBuf,
    /// 音频时长（秒）
    pub duration_secs: f64,
    /// 说话人标识（如果有说话人分离信息）
    pub speaker_id: Option<String>,
    /// 采样率
    pub sample_rate: u32,
    /// 参考音频对应的提示文本（用于零样本克隆）
    pub prompt_text: String,
}

impl ReferenceAudio {
    /// 创建新的参考音频
    #[must_use]
    pub fn new(path: PathBuf, duration_secs: f64, sample_rate: u32, prompt_text: String) -> Self {
        Self {
            path,
            duration_secs,
            speaker_id: None,
            sample_rate,
            prompt_text,
        }
    }

    /// 设置说话人标识
    #[must_use]
    pub fn with_speaker_id(mut self, speaker_id: impl Into<String>) -> Self {
        self.speaker_id = Some(speaker_id.into());
        self
    }
}

// ─── 参考音频提取器 ───────────────────────────────────────

/// 参考音频提取器
///
/// 从完整音频中自动提取适合声音克隆的参考音频片段。
/// 支持人声增强、静音修剪和音量归一化。
///
/// # 配置
/// 通过 [`VoiceExtractorConfig`] 配置提取行为：
/// - `enable_enhancement`: 是否使用 ffmpeg 人声增强
/// - `enable_silence_trim`: 是否修剪首尾静音
/// - `enable_normalization`: 是否归一化音量
///
/// # 示例
/// ```no_run
/// use vt_core::voice_extractor::VoiceExtractor;
/// use vt_core::config::VoiceExtractorConfig;
///
/// let config = VoiceExtractorConfig {
///     enable_enhancement: true,
///     enable_silence_trim: true,
///     ..Default::default()
/// };
/// let extractor = VoiceExtractor::new(config);
/// ```
pub struct VoiceExtractor {
    /// 提取配置
    config: VoiceExtractorConfig,
}

impl VoiceExtractor {
    /// 创建新的参考音频提取器
    ///
    /// # 参数
    /// - `config`: 提取配置
    #[must_use]
    pub fn new(config: VoiceExtractorConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建提取器
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(VoiceExtractorConfig::default())
    }

    /// 获取配置引用
    #[must_use]
    pub fn config(&self) -> &VoiceExtractorConfig {
        &self.config
    }

    /// 从完整音频中提取参考音频
    ///
    /// # 工作流程
    /// 1. 根据 ASR Segment 列表选择最佳片段（时长最接近理想值）
    /// 2. 从完整 WAV 中截取对应音频段
    /// 3. 依次执行人声增强、静音修剪、音量归一化
    /// 4. 返回 [`ReferenceAudio`] 结果
    ///
    /// # 参数
    /// - `full_wav_path`: 完整音频 WAV 文件路径（16kHz mono）
    /// - `segments`: ASR 产生的 Segment 列表
    /// - `output_path`: 输出参考音频文件路径
    ///
    /// # 返回
    /// - `Ok(Some(reference_audio))`: 成功提取
    /// - `Ok(None)`: 没有合适的片段
    ///
    /// # 错误
    /// - [`AppError::AudioDecodeError`][]: WAV 读取失败
    /// - [`AppError::VoiceCloningError`][]: 音频处理失败
    pub fn extract_reference_audio(
        &self,
        full_wav_path: &Path,
        segments: &[Segment],
        output_path: &Path,
    ) -> AppResult<Option<ReferenceAudio>> {
        if segments.is_empty() {
            tracing::warn!("VoiceExtractor: no segments available");
            return Ok(None);
        }

        // 选择最佳参考片段
        let best = self.select_best_segment(segments);
        let Some(segment) = best else {
            tracing::warn!(
                "VoiceExtractor: no suitable segment found \
                (need {:.1}–{:.1}s speech with text)",
                self.config.min_duration_secs,
                self.config.max_duration_secs
            );
            return Ok(None);
        };

        let prompt_text = segment.source_text.clone();
        let start = segment.start;
        let end = segment.end;
        let duration = end - start;

        tracing::info!(
            "VoiceExtractor: selected segment {} ({:.1}s–{:.1}s, {:.1}s), prompt: \"{}\"",
            segment.id,
            start,
            end,
            duration,
            if prompt_text.len() > 60 {
                format!("{}...", prompt_text.chars().take(60).collect::<String>())
            } else {
                prompt_text.clone()
            }
        );

        // 读取完整音频
        let (samples, sample_rate) = read_wav_mono(full_wav_path)?;

        // 截取目标段
        let start_sample = ((start * sample_rate as f64) as usize).min(samples.len());
        let end_sample = ((end * sample_rate as f64) as usize).min(samples.len());

        if start_sample >= end_sample {
            tracing::warn!(
                "VoiceExtractor: invalid sample range {}..{} for segment {}",
                start_sample,
                end_sample,
                segment.id
            );
            return Ok(None);
        }

        let mut ref_samples = samples[start_sample..end_sample].to_vec();

        // 确保输出目录存在
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to create reference dir: {e}"))
            })?;
        }

        // 静音修剪
        if self.config.enable_silence_trim {
            let original_len = ref_samples.len();
            self.trim_silence(&mut ref_samples, sample_rate);
            if ref_samples.len() < original_len {
                tracing::debug!(
                    "VoiceExtractor: trimmed silence ({} → {} samples, {:.1}s → {:.1}s)",
                    original_len,
                    ref_samples.len(),
                    original_len as f64 / sample_rate as f64,
                    ref_samples.len() as f64 / sample_rate as f64
                );
            }
        }

        // ── 硬校验：确保提取后音频满足最低要求 ──
        // 静音修剪可能导致音频过短，需要验证
        let post_trim_duration = ref_samples.len() as f64 / sample_rate as f64;
        if post_trim_duration < 1.0 {
            tracing::warn!(
                "VoiceExtractor: reference audio too short after trim ({:.1}s < 1.0s), \
                skipping silence trim and using original segment",
                post_trim_duration
            );
            // 重新截取原始段（不修剪静音）
            ref_samples = samples[start_sample..end_sample].to_vec();
            // 仍然归一化
            if self.config.enable_normalization {
                self.normalize_rms(&mut ref_samples);
            }
        }

        // 验证音频能量（非全静音）
        let audio_rms = rms(&ref_samples);
        if audio_rms < 0.001 {
            tracing::warn!(
                "VoiceExtractor: reference audio RMS too low ({:.6}), \
                may be silence or noise — cloning quality may degrade",
                audio_rms
            );
        }

        // 验证音频时长上限（避免过长的参考导致引擎超时）
        let post_trim_duration = ref_samples.len() as f64 / sample_rate as f64;
        if post_trim_duration > 30.0 {
            tracing::warn!(
                "VoiceExtractor: reference audio too long ({:.1}s > 30.0s), \
                truncating to 30s",
                post_trim_duration
            );
            let max_samples = (30.0 * sample_rate as f64) as usize;
            ref_samples.truncate(max_samples);
        }

        // 音量归一化
        if self.config.enable_normalization {
            self.normalize_rms(&mut ref_samples);
            tracing::debug!(
                "VoiceExtractor: normalized to target RMS {:.1}dBFS",
                self.config.target_rms_db
            );
        }

        // ── 最终硬校验 ──
        // 确保有足够的采样数据
        let min_required_samples = (sample_rate as f64 * 0.5) as usize; // 至少 0.5 秒
        if ref_samples.len() < min_required_samples {
            tracing::warn!(
                "VoiceExtractor: reference audio has too few samples ({}, need at least {}), \
                cloning may fail",
                ref_samples.len(),
                min_required_samples
            );
        }

        // 写入 WAV 文件
        self.write_wav(output_path, &ref_samples, sample_rate)?;

        let final_duration = ref_samples.len() as f64 / sample_rate as f64;

        // 可选：ffmpeg 人声增强
        if self.config.enable_enhancement {
            if let Err(e) = self.enhance_with_ffmpeg(output_path) {
                tracing::warn!(
                    "VoiceExtractor: ffmpeg enhancement failed (using unenhanced audio): {}",
                    e
                );
                // 增强失败不影响整体流程，使用未增强的音频
            }
        }

        tracing::info!(
            "VoiceExtractor: reference audio saved to {:?} ({:.1}s, {}Hz)",
            output_path,
            final_duration,
            sample_rate
        );

        Ok(Some(ReferenceAudio::new(
            output_path.to_path_buf(),
            final_duration,
            sample_rate,
            prompt_text,
        )))
    }

    /// 选择最佳参考片段
    ///
    /// 根据时长和文本内容选择最适合的 Segment：
    /// 1. 优先选择时长在 min_duration_secs – max_duration_secs 范围内
    ///    且 source_text 非空的片段
    /// 2. 在候选中，选择时长最接近 ideal_duration_secs 的片段
    /// 3. 如果没有严格匹配的，放宽到 2–15 秒范围
    fn select_best_segment<'a>(&self, segments: &'a [Segment]) -> Option<&'a Segment> {
        let min_dur = self.config.min_duration_secs;
        let max_dur = self.config.max_duration_secs;
        let ideal_dur = self.config.ideal_duration_secs;

        // 严格范围：min_dur – max_dur
        let best = segments
            .iter()
            .filter(|s| {
                let dur = s.end - s.start;
                dur >= min_dur && dur <= max_dur && !s.source_text.trim().is_empty()
            })
            .min_by_key(|s| {
                let dur = s.end - s.start;
                ((dur - ideal_dur).abs() * 100.0) as u64
            });

        if best.is_some() {
            return best;
        }

        // 放宽范围：2 – 15 秒
        segments
            .iter()
            .filter(|s| {
                let dur = s.end - s.start;
                dur >= 2.0 && dur <= 15.0 && !s.source_text.trim().is_empty()
            })
            .min_by_key(|s| {
                let dur = s.end - s.start;
                ((dur - ideal_dur).abs() * 100.0) as u64
            })
    }

    /// 修剪首尾静音
    ///
    /// 检测音频开头和结尾的静音段并裁剪。
    /// 静音判断标准：振幅低于 `silence_threshold_db` 对应的幅度。
    fn trim_silence(&self, samples: &mut Vec<f32>, sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        let threshold_amp = db_to_amplitude(self.config.silence_threshold_db) as f32;
        let frame_size = (sample_rate as f64 * 0.02) as usize; // 20ms 帧
        if frame_size == 0 {
            return;
        }

        // 检测开头静音
        let mut start = 0;
        for (i, chunk) in samples.chunks(frame_size).enumerate() {
            let frame_rms = rms(chunk);
            if frame_rms > threshold_amp {
                start = i * frame_size;
                break;
            }
        }

        // 检测结尾静音
        let mut end = samples.len();
        for (i, chunk) in samples.chunks(frame_size).enumerate().rev() {
            let frame_rms = rms(chunk);
            if frame_rms > threshold_amp {
                end = (i + 1) * frame_size;
                break;
            }
        }

        if start >= end {
            // 整段都是静音，保留原始数据
            return;
        }

        // 保留开头 100ms 的缓冲（避免截掉有效语音的起始部分）
        let buffer = (sample_rate as f64 * 0.1) as usize;
        let start = start.saturating_sub(buffer);
        let end = (end + buffer).min(samples.len());

        *samples = samples[start..end].to_vec();
    }

    /// RMS 音量归一化
    ///
    /// 将音频的 RMS 电平调整到目标值。
    ///
    /// # 算法
    /// 1. 计算当前 RMS
    /// 2. 计算目标 RMS
    /// 3. 计算增益系数 = target_rms / current_rms
    /// 4. 对所有样本乘以增益系数
    /// 5. 限幅（clamp 到 [-1.0, 1.0]）
    fn normalize_rms(&self, samples: &mut Vec<f32>) {
        if samples.is_empty() {
            return;
        }

        let current_rms = rms(samples);
        if current_rms < f32::EPSILON {
            // 全静音，跳过
            return;
        }

        let target_rms = db_to_amplitude(self.config.target_rms_db) as f32;
        let gain = target_rms / current_rms;

        // 限制增益范围（避免过度放大）
        let gain = gain.clamp(0.1, 10.0);

        for sample in samples.iter_mut() {
            *sample = (*sample * gain).clamp(-1.0, 1.0);
        }
    }

    /// 使用 ffmpeg 人声增强
    ///
    /// 通过 ffmpeg 滤波器链增强人声质量：
    /// - `highpass=f=80`: 去除 80Hz 以下的低频噪声
    /// - `lowpass=f=8000`: 去除 8kHz 以上的高频杂音
    /// - `loudnorm`: 响度归一化
    ///
    /// # 参数
    /// - `wav_path`: WAV 文件路径（原地替换）
    ///
    /// # 错误
    /// - ffmpeg 不可用或执行失败
    fn enhance_with_ffmpeg(&self, wav_path: &Path) -> AppResult<()> {
        let temp_path = wav_path.with_extension("enhanced_tmp.wav");

        let output = std::process::Command::new("ffmpeg")
            .arg("-y")
            .arg("-i")
            .arg(wav_path)
            .arg("-af")
            .arg("highpass=f=80,lowpass=f=8000,loudnorm=I=-20:TP=-1:LRA=7")
            .arg("-ar")
            .arg("16000")
            .arg("-ac")
            .arg("1")
            .arg("-sample_fmt")
            .arg("s16")
            .arg(&temp_path)
            .output();

        match output {
            Ok(result) => {
                if !result.status.success() {
                    let stderr = String::from_utf8_lossy(&result.stderr);
                    return Err(AppError::VoiceCloningError(format!(
                        "ffmpeg enhancement failed: {stderr}"
                    )));
                }
                // 替换原文件
                std::fs::rename(&temp_path, wav_path).map_err(|e| {
                    AppError::VoiceCloningError(format!("Failed to replace enhanced audio: {e}"))
                })?;
                tracing::debug!("VoiceExtractor: ffmpeg enhancement completed");
                Ok(())
            }
            Err(e) => Err(AppError::VoiceCloningError(format!(
                "ffmpeg not available or failed to execute: {e}"
            ))),
        }
    }

    /// 写入 WAV 文件
    ///
    /// 将 f32 采样数据写入 16-bit PCM WAV 文件。
    fn write_wav(&self, path: &Path, samples: &[f32], sample_rate: u32) -> AppResult<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to create WAV writer: {e}"))
        })?;

        for sample in samples {
            let i16_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer
                .write_sample(i16_sample)
                .map_err(|e| AppError::VoiceCloningError(format!("Failed to write sample: {e}")))?;
        }

        writer
            .finalize()
            .map_err(|e| AppError::VoiceCloningError(format!("Failed to finalize WAV: {e}")))?;

        Ok(())
    }
}

impl std::fmt::Debug for VoiceExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceExtractor")
            .field("enable_enhancement", &self.config.enable_enhancement)
            .field("enable_silence_trim", &self.config.enable_silence_trim)
            .field("enable_normalization", &self.config.enable_normalization)
            .finish()
    }
}

// ─── 音频处理工具函数 ─────────────────────────────────────

/// 计算 RMS（均方根）
///
/// 用于衡量音频的平均能量。
#[must_use]
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// dB 转幅度
///
/// `amplitude = 10^(db/20)`
#[must_use]
fn db_to_amplitude(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// 幅度转 dB
///
/// `db = 20 * log10(amplitude)`
#[must_use]
#[allow(dead_code)]
fn amplitude_to_db(amp: f64) -> f64 {
    if amp < f64::EPSILON {
        return f64::NEG_INFINITY;
    }
    20.0 * amp.log10()
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VoiceExtractorConfig;

    /// 创建测试用 WAV 文件
    fn create_test_wav(path: &Path, duration_secs: f64, sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV");
        let total_samples = (sample_rate as f64 * duration_secs) as usize;
        for i in 0..total_samples {
            // 模拟语音信号（正弦波 + 幅度变化）
            let t = i as f64 / sample_rate as f64;
            let envelope = if t < 0.5 || t > duration_secs - 0.5 {
                0.0 // 首尾静音
            } else {
                0.3
            };
            let sample =
                ((t * 220.0 * 2.0 * std::f64::consts::PI).sin() * envelope * 32767.0) as i16;
            writer.write_sample(sample).expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize WAV");
    }

    // ── ReferenceAudio 测试 ──────────────────────────

    #[test]
    fn test_reference_audio_new() {
        let ref_audio = ReferenceAudio::new(
            PathBuf::from("ref.wav"),
            5.0,
            16000,
            "Hello world".to_string(),
        );

        assert_eq!(ref_audio.path, PathBuf::from("ref.wav"));
        assert!((ref_audio.duration_secs - 5.0).abs() < f64::EPSILON);
        assert_eq!(ref_audio.sample_rate, 16000);
        assert_eq!(ref_audio.prompt_text, "Hello world");
        assert!(ref_audio.speaker_id.is_none());
    }

    #[test]
    fn test_reference_audio_with_speaker_id() {
        let ref_audio =
            ReferenceAudio::new(PathBuf::from("ref.wav"), 5.0, 16000, "Hello".to_string())
                .with_speaker_id("SPEAKER_00");

        assert_eq!(ref_audio.speaker_id.as_deref(), Some("SPEAKER_00"));
    }

    // ── VoiceExtractor 测试 ──────────────────────────

    #[test]
    fn test_voice_extractor_new() {
        let config = VoiceExtractorConfig::default();
        let extractor = VoiceExtractor::new(config);

        assert!(extractor.config().enable_enhancement);
        assert!(extractor.config().enable_silence_trim);
        assert!(extractor.config().enable_normalization);
    }

    #[test]
    fn test_voice_extractor_with_default_config() {
        let extractor = VoiceExtractor::with_default_config();

        assert!((extractor.config().ideal_duration_secs - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extract_reference_audio_success() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        // 创建 30 秒音频
        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 30.0, 16000);

        // 创建 ASR Segment（5 秒，有文本）
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 2.0, "Hi".into()),
            Segment::new(
                "seg-1".into(),
                2.0,
                7.0,
                "Hello everyone, welcome to this presentation.".into(),
            ),
            Segment::new("seg-2".into(), 7.0, 12.0, "Today we will discuss.".into()),
        ];

        // 禁用 ffmpeg 增强（测试环境可能没有 ffmpeg）
        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let extractor = VoiceExtractor::new(config);

        let ref_output = dir.path().join("auto_reference.wav");
        let result = extractor.extract_reference_audio(&full_wav, &segments, &ref_output);

        assert!(result.is_ok(), "Should extract successfully");
        let ref_audio = result.unwrap().expect("Should find a suitable segment");
        assert!(ref_output.exists(), "Reference WAV should be created");
        assert!(
            !ref_audio.prompt_text.is_empty(),
            "prompt_text should not be empty"
        );
        assert!(ref_audio.duration_secs > 0.0, "Duration should be positive");
        assert_eq!(ref_audio.sample_rate, 16000);
    }

    #[test]
    fn test_extract_reference_audio_picks_best_duration() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        // 多个候选，验证选择了最接近 5 秒的
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 8.0, "Long segment".into()), // 8s
            Segment::new("seg-1".into(), 8.0, 13.0, "Medium segment".into()), // 5s ← best
            Segment::new("seg-2".into(), 13.0, 16.0, "Short".into()),      // 3s
        ];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let extractor = VoiceExtractor::new(config);

        let ref_output = dir.path().join("ref.wav");
        let result = extractor
            .extract_reference_audio(&full_wav, &segments, &ref_output)
            .unwrap();

        assert!(result.is_some(), "Should find a segment");
        let ref_audio = result.unwrap();
        assert_eq!(ref_audio.prompt_text, "Medium segment");
    }

    #[test]
    fn test_extract_reference_audio_no_suitable_segment() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        // 所有 segment 都太短
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 0.5, "A".into()),
            Segment::new("seg-1".into(), 0.5, 1.0, "B".into()),
        ];

        let extractor = VoiceExtractor::with_default_config();

        let ref_output = dir.path().join("ref.wav");
        let result = extractor
            .extract_reference_audio(&full_wav, &segments, &ref_output)
            .unwrap();

        assert!(
            result.is_none(),
            "Should return None when no suitable segment"
        );
    }

    #[test]
    fn test_extract_reference_audio_empty_segments() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 10.0, 16000);

        let extractor = VoiceExtractor::with_default_config();

        let ref_output = dir.path().join("ref.wav");
        let result = extractor.extract_reference_audio(&full_wav, &[], &ref_output);

        assert!(result.is_ok(), "Should handle empty segments gracefully");
        assert!(
            result.unwrap().is_none(),
            "Should return None for empty segments"
        );
    }

    #[test]
    fn test_extract_reference_audio_fallback_to_wider_range() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        // 没有严格匹配 3-10s 的，但有 2-15s 的
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 12.0, "Twelve seconds".into()),
            Segment::new("seg-1".into(), 12.0, 14.0, "Two seconds".into()),
        ];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let extractor = VoiceExtractor::new(config);

        let ref_output = dir.path().join("ref.wav");
        let result = extractor
            .extract_reference_audio(&full_wav, &segments, &ref_output)
            .unwrap();

        assert!(result.is_some(), "Should fallback to wider range");
    }

    #[test]
    fn test_extract_reference_audio_with_silence_trim() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 30.0, 16000);

        let segments = vec![Segment::new("seg-0".into(), 0.0, 7.0, "Hello world".into())];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            enable_silence_trim: true,
            enable_normalization: false,
            ..Default::default()
        };
        let extractor = VoiceExtractor::new(config);

        let ref_output = dir.path().join("ref_trimmed.wav");
        let result = extractor
            .extract_reference_audio(&full_wav, &segments, &ref_output)
            .unwrap();

        assert!(result.is_some());
        // 修剪后时长应 <= 原始时长（7秒）
        let ref_audio = result.unwrap();
        assert!(
            ref_audio.duration_secs <= 7.0,
            "Duration after trim should be <= original"
        );
    }

    #[test]
    fn test_extract_reference_audio_with_normalization() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 30.0, 16000);

        let segments = vec![Segment::new("seg-0".into(), 0.0, 7.0, "Hello world".into())];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            enable_silence_trim: false,
            enable_normalization: true,
            ..Default::default()
        };
        let extractor = VoiceExtractor::new(config);

        let ref_output = dir.path().join("ref_normalized.wav");
        let result = extractor
            .extract_reference_audio(&full_wav, &segments, &ref_output)
            .unwrap();

        assert!(result.is_some());
        assert!(ref_output.exists());

        // 验证 WAV 文件可正常读取
        let reader = hound::WavReader::open(&ref_output).expect("Failed to open WAV");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 16000);
    }

    #[test]
    fn test_extract_reference_audio_custom_durations() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 4.0, "Four seconds".into()),
            Segment::new("seg-1".into(), 4.0, 8.0, "Four seconds again".into()),
            Segment::new("seg-2".into(), 8.0, 20.0, "Twelve seconds".into()),
        ];

        // 自定义时长范围：4-8 秒，理想 6 秒
        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            min_duration_secs: 4.0,
            max_duration_secs: 8.0,
            ideal_duration_secs: 6.0,
            ..Default::default()
        };
        let extractor = VoiceExtractor::new(config);

        let ref_output = dir.path().join("ref_custom.wav");
        let result = extractor
            .extract_reference_audio(&full_wav, &segments, &ref_output)
            .unwrap();

        assert!(result.is_some());
        // 应选择 4 秒的片段（最接近 6 秒的候选是 4 秒和 4 秒）
        let ref_audio = result.unwrap();
        assert_eq!(ref_audio.prompt_text, "Four seconds");
    }

    // ── 工具函数测试 ──────────────────────────────────

    #[test]
    fn test_rms_empty() {
        assert!((rms(&[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rms_constant() {
        let samples = vec![0.5; 100];
        let result = rms(&samples);
        assert!((result - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_rms_sine_wave() {
        // 正弦波的 RMS = amplitude / sqrt(2)
        let samples: Vec<f32> = (0..1000)
            .map(|i| ((i as f32 / 1000.0 * 2.0 * std::f32::consts::PI * 10.0).sin()) * 0.5)
            .collect();
        let result = rms(&samples);
        let expected = 0.5 / 2.0_f32.sqrt();
        assert!(
            (result - expected).abs() < 0.01,
            "RMS of sine wave should be amplitude/sqrt(2): got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_db_to_amplitude() {
        assert!((db_to_amplitude(0.0) - 1.0).abs() < 0.001);
        assert!((db_to_amplitude(-20.0) - 0.1).abs() < 0.01);
        assert!((db_to_amplitude(-40.0) - 0.01).abs() < 0.001);
        assert!((db_to_amplitude(-6.0) - 0.501).abs() < 0.01);
    }

    #[test]
    fn test_amplitude_to_db() {
        assert!((amplitude_to_db(1.0) - 0.0).abs() < 0.001);
        assert!((amplitude_to_db(0.1) - (-20.0)).abs() < 0.01);
        assert!((amplitude_to_db(0.01) - (-40.0)).abs() < 0.01);
    }

    #[test]
    fn test_db_amplitude_roundtrip() {
        let values = [0.001, 0.01, 0.1, 0.5, 1.0];
        for v in values {
            let db = amplitude_to_db(v);
            let restored = db_to_amplitude(db);
            assert!(
                (restored - v).abs() < 0.001,
                "Roundtrip failed for {v}: got {restored}"
            );
        }
    }
}

// ─── 长音频智能裁剪 — 借鉴 OmniVoice 智能片段选择 ──────────

/// 智能裁剪长音频
///
/// 借鉴 OmniVoice 的智能片段选择逻辑：当参考音频过长时（> 30s），
/// 不是简单截断，而是通过滑动窗口找到能量最高、静音最少的片段。
///
/// # 算法
/// 1. 将音频分成 `target_duration` 秒的滑动窗口
/// 2. 计算每个窗口的能量得分（RMS + 非静音比例）
/// 3. 选择得分最高的窗口作为参考音频
///
/// # 参数
/// - `samples`: 音频波形
/// - `sample_rate`: 采样率
/// - `target_duration`: 目标时长（秒，默认 10.0）
/// - `silence_threshold_db`: 静音阈值（dBFS，默认 -50）
///
/// # 返回
/// 裁剪后的音频波形
#[must_use]
pub fn smart_trim_long_audio(
    samples: &[f32],
    sample_rate: u32,
    target_duration: f64,
    silence_threshold_db: f32,
) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
    }

    let target_samples = (target_duration * sample_rate as f64) as usize;
    if samples.len() <= target_samples {
        return samples.to_vec();
    }

    let threshold_amp = 10.0f32.powf(silence_threshold_db / 20.0);
    let frame_size = (sample_rate as f64 * 0.02) as usize; // 20ms 帧
    if frame_size == 0 {
        return samples[..target_samples].to_vec();
    }

    // 滑动窗口步长（每次移动 1 秒）
    let step_samples = sample_rate as usize;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_start = 0usize;

    let mut start = 0usize;
    while start + target_samples <= samples.len() {
        let window = &samples[start..start + target_samples];

        // 计算窗口能量得分
        let mut energy_sum = 0.0f64;
        let mut non_silent_frames = 0usize;
        let mut total_frames = 0usize;

        for frame in window.chunks(frame_size) {
            let frame_rms = rms(frame);
            energy_sum += frame_rms as f64;
            total_frames += 1;
            if frame_rms > threshold_amp {
                non_silent_frames += 1;
            }
        }

        if total_frames == 0 {
            break;
        }

        let avg_energy = energy_sum / total_frames as f64;
        let non_silent_ratio = non_silent_frames as f64 / total_frames as f64;
        // 综合得分：能量 × 非静音比例
        let score = avg_energy * non_silent_ratio;

        if score > best_score {
            best_score = score;
            best_start = start;
        }

        start += step_samples;
    }

    // 返回最佳窗口
    let end = (best_start + target_samples).min(samples.len());
    samples[best_start..end].to_vec()
}

/// 自动从多个 ASR 片段构建参考文本
///
/// 借鉴 OmniVoice 的参考文本选择逻辑：当没有明确的参考文本时，
/// 从 ASR 片段中选择文本最清晰、长度适中的片段作为参考文本。
///
/// # 参数
/// - `segments`: ASR 片段列表
/// - `ideal_chars`: 理想文本长度（字符数，默认 20-100）
///
/// # 返回
/// 最佳参考文本，如果没有合适的则返回 None
#[must_use]
pub fn auto_select_reference_text(segments: &[Segment]) -> Option<String> {
    if segments.is_empty() {
        return None;
    }

    // 评分每个片段的文本质量
    let mut best_score = f64::NEG_INFINITY;
    let mut best_text = None;

    for seg in segments {
        let text = seg.source_text.trim();
        if text.is_empty() {
            continue;
        }

        let char_count = text.chars().count();
        if char_count < 5 || char_count > 200 {
            continue;
        }

        // 评分：长度接近 50 字符最好，避免有特殊字符
        let length_score = 1.0 - ((char_count as f64 - 50.0).abs() / 50.0);
        let special_count = text
            .chars()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace() && !c.is_ascii_punctuation())
            .count();
        let special_penalty = special_count as f64 * 0.1;
        let score = length_score - special_penalty;

        if score > best_score {
            best_score = score;
            best_text = Some(text.to_string());
        }
    }

    best_text
}

#[cfg(test)]
mod omni_smart_trim_tests {
    use super::*;

    #[test]
    fn test_smart_trim_short_audio() {
        // 短音频不需要裁剪
        let samples = vec![0.5; 4800]; // 0.2s @ 24kHz
        let result = smart_trim_long_audio(&samples, 24000, 10.0, -50.0);
        assert_eq!(result.len(), samples.len());
    }

    #[test]
    fn test_smart_trim_long_audio() {
        // 30s 音频，目标 10s
        let sample_rate = 16000u32;
        let total_samples = sample_rate as usize * 30;
        // 创建有变化能量的音频：5-15s 有信号，其他静音
        let samples: Vec<f32> = (0..total_samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                if t >= 5.0 && t <= 15.0 {
                    (t * 220.0 * 2.0 * std::f64::consts::PI).sin() as f32 * 0.3
                } else {
                    0.0
                }
            })
            .collect();

        let result = smart_trim_long_audio(&samples, sample_rate, 10.0, -50.0);
        // 应该裁剪到 ~10s
        assert!(result.len() <= sample_rate as usize * 10 + sample_rate as usize);
        assert!(result.len() >= sample_rate as usize * 9);
        // 最佳窗口应该在 5-15s 范围内（有信号的区域）
        let result_rms = rms(&result);
        assert!(
            result_rms > 0.01,
            "Selected window should have signal energy"
        );
    }

    #[test]
    fn test_smart_trim_empty() {
        assert!(smart_trim_long_audio(&[], 24000, 10.0, -50.0).is_empty());
    }

    #[test]
    fn test_auto_select_reference_text_basic() {
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 3.0, "Hello world".into()),
            Segment::new(
                "seg-1".into(),
                3.0,
                8.0,
                "Welcome to this presentation about AI".into(),
            ),
            Segment::new("seg-2".into(), 8.0, 11.0, "Hi".into()),
        ];
        let result = auto_select_reference_text(&segments);
        assert!(result.is_some());
        // "Welcome to this presentation about AI" (38 chars) should be preferred over "Hello world" (11) and "Hi" (2)
        assert!(result.unwrap().len() > 10);
    }

    #[test]
    fn test_auto_select_reference_text_empty() {
        assert!(auto_select_reference_text(&[]).is_none());
    }

    #[test]
    fn test_auto_select_reference_text_all_empty() {
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 3.0, "".into()),
            Segment::new("seg-1".into(), 3.0, 8.0, "   ".into()),
        ];
        assert!(auto_select_reference_text(&segments).is_none());
    }

    #[test]
    fn test_auto_select_reference_text_too_short() {
        let segments = vec![Segment::new("seg-0".into(), 0.0, 3.0, "Hi".into())];
        assert!(auto_select_reference_text(&segments).is_none());
    }

    #[test]
    fn test_auto_select_reference_text_too_long() {
        let long_text = "a".repeat(250);
        let segments = vec![Segment::new("seg-0".into(), 0.0, 3.0, long_text)];
        assert!(auto_select_reference_text(&segments).is_none());
    }
}
