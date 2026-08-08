//! 音视频处理模块
//!
//! 基于 FFmpeg 命令行工具实现音频提取与视频合成功能。
//!
//! # 功能概览
//! - [`AudioExtractor`] trait + [`FfmpegAudioExtractor`]：从视频中提取 16kHz 单声道 WAV 音频
//! - [`VideoComposer`] trait + [`FfmpegVideoComposer`]：将新音轨合成到视频中（替换原音轨，可选烧录字幕）
//! - [`probe_media`]：探测媒体文件元数据（时长、流信息）
//!
//! # 硬件加速
//! [`FfmpegVideoComposer`] 默认使用 `h264_videotoolbox` 编码器（macOS M 系列芯片），
//! 可通过 [`FfmpegVideoComposer::with_encoder`] 指定其他编码器（如 `libx264`）。
//!
//! # 错误处理
//! 所有公共方法返回 [`AppResult<T>`]，对文件不存在、FFmpeg 执行失败等情况
//! 返回清晰的 [`AppError`] 变体。

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::config::AudioSyncMode;
use crate::error::{AppError, AppResult};
use crate::sola::{sola_write_into_buffer, DEFAULT_OVERLAP_SAMPLES};

// ─── ffprobe JSON 反序列化（内部类型） ─────────────────────

/// ffprobe JSON 输出的根结构（内部使用）
#[derive(Deserialize)]
struct FfprobeOutput {
    /// 媒体流列表
    streams: Vec<FfprobeStream>,
    /// 容器格式信息
    #[serde(default)]
    format: FfprobeFormat,
}

/// ffprobe 单个流信息（内部使用）
#[derive(Deserialize)]
struct FfprobeStream {
    /// 流类型（"video" 或 "audio"）
    codec_type: String,
    /// 编解码器名称
    codec_name: String,
    /// 音频采样率（字符串形式，如 "16000"）
    sample_rate: Option<String>,
    /// 音频通道数
    channels: Option<i32>,
    /// 视频宽度（像素）
    width: Option<i32>,
    /// 视频高度（像素）
    height: Option<i32>,
}

/// ffprobe 格式信息（内部使用）
#[derive(Deserialize, Default)]
struct FfprobeFormat {
    /// 媒体时长（字符串形式，如 "30.000000"）
    duration: Option<String>,
}

// ─── 公共数据结构 ────────────────────────────────────────

/// 媒体文件探测信息
///
/// 通过 [`probe_media`] 获取，包含媒体时长和所有流的信息。
#[derive(Debug, Clone)]
pub struct MediaInfo {
    /// 媒体时长（秒）
    pub duration: f64,
    /// 流信息列表（视频流、音频流等）
    pub streams: Vec<StreamInfo>,
}

/// 单个媒体流信息
///
/// 描述一个视频流或音频流的编解码器、分辨率/采样率等属性。
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// 流类型（`"video"` 或 `"audio"`）
    pub codec_type: String,
    /// 编解码器名称（如 `h264`、`aac`、`pcm_s16le`）
    pub codec_name: String,
    /// 音频采样率（Hz），仅音频流有值
    pub sample_rate: Option<i32>,
    /// 音频通道数，仅音频流有值
    pub channels: Option<i32>,
    /// 视频宽度（像素），仅视频流有值
    pub width: Option<i32>,
    /// 视频高度（像素），仅视频流有值
    pub height: Option<i32>,
}

impl From<FfprobeStream> for StreamInfo {
    /// 将 ffprobe 内部流结构转换为公共 [`StreamInfo`]
    fn from(s: FfprobeStream) -> Self {
        Self {
            codec_type: s.codec_type,
            codec_name: s.codec_name,
            sample_rate: s.sample_rate.and_then(|sr| sr.parse().ok()),
            channels: s.channels,
            width: s.width,
            height: s.height,
        }
    }
}

// ─── 公共 Trait 定义 ─────────────────────────────────────

/// 音频提取器接口
///
/// 定义从视频中提取音频的标准接口，具体实现见 [`FfmpegAudioExtractor`]。
///
/// # 线程安全
/// 实现者必须满足 `Send + Sync`，以支持异步流水线中的并行处理。
pub trait AudioExtractor: Send + Sync {
    /// 从视频中提取音频为 16kHz 单声道 PCM WAV
    ///
    /// # 参数
    /// - `input_path`: 输入视频文件路径
    /// - `output_path`: 输出 WAV 文件路径
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 输入文件不存在
    /// - [`AppError::FFmpeg`][]: FFmpeg 执行失败（非零退出码）
    /// - [`AppError::Io`][]: 命令启动失败（如 ffmpeg 未安装）
    fn extract_audio(&self, input_path: &Path, output_path: &Path) -> AppResult<()>;
}

/// 视频合成器接口
///
/// 定义将新音轨与视频合成的标准接口，具体实现见 [`FfmpegVideoComposer`]。
///
/// # 线程安全
/// 实现者必须满足 `Send + Sync`，以支持异步流水线中的并行处理。
pub trait VideoComposer: Send + Sync {
    /// 将新音轨与视频合成，替换原音轨
    ///
    /// # 参数
    /// - `video_path`: 原始视频文件路径
    /// - `new_audio_path`: 新音频文件路径
    /// - `output_path`: 输出视频文件路径
    /// - `burn_subtitles`: 是否烧录字幕到视频画面
    /// - `subtitle_path`: 字幕文件路径（当 `burn_subtitles` 为 `true` 时必须提供）
    /// - `video_stretch_factor`: 视频拉伸因子（1.0 = 原速，>1.0 = 慢放以匹配较长音频）
    ///
    /// # 行为
    /// - `video_stretch_factor > 1.0` 时，使用 `setpts=PTS/factor` 慢放视频
    /// - 不烧录字幕且不慢放时，视频流直接复制（`-c:v copy`），速度更快
    /// - 烧录字幕或慢放时，使用滤镜并重新编码视频
    /// - 音频始终重新编码为 AAC
    /// - 使用 `-map 0:v:0` 和 `-map 1:a:0` 只映射第一个视频流和音频流
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 输入文件不存在
    /// - [`AppError::FFmpeg`][]: FFmpeg 执行失败或参数不一致
    /// - [`AppError::Io`][]: 命令启动失败
    fn compose_video(
        &self,
        video_path: &Path,
        new_audio_path: &Path,
        output_path: &Path,
        burn_subtitles: bool,
        subtitle_path: Option<&Path>,
        video_stretch_factor: f64,
    ) -> AppResult<()>;
}

// ─── FFmpeg 音频提取器实现 ───────────────────────────────

/// 基于 FFmpeg 的音频提取器
///
/// 将视频中的音频提取为 **16kHz 单声道 PCM WAV** 格式，
/// 适配后续 ASR（语音识别）处理要求。
///
/// # 示例
/// ```no_run
/// use std::path::Path;
/// use vt_core::media::{AudioExtractor, FfmpegAudioExtractor};
///
/// let extractor = FfmpegAudioExtractor::new();
/// extractor.extract_audio(Path::new("input.mp4"), Path::new("output.wav"))
///     .expect("Audio extraction failed");
/// ```
pub struct FfmpegAudioExtractor;

impl FfmpegAudioExtractor {
    /// 创建新的 FFmpeg 音频提取器实例
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for FfmpegAudioExtractor {
    /// 返回默认的音频提取器
    fn default() -> Self {
        Self::new()
    }
}

impl AudioExtractor for FfmpegAudioExtractor {
    #[tracing::instrument(skip(self), fields(input = ?input_path, output = ?output_path))]
    fn extract_audio(&self, input_path: &Path, output_path: &Path) -> AppResult<()> {
        if !input_path.exists() {
            return Err(AppError::FileNotFound(input_path.to_path_buf()));
        }

        tracing::debug!(
            "Extracting audio from {:?} to {:?}",
            input_path,
            output_path
        );

        let output = Command::new("ffmpeg")
            .arg("-y")
            .args(["-i"])
            .arg(input_path)
            .args(["-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
            .arg(output_path)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AppError::FFmpeg(format!(
                "Audio extraction failed (exit code {:?}): {stderr}",
                output.status.code()
            )));
        }

        tracing::debug!("Audio extracted successfully to {:?}", output_path);
        Ok(())
    }
}

// ─── FFmpeg 视频合成器实现 ───────────────────────────────

/// 基于 FFmpeg 的视频合成器
///
/// 支持将新音轨替换到视频中，可选烧录软字幕（SRT/ASS）。
///
/// # 硬件加速
/// 默认使用 `h264_videotoolbox` 编码器（macOS VideoToolbox 硬件加速）。
/// 在不支持 VideoToolbox 的环境，可通过 [`with_encoder`](Self::with_encoder)
/// 指定 `libx264` 等软件编码器。
///
/// # 示例
/// ```no_run
/// use std::path::Path;
/// use vt_core::media::{FfmpegVideoComposer, VideoComposer};
///
/// // 使用默认 VideoToolbox 硬件加速
/// let composer = FfmpegVideoComposer::new();
/// composer
///     .compose_video(
///         Path::new("input.mp4"),
///         Path::new("new_audio.wav"),
///         Path::new("output.mp4"),
///         false,
///         None,
///         1.0,
///     )
///     .expect("Video composition failed");
/// ```
pub struct FfmpegVideoComposer {
    /// 视频编码器名称（如 `h264_videotoolbox`、`libx264`）
    video_encoder: String,
    /// 视频质量参数（仅重新编码时生效）
    /// - `h264_videotoolbox`: 使用 `-q:v` (1-100，值越小质量越高，推荐 30-50)
    /// - `libx264`: 使用 `-crf` (0-51，值越小质量越高，推荐 18-28)
    /// - `None`: 使用编码器默认码率（不推荐，可能导致体积暴涨）
    video_quality: Option<i32>,
}

impl FfmpegVideoComposer {
    /// 创建使用 VideoToolbox 硬件加速的合成器
    ///
    /// 在 macOS M 系列芯片上自动启用硬件加速编码。
    #[must_use]
    pub fn new() -> Self {
        Self {
            video_encoder: "h264_videotoolbox".to_string(),
            // h264_videotoolbox 的 -q:v 默认 35：质量与体积的良好平衡
            // 对比测试：默认无参数 → 2 Mbps+，-q:v 35 → ~500 kbps (1080p 屏幕录制)
            video_quality: Some(35),
        }
    }

    /// 创建使用指定视频编码器的合成器
    ///
    /// # 参数
    /// - `video_encoder`: 编码器名称（如 `"libx264"`、`"libx265"`、`"h264_videotoolbox"`）
    ///
    /// # 示例
    /// ```no_run
    /// use vt_core::media::FfmpegVideoComposer;
    ///
    /// let composer = FfmpegVideoComposer::with_encoder("libx264");
    /// ```
    #[must_use]
    pub fn with_encoder(video_encoder: impl Into<String>) -> Self {
        Self {
            video_encoder: video_encoder.into(),
            video_quality: Some(35),
        }
    }

    /// 获取当前使用的视频编码器名称
    #[must_use]
    pub fn encoder(&self) -> &str {
        &self.video_encoder
    }

    /// 设置视频质量参数
    ///
    /// # 参数
    /// - `quality`: 质量值
    ///   - `h264_videotoolbox`: 1-100，值越小质量越高，推荐 30-50
    ///   - `libx264`: 0-51 (CRF)，值越小质量越高，推荐 18-28
    ///   - 传入 `None` 使用编码器默认码率（不推荐）
    #[must_use]
    pub fn with_quality(mut self, quality: Option<i32>) -> Self {
        self.video_quality = quality;
        self
    }
}

impl Default for FfmpegVideoComposer {
    /// 返回使用默认 VideoToolbox 编码器的合成器
    fn default() -> Self {
        Self::new()
    }
}

impl VideoComposer for FfmpegVideoComposer {
    #[tracing::instrument(skip(self), fields(video = ?video_path, audio = ?new_audio_path, output = ?output_path, burn_subs = burn_subtitles))]
    fn compose_video(
        &self,
        video_path: &Path,
        new_audio_path: &Path,
        output_path: &Path,
        burn_subtitles: bool,
        subtitle_path: Option<&Path>,
        video_stretch_factor: f64,
    ) -> AppResult<()> {
        // 验证输入文件存在
        if !video_path.exists() {
            return Err(AppError::FileNotFound(video_path.to_path_buf()));
        }
        if !new_audio_path.exists() {
            return Err(AppError::FileNotFound(new_audio_path.to_path_buf()));
        }

        // 验证字幕参数一致性
        if burn_subtitles && subtitle_path.is_none() {
            return Err(AppError::FFmpeg(
                "burn_subtitles is true but no subtitle_path provided".to_string(),
            ));
        }
        if let Some(subs) = subtitle_path {
            if !subs.exists() {
                return Err(AppError::FileNotFound(subs.to_path_buf()));
            }
        }

        let need_slow = video_stretch_factor > 1.001;
        let need_reencode = burn_subtitles || need_slow;

        tracing::debug!(
            "Composing video: video={:?}, audio={:?}, output={:?}, burn_subtitles={}, stretch={:.3}",
            video_path,
            new_audio_path,
            output_path,
            burn_subtitles,
            video_stretch_factor
        );

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .args(["-i"])
            .arg(video_path)
            .args(["-i"])
            .arg(new_audio_path);

        if need_reencode {
            // 构建视频滤镜链
            let mut filters: Vec<String> = Vec::new();

            if need_slow {
                // setpts=PTS/factor 慢放视频
                filters.push(format!("setpts=PTS/{video_stretch_factor:.6}"));
            }

            if burn_subtitles {
                if let Some(subs) = subtitle_path {
                    let subs_str = subs.to_str().ok_or_else(|| {
                        AppError::FFmpeg(format!("Subtitle path contains invalid UTF-8: {subs:?}"))
                    })?;
                    filters.push(format!("subtitles={subs_str}"));
                }
            }

            let filter_chain = filters.join(",");
            cmd.args(["-vf", &filter_chain]);
            cmd.args(["-c:v", self.video_encoder.as_str()]);

            // 添加视频质量控制参数，防止输出体积暴涨
            if let Some(q) = &self.video_quality {
                if self.video_encoder == "h264_videotoolbox" {
                    // h264_videotoolbox 使用 -q:v (1-100，值越小质量越高)
                    cmd.args(["-q:v", &q.to_string()]);
                } else if self.video_encoder == "libx264" || self.video_encoder == "libx265" {
                    // libx264/libx265 使用 -crf (0-51，值越小质量越高)
                    cmd.args(["-crf", &q.to_string()]);
                }
            }
        } else {
            // 不烧录字幕且不慢放时直接复制视频流
            cmd.args(["-c:v", "copy"]);
        }

        // 音频始终重新编码为 AAC，映射第一个视频流(0:v:0)和新音频流(1:a:0)
        cmd.args(["-c:a", "aac"])
            .args(["-map", "0:v:0"])
            .args(["-map", "1:a:0"])
            .arg(output_path);

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AppError::FFmpeg(format!(
                "Video composition failed (exit code {:?}): {stderr}",
                output.status.code()
            )));
        }

        tracing::debug!("Video composed successfully to {:?}", output_path);
        Ok(())
    }
}

// ─── 视频/音频时长对齐辅助函数 ─────────────────────────────

/// 视频定格延长：当配音音频长于视频时，定格最后一帧等待音频结束
///
/// 参考 pyvideotrans 的 `_video_extend()` 方法，
/// 使用 ffmpeg `tpad` 滤镜在视频末尾定格最后一帧。
///
/// # 参数
/// - `video_path`: 视频文件路径（原地修改）
/// - `extend_secs`: 需要延长的秒数
///
/// # 错误
/// - [`AppError::FFmpeg`][]: ffmpeg 执行失败
pub fn extend_video_freeze_frame(video_path: &Path, extend_secs: f64) -> AppResult<()> {
    if extend_secs < 0.01 {
        return Ok(());
    }

    let tmp_path = video_path.with_extension("extended.mp4");
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y")
        .arg("-i")
        .arg(video_path)
        .arg("-vf")
        .arg(format!(
            "tpad=stop_mode=clone:stop_duration={extend_secs:.3}"
        ))
        .arg("-c:v")
        .arg("libx264")
        .arg("-crf")
        .arg("23")
        .arg("-preset")
        .arg("veryfast")
        .arg("-an")
        .arg(&tmp_path);

    let output = cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::warn!("Video freeze-frame extension failed (non-fatal): {stderr}");
        let _ = std::fs::remove_file(&tmp_path);
        return Ok(());
    }

    std::fs::rename(&tmp_path, video_path)?;
    tracing::debug!("Video extended by {extend_secs:.3}s with freeze frame");
    Ok(())
}

/// 音频末尾静音填充：当视频长于音频时，在音频末尾补静音
///
/// 参考 pyvideotrans 的 `apad` 用法，
/// 使用 ffmpeg `apad` 滤镜在音频末尾添加指定时长的静音。
///
/// # 参数
/// - `audio_path`: 音频文件路径（原地修改）
/// - `pad_secs`: 需要填充的静音秒数
///
/// # 错误
/// - [`AppError::FFmpeg`][]: ffmpeg 执行失败
pub fn pad_audio_with_silence(audio_path: &Path, pad_secs: f64) -> AppResult<()> {
    if pad_secs < 0.01 {
        return Ok(());
    }

    let tmp_path = audio_path.with_extension("padded.wav");
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y")
        .arg("-i")
        .arg(audio_path)
        .arg("-af")
        .arg(format!("apad=pad_dur={pad_secs:.3}"))
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg(&tmp_path);

    let output = cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::warn!("Audio silence padding failed (non-fatal): {stderr}");
        let _ = std::fs::remove_file(&tmp_path);
        return Ok(());
    }

    std::fs::rename(&tmp_path, audio_path)?;
    tracing::debug!("Audio padded with {pad_secs:.3}s of silence");
    Ok(())
}

/// 获取音频文件时长（秒）
///
/// 使用 ffprobe 获取音频时长。
pub fn get_audio_duration(audio_path: &Path) -> AppResult<f64> {
    if !audio_path.exists() {
        return Err(AppError::FileNotFound(audio_path.to_path_buf()));
    }

    let output = Command::new("ffprobe")
        .args(["-v", "quiet", "-show_entries", "format=duration"])
        .args(["-of", "default=noprint_wrappers=1:nokey=1"])
        .arg(audio_path)
        .output()?;

    if !output.status.success() {
        return Err(AppError::MediaProbe(format!(
            "ffprobe failed for {:?}",
            audio_path
        )));
    }

    let duration_str = String::from_utf8_lossy(&output.stdout);
    duration_str
        .trim()
        .parse::<f64>()
        .map_err(|e| AppError::MediaProbe(format!("Failed to parse audio duration: {e}")))
}

// ─── 媒体探测 ────────────────────────────────────────────

/// 探测媒体文件信息
///
/// 使用 `ffprobe` 获取媒体文件的时长、流信息等元数据。
///
/// # 参数
/// - `path`: 媒体文件路径
///
/// # 返回
/// 包含时长和流信息的 [`MediaInfo`]。
///
/// # 错误
/// - [`AppError::FileNotFound`][]: 文件不存在
/// - [`AppError::MediaProbe`][]: ffprobe 执行失败或 JSON 输出解析失败
/// - [`AppError::Io`][]: ffprobe 命令启动失败
///
/// # 示例
/// ```no_run
/// use std::path::Path;
/// use vt_core::media::probe_media;
///
/// let info = probe_media(Path::new("video.mp4")).expect("Probe failed");
/// println!("Duration: {}s", info.duration);
/// ```
#[tracing::instrument(fields(path = ?path))]
pub fn probe_media(path: &Path) -> AppResult<MediaInfo> {
    if !path.exists() {
        return Err(AppError::FileNotFound(path.to_path_buf()));
    }

    let output = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::MediaProbe(format!(
            "ffprobe failed (exit code {:?}): {stderr}",
            output.status.code()
        )));
    }

    let probe: FfprobeOutput = serde_json::from_slice(&output.stdout)
        .map_err(|e| AppError::MediaProbe(format!("Failed to parse ffprobe JSON output: {e}")))?;

    let duration = probe
        .format
        .duration
        .as_deref()
        .and_then(|d| d.parse::<f64>().ok())
        .unwrap_or(0.0);

    let streams: Vec<StreamInfo> = probe.streams.into_iter().map(StreamInfo::from).collect();

    Ok(MediaInfo { duration, streams })
}

// ─── 音频混合（多模式同步） ─────────────────────────────

/// 音频混合统计信息
#[derive(Debug, Clone, Default, Serialize)]
pub struct AudioMixStats {
    /// 总片段数
    pub total: usize,
    /// 被截断的片段数（Trim 模式）
    pub trimmed: usize,
    /// 被加速的片段数（SpeedUp / Hybrid 模式）
    pub sped_up: usize,
    /// 因音频溢出导致视频慢放的片段数（VideoSlow / Hybrid 模式）
    pub video_slowed: usize,
}

/// 音频混合结果
#[derive(Debug, Clone, Serialize)]
pub struct AudioMixResult {
    /// 混合后音频的实际时长（秒）
    /// 在 VideoSlow / Hybrid 模式下可能大于原视频时长
    pub audio_duration_secs: f64,
    /// 视频拉伸因子（1.0 = 不拉伸，>1.0 = 视频需要慢放）
    pub video_stretch_factor: f64,
    /// 统计信息
    pub stats: AudioMixStats,
}

/// 将多个 TTS 音频片段按同步策略混合到一条音轨中
///
/// 根据不同的 [`AudioSyncMode`]，采用不同的策略对齐 TTS 音频与原视频时间轴：
///
/// # 同步模式
/// - **Trim**: TTS 音频超出时间槽时截断尾部，不足时补静音。`video_stretch = 1.0`
/// - **SpeedUp**: 使用线性重采样加速 TTS 音频以适应时间槽。`video_stretch = 1.0`
/// - **VideoSlow**: 不修改 TTS 音频，采用"涟漪"放置（溢出部分推后后续片段），
///   视频慢放以匹配较长音频。`video_stretch = audio_duration / video_duration`
/// - **Hybrid**: 先加速至上限 `max_speed_ratio`，剩余溢出通过视频慢放补偿
///
/// # 参数
/// - `segments`: `[(起始秒, 结束秒, 音频文件路径), ...]`
/// - `video_duration_secs`: 原视频时长（秒）
/// - `output_path`: 输出 WAV 文件路径
/// - `mode`: 同步模式
/// - `max_speed_ratio`: 加速上限倍率（仅 SpeedUp / Hybrid 生效）
pub fn mix_audio_segments(
    segments: &[(f64, f64, &Path)],
    video_duration_secs: f64,
    output_path: &Path,
    mode: AudioSyncMode,
    max_speed_ratio: f32,
) -> AppResult<AudioMixResult> {
    const SAMPLE_RATE: u32 = 16000;
    let mut stats = AudioMixStats {
        total: segments.len(),
        ..Default::default()
    };

    // 第一遍：读取所有 TTS 音频并测量时长
    struct SegmentAudio {
        start: f64,
        end: f64,
        samples: Vec<f32>,
        tts_duration: f64,
    }

    let mut audios: Vec<SegmentAudio> = Vec::with_capacity(segments.len());
    for &(start, end, path) in segments {
        if !path.exists() {
            tracing::warn!("Skipping missing audio file: {:?}", path);
            continue;
        }
        let (samples, _sr) = read_wav_mono_16k(path)?;
        if samples.is_empty() {
            continue;
        }
        let tts_duration = samples.len() as f64 / SAMPLE_RATE as f64;
        audios.push(SegmentAudio {
            start,
            end,
            samples,
            tts_duration,
        });
    }

    if audios.is_empty() {
        // 无音频，输出静音
        let total_samples = (video_duration_secs * SAMPLE_RATE as f64) as usize;
        write_pcm_wav(output_path, &vec![0.0f32; total_samples], SAMPLE_RATE)?;
        return Ok(AudioMixResult {
            audio_duration_secs: video_duration_secs,
            video_stretch_factor: 1.0,
            stats,
        });
    }

    // 根据模式处理每段音频
    let mut processed: Vec<(f64, Vec<f32>)> = Vec::with_capacity(audios.len()); // (placement_time, samples)

    match mode {
        AudioSyncMode::Trim | AudioSyncMode::SpeedUp => {
            // 固定时间槽模式：每段音频放在原始 start 位置，限制在 [start, end] 内
            for audio in &audios {
                let slot_duration = (audio.end - audio.start).max(0.0);
                let slot_samples = (slot_duration * SAMPLE_RATE as f64) as usize;
                let tts_samples = &audio.samples;

                if slot_samples == 0 {
                    continue;
                }

                let final_samples = if tts_samples.len() > slot_samples {
                    // TTS 比时间槽长
                    if mode == AudioSyncMode::SpeedUp {
                        // 加速模式：重采样压缩
                        stats.sped_up += 1;
                        let ratio = tts_samples.len() as f64 / slot_samples as f64;
                        tracing::info!(
                            "SpeedUp: tts={:.2}s → slot={:.2}s (ratio={:.2}x)",
                            audio.tts_duration,
                            slot_duration,
                            ratio
                        );
                        linear_resample(tts_samples, slot_samples)
                    } else {
                        // Trim 模式：截断
                        stats.trimmed += 1;
                        tracing::info!(
                            "Trim: tts={:.2}s → slot={:.2}s (truncated {:.2}s)",
                            audio.tts_duration,
                            slot_duration,
                            audio.tts_duration - slot_duration
                        );
                        tts_samples[..slot_samples].to_vec()
                    }
                } else {
                    // TTS 比时间槽短：补静音
                    let mut padded = tts_samples.clone();
                    padded.resize(slot_samples, 0.0);
                    padded
                };

                processed.push((audio.start, final_samples));
            }
        }

        AudioSyncMode::VideoSlow | AudioSyncMode::Hybrid => {
            // 涟漪放置模式：音频可以溢出，后续片段被推后
            let mut cursor = 0.0f64; // 当前放置游标

            for audio in &audios {
                // 不在原始 start 之前开始（保留前导静音）
                let placement = audio.start.max(cursor);

                let final_samples = if mode == AudioSyncMode::Hybrid {
                    // Hybrid: 先尝试加速到 max_speed_ratio
                    let slot_duration = (audio.end - audio.start).max(0.01);
                    let speed_ratio = audio.tts_duration / slot_duration;

                    if speed_ratio > max_speed_ratio as f64 {
                        // 需要加速 + 视频慢放补偿
                        stats.sped_up += 1;
                        stats.video_slowed += 1;
                        let target_duration = audio.tts_duration / max_speed_ratio as f64;
                        let target_samples = (target_duration * SAMPLE_RATE as f64) as usize;
                        tracing::info!(
                            "Hybrid: tts={:.2}s, slot={:.2}s, speed={:.2}x→{:.2}x, remaining={:.2}s via video-slow",
                            audio.tts_duration,
                            slot_duration,
                            speed_ratio,
                            max_speed_ratio,
                            target_duration - slot_duration
                        );
                        linear_resample(&audio.samples, target_samples)
                    } else if speed_ratio > 1.0 {
                        // 只需加速，无需视频慢放
                        stats.sped_up += 1;
                        let target_samples = (slot_duration * SAMPLE_RATE as f64) as usize;
                        linear_resample(&audio.samples, target_samples)
                    } else {
                        // TTS 比时间槽短，直接使用
                        audio.samples.clone()
                    }
                } else {
                    // VideoSlow: 不修改音频
                    let slot_duration = (audio.end - audio.start).max(0.01);
                    if audio.tts_duration > slot_duration {
                        stats.video_slowed += 1;
                        tracing::info!(
                            "VideoSlow: tts={:.2}s > slot={:.2}s, overflow={:.2}s (video will slow down)",
                            audio.tts_duration,
                            slot_duration,
                            audio.tts_duration - slot_duration
                        );
                    }
                    audio.samples.clone()
                };

                let final_duration = final_samples.len() as f64 / SAMPLE_RATE as f64;
                cursor = placement + final_duration;
                processed.push((placement, final_samples));
            }
        }
    }

    // 计算最终音频时长
    let final_audio_duration = processed
        .iter()
        .map(|(start, samples)| start + samples.len() as f64 / SAMPLE_RATE as f64)
        .fold(0.0f64, f64::max)
        .max(video_duration_secs);

    let video_stretch_factor = (final_audio_duration / video_duration_secs).max(1.0);

    // 分配缓冲区
    let total_samples = (final_audio_duration * SAMPLE_RATE as f64) as usize;
    let mut buffer: Vec<f32> = vec![0.0f32; total_samples];

    // 将处理后的音频使用 SOLA 算法放入缓冲区
    // SOLA 在重叠区域通过互相关找到最佳拼接点并使用 Hann 窗淡入淡出，消除拼接痕迹
    let overlap_samples = DEFAULT_OVERLAP_SAMPLES; // 20ms@16kHz
    for (placement, samples) in &processed {
        let offset = (placement * SAMPLE_RATE as f64) as usize;
        if offset >= total_samples {
            continue;
        }
        sola_write_into_buffer(&mut buffer, offset, samples, overlap_samples);
    }

    // 归一化
    let max_abs = buffer.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if max_abs > 1.0 {
        let scale = 1.0 / max_abs;
        tracing::warn!(
            "Audio peak {:.3} > 1.0, applying normalization scale {:.4}",
            max_abs,
            scale
        );
        for s in &mut buffer {
            *s *= scale;
        }
    }

    write_pcm_wav(output_path, &buffer, SAMPLE_RATE)?;

    tracing::info!(
        "Audio mix [{:?}]: {} segments, audio={:.1}s, video={:.1}s, stretch={:.3}x, trimmed={}, sped_up={}, video_slowed={}",
        mode,
        segments.len(),
        final_audio_duration,
        video_duration_secs,
        video_stretch_factor,
        stats.trimmed,
        stats.sped_up,
        stats.video_slowed
    );

    Ok(AudioMixResult {
        audio_duration_secs: final_audio_duration,
        video_stretch_factor,
        stats,
    })
}

/// 线性重采样：将 `input` 采样数据重采样为 `target_len` 个样本
///
/// 通过线性插值实现，改变速度但不改变音高。
fn linear_resample(input: &[f32], target_len: usize) -> Vec<f32> {
    if input.is_empty() || target_len == 0 {
        return Vec::new();
    }
    if input.len() == target_len {
        return input.to_vec();
    }

    let ratio = input.len() as f64 / target_len as f64;
    let mut output = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        let s1 = input[src_idx];
        let s2 = if src_idx + 1 < input.len() {
            input[src_idx + 1]
        } else {
            input[input.len() - 1]
        };

        output.push((s1 as f64 * (1.0 - frac) + s2 as f64 * frac) as f32);
    }

    output
}

/// 读取 mono WAV 文件为 f32 采样数据，自动重采样到 16kHz（内部使用）
///
/// TTS 引擎可能输出 24kHz/48kHz 等不同采样率的 WAV 文件，
/// 此函数在读取后自动通过线性插值重采样到 16kHz，以统一后续混合处理。
///
/// # 参数
/// - `path`: WAV 文件路径
///
/// # 返回
/// `(samples, 16000)` — 重采样后的 f32 采样数据和固定 16000Hz 采样率
///
/// # 错误
/// - [`AppError::FileNotFound`][]: 文件不存在
/// - [`AppError::AudioDecodeError`][]: WAV 格式损坏或读取失败
fn read_wav_mono_16k(path: &Path) -> AppResult<(Vec<f32>, u32)> {
    const TARGET_RATE: u32 = 16000;

    if !path.exists() {
        return Err(AppError::FileNotFound(path.to_path_buf()));
    }

    let mut reader = hound::WavReader::open(path)
        .map_err(|e| AppError::AudioDecodeError(format!("Failed to open WAV {path:?}: {e}")))?;

    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(AppError::AudioDecodeError(format!(
            "Expected mono, got {} channels for {path:?}",
            spec.channels
        )));
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| {
            s.map(|v| v as f32 / 32768.0)
                .map_err(|e| AppError::AudioDecodeError(format!("Sample read error: {e}")))
        })
        .collect::<AppResult<Vec<_>>>()?;

    // 如果采样率不是 16kHz，通过线性插值重采样
    let samples = if spec.sample_rate != TARGET_RATE {
        tracing::debug!(
            "Resampling TTS audio from {}Hz to {}Hz ({} samples → ~{} samples)",
            spec.sample_rate,
            TARGET_RATE,
            samples.len(),
            (samples.len() as f64 * TARGET_RATE as f64 / spec.sample_rate as f64) as usize
        );
        resample_linear(&samples, spec.sample_rate, TARGET_RATE)
    } else {
        samples
    };

    Ok((samples, TARGET_RATE))
}

/// 线性插值重采样：将输入采样数据从 `src_rate` 重采样到 `dst_rate`
///
/// 通过线性插值实现，改变采样率但不改变音高。
fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return input.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let target_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        let s1 = input[src_idx];
        let s2 = if src_idx + 1 < input.len() {
            input[src_idx + 1]
        } else {
            input[input.len() - 1]
        };

        output.push((s1 as f64 * (1.0 - frac) + s2 as f64 * frac) as f32);
    }

    output
}

/// 将 f32 采样数据写入 16kHz mono 16-bit PCM WAV 文件
fn write_pcm_wav(path: &Path, samples: &[f32], sample_rate: u32) -> AppResult<()> {
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

/// 在媒体信息中查找第一个音频流
///
/// # 返回
/// 第一个音频流的引用，如果没有音频流则返回 `None`。
#[must_use]
pub fn find_audio_stream(info: &MediaInfo) -> Option<&StreamInfo> {
    info.streams.iter().find(|s| s.codec_type == "audio")
}

/// 在媒体信息中查找第一个视频流
///
/// # 返回
/// 第一个视频流的引用，如果没有视频流则返回 `None`。
#[must_use]
pub fn find_video_stream(info: &MediaInfo) -> Option<&StreamInfo> {
    info.streams.iter().find(|s| s.codec_type == "video")
}
