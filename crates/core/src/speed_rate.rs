//! 声画对齐 (SpeedRate) 模块
//!
//! 参考 pyvideotrans 的 `task/_rate.py`，实现片段级别的声画对齐：
//!
//! # 三种模式
//! - **AudioSpeedUp**: 仅加速配音音频（ffmpeg atempo），不改变视频
//! - **VideoSlowDown**: 仅慢放视频（setpts），不改变音频
//! - **Hybrid**: 倍率<1.2 仅音频加速，>1.2 音频+视频各分担一半
//!
//! # 工作流
//! 1. 对每个 segment，比较 source_duration 和 dubb_duration
//! 2. 计算需要的倍率 = dubb_duration / source_duration
//! 3. 根据模式选择策略
//! 4. 对音频应用 atempo 变速（可链式组合，如 2.0 = atempo=1.5,atempo=1.33）
//! 5. 对视频应用 setpts 变速
//! 6. 拼接所有片段并输出

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};
use crate::models::segment::Segment;

/// 声画对齐模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeedRateMode {
    /// 仅音频加速
    AudioSpeedUp,
    /// 仅视频慢放
    VideoSlowDown,
    /// 混合模式（音频+视频各分担一部分）
    #[default]
    Hybrid,
}

/// 声画对齐配置
#[derive(Debug, Clone)]
pub struct SpeedRateConfig {
    /// 对齐模式
    pub mode: SpeedRateMode,
    /// 最大音频加速倍率（超过此值进入 Hybrid 模式的视频慢放部分）
    pub max_audio_speed: f64,
    /// 最大视频慢放倍率
    pub max_video_slow: f64,
    /// 最小处理阈值（倍率差异小于此值不处理）
    pub min_threshold: f64,
}

impl Default for SpeedRateConfig {
    fn default() -> Self {
        Self {
            mode: SpeedRateMode::Hybrid,
            max_audio_speed: 1.3,
            max_video_slow: 2.0,
            min_threshold: 0.05,
        }
    }
}

/// 读取 WAV 文件时长（秒）
///
/// 使用 hound 库读取 WAV 头信息，无需启动 ffmpeg。
/// 用于在 pipeline TTS 阶段快速获取 TTS 音频时长。
fn read_wav_duration(path: &Path) -> AppResult<f64> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| AppError::AudioDecodeError(format!("Failed to open WAV for duration: {e}")))?;
    let spec = reader.spec();
    Ok(reader.duration() as f64 / spec.sample_rate as f64)
}

/// 声画对齐处理器
#[derive(Clone)]
pub struct SpeedRateProcessor {
    config: SpeedRateConfig,
}

impl SpeedRateProcessor {
    /// 创建处理器
    pub fn new(config: SpeedRateConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建
    pub fn default() -> Self {
        Self::new(SpeedRateConfig::default())
    }

    /// 对单个 segment 计算需要的倍率
    ///
    /// 返回值：(音频倍率, 视频倍率)
    /// - 音频倍率 > 1.0 = 加速, < 1.0 = 慢速
    /// - 视频倍率 > 1.0 = 慢放, < 1.0 = 加速
    /// - 倍率 = 1.0 = 不变
    pub fn compute_rates(&self, source_dur: f64, dubb_dur: f64) -> (f64, f64) {
        if source_dur < 0.01 || dubb_dur < 0.01 {
            return (1.0, 1.0);
        }

        let ratio = dubb_dur / source_dur;
        // 倍率 = 配音时长 / 原始时长
        // ratio > 1.0: 配音比原文长，需要加速配音或慢放视频
        // ratio < 1.0: 配音比原文短，需要慢速配音或加速视频

        if (ratio - 1.0).abs() < self.config.min_threshold {
            return (1.0, 1.0);
        }

        match self.config.mode {
            SpeedRateMode::AudioSpeedUp => {
                // 仅音频变速
                let audio_rate = ratio.min(self.config.max_audio_speed);
                (audio_rate, 1.0)
            }
            SpeedRateMode::VideoSlowDown => {
                // 仅视频变速（配音长则视频慢放）
                if ratio > 1.0 {
                    let video_slow = ratio.min(self.config.max_video_slow);
                    (1.0, video_slow)
                } else {
                    // 配音短则视频加速（不太常见）
                    (1.0, 1.0 / ratio)
                }
            }
            SpeedRateMode::Hybrid => {
                if ratio <= self.config.max_audio_speed {
                    // 倍率在音频加速范围内，仅音频加速
                    (ratio, 1.0)
                } else {
                    // 超出音频加速上限，音频加速到上限，剩余通过视频慢放补偿
                    let audio_rate = self.config.max_audio_speed;
                    // 配音加速后剩余的时长 = dubb_dur / audio_rate
                    // 需要视频慢放 = 配音加速后时长 / source_dur
                    let remaining_after_audio = dubb_dur / audio_rate;
                    let video_slow =
                        (remaining_after_audio / source_dur).min(self.config.max_video_slow);
                    (audio_rate, video_slow)
                }
            }
        }
    }

    /// 使用 ffmpeg atempo 变速音频
    ///
    /// atempo 支持范围 0.5-2.0，超出需要链式组合
    /// 例：2.5x = atempo=2.0,atempo=1.25
    /// 例：0.4x = atempo=0.5,atempo=0.8
    pub fn speed_up_audio(input: &Path, output: &Path, rate: f64) -> AppResult<()> {
        if (rate - 1.0).abs() < 0.01 {
            // 不需要变速，直接复制
            std::fs::copy(input, output)?;
            return Ok(());
        }

        let filter = build_atempo_chain(rate);
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .arg("-i")
            .arg(input)
            .arg("-af")
            .arg(&filter)
            .arg("-c:a")
            .arg("pcm_s16le")
            .arg(output);

        let result = cmd.output()?;
        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(AppError::FFmpeg(format!(
                "Audio speed-up failed (rate={rate}): {stderr}"
            )));
        }
        tracing::debug!("Audio sped up by {rate:.3}x with filter: {filter}");
        Ok(())
    }

    /// 使用 ffmpeg setpts 慢放/加速视频
    ///
    /// setpts=PTS/rate: rate>1 慢放, rate<1 加速
    pub fn slow_down_video(input: &Path, output: &Path, rate: f64) -> AppResult<()> {
        if (rate - 1.0).abs() < 0.01 {
            std::fs::copy(input, output)?;
            return Ok(());
        }

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .arg("-i")
            .arg(input)
            .arg("-vf")
            .arg(format!("setpts=PTS/{rate:.6}"))
            .arg("-c:v")
            .arg("libx264")
            .arg("-crf")
            .arg("23")
            .arg("-preset")
            .arg("veryfast")
            .arg("-an")
            .arg(output);

        let result = cmd.output()?;
        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(AppError::FFmpeg(format!(
                "Video slow-down failed (rate={rate}): {stderr}"
            )));
        }
        tracing::debug!("Video slowed down by {rate:.3}x");
        Ok(())
    }

    /// 处理单个 TTS 音频文件：读取时长、计算倍率、应用变速
    ///
    /// 在 pipeline TTS 阶段调用，对每段 TTS 音频进行逐段声画对齐。
    /// 如果音频需要变速（倍率偏离 1.0 超过阈值），使用 ffmpeg atempo 变速，
    /// 结果文件替换原文件。
    ///
    /// # 参数
    /// - `audio_path`: TTS 音频文件路径（WAV 格式，原地修改）
    /// - `source_dur`: 原始 segment 时长（秒），即 `seg.end - seg.start`
    ///
    /// # 返回
    /// 应用的音频变速倍率（1.0 = 未变速）
    pub fn process_audio(&self, audio_path: &Path, source_dur: f64) -> AppResult<f64> {
        let dubb_dur = read_wav_duration(audio_path)?;
        let (audio_rate, _video_rate) = self.compute_rates(source_dur, dubb_dur);

        if (audio_rate - 1.0).abs() >= 0.01 {
            let tmp_path = audio_path.with_extension("speedup_tmp.wav");
            Self::speed_up_audio(audio_path, &tmp_path, audio_rate)?;
            std::fs::rename(&tmp_path, audio_path)?;
            tracing::info!(
                "SpeedRate: segment audio sped up by {audio_rate:.3}x \
                (source={source_dur:.1}s, dubb={dubb_dur:.1}s → {:.1}s)",
                dubb_dur / audio_rate
            );
            Ok(audio_rate)
        } else {
            Ok(1.0)
        }
    }

    /// 处理 segments 列表，返回每个 segment 的倍率信息
    ///
    /// 不执行实际的 ffmpeg 操作，仅计算每个 segment 需要的倍率
    pub fn compute_segment_rates(&self, segments: &[Segment]) -> Vec<SegmentRateInfo> {
        segments
            .iter()
            .map(|seg| {
                let source_dur = seg.end - seg.start;
                let dubb_dur = seg
                    .tts_audio_path
                    .as_ref()
                    .map(|_| {
                        // 这里无法直接获取配音时长，返回占位值
                        // 实际使用时应在 pipeline 中先测量配音时长
                        source_dur
                    })
                    .unwrap_or(source_dur);

                let (audio_rate, video_rate) = self.compute_rates(source_dur, dubb_dur);
                SegmentRateInfo {
                    segment_id: seg.id.clone(),
                    source_duration: source_dur,
                    dubb_duration: dubb_dur,
                    audio_rate,
                    video_rate,
                }
            })
            .collect()
    }
}

/// 单个 segment 的倍率信息
#[derive(Debug, Clone)]
pub struct SegmentRateInfo {
    /// Segment ID
    pub segment_id: String,
    /// 原始时长（秒）
    pub source_duration: f64,
    /// 配音时长（秒）
    pub dubb_duration: f64,
    /// 音频变速倍率（1.0 = 不变）
    pub audio_rate: f64,
    /// 视频变速倍率（1.0 = 不变, >1.0 = 慢放）
    pub video_rate: f64,
}

/// 构建 ffmpeg atempo 滤镜链
///
/// atempo 滤镜支持 0.5-2.0 范围，超出需要链式组合：
/// - 2.5x = atempo=2.0,atempo=1.25
/// - 3.0x = atempo=2.0,atempo=1.5
/// - 0.3x = atempo=0.5,atempo=0.6
fn build_atempo_chain(rate: f64) -> String {
    let mut remaining = rate;
    let mut filters: Vec<String> = Vec::new();

    while remaining > 2.0 {
        filters.push("atempo=2.0".to_string());
        remaining /= 2.0;
    }

    while remaining < 0.5 {
        filters.push("atempo=0.5".to_string());
        remaining /= 0.5;
    }

    // 剩余部分在 [0.5, 2.0] 范围内
    if (remaining - 1.0).abs() > 0.01 {
        filters.push(format!("atempo={remaining:.4}"));
    }

    if filters.is_empty() {
        "anull".to_string()
    } else {
        filters.join(",")
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rates_no_adjustment() {
        let proc = SpeedRateProcessor::default();
        let (audio, video) = proc.compute_rates(5.0, 5.02);
        assert!((audio - 1.0).abs() < 0.01);
        assert!((video - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_rates_audio_speedup_only() {
        let proc = SpeedRateProcessor::default();
        // 配音比原文长 20%
        let (audio, video) = proc.compute_rates(5.0, 6.0);
        // Hybrid 模式下 1.2 < 1.3 max_audio_speed，所以仅音频加速
        assert!((audio - 1.2).abs() < 0.01);
        assert!((video - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_rates_hybrid_mode() {
        let proc = SpeedRateProcessor::default();
        // 配音比原文长 50%，超出 max_audio_speed=1.3
        let (audio, video) = proc.compute_rates(10.0, 15.0);
        // 音频应加速到 max_audio_speed=1.3
        assert!((audio - 1.3).abs() < 0.01);
        // 视频应慢放来补偿剩余
        assert!(video > 1.0);
        // 验证：配音加速后 = 15.0/1.3 = 11.54s
        // 视频慢放 = 11.54/10.0 = 1.154
        assert!((video - 1.154).abs() < 0.01);
    }

    #[test]
    fn test_compute_rates_shorter_dubb() {
        let proc = SpeedRateProcessor::default();
        // 配音比原文短
        let (audio, video) = proc.compute_rates(5.0, 3.0);
        // Hybrid 模式：音频加速 0.6x
        assert!((audio - 0.6).abs() < 0.01);
        assert!((video - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_build_atempo_chain_normal() {
        let chain = build_atempo_chain(1.5);
        assert_eq!(chain, "atempo=1.5000");
    }

    #[test]
    fn test_build_atempo_chain_high() {
        let chain = build_atempo_chain(3.0);
        assert!(chain.contains("atempo=2.0"));
        assert!(chain.contains("atempo=1.5000"));
    }

    #[test]
    fn test_build_atempo_chain_very_high() {
        let chain = build_atempo_chain(5.0);
        // 5.0 = 2.0 * 2.0 * 1.25
        assert!(chain.contains("atempo=2.0"));
        // Should have 3 filters
        assert_eq!(chain.matches("atempo").count(), 3);
    }

    #[test]
    fn test_build_atempo_chain_low() {
        let chain = build_atempo_chain(0.3);
        // 0.3 = 0.5 * 0.6
        assert!(chain.contains("atempo=0.5"));
        assert!(chain.contains("atempo=0.6000"));
    }

    #[test]
    fn test_build_atempo_chain_no_change() {
        let chain = build_atempo_chain(1.0);
        assert_eq!(chain, "anull");
    }

    #[test]
    fn test_compute_rates_audio_only_mode() {
        let proc = SpeedRateProcessor::new(SpeedRateConfig {
            mode: SpeedRateMode::AudioSpeedUp,
            ..Default::default()
        });
        // 配音比原文长 50%
        let (audio, video) = proc.compute_rates(10.0, 15.0);
        // 音频加速到 max_audio_speed=1.3
        assert!((audio - 1.3).abs() < 0.01);
        // 视频不变
        assert!((video - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_rates_video_only_mode() {
        let proc = SpeedRateProcessor::new(SpeedRateConfig {
            mode: SpeedRateMode::VideoSlowDown,
            ..Default::default()
        });
        // 配音比原文长 50%
        let (audio, video) = proc.compute_rates(10.0, 15.0);
        // 音频不变
        assert!((audio - 1.0).abs() < 0.01);
        // 视频慢放 1.5x
        assert!((video - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_segment_rates() {
        let proc = SpeedRateProcessor::default();
        let segments = vec![
            Segment::new("s1".into(), 0.0, 5.0, "hello".into()),
            Segment::new("s2".into(), 5.0, 10.0, "world".into()),
        ];
        let rates = proc.compute_segment_rates(&segments);
        assert_eq!(rates.len(), 2);
        assert_eq!(rates[0].segment_id, "s1");
        assert!((rates[0].source_duration - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_zero_duration() {
        let proc = SpeedRateProcessor::default();
        let (audio, video) = proc.compute_rates(0.0, 5.0);
        assert!((audio - 1.0).abs() < 0.01);
        assert!((video - 1.0).abs() < 0.01);
    }
}
