//! 说话人分离模块
//!
//! 自动识别视频中不同说话人，为每个 Segment 标记 speaker 属性。
//!
//! # 功能概览
//! - [`DiarizationEngine`][] trait：定义说话人分离的标准接口
//! - [`SpeakerSegment`][]：说话人时间片段数据结构
//! - [`DiarizationResult`][]：分离结果，包含所有说话人片段
//! - [`MockDiarizationEngine`][]：用于测试的 Mock 实现
//! - [`assign_speakers_to_segments`][]：将说话人标签映射回流水线 Segment
//!
//! # 引擎实现
//! 目前提供 [`MockDiarizationEngine`] 用于测试和开发。
//! 生产环境中可接入 `speakrs`、`polyvoice`、`pyannote-rs` 等引擎，
//! 通过实现 [`DiarizationEngine`] trait 即可集成。
//!
//! # 性能要求
//! - M1 Pro 上处理 1 小时音频 ≤ 2 分钟
//! - CoreML 加速下 DER（错误率）≤ 7.1%
//!
//! # 示例
//! ```no_run
//! use vt_core::diarization::{DiarizationEngine, MockDiarizationEngine};
//! use std::path::Path;
//!
//! let engine = MockDiarizationEngine::new(2); // 2 个说话人
//! let result = engine.diarize(Path::new("audio.wav"))
//!     .expect("Diarization failed");
//!
//! println!("Found {} speaker segments", result.segments.len());
//! ```

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};
use crate::models::segment::Segment;

// ─── 说话人片段 ───────────────────────────────────────────

/// 说话人时间片段
///
/// 表示某个说话人在一段时间内的语音活动。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerSegment {
    /// 说话人标识（如 `SPEAKER_00`、`SPEAKER_01`）
    pub speaker_id: String,
    /// 片段起始时间（秒）
    pub start: f64,
    /// 片段结束时间（秒）
    pub end: f64,
    /// 置信度（0.0–1.0，可选）
    pub confidence: Option<f64>,
}

impl SpeakerSegment {
    /// 创建新的说话人片段
    ///
    /// # 参数
    /// - `speaker_id`: 说话人标识
    /// - `start`: 起始时间（秒）
    /// - `end`: 结束时间（秒）
    #[must_use]
    pub fn new(speaker_id: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            speaker_id: speaker_id.into(),
            start,
            end,
            confidence: None,
        }
    }

    /// 设置置信度
    ///
    /// # 参数
    /// - `confidence`: 置信度值（0.0–1.0）
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// 检查两个片段是否时间重叠
    ///
    /// # 参数
    /// - `other`: 另一个说话人片段
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// 获取片段时长（秒）
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

// ─── 分离结果 ─────────────────────────────────────────────

/// 说话人分离结果
///
/// 包含所有检测到的说话人片段和说话人总数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationResult {
    /// 所有说话人时间片段（按时间排序）
    pub segments: Vec<SpeakerSegment>,
    /// 检测到的说话人总数
    pub speaker_count: usize,
    /// 处理耗时（秒）
    pub processing_time_secs: f64,
}

impl DiarizationResult {
    /// 创建新的分离结果
    ///
    /// # 参数
    /// - `segments`: 说话人片段列表
    /// - `processing_time_secs`: 处理耗时
    #[must_use]
    pub fn new(segments: Vec<SpeakerSegment>, processing_time_secs: f64) -> Self {
        let speaker_count = segments
            .iter()
            .map(|s| &s.speaker_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        Self {
            segments,
            speaker_count,
            processing_time_secs,
        }
    }

    /// 验证结果：检查片段不重叠且说话人数 ≥ 1
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.segments.is_empty() {
            return false;
        }

        if self.speaker_count == 0 {
            return false;
        }

        // 检查同一说话人的片段不重叠
        let mut sorted = self.segments.clone();
        sorted.sort_by(|a, b| {
            a.speaker_id.cmp(&b.speaker_id).then(
                a.start
                    .partial_cmp(&b.start)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        for window in sorted.windows(2) {
            if window[0].speaker_id == window[1].speaker_id && window[0].overlaps(&window[1]) {
                return false;
            }
        }

        true
    }

    /// 获取指定说话人的所有片段
    #[must_use]
    pub fn segments_for_speaker(&self, speaker_id: &str) -> Vec<&SpeakerSegment> {
        self.segments
            .iter()
            .filter(|s| s.speaker_id == speaker_id)
            .collect()
    }

    /// 获取所有说话人 ID
    #[must_use]
    pub fn speaker_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .segments
            .iter()
            .map(|s| s.speaker_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort();
        ids
    }
}

// ─── DiarizationEngine trait ─────────────────────────────

/// 说话人分离引擎 trait
///
/// 定义说话人分离的标准接口，各引擎实现此 trait 即可集成到流水线中。
///
/// # 实现要求
/// - 输入：16kHz mono WAV 音频文件路径
/// - 输出：[`DiarizationResult`]，包含所有说话人片段
/// - 性能：处理 1 小时音频 ≤ 2 分钟（M1 Pro）
pub trait DiarizationEngine: Send + Sync {
    /// 执行说话人分离
    ///
    /// # 参数
    /// - `audio_path`: 16kHz mono WAV 音频文件路径
    ///
    /// # 错误
    /// - [`AppError::DiarizationError`][]: 分离过程中的错误
    fn diarize(&self, audio_path: &Path) -> AppResult<DiarizationResult>;

    /// 获取引擎名称
    fn name(&self) -> &str;
}

// ─── Mock 引擎 ────────────────────────────────────────────

/// Mock 说话人分离引擎
///
/// 用于测试和开发，根据音频时长生成模拟的说话人片段。
/// 可配置说话人数量和交替模式。
pub struct MockDiarizationEngine {
    /// 模拟的说话人数量
    speaker_count: usize,
    /// 每个说话人片段的时长（秒）
    segment_duration: f64,
}

impl MockDiarizationEngine {
    /// 创建新的 Mock 引擎
    ///
    /// # 参数
    /// - `speaker_count`: 模拟的说话人数量
    #[must_use]
    pub fn new(speaker_count: usize) -> Self {
        Self {
            speaker_count: speaker_count.max(1),
            segment_duration: 5.0,
        }
    }

    /// 设置每个片段的时长
    ///
    /// # 参数
    /// - `duration`: 片段时长（秒）
    #[must_use]
    pub fn with_segment_duration(mut self, duration: f64) -> Self {
        self.segment_duration = duration.max(0.1);
        self
    }

    /// 从 WAV 文件读取音频时长
    fn read_audio_duration(path: &Path) -> AppResult<f64> {
        let reader =
            hound::WavReader::open(path).map_err(|e| AppError::AudioDecodeError(e.to_string()))?;
        let spec = reader.spec();
        let duration = reader.duration() as f64 / spec.sample_rate as f64;
        Ok(duration)
    }
}

impl DiarizationEngine for MockDiarizationEngine {
    fn diarize(&self, audio_path: &Path) -> AppResult<DiarizationResult> {
        let start_time = std::time::Instant::now();

        if !audio_path.exists() {
            return Err(AppError::FileNotFound(audio_path.to_path_buf()));
        }

        let duration = Self::read_audio_duration(audio_path)?;

        // 生成交替的说话人片段
        let mut segments = Vec::new();
        let mut current_time = 0.0;

        while current_time < duration {
            let speaker_idx = segments.len() % self.speaker_count;
            let speaker_id = format!("SPEAKER_{speaker_idx:02}");
            let end_time = (current_time + self.segment_duration).min(duration);

            segments.push(SpeakerSegment::new(speaker_id, current_time, end_time));
            current_time = end_time;
        }

        let processing_time = start_time.elapsed().as_secs_f64();

        Ok(DiarizationResult::new(segments, processing_time))
    }

    fn name(&self) -> &str {
        "mock"
    }
}

impl std::fmt::Debug for MockDiarizationEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockDiarizationEngine")
            .field("speaker_count", &self.speaker_count)
            .field("segment_duration", &self.segment_duration)
            .finish()
    }
}

// ─── 说话人映射 ───────────────────────────────────────────

/// 将说话人标签映射到流水线 Segment
///
/// 根据时间戳重叠，为每个 Segment 分配最匹配的说话人。
///
/// # 算法
/// 对每个 Segment，找到时间重叠最大的说话人片段，
/// 将其 `speaker_id` 赋值给 Segment 的 `speaker` 字段。
///
/// # 参数
/// - `segments`: 流水线 Segment 列表（可变引用，会被修改）
/// - `diarization`: 说话人分离结果
///
/// # 返回
/// 被标记了说话人的 Segment 数量。
pub fn assign_speakers_to_segments(
    segments: &mut [Segment],
    diarization: &DiarizationResult,
) -> usize {
    let mut assigned_count = 0;

    for seg in segments.iter_mut() {
        let mut best_speaker: Option<String> = None;
        let mut best_overlap = 0.0f64;

        for spk_seg in &diarization.segments {
            // 计算时间重叠
            let overlap_start = seg.start.max(spk_seg.start);
            let overlap_end = seg.end.min(spk_seg.end);
            let overlap = (overlap_end - overlap_start).max(0.0);

            if overlap > best_overlap {
                best_overlap = overlap;
                best_speaker = Some(spk_seg.speaker_id.clone());
            }
        }

        if let Some(speaker) = best_speaker {
            seg.speaker = Some(speaker);
            assigned_count += 1;
        }
    }

    assigned_count
}

/// 从说话人片段中提取参考音频
///
/// 取指定说话人最清晰（时长最长）的一段语音作为参考音频。
///
/// # 参数
/// - `diarization`: 说话人分离结果
/// - `speaker_id`: 目标说话人 ID
/// - `audio_path`: 完整音频文件路径
/// - `output_dir`: 参考音频输出目录
///
/// # 返回
/// 提取的参考音频文件路径。
///
/// # 错误
/// - [`AppError::DiarizationError`][]: 找不到该说话人或提取失败
pub fn extract_speaker_reference(
    diarization: &DiarizationResult,
    speaker_id: &str,
    audio_path: &Path,
    output_dir: &Path,
) -> AppResult<PathBuf> {
    let speaker_segments = diarization.segments_for_speaker(speaker_id);

    if speaker_segments.is_empty() {
        return Err(AppError::DiarizationError(format!(
            "No segments found for speaker {speaker_id}"
        )));
    }

    // 找最长的片段作为参考
    let best_segment = speaker_segments
        .iter()
        .max_by(|a, b| {
            a.duration()
                .partial_cmp(&b.duration())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("speaker_segments is non-empty");

    // 确保输出目录存在
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)
            .map_err(|e| AppError::DiarizationError(format!("Failed to create output dir: {e}")))?;
    }

    // 使用 FFmpeg 提取片段
    let output_path = output_dir.join(format!("{speaker_id}_reference.wav"));

    let start = format!("{:.3}", best_segment.start);
    let duration = format!("{:.3}", best_segment.duration());

    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            audio_path.to_str().unwrap_or(""),
            "-ss",
            &start,
            "-t",
            &duration,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            output_path.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(|e| AppError::DiarizationError(format!("FFmpeg failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::DiarizationError(format!(
            "FFmpeg extraction failed: {stderr}"
        )));
    }

    tracing::info!(
        "Extracted reference audio for {speaker_id}: {:?} ({:.1}s)",
        output_path,
        best_segment.duration()
    );

    Ok(output_path)
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 创建测试用 WAV 文件
    fn create_test_wav(path: &Path, duration_secs: f64, sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV");
        let num_samples = (duration_secs * sample_rate as f64) as usize;

        for i in 0..num_samples {
            let sample = ((i as f64 * 0.1).sin() * 16000.0) as i16;
            writer.write_sample(sample).expect("Failed to write sample");
        }

        writer.finalize().expect("Failed to finalize WAV");
    }

    // ── SpeakerSegment 测试 ───────────────────────────

    #[test]
    fn test_speaker_segment_new() {
        let seg = SpeakerSegment::new("SPEAKER_00", 1.0, 5.0);

        assert_eq!(seg.speaker_id, "SPEAKER_00");
        assert!((seg.start - 1.0).abs() < f64::EPSILON);
        assert!((seg.end - 5.0).abs() < f64::EPSILON);
        assert!(seg.confidence.is_none());
    }

    #[test]
    fn test_speaker_segment_with_confidence() {
        let seg = SpeakerSegment::new("SPEAKER_00", 0.0, 5.0).with_confidence(0.85);

        assert_eq!(seg.confidence, Some(0.85));
    }

    #[test]
    fn test_speaker_segment_with_confidence_clamped() {
        let seg = SpeakerSegment::new("SPEAKER_00", 0.0, 5.0).with_confidence(1.5);
        assert_eq!(seg.confidence, Some(1.0));

        let seg = SpeakerSegment::new("SPEAKER_00", 0.0, 5.0).with_confidence(-0.5);
        assert_eq!(seg.confidence, Some(0.0));
    }

    #[test]
    fn test_speaker_segment_overlaps() {
        let seg1 = SpeakerSegment::new("SPEAKER_00", 0.0, 5.0);
        let seg2 = SpeakerSegment::new("SPEAKER_01", 3.0, 8.0);
        let seg3 = SpeakerSegment::new("SPEAKER_01", 5.0, 10.0);

        assert!(seg1.overlaps(&seg2), "0-5 should overlap 3-8");
        assert!(!seg1.overlaps(&seg3), "0-5 should not overlap 5-10");
    }

    #[test]
    fn test_speaker_segment_duration() {
        let seg = SpeakerSegment::new("SPEAKER_00", 1.5, 4.5);
        assert!((seg.duration() - 3.0).abs() < f64::EPSILON);
    }

    // ── DiarizationResult 测试 ────────────────────────

    #[test]
    fn test_diarization_result_new() {
        let segments = vec![
            SpeakerSegment::new("SPEAKER_00", 0.0, 5.0),
            SpeakerSegment::new("SPEAKER_01", 5.0, 10.0),
            SpeakerSegment::new("SPEAKER_00", 10.0, 15.0),
        ];

        let result = DiarizationResult::new(segments, 0.5);

        assert_eq!(result.segments.len(), 3);
        assert_eq!(result.speaker_count, 2);
        assert!((result.processing_time_secs - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diarization_result_is_valid() {
        let segments = vec![
            SpeakerSegment::new("SPEAKER_00", 0.0, 5.0),
            SpeakerSegment::new("SPEAKER_01", 5.0, 10.0),
        ];

        let result = DiarizationResult::new(segments, 0.1);
        assert!(
            result.is_valid(),
            "Non-overlapping segments should be valid"
        );
    }

    #[test]
    fn test_diarization_result_invalid_empty() {
        let result = DiarizationResult::new(Vec::new(), 0.0);
        assert!(!result.is_valid(), "Empty result should be invalid");
    }

    #[test]
    fn test_diarization_result_invalid_overlap() {
        // 同一说话人的重叠片段
        let segments = vec![
            SpeakerSegment::new("SPEAKER_00", 0.0, 5.0),
            SpeakerSegment::new("SPEAKER_00", 3.0, 8.0),
        ];

        let result = DiarizationResult::new(segments, 0.1);
        assert!(
            !result.is_valid(),
            "Overlapping segments for same speaker should be invalid"
        );
    }

    #[test]
    fn test_diarization_result_segments_for_speaker() {
        let segments = vec![
            SpeakerSegment::new("SPEAKER_00", 0.0, 5.0),
            SpeakerSegment::new("SPEAKER_01", 5.0, 10.0),
            SpeakerSegment::new("SPEAKER_00", 10.0, 15.0),
        ];

        let result = DiarizationResult::new(segments, 0.1);

        let speaker_00 = result.segments_for_speaker("SPEAKER_00");
        assert_eq!(speaker_00.len(), 2);

        let speaker_01 = result.segments_for_speaker("SPEAKER_01");
        assert_eq!(speaker_01.len(), 1);

        let speaker_99 = result.segments_for_speaker("SPEAKER_99");
        assert_eq!(speaker_99.len(), 0);
    }

    #[test]
    fn test_diarization_result_speaker_ids() {
        let segments = vec![
            SpeakerSegment::new("SPEAKER_01", 0.0, 5.0),
            SpeakerSegment::new("SPEAKER_00", 5.0, 10.0),
            SpeakerSegment::new("SPEAKER_01", 10.0, 15.0),
        ];

        let result = DiarizationResult::new(segments, 0.1);
        let ids = result.speaker_ids();

        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], "SPEAKER_00"); // 排序后
        assert_eq!(ids[1], "SPEAKER_01");
    }

    // ── MockDiarizationEngine 测试 ────────────────────

    #[test]
    fn test_mock_engine_diarize() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let wav_path = dir.path().join("test.wav");
        create_test_wav(&wav_path, 10.0, 16000);

        let engine = MockDiarizationEngine::new(2);
        let result = engine.diarize(&wav_path).expect("diarize failed");

        assert_eq!(result.speaker_count, 2, "Should detect 2 speakers");
        assert!(!result.segments.is_empty(), "Should have segments");
        assert!(result.is_valid(), "Result should be valid");
    }

    #[test]
    fn test_mock_engine_diarize_single_speaker() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let wav_path = dir.path().join("test.wav");
        create_test_wav(&wav_path, 10.0, 16000);

        let engine = MockDiarizationEngine::new(1);
        let result = engine.diarize(&wav_path).expect("diarize failed");

        assert_eq!(result.speaker_count, 1);
        assert!(result.segments.iter().all(|s| s.speaker_id == "SPEAKER_00"));
    }

    #[test]
    fn test_mock_engine_diarize_file_not_found() {
        let engine = MockDiarizationEngine::new(2);
        let result = engine.diarize(Path::new("/nonexistent/audio.wav"));

        assert!(result.is_err(), "Should error for nonexistent file");
    }

    #[test]
    fn test_mock_engine_diarize_alternating_speakers() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let wav_path = dir.path().join("test.wav");
        create_test_wav(&wav_path, 20.0, 16000);

        let engine = MockDiarizationEngine::new(2).with_segment_duration(5.0);
        let result = engine.diarize(&wav_path).expect("diarize failed");

        // 20 秒音频，每段 5 秒 → 4 段，交替 SPEAKER_00 和 SPEAKER_01
        assert_eq!(result.segments.len(), 4);
        assert_eq!(result.segments[0].speaker_id, "SPEAKER_00");
        assert_eq!(result.segments[1].speaker_id, "SPEAKER_01");
        assert_eq!(result.segments[2].speaker_id, "SPEAKER_00");
        assert_eq!(result.segments[3].speaker_id, "SPEAKER_01");
    }

    #[test]
    fn test_mock_engine_name() {
        let engine = MockDiarizationEngine::new(2);
        assert_eq!(engine.name(), "mock");
    }

    // ── assign_speakers_to_segments 测试 ──────────────

    #[test]
    fn test_assign_speakers_basic() {
        let mut segments = vec![
            Segment::new("seg-0".into(), 0.0, 5.0, "Hello".into()),
            Segment::new("seg-1".into(), 5.0, 10.0, "World".into()),
            Segment::new("seg-2".into(), 10.0, 15.0, "Test".into()),
        ];

        let diarization = DiarizationResult::new(
            vec![
                SpeakerSegment::new("SPEAKER_00", 0.0, 5.0),
                SpeakerSegment::new("SPEAKER_01", 5.0, 10.0),
                SpeakerSegment::new("SPEAKER_00", 10.0, 15.0),
            ],
            0.1,
        );

        let assigned = assign_speakers_to_segments(&mut segments, &diarization);

        assert_eq!(assigned, 3, "All segments should be assigned");
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[2].speaker.as_deref(), Some("SPEAKER_00"));
    }

    #[test]
    fn test_assign_speakers_partial_overlap() {
        let mut segments = vec![Segment::new("seg-0".into(), 2.0, 7.0, "Hello".into())];

        let diarization = DiarizationResult::new(
            vec![
                SpeakerSegment::new("SPEAKER_00", 0.0, 5.0),
                SpeakerSegment::new("SPEAKER_01", 5.0, 10.0),
            ],
            0.1,
        );

        // Segment 2-7 overlaps both, but more with SPEAKER_01 (5-7=2s vs 2-5=3s)
        // Actually: overlap with SPEAKER_00 = 3s (2-5), with SPEAKER_01 = 2s (5-7)
        // So SPEAKER_00 should win
        let assigned = assign_speakers_to_segments(&mut segments, &diarization);

        assert_eq!(assigned, 1);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
    }

    #[test]
    fn test_assign_speakers_no_overlap() {
        let mut segments = vec![Segment::new("seg-0".into(), 0.0, 5.0, "Hello".into())];

        // 说话人片段在 10-20s，Segment 在 0-5s，无重叠
        let diarization =
            DiarizationResult::new(vec![SpeakerSegment::new("SPEAKER_00", 10.0, 20.0)], 0.1);

        let assigned = assign_speakers_to_segments(&mut segments, &diarization);

        assert_eq!(assigned, 0, "No segments should be assigned");
        assert!(segments[0].speaker.is_none());
    }

    #[test]
    fn test_assign_speakers_empty_diarization() {
        let mut segments = vec![Segment::new("seg-0".into(), 0.0, 5.0, "Hello".into())];

        let diarization = DiarizationResult::new(Vec::new(), 0.0);

        let assigned = assign_speakers_to_segments(&mut segments, &diarization);

        assert_eq!(assigned, 0);
        assert!(segments[0].speaker.is_none());
    }

    #[test]
    fn test_assign_speakers_empty_segments() {
        let mut segments: Vec<Segment> = Vec::new();

        let diarization =
            DiarizationResult::new(vec![SpeakerSegment::new("SPEAKER_00", 0.0, 5.0)], 0.1);

        let assigned = assign_speakers_to_segments(&mut segments, &diarization);

        assert_eq!(assigned, 0);
    }

    // ── 集成测试：Mock 引擎 + 说话人映射 ─────────────

    #[test]
    fn test_integration_diarize_and_assign() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let wav_path = dir.path().join("test.wav");
        create_test_wav(&wav_path, 20.0, 16000);

        // 1. 说话人分离
        let engine = MockDiarizationEngine::new(2).with_segment_duration(5.0);
        let result = engine.diarize(&wav_path).expect("diarize failed");

        assert!(
            result.speaker_count >= 2,
            "Should detect at least 2 speakers"
        );

        // 2. 映射到 Segment
        let mut segments: Vec<Segment> = (0..4)
            .map(|i| {
                Segment::new(
                    format!("seg-{i}"),
                    i as f64 * 5.0,
                    (i + 1) as f64 * 5.0,
                    format!("text-{i}"),
                )
            })
            .collect();

        let assigned = assign_speakers_to_segments(&mut segments, &result);

        assert!(assigned > 0, "At least some segments should be assigned");
        assert!(
            segments.iter().all(|s| s.speaker.is_some()),
            "All should have speaker"
        );

        // 3. 验证说话人交替
        assert_eq!(
            segments[0].speaker, segments[2].speaker,
            "Alternating pattern"
        );
        assert_eq!(
            segments[1].speaker, segments[3].speaker,
            "Alternating pattern"
        );
        assert_ne!(
            segments[0].speaker, segments[1].speaker,
            "Different speakers"
        );
    }

    #[test]
    fn test_diarization_result_serde_roundtrip() {
        let result = DiarizationResult::new(
            vec![
                SpeakerSegment::new("SPEAKER_00", 0.0, 5.0).with_confidence(0.9),
                SpeakerSegment::new("SPEAKER_01", 5.0, 10.0),
            ],
            1.23,
        );

        let json = serde_json::to_string(&result).expect("serialize failed");
        let restored: DiarizationResult = serde_json::from_str(&json).expect("deserialize failed");

        assert_eq!(restored.speaker_count, result.speaker_count);
        assert_eq!(restored.segments.len(), result.segments.len());
        assert!((restored.processing_time_secs - result.processing_time_secs).abs() < f64::EPSILON);
    }
}
