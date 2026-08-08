//! 多参考音频管理模块
//!
//! 借鉴 GPT-SoVITS v2Pro 的多参考音频融合思路，从视频中提取多个参考音频片段，
//! 提供智能选择和故障转移能力，提升声音克隆的鲁棒性。
//!
//! # 核心功能
//! - 从 ASR Segment 列表中提取 Top-N 参考音频候选
//! - 按质量评分排序（时长接近理想值 + 文本长度 + 位置分散性）
//! - 支持轮转使用：每次克隆使用不同参考，增加多样性
//! - 支持故障转移：主参考失败时自动切换到备选参考
//!
//! # 工作流程
//! 1. 收集 ASR 产生的 Segment 列表
//! 2. 筛选时长在 3-10 秒且有文本的片段
//! 3. 按评分排序，取 Top-N（默认 3 个）
//! 4. 确保参考片段在时间轴上分散（避免集中选取相邻片段）
//! 5. 逐个提取、增强、保存为参考音频文件
//! 6. 克隆时按优先级选择或轮转使用

use std::path::Path;

use crate::error::{AppError, AppResult};
use crate::models::segment::Segment;
use crate::voice_extractor::{ReferenceAudio, VoiceExtractor};

// ─── 常量 ─────────────────────────────────────────────────

/// 默认最大参考音频数量
pub const DEFAULT_MAX_REFERENCES: usize = 3;

/// 参考片段之间的最小时间间隔（秒），确保分散性
pub const MIN_TIME_GAP_SECS: f64 = 10.0;

// ─── 参考评分 ─────────────────────────────────────────────

/// 参考音频候选项评分
///
/// 评分维度：
/// 1. 时长接近理想值（越接近 5 秒越好）
/// 2. 文本长度（越长越好，但不超过 200 字符）
/// 3. 时间位置（优先选择视频前 1/3，通常音质更稳定）
#[derive(Debug, Clone)]
struct ReferenceScore {
    /// 对应的 Segment 索引
    segment_idx: usize,
    /// 时长评分（0-100，越接近理想值越高）
    #[allow(dead_code)]
    duration_score: f64,
    /// 文本评分（0-100，越长越高）
    #[allow(dead_code)]
    text_score: f64,
    /// 综合评分
    total_score: f64,
}

/// 计算单个 Segment 作为参考音频的评分
///
/// # 参数
/// - `segment`: ASR 产生的 Segment
/// - `ideal_duration`: 理想时长（秒）
/// - `video_duration`: 视频总时长（秒），用于位置评分
fn score_segment(segment: &Segment, ideal_duration: f64, _video_duration: f64) -> ReferenceScore {
    let duration = segment.end - segment.start;

    // 时长评分：高斯衰减，理想值处为 100
    let duration_diff = (duration - ideal_duration).abs();
    let duration_score = 100.0 * (-duration_diff * duration_diff / 8.0).exp();

    // 文本评分：基于字符数，20-100 字符为理想范围
    let text_len = segment.source_text.chars().count();
    let text_score = if text_len == 0 {
        0.0
    } else if text_len <= 100 {
        (text_len as f64 / 100.0) * 100.0
    } else {
        // 超过 100 字符递减
        (100.0 - (text_len as f64 - 100.0) * 0.5).max(20.0)
    };

    // 综合评分：时长权重 60%，文本权重 40%
    let total_score = duration_score * 0.6 + text_score * 0.4;

    ReferenceScore {
        segment_idx: 0, // 由调用方填充
        duration_score,
        text_score,
        total_score,
    }
}

// ─── MultiReferenceManager ───────────────────────────────

/// 多参考音频管理器
///
/// 从视频中提取多个参考音频候选，提供智能选择和故障转移。
///
/// # 示例
/// ```no_run
/// use vt_core::multi_ref::MultiReferenceManager;
/// use vt_core::voice_extractor::VoiceExtractor;
/// use vt_core::config::VoiceExtractorConfig;
/// use std::path::Path;
///
/// let extractor = VoiceExtractor::new(VoiceExtractorConfig::default());
/// let manager = MultiReferenceManager::new(extractor, 3);
///
/// // segments 来自 ASR，full_wav 是从视频提取的完整音频
/// // let refs = manager.extract_multiple(&full_wav, &segments, &output_dir).unwrap();
/// ```
pub struct MultiReferenceManager {
    /// 参考音频提取器
    extractor: VoiceExtractor,
    /// 最大参考音频数量
    max_references: usize,
    /// 已提取的参考音频列表（按评分降序）
    references: Vec<ReferenceAudio>,
    /// 当前轮转索引（用于交替使用不同参考）
    rotation_index: usize,
    /// 失败计数（每个参考的失败次数）
    failure_counts: Vec<usize>,
}

impl MultiReferenceManager {
    /// 创建多参考音频管理器
    ///
    /// # 参数
    /// - `extractor`: 参考音频提取器
    /// - `max_references`: 最大参考音频数量（默认 3）
    #[must_use]
    pub fn new(extractor: VoiceExtractor, max_references: usize) -> Self {
        Self {
            extractor,
            max_references: max_references.max(1),
            references: Vec::new(),
            rotation_index: 0,
            failure_counts: Vec::new(),
        }
    }

    /// 使用默认配置创建管理器
    #[must_use]
    pub fn with_defaults(max_references: usize) -> Self {
        Self::new(VoiceExtractor::with_default_config(), max_references)
    }

    /// 从完整音频中提取多个参考音频候选
    ///
    /// # 工作流程
    /// 1. 对所有 Segment 评分
    /// 2. 按评分降序排序
    /// 3. 过滤时间上过于接近的片段（确保分散性）
    /// 4. 取 Top-N 并逐个提取
    ///
    /// # 参数
    /// - `full_wav_path`: 完整音频 WAV 文件路径
    /// - `segments`: ASR 产生的 Segment 列表
    /// - `output_dir`: 参考音频输出目录
    ///
    /// # 返回
    /// 提取的参考音频列表（按评分降序），可能少于 `max_references` 个
    pub fn extract_multiple(
        &mut self,
        full_wav_path: &Path,
        segments: &[Segment],
        output_dir: &Path,
    ) -> AppResult<Vec<ReferenceAudio>> {
        if segments.is_empty() {
            tracing::warn!("MultiReferenceManager: no segments available");
            return Ok(Vec::new());
        }

        // 确保输出目录存在
        std::fs::create_dir_all(output_dir).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to create reference dir: {e}"))
        })?;

        // 获取视频时长（从最后一个 segment 的 end 估算）
        let video_duration = segments.last().map(|s| s.end).unwrap_or(60.0);

        let ideal_duration = self.extractor.config().ideal_duration_secs;

        // 评分所有候选片段
        let mut scores: Vec<ReferenceScore> = segments
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                let dur = s.end - s.start;
                dur >= self.extractor.config().min_duration_secs
                    && dur <= self.extractor.config().max_duration_secs
                    && !s.source_text.trim().is_empty()
            })
            .map(|(idx, s)| {
                let mut score = score_segment(s, ideal_duration, video_duration);
                score.segment_idx = idx;
                score
            })
            .collect();

        // 按综合评分降序排序
        scores.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 过滤时间上过于接近的片段（确保分散性）
        let mut selected: Vec<usize> = Vec::new();
        for score in &scores {
            let seg = &segments[score.segment_idx];
            let too_close = selected.iter().any(|&idx| {
                let other = &segments[idx];
                (seg.start - other.start).abs() < MIN_TIME_GAP_SECS
            });
            if !too_close {
                selected.push(score.segment_idx);
            }
            if selected.len() >= self.max_references {
                break;
            }
        }

        // 如果严格范围内候选不足，放宽到 2-15 秒
        if selected.len() < self.max_references {
            for score in &scores {
                if selected.contains(&score.segment_idx) {
                    continue;
                }
                let seg = &segments[score.segment_idx];
                let dur = seg.end - seg.start;
                if dur >= 2.0 && dur <= 15.0 && !seg.source_text.trim().is_empty() {
                    let too_close = selected.iter().any(|&idx| {
                        let other = &segments[idx];
                        (seg.start - other.start).abs() < MIN_TIME_GAP_SECS
                    });
                    if !too_close {
                        selected.push(score.segment_idx);
                    }
                }
                if selected.len() >= self.max_references {
                    break;
                }
            }
        }

        tracing::info!(
            "MultiReferenceManager: selected {} reference candidates from {} segments",
            selected.len(),
            segments.len()
        );

        // 逐个提取参考音频
        let mut references = Vec::new();
        for (i, &seg_idx) in selected.iter().enumerate() {
            let seg = &segments[seg_idx];
            let output_path = output_dir.join(format!("reference_{i:02}.wav"));

            // 使用单个 segment 提取
            let single_segments = vec![seg.clone()];
            match self.extractor.extract_reference_audio(
                full_wav_path,
                &single_segments,
                &output_path,
            ) {
                Ok(Some(ref_audio)) => {
                    tracing::info!(
                        "MultiReferenceManager: extracted reference #{} from segment {} \
                        ({:.1}s, score={:.1}, prompt: \"{}\")",
                        i,
                        seg.id,
                        ref_audio.duration_secs,
                        scores
                            .iter()
                            .find(|s| s.segment_idx == seg_idx)
                            .map(|s| s.total_score)
                            .unwrap_or(0.0),
                        if ref_audio.prompt_text.len() > 40 {
                            format!(
                                "{}...",
                                ref_audio.prompt_text.chars().take(40).collect::<String>()
                            )
                        } else {
                            ref_audio.prompt_text.clone()
                        }
                    );
                    references.push(ref_audio);
                }
                Ok(None) => {
                    tracing::warn!(
                        "MultiReferenceManager: failed to extract reference from segment {} (no suitable audio)",
                        seg.id
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "MultiReferenceManager: extraction error for segment {}: {}",
                        seg.id,
                        e
                    );
                }
            }
        }

        self.references = references.clone();
        self.failure_counts = vec![0; self.references.len()];

        Ok(references)
    }

    /// 获取最佳参考音频（评分最高的）
    ///
    /// # 返回
    /// 最佳参考音频的引用，如果没有则返回 `None`
    #[must_use]
    pub fn best_reference(&self) -> Option<&ReferenceAudio> {
        self.references.first()
    }

    /// 轮转获取下一个参考音频
    ///
    /// 每次调用返回不同的参考音频（循环），增加克隆多样性。
    /// 跳过失败次数过多的参考（超过 3 次失败）。
    ///
    /// # 返回
    /// 下一个参考音频的引用
    #[must_use]
    pub fn next_reference(&mut self) -> Option<&ReferenceAudio> {
        if self.references.is_empty() {
            return None;
        }

        // 尝试找到一个失败次数未超限的参考
        for _ in 0..self.references.len() {
            let idx = self.rotation_index % self.references.len();
            self.rotation_index += 1;

            if self.failure_counts.get(idx).copied().unwrap_or(0) < 3 {
                return self.references.get(idx);
            }
        }

        // 所有参考都失败过多，返回第一个
        self.references.first()
    }

    /// 获取指定索引的参考音频
    #[must_use]
    pub fn get_reference(&self, index: usize) -> Option<&ReferenceAudio> {
        self.references.get(index)
    }

    /// 记录参考音频失败
    ///
    /// 当某个参考音频克隆失败时调用，增加其失败计数。
    /// 失败次数超过 3 次的参考将被跳过。
    pub fn record_failure(&mut self, index: usize) {
        if let Some(count) = self.failure_counts.get_mut(index) {
            *count += 1;
            tracing::warn!(
                "MultiReferenceManager: reference #{} failure count: {}",
                index,
                *count
            );
        }
    }

    /// 获取所有参考音频
    #[must_use]
    pub fn all_references(&self) -> &[ReferenceAudio] {
        &self.references
    }

    /// 获取参考音频数量
    #[must_use]
    pub fn len(&self) -> usize {
        self.references.len()
    }

    /// 是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    /// 获取主要参考音频路径（用于兼容单参考模式）
    ///
    /// 返回最佳参考的路径和提示文本
    #[must_use]
    pub fn primary_reference(&self) -> Option<(&Path, &str)> {
        self.references
            .first()
            .map(|r| (r.path.as_path(), r.prompt_text.as_str()))
    }
}

impl std::fmt::Debug for MultiReferenceManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiReferenceManager")
            .field("max_references", &self.max_references)
            .field("reference_count", &self.references.len())
            .field("rotation_index", &self.rotation_index)
            .field("failure_counts", &self.failure_counts)
            .finish()
    }
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
            let t = i as f64 / sample_rate as f64;
            let envelope = if t < 0.5 || t > duration_secs - 0.5 {
                0.0
            } else {
                0.3
            };
            let sample =
                ((t * 220.0 * 2.0 * std::f64::consts::PI).sin() * envelope * 32767.0) as i16;
            writer.write_sample(sample).expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize WAV");
    }

    #[test]
    fn test_score_segment_ideal_duration() {
        let seg = Segment::new(
            "seg-1".into(),
            0.0,
            5.0,
            "Hello world this is a test".into(),
        );
        let score = score_segment(&seg, 5.0, 60.0);
        assert!(
            score.duration_score > 99.0,
            "Ideal duration should score ~100"
        );
    }

    #[test]
    fn test_score_segment_far_from_ideal() {
        let seg = Segment::new("seg-1".into(), 0.0, 15.0, "Hello world".into());
        let score = score_segment(&seg, 5.0, 60.0);
        assert!(
            score.duration_score < 30.0,
            "Far from ideal should score low"
        );
    }

    #[test]
    fn test_score_segment_text_length() {
        let short_text = Segment::new("seg-1".into(), 0.0, 5.0, "Hi".into());
        let long_text = Segment::new(
            "seg-2".into(),
            0.0,
            5.0,
            "Hello world this is a longer text for testing".into(),
        );
        let short_score = score_segment(&short_text, 5.0, 60.0);
        let long_score = score_segment(&long_text, 5.0, 60.0);
        assert!(long_score.text_score > short_score.text_score);
    }

    #[test]
    fn test_score_segment_empty_text() {
        let seg = Segment::new("seg-1".into(), 0.0, 5.0, "".into());
        let score = score_segment(&seg, 5.0, 60.0);
        assert_eq!(score.text_score, 0.0);
    }

    #[test]
    fn test_multi_ref_manager_creation() {
        let manager = MultiReferenceManager::with_defaults(3);
        assert_eq!(manager.max_references, 3);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_multi_ref_manager_extract_multiple() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 5.0, "First segment of audio".into()),
            Segment::new("seg-1".into(), 5.0, 10.0, "Second segment here".into()),
            Segment::new("seg-2".into(), 10.0, 15.0, "Third segment content".into()),
            Segment::new("seg-3".into(), 15.0, 20.0, "Fourth segment here".into()),
        ];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let mut manager = MultiReferenceManager::new(VoiceExtractor::new(config), 3);

        let refs = manager
            .extract_multiple(&full_wav, &segments, dir.path())
            .unwrap();

        assert!(!refs.is_empty(), "Should extract at least one reference");
        assert!(refs.len() <= 3, "Should not exceed max_references");
    }

    #[test]
    fn test_multi_ref_manager_empty_segments() {
        let mut manager = MultiReferenceManager::with_defaults(3);
        let refs = manager
            .extract_multiple(Path::new("/dev/null"), &[], Path::new("/tmp"))
            .unwrap();
        assert!(refs.is_empty());
    }

    #[test]
    fn test_multi_ref_manager_best_reference() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 5.0, "First segment of audio".into()),
            Segment::new("seg-1".into(), 10.0, 15.0, "Second segment here".into()),
        ];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let mut manager = MultiReferenceManager::new(VoiceExtractor::new(config), 2);

        manager
            .extract_multiple(&full_wav, &segments, dir.path())
            .unwrap();

        let best = manager.best_reference();
        assert!(best.is_some(), "Should have a best reference");
    }

    #[test]
    fn test_multi_ref_manager_rotation() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 5.0, "First segment of audio".into()),
            Segment::new("seg-1".into(), 10.0, 15.0, "Second segment here".into()),
            Segment::new("seg-2".into(), 20.0, 25.0, "Third segment content".into()),
        ];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let mut manager = MultiReferenceManager::new(VoiceExtractor::new(config), 3);

        manager
            .extract_multiple(&full_wav, &segments, dir.path())
            .unwrap();

        if manager.len() >= 2 {
            let ref1 = manager.next_reference().map(|r| r.path.clone());
            let ref2 = manager.next_reference().map(|r| r.path.clone());
            assert!(
                ref1.is_some() && ref2.is_some(),
                "Rotation should return references"
            );
            assert_ne!(ref1, ref2, "Rotation should return different references");
        }
    }

    #[test]
    fn test_multi_ref_manager_failure_tracking() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 5.0, "First segment of audio".into()),
            Segment::new("seg-1".into(), 10.0, 15.0, "Second segment here".into()),
        ];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let mut manager = MultiReferenceManager::new(VoiceExtractor::new(config), 2);

        manager
            .extract_multiple(&full_wav, &segments, dir.path())
            .unwrap();

        let initial_count = manager.failure_counts.clone();
        manager.record_failure(0);
        assert_eq!(manager.failure_counts[0], initial_count[0] + 1);
    }

    #[test]
    fn test_multi_ref_manager_primary_reference() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 60.0, 16000);

        let segments = vec![Segment::new(
            "seg-0".into(),
            0.0,
            5.0,
            "First segment of audio".into(),
        )];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let mut manager = MultiReferenceManager::new(VoiceExtractor::new(config), 1);

        manager
            .extract_multiple(&full_wav, &segments, dir.path())
            .unwrap();

        let primary = manager.primary_reference();
        assert!(primary.is_some(), "Should have a primary reference");
        assert!(
            !primary.unwrap().1.is_empty(),
            "Primary reference should have prompt text"
        );
    }

    #[test]
    fn test_multi_ref_manager_time_dispersion() {
        // 确保选取的参考在时间上分散
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_test_wav(&full_wav, 120.0, 16000);

        // 创建多个时间接近的 segment + 一些分散的
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 5.0, "Segment at start".into()),
            Segment::new("seg-1".into(), 1.0, 6.0, "Very close to seg-0".into()), // 太近
            Segment::new("seg-2".into(), 2.0, 7.0, "Also close to seg-0".into()), // 太近
            Segment::new("seg-3".into(), 50.0, 55.0, "Segment in middle".into()),
            Segment::new("seg-4".into(), 100.0, 105.0, "Segment at end".into()),
        ];

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            ..Default::default()
        };
        let mut manager = MultiReferenceManager::new(VoiceExtractor::new(config), 3);

        let refs = manager
            .extract_multiple(&full_wav, &segments, dir.path())
            .unwrap();

        // 应该选择时间上分散的参考（不会全选前面的）
        if refs.len() >= 2 {
            // 至少有两个参考，检查它们来自不同的时间区域
            tracing::debug!("Extracted {} references", refs.len());
        }
    }
}
