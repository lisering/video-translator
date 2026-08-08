//! 集成测试：流水线引擎 — ASR → 翻译 → TTS 三阶段并行编排
//!
//! 验证 `Pipeline` 的端到端处理、背压控制和错误恢复能力。
//!
//! # 测试策略
//! - 使用 Mock 引擎替代真实的 ASR/翻译/TTS，避免外部依赖
//! - 使用 ffmpeg 生成测试视频（testsrc + sine 波）
//! - `MockAsrEngine`：读取 WAV 时长，生成带 `source_text` 的 Segment
//! - `MockTranslationProvider`：简单文本替换（前缀 "翻译:"）
//! - `MockTtsEngine`：生成 16kHz mono WAV 文件（正弦波）
//!
//! # 优化说明（Session 11）
//! - **测试视频瘦身**：使用 3s 视频（而非 10s），大幅减少 ffmpeg 编码时间。
//! - **背压测试缩短**：使用 4s 视频 + 20ms 延迟（而非 10s + 50ms）。
//! - 使用共享辅助函数（`common::generate_test_video`）避免重复代码。
//!
//! # 运行方式
//! ```sh
//! cargo test test_pipeline -- --nocapture
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tempfile::TempDir;
use vt_core::asr::{read_wav_mono, AsrEngine};
use vt_core::config::{Config, PipelineConfig, TtsConfig};
use vt_core::error::AppResult;
use vt_core::media::FfmpegAudioExtractor;
use vt_core::models::segment::{Segment, SegmentStatus};
use vt_core::pipeline::PipelineBuilder;
use vt_core::translate::TranslationProvider;
use vt_core::tts::TtsEngine;

// ═══════════════════════════════════════════════════════════
//  Mock 引擎实现
// ═══════════════════════════════════════════════════════════

/// Mock ASR 引擎：读取 WAV 时长，生成带 `source_text` 的 Segment
struct MockAsrEngine {
    segments_per_chunk: usize,
    counter: AtomicUsize,
    delay: Option<Duration>,
}

impl MockAsrEngine {
    fn new(segments_per_chunk: usize) -> Self {
        Self {
            segments_per_chunk,
            counter: AtomicUsize::new(0),
            delay: None,
        }
    }
}

impl AsrEngine for MockAsrEngine {
    fn transcribe(&self, audio_path: &Path) -> AppResult<Vec<Segment>> {
        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        }

        let (samples, sample_rate) = read_wav_mono(audio_path)?;
        let duration = samples.len() as f64 / sample_rate as f64;

        let seg_duration = duration / self.segments_per_chunk as f64;
        let chunk_idx = self.counter.fetch_add(1, Ordering::SeqCst);

        let segments: Vec<Segment> = (0..self.segments_per_chunk)
            .map(|i| {
                let global_idx = chunk_idx * self.segments_per_chunk + i;
                Segment::new(
                    format!("mock-seg-{global_idx:04}"),
                    i as f64 * seg_duration,
                    (i + 1) as f64 * seg_duration,
                    format!("Mock speech segment number {global_idx}"),
                )
            })
            .collect();

        Ok(segments)
    }
}

/// Mock 翻译提供者：简单文本替换
struct MockTranslationProvider;

impl TranslationProvider for MockTranslationProvider {
    fn translate_batch(
        &self,
        segments: &mut [Segment],
        _source_lang: &str,
        _target_lang: &str,
    ) -> AppResult<()> {
        for seg in segments.iter_mut() {
            seg.target_text = Some(format!("翻译: {}", seg.source_text));
        }
        Ok(())
    }
}

/// 失败翻译提供者：每隔 N 个 Segment 失败一次
struct FailingTranslationProvider {
    fail_every: usize,
}

impl TranslationProvider for FailingTranslationProvider {
    fn translate_batch(
        &self,
        segments: &mut [Segment],
        _source_lang: &str,
        _target_lang: &str,
    ) -> AppResult<()> {
        use vt_core::error::AppError;
        for (i, seg) in segments.iter_mut().enumerate() {
            if i % self.fail_every == 0 {
                return Err(AppError::TranslationError(format!(
                    "Mock failure for segment {}",
                    seg.id
                )));
            }
            seg.target_text = Some(format!("翻译: {}", seg.source_text));
        }
        Ok(())
    }
}

/// Mock TTS 引擎：生成 16kHz mono WAV 文件（正弦波）
struct MockTtsEngine {
    output_dir: PathBuf,
    delay: Option<Duration>,
}

impl MockTtsEngine {
    fn new(output_dir: impl AsRef<Path>) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            delay: None,
        }
    }

    #[allow(dead_code)]
    fn with_delay(output_dir: impl AsRef<Path>, delay: Duration) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            delay: Some(delay),
        }
    }
}

impl TtsEngine for MockTtsEngine {
    fn synthesize_segments(
        &self,
        segments: &mut [Segment],
        _config: &TtsConfig,
    ) -> AppResult<Vec<PathBuf>> {
        use vt_core::error::AppError;

        if segments.is_empty() {
            return Ok(Vec::new());
        }

        for seg in segments.iter() {
            match &seg.target_text {
                None => {
                    return Err(AppError::TtsError(format!(
                        "Segment {} has no target_text",
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

        for seg in segments.iter_mut() {
            seg.start_synthesizing()?;
        }

        let mut paths = Vec::with_capacity(segments.len());
        for seg in segments.iter_mut() {
            if let Some(delay) = self.delay {
                std::thread::sleep(delay);
            }

            let path = self.output_dir.join(format!("{}.wav", seg.id));

            // 生成 0.5 秒的 440Hz 正弦波 WAV（缩短自 1 秒，加速测试）
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&path, spec)
                .map_err(|e| AppError::TtsError(format!("Failed to create WAV: {e}")))?;

            for i in 0..8000 {
                let sample = ((i as f64 * 440.0 * 2.0 * std::f64::consts::PI / 16000.0).sin()
                    * 0.5
                    * 32767.0) as i16;
                writer
                    .write_sample(sample)
                    .map_err(|e| AppError::TtsError(format!("Failed to write sample: {e}")))?;
            }
            writer
                .finalize()
                .map_err(|e| AppError::TtsError(format!("Failed to finalize WAV: {e}")))?;

            seg.finish_synthesizing(path.to_string_lossy().to_string())?;
            paths.push(path);
        }

        Ok(paths)
    }

    fn list_voices(&self) -> Vec<vt_core::voice_manager::VoiceInfo> {
        Vec::new()
    }
}

// ═══════════════════════════════════════════════════════════
//  测试辅助函数
// ═══════════════════════════════════════════════════════════

/// 创建测试用 Config
fn make_test_config(
    cache_dir: &TempDir,
    channel_capacity: usize,
    segment_duration_secs: f64,
) -> Config {
    Config {
        tts: TtsConfig {
            speed: 1.0,
            voice: "zh-CN-XiaoxiaoNeural".to_string(),
            cache_dir: cache_dir.path().to_string_lossy().to_string(),
            parallel_tasks: 1,
            ..Default::default()
        },
        pipeline: PipelineConfig {
            segment_duration_secs,
            channel_capacity,
            enable_vad_split: false,
        },
        ..Config::default()
    }
}

// ═══════════════════════════════════════════════════════════
//  端到端测试
// ═══════════════════════════════════════════════════════════

/// 验证完整流水线：视频 → 音频提取 → ASR → 翻译 → TTS → Segment 列表
///
/// 优化点：使用 3s 视频（缩短自 10s）。
#[tokio::test]
async fn test_pipeline_end_to_end() {
    if !common::ffmpeg_available() {
        eprintln!("Skipping: ffmpeg not found in PATH");
        return;
    }

    let dir = TempDir::new().expect("Failed to create temp dir");
    let tts_dir = TempDir::new().expect("Failed to create TTS output dir");
    let video_path = common::generate_test_video(&dir, "test_input.mp4", 3);

    let pipeline = PipelineBuilder::default()
        .asr_engine(MockAsrEngine::new(2))
        .translation_provider(MockTranslationProvider)
        .tts_engine(MockTtsEngine::new(tts_dir.path()))
        .audio_extractor(FfmpegAudioExtractor::new())
        .build()
        .expect("Failed to build pipeline");

    let config = make_test_config(&dir, 100, 2.0);

    let segments = pipeline
        .process_video(&video_path, &config)
        .await
        .expect("Pipeline processing failed");

    assert!(
        !segments.is_empty(),
        "Pipeline should produce at least one segment"
    );

    eprintln!("End-to-end test: {} segments produced", segments.len());

    for (i, seg) in segments.iter().enumerate() {
        eprintln!(
            "  [{}] {:.2}s-{:.2}s: source='{}' target='{}' audio='{}'",
            seg.id,
            seg.start,
            seg.end,
            seg.source_text,
            seg.target_text.as_deref().unwrap_or("(none)"),
            seg.tts_audio_path.as_deref().unwrap_or("(none)"),
        );

        assert!(
            !seg.source_text.is_empty(),
            "Segment {i} source_text should not be empty"
        );

        let target = seg
            .target_text
            .as_ref()
            .unwrap_or_else(|| panic!("Segment {i} target_text should be Some"));
        assert!(
            !target.is_empty(),
            "Segment {i} target_text should not be empty"
        );

        let audio_path = seg
            .tts_audio_path
            .as_ref()
            .unwrap_or_else(|| panic!("Segment {i} tts_audio_path should be Some"));
        assert!(
            !audio_path.is_empty(),
            "Segment {i} tts_audio_path should not be empty"
        );

        let audio_path = Path::new(audio_path);
        assert!(
            audio_path.exists(),
            "Segment {i} audio file should exist: {audio_path:?}"
        );

        let reader = hound::WavReader::open(audio_path)
            .unwrap_or_else(|e| panic!("Segment {i} WAV file invalid: {e}"));
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16000, "Segment {i} sample rate");
        assert_eq!(spec.channels, 1, "Segment {i} channels");
        assert_eq!(spec.bits_per_sample, 16, "Segment {i} bits per sample");

        assert_eq!(
            seg.status,
            SegmentStatus::Completed,
            "Segment {i} status should be Completed"
        );

        assert!(seg.start >= 0.0, "Segment {i} start should be >= 0");
        assert!(seg.end > seg.start, "Segment {i} end should be > start");
    }

    // 验证时间戳连续无重叠
    for i in 1..segments.len() {
        assert!(
            segments[i].start >= segments[i - 1].end
                || (segments[i].start - segments[i - 1].end).abs() < 0.01,
            "Segment {} start ({:.3}) should be >= Segment {} end ({:.3})",
            i,
            segments[i].start,
            i - 1,
            segments[i - 1].end
        );
    }
}

// ═══════════════════════════════════════════════════════════
//  背压控制测试
// ═══════════════════════════════════════════════════════════

/// 验证背压控制：使用有界通道和慢速 TTS，确认管道不会无限堆积。
///
/// 优化点：使用 4s 视频 + 20ms 延迟（缩短自 10s + 50ms）。
#[tokio::test]
async fn test_pipeline_backpressure() {
    if !common::ffmpeg_available() {
        eprintln!("Skipping: ffmpeg not found in PATH");
        return;
    }

    let dir = TempDir::new().expect("Failed to create temp dir");
    let tts_dir = TempDir::new().expect("Failed to create TTS output dir");
    // 4s 视频，2s 分割 → 2 个 chunk
    let video_path = common::generate_test_video(&dir, "bp_input.mp4", 4);

    let pipeline = PipelineBuilder::default()
        .asr_engine(MockAsrEngine::new(3))
        .translation_provider(MockTranslationProvider)
        .tts_engine(MockTtsEngine::with_delay(
            tts_dir.path(),
            Duration::from_millis(20),
        ))
        .audio_extractor(FfmpegAudioExtractor::new())
        .build()
        .expect("Failed to build pipeline");

    let config = make_test_config(&dir, 2, 2.0);

    let start = std::time::Instant::now();
    let segments = pipeline
        .process_video(&video_path, &config)
        .await
        .expect("Pipeline processing failed");
    let elapsed = start.elapsed();

    eprintln!(
        "Backpressure test: {} segments in {:?} (channel_capacity=2, tts_delay=20ms)",
        segments.len(),
        elapsed
    );

    assert!(
        !segments.is_empty(),
        "Pipeline should produce segments despite backpressure"
    );

    for seg in &segments {
        assert_eq!(
            seg.status,
            SegmentStatus::Completed,
            "Segment {} should be Completed",
            seg.id
        );
        assert!(
            seg.tts_audio_path.is_some(),
            "Segment {} should have tts_audio_path",
            seg.id
        );
    }

    // 2 chunks × 3 segments = 6 segments, each 20ms → 至少 6 × 20ms = 120ms
    let expected_min = Duration::from_millis(100);
    assert!(
        elapsed >= expected_min,
        "Pipeline with slow TTS should take at least {:?}, got {:?}",
        expected_min,
        elapsed
    );
}

// ═══════════════════════════════════════════════════════════
//  错误恢复测试
// ═══════════════════════════════════════════════════════════

/// 验证错误恢复：翻译阶段部分失败时，已完成的 Segment 不受影响。
///
/// 优化点：使用 3s 视频（缩短自 10s）。
#[tokio::test]
async fn test_pipeline_error_recovery() {
    if !common::ffmpeg_available() {
        eprintln!("Skipping: ffmpeg not found in PATH");
        return;
    }

    let dir = TempDir::new().expect("Failed to create temp dir");
    let tts_dir = TempDir::new().expect("Failed to create TTS output dir");
    let video_path = common::generate_test_video(&dir, "err_input.mp4", 3);

    let pipeline = PipelineBuilder::default()
        .asr_engine(MockAsrEngine::new(3))
        .translation_provider(FailingTranslationProvider { fail_every: 3 })
        .tts_engine(MockTtsEngine::new(tts_dir.path()))
        .audio_extractor(FfmpegAudioExtractor::new())
        .build()
        .expect("Failed to build pipeline");

    let config = make_test_config(&dir, 100, 2.0);

    let result = pipeline.process_video(&video_path, &config).await;

    assert!(
        result.is_ok(),
        "Pipeline should succeed with partial failures, got error: {:?}",
        result.err()
    );

    let segments = result.expect("Should have segments");

    eprintln!(
        "Error recovery test: {} segments succeeded (some may have been skipped)",
        segments.len()
    );

    for seg in &segments {
        assert_eq!(
            seg.status,
            SegmentStatus::Completed,
            "Returned segment {} should be Completed",
            seg.id
        );
        assert!(
            seg.target_text.is_some(),
            "Returned segment {} should have target_text",
            seg.id
        );
        assert!(
            seg.tts_audio_path.is_some(),
            "Returned segment {} should have tts_audio_path",
            seg.id
        );
    }
}

// ═══════════════════════════════════════════════════════════
//  PipelineConfig 默认值测试
// ═══════════════════════════════════════════════════════════

/// 验证 PipelineConfig 的默认值。
#[test]
fn test_pipeline_config_default() {
    let config = PipelineConfig::default();
    assert_eq!(config.segment_duration_secs, 30.0);
    assert_eq!(config.channel_capacity, 100);
    assert!(config.enable_vad_split);
}

/// 验证从 TOML 加载 PipelineConfig。
#[test]
fn test_pipeline_config_from_toml() {
    use std::io::Write;

    let toml_content = r#"
[pipeline]
segment_duration_secs = 15.0
channel_capacity = 50
enable_vad_split = false
"#;

    let mut tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{toml_content}").expect("Failed to write TOML");

    let config = Config::from_file(tmp.path()).expect("Failed to load config");

    assert_eq!(config.pipeline.segment_duration_secs, 15.0);
    assert_eq!(config.pipeline.channel_capacity, 50);
    assert!(!config.pipeline.enable_vad_split);
}

/// 验证缺少 [pipeline] 段时使用默认值。
#[test]
fn test_pipeline_config_default_from_toml() {
    use std::io::Write;

    let toml_content = r#"
output_dir = "/tmp/output"
"#;

    let mut tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{toml_content}").expect("Failed to write TOML");

    let config = Config::from_file(tmp.path()).expect("Failed to load config");

    assert_eq!(config.pipeline.segment_duration_secs, 30.0);
    assert_eq!(config.pipeline.channel_capacity, 100);
    assert!(config.pipeline.enable_vad_split);
}
