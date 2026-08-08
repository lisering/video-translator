//! 集成测试：说话人分离
//!
//! 验证 Mock 说话人分离引擎和说话人标签映射逻辑。

use std::path::Path;
use vt_core::diarization::{
    assign_speakers_to_segments, DiarizationEngine, DiarizationResult, MockDiarizationEngine,
    SpeakerSegment,
};
use vt_core::models::segment::Segment;

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

/// 验证 Mock 引擎返回多个说话人
#[test]
fn test_diarize_returns_multiple_speakers() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let wav_path = dir.path().join("test.wav");
    create_test_wav(&wav_path, 20.0, 16000);

    let engine = MockDiarizationEngine::new(2);
    let result = engine.diarize(&wav_path).expect("diarize failed");

    assert!(
        result.speaker_count >= 2,
        "Should detect at least 2 speakers"
    );
}

/// 验证说话人片段时间戳不重叠（同一说话人）
#[test]
fn test_speaker_segments_no_overlap() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let wav_path = dir.path().join("test.wav");
    create_test_wav(&wav_path, 30.0, 16000);

    let engine = MockDiarizationEngine::new(2);
    let result = engine.diarize(&wav_path).expect("diarize failed");

    // 验证同一说话人的片段不重叠
    let speaker_ids = result.speaker_ids();
    for speaker_id in &speaker_ids {
        let segments = result.segments_for_speaker(speaker_id);
        for window in segments.windows(2) {
            assert!(
                !window[0].overlaps(window[1]),
                "Segments for {} should not overlap",
                speaker_id
            );
        }
    }
}

/// 验证说话人标签映射到 Segment
#[test]
fn test_assign_speakers_to_segments() {
    let mut segments = vec![
        Segment::new("seg-0".into(), 0.0, 5.0, "Hello".into()),
        Segment::new("seg-1".into(), 5.0, 10.0, "World".into()),
    ];

    let diarization = DiarizationResult::new(
        vec![
            SpeakerSegment::new("SPEAKER_00", 0.0, 5.0),
            SpeakerSegment::new("SPEAKER_01", 5.0, 10.0),
        ],
        0.1,
    );

    let assigned = assign_speakers_to_segments(&mut segments, &diarization);

    assert_eq!(assigned, 2);
    assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_01"));
}

/// 验证完整流程：分离 + 映射
#[test]
fn test_integration_diarize_and_assign() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let wav_path = dir.path().join("test.wav");
    create_test_wav(&wav_path, 20.0, 16000);

    // 1. 说话人分离
    let engine = MockDiarizationEngine::new(2).with_segment_duration(5.0);
    let result = engine.diarize(&wav_path).expect("diarize failed");

    assert!(result.is_valid(), "Diarization result should be valid");

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

    assert!(assigned > 0, "Should assign speakers to segments");
    assert!(
        segments.iter().all(|s| s.speaker.is_some()),
        "All segments should have speaker"
    );

    // 3. 验证说话人交替
    assert_ne!(
        segments[0].speaker, segments[1].speaker,
        "Adjacent segments should have different speakers"
    );
}

/// 验证单说话人场景
#[test]
fn test_single_speaker() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let wav_path = dir.path().join("test.wav");
    create_test_wav(&wav_path, 10.0, 16000);

    let engine = MockDiarizationEngine::new(1);
    let result = engine.diarize(&wav_path).expect("diarize failed");

    assert_eq!(result.speaker_count, 1);
    assert!(result.segments.iter().all(|s| s.speaker_id == "SPEAKER_00"));
}

/// 验证结果序列化
#[test]
fn test_result_serde() {
    let result = DiarizationResult::new(
        vec![
            SpeakerSegment::new("SPEAKER_00", 0.0, 5.0).with_confidence(0.9),
            SpeakerSegment::new("SPEAKER_01", 5.0, 10.0),
        ],
        1.5,
    );

    let json = serde_json::to_string(&result).expect("serialize");
    let restored: DiarizationResult = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(restored.speaker_count, 2);
    assert_eq!(restored.segments.len(), 2);
}
