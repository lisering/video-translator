//! 集成测试：Segment 数据模型
//!
//! 验证 Segment 的 JSON 序列化/反序列化以及状态机转换逻辑。

use vt_core::models::segment::{Segment, SegmentStatus};

/// 验证 Segment 的 JSON 序列化与反序列化往返（round-trip）正确。
#[test]
fn test_segment_serialization() {
    let segment = Segment::new("seg-001".to_string(), 0.0, 5.5, "Hello, world!".to_string());

    // 序列化为 JSON
    let json = serde_json::to_string(&segment).unwrap();
    assert!(!json.is_empty());

    // 反序列化回来
    let deserialized: Segment = serde_json::from_str(&json).unwrap();

    // 验证所有字段一致
    assert_eq!(deserialized.id, "seg-001");
    assert_eq!(deserialized.start, 0.0);
    assert_eq!(deserialized.end, 5.5);
    assert_eq!(deserialized.source_text, "Hello, world!");
    assert_eq!(deserialized.status, SegmentStatus::Pending);
    assert!(deserialized.speaker.is_none());
    assert!(deserialized.target_text.is_none());
    assert!(deserialized.tts_audio_path.is_none());
}

/// 验证 Segment 带可选项的序列化/反序列化。
#[test]
fn test_segment_serialization_with_options() {
    let mut segment = Segment::new(
        "seg-002".to_string(),
        10.0,
        20.5,
        "Rust is awesome".to_string(),
    );
    segment.speaker = Some("Speaker A".to_string());
    segment.target_text = Some("Rust 太棒了".to_string());
    segment.tts_audio_path = Some("/tmp/audio/seg-002.wav".to_string());
    segment.status = SegmentStatus::Completed;

    let json = serde_json::to_string_pretty(&segment).unwrap();
    let deserialized: Segment = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.speaker, Some("Speaker A".to_string()));
    assert_eq!(deserialized.target_text, Some("Rust 太棒了".to_string()));
    assert_eq!(
        deserialized.tts_audio_path,
        Some("/tmp/audio/seg-002.wav".to_string())
    );
    assert_eq!(deserialized.status, SegmentStatus::Completed);
}

/// 验证 Segment 状态机的正常转换流程：
/// Pending → Transcribing → Translated → Synthesizing → Completed
#[test]
fn test_segment_status_transition() {
    let mut segment = Segment::new("seg-003".to_string(), 0.0, 3.0, "Test text".to_string());

    // 初始状态为 Pending
    assert_eq!(segment.status, SegmentStatus::Pending);

    // Pending → Transcribing
    segment.start_transcribing().unwrap();
    assert_eq!(segment.status, SegmentStatus::Transcribing);

    // Transcribing → Translated（附带目标文本）
    segment.finish_transcribing("测试文本".to_string()).unwrap();
    assert_eq!(segment.status, SegmentStatus::Translated);
    assert_eq!(segment.target_text, Some("测试文本".to_string()));

    // Translated → Synthesizing
    segment.start_synthesizing().unwrap();
    assert_eq!(segment.status, SegmentStatus::Synthesizing);

    // Synthesizing → Completed（附带音频路径）
    segment
        .finish_synthesizing("/tmp/audio/seg-003.wav".to_string())
        .unwrap();
    assert_eq!(segment.status, SegmentStatus::Completed);
    assert_eq!(
        segment.tts_audio_path,
        Some("/tmp/audio/seg-003.wav".to_string())
    );
}

/// 验证任意状态都可以转换为 Failed。
#[test]
fn test_segment_status_transition_to_failed() {
    let mut segment = Segment::new("seg-004".to_string(), 0.0, 3.0, "Will fail".to_string());

    // Pending → Failed
    segment.fail().unwrap();
    assert_eq!(segment.status, SegmentStatus::Failed);

    // 从 Transcribing 状态也可以 Failed
    let mut segment2 = Segment::new("seg-005".to_string(), 0.0, 3.0, "Will fail too".to_string());
    segment2.start_transcribing().unwrap();
    segment2.fail().unwrap();
    assert_eq!(segment2.status, SegmentStatus::Failed);
}

/// 验证非法状态转换会返回错误。
#[test]
fn test_segment_invalid_status_transition() {
    let mut segment = Segment::new(
        "seg-006".to_string(),
        0.0,
        3.0,
        "Invalid transition".to_string(),
    );

    // 不能从 Pending 直接跳到 Translated
    let result = segment.finish_transcribing("跳过步骤".to_string());
    assert!(result.is_err());

    // 不能从 Pending 直接跳到 Synthesizing
    let result = segment.start_synthesizing();
    assert!(result.is_err());

    // 状态应该仍然是 Pending（转换失败不改变状态）
    assert_eq!(segment.status, SegmentStatus::Pending);
}

/// 验证 Segment 的 Default 实现。
#[test]
fn test_segment_default() {
    let segment = Segment::default();
    assert_eq!(segment.id, "");
    assert_eq!(segment.start, 0.0);
    assert_eq!(segment.end, 0.0);
    assert!(segment.speaker.is_none());
    assert_eq!(segment.source_text, "");
    assert!(segment.target_text.is_none());
    assert!(segment.tts_audio_path.is_none());
    assert_eq!(segment.status, SegmentStatus::Pending);
}

/// 验证 SegmentStatus 的 Default 实现。
#[test]
fn test_segment_status_default() {
    let status = SegmentStatus::default();
    assert_eq!(status, SegmentStatus::Pending);
}
