//! 集成测试：声音克隆
//!
//! 验证 Mock 声音克隆引擎和流水线集成逻辑。

use std::path::Path;
use vt_core::cloning::{CloningConfig, CloningIntegration, MockCloningEngine, VoiceCloningEngine};
use vt_core::models::segment::Segment;

/// 创建测试用参考音频
fn create_reference_audio(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV");
    for i in 0..16000 * 5 {
        let sample = ((i as f64 * 0.1).sin() * 16000.0) as i16;
        writer.write_sample(sample).expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");
}

/// 验证克隆合成生成可播放的音频文件
#[test]
fn test_clone_and_synthesize_produces_audio() {
    let dir = tempfile::TempDir::new().expect("temp dir");

    let ref_path = dir.path().join("reference.wav");
    create_reference_audio(&ref_path);

    let engine = MockCloningEngine::new();
    let config = CloningConfig {
        output_dir: dir.path().join("output").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let result = engine.clone_and_synthesize(&ref_path, "你好，这是克隆的声音", &config);

    assert!(result.is_ok(), "Should synthesize successfully");
    let output_path = result.unwrap();
    assert!(output_path.exists(), "Output file should exist");

    // 验证 WAV 格式正确
    let reader = hound::WavReader::open(&output_path).expect("Failed to open WAV");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "Should be mono");
    assert_eq!(spec.sample_rate, 24000, "Should be 24kHz");
    assert_eq!(spec.bits_per_sample, 16, "Should be 16-bit");

    // 验证音频有内容
    let duration = reader.duration() as f64 / spec.sample_rate as f64;
    assert!(duration > 0.0, "Audio should have duration > 0");
}

/// 验证批量克隆合成
#[test]
fn test_batch_synthesize() {
    let dir = tempfile::TempDir::new().expect("temp dir");

    let ref_path = dir.path().join("reference.wav");
    create_reference_audio(&ref_path);

    let engine = MockCloningEngine::new();
    let config = CloningConfig {
        output_dir: dir.path().join("output").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let texts = vec![
        "第一段文本".to_string(),
        "第二段文本".to_string(),
        "第三段文本".to_string(),
    ];

    let results = engine
        .clone_and_synthesize_batch(&ref_path, &texts, &config)
        .expect("batch failed");

    assert_eq!(results.len(), 3);
    for path in &results {
        assert!(path.exists(), "Each output should exist");
    }

    // 验证文件名唯一
    let mut paths: Vec<_> = results.iter().collect();
    paths.sort();
    paths.dedup();
    assert_eq!(paths.len(), 3, "All paths should be unique");
}

/// 验证流水线集成：从 Segment 合成克隆语音
#[test]
fn test_pipeline_integration() {
    let dir = tempfile::TempDir::new().expect("temp dir");

    let ref_path = dir.path().join("reference.wav");
    create_reference_audio(&ref_path);

    // 创建已翻译的 Segment
    let mut segment = Segment::new("seg-1".into(), 0.0, 5.0, "Hello world".into());
    segment.start_transcribing().expect("start");
    segment
        .finish_transcribing("你好世界".into())
        .expect("finish");

    let config = CloningConfig {
        output_dir: dir.path().join("output").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let integration = CloningIntegration::new(Box::new(MockCloningEngine::new()), config);

    let result = integration.synthesize_for_segment(&segment, &ref_path);
    assert!(result.is_ok(), "Should synthesize for segment");

    let path = result.unwrap();
    assert!(path.exists(), "Output should exist");
}

/// 验证优雅降级：参考音频不存在时返回 None
#[test]
fn test_graceful_degradation() {
    let _dir = tempfile::TempDir::new().expect("temp dir");

    let mut segment = Segment::new("seg-1".into(), 0.0, 5.0, "Hello".into());
    segment.start_transcribing().expect("start");
    segment.finish_transcribing("你好".into()).expect("finish");

    let integration =
        CloningIntegration::new(Box::new(MockCloningEngine::new()), CloningConfig::default());

    // 使用不存在的参考音频
    let result = integration.try_synthesize(&segment, Path::new("/nonexistent/ref.wav"));

    assert!(result.is_ok(), "try_synthesize should not hard-fail");
    assert!(
        result.unwrap().is_none(),
        "Should return None for graceful degradation"
    );
}

/// 验证不同语速和音调配置
#[test]
fn test_different_configs() {
    let dir = tempfile::TempDir::new().expect("temp dir");

    let ref_path = dir.path().join("reference.wav");
    create_reference_audio(&ref_path);

    let engine = MockCloningEngine::new();

    // 快速高音调
    let fast_config = CloningConfig {
        speed: 2.0,
        pitch_shift: 5.0,
        output_dir: dir.path().join("fast").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let fast_result = engine
        .clone_and_synthesize(&ref_path, "快速测试", &fast_config)
        .expect("fast synthesize failed");
    assert!(fast_result.exists());

    // 慢速低音调
    let slow_config = CloningConfig {
        speed: 0.5,
        pitch_shift: -5.0,
        output_dir: dir.path().join("slow").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let slow_result = engine
        .clone_and_synthesize(&ref_path, "慢速测试", &slow_config)
        .expect("slow synthesize failed");
    assert!(slow_result.exists());
}

/// 验证引擎名称
#[test]
fn test_engine_name() {
    let engine = MockCloningEngine::new();
    assert_eq!(engine.name(), "mock-cloning");
}

/// 验证缺少 target_text 时报错
#[test]
fn test_missing_target_text() {
    let dir = tempfile::TempDir::new().expect("temp dir");

    let ref_path = dir.path().join("reference.wav");
    create_reference_audio(&ref_path);

    // 未翻译的 Segment（没有 target_text）
    let segment = Segment::new("seg-1".into(), 0.0, 5.0, "Hello".into());

    let integration =
        CloningIntegration::new(Box::new(MockCloningEngine::new()), CloningConfig::default());

    let result = integration.synthesize_for_segment(&segment, &ref_path);
    assert!(result.is_err(), "Should fail without target_text");
}
