//! 声音克隆集成测试
//!
//! 测试 VoiceExtractor、SubprocessCloneEngine 和相关配置的完整功能。
//!
//! # 测试覆盖
//! - VoiceExtractor 参考音频提取（含静音修剪、归一化）
//! - SubprocessCloneEngine 引擎创建和预设参数
//! - VoiceExtractorConfig 序列化/反序列化
//! - CloningConfig 扩展字段验证
//! - 端到端流程测试

use std::path::{Path, PathBuf};

use vt_core::cloning::{
    CloningConfig, CloningIntegration, MockCloningEngine, SubprocessCloneEngine, VoiceCloningEngine,
};
use vt_core::config::{CloningConfig as CloningEngineConfig, VoiceExtractorConfig};
use vt_core::models::segment::Segment;
use vt_core::voice_extractor::{ReferenceAudio, VoiceExtractor};

// ─── 测试工具函数 ─────────────────────────────────────────

/// 创建测试用 WAV 文件（模拟从视频提取的音频）
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
        // 模拟语音信号：有静音段 + 语音段
        let envelope = if t < 0.5 || (t > 3.0 && t < 4.0) || t > duration_secs - 0.5 {
            0.0 // 静音段
        } else {
            0.3 * (1.0 + (t * 3.0).sin() * 0.2) // 语音段（有幅度变化）
        };
        let sample = ((t * 220.0 * 2.0 * std::f64::consts::PI).sin() * envelope * 32767.0) as i16;
        writer.write_sample(sample).expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");
}

/// 创建简短参考音频（用于克隆引擎测试）
fn create_reference_audio(path: &Path) {
    create_test_wav(path, 5.0, 16000);
}

// ─── VoiceExtractor 集成测试 ─────────────────────────────

#[test]
fn test_voice_extractor_full_extraction_flow() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    // 1. 创建完整音频（30 秒）
    let full_wav = dir.path().join("full_audio.wav");
    create_test_wav(&full_wav, 30.0, 16000);

    // 2. 创建 ASR Segment
    let segments = vec![
        Segment::new("seg-0".into(), 0.0, 3.0, "Hello world".into()),
        Segment::new(
            "seg-1".into(),
            3.0,
            8.0,
            "Welcome to this video tutorial".into(),
        ),
        Segment::new("seg-2".into(), 8.0, 15.0, "Let's begin".into()),
    ];

    // 3. 配置 VoiceExtractor（禁用 ffmpeg 增强）
    let config = VoiceExtractorConfig {
        enable_enhancement: false,
        enable_silence_trim: true,
        enable_normalization: true,
        ..Default::default()
    };
    let extractor = VoiceExtractor::new(config);

    // 4. 提取参考音频
    let ref_output = dir.path().join("reference.wav");
    let result = extractor
        .extract_reference_audio(&full_wav, &segments, &ref_output)
        .expect("extraction failed");

    // 5. 验证结果
    assert!(result.is_some(), "Should find a suitable segment");
    let ref_audio = result.unwrap();
    assert!(ref_output.exists(), "Output file should exist");
    assert!(
        !ref_audio.prompt_text.is_empty(),
        "prompt_text should not be empty"
    );
    assert!(ref_audio.duration_secs > 0.0, "Duration should be positive");
    assert_eq!(ref_audio.sample_rate, 16000);

    // 6. 验证 WAV 文件可读取
    let reader = hound::WavReader::open(&ref_output).expect("Failed to open WAV");
    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.spec().sample_rate, 16000);
}

#[test]
fn test_voice_extractor_selects_ideal_duration() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let full_wav = dir.path().join("full.wav");
    create_test_wav(&full_wav, 60.0, 16000);

    // 多个候选，理想时长 5 秒
    let segments = vec![
        Segment::new("seg-0".into(), 0.0, 8.0, "Eight seconds".into()), // 8s
        Segment::new("seg-1".into(), 8.0, 13.0, "Five seconds".into()), // 5s ← best
        Segment::new("seg-2".into(), 13.0, 16.0, "Three seconds".into()), // 3s
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

    assert!(result.is_some());
    let ref_audio = result.unwrap();
    assert_eq!(ref_audio.prompt_text, "Five seconds");
}

#[test]
fn test_voice_extractor_silence_trim_reduces_duration() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let full_wav = dir.path().join("full.wav");
    create_test_wav(&full_wav, 30.0, 16000);

    // 选择一个有首尾静音的片段
    let segments = vec![Segment::new("seg-0".into(), 0.0, 5.0, "Has silence".into())];

    // 启用静音修剪
    let config_with_trim = VoiceExtractorConfig {
        enable_enhancement: false,
        enable_silence_trim: true,
        enable_normalization: false,
        ..Default::default()
    };
    let extractor_trim = VoiceExtractor::new(config_with_trim);

    let ref_trimmed = dir.path().join("ref_trimmed.wav");
    let result_trimmed = extractor_trim
        .extract_reference_audio(&full_wav, &segments, &ref_trimmed)
        .unwrap()
        .unwrap();

    // 禁用静音修剪
    let config_no_trim = VoiceExtractorConfig {
        enable_enhancement: false,
        enable_silence_trim: false,
        enable_normalization: false,
        ..Default::default()
    };
    let extractor_no_trim = VoiceExtractor::new(config_no_trim);

    let ref_not_trimmed = dir.path().join("ref_not_trimmed.wav");
    let result_not_trimmed = extractor_no_trim
        .extract_reference_audio(&full_wav, &segments, &ref_not_trimmed)
        .unwrap()
        .unwrap();

    // 修剪后应 <= 未修剪
    assert!(
        result_trimmed.duration_secs <= result_not_trimmed.duration_secs,
        "Trimmed duration ({:.2}s) should be <= untrimmed ({:.2}s)",
        result_trimmed.duration_secs,
        result_not_trimmed.duration_secs
    );
}

#[test]
fn test_voice_extractor_normalization_produces_valid_audio() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let full_wav = dir.path().join("full.wav");
    create_test_wav(&full_wav, 30.0, 16000);

    let segments = vec![Segment::new("seg-0".into(), 0.0, 5.0, "Test text".into())];

    let config = VoiceExtractorConfig {
        enable_enhancement: false,
        enable_silence_trim: false,
        enable_normalization: true,
        target_rms_db: -15.0,
        ..Default::default()
    };
    let extractor = VoiceExtractor::new(config);

    let ref_output = dir.path().join("ref_norm.wav");
    let result = extractor
        .extract_reference_audio(&full_wav, &segments, &ref_output)
        .unwrap();

    assert!(result.is_some());
    assert!(ref_output.exists());

    // 验证 WAV 可读取且非空
    let mut reader = hound::WavReader::open(&ref_output).expect("Failed to open WAV");
    let samples: Vec<i16> = reader.samples().filter_map(|s| s.ok()).collect();
    assert!(!samples.is_empty(), "Normalized audio should have samples");

    // 验证归一化后音频有合理振幅（非全零）
    let max_amp = samples.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    assert!(
        max_amp > 100,
        "Normalized audio should have non-trivial amplitude (max: {max_amp})"
    );
}

#[test]
fn test_voice_extractor_custom_duration_range() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let full_wav = dir.path().join("full.wav");
    create_test_wav(&full_wav, 60.0, 16000);

    let segments = vec![
        Segment::new("seg-0".into(), 0.0, 4.0, "Four seconds".into()),
        Segment::new("seg-1".into(), 4.0, 8.0, "Four seconds again".into()),
        Segment::new("seg-2".into(), 8.0, 20.0, "Twelve seconds".into()),
    ];

    // 自定义范围：4-8 秒
    let config = VoiceExtractorConfig {
        enable_enhancement: false,
        min_duration_secs: 4.0,
        max_duration_secs: 8.0,
        ideal_duration_secs: 6.0,
        ..Default::default()
    };
    let extractor = VoiceExtractor::new(config);

    let ref_output = dir.path().join("ref.wav");
    let result = extractor
        .extract_reference_audio(&full_wav, &segments, &ref_output)
        .unwrap();

    assert!(result.is_some());
    // 12 秒的 segment 应被排除（超出 max_duration_secs）
    let ref_audio = result.unwrap();
    assert!(
        ref_audio.prompt_text != "Twelve seconds",
        "Should not select segment outside custom range"
    );
}

// ─── SubprocessCloneEngine 测试 ──────────────────────────

#[test]
fn test_subprocess_engine_new() {
    let engine = SubprocessCloneEngine::new(
        "/usr/bin/echo".to_string(),
        Some("/path/to/model".to_string()),
        vec!["{text}".to_string(), "{output}".to_string()],
        60,
    );

    assert_eq!(engine.name(), "subprocess");
}

#[test]
fn test_subprocess_engine_from_config_with_args() {
    let config = CloningEngineConfig {
        enabled: true,
        engine: "subprocess".to_string(),
        clone_command: Some("/path/to/tool".to_string()),
        clone_model_path: Some("/path/to/model".to_string()),
        clone_args: vec![
            "synthesize".to_string(),
            "--text".to_string(),
            "{text}".to_string(),
            "--voice".to_string(),
            "{ref_audio}".to_string(),
            "--output".to_string(),
            "{output}".to_string(),
        ],
        clone_timeout_secs: 30,
        ..Default::default()
    };

    let engine = SubprocessCloneEngine::from_config(&config);
    assert!(engine.is_ok(), "Should create engine from config");
}

#[test]
fn test_subprocess_engine_from_config_missing_command() {
    let config = CloningEngineConfig {
        enabled: true,
        engine: "subprocess".to_string(),
        clone_command: None,
        ..Default::default()
    };

    let result = SubprocessCloneEngine::from_config(&config);
    assert!(result.is_err(), "Should fail without clone_command");
}

#[test]
fn test_subprocess_engine_from_config_with_preset() {
    // 测试 IndexTTS 预设
    let config_indextts = CloningEngineConfig {
        enabled: true,
        engine: "indextts".to_string(),
        clone_command: Some("/path/to/indextts".to_string()),
        clone_args: vec![], // 空参数，应使用预设
        ..Default::default()
    };
    let engine = SubprocessCloneEngine::from_config(&config_indextts);
    assert!(engine.is_ok(), "Should create engine with indextts preset");

    // 测试 qwen3-tts 预设
    let config_qwen = CloningEngineConfig {
        enabled: true,
        engine: "qwen3-tts".to_string(),
        clone_command: Some("/path/to/voice_clone".to_string()),
        clone_args: vec![],
        clone_model_path: Some("/path/to/model".to_string()),
        ..Default::default()
    };
    let engine = SubprocessCloneEngine::from_config(&config_qwen);
    assert!(engine.is_ok(), "Should create engine with qwen3-tts preset");
}

#[test]
fn test_subprocess_engine_set_prompt_text() {
    let engine = SubprocessCloneEngine::new(
        "/usr/bin/echo".to_string(),
        None,
        vec!["{text}".to_string()],
        60,
    );

    // 不应 panic
    engine.set_prompt_text("This is a prompt text for testing.");
}

#[test]
fn test_subprocess_engine_clone_nonexistent_command() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let ref_path = dir.path().join("reference.wav");
    create_reference_audio(&ref_path);

    let engine = SubprocessCloneEngine::new(
        "/nonexistent/command/path".to_string(),
        None,
        vec!["{text}".to_string(), "{output}".to_string()],
        10,
    );

    let config = CloningConfig {
        output_dir: dir.path().join("output").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let result = engine.clone_and_synthesize(&ref_path, "test text", &config);
    assert!(result.is_err(), "Should fail for nonexistent command");
}

#[test]
fn test_subprocess_engine_clone_nonexistent_reference() {
    let engine = SubprocessCloneEngine::new(
        "/usr/bin/echo".to_string(),
        None,
        vec!["{text}".to_string()],
        10,
    );

    let config = CloningConfig::default();

    let result = engine.clone_and_synthesize(Path::new("/nonexistent/ref.wav"), "test", &config);
    assert!(result.is_err(), "Should fail for nonexistent reference");
}

#[test]
fn test_subprocess_engine_with_echo_command() {
    // 使用 echo 命令测试（它不会生成输出文件，但能验证命令执行流程）
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let ref_path = dir.path().join("reference.wav");
    create_reference_audio(&ref_path);

    let engine = SubprocessCloneEngine::new(
        "/usr/bin/echo".to_string(),
        None,
        vec!["clone".to_string(), "{text}".to_string()],
        10,
    );

    let config = CloningConfig {
        output_dir: dir.path().join("output").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let result = engine.clone_and_synthesize(&ref_path, "hello", &config);

    // echo 命令会成功执行但不会生成输出文件
    assert!(
        result.is_err(),
        "Should fail because echo doesn't produce output file"
    );
}

// ─── VoiceExtractorConfig 序列化测试 ─────────────────────

#[test]
fn test_voice_extractor_config_serde_roundtrip() {
    let config = VoiceExtractorConfig {
        enable_enhancement: false,
        enable_silence_trim: true,
        enable_normalization: false,
        silence_threshold_db: -50.0,
        target_rms_db: -15.0,
        min_duration_secs: 2.0,
        max_duration_secs: 12.0,
        ideal_duration_secs: 6.0,
    };

    let json = serde_json::to_string(&config).expect("serialize failed");
    let restored: VoiceExtractorConfig = serde_json::from_str(&json).expect("deserialize failed");

    assert_eq!(restored.enable_enhancement, config.enable_enhancement);
    assert_eq!(restored.enable_silence_trim, config.enable_silence_trim);
    assert_eq!(restored.enable_normalization, config.enable_normalization);
    assert!((restored.silence_threshold_db - config.silence_threshold_db).abs() < f64::EPSILON);
    assert!((restored.target_rms_db - config.target_rms_db).abs() < f64::EPSILON);
    assert!((restored.min_duration_secs - config.min_duration_secs).abs() < f64::EPSILON);
    assert!((restored.max_duration_secs - config.max_duration_secs).abs() < f64::EPSILON);
    assert!((restored.ideal_duration_secs - config.ideal_duration_secs).abs() < f64::EPSILON);
}

#[test]
fn test_voice_extractor_config_toml_roundtrip() {
    let config = VoiceExtractorConfig {
        enable_enhancement: true,
        enable_silence_trim: false,
        enable_normalization: true,
        silence_threshold_db: -35.0,
        target_rms_db: -18.0,
        min_duration_secs: 4.0,
        max_duration_secs: 8.0,
        ideal_duration_secs: 6.0,
    };

    let toml_str = toml::to_string(&config).expect("toml serialize failed");
    let restored: VoiceExtractorConfig =
        toml::from_str(&toml_str).expect("toml deserialize failed");

    assert_eq!(restored.enable_enhancement, config.enable_enhancement);
    assert_eq!(restored.enable_silence_trim, config.enable_silence_trim);
}

#[test]
fn test_cloning_config_with_subprocess_fields() {
    let toml_str = r#"
enabled = true
engine = "subprocess"
clone_command = "/path/to/indextts"
clone_model_path = "/path/to/model"
clone_args = ["synthesize", "--text", "{text}", "--voice", "{ref_audio}", "--output", "{output}"]
clone_timeout_secs = 60

[voice_extractor]
enable_enhancement = true
enable_silence_trim = true
enable_normalization = true
silence_threshold_db = -40.0
target_rms_db = -20.0
min_duration_secs = 3.0
max_duration_secs = 10.0
ideal_duration_secs = 5.0
"#;

    let config: CloningEngineConfig = toml::from_str(toml_str).expect("toml parse failed");

    assert!(config.enabled);
    assert_eq!(config.engine, "subprocess");
    assert_eq!(config.clone_command.as_deref(), Some("/path/to/indextts"));
    assert_eq!(config.clone_model_path.as_deref(), Some("/path/to/model"));
    assert_eq!(config.clone_args.len(), 7);
    assert_eq!(config.clone_timeout_secs, 60);
    assert!(config.voice_extractor.enable_enhancement);
    assert!(config.voice_extractor.enable_silence_trim);
    assert!(config.voice_extractor.enable_normalization);
}

// ─── CloningIntegration + SubprocessCloneEngine 测试 ─────

#[test]
fn test_cloning_integration_with_subprocess_engine() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let ref_path = dir.path().join("reference.wav");
    create_reference_audio(&ref_path);

    // 创建一个会失败的 SubprocessCloneEngine（命令不存在）
    let engine = SubprocessCloneEngine::new(
        "/nonexistent/tool".to_string(),
        None,
        vec!["{text}".to_string()],
        10,
    );

    let config = CloningConfig {
        output_dir: dir.path().join("output").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let integration = CloningIntegration::new(Box::new(engine), config);

    let mut segment = Segment::new("seg-1".into(), 0.0, 5.0, "Hello".into());
    segment.start_transcribing().expect("start_transcribing");
    segment
        .finish_transcribing("你好".into())
        .expect("finish_transcribing");

    // try_synthesize 应优雅降级
    let result = integration.try_synthesize(&segment, &ref_path);
    assert!(result.is_ok(), "try_synthesize should not hard-fail");
    assert!(
        result.unwrap().is_none(),
        "Should return None on failure (graceful degradation)"
    );
}

#[test]
fn test_cloning_integration_engine_name_subprocess() {
    let engine = SubprocessCloneEngine::new(
        "/path/to/tool".to_string(),
        None,
        vec!["{text}".to_string()],
        60,
    );

    let integration = CloningIntegration::new(Box::new(engine), CloningConfig::default());
    assert_eq!(integration.engine_name(), "subprocess");
}

// ─── ReferenceAudio 测试 ─────────────────────────────────

#[test]
fn test_reference_audio_builder_pattern() {
    let ref_audio = ReferenceAudio::new(
        PathBuf::from("/tmp/ref.wav"),
        5.5,
        24000,
        "Hello world".to_string(),
    )
    .with_speaker_id("SPEAKER_01");

    assert_eq!(ref_audio.path, PathBuf::from("/tmp/ref.wav"));
    assert!((ref_audio.duration_secs - 5.5).abs() < f64::EPSILON);
    assert_eq!(ref_audio.sample_rate, 24000);
    assert_eq!(ref_audio.prompt_text, "Hello world");
    assert_eq!(ref_audio.speaker_id.as_deref(), Some("SPEAKER_01"));
}

// ─── 端到端流程测试 ───────────────────────────────────────

#[test]
fn test_end_to_end_extraction_and_cloning_flow() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    // 1. 创建模拟视频音频
    let full_wav = dir.path().join("full_audio.wav");
    create_test_wav(&full_wav, 30.0, 16000);

    // 2. 创建 ASR Segment（模拟 ASR 输出）
    let segments = vec![
        Segment::new(
            "seg-0".into(),
            0.0,
            5.0,
            "Hello, welcome to this video.".into(),
        ),
        Segment::new(
            "seg-1".into(),
            5.0,
            10.0,
            "Today we'll learn about Rust.".into(),
        ),
    ];

    // 3. 使用 VoiceExtractor 提取参考音频
    let config = VoiceExtractorConfig {
        enable_enhancement: false,
        enable_silence_trim: true,
        enable_normalization: true,
        ..Default::default()
    };
    let extractor = VoiceExtractor::new(config);

    let ref_output = dir.path().join("extracted_ref.wav");
    let ref_audio = extractor
        .extract_reference_audio(&full_wav, &segments, &ref_output)
        .expect("extraction failed")
        .expect("should find a segment");

    // 4. 验证提取结果
    assert!(ref_output.exists());
    assert!(!ref_audio.prompt_text.is_empty());

    // 5. 使用 MockCloningEngine 进行克隆合成
    let synth_config = CloningConfig {
        output_dir: dir.path().join("cloned").to_string_lossy().into_owned(),
        ..Default::default()
    };

    let integration = CloningIntegration::new(Box::new(MockCloningEngine::new()), synth_config);

    // 设置 prompt_text
    integration.set_prompt_text(&ref_audio.prompt_text);

    // 6. 为每个 segment 合成
    let mut seg = segments[0].clone();
    seg.start_transcribing().expect("start");
    seg.finish_transcribing("你好，欢迎观看这个视频。".to_string())
        .expect("finish");

    let result = integration.synthesize_for_segment(&seg, &ref_audio.path);
    assert!(result.is_ok(), "Should synthesize for segment");
    assert!(result.unwrap().exists(), "Output file should exist");
}

#[test]
fn test_extraction_with_multiple_speakers_metadata() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let full_wav = dir.path().join("full.wav");
    create_test_wav(&full_wav, 60.0, 16000);

    // 模拟多说话人场景
    let segments = vec![
        Segment::new("seg-0".into(), 0.0, 5.0, "Speaker one says hello".into()),
        Segment::new("seg-1".into(), 5.0, 12.0, "Speaker two responds".into()),
        Segment::new("seg-2".into(), 12.0, 17.0, "Speaker one again".into()),
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

    assert!(result.is_some());
    // 即使有多个 segment，也应该选择最理想时长的那个
    let ref_audio = result.unwrap();
    assert!(!ref_audio.prompt_text.is_empty());
}
