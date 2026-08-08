//! 集成测试：配置管理
//!
//! 验证默认配置值以及从 TOML 文件加载配置的逻辑。

use std::io::Write;

use tempfile::NamedTempFile;
use vt_core::config::{
    AsrConfig, BatchConfig, CheckpointConfig, CloningConfig, Config, DiarizationConfig,
    PerformanceConfig, TtsConfig,
};

/// 验证默认配置值符合预期。
#[test]
fn test_default_config() {
    let config = Config::default();

    // ASR 默认配置
    assert!(!config.asr.model.is_empty());
    assert!(config.asr.use_metal, "use_metal should default to true");
    assert_eq!(config.asr.language, "en");

    // TTS 默认配置
    assert_eq!(config.tts.speed, 1.0);
    assert!(!config.tts.voice.is_empty());

    // 顶层配置
    assert!(!config.output_dir.is_empty());
    assert!(config.max_concurrent_tasks > 0);

    // 新增模块默认配置
    assert!(
        !config.diarization.enabled,
        "diarization should default to disabled"
    );
    assert!(config.cloning.enabled, "cloning should default to enabled");
    assert!(
        config.checkpoint.enabled,
        "checkpoint should default to enabled"
    );
    assert!(config.batch.max_concurrent > 0);
}

/// 验证 AsrConfig 的默认值。
#[test]
fn test_default_asr_config() {
    let asr = AsrConfig::default();
    assert!(!asr.model.is_empty());
    assert!(asr.use_metal);
    assert_eq!(asr.language, "en");
}

/// 验证 TtsConfig 的默认值。
#[test]
fn test_default_tts_config() {
    let tts = TtsConfig::default();
    assert_eq!(tts.speed, 1.0);
    assert!(!tts.voice.is_empty());
}

/// 验证 DiarizationConfig 的默认值。
#[test]
fn test_default_diarization_config() {
    let dia = DiarizationConfig::default();
    assert!(!dia.enabled);
    assert_eq!(dia.engine, "speakrs");
    assert!(dia.use_coreml);
}

/// 验证 CloningConfig 的默认值。
#[test]
fn test_default_cloning_config() {
    let clone = CloningConfig::default();
    assert!(clone.enabled);
    assert_eq!(clone.engine, "indextts");
    assert!(clone.auto_extract_speaker);
}

/// 验证 BatchConfig 的默认值。
#[test]
fn test_default_batch_config() {
    let batch = BatchConfig::default();
    assert!(batch.max_concurrent > 0);
    assert!((batch.memory_threshold - 80.0).abs() < f64::EPSILON);
    assert!(batch.enable_priority);
}

/// 验证 CheckpointConfig 的默认值。
#[test]
fn test_default_checkpoint_config() {
    let cp = CheckpointConfig::default();
    assert!(cp.enabled);
    assert_eq!(cp.retention_days, 7);
}

/// 验证 PerformanceConfig 的默认值。
#[test]
fn test_default_performance_config() {
    let perf = PerformanceConfig::default();
    assert!(!perf.enable_profiling);
}

/// 验证从完整的 TOML 文件加载配置。
#[test]
fn test_config_from_toml() {
    let toml_content = r#"
output_dir = "/tmp/video-output"
max_concurrent_tasks = 8

[asr]
model = "whisper-medium"
use_metal = true
language = "en"

[tts]
speed = 1.5
voice = "zh-CN-YunxiNeural"

[diarization]
enabled = true
engine = "speakrs"
use_coreml = true

[cloning]
enabled = true
engine = "indextts"
reference_audio_dir = "./refs"
auto_extract_speaker = true

[batch]
max_concurrent = 5
memory_threshold = 75.0
enable_priority = false

[checkpoint]
enabled = false
dir = "/tmp/checkpoints"
retention_days = 14

[performance]
enable_profiling = true
flamegraph_output = "./prof.svg"
"#;

    let mut tmp_file = NamedTempFile::new().unwrap();
    write!(tmp_file, "{}", toml_content).unwrap();

    let config = Config::from_file(tmp_file.path()).unwrap();

    // ASR
    assert_eq!(config.asr.model, "whisper-medium");
    assert!(config.asr.use_metal);
    assert_eq!(config.asr.language, "en");

    // TTS
    assert_eq!(config.tts.speed, 1.5);
    assert_eq!(config.tts.voice, "zh-CN-YunxiNeural");

    // 顶层
    assert_eq!(config.output_dir, "/tmp/video-output");
    assert_eq!(config.max_concurrent_tasks, 8);

    // Diarization
    assert!(config.diarization.enabled);
    assert_eq!(config.diarization.engine, "speakrs");
    assert!(config.diarization.use_coreml);

    // Cloning
    assert!(config.cloning.enabled);
    assert_eq!(config.cloning.engine, "indextts");
    assert_eq!(config.cloning.reference_audio_dir, "./refs");

    // Batch
    assert_eq!(config.batch.max_concurrent, 5);
    assert!((config.batch.memory_threshold - 75.0).abs() < f64::EPSILON);
    assert!(!config.batch.enable_priority);

    // Checkpoint
    assert!(!config.checkpoint.enabled);
    assert_eq!(config.checkpoint.dir, "/tmp/checkpoints");
    assert_eq!(config.checkpoint.retention_days, 14);

    // Performance
    assert!(config.performance.enable_profiling);
    assert_eq!(config.performance.flamegraph_output, "./prof.svg");
}

/// 验证当 TOML 文件缺少 [asr] 和 [tts] 段时，使用默认值进行合并。
#[test]
fn test_config_from_toml_with_defaults() {
    let toml_content = r#"
output_dir = "/custom/output"
max_concurrent_tasks = 2
"#;

    let mut tmp_file = NamedTempFile::new().unwrap();
    write!(tmp_file, "{}", toml_content).unwrap();

    let config = Config::from_file(tmp_file.path()).unwrap();

    // 缺少的 [asr] 段应使用默认值
    assert!(config.asr.use_metal, "use_metal should default to true");
    assert_eq!(config.asr.language, "en");

    // 缺少的 [tts] 段应使用默认值
    assert_eq!(config.tts.speed, 1.0);

    // 提供的值应正确加载
    assert_eq!(config.output_dir, "/custom/output");
    assert_eq!(config.max_concurrent_tasks, 2);

    // 新增模块应使用默认值
    assert!(!config.diarization.enabled);
    assert!(config.checkpoint.enabled);
    assert!(config.batch.max_concurrent > 0);
}

/// 验证当 TOML 文件中 [asr] 段只提供部分字段时，缺失字段使用默认值。
#[test]
fn test_config_from_toml_partial_asr() {
    let toml_content = r#"
output_dir = "./out"
max_concurrent_tasks = 1

[asr]
model = "whisper-tiny"
"#;

    let mut tmp_file = NamedTempFile::new().unwrap();
    write!(tmp_file, "{}", toml_content).unwrap();

    let config = Config::from_file(tmp_file.path()).unwrap();

    // 提供的字段
    assert_eq!(config.asr.model, "whisper-tiny");

    // 缺失字段应使用默认值
    assert!(config.asr.use_metal, "use_metal should default to true");
    assert_eq!(config.asr.language, "en");

    // 顶层字段应正确加载
    assert_eq!(config.output_dir, "./out");
    assert_eq!(config.max_concurrent_tasks, 1);
}

/// 验证 Config 的序列化/反序列化往返。
#[test]
fn test_config_serde_roundtrip() {
    let config = Config::default();

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: Config = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.asr.model, config.asr.model);
    assert_eq!(deserialized.asr.use_metal, config.asr.use_metal);
    assert_eq!(deserialized.tts.speed, config.tts.speed);
    assert_eq!(deserialized.tts.voice, config.tts.voice);
    assert_eq!(deserialized.output_dir, config.output_dir);
    assert_eq!(
        deserialized.max_concurrent_tasks,
        config.max_concurrent_tasks
    );

    // 新增模块
    assert_eq!(deserialized.diarization.enabled, config.diarization.enabled);
    assert_eq!(deserialized.cloning.enabled, config.cloning.enabled);
    assert_eq!(deserialized.checkpoint.enabled, config.checkpoint.enabled);
    assert_eq!(
        deserialized.batch.max_concurrent,
        config.batch.max_concurrent
    );
}

/// 验证加载不存在的配置文件时返回错误。
#[test]
fn test_config_from_file_not_found() {
    let result = Config::from_file("/nonexistent/path/config.toml");
    assert!(result.is_err());
}

/// 验证部分新增模块配置可以从 TOML 加载。
#[test]
fn test_config_from_toml_new_modules() {
    let toml_content = r#"
[diarization]
enabled = true

[batch]
max_concurrent = 8
"#;

    let mut tmp_file = NamedTempFile::new().unwrap();
    write!(tmp_file, "{}", toml_content).unwrap();

    let config = Config::from_file(tmp_file.path()).unwrap();

    // diarization: enabled 被设置，其余用默认值
    assert!(config.diarization.enabled);
    assert_eq!(config.diarization.engine, "speakrs"); // 默认值
    assert!(config.diarization.use_coreml); // 默认值

    // batch: max_concurrent 被设置，其余用默认值
    assert_eq!(config.batch.max_concurrent, 8);
    assert!((config.batch.memory_threshold - 80.0).abs() < f64::EPSILON); // 默认值
    assert!(config.batch.enable_priority); // 默认值

    // cloning 和 checkpoint 未设置，应使用默认值
    assert!(config.cloning.enabled);
    assert!(config.checkpoint.enabled);
}
