//! 集成测试：语音合成 (TTS) — macOS `say` 离线引擎
//!
//! 验证 `SayEngine` 的语音合成功能，包括：
//! - 单段中文文本合成（WAV 文件生成、格式验证、时长合理性）
//! - 批量合成多个 Segment（并行处理、路径填充）
//! - 缓存机制（相同文本复用缓存、不重复合成）
//! - 错误处理（空文本、无效配置等）
//!
//! # 离线运行
//! 所有测试使用 macOS 内置的 `say` 命令，无需网络连接。
//!
//! # 运行方式
//! ```sh
//! cargo test test_tts -- --nocapture
//! ```

mod common;

use std::path::PathBuf;
use std::time::SystemTime;

use tempfile::TempDir;
use vt_core::config::TtsConfig;
use vt_core::error::AppError;
use vt_core::models::segment::{Segment, SegmentStatus};
use vt_core::tts::{SayEngine, TtsEngine};

// ─── 测试辅助函数 ─────────────────────────────────────────

/// 创建测试用 Segment（已翻译状态，包含中文目标文本）。
fn make_translated_segment(id: &str, target_text: &str) -> Segment {
    let mut seg = Segment::new(id.to_string(), 0.0, 5.0, "Hello world".to_string());
    seg.start_transcribing().expect("start_transcribing failed");
    seg.finish_transcribing(target_text.to_string())
        .expect("finish_transcribing failed");
    seg
}

/// 获取文件的最后修改时间（用于缓存测试）。
fn file_mtime(path: &std::path::Path) -> SystemTime {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// 验证 WAV 文件格式正确（24kHz mono 16-bit PCM，默认采样率）。
fn validate_wav_format(path: &std::path::Path) {
    assert!(path.exists(), "WAV file should exist: {:?}", path);

    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();

    assert_eq!(
        spec.sample_rate, 24000,
        "Sample rate should be 24000 Hz (default), got {}",
        spec.sample_rate
    );
    assert_eq!(
        spec.channels, 1,
        "Channels should be 1 (mono), got {}",
        spec.channels
    );
    assert_eq!(
        spec.bits_per_sample, 16,
        "Bits per sample should be 16, got {}",
        spec.bits_per_sample
    );

    // 确保有实际音频数据
    let sample_count: usize = reader.samples::<i16>().count();
    assert!(sample_count > 0, "WAV file should contain audio samples");
}

// ═══════════════════════════════════════════════════════════
//  单段合成测试
// ═══════════════════════════════════════════════════════════

/// 验证单个 Segment 的中文文本合成：
/// - WAV 文件存在
/// - WAV 格式正确（24kHz mono 16-bit）
/// - 音频时长与文本长度匹配
/// - Segment 的 `tts_audio_path` 被填充
/// - Segment 状态变为 `Completed`
#[test]
fn test_tts_single() {
    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: TTS engine not available");
            return;
        }
    };
    let config = common::shared_tts_config();

    let text = "你好，世界。这是一个语音合成测试。";
    let mut segments = vec![make_translated_segment("seg-0001", text)];

    let audio_paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("TTS synthesis failed");

    // 验证返回的路径列表
    assert_eq!(audio_paths.len(), 1, "Should return 1 audio path");
    assert!(
        audio_paths[0].exists(),
        "Audio file should exist: {:?}",
        audio_paths[0]
    );

    // 验证 Segment 状态和路径
    assert_eq!(segments[0].status, SegmentStatus::Completed);
    assert!(
        segments[0].tts_audio_path.is_some(),
        "tts_audio_path should be set"
    );

    // 验证 WAV 格式
    validate_wav_format(&audio_paths[0]);

    // 验证音频时长合理性
    let mut reader =
        hound::WavReader::open(&audio_paths[0]).expect("Failed to open WAV for duration check");
    let sample_count: usize = reader.samples::<i16>().count();
    let duration_secs = sample_count as f64 / 16000.0;

    eprintln!(
        "TTS single: text='{}', samples={}, duration={:.2}s",
        text, sample_count, duration_secs
    );

    assert!(
        duration_secs > 1.0,
        "Audio duration should be > 1s for text '{}', got {:.2}s",
        text,
        duration_secs
    );
    assert!(
        duration_secs < 30.0,
        "Audio duration should be < 30s, got {:.2}s",
        duration_secs
    );
}

// ═══════════════════════════════════════════════════════════
//  批量合成测试
// ═══════════════════════════════════════════════════════════

/// 验证批量合成多个 Segment：
/// - 所有 Segment 的 `tts_audio_path` 都被填充
/// - 所有 WAV 文件存在且格式正确
/// - 所有 Segment 状态变为 `Completed`
#[test]
fn test_tts_batch() {
    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: TTS engine not available");
            return;
        }
    };
    let config = common::shared_tts_config();

    let texts = [
        "欢迎使用视频翻译工具。",
        "语音识别模块已完成。",
        "翻译模块支持多种语言。",
    ];

    let mut segments: Vec<Segment> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| make_translated_segment(&format!("seg-{i:04}"), text))
        .collect();

    let audio_paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("TTS batch synthesis failed");

    // 验证返回路径数量
    assert_eq!(
        audio_paths.len(),
        texts.len(),
        "Should return {} audio paths",
        texts.len()
    );

    // 验证每个 Segment
    for (i, seg) in segments.iter().enumerate() {
        assert_eq!(
            seg.status,
            SegmentStatus::Completed,
            "Segment {} should be Completed",
            i
        );
        assert!(
            seg.tts_audio_path.is_some(),
            "Segment {} tts_audio_path should be set",
            i
        );
    }

    // 验证每个 WAV 文件
    for (i, path) in audio_paths.iter().enumerate() {
        assert!(path.exists(), "Audio file {} should exist: {:?}", i, path);
        validate_wav_format(path);

        eprintln!(
            "TTS batch[{}]: {:?} ({} bytes)",
            i,
            path,
            std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
        );
    }
}

// ═══════════════════════════════════════════════════════════
//  缓存机制测试
// ═══════════════════════════════════════════════════════════

/// 验证缓存机制：相同文本第二次合成时复用缓存，不重新生成文件。
#[test]
fn test_tts_cache() {
    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: TTS engine not available");
            return;
        }
    };
    let config = common::shared_tts_config();

    let text = "这是一段用于缓存测试的中文文本。";
    let mut segments1 = vec![make_translated_segment("seg-cache-1", text)];

    // 第一次合成（可能命中持久化缓存）
    let paths1 = engine
        .synthesize_segments(&mut segments1, &config)
        .expect("First synthesis failed");

    assert_eq!(paths1.len(), 1);
    let cached_path = paths1[0].clone();
    assert!(
        cached_path.exists(),
        "Cached file should exist after first synthesis"
    );

    // 记录第一次合成的文件修改时间
    let mtime1 = file_mtime(&cached_path);

    // 等待一小段时间确保修改时间精度
    std::thread::sleep(std::time::Duration::from_millis(100));

    // 第二次合成相同文本（应命中缓存）
    let mut segments2 = vec![make_translated_segment("seg-cache-2", text)];
    let paths2 = engine
        .synthesize_segments(&mut segments2, &config)
        .expect("Second synthesis failed");

    assert_eq!(paths2.len(), 1);

    // 验证缓存命中：文件路径应相同
    assert_eq!(
        paths2[0], cached_path,
        "Second synthesis should reuse cached file (same path)"
    );

    // 验证文件修改时间不变（缓存命中，未重新写入）
    let mtime2 = file_mtime(&cached_path);
    assert_eq!(
        mtime1, mtime2,
        "File modification time should be unchanged (cache hit)"
    );

    eprintln!("TTS cache: reused file {:?} (mtime unchanged)", cached_path);
}

// ═══════════════════════════════════════════════════════════
//  语速配置测试
// ═══════════════════════════════════════════════════════════

/// 验证不同语速配置产生不同时长的音频。
#[test]
fn test_tts_speed_variation() {
    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: TTS engine not available");
            return;
        }
    };

    let text = "这是一段用于测试语速变化的中文文本内容。";

    // 正常语速
    let config1 = common::shared_tts_config();
    let mut segs1 = vec![make_translated_segment("seg-speed-1", text)];
    let paths1 = engine
        .synthesize_segments(&mut segs1, &config1)
        .expect("Synthesis at speed 1.0 failed");

    // 快语速
    let config2 = TtsConfig {
        speed: 2.0,
        ..common::shared_tts_config()
    };
    let mut segs2 = vec![make_translated_segment("seg-speed-2", text)];
    let paths2 = engine
        .synthesize_segments(&mut segs2, &config2)
        .expect("Synthesis at speed 2.0 failed");

    // 读取音频时长
    let get_duration = |path: &PathBuf| -> f64 {
        let mut reader = hound::WavReader::open(path).expect("Failed to open WAV");
        let spec = reader.spec();
        let samples: usize = reader.samples::<i16>().count();
        samples as f64 / spec.sample_rate as f64
    };

    let duration_normal = get_duration(&paths1[0]);
    let duration_fast = get_duration(&paths2[0]);

    eprintln!(
        "TTS speed: normal={:.2}s, fast={:.2}s (speed=2.0)",
        duration_normal, duration_fast
    );

    // 快语速的音频应明显短于正常语速
    assert!(
        duration_fast < duration_normal,
        "Fast speech (speed=2.0) should be shorter than normal (speed=1.0): \
         fast={:.2}s vs normal={:.2}s",
        duration_fast,
        duration_normal
    );
}

// ═══════════════════════════════════════════════════════════
//  错误处理测试
// ═══════════════════════════════════════════════════════════

/// 验证空 Segment 列表直接返回成功（空路径列表）。
#[test]
fn test_tts_empty_segments() {
    let cache_dir = TempDir::new().expect("Failed to create temp dir");
    let config = TtsConfig {
        speed: 1.0,
        voice: "Tingting".to_string(),
        cache_dir: cache_dir.path().to_string_lossy().to_string(),
        parallel_tasks: 2,
        ..Default::default()
    };
    let engine = SayEngine::new(&config).expect("Failed to create engine");

    let mut segments: Vec<Segment> = vec![];
    let result = engine.synthesize_segments(&mut segments, &config);

    assert!(result.is_ok(), "Empty segments should succeed");
    let paths = result.expect("Should be Ok");
    assert!(paths.is_empty(), "Should return empty path list");
}

/// 验证 Segment 的 `target_text` 为空时返回错误。
#[test]
fn test_tts_empty_text_error() {
    let cache_dir = TempDir::new().expect("Failed to create temp dir");
    let config = TtsConfig {
        speed: 1.0,
        voice: "Tingting".to_string(),
        cache_dir: cache_dir.path().to_string_lossy().to_string(),
        parallel_tasks: 2,
        ..Default::default()
    };
    let engine = SayEngine::new(&config).expect("Failed to create engine");

    let mut seg = Segment::new("seg-empty".to_string(), 0.0, 5.0, "Hello".to_string());
    seg.start_transcribing().expect("start failed");
    seg.finish_transcribing("".to_string())
        .expect("finish failed");

    let mut segments = vec![seg];
    let result = engine.synthesize_segments(&mut segments, &config);

    assert!(result.is_err(), "Empty target_text should return error");
    assert!(
        matches!(result, Err(AppError::TtsError(_))),
        "Expected TtsError for empty text, got {:?}",
        result
    );
}

/// 验证 Segment 未翻译（target_text 为 None）时返回错误。
#[test]
fn test_tts_untranslated_segment_error() {
    let cache_dir = TempDir::new().expect("Failed to create temp dir");
    let config = TtsConfig {
        speed: 1.0,
        voice: "Tingting".to_string(),
        cache_dir: cache_dir.path().to_string_lossy().to_string(),
        parallel_tasks: 2,
        ..Default::default()
    };
    let engine = SayEngine::new(&config).expect("Failed to create engine");

    let mut segments = vec![Segment::new(
        "seg-untranslated".to_string(),
        0.0,
        5.0,
        "Hello world".to_string(),
    )];

    let result = engine.synthesize_segments(&mut segments, &config);

    assert!(result.is_err(), "Untranslated segment should return error");
    assert!(
        matches!(result, Err(AppError::TtsError(_))),
        "Expected TtsError for untranslated segment, got {:?}",
        result
    );
}

// ═══════════════════════════════════════════════════════════
//  TtsConfig 默认值测试
// ═══════════════════════════════════════════════════════════

/// 验证 TtsConfig 默认值。
#[test]
fn test_tts_config_default() {
    let config = TtsConfig::default();
    assert_eq!(config.speed, 1.0, "Default speed should be 1.0");
    assert!(
        !config.voice.is_empty(),
        "Default voice should not be empty"
    );
    assert!(
        !config.cache_dir.is_empty(),
        "Default cache_dir should not be empty"
    );
    assert!(
        config.parallel_tasks > 0,
        "Default parallel_tasks should be > 0"
    );
}

/// 验证从 TOML 加载 TTS 配置。
#[test]
fn test_tts_config_from_toml() {
    use std::io::Write;
    use vt_core::config::Config;

    let toml_content = r#"
[tts]
speed = 1.5
voice = "Meijia"
cache_dir = "/tmp/test-tts-cache"
parallel_tasks = 8
"#;

    let mut tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{toml_content}").expect("Failed to write TOML");

    let config = Config::from_file(tmp.path()).expect("Failed to load config");

    assert_eq!(config.tts.speed, 1.5);
    assert_eq!(config.tts.voice, "Meijia");
    assert_eq!(config.tts.cache_dir, "/tmp/test-tts-cache");
    assert_eq!(config.tts.parallel_tasks, 8);
}

// ═══════════════════════════════════════════════════════════
//  KokoroEngine 集成测试
// ═══════════════════════════════════════════════════════════

/// 验证 KokoroEngine 在 fallback_to_say=true 时降级到 SayEngine。
#[test]
fn test_kokoro_fallback_to_say() {
    let config = vt_core::config::TtsConfig::default();
    let engine = vt_core::tts::KokoroEngine::new(&config)
        .expect("KokoroEngine should fall back to SayEngine");
    assert_eq!(engine.backend_name(), "SayEngine");
}

/// 验证 KokoroEngine 在 fallback_to_say=false 时返回错误。
#[test]
fn test_kokoro_no_fallback_error() {
    let config = vt_core::config::TtsConfig {
        fallback_to_say: false,
        ..Default::default()
    };
    let result = vt_core::tts::KokoroEngine::new(&config);
    assert!(result.is_err());
}

/// 验证 KokoroEngine Debug 输出包含后端信息。
#[test]
fn test_kokoro_debug_output() {
    let config = vt_core::config::TtsConfig::default();
    let engine = vt_core::tts::KokoroEngine::new(&config).expect("Should succeed");
    let debug = format!("{engine:?}");
    assert!(debug.contains("KokoroEngine"));
    assert!(debug.contains("SayEngine"));
}

/// 验证 KokoroEngine 通过 TtsEngine trait 合成文本。
#[test]
fn test_kokoro_synthesize_via_trait() {
    use vt_core::tts::TtsEngine;

    let config = vt_core::config::TtsConfig::default();
    let engine = vt_core::tts::KokoroEngine::new(&config).expect("KokoroEngine should succeed");

    let mut segments = vec![vt_core::models::segment::Segment::new(
        "kokoro-test".to_string(),
        0.0,
        2.0,
        "Hello world".to_string(),
    )];
    segments[0].start_transcribing().expect("start failed");
    segments[0]
        .finish_transcribing("你好世界".to_string())
        .expect("finish failed");

    let paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("Synthesis should succeed");

    assert_eq!(paths.len(), 1);
    assert!(paths[0].exists(), "Audio file should exist");
}
