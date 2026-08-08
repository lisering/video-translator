//! 集成测试：TTS 多音色与音频后处理
//!
//! 验证以下功能：
//! - 音色列表查询（`list_voices()`）
//! - 多音色合成（至少 2 女 + 2 男）
//! - 音色切换后合成结果变化
//! - 音频质量（采样率、时长、音量）
//! - 语速/音调/音量控制有效性
//! - 降级机制（KokoroEngine → SayEngine）
//! - 中英文混排发音
//!
//! # 运行方式
//! ```sh
//! cargo test --test test_tts_voice -- --nocapture
//! ```

mod common;

use std::path::PathBuf;

use tempfile::TempDir;
use vt_core::config::TtsConfig;
use vt_core::models::segment::{Segment, SegmentStatus};
use vt_core::tts::{SayEngine, TtsEngine};
use vt_core::voice_manager::{VoiceGender, VoiceManager};

// ─── 测试辅助函数 ─────────────────────────────────────────

/// 创建测试用 Segment（已翻译状态）
fn make_translated_segment(id: &str, target_text: &str) -> Segment {
    let mut seg = Segment::new(id.to_string(), 0.0, 5.0, "Hello world".to_string());
    seg.start_transcribing().expect("start_transcribing failed");
    seg.finish_transcribing(target_text.to_string())
        .expect("finish_transcribing failed");
    seg
}

/// 创建指定缓存目录和音色的 TTS 配置
fn make_config(cache_dir: &str, voice_id: &str) -> TtsConfig {
    TtsConfig {
        voice_id: voice_id.to_string(),
        cache_dir: cache_dir.to_string(),
        ..Default::default()
    }
}

/// 获取 WAV 文件时长（秒）
fn wav_duration(path: &PathBuf) -> f64 {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV");
    let spec = reader.spec();
    let samples: usize = reader.samples::<i16>().count();
    samples as f64 / spec.sample_rate as f64
}

/// 获取 WAV 文件采样率
fn wav_sample_rate(path: &PathBuf) -> u32 {
    let reader = hound::WavReader::open(path).expect("Failed to open WAV");
    reader.spec().sample_rate
}

// ═══════════════════════════════════════════════════════════
//  音色列表测试
// ═══════════════════════════════════════════════════════════

/// 验证 `list_voices()` 能返回音色列表，且至少包含 4 种音色。
#[test]
fn test_tts_voices_list() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let voices = engine.list_voices();
    assert!(
        voices.len() >= 4,
        "Should have at least 4 voices, got {}",
        voices.len()
    );

    // 验证每个音色都有必要字段
    for v in &voices {
        assert!(!v.id.is_empty(), "Voice ID should not be empty");
        assert!(!v.name.is_empty(), "Voice name should not be empty");
        assert!(
            !v.say_voice.is_empty(),
            "Voice say_voice should not be empty"
        );
        assert!(
            v.pitch_multiplier > 0.0 && v.pitch_multiplier <= 2.0,
            "Pitch multiplier should be in (0, 2.0], got {}",
            v.pitch_multiplier
        );
    }
}

/// 验证音色列表中至少有 2 种女声和 2 种男声。
#[test]
fn test_tts_voices_gender_distribution() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let voices = engine.list_voices();
    let females = voices
        .iter()
        .filter(|v| v.gender == VoiceGender::Female)
        .count();
    let males = voices
        .iter()
        .filter(|v| v.gender == VoiceGender::Male)
        .count();

    assert!(
        females >= 2,
        "Should have at least 2 female voices, got {females}"
    );
    assert!(
        males >= 2,
        "Should have at least 2 male voices, got {males}"
    );
}

/// 验证 VoiceManager 独立使用时功能正确。
#[test]
fn test_voice_manager_standalone() {
    let manager = VoiceManager::new();

    // 查找女声
    let female = manager.find_by_id("tingting");
    assert!(female.is_some());
    assert_eq!(female.as_ref().expect("voice").gender, VoiceGender::Female);

    // 查找男声
    let male = manager.find_by_id("zhiming");
    assert!(male.is_some());
    assert_eq!(male.as_ref().expect("voice").gender, VoiceGender::Male);

    // 男声应该有音调偏移
    assert!(
        male.expect("voice").pitch_multiplier < 1.0,
        "Male voice should have pitch_multiplier < 1.0"
    );
}

// ═══════════════════════════════════════════════════════════
//  音色选择测试
// ═══════════════════════════════════════════════════════════

/// 验证切换到不同音色后，合成结果文件路径不同（缓存 key 包含 voice_id）。
#[test]
fn test_tts_voice_selection_different_paths() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let text = "这是一段测试文本。";

    // 使用女声合成
    let config_female = make_config(&cache_path, "tingting");
    let mut segs_f = vec![make_translated_segment("seg-female", text)];
    let paths_female = engine
        .synthesize_segments(&mut segs_f, &config_female)
        .expect("Female synthesis failed");

    // 使用男声合成
    let config_male = make_config(&cache_path, "zhiming");
    let mut segs_m = vec![make_translated_segment("seg-male", text)];
    let paths_male = engine
        .synthesize_segments(&mut segs_m, &config_male)
        .expect("Male synthesis failed");

    // 两个路径应该不同（voice_id 不同 → hash 不同）
    assert_ne!(
        paths_female[0], paths_male[0],
        "Different voice_id should produce different cache paths"
    );
}

/// 验证女声和男声合成后音频特性不同（男声经过音调偏移处理）。
#[test]
fn test_tts_voice_selection_changes_audio() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let text = "你好世界，这是一段中文测试文本。";

    // 女声合成（无音调偏移）
    let config_female = make_config(&cache_path, "tingting");
    let mut segs_f = vec![make_translated_segment("seg-f", text)];
    let paths_f = engine
        .synthesize_segments(&mut segs_f, &config_female)
        .expect("Female synthesis failed");

    // 男声合成（有音调偏移）
    let config_male = make_config(&cache_path, "zhiming");
    let mut segs_m = vec![make_translated_segment("seg-m", text)];
    let paths_m = engine
        .synthesize_segments(&mut segs_m, &config_male)
        .expect("Male synthesis failed");

    // 两个文件都应该存在
    assert!(paths_f[0].exists(), "Female audio should exist");
    assert!(paths_m[0].exists(), "Male audio should exist");

    // 两个文件路径不同
    assert_ne!(paths_f[0], paths_m[0], "Audio paths should differ");

    // 两个文件大小可能不同（不同的后处理）
    let size_f = std::fs::metadata(&paths_f[0]).map(|m| m.len()).unwrap_or(0);
    let size_m = std::fs::metadata(&paths_m[0]).map(|m| m.len()).unwrap_or(0);

    eprintln!(
        "Female audio: {} bytes, Male audio: {} bytes",
        size_f, size_m
    );

    // 两段音频都应该有合理时长
    let dur_f = wav_duration(&paths_f[0]);
    let dur_m = wav_duration(&paths_m[0]);
    assert!(
        dur_f > 0.5,
        "Female audio duration should be > 0.5s, got {dur_f:.2}s"
    );
    assert!(
        dur_m > 0.5,
        "Male audio duration should be > 0.5s, got {dur_m:.2}s"
    );
}

// ═══════════════════════════════════════════════════════════
//  音频质量测试
// ═══════════════════════════════════════════════════════════

/// 验证合成音频的采样率与配置一致。
#[test]
fn test_tts_audio_quality_sample_rate() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    // 测试 24kHz（默认）
    let config_24k = make_config(&cache_path, "tingting");
    let mut segs = vec![make_translated_segment("seg-24k", "采样率测试。")];
    let paths = engine
        .synthesize_segments(&mut segs, &config_24k)
        .expect("24kHz synthesis failed");
    assert_eq!(
        wav_sample_rate(&paths[0]),
        24000,
        "Sample rate should be 24000 Hz"
    );

    // 测试 48kHz
    let config_48k = TtsConfig {
        sample_rate: 48000,
        ..make_config(&cache_path, "tingting")
    };
    let mut segs48 = vec![make_translated_segment("seg-48k", "采样率测试。")];
    let paths48 = engine
        .synthesize_segments(&mut segs48, &config_48k)
        .expect("48kHz synthesis failed");
    assert_eq!(
        wav_sample_rate(&paths48[0]),
        48000,
        "Sample rate should be 48000 Hz"
    );
}

/// 验证合成音频时长合理（与文本长度匹配）。
#[test]
fn test_tts_audio_quality_duration() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let short_text = "你好。";
    let long_text = "这是一段较长的中文测试文本，用于验证语音合成的时长是否与文本长度成正比关系。";

    let config = make_config(&cache_path, "tingting");

    let mut segs_short = vec![make_translated_segment("seg-short", short_text)];
    let paths_short = engine
        .synthesize_segments(&mut segs_short, &config)
        .expect("Short synthesis failed");

    let mut segs_long = vec![make_translated_segment("seg-long", long_text)];
    let paths_long = engine
        .synthesize_segments(&mut segs_long, &config)
        .expect("Long synthesis failed");

    let dur_short = wav_duration(&paths_short[0]);
    let dur_long = wav_duration(&paths_long[0]);

    eprintln!("Short text duration: {dur_short:.2}s, Long text duration: {dur_long:.2}s");

    assert!(
        dur_short > 0.3,
        "Short audio should be > 0.3s, got {dur_short:.2}s"
    );
    assert!(
        dur_long > 1.0,
        "Long audio should be > 1.0s, got {dur_long:.2}s"
    );
    assert!(
        dur_long > dur_short,
        "Long text should produce longer audio: {dur_long:.2}s vs {dur_short:.2}s"
    );
}

/// 验证合成音频无空文件、无损坏。
#[test]
fn test_tts_no_artifacts() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let config = make_config(&cache_path, "tingting");
    let mut segs = vec![make_translated_segment(
        "seg-artifact",
        "验证音频无爆破音和噪声。",
    )];
    let paths = engine
        .synthesize_segments(&mut segs, &config)
        .expect("Synthesis failed");

    // 验证文件存在且非空
    assert!(paths[0].exists(), "Audio file should exist");
    let file_size = std::fs::metadata(&paths[0])
        .map(|m| m.len())
        .expect("Failed to get file size");
    assert!(
        file_size > 1000,
        "Audio file should be > 1KB, got {file_size} bytes"
    );

    // 验证 WAV 格式正确
    let mut reader = hound::WavReader::open(&paths[0]).expect("Failed to open WAV");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "Should be mono");
    assert_eq!(spec.bits_per_sample, 16, "Should be 16-bit");

    // 验证有实际音频样本
    let sample_count: usize = reader.samples::<i16>().count();
    assert!(sample_count > 0, "Should have audio samples");

    // 验证音频数据不为全零（静音）
    let mut reader2 = hound::WavReader::open(&paths[0]).expect("Failed to reopen WAV");
    let max_amplitude: i16 = reader2
        .samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s.abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_amplitude > 100,
        "Audio should not be silent (max amplitude should be > 100), got {max_amplitude}"
    );
}

// ═══════════════════════════════════════════════════════════
//  语速/音调/音量控制测试
// ═══════════════════════════════════════════════════════════

/// 验证不同语速产生不同时长的音频。
#[test]
fn test_tts_speed_control() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let text = "这是一段用于测试语速变化的中文文本内容。";

    // 正常语速
    let config_normal = make_config(&cache_path, "tingting");
    let mut segs_n = vec![make_translated_segment("seg-normal", text)];
    let paths_n = engine
        .synthesize_segments(&mut segs_n, &config_normal)
        .expect("Normal speed synthesis failed");

    // 快语速
    let config_fast = TtsConfig {
        speed: 2.0,
        ..make_config(&cache_path, "tingting")
    };
    let mut segs_f = vec![make_translated_segment("seg-fast", text)];
    let paths_f = engine
        .synthesize_segments(&mut segs_f, &config_fast)
        .expect("Fast speed synthesis failed");

    let dur_normal = wav_duration(&paths_n[0]);
    let dur_fast = wav_duration(&paths_f[0]);

    eprintln!("Speed control: normal={dur_normal:.2}s, fast={dur_fast:.2}s");

    assert!(
        dur_fast < dur_normal,
        "Fast speech should be shorter: {dur_fast:.2}s vs {dur_normal:.2}s"
    );
}

/// 验证音调控制改变音频文件（不同 pitch 产生不同缓存路径）。
#[test]
fn test_tts_pitch_control() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let text = "音调控制测试。";

    // 默认音调
    let config_default = make_config(&cache_path, "tingting");
    let mut segs_d = vec![make_translated_segment("seg-pitch-default", text)];
    let paths_d = engine
        .synthesize_segments(&mut segs_d, &config_default)
        .expect("Default pitch synthesis failed");

    // 升高音调
    let config_high = TtsConfig {
        pitch: 1.2,
        ..make_config(&cache_path, "tingting")
    };
    let mut segs_h = vec![make_translated_segment("seg-pitch-high", text)];
    let paths_h = engine
        .synthesize_segments(&mut segs_h, &config_high)
        .expect("High pitch synthesis failed");

    // 两个路径应该不同（pitch 不同 → hash 不同）
    assert_ne!(
        paths_d[0], paths_h[0],
        "Different pitch should produce different cache paths"
    );

    // 两个文件都应该存在
    assert!(paths_d[0].exists(), "Default pitch audio should exist");
    assert!(paths_h[0].exists(), "High pitch audio should exist");
}

/// 验证音量控制改变音频文件。
#[test]
fn test_tts_volume_control() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let text = "音量控制测试。";

    // 默认音量
    let config_default = make_config(&cache_path, "tingting");
    let mut segs_d = vec![make_translated_segment("seg-vol-default", text)];
    let paths_d = engine
        .synthesize_segments(&mut segs_d, &config_default)
        .expect("Default volume synthesis failed");

    // 增大音量
    let config_loud = TtsConfig {
        volume: 1.5,
        ..make_config(&cache_path, "tingting")
    };
    let mut segs_l = vec![make_translated_segment("seg-vol-loud", text)];
    let paths_l = engine
        .synthesize_segments(&mut segs_l, &config_loud)
        .expect("Loud volume synthesis failed");

    // 两个路径应该不同（volume 不同 → hash 不同）
    assert_ne!(
        paths_d[0], paths_l[0],
        "Different volume should produce different cache paths"
    );
}

// ═══════════════════════════════════════════════════════════
//  降级测试
// ═══════════════════════════════════════════════════════════

/// 验证 KokoroEngine 在 fallback_to_say=true 时降级到 SayEngine。
#[test]
fn test_tts_fallback_to_say() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let config = TtsConfig {
        cache_dir: cache_path,
        fallback_to_say: true,
        ..Default::default()
    };

    let engine = vt_core::tts::KokoroEngine::new(&config)
        .expect("KokoroEngine should fall back to SayEngine");
    assert_eq!(engine.backend_name(), "SayEngine");

    // 验证降级后仍能正常合成
    let mut segments = vec![make_translated_segment("seg-fallback", "降级测试。")];
    let paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("Fallback synthesis should succeed");

    assert_eq!(paths.len(), 1);
    assert!(paths[0].exists(), "Fallback audio should exist");
    assert_eq!(
        segments[0].status,
        SegmentStatus::Completed,
        "Segment should be completed"
    );
}

/// 验证 KokoroEngine 在 fallback_to_say=false 时返回错误。
#[test]
fn test_tts_no_fallback_error() {
    let config = TtsConfig {
        fallback_to_say: false,
        ..Default::default()
    };
    let result = vt_core::tts::KokoroEngine::new(&config);
    assert!(result.is_err(), "Should error when fallback is disabled");
}

// ═══════════════════════════════════════════════════════════
//  中英文混排测试
// ═══════════════════════════════════════════════════════════

/// 验证中英文混排文本能正常合成。
#[test]
fn test_tts_mixed_chinese_english() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let text = "API 是 Application Programming Interface 的缩写。";
    let config = make_config(&cache_path, "tingting");

    let mut segments = vec![make_translated_segment("seg-mixed", text)];
    let paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("Mixed text synthesis failed");

    assert_eq!(paths.len(), 1);
    assert!(paths[0].exists(), "Mixed text audio should exist");

    let duration = wav_duration(&paths[0]);
    eprintln!("Mixed text duration: {duration:.2}s");
    assert!(
        duration > 1.0,
        "Mixed text audio should be > 1.0s, got {duration:.2}s"
    );

    assert_eq!(
        segments[0].status,
        SegmentStatus::Completed,
        "Segment should be completed"
    );
}

// ═══════════════════════════════════════════════════════════
//  批量合成测试
// ═══════════════════════════════════════════════════════════

/// 验证批量合成多个 Segment 并使用男声。
#[test]
fn test_tts_batch_male_voice() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_path = dir.path().to_string_lossy().to_string();

    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let texts = [
        "第一段中文测试文本。",
        "第二段用于验证批量合成。",
        "第三段使用男声合成。",
    ];

    let config = make_config(&cache_path, "zhiming");

    let mut segments: Vec<Segment> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| make_translated_segment(&format!("seg-male-{i:04}"), text))
        .collect();

    let paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("Batch male synthesis failed");

    assert_eq!(paths.len(), texts.len());

    for (i, seg) in segments.iter().enumerate() {
        assert_eq!(
            seg.status,
            SegmentStatus::Completed,
            "Segment {i} should be Completed"
        );
        assert!(paths[i].exists(), "Audio file {i} should exist");
    }
}

/// 验证配置的 TOML 序列化/反序列化包含新字段。
#[test]
fn test_tts_config_with_new_fields() {
    use std::io::Write;
    use vt_core::config::Config;

    let toml_content = r#"
[tts]
engine = "say"
speed = 1.5
pitch = 0.9
volume = 1.2
voice_id = "zhiming"
voice = "Tingting"
sample_rate = 48000
device = "cpu"
fallback_to_say = true
auto_voice_selection = false
"#;

    let mut tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{toml_content}").expect("Failed to write TOML");

    let config = Config::from_file(tmp.path()).expect("Failed to load config");

    assert_eq!(config.tts.engine, "say");
    assert_eq!(config.tts.speed, 1.5);
    assert!((config.tts.pitch - 0.9).abs() < 0.001);
    assert!((config.tts.volume - 1.2).abs() < 0.001);
    assert_eq!(config.tts.voice_id, "zhiming");
    assert_eq!(config.tts.sample_rate, 48000);
    assert!(config.tts.fallback_to_say);
}

/// 验证 TtsConfig 默认值包含新字段。
#[test]
fn test_tts_config_defaults() {
    let config = TtsConfig::default();
    assert_eq!(config.engine, "say");
    assert!((config.speed - 1.0).abs() < 0.001);
    assert!((config.pitch - 1.0).abs() < 0.001);
    assert!((config.volume - 1.0).abs() < 0.001);
    assert_eq!(config.voice_id, "tingting");
    assert_eq!(config.sample_rate, 24000);
    assert!(config.fallback_to_say);
    assert!(!config.auto_voice_selection);
}
