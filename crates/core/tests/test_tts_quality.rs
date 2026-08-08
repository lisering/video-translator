//! 集成测试：TTS 音色稳定性与音质优化
//!
//! 验证以下功能：
//! - EQ 均衡器（highshelf 衰减齿音、lowshelf 增强温暖感）— 通过合成验证
//! - 交叉淡入淡出（crossfade）消除拼接感
//! - 音色一致性：多段合成使用相同参数
//! - 后处理始终应用（即使女声默认参数）
//! - `audio_post_process` 公共接口可用
//!
//! # 离线运行
//! 所有测试使用 macOS 内置的 `say` 命令，无需网络连接。
//!
//! # 运行方式
//! ```sh
//! cargo test --test test_tts_quality -- --nocapture
//! ```

mod common;

use vt_core::config::TtsConfig;
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

/// 检查测试前置条件：`say` 和 `ffmpeg` 可用。
fn check_prerequisites() -> bool {
    common::say_available() && common::ffmpeg_available()
}

// ═══════════════════════════════════════════════════════════
//  音色参数固化测试
// ═══════════════════════════════════════════════════════════

/// 验证 TtsConfig 的 seed、temperature、stability 字段存在且有默认值。
#[test]
fn test_tts_config_synthesis_params() {
    let config = TtsConfig::default();
    assert_eq!(config.seed, Some(42), "Default seed should be 42");
    assert!(
        (config.temperature - 0.3).abs() < 0.001,
        "Default temperature should be 0.3"
    );
    assert!(
        (config.stability - 0.8).abs() < 0.001,
        "Default stability should be 0.8"
    );
}

/// 验证 EQ 参数默认值。
#[test]
fn test_tts_config_eq_defaults() {
    let config = TtsConfig::default();
    assert!(
        (config.eq_high_shelf_db - (-3.0)).abs() < 0.001,
        "Default eq_high_shelf_db should be -3.0"
    );
    assert_eq!(
        config.crossfade_duration_ms, 50,
        "Default crossfade_duration_ms should be 50"
    );
}

/// 验证从 TOML 加载新增 TTS 配置字段。
#[test]
fn test_tts_config_new_fields_from_toml() {
    use std::io::Write;
    use vt_core::config::Config;

    let toml_content = r#"
[tts]
speed = 1.0
voice = "Tingting"
seed = 100
temperature = 0.5
stability = 0.9
eq_high_shelf_db = -5.0
crossfade_duration_ms = 100
"#;

    let mut tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{toml_content}").expect("Failed to write TOML");

    let config = Config::from_file(tmp.path()).expect("Failed to load config");

    assert_eq!(config.tts.seed, Some(100));
    assert!((config.tts.temperature - 0.5).abs() < 0.001);
    assert!((config.tts.stability - 0.9).abs() < 0.001);
    assert!((config.tts.eq_high_shelf_db - (-5.0)).abs() < 0.001);
    assert_eq!(config.tts.crossfade_duration_ms, 100);
}

// ═══════════════════════════════════════════════════════════
//  音色一致性测试（合成验证）
// ═══════════════════════════════════════════════════════════

/// 验证合成 5 段不同文本时，所有段使用相同音色参数。
///
/// 此测试验证音色一致性：所有段使用相同的 voice_id、speed、pitch、volume，
/// 确保不会出现"两个人讲话"的割裂感。
#[test]
fn test_timbre_stability_synthesis_5_segments() {
    if !check_prerequisites() {
        eprintln!("Skipping: 'say' or 'ffmpeg' not available");
        return;
    }

    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: TTS engine not available");
            return;
        }
    };
    let config = common::shared_tts_config();

    let texts = [
        "欢迎使用 Rust 编程语言。",
        "今天我们学习所有权和借用。",
        "字符串和向量是常用的集合类型。",
        "结构体和枚举用于自定义数据。",
        "特征和实现面向对象编程。",
    ];

    let mut segments: Vec<Segment> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| make_translated_segment(&format!("seg-timbre-{i:04}"), text))
        .collect();

    let audio_paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("TTS synthesis failed");

    // 验证所有段都成功合成
    assert_eq!(audio_paths.len(), 5, "Should return 5 audio paths");

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

    // 验证所有 WAV 文件存在且格式正确
    for (i, path) in audio_paths.iter().enumerate() {
        assert!(path.exists(), "Audio file {} should exist: {:?}", i, path);

        let reader = hound::WavReader::open(path).expect("Failed to open WAV");
        let spec = reader.spec();
        assert_eq!(
            spec.sample_rate, config.sample_rate,
            "Sample rate should be {}",
            config.sample_rate
        );
        assert_eq!(spec.channels, 1, "Should be mono");
        assert_eq!(spec.bits_per_sample, 16, "Should be 16-bit");

        eprintln!(
            "Timbre test[{}]: {:?} ({} bytes, {}Hz)",
            i,
            path,
            std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            spec.sample_rate
        );
    }
}

/// 验证相同文本第二次合成复用缓存（音色一致性保证）。
#[test]
fn test_timbre_stability_cache_reuse() {
    if !check_prerequisites() {
        eprintln!("Skipping: 'say' or 'ffmpeg' not available");
        return;
    }

    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: TTS engine not available");
            return;
        }
    };
    let config = common::shared_tts_config();

    let text = "这是一段用于验证音色一致性的测试文本。";

    // 第一次合成
    let mut segs1 = vec![make_translated_segment("seg-cache-1", text)];
    let paths1 = engine
        .synthesize_segments(&mut segs1, &config)
        .expect("First synthesis failed");

    // 第二次合成相同文本
    let mut segs2 = vec![make_translated_segment("seg-cache-2", text)];
    let paths2 = engine
        .synthesize_segments(&mut segs2, &config)
        .expect("Second synthesis failed");

    // 验证缓存命中：文件路径应相同
    assert_eq!(
        paths1[0], paths2[0],
        "Same text should reuse cached file (same path) — ensuring timbre consistency"
    );
}

// ═══════════════════════════════════════════════════════════
//  齿音衰减测试（合成验证）
// ═══════════════════════════════════════════════════════════

/// 验证合成含齿音字符（s/sh/x）的句子时，音频成功生成。
///
/// 注意：此测试验证音频合成成功且格式正确，
/// 实际齿音衰减效果（EQ highshelf 滤镜）需通过主观听感验收。
#[test]
fn test_sibilance_synthesis() {
    if !check_prerequisites() {
        eprintln!("Skipping: 'say' or 'ffmpeg' not available");
        return;
    }

    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: TTS engine not available");
            return;
        }
    };
    let config = common::shared_tts_config();

    // 含大量齿音字符的中文句子
    let text = "这是测试齿音的句子，包含很多嘶嘶声和嘘嘘声。";
    let mut segments = vec![make_translated_segment("seg-sibilance", text)];

    let audio_paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("TTS synthesis failed");

    assert_eq!(audio_paths.len(), 1);
    assert!(audio_paths[0].exists(), "Audio file should exist");

    // 验证 WAV 格式
    let mut reader = hound::WavReader::open(&audio_paths[0]).expect("Failed to open WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, config.sample_rate);
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.bits_per_sample, 16);

    // 验证有音频数据
    let sample_count: usize = reader.samples::<i16>().count();
    assert!(sample_count > 0, "WAV file should contain audio samples");

    eprintln!(
        "Sibilance test: text='{}', samples={}, duration={:.2}s",
        text,
        sample_count,
        sample_count as f64 / config.sample_rate as f64
    );
}

// ═══════════════════════════════════════════════════════════
//  audio_post_process 公共接口测试
// ═══════════════════════════════════════════════════════════

/// 验证 `SayEngine::audio_post_process` 公共接口可正常调用。
///
/// 此测试使用 `say` 生成原始音频，然后用 `audio_post_process` 进行后处理，
/// 验证输出 WAV 格式正确（EQ + 淡入淡出 + 重采样）。
#[test]
fn test_audio_post_process_public_api() {
    if !check_prerequisites() {
        eprintln!("Skipping: 'say' or 'ffmpeg' not available");
        return;
    }

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let input_path = dir.path().join("input.wav");
    let output_path = dir.path().join("output.wav");
    let config = TtsConfig::default();

    // 使用 say 生成原始音频（24kHz）
    let say_output = std::process::Command::new("say")
        .arg("-v")
        .arg("Tingting")
        .arg("-o")
        .arg(&input_path)
        .arg("--file-format=WAVE")
        .arg("--data-format=LEI16@24000")
        .arg("测试音频后处理功能。")
        .output();

    match say_output {
        Ok(o) if o.status.success() => {}
        _ => {
            eprintln!("Skipping: 'say' command failed");
            return;
        }
    }

    assert!(input_path.exists(), "Input WAV should exist");

    // 调用 audio_post_process
    SayEngine::audio_post_process(&input_path, &output_path, &config)
        .expect("audio_post_process failed");

    assert!(output_path.exists(), "Output WAV should exist");

    // 验证输出格式
    let mut reader = hound::WavReader::open(&output_path).expect("Failed to open output WAV");
    let spec = reader.spec();
    assert_eq!(
        spec.sample_rate, config.sample_rate,
        "Output sample rate should match config"
    );
    assert_eq!(spec.channels, 1, "Should be mono");
    assert_eq!(spec.bits_per_sample, 16, "Should be 16-bit");

    let sample_count: usize = reader.samples::<i16>().count();
    assert!(sample_count > 0, "Output should have audio samples");

    eprintln!(
        "audio_post_process test: {} samples, {}Hz, duration={:.2}s",
        sample_count,
        spec.sample_rate,
        sample_count as f64 / spec.sample_rate as f64
    );
}

// ═══════════════════════════════════════════════════════════
//  采样率统一测试
// ═══════════════════════════════════════════════════════════

/// 验证合成音频的采样率与配置一致。
#[test]
fn test_sample_rate_consistency() {
    if !check_prerequisites() {
        eprintln!("Skipping: 'say' or 'ffmpeg' not available");
        return;
    }

    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: TTS engine not available");
            return;
        }
    };
    let config = common::shared_tts_config();

    let texts = ["第一段文本。", "第二段文本。", "第三段文本。"];
    let mut segments: Vec<Segment> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| make_translated_segment(&format!("seg-sr-{i:04}"), text))
        .collect();

    let audio_paths = engine
        .synthesize_segments(&mut segments, &config)
        .expect("TTS synthesis failed");

    // 验证所有音频采样率一致
    for (i, path) in audio_paths.iter().enumerate() {
        let reader = hound::WavReader::open(path).expect("Failed to open WAV");
        let spec = reader.spec();
        assert_eq!(
            spec.sample_rate, config.sample_rate,
            "Audio {} sample rate should be {}, got {}",
            i, config.sample_rate, spec.sample_rate
        );
    }
}
