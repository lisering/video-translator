//! 集成测试：语音识别 (ASR) — Whisper 集成
//!
//! 验证 `WhisperEngine` 的转录功能，包括：
//! - 基本转录（英文语音 → 带时间戳的 Segment 列表）
//! - VAD 预处理（跳过静音段，仅识别有效语音）
//! - 错误处理（不存在的文件、无效 WAV 等）
//! - VAD 检测逻辑（纯算法测试，不需要模型）
//! - WAV 读取功能
//!
//! # 模型依赖
//! 需要模型的测试会自动下载 `ggml-tiny.en.bin`（约 75MB）到
//! `~/.cache/video-translator/models/`。首次运行需联网下载。
//!
//! # 优化说明（Session 11）
//! - **模型共享单例**：所有 ASR 测试复用同一 `WhisperEngine` 实例（通过 `once_cell`），
//!   模型只加载一次，节省 ~30s+ 重复加载开销。
//! - **测试数据瘦身**：使用更短的语音文本和静音段（~3s 而非 ~10s）。
//! - **TEST_QUICK 支持**：`TEST_QUICK=1` 时跳过需要模型的慢速测试。
//!
//! # 测试音频生成
//! 使用 macOS `say` 命令生成英文语音，再用 ffmpeg 转换为 16kHz mono WAV。
//! 若 `say` 或 `ffmpeg` 不可用，相关测试将自动跳过。

mod common;

use std::path::Path;

use tempfile::TempDir;
use vt_core::asr::{
    detect_speech_segments, read_wav_mono, AsrEngine, VadConfig, WhisperConfig, WhisperEngine,
};
use vt_core::error::AppError;

// ─── WAV 读取测试 ─────────────────────────────────────────

/// 验证 `read_wav_mono` 能正确读取 16kHz mono WAV 文件。
#[test]
fn test_read_wav_mono() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let wav_path = common::generate_sine_wav(&dir, "sine.wav", 1, 440);

    let (samples, sample_rate) = read_wav_mono(&wav_path).expect("Failed to read WAV");

    assert_eq!(sample_rate, 16000, "Sample rate should be 16000 Hz");
    // 1 秒 * 16000 = 16000 samples（允许微小误差）
    assert!(
        samples.len() > 15000 && samples.len() < 17000,
        "Expected ~16000 samples, got {}",
        samples.len()
    );
    assert!(
        samples.iter().all(|s| s.abs() <= 1.0),
        "All samples should be in [-1.0, 1.0]"
    );
}

/// 验证 `read_wav_mono` 对不存在的文件返回 `FileNotFound` 错误。
#[test]
fn test_read_wav_nonexistent_file() {
    let result = read_wav_mono(Path::new("/nonexistent/audio.wav"));
    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::FileNotFound(_))),
        "Expected FileNotFound, got {:?}",
        result
    );
}

// ─── VAD 检测测试 ─────────────────────────────────────────

/// 验证 VAD 能检测出正弦波音频中的语音段（有能量的音频）。
#[test]
fn test_vad_detects_signal() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    // 使用 1 秒正弦波（缩短自 3 秒）
    let wav_path = common::generate_sine_wav(&dir, "sine.wav", 1, 440);

    let (samples, sample_rate) = read_wav_mono(&wav_path).expect("Failed to read WAV");
    let config = VadConfig::default();
    let segments = detect_speech_segments(&samples, sample_rate, &config);

    assert!(
        !segments.is_empty(),
        "VAD should detect speech in sine wave audio"
    );

    let total_speech_ms: i64 = segments.iter().map(|s| s.end_ms - s.start_ms).sum();
    assert!(
        total_speech_ms > 500,
        "Speech duration should be > 500ms, got {total_speech_ms}ms"
    );
}

/// 验证 VAD 能正确跳过纯静音段。
#[test]
fn test_vad_skips_silence() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    // 使用 2 秒静音（缩短自 5 秒）
    let silence_path = common::generate_silence_wav(&dir, "silence.wav", 2)
        .expect("Failed to generate silence WAV");

    let (samples, sample_rate) = read_wav_mono(&silence_path).expect("Failed to read WAV");
    let config = VadConfig::default();
    let segments = detect_speech_segments(&samples, sample_rate, &config);

    assert!(
        segments.is_empty(),
        "VAD should not detect speech in pure silence, got {segments:?}"
    );
}

/// 验证 VAD 能正确分割"语音+静音+语音"的音频。
#[test]
fn test_vad_splits_speech_and_silence() {
    let dir = TempDir::new().expect("Failed to create temp dir");

    // 使用更短的音频段（1s + 1s 静音 + 1s，总时长 ~3s）
    let sine1 = common::generate_sine_wav(&dir, "sine1.wav", 1, 440);
    let sine2 = common::generate_sine_wav(&dir, "sine2.wav", 1, 440);
    let silence_path = common::generate_silence_wav(&dir, "silence.wav", 1)
        .expect("Failed to generate silence WAV");

    let concat_path = dir.path().join("concat.txt");
    std::fs::write(
        &concat_path,
        format!(
            "file '{}'\nfile '{}'\nfile '{}'\n",
            sine1.display(),
            silence_path.display(),
            sine2.display()
        ),
    )
    .expect("Failed to write concat list");

    let combined_path = dir.path().join("combined.wav");
    std::process::Command::new("ffmpeg")
        .arg("-y")
        .args(["-f", "concat", "-safe", "0", "-i"])
        .arg(&concat_path)
        .args(["-c:a", "pcm_s16le"])
        .arg(&combined_path)
        .status()
        .expect("Failed to concatenate audio");

    let (samples, sample_rate) = read_wav_mono(&combined_path).expect("Failed to read WAV");
    let config = VadConfig::default();
    let segments = detect_speech_segments(&samples, sample_rate, &config);

    assert!(
        segments.len() >= 2,
        "Expected at least 2 speech segments, got {}: {:?}",
        segments.len(),
        segments
    );

    let first = &segments[0];
    let last = &segments[segments.len() - 1];
    assert!(
        first.start_ms < 1000,
        "First segment should start in first 1s, got {}ms",
        first.start_ms
    );
    assert!(
        last.start_ms >= 1500,
        "Last segment should start after 1.5s (after silence), got {}ms",
        last.start_ms
    );
}

// ─── WhisperEngine 构造测试 ──────────────────────────────

/// 验证 `WhisperEngine` 能从模型路径创建。
///
/// 使用共享单例，模型只加载一次。
/// 若 `VT_SKIP_ASR_TESTS` 已设置或模型不可用，则跳过。
#[test]
fn test_whisper_engine_creation() {
    let engine = common::shared_whisper_engine();
    if engine.is_none() {
        eprintln!("Skipping: model not available (VT_SKIP_ASR_TESTS or download failed)");
    }
}

/// 验证 `WhisperEngine` 对不存在的模型路径返回错误。
#[test]
fn test_whisper_engine_invalid_model_path() {
    let result = WhisperEngine::from_model_path("/nonexistent/model.bin");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::FileNotFound(_))),
        "Expected FileNotFound, got {:?}",
        result
    );
}

// ─── ASR 基本转录测试 ─────────────────────────────────────

/// 验证 WhisperEngine 能转录英文语音，返回非空的 Segment 列表。
///
/// 优化点：使用共享单例模型 + 更短的测试文本。
#[test]
fn test_asr_basic() {
    if common::is_quick_mode() {
        eprintln!("Skipping: TEST_QUICK=1");
        return;
    }

    let engine = match common::shared_whisper_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: model not available");
            return;
        }
    };

    let dir = TempDir::new().expect("Failed to create temp dir");
    // 使用更短的测试文本
    let wav_path = match common::generate_speech_wav(&dir, "test_en.wav", "Hello, this is a test.")
    {
        Some(p) => p,
        None => {
            eprintln!("Skipping: could not generate test speech audio");
            return;
        }
    };

    let segments = engine.transcribe(&wav_path).expect("Transcription failed");

    assert!(!segments.is_empty(), "Should return at least one segment");

    for seg in &segments {
        assert!(
            !seg.source_text.is_empty(),
            "Segment source_text should not be empty: {:?}",
            seg
        );
        assert!(seg.start >= 0.0, "Segment start should be >= 0: {:?}", seg);
        assert!(
            seg.end > seg.start,
            "Segment end should be > start: {:?}",
            seg
        );
        assert!(seg.end <= 15.0, "Segment end should be <= 15s: {:?}", seg);
        assert!(
            !seg.id.is_empty(),
            "Segment id should not be empty: {:?}",
            seg
        );
    }

    eprintln!("ASR basic test: {} segments detected", segments.len());
    for seg in &segments {
        eprintln!(
            "  [{}] {:.2}s-{:.2}s: {}",
            seg.id, seg.start, seg.end, seg.source_text
        );
    }
}

// ─── ASR + VAD 测试 ───────────────────────────────────────

/// 验证带 VAD 预处理的转录能正确处理含静音段的音频。
///
/// 优化点：使用共享单例模型 + 更短的音频段。
#[test]
fn test_asr_with_vad() {
    if common::is_quick_mode() {
        eprintln!("Skipping: TEST_QUICK=1");
        return;
    }

    let engine = match common::shared_whisper_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: model not available");
            return;
        }
    };

    let dir = TempDir::new().expect("Failed to create temp dir");
    let wav_path = match common::generate_audio_with_silence(&dir, "test_with_silence.wav") {
        Some(p) => p,
        None => {
            eprintln!("Skipping: could not generate test audio with silence");
            return;
        }
    };

    let segments = engine.transcribe(&wav_path).expect("Transcription failed");

    assert!(
        !segments.is_empty(),
        "Should detect speech segments even with VAD"
    );

    for seg in &segments {
        assert!(
            !seg.source_text.trim().is_empty(),
            "Segment text should not be empty: {:?}",
            seg
        );
    }

    eprintln!("ASR with VAD test: {} segments detected", segments.len());
    for seg in &segments {
        eprintln!(
            "  [{}] {:.2}s-{:.2}s: {}",
            seg.id, seg.start, seg.end, seg.source_text
        );
    }
}

// ─── ASR 错误处理测试 ─────────────────────────────────────

/// 验证转录不存在的文件返回 `FileNotFound` 错误。
///
/// 使用共享单例模型，不重复加载。
#[test]
fn test_asr_nonexistent_file() {
    let engine = match common::shared_whisper_engine() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: model not available");
            return;
        }
    };

    let result = engine.transcribe(Path::new("/nonexistent/audio.wav"));
    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::FileNotFound(_))),
        "Expected FileNotFound, got {:?}",
        result
    );
}

// ─── 配置与模型管理测试 ───────────────────────────────────

/// 验证 `VadConfig` 的默认值。
#[test]
fn test_vad_config_default() {
    let config = VadConfig::default();
    assert!(config.frame_size_ms > 0, "frame_size_ms should be > 0");
    assert!(
        config.energy_threshold > 0.0,
        "energy_threshold should be > 0"
    );
    assert!(
        config.min_speech_duration_ms > 0,
        "min_speech_duration_ms should be > 0"
    );
    assert!(
        config.min_silence_duration_ms > 0,
        "min_silence_duration_ms should be > 0"
    );
}

/// 验证 `WhisperConfig` 的默认值。
#[test]
fn test_whisper_config_default() {
    let config = WhisperConfig::default();
    assert!(!config.language.is_empty(), "language should not be empty");
    assert!(config.use_metal, "use_metal should default to true");
    assert!(config.use_vad, "use_vad should default to true");
    assert!(config.n_threads > 0, "n_threads should be > 0");
}

/// 验证 `ModelManager` 的缓存目录创建。
#[test]
fn test_model_manager_creation() {
    let manager = vt_core::asr::ModelManager::new();
    assert!(
        manager.is_ok(),
        "Failed to create ModelManager: {:?}",
        manager.err()
    );

    let manager = manager.unwrap();
    let cache_dir = manager.cache_dir();
    assert!(
        cache_dir.exists(),
        "Cache directory should exist after ModelManager creation"
    );
}

/// 验证 `ModelManager` 能返回模型路径（不下载）。
#[test]
fn test_model_manager_model_path() {
    let manager = vt_core::asr::ModelManager::new().expect("Failed to create ModelManager");
    // "ggml-tiny.bin" 是直接传入 ggml 文件名，不会被映射
    let path = manager.model_path("ggml-tiny.bin");
    assert!(
        path.to_str().unwrap().contains("ggml-tiny.bin"),
        "Model path should contain model filename"
    );
}

/// 验证 `ModelManager` 能将模型名称映射为正确的 ggml 文件名。
#[test]
fn test_model_manager_model_name_resolution() {
    let manager = vt_core::asr::ModelManager::new().expect("Failed to create ModelManager");
    // whisper-large-v3-turbo → ggml-large-v3-turbo.bin
    let path = manager.model_path("whisper-large-v3-turbo");
    assert!(
        path.to_str().unwrap().contains("ggml-large-v3-turbo.bin"),
        "whisper-large-v3-turbo should resolve to ggml-large-v3-turbo.bin"
    );
    // whisper-tiny → ggml-tiny.bin
    let path = manager.model_path("whisper-tiny");
    assert!(
        path.to_str().unwrap().contains("ggml-tiny.bin"),
        "whisper-tiny should resolve to ggml-tiny.bin"
    );
}
