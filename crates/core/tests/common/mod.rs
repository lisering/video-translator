//! 测试共享工具模块
//!
//! 提供跨测试文件复用的共享资源，包括：
//! - **WhisperEngine 全局单例**：所有 ASR 测试复用同一模型实例，避免重复加载（节省 ~30s+）
//! - **SayEngine 全局单例**：所有 TTS 测试复用同一引擎和持久化缓存目录
//! - **音频/视频生成辅助函数**：生成最短有效测试数据
//! - **TEST_QUICK 环境变量检查**：开发模式下跳过慢速测试
//!
//! # 设计原则
//! - 模型/引擎加载在测试套件初始化时**只执行一次**（`OnceLock`）
//! - 所有测试共享同一引擎实例，线程安全（`Send + Sync`）
//! - 精度验证不受影响：真实模型、真实推理

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use tempfile::TempDir;
use vt_core::asr::{ModelManager, WhisperConfig, WhisperEngine};
use vt_core::config::TtsConfig;
use vt_core::tts::SayEngine;

// ─── 环境变量检查 ─────────────────────────────────────────

/// 检查 `TEST_QUICK=1` 是否设置（开发模式下跳过慢速测试）。
///
/// CI 环境中不设置此变量，确保全量测试运行。
pub fn is_quick_mode() -> bool {
    std::env::var("TEST_QUICK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// 检查 `VT_SKIP_ASR_TESTS` 是否设置（CI 跳过需要模型的测试）。
pub fn asr_tests_disabled() -> bool {
    std::env::var("VT_SKIP_ASR_TESTS").is_ok()
}

/// 检查 `VT_SKIP_TTS_TESTS` 是否设置（CI 跳过需要网络的测试）。
pub fn tts_tests_disabled() -> bool {
    std::env::var("VT_SKIP_TTS_TESTS").is_ok()
}

/// 检查 `say` 命令是否可用（仅 macOS）。
pub fn say_available() -> bool {
    Command::new("say").arg("-v").arg("?").output().is_ok()
}
/// 检查 `ffmpeg` 是否可用。
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

// ─── Whisper 模型全局单例 ────────────────────────────────

/// 测试用模型文件名（最小英文模型，约 75MB）
const TEST_MODEL: &str = "ggml-tiny.en.bin";

/// 全局 WhisperEngine 单例（线程安全，只加载一次）。
///
/// 所有 ASR 集成测试复用此实例，避免重复加载模型文件。
/// 首次访问时触发模型下载和加载（约 5-10 秒），后续访问即时返回。
static WHISPER_ENGINE: OnceLock<Option<WhisperEngine>> = OnceLock::new();

/// 获取共享的 WhisperEngine 实例。
///
/// 首次调用时下载并加载模型，后续调用直接返回缓存的实例。
/// 若模型下载失败或测试被禁用，返回 `None`。
///
/// # 线程安全
/// 使用 `OnceLock` 保证只初始化一次，`WhisperEngine` 实现了 `Send + Sync`。
pub fn shared_whisper_engine() -> Option<&'static WhisperEngine> {
    WHISPER_ENGINE
        .get_or_init(|| {
            if asr_tests_disabled() {
                eprintln!("[shared_whisper_engine] VT_SKIP_ASR_TESTS is set, skipping");
                return None;
            }

            let manager = match ModelManager::new() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[shared_whisper_engine] Failed to create ModelManager: {e}");
                    return None;
                }
            };

            let model_path = match manager.ensure_model(TEST_MODEL) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[shared_whisper_engine] Failed to download model: {e}");
                    return None;
                }
            };

            eprintln!("[shared_whisper_engine] Loading Whisper model (one-time)...");
            let start = std::time::Instant::now();

            let config = WhisperConfig::default()
                .with_model_path(&model_path)
                .with_vad(true);

            match WhisperEngine::new(config) {
                Ok(engine) => {
                    eprintln!(
                        "[shared_whisper_engine] Model loaded in {:?} (shared across all ASR tests)",
                        start.elapsed()
                    );
                    Some(engine)
                }
                Err(e) => {
                    eprintln!("[shared_whisper_engine] Failed to create engine: {e}");
                    None
                }
            }
        })
        .as_ref()
}

// ─── 音频生成辅助函数 ─────────────────────────────────────

/// 使用 macOS `say` 生成英文语音 WAV 文件（16kHz mono PCM）。
///
/// 返回生成的 WAV 文件路径。若 `say` 或 `ffmpeg` 不可用则返回 `None`。
///
/// 优化点：使用最短有效文本，减少音频时长（~3s 而非 ~10s）。
pub fn generate_speech_wav(dir: &TempDir, name: &str, text: &str) -> Option<PathBuf> {
    if !say_available() || !ffmpeg_available() {
        eprintln!("Skipping: 'say' or 'ffmpeg' not available");
        return None;
    }

    let wav_path = dir.path().join(name);
    let aiff_path = dir.path().join("temp_speech.aiff");

    let say_status = Command::new("say")
        .args(["-o"])
        .arg(&aiff_path)
        .arg(text)
        .status()
        .ok()?;

    if !say_status.success() {
        eprintln!("Failed to generate speech with 'say'");
        return None;
    }

    let ff_status = Command::new("ffmpeg")
        .arg("-y")
        .args(["-i"])
        .arg(&aiff_path)
        .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
        .arg(&wav_path)
        .status()
        .ok()?;

    if !ff_status.success() {
        eprintln!("Failed to convert speech to WAV with ffmpeg");
        return None;
    }

    // 清理临时 AIFF 文件
    let _ = std::fs::remove_file(&aiff_path);

    Some(wav_path)
}

/// 生成正弦波 WAV（用于 VAD 测试，模拟语音能量）。
pub fn generate_sine_wav(dir: &TempDir, name: &str, duration_secs: u32, freq: u32) -> PathBuf {
    let path = dir.path().join(name);
    let src = format!("sine=frequency={freq}:duration={duration_secs}");
    Command::new("ffmpeg")
        .arg("-y")
        .args(["-f", "lavfi", "-i"])
        .arg(&src)
        .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
        .arg(&path)
        .status()
        .expect("Failed to generate sine WAV");
    assert!(path.exists(), "Sine WAV file was not created");
    path
}

/// 生成纯静音 WAV（用于 VAD 测试）。
pub fn generate_silence_wav(dir: &TempDir, name: &str, duration_secs: u32) -> Option<PathBuf> {
    if !ffmpeg_available() {
        return None;
    }
    let path = dir.path().join(name);
    let src = format!("anullsrc=duration={duration_secs}:channel_layout=mono:sample_rate=16000");
    let status = Command::new("ffmpeg")
        .arg("-y")
        .args(["-f", "lavfi", "-i"])
        .arg(&src)
        .args(["-c:a", "pcm_s16le"])
        .arg(&path)
        .status()
        .ok()?;
    if status.success() {
        Some(path)
    } else {
        None
    }
}

/// 生成包含静音段的测试音频（语音 + 静音 + 语音）。
///
/// 优化点：使用更短的语音段（1s + 1s 静音 + 1s），总时长 ~3s。
pub fn generate_audio_with_silence(dir: &TempDir, name: &str) -> Option<PathBuf> {
    if !say_available() || !ffmpeg_available() {
        eprintln!("Skipping: 'say' or 'ffmpeg' not available");
        return None;
    }

    let output_path = dir.path().join(name);
    let part1_aiff = dir.path().join("part1.aiff");
    let part2_aiff = dir.path().join("part2.aiff");
    let part1_wav = dir.path().join("part1.wav");
    let silence_wav = dir.path().join("silence.wav");
    let part2_wav = dir.path().join("part2.wav");

    // 生成两段短语音
    let s1 = Command::new("say")
        .args(["-o"])
        .arg(&part1_aiff)
        .arg("Hello first segment.")
        .status()
        .ok()?;
    if !s1.success() {
        return None;
    }

    let s2 = Command::new("say")
        .args(["-o"])
        .arg(&part2_aiff)
        .arg("Second segment.")
        .status()
        .ok()?;
    if !s2.success() {
        return None;
    }

    // 转换为 WAV
    let c1 = Command::new("ffmpeg")
        .arg("-y")
        .args(["-i"])
        .arg(&part1_aiff)
        .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
        .arg(&part1_wav)
        .status()
        .ok()?;
    if !c1.success() {
        return None;
    }

    let c2 = Command::new("ffmpeg")
        .arg("-y")
        .args(["-i"])
        .arg(&part2_aiff)
        .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
        .arg(&part2_wav)
        .status()
        .ok()?;
    if !c2.success() {
        return None;
    }

    // 生成 1 秒静音（缩短自 3 秒）
    let cs = Command::new("ffmpeg")
        .arg("-y")
        .args(["-f", "lavfi", "-i"])
        .arg("anullsrc=duration=1:channel_layout=mono:sample_rate=16000")
        .args(["-c:a", "pcm_s16le"])
        .arg(&silence_wav)
        .status()
        .ok()?;
    if !cs.success() {
        return None;
    }

    // 拼接：part1 + silence + part2
    let concat_list = dir.path().join("concat.txt");
    std::fs::write(
        &concat_list,
        format!(
            "file '{}'\nfile '{}'\nfile '{}'\n",
            part1_wav.display(),
            silence_wav.display(),
            part2_wav.display()
        ),
    )
    .ok()?;

    let cc = Command::new("ffmpeg")
        .arg("-y")
        .args(["-f", "concat", "-safe", "0", "-i"])
        .arg(&concat_list)
        .args(["-c:a", "pcm_s16le"])
        .arg(&output_path)
        .status()
        .ok()?;

    if !cc.success() {
        return None;
    }

    // 清理临时文件
    let _ = std::fs::remove_file(&part1_aiff);
    let _ = std::fs::remove_file(&part2_aiff);

    Some(output_path)
}

// ─── 视频生成辅助函数 ─────────────────────────────────────

/// 生成测试视频（指定时长，小尺寸以加速编码）。
///
/// 优化点：使用 320x240 分辨率（而非 1280x720），大幅减少编码时间。
pub fn generate_test_video(dir: &TempDir, name: &str, duration: u32) -> PathBuf {
    let path = dir.path().join(name);
    let video_src = format!("testsrc=duration={duration}:size=320x240:rate=30");
    let audio_src = format!("sine=frequency=440:duration={duration}");

    let status = Command::new("ffmpeg")
        .arg("-y")
        .args(["-f", "lavfi", "-i"])
        .arg(&video_src)
        .args(["-f", "lavfi", "-i"])
        .arg(&audio_src)
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
        .arg(&path)
        .status()
        .expect("Failed to spawn ffmpeg for test video generation");

    assert!(status.success(), "ffmpeg failed to generate test video");
    assert!(path.exists(), "Test video file was not created");
    path
}

/// 生成测试 WAV 音频（指定时长，16kHz 单声道 PCM）。
pub fn generate_test_wav(dir: &TempDir, name: &str, duration: u32) -> PathBuf {
    let path = dir.path().join(name);
    let audio_src = format!("sine=frequency=500:duration={duration}");

    let status = Command::new("ffmpeg")
        .arg("-y")
        .args(["-f", "lavfi", "-i"])
        .arg(&audio_src)
        .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
        .arg(&path)
        .status()
        .expect("Failed to spawn ffmpeg for test audio generation");

    assert!(status.success(), "ffmpeg failed to generate test audio");
    assert!(path.exists(), "Test audio file was not created");
    path
}

// ─── TTS 共享引擎单例 ─────────────────────────────────────

/// 检查 macOS `say` 命令和中文语音是否可用。
pub fn should_skip_tts() -> bool {
    if tts_tests_disabled() {
        eprintln!("Skipping: VT_SKIP_TTS_TESTS is set");
        return true;
    }
    if !say_available() {
        eprintln!("Skipping: 'say' command not available (non-macOS?)");
        return true;
    }
    false
}

/// 全局共享 TTS 引擎单例（线程安全，只创建一次）。
///
/// 所有 TTS 集成测试复用此实例及其缓存目录，
/// 相同文本+音色+语速的合成结果会被缓存复用。
static SHARED_TTS_ENGINE: OnceLock<Option<SayEngine>> = OnceLock::new();

/// 获取共享的 SayEngine 实例。
///
/// 首次调用时创建引擎并初始化持久化缓存目录，
/// 后续调用直接返回缓存的实例。
/// 若 TTS 测试被禁用或 `say` 不可用，返回 `None`。
pub fn shared_tts_engine() -> Option<&'static SayEngine> {
    SHARED_TTS_ENGINE
        .get_or_init(|| {
            if should_skip_tts() {
                return None;
            }

            // 使用持久化缓存目录，跨测试运行也能命中缓存
            let cache_dir = dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("video-translator")
                .join("test_tts_cache");

            let config = TtsConfig {
                speed: 1.0,
                voice: "Tingting".to_string(),
                cache_dir: cache_dir.to_string_lossy().to_string(),
                parallel_tasks: 2,
                ..Default::default()
            };

            match SayEngine::new(&config) {
                Ok(engine) => {
                    eprintln!(
                        "[shared_tts_engine] Engine created with cache: {:?} (shared across all TTS tests)",
                        cache_dir
                    );
                    Some(engine)
                }
                Err(e) => {
                    eprintln!("[shared_tts_engine] Failed to create engine: {e}");
                    None
                }
            }
        })
        .as_ref()
}

/// 获取共享 TTS 引擎的配置。
pub fn shared_tts_config() -> TtsConfig {
    TtsConfig {
        speed: 1.0,
        voice: "Tingting".to_string(),
        cache_dir: dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("video-translator")
            .join("test_tts_cache")
            .to_string_lossy()
            .to_string(),
        parallel_tasks: 2,
        ..Default::default()
    }
}
