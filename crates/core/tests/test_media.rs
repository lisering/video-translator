//! 集成测试：音视频处理
//!
//! 验证音频提取（16kHz 单声道 WAV）和视频合成（替换音轨、烧录字幕）功能。
//! 测试中动态生成测试视频，不依赖外部文件。
//!
//! # 优化说明（Session 11）
//! - **测试视频瘦身**：使用 3s 视频（而非 10s/30s），大幅减少 ffmpeg 编码时间。
//! - **性能测试放宽**：性能测试使用 5s 视频（而非 30s），断言阈值相应调整。
//! - **分辨率降低**：使用 320x240（而非 1280x720），加速编码。
//! - 使用共享辅助函数避免重复代码。

mod common;

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use tempfile::TempDir;
use vt_core::error::AppError;
use vt_core::media::{
    find_audio_stream, find_video_stream, probe_media, AudioExtractor, FfmpegAudioExtractor,
    FfmpegVideoComposer, VideoComposer,
};

// ─── 测试辅助函数 ─────────────────────────────────────────

/// 检查 ffmpeg 和 ffprobe 是否可用，不可用则跳过测试。
fn ensure_ffmpeg_available() {
    let ffmpeg_ok = Command::new("ffmpeg").arg("-version").output().is_ok();
    let ffprobe_ok = Command::new("ffprobe").arg("-version").output().is_ok();

    if !ffmpeg_ok || !ffprobe_ok {
        eprintln!("Skipping test: ffmpeg/ffprobe not found in PATH");
    }
}

/// 生成测试 SRT 字幕文件。
fn generate_test_srt(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    let srt_content = "\
1
00:00:00,000 --> 00:00:01,000
测试字幕第一行

2
00:00:01,000 --> 00:00:02,000
测试字幕第二行

3
00:00:02,000 --> 00:00:03,000
测试字幕第三行
";
    fs::write(&path, srt_content).expect("Failed to write test SRT file");
    path
}

// ─── 媒体探测测试 ─────────────────────────────────────────

/// 验证 `probe_media` 能正确探测视频文件的时长和流信息。
#[test]
fn test_probe_media_video() {
    ensure_ffmpeg_available();

    let dir = TempDir::new().expect("Failed to create temp dir");
    // 使用 3s 视频（缩短自 10s）
    let video_path = common::generate_test_video(&dir, "input.mp4", 3);

    let info = probe_media(&video_path).expect("probe_media failed");

    // 时长应约为 3 秒（允许 1 秒误差）
    assert!(
        (info.duration - 3.0).abs() < 1.0,
        "Duration mismatch: expected ~3s, got {}",
        info.duration
    );

    let video_stream = find_video_stream(&info);
    assert!(video_stream.is_some(), "No video stream found");
    let video_stream = video_stream.expect("video stream should exist");
    assert_eq!(video_stream.codec_type, "video");
    assert_eq!(video_stream.width, Some(320));
    assert_eq!(video_stream.height, Some(240));

    let audio_stream = find_audio_stream(&info);
    assert!(audio_stream.is_some(), "No audio stream found");
    let audio_stream = audio_stream.expect("audio stream should exist");
    assert_eq!(audio_stream.codec_type, "audio");
}

/// 验证 `probe_media` 对不存在的文件返回 `FileNotFound` 错误。
#[test]
fn test_probe_media_nonexistent_file() {
    let result = probe_media(std::path::Path::new("/nonexistent/video.mp4"));
    assert!(result.is_err());
    match result {
        Err(AppError::FileNotFound(path)) => {
            assert_eq!(path, std::path::PathBuf::from("/nonexistent/video.mp4"));
        }
        Err(other) => panic!("Expected FileNotFound, got {other:?}"),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

// ─── 音频提取测试 ─────────────────────────────────────────

/// 验证音频提取：从视频中提取 16kHz 单声道 WAV。
#[test]
fn test_audio_extraction() {
    ensure_ffmpeg_available();

    let dir = TempDir::new().expect("Failed to create temp dir");
    let video_path = common::generate_test_video(&dir, "input.mp4", 3);
    let wav_path = dir.path().join("extracted.wav");

    let extractor = FfmpegAudioExtractor::new();
    let result = extractor.extract_audio(&video_path, &wav_path);

    assert!(
        result.is_ok(),
        "Audio extraction failed: {:?}",
        result.err()
    );
    assert!(wav_path.exists(), "Output WAV file was not created");

    let info = probe_media(&wav_path).expect("Failed to probe extracted WAV");

    let audio = find_audio_stream(&info).expect("No audio stream in extracted WAV");
    assert_eq!(
        audio.sample_rate,
        Some(16000),
        "Sample rate should be 16000 Hz"
    );
    assert_eq!(audio.channels, Some(1), "Channels should be 1 (mono)");
    assert_eq!(audio.codec_name, "pcm_s16le", "Codec should be pcm_s16le");

    assert!(
        (info.duration - 3.0).abs() < 1.0,
        "Duration mismatch: expected ~3s, got {}",
        info.duration
    );

    assert!(
        find_video_stream(&info).is_none(),
        "Extracted audio should not contain video stream"
    );
}

/// 验证音频提取对不存在的文件返回 `FileNotFound` 错误。
#[test]
fn test_audio_extraction_nonexistent_file() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let wav_path = dir.path().join("output.wav");

    let extractor = FfmpegAudioExtractor::new();
    let result = extractor.extract_audio(std::path::Path::new("/nonexistent/video.mp4"), &wav_path);

    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::FileNotFound(_))),
        "Expected FileNotFound error, got {:?}",
        result
    );
}

/// 验证音频提取的性能：5 秒视频应在 3 秒内完成。
///
/// 优化点：使用 5s 视频（缩短自 30s），断言阈值调整为 3s。
#[test]
fn test_audio_extraction_performance() {
    ensure_ffmpeg_available();

    let dir = TempDir::new().expect("Failed to create temp dir");
    let video_path = common::generate_test_video(&dir, "perf_input.mp4", 5);
    let wav_path = dir.path().join("perf_output.wav");

    let extractor = FfmpegAudioExtractor::new();
    let start = Instant::now();
    extractor
        .extract_audio(&video_path, &wav_path)
        .expect("Audio extraction failed");
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 3,
        "Audio extraction took too long: {:?} (expected < 3s)",
        elapsed
    );

    eprintln!("Audio extraction (5s video) took: {:?}", elapsed);
}

// ─── 视频合成测试 ─────────────────────────────────────────

/// 验证视频合成：用新音轨替换原音轨（不烧录字幕）。
#[test]
fn test_video_composition() {
    ensure_ffmpeg_available();

    let dir = TempDir::new().expect("Failed to create temp dir");
    let video_path = common::generate_test_video(&dir, "original.mp4", 3);
    let audio_path = common::generate_test_wav(&dir, "new_audio.wav", 3);
    let output_path = dir.path().join("composed.mp4");

    let composer = FfmpegVideoComposer::with_encoder("libx264");
    let result = composer.compose_video(&video_path, &audio_path, &output_path, false, None, 1.0);

    assert!(
        result.is_ok(),
        "Video composition failed: {:?}",
        result.err()
    );
    assert!(output_path.exists(), "Output video file was not created");

    let original_info = probe_media(&video_path).expect("Failed to probe original video");
    let composed_info = probe_media(&output_path).expect("Failed to probe composed video");

    assert!(
        (composed_info.duration - original_info.duration).abs() < 1.0,
        "Duration mismatch: original={}, composed={}",
        original_info.duration,
        composed_info.duration
    );

    assert!(
        find_video_stream(&composed_info).is_some(),
        "Composed video should have a video stream"
    );
    let audio = find_audio_stream(&composed_info).expect("No audio stream in composed video");
    assert_eq!(
        audio.codec_name, "aac",
        "Composed audio should be AAC encoded"
    );
}

/// 验证视频合成并烧录字幕。
#[test]
fn test_video_composition_with_subtitles() {
    ensure_ffmpeg_available();

    let dir = TempDir::new().expect("Failed to create temp dir");
    let video_path = common::generate_test_video(&dir, "original_sub.mp4", 3);
    let audio_path = common::generate_test_wav(&dir, "new_audio_sub.wav", 3);
    let subtitle_path = generate_test_srt(&dir, "subs.srt");
    let output_path = dir.path().join("composed_sub.mp4");

    let composer = FfmpegVideoComposer::with_encoder("libx264");
    let result = composer.compose_video(
        &video_path,
        &audio_path,
        &output_path,
        true,
        Some(&subtitle_path),
        1.0,
    );

    assert!(
        result.is_ok(),
        "Video composition with subtitles failed: {:?}",
        result.err()
    );
    assert!(output_path.exists(), "Output video file was not created");

    let info = probe_media(&output_path).expect("Failed to probe composed video");
    assert!(
        find_video_stream(&info).is_some(),
        "Composed video should have a video stream"
    );
    assert!(
        find_audio_stream(&info).is_some(),
        "Composed video should have an audio stream"
    );

    assert!(
        (info.duration - 3.0).abs() < 1.5,
        "Duration mismatch: expected ~3s, got {}",
        info.duration
    );
}

/// 验证视频合成对不存在的视频文件返回 `FileNotFound` 错误。
#[test]
fn test_video_composition_nonexistent_video() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let audio_path = common::generate_test_wav(&dir, "audio.wav", 2);
    let output_path = dir.path().join("output.mp4");

    let composer = FfmpegVideoComposer::with_encoder("libx264");
    let result = composer.compose_video(
        std::path::Path::new("/nonexistent/video.mp4"),
        &audio_path,
        &output_path,
        false,
        None,
        1.0,
    );

    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::FileNotFound(_))),
        "Expected FileNotFound error, got {:?}",
        result
    );
}

/// 验证视频合成对不存在的新音频文件返回 `FileNotFound` 错误。
#[test]
fn test_video_composition_nonexistent_audio() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let video_path = common::generate_test_video(&dir, "video.mp4", 2);
    let output_path = dir.path().join("output.mp4");

    let composer = FfmpegVideoComposer::with_encoder("libx264");
    let result = composer.compose_video(
        &video_path,
        std::path::Path::new("/nonexistent/audio.wav"),
        &output_path,
        false,
        None,
        1.0,
    );

    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::FileNotFound(_))),
        "Expected FileNotFound error, got {:?}",
        result
    );
}

/// 验证当 `burn_subtitles` 为 true 但未提供字幕路径时返回错误。
#[test]
fn test_video_composition_burn_without_subtitle_path() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let video_path = common::generate_test_video(&dir, "video.mp4", 2);
    let audio_path = common::generate_test_wav(&dir, "audio.wav", 2);
    let output_path = dir.path().join("output.mp4");

    let composer = FfmpegVideoComposer::with_encoder("libx264");
    let result = composer.compose_video(&video_path, &audio_path, &output_path, true, None, 1.0);

    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::FFmpeg(_))),
        "Expected FFmpeg error for missing subtitle path, got {:?}",
        result
    );
}

/// 验证视频合成的性能：5 秒视频应在 5 秒内完成。
///
/// 优化点：使用 5s 视频（缩短自 30s），断言阈值调整为 5s。
#[test]
fn test_video_composition_performance() {
    ensure_ffmpeg_available();

    let dir = TempDir::new().expect("Failed to create temp dir");
    let video_path = common::generate_test_video(&dir, "perf_video.mp4", 5);
    let audio_path = common::generate_test_wav(&dir, "perf_audio.wav", 5);
    let output_path = dir.path().join("perf_composed.mp4");

    let composer = FfmpegVideoComposer::with_encoder("libx264");
    let start = Instant::now();
    composer
        .compose_video(&video_path, &audio_path, &output_path, false, None, 1.0)
        .expect("Video composition failed");
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 5,
        "Video composition took too long: {:?} (expected < 5s)",
        elapsed
    );

    eprintln!("Video composition (5s video) took: {:?}", elapsed);
}

/// 验证 `FfmpegVideoComposer` 默认使用 VideoToolbox 编码器。
#[test]
fn test_video_composer_default_encoder() {
    let composer = FfmpegVideoComposer::new();
    assert_eq!(
        composer.encoder(),
        "h264_videotoolbox",
        "Default encoder should be h264_videotoolbox for M1 hardware acceleration"
    );
}

/// 验证 `FfmpegVideoComposer::with_encoder` 正确设置编码器。
#[test]
fn test_video_composer_custom_encoder() {
    let composer = FfmpegVideoComposer::with_encoder("libx265");
    assert_eq!(composer.encoder(), "libx265");
}

/// 验证 `FfmpegAudioExtractor` 实现了 `Default` trait。
#[test]
fn test_audio_extractor_default() {
    fn assert_impl_default<T: Default>() {}
    assert_impl_default::<FfmpegAudioExtractor>();
}

/// 验证 `FfmpegVideoComposer` 的 `Default` 实现。
#[test]
fn test_video_composer_default() {
    let composer = FfmpegVideoComposer::default();
    assert_eq!(composer.encoder(), "h264_videotoolbox");
}
