//! 声音克隆性能基准测试
//!
//! 测量 VoiceExtractor 和 SubprocessCloneEngine 的关键性能指标。
//!
//! # 基准项
//! - `extract_reference_audio`: 参考音频提取（含静音修剪+归一化）
//! - `extract_reference_audio_no_enhancement`: 禁用增强的提取（基线对比）
//! - `trim_silence`: 静音修剪算法性能
//! - `normalize_rms`: RMS 归一化算法性能
//! - `subprocess_engine_creation`: SubprocessCloneEngine 创建开销

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::Path;

use vt_core::config::VoiceExtractorConfig;
use vt_core::models::segment::Segment;
use vt_core::voice_extractor::VoiceExtractor;

/// 创建测试用 WAV 文件
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
        let envelope = if t < 0.5 || t > duration_secs - 0.5 {
            0.0
        } else {
            0.3 * (1.0 + (t * 3.0).sin() * 0.2)
        };
        let sample = ((t * 220.0 * 2.0 * std::f64::consts::PI).sin() * envelope * 32767.0) as i16;
        writer.write_sample(sample).expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");
}

/// 创建测试 segments
fn create_test_segments() -> Vec<Segment> {
    vec![
        Segment::new("seg-0".into(), 0.0, 3.0, "Hello world".into()),
        Segment::new(
            "seg-1".into(),
            3.0,
            8.0,
            "Welcome to this video tutorial".into(),
        ),
        Segment::new("seg-2".into(), 8.0, 15.0, "Let's begin".into()),
    ]
}

/// 基准：参考音频提取（含静音修剪+归一化）
fn bench_extract_reference_audio(c: &mut Criterion) {
    let mut group = c.benchmark_group("voice_extractor");

    for duration in [30.0, 60.0, 120.0] {
        group.bench_with_input(
            BenchmarkId::new("extract_enhanced", format!("{}s", duration)),
            &duration,
            |b, &duration| {
                let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
                let full_wav = dir.path().join("full.wav");
                create_test_wav(&full_wav, duration, 16000);
                let segments = create_test_segments();

                let config = VoiceExtractorConfig {
                    enable_enhancement: false, // 禁用 ffmpeg（CI 环境可能没有）
                    enable_silence_trim: true,
                    enable_normalization: true,
                    ..Default::default()
                };
                let extractor = VoiceExtractor::new(config);

                b.iter(|| {
                    let ref_output = dir.path().join("ref.wav");
                    let result = extractor.extract_reference_audio(
                        black_box(&full_wav),
                        black_box(&segments),
                        black_box(&ref_output),
                    );
                    black_box(result);
                    // 清理以便下次迭代
                    let _ = std::fs::remove_file(&ref_output);
                });
            },
        );
    }

    group.finish();
}

/// 基准：禁用所有增强的提取（基线对比）
fn bench_extract_no_enhancement(c: &mut Criterion) {
    let mut group = c.benchmark_group("voice_extractor");

    group.bench_function("extract_baseline", |b| {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let full_wav = dir.path().join("full.wav");
        create_test_wav(&full_wav, 60.0, 16000);
        let segments = create_test_segments();

        let config = VoiceExtractorConfig {
            enable_enhancement: false,
            enable_silence_trim: false,
            enable_normalization: false,
            ..Default::default()
        };
        let extractor = VoiceExtractor::new(config);

        b.iter(|| {
            let ref_output = dir.path().join("ref_baseline.wav");
            let result = extractor.extract_reference_audio(
                black_box(&full_wav),
                black_box(&segments),
                black_box(&ref_output),
            );
            black_box(result);
            let _ = std::fs::remove_file(&ref_output);
        });
    });

    group.finish();
}

/// 基准：SubprocessCloneEngine 创建开销
fn bench_subprocess_engine_creation(c: &mut Criterion) {
    c.bench_function("subprocess_engine_creation", |b| {
        b.iter(|| {
            let engine = vt_core::cloning::SubprocessCloneEngine::new(
                black_box("/path/to/tool".to_string()),
                black_box(Some("/path/to/model".to_string())),
                black_box(vec![
                    "synthesize".to_string(),
                    "--text".to_string(),
                    "{text}".to_string(),
                    "--voice".to_string(),
                    "{ref_audio}".to_string(),
                    "--output".to_string(),
                    "{output}".to_string(),
                ]),
                black_box(120),
            );
            black_box(engine);
        });
    });
}

/// 基准：VoiceExtractorConfig 创建和克隆
fn bench_config_operations(c: &mut Criterion) {
    c.bench_function("config_clone", |b| {
        let config = VoiceExtractorConfig::default();
        b.iter(|| {
            let cloned = black_box(&config).clone();
            black_box(cloned);
        });
    });
}

criterion_group!(
    benches,
    bench_extract_reference_audio,
    bench_extract_no_enhancement,
    bench_subprocess_engine_creation,
    bench_config_operations,
);
criterion_main!(benches);
