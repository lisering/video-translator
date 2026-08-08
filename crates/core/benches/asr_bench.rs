//! ASR 模块性能基准测试
//!
//! 测试内容:
//! - VAD 语音活动检测（不同时长音频）
//! - WAV 文件读取
//! - 音频采样数据生成
//! - VAD 配置参数影响
//!
//! # 运行方式
//! ```sh
//! cargo bench --bench asr_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::Path;
use vt_core::asr::{detect_speech_segments, read_wav_mono, VadConfig};

// ─── 辅助函数 ─────────────────────────────────────────────

/// 生成模拟语音音频（包含语音段和静音段）
fn generate_speech_like_audio(duration_secs: f64, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f64) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // 模拟 1s 语音 + 0.5s 静音 的周期
            let cycle_pos = t % 1.5;
            if cycle_pos < 1.0 {
                // 语音段：440Hz + 880Hz 混合
                let wave = (t * 440.0 * 2.0 * std::f64::consts::PI).sin() * 0.3
                    + (t * 880.0 * 2.0 * std::f64::consts::PI).sin() * 0.15;
                wave as f32
            } else {
                // 静音段
                0.0
            }
        })
        .collect()
}

/// 创建测试 WAV 文件
fn create_test_wav(path: &Path, duration_secs: f64, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV");
    let samples = generate_speech_like_audio(duration_secs, sample_rate);

    for s in &samples {
        let i16_sample = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer
            .write_sample(i16_sample)
            .expect("Failed to write sample");
    }

    writer.finalize().expect("Failed to finalize WAV");
}

// ─── VAD 检测基准 ─────────────────────────────────────────

fn bench_vad_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_detection");
    group.sample_size(20);

    for duration in [10.0, 30.0, 60.0, 120.0] {
        let samples = generate_speech_like_audio(duration, 16000);
        let config = VadConfig::default();

        group.bench_with_input(
            BenchmarkId::new("default_config", format!("{duration}s")),
            &samples,
            |b, samples| {
                b.iter(|| {
                    let segments = detect_speech_segments(black_box(samples), 16000, &config);
                    black_box(segments.len());
                });
            },
        );
    }

    group.finish();
}

// ─── VAD 配置灵敏度对比 ───────────────────────────────────

fn bench_vad_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_sensitivity");
    group.sample_size(20);

    let samples = generate_speech_like_audio(60.0, 16000);

    // 低灵敏度（大段）
    let low_sensitivity = VadConfig {
        energy_threshold: 0.05,
        min_speech_duration_ms: 800,
        min_silence_duration_ms: 1000,
        speech_pad_ms: 300,
        ..Default::default()
    };

    // 默认灵敏度
    let default_config = VadConfig::default();

    // 高灵敏度（细粒度）
    let high_sensitivity = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 100,
        speech_pad_ms: 30,
        ..Default::default()
    };

    let configs: [(&str, VadConfig); 3] = [
        ("low_sensitivity", low_sensitivity),
        ("default", default_config),
        ("high_sensitivity", high_sensitivity),
    ];

    for (name, config) in &configs {
        group.bench_with_input(BenchmarkId::new("60s_audio", *name), config, |b, config| {
            b.iter(|| {
                let segments = detect_speech_segments(black_box(&samples), 16000, config);
                black_box(segments.len());
            });
        });
    }

    group.finish();
}

// ─── WAV 读取基准 ─────────────────────────────────────────

fn bench_wav_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("asr_wav_read");

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    for duration in [1.0, 10.0, 60.0] {
        let wav_path = dir.path().join(format!("test_{duration}.wav"));
        create_test_wav(&wav_path, duration, 16000);

        group.bench_with_input(
            BenchmarkId::new("read_wav_mono", format!("{duration}s")),
            &wav_path,
            |b, path| {
                b.iter(|| {
                    let (samples, sample_rate) = read_wav_mono(black_box(path)).unwrap();
                    black_box((samples.len(), sample_rate));
                });
            },
        );
    }

    group.finish();
}

// ─── 音频采样生成基准 ─────────────────────────────────────

fn bench_sample_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_sample_generation");

    for duration in [1.0, 10.0, 60.0] {
        group.bench_with_input(
            BenchmarkId::new("speech_like", format!("{duration}s")),
            &duration,
            |b, &duration| {
                b.iter(|| {
                    let samples = generate_speech_like_audio(black_box(duration), 16000);
                    black_box(samples.len());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    asr_benches,
    bench_vad_detection,
    bench_vad_sensitivity,
    bench_wav_read,
    bench_sample_generation,
);

criterion_main!(asr_benches);
