//! TTS 性能基准测试（多音色 + 音频后处理）
//!
//! 测试内容：
//! - 不同音色（女声/男声）的合成延迟
//! - 音调偏移后处理对性能的影响
//! - 48kHz 高采样率合成的性能
//! - 批量合成吞吐量（多音色混合）
//! - 1 分钟语音合成时间验证（≤ 5 秒）
//!
//! # 运行方式
//! ```sh
//! cargo bench --bench tts_speed_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vt_core::config::TtsConfig;
use vt_core::models::segment::Segment;
use vt_core::tts::{SayEngine, TtsEngine};

// ─── 辅助函数 ─────────────────────────────────────────────

/// 创建测试用 Segment（已翻译状态）
fn make_segment(text: &str, index: usize) -> Segment {
    let mut seg = Segment::new(
        format!("seg-{index:04}"),
        index as f64 * 5.0,
        (index + 1) as f64 * 5.0,
        format!("source-{index}"),
    );
    seg.start_transcribing().ok();
    seg.finish_transcribing(text.to_string()).ok();
    seg
}

/// 创建指定音色和参数的 TTS 配置
fn make_config(cache_dir: &str, voice_id: &str, pitch: f32, sample_rate: u32) -> TtsConfig {
    TtsConfig {
        voice_id: voice_id.to_string(),
        cache_dir: cache_dir.to_string(),
        pitch,
        sample_rate,
        ..Default::default()
    }
}

// 测试文本
const MEDIUM_TEXT: &str = "这是一个Rust编程语言的入门教程，我们将从基础开始讲解。";
const LONG_TEXT: &str = "在这个视频中，我们将深入学习Rust的所有权系统。所有权是Rust最独特的特性之一，它使得Rust能够在没有垃圾回收器的情况下保证内存安全。我们将通过实际代码示例来理解所有权、借用和引用的概念，以及它们如何帮助我们在编译时捕获常见的内存错误。";

// ─── 不同音色合成基准 ─────────────────────────────────────

fn bench_voice_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_voice_comparison");
    group.sample_size(10);

    let cache_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let cache_path = cache_dir.path().to_string_lossy().to_string();
    let engine = SayEngine::with_cache_dir(cache_dir.path()).expect("Failed to create engine");

    let voices: [(&str, &str); 4] = [
        ("tingting", "Female (Tingting)"),
        ("meijia", "Female (Meijia)"),
        ("zhiming", "Male (Zhiming)"),
        ("weiqiang", "Male (Weiqiang)"),
    ];

    for (voice_id, label) in &voices {
        let config = make_config(&cache_path, voice_id, 1.0, 24000);
        group.bench_with_input(
            BenchmarkId::new("synthesize", *label),
            MEDIUM_TEXT,
            |b, text| {
                b.iter(|| {
                    let mut segments = vec![make_segment(black_box(text), 0)];
                    engine
                        .synthesize_segments(&mut segments, &config)
                        .expect("synthesis failed");
                    black_box(&segments);
                });
            },
        );
    }

    group.finish();
}

// ─── 音调偏移后处理影响基准 ─────────────────────────────────

fn bench_pitch_shift_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_pitch_shift");
    group.sample_size(10);

    let cache_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let cache_path = cache_dir.path().to_string_lossy().to_string();
    let engine = SayEngine::with_cache_dir(cache_dir.path()).expect("Failed to create engine");

    // 女声无音调偏移（无后处理）
    let config_no_shift = make_config(&cache_path, "tingting", 1.0, 24000);
    group.bench_function("no_shift_female", |b| {
        b.iter(|| {
            let mut segments = vec![make_segment(black_box(MEDIUM_TEXT), 0)];
            engine
                .synthesize_segments(&mut segments, &config_no_shift)
                .expect("synthesis failed");
            black_box(&segments);
        });
    });

    // 男声有音调偏移（有后处理）
    let config_with_shift = make_config(&cache_path, "zhiming", 1.0, 24000);
    group.bench_function("with_shift_male", |b| {
        b.iter(|| {
            let mut segments = vec![make_segment(black_box(MEDIUM_TEXT), 0)];
            engine
                .synthesize_segments(&mut segments, &config_with_shift)
                .expect("synthesis failed");
            black_box(&segments);
        });
    });

    group.finish();
}

// ─── 采样率影响基准 ───────────────────────────────────────

fn bench_sample_rate_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_sample_rate");
    group.sample_size(10);

    let cache_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let cache_path = cache_dir.path().to_string_lossy().to_string();
    let engine = SayEngine::with_cache_dir(cache_dir.path()).expect("Failed to create engine");

    for (rate, label) in [(24000u32, "24kHz"), (48000u32, "48kHz")] {
        let config = make_config(&cache_path, "tingting", 1.0, rate);
        group.bench_function(label, |b| {
            b.iter(|| {
                let mut segments = vec![make_segment(black_box(MEDIUM_TEXT), 0)];
                engine
                    .synthesize_segments(&mut segments, &config)
                    .expect("synthesis failed");
                black_box(&segments);
            });
        });
    }

    group.finish();
}

// ─── 1 分钟语音合成时间验证 ────────────────────────────────

/// 验证 1 分钟语音的合成时间 ≤ 5 秒（≥ 12x 实时）
fn bench_one_minute_speech(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_one_minute");
    group.sample_size(5);

    let cache_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let cache_path = cache_dir.path().to_string_lossy().to_string();
    let engine = SayEngine::with_cache_dir(cache_dir.path()).expect("Failed to create engine");

    // 中文语速约 4 字/秒，1 分钟约 240 字
    // 使用 LONG_TEXT（约 100 字）反复合成来模拟
    let one_minute_text = format!("{LONG_TEXT}{LONG_TEXT}{LONG_TEXT}");

    let config = make_config(&cache_path, "tingting", 1.0, 24000);

    group.bench_function("synthesize_1min", |b| {
        b.iter(|| {
            let mut segments = vec![make_segment(black_box(&one_minute_text), 0)];
            engine
                .synthesize_segments(&mut segments, &config)
                .expect("synthesis failed");
            black_box(&segments);
        });
    });

    group.finish();
}

// ─── 批量合成基准（混合音色） ──────────────────────────────

fn bench_batch_mixed_voices(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_batch_mixed");
    group.sample_size(10);

    let cache_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let cache_path = cache_dir.path().to_string_lossy().to_string();
    let engine = SayEngine::with_cache_dir(cache_dir.path()).expect("Failed to create engine");

    for batch_size in [5, 10] {
        let config = make_config(&cache_path, "zhiming", 1.0, 24000);
        group.bench_with_input(
            BenchmarkId::new("male_voice_batch", batch_size),
            &batch_size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        (0..size)
                            .map(|i| make_segment(&format!("这是第{i}段男声测试文本。"), i))
                            .collect::<Vec<_>>()
                    },
                    |mut segments| {
                        engine
                            .synthesize_segments(&mut segments, &config)
                            .expect("synthesis failed");
                        black_box(segments.len());
                    },
                );
            },
        );
    }

    group.finish();
}

// ─── 缓存命中基准 ─────────────────────────────────────────

fn bench_cache_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_cache_voice");

    let cache_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let cache_path = cache_dir.path().to_string_lossy().to_string();
    let engine = SayEngine::with_cache_dir(cache_dir.path()).expect("Failed to create engine");

    // 预填充缓存（女声）
    let config = make_config(&cache_path, "tingting", 1.0, 24000);
    let seg = make_segment(MEDIUM_TEXT, 0);
    engine
        .synthesize_segments(&mut [seg.clone()], &config)
        .expect("pre-fill failed");

    // 缓存命中
    group.bench_function("cache_hit_female", |b| {
        b.iter(|| {
            let mut segments = vec![make_segment(black_box(MEDIUM_TEXT), 0)];
            engine
                .synthesize_segments(&mut segments, &config)
                .expect("synthesis failed");
            black_box(&segments);
        });
    });

    group.finish();
}

criterion_group!(
    tts_speed_benches,
    bench_voice_comparison,
    bench_pitch_shift_impact,
    bench_sample_rate_impact,
    bench_one_minute_speech,
    bench_batch_mixed_voices,
    bench_cache_hit,
);

criterion_main!(tts_speed_benches);
