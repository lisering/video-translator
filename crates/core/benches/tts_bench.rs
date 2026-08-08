//! TTS 模块性能基准测试
//!
//! 测试内容:
//! - SayEngine 合成延迟（macOS `say` 命令）
//! - TTS 缓存命中/未命中性能
//! - 批量合成吞吐量
//! - 不同文本长度的影响
//!
//! # 运行方式
//! ```sh
//! cargo bench --bench tts_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vt_core::config::TtsConfig;
use vt_core::models::segment::Segment;
use vt_core::tts::{SayEngine, TtsEngine};

// ─── 辅助函数 ─────────────────────────────────────────────

/// 创建测试用 Segment（需设置 target_text 供 TTS 使用）
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

/// 测试文本集（不同长度）
const SHORT_TEXT: &str = "你好";
const MEDIUM_TEXT: &str = "这是一个Rust编程语言的入门教程，我们将从基础开始讲解。";
const LONG_TEXT: &str = "在这个视频中，我们将深入学习Rust的所有权系统。所有权是Rust最独特的特性之一，它使得Rust能够在没有垃圾回收器的情况下保证内存安全。我们将通过实际代码示例来理解所有权、借用和引用的概念，以及它们如何帮助我们在编译时捕获常见的内存错误。";

// ─── SayEngine 单条合成基准 ───────────────────────────────

fn bench_say_synthesize_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_say_single");
    group.sample_size(10);

    let config = TtsConfig::default();
    let engine = SayEngine::new(&config).expect("Failed to create SayEngine");

    let test_cases: [(&str, &str); 3] = [
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ];

    for (name, text) in &test_cases {
        group.bench_with_input(BenchmarkId::new("synthesize", *name), text, |b, text| {
            b.iter(|| {
                let mut segments = vec![make_segment(black_box(text), 0)];
                engine.synthesize_segments(&mut segments, &config).unwrap();
                black_box(&segments);
            });
        });
    }

    group.finish();
}

// ─── TTS 缓存命中基准 ─────────────────────────────────────

fn bench_tts_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_cache");

    let config = TtsConfig::default();
    let engine = SayEngine::new(&config).expect("Failed to create SayEngine");

    // 预填充缓存
    let seg = make_segment(MEDIUM_TEXT, 0);
    engine
        .synthesize_segments(&mut [seg.clone()], &config)
        .unwrap();

    // 缓存命中：相同文本再次合成
    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let mut segments = vec![make_segment(black_box(MEDIUM_TEXT), 0)];
            engine.synthesize_segments(&mut segments, &config).unwrap();
            black_box(&segments);
        });
    });

    // 缓存未命中：每次使用不同文本
    let counter = std::cell::Cell::new(0usize);
    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            let i = counter.get();
            counter.set(i + 1);
            let text = format!("测试文本编号{i}");
            let mut segments = vec![make_segment(&text, i)];
            engine.synthesize_segments(&mut segments, &config).unwrap();
            black_box(&segments);
        });
    });

    group.finish();
}

// ─── 批量合成基准 ─────────────────────────────────────────

fn bench_tts_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("tts_batch");
    group.sample_size(10);

    let config = TtsConfig::default();
    let engine = SayEngine::new(&config).expect("Failed to create SayEngine");

    for batch_size in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("synthesize_batch", batch_size),
            &batch_size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        (0..size)
                            .map(|i| make_segment(&format!("这是第{i}段测试文本。"), i))
                            .collect::<Vec<_>>()
                    },
                    |mut segments| {
                        engine.synthesize_segments(&mut segments, &config).unwrap();
                        black_box(segments.len());
                    },
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    tts_benches,
    bench_say_synthesize_single,
    bench_tts_cache,
    bench_tts_batch,
);

criterion_main!(tts_benches);
