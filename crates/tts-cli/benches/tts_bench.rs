//! TTS 引擎性能基准测试
//!
//! 测试核心组件的吞吐量和延迟:
//! - KVCache 更新/回滚
//! - 采样 (top-k + repetition penalty)
//! - Attention forward
//! - QLinear forward
//!
//! 运行: cargo bench -p vt-tts --features metal

#![cfg(feature = "metal")]

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use vt_tts::transformer::KVCache;

fn bench_kvcache_update(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("kvcache");

    // 单步追加
    group.bench_function("update_single", |b| {
        b.iter_batched(
            || {
                let mut cache = KVCache::new();
                let k = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
                let v = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
                cache.update(&k, &v).unwrap();
                cache
            },
            |mut cache| {
                let k = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
                let v = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
                cache.update(&k, &v).unwrap();
                black_box(cache);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // 批量追加 (10 步)
    group.bench_function("update_batch_10", |b| {
        b.iter(|| {
            let mut cache = KVCache::new();
            for _ in 0..10 {
                let k = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
                let v = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
                cache.update(&k, &v).unwrap();
            }
            black_box(cache);
        });
    });

    // 回滚
    group.bench_function("rollback", |b| {
        b.iter_batched(
            || {
                let mut cache = KVCache::new();
                for _ in 0..20 {
                    let k = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
                    let v = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
                    cache.update(&k, &v).unwrap();
                }
                cache
            },
            |mut cache| {
                cache.rollback(1).unwrap();
                black_box(cache);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_tensor_ops(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("tensor_ops");

    // Matmul F32 [1, 64] x [64, 128]
    group.bench_function("matmul_f32_64x128", |b| {
        let a = Tensor::randn(0.0f32, 1.0, (1, 64), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        b.iter(|| {
            let c = black_box(&a).matmul(black_box(&b)).unwrap();
            black_box(c);
        });
    });

    // Softmax [1, 100]
    group.bench_function("softmax_100", |b| {
        let x = Tensor::randn(0.0f32, 1.0, (1, 100), &device).unwrap();
        b.iter(|| {
            let s = candle_nn::ops::softmax(black_box(&x), candle_core::D::Minus1).unwrap();
            black_box(s);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_kvcache_update, bench_tensor_ops);
criterion_main!(benches);
