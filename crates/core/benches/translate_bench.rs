//! 翻译模块性能基准测试
//!
//! 测试内容:
//! - BLEU 评估性能（不同文本长度和批量大小）
//! - 术语管理器性能（占位符应用/还原）
//! - Mock 翻译引擎延迟
//! - 术语增强翻译性能
//!
//! # 运行方式
//! ```sh
//! cargo bench --bench translate_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vt_core::models::segment::Segment;
use vt_core::translate::{
    BleuEvaluator, GlossaryEntry, LocalTranslationEngine, MockInferenceBackend, TerminologyManager,
    TranslationProvider,
};

// ─── 辅助函数 ─────────────────────────────────────────────

fn make_segments(texts: &[&str]) -> Vec<Segment> {
    texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            Segment::new(
                format!("seg-{i:04}"),
                i as f64,
                (i + 1) as f64,
                text.to_string(),
            )
        })
        .collect()
}

fn it_terminology() -> TerminologyManager {
    TerminologyManager::from_entries(vec![
        GlossaryEntry::new("GPU", "图形处理器"),
        GlossaryEntry::new("API", "应用程序接口"),
        GlossaryEntry::new("Docker", "容器引擎"),
        GlossaryEntry::new("Kubernetes", "容器编排系统"),
        GlossaryEntry::new("machine learning", "机器学习"),
        GlossaryEntry::new("neural network", "神经网络"),
        GlossaryEntry::new("container", "容器"),
        GlossaryEntry::new("microservice", "微服务"),
        GlossaryEntry::new("CI/CD", "持续集成/持续部署"),
        GlossaryEntry::new("RESTful", "REST风格"),
    ])
    .expect("Failed to create terminology")
}

// ─── BLEU 评估基准 ────────────────────────────────────────

fn bench_bleu_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bleu_evaluation");

    let evaluator = BleuEvaluator::new();

    // 单句 BLEU 评估
    let test_cases: [(&str, &str); 3] = [
        ("short", "你好"),
        ("medium", "这是一个Rust编程语言的入门教程"),
        (
            "long",
            "在这个视频中我们将深入学习Rust的所有权系统这是Rust最独特的特性之一",
        ),
    ];
    let references: [&str; 3] = [
        "你好",
        "这是一个Rust编程语言的入门教程",
        "在这个视频中我们将深入学习Rust的所有权系统这是Rust最独特的特性之一",
    ];

    for (i, (name, candidate)) in test_cases.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("single", *name),
            candidate,
            |b, candidate| {
                b.iter(|| {
                    let score = evaluator.evaluate(black_box(candidate), &[references[i]]);
                    black_box(score);
                });
            },
        );
    }

    // 批量 BLEU 评估
    for batch_size in [10, 50, 100] {
        let candidates: Vec<String> = (0..batch_size)
            .map(|i| format!("测试翻译结果第{i}条"))
            .collect();
        let ref_strings: Vec<String> = (0..batch_size)
            .map(|i| format!("测试翻译结果第{i}条"))
            .collect();
        let references: Vec<Vec<&str>> = ref_strings.iter().map(|s| vec![s.as_str()]).collect();

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &(candidates, references),
            |b, (candidates, references)| {
                b.iter(|| {
                    let score =
                        evaluator.evaluate_batch(black_box(candidates), black_box(references));
                    black_box(score);
                });
            },
        );
    }

    group.finish();
}

// ─── 术语管理器基准 ───────────────────────────────────────

fn bench_terminology(c: &mut Criterion) {
    let mut group = c.benchmark_group("terminology_manager");

    let terminology = it_terminology();

    // 占位符应用
    let texts_with_terms: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "The GPU and API are used in Docker container {i} for machine learning with Kubernetes."
            )
        })
        .collect();

    group.bench_function("apply_placeholders_100", |b| {
        b.iter(|| {
            let mut count = 0;
            for text in &texts_with_terms {
                let (modified, mapping) = terminology.apply_placeholders(black_box(text));
                count += mapping.len();
                black_box(modified);
            }
            black_box(count);
        });
    });

    // 占位符还原
    let modified_texts: Vec<(String, Vec<(String, String)>)> = texts_with_terms
        .iter()
        .map(|text| terminology.apply_placeholders(text))
        .collect();

    group.bench_function("restore_placeholders_100", |b| {
        b.iter(|| {
            let mut count = 0;
            for (modified, mapping) in &modified_texts {
                let restored =
                    terminology.restore_placeholders(black_box(modified), black_box(mapping));
                count += restored.len();
            }
            black_box(count);
        });
    });

    // 单条术语占位符应用
    group.bench_function("apply_single", |b| {
        b.iter(|| {
            let (modified, mapping) =
                terminology.apply_placeholders(black_box("The GPU renders graphics"));
            black_box((modified, mapping.len()));
        });
    });

    group.finish();
}

// ─── Mock 翻译引擎延迟基准 ────────────────────────────────

fn bench_mock_translation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mock_translation");

    let pairs: Vec<(&str, &str)> = vec![
        ("Hello, world", "你好，世界"),
        ("Good morning", "早上好"),
        ("The GPU renders graphics", "图形处理器渲染图形"),
        (
            "Docker containers package applications",
            "容器引擎打包应用程序",
        ),
        ("Machine learning is powerful", "机器学习很强大"),
        ("Artificial intelligence", "人工智能"),
        ("Deep learning", "深度学习"),
        ("Open source software", "开源软件"),
        ("Cloud computing", "云计算"),
        ("Database", "数据库"),
    ];

    let backend = MockInferenceBackend::from_pairs(&pairs);
    let engine = LocalTranslationEngine::new(backend);

    // 单条翻译
    group.bench_function("single_translation", |b| {
        b.iter(|| {
            let mut segments = make_segments(&["Hello, world"]);
            engine
                .translate_batch(black_box(&mut segments), "en", "zh")
                .unwrap();
            black_box(segments);
        });
    });

    // 批量翻译（不同大小）
    for batch_size in [5, 10, 50] {
        let sources: Vec<&str> = pairs.iter().take(batch_size).map(|(s, _)| *s).collect();

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &sources,
            |b, sources| {
                b.iter(|| {
                    let mut segments = make_segments(black_box(sources));
                    engine
                        .translate_batch(black_box(&mut segments), "en", "zh")
                        .unwrap();
                    black_box(segments);
                });
            },
        );
    }

    group.finish();
}

// ─── 术语增强翻译基准 ─────────────────────────────────────

fn bench_translation_with_terminology(c: &mut Criterion) {
    let mut group = c.benchmark_group("translation_with_terminology");

    let terminology = it_terminology();

    let backend = MockInferenceBackend::from_pairs(&[
        ("The [[T0]] renders graphics", "[[T0]]渲染图形"),
        (
            "[[T1]] containers package applications",
            "[[T1]]打包应用程序",
        ),
        ("[[T4]] is powerful", "[[T4]]很强大"),
    ]);

    let engine = LocalTranslationEngine::new(backend).with_terminology(terminology);

    let test_texts: Vec<&str> = vec![
        "The GPU renders graphics",
        "Docker containers package applications",
        "machine learning is powerful",
    ];

    group.bench_function("batch_with_terminology", |b| {
        b.iter(|| {
            let mut segments = make_segments(black_box(&test_texts));
            engine
                .translate_batch(black_box(&mut segments), "en", "zh")
                .unwrap();
            black_box(segments);
        });
    });

    group.finish();
}

criterion_group!(
    translate_benches,
    bench_bleu_evaluation,
    bench_terminology,
    bench_mock_translation,
    bench_translation_with_terminology,
);

criterion_main!(translate_benches);
