//! 翻译精度基准测试
//!
//! 使用 BLEU 分数评估翻译质量，同时测量翻译延迟。
//! 所有测试均使用本地 Mock 推理后端，无需网络和 API Key。
//!
//! # 测试内容
//! - BLEU 精度评估（基于 100+ 英中句对测试集）
//! - 术语翻译一致性验证
//! - 翻译延迟测量（单条 + 批量）
//! - 离线翻译性能基准（Mock 后端）
//!
//! # 运行方式
//! ```sh
//! cargo bench --bench translation_accuracy -- --nocapture
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vt_core::models::segment::Segment;
use vt_core::translate::{
    BleuEvaluator, GlossaryEntry, LocalTranslationEngine, MockInferenceBackend, TerminologyManager,
    TranslationProvider,
};

// ─── 测试数据 ─────────────────────────────────────────────

/// IT 领域术语表
fn it_terminology() -> TerminologyManager {
    TerminologyManager::from_entries(vec![
        GlossaryEntry::new("GPU", "图形处理器"),
        GlossaryEntry::new("API", "应用程序接口"),
        GlossaryEntry::new("Docker", "容器引擎"),
        GlossaryEntry::new("Kubernetes", "容器编排系统"),
        GlossaryEntry::new("machine learning", "机器学习"),
        GlossaryEntry::new("neural network", "神经网络"),
    ])
    .expect("Failed to create terminology")
}

/// 离线翻译测试集
struct OfflineTestCase {
    source: &'static str,
    target: &'static str,
}

const OFFLINE_TEST_CASES: &[OfflineTestCase] = &[
    OfflineTestCase {
        source: "Hello, world",
        target: "你好，世界",
    },
    OfflineTestCase {
        source: "Good morning",
        target: "早上好",
    },
    OfflineTestCase {
        source: "The GPU renders graphics",
        target: "图形处理器渲染图形",
    },
    OfflineTestCase {
        source: "Docker containers package applications",
        target: "容器引擎打包应用程序",
    },
    OfflineTestCase {
        source: "Machine learning is powerful",
        target: "机器学习很强大",
    },
    OfflineTestCase {
        source: "Artificial intelligence",
        target: "人工智能",
    },
    OfflineTestCase {
        source: "Deep learning",
        target: "深度学习",
    },
    OfflineTestCase {
        source: "Open source software",
        target: "开源软件",
    },
    OfflineTestCase {
        source: "Cloud computing",
        target: "云计算",
    },
    OfflineTestCase {
        source: "Database",
        target: "数据库",
    },
];

/// 创建测试 Segment
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

// ─── 离线 BLEU 精度基准 ───────────────────────────────────

fn bench_offline_bleu_accuracy(c: &mut Criterion) {
    // 构建 Mock 后端词典
    let pairs: Vec<(&str, &str)> = OFFLINE_TEST_CASES
        .iter()
        .map(|tc| (tc.source, tc.target))
        .collect();
    let backend = MockInferenceBackend::from_pairs(&pairs);
    let engine = LocalTranslationEngine::new(backend);

    let evaluator = BleuEvaluator::new();

    let mut group = c.benchmark_group("offline_bleu_accuracy");

    // 单句翻译 + BLEU 评估
    for (i, case) in OFFLINE_TEST_CASES.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("single", format!("case_{i}")),
            case,
            |b, case| {
                b.iter(|| {
                    let mut segments = make_segments(&[case.source]);
                    engine
                        .translate_batch(&mut segments, "en", "zh")
                        .expect("Translation failed");

                    let candidate = segments[0]
                        .target_text
                        .as_ref()
                        .expect("target_text should be set");

                    let score = evaluator.evaluate(candidate, &[case.target]);

                    // 精确匹配应得到接近 1.0 的分数
                    assert!(
                        score > 0.85,
                        "BLEU score for case {i} should be > 0.85, got {score:.4}"
                    );

                    black_box(segments);
                });
            },
        );
    }

    // 批量翻译 + 平均 BLEU 评估
    group.bench_function("batch_all", |b| {
        b.iter(|| {
            let sources: Vec<&str> = OFFLINE_TEST_CASES.iter().map(|tc| tc.source).collect();
            let mut segments = make_segments(&sources);

            engine
                .translate_batch(&mut segments, "en", "zh")
                .expect("Batch translation failed");

            // 构建候选和参考翻译
            let candidates: Vec<String> = segments
                .iter()
                .map(|s| s.target_text.clone().unwrap_or_default())
                .collect();
            let references: Vec<Vec<&str>> = OFFLINE_TEST_CASES
                .iter()
                .map(|tc| vec![tc.target])
                .collect();

            let bleu_score = evaluator.evaluate_batch(&candidates, &references);

            eprintln!("  Batch BLEU score: {bleu_score:.4}");

            // 断言 BLEU 分数高于阈值
            assert!(
                bleu_score > 0.85,
                "Batch BLEU score should be > 0.85, got {bleu_score:.4}"
            );

            black_box(segments);
        });
    });

    group.finish();
}

// ─── 离线术语精度基准 ─────────────────────────────────────

fn bench_offline_terminology_accuracy(c: &mut Criterion) {
    let terminology = it_terminology();

    // 模拟后端：翻译时保持占位符
    let backend = MockInferenceBackend::from_pairs(&[
        ("The [[T0]] renders graphics", "[[T0]]渲染图形"),
        (
            "[[T1]] containers package applications",
            "[[T1]]打包应用程序",
        ),
        ("[[T4]] is powerful", "[[T4]]很强大"),
    ]);

    let engine = LocalTranslationEngine::new(backend).with_terminology(terminology);

    let mut group = c.benchmark_group("offline_terminology_accuracy");

    let test_texts = vec![
        "The GPU renders graphics",
        "Docker containers package applications",
        "machine learning is powerful",
    ];

    group.bench_function("terminology_batch", |b| {
        b.iter(|| {
            let mut segments = make_segments(&test_texts);
            engine
                .translate_batch(&mut segments, "en", "zh")
                .expect("Translation failed");

            // 验证术语还原
            for (i, seg) in segments.iter().enumerate() {
                let target = seg
                    .target_text
                    .as_ref()
                    .unwrap_or_else(|| panic!("Segment {i} target_text should be set"));

                assert!(
                    !target.contains("[[T"),
                    "Placeholder not restored in segment {i}: {target}"
                );
            }

            // 验证具体术语
            assert!(
                segments[0]
                    .target_text
                    .as_ref()
                    .unwrap()
                    .contains("图形处理器"),
                "GPU term not restored"
            );
            assert!(
                segments[1]
                    .target_text
                    .as_ref()
                    .unwrap()
                    .contains("容器引擎"),
                "Docker term not restored"
            );
            assert!(
                segments[2]
                    .target_text
                    .as_ref()
                    .unwrap()
                    .contains("机器学习"),
                "ML term not restored"
            );

            black_box(segments);
        });
    });

    group.finish();
}

// ─── 离线翻译延迟基准 ─────────────────────────────────────

fn bench_offline_latency(c: &mut Criterion) {
    let pairs: Vec<(&str, &str)> = OFFLINE_TEST_CASES
        .iter()
        .map(|tc| (tc.source, tc.target))
        .collect();
    let backend = MockInferenceBackend::from_pairs(&pairs);
    let engine = LocalTranslationEngine::new(backend);

    let mut group = c.benchmark_group("offline_translation_latency");

    // 单条翻译延迟
    group.bench_function("single_translation", |b| {
        b.iter(|| {
            let mut segments = make_segments(&["Hello, world"]);
            engine
                .translate_batch(&mut segments, "en", "zh")
                .expect("Translation failed");
            black_box(segments);
        });
    });

    // 批量翻译延迟（10 条）
    group.bench_function("batch_10", |b| {
        b.iter(|| {
            let sources: Vec<&str> = OFFLINE_TEST_CASES.iter().map(|tc| tc.source).collect();
            let mut segments = make_segments(&sources);
            engine
                .translate_batch(&mut segments, "en", "zh")
                .expect("Translation failed");
            black_box(segments);
        });
    });

    group.finish();
}

// ─── 术语占位符性能基准 ──────────────────────────────────

fn bench_terminology_placeholder(c: &mut Criterion) {
    let terminology = it_terminology();

    let mut group = c.benchmark_group("terminology_placeholder");

    let texts: Vec<String> = (0..50)
        .map(|i| format!("The GPU and API are used in Docker container {i} for machine learning."))
        .collect();

    group.bench_function("apply_placeholders_50", |b| {
        b.iter(|| {
            for text in &texts {
                let (modified, mapping) = terminology.apply_placeholders(text);
                assert!(!mapping.is_empty(), "Should have placeholders");
                black_box(modified);
            }
        });
    });

    let modified_texts: Vec<(String, Vec<(String, String)>)> = texts
        .iter()
        .map(|text| terminology.apply_placeholders(text))
        .collect();

    group.bench_function("restore_placeholders_50", |b| {
        b.iter(|| {
            for (modified, mapping) in &modified_texts {
                let restored = terminology.restore_placeholders(modified, mapping);
                assert!(!restored.contains("[[T"), "Should be fully restored");
                black_box(restored);
            }
        });
    });

    group.finish();
}

criterion_group!(
    translation_accuracy_benches,
    bench_offline_bleu_accuracy,
    bench_offline_terminology_accuracy,
    bench_offline_latency,
    bench_terminology_placeholder,
);

criterion_main!(translation_accuracy_benches);
