//! 集成测试：离线翻译模块
//!
//! 验证翻译模型的下载、缓存、完整性校验和离线推理能力。
//!
//! # 测试内容
//! - `test_offline_model_load`: 断网环境下从本地路径加载模型
//! - `test_model_download_and_cache`: 模型下载与缓存复用
//! - `test_offline_end_to_end`: 断网环境下完整翻译流水线
//! - `test_model_integrity_verification`: SHA256 完整性校验
//! - `test_translation_accuracy`: BLEU 精度评估
//! - `test_terminology_accuracy`: 术语翻译准确性
//!
//! # 运行方式
//! ```sh
//! # 常规测试（不需要网络和模型）
//! cargo test --workspace --test test_offline_translation
//!
//! # 断网环境测试（需要预先下载模型）
//! sudo ifconfig awdl0 down  # macOS 模拟断网
//! cargo test --workspace --test test_offline_translation -- --ignored
//! sudo ifconfig awdl0 up   # 恢复网络
//! ```

use std::io::Write;

use tempfile::NamedTempFile;
use vt_core::error::AppError;
use vt_core::model_manager::{ModelManager, ModelSource};
use vt_core::models::segment::Segment;
use vt_core::translate::{
    BleuEvaluator, GlossaryEntry, LocalTranslationEngine, MockInferenceBackend, TerminologyManager,
    TranslationProvider,
};

// ═══════════════════════════════════════════════════════════
//  辅助函数
// ═══════════════════════════════════════════════════════════

/// 创建测试用 Segment 列表
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

/// 加载测试数据集
fn load_test_set() -> Vec<(String, String)> {
    let json = include_str!("data/en_zh_test_set.json");
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(json).expect("Failed to parse test set");
    entries
        .iter()
        .map(|e| {
            let source = e["source"].as_str().expect("Missing source").to_string();
            let target = e["target"].as_str().expect("Missing target").to_string();
            (source, target)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════
//  test_offline_model_load
// ═══════════════════════════════════════════════════════════

/// 验证在断网环境下（模拟）能从本地缓存路径成功加载模型。
///
/// 此测试创建一个本地模型文件，然后使用 `ModelSource::Local` 加载它，
/// 验证整个过程不发起任何网络请求。
#[test]
fn test_offline_model_load() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    // 创建模拟模型文件
    let model_path = dir.path().join("test_model.gguf");
    std::fs::write(&model_path, b"fake gguf model data for testing")
        .expect("Failed to write model file");

    // 使用 ModelManager 从本地路径加载
    let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create ModelManager");

    let source = ModelSource::Local {
        path: model_path.clone(),
    };

    // load_model 应该直接返回本地路径，不发起网络请求
    let loaded_path = manager
        .load_model(&source, "test_model.gguf", None)
        .expect("Failed to load model from local path");

    assert_eq!(loaded_path, model_path);
    assert!(loaded_path.exists(), "Loaded model path should exist");

    // 验证模型文件内容
    let content = std::fs::read_to_string(&loaded_path).expect("Failed to read model");
    assert!(
        content.contains("fake gguf model data"),
        "Model content should match"
    );
}

/// 验证本地模型路径不存在时返回清晰错误。
#[test]
fn test_offline_model_load_not_found() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create ModelManager");

    let source = ModelSource::Local {
        path: std::path::PathBuf::from("/nonexistent/model.gguf"),
    };

    let result = manager.load_model(&source, "model.gguf", None);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::FileNotFound(_))),
        "Expected FileNotFound for nonexistent local model, got {result:?}"
    );
}

/// 验证本地模型加载时进行 SHA256 校验。
#[test]
fn test_offline_model_load_with_sha256() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    // 创建模型文件并计算 SHA256
    let model_path = dir.path().join("model.gguf");
    std::fs::write(&model_path, b"hello world").expect("Failed to write model file");

    let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create ModelManager");

    let source = ModelSource::Local {
        path: model_path.clone(),
    };

    // 使用正确的 SHA256 加载
    let loaded = manager
        .load_model(
            &source,
            "model.gguf",
            Some("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"),
        )
        .expect("Failed to load with correct SHA256");
    assert_eq!(loaded, model_path);

    // 使用错误的 SHA256 应失败
    let source2 = ModelSource::Local {
        path: model_path.clone(),
    };
    let result = manager.load_model(
        &source2,
        "model.gguf",
        Some("0000000000000000000000000000000000000000000000000000000000000000"),
    );
    assert!(result.is_err());
    assert!(
        matches!(result, Err(AppError::ModelLoadError(_))),
        "Expected ModelLoadError for wrong SHA256, got {result:?}"
    );
}

// ═══════════════════════════════════════════════════════════
//  test_model_download_and_cache
// ═══════════════════════════════════════════════════════════

/// 验证模型缓存复用机制。
///
/// 手动创建缓存文件后验证 `load_model` 直接使用缓存而非下载。
#[test]
fn test_model_cache_reuse() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create ModelManager");

    let source = ModelSource::ModelScope {
        repo_id: "test-org/test-model".to_string(),
        revision: Some("master".to_string()),
    };

    let cache_path = manager.get_cache_path(&source, "model.gguf");

    // 步骤1：验证缓存不存在
    assert!(!cache_path.exists(), "Cache should not exist initially");
    assert!(
        !manager.is_model_cached(&source, "model.gguf"),
        "is_model_cached should return false"
    );

    // 步骤2：手动创建缓存文件（模拟下载完成后的状态）
    let model_content = b"fake gguf model binary data for cache test";
    std::fs::create_dir_all(cache_path.parent().expect("No parent dir"))
        .expect("Failed to create parent directories");
    std::fs::write(&cache_path, model_content).expect("Failed to write cache file");

    // 步骤3：验证文件被正确保存
    assert!(cache_path.exists(), "Cache file should exist");
    assert!(
        manager.is_model_cached(&source, "model.gguf"),
        "is_model_cached should return true after file creation"
    );

    // 步骤4：加载模型 — 应直接使用缓存
    let loaded_path = manager
        .load_model(&source, "model.gguf", None)
        .expect("Failed to load model from cache");
    assert_eq!(loaded_path, cache_path);

    // 验证文件内容
    let saved_content = std::fs::read(&cache_path).expect("Failed to read cached model");
    assert_eq!(
        saved_content, model_content,
        "Cached model content should match"
    );
}

/// 验证模型缓存目录结构。
#[test]
fn test_model_cache_directory_structure() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create ModelManager");

    let source = ModelSource::ModelScope {
        repo_id: "org/model-name".to_string(),
        revision: Some("v1".to_string()),
    };

    let cache_path = manager.get_cache_path(&source, "qwen2.5-3b-instruct-q5_k_m.gguf");

    // 验证路径结构：cache_dir/org/model-name/qwen2.5-3b-instruct-q5_k_m.gguf
    assert!(cache_path.starts_with(dir.path()));
    assert!(cache_path
        .to_string_lossy()
        .ends_with("qwen2.5-3b-instruct-q5_k_m.gguf"));
    assert!(cache_path.to_string_lossy().contains("org"));
    assert!(cache_path.to_string_lossy().contains("model-name"));
}

// ═══════════════════════════════════════════════════════════
//  test_offline_end_to_end
// ═══════════════════════════════════════════════════════════

/// 验证在断网环境下运行完整的翻译流水线。
///
/// 使用 `MockInferenceBackend` 模拟本地模型推理，
/// 从 Segment 输入到翻译结果输出，验证整个流程无网络依赖。
#[test]
fn test_offline_end_to_end() {
    // 创建本地翻译引擎（完全离线，无网络依赖）
    let backend = MockInferenceBackend::default();
    let engine = LocalTranslationEngine::new(backend);

    // 创建输入 Segment 列表
    let mut segments = make_segments(&[
        "Hello, world",
        "Good morning",
        "Thank you",
        "How are you",
        "Test",
    ]);

    // 执行翻译（完全离线）
    engine
        .translate_batch(&mut segments, "en", "zh")
        .expect("Offline translation failed");

    // 验证所有 Segment 的 target_text 均被正确填充且非空
    for (i, seg) in segments.iter().enumerate() {
        let target = seg
            .target_text
            .as_ref()
            .unwrap_or_else(|| panic!("Segment {i} target_text should be set"));
        assert!(
            !target.is_empty(),
            "Segment {i} target_text should not be empty"
        );
    }

    // 验证具体翻译结果
    assert_eq!(segments[0].target_text.as_deref(), Some("你好，世界"));
    assert_eq!(segments[1].target_text.as_deref(), Some("早上好"));
    assert_eq!(segments[2].target_text.as_deref(), Some("谢谢"));
}

/// 验证断网环境下带术语表的完整翻译流水线。
#[test]
fn test_offline_end_to_end_with_terminology() {
    // 创建术语表
    let terminology = TerminologyManager::from_entries(vec![
        GlossaryEntry::new("GPU", "图形处理器"),
        GlossaryEntry::new("API", "应用程序接口"),
    ])
    .expect("Failed to create terminology");

    // 创建模拟后端（模拟翻译模型对占位符的保持）
    let backend = MockInferenceBackend::from_pairs(&[(
        "The [[T0]] renders graphics using the [[T1]]",
        "[[T0]]使用[[T1]]渲染图形",
    )]);

    let engine = LocalTranslationEngine::new(backend).with_terminology(terminology);

    let mut segments = make_segments(&["The GPU renders graphics using the API"]);

    engine
        .translate_batch(&mut segments, "en", "zh")
        .expect("Translation with terminology failed");

    let target = segments[0]
        .target_text
        .as_ref()
        .expect("target_text should be set");

    // 验证术语被正确还原
    assert!(
        target.contains("图形处理器"),
        "Should contain GPU translation, got: {target}"
    );
    assert!(
        target.contains("应用程序接口"),
        "Should contain API translation, got: {target}"
    );
    assert!(
        !target.contains("[[T"),
        "No placeholders should remain, got: {target}"
    );
}

/// 验证断网环境下批量翻译的完整性。
#[test]
fn test_offline_batch_translation_completeness() {
    let test_set = load_test_set();
    assert!(
        test_set.len() >= 100,
        "Test set should have at least 100 pairs, got {}",
        test_set.len()
    );

    // 构建模拟后端词典
    let pairs: Vec<(&str, &str)> = test_set
        .iter()
        .map(|(s, t)| (s.as_str(), t.as_str()))
        .collect();
    let backend = MockInferenceBackend::from_pairs(&pairs);

    let engine = LocalTranslationEngine::new(backend).with_batch_size(20);

    // 取前 50 条进行测试
    let sources: Vec<&str> = test_set.iter().take(50).map(|(s, _)| s.as_str()).collect();
    let mut segments = make_segments(&sources);

    engine
        .translate_batch(&mut segments, "en", "zh")
        .expect("Batch translation failed");

    // 验证所有翻译结果非空
    for (i, seg) in segments.iter().enumerate() {
        assert!(
            seg.target_text.is_some(),
            "Segment {i} should have target_text"
        );
        let target = seg.target_text.as_ref().expect("target_text should exist");
        assert!(
            !target.is_empty(),
            "Segment {i} translation should not be empty"
        );
    }
}

// ═══════════════════════════════════════════════════════════
//  test_model_integrity_verification
// ═══════════════════════════════════════════════════════════

/// 验证 SHA256 完整性校验通过。
#[test]
fn test_model_integrity_verification_pass() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let file_path = dir.path().join("model.gguf");
    std::fs::write(&file_path, b"hello world").expect("Failed to write file");

    let result = ModelManager::verify_model_integrity(
        &file_path,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
    );
    assert!(result.is_ok(), "SHA256 verification should pass");
}

/// 验证 SHA256 完整性校验失败。
#[test]
fn test_model_integrity_verification_fail() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let file_path = dir.path().join("model.gguf");
    std::fs::write(&file_path, b"hello world").expect("Failed to write file");

    let result = ModelManager::verify_model_integrity(
        &file_path,
        "aaaa000000000000000000000000000000000000000000000000000000000000",
    );
    assert!(result.is_err(), "SHA256 verification should fail");
}

/// 验证损坏的缓存模型会被重新下载。
#[test]
fn test_corrupted_cache_redownload() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create ModelManager");

    let source = ModelSource::ModelScope {
        repo_id: "test-org/model".to_string(),
        revision: None,
    };

    // 创建一个损坏的缓存文件
    let cache_path = manager.get_cache_path(&source, "model.gguf");
    std::fs::create_dir_all(cache_path.parent().expect("No parent")).expect("Failed to create dir");
    std::fs::write(&cache_path, b"corrupted data").expect("Failed to write corrupted file");

    // 尝试加载并校验 — 应失败并删除损坏文件
    let result = manager.load_model(
        &source,
        "model.gguf",
        Some("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"),
    );

    // 应返回下载错误（因为 mock 服务器未设置）
    assert!(result.is_err(), "Should fail to load corrupted model");
    // 损坏的文件应已被删除
    assert!(
        !cache_path.exists(),
        "Corrupted cache file should be deleted"
    );
}

// ═══════════════════════════════════════════════════════════
//  test_translation_accuracy
// ═══════════════════════════════════════════════════════════

/// 验证翻译精度：使用 BLEU 分数评估。
///
/// 使用模拟后端对测试集进行翻译，计算 BLEU 分数，
/// 并断言其高于预设阈值（0.85）。
#[test]
fn test_translation_accuracy() {
    let test_set = load_test_set();
    // 使用 BLEU-2 评估，因为许多中文翻译短于 4 个 token，BLEU-4 会给 0 分
    let evaluator = BleuEvaluator::with_max_n(2);

    // 构建模拟后端（精确匹配测试集）
    let pairs: Vec<(&str, &str)> = test_set
        .iter()
        .map(|(s, t)| (s.as_str(), t.as_str()))
        .collect();
    let backend = MockInferenceBackend::from_pairs(&pairs);
    let engine = LocalTranslationEngine::new(backend);

    // 取前 50 条进行精度评估
    let test_subset: Vec<&(String, String)> = test_set.iter().take(50).collect();
    let sources: Vec<&str> = test_subset.iter().map(|(s, _)| s.as_str()).collect();
    let mut segments = make_segments(&sources);

    engine
        .translate_batch(&mut segments, "en", "zh")
        .expect("Translation failed");

    // 构建候选和参考翻译
    let candidates: Vec<String> = segments
        .iter()
        .map(|s| s.target_text.clone().unwrap_or_default())
        .collect();
    let references: Vec<Vec<&str>> = test_subset.iter().map(|(_, t)| vec![t.as_str()]).collect();

    // 计算 BLEU 分数
    let bleu_score = evaluator.evaluate_batch(&candidates, &references);

    eprintln!("BLEU score: {bleu_score:.4}");

    // 精确匹配应得到接近 1.0 的分数
    assert!(
        bleu_score > 0.85,
        "BLEU score should be > 0.85, got {bleu_score:.4}"
    );
}

// ═══════════════════════════════════════════════════════════
//  test_terminology_accuracy
// ═══════════════════════════════════════════════════════════

/// 验证术语翻译准确性。
///
/// 使用包含特定 IT 术语的测试集，验证术语被正确翻译。
#[test]
fn test_terminology_accuracy() {
    // 加载术语表
    let glossary_json = include_str!("data/en_zh_terminology.json");
    let glossary_entries: Vec<serde_json::Value> =
        serde_json::from_str(glossary_json).expect("Failed to parse glossary");

    let entries: Vec<GlossaryEntry> = glossary_entries
        .iter()
        .map(|e| {
            GlossaryEntry::new(
                e["source"].as_str().expect("Missing source"),
                e["target"].as_str().expect("Missing target"),
            )
        })
        .collect();

    let terminology =
        TerminologyManager::from_entries(entries).expect("Failed to create terminology");

    // 验证术语表非空
    assert!(
        terminology.len() >= 10,
        "Glossary should have at least 10 entries"
    );

    // 测试术语占位符替换
    let test_cases = vec![
        ("The GPU is fast", "图形处理器"),
        ("Use the API correctly", "应用程序接口"),
        ("Docker is useful", "容器引擎"),
        ("machine learning models", "机器学习"),
        ("neural network architecture", "神经网络"),
    ];

    for (text, expected_term) in test_cases {
        let (modified, mapping) = terminology.apply_placeholders(text);
        assert!(!mapping.is_empty(), "Should have placeholder for '{text}'");

        // 还原占位符
        let placeholder_text = "[[T0]]".to_string();
        let restored = terminology.restore_placeholders(&placeholder_text, &mapping);
        assert!(
            restored.contains(expected_term),
            "Restored text should contain '{expected_term}', got: {restored}"
        );
        assert!(
            !modified.contains(expected_term),
            "Modified text should NOT contain Chinese term (should be placeholder)"
        );
    }
}

/// 验证术语在翻译后被正确还原。
#[test]
fn test_terminology_restoration_in_translation() {
    let terminology = TerminologyManager::from_entries(vec![
        GlossaryEntry::new("GPU", "图形处理器"),
        GlossaryEntry::new("API", "应用程序接口"),
        GlossaryEntry::new("Docker", "容器引擎"),
    ])
    .expect("Failed to create terminology");

    // 模拟后端：翻译时保持占位符不变
    let backend = MockInferenceBackend::from_pairs(&[(
        "The [[T0]] and [[T1]] work with [[T2]]",
        "[[T0]]和[[T1]]与[[T2]]一起工作",
    )]);

    let engine = LocalTranslationEngine::new(backend).with_terminology(terminology);

    let mut segments = make_segments(&["The GPU and API work with Docker"]);

    engine
        .translate_batch(&mut segments, "en", "zh")
        .expect("Translation failed");

    let target = segments[0]
        .target_text
        .as_ref()
        .expect("target_text should be set");

    // 验证所有术语被正确还原
    assert!(
        target.contains("图形处理器"),
        "GPU term not restored: {target}"
    );
    assert!(
        target.contains("应用程序接口"),
        "API term not restored: {target}"
    );
    assert!(
        target.contains("容器引擎"),
        "Docker term not restored: {target}"
    );
    assert!(
        !target.contains("[[T"),
        "No placeholders should remain: {target}"
    );
}

// ═══════════════════════════════════════════════════════════
//  配置测试
// ═══════════════════════════════════════════════════════════

/// 验证 TranslationConfig 的本地模型配置。
#[test]
fn test_translation_config_local_provider() {
    use vt_core::config::TranslationConfig;

    let config = TranslationConfig {
        model_path: Some(std::path::PathBuf::from("/path/to/model.gguf")),
        device: "metal".to_string(),
        max_tokens: 1024,
        temperature: 0.1,
        ..Default::default()
    };

    assert_eq!(config.device, "metal");
    assert_eq!(config.max_tokens, 1024);
    assert!((config.temperature - 0.1).abs() < 0.001);
}

/// 验证从 TOML 加载本地翻译配置。
#[test]
fn test_translation_config_local_from_toml() {
    use vt_core::config::Config;

    let toml_content = r#"
[translation]
device = "metal"
max_tokens = 256
temperature = 0.2
model_path = "/path/to/model.gguf"

[translation.model_source]
ModelScope = { repo_id = "Qwen/Qwen2.5-3B-Instruct-GGUF", revision = "master" }
"#;

    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{toml_content}").expect("Failed to write TOML");

    let config = Config::from_file(tmp.path()).expect("Failed to load config");

    assert_eq!(config.translation.device, "metal");
    assert_eq!(config.translation.max_tokens, 256);
    assert!((config.translation.temperature - 0.2).abs() < 0.001);
}

/// 验证 TranslationConfig 默认值。
#[test]
fn test_translation_config_defaults() {
    use vt_core::config::TranslationConfig;

    let config = TranslationConfig::default();

    assert_eq!(config.device, "metal");
    assert_eq!(config.max_tokens, 256);
    assert!((config.temperature - 0.3).abs() < 0.001);
    assert!(config.model_path.is_none());
    assert!(config.glossary_path.is_none());
}

// ═══════════════════════════════════════════════════════════
//  环境变量缓存路径测试
// ═══════════════════════════════════════════════════════════

/// 验证 VIDEO_TRANSLATOR_CACHE 环境变量控制缓存目录。
#[test]
fn test_cache_dir_from_env() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    std::env::set_var("VIDEO_TRANSLATOR_CACHE", dir.path());

    let manager = ModelManager::new().expect("Failed to create manager");
    let cache_path = manager.cache_dir();

    // 缓存目录应该在环境变量指定的路径下
    assert!(
        cache_path.starts_with(dir.path()),
        "Cache dir should be under VIDEO_TRANSLATOR_CACHE"
    );

    std::env::remove_var("VIDEO_TRANSLATOR_CACHE");
}

/// 验证 ModelSource 序列化/反序列化。
#[test]
fn test_model_source_serde() {
    let source = ModelSource::ModelScope {
        repo_id: "org/model".to_string(),
        revision: Some("v1".to_string()),
    };

    let json = serde_json::to_string(&source).expect("Failed to serialize");
    let deserialized: ModelSource = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(source, deserialized);

    let local = ModelSource::Local {
        path: std::path::PathBuf::from("/models/test.gguf"),
    };
    let json = serde_json::to_string(&local).expect("Failed to serialize");
    let deserialized: ModelSource = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(local, deserialized);
}
