//! 集成测试：术语翻译增强（编程术语映射 + 后校正）
//!
//! 验证以下功能：
//! - 内置编程术语映射表覆盖 Rust 常见术语
//! - `TerminologyManager` 正确应用占位符替换和还原
//! - `post_correct()` 修正翻译模型的常见误译
//! - `apply_post_correction()` 对 Segment 列表进行批量校正
//! - `LocalTranslationEngine` 集成内置术语后翻译结果正确
//!
//! # 运行方式
//! ```sh
//! cargo test --test test_terminology -- --nocapture
//! ```

mod common;

use vt_core::error::AppResult;
use vt_core::models::segment::Segment;
use vt_core::translate::{
    apply_post_correction, builtin_programming_terms, post_correct, LocalTranslationEngine,
    MockInferenceBackend, TerminologyManager, TranslationProvider,
};

// ═══════════════════════════════════════════════════════════
//  内置编程术语映射表测试
// ═══════════════════════════════════════════════════════════

/// 验证内置编程术语表至少包含 10 个条目。
#[test]
fn test_builtin_terms_count() {
    let terms = builtin_programming_terms();
    assert!(
        terms.len() >= 10,
        "Should have at least 10 builtin terms, got {}",
        terms.len()
    );
}

/// 验证 `println` 映射为"打印并换行"。
#[test]
fn test_builtin_terms_println() {
    let terms = builtin_programming_terms();
    let entry = terms
        .iter()
        .find(|e| e.source == "println")
        .expect("Should contain 'println' term");
    assert_eq!(entry.target, "打印并换行");
}

/// 验证 `println!` 映射为"打印并换行宏"。
#[test]
fn test_builtin_terms_println_macro() {
    let terms = builtin_programming_terms();
    let entry = terms
        .iter()
        .find(|e| e.source == "println!")
        .expect("Should contain 'println!' term");
    assert_eq!(entry.target, "打印并换行宏");
}

/// 验证 `format!` 映射为"格式化宏"。
#[test]
fn test_builtin_terms_format_macro() {
    let terms = builtin_programming_terms();
    let entry = terms
        .iter()
        .find(|e| e.source == "format!")
        .expect("Should contain 'format!' term");
    assert_eq!(entry.target, "格式化宏");
}

/// 验证 `String` 映射为"字符串"。
#[test]
fn test_builtin_terms_string() {
    let terms = builtin_programming_terms();
    let entry = terms
        .iter()
        .find(|e| e.source == "String")
        .expect("Should contain 'String' term");
    assert_eq!(entry.target, "字符串");
}

/// 验证 `Vec` 映射为"向量"。
#[test]
fn test_builtin_terms_vec() {
    let terms = builtin_programming_terms();
    let entry = terms
        .iter()
        .find(|e| e.source == "Vec")
        .expect("Should contain 'Vec' term");
    assert_eq!(entry.target, "向量");
}

/// 验证内置术语表无重复 source。
#[test]
fn test_builtin_terms_no_duplicates() {
    let terms = builtin_programming_terms();
    let mut sources: Vec<&str> = terms.iter().map(|e| e.source.as_str()).collect();
    sources.sort();
    let duplicates: Vec<&str> = sources
        .windows(2)
        .filter(|w| w[0] == w[1])
        .map(|w| w[0])
        .collect();
    assert!(
        duplicates.is_empty(),
        "Should have no duplicate sources, found: {:?}",
        duplicates
    );
}

// ═══════════════════════════════════════════════════════════
//  TerminologyManager 编程术语集成测试
// ═══════════════════════════════════════════════════════════

/// 验证 TerminologyManager 使用内置编程术语正确替换和还原。
#[test]
fn test_terminology_rust_println() -> AppResult<()> {
    let term = TerminologyManager::from_entries(builtin_programming_terms())?;

    let text = "Use println to print with a newline";
    let (modified, mapping) = term.apply_placeholders(text);

    // println 应被替换为占位符
    assert!(
        !modified.contains("println"),
        "println should be replaced by placeholder, got: {modified}"
    );
    assert!(!mapping.is_empty(), "Should have at least one mapping");

    // 模拟翻译：占位符应被保留
    let translated = format!("使用{placeholder}打印并换行", placeholder = mapping[0].0);

    // 还原占位符
    let restored = term.restore_placeholders(&translated, &mapping);
    assert!(
        restored.contains("打印并换行"),
        "Restored text should contain correct term, got: {restored}"
    );
    assert!(
        !restored.contains("[[T"),
        "Restored text should not contain placeholder, got: {restored}"
    );

    Ok(())
}

/// 验证 TerminologyManager 正确处理 `format!` 宏术语。
#[test]
fn test_terminology_rust_format_macro() -> AppResult<()> {
    let term = TerminologyManager::from_entries(builtin_programming_terms())?;

    let text = "Use format! macro to create strings";
    let (modified, mapping) = term.apply_placeholders(text);

    assert!(
        !modified.contains("format!"),
        "format! should be replaced by placeholder"
    );

    let translated = format!("使用{p}创建字符串", p = mapping[0].0);
    let restored = term.restore_placeholders(&translated, &mapping);
    assert!(
        restored.contains("格式化宏"),
        "Should contain '格式化宏', got: {restored}"
    );

    Ok(())
}

/// 验证 TerminologyManager 正确处理多个编程术语。
#[test]
fn test_terminology_rust_multiple_terms() -> AppResult<()> {
    let term = TerminologyManager::from_entries(builtin_programming_terms())?;

    let text = "The String and Vec types are important in Rust";
    let (modified, mapping) = term.apply_placeholders(text);

    assert!(mapping.len() >= 2, "Should have at least 2 mappings");

    let restored = term.restore_placeholders(&modified, &mapping);
    assert!(
        restored.contains("字符串") || restored.contains("向量"),
        "Restored text should contain Chinese terms"
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════
//  post_correct() 后校正测试
// ═══════════════════════════════════════════════════════════

/// 验证 "打印行" 被修正为 "打印并换行"。
#[test]
fn test_post_correction_println_line() {
    let corrected = post_correct("使用打印行输出内容");
    assert_eq!(corrected, "使用打印并换行输出内容");
}

/// 验证 "打印线" 被修正为 "打印并换行"。
#[test]
fn test_post_correction_println_wire() {
    let corrected = post_correct("这是打印线函数");
    assert_eq!(corrected, "这是打印并换行函数");
}

/// 验证 "输出行" 被修正为 "打印并换行"。
#[test]
fn test_post_correction_output_line() {
    let corrected = post_correct("调用输出行宏");
    assert_eq!(corrected, "调用打印并换行宏");
}

/// 验证 "格式化感叹号" 被修正为 "格式化宏"。
#[test]
fn test_post_correction_format_exclamation() {
    let corrected = post_correct("这是格式化感叹号");
    assert_eq!(corrected, "这是格式化宏");
}

/// 验证 "格式化叹号" 被修正为 "格式化宏"。
#[test]
fn test_post_correction_format_exclamation_short() {
    let corrected = post_correct("使用格式化叹号");
    assert_eq!(corrected, "使用格式化宏");
}

/// 验证多个错误术语同时被修正。
#[test]
fn test_post_correction_multiple_errors() {
    let corrected = post_correct("打印行和格式化感叹号都需要修正");
    assert_eq!(corrected, "打印并换行和格式化宏都需要修正");
}

/// 验证正常文本不被修改。
#[test]
fn test_post_correction_no_change() {
    let text = "这是一段正常的中文文本，不包含错误术语。";
    let corrected = post_correct(text);
    assert_eq!(corrected, text);
}

/// 验证空文本不被修改。
#[test]
fn test_post_correction_empty() {
    let corrected = post_correct("");
    assert_eq!(corrected, "");
}

// ═══════════════════════════════════════════════════════════
//  apply_post_correction() Segment 批量校正测试
// ═══════════════════════════════════════════════════════════

/// 验证 apply_post_correction 正确修正 Segment 的 target_text。
#[test]
fn test_apply_post_correction_segment() {
    let mut seg = Segment::new("s1".into(), 0.0, 5.0, "println".into());
    seg.start_transcribing().ok();
    seg.finish_transcribing("使用打印行输出".into()).ok();

    let mut segments = vec![seg];
    apply_post_correction(&mut segments);

    assert_eq!(
        segments[0].target_text.as_deref(),
        Some("使用打印并换行输出")
    );
}

/// 验证 apply_post_correction 对多个 Segment 批量校正。
#[test]
fn test_apply_post_correction_multiple_segments() {
    let make_seg = |id: &str, text: &str| -> Segment {
        let mut s = Segment::new(id.into(), 0.0, 5.0, "source".into());
        s.start_transcribing().ok();
        s.finish_transcribing(text.into()).ok();
        s
    };

    let mut segments = vec![
        make_seg("s1", "使用打印行输出"),
        make_seg("s2", "这是格式化感叹号"),
        make_seg("s3", "正常文本不变"),
    ];

    apply_post_correction(&mut segments);

    assert_eq!(
        segments[0].target_text.as_deref(),
        Some("使用打印并换行输出")
    );
    assert_eq!(segments[1].target_text.as_deref(), Some("这是格式化宏"));
    assert_eq!(segments[2].target_text.as_deref(), Some("正常文本不变"));
}

/// 验证 apply_post_correction 对空 Segment 列表不报错。
#[test]
fn test_apply_post_correction_empty() {
    let mut segments: Vec<Segment> = vec![];
    apply_post_correction(&mut segments);
}

/// 验证 apply_post_correction 对 None target_text 不报错。
#[test]
fn test_apply_post_correction_none_target() {
    let seg = Segment::new("s1".into(), 0.0, 5.0, "source".into());
    let mut segments = vec![seg];
    apply_post_correction(&mut segments);
    // target_text 仍为 None
    assert!(segments[0].target_text.is_none());
}

// ═══════════════════════════════════════════════════════════
//  LocalTranslationEngine 内置术语集成测试
// ═══════════════════════════════════════════════════════════

/// 验证 LocalTranslationEngine 使用内置编程术语正确翻译含 println 的文本。
#[test]
fn test_engine_with_builtin_terms_println() -> AppResult<()> {
    let term = TerminologyManager::from_entries(builtin_programming_terms())?;

    // Mock 后端：将占位符直接返回（模拟翻译模型保持占位符不变）
    let backend = MockInferenceBackend::from_pairs(&[
        ("Use [[T0]] to print", "使用[[T0]]进行打印"),
        ("[[T0]]", "[[T0]]"),
    ]);

    let engine = LocalTranslationEngine::new(backend).with_terminology(term);

    let mut segments = vec![Segment::new(
        "s1".into(),
        0.0,
        5.0,
        "Use println to print".into(),
    )];
    engine.translate_batch(&mut segments, "en", "zh")?;

    let target = segments[0]
        .target_text
        .as_ref()
        .expect("target_text should be set");
    assert!(
        target.contains("打印并换行"),
        "Should contain '打印并换行', got: {target}"
    );

    Ok(())
}

/// 验证 LocalTranslationEngine 使用内置编程术语正确翻译含 format! 的文本。
#[test]
fn test_engine_with_builtin_terms_format() -> AppResult<()> {
    let term = TerminologyManager::from_entries(builtin_programming_terms())?;

    let backend = MockInferenceBackend::from_pairs(&[("Use [[T0]] macro", "使用[[T0]]宏")]);

    let engine = LocalTranslationEngine::new(backend).with_terminology(term);

    let mut segments = vec![Segment::new(
        "s1".into(),
        0.0,
        5.0,
        "Use format! macro".into(),
    )];
    engine.translate_batch(&mut segments, "en", "zh")?;

    let target = segments[0]
        .target_text
        .as_ref()
        .expect("target_text should be set");
    assert!(
        target.contains("格式化宏"),
        "Should contain '格式化宏', got: {target}"
    );

    Ok(())
}

/// 验证翻译后术语后校正的完整流程。
#[test]
fn test_full_pipeline_terminology_correction() -> AppResult<()> {
    // 步骤1：使用术语表翻译
    let term = TerminologyManager::from_entries(builtin_programming_terms())?;

    // Mock 后端模拟翻译模型将 println 误译为"打印行"
    let backend =
        MockInferenceBackend::from_pairs(&[("Use [[T0]] to print", "使用打印行进行打印")]);

    let engine = LocalTranslationEngine::new(backend).with_terminology(term);

    let mut segments = vec![Segment::new(
        "s1".into(),
        0.0,
        5.0,
        "Use println to print".into(),
    )];
    engine.translate_batch(&mut segments, "en", "zh")?;

    // 步骤2：术语后校正
    apply_post_correction(&mut segments);

    let target = segments[0]
        .target_text
        .as_ref()
        .expect("target_text should be set");

    // "打印行" 应被修正为 "打印并换行"
    assert!(
        target.contains("打印并换行"),
        "Should contain corrected term '打印并换行', got: {target}"
    );
    assert!(
        !target.contains("打印行"),
        "Should not contain incorrect term '打印行', got: {target}"
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════
//  配置字段测试
// ═══════════════════════════════════════════════════════════

/// 验证 TranslationConfig 默认启用 force_glossary 和 post_correction。
#[test]
fn test_translation_config_defaults() {
    use vt_core::config::TranslationConfig;

    let config = TranslationConfig::default();
    assert!(
        config.force_glossary,
        "force_glossary should be true by default"
    );
    assert!(
        config.post_correction_enabled,
        "post_correction_enabled should be true by default"
    );
}

/// 验证 TtsConfig 新增字段默认值正确。
#[test]
fn test_tts_config_new_defaults() {
    use vt_core::config::TtsConfig;

    let config = TtsConfig::default();
    assert_eq!(config.seed, Some(42), "Default seed should be 42");
    assert!(
        (config.temperature - 0.3).abs() < 0.001,
        "Default temperature should be 0.3"
    );
    assert!(
        (config.stability - 0.8).abs() < 0.001,
        "Default stability should be 0.8"
    );
    assert!(
        (config.eq_high_shelf_db - (-3.0)).abs() < 0.001,
        "Default eq_high_shelf_db should be -3.0"
    );
    assert_eq!(
        config.crossfade_duration_ms, 50,
        "Default crossfade_duration_ms should be 50"
    );
}

/// 验证从 TOML 加载新增 TTS 配置字段。
#[test]
fn test_tts_config_new_fields_from_toml() {
    use std::io::Write;
    use vt_core::config::Config;

    let toml_content = r#"
[tts]
speed = 1.0
voice = "Tingting"
seed = 100
temperature = 0.5
stability = 0.9
eq_high_shelf_db = -5.0
crossfade_duration_ms = 100

[translation]
force_glossary = false
post_correction_enabled = false
"#;

    let mut tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{toml_content}").expect("Failed to write TOML");

    let config = Config::from_file(tmp.path()).expect("Failed to load config");

    assert_eq!(config.tts.seed, Some(100));
    assert!((config.tts.temperature - 0.5).abs() < 0.001);
    assert!((config.tts.stability - 0.9).abs() < 0.001);
    assert!((config.tts.eq_high_shelf_db - (-5.0)).abs() < 0.001);
    assert_eq!(config.tts.crossfade_duration_ms, 100);
    assert!(!config.translation.force_glossary);
    assert!(!config.translation.post_correction_enabled);
}
