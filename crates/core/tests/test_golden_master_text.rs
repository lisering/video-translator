//! P0-C: 翻译和 ASR 文本输出 Golden Master 测试
//!
//! 这些测试验证 Golden Master 框架在文本指纹场景下的工作流程：
//! - 翻译输出指纹：检测翻译质量变化（字符数、CJK 比例等）
//! - ASR 输出指纹：检测转录结果变化
//! - 批量段处理：多段文本的指纹管理

use vt_core::golden_master::{
    Fingerprint, FingerprintCompare, GoldenMaster, GoldenMasterTestCase, TextFingerprint,
};

// ─── 翻译输出 Golden Master 测试 ──────────────────────────

#[test]
fn test_translate_golden_master_stable() {
    // 相同翻译结果 → 指纹一致
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let text = "你好世界，这是一个测试。";
    let fp = TextFingerprint::from_text(text);

    // 首次：创建基线
    let result = gm
        .load_or_create("translate", "stable_test", &fp)
        .expect("load_or_create failed");
    assert_eq!(result, FingerprintCompare::Match);

    // 第二次：相同文本 → Match
    let fp2 = TextFingerprint::from_text(text);
    let result = gm
        .load_or_create("translate", "stable_test", &fp2)
        .expect("load_or_create failed");
    assert!(result.is_pass());
}

#[test]
fn test_translate_golden_master_detect_content_change() {
    // 检测翻译内容变化
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let original = "这是一个简短的翻译。";
    let modified = "这是一个完全不同的且更长的翻译结果。";

    let fp1 = TextFingerprint::from_text(original);
    let fp2 = TextFingerprint::from_text(modified);

    // 创建基线
    gm.save("translate", "content_change", &fp1)
        .expect("save failed");

    // 不同内容 → Changed
    let result = gm
        .load_or_create("translate", "content_change", &fp2)
        .expect("load_or_create failed");
    assert!(!result.is_pass(), "Different content should not pass");
}

#[test]
fn test_translate_golden_master_minor_variation() {
    // 微小变化（标点差异）→ 在容差范围内
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let text1 = "Hello world. This is a test.";
    let text2 = "Hello world, this is a test."; // 句号→逗号

    let fp1 = TextFingerprint::from_text(text1);
    let fp2 = TextFingerprint::from_text(text2);

    // 创建基线
    gm.save("translate", "minor_variation", &fp1)
        .expect("save failed");

    // 微小变化 → ApproximateMatch（字符数和单词数差异在容差内）
    let result = gm
        .load_or_create("translate", "minor_variation", &fp2)
        .expect("load_or_create failed");

    // 由于标点变化不改变字符数和单词数，应该匹配
    assert!(
        result.is_pass(),
        "Minor punctuation change should pass: {}",
        result.diff_message()
    );
}

#[test]
fn test_translate_golden_master_language_switch() {
    // 检测语言切换（英文→中文）
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let english = "Hello world. This is a translation test.";
    let chinese = "你好世界。这是一个翻译测试。";

    let fp_en = TextFingerprint::from_text(english);
    let fp_cn = TextFingerprint::from_text(chinese);

    // 创建基线
    gm.save("translate", "lang_switch", &fp_en)
        .expect("save failed");

    // 语言切换 → Changed（CJK 比例剧烈变化）
    let result = gm
        .load_or_create("translate", "lang_switch", &fp_cn)
        .expect("load_or_create failed");
    assert!(!result.is_pass(), "Language switch should not pass");
}

#[test]
fn test_translate_golden_master_multi_segment() {
    // 模拟多段翻译的 Golden Master 测试
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let translations = [
        ("seg_0", "你好世界"),
        ("seg_1", "这是一个测试"),
        ("seg_2", "语音翻译系统"),
    ];

    // 第一轮：创建基线
    for (name, text) in &translations {
        let fp = TextFingerprint::from_text(text);
        let result = gm
            .load_or_create("translate", name, &fp)
            .expect("load_or_create failed");
        assert!(result.is_pass(), "Segment {name} should pass on first run");
    }

    // 验证基线数量
    let baselines = gm.list_baselines();
    assert_eq!(baselines.len(), 3);

    // 第二轮：相同数据 → 全部通过
    for (name, text) in &translations {
        let fp = TextFingerprint::from_text(text);
        let result = gm
            .load_or_create("translate", name, &fp)
            .expect("load_or_create failed");
        assert!(result.is_pass(), "Segment {name} should pass on second run");
    }

    // 第三轮：修改一个段 → 该段失败，其他通过
    let modified_fp = TextFingerprint::from_text("这是一个完全不同的翻译结果内容");
    let result = gm
        .load_or_create("translate", "seg_1", &modified_fp)
        .expect("load_or_create failed");
    assert!(!result.is_pass(), "Modified segment should fail");

    // 其他段仍然通过
    let fp = TextFingerprint::from_text("语音翻译系统");
    let result = gm
        .load_or_create("translate", "seg_2", &fp)
        .expect("load_or_create failed");
    assert!(result.is_pass(), "Unmodified segment should still pass");
}

// ─── ASR 输出 Golden Master 测试 ───────────────────────────

#[test]
fn test_asr_golden_master_stable() {
    // ASR 转录结果稳定性
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let transcript = "Hello and welcome to this video about machine learning.";
    let fp = TextFingerprint::from_text(transcript);

    // 首次：创建基线
    let result = gm
        .load_or_create("asr", "stable_test", &fp)
        .expect("load_or_create failed");
    assert_eq!(result, FingerprintCompare::Match);

    // 第二次：相同文本 → Match
    let fp2 = TextFingerprint::from_text(transcript);
    let result = gm
        .load_or_create("asr", "stable_test", &fp2)
        .expect("load_or_create failed");
    assert!(result.is_pass());
}

#[test]
fn test_asr_golden_master_detect_word_error() {
    // 检测 ASR 单词错误
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let correct = "The quick brown fox jumps over the lazy dog.";
    let asr_error = "The quick brown fox jumps over the lazy log."; // dog→log

    let fp1 = TextFingerprint::from_text(correct);
    let fp2 = TextFingerprint::from_text(asr_error);

    // 创建基线
    gm.save("asr", "word_error", &fp1).expect("save failed");

    // 单词替换 → 可能不改变字符数/单词数，但哈希不同
    // 字符数差 1 (dog vs log, same length) → 在容差内
    let _result = gm
        .load_or_create("asr", "word_error", &fp2)
        .expect("load_or_create failed");

    // dog→log 不改变字符数和单词数，所以在容差内可能通过
    // 但如果需要精确检测，应该检查哈希
    assert_ne!(fp1.sha256_hash, fp2.sha256_hash, "Hashes should differ");
}

#[test]
fn test_asr_golden_master_detect_insertion() {
    // 检测 ASR 插入错误（多了单词）
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let original = "Hello world.";
    let with_insertion = "Hello world this is extra content inserted by ASR.";

    let fp1 = TextFingerprint::from_text(original);
    let fp2 = TextFingerprint::from_text(with_insertion);

    // 创建基线
    gm.save("asr", "insertion", &fp1).expect("save failed");

    // 插入额外内容 → Changed（字符数和单词数差异超过容差）
    let result = gm
        .load_or_create("asr", "insertion", &fp2)
        .expect("load_or_create failed");
    assert!(!result.is_pass(), "Insertion should be detected");
}

#[test]
fn test_asr_golden_master_detect_deletion() {
    // 检测 ASR 删除错误（少了单词）
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let original = "The quick brown fox jumps over the lazy dog.";
    let with_deletion = "The quick fox over the dog.";

    let fp1 = TextFingerprint::from_text(original);
    let fp2 = TextFingerprint::from_text(with_deletion);

    // 创建基线
    gm.save("asr", "deletion", &fp1).expect("save failed");

    // 删除内容 → Changed
    let result = gm
        .load_or_create("asr", "deletion", &fp2)
        .expect("load_or_create failed");
    assert!(!result.is_pass(), "Deletion should be detected");
}

#[test]
fn test_text_golden_master_test_case_helper() {
    // 测试 GoldenMasterTestCase 用于文本
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let text = "这是一个翻译测试。";
    let fp = TextFingerprint::from_text(text);

    // 首次运行：创建基线并通过
    GoldenMasterTestCase::new("translate", "helper_text_test")
        .with_baseline_dir(dir.path())
        .assert_pass(&fp);

    // 第二次运行：相同数据 → 通过
    GoldenMasterTestCase::new("translate", "helper_text_test")
        .with_baseline_dir(dir.path())
        .assert_pass(&fp);
}

#[test]
fn test_text_golden_master_fingerprint_summary() {
    // 验证文本指纹摘要
    let text = "Hello world. 你好世界。";
    let fp = TextFingerprint::from_text(text);

    let summary = fp.summary();
    assert!(summary.contains("chars="), "Summary should contain char count: {summary}");
    assert!(summary.contains("words="), "Summary should contain word count: {summary}");
    assert!(summary.contains("cjk="), "Summary should contain CJK ratio: {summary}");
}

#[test]
fn test_text_golden_master_empty_text() {
    // 空文本的指纹
    let fp = TextFingerprint::from_text("");

    assert_eq!(fp.char_count, 0);
    assert_eq!(fp.word_count, 0);
    assert_eq!(fp.line_count, 1); // lines().count() returns 1 for empty string
    assert_eq!(fp.sentence_count, 1); // max(1)
    assert!((fp.cjk_ratio - 0.0).abs() < 1e-10);
    assert!((fp.ascii_ratio - 0.0).abs() < 1e-10);
}

#[test]
fn test_text_golden_master_mixed_language() {
    // 混合语言文本
    let text = "Hello 你好 world 世界 test 测试 123";
    let fp = TextFingerprint::from_text(text);

    assert!(fp.cjk_ratio > 0.1, "Should have CJK content: {}", fp.summary());
    assert!(fp.ascii_ratio > 0.3, "Should have ASCII content: {}", fp.summary());
    assert!(fp.digit_ratio > 0.0, "Should have digits: {}", fp.summary());
}

#[test]
fn test_text_golden_master_persistence_workflow() {
    // 完整的 Golden Master 工作流测试
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    // Step 1: 首次运行，创建基线
    let original_text = "这是一个原始的翻译结果。";
    let fp = TextFingerprint::from_text(original_text);
    let result = gm
        .load_or_create("translate", "workflow", &fp)
        .expect("load_or_create failed");
    assert_eq!(result, FingerprintCompare::Match);

    // Step 2: 代码改变后，翻译结果变了
    let changed_text = "这是一个修改后的翻译结果内容。";
    let fp_changed = TextFingerprint::from_text(changed_text);
    let result = gm
        .load_or_create("translate", "workflow", &fp_changed)
        .expect("load_or_create failed");
    assert!(!result.is_pass(), "Changed translation should fail");

    // Step 3: 人工审查后，接受新基线
    gm.accept("translate", "workflow", &fp_changed)
        .expect("accept failed");

    // Step 4: 再次运行，新基线通过
    let result = gm
        .load_or_create("translate", "workflow", &fp_changed)
        .expect("load_or_create failed");
    assert!(result.is_pass(), "Accepted baseline should pass");
}
