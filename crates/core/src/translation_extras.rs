//! 翻译增强模块
//!
//! 包含以下功能（参考 pyvideotrans）：
//! - LLM 重断句提示词模板 (P7)
//! - SRT 格式翻译 (P8)
//! - 双语字幕生成 (P9)
//! - 语言特定翻译提示词 (P10)
//! - 术语表 Markdown 表格注入 (P11)

use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

use crate::models::segment::Segment;
use crate::translate::GlossaryEntry;

// ─── P7: LLM 重断句提示词 ─────────────────────────────────

/// 构建 LLM 重断句提示词
///
/// 参考 pyvideotrans 的 `prompts/resegment/llm.txt`，
/// 使用 LLM 对 ASR 输出进行重新断句，改善字幕质量。
pub fn build_resegment_prompt(
    raw_text: &str,
    language: &str,
    target_language: Option<&str>,
) -> String {
    let task_desc = if let Some(target) = target_language {
        format!(
            "You are an expert subtitle editor. The following text was transcribed by ASR \
             and may have poor segmentation. Re-segment it into natural subtitle lines.\n\
             The text is in {language} and should remain in {language}.\n\
             If translation to {target} is needed, translate each line."
        )
    } else {
        format!(
            "You are an expert subtitle editor. The following text was transcribed by ASR \
             and may have poor segmentation. Re-segment it into natural subtitle lines \
             in {language}.\n\
             Rules:\n\
             1. Each line should be 1-15 words (or 5-30 characters for CJK)\n\
             2. Split at natural pause points (commas, periods, conjunctions)\n\
             3. Keep related clauses together\n\
             4. Do NOT merge unrelated sentences\n\
             5. Preserve all content, do not add or remove information\n\
             6. Output each line on a new line, numbered sequentially"
        )
    };

    format!(
        "{task_desc}\n\n\
        # ASR Output (may have poor segmentation)\n\
        {raw_text}\n\n\
        # Output Format\n\
        Output each re-segmented line on a new line, prefixed with line number:\n\
        1. First line\n\
        2. Second line\n\
        ..."
    )
}

// ─── P8: SRT 格式翻译 ──────────────────────────────────────

/// 将 segments 格式化为 SRT 格式字符串
///
/// 参考 pyvideotrans 的 `_run_srt()` 方法，
/// 将多个 segment 组成 SRT 格式文本发送给 LLM 翻译，
/// 翻译后能保持时间戳和行号对齐。
pub fn segments_to_srt(segments: &[Segment]) -> String {
    let mut srt = String::new();
    for (i, seg) in segments.iter().enumerate() {
        let _ = writeln!(srt, "{}", i + 1);
        let _ = writeln!(
            srt,
            "{} --> {}",
            format_srt_time(seg.start),
            format_srt_time(seg.end)
        );
        let _ = writeln!(srt, "{}", seg.source_text);
        let _ = writeln!(srt);
    }
    srt
}

/// 从 SRT 格式字符串解析回 segments
///
/// 解析 LLM 返回的 SRT 格式翻译结果
pub fn srt_to_segments(srt: &str, original_segments: &[Segment]) -> Vec<Segment> {
    let mut results = Vec::new();
    let blocks: Vec<&str> = srt.split("\n\n").collect();

    for (i, block) in blocks.iter().enumerate() {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 {
            continue;
        }

        // 跳过行号
        let _time_line = lines.get(1).unwrap_or(&"");
        let text_lines = &lines[2..];
        let text = text_lines.join("\n").trim().to_string();

        // 从原 segments 获取时间戳
        let (start, end) = if i < original_segments.len() {
            (original_segments[i].start, original_segments[i].end)
        } else {
            (0.0, 0.0)
        };

        let mut seg = if i < original_segments.len() {
            original_segments[i].clone()
        } else {
            Segment::new(format!("seg-{:04}", i + 1), start, end, String::new())
        };
        seg.target_text = Some(text);
        results.push(seg);
    }

    results
}

/// 格式化秒为 SRT 时间格式 (HH:MM:SS,mmm)
fn format_srt_time(secs: f64) -> String {
    let total_ms = (secs * 1000.0) as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

// ─── P9: 双语字幕生成 ──────────────────────────────────────

/// 字幕类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubtitleType {
    /// 不嵌入字幕
    #[default]
    None,
    /// 硬字幕（烧录到画面）
    Hard,
    /// 软字幕（容器内嵌）
    Soft,
    /// 双语硬字幕
    HardBilingual,
    /// 双语软字幕
    SoftBilingual,
}

/// 生成双语 SRT 文件内容
///
/// 参考 pyvideotrans 的双语字幕格式：
/// 源语言在上，目标语言在下
pub fn generate_bilingual_srt(
    segments: &[Segment],
    _source_lang_name: &str,
    _target_lang_name: &str,
) -> String {
    let mut srt = String::new();
    for (i, seg) in segments.iter().enumerate() {
        let _ = writeln!(srt, "{}", i + 1);
        let _ = writeln!(
            srt,
            "{} --> {}",
            format_srt_time(seg.start),
            format_srt_time(seg.end)
        );
        // 源语言在上
        let _ = writeln!(srt, "{}", seg.source_text);
        // 目标语言在下
        if let Some(ref target) = seg.target_text {
            let _ = writeln!(srt, "{target}");
        }
        let _ = writeln!(srt);
    }
    srt
}

/// 生成单语 SRT 文件内容
pub fn generate_srt(segments: &[Segment], use_target: bool) -> String {
    let mut srt = String::new();
    for (i, seg) in segments.iter().enumerate() {
        let _ = writeln!(srt, "{}", i + 1);
        let _ = writeln!(
            srt,
            "{} --> {}",
            format_srt_time(seg.start),
            format_srt_time(seg.end)
        );
        let text = if use_target {
            seg.target_text.as_deref().unwrap_or("")
        } else {
            &seg.source_text
        };
        let _ = writeln!(srt, "{text}");
        let _ = writeln!(srt);
    }
    srt
}

// ─── P10: 语言特定翻译提示词 ───────────────────────────────

/// 语言特定提示词配置
pub struct LanguagePromptConfig {
    /// 语言名称
    pub name: &'static str,
    /// 语言代码
    pub code: &'static str,
    /// 翻译提示词中的语言特定规则
    pub rules: &'static str,
}

/// 获取语言特定提示词
///
/// 参考 pyvideotrans 的 `prompts/language_prompts/` 目录，
/// 为不同目标语言提供特定的翻译规则。
pub fn get_language_prompt(target_lang: &str) -> &'static str {
    match target_lang.split('-').next().unwrap_or("") {
        "zh" => "\
- Use Simplified Chinese (简体中文)\n\
- For technical terms, use established Chinese translations (e.g., API → 接口, function → 函数)\n\
- Keep English origin technical terms (API, GPU, CPU, Docker, Kubernetes) in English when no well-known Chinese equivalent exists\n\
- Use natural spoken Chinese, not written/formal style\n\
- Add appropriate measure words (量词)\n\
- Numbers: use Arabic numerals for technical contexts, Chinese numerals for colloquial",
        "ja" => "\
- Use natural Japanese (自然な日本語)\n\
- Use です/ます form for polite narration\n\
- For technical terms, use katakana for loan words (e.g., API → エーピーアイ)\n\
- Keep English technical terms in English when appropriate",
        "ko" => "\
- Use natural Korean (자연스러운 한국어)\n\
- Use 해요체 (polite informal) for narration\n\
- For technical terms, use established Korean translations\n\
- Keep English technical terms in English when appropriate",
        "fr" => "\
- Use natural French (français naturel)\n\
- Use formal address (vous) for professional content\n\
- Adapt sentence structure to French syntax, do not calque English structures",
        "de" => "\
- Use natural German (natürliches Deutsch)\n\
- Use Sie-form for professional content\n\
- Capitalize all nouns\n\
- Use established German technical translations",
        "es" => "\
- Use natural Spanish (español natural)\n\
- Use tú form for casual, usted for formal content\n\
- Use established Spanish technical translations",
        "pt" => "\
- Use natural Portuguese (português natural)\n\
- Use Brazilian Portuguese unless European is specified",
        "ru" => "\
- Use natural Russian (естественный русский язык)\n\
- Use established Russian technical translations",
        "it" => "\
- Use natural Italian (italiano naturale)\n\
- Use established Italian technical translations",
        _ => "\
- Use natural, spoken language appropriate for voice-over narration\n\
- Keep technical terms in English when no well-known local equivalent exists",
    }
}

// ─── P11: 术语表 Markdown 表格注入 ─────────────────────────

/// 将术语表格式化为 Markdown 表格
///
/// 参考 pyvideotrans 的术语表注入方式，
/// 将术语表格式化为 Markdown 表格并注入提示词，
/// 让 LLM 严格遵循术语翻译。
pub fn format_glossary_as_markdown(entries: &[GlossaryEntry]) -> String {
    if entries.is_empty() {
        return String::new();
    }

    let mut table = String::new();
    table.push_str("# Glossary\n\n");
    table.push_str("| Source Term | Target Translation |\n");
    table.push_str("|---|---|\n");
    for entry in entries {
        table.push_str(&format!("| {} | {} |\n", entry.source, entry.target));
    }
    table.push_str(
        "\nIMPORTANT: You MUST strictly follow the glossary above when translating. \
         If a source term appears in the text, use the exact target translation from the table.\n",
    );
    table
}

/// 构建包含术语表的完整提示词
pub fn build_prompt_with_glossary(
    system_prompt: &str,
    entries: &[GlossaryEntry],
    target_lang: &str,
) -> String {
    let glossary = format_glossary_as_markdown(entries);
    let lang_rules = get_language_prompt(target_lang);

    format!(
        "{system_prompt}\n\n\
        {glossary}\n\n\
        # Language-Specific Rules\n\
        {lang_rules}"
    )
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segments_to_srt() {
        let segs = vec![
            Segment::new("1".into(), 0.0, 3.0, "Hello".into()),
            Segment::new("2".into(), 3.0, 6.0, "World".into()),
        ];
        let srt = segments_to_srt(&segs);
        assert!(srt.contains("1"));
        assert!(srt.contains("00:00:00,000 --> 00:00:03,000"));
        assert!(srt.contains("Hello"));
        assert!(srt.contains("World"));
    }

    #[test]
    fn test_srt_to_segments() {
        let srt =
            "1\n00:00:00,000 --> 00:00:03,000\n你好\n\n2\n00:00:03,000 --> 00:00:06,000\n世界\n";
        let orig = vec![
            Segment::new("1".into(), 0.0, 3.0, "Hello".into()),
            Segment::new("2".into(), 3.0, 6.0, "World".into()),
        ];
        let result = srt_to_segments(srt, &orig);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].target_text.as_deref(), Some("你好"));
        assert_eq!(result[1].target_text.as_deref(), Some("世界"));
        // 时间戳应从原 segments 继承
        assert!((result[0].start - 0.0).abs() < 0.01);
        assert!((result[0].end - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.5), "00:00:01,500");
        assert_eq!(format_srt_time(65.5), "00:01:05,500");
        assert_eq!(format_srt_time(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_generate_bilingual_srt() {
        let mut seg = Segment::new("1".into(), 0.0, 3.0, "Hello".into());
        seg.target_text = Some("你好".to_string());
        let segs = vec![seg];
        let srt = generate_bilingual_srt(&segs, "English", "Chinese");
        assert!(srt.contains("Hello"));
        assert!(srt.contains("你好"));
        // Source should come before target
        let source_pos = srt.find("Hello").unwrap();
        let target_pos = srt.find("你好").unwrap();
        assert!(source_pos < target_pos);
    }

    #[test]
    fn test_generate_srt_source() {
        let segs = vec![Segment::new("1".into(), 0.0, 3.0, "Hello".into())];
        let srt = generate_srt(&segs, false);
        assert!(srt.contains("Hello"));
    }

    #[test]
    fn test_generate_srt_target() {
        let mut seg = Segment::new("1".into(), 0.0, 3.0, "Hello".into());
        seg.target_text = Some("你好".to_string());
        let srt = generate_srt(&[seg], true);
        assert!(srt.contains("你好"));
        assert!(!srt.contains("Hello"));
    }

    #[test]
    fn test_get_language_prompt() {
        let prompt = get_language_prompt("zh");
        assert!(prompt.contains("Simplified Chinese"));
        assert!(prompt.contains("量词"));

        let prompt = get_language_prompt("ja");
        assert!(prompt.contains("日本語"));

        let prompt = get_language_prompt("unknown");
        assert!(prompt.contains("natural"));
    }

    #[test]
    fn test_format_glossary_as_markdown() {
        let entries = vec![
            GlossaryEntry::new("API", "接口"),
            GlossaryEntry::new("GPU", "图形处理器"),
        ];
        let table = format_glossary_as_markdown(&entries);
        assert!(table.contains("Glossary"));
        assert!(table.contains("API"));
        assert!(table.contains("接口"));
        assert!(table.contains("GPU"));
        assert!(table.contains("图形处理器"));
        assert!(table.contains("|---|---|"));
    }

    #[test]
    fn test_format_glossary_empty() {
        let table = format_glossary_as_markdown(&[]);
        assert!(table.is_empty());
    }

    #[test]
    fn test_build_prompt_with_glossary() {
        let entries = vec![GlossaryEntry::new("API", "接口")];
        let prompt = build_prompt_with_glossary("You are a translator.", &entries, "zh");
        assert!(prompt.contains("Glossary"));
        assert!(prompt.contains("API"));
        assert!(prompt.contains("接口"));
        assert!(prompt.contains("Simplified Chinese"));
    }

    #[test]
    fn test_resegment_prompt() {
        let prompt = build_resegment_prompt("hello world this is a test", "en", None);
        assert!(prompt.contains("subtitle editor"));
        assert!(prompt.contains("hello world"));
        assert!(prompt.contains("natural"));
    }

    #[test]
    fn test_resegment_prompt_with_translation() {
        let prompt = build_resegment_prompt("hello world", "en", Some("zh"));
        assert!(prompt.contains("translation"));
    }
}
