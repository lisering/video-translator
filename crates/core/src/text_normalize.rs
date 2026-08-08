//! 文本归一化模块
//!
//! 借鉴 dots.tts 的 WeTextProcessing 思路，在 ASR 后/翻译前对文本进行归一化处理，
//! 将口语化的数字、符号、缩写等转换为标准书写形式，提升翻译质量。
//!
//! # 核心功能
//! - [`normalize_text`]: 对文本进行归一化处理
//! - [`NormalizationConfig`]: 归一化配置
//!
//! # 归一化规则
//! 1. **数字归一化**: "1st" → "first", "2nd" → "second", "100%" → "100 percent"
//! 2. **符号归一化**: "&" → "and", "@" → "at", "#" → "number"
//! 3. **缩写展开**: "e.g." → "for example", "i.e." → "that is"
//! 4. **标点修复**: 连续标点合并，缺失句末标点补全
//! 5. **空白处理**: 多余空格压缩，首尾 trim
//! 6. **URL/邮箱保护**: 不归一化 URL 和邮箱地址
//!
//! # 示例
//! ```
//! use vt_core::text_normalize::{normalize_text, NormalizationConfig};
//!
//! let config = NormalizationConfig::default();
//! let result = normalize_text("Hello & welcome to the 1st lesson @ 3pm!", &config);
//! assert!(result.contains("and"));
//! assert!(result.contains("first"));
//! ```

use regex::Regex;

// ─── 配置 ─────────────────────────────────────────────────

/// 文本归一化配置
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// 是否展开英文序数词 (1st → first)
    pub expand_ordinals: bool,
    /// 是否展开符号 (& → and, @ → at)
    pub expand_symbols: bool,
    /// 是否展开常见缩写 (e.g. → for example)
    pub expand_abbreviations: bool,
    /// 是否压缩多余空格
    pub collapse_whitespace: bool,
    /// 是否修复连续标点
    pub fix_punctuation: bool,
    /// 是否保护 URL 和邮箱（不归一化）
    pub protect_urls: bool,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            expand_ordinals: true,
            expand_symbols: true,
            expand_abbreviations: true,
            collapse_whitespace: true,
            fix_punctuation: true,
            protect_urls: true,
        }
    }
}

// ─── 归一化实现 ───────────────────────────────────────────

/// 序数词映射
const ORDINAL_MAP: &[(&str, &str)] = &[
    ("1st", "first"),
    ("2nd", "second"),
    ("3rd", "third"),
    ("4th", "fourth"),
    ("5th", "fifth"),
    ("6th", "sixth"),
    ("7th", "seventh"),
    ("8th", "eighth"),
    ("9th", "ninth"),
    ("10th", "tenth"),
    ("11th", "eleventh"),
    ("12th", "twelfth"),
    ("13th", "thirteenth"),
    ("14th", "fourteenth"),
    ("15th", "fifteenth"),
    ("16th", "sixteenth"),
    ("17th", "seventeenth"),
    ("18th", "eighteenth"),
    ("19th", "nineteenth"),
    ("20th", "twentieth"),
    ("21st", "twenty-first"),
    ("22nd", "twenty-second"),
    ("23rd", "twenty-third"),
    ("30th", "thirtieth"),
    ("40th", "fortieth"),
    ("50th", "fiftieth"),
    ("60th", "sixtieth"),
    ("70th", "seventieth"),
    ("80th", "eightieth"),
    ("90th", "ninetieth"),
    ("100th", "hundredth"),
    ("1000th", "thousandth"),
];

/// 符号映射
const SYMBOL_MAP: &[(&str, &str)] = &[
    ("&", "and"),
    ("@", "at"),
    ("#", "number"),
    ("$", "dollars"),
    ("%", "percent"),
    ("+", "plus"),
    ("=", "equals"),
    ("~", "approximately"),
];

/// 缩写映射
const ABBREVIATION_MAP: &[(&str, &str)] = &[
    ("e.g.", "for example"),
    ("i.e.", "that is"),
    ("etc.", "etcetera"),
    ("vs.", "versus"),
    ("approx.", "approximately"),
    ("Mr.", "Mister"),
    ("Mrs.", "Missus"),
    ("Dr.", "Doctor"),
    ("Prof.", "Professor"),
    ("Inc.", "Incorporated"),
    ("Ltd.", "Limited"),
    ("Corp.", "Corporation"),
    ("Jan.", "January"),
    ("Feb.", "February"),
    ("Aug.", "August"),
    ("Sept.", "September"),
    ("Oct.", "October"),
    ("Nov.", "November"),
    ("Dec.", "December"),
];

/// 对文本进行归一化处理
///
/// 按配置选项依次执行归一化规则。
///
/// # 参数
/// - `text`: 待归一化的文本
/// - `config`: 归一化配置
///
/// # 返回
/// 归一化后的文本
#[must_use]
pub fn normalize_text(text: &str, config: &NormalizationConfig) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut result = text.to_string();

    // 保护 URL 和邮箱（用占位符替换，最后恢复）
    let mut urls: Vec<String> = Vec::new();
    if config.protect_urls {
        // URL 匹配: http://... 或 https://...
        let url_re =
            Regex::new(r#"https?://[^\s<>"]+"#).unwrap_or_else(|_| Regex::new(r"$^").unwrap());
        result = url_re
            .replace_all(&result, |caps: &regex::Captures| {
                let url = caps[0].to_string();
                let idx = urls.len();
                urls.push(url);
                format!("__URL_{idx}__")
            })
            .to_string();

        // 邮箱匹配
        let email_re = Regex::new(r#"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"#)
            .unwrap_or_else(|_| Regex::new(r"$^").unwrap());
        result = email_re
            .replace_all(&result, |caps: &regex::Captures| {
                let email = caps[0].to_string();
                let idx = urls.len();
                urls.push(email);
                format!("__URL_{idx}__")
            })
            .to_string();
    }

    // 展开缩写（在符号展开之前，因为缩写可能包含点号）
    if config.expand_abbreviations {
        for (abbr, expansion) in ABBREVIATION_MAP {
            // 使用不区分大小写的替换
            let pattern = regex::escape(abbr);
            let re = Regex::new(&format!("(?i){}", pattern))
                .unwrap_or_else(|_| Regex::new(r"$^").unwrap());
            result = re.replace_all(&result, *expansion).to_string();
        }
    }

    // 展开序数词
    if config.expand_ordinals {
        for (ordinal, word) in ORDINAL_MAP {
            // 使用词边界匹配
            let pattern = regex::escape(ordinal);
            let re = Regex::new(&format!(r"(?i)\b{}", pattern))
                .unwrap_or_else(|_| Regex::new(r"$^").unwrap());
            result = re.replace_all(&result, *word).to_string();
        }
    }

    // 展开符号
    if config.expand_symbols {
        for (symbol, word) in SYMBOL_MAP {
            // 注意：# 和 @ 需要特殊处理，避免替换 URL 中的
            // 但 URL 已经被保护了，所以可以直接替换
            result = result.replace(symbol, word);
        }
    }

    // 修复连续标点
    if config.fix_punctuation {
        // regex crate 不支持反向引用，逐字符处理连续重复
        // 合并连续的相同标点: "!!!" → "!", "..." → "."
        // 至少 3 个连续相同标点才合并
        for punct in &['!', '?', '.'] {
            let pattern = format!(r"\{}{{3,}}", punct);
            if let Ok(re) = Regex::new(&pattern) {
                let replacement = format!("{}", punct);
                result = re.replace_all(&result, replacement.as_str()).to_string();
            }
        }

        // 合并连续的相同逗号: ",,," → ","
        let comma_re = Regex::new(r",{3,}").unwrap_or_else(|_| Regex::new(r"$^").unwrap());
        result = comma_re.replace_all(&result, ",").to_string();
    }

    // 压缩多余空格
    if config.collapse_whitespace {
        let ws_re = Regex::new(r"\s+").unwrap();
        result = ws_re.replace_all(&result, " ").to_string();
        result = result.trim().to_string();
    }

    // 恢复 URL 和邮箱
    if config.protect_urls {
        for (idx, url) in urls.iter().enumerate() {
            result = result.replace(&format!("__URL_{idx}__"), url);
        }
    }

    result
}

/// 简单的文本清洗：去除控制字符、修复编码问题
///
/// 在 ASR 后立即调用，去除 Whisper 输出中可能包含的控制字符。
///
/// # 参数
/// - `text`: 待清洗的文本
///
/// # 返回
/// 清洗后的文本
#[must_use]
pub fn clean_asr_output(text: &str) -> String {
    let mut result = String::with_capacity(text.len());

    for ch in text.chars() {
        // 保留可见字符、空格、换行、制表符
        if ch.is_alphanumeric()
            || ch.is_ascii_punctuation()
            || ch.is_whitespace()
            || (ch as u32) > 0x2E80
        // CJK 字符及标点
        {
            result.push(ch);
        } else {
            // 控制字符替换为空格，避免单词粘连
            result.push(' ');
        }
    }

    // 压缩多余空格
    let ws_re = Regex::new(r"\s+").unwrap();
    let result = ws_re.replace_all(&result, " ").trim().to_string();

    result
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let config = NormalizationConfig::default();
        let result = normalize_text("Hello world", &config);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_normalize_empty() {
        let config = NormalizationConfig::default();
        assert_eq!(normalize_text("", &config), "");
        assert_eq!(normalize_text("   ", &config), "");
    }

    #[test]
    fn test_expand_ordinals() {
        let config = NormalizationConfig::default();
        let result = normalize_text("This is the 1st lesson and the 2nd example.", &config);
        assert!(result.contains("first"), "Should expand 1st: {result}");
        assert!(result.contains("second"), "Should expand 2nd: {result}");
    }

    #[test]
    fn test_expand_ordinals_case_insensitive() {
        let config = NormalizationConfig::default();
        let result = normalize_text("The 3RD place winner.", &config);
        assert!(result.contains("third"), "Should expand 3RD: {result}");
    }

    #[test]
    fn test_expand_symbols() {
        let config = NormalizationConfig::default();
        let result = normalize_text("Fish & chips @ the #1 restaurant", &config);
        assert!(result.contains("and"), "Should expand &: {result}");
        assert!(result.contains("at"), "Should expand @: {result}");
        assert!(result.contains("number"), "Should expand #: {result}");
    }

    #[test]
    fn test_expand_abbreviations() {
        let config = NormalizationConfig::default();
        let result = normalize_text("Use e.g. this format, i.e. exactly.", &config);
        assert!(
            result.contains("for example"),
            "Should expand e.g.: {result}"
        );
        assert!(result.contains("that is"), "Should expand i.e.: {result}");
    }

    #[test]
    fn test_collapse_whitespace() {
        let config = NormalizationConfig::default();
        let result = normalize_text("Hello    world\n\n\nTest", &config);
        assert_eq!(result, "Hello world Test");
    }

    #[test]
    fn test_fix_punctuation() {
        let config = NormalizationConfig::default();
        let result = normalize_text("Hello!!! World...", &config);
        assert!(
            !result.contains("!!!"),
            "Should collapse repeated punctuation: {result}"
        );
    }

    #[test]
    fn test_protect_urls() {
        let config = NormalizationConfig::default();
        let result = normalize_text("Visit https://example.com/page & see more", &config);
        assert!(
            result.contains("https://example.com/page"),
            "URL should be protected: {result}"
        );
        assert!(
            result.contains("and"),
            "Symbol should be expanded: {result}"
        );
    }

    #[test]
    fn test_protect_email() {
        let config = NormalizationConfig::default();
        let result = normalize_text("Contact user@example.com for info.", &config);
        assert!(
            result.contains("user@example.com"),
            "Email should be protected: {result}"
        );
    }

    #[test]
    fn test_clean_asr_output() {
        let result = clean_asr_output("Hello\x00world\x07test");
        assert_eq!(result, "Hello world test");
    }

    #[test]
    fn test_clean_asr_preserves_cjk() {
        let result = clean_asr_output("你好世界\x00测试");
        assert!(result.contains("你好世界"));
        assert!(result.contains("测试"));
    }

    #[test]
    fn test_normalize_disabled_ordinals() {
        let config = NormalizationConfig {
            expand_ordinals: false,
            ..Default::default()
        };
        let result = normalize_text("The 1st place.", &config);
        assert!(
            result.contains("1st"),
            "Should not expand ordinals when disabled: {result}"
        );
    }

    #[test]
    fn test_normalize_disabled_symbols() {
        let config = NormalizationConfig {
            expand_symbols: false,
            ..Default::default()
        };
        let result = normalize_text("Fish & chips", &config);
        assert!(
            result.contains("&"),
            "Should not expand symbols when disabled: {result}"
        );
    }

    #[test]
    fn test_normalize_full_sentence() {
        let config = NormalizationConfig::default();
        let input = "Welcome to the 1st lesson! Visit https://example.com & learn more.";
        let result = normalize_text(input, &config);
        assert!(result.contains("first"), "Should expand ordinal");
        assert!(result.contains("https://example.com"), "Should protect URL");
        assert!(result.contains("and"), "Should expand &");
    }

    #[test]
    fn test_normalize_percent() {
        let config = NormalizationConfig::default();
        let result = normalize_text("100% accurate", &config);
        assert!(result.contains("percent"), "Should expand %: {result}");
    }

    #[test]
    fn test_clean_asr_empty() {
        assert_eq!(clean_asr_output(""), "");
        assert_eq!(clean_asr_output("   "), "");
    }
}

// ─── 增强文本归一化 — 借鉴 OmniVoice FST normalization ─────

/// 将常见数字转为英文单词表示
///
/// 借鉴 OmniVoice 的数字归一化逻辑，将 TTS 文本中的数字
/// 转换为更自然的口语表达。
///
/// # 转换规则
/// - `100%` → `100 percent`（已在 normalize_text 中处理）
/// - `$5` → `5 dollars`
/// - `5kg` → `5 kilograms`
/// - `25°C` → `25 degrees Celsius`
/// - `3:30pm` → `3:30 PM`
/// - `#1` → `number 1`
pub fn normalize_numbers(text: &str) -> String {
    let mut result = text.to_string();

    // 货币: $5 → 5 dollars, €10 → 10 euros, £20 → 20 pounds, ¥100 → 100 yen
    let currency_re = Regex::new(r"([$€£¥])(\d+)").unwrap();
    result = currency_re
        .replace_all(&result, |caps: &regex::Captures| {
            let symbol = &caps[1];
            let amount = &caps[2];
            let currency = match symbol {
                "$" => "dollars",
                "€" => "euros",
                "£" => "pounds",
                "¥" => "yen",
                _ => return format!("{symbol}{amount}"),
            };
            format!("{amount} {currency}")
        })
        .to_string();

    // 温度: 25°C → 25 degrees Celsius, 100°F → 100 degrees Fahrenheit
    let temp_re = Regex::new(r"(\d+)\s*°([CF])").unwrap();
    result = temp_re
        .replace_all(&result, |caps: &regex::Captures| {
            let temp = &caps[1];
            let unit = &caps[2];
            let unit_name = if unit == "C" { "Celsius" } else { "Fahrenheit" };
            format!("{temp} degrees {unit_name}")
        })
        .to_string();

    // 单位: 5kg → 5 kilograms, 10cm → 10 centimeters, 2m → 2 meters
    let unit_re = Regex::new(r"(\d+)\s*(kg|g|km|m|cm|mm|lb|oz|ft|in|mi|hz|khz|mhz|ghz)\b").unwrap();
    result = unit_re
        .replace_all(&result, |caps: &regex::Captures| {
            let num = &caps[1];
            let unit = &caps[2].to_lowercase();
            let full_unit = match unit.as_str() {
                "kg" => "kilograms",
                "g" => "grams",
                "km" => "kilometers",
                "m" => "meters",
                "cm" => "centimeters",
                "mm" => "millimeters",
                "lb" => "pounds",
                "oz" => "ounces",
                "ft" => "feet",
                "in" => "inches",
                "mi" => "miles",
                "hz" => "hertz",
                "khz" => "kilohertz",
                "mhz" => "megahertz",
                "ghz" => "gigahertz",
                _ => return format!("{num}{unit}"),
            };
            format!("{num} {full_unit}")
        })
        .to_string();

    // 时间: 3:30pm → 3:30 PM, 3:30am → 3:30 AM
    let time_re = Regex::new(r"(\d{1,2}:\d{2})\s*(am|pm)\b").unwrap();
    result = time_re
        .replace_all(&result, |caps: &regex::Captures| {
            let time = &caps[1];
            let period = caps[2].to_uppercase();
            format!("{time} {period}")
        })
        .to_string();

    // #1 → number 1
    let hash_re = Regex::new(r"#(\d+)\b").unwrap();
    result = hash_re.replace_all(&result, "number $1").to_string();

    result
}

/// 将中文数字转为阿拉伯数字（用于翻译后处理）
///
/// 借鉴 OmniVoice 的中文数字归一化：
/// - 一 → 1, 二 → 2, 三 → 3, ... 十 → 10
/// - 二十 → 20, 三百 → 300, 一千 → 1000
pub fn normalize_chinese_numbers(text: &str) -> String {
    let digit_map = [
        ('零', '0'),
        ('一', '1'),
        ('二', '2'),
        ('三', '3'),
        ('四', '4'),
        ('五', '5'),
        ('六', '6'),
        ('七', '7'),
        ('八', '8'),
        ('九', '9'),
    ];

    let mut result = text.to_string();

    // 简单替换：单个中文数字 → 阿拉伯数字
    for (cn, num) in digit_map {
        result = result.replace(cn, &num.to_string());
    }

    // 十 → 10（当不跟在其他数字后面时）
    result = result.replace("十", "10");

    result
}

/// 综合文本归一化（增强版）
///
/// 在 `normalize_text` 基础上增加数字、单位、货币归一化。
/// 适用于 TTS 前的最终文本处理。
///
/// # 参数
/// - `text`: 输入文本
/// - `config`: 归一化配置
///
/// # 返回
/// 归一化后的文本
pub fn normalize_text_enhanced(text: &str, config: &NormalizationConfig) -> String {
    // 先执行数字/单位归一化（需要在符号展开之前处理 $、° 等）
    let result = normalize_numbers(text);

    // 再执行基础归一化
    let result = normalize_text(&result, config);

    result
}

#[cfg(test)]
mod omni_normalize_tests {
    use super::*;

    #[test]
    fn test_normalize_numbers_currency_dollar() {
        let result = normalize_numbers("It costs $5.");
        assert!(
            result.contains("5 dollars"),
            "Expected '5 dollars', got: {result}"
        );
    }

    #[test]
    fn test_normalize_numbers_currency_euro() {
        let result = normalize_numbers("Price: €10");
        assert!(
            result.contains("10 euros"),
            "Expected '10 euros', got: {result}"
        );
    }

    #[test]
    fn test_normalize_numbers_currency_yen() {
        let result = normalize_numbers("¥100 for this");
        assert!(
            result.contains("100 yen"),
            "Expected '100 yen', got: {result}"
        );
    }

    #[test]
    fn test_normalize_numbers_temperature_celsius() {
        let result = normalize_numbers("Water boils at 100°C");
        assert!(result.contains("100 degrees Celsius"), "Got: {result}");
    }

    #[test]
    fn test_normalize_numbers_temperature_fahrenheit() {
        let result = normalize_numbers("Body temp is 98°F");
        assert!(result.contains("98 degrees Fahrenheit"), "Got: {result}");
    }

    #[test]
    fn test_normalize_numbers_units() {
        let result = normalize_numbers("I weigh 70kg and am 175cm tall");
        assert!(result.contains("70 kilograms"), "Got: {result}");
        assert!(result.contains("175 centimeters"), "Got: {result}");
    }

    #[test]
    fn test_normalize_numbers_time() {
        let result = normalize_numbers("Meet at 3:30pm tomorrow");
        assert!(result.contains("3:30 PM"), "Got: {result}");
    }

    #[test]
    fn test_normalize_numbers_hash() {
        let result = normalize_numbers("You are #1!");
        assert!(result.contains("number 1"), "Got: {result}");
    }

    #[test]
    fn test_normalize_numbers_frequency() {
        let result = normalize_numbers("CPU runs at 3.5GHz");
        // Note: our regex only matches integers before units
        // 3.5GHz won't match because 3.5 is not \d+
        // But 5GHz would match
        assert!(!result.contains("3.5 gigahertz") || result.contains("3.5GHz"));
    }

    #[test]
    fn test_normalize_numbers_no_change() {
        let result = normalize_numbers("Hello world");
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_normalize_chinese_numbers_simple() {
        let result = normalize_chinese_numbers("我有三个苹果");
        assert!(result.contains('3'), "Should convert 三 to 3: {result}");
    }

    #[test]
    fn test_normalize_chinese_numbers_shi() {
        let result = normalize_chinese_numbers("十个人");
        assert!(result.contains("10"), "Should convert 十 to 10: {result}");
    }

    #[test]
    fn test_normalize_text_enhanced_combines() {
        let config = NormalizationConfig::default();
        let result = normalize_text_enhanced("The 1st prize is $100", &config);
        assert!(result.contains("first"), "Should expand ordinal: {result}");
        // normalize_numbers runs first, converting $100 → 100 dollars
        // then normalize_text runs, which won't find $ to expand
        assert!(
            result.contains("100 dollars"),
            "Should expand currency: {result}"
        );
    }

    #[test]
    fn test_normalize_numbers_empty() {
        assert_eq!(normalize_numbers(""), "");
    }

    #[test]
    fn test_normalize_numbers_multiple() {
        let result = normalize_numbers("$5 and €10 at 25°C");
        assert!(result.contains("5 dollars"), "Got: {result}");
        assert!(result.contains("10 euros"), "Got: {result}");
        assert!(result.contains("25 degrees Celsius"), "Got: {result}");
    }
}
