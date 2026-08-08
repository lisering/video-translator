//! TTS 语音控制参数 — 借鉴 MOSS-TTS 的结构化 prompt 方案
//!
//! MOSS-TTS 支持 language/tokens/quality 等控制参数，
//! 此模块定义这些参数在 video-translator 中的传递方式。
//!
//! 对应 MOSS-TTS 项目:
//! - `moss_tts_delay/processing_moss_tts.py` 中的 `UserMessage` 参数
//! - `moss_tts_local_v1.5/streaming.py` 中的语言标签映射
//!
//! # 语言标签
//! MOSS-TTS 支持 31 种语言标签，传递给 TTS 可以确保正确发音。
//! 视频翻译中，翻译目标语言即为 TTS 语言标签。
//!
//! # Token 时长控制
//! MOSS-TTS 通过 `tokens` 参数控制生成音频长度。
//! 每秒约 12.5 个音频帧（24kHz / 80 下采样率）。
//! 可根据原始视频段时长计算所需 token 数量。

use serde::{Deserialize, Serialize};

/// 多语言 token 速率（每字符生成的音频 token 数）
///
/// 借鉴 MOSS-TTS `moss_tts_local_v1.5/streaming.py` 的 `TEXT_TO_AUDIO_TOKENS_PER_CHAR` 表
pub fn tokens_per_char(language: &str) -> f64 {
    let lang = language.to_lowercase();
    match lang.as_str() {
        "zh" | "cmn" | "chinese" | "yue" | "cantonese" => 3.098,
        "en" | "english" => 0.867,
        "fr" | "french" => 0.9,
        "ja" | "japanese" => 2.2,
        "ko" | "korean" => 1.8,
        "de" | "german" => 0.9,
        "es" | "spanish" => 0.9,
        "it" | "italian" => 0.9,
        "pt" | "portuguese" => 0.9,
        "ru" | "russian" => 1.0,
        "ar" | "arabic" => 1.0,
        "th" | "thai" => 1.5,
        "vi" | "vietnamese" => 1.2,
        "id" | "indonesian" => 0.9,
        "tr" | "turkish" => 0.9,
        "nl" | "dutch" => 0.9,
        "pl" | "polish" => 0.9,
        "uk" | "ukrainian" => 1.0,
        "cs" | "czech" | "da" | "danish" | "fi" | "finnish" | "el" | "greek" | "he" | "hebrew"
        | "hu" | "hungarian" | "no" | "norwegian" | "ro" | "romanian" | "sk" | "slovak" | "sv"
        | "swedish" => 0.9,
        _ => 1.0, // 默认
    }
}

/// 根据目标时长估算所需 token 数
///
/// MOSS-TTS 中 1 秒 ≈ 12.5 音频帧（帧率 = 24000Hz / 80 = 300 帧/秒... 不对，
/// 实际帧率取决于下采样率。MOSS-TTS 24kHz / 1920 = 12.5 frames/sec。
/// 实际值 = sample_rate / hop_length = 24000 / 1920 = 12.5
///
/// # 参数
/// - `duration_secs`: 目标音频时长（秒）
/// - `sample_rate`: 采样率（默认 24000）
/// - `hop_length`: 帧步长（默认 1920）
pub fn estimate_tokens_for_duration(
    duration_secs: f64,
    sample_rate: u32,
    hop_length: u32,
) -> usize {
    let frames_per_sec = sample_rate as f64 / hop_length as f64;
    (duration_secs * frames_per_sec) as usize
}

/// 根据文本长度估算生成音频时长
///
/// # 参数
/// - `text`: 要合成的文本
/// - `language`: 语言标签
pub fn estimate_audio_duration(text: &str, language: &str) -> f64 {
    let char_count = text.chars().count() as f64;
    let tpc = tokens_per_char(language);
    let total_tokens = char_count * tpc;
    // 帧率 12.5 fps
    total_tokens / 12.5
}

/// TTS 语音控制参数
///
/// 借鉴 MOSS-TTS 的结构化 prompt 格式，将语言标签、token 数等
/// 控制参数传递给 TTS 引擎（Python subprocess 或未来 Rust 原生实现）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsControlParams {
    /// 目标语言标签（如 "zh", "en", "ja"）
    #[serde(default)]
    pub language: Option<String>,

    /// 期望的音频时长（秒），用于 token 数控制
    #[serde(default)]
    pub target_duration_secs: Option<f64>,

    /// 质量标签
    #[serde(default)]
    pub quality: Option<String>,

    /// 指令文本
    #[serde(default)]
    pub instruction: Option<String>,

    /// 最大 token 数
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

impl Default for TtsControlParams {
    fn default() -> Self {
        Self {
            language: Some("zh".to_string()),
            target_duration_secs: None,
            quality: None,
            instruction: None,
            max_tokens: None,
        }
    }
}

impl TtsControlParams {
    /// 从视频段信息构建控制参数
    ///
    /// 根据原始视频段时长计算 token 数，
    /// 确保生成的 TTS 音频与原始视频时长匹配。
    pub fn from_segment(text: &str, target_language: &str, segment_duration: f64) -> Self {
        let estimated_duration = estimate_audio_duration(text, target_language);
        let tokens = if estimated_duration > segment_duration * 1.1 {
            // 文本太长，需要 token 控制
            Some(estimate_tokens_for_duration(segment_duration, 24000, 1920))
        } else {
            None
        };

        Self {
            language: Some(target_language.to_string()),
            target_duration_secs: Some(segment_duration),
            quality: None,
            instruction: None,
            max_tokens: tokens,
        }
    }

    /// 序列化为 JSON 对象（传递给 Python TTS server）
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "language": self.language,
            "target_duration_secs": self.target_duration_secs,
            "quality": self.quality,
            "instruction": self.instruction,
            "max_tokens": self.max_tokens,
        })
    }
}

/// 将语言代码映射为 MOSS-TTS 语言标签
pub fn to_moss_language_tag(lang: &str) -> &str {
    match lang.to_lowercase().as_str() {
        "zh" | "chinese" | "cmn" => "Chinese",
        "yue" | "cantonese" => "Cantonese",
        "en" | "english" => "English",
        "ja" | "japanese" => "Japanese",
        "ko" | "korean" => "Korean",
        "fr" | "french" => "French",
        "de" | "german" => "German",
        "es" | "spanish" => "Spanish",
        "it" | "italian" => "Italian",
        "pt" | "portuguese" => "Portuguese",
        "ru" | "russian" => "Russian",
        "ar" | "arabic" => "Arabic",
        "th" | "thai" => "Thai",
        "vi" | "vietnamese" => "Vietnamese",
        "id" | "indonesian" => "Indonesian",
        "tr" | "turkish" => "Turkish",
        "nl" | "dutch" => "Dutch",
        "pl" | "polish" => "Polish",
        "uk" | "ukrainian" => "Ukrainian",
        "cs" | "czech" => "Czech",
        "da" | "danish" => "Danish",
        "fi" | "finnish" => "Finnish",
        "el" | "greek" => "Greek",
        "he" | "hebrew" => "Hebrew",
        "hu" | "hungarian" => "Hungarian",
        "no" | "norwegian" => "Norwegian",
        "ro" | "romanian" => "Romanian",
        "sk" | "slovak" => "Slovak",
        "sv" | "swedish" => "Swedish",
        _ => "Chinese", // 默认中文
    }
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_per_char_chinese() {
        let tpc = tokens_per_char("zh");
        assert!((tpc - 3.098).abs() < 0.01);
    }

    #[test]
    fn test_tokens_per_char_english() {
        let tpc = tokens_per_char("en");
        assert!((tpc - 0.867).abs() < 0.01);
    }

    #[test]
    fn test_tokens_per_char_japanese() {
        let tpc = tokens_per_char("ja");
        assert!((tpc - 2.2).abs() < 0.01);
    }

    #[test]
    fn test_tokens_per_char_unknown() {
        let tpc = tokens_per_char("unknown");
        assert!((tpc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_tokens_for_duration() {
        // 5 seconds at 12.5 fps = 62.5 → 62 tokens
        let tokens = estimate_tokens_for_duration(5.0, 24000, 1920);
        assert_eq!(tokens, 62);
    }

    #[test]
    fn test_estimate_audio_duration_chinese() {
        // 100 Chinese chars → 100 * 3.098 = 309.8 tokens / 12.5 = 24.78 seconds
        let dur = estimate_audio_duration("你好世界".repeat(25).as_str(), "zh");
        assert!(dur > 20.0 && dur < 30.0);
    }

    #[test]
    fn test_estimate_audio_duration_english() {
        // 100 English chars → 100 * 0.867 = 86.7 tokens / 12.5 = 6.94 seconds
        let dur = estimate_audio_duration(&"a".repeat(100), "en");
        assert!(dur > 5.0 && dur < 8.0);
    }

    #[test]
    fn test_moss_language_tag() {
        assert_eq!(to_moss_language_tag("zh"), "Chinese");
        assert_eq!(to_moss_language_tag("en"), "English");
        assert_eq!(to_moss_language_tag("ja"), "Japanese");
        assert_eq!(to_moss_language_tag("unknown"), "Chinese");
    }

    #[test]
    fn test_tts_control_params_default() {
        let params = TtsControlParams::default();
        assert_eq!(params.language.as_deref(), Some("zh"));
    }

    #[test]
    fn test_tts_control_params_from_segment() {
        let params = TtsControlParams::from_segment("这是一段测试文本", "zh", 5.0);
        assert_eq!(params.language.as_deref(), Some("zh"));
        assert!(params.target_duration_secs.is_some());
    }

    #[test]
    fn test_tts_control_params_json() {
        let params = TtsControlParams {
            language: Some("zh".to_string()),
            target_duration_secs: Some(5.0),
            quality: Some("high".to_string()),
            instruction: None,
            max_tokens: Some(62),
        };
        let json = params.to_json();
        assert_eq!(json["language"], "zh");
        assert_eq!(json["target_duration_secs"], 5.0);
        assert_eq!(json["max_tokens"], 62);
    }
}
