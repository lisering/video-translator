//! Language 和 Speaker 类型定义

/// 语言 ID
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Japanese,
    Korean,
}

impl Language {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "chinese" | "zh" => Language::Chinese,
            "japanese" | "ja" => Language::Japanese,
            "korean" | "ko" => Language::Korean,
            _ => Language::English,
        }
    }

    pub fn token_id(&self) -> u32 {
        match self {
            Language::Chinese => 2055,
            Language::English => 2050,
            Language::Japanese => 2058,
            Language::Korean => 2064,
        }
    }

    /// 从文本内容自动检测语言
    ///
    /// 通过检查 CJK 字符的存在来判断语言。
    /// - 包含 Hiragana/Katakana → Japanese
    /// - 包含 CJK 汉字 → Chinese
    /// - 其他 → English
    pub fn detect_from_text(text: &str) -> Self {
        let mut has_cjk = false;
        let mut has_kana = false;
        for c in text.chars() {
            let cp = c as u32;
            // CJK Unified Ideographs + Extension A
            if (0x4E00..=0x9FFF).contains(&cp) || (0x3400..=0x4DBF).contains(&cp) {
                has_cjk = true;
            }
            // Hiragana
            if (0x3040..=0x309F).contains(&cp) {
                has_kana = true;
            }
            // Katakana
            if (0x30A0..=0x30FF).contains(&cp) {
                has_kana = true;
            }
        }
        if has_kana {
            Language::Japanese
        } else if has_cjk {
            Language::Chinese
        } else {
            Language::English
        }
    }
}

/// 预置说话人 (CustomVoice 模型)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Serena,
    Vivian,
    UncleFu,
    Ryan,
    Aiden,
    OnoAnna,
    Sohee,
    Eric,
    Dylan,
}

impl Speaker {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ryan" => Some(Speaker::Ryan),
            "serena" => Some(Speaker::Serena),
            "vivian" => Some(Speaker::Vivian),
            "aiden" => Some(Speaker::Aiden),
            "uncle_fu" | "unclefu" => Some(Speaker::UncleFu),
            "ono_anna" | "onoanna" => Some(Speaker::OnoAnna),
            "sohee" => Some(Speaker::Sohee),
            "eric" => Some(Speaker::Eric),
            "dylan" => Some(Speaker::Dylan),
            _ => None,
        }
    }

    pub fn token_id(&self) -> u32 {
        match self {
            Speaker::Serena => 3066,
            Speaker::Vivian => 3065,
            Speaker::UncleFu => 3010,
            Speaker::Ryan => 3061,
            Speaker::Aiden => 2861,
            Speaker::OnoAnna => 2873,
            Speaker::Sohee => 2864,
            Speaker::Eric => 2875,
            Speaker::Dylan => 2878,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Language 检测 ───

    #[test]
    fn test_language_from_str() {
        assert_eq!(Language::from_str("chinese"), Language::Chinese);
        assert_eq!(Language::from_str("zh"), Language::Chinese);
        assert_eq!(Language::from_str("ZH"), Language::Chinese);
        assert_eq!(Language::from_str("Chinese"), Language::Chinese);
        assert_eq!(Language::from_str("english"), Language::English);
        assert_eq!(Language::from_str("en"), Language::English);
        assert_eq!(Language::from_str("japanese"), Language::Japanese);
        assert_eq!(Language::from_str("ja"), Language::Japanese);
        assert_eq!(Language::from_str("korean"), Language::Korean);
        assert_eq!(Language::from_str("ko"), Language::Korean);
        assert_eq!(Language::from_str("unknown"), Language::English);
        assert_eq!(Language::from_str(""), Language::English);
    }

    #[test]
    fn test_language_token_id() {
        assert_eq!(Language::Chinese.token_id(), 2055);
        assert_eq!(Language::English.token_id(), 2050);
        assert_eq!(Language::Japanese.token_id(), 2058);
        assert_eq!(Language::Korean.token_id(), 2064);
    }

    #[test]
    fn test_detect_from_text_chinese() {
        assert_eq!(Language::detect_from_text("你好世界"), Language::Chinese);
        assert_eq!(Language::detect_from_text("Hello 世界"), Language::Chinese);
        assert_eq!(Language::detect_from_text("测试"), Language::Chinese);
    }

    #[test]
    fn test_detect_from_text_japanese() {
        assert_eq!(Language::detect_from_text("こんにちは"), Language::Japanese);
        assert_eq!(Language::detect_from_text("カタカナ"), Language::Japanese);
        assert_eq!(Language::detect_from_text("Hello です"), Language::Japanese);
    }

    #[test]
    fn test_detect_from_text_english() {
        assert_eq!(Language::detect_from_text("Hello world"), Language::English);
        assert_eq!(Language::detect_from_text("123 abc"), Language::English);
        assert_eq!(Language::detect_from_text(""), Language::English);
    }

    #[test]
    fn test_detect_from_text_mixed_cjk_kana() {
        // 日文假名优先级高于 CJK 汉字
        assert_eq!(
            Language::detect_from_text("日本語テスト"),
            Language::Japanese
        );
    }

    #[test]
    fn test_detect_from_text_korean() {
        // 韩文目前回退到 English（未实现韩文检测）
        assert_eq!(Language::detect_from_text("안녕하세요"), Language::English);
    }

    // ─── Speaker 解析 ───

    #[test]
    fn test_speaker_from_str() {
        assert_eq!(Speaker::from_str("ryan"), Some(Speaker::Ryan));
        assert_eq!(Speaker::from_str("Ryan"), Some(Speaker::Ryan));
        assert_eq!(Speaker::from_str("RYAN"), Some(Speaker::Ryan));
        assert_eq!(Speaker::from_str("serena"), Some(Speaker::Serena));
        assert_eq!(Speaker::from_str("vivian"), Some(Speaker::Vivian));
        assert_eq!(Speaker::from_str("uncle_fu"), Some(Speaker::UncleFu));
        assert_eq!(Speaker::from_str("unclefu"), Some(Speaker::UncleFu));
        assert_eq!(Speaker::from_str("ono_anna"), Some(Speaker::OnoAnna));
        assert_eq!(Speaker::from_str("onoanna"), Some(Speaker::OnoAnna));
        assert_eq!(Speaker::from_str("sohee"), Some(Speaker::Sohee));
        assert_eq!(Speaker::from_str("eric"), Some(Speaker::Eric));
        assert_eq!(Speaker::from_str("dylan"), Some(Speaker::Dylan));
        assert_eq!(Speaker::from_str("unknown"), None);
        assert_eq!(Speaker::from_str(""), None);
    }

    #[test]
    fn test_speaker_token_id_unique() {
        let speakers = [
            Speaker::Serena,
            Speaker::Vivian,
            Speaker::UncleFu,
            Speaker::Ryan,
            Speaker::Aiden,
            Speaker::OnoAnna,
            Speaker::Sohee,
            Speaker::Eric,
            Speaker::Dylan,
        ];
        let ids: Vec<u32> = speakers.iter().map(|s| s.token_id()).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            speakers.len(),
            "Speaker token IDs must be unique"
        );
    }
}
