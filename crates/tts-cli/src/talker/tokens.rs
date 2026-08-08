//! 特殊 Token IDs — ChatML / TTS / Codec

/// ChatML 特殊 token IDs
pub mod special_tokens {
    pub const IM_START: u32 = 151644;
    pub const IM_END: u32 = 151645;
    pub const ASSISTANT: u32 = 77091;
    pub const NEWLINE: u32 = 198;
}

/// TTS 特殊 token IDs
pub mod tts_tokens {
    pub const TTS_PAD: u32 = 151671;
    pub const TTS_BOS: u32 = 151672;
    pub const TTS_EOS: u32 = 151673;
}

/// Codec 特殊 token IDs
pub mod codec_tokens {
    pub const CODEC_PAD: u32 = 2148;
    pub const CODEC_BOS: u32 = 2149;
    pub const CODEC_EOS: u32 = 2150;
    pub const CODEC_THINK: u32 = 2154;
    pub const CODEC_NOTHINK: u32 = 2155;
    pub const CODEC_THINK_BOS: u32 = 2156;
    pub const CODEC_THINK_EOS: u32 = 2157;
    pub const CODEC_VOCAB_SIZE: usize = 3072;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── 特殊 Token 常量 ───

    #[test]
    fn test_special_tokens_constants() {
        assert_eq!(special_tokens::IM_START, 151644);
        assert_eq!(special_tokens::IM_END, 151645);
        assert_ne!(special_tokens::IM_START, special_tokens::IM_END);
    }

    #[test]
    fn test_tts_tokens_constants() {
        assert_eq!(tts_tokens::TTS_PAD, 151671);
        assert_eq!(tts_tokens::TTS_BOS, 151672);
        assert_eq!(tts_tokens::TTS_EOS, 151673);
        assert!(tts_tokens::TTS_BOS < tts_tokens::TTS_EOS);
    }

    #[test]
    fn test_codec_tokens_constants() {
        assert_eq!(codec_tokens::CODEC_PAD, 2148);
        assert_eq!(codec_tokens::CODEC_BOS, 2149);
        assert_eq!(codec_tokens::CODEC_EOS, 2150);
        assert!(codec_tokens::CODEC_BOS < codec_tokens::CODEC_EOS);
        assert!(codec_tokens::CODEC_THINK != codec_tokens::CODEC_NOTHINK);
    }

    #[test]
    fn test_codec_vocab_size() {
        assert_eq!(codec_tokens::CODEC_VOCAB_SIZE, 3072);
    }
}
