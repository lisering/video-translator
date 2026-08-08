//! 属性测试 (Property-based testing)
//!
//! 使用 proptest 框架自动生成大量测试输入，覆盖边界条件和随机组合。
//! 测试不依赖 Metal 设备或模型文件，纯 CPU 逻辑。

use proptest::prelude::*;
use vt_tts::audio::{extract_mel_spectrogram, AudioBuffer, MelConfig};
use vt_tts::config::TtsEngineConfig;
use vt_tts::talker::{parse_quantize, Language, Speaker};
use vt_tts::tokenizer::TextTokenizer;

// ─── Tokenizer Fallback 往返测试 ───────────────────────────

proptest! {
    /// 任意 ASCII 文本经过 fallback tokenizer 编码→解码后应保持不变
    #[test]
    fn prop_fallback_tokenizer_roundtrip(text in "[\\x00-\\x7F]{0,100}") {
        let tok = TextTokenizer::fallback();
        let ids = tok.encode(&text).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        prop_assert_eq!(decoded, text);
    }

    /// Fallback tokenizer 的编码长度应等于字节数
    #[test]
    fn prop_fallback_encode_length(text in "[\\x00-\\x7F]{0,50}") {
        let tok = TextTokenizer::fallback();
        let ids = tok.encode(&text).unwrap();
        prop_assert_eq!(ids.len(), text.len());
    }

    /// 多语言文本往返测试
    #[test]
    fn prop_fallback_unicode_roundtrip(text in "[\\p{L}\\p{N}\\p{P}\\p{S} ]{0,30}") {
        let tok = TextTokenizer::fallback();
        let ids = tok.encode(&text).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        prop_assert_eq!(decoded, text);
    }
}

// ─── Language 检测属性 ─────────────────────────────────────

proptest! {
    /// 包含中文字符的文本应检测为 Chinese 或 Japanese
    #[test]
    fn prop_detect_cjk_is_cjk_or_japanese(
        prefix in "[a-zA-Z0-9 ]{0,20}",
        cjk in "[\u{4E00}-\u{9FFF}]{1,5}",
        suffix in "[a-zA-Z0-9 ]{0,20}"
    ) {
        let text = format!("{}{}{}", prefix, cjk, suffix);
        let lang = Language::detect_from_text(&text);
        prop_assert!(
            lang == Language::Chinese || lang == Language::Japanese,
            "CJK text should be Chinese or Japanese, got {:?} for '{}'",
            lang, text
        );
    }

    /// 纯英文/数字文本应检测为 English
    #[test]
    fn prop_detect_ascii_is_english(text in "[a-zA-Z0-9 .,!?'-]{1,50}") {
        let lang = Language::detect_from_text(&text);
        prop_assert_eq!(lang, Language::English);
    }
}

// ─── Speaker 解析属性 ──────────────────────────────────────

proptest! {
    /// 所有已知 Speaker 名称应该能被正确解析（小写）
    #[test]
    fn prop_speaker_from_str_valid(
        name in prop::sample::select(vec![
            "serena", "vivian", "uncle_fu", "unclefu", "ryan", "aiden",
            "ono_anna", "onoanna", "sohee", "eric", "dylan"
        ])
    ) {
        let speaker = Speaker::from_str(name);
        prop_assert!(speaker.is_some(), "Valid speaker name '{}' should parse", name);
    }

    /// 大写 Speaker 名称也应该能被正确解析
    #[test]
    fn prop_speaker_from_str_uppercase(
        name in prop::sample::select(vec![
            "SERENA", "VIVIAN", "UNCLE_FU", "UNCLEFU", "RYAN", "AIDEN",
            "ONO_ANNA", "ONOANNA", "SOHEE", "ERIC", "DYLAN"
        ])
    ) {
        let speaker = Speaker::from_str(name);
        prop_assert!(speaker.is_some(), "Uppercase speaker name '{}' should parse", name);
    }

    /// 未知字符串不应解析为 Speaker
    #[test]
    fn prop_speaker_from_str_invalid(name in "[a-z]{1,10}") {
        prop_assume!(!matches!(
            name.as_str(),
            "serena" | "vivian" | "unclefu" | "uncle_fu" |
            "ryan" | "aiden" | "onoanna" | "ono_anna" |
            "sohee" | "eric" | "dylan"
        ));
        let speaker = Speaker::from_str(&name);
        prop_assert!(speaker.is_none(), "Invalid speaker name '{}' should return None", name);
    }
}

// ─── Config 序列化属性 ─────────────────────────────────────

proptest! {
    /// Config 序列化→反序列化应保持一致
    #[test]
    fn prop_config_serde_roundtrip(
        device in prop::sample::select(vec!["cpu", "metal", "cuda"]),
        temperature in 0.0f32..2.0,
        top_k in 1usize..100,
        repetition_penalty in 1.0f32..2.0,
        no_repeat_ngram_size in 0usize..5,
        seed in 0u64..u64::MAX,
        max_codes in 10usize..1000,
        sr_multiple in 8u32..48,
        language in prop::sample::select(vec!["auto", "chinese", "english", "japanese", "korean"]),
        mixed_precision in proptest::bool::ANY,
        quantize in proptest::option::of(prop::sample::select(vec![
            "q8_0", "q4_0", "q4k", "q6k", "q5_0", "q5k", "none"
        ])),
        decode_device in proptest::option::of(prop::sample::select(vec!["cpu", "metal", "cuda"]))
    ) {
        let output_sample_rate = sr_multiple * 1000;
        let cfg = TtsEngineConfig {
            model_dir: "/tmp/model".into(),
            device: device.to_string(),
            temperature,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            seed: Some(seed),
            max_codes,
            output_sample_rate,
            language: language.to_string(),
            mixed_precision,
            quantize: quantize.map(|s| s.to_string()),
            decode_device: decode_device.map(|s| s.to_string()),
        };

        let json = serde_json::to_string(&cfg).unwrap();
        let de: TtsEngineConfig = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(de.device, device);
        prop_assert!((de.temperature - temperature).abs() < 1e-6);
        prop_assert_eq!(de.top_k, top_k);
        prop_assert!((de.repetition_penalty - repetition_penalty).abs() < 1e-6);
        prop_assert_eq!(de.no_repeat_ngram_size, no_repeat_ngram_size);
        prop_assert_eq!(de.seed, Some(seed));
        prop_assert_eq!(de.max_codes, max_codes);
        prop_assert_eq!(de.output_sample_rate, output_sample_rate);
        prop_assert_eq!(de.language, language);
        prop_assert_eq!(de.mixed_precision, mixed_precision);
        prop_assert_eq!(de.quantize.as_deref(), quantize);
        prop_assert_eq!(de.decode_device.as_deref(), decode_device);
    }
}

// ─── parse_quantize 属性 ───────────────────────────────────

proptest! {
    /// 量化格式字符串解析应与预期一致
    #[test]
    fn prop_parse_quantize_valid(
        s in prop::sample::select(vec!["q8_0", "q4_0", "q4k", "q6k", "q5_0", "q5k", "none"])
    ) {
        let result = parse_quantize(&Some(s.to_string()));
        if s == "none" {
            prop_assert!(result.is_none());
        } else {
            prop_assert!(result.is_some(), "Valid quantize '{}' should parse", s);
        }
    }

    /// 大小写不敏感
    #[test]
    fn prop_parse_quantize_case_insensitive(
        s in prop::sample::select(vec!["Q8_0", "Q4_0", "Q4K", "Q6K", "Q5_0", "Q5K"])
    ) {
        let result = parse_quantize(&Some(s.to_string()));
        prop_assert!(result.is_some(), "Quantize '{}' should parse case-insensitively", s);
    }
}

// ─── Audio 重采样属性 ──────────────────────────────────────

proptest! {
    /// 相同采样率的重采样应返回原数据
    #[test]
    fn prop_resample_same_rate_noop(
        n in 1usize..500,
        sr_multiple in 8u32..48
    ) {
        let sr = sr_multiple * 1000;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let buf = AudioBuffer::from_samples(samples.clone(), sr);
        let resampled = buf.resample_linear(sr);
        prop_assert_eq!(resampled.samples, samples);
        prop_assert_eq!(resampled.sample_rate, sr);
    }

    /// 上采样后样本数应增加
    #[test]
    fn prop_resample_upsample_increases_length(
        n in 10usize..200,
        src_sr in 8000u32..16000,
        dst_sr in 16001u32..48000
    ) {
        let samples = vec![0.0f32; n];
        let buf = AudioBuffer::from_samples(samples, src_sr);
        let resampled = buf.resample_linear(dst_sr);
        prop_assert!(
            resampled.samples.len() >= n,
            "Upsampling should produce >= {} samples, got {}",
            n, resampled.samples.len()
        );
    }

    /// 重采样后所有样本应在合理范围内
    #[test]
    fn prop_resample_samples_in_range(
        n in 10usize..200,
        src_sr in 8000u32..24000,
        dst_sr in 8000u32..48000
    ) {
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.8).collect();
        let buf = AudioBuffer::from_samples(samples, src_sr);
        let resampled = buf.resample_linear(dst_sr);
        for &s in &resampled.samples {
            prop_assert!(s >= -1.0 && s <= 1.0, "Resampled sample {} out of range", s);
        }
    }
}

// ─── Mel 频谱图属性 ────────────────────────────────────────

proptest! {
    /// Mel 频谱图的行数应等于 n_mels
    #[test]
    fn prop_mel_spectrogram_n_mels(n_samples in 512usize..8000) {
        let cfg = MelConfig::speaker_encoder();
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let mel = extract_mel_spectrogram(&samples, &cfg);
        prop_assert_eq!(mel.len(), cfg.n_mels);
    }

    /// Mel 频谱图的所有值应有限
    #[test]
    fn prop_mel_spectrogram_finite(n_samples in 512usize..4000) {
        let cfg = MelConfig::speaker_encoder();
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (i as f32 * 0.05).sin() * 0.5)
            .collect();
        let mel = extract_mel_spectrogram(&samples, &cfg);
        for row in &mel {
            for &val in row {
                prop_assert!(val.is_finite(), "Mel value {} should be finite", val);
            }
        }
    }
}
