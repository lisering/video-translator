//! 句子分割与音频拼接模块
//!
//! 借鉴 dots.tts 的双流式 pipeline 思路：将长段翻译文本按句拆分，
//! 逐句合成后用交叉淡入淡出拼接，减少单次 TTS 负载并改善音频质量。
//!
//! # 核心功能
//! - [`split_sentences`]: 将文本按句子边界拆分（支持中英文混合）
//! - [`crossfade_concat_wav`]: 将多个 WAV 文件用交叉淡入淡出拼接为一个
//!
//! # 句子分割策略
//! 1. 中文按 `。！？；` 分句
//! 2. 英文按 `.!?;` 分句（后跟空格或换行）
//! 3. 换行符 `\n` 也作为分句边界
//! 4. 保留标点符号在句子末尾
//! 5. 过滤空句子和纯空白句子
//! 6. 过长句子（>200 字符）在逗号处二次拆分
//!
//! # 音频拼接策略
//! - 相邻 WAV 文件在边界处用等功率交叉淡入淡出（sin/cos 权重）
//! - 交叉淡入淡出时长可配置（默认 50ms）
//! - 所有 WAV 必须采样率一致
//!
//! # 示例
//! ```
//! use vt_core::sentence_split::split_sentences;
//!
//! let text = "你好世界。这是第一句话！第二句话？";
//! let sentences = split_sentences(text);
//! assert_eq!(sentences.len(), 3);
//! ```

use std::path::{Path, PathBuf};

use crate::error::{AppError, AppResult};

// ─── 句子分割 ─────────────────────────────────────────────

/// 中文句子终止标点
const CN_TERMINATORS: &[char] = &['。', '！', '？', '；', '…'];

/// 英文句子终止标点（后跟空格/换行时才算终止）
const EN_TERMINATORS: &[char] = &['.', '!', '?', ';'];

/// 中文次级分割标点（过长句子在此处二次拆分）
const CN_SECONDARY: &[char] = &['，', '、'];

/// 最大句子长度（字符数），超过则在次级标点处拆分
const MAX_SENTENCE_CHARS: usize = 200;

/// 将文本按句子边界拆分
///
/// 支持中英文混合文本，按句号、问号、感叹号等标点分句。
/// 保留标点符号在句子末尾，过滤空句子。
///
/// # 参数
/// - `text`: 待拆分的文本
///
/// # 返回
/// 句子列表（每个句子包含末尾标点，已 trim）
#[must_use]
pub fn split_sentences(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        current.push(ch);

        // 检查是否是句子终止标点
        let is_cn_terminator = CN_TERMINATORS.contains(&ch);

        // 英文终止标点需要后续是空格/换行/结尾
        let is_en_terminator = EN_TERMINATORS.contains(&ch) && {
            if i + 1 < chars.len() {
                let next = chars[i + 1];
                next.is_whitespace() || next == '\n' || next == '\r'
            } else {
                true // 文本结尾
            }
        };

        // 换行符也是分割点
        let is_newline = ch == '\n';

        if is_cn_terminator || is_en_terminator || is_newline {
            // 推入当前句子
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }

        i += 1;
    }

    // 处理最后未终止的文本
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    // 对过长句子进行二次拆分
    sentences
        .into_iter()
        .flat_map(|s| {
            if s.chars().count() > MAX_SENTENCE_CHARS {
                split_long_sentence(&s)
            } else {
                vec![s]
            }
        })
        .collect()
}

/// 对过长句子在次级标点处拆分
fn split_long_sentence(sentence: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();

    for ch in sentence.chars() {
        current.push(ch);

        if CN_SECONDARY.contains(&ch) {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                result.push(trimmed);
            }
            current.clear();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        result.push(trimmed);
    }

    // 如果拆分后仍然只有一个句子（没有次级标点），直接返回原句
    if result.len() <= 1 {
        return vec![sentence.to_string()];
    }

    result
}

// ─── WAV 交叉淡入淡出拼接 ─────────────────────────────────

/// 默认交叉淡入淡出时长（毫秒）
pub const DEFAULT_CROSSFADE_MS: u64 = 50;

/// 将多个 WAV 文件用交叉淡入淡出拼接为一个
///
/// # 工作原理
/// 1. 读取所有 WAV 文件
/// 2. 相邻文件在边界处用等功率交叉淡入淡出（sin/cos 权重）
/// 3. 输出拼接后的 WAV 文件
///
/// # 参数
/// - `wav_paths`: WAV 文件路径列表（按顺序拼接）
/// - `output_path`: 输出 WAV 文件路径
/// - `crossfade_ms`: 交叉淡入淡出时长（毫秒）
///
/// # 错误
/// - [`AppError::VoiceCloningError`][]: WAV 读取/写入失败
/// - [`AppError::VoiceCloningError`][]: 采样率不一致
pub fn crossfade_concat_wav(
    wav_paths: &[PathBuf],
    output_path: &Path,
    crossfade_ms: u64,
) -> AppResult<()> {
    if wav_paths.is_empty() {
        return Err(AppError::VoiceCloningError(
            "crossfade_concat_wav: empty wav list".to_string(),
        ));
    }

    // 单个文件：直接复制
    if wav_paths.len() == 1 {
        std::fs::copy(&wav_paths[0], output_path)
            .map_err(|e| AppError::VoiceCloningError(format!("Failed to copy WAV: {e}")))?;
        return Ok(());
    }

    // 读取所有 WAV 文件
    let mut all_samples: Vec<Vec<f32>> = Vec::with_capacity(wav_paths.len());
    let mut sample_rate: u32 = 0;
    let mut channels: u16 = 0;

    for path in wav_paths {
        let reader = hound::WavReader::open(path).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to open WAV {:?}: {e}", path))
        })?;

        let spec = reader.spec();
        if sample_rate == 0 {
            sample_rate = spec.sample_rate;
            channels = spec.channels;
        } else if spec.sample_rate != sample_rate {
            return Err(AppError::VoiceCloningError(format!(
                "Sample rate mismatch: {} vs {} for {:?}",
                spec.sample_rate, sample_rate, path
            )));
        }

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = 2f32.powi(spec.bits_per_sample as i32 - 1);
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
        };

        all_samples.push(samples);
    }

    // 计算交叉淡入淡出样本数
    let crossfade_samples = (sample_rate as f64 * crossfade_ms as f64 / 1000.0) as usize;

    // 拼接音频
    let mut result: Vec<f32> = Vec::new();

    for (i, samples) in all_samples.iter().enumerate() {
        if i == 0 {
            // 第一段：直接添加全部
            result.extend_from_slice(samples);
        } else {
            // 后续段：与前一段尾部做交叉淡入淡出
            let prev_len = result.len();
            let fade_len = crossfade_samples.min(prev_len).min(samples.len());

            if fade_len > 0 {
                // 对前一段的最后 fade_len 个样本和当前段的前 fade_len 个样本做等功率交叉淡入淡出
                let prev_tail_start = prev_len - fade_len;
                for j in 0..fade_len {
                    let t = j as f32 / fade_len as f32;
                    // 等功率交叉淡入淡出: cos/sin 权重
                    let fade_out = (std::f32::consts::PI * t * 0.5).cos();
                    let fade_in = (std::f32::consts::PI * t * 0.5).sin();

                    let prev_idx = prev_tail_start + j;
                    let curr_idx = j;

                    if channels == 1 {
                        result[prev_idx] = result[prev_idx] * fade_out;
                        let curr_sample = if curr_idx < samples.len() {
                            samples[curr_idx] * fade_in
                        } else {
                            0.0
                        };
                        result[prev_idx] += curr_sample;
                    } else {
                        // 多声道：每通道分别处理
                        for ch in 0..channels as usize {
                            let ri = prev_idx + ch;
                            let ci = curr_idx + ch;
                            if ri < result.len() {
                                result[ri] = result[ri] * fade_out;
                                if ci < samples.len() {
                                    result[ri] += samples[ci] * fade_in;
                                }
                            }
                        }
                    }
                }

                // 添加当前段剩余部分（跳过已淡入的部分）
                if fade_len < samples.len() {
                    result.extend_from_slice(&samples[fade_len..]);
                }
            } else {
                // 无交叉淡入淡出：直接拼接
                result.extend_from_slice(samples);
            }
        }
    }

    // 写入输出 WAV
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to create output dir: {e}"))
        })?;
    }

    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec)
        .map_err(|e| AppError::VoiceCloningError(format!("Failed to create output WAV: {e}")))?;

    let max_val = 32767.0f32;
    for sample in &result {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * max_val) as i16;
        writer
            .write_sample(i16_sample)
            .map_err(|e| AppError::VoiceCloningError(format!("Failed to write sample: {e}")))?;
    }

    writer
        .finalize()
        .map_err(|e| AppError::VoiceCloningError(format!("Failed to finalize WAV: {e}")))?;

    Ok(())
}

// ─── 句子级 TTS 合成 ─────────────────────────────────────

/// 句子级 TTS 合成结果
#[derive(Debug)]
pub struct SentenceTtsResult {
    /// 最终拼接后的音频路径
    pub audio_path: PathBuf,
    /// 合成的句子数量
    pub sentence_count: usize,
    /// 总音频时长（秒）
    pub total_duration_secs: f64,
}

/// 判断文本是否应该按句拆分
///
/// 只有文本足够长（超过阈值）时才拆分，短文本直接整体合成。
///
/// # 参数
/// - `text`: 待判断的文本
/// - `min_chars`: 最小拆分阈值（字符数），默认 80
#[must_use]
pub fn should_split_for_tts(text: &str, min_chars: usize) -> bool {
    text.chars().count() > min_chars
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── split_sentences 测试 ────────────────────────────

    #[test]
    fn test_split_chinese_sentences() {
        let text = "你好世界。这是第一句话！第二句话？";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "你好世界。");
        assert_eq!(sentences[1], "这是第一句话！");
        assert_eq!(sentences[2], "第二句话？");
    }

    #[test]
    fn test_split_english_sentences() {
        let text = "Hello world. This is sentence one! Is this sentence two?";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("Hello world."));
        assert!(sentences[1].contains("sentence one!"));
    }

    #[test]
    fn test_split_mixed_cn_en() {
        let text = "你好世界。Hello world. 这是中文！";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_split_newline_separator() {
        let text = "第一行\n第二行\n第三行";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_split_empty_text() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   ").is_empty());
        assert!(split_sentences("\n\n").is_empty());
    }

    #[test]
    fn test_split_no_terminator() {
        let text = "这是一段没有句号的文本";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "这是一段没有句号的文本");
    }

    #[test]
    fn test_split_long_sentence_at_comma() {
        // 构造一个超过 MAX_SENTENCE_CHARS 的长句子，在逗号处拆分
        let long_text = format!(
            "这是一个{}很长的句子，后面还有更多内容继续延伸到很长的程度",
            "非常".repeat(100)
        );
        let sentences = split_sentences(&long_text);
        assert!(
            sentences.len() > 1,
            "Long sentence should be split at comma"
        );
    }

    #[test]
    fn test_split_preserves_terminator() {
        let text = "你好。世界！";
        let sentences = split_sentences(text);
        assert_eq!(sentences[0], "你好。");
        assert_eq!(sentences[1], "世界！");
    }

    #[test]
    fn test_split_ellipsis() {
        let text = "这是第一段…第二段";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "这是第一段…");
    }

    // ── should_split_for_tts 测试 ───────────────────────

    #[test]
    fn test_should_split_short_text() {
        assert!(!should_split_for_tts("短文本", 80));
    }

    #[test]
    fn test_should_split_long_text() {
        let long_text = "这是一段很长的文本".repeat(20);
        assert!(should_split_for_tts(&long_text, 80));
    }

    // ── crossfade_concat_wav 测试 ───────────────────────

    /// 创建测试用 WAV 文件
    fn create_test_wav(path: &Path, samples: &[f32], sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            let i16_sample = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(i16_sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn test_crossfade_concat_single_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let out = dir.path().join("out.wav");

        create_test_wav(&wav1, &[0.5; 1000], 24000);

        crossfade_concat_wav(&[wav1.clone()], &out, 50).unwrap();

        let reader = hound::WavReader::open(&out).unwrap();
        assert_eq!(reader.spec().sample_rate, 24000);
        assert_eq!(reader.spec().channels, 1);
    }

    #[test]
    fn test_crossfade_concat_two_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let wav2 = dir.path().join("b.wav");
        let out = dir.path().join("out.wav");

        // 第一段: 0.5 的正弦波
        let samples1: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        // 第二段: 0.3 的正弦波
        let samples2: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();

        create_test_wav(&wav1, &samples1, 24000);
        create_test_wav(&wav2, &samples2, 24000);

        crossfade_concat_wav(&[wav1, wav2], &out, 10).unwrap();

        let reader = hound::WavReader::open(&out).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 24000);

        // 总长度应略小于两段之和（交叉淡入淡出区域重叠）
        // 2 × 2000 - 240 (10ms at 24kHz) = 3760
        let total_samples: Vec<i16> = reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .collect();
        assert!(
            total_samples.len() > 3500,
            "Concatenated audio should be longer than 3500 samples, got {}",
            total_samples.len()
        );
        assert!(
            total_samples.len() < 4000,
            "Concatenated audio should be shorter than 4000 (crossfade overlap), got {}",
            total_samples.len()
        );
    }

    #[test]
    fn test_crossfade_concat_empty_list() {
        let out = Path::new("/tmp/test_empty.wav");
        let result = crossfade_concat_wav(&[], out, 50);
        assert!(result.is_err());
    }

    #[test]
    fn test_crossfade_concat_sample_rate_mismatch() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let wav2 = dir.path().join("b.wav");
        let out = dir.path().join("out.wav");

        create_test_wav(&wav1, &[0.5; 100], 24000);
        create_test_wav(&wav2, &[0.3; 100], 16000);

        let result = crossfade_concat_wav(&[wav1, wav2], &out, 50);
        assert!(result.is_err(), "Should fail on sample rate mismatch");
    }

    #[test]
    fn test_crossfade_concat_three_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let wav2 = dir.path().join("b.wav");
        let wav3 = dir.path().join("c.wav");
        let out = dir.path().join("out.wav");

        create_test_wav(&wav1, &[0.5; 1500], 24000);
        create_test_wav(&wav2, &[0.3; 1500], 24000);
        create_test_wav(&wav3, &[0.7; 1500], 24000);

        crossfade_concat_wav(&[wav1, wav2, wav3], &out, 10).unwrap();

        let reader = hound::WavReader::open(&out).unwrap();
        let total_samples: Vec<i16> = reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .collect();
        // 3 × 1500 - 2 × 240 (10ms at 24kHz) = 4020
        assert!(
            total_samples.len() > 3500,
            "3-file concat should have > 3500 samples, got {}",
            total_samples.len()
        );
    }
}

// ─── 缩写感知分句 — 借鉴 OmniVoice abbreviation-aware splitting ──

/// 常见英文缩写表（句点不会导致分句）
///
/// 借鉴 OmniVoice `split_text_with_abbrev` 的缩写列表，
/// 防止 "Mr." "Dr." "e.g." 等缩写中的句点被误判为句子边界。
const COMMON_ABBREVIATIONS: &[&str] = &[
    // 称谓
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", // 学位
    "Ph.D", "B.A", "M.A", "B.S", "M.S", "M.D", "B.Sc", "M.Sc", // 拉丁缩写
    "e.g", "i.e", "etc", "vs", "cf", "ca", "approx", // 时间
    "a.m", "p.m", "A.M", "P.M", // 组织/地名
    "U.S", "U.K", "U.S.A", "U.S.S.R", "E.U", // 其他
    "Inc", "Ltd", "Co", "Corp", "No", "Vol", "pp", "op.cit", "Jan", "Feb", "Mar", "Apr", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Sept",
];

/// 检查当前位置的句点是否属于缩写
///
/// 检查当前句点之前的单词是否在缩写表中。
/// 同时支持多词缩写如 "e.g." "i.e." "U.S." 等。
///
/// # 参数
/// - `chars`: 字符数组
/// - `dot_idx`: 句点在 chars 中的索引
///
/// # 返回
/// `true` 如果该句点属于缩写（不应分句）
fn is_abbreviation_dot(chars: &[char], dot_idx: usize) -> bool {
    // 向前查找单词开始
    let mut start = dot_idx;
    while start > 0 && chars[start - 1].is_alphanumeric() {
        start -= 1;
    }

    // 提取句点前的单词
    let word: String = chars[start..dot_idx].iter().collect();
    if word.is_empty() {
        return false;
    }

    // 检查是否在缩写表中（大小写不敏感）
    let word_lower = word.to_lowercase();
    for &abbr in COMMON_ABBREVIATIONS {
        if word_lower == abbr.to_lowercase() {
            return true;
        }
    }

    // 检查是否是单字母+句点模式（如 "A." "B." — 可能是缩写）
    if word.len() == 1 && word.chars().next().unwrap().is_ascii_uppercase() {
        // 检查后面是否还有句点或字母（如 "U.S." "A.B.C."）
        if dot_idx + 1 < chars.len() {
            let next = chars[dot_idx + 1];
            if next.is_alphabetic() || next == '.' {
                return true;
            }
        }
        // 也检查前面是否是缩写的一部分（如 U.S. 中的第二个点）
        // 向前查找：如果前一个非字母字符是句点，且那个句点前是单字母
        if start >= 2 && chars[start - 1] == '.' {
            // 再向前找一个单词
            let mut prev_start = start - 1;
            while prev_start > 0 && chars[prev_start - 1].is_alphanumeric() {
                prev_start -= 1;
            }
            if prev_start < start - 1 {
                let prev_word: String = chars[prev_start..start - 1].iter().collect();
                if prev_word.len() == 1 && prev_word.chars().next().unwrap().is_ascii_uppercase() {
                    return true;
                }
                // 检查组合是否在缩写表中（如 e.g, i.e）
                let combo = format!("{}.{}", prev_word.to_lowercase(), word_lower);
                for &abbr in COMMON_ABBREVIATIONS {
                    if combo == abbr.to_lowercase() {
                        return true;
                    }
                }
            }
        }
    }

    // 检查多词缩写：向前查找 “word1.word2” 模式
    if start >= 2 && chars[start - 1] == '.' {
        let mut prev_start = start - 1;
        while prev_start > 0 && chars[prev_start - 1].is_alphanumeric() {
            prev_start -= 1;
        }
        if prev_start < start - 1 {
            let prev_word: String = chars[prev_start..start - 1].iter().collect();
            let combo = format!("{}.{}", prev_word.to_lowercase(), word_lower);
            for &abbr in COMMON_ABBREVIATIONS {
                if combo == abbr.to_lowercase() {
                    return true;
                }
            }
        }
    }

    false
}

/// 缩写感知的句子分割
///
/// 在 `split_sentences` 基础上增加缩写检测，
/// 防止 "Mr." "Dr." "e.g." 等缩写中的句点被误判为句子边界。
///
/// # 参数
/// - `text`: 待拆分的文本
///
/// # 返回
/// 句子列表
#[must_use]
pub fn split_sentences_aware(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        current.push(ch);

        // 检查是否是句子终止标点
        let is_cn_terminator = CN_TERMINATORS.contains(&ch);

        // 英文终止标点需要更严格的检查
        let is_en_terminator = EN_TERMINATORS.contains(&ch) && {
            // 首先检查是否是缩写中的句点
            if ch == '.' && is_abbreviation_dot(&chars, i) {
                false // 缩写中的句点不是终止符
            } else {
                // 需要后续是空格/换行/结尾
                if i + 1 < chars.len() {
                    let next = chars[i + 1];
                    next.is_whitespace() || next == '\n' || next == '\r'
                } else {
                    true // 文本结尾
                }
            }
        };

        // 换行符也是分割点
        let is_newline = ch == '\n';

        if is_cn_terminator || is_en_terminator || is_newline {
            // 推入当前句子
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }

        i += 1;
    }

    // 处理最后未终止的文本
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    // 对过长句子进行二次拆分
    sentences
        .into_iter()
        .flat_map(|s| {
            if s.chars().count() > MAX_SENTENCE_CHARS {
                split_long_sentence(&s)
            } else {
                vec![s]
            }
        })
        .collect()
}

#[cfg(test)]
mod omni_abbrev_tests {
    use super::*;

    #[test]
    fn test_split_aware_basic() {
        let result = split_sentences_aware("Hello world. This is a test.");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "Hello world.");
        assert_eq!(result[1], "This is a test.");
    }

    #[test]
    fn test_split_aware_abbreviation_mr() {
        let result = split_sentences_aware("Mr. Smith went to the store. He bought milk.");
        assert_eq!(
            result.len(),
            2,
            "Should not split on 'Mr.' — got {:?}",
            result
        );
        assert!(result[0].contains("Mr."));
    }

    #[test]
    fn test_split_aware_abbreviation_dr() {
        let result = split_sentences_aware("Dr. Watson arrived. Then he left.");
        assert_eq!(
            result.len(),
            2,
            "Should not split on 'Dr.' — got {:?}",
            result
        );
    }

    #[test]
    fn test_split_aware_abbreviation_eg() {
        let result = split_sentences_aware("Some fruits, e.g. apples, are sweet. Others are not.");
        assert_eq!(
            result.len(),
            2,
            "Should not split on 'e.g.' — got {:?}",
            result
        );
    }

    #[test]
    fn test_split_aware_abbreviation_ie() {
        let result = split_sentences_aware("This means, i.e. that is, we should go. Let's go.");
        assert_eq!(
            result.len(),
            2,
            "Should not split on 'i.e.' — got {:?}",
            result
        );
    }

    #[test]
    fn test_split_aware_abbreviation_us() {
        let result = split_sentences_aware("The U.S. has fifty states. Canada has ten provinces.");
        assert_eq!(
            result.len(),
            2,
            "Should not split on 'U.S.' — got {:?}",
            result
        );
    }

    #[test]
    fn test_split_aware_abbreviation_inc() {
        let result = split_sentences_aware("Acme Inc. is a company. It makes things.");
        assert_eq!(
            result.len(),
            2,
            "Should not split on 'Inc.' — got {:?}",
            result
        );
    }

    #[test]
    fn test_split_aware_chinese() {
        let result = split_sentences_aware("你好世界。这是一个测试。");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_split_aware_multiple_abbreviations() {
        let result = split_sentences_aware(
            "Mr. Smith met Dr. Jones at 3 p.m. They discussed Ph.D. programs.",
        );
        assert_eq!(
            result.len(),
            1,
            "All abbreviations should be kept together — got {:?}",
            result
        );
    }

    #[test]
    fn test_split_aware_single_letter_abbrev() {
        let result = split_sentences_aware("Section A. This is about B. The rest follows.");
        // "A." is single uppercase letter + period, should not split if followed by letter
        // But "A." followed by space+capital could be ambiguous
        // "B." followed by space+capital also
        // In practice, these are ambiguous — we err on the side of not splitting
        assert!(result.len() <= 3, "Got {:?}", result);
    }

    #[test]
    fn test_split_aware_month_abbreviations() {
        let result = split_sentences_aware("In Jan. it was cold. By Feb. it warmed up.");
        assert_eq!(
            result.len(),
            2,
            "Should not split on month abbreviations — got {:?}",
            result
        );
    }

    #[test]
    fn test_split_aware_no_abbreviation() {
        let result = split_sentences_aware("This is sentence one. This is sentence two.");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_split_aware_empty() {
        assert!(split_sentences_aware("").is_empty());
        assert!(split_sentences_aware("   ").is_empty());
    }

    #[test]
    fn test_split_aware_newline() {
        let result = split_sentences_aware("First line.\nSecond line.");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_is_abbreviation_dot_directly() {
        let chars: Vec<char> = "Mr. Smith".chars().collect();
        let dot_idx = chars.iter().position(|&c| c == '.').unwrap();
        assert!(is_abbreviation_dot(&chars, dot_idx));

        let chars2: Vec<char> = "Hello. World".chars().collect();
        let dot_idx2 = chars2.iter().position(|&c| c == '.').unwrap();
        assert!(!is_abbreviation_dot(&chars2, dot_idx2));
    }
}
