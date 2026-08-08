//! 生成 prompt 构建 + 输出解析 — 从 MOSS-TTS `processor.py` 移植
//!
//! 构建多通道 input_ids (S, 33) 用于 TTS 生成，
//! 以及解析生成输出为文本 + 音频码。
//!
//! 对应 MOSS-TTS 项目 `moss_tts_delay/llama_cpp/processor.py`。

use super::constants::MossTtsConstants;
use super::delay_state::{apply_delay_pattern, extract_audio_segments};
use crate::error::{AppError, AppResult};

/// 音频 placeholder 字符串
const AUDIO_PLACEHOLDER: &str = "<|audio|>";

/// MOSS-TTS tokenizer 封装
///
/// 使用 BPE tokenizer 编码/解码文本。
/// 优先使用 HuggingFace `tokenizers` 库（通过 subprocess 或 FFI）。
pub struct MossTtsTokenizer {
    /// tokenizer 路径
    pub tokenizer_dir: String,
    /// 特殊 token 字符串映射
    pub special_tokens: SpecialTokens,
}

/// 特殊 token 字符串
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub im_start: String,
    pub im_end: String,
    pub audio_start: String,
    pub audio_end: String,
    pub audio_user_slot: String,
    pub audio_assistant_gen_slot: String,
    pub audio_assistant_delay_slot: String,
}

impl SpecialTokens {
    /// 从常量获取 token 字符串
    ///
    /// 注意：这些字符串需要从 tokenizer 的 vocab 中查找。
    /// 这里使用 MOSS-TTS 的标准 token 字符串。
    pub fn from_constants(_consts: &MossTtsConstants) -> Self {
        // MOSS-TTS 使用的标准特殊 token 字符串
        // 这些与 Qwen3 tokenizer 配置一致
        Self {
            im_start: "<|im_start|>".to_string(),
            im_end: "<|im_end|>".to_string(),
            audio_start: "<|audio_start|>".to_string(),
            audio_end: "<|audio_end|>".to_string(),
            audio_user_slot: "<|audio_user|>".to_string(),
            audio_assistant_gen_slot: "<|audio_assistant_gen|>".to_string(),
            audio_assistant_delay_slot: "<|audio_assistant_delay|>".to_string(),
        }
    }
}

/// 生成 prompt 构建参数
#[derive(Debug, Clone)]
pub struct GenerationPromptParams<'a> {
    /// 要合成的文本
    pub text: &'a str,
    /// 参考音频码（可选，用于声音克隆）
    pub reference_codes: Option<&'a [Vec<i64>]>,
    /// 指令（可选）
    pub instruction: Option<&'a str>,
    /// token 数量控制（可选）
    pub tokens: Option<usize>,
    /// 质量标签（可选）
    pub quality: Option<&'a str>,
    /// 语言标签（可选）
    pub language: Option<&'a str>,
}

/// 构建生成的多通道 input_ids
///
/// 返回: [S, 1+n_vq] int64
///
/// 格式与 MOSS-TTS 的 `build_generation_prompt()` 一致。
pub fn build_generation_prompt(
    tokenizer: &MossTtsTokenizer,
    params: GenerationPromptParams,
    consts: &MossTtsConstants,
) -> AppResult<Vec<Vec<i64>>> {
    let special = &tokenizer.special_tokens;
    let n_vq = consts.n_vq;
    let pad_code = consts.audio_pad_code as i64;

    // 构建 user_inst 内容
    let has_ref = params
        .reference_codes
        .is_some_and(|codes| !codes.is_empty());
    let ref_str = if has_ref {
        format!("[S1]:\n{AUDIO_PLACEHOLDER}")
    } else {
        "None".to_string()
    };

    let user_content = format!(
        "<user_inst>\n\
         - Reference(s):\n{ref_str}\n\
         - Instruction:\n{}\n\
         - Tokens:\n{}\n\
         - Quality:\n{}\n\
         - Sound Event:\n{}\n\
         - Ambient Sound:\n{}\n\
         - Language:\n{}\n\
         - Text:\n{}\n\
         </user_inst>",
        params.instruction.unwrap_or("None"),
        params
            .tokens
            .map(|t| t.to_string())
            .unwrap_or_else(|| "None".to_string()),
        params.quality.unwrap_or("None"),
        "None", // sound_event
        "None", // ambient_sound
        params.language.unwrap_or("None"),
        params.text,
    );

    // 替换 audio placeholder 为特殊 token 序列
    let user_content = replace_audio_placeholders(
        &user_content,
        if has_ref {
            vec![params.reference_codes.unwrap().len()]
        } else {
            vec![]
        },
        n_vq,
        &special.audio_user_slot,
        &special.audio_start,
        &special.audio_end,
    );

    // 完整 prompt: im_start + user + content + im_end + im_start + assistant
    let full_text = format!(
        "{}user\n{}{}\n{}assistant\n{}",
        special.im_start, user_content, special.im_end, special.im_start, special.audio_start,
    );

    // 编码文本为 token ID
    let text_ids = tokenizer.encode(&full_text)?;

    // 构建 multi-channel input_ids
    let ref_codes_list: Vec<&[Vec<i64>]> = if has_ref {
        vec![params.reference_codes.unwrap()]
    } else {
        vec![]
    };

    let unified = build_unified_codes(&text_ids, &ref_codes_list, n_vq, pad_code, consts);

    // 追加 assistant gen slot
    let gen_ids = tokenizer.encode(&special.audio_start)?;
    let mut gen_multi = vec![vec![pad_code; 1 + n_vq]; gen_ids.len()];
    for (i, &tid) in gen_ids.iter().enumerate() {
        gen_multi[i][0] = tid as i64;
    }

    // 拼接
    let mut result = unified;
    result.extend(gen_multi);

    Ok(result)
}

/// 解析生成输出
///
/// 返回: (text, audio_codes [T, n_vq])
pub fn parse_generation_output(
    tokenizer: &MossTtsTokenizer,
    generation_ids: &[Vec<i64>],
    prompt_len: usize,
    consts: &MossTtsConstants,
) -> AppResult<(String, Vec<Vec<i64>>)> {
    if generation_ids.is_empty() || prompt_len >= generation_ids.len() {
        return Ok((String::new(), vec![]));
    }

    let gen_part = &generation_ids[prompt_len..];
    let text_channel: Vec<i64> = gen_part.iter().map(|row| row[0]).collect();
    let audio_channels: Vec<Vec<i64>> = gen_part.iter().map(|row| row[1..].to_vec()).collect();

    // 解码文本
    let text = tokenizer.decode(&text_channel);

    // 提取音频段
    let segments = extract_audio_segments(&audio_channels, consts.audio_pad_code as i64);

    let audio_codes = if segments.is_empty() {
        vec![]
    } else {
        // 拼接所有段
        let mut combined = vec![];
        for seg in segments {
            combined.extend(seg);
        }
        combined
    };

    Ok((text, audio_codes))
}

// ─── 内部辅助函数 ─────────────────────────────────────────

/// 替换 <|audio|> placeholder 为特殊 token 序列
fn replace_audio_placeholders(
    content: &str,
    lengths: Vec<usize>,
    n_vq: usize,
    gen_slot: &str,
    audio_start: &str,
    audio_end: &str,
) -> String {
    let mut result = content.to_string();
    for &len in &lengths {
        let block = if len == 0 {
            format!("{audio_start}{audio_end}")
        } else {
            let slots = gen_slot.repeat(len);
            let delay_slots = gen_slot.repeat(n_vq.saturating_sub(1));
            format!("{audio_start}{slots}{delay_slots}{audio_end}")
        };
        result = result.replacen(AUDIO_PLACEHOLDER, &block, 1);
    }
    result
}

/// 构建统一编码（文本 + 音频混合序列）
fn build_unified_codes(
    text_ids: &[u32],
    audio_codes_list: &[&[Vec<i64>]],
    n_vq: usize,
    pad_code: i64,
    consts: &MossTtsConstants,
) -> Vec<Vec<i64>> {
    let n_text = text_ids.len();

    if audio_codes_list.is_empty() {
        // 无音频：全部 padding
        return (0..n_text)
            .map(|i| {
                let mut row = vec![pad_code; 1 + n_vq];
                row[0] = text_ids[i] as i64;
                row
            })
            .collect();
    }

    // 有音频：需要 delay pattern 编码
    // 简化版：将音频码通过 delay pattern 偏移后嵌入
    let audio_start_id = consts.audio_start_token_id as i64;
    let audio_end_id = consts.audio_end_token_id as i64;

    let mut result: Vec<Vec<i64>> = Vec::with_capacity(n_text);
    let mut audio_idx = 0;

    for (i, &tid) in text_ids.iter().enumerate() {
        let mut row = vec![pad_code; 1 + n_vq];
        row[0] = tid as i64;

        // 在 audio_start 和 audio_end 之间嵌入 delay pattern 编码的音频
        if tid as i64 == audio_start_id {
            audio_idx = i;
        }

        // 检查当前位置是否在音频段内
        if !audio_codes_list.is_empty() {
            let codes = audio_codes_list[0];
            let delayed = apply_delay_pattern(codes, pad_code);

            // 尝试在当前位置插入延迟码
            let delay_offset = i.saturating_sub(audio_idx + 1);
            if delay_offset < delayed.len() {
                for j in 0..n_vq.min(delayed[delay_offset].len()) {
                    row[1 + j] = delayed[delay_offset][j];
                }
            }
        }

        if tid as i64 == audio_end_id {
            audio_idx = 0; // reset
        }

        result.push(row);
    }

    result
}

impl MossTtsTokenizer {
    /// 从目录加载 tokenizer
    ///
    /// 需要 `tokenizer.json` 文件。实际 BPE 编码通过外部工具完成
    /// （如 Python `tokenizers` 库或 Rust `tokenizers` crate）。
    pub fn new(tokenizer_dir: &str) -> AppResult<Self> {
        let path = std::path::Path::new(tokenizer_dir).join("tokenizer.json");
        if !path.exists() {
            return Err(AppError::VoiceCloningError(format!(
                "tokenizer.json not found in {tokenizer_dir}"
            )));
        }
        let consts = MossTtsConstants::moss_tts_defaults();
        Ok(Self {
            tokenizer_dir: tokenizer_dir.to_string(),
            special_tokens: SpecialTokens::from_constants(&consts),
        })
    }

    /// 编码文本为 token ID
    ///
    /// 注意：实际 BPE 编码需要 `tokenizers` 库。
    /// 此方法为接口占位，实际使用时需要通过 Python subprocess 或 Rust crate 实现。
    pub fn encode(&self, _text: &str) -> AppResult<Vec<u32>> {
        // TODO: 使用 Rust `tokenizers` crate 或 Python subprocess 实现 BPE 编码
        // 当前返回空结果作为占位
        tracing::warn!(
            "MossTtsTokenizer::encode not fully implemented — requires tokenizers crate"
        );
        Err(AppError::VoiceCloningError(
            "BPE encoding not implemented — install tokenizers crate or use Python subprocess"
                .to_string(),
        ))
    }

    /// 解码 token ID 为文本
    pub fn decode(&self, _ids: &[i64]) -> String {
        // TODO: 同上
        String::new()
    }
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens() {
        let consts = MossTtsConstants::moss_tts_defaults();
        let special = SpecialTokens::from_constants(&consts);
        assert_eq!(special.im_start, "<|im_start|>");
        assert_eq!(special.audio_start, "<|audio_start|>");
    }

    #[test]
    fn test_replace_audio_placeholders() {
        let result = replace_audio_placeholders(
            "ref: <|audio|>",
            vec![5],
            32,
            "<|audio_user|>",
            "<|audio_start|>",
            "<|audio_end|>",
        );
        assert!(result.contains("<|audio_start|>"));
        assert!(result.contains("<|audio_end|>"));
        assert!(!result.contains("<|audio|>"));
    }

    #[test]
    fn test_replace_no_audio() {
        let result = replace_audio_placeholders(
            "ref: None",
            vec![],
            32,
            "<|audio_user|>",
            "<|audio_start|>",
            "<|audio_end|>",
        );
        assert_eq!(result, "ref: None");
    }

    #[test]
    fn test_parse_empty_output() {
        let consts = MossTtsConstants::moss_tts_defaults();
        let tokenizer = MossTtsTokenizer {
            tokenizer_dir: "/nonexistent".to_string(),
            special_tokens: SpecialTokens::from_constants(&consts),
        };
        let (text, audio) = parse_generation_output(&tokenizer, &[], 0, &consts).unwrap();
        assert!(text.is_empty());
        assert!(audio.is_empty());
    }
}
