//! 采样工具 — argmax / top-k / GPU-native 采样 / 推测解码辅助函数

use std::collections::HashSet;

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Tensor};
use rand::{Rng, SeedableRng};

/// Argmax: 返回最大值的索引
pub fn argmax(logits: &Tensor) -> Result<u32> {
    let idx = logits.argmax(0)?;
    let idx = idx.to_scalar::<u32>()?;
    Ok(idx)
}

/// GPU-native batch argmax: 对 [T, vocab_size] 的每一行取 argmax
///
/// 返回 T 个 token ID，全程在 GPU 上计算，仅将结果索引 (T 个 u32) 传回 CPU。
/// 比 `for t in 0..T { argmax(logits.i(t)) }` 减少 T 次 kernel launch + GPU→CPU 同步。
pub fn argmax_on_device(logits: &Tensor) -> Result<Vec<u32>> {
    // logits: [T, vocab_size]
    // argmax(Dim) → [T]
    let indices = logits.argmax(1)?;
    Ok(indices.to_vec1::<u32>()?)
}

/// GPU-native 采样路径
///
/// 将温度缩放、argmax/top-k 选择等操作尽可能在 GPU 上完成，
/// 减少 GPU↔CPU 同步开销。重复惩罚和 n-gram ban 仍在 CPU 上执行
/// (需要遍历历史序列)，但 logits 的 GPU→CPU 传输仅一次。
///
/// 当 `top_k == 1` 且无重复惩罚时，直接用 GPU argmax，零 CPU 排序。
pub fn sample_top_k_gpu(
    logits: &Tensor,
    top_k: usize,
    temperature: f32,
    seed: Option<u64>,
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    history: &[u32],
) -> Result<u32> {
    // 当无惩罚 + 无 ngram + 温度=1.0 时，直接 GPU argmax (零 CPU 排序)
    if repetition_penalty <= 1.0 && no_repeat_ngram_size == 0 && temperature == 1.0 {
        return Ok(argmax(logits)?);
    }

    // 当 top_k == 1 (greedy) 且无 ngram ban 时，GPU argmax (仅需 CPU 重复惩罚)
    if top_k == 1 && no_repeat_ngram_size == 0 {
        // 如果有重复惩罚，需要先在 CPU 上修改 logits
        if repetition_penalty > 1.0 && !history.is_empty() {
            let logits_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mut logits_mod = logits_vec;
            let history_set: HashSet<usize> = history.iter().map(|&t| t as usize).collect();
            for &token_idx in &history_set {
                if token_idx < logits_mod.len() {
                    if logits_mod[token_idx] > 0.0 {
                        logits_mod[token_idx] /= repetition_penalty;
                    } else {
                        logits_mod[token_idx] *= repetition_penalty;
                    }
                }
            }
            // argmax on CPU (vocab_size=3072, 很快)
            let best = logits_mod
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            return Ok(best);
        }
        return Ok(argmax(logits)?);
    }

    // 通用路径: 回退到 CPU sample_top_k
    sample_top_k(
        logits,
        top_k,
        temperature,
        seed,
        repetition_penalty,
        no_repeat_ngram_size,
        history,
    )
}

/// Top-K 采样 (手动实现，不依赖 candle topk)
///
/// 支持两种重复抑制策略:
///
/// 1. **重复惩罚 (Repetition Penalty)**: 对已出现的 token logits 施加惩罚。
///    参考 HF Transformers: `logits[t] = logits[t] / penalty if logits[t] > 0`
///    `else logits[t] * penalty`。惩罚值 >1 降低重复概率，=1.0 禁用。
///
/// 2. **No-repeat n-gram**: 禁止已出现的 n-gram 序列再次出现。
/// 对匹配当前 n-1 前缀的历史 n-gram 的后续 token，设置 logits 为 -inf。
pub fn sample_top_k(
    logits: &Tensor,
    top_k: usize,
    temperature: f32,
    seed: Option<u64>,
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    history: &[u32],
) -> Result<u32> {
    let vocab_size = logits.dim(0)?;
    let k = top_k.min(vocab_size);

    let mut logits_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    // ── 1. 重复惩罚 ──
    if repetition_penalty > 1.0 && !history.is_empty() {
        let history_set: HashSet<usize> = history.iter().map(|&t| t as usize).collect();
        for &token_idx in &history_set {
            if token_idx < logits_vec.len() {
                if logits_vec[token_idx] > 0.0 {
                    logits_vec[token_idx] /= repetition_penalty;
                } else {
                    logits_vec[token_idx] *= repetition_penalty;
                }
            }
        }
    }

    // ── 2. No-repeat n-gram ban ──
    if no_repeat_ngram_size >= 2 && history.len() >= no_repeat_ngram_size {
        let n = no_repeat_ngram_size;
        // 当前 n-1 前缀 (序列的最后 n-1 个 token)
        let current_prefix = &history[history.len() - (n - 1)..];
        // 搜索历史中所有匹配当前前缀的 n-gram，禁掉它们的后续 token
        for j in 0..=(history.len() - n) {
            if &history[j..j + n - 1] == current_prefix {
                let banned_token = history[j + n - 1] as usize;
                if banned_token < logits_vec.len() {
                    logits_vec[banned_token] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 应用温度
    let scaled: Vec<f32> = if temperature != 1.0 {
        logits_vec.iter().map(|&v| v / temperature).collect()
    } else {
        logits_vec
    };

    // 手动 top-k: 排序后取前 k 个
    let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k_items: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

    // Softmax over top-k
    let max_logit = top_k_items[0].1;
    let exp_sum: f32 = top_k_items.iter().map(|(_, v)| (v - max_logit).exp()).sum();
    let probs: Vec<f32> = top_k_items
        .iter()
        .map(|(_, v)| (v - max_logit).exp() / exp_sum)
        .collect();

    // 采样
    let mut rng = if let Some(s) = seed {
        rand::rngs::StdRng::seed_from_u64(s)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    let dist = rand::distributions::WeightedIndex::new(&probs)
        .map_err(|e| anyhow::anyhow!("WeightedIndex failed: {}", e))?;
    let sample_idx = rng.sample(&dist);

    Ok(top_k_items[sample_idx].0 as u32)
}

// ──────────────────────────── 推测解码辅助函数 ────────────────────────────

/// 检查 token 是否被 no-repeat-ngram 禁止
///
/// 如果 tokens 的最后 (n-1) 个 token 加上 `token` 构成的 n-gram
/// 在历史中已出现过，则该 token 被禁止。
pub fn is_ngram_banned(tokens: &[u32], token: u32, ngram_size: usize) -> bool {
    if ngram_size < 2 || tokens.len() < ngram_size {
        return false;
    }
    let n = ngram_size;
    let current_prefix = &tokens[tokens.len() - (n - 1)..];
    for j in 0..=(tokens.len() - n) {
        if &tokens[j..j + n - 1] == current_prefix && tokens[j + n - 1] == token {
            return true;
        }
    }
    false
}

/// 更新 n-gram 推测表
///
/// 将最近生成的 token 加入推测表: (n-1)-token 前缀 → 最近出现的下一个 token。
/// 使用"最近优先"策略: 如果同一个前缀出现多次，保留最近一次的后续 token。
pub fn update_ngram_table(
    table: &mut std::collections::HashMap<Vec<u32>, u32>,
    tokens: &[u32],
    ngram_size: usize,
) {
    if tokens.len() >= ngram_size {
        let prefix = &tokens[tokens.len() - ngram_size..tokens.len() - 1];
        let next_token = *tokens.last().unwrap();
        table.insert(prefix.to_vec(), next_token);
    }
}

// ──────────────────────────── 权重量化 ────────────────────────────

/// 解析量化格式字符串
///
/// 支持的格式:
/// - "q8_0": 8-bit 量化 (block_size=32, 精度损失极小, 权重 1/4)
/// - "q4_0": 4-bit 量化 (block_size=32, 精度损失较小, 权重 1/8)
/// - "q4k":  4-bit K-量化 (block_size=256, 精度更好, 权重 1/8)
/// - "none" / None: 不量化
pub fn parse_quantize(s: &Option<String>) -> Option<GgmlDType> {
    match s.as_deref().map(|s| s.to_lowercase()).as_deref() {
        Some("q8_0") => Some(GgmlDType::Q8_0),
        Some("q4_0") => Some(GgmlDType::Q4_0),
        Some("q4k") => Some(GgmlDType::Q4K),
        Some("q6k") => Some(GgmlDType::Q6K),
        Some("q5_0") => Some(GgmlDType::Q5_0),
        Some("q5k") => Some(GgmlDType::Q5K),
        Some("none") | None => None,
        Some(other) => {
            tracing::warn!("Unknown quantize format '{}', ignoring. Supported: q8_0, q4_0, q4k, q6k, q5_0, q5k", other);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── parse_quantize ───

    #[test]
    fn test_parse_quantize_none() {
        assert_eq!(parse_quantize(&None), None);
    }

    #[test]
    fn test_parse_quantize_empty_string() {
        assert_eq!(parse_quantize(&Some("".to_string())), None);
    }

    #[test]
    fn test_parse_quantize_explicit_none() {
        assert_eq!(parse_quantize(&Some("none".to_string())), None);
    }

    #[test]
    fn test_parse_quantize_q8_0() {
        assert_eq!(
            parse_quantize(&Some("q8_0".to_string())),
            Some(GgmlDType::Q8_0)
        );
        assert_eq!(
            parse_quantize(&Some("Q8_0".to_string())),
            Some(GgmlDType::Q8_0)
        );
    }

    #[test]
    fn test_parse_quantize_q4_0() {
        assert_eq!(
            parse_quantize(&Some("q4_0".to_string())),
            Some(GgmlDType::Q4_0)
        );
    }

    #[test]
    fn test_parse_quantize_q4k() {
        assert_eq!(
            parse_quantize(&Some("q4k".to_string())),
            Some(GgmlDType::Q4K)
        );
    }

    #[test]
    fn test_parse_quantize_q6k() {
        assert_eq!(
            parse_quantize(&Some("q6k".to_string())),
            Some(GgmlDType::Q6K)
        );
    }

    #[test]
    fn test_parse_quantize_q5_0() {
        assert_eq!(
            parse_quantize(&Some("q5_0".to_string())),
            Some(GgmlDType::Q5_0)
        );
    }

    #[test]
    fn test_parse_quantize_q5k() {
        assert_eq!(
            parse_quantize(&Some("q5k".to_string())),
            Some(GgmlDType::Q5K)
        );
    }

    #[test]
    fn test_parse_quantize_invalid() {
        assert_eq!(parse_quantize(&Some("invalid".to_string())), None);
        assert_eq!(parse_quantize(&Some("q3".to_string())), None);
    }

    // ─── is_ngram_banned ───

    #[test]
    fn test_is_ngram_banned_empty() {
        assert!(!is_ngram_banned(&[], 42, 3));
    }

    #[test]
    fn test_is_ngram_banned_too_short() {
        assert!(!is_ngram_banned(&[1, 2], 3, 3));
    }

    #[test]
    fn test_is_ngram_banned_disabled() {
        assert!(!is_ngram_banned(&[1, 2, 3], 4, 1));
        assert!(!is_ngram_banned(&[1, 2, 3], 4, 0));
    }

    #[test]
    fn test_is_ngram_banned_match() {
        // 历史: [1, 2, 3, 1, 2], 检查 token 3, ngram=3
        // 前缀 [1, 2] 在位置 0 后跟 3, 当前前缀也是 [1, 2], token=3 → banned
        assert!(is_ngram_banned(&[1, 2, 3, 1, 2], 3, 3));
    }

    #[test]
    fn test_is_ngram_banned_no_match() {
        // 历史: [1, 2, 3, 1, 2], 检查 token 4, ngram=3
        // 前缀 [1, 2] 后跟 3，不是 4 → not banned
        assert!(!is_ngram_banned(&[1, 2, 3, 1, 2], 4, 3));
    }

    #[test]
    fn test_is_ngram_banned_multiple_occurrences() {
        // 历史: [1, 2, 3, 4, 2, 3], 检查 token 4, ngram=3
        // 前缀 [2, 3] (最后 2 个 token) 在位置 1 也出现, 后跟 4 = token → banned
        assert!(is_ngram_banned(&[1, 2, 3, 4, 2, 3], 4, 3));
    }

    // ─── update_ngram_table ───

    #[test]
    fn test_update_ngram_table_basic() {
        let mut table: std::collections::HashMap<Vec<u32>, u32> = std::collections::HashMap::new();
        let tokens = vec![1, 2, 3, 4];
        // ngram_size=3, 前缀 [2, 3] → next=4
        update_ngram_table(&mut table, &tokens, 3);
        assert_eq!(table.get(&vec![2, 3]), Some(&4));
    }

    #[test]
    fn test_update_ngram_table_too_short() {
        let mut table: std::collections::HashMap<Vec<u32>, u32> = std::collections::HashMap::new();
        let tokens = vec![1, 2];
        update_ngram_table(&mut table, &tokens, 3);
        assert!(table.is_empty());
    }

    #[test]
    fn test_update_ngram_table_recent_wins() {
        let mut table: std::collections::HashMap<Vec<u32>, u32> = std::collections::HashMap::new();
        // 第一次: [1, 2, 3] → 前缀 [1, 2] → 3
        update_ngram_table(&mut table, &[1, 2, 3], 3);
        assert_eq!(table.get(&vec![1, 2]), Some(&3));
        // 第二次: [1, 2, 5] → 前缀 [1, 2] → 5 (覆盖)
        update_ngram_table(&mut table, &[1, 2, 5], 3);
        assert_eq!(table.get(&vec![1, 2]), Some(&5));
    }
}
