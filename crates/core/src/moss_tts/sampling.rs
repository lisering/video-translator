//! 采样函数 — 从 MOSS-TTS `inference_utils.py` / `sampling.py` 移植
//!
//! 纯 Rust 实现的 top_k / top_p / softmax / multinomial / repetition_penalty，
//! 无 PyTorch / NumPy 依赖。
//!
//! 对应 MOSS-TTS 项目:
//! - `moss_tts_delay/inference_utils.py` (PyTorch 版)
//! - `moss_tts_delay/llama_cpp/sampling.py` (NumPy 版)

use rayon::prelude::*;

/// 数值稳定的 softmax（沿最后一维）
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Top-K 过滤：只保留最高 K 个 logits，其余设为 -inf
pub fn apply_top_k(logits: &mut [f32], top_k: usize) {
    let k = top_k.min(logits.len());
    if k == 0 || k >= logits.len() {
        return;
    }
    // 找到第 K 大的值作为阈值
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[indices[k - 1]];
    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Top-P (nucleus) 过滤：保留累积概率 <= top_p 的 token
pub fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 || top_p <= 0.0 || logits.is_empty() {
        return;
    }
    let probs = softmax(logits);
    // 按概率降序排列索引
    let mut sorted_idx: Vec<usize> = (0..logits.len()).collect();
    sorted_idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 累积概率
    let mut cum_prob = 0.0f32;
    let mut remove = vec![false; logits.len()];
    for &idx in &sorted_idx {
        cum_prob += probs[idx];
        if cum_prob > top_p {
            remove[idx] = true;
        }
    }
    // 第一个（最高概率）永远不移除
    if let Some(&first) = sorted_idx.first() {
        remove[first] = false;
    }

    for (i, &r) in remove.iter().enumerate() {
        if r {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// 重复惩罚：对已出现 token 的 logits 施加惩罚
///
/// - 正 logits: 除以 penalty（降低概率）
/// - 负 logits: 乘以 penalty（提高概率）
pub fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 || prev_tokens.is_empty() {
        return;
    }
    // 获取唯一 token ID
    let mut unique: Vec<u32> = prev_tokens
        .iter()
        .cloned()
        .filter(|&t| (t as usize) < logits.len())
        .collect();
    unique.sort_unstable();
    unique.dedup();

    for token_id in unique {
        let idx = token_id as usize;
        if logits[idx] > 0.0 {
            logits[idx] /= penalty;
        } else {
            logits[idx] *= penalty;
        }
    }
}

/// 多项分布采样：根据概率随机选择一个 token
pub fn multinomial(probs: &[f32]) -> usize {
    if probs.is_empty() {
        return 0;
    }
    // 累积概率
    let mut cum = 0.0f32;
    let r: f32 = rand::random();
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    // fallback: 返回最后一个
    probs.len().saturating_sub(1)
}

/// 采样 token：支持 top_k / top_p / repetition_penalty / temperature
///
/// 返回采样到的 token ID
pub fn sample_token(
    logits: &[f32],
    prev_tokens: Option<&[u32]>,
    repetition_penalty: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    do_sample: bool,
    temperature: f32,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let mut filtered: Vec<f32> = logits.to_vec();

    // 1. 重复惩罚
    if let Some(prev) = prev_tokens {
        apply_repetition_penalty(&mut filtered, prev, repetition_penalty);
    }

    // 2. 贪心采样
    if !do_sample || temperature <= 0.0 {
        return argmax(&filtered) as u32;
    }

    // 3. 温度缩放
    let temp = if temperature > 0.0 { temperature } else { 1.0 };
    for logit in filtered.iter_mut() {
        *logit /= temp;
    }

    // 4. Top-K 过滤
    if let Some(k) = top_k {
        if k > 0 {
            apply_top_k(&mut filtered, k);
        }
    }

    // 5. Top-P 过滤
    if let Some(p) = top_p {
        if p < 1.0 {
            apply_top_p(&mut filtered, p);
        }
    }

    // 6. Softmax → 概率
    let probs = softmax(&filtered);

    // 7. 多项分布采样
    multinomial(&probs) as u32
}

/// Argmax：返回最大值索引
pub fn argmax(arr: &[f32]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─── 批量采样（多个头并行）────────────────────────────────

/// 批量采样：对多个 logits 头并行采样
///
/// 用于 delay pattern 中同时采样 32 个音频码本的 token。
/// 使用 Rayon 并行处理。
pub fn sample_tokens_batch(
    logits_batch: &[Vec<f32>],              // [n_heads, vocab_size]
    prev_tokens_batch: Option<&[Vec<u32>]>, // [n_heads, T]
    repetition_penalty: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    do_sample: bool,
    temperature: f32,
) -> Vec<u32> {
    logits_batch
        .par_iter()
        .enumerate()
        .map(|(i, logits)| {
            let prev = prev_tokens_batch.and_then(|batch| batch.get(i).map(|v| v.as_slice()));
            sample_token(
                logits,
                prev,
                repetition_penalty,
                top_p,
                top_k,
                do_sample,
                temperature,
            )
        })
        .collect()
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let result = softmax(&[1.0, 2.0, 3.0]);
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let result = softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_overflow() {
        let result = softmax(&[1e30, 1e30, 1e30]);
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn test_apply_top_k() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        assert_eq!(logits.iter().filter(|&&l| l > f32::NEG_INFINITY).count(), 2);
        assert!(logits[1] > f32::NEG_INFINITY); // 5.0
        assert!(logits[4] > f32::NEG_INFINITY); // 4.0
    }

    #[test]
    fn test_apply_top_p() {
        let mut logits = vec![10.0, 1.0, 1.0, 1.0];
        apply_top_p(&mut logits, 0.5);
        // 最高概率 token 应保留
        assert!(logits[0] > f32::NEG_INFINITY);
        // 低概率 token 应被过滤
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        // 正 logit: 2.0 / 2.0 = 1.0
        assert!((logits[0] - 1.0).abs() < 1e-5);
        // 负 logit: -2.0 * 2.0 = -4.0
        assert!((logits[1] - (-4.0)).abs() < 1e-5);
        // 未出现: 不变
        assert!((logits[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = vec![1.0, 5.0, 2.0];
        let token = sample_token(&logits, None, 1.0, None, None, false, 1.0);
        assert_eq!(token, 1); // argmax
    }

    #[test]
    fn test_sample_token_with_temp() {
        let logits = vec![1.0, 5.0, 2.0];
        // 高温 + top_k=1 = 确定性选择最大
        let token = sample_token(&logits, None, 1.0, None, Some(1), true, 1.0);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_batch_sampling() {
        let batch = vec![vec![1.0, 5.0, 2.0], vec![3.0, 1.0, 2.0]];
        let tokens = sample_tokens_batch(&batch, None, 1.0, None, Some(1), true, 1.0);
        assert_eq!(tokens, vec![1, 0]); // argmax of each
    }
}
