//! Delay Pattern 状态机 — 从 MOSS-TTS `delay_state.py` 移植
//!
//! 实现纯 Rust 的 delay pattern 自回归生成逻辑。
//! 对应 MOSS-TTS 项目 `moss_tts_delay/llama_cpp/delay_state.py`。
//!
//! # Delay Pattern 原理
//! 32 个音频码本通过对角线偏移（delay scheduling）实现并行预测：
//! - Head 0 预测文本 token at t
//! - Head k 预测 codebook k-1 at t-(k-1)
//! - 音频结束后需要 n_vq 步 drain 阶梯
//!
//! # 状态机
//! - `audio_length`: 已生成的音频帧数
//! - `delayed_length`: delay 偏移计数（INT64_MAX = 未开始）
//! - `is_audio`: 是否在音频生成阶段
//! - `is_stopping`: 是否已收到停止信号

use super::constants::MossTtsConstants;
use super::sampling::{sample_token, sample_tokens_batch};

/// i64 最大值常量
const INT64_MAX: i64 = i64::MAX;

/// 采样配置
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// 文本采样温度
    pub text_temperature: f32,
    /// 文本 top_p
    pub text_top_p: f32,
    /// 文本 top_k
    pub text_top_k: usize,
    /// 音频采样温度
    pub audio_temperature: f32,
    /// 音频 top_p
    pub audio_top_p: f32,
    /// 音频 top_k
    pub audio_top_k: usize,
    /// 音频重复惩罚
    pub audio_repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            text_temperature: 1.5,
            text_top_p: 1.0,
            text_top_k: 50,
            audio_temperature: 1.7,
            audio_top_p: 0.8,
            audio_top_k: 25,
            audio_repetition_penalty: 1.0,
        }
    }
}

/// Delay 状态机（batch_size=1）
pub struct DelayState {
    /// 已生成音频帧数
    pub audio_length: i64,
    /// delay 偏移计数
    pub delayed_length: i64,
    /// 是否在音频阶段
    pub is_audio: bool,
    /// 是否停止
    pub is_stopping: bool,
    /// 时间步
    pub time_step: usize,
    /// 文本历史
    pub text_history: Vec<u32>,
    /// 音频历史（动态增长缓冲）
    audio_buf: Vec<Vec<i64>>,
    /// n_vq（保留供未来使用）
    #[allow(dead_code)]
    n_vq: usize,
}

impl DelayState {
    /// 创建新状态
    pub fn new(n_vq: usize) -> Self {
        Self {
            audio_length: 0,
            delayed_length: INT64_MAX,
            is_audio: false,
            is_stopping: false,
            time_step: 0,
            text_history: Vec::new(),
            audio_buf: Vec::new(),
            n_vq,
        }
    }

    /// 获取音频历史
    pub fn audio_history(&self) -> Option<&[Vec<i64>]> {
        if self.audio_buf.is_empty() {
            None
        } else {
            Some(&self.audio_buf)
        }
    }

    /// 追加音频帧
    fn append_audio(&mut self, codes: Vec<i64>) {
        self.audio_buf.push(codes);
    }

    /// 获取特定码本的历史 token 列表（用于重复惩罚）
    pub fn get_audio_channel_history(&self, channel: usize) -> Vec<u32> {
        self.audio_buf
            .iter()
            .filter_map(|frame| frame.get(channel).map(|&v| v as u32))
            .collect()
    }
}

/// 从 prefill input_ids 初始化 delay 状态
///
/// `input_ids`: [S, 1+n_vq] — 完整 prompt 序列
pub fn init_delay_state(input_ids: &[Vec<i64>], consts: &MossTtsConstants) -> DelayState {
    let n_vq = consts.n_vq;
    let mut state = DelayState::new(n_vq);

    let seq_len = input_ids.len();
    if seq_len == 0 {
        return state;
    }

    let text_channel: Vec<i64> = input_ids.iter().map(|row| row[0]).collect();
    let last_text_token = text_channel[seq_len - 1];

    let is_continuation = last_text_token == consts.audio_start_token_id as i64
        || last_text_token == consts.audio_assistant_gen_slot_token_id as i64;

    if is_continuation {
        // 查找最后一个 audio_start_token
        let audio_start_idx = text_channel
            .iter()
            .rposition(|&t| t == consts.audio_start_token_id as i64);

        if let Some(idx) = audio_start_idx {
            state.audio_length = (seq_len - idx) as i64;
            state.is_audio = true;
        }
    }

    state.text_history = text_channel.iter().map(|&t| t as u32).collect();
    state.audio_buf = input_ids.iter().map(|row| row[1..].to_vec()).collect();

    state
}

/// 执行单步自回归生成
///
/// 返回 `next_input_ids`: [1+n_vq] — [text_token, audio_0, ..., audio_{n_vq-1}]
pub fn step(
    state: &mut DelayState,
    text_logits: &[f32],
    audio_logits: &[Vec<f32>], // [n_vq, audio_vocab_size]
    config: &SamplingConfig,
    consts: &MossTtsConstants,
) -> Vec<i64> {
    let n_vq = consts.n_vq;

    if state.is_stopping {
        let mut result = vec![consts.audio_pad_code as i64; 1 + n_vq];
        result[0] = consts.pad_token_id as i64;
        return result;
    }

    // ── 文本 token 决策 ──
    let next_text: i64;
    if state.delayed_length < n_vq as i64 {
        next_text = consts.audio_assistant_delay_slot_token_id as i64;
    } else if state.delayed_length == n_vq as i64 {
        next_text = consts.audio_end_token_id as i64;
        state.is_audio = false;
    } else {
        let text_temp = if config.text_temperature > 0.0 {
            config.text_temperature
        } else {
            1.0
        };
        let text_do_sample = config.text_temperature > 0.0;

        let mut scaled: Vec<f32> = text_logits.iter().map(|&l| l / text_temp).collect();

        // mask 非音频/音频 token
        if !state.is_audio {
            // 非音频状态：排除 pad / gen_slot / delay_slot / end
            for &id in &super::constants::pre_exclude_ids(consts) {
                let idx = id as usize;
                if idx < scaled.len() {
                    scaled[idx] = f32::NEG_INFINITY;
                }
            }
        } else {
            // 音频状态：只允许 gen_slot / delay_slot
            let allowed = super::constants::audio_allowed_ids(consts);
            let mask: Vec<bool> = (0..scaled.len())
                .map(|i| !allowed.contains(&(i as u32)))
                .collect();
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    scaled[i] = f32::NEG_INFINITY;
                }
            }
        }

        // 第 0 步排除 delay_slot
        if state.time_step == 0 {
            let idx = consts.audio_assistant_delay_slot_token_id as usize;
            if idx < scaled.len() {
                scaled[idx] = f32::NEG_INFINITY;
            }
        }
        // 前 n_vq 步排除 im_end
        if state.time_step <= n_vq {
            let idx = consts.im_end_token_id as usize;
            if idx < scaled.len() {
                scaled[idx] = f32::NEG_INFINITY;
            }
        }

        let token = sample_token(
            &scaled,
            Some(&state.text_history.clone()),
            1.0, // text 没有 repetition penalty
            Some(config.text_top_p),
            Some(config.text_top_k),
            text_do_sample,
            1.0, // 温度已应用
        );
        next_text = token as i64;
    }

    // 更新音频/停止状态
    if next_text == consts.audio_start_token_id as i64 {
        state.is_audio = true;
    }
    if next_text == consts.im_end_token_id as i64 {
        state.is_stopping = true;
    }

    // ── 音频 token 决策 ──
    let mut next_audio = vec![consts.audio_pad_code as i64; n_vq];

    // pre_audio_mask: codebook index < audio_length
    let pre_audio_mask: Vec<bool> = (0..n_vq as i64).map(|i| i < state.audio_length).collect();

    // post_audio_mask: codebook index > delayed_length - 1
    let post_audio_mask: Vec<bool> = if state.delayed_length == INT64_MAX {
        vec![true; n_vq]
    } else {
        (0..n_vq as i64)
            .map(|i| i > state.delayed_length - 1)
            .collect()
    };

    // sampling_mask = pre & post
    let sampling_mask: Vec<bool> = pre_audio_mask
        .iter()
        .zip(post_audio_mask.iter())
        .map(|(a, b)| *a && *b)
        .collect();

    if sampling_mask.iter().any(|&m| m) {
        let audio_temp = if config.audio_temperature > 0.0 {
            config.audio_temperature
        } else {
            1.0
        };
        let audio_do_sample = config.audio_temperature > 0.0;

        // 对每个需要采样的码本处理
        let mask_indices: Vec<usize> = sampling_mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i)
            .collect();

        // 准备每个码本的 logits
        let masked_logits: Vec<Vec<f32>> = mask_indices
            .iter()
            .map(|&i| {
                let mut logits = audio_logits[i].clone();
                // 温度缩放
                for l in logits.iter_mut() {
                    *l /= audio_temp;
                }
                // 排除 pad_code
                let pad_idx = consts.audio_pad_code as usize;
                if pad_idx < logits.len() {
                    logits[pad_idx] = f32::NEG_INFINITY;
                }
                logits
            })
            .collect();

        // 准备每个码本的历史 token（用于重复惩罚）
        let prev_tokens: Vec<Vec<u32>> = mask_indices
            .iter()
            .map(|&i| state.get_audio_channel_history(i))
            .collect();

        // 批量采样
        let sampled = sample_tokens_batch(
            &masked_logits,
            Some(&prev_tokens),
            config.audio_repetition_penalty,
            Some(config.audio_top_p),
            Some(config.audio_top_k),
            audio_do_sample,
            1.0, // 温度已应用
        );

        for (k, &idx) in mask_indices.iter().enumerate() {
            next_audio[idx] = sampled[k] as i64;
        }
    }

    // ── 状态更新 ──
    let next_text_u32 = next_text as u32;
    if next_text == consts.audio_start_token_id as i64
        || next_text == consts.audio_assistant_gen_slot_token_id as i64
        || next_text == consts.audio_assistant_delay_slot_token_id as i64
    {
        state.audio_length += 1;
    }
    if next_text == consts.audio_end_token_id as i64 {
        state.audio_length = 0;
    }

    if state.delayed_length == INT64_MAX
        && next_text == consts.audio_assistant_delay_slot_token_id as i64
    {
        state.delayed_length = 0;
    }
    if state.delayed_length != INT64_MAX {
        state.delayed_length += 1;
    }
    if state.delayed_length > n_vq as i64 {
        state.delayed_length = INT64_MAX;
    }

    state.time_step += 1;
    state.text_history.push(next_text_u32);
    state.append_audio(next_audio.clone());

    let mut result = Vec::with_capacity(1 + n_vq);
    result.push(next_text);
    result.extend(next_audio);
    result
}

// ─── Delay Pattern 编解码 ─────────────────────────────────

/// 应用 delay pattern 到音频码本
///
/// `codes`: [T, n_vq] — 原始音频码
/// 返回: [T + n_vq - 1, n_vq] — 偏移后的码
pub fn apply_delay_pattern(codes: &[Vec<i64>], pad_code: i64) -> Vec<Vec<i64>> {
    if codes.is_empty() {
        return vec![];
    }
    let t = codes.len();
    let n_vq = codes[0].len();
    let mut delayed = vec![vec![pad_code; n_vq]; t + n_vq.saturating_sub(1)];

    for i in 0..n_vq {
        for j in 0..t {
            if i + j < delayed.len() {
                delayed[i + j][i] = codes[j][i];
            }
        }
    }
    delayed
}

/// 移除 delay pattern
///
/// `delay_codes`: [T + n_vq - 1, n_vq]
/// 返回: [T, n_vq]
pub fn apply_de_delay_pattern(delay_codes: &[Vec<i64>]) -> Vec<Vec<i64>> {
    if delay_codes.is_empty() {
        return vec![];
    }
    let total_len = delay_codes.len();
    let n_vq = delay_codes[0].len();
    let t = total_len.saturating_sub(n_vq.saturating_sub(1));
    if t == 0 {
        return vec![];
    }

    let mut codes = vec![vec![0i64; n_vq]; t];
    for i in 0..n_vq {
        for j in 0..t {
            if i + j < delay_codes.len() {
                codes[j][i] = delay_codes[i + j][i];
            }
        }
    }
    codes
}

/// 提取非 padding 音频段（de-delay 后）
pub fn extract_audio_segments(
    generation_audio: &[Vec<i64>],
    audio_pad_code: i64,
) -> Vec<Vec<Vec<i64>>> {
    let codes = apply_de_delay_pattern(generation_audio);
    if codes.is_empty() {
        return vec![];
    }

    let is_pad: Vec<bool> = codes
        .iter()
        .map(|row| row.iter().all(|&v| v == audio_pad_code))
        .collect();

    let non_pad_idx: Vec<usize> = (0..codes.len()).filter(|&i| !is_pad[i]).collect();

    if non_pad_idx.is_empty() {
        return vec![];
    }

    let mut segments = vec![];
    let mut start = non_pad_idx[0];
    for i in 1..non_pad_idx.len() {
        if non_pad_idx[i] != non_pad_idx[i - 1] + 1 {
            segments.push(codes[start..non_pad_idx[i - 1] + 1].to_vec());
            start = non_pad_idx[i];
        }
    }
    segments.push(codes[start..non_pad_idx[non_pad_idx.len() - 1] + 1].to_vec());
    segments
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::constants::MossTtsConstants;
    use super::*;

    #[test]
    fn test_init_delay_state_empty() {
        let consts = MossTtsConstants::moss_tts_defaults();
        let state = init_delay_state(&[], &consts);
        assert_eq!(state.audio_length, 0);
        assert!(!state.is_audio);
    }

    #[test]
    fn test_delay_state_default() {
        let state = DelayState::new(32);
        assert_eq!(state.delayed_length, INT64_MAX);
        assert!(!state.is_audio);
        assert!(!state.is_stopping);
    }

    #[test]
    fn test_delay_pattern_roundtrip() {
        let codes = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let delayed = apply_delay_pattern(&codes, 1024);
        let recovered = apply_de_delay_pattern(&delayed);
        assert_eq!(recovered, codes);
    }

    #[test]
    fn test_delay_pattern_shape() {
        let codes = vec![vec![1, 2, 3]; 5];
        let delayed = apply_delay_pattern(&codes, 1024);
        // T=5, n_vq=3 → delayed_len = 5 + 3 - 1 = 7
        assert_eq!(delayed.len(), 7);
        assert_eq!(delayed[0].len(), 3);
    }

    #[test]
    fn test_extract_audio_segments_all_pad() {
        let audio = vec![vec![1024; 3]; 5];
        let segs = extract_audio_segments(&audio, 1024);
        assert!(segs.is_empty());
    }

    #[test]
    fn test_extract_audio_segments_no_pad() {
        // Need total_len >= n_vq for de_delay_pattern to produce non-empty output
        // total_len=5, n_vq=3 → T = 5 - 3 + 1 = 3
        let audio = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
            vec![13, 14, 15],
        ];
        let segs = extract_audio_segments(&audio, 1024);
        assert_eq!(segs.len(), 1);
        assert!(!segs[0].is_empty());
    }

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert!((config.text_temperature - 1.5).abs() < 1e-5);
        assert!((config.audio_temperature - 1.7).abs() < 1e-5);
        assert_eq!(config.audio_top_k, 25);
    }

    #[test]
    fn test_step_stopping() {
        let consts = MossTtsConstants::moss_tts_defaults();
        let mut state = DelayState::new(consts.n_vq);
        state.is_stopping = true;
        let config = SamplingConfig::default();
        let text_logits = vec![0.0; 200000];
        let audio_logits = vec![vec![0.0; 1025]; consts.n_vq];
        let result = step(&mut state, &text_logits, &audio_logits, &config, &consts);
        assert_eq!(result[0], consts.pad_token_id as i64);
    }
}
