//! Token ID 常量 — 从 MOSS-TTS `configuration_moss_tts.py` 移植
//!
//! 对应 MOSS-TTS 项目 `moss_tts_delay/llama_cpp/_constants.py`
//! 以及 `moss_tts_delay/configuration_moss_tts.py` 的默认值。

use serde::Deserialize;
use std::path::Path;

/// 默认 Token ID 常量（与 MOSS-TTS v1.5 一致）
///
/// 这些值来自 `MossTTSDelayConfig.__init__()` 的默认参数。
#[derive(Debug, Clone, Deserialize)]
pub struct MossTtsConstants {
    /// 音频 VQ 码本数量（32 = 32 层 RVQ）
    pub n_vq: usize,
    /// 文本 padding token ID
    pub pad_token_id: u32,
    /// 对话开始 token ID
    pub im_start_token_id: u32,
    /// 对话结束 token ID
    pub im_end_token_id: u32,
    /// 音频段开始 token ID
    pub audio_start_token_id: u32,
    /// 音频段结束 token ID
    pub audio_end_token_id: u32,
    /// 用户侧音频 slot token ID
    pub audio_user_slot_token_id: u32,
    /// 助手生成 slot token ID
    pub audio_assistant_gen_slot_token_id: u32,
    /// 延迟 slot token ID
    pub audio_assistant_delay_slot_token_id: u32,
    /// 音频码本 padding 值
    pub audio_pad_code: u32,
    /// 音频词汇表大小
    pub audio_vocab_size: u32,
    /// 采样率（Hz）
    pub sampling_rate: u32,
}

impl Default for MossTtsConstants {
    fn default() -> Self {
        Self::moss_tts_defaults()
    }
}

impl MossTtsConstants {
    /// MOSS-TTS 默认常量
    pub fn moss_tts_defaults() -> Self {
        Self {
            n_vq: 32,
            pad_token_id: 151643,
            im_start_token_id: 151644,
            im_end_token_id: 151645,
            audio_start_token_id: 151652,
            audio_end_token_id: 151653,
            audio_user_slot_token_id: 151654,
            audio_assistant_gen_slot_token_id: 151656,
            audio_assistant_delay_slot_token_id: 151662,
            audio_pad_code: 1024,
            audio_vocab_size: 1024,
            sampling_rate: 24000,
        }
    }

    /// 从 config.json 加载常量（覆盖默认值）
    pub fn from_config_json(path: &Path) -> Self {
        let defaults = Self::moss_tts_defaults();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                #[derive(Deserialize)]
                struct ConfigFile {
                    n_vq: Option<usize>,
                    pad_token_id: Option<u32>,
                    im_start_token_id: Option<u32>,
                    im_end_token_id: Option<u32>,
                    audio_start_token_id: Option<u32>,
                    audio_end_token_id: Option<u32>,
                    audio_user_slot_token_id: Option<u32>,
                    audio_assistant_gen_slot_token_id: Option<u32>,
                    audio_assistant_delay_slot_token_id: Option<u32>,
                    audio_pad_code: Option<u32>,
                    audio_vocab_size: Option<u32>,
                    sampling_rate: Option<u32>,
                }
                match serde_json::from_str::<ConfigFile>(&content) {
                    Ok(cfg) => Self {
                        n_vq: cfg.n_vq.unwrap_or(defaults.n_vq),
                        pad_token_id: cfg.pad_token_id.unwrap_or(defaults.pad_token_id),
                        im_start_token_id: cfg
                            .im_start_token_id
                            .unwrap_or(defaults.im_start_token_id),
                        im_end_token_id: cfg.im_end_token_id.unwrap_or(defaults.im_end_token_id),
                        audio_start_token_id: cfg
                            .audio_start_token_id
                            .unwrap_or(defaults.audio_start_token_id),
                        audio_end_token_id: cfg
                            .audio_end_token_id
                            .unwrap_or(defaults.audio_end_token_id),
                        audio_user_slot_token_id: cfg
                            .audio_user_slot_token_id
                            .unwrap_or(defaults.audio_user_slot_token_id),
                        audio_assistant_gen_slot_token_id: cfg
                            .audio_assistant_gen_slot_token_id
                            .unwrap_or(defaults.audio_assistant_gen_slot_token_id),
                        audio_assistant_delay_slot_token_id: cfg
                            .audio_assistant_delay_slot_token_id
                            .unwrap_or(defaults.audio_assistant_delay_slot_token_id),
                        audio_pad_code: cfg.audio_pad_code.unwrap_or(defaults.audio_pad_code),
                        audio_vocab_size: cfg.audio_vocab_size.unwrap_or(defaults.audio_vocab_size),
                        sampling_rate: cfg.sampling_rate.unwrap_or(defaults.sampling_rate),
                    },
                    Err(e) => {
                        tracing::warn!("Failed to parse config.json ({}), using defaults", e);
                        defaults
                    }
                }
            }
            Err(_) => defaults,
        }
    }
}

// ─── 全局常量（使用默认值）──────────────────────────────────

/// 音频 VQ 码本数量
pub const N_VQ: usize = 32;
/// 采样率
pub const SAMPLE_RATE: u32 = 24000;
/// 音频 padding 码
pub const AUDIO_PAD_CODE: u32 = 1024;

// ─── 预排除 token ID 列表 ──────────────────────────────────

/// 非音频状态下排除的 token ID 列表
pub fn pre_exclude_ids(consts: &MossTtsConstants) -> Vec<u32> {
    vec![
        consts.pad_token_id,
        consts.audio_assistant_gen_slot_token_id,
        consts.audio_assistant_delay_slot_token_id,
        consts.audio_end_token_id,
    ]
}

/// 音频状态下允许的 token ID 列表
pub fn audio_allowed_ids(consts: &MossTtsConstants) -> Vec<u32> {
    vec![
        consts.audio_assistant_gen_slot_token_id,
        consts.audio_assistant_delay_slot_token_id,
    ]
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let c = MossTtsConstants::moss_tts_defaults();
        assert_eq!(c.n_vq, 32);
        assert_eq!(c.sampling_rate, 24000);
        assert_eq!(c.audio_pad_code, 1024);
        assert_eq!(c.audio_vocab_size, 1024);
        assert_eq!(c.pad_token_id, 151643);
        assert_eq!(c.im_start_token_id, 151644);
        assert_eq!(c.im_end_token_id, 151645);
        assert_eq!(c.audio_start_token_id, 151652);
        assert_eq!(c.audio_end_token_id, 151653);
        assert_eq!(c.audio_assistant_gen_slot_token_id, 151656);
        assert_eq!(c.audio_assistant_delay_slot_token_id, 151662);
    }

    #[test]
    fn test_pre_exclude_ids() {
        let c = MossTtsConstants::moss_tts_defaults();
        let ids = pre_exclude_ids(&c);
        assert_eq!(ids.len(), 4);
        assert!(ids.contains(&c.pad_token_id));
        assert!(ids.contains(&c.audio_end_token_id));
    }

    #[test]
    fn test_audio_allowed_ids() {
        let c = MossTtsConstants::moss_tts_defaults();
        let ids = audio_allowed_ids(&c);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&c.audio_assistant_gen_slot_token_id));
        assert!(ids.contains(&c.audio_assistant_delay_slot_token_id));
    }

    #[test]
    fn test_from_config_json_nonexistent() {
        let c = MossTtsConstants::from_config_json(Path::new("/nonexistent/config.json"));
        assert_eq!(c.n_vq, 32);
    }
}
