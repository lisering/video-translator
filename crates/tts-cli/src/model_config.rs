//! 模型配置模块
//!
//! 从 HuggingFace config.json 解析 Qwen3-TTS 模型配置。
//! 参考 TrevorS/qwen3-tts-rs 的 `src/models/config.rs`。

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ──────────────────────────── 层级配置 (用于构建 DecoderLayer) ────────────────────────────

/// Transformer 层配置（用于 Attention + MLP 构建）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TTSConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim_override: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl Default for Qwen3TTSConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            head_dim_override: Some(128),
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            sliding_window: None,
        }
    }
}

impl Qwen3TTSConfig {
    /// KV heads 数量（默认 = attention heads）
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// head 维度
    pub fn head_dim(&self) -> usize {
        self.head_dim_override
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

// ──────────────────────────── Talker 配置 ────────────────────────────

/// Talker 模型配置
#[derive(Debug, Clone)]
pub struct TalkerConfig {
    /// 文本词表大小 (151936)
    pub text_vocab_size: usize,
    /// 文本嵌入维度 (2048)
    pub text_embed_dim: usize,
    /// 隐藏维度 (1024 for 0.6B, 2048 for 1.7B)
    pub hidden_size: usize,
    /// 文本投影中间维度 (2048)
    pub text_proj_intermediate: usize,
    /// MLP 中间维度 (3072 for 0.6B, 6144 for 1.7B)
    pub intermediate_size: usize,
    /// Transformer 层数 (28)
    pub num_hidden_layers: usize,
    /// 注意力头数 (16)
    pub num_attention_heads: usize,
    /// KV 头数 (8, GQA)
    pub num_key_value_heads: usize,
    /// 头维度 (128)
    pub head_dim: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// 最大位置嵌入数
    pub max_position_embeddings: usize,
    /// Codec 词表大小 (3072)
    pub codec_vocab_size: usize,
    /// MRoPE section [T, H, W]
    pub mrope_section: Option<[usize; 3]>,
}

impl Default for TalkerConfig {
    /// 0.6B 默认配置
    fn default() -> Self {
        Self {
            text_vocab_size: 151936,
            text_embed_dim: 2048,
            hidden_size: 1024,
            text_proj_intermediate: 2048,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            codec_vocab_size: 3072,
            mrope_section: Some([24, 20, 20]),
        }
    }
}

impl TalkerConfig {
    /// 1.7B 配置
    pub fn large() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 6144,
            ..Default::default()
        }
    }

    /// 转换为层级配置
    pub fn to_layer_config(&self) -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            head_dim_override: Some(self.head_dim),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            sliding_window: None,
        }
    }
}

// ──────────────────────────── CodePredictor 配置 ────────────────────────────

/// CodePredictor 配置
#[derive(Debug, Clone)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Codec 词表大小 (2048)
    pub vocab_size: usize,
    /// Codebook 组数 (16)
    pub num_code_groups: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// 当 talker hidden != 1024 时的投影维度
    pub codec_embed_dim: Option<usize>,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 5,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 2048,
            num_code_groups: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            codec_embed_dim: None,
        }
    }
}

impl CodePredictorConfig {
    pub fn to_layer_config(&self) -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            head_dim_override: Some(self.head_dim),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            sliding_window: None,
        }
    }
}

// ──────────────────────────── Speaker Encoder 配置 ────────────────────────────

/// ECAPA-TDNN 说话人编码器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEncoderConfig {
    /// 输入 mel 通道数 (128)
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    /// 输出嵌入维度 (1024)
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,
    /// 每块通道数 [initial_out, se_res2net_0, se_res2net_1, se_res2net_2, mfa_out]
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,
    /// 每块卷积核大小
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    /// 每块膨胀率
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    /// ASP 注意力通道数
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,
    /// Res2Net 缩放因子
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    /// SE block 瓶颈通道数
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,
    /// 音频采样率
    #[serde(default = "default_speaker_sample_rate")]
    pub sample_rate: u32,
}

fn default_mel_dim() -> usize {
    128
}
fn default_enc_dim() -> usize {
    1024
}
fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}
fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}
fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}
fn default_enc_attention_channels() -> usize {
    128
}
fn default_enc_res2net_scale() -> usize {
    8
}
fn default_enc_se_channels() -> usize {
    128
}
fn default_speaker_sample_rate() -> u32 {
    24000
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            mel_dim: default_mel_dim(),
            enc_dim: default_enc_dim(),
            enc_channels: default_enc_channels(),
            enc_kernel_sizes: default_enc_kernel_sizes(),
            enc_dilations: default_enc_dilations(),
            enc_attention_channels: default_enc_attention_channels(),
            enc_res2net_scale: default_enc_res2net_scale(),
            enc_se_channels: default_enc_se_channels(),
            sample_rate: default_speaker_sample_rate(),
        }
    }
}

// ──────────────────────────── 模型变体 ────────────────────────────

/// 模型变体类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// 声音克隆 (含 speaker encoder)
    Base,
    /// 9 预置说话人
    CustomVoice,
    /// 文本描述创建新声音
    VoiceDesign,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base => write!(f, "base"),
            Self::CustomVoice => write!(f, "custom_voice"),
            Self::VoiceDesign => write!(f, "voice_design"),
        }
    }
}

// ──────────────────────────── ParsedModelConfig ────────────────────────────

/// 从 config.json 解析的完整模型配置
#[derive(Debug, Clone)]
pub struct ParsedModelConfig {
    pub model_type: ModelType,
    pub model_size: String,

    // Talker
    pub talker_hidden_size: usize,
    pub talker_intermediate_size: usize,
    pub talker_num_hidden_layers: usize,
    pub talker_num_attention_heads: usize,
    pub talker_num_key_value_heads: usize,
    pub talker_head_dim: usize,
    pub talker_vocab_size: usize,
    pub talker_text_vocab_size: usize,
    pub talker_text_hidden_size: usize,
    pub talker_rms_norm_eps: f64,
    pub talker_rope_theta: f64,
    pub talker_max_position_embeddings: usize,
    pub mrope_section: Option<[usize; 3]>,

    // Code predictor
    pub cp_hidden_size: usize,
    pub cp_intermediate_size: usize,
    pub cp_num_hidden_layers: usize,
    pub cp_num_attention_heads: usize,
    pub cp_num_key_value_heads: usize,
    pub cp_head_dim: usize,
    pub cp_vocab_size: usize,
    pub cp_num_code_groups: usize,
    pub cp_rms_norm_eps: f64,
    pub cp_rope_theta: f64,

    // Speaker encoder
    pub speaker_encoder_config: Option<SpeakerEncoderConfig>,
}

impl ParsedModelConfig {
    /// 从 config.json 文件解析
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config from {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config from {}", path.display()))?;

        let model_type = match v["tts_model_type"].as_str().unwrap_or("base") {
            "custom_voice" => ModelType::CustomVoice,
            "voice_design" => ModelType::VoiceDesign,
            _ => ModelType::Base,
        };

        let model_size = v["tts_model_size"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        let t = &v["talker_config"];
        let cp = &t["code_predictor_config"];

        let talker_hidden_size = t["hidden_size"].as_u64().unwrap_or(1024) as usize;
        let talker_intermediate_size = t["intermediate_size"].as_u64().unwrap_or(3072) as usize;
        let talker_num_hidden_layers = t["num_hidden_layers"].as_u64().unwrap_or(28) as usize;
        let talker_num_attention_heads = t["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        let talker_num_key_value_heads = t["num_key_value_heads"].as_u64().unwrap_or(8) as usize;
        let talker_head_dim = t["head_dim"].as_u64().unwrap_or(128) as usize;
        let talker_vocab_size = t["vocab_size"].as_u64().unwrap_or(3072) as usize;
        let talker_text_vocab_size = t["text_vocab_size"].as_u64().unwrap_or(151936) as usize;
        let talker_text_hidden_size = t["text_hidden_size"].as_u64().unwrap_or(2048) as usize;
        let talker_rms_norm_eps = t["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let talker_rope_theta = t["rope_theta"].as_f64().unwrap_or(1_000_000.0);
        let talker_max_position_embeddings =
            t["max_position_embeddings"].as_u64().unwrap_or(32768) as usize;

        let mrope_section = t["rope_scaling"]["mrope_section"]
            .as_array()
            .and_then(|arr| {
                if arr.len() == 3 {
                    Some([
                        arr[0].as_u64()? as usize,
                        arr[1].as_u64()? as usize,
                        arr[2].as_u64()? as usize,
                    ])
                } else {
                    None
                }
            });

        let cp_hidden_size = cp["hidden_size"].as_u64().unwrap_or(1024) as usize;
        let cp_intermediate_size = cp["intermediate_size"].as_u64().unwrap_or(3072) as usize;
        let cp_num_hidden_layers = cp["num_hidden_layers"].as_u64().unwrap_or(5) as usize;
        let cp_num_attention_heads = cp["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        let cp_num_key_value_heads = cp["num_key_value_heads"].as_u64().unwrap_or(8) as usize;
        let cp_head_dim = cp["head_dim"].as_u64().unwrap_or(128) as usize;
        let cp_vocab_size = cp["vocab_size"].as_u64().unwrap_or(2048) as usize;
        let cp_num_code_groups = cp["num_code_groups"].as_u64().unwrap_or(16) as usize;
        let cp_rms_norm_eps = cp["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let cp_rope_theta = cp["rope_theta"].as_f64().unwrap_or(1_000_000.0);

        let speaker_encoder_config = if v["speaker_encoder_config"].is_object() {
            let se = &v["speaker_encoder_config"];
            Some(SpeakerEncoderConfig {
                enc_dim: se["enc_dim"].as_u64().unwrap_or(1024) as usize,
                sample_rate: se["sample_rate"].as_u64().unwrap_or(24000) as u32,
                ..Default::default()
            })
        } else {
            None
        };

        Ok(Self {
            model_type,
            model_size,
            talker_hidden_size,
            talker_intermediate_size,
            talker_num_hidden_layers,
            talker_num_attention_heads,
            talker_num_key_value_heads,
            talker_head_dim,
            talker_vocab_size,
            talker_text_vocab_size,
            talker_text_hidden_size,
            talker_rms_norm_eps,
            talker_rope_theta,
            talker_max_position_embeddings,
            mrope_section,
            cp_hidden_size,
            cp_intermediate_size,
            cp_num_hidden_layers,
            cp_num_attention_heads,
            cp_num_key_value_heads,
            cp_head_dim,
            cp_vocab_size,
            cp_num_code_groups,
            cp_rms_norm_eps,
            cp_rope_theta,
            speaker_encoder_config,
        })
    }

    /// 人类可读标签，如 "0.6B Base"
    pub fn label(&self) -> String {
        let size = match self.model_size.as_str() {
            "0b6" => "0.6B",
            "1b7" => "1.7B",
            other => other,
        };
        let variant = match self.model_type {
            ModelType::Base => "Base",
            ModelType::CustomVoice => "CustomVoice",
            ModelType::VoiceDesign => "VoiceDesign",
        };
        format!("{} {}", size, variant)
    }

    /// 构建 TalkerConfig
    pub fn to_talker_config(&self) -> TalkerConfig {
        TalkerConfig {
            text_vocab_size: self.talker_text_vocab_size,
            text_embed_dim: self.talker_text_hidden_size,
            hidden_size: self.talker_hidden_size,
            text_proj_intermediate: self.talker_text_hidden_size,
            intermediate_size: self.talker_intermediate_size,
            num_hidden_layers: self.talker_num_hidden_layers,
            num_attention_heads: self.talker_num_attention_heads,
            num_key_value_heads: self.talker_num_key_value_heads,
            head_dim: self.talker_head_dim,
            rms_norm_eps: self.talker_rms_norm_eps,
            rope_theta: self.talker_rope_theta,
            max_position_embeddings: self.talker_max_position_embeddings,
            codec_vocab_size: self.talker_vocab_size,
            mrope_section: self.mrope_section,
        }
    }

    /// 构建 CodePredictorConfig
    pub fn to_code_predictor_config(&self) -> CodePredictorConfig {
        CodePredictorConfig {
            hidden_size: self.cp_hidden_size,
            intermediate_size: self.cp_intermediate_size,
            num_hidden_layers: self.cp_num_hidden_layers,
            num_attention_heads: self.cp_num_attention_heads,
            num_key_value_heads: self.cp_num_key_value_heads,
            head_dim: self.cp_head_dim,
            vocab_size: self.cp_vocab_size,
            num_code_groups: self.cp_num_code_groups,
            rms_norm_eps: self.cp_rms_norm_eps,
            rope_theta: self.cp_rope_theta,
            codec_embed_dim: if self.talker_hidden_size != self.cp_hidden_size {
                Some(self.talker_hidden_size)
            } else {
                None
            },
        }
    }
}
