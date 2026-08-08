//! 解码器配置 — 从 `speech_tokenizer/config.json` 的 `decoder_config` 解析

/// 解码器配置 (从 `speech_tokenizer/config.json` 的 `decoder_config` 解析)
#[derive(Debug, Clone)]
pub(crate) struct DecoderConfig {
    /// latent 维度 (1024)
    pub latent_dim: usize,
    /// codebook 维度 (512)
    pub codebook_dim: usize,
    /// codebook 大小 (2048)
    pub codebook_size: usize,
    /// decoder 初始通道数 (1536)
    pub decoder_dim: usize,
    /// transformer hidden_size (512)
    pub hidden_size: usize,
    /// transformer intermediate_size (1024)
    pub intermediate_size: usize,
    /// transformer 层数 (8)
    pub num_hidden_layers: usize,
    /// 注意力头数 (16)
    pub num_attention_heads: usize,
    /// KV 头数 (16)
    pub num_key_value_heads: usize,
    /// head 维度 (64)
    pub head_dim: usize,
    /// 上采样率 [8, 5, 4, 3]
    pub upsample_rates: Vec<usize>,
    /// 上采样比率 [2, 2] (ConvTranspose1d 阶段)
    pub upsampling_ratios: Vec<usize>,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// 滑动窗口 (72)
    pub sliding_window: usize,
    /// 量化器数量 (16)
    #[allow(dead_code)]
    pub num_quantizers: usize,
    /// VQ 隐藏维度 (256)
    pub vq_hidden_dim: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            latent_dim: 1024,
            codebook_dim: 512,
            codebook_size: 2048,
            decoder_dim: 1536,
            hidden_size: 512,
            intermediate_size: 1024,
            num_hidden_layers: 8,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 64,
            upsample_rates: vec![8, 5, 4, 3],
            upsampling_ratios: vec![2, 2],
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            sliding_window: 72,
            num_quantizers: 16,
            vq_hidden_dim: 256,
        }
    }
}
