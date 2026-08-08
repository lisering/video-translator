//! DecoderLayer — Transformer 解码器层

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{Module, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

use crate::model_config::Qwen3TTSConfig;
use crate::transformer::attention::Attention;
use crate::transformer::kv_cache::AnyKVCache;
use crate::transformer::mlp::Mlp;
use crate::transformer::rope::RoPEType;

/// Transformer 解码器层
///
/// ```text
/// h = x + self_attn(input_layernorm(x))
/// out = h + mlp(post_attention_layernorm(h))
/// ```
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(
        config: &Qwen3TTSConfig,
        vb: VarBuilder,
        quantize: Option<GgmlDType>,
    ) -> Result<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"), quantize)?;
        let mlp = Mlp::new(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("mlp"),
            quantize,
        )?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// 前向推理
    ///
    /// - `hidden_states`: [batch, seq_len, hidden_size]
    /// - `rope`: 旋转位置编码
    /// - `kv_cache`: 可选 KV 缓存
    /// - `offset`: 序列偏移
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RoPEType,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
        causal: bool,
    ) -> Result<Tensor> {
        // Self-attention
        let residual = hidden_states;
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_out = self
            .self_attn
            .forward(&normed, rope, kv_cache, offset, causal)?;
        let hidden = (residual + attn_out)?;

        // MLP
        let residual = &hidden;
        let normed = self.post_attention_layernorm.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let out = (residual + mlp_out)?;

        Ok(out)
    }
}
