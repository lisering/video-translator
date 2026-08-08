//! PreTransformer — 8 层 Transformer (hidden=512, SwiGLU, layer_scale, 滑动窗口注意力)

use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{linear, linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use super::config::DecoderConfig;
use super::helpers::{repeat_kv, LayerScale};
use super::rope::RotaryEmbeddingForDecoder;

// ──────────────────────────── pre_transformer 层 ────────────────────────────

pub(crate) struct PreTransformerLayer {
    input_layernorm: RmsNorm,
    /// 融合 QKV 投影: [hidden, q_dim + 2*kv_dim]
    self_attn_qkv: Linear,
    self_attn_o: Linear,
    attn_layer_scale: LayerScale,
    post_attention_layernorm: RmsNorm,
    /// 融合 gate+up 投影: [hidden, 2*intermediate]
    mlp_gate_up: Linear,
    mlp_down: Linear,
    mlp_layer_scale: LayerScale,
    rope: RotaryEmbeddingForDecoder,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Q 维度 = num_heads * head_dim
    q_dim: usize,
    /// KV 维度 = num_kv_heads * head_dim
    kv_dim: usize,
    /// MLP 中间层维度
    intermediate_size: usize,
    scale: f64,
    /// 滑动窗口大小 (0 = 全注意力, >0 = 限制每个位置只看到 ±sw/2 范围)
    sliding_window: usize,
}

impl PreTransformerLayer {
    pub(crate) fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let inter = config.intermediate_size;
        let eps = config.rms_norm_eps;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let input_layernorm = rms_norm(hidden, eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(hidden, eps, vb.pp("post_attention_layernorm"))?;

        // 融合 QKV: 拼接三个 [out_dim, hidden] 权重为 [q_dim + 2*kv_dim, hidden]
        let q_weight = vb.pp("self_attn.q_proj").get((q_dim, hidden), "weight")?;
        let k_weight = vb.pp("self_attn.k_proj").get((kv_dim, hidden), "weight")?;
        let v_weight = vb.pp("self_attn.v_proj").get((kv_dim, hidden), "weight")?;
        let qkv_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
        let self_attn_qkv = Linear::new(qkv_weight, None);

        let self_attn_o = linear_no_bias(q_dim, hidden, vb.pp("self_attn.o_proj"))?;

        // 融合 gate+up: 拼接两个 [inter, hidden] 权重为 [2*inter, hidden]
        let gate_weight = vb.pp("mlp.gate_proj").get((inter, hidden), "weight")?;
        let up_weight = vb.pp("mlp.up_proj").get((inter, hidden), "weight")?;
        let gate_up_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
        let mlp_gate_up = Linear::new(gate_up_weight, None);

        let mlp_down = linear_no_bias(inter, hidden, vb.pp("mlp.down_proj"))?;

        let attn_layer_scale = LayerScale::new(hidden, vb.clone(), "self_attn_layer_scale")?;
        let mlp_layer_scale = LayerScale::new(hidden, vb.clone(), "mlp_layer_scale")?;

        let rope = RotaryEmbeddingForDecoder::new(head_dim, 8000, config.rope_theta, vb.device())?;

        Ok(Self {
            input_layernorm,
            self_attn_qkv,
            self_attn_o,
            attn_layer_scale,
            post_attention_layernorm,
            mlp_gate_up,
            mlp_down,
            mlp_layer_scale,
            rope,
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            q_dim,
            kv_dim,
            intermediate_size: inter,
            scale: 1.0 / (head_dim as f64).sqrt(),
            sliding_window: config.sliding_window,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Self-attention (non-causal / bidirectional)
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;

        // 融合 QKV: 单次 matmul 替代 3 次
        let qkv = self.self_attn_qkv.forward(&normed)?; // [batch, seq, q_dim + 2*kv_dim]
        let q = qkv.narrow(D::Minus1, 0, self.q_dim)?;
        let k = qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
        let v = qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;

        // Metal matmul 要求连续张量，保留 .contiguous()
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rope.apply(&q, &k, 0)?;

        let k = if self.num_kv_heads != self.num_heads {
            repeat_kv(&k, self.num_heads / self.num_kv_heads)?
        } else {
            k
        };
        let v = if self.num_kv_heads != self.num_heads {
            repeat_kv(&v, self.num_heads / self.num_kv_heads)?
        } else {
            v
        };

        // Metal matmul 要求连续张量
        let att = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let att = (att * self.scale)?;

        // 滑动窗口注意力 (bidirectional)
        // 当 seq_len > sliding_window 时，限制每个位置只看到 ±sw/2 范围
        // 模型配置中 sliding_window=72，训练时使用了滑动窗口注意力
        let att = if self.sliding_window > 0 && seq_len > self.sliding_window {
            let half_sw = self.sliding_window / 2;
            let mut mask_vals = vec![f32::NEG_INFINITY; seq_len * seq_len];
            for i in 0..seq_len {
                let start = i.saturating_sub(half_sw);
                let end = (i + half_sw + 1).min(seq_len);
                for j in start..end {
                    mask_vals[i * seq_len + j] = 0.0;
                }
            }
            let mask = Tensor::new(mask_vals.as_slice(), &att.device())?
                .reshape((1, 1, seq_len, seq_len))?
                .to_dtype(att.dtype())?
                .broadcast_as(att.shape())?;
            att.broadcast_add(&mask)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;

        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        let out = self.self_attn_o.forward(&out)?;
        let out = self.attn_layer_scale.forward(&out)?;
        let hidden = (residual + out)?;

        // MLP (SwiGLU) — 融合 gate+up: 单次 matmul 替代 2 次
        let residual = &hidden;
        let normed = self.post_attention_layernorm.forward(&hidden)?;
        let gate_up = self.mlp_gate_up.forward(&normed)?; // [batch, seq, 2*inter]
        let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let prod = gate.broadcast_mul(&up)?;
        let out = self.mlp_down.forward(&prod)?;
        let out = self.mlp_layer_scale.forward(&out)?;

        Ok((residual + out)?)
    }
}

// ──────────────────────────── pre_transformer ────────────────────────────

pub(crate) struct PreTransformer {
    input_proj: Linear,
    layers: Vec<PreTransformerLayer>,
    norm: RmsNorm,
    output_proj: Linear,
}

impl PreTransformer {
    pub(crate) fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let input_proj = linear(config.latent_dim, config.hidden_size, vb.pp("input_proj"))?;
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                PreTransformerLayer::new(config, vb.pp(format!("layers.{i}")))
                    .map_err(|e| anyhow::anyhow!("PreTransformer layer {i} init failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let output_proj = linear(config.hidden_size, config.latent_dim, vb.pp("output_proj"))?;
        Ok(Self {
            input_proj,
            layers,
            norm,
            output_proj,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, _t) = x.dims3()?;
        let x = x.transpose(1, 2)?;
        let x = self.input_proj.forward(&x)?;
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        let x = self.norm.forward(&x)?;
        let x = self.output_proj.forward(&x)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        Ok(x)
    }
}
