//! Attention — 多头注意力 + GQA + QK 归一化

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{Module, Tensor, D};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

use crate::model_config::Qwen3TTSConfig;
use crate::transformer::kv_cache::AnyKVCache;
use crate::transformer::qlinear::QLinear;
use crate::transformer::rope::RoPEType;

/// 多头注意力 + GQA + QK 归一化
///
/// 参考 Qwen3-TTS 的 Attention 实现：
/// - Q/K 在投影后进行 per-head RMSNorm
/// - 支持 grouped-query attention (num_kv_heads < num_heads)
/// - 使用 RoPE 旋转位置编码
/// - **QKV 融合**: 将 q_proj + k_proj + v_proj 合并为单次 matmul，减少 kernel launch
/// - **权重量化**: 可选 Q8_0/Q4_0/Q4K 量化，减少内存带宽
pub struct Attention {
    /// 融合 QKV 投影: [hidden, q_dim + 2*kv_dim] (可选量化)
    qkv_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Q 维度 = num_heads * head_dim
    q_dim: usize,
    /// KV 维度 = num_kv_heads * head_dim
    kv_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn new(
        config: &Qwen3TTSConfig,
        vb: VarBuilder,
        quantize: Option<GgmlDType>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // 融合 QKV: 加载三个权重并在 dim 0 拼接
        // 权重形状: [out_dim, in_dim]，拼接后 [q_dim + 2*kv_dim, hidden]
        let q_weight = vb.pp("q_proj").get((q_dim, hidden_size), "weight")?;
        let k_weight = vb.pp("k_proj").get((kv_dim, hidden_size), "weight")?;
        let v_weight = vb.pp("v_proj").get((kv_dim, hidden_size), "weight")?;
        let qkv_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
        let qkv_proj = QLinear::from_weight(qkv_weight, quantize)?;

        let o_weight = vb.pp("o_proj").get((hidden_size, q_dim), "weight")?;
        let o_proj = QLinear::from_weight(o_weight, quantize)?;

        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim,
            kv_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// 前向推理
    ///
    /// - `hidden_states`: [batch, seq_len, hidden_size]
    /// - `rope`: 旋转位置编码
    /// - `kv_cache`: KV 缓存
    /// - `offset`: 序列偏移量
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RoPEType,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
        causal: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // 融合 QKV 投影: 单次 matmul 替代 3 次
        let qkv = self.qkv_proj.forward(hidden_states)?; // [batch, seq, q_dim + 2*kv_dim]
        let q = qkv.narrow(D::Minus1, 0, self.q_dim)?;
        let k = qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
        let v = qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;

        // Reshape to [batch, seq_len, heads, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // Transpose to [batch, heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // QK normalization (per-head RmsNorm)
        // q_norm/k_norm 的维度是 head_dim，对最后一维归一化
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // 应用 RoPE
        let (q, k) = rope.apply(&q, &k, offset)?;

        // 更新 KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k, v)
        };

        // GQA: 如果 num_kv_heads < num_heads，需要 repeat KV
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

        // Attention: softmax(Q @ K^T * scale) @ V
        let att = q.matmul(&k.transpose(2, 3)?)?; // [batch, heads, seq_len, kv_len]
        let att = (att * self.scale)?;

        // 因果遮罩: 当 causal=true 且 seq_len > 1 时，遮蔽未来位置
        // 用于自回归生成 (TalkerModel)，确保位置 i 只看到位置 <= i
        // CodePredictor 使用双向注意力 (causal=false)
        let att = if causal && seq_len > 1 {
            let kv_len = att.dim(3)?;
            let prev_kv_len = kv_len - seq_len; // KV cache 中已有的条目数
                                                // mask[i, j] = 0 if j <= prev_kv_len + i, -inf otherwise
            let mut mask_vals = vec![0.0f32; seq_len * kv_len];
            for i in 0..seq_len {
                for j in (prev_kv_len + i + 1)..kv_len {
                    mask_vals[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::new(mask_vals.as_slice(), &att.device())?
                .reshape((1, 1, seq_len, kv_len))?
                .to_dtype(att.dtype())?
                .broadcast_as(att.shape())?;
            att.broadcast_add(&mask)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;

        let out = att.matmul(&v)?; // [batch, heads, seq_len, head_dim]

        // Transpose back and reshape
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // 输出投影
        Ok(self.o_proj.forward(&out)?)
    }
}

/// GQA: 将 KV heads 重复以匹配 Q heads
///
/// 输入: [batch, num_kv_heads, seq_len, head_dim]
/// 输出: [batch, num_heads, seq_len, head_dim]
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x.unsqueeze(2)?; // [batch, num_kv_heads, 1, seq_len, head_dim]
    let x = x.broadcast_as((batch, num_kv_heads, n_rep, seq_len, head_dim))?;
    let x = x.reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;
    Ok(x)
}
