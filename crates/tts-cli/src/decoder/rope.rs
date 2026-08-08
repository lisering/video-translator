//! RoPE (旋转位置编码) — 解码器 PreTransformer 专用简化版

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};

use super::helpers::apply_rope_dec;

pub(crate) struct RotaryEmbeddingForDecoder {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbeddingForDecoder {
    pub(crate) fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions = Tensor::arange(0, max_seq_len as i64, device)?
            .to_dtype(candle_core::DType::F32)?
            .unsqueeze(1)?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    pub(crate) fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq_len)?;
        let sin = self.sin.i(offset..offset + seq_len)?;
        let q_rot = apply_rope_dec(q, &cos, &sin)?;
        let k_rot = apply_rope_dec(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}
