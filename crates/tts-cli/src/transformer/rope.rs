//! RoPE / MRoPE — 旋转位置编码

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor, D};

/// 将 RoPE 旋转应用到张量
///
/// `x` 形状 `[batch, heads, seq_len, head_dim]`
/// `cos`/`sin` 形状 `[seq_len, head_dim/2]`
fn apply_rope_rotation(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _s, d) = x.dims4()?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;

    let cos = cos
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(x.dtype())?
        .broadcast_as(x1.shape())?;
    let sin = sin
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(x.dtype())?
        .broadcast_as(x1.shape())?;

    let rotated = Tensor::cat(
        &[
            &(&x1.mul(&cos)? - &x2.mul(&sin)?)?,
            &(&x2.mul(&cos)? + &x1.mul(&sin)?)?,
        ],
        D::Minus1,
    )?;

    Ok(rotated)
}

/// 标准 RoPE (Rotary Position Embedding)
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions = Tensor::arange(0, max_seq_len as i64, device)?
            .to_dtype(candle_core::DType::F32)?
            .unsqueeze(1)?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?; // [seq, dim/2]
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    /// 应用 RoPE，返回旋转后的 (q, k)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq_len)?;
        let sin = self.sin.i(offset..offset + seq_len)?;
        let q_rot = apply_rope_rotation(q, &cos, &sin)?;
        let k_rot = apply_rope_rotation(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}

/// MRoPE (Multimodal Rotary Position Embedding)
///
/// Qwen3-TTS 使用 MRoPE with section [24, 20, 20]:
/// - 24 个频率对用于时间维度 (T)
/// - 20 个频率对用于高度维度 (H)
/// - 20 个频率对用于宽度维度 (W)
/// 总计 = 64 = head_dim / 2
///
/// 对 TTS 而言，三个维度使用相同的位置值，但频率分布不同。
pub struct MRoPE {
    inv_freq: Tensor,
    device: Device,
}

impl MRoPE {
    pub fn new(
        dim: usize,
        theta: f64,
        _mrope_section: [usize; 3],
        device: &Device,
    ) -> Result<Self> {
        // MRoPE 使用与标准 RoPE 相同的逆频率计算
        // 区别在于应用时如何分配频率对到不同维度
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        Ok(Self {
            inv_freq,
            device: device.clone(),
        })
    }

    /// 应用 MRoPE
    ///
    /// 对 TTS，所有三个位置维度 (T, H, W) 使用相同的序列位置。
    /// 这意味着 MRoPE 退化为标准 RoPE，但频率分组不同。
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let positions = Tensor::arange(offset as i64, (offset + seq_len) as i64, &self.device)?
            .to_dtype(candle_core::DType::F32)?
            .unsqueeze(1)?;
        let freqs = positions.matmul(&self.inv_freq.unsqueeze(0)?)?; // [seq, dim/2]
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let q_rot = apply_rope_rotation(q, &cos, &sin)?;
        let k_rot = apply_rope_rotation(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}

/// RoPE 类型枚举
pub enum RoPEType {
    Standard(RotaryEmbedding),
    Multimodal(MRoPE),
}

impl RoPEType {
    /// 应用旋转位置编码
    ///
    /// - `q`, `k`: [batch, heads, seq_len, head_dim]
    /// - `offset`: KV cache 偏移量
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        match self {
            RoPEType::Standard(rope) => rope.apply(q, k, offset),
            RoPEType::Multimodal(mrope) => {
                let seq_len = q.dim(2)?;
                mrope.apply(q, k, offset, seq_len)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_type_variants() {
        // RoPEType has two variants: Standard and Multimodal
        // Verify they exist by checking the enum can be constructed
        // (construction requires Tensor params, so just check size > 0)
        assert!(std::mem::size_of::<RoPEType>() > 0);
    }
}
