//! 解码器块 — ConvNeXtBlock, UpsampleBlock, DecoderConvBlock, DecoderUpsampleBlock

use std::time::Instant;

use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::conv::{Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use candle_nn::VarBuilder;
use rayon::prelude::*;

use super::conv_ops::{FastConv1d, FastConvTranspose1d};
use super::helpers::{affine_norm_fused_cpu, gelu, SnakeBeta};
use candle_core::DType;

// ──────────────────────────── ConvNeXt block (upsample) ────────────────────────────

pub(crate) struct ConvNeXtBlock {
    /// 深度卷积 — CausalConvNet(k=7, groups=dim, dilation=1): left_pad=6, right_pad=0
    dwconv: FastConv1d,
    /// LayerNorm 权重 (手动 dim-1 归一化，避免 transpose+contiguous)
    norm_weight: Tensor,
    norm_bias: Tensor,
    /// 1×1 Conv1d 替代 Linear — 使用 FastConv1d (CPU: 直接 matmul, GPU: Candle Conv1d)
    pwconv1: FastConv1d,
    pwconv2: FastConv1d,
    /// Pre-reshaped to [1, C, 1]
    gamma: Tensor,
    eps: f64,
}

impl ConvNeXtBlock {
    pub(crate) fn new(channels: usize, hidden_mult: usize, vb: VarBuilder) -> Result<Self> {
        // CausalConvNet: k=7, dilation=1 → left_pad = (7-1)*1 = 6
        // Conv1dConfig padding=6 (symmetric), crop right 6 after forward for causal
        let dw_weight = vb.get((channels, 1, 7), "dwconv.conv.weight")?;
        let dw_bias = vb.get(channels, "dwconv.conv.bias")?;
        let dwconv = FastConv1d::new(
            dw_weight,
            Some(dw_bias),
            Conv1dConfig {
                padding: 6,
                groups: channels,
                ..Default::default()
            },
            vb.device(),
        )?;

        // LayerNorm 权重 — 手动 dim-1 归一化，避免 transpose
        // Python uses eps=1e-6 for ConvNeXtBlock.norm
        let norm_weight = vb.get(channels, "norm.weight")?;
        let norm_bias = vb.get(channels, "norm.bias")?;

        // 1×1 Conv1d 替代 Linear — 使用 FastConv1d (CPU: 直接 matmul, GPU: Candle Conv1d)
        // 权重从 [out, in] reshape 为 [out, in, 1] (零拷贝元数据变更)
        let hidden_dim = channels * hidden_mult;
        let pw1_weight = vb
            .get((hidden_dim, channels), "pwconv1.weight")?
            .reshape((hidden_dim, channels, 1))?;
        let pw1_bias = vb.get(hidden_dim, "pwconv1.bias")?;
        let pwconv1 = FastConv1d::new(
            pw1_weight,
            Some(pw1_bias),
            Conv1dConfig::default(),
            vb.device(),
        )?;

        let pw2_weight = vb
            .get((channels, hidden_dim), "pwconv2.weight")?
            .reshape((channels, hidden_dim, 1))?;
        let pw2_bias = vb.get(channels, "pwconv2.bias")?;
        let pwconv2 = FastConv1d::new(
            pw2_weight,
            Some(pw2_bias),
            Conv1dConfig::default(),
            vb.device(),
        )?;

        // gamma 预 reshape
        let gamma = vb.get(channels, "gamma")?.reshape((1, (), 1))?;

        Ok(Self {
            dwconv,
            norm_weight,
            norm_bias,
            pwconv1,
            pwconv2,
            gamma,
            eps: 1e-6,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x; // [batch, C, L]
        let l_in = x.dim(2)?;
        // CausalConvNet: padding=6 (symmetric), crop right 6 for causal
        let x = self.dwconv.forward(x)?; // [batch, C, L+6]
        let x = x.narrow(2, 0, l_in)?.contiguous()?; // crop right → [batch, C, L]

        // LayerNorm (dim-1, 通道维度) — CPU: 融合原始 Vec + Rayon; GPU: Candle ops
        let x = if x.device().is_cpu() && x.dtype() == DType::F32 {
            let nw = self.norm_weight.flatten_all()?.to_vec1::<f32>()?;
            let nb = self.norm_bias.flatten_all()?.to_vec1::<f32>()?;
            affine_norm_fused_cpu(&x, &nw, &nb, self.eps)?
        } else {
            let dims = x.dims();
            let c = dims[1];
            let mean = (x.sum_keepdim(1)? / c as f64)?;
            let centered = x.broadcast_sub(&mean)?;
            let var = (centered.sqr()?.sum_keepdim(1)? / c as f64)?;
            let normed = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
            let nw = self
                .norm_weight
                .reshape((1, (), 1))?
                .to_dtype(normed.dtype())?
                .broadcast_as(normed.shape())?;
            let nb = self
                .norm_bias
                .reshape((1, (), 1))?
                .to_dtype(normed.dtype())?
                .broadcast_as(normed.shape())?;
            normed.broadcast_mul(&nw)?.broadcast_add(&nb)?
        };

        // 1×1 Conv1d — 无需 transpose，直接在 [batch, C, L] 上操作
        let x = self.pwconv1.forward(&x)?; // [batch, 4C, L]
        let x = gelu(&x)?; // [batch, 4C, L]
        let x = self.pwconv2.forward(&x)?; // [batch, C, L]

        // gamma — 已预 reshape
        let gamma = self.gamma.to_dtype(x.dtype())?.broadcast_as(x.shape())?;
        let x = x.broadcast_mul(&gamma)?;
        Ok((residual + x)?)
    }
}

// ──────────────────────────── Upsample block ────────────────────────────

pub(crate) struct UpsampleBlock {
    conv_transpose: ConvTranspose1d,
    convnext: ConvNeXtBlock,
}

impl UpsampleBlock {
    pub(crate) fn new(channels: usize, ratio: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((channels, channels, ratio), "0.conv.weight")?;
        let bias = vb.get(channels, "0.conv.bias")?;
        let conv_transpose = ConvTranspose1d::new(
            weight,
            Some(bias),
            ConvTranspose1dConfig {
                stride: ratio,
                ..Default::default()
            },
        );
        let convnext = ConvNeXtBlock::new(channels, 4, vb.pp("1"))?;
        Ok(Self {
            conv_transpose,
            convnext,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose.forward(x)?;
        self.convnext.forward(&x)
    }
}

// ──────────────────────────── decoder.decoder ConvNeXt-like block ────────────────────────────

/// DecoderDecoderResidualUnit — matches Python Qwen3TTSTokenizerV2DecoderDecoderResidualUnit.
///
/// Python structure:
///   act1 = SnakeBeta(dim)
///   conv1 = CausalConvNet(dim, dim, k=7, dilation=d)  ← causal padding = (k-1)*d
///   act2 = SnakeBeta(dim)
///   conv2 = CausalConvNet(dim, dim, k=1)
///
/// Our code uses SnakeBeta (NOT LearnedActivation!) and causal padding (NOT symmetric!).
pub(crate) struct DecoderConvBlock {
    /// SnakeBeta activation (NOT LearnedActivation!)
    act1: SnakeBeta,
    /// CausalConvNet(k=7, dilation=d): padding = 6*d, crop right 6*d
    conv1: FastConv1d,
    /// SnakeBeta activation (NOT LearnedActivation!)
    act2: SnakeBeta,
    /// k=1 Conv1d — 使用 FastConv1d (CPU: 直接 matmul, GPU: Candle Conv1d)
    conv2: FastConv1d,
    /// Dilation for conv1 (1, 3, or 9)
    dilation: usize,
}

impl DecoderConvBlock {
    pub(crate) fn new(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        // SnakeBeta activations (NOT LearnedActivation!)
        let act1 = SnakeBeta::from_vb(channels, vb.clone(), "act1")?;

        // CausalConvNet(k=7, dilation=d): left_pad = (7-1)*d = 6*d
        // Conv1dConfig padding = 6*d (symmetric), crop right 6*d after forward for causal
        let c1_weight = vb.get((channels, channels, 7), "conv1.conv.weight")?;
        let c1_bias = vb.get(channels, "conv1.conv.bias")?;
        let conv1 = FastConv1d::new(
            c1_weight,
            Some(c1_bias),
            Conv1dConfig {
                padding: 6 * dilation,
                dilation,
                ..Default::default()
            },
            vb.device(),
        )?;

        let act2 = SnakeBeta::from_vb(channels, vb.clone(), "act2")?;

        let c2_weight = vb.get((channels, channels, 1), "conv2.conv.weight")?;
        let c2_bias = vb.get(channels, "conv2.conv.bias")?;
        let conv2 = FastConv1d::new(
            c2_weight,
            Some(c2_bias),
            Conv1dConfig::default(),
            vb.device(),
        )?;

        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
            dilation,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let l_in = x.dim(2)?;
        let x = self.act1.forward(x)?;
        // CausalConvNet: padding=6*dilation (symmetric), crop right 6*dilation for causal
        let x = self.conv1.forward(&x)?;
        let x = x.narrow(2, 0, l_in)?.contiguous()?; // crop right → causal
        let x = self.act2.forward(&x)?;
        let x = self.conv2.forward(&x)?;
        Ok((residual + x)?)
    }

    /// CPU 融合 forward (Vec 输入): 保持数据在 Vec 空间, 消除 Tensor ↔ Vec 转换开销
    ///
    /// 链式调用: act1 → conv1 (with causal crop) → act2 → conv2 → residual add
    /// 所有操作在 Vec 空间完成, 仅 conv1/conv2 内部创建 Tensor 用于 BLAS matmul.
    #[allow(dead_code)]
    pub(crate) fn forward_fused_vec(
        &self,
        x: &[f32],
        c: usize,
        l: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        // act1: SnakeBeta &[f32] → Vec<f32>
        let h = self.act1.forward_vec(x, c, l)?;

        // conv1: CausalConvNet with dilation, padding=6*dilation
        // Output length = l + 6*dilation (symmetric padding adds 6*d on both sides,
        // conv with dilation reduces by (k-1)*d = 6*d, net = l + 6*d)
        let h = self.conv1.forward_vec(&h, c, l, device)?;
        // Crop right 6*dilation per channel (causal: keep first l samples)
        let crop = 6 * self.dilation;
        let l_out = l; // after crop, length is back to l
        let l_conv = l + crop; // conv1 output length per channel
        let mut h_cropped = vec![0.0f32; c * l_out];
        for ci in 0..c {
            let src_start = ci * l_conv;
            let dst_start = ci * l_out;
            h_cropped[dst_start..dst_start + l_out]
                .copy_from_slice(&h[src_start..src_start + l_out]);
        }

        // act2: SnakeBeta &[f32] → Vec<f32>
        let h = self.act2.forward_vec(&h_cropped, c, l_out)?;

        // conv2: k=1, no padding needed
        let mut h = self.conv2.forward_vec_owned(h, c, l_out, device)?;

        // residual add: x (still borrowed) + h (Vec, in-place)
        h.par_iter_mut().zip(x.par_iter()).for_each(|(o, &r)| {
            *o += r;
        });

        Ok(h)
    }
}

// ──────────────────────────── decoder.decoder upsample block ────────────────────────────

/// DecoderDecoderBlock — matches Python Qwen3TTSTokenizerV2DecoderDecoderBlock.
///
/// Python structure:
///   block = [
///     SnakeBeta(in_dim),                                    # block.0 — SnakeBeta (NOT LayerNorm!)
///     CausalTransConvNet(in_dim, out_dim, 2*stride, stride), # block.1 — no padding, crop right
///     DecoderDecoderResidualUnit(out_dim, dilation=1),       # block.2
///     DecoderDecoderResidualUnit(out_dim, dilation=3),       # block.3
///     DecoderDecoderResidualUnit(out_dim, dilation=9),       # block.4
///   ]
pub(crate) struct DecoderUpsampleBlock {
    /// SnakeBeta activation for block.0 (NOT LayerNorm!)
    snake_beta: SnakeBeta,
    /// CausalTransConvNet: ConvTranspose1d with padding=0, crop right (kernel-stride)
    conv_transpose: FastConvTranspose1d,
    /// 3 residual units with dilation 1, 3, 9
    conv_blocks: Vec<DecoderConvBlock>,
    /// Input channel count
    #[allow(dead_code)]
    in_channels: usize,
    /// Output channel count (for CPU fused Vec path — computing l_out from Vec length)
    pub(crate) out_channels: usize,
    /// ConvTranspose1d crop amount = kernel - stride
    conv_transpose_crop: usize,
}

impl DecoderUpsampleBlock {
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // block.0: SnakeBeta (NOT LayerNorm!)
        let snake_beta = SnakeBeta::from_vb(in_channels, vb.clone(), "block.0")?;

        // block.1: CausalTransConvNet — no padding, crop right (kernel - stride)
        // Python: ConvTranspose1d(in, out, kernel=2*stride, stride=stride), no padding
        //         then crop right (kernel - stride) = stride
        let kernel = stride * 2;
        let conv_transpose_crop = kernel - stride; // = stride
        let weight = vb.get((in_channels, out_channels, kernel), "block.1.conv.weight")?;
        let bias = vb.get(out_channels, "block.1.conv.bias")?;
        let conv_transpose = FastConvTranspose1d::new(
            weight,
            Some(bias),
            ConvTranspose1dConfig {
                stride,
                padding: 0,
                output_padding: 0,
                ..Default::default()
            },
            vb.device(),
        )?;

        // block.2-4: DecoderDecoderResidualUnit with dilation 1, 3, 9
        let dilations = [1usize, 3, 9];
        let conv_blocks = (2..=4)
            .zip(dilations.iter())
            .map(|(i, &dil)| DecoderConvBlock::new(out_channels, dil, vb.pp(format!("block.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            snake_beta,
            conv_transpose,
            conv_blocks,
            in_channels,
            out_channels,
            conv_transpose_crop,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // CPU fused Vec path: snake_beta → conv_transpose → conv_blocks, all in Vec space
        if x.device().is_cpu() && x.dtype() == DType::F32 {
            let (_batch, c_in, l_in) = x.dims3()?;
            let device = x.device();
            let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
            let out_vec = self.forward_vec_impl(x_vec, c_in, l_in, device)?;
            let c = self.out_channels;
            let l = out_vec.len() / c;
            Ok(Tensor::from_vec(out_vec, (1, c, l), device)?)
        } else {
            // GPU path: SnakeBeta + conv_transpose (with crop) + conv_blocks
            let x = self.snake_beta.forward(x)?;
            let x = self.conv_transpose.forward(&x)?;
            // Crop right (kernel - stride) for CausalTransConvNet
            let l_in = x.dim(2)?;
            let l_out = l_in - self.conv_transpose_crop;
            let x = x.narrow(2, 0, l_out)?.contiguous()?;
            let mut x = x;
            for block in &self.conv_blocks {
                x = block.forward(&x)?;
            }
            Ok(x)
        }
    }

    /// CPU 全 Vec 链式 forward: 接受 Vec<f32> 输入, 返回 Vec<f32> 输出
    #[allow(dead_code)]
    pub(crate) fn forward_vec(
        &self,
        x: Vec<f32>,
        c_in: usize,
        l_in: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        self.forward_vec_impl(x, c_in, l_in, device)
    }

    /// Shared implementation for forward (CPU) and forward_vec
    fn forward_vec_impl(
        &self,
        x: Vec<f32>,
        c_in: usize,
        l_in: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        // SnakeBeta: Vec → Vec (no Tensor roundtrip)
        let x = self.snake_beta.forward_vec(&x, c_in, l_in)?;

        // ConvTranspose1d — padding=0, output = (l_in-1)*stride + kernel
        let x = self.conv_transpose.forward_vec(x, c_in, l_in, device)?;

        // Crop right (kernel - stride) per channel for CausalTransConvNet
        let c = self.out_channels;
        let l_conv = x.len() / c; // = (l_in-1)*stride + kernel
        let l_out = l_conv - self.conv_transpose_crop; // = l_in * stride
        let mut x_cropped = vec![0.0f32; c * l_out];
        for ci in 0..c {
            let src_start = ci * l_conv;
            let dst_start = ci * l_out;
            x_cropped[dst_start..dst_start + l_out]
                .copy_from_slice(&x[src_start..src_start + l_out]);
        }

        // ConvNeXt-like blocks (residual) — fused Vec path
        let mut x = x_cropped;
        for block in &self.conv_blocks {
            x = block.forward_fused_vec(&x, c, l_out, device)?;
        }
        Ok(x)
    }

    /// 带细粒度计时的 Vec forward — 用于准确瓶颈分析 (生产路径)
    #[allow(dead_code)]
    pub(crate) fn forward_timed_vec(
        &self,
        x: Vec<f32>,
        c_in: usize,
        l_in: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        // SnakeBeta: Vec → Vec
        let t1 = Instant::now();
        let x_vec = self.snake_beta.forward_vec(&x, c_in, l_in)?;
        let t_norm = t1.elapsed();

        // conv_transpose: Vec → Vec (padding=0)
        let t2 = Instant::now();
        let x_vec = self.conv_transpose.forward_vec(x_vec, c_in, l_in, device)?;
        let t_conv_t = t2.elapsed();

        // Crop right (kernel - stride) per channel
        let c = self.out_channels;
        let l_conv = x_vec.len() / c;
        let l = l_conv - self.conv_transpose_crop;
        let mut x_cropped = vec![0.0f32; c * l];
        for ci in 0..c {
            let src_start = ci * l_conv;
            let dst_start = ci * l;
            x_cropped[dst_start..dst_start + l].copy_from_slice(&x_vec[src_start..src_start + l]);
        }
        let mut x_vec = x_cropped;

        // Per-operation timed conv_blocks (all in Vec space)
        for (i, block) in self.conv_blocks.iter().enumerate() {
            // act1: SnakeBeta &[f32] → Vec<f32>
            let ta = Instant::now();
            let h = block.act1.forward_vec(&x_vec, c, l)?;
            let t_act1 = ta.elapsed();

            // conv1: CausalConvNet with dilation
            let tb2 = Instant::now();
            let h = block.conv1.forward_vec(&h, c, l, device)?;
            let t_conv1 = tb2.elapsed();

            // Crop right 6*dilation per channel
            let crop = 6 * block.dilation;
            let l_conv1 = l + crop;
            let mut h_cropped = vec![0.0f32; c * l];
            for ci in 0..c {
                let src_start = ci * l_conv1;
                let dst_start = ci * l;
                h_cropped[dst_start..dst_start + l].copy_from_slice(&h[src_start..src_start + l]);
            }

            // act2: SnakeBeta &[f32] → Vec<f32>
            let tc = Instant::now();
            let h = block.act2.forward_vec(&h_cropped, c, l)?;
            let t_act2 = tc.elapsed();

            // conv2: k=1, no padding
            let td = Instant::now();
            let mut h = block.conv2.forward_vec_owned(h, c, l, device)?;
            let t_conv2 = td.elapsed();

            // residual add: in-place Rayon parallel
            let te = Instant::now();
            h.par_iter_mut().zip(x_vec.par_iter()).for_each(|(o, &r)| {
                *o += r;
            });
            let t_add = te.elapsed();

            x_vec = h;

            tracing::info!(
                "    conv_block {}: act1={:.1}ms, conv1={:.1}ms, act2={:.1}ms, conv2={:.1}ms, add={:.1}ms",
                i,
                t_act1.as_secs_f64() * 1000.0,
                t_conv1.as_secs_f64() * 1000.0,
                t_act2.as_secs_f64() * 1000.0,
                t_conv2.as_secs_f64() * 1000.0,
                t_add.as_secs_f64() * 1000.0,
            );
        }

        tracing::info!(
            "  block timing (Vec): snake_beta={:.1}ms, conv_transpose={:.1}ms",
            t_norm.as_secs_f64() * 1000.0,
            t_conv_t.as_secs_f64() * 1000.0,
        );

        Ok(x_vec)
    }

    /// 带细粒度计时的 forward — Tensor 包装版, 用于 GPU 或非 Vec 链式路径
    #[allow(dead_code)]
    pub(crate) fn forward_timed(&self, x: &Tensor) -> Result<Tensor> {
        // CPU fused Vec timed path
        if x.device().is_cpu() && x.dtype() == DType::F32 {
            let (_batch, c_in, l_in) = x.dims3()?;
            let device = x.device();
            let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
            let out_vec = self.forward_timed_vec(x_vec, c_in, l_in, device)?;
            let c = self.out_channels;
            let l = out_vec.len() / c;
            Ok(Tensor::from_vec(out_vec, (1, c, l), device)?)
        } else {
            // GPU fallback: SnakeBeta + conv_transpose (with crop) + conv_blocks
            let x = self.snake_beta.forward(x)?;
            let x = self.conv_transpose.forward(&x)?;
            let l_in = x.dim(2)?;
            let l_out = l_in - self.conv_transpose_crop;
            let x = x.narrow(2, 0, l_out)?.contiguous()?;
            let mut x = x;
            for block in &self.conv_blocks {
                x = block.forward(&x)?;
            }
            Ok(x)
        }
    }
}
