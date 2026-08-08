//! Conv1d / ConvTranspose1d 优化实现
//!
//! - `MatmulConv1d`: CPU 优化的 Conv1d (im2col + BLAS matmul) — kernel > 1
//! - `MatmulConv1dK1`: CPU 优化的 Conv1d (直接 matmul) — kernel = 1
//! - `FastConv1d`: 枚举自动选择 CPU Matmul / GPU Candle Conv1d
//! - `MatmulConvTranspose1d`: CPU 优化的 ConvTranspose1d (单次 matmul + col2im scatter)
//! - `FastConvTranspose1d`: 枚举自动选择 CPU Matmul / GPU Candle ConvTranspose1d

use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use rayon::prelude::*;

// ──────────────────────────── Direct BLAS (Accelerate) ────────────────────────────

/// 直接 BLAS sgemm 调用 — 绕过 Candle Tensor API, 消除 Tensor 创建/提取开销。
///
/// 当 `accelerate` feature 启用时, 使用 FFI 直接调用 Apple Accelerate 的 `cblas_sgemm`,
/// 将 matmul 结果直接写入预分配的 Vec<f32>, 并支持 bias 融合 (beta=1.0)。
///
/// 消除的开销 (每次 forward_vec 调用):
/// - Tensor::from_vec 元数据创建 (im2col → Tensor)
/// - Candle matmul 抽象层开销 (shape 检查, device 检查, dispatch)
/// - result_2d.flatten_all().to_vec1() 内存拷贝 (54-108MB)
/// - 独立 bias_add 步骤 (融合到 sgemm 的 beta 参数)
///
/// 当 `accelerate` feature 未启用时, 回退到 Candle Tensor matmul 路径。
#[cfg(feature = "accelerate")]
mod cblas {
    const ROW_MAJOR: u32 = 101;
    const NO_TRANS: u32 = 111;
    const TRANS: u32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: u32,
            trans_a: u32,
            trans_b: u32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    /// Row-major sgemm: C[M,N] = alpha * op(A) @ op(B) + beta * C
    ///
    /// - A: stored row-major as [M, K] if !trans_a, [K, M] if trans_a
    /// - B: stored row-major as [K, N] if !trans_b, [N, K] if trans_b
    /// - C: stored row-major as [M, N], must be pre-allocated with size M*N
    ///
    /// 当 beta=1.0 时, sgemm 计算 C = A@B + C, 可用于 bias 融合
    /// (预先将 C 填充为 bias 值, sgemm 自动加上 matmul 结果)。
    pub(crate) fn sgemm(
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) {
        let (ta, lda) = if trans_a {
            (TRANS, m) // A stored [K, M] row-major, lda = M
        } else {
            (NO_TRANS, k) // A stored [M, K] row-major, lda = K
        };
        let (tb, ldb) = if trans_b {
            (TRANS, k) // B stored [N, K] row-major, ldb = K
        } else {
            (NO_TRANS, n) // B stored [K, N] row-major, ldb = N
        };
        let ldc = n;

        debug_assert!(
            a.len() >= m * k,
            "sgemm: a too small: {} < {}",
            a.len(),
            m * k
        );
        debug_assert!(
            b.len() >= k * n,
            "sgemm: b too small: {} < {}",
            b.len(),
            k * n
        );
        debug_assert!(
            c.len() >= m * n,
            "sgemm: c too small: {} < {}",
            c.len(),
            m * n
        );

        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                ta,
                tb,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }
}

// ──────────────────────────── MatmulConv1d (CPU 优化) ────────────────────────────

/// CPU 优化的 Conv1d — 使用 im2col + BLAS matmul 替代 Candle 的直接卷积。
///
/// 适用于 groups=1, dilation=1, stride=1, kernel>1 的标准卷积。
/// 在 CPU 上比 Candle Conv1d 快 4-10x (Accelerate BLAS vs 直接卷积)。
///
/// 原理:
/// 1. 权重 [C_out, C_in, K] → 转置 → [C_out, K, C_in] → 重塑 → [C_out, K*C_in]
/// 2. 输入 im2col: 融合零填充 + Rayon 并行构建 [K*C_in, L] 矩阵
/// 3. 单次 matmul: [C_out, K*C_in] @ [K*C_in, L] = [C_out, L]
///
/// im2col 构建: raw Vec + Rayon 并行 (按行 K*C_in 分配到多线程),
/// 融合零填充 (边界检查内联), 消除 Tensor::narrow + Tensor::cat 开销。
/// BLAS matmul 利用 SIMD (Accelerate) 和缓存优化, 比直接卷积快得多。
pub(crate) struct MatmulConv1d {
    /// 预处理权重: [C_out, K*C_in] (转置后重塑, 连续化)
    weight: Tensor,
    /// CPU-optimized: pre-extracted weight [C_out, K*C_in] row-major (for direct BLAS)
    #[cfg_attr(not(feature = "accelerate"), allow(dead_code))]
    weight_vec: Vec<f32>,
    bias: Option<Tensor>,
    /// CPU-optimized: pre-extracted bias [C_out]
    bias_vec: Option<Vec<f32>>,
    kernel_size: usize,
    padding: usize,
    c_out: usize,
    #[allow(dead_code)]
    c_in: usize,
}

impl MatmulConv1d {
    fn new(conv_weight: Tensor, bias: Option<Tensor>, padding: usize) -> Result<Self> {
        let dims = conv_weight.dims();
        let c_out = dims[0];
        let c_in = dims[1];
        let k = dims[2];
        // 转置 [C_out, C_in, K] → [C_out, K, C_in] → 重塑 [C_out, K*C_in]
        // 这样 im2col 按 tap k 顺序拼接时, 权重行索引 = k*C_in + c 匹配
        let weight = conv_weight
            .transpose(1, 2)?
            .contiguous()?
            .reshape((c_out, k * c_in))?;
        // Pre-extract weight to Vec for direct BLAS sgemm path
        let weight_vec = weight.flatten_all()?.to_vec1::<f32>()?;
        let bias_vec = if let Some(ref b) = bias {
            Some(b.flatten_all()?.to_vec1::<f32>()?)
        } else {
            None
        };
        Ok(Self {
            weight,
            weight_vec,
            bias,
            bias_vec,
            kernel_size: k,
            padding,
            c_out,
            c_in,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, C_in, L] — batch is always 1 in our decoder
        let (batch, c_in, l) = x.dims3()?;
        debug_assert_eq!(batch, 1, "MatmulConv1d currently only supports batch=1");
        let device = x.device();

        // 输出长度: L + 2*padding - kernel_size + 1
        let out_len = l + 2 * self.padding - self.kernel_size + 1;
        let k = self.kernel_size;
        let pad = self.padding;

        // 1. 提取输入到 Vec<f32> — [C_in * L] (通道主序: x_vec[c*l + i] = x[c, i])
        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;

        // 2. 直接构建 im2col 矩阵: [K*C_in, out_len] (行主序)
        //    im2col[(tap*C_in + ch) * out_len + li] = x_padded[ch, li + tap]
        //      = x[ch, li + tap - pad] if 0 <= li + tap - pad < L, else 0
        //
        //    相比原实现 (Tensor::narrow + Tensor::cat):
        //    - 消除零填充 Tensor 分配 (54MB for Block 4)
        //    - 消除 7 个 narrow view + Tensor::cat (376MB 串行拷贝非连续数据)
        //    - Rayon 并行填充: 按行 (K*C_in 行) 分配到多线程
        //    - 融合零填充: 边界检查内联, 越界位置保持 0.0
        let im2col_size = k * c_in * out_len;
        let mut im2col_vec = vec![0.0f32; im2col_size];

        // Rayon 并行: 每行 out_len 元素, 共 K*C_in 行
        // 每行是一个通道在某个 tap 下的滑动窗口视图
        im2col_vec
            .par_chunks_mut(out_len)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let tap = row_idx / c_in;
                let ch = row_idx % c_in;
                let x_base = ch * l;
                for li in 0..out_len {
                    // 源索引 (在原始未填充空间中)
                    let src = li as isize + tap as isize - pad as isize;
                    if src >= 0 && (src as usize) < l {
                        row[li] = x_vec[x_base + src as usize];
                    }
                    // else: 保持 0.0 (零填充)
                }
            });

        // 3. 创建 Tensor (零拷贝: from_vec 取得 Vec 所有权)
        let im2col_2d = Tensor::from_vec(im2col_vec, (k * c_in, out_len), device)?;

        // 4. matmul: [C_out, K*C_in] @ [K*C_in, out_len] = [C_out, out_len]
        let result_2d = self.weight.matmul(&im2col_2d)?; // [C_out, out_len]
        let mut result = result_2d.unsqueeze(0)?; // [1, C_out, out_len]

        // 5. 加偏置
        if let Some(ref bias) = self.bias {
            let b = bias
                .reshape((1, (), 1))?
                .to_dtype(result.dtype())?
                .broadcast_as(result.shape())?;
            result = result.broadcast_add(&b)?;
        }

        Ok(result)
    }

    /// CPU forward (Vec 输入): 跳过输入 to_vec1, 偏置在 Vec 空间添加
    ///
    /// 用于 DecoderConvBlock::forward_fused_vec 链式调用.
    /// 输入: &[f32] 通道主序 [C_in * L], 输出: Vec<f32> 通道主序 [C_out * out_len]
    ///
    /// accelerate 路径: 直接 cblas_sgemm, bias 融合 (beta=1.0), 零 Tensor 开销
    /// 非 accelerate 路径: Tensor::from_vec + matmul + to_vec1 + Vec bias add
    fn forward_vec(&self, x: &[f32], c_in: usize, l: usize, device: &Device) -> Result<Vec<f32>> {
        let out_len = l + 2 * self.padding - self.kernel_size + 1;
        let k = self.kernel_size;
        let pad = self.padding;

        // Build im2col: [K*C_in, out_len] — Rayon parallel fill (same for both paths)
        let im2col_size = k * c_in * out_len;
        let mut im2col_vec = vec![0.0f32; im2col_size];
        im2col_vec
            .par_chunks_mut(out_len)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let tap = row_idx / c_in;
                let ch = row_idx % c_in;
                let x_base = ch * l;
                for li in 0..out_len {
                    let src = li as isize + tap as isize - pad as isize;
                    if src >= 0 && (src as usize) < l {
                        row[li] = x[x_base + src as usize];
                    }
                }
            });

        let result_size = self.c_out * out_len;

        #[cfg(feature = "accelerate")]
        {
            let _ = device; // unused in accelerate path
                            // Direct BLAS sgemm — bypasses Candle Tensor API entirely
                            // Bias fused via beta=1.0: C = 1.0 * A@B + 1.0 * C (C pre-filled with bias)
            let mut result_vec = if let Some(ref bias_vec) = self.bias_vec {
                let mut v = vec![0.0f32; result_size];
                v.par_chunks_mut(out_len)
                    .enumerate()
                    .for_each(|(co, chunk)| {
                        let b = bias_vec[co];
                        for val in chunk.iter_mut() {
                            *val = b;
                        }
                    });
                v
            } else {
                vec![0.0f32; result_size]
            };
            let beta = if self.bias_vec.is_some() { 1.0 } else { 0.0 };

            cblas::sgemm(
                false,            // trans_a: weight [C_out, K*C_in] NoTrans
                false,            // trans_b: im2col [K*C_in, out_len] NoTrans
                self.c_out,       // M
                out_len,          // N
                k * c_in,         // K
                1.0,              // alpha
                &self.weight_vec, // A
                &im2col_vec,      // B
                beta,             // beta (1.0 for bias fusion)
                &mut result_vec,  // C
            );
            Ok(result_vec)
        }

        #[cfg(not(feature = "accelerate"))]
        {
            // Tensor fallback — create Tensor, matmul, extract to Vec
            let im2col_2d = Tensor::from_vec(im2col_vec, (k * c_in, out_len), device)?;
            let result_2d = self.weight.matmul(&im2col_2d)?;
            let mut result_vec = result_2d.flatten_all()?.to_vec1::<f32>()?;
            debug_assert_eq!(result_vec.len(), result_size);
            if let Some(ref bias_vec) = self.bias_vec {
                result_vec
                    .par_chunks_mut(out_len)
                    .enumerate()
                    .for_each(|(co, chunk)| {
                        let b = bias_vec[co];
                        for v in chunk.iter_mut() {
                            *v += b;
                        }
                    });
            }
            Ok(result_vec)
        }
    }
}

// ──────────────────────────── MatmulConv1dK1 (kernel=1 直接 matmul) ────────────────────────────

/// CPU 优化的 Conv1d (kernel=1) — 直接 BLAS matmul, 无需 im2col。
///
/// kernel=1 的 Conv1d 等价于矩阵乘法:
/// - 权重 [C_out, C_in, 1] → reshape → [C_out, C_in]
/// - 输入 [1, C_in, L] → squeeze → [C_in, L]
/// - 输出 = [C_out, C_in] @ [C_in, L] = [C_out, L]
///
/// 与 MatmulConv1d (kernel>1) 相比:
/// - 跳过 im2col 步骤 (无需 narrow + cat)
/// - 权重无需 transpose (直接 reshape)
/// - 适用于 ConvNeXtBlock pwconv1/pwconv2, DecoderConvBlock conv2 等所有 k=1 Conv1d
pub(crate) struct MatmulConv1dK1 {
    /// 预处理权重: [C_out, C_in] (从 [C_out, C_in, 1] reshape, 连续化)
    weight: Tensor,
    /// CPU-optimized: pre-extracted weight [C_out, C_in] row-major (for direct BLAS)
    #[cfg_attr(not(feature = "accelerate"), allow(dead_code))]
    weight_vec: Vec<f32>,
    bias: Option<Tensor>,
    /// CPU-optimized: pre-extracted bias [C_out]
    bias_vec: Option<Vec<f32>>,
    c_out: usize,
    c_in: usize,
}

impl MatmulConv1dK1 {
    fn new(conv_weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = conv_weight.dims();
        let c_out = dims[0];
        let c_in = dims[1];
        // [C_out, C_in, 1] → [C_out, C_in] (零拷贝 reshape + 连续化确保 BLAS 效率)
        let weight = conv_weight.reshape((c_out, c_in))?.contiguous()?;
        // Pre-extract weight to Vec for direct BLAS sgemm path
        let weight_vec = weight.flatten_all()?.to_vec1::<f32>()?;
        let bias_vec = if let Some(ref b) = bias {
            Some(b.flatten_all()?.to_vec1::<f32>()?)
        } else {
            None
        };
        Ok(Self {
            weight,
            weight_vec,
            bias,
            bias_vec,
            c_out,
            c_in,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [1, C_in, L] — batch is always 1 in our decoder
        let (batch, _c_in, _l) = x.dims3()?;
        debug_assert_eq!(batch, 1, "MatmulConv1dK1 currently only supports batch=1");

        // squeeze batch: [1, C_in, L] → [C_in, L]
        let x_2d = x.squeeze(0)?;

        // matmul: [C_out, C_in] @ [C_in, L] = [C_out, L]
        let result_2d = self.weight.matmul(&x_2d)?;

        // unsqueeze: [C_out, L] → [1, C_out, L]
        let mut result = result_2d.unsqueeze(0)?;

        // 加偏置
        if let Some(ref bias) = self.bias {
            let b = bias
                .reshape((1, (), 1))?
                .to_dtype(result.dtype())?
                .broadcast_as(result.shape())?;
            result = result.broadcast_add(&b)?;
        }

        Ok(result)
    }

    /// CPU forward (Vec 输入): 跳过 to_vec1, 偏置在 Vec 空间添加
    ///
    /// 用于 DecoderConvBlock::forward_fused_vec 链式调用.
    /// k=1: 输出长度 = 输入长度, 无 padding, 无 im2col.
    ///
    /// accelerate 路径: 直接 cblas_sgemm, bias 融合 (beta=1.0), 零 Tensor 开销
    /// 非 accelerate 路径: Tensor::from_vec + matmul + to_vec1 + Vec bias add
    fn forward_vec(&self, x: &[f32], c_in: usize, l: usize, device: &Device) -> Result<Vec<f32>> {
        #[cfg(feature = "accelerate")]
        {
            let _ = device; // unused in accelerate path
            let result_size = self.c_out * l;
            let mut result_vec = if let Some(ref bias_vec) = self.bias_vec {
                let mut v = vec![0.0f32; result_size];
                v.par_chunks_mut(l).enumerate().for_each(|(co, chunk)| {
                    let b = bias_vec[co];
                    for val in chunk.iter_mut() {
                        *val = b;
                    }
                });
                v
            } else {
                vec![0.0f32; result_size]
            };
            let beta = if self.bias_vec.is_some() { 1.0 } else { 0.0 };

            cblas::sgemm(
                false,            // trans_a: weight [C_out, C_in] NoTrans
                false,            // trans_b: x [C_in, L] NoTrans
                self.c_out,       // M
                l,                // N
                c_in,             // K
                1.0,              // alpha
                &self.weight_vec, // A
                x,                // B (borrowed slice)
                beta,             // beta
                &mut result_vec,  // C
            );
            Ok(result_vec)
        }

        #[cfg(not(feature = "accelerate"))]
        {
            let x_2d = Tensor::from_vec(x.to_vec(), (c_in, l), device)?;
            let result_2d = self.weight.matmul(&x_2d)?;
            let mut result_vec = result_2d.flatten_all()?.to_vec1::<f32>()?;
            debug_assert_eq!(result_vec.len(), self.c_out * l);
            if let Some(ref bias_vec) = self.bias_vec {
                result_vec
                    .par_chunks_mut(l)
                    .enumerate()
                    .for_each(|(co, chunk)| {
                        let b = bias_vec[co];
                        for v in chunk.iter_mut() {
                            *v += b;
                        }
                    });
            }
            Ok(result_vec)
        }
    }

    /// CPU forward (Vec owned 输入): 避免 clone, 直接将 Vec 转为 Tensor
    ///
    /// 用于 conv2 (k=1) 在 fused Vec 路径中, act2 输出 Vec 直接传入.
    ///
    /// accelerate 路径: 直接 cblas_sgemm, bias 融合 (beta=1.0), 零 Tensor 开销
    /// 非 accelerate 路径: Tensor::from_vec + matmul + to_vec1 + Vec bias add
    fn forward_vec_owned(
        &self,
        x: Vec<f32>,
        _c_in: usize,
        l: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "accelerate")]
        {
            let _ = device; // unused in accelerate path
            let result_size = self.c_out * l;
            let mut result_vec = if let Some(ref bias_vec) = self.bias_vec {
                let mut v = vec![0.0f32; result_size];
                v.par_chunks_mut(l).enumerate().for_each(|(co, chunk)| {
                    let b = bias_vec[co];
                    for val in chunk.iter_mut() {
                        *val = b;
                    }
                });
                v
            } else {
                vec![0.0f32; result_size]
            };
            let beta = if self.bias_vec.is_some() { 1.0 } else { 0.0 };

            cblas::sgemm(
                false,            // trans_a: weight [C_out, C_in] NoTrans
                false,            // trans_b: x [C_in, L] NoTrans
                self.c_out,       // M
                l,                // N
                self.c_in,        // K
                1.0,              // alpha
                &self.weight_vec, // A
                &x,               // B (borrowed from owned Vec)
                beta,             // beta
                &mut result_vec,  // C
            );
            Ok(result_vec)
        }

        #[cfg(not(feature = "accelerate"))]
        {
            // Tensor::from_vec takes ownership — zero-copy
            let x_2d = Tensor::from_vec(x, (self.c_in, l), device)?;
            let result_2d = self.weight.matmul(&x_2d)?;
            let mut result_vec = result_2d.flatten_all()?.to_vec1::<f32>()?;
            if let Some(ref bias_vec) = self.bias_vec {
                result_vec
                    .par_chunks_mut(l)
                    .enumerate()
                    .for_each(|(co, chunk)| {
                        let b = bias_vec[co];
                        for v in chunk.iter_mut() {
                            *v += b;
                        }
                    });
            }
            Ok(result_vec)
        }
    }
}

// ──────────────────────────── DepthwiseConv1d (CPU 优化, Rayon 并行) ────────────────────────────

/// CPU 优化的深度卷积 (depthwise Conv1d) — 使用 Rayon 并行替代 Candle 的直接卷积。
///
/// 适用于 groups = channels 的 Conv1d (如 ConvNeXtBlock 的 dwconv)。
/// 权重形状: [C, 1, K], 每个通道有自己的 1D 卷积核。
///
/// 在 CPU 上, Candle 的 depthwise Conv1d 使用直接卷积 (逐元素乘加),
/// 无法利用 BLAS。本实现使用 Rayon 按通道并行, 每个通道内顺序卷积,
/// 充分利用多核加速。
///
/// 原理:
/// 1. 权重 [C, 1, K] → 预提取为 Vec<f32> [C * K]
/// 2. 输入 [1, C, L] → 预提取为 Vec<f32> [C * L] (含零填充)
/// 3. 每个通道独立卷积: out[c, l] = bias[c] + sum_k weight[c, k] * x_padded[c, l+k]
/// 4. Rayon 并行: par_chunks_mut(C) 按通道分配线程
pub(crate) struct DepthwiseConv1d {
    /// 预提取权重: [C * K] (通道主序)
    weight: Vec<f32>,
    /// 预提取偏置: [C]
    bias: Option<Vec<f32>>,
    kernel_size: usize,
    padding: usize,
    channels: usize,
}

impl DepthwiseConv1d {
    fn new(conv_weight: Tensor, bias: Option<Tensor>, padding: usize) -> Result<Self> {
        let dims = conv_weight.dims();
        let channels = dims[0];
        let k = dims[2];
        // [C, 1, K] → flatten → [C * K] (通道主序: weight[c*K + k])
        let weight = conv_weight.flatten_all()?.to_vec1::<f32>()?;
        let bias = if let Some(b) = bias {
            Some(b.flatten_all()?.to_vec1::<f32>()?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            kernel_size: k,
            padding,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [1, C, L] — batch is always 1
        let (batch, c, l) = x.dims3()?;
        debug_assert_eq!(batch, 1, "DepthwiseConv1d currently only supports batch=1");
        debug_assert_eq!(c, self.channels, "channel count mismatch");
        let device = x.device();

        // 输出长度: L + 2*padding - kernel_size + 1
        let out_len = l + 2 * self.padding - self.kernel_size + 1;

        // 提取输入到 Vec
        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0.0f32; c * out_len];

        let k_size = self.kernel_size;
        let padding = self.padding;
        let weight = &self.weight;

        // Rayon 并行: 每个线程处理一个通道
        // 通道 c 的数据: x_vec[c*l .. c*l+l], output[c*out_len .. c*out_len+out_len]
        output
            .par_chunks_mut(out_len)
            .enumerate()
            .for_each(|(ci, out_chunk)| {
                let x_base = ci * l;
                let w_base = ci * k_size;
                let b = self.bias.as_ref().map(|bv| bv[ci]).unwrap_or(0.0);

                for li in 0..out_len {
                    let mut sum = b;
                    for ki in 0..k_size {
                        // 输入位置: li + ki - padding (在原始未填充空间中)
                        let src_idx = li as isize + ki as isize - padding as isize;
                        if src_idx >= 0 && (src_idx as usize) < l {
                            sum += weight[w_base + ki] * x_vec[x_base + src_idx as usize];
                        }
                    }
                    out_chunk[li] = sum;
                }
            });

        Ok(Tensor::from_vec(output, (1, c, out_len), device)?)
    }

    /// CPU forward (Vec 输入): 跳过 to_vec1/from_vec
    fn forward_vec(&self, x: &[f32], c: usize, l: usize, _device: &Device) -> Result<Vec<f32>> {
        debug_assert_eq!(c, self.channels, "channel count mismatch");
        let out_len = l + 2 * self.padding - self.kernel_size + 1;
        let mut output = vec![0.0f32; c * out_len];
        let k_size = self.kernel_size;
        let padding = self.padding;
        let weight = &self.weight;

        output
            .par_chunks_mut(out_len)
            .enumerate()
            .for_each(|(ci, out_chunk)| {
                let x_base = ci * l;
                let w_base = ci * k_size;
                let b = self.bias.as_ref().map(|bv| bv[ci]).unwrap_or(0.0);
                for li in 0..out_len {
                    let mut sum = b;
                    for ki in 0..k_size {
                        let src_idx = li as isize + ki as isize - padding as isize;
                        if src_idx >= 0 && (src_idx as usize) < l {
                            sum += weight[w_base + ki] * x[x_base + src_idx as usize];
                        }
                    }
                    out_chunk[li] = sum;
                }
            });

        Ok(output)
    }
}

// ──────────────────────────── FastConv1d ────────────────────────────

/// Conv1d 实现: 自动选择 CPU matmul 或 GPU Candle Conv1d。
///
/// 在 CPU 上:
/// - groups=1, kernel=1: 使用 MatmulConv1dK1 (直接 BLAS matmul)
/// - groups=1, kernel>1: 使用 MatmulConv1d (im2col + BLAS matmul)
/// - groups>1 (depthwise): 使用 DepthwiseConv1d (Rayon 并行)
/// 在 GPU (Metal/CUDA) 上, 使用 Candle 原生 Conv1d (已有 GPU 优化)。
pub(crate) enum FastConv1d {
    Candle(Conv1d),
    Matmul(MatmulConv1d),
    MatmulK1(MatmulConv1dK1),
    Depthwise(DepthwiseConv1d),
}

impl FastConv1d {
    pub(crate) fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        config: Conv1dConfig,
        device: &Device,
    ) -> Result<Self> {
        let kernel_size = weight.dim(2).unwrap_or(1);
        let use_matmul = matches!(device, Device::Cpu)
            && config.groups == 1
            && config.dilation == 1
            && config.stride == 1;

        // CPU depthwise: groups > 1 (typically groups = channels)
        let use_depthwise = matches!(device, Device::Cpu)
            && config.groups > 1
            && config.dilation == 1
            && config.stride == 1;

        if use_matmul {
            if kernel_size == 1 {
                // k=1: 直接 matmul, 无需 im2col
                let m = MatmulConv1dK1::new(weight, bias)?;
                tracing::debug!(
                    "FastConv1d: using MatmulConv1dK1 (CPU, k=1, in={}, out={})",
                    m.weight.dim(1).unwrap_or(0),
                    m.weight.dim(0).unwrap_or(0)
                );
                Ok(FastConv1d::MatmulK1(m))
            } else {
                // k>1: im2col + matmul
                let m = MatmulConv1d::new(weight, bias, config.padding)?;
                tracing::debug!(
                    "FastConv1d: using MatmulConv1d (CPU, k={}, in={}, out={})",
                    kernel_size,
                    m.weight.dim(1).unwrap_or(0) / kernel_size,
                    m.weight.dim(0).unwrap_or(0)
                );
                Ok(FastConv1d::Matmul(m))
            }
        } else if use_depthwise {
            // depthwise: Rayon 并行卷积
            let m = DepthwiseConv1d::new(weight, bias, config.padding)?;
            tracing::debug!(
                "FastConv1d: using DepthwiseConv1d (CPU, k={}, channels={}, groups={})",
                kernel_size,
                m.channels,
                config.groups
            );
            Ok(FastConv1d::Depthwise(m))
        } else {
            Ok(FastConv1d::Candle(Conv1d::new(weight, bias, config)))
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            FastConv1d::Candle(c) => Ok(c.forward(x)?),
            FastConv1d::Matmul(m) => m.forward(x),
            FastConv1d::MatmulK1(m) => m.forward(x),
            FastConv1d::Depthwise(m) => m.forward(x),
        }
    }

    /// CPU forward (Vec borrowed 输入): 用于 conv1 (k>1) 和 depthwise
    ///
    /// 输入: &[f32] 通道主序, 输出: Vec<f32> 通道主序.
    /// Candle 变体回退到 Tensor 路径.
    pub(crate) fn forward_vec(
        &self,
        x: &[f32],
        c_in: usize,
        l: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        match self {
            FastConv1d::Candle(c) => {
                let x_tensor = Tensor::from_vec(x.to_vec(), (1, c_in, l), device)?;
                let result = c.forward(&x_tensor)?;
                Ok(result.flatten_all()?.to_vec1::<f32>()?)
            }
            FastConv1d::Matmul(m) => m.forward_vec(x, c_in, l, device),
            FastConv1d::MatmulK1(m) => m.forward_vec(x, c_in, l, device),
            FastConv1d::Depthwise(m) => m.forward_vec(x, c_in, l, device),
        }
    }

    /// CPU forward (Vec owned 输入): 用于 conv2 (k=1), 避免 clone
    ///
    /// 输入: Vec<f32> (moved, no clone), 输出: Vec<f32>.
    /// MatmulConv1dK1 直接将 Vec 转为 Tensor (zero-copy).
    pub(crate) fn forward_vec_owned(
        &self,
        x: Vec<f32>,
        c_in: usize,
        l: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        match self {
            FastConv1d::Candle(c) => {
                let x_tensor = Tensor::from_vec(x, (1, c_in, l), device)?;
                let result = c.forward(&x_tensor)?;
                Ok(result.flatten_all()?.to_vec1::<f32>()?)
            }
            FastConv1d::Matmul(m) => m.forward_vec(&x, c_in, l, device),
            FastConv1d::MatmulK1(m) => m.forward_vec_owned(x, c_in, l, device),
            FastConv1d::Depthwise(m) => m.forward_vec(&x, c_in, l, device),
        }
    }
}

// ──────────────────────────── MatmulConvTranspose1d (CPU 优化) ────────────────────────────

/// CPU 优化的 ConvTranspose1d — 使用单次 BLAS matmul + col2im scatter 替代 Candle 的慢速路径。
///
/// 当 padding > 0 时, Candle 的 ConvTranspose1d 会使用慢速路径 (直接嵌套循环卷积),
/// 本实现使用 matmul + col2im 替代, 利用 Accelerate BLAS 加速。
///
/// 原理:
/// 1. 权重 [C_in, C_out, K] → 重塑 → [C_in, C_out * K] (零拷贝)
/// 2. 输入 [1, C_in, L_in] → 转置 → [L_in, C_in]
/// 3. 单次 matmul: [L_in, C_in] @ [C_in, C_out * K] = [L_in, C_out * K]
/// 4. col2im scatter: 每个 (l_in_i, k_i) 贡献到输出位置 l_in_i * stride + k_i - padding
///
/// 与 MatmulConv1d 的 im2col 方向相反: Conv1d 是 im2col→matmul (收集),
/// ConvTranspose1d 是 matmul→col2im (散射)。
pub(crate) struct MatmulConvTranspose1d {
    /// 预处理权重: [C_in, C_out * K] (连续化)
    weight: Tensor,
    /// CPU-optimized: pre-extracted weight [C_in, C_out*K] row-major (for direct BLAS)
    #[cfg_attr(not(feature = "accelerate"), allow(dead_code))]
    weight_vec: Vec<f32>,
    bias: Option<Tensor>,
    /// CPU-optimized: pre-extracted bias [C_out]
    #[cfg_attr(not(feature = "accelerate"), allow(dead_code))]
    bias_vec: Option<Vec<f32>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    kernel_size: usize,
    c_in: usize,
    c_out: usize,
}

impl MatmulConvTranspose1d {
    fn new(weight: Tensor, bias: Option<Tensor>, config: ConvTranspose1dConfig) -> Result<Self> {
        let dims = weight.dims();
        let c_in = dims[0];
        let c_out = dims[1];
        let k = dims[2];
        // Reshape [C_in, C_out, K] → [C_in, C_out * K] (零拷贝, 仅元数据变更)
        let weight = weight.reshape((c_in, c_out * k))?.contiguous()?;
        // Pre-extract weight and bias to Vec for direct BLAS sgemm path
        let weight_vec = weight.flatten_all()?.to_vec1::<f32>()?;
        let bias_vec = if let Some(ref b) = bias {
            Some(b.flatten_all()?.to_vec1::<f32>()?)
        } else {
            None
        };
        Ok(Self {
            weight,
            weight_vec,
            bias,
            bias_vec,
            stride: config.stride,
            padding: config.padding,
            output_padding: config.output_padding,
            kernel_size: k,
            c_in,
            c_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [1, C_in, L_in] — batch 在解码器中始终为 1
        let (batch, c_in, l_in) = x.dims3()?;
        debug_assert_eq!(
            batch, 1,
            "MatmulConvTranspose1d currently only supports batch=1"
        );
        debug_assert_eq!(c_in, self.c_in);
        let device = x.device();
        let dtype = x.dtype();

        // 输出长度: (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
        let l_out =
            (l_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding;

        // 1. 转置 + 去 batch: [1, C_in, L_in] → [L_in, C_in]
        let x_2d = x.transpose(1, 2)?.squeeze(0)?.contiguous()?;

        // 2. 单次 BLAS matmul: [L_in, C_in] @ [C_in, C_out * K] → [L_in, C_out * K]
        let col = x_2d.matmul(&self.weight)?;

        // 3. col2im: 将 [L_in, C_out * K] 散射到 [C_out, L_out] (with padding)
        //    使用原始 Vec 操作实现高效 scatter-add (避免 K 次张量分配)
        //    Rayon 并行: 按输出通道并行, 每个通道写入不重叠的输出区域
        let col_vec = col.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0.0f32; self.c_out * l_out];

        // 并行 col2im: 每个线程处理一个输出通道, 写入 output[c_i * l_out .. (c_i+1) * l_out]
        let stride = self.stride;
        let padding = self.padding;
        let k_size = self.kernel_size;
        let c_out = self.c_out;
        let col_csk = c_out * k_size; // col_vec 中每个 l_in_i 的步长

        output
            .par_chunks_mut(l_out)
            .enumerate()
            .for_each(|(c_i, out_chunk)| {
                for l_in_i in 0..l_in {
                    let base_src = l_in_i * col_csk + c_i * k_size;
                    for k_i in 0..k_size {
                        let raw_pos = l_in_i * stride + k_i;
                        if raw_pos >= padding && raw_pos - padding < l_out {
                            out_chunk[raw_pos - padding] += col_vec[base_src + k_i];
                        }
                    }
                }
            });

        // 4. 转回张量: [1, C_out, L_out]
        let result = Tensor::from_vec(output, (1, c_out, l_out), device)?.to_dtype(dtype)?;

        // 5. 加偏置
        let result = if let Some(ref bias) = self.bias {
            let b = bias
                .reshape((1, c_out, 1))?
                .to_dtype(dtype)?
                .broadcast_as(result.shape())?;
            result.broadcast_add(&b)?
        } else {
            result
        };

        Ok(result)
    }

    /// CPU forward (Vec owned 输入): 直接 cblas_sgemm + col2im, 返回 Vec<f32>
    ///
    /// 用于 DecoderUpsampleBlock 的 CPU 融合 Vec 路径.
    /// 输入: Vec<f32> 通道主序 [C_in * L_in], 输出: Vec<f32> 通道主序 [C_out * L_out]
    ///
    /// accelerate 路径: 直接 cblas_sgemm (TransA), 消除 transpose+contiguous+to_vec1
    /// 非 accelerate 路径: Tensor transpose + matmul + to_vec1
    fn forward_vec(
        &self,
        x: Vec<f32>,
        c_in: usize,
        l_in: usize,
        _device: &Device,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(c_in, self.c_in, "c_in mismatch");
        let l_out =
            (l_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding;

        // Matmul: col[L_in, C_out*K] = x^T @ weight = [L_in, C_in] @ [C_in, C_out*K]
        // x is stored as [C_in, L_in] row-major → need TransA to get [L_in, C_in]

        #[cfg(feature = "accelerate")]
        {
            let col_size = l_in * self.c_out * self.kernel_size;
            let mut col_vec = vec![0.0f32; col_size];
            // sgemm: C = alpha * op(A) @ op(B) + beta * C
            // A = x [C_in, L_in] row-major, trans_a=true → [L_in, C_in]
            // B = weight_vec [C_in, C_out*K] row-major, trans_b=false
            // C = col_vec [L_in, C_out*K] row-major
            cblas::sgemm(
                true,                          // trans_a: x is [C_in, L_in], need [L_in, C_in]
                false,                         // trans_b: weight is [C_in, C_out*K]
                l_in,                          // M
                self.c_out * self.kernel_size, // N
                c_in,                          // K
                1.0,                           // alpha
                &x,                            // A (owned Vec, borrowed as slice)
                &self.weight_vec,              // B
                0.0,          // beta (no bias fusion here, bias added after col2im)
                &mut col_vec, // C
            );
            self.col2im_and_bias(col_vec, l_in, l_out)
        }

        #[cfg(not(feature = "accelerate"))]
        {
            // Tensor fallback: transpose + matmul + to_vec1
            let x_2d = Tensor::from_vec(x, (c_in, l_in), &Device::Cpu)?
                .transpose(0, 1)?
                .contiguous()?;
            let weight_tensor = Tensor::from_vec(
                self.weight_vec.clone(),
                (c_in, self.c_out * self.kernel_size),
                &Device::Cpu,
            )?;
            let col = x_2d.matmul(&weight_tensor)?;
            let col_vec = col.flatten_all()?.to_vec1::<f32>()?;
            self.col2im_and_bias(col_vec, l_in, l_out)
        }
    }

    /// col2im scatter + bias addition (shared between accelerate and fallback paths)
    fn col2im_and_bias(&self, col_vec: Vec<f32>, l_in: usize, l_out: usize) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; self.c_out * l_out];
        let stride = self.stride;
        let padding = self.padding;
        let k_size = self.kernel_size;
        let c_out = self.c_out;
        let col_csk = c_out * k_size;

        output
            .par_chunks_mut(l_out)
            .enumerate()
            .for_each(|(c_i, out_chunk)| {
                for l_in_i in 0..l_in {
                    let base_src = l_in_i * col_csk + c_i * k_size;
                    for k_i in 0..k_size {
                        let raw_pos = l_in_i * stride + k_i;
                        if raw_pos >= padding && raw_pos - padding < l_out {
                            out_chunk[raw_pos - padding] += col_vec[base_src + k_i];
                        }
                    }
                }
            });

        // Add bias in Vec space
        if let Some(ref bias_vec) = self.bias_vec {
            output
                .par_chunks_mut(l_out)
                .enumerate()
                .for_each(|(co, chunk)| {
                    let b = bias_vec[co];
                    for v in chunk.iter_mut() {
                        *v += b;
                    }
                });
        }

        Ok(output)
    }
}

// ──────────────────────────── FastConvTranspose1d ────────────────────────────

/// ConvTranspose1d 实现: 自动选择 CPU matmul 或 GPU Candle ConvTranspose1d。
///
/// 在 CPU 上, 对 stride > 1, groups=1, dilation=1 的转置卷积使用 MatmulConvTranspose1d。
/// 在 GPU (Metal/CUDA) 上, 使用 Candle 原生 ConvTranspose1d。
pub(crate) enum FastConvTranspose1d {
    Candle(ConvTranspose1d),
    Matmul(MatmulConvTranspose1d),
}

impl FastConvTranspose1d {
    pub(crate) fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        config: ConvTranspose1dConfig,
        device: &Device,
    ) -> Result<Self> {
        let use_matmul = matches!(device, Device::Cpu)
            && config.groups == 1
            && config.dilation == 1
            && config.stride > 1;

        if use_matmul {
            let m = MatmulConvTranspose1d::new(weight, bias, config)?;
            tracing::debug!(
                "FastConvTranspose1d: using MatmulConvTranspose1d (CPU, stride={}, k={}, c_in={}, c_out={})",
                m.stride, m.kernel_size, m.c_in, m.c_out
            );
            Ok(FastConvTranspose1d::Matmul(m))
        } else {
            Ok(FastConvTranspose1d::Candle(ConvTranspose1d::new(
                weight, bias, config,
            )))
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            FastConvTranspose1d::Candle(c) => Ok(c.forward(x)?),
            FastConvTranspose1d::Matmul(m) => m.forward(x),
        }
    }

    /// CPU forward (Vec owned 输入): 用于 DecoderUpsampleBlock 的 CPU 融合 Vec 路径
    ///
    /// 输入: Vec<f32> 通道主序 [C_in * L_in], 输出: Vec<f32> 通道主序 [C_out * L_out]
    /// Matmul 变体: 直接 cblas_sgemm (accelerate) 或 Tensor matmul (fallback)
    /// Candle 变体: 回退到 Tensor 路径
    pub(crate) fn forward_vec(
        &self,
        x: Vec<f32>,
        c_in: usize,
        l_in: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        match self {
            FastConvTranspose1d::Candle(c) => {
                let x_tensor = Tensor::from_vec(x, (1, c_in, l_in), device)?;
                let result = c.forward(&x_tensor)?;
                Ok(result.flatten_all()?.to_vec1::<f32>()?)
            }
            FastConvTranspose1d::Matmul(m) => m.forward_vec(x, c_in, l_in, device),
        }
    }
}

// ──────────────────────────── 测试 ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// 辅助: 比较两个 Tensor 的最大绝对差
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = a.sub(b).unwrap();
        let abs_diff = diff.abs().unwrap();
        let vec = abs_diff.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        vec.iter().fold(0.0f32, |m, &v| m.max(v))
    }

    /// 测试: MatmulConv1d 与 Conv1d 在 kernel=7, padding=3 下输出一致
    #[test]
    fn test_matmul_conv1d_matches_candle_k7() {
        let device = Device::Cpu;
        let c_out = 16;
        let c_in = 16;
        let kernel = 7;
        let padding = 3;
        let seq_len = 100;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1d::new(weight, Some(bias), padding).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(candle_out.dims(), matmul_out.dims());
        assert_eq!(matmul_out.dim(2).unwrap(), seq_len);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(diff < 1e-4, "MatmulConv1d vs Conv1d max diff: {diff}");
    }

    /// 测试: MatmulConv1d 与 Conv1d 在 kernel=3, padding=1 下输出一致
    #[test]
    fn test_matmul_conv1d_matches_candle_k3() {
        let device = Device::Cpu;
        let c_out = 32;
        let c_in = 64;
        let kernel = 3;
        let padding = 1;
        let seq_len = 50;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1d::new(weight, Some(bias), padding).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(candle_out.dims(), matmul_out.dims());
        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(diff < 1e-4, "MatmulConv1d vs Conv1d max diff (k=3): {diff}");
    }

    /// 测试: MatmulConv1d 无偏置也能工作
    #[test]
    fn test_matmul_conv1d_no_bias() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (8, 8, 7), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 50), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            None,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1d::new(weight, None, 3).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-4,
            "MatmulConv1d (no bias) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: MatmulConv1d 在大尺寸下与 Conv1d 一致 (模拟实际解码器尺寸)
    #[test]
    fn test_matmul_conv1d_large_size() {
        let device = Device::Cpu;
        let c_out = 96;
        let c_in = 96;
        let kernel = 7;
        let padding = 3;
        let seq_len = 1000; // 缩小版, 实际为 140K+

        let weight = Tensor::randn(0.0f32, 0.01, (c_out, c_in, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.01, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1d::new(weight, Some(bias), padding).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(matmul_out.dim(2).unwrap(), seq_len);
        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-3,
            "MatmulConv1d (large) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: FastConv1d 在 CPU 上选择 MatmulConv1d
    #[test]
    fn test_fast_conv1d_cpu_uses_matmul() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (8, 8, 7), &device).unwrap();
        let bias = Tensor::zeros(8, DType::F32, &device).unwrap();

        let fast = FastConv1d::new(
            weight,
            Some(bias),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        assert!(matches!(fast, FastConv1d::Matmul(_)));
    }

    /// 测试: FastConv1d 对 kernel=1 在 CPU 上使用 MatmulConv1dK1
    #[test]
    fn test_fast_conv1d_kernel1_uses_matmul_k1() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (8, 8, 1), &device).unwrap();
        let bias = Tensor::zeros(8, DType::F32, &device).unwrap();

        let fast = FastConv1d::new(weight, Some(bias), Conv1dConfig::default(), &device).unwrap();

        assert!(matches!(fast, FastConv1d::MatmulK1(_)));
    }

    /// 测试: MatmulConv1dK1 与 Conv1d 在 kernel=1 下输出一致
    #[test]
    fn test_matmul_conv1d_k1_matches_candle() {
        let device = Device::Cpu;
        let c_out = 32;
        let c_in = 64;
        let seq_len = 100;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, 1), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let conv = Conv1d::new(weight.clone(), Some(bias.clone()), Conv1dConfig::default());
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1dK1::new(weight, Some(bias)).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(candle_out.dims(), matmul_out.dims());
        assert_eq!(matmul_out.dim(2).unwrap(), seq_len);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(diff < 1e-5, "MatmulConv1dK1 vs Conv1d max diff: {diff}");
    }

    /// 测试: MatmulConv1dK1 无偏置也能工作
    #[test]
    fn test_matmul_conv1d_k1_no_bias() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (16, 16, 1), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, 16, 50), &device).unwrap();

        let conv = Conv1d::new(weight.clone(), None, Conv1dConfig::default());
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1dK1::new(weight, None).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-5,
            "MatmulConv1dK1 (no bias) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: MatmulConv1dK1 非对称通道 (C_in != C_out) 也能正确工作
    #[test]
    fn test_matmul_conv1d_k1_asymmetric_channels() {
        let device = Device::Cpu;
        let c_in = 96;
        let c_out = 384;
        let seq_len = 200;

        let weight = Tensor::randn(0.0f32, 0.05, (c_out, c_in, 1), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.02, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let conv = Conv1d::new(weight.clone(), Some(bias.clone()), Conv1dConfig::default());
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1dK1::new(weight, Some(bias)).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(matmul_out.dim(1).unwrap(), c_out);
        assert_eq!(matmul_out.dim(2).unwrap(), seq_len);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-5,
            "MatmulConv1dK1 (asym) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: MatmulConv1dK1 在大尺寸下与 Conv1d 一致 (模拟实际解码器尺寸)
    /// Block 4: 96 channels, ~140K samples
    #[test]
    fn test_matmul_conv1d_k1_large_size() {
        let device = Device::Cpu;
        let c_out = 96;
        let c_in = 96;
        let seq_len = 5000; // 缩小版, 实际为 140K+

        let weight = Tensor::randn(0.0f32, 0.01, (c_out, c_in, 1), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.01, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let conv = Conv1d::new(weight.clone(), Some(bias.clone()), Conv1dConfig::default());
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1dK1::new(weight, Some(bias)).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(matmul_out.dim(2).unwrap(), seq_len);
        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-4,
            "MatmulConv1dK1 (large) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: FastConv1d (MatmulK1 variant) forward 正确性
    #[test]
    fn test_fast_conv1d_k1_forward_correctness() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (32, 64, 1), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (32,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, 64, 200), &device).unwrap();

        let conv = Conv1d::new(weight.clone(), Some(bias.clone()), Conv1dConfig::default());
        let candle_out = conv.forward(&input).unwrap();

        let fast = FastConv1d::new(weight, Some(bias), Conv1dConfig::default(), &device).unwrap();
        assert!(matches!(fast, FastConv1d::MatmulK1(_)));
        let fast_out = fast.forward(&input).unwrap();

        let diff = max_abs_diff(&candle_out, &fast_out);
        assert!(
            diff < 1e-5,
            "FastConv1d (MatmulK1) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: FastConv1d 对 depthwise (groups=channels) 在 CPU 上使用 DepthwiseConv1d
    #[test]
    fn test_fast_conv1d_depthwise_uses_depthwise() {
        let device = Device::Cpu;
        let channels = 8;
        let weight = Tensor::randn(0.0f32, 0.1, (channels, 1, 7), &device).unwrap();
        let bias = Tensor::zeros(channels, DType::F32, &device).unwrap();

        let fast = FastConv1d::new(
            weight,
            Some(bias),
            Conv1dConfig {
                padding: 3,
                groups: channels,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        assert!(matches!(fast, FastConv1d::Depthwise(_)));
    }

    /// 测试: DepthwiseConv1d 与 Candle Conv1d (depthwise, k=7, pad=3) 输出一致
    #[test]
    fn test_depthwise_conv1d_matches_candle_k7() {
        let device = Device::Cpu;
        let channels = 16;
        let kernel = 7;
        let padding = 3;
        let seq_len = 100;

        let weight = Tensor::randn(0.0f32, 0.1, (channels, 1, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (channels,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, channels, seq_len), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding,
                groups: channels,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let dw = DepthwiseConv1d::new(weight, Some(bias), padding).unwrap();
        let dw_out = dw.forward(&input).unwrap();

        assert_eq!(candle_out.dims(), dw_out.dims());
        assert_eq!(dw_out.dim(2).unwrap(), seq_len);

        let diff = max_abs_diff(&candle_out, &dw_out);
        assert!(
            diff < 1e-5,
            "DepthwiseConv1d vs Conv1d (depthwise) max diff: {diff}"
        );
    }

    /// 测试: DepthwiseConv1d 无偏置也能工作
    #[test]
    fn test_depthwise_conv1d_no_bias() {
        let device = Device::Cpu;
        let channels = 8;
        let weight = Tensor::randn(0.0f32, 0.1, (channels, 1, 7), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, channels, 50), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            None,
            Conv1dConfig {
                padding: 3,
                groups: channels,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let dw = DepthwiseConv1d::new(weight, None, 3).unwrap();
        let dw_out = dw.forward(&input).unwrap();

        let diff = max_abs_diff(&candle_out, &dw_out);
        assert!(
            diff < 1e-5,
            "DepthwiseConv1d (no bias) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: DepthwiseConv1d 在大尺寸下与 Conv1d 一致 (模拟 ConvNeXtBlock 实际尺寸)
    /// channels=512, seq_len=5000 (缩小版, 实际为 18K+)
    #[test]
    fn test_depthwise_conv1d_large_size() {
        let device = Device::Cpu;
        let channels = 64;
        let kernel = 7;
        let padding = 3;
        let seq_len = 2000;

        let weight = Tensor::randn(0.0f32, 0.01, (channels, 1, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.01, (channels,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, channels, seq_len), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding,
                groups: channels,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let dw = DepthwiseConv1d::new(weight, Some(bias), padding).unwrap();
        let dw_out = dw.forward(&input).unwrap();

        assert_eq!(dw_out.dim(2).unwrap(), seq_len);
        let diff = max_abs_diff(&candle_out, &dw_out);
        assert!(
            diff < 1e-4,
            "DepthwiseConv1d (large) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: DepthwiseConv1d padding=0 时也能工作
    #[test]
    fn test_depthwise_conv1d_no_padding() {
        let device = Device::Cpu;
        let channels = 4;
        let kernel = 3;
        let weight = Tensor::randn(0.0f32, 0.1, (channels, 1, kernel), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, channels, 20), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            None,
            Conv1dConfig {
                groups: channels,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let dw = DepthwiseConv1d::new(weight, None, 0).unwrap();
        let dw_out = dw.forward(&input).unwrap();

        // padding=0, kernel=3 → 输出长度 = 20 - 3 + 1 = 18
        assert_eq!(candle_out.dim(2).unwrap(), 18);
        assert_eq!(dw_out.dim(2).unwrap(), 18);

        let diff = max_abs_diff(&candle_out, &dw_out);
        assert!(
            diff < 1e-5,
            "DepthwiseConv1d (no pad) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: FastConv1d (Depthwise variant) forward 正确性
    #[test]
    fn test_fast_conv1d_depthwise_forward_correctness() {
        let device = Device::Cpu;
        let channels = 16;
        let weight = Tensor::randn(0.0f32, 0.1, (channels, 1, 7), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (channels,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, channels, 200), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding: 3,
                groups: channels,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let fast = FastConv1d::new(
            weight,
            Some(bias),
            Conv1dConfig {
                padding: 3,
                groups: channels,
                ..Default::default()
            },
            &device,
        )
        .unwrap();
        assert!(matches!(fast, FastConv1d::Depthwise(_)));
        let fast_out = fast.forward(&input).unwrap();

        let diff = max_abs_diff(&candle_out, &fast_out);
        assert!(
            diff < 1e-5,
            "FastConv1d (Depthwise) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: FastConv1d 的 forward 在 CPU 上正确工作
    #[test]
    fn test_fast_conv1d_forward_correctness() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (16, 16, 7), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (16,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, 16, 200), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let fast = FastConv1d::new(
            weight,
            Some(bias),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            &device,
        )
        .unwrap();
        let fast_out = fast.forward(&input).unwrap();

        let diff = max_abs_diff(&candle_out, &fast_out);
        assert!(diff < 1e-4, "FastConv1d vs Conv1d max diff: {diff}");
    }

    /// 测试: MatmulConv1d padding=0 时也能工作
    #[test]
    fn test_matmul_conv1d_no_padding() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (4, 4, 3), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, 4, 20), &device).unwrap();

        let conv = Conv1d::new(weight.clone(), None, Conv1dConfig::default());
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1d::new(weight, None, 0).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        // padding=0, kernel=3 → 输出长度 = 20 - 3 + 1 = 18
        assert_eq!(candle_out.dim(2).unwrap(), 18);
        assert_eq!(matmul_out.dim(2).unwrap(), 18);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-4,
            "MatmulConv1d (no pad) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: 非对称通道 (C_in != C_out) 也能正确工作
    #[test]
    fn test_matmul_conv1d_asymmetric_channels() {
        let device = Device::Cpu;
        let c_in = 64;
        let c_out = 32;
        let kernel = 7;
        let padding = 3;
        let seq_len = 100;

        let weight = Tensor::randn(0.0f32, 0.05, (c_out, c_in, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.02, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1d::new(weight, Some(bias), padding).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(matmul_out.dim(1).unwrap(), c_out);
        assert_eq!(matmul_out.dim(2).unwrap(), seq_len);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-4,
            "MatmulConv1d (asym) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: raw Vec im2col 边界正确性 — 验证零填充区域输出正确
    /// 使用已知输入, 手动验证输出边界值
    #[test]
    fn test_matmul_conv1d_raw_vec_boundary() {
        let device = Device::Cpu;
        // 1 通道, k=3, pad=1, 输入 [1, 1, 5] = [1, 2, 3, 4, 5]
        // 零填充后: [0, 1, 2, 3, 4, 5, 0]
        // 卷积 (weight=1, bias=0): out[i] = sum(x_padded[i+i+k] for k in 0..3)
        // out[0] = 0+1+2 = 3, out[1] = 1+2+3 = 6, ..., out[4] = 4+5+0 = 9
        let weight = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (1, 1, 3), &device).unwrap();
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), &device).unwrap();

        let matmul = MatmulConv1d::new(weight, None, 1).unwrap();
        let out = matmul.forward(&input).unwrap();

        let vec = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(vec.len(), 5);
        assert!(
            (vec[0] - 3.0).abs() < 1e-5,
            "out[0] should be 3.0 (0+1+2), got {}",
            vec[0]
        );
        assert!(
            (vec[1] - 6.0).abs() < 1e-5,
            "out[1] should be 6.0 (1+2+3), got {}",
            vec[1]
        );
        assert!(
            (vec[2] - 9.0).abs() < 1e-5,
            "out[2] should be 9.0 (2+3+4), got {}",
            vec[2]
        );
        assert!(
            (vec[3] - 12.0).abs() < 1e-5,
            "out[3] should be 12.0 (3+4+5), got {}",
            vec[3]
        );
        assert!(
            (vec[4] - 9.0).abs() < 1e-5,
            "out[4] should be 9.0 (4+5+0), got {}",
            vec[4]
        );
    }

    /// 测试: raw Vec im2col 多通道边界正确性
    #[test]
    fn test_matmul_conv1d_raw_vec_multi_channel_boundary() {
        let device = Device::Cpu;
        // 2 通道, k=3, pad=1
        // ch0: [1, 2, 3], ch1: [4, 5, 6]
        // weight: out = ch0 * [1,0,0] + ch1 * [0,1,0] (即 out = ch1 shifted by 1)
        let weight = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, // out_ch0: tap0 from in_ch0
                0.0, 1.0, 0.0,
            ], // out_ch0: tap1 from in_ch1
            (1, 2, 3),
            &device,
        )
        .unwrap();
        let input =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3), &device).unwrap();

        let matmul = MatmulConv1d::new(weight, None, 1).unwrap();
        let out = matmul.forward(&input).unwrap();

        // 与 Candle 对比
        let conv = Conv1d::new(
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (1, 2, 3), &device).unwrap(),
            None,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let diff = max_abs_diff(&candle_out, &out);
        assert!(
            diff < 1e-5,
            "Multi-channel boundary vs Candle max diff: {diff}"
        );
    }

    /// 测试: raw Vec im2col 在 Block 4 规模下正确 (96ch, k=7, pad=3, 10K samples)
    #[test]
    fn test_matmul_conv1d_raw_vec_block4_scale() {
        let device = Device::Cpu;
        let c = 96;
        let kernel = 7;
        let padding = 3;
        let seq_len = 10_000;

        let weight = Tensor::randn(0.0f32, 0.01, (c, c, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.01, (c,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c, seq_len), &device).unwrap();

        let conv = Conv1d::new(
            weight.clone(),
            Some(bias.clone()),
            Conv1dConfig {
                padding,
                ..Default::default()
            },
        );
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1d::new(weight, Some(bias), padding).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(matmul_out.dim(2).unwrap(), seq_len);
        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-3,
            "MatmulConv1d (Block 4 scale) vs Conv1d max diff: {diff}"
        );
    }

    /// 测试: raw Vec im2col padding=0 时也正确 (无零填充, 无边界分支)
    #[test]
    fn test_matmul_conv1d_raw_vec_no_padding_correct() {
        let device = Device::Cpu;
        // k=3, pad=0, 输入 [1, 2, 5] → 输出长度 = 5 - 3 + 1 = 3
        let weight =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (1, 2, 3), &device).unwrap();
        let input = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            (1, 2, 5),
            &device,
        )
        .unwrap();

        let conv = Conv1d::new(weight.clone(), None, Conv1dConfig::default());
        let candle_out = conv.forward(&input).unwrap();

        let matmul = MatmulConv1d::new(weight, None, 0).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        assert_eq!(candle_out.dim(2).unwrap(), 3);
        assert_eq!(matmul_out.dim(2).unwrap(), 3);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-5,
            "MatmulConv1d (raw vec, no pad) vs Conv1d max diff: {diff}"
        );
    }

    // ──────────────────── MatmulConvTranspose1d 测试 ────────────────────

    /// 测试: MatmulConvTranspose1d 与 ConvTranspose1d 在 stride=3, kernel=6, padding=1 下输出一致
    /// (模拟 Block 4: 192→96, upsample_rate=3)
    #[test]
    fn test_matmul_conv_transpose1d_matches_candle_s3() {
        let device = Device::Cpu;
        let c_in = 8;
        let c_out = 4;
        let stride = 3;
        let kernel = stride * 2;
        let padding = stride / 2;
        let l_in = 20;

        let weight = Tensor::randn(0.0f32, 0.05, (c_in, c_out, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.02, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, l_in), &device).unwrap();

        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: padding,
            ..Default::default()
        };

        let candle = ConvTranspose1d::new(weight.clone(), Some(bias.clone()), config);
        let candle_out = candle.forward(&input).unwrap();

        let matmul = MatmulConvTranspose1d::new(weight, Some(bias), config).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        let expected_l_out = (l_in - 1) * stride - 2 * padding + kernel + padding;
        assert_eq!(matmul_out.dim(2).unwrap(), expected_l_out);
        assert_eq!(candle_out.dim(2).unwrap(), expected_l_out);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-4,
            "MatmulConvTranspose1d (s3) vs ConvTranspose1d max diff: {diff}"
        );
    }

    /// 测试: MatmulConvTranspose1d 与 ConvTranspose1d 在 stride=8, kernel=16, padding=4 下输出一致
    /// (模拟 Block 1: 1536→768, upsample_rate=8)
    #[test]
    fn test_matmul_conv_transpose1d_matches_candle_s8() {
        let device = Device::Cpu;
        let c_in = 16;
        let c_out = 8;
        let stride = 8;
        let kernel = stride * 2;
        let padding = stride / 2;
        let l_in = 10;

        let weight = Tensor::randn(0.0f32, 0.05, (c_in, c_out, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.02, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, l_in), &device).unwrap();

        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: padding,
            ..Default::default()
        };

        let candle = ConvTranspose1d::new(weight.clone(), Some(bias.clone()), config);
        let candle_out = candle.forward(&input).unwrap();

        let matmul = MatmulConvTranspose1d::new(weight, Some(bias), config).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        let expected_l_out = (l_in - 1) * stride - 2 * padding + kernel + padding;
        assert_eq!(matmul_out.dim(2).unwrap(), expected_l_out);
        assert_eq!(candle_out.dim(2).unwrap(), expected_l_out);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-4,
            "MatmulConvTranspose1d (s8) vs ConvTranspose1d max diff: {diff}"
        );
    }

    /// 测试: MatmulConvTranspose1d 无偏置也能工作
    #[test]
    fn test_matmul_conv_transpose1d_no_bias() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (8, 4, 6), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 50), &device).unwrap();

        let config = ConvTranspose1dConfig {
            stride: 3,
            padding: 1,
            output_padding: 1,
            ..Default::default()
        };

        let candle = ConvTranspose1d::new(weight.clone(), None, config);
        let candle_out = candle.forward(&input).unwrap();

        let matmul = MatmulConvTranspose1d::new(weight, None, config).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-4,
            "MatmulConvTranspose1d (no bias) vs ConvTranspose1d max diff: {diff}"
        );
    }

    /// 测试: MatmulConvTranspose1d 在大尺寸下与 ConvTranspose1d 一致
    /// (模拟实际解码器 Block 4: 192→96, l_in=500)
    #[test]
    fn test_matmul_conv_transpose1d_large_size() {
        let device = Device::Cpu;
        let c_in = 16;
        let c_out = 8;
        let stride = 3;
        let kernel = 6;
        let padding = 1;
        let l_in = 500;

        let weight = Tensor::randn(0.0f32, 0.01, (c_in, c_out, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.005, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, l_in), &device).unwrap();

        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: padding,
            ..Default::default()
        };

        let candle = ConvTranspose1d::new(weight.clone(), Some(bias.clone()), config);
        let candle_out = candle.forward(&input).unwrap();

        let matmul = MatmulConvTranspose1d::new(weight, Some(bias), config).unwrap();
        let matmul_out = matmul.forward(&input).unwrap();

        let expected_l_out = (l_in - 1) * stride - 2 * padding + kernel + padding;
        assert_eq!(matmul_out.dim(2).unwrap(), expected_l_out);
        assert_eq!(candle_out.dim(2).unwrap(), expected_l_out);

        let diff = max_abs_diff(&candle_out, &matmul_out);
        assert!(
            diff < 1e-3,
            "MatmulConvTranspose1d (large) vs ConvTranspose1d max diff: {diff}"
        );
    }

    /// 测试: FastConvTranspose1d 在 CPU 上选择 MatmulConvTranspose1d
    #[test]
    fn test_fast_conv_transpose1d_cpu_uses_matmul() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.1, (8, 4, 6), &device).unwrap();
        let bias = Tensor::zeros(4, DType::F32, &device).unwrap();

        let fast = FastConvTranspose1d::new(
            weight,
            Some(bias),
            ConvTranspose1dConfig {
                stride: 3,
                padding: 1,
                output_padding: 1,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        assert!(matches!(fast, FastConvTranspose1d::Matmul(_)));
    }

    /// 测试: FastConvTranspose1d forward 正确性 (对比 ConvTranspose1d)
    #[test]
    fn test_fast_conv_transpose1d_forward_correctness() {
        let device = Device::Cpu;
        let c_in = 16;
        let c_out = 8;
        let stride = 4;
        let kernel = 8;
        let padding = 2;
        let l_in = 100;

        let weight = Tensor::randn(0.0f32, 0.05, (c_in, c_out, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.02, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, l_in), &device).unwrap();

        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: padding,
            ..Default::default()
        };

        let candle = ConvTranspose1d::new(weight.clone(), Some(bias.clone()), config);
        let candle_out = candle.forward(&input).unwrap();

        let fast = FastConvTranspose1d::new(weight, Some(bias), config, &device).unwrap();
        let fast_out = fast.forward(&input).unwrap();

        let expected_l_out = (l_in - 1) * stride - 2 * padding + kernel + padding;
        assert_eq!(fast_out.dim(2).unwrap(), expected_l_out);

        let diff = max_abs_diff(&candle_out, &fast_out);
        assert!(
            diff < 1e-4,
            "FastConvTranspose1d vs ConvTranspose1d max diff: {diff}"
        );
    }

    // ──────────────────── forward_vec 测试 ────────────────────

    /// 辅助: 比较两个 &[f32] 的最大绝对差
    fn max_abs_diff_vec(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
    }

    /// 测试: MatmulConv1d::forward_vec 与 forward (Tensor) 输出一致
    #[test]
    fn test_matmul_conv1d_forward_vec_matches_forward() {
        let device = Device::Cpu;
        let c_out = 16;
        let c_in = 16;
        let kernel = 7;
        let padding = 3;
        let seq_len = 200;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let matmul = MatmulConv1d::new(weight, Some(bias), padding).unwrap();
        let tensor_out = matmul.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = matmul.forward_vec(&x_vec, c_in, seq_len, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-4,
            "MatmulConv1d forward_vec vs forward max diff: {diff}"
        );
    }

    /// 测试: MatmulConv1dK1::forward_vec 与 forward (Tensor) 输出一致
    #[test]
    fn test_matmul_conv1d_k1_forward_vec_matches_forward() {
        let device = Device::Cpu;
        let c_out = 32;
        let c_in = 64;
        let seq_len = 200;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, 1), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let matmul = MatmulConv1dK1::new(weight, Some(bias)).unwrap();
        let tensor_out = matmul.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = matmul.forward_vec(&x_vec, c_in, seq_len, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-5,
            "MatmulConv1dK1 forward_vec vs forward max diff: {diff}"
        );
    }

    /// 测试: MatmulConv1dK1::forward_vec_owned 与 forward (Tensor) 输出一致
    #[test]
    fn test_matmul_conv1d_k1_forward_vec_owned_matches_forward() {
        let device = Device::Cpu;
        let c_out = 32;
        let c_in = 64;
        let seq_len = 200;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, 1), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let matmul = MatmulConv1dK1::new(weight, Some(bias)).unwrap();
        let tensor_out = matmul.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = matmul
            .forward_vec_owned(x_vec, c_in, seq_len, &device)
            .unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-5,
            "MatmulConv1dK1 forward_vec_owned vs forward max diff: {diff}"
        );
    }

    /// 测试: DepthwiseConv1d::forward_vec 与 forward (Tensor) 输出一致
    #[test]
    fn test_depthwise_conv1d_forward_vec_matches_forward() {
        let device = Device::Cpu;
        let channels = 16;
        let kernel = 7;
        let padding = 3;
        let seq_len = 100;

        let weight = Tensor::randn(0.0f32, 0.1, (channels, 1, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (channels,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, channels, seq_len), &device).unwrap();

        let dw = DepthwiseConv1d::new(weight, Some(bias), padding).unwrap();
        let tensor_out = dw.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = dw.forward_vec(&x_vec, channels, seq_len, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-5,
            "DepthwiseConv1d forward_vec vs forward max diff: {diff}"
        );
    }

    /// 测试: FastConv1d::forward_vec 与 forward (Tensor) 输出一致
    #[test]
    fn test_fast_conv1d_forward_vec_matches_forward() {
        let device = Device::Cpu;
        let c_out = 16;
        let c_in = 16;
        let kernel = 7;
        let padding = 3;
        let seq_len = 200;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let fast = FastConv1d::new(
            weight,
            Some(bias),
            Conv1dConfig {
                padding,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let tensor_out = fast.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = fast.forward_vec(&x_vec, c_in, seq_len, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-4,
            "FastConv1d forward_vec vs forward max diff: {diff}"
        );
    }

    /// 测试: FastConv1d::forward_vec_owned (k=1) 与 forward (Tensor) 输出一致
    #[test]
    fn test_fast_conv1d_forward_vec_owned_matches_forward() {
        let device = Device::Cpu;
        let c_out = 32;
        let c_in = 64;
        let seq_len = 200;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, 1), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.05, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let fast = FastConv1d::new(weight, Some(bias), Conv1dConfig::default(), &device).unwrap();
        assert!(matches!(fast, FastConv1d::MatmulK1(_)));

        let tensor_out = fast.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = fast
            .forward_vec_owned(x_vec, c_in, seq_len, &device)
            .unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-5,
            "FastConv1d forward_vec_owned vs forward max diff: {diff}"
        );
    }

    // ──────────────────── Direct BLAS sgemm 测试 ────────────────────

    /// 测试: MatmulConv1d::forward_vec 无偏置时正确 (测试 beta=0.0 路径)
    #[test]
    fn test_matmul_conv1d_forward_vec_no_bias() {
        let device = Device::Cpu;
        let c_out = 16;
        let c_in = 16;
        let kernel = 7;
        let padding = 3;
        let seq_len = 200;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, kernel), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let matmul = MatmulConv1d::new(weight, None, padding).unwrap();
        let tensor_out = matmul.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = matmul.forward_vec(&x_vec, c_in, seq_len, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-4,
            "MatmulConv1d forward_vec (no bias) vs forward max diff: {diff}"
        );
    }

    /// 测试: MatmulConv1dK1::forward_vec 无偏置时正确 (测试 beta=0.0 路径)
    #[test]
    fn test_matmul_conv1d_k1_forward_vec_no_bias() {
        let device = Device::Cpu;
        let c_out = 32;
        let c_in = 64;
        let seq_len = 200;

        let weight = Tensor::randn(0.0f32, 0.1, (c_out, c_in, 1), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let matmul = MatmulConv1dK1::new(weight, None).unwrap();
        let tensor_out = matmul.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = matmul.forward_vec(&x_vec, c_in, seq_len, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-5,
            "MatmulConv1dK1 forward_vec (no bias) vs forward max diff: {diff}"
        );
    }

    /// 测试: MatmulConvTranspose1d::forward_vec 与 forward (Tensor) 输出一致
    /// 验证直接 cblas_sgemm (TransA) + col2im 路径正确性
    #[test]
    fn test_matmul_conv_transpose1d_forward_vec_matches_forward() {
        let device = Device::Cpu;
        let c_in = 16;
        let c_out = 8;
        let stride = 3;
        let kernel = 6;
        let padding = 1;
        let l_in = 100;

        let weight = Tensor::randn(0.0f32, 0.05, (c_in, c_out, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.02, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, l_in), &device).unwrap();

        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: padding,
            ..Default::default()
        };

        let matmul = MatmulConvTranspose1d::new(weight, Some(bias), config).unwrap();
        // Tensor path
        let tensor_out = matmul.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Vec path
        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = matmul.forward_vec(x_vec, c_in, l_in, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-4,
            "MatmulConvTranspose1d forward_vec vs forward max diff: {diff}"
        );
    }

    /// 测试: MatmulConvTranspose1d::forward_vec 无偏置时正确
    #[test]
    fn test_matmul_conv_transpose1d_forward_vec_no_bias() {
        let device = Device::Cpu;
        let c_in = 16;
        let c_out = 8;
        let stride = 4;
        let kernel = 8;
        let padding = 2;
        let l_in = 50;

        let weight = Tensor::randn(0.0f32, 0.05, (c_in, c_out, kernel), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, l_in), &device).unwrap();

        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: padding,
            ..Default::default()
        };

        let matmul = MatmulConvTranspose1d::new(weight, None, config).unwrap();
        let tensor_out = matmul.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = matmul.forward_vec(x_vec, c_in, l_in, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-4,
            "MatmulConvTranspose1d forward_vec (no bias) vs forward max diff: {diff}"
        );
    }

    /// 测试: FastConvTranspose1d::forward_vec 与 forward (Tensor) 输出一致
    #[test]
    fn test_fast_conv_transpose1d_forward_vec_matches_forward() {
        let device = Device::Cpu;
        let c_in = 16;
        let c_out = 8;
        let stride = 3;
        let kernel = 6;
        let padding = 1;
        let l_in = 100;

        let weight = Tensor::randn(0.0f32, 0.05, (c_in, c_out, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.02, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, l_in), &device).unwrap();

        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: padding,
            ..Default::default()
        };

        let fast = FastConvTranspose1d::new(weight, Some(bias), config, &device).unwrap();
        assert!(matches!(fast, FastConvTranspose1d::Matmul(_)));

        let tensor_out = fast.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = fast.forward_vec(x_vec, c_in, l_in, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-4,
            "FastConvTranspose1d forward_vec vs forward max diff: {diff}"
        );
    }

    /// 测试: MatmulConv1d::forward_vec 在大尺寸下与 forward 一致 (Block 4 规模)
    /// 验证直接 cblas_sgemm 在大矩阵下的数值稳定性
    #[test]
    fn test_matmul_conv1d_forward_vec_large_size() {
        let device = Device::Cpu;
        let c_out = 96;
        let c_in = 96;
        let kernel = 7;
        let padding = 3;
        let seq_len = 5000;

        let weight = Tensor::randn(0.0f32, 0.01, (c_out, c_in, kernel), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.01, (c_out,), &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, c_in, seq_len), &device).unwrap();

        let matmul = MatmulConv1d::new(weight, Some(bias), padding).unwrap();
        let tensor_out = matmul.forward(&input).unwrap();
        let tensor_vec = tensor_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_vec = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vec_out = matmul.forward_vec(&x_vec, c_in, seq_len, &device).unwrap();

        assert_eq!(tensor_vec.len(), vec_out.len());
        let diff = max_abs_diff_vec(&tensor_vec, &vec_out);
        assert!(
            diff < 1e-3,
            "MatmulConv1d forward_vec (large) vs forward max diff: {diff}"
        );
    }
}
