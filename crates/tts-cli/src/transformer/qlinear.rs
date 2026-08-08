//! QLinear — 可量化的线性层

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Module, Tensor};
use candle_nn::Linear;

/// 可量化的线性层 — 支持常规 F32 Linear 或量化 QMatMul
///
/// 当 `quantize` 参数为 `Some(dtype)` 时，将权重量化为 GGML 格式 (Q8_0/Q4_0/Q4K 等)，
/// 使用 Metal 原生量化 matmul 内核。量化可将权重大小减少 2-8x，
/// 显著降低自回归生成阶段的内存带宽瓶颈。
///
/// **适用场景**: TalkerModel 的 28 层 Transformer (qkv_proj, o_proj, gate_up_proj, down_proj)。
/// 每步生成需加载 ~1.65GB F32 权重，Q8_0 可减至 ~420MB (4x)，Q4_0 可减至 ~210MB (8x)。
///
/// **限制**: Metal 量化 matmul 要求 F32 输入，输出始终为 F32。
/// 因此启用量化时强制使用 F32 dtype (不支持与 mixed_precision 同时使用)。
#[derive(Debug)]
pub enum QLinear {
    /// 常规 Linear 层 (F32/F16)
    Linear(Linear),
    /// 量化 matmul (Q8_0/Q4_0/Q4K 等)
    Quantized(QMatMul),
}

impl QLinear {
    /// 从权重张量创建 QLinear
    ///
    /// - `weight`: 权重张量 [out_dim, in_dim]
    /// - `quantize`: 量化格式 (None = 不量化，使用常规 Linear)
    pub fn from_weight(weight: Tensor, quantize: Option<GgmlDType>) -> candle_core::Result<Self> {
        if let Some(dtype) = quantize {
            let qtensor = QTensor::quantize(&weight, dtype)?;
            let qmatmul = QMatMul::from_qtensor(qtensor)?;
            Ok(Self::Quantized(qmatmul))
        } else {
            Ok(Self::Linear(Linear::new(weight, None)))
        }
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Linear(l) => l.forward(xs),
            Self::Quantized(q) => q.forward(xs),
        }
    }
}
