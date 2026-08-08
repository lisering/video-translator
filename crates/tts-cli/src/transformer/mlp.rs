//! MLP (SwiGLU) — 融合 gate/up 投影

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{Module, Tensor, D};
use candle_nn::VarBuilder;

use crate::transformer::qlinear::QLinear;

/// SwiGLU MLP — 融合 gate/up 投影
///
/// ```text
/// down_proj(silu(gate_up(x)[..inter]) * gate_up(x)[inter..])
/// ```
/// gate_proj 和 up_proj 合并为单次 matmul，减少 kernel launch
pub struct Mlp {
    /// 融合 gate+up 投影: [hidden, 2*intermediate] (可选量化)
    gate_up_proj: QLinear,
    down_proj: QLinear,
    intermediate_size: usize,
}

impl Mlp {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        quantize: Option<GgmlDType>,
    ) -> Result<Self> {
        // 融合 gate + up: 拼接两个 [inter, hidden] 权重为 [2*inter, hidden]
        let gate_weight = vb
            .pp("gate_proj")
            .get((intermediate_size, hidden_size), "weight")?;
        let up_weight = vb
            .pp("up_proj")
            .get((intermediate_size, hidden_size), "weight")?;
        let gate_up_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
        let gate_up_proj = QLinear::from_weight(gate_up_weight, quantize)?;

        let down_weight = vb
            .pp("down_proj")
            .get((hidden_size, intermediate_size), "weight")?;
        let down_proj = QLinear::from_weight(down_weight, quantize)?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 融合 gate+up: 单次 matmul 替代 2 次
        let gate_up = self.gate_up_proj.forward(x)?; // [..., 2*inter]
        let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let prod = gate.mul(&up)?;
        Ok(self.down_proj.forward(&prod)?)
    }
}
