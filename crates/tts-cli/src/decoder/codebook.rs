//! VQ Codebook — 向量量化码本查表

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{Device, Tensor};

pub(crate) struct VQCodebook {
    embeddings: Tensor,
}

impl VQCodebook {
    pub(crate) fn from_weights(embedding_sum: &Tensor, cluster_usage: &Tensor) -> Result<Self> {
        // Reference: Qwen3-TTS Python (EuclideanCodebook.decode):
        //   embedding = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]
        //
        // cluster_usage is an EMA of how often each entry was used during training.
        // embedding_sum is the corresponding EMA of embedding vectors.
        // Their ratio gives the actual codebook entry.
        let epsilon = 1e-5f32;
        let usage_vals = cluster_usage.to_vec1::<f32>()?;
        let usage_clamped: Vec<f32> = usage_vals.iter().map(|&v| v.max(epsilon)).collect();
        let usage_tensor = Tensor::new(usage_clamped.as_slice(), cluster_usage.device())?;
        let embeddings = embedding_sum.broadcast_div(&usage_tensor.unsqueeze(1)?)?;

        // Debug: log codebook stats
        let emb_vals = embeddings.flatten_all()?.to_vec1::<f32>()?;
        let min_v = emb_vals.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_v = emb_vals.iter().fold(0.0f32, |a, &b| a.max(b));
        let mean_v = emb_vals.iter().sum::<f32>() / emb_vals.len() as f32;
        tracing::info!(
            "Codebook: shape={:?}, min={:.4}, max={:.4}, mean={:.6}, entries={}",
            embeddings.dims(),
            min_v,
            max_v,
            mean_v,
            embedding_sum.dim(0)?
        );

        Ok(Self { embeddings })
    }

    pub(crate) fn lookup(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embeddings
            .index_select(tokens, 0)
            .map_err(|e| anyhow::anyhow!("codebook lookup: {e}"))
    }
}

// ──────────────────────────── 辅助函数 ────────────────────────────

#[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
pub(crate) fn load_codebook(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    device: &Device,
) -> Result<Option<VQCodebook>> {
    let embed_sum_key = format!("{prefix}.embedding_sum");
    let usage_key = format!("{prefix}.cluster_usage");

    let embedding_sum = match weights.get(&embed_sum_key) {
        Some(t) => t,
        None => return Ok(None),
    };
    let cluster_usage = match weights.get(&usage_key) {
        Some(t) => t,
        None => return Ok(None),
    };

    let embedding_sum = embedding_sum.to_device(device)?;
    let cluster_usage = cluster_usage.to_device(device)?;

    VQCodebook::from_weights(&embedding_sum, &cluster_usage).map(Some)
}
