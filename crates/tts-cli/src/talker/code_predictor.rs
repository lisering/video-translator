//! CodePredictor — 语义 token → 声学 token (16 codebooks)

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::model_config::CodePredictorConfig;
use crate::transformer::{AnyKVCache, DecoderLayer, RoPEType, RotaryEmbedding};

use super::sampling::argmax_on_device;

/// CodePredictor — 语义 token → 声学 token (16 codebooks)
pub struct CodePredictor {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    heads: Vec<Linear>,
    small_to_mtp_projection: Option<Linear>,
    rope: RoPEType,
    config: CodePredictorConfig,
    device: Device,
}

impl CodePredictor {
    pub fn new(
        config: CodePredictorConfig,
        weights: &HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = VarBuilder::from_tensors(weights.clone(), dtype, device);
        let cp = vb.pp("talker.code_predictor");
        let model = cp.pp("model");

        let layer_config = config.to_layer_config();

        let embed_dim = config.codec_embed_dim.unwrap_or(config.hidden_size);
        // 尝试加载 codec_embedding.0 (实际权重命名) 或 embed_tokens (后备)
        let embed_tokens =
            embedding(config.vocab_size, embed_dim, model.pp("codec_embedding.0"))
                .or_else(|_| embedding(config.vocab_size, embed_dim, model.pp("embed_tokens")))?;

        let small_to_mtp_projection = None;

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, model.pp("norm"))?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                DecoderLayer::new(&layer_config, model.pp(format!("layers.{}", i)), None)
                    .map_err(|e| anyhow::anyhow!("CP layer {} init failed: {}", i, e))
            })
            .collect::<Result<Vec<_>>>()?;

        // 实际权重命名为 lm_head.{i}，而非 heads.{i}
        // CodePredictor 有 num_code_groups-1 个 lm_head (第一个 codebook 由 Talker 生成)
        let max_heads = config.num_code_groups;
        let mut heads = Vec::with_capacity(max_heads);
        for i in 0..max_heads {
            let head = linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                cp.pp(format!("lm_head.{}", i)),
            )
            .or_else(|_| {
                linear_no_bias(
                    config.hidden_size,
                    config.vocab_size,
                    cp.pp(format!("heads.{}", i)),
                )
            });
            match head {
                Ok(h) => heads.push(h),
                Err(_) => {
                    if i == 0 {
                        return Err(anyhow::anyhow!(
                            "CP head 0 init failed: no lm_head or heads found"
                        ));
                    }
                    tracing::debug!("CP head {} not found, loaded {} heads", i, heads.len());
                    break;
                }
            }
        }

        let rope = RoPEType::Standard(RotaryEmbedding::new(
            config.head_dim,
            4096,
            config.rope_theta,
            device,
        )?);

        tracing::info!(
            "CodePredictor loaded: {} layers, {} heads, hidden={}",
            config.num_hidden_layers,
            config.num_code_groups,
            config.hidden_size
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            heads,
            small_to_mtp_projection,
            rope,
            config,
            device: device.clone(),
        })
    }

    /// 从语义 token 生成声学 token
    ///
    /// 输入: 语义 token 序列 [T]
    /// 输出: [T, num_code_groups] 的 codec 帧
    ///
    /// **安全检查**: 所有输入 token 会被限制在 [0, vocab_size) 范围内，
    /// 防止特殊 token (如 CODEC_EOS=2150) 导致 embedding 查表越界崩溃。
    pub fn generate(
        &self,
        semantic_tokens: &[u32],
        talker_hidden: Option<&Tensor>,
    ) -> Result<Vec<Vec<u32>>> {
        if semantic_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // 验证并限制 token 范围: CodePredictor 的 embed_tokens 词表只有 vocab_size 个条目
        // TalkerModel 可能生成特殊 token (>= vocab_size) 如 CODEC_EOS=2150, CODEC_THINK=2154 等
        // 这些 token 会导致 embedding 查表越界，在 Metal 上触发 crash (exit code 101)
        let vocab_size = self.config.vocab_size;
        let validated_tokens: Vec<u32> = semantic_tokens
            .iter()
            .map(|&t| {
                if (t as usize) >= vocab_size {
                    tracing::warn!(
                        "Token {} exceeds CodePredictor vocab_size {}, clamping to 0",
                        t,
                        vocab_size
                    );
                    0
                } else {
                    t
                }
            })
            .collect();

        let seq_len = validated_tokens.len();
        let tokens = Tensor::new(validated_tokens.as_slice(), &self.device)?;
        let mut h = self.embed_tokens.forward(&tokens)?;
        h = h.unsqueeze(0)?;

        if let Some(ref proj) = self.small_to_mtp_projection {
            h = proj.forward(&h)?;
        }

        // 注入 talker hidden states (add)
        if let Some(th) = talker_hidden {
            let th_len = th.dim(1)?;
            let start = th_len.saturating_sub(seq_len);
            let th_slice = th.i((.., start.., ..))?;
            if th_slice.dim(1)? == seq_len {
                h = h.add(&th_slice)?;
            }
        }

        let mut kv_caches: Vec<AnyKVCache> =
            self.layers.iter().map(|_| AnyKVCache::new()).collect();

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &self.rope, Some(&mut kv_caches[i]), 0, false)?;
        }
        let h = self.norm.forward(&h)?;

        // ── 批量推理: 对每个 head 一次性处理所有 T 帧 ──
        // 旧实现: for t in 0..T { for head { head.forward(h.i(t)) } } = T×H 次串行 kernel
        // 新实现: for head { head.forward(h) } + argmax_on_device = H 次串行 kernel
        let h_2d = h.squeeze(0)?; // [T, hidden]
        let num_heads = self.heads.len();
        let mut frames: Vec<Vec<u32>> = (0..seq_len)
            .map(|_| Vec::with_capacity(num_heads))
            .collect();

        for head in &self.heads {
            // [T, hidden] → [T, vocab_size]
            let logits = head.forward(&h_2d)?;
            // GPU-native batch argmax: [T, vocab] → [T]
            let tokens = argmax_on_device(&logits)?;
            for (t, &token) in tokens.iter().enumerate() {
                frames[t].push(token);
            }
        }

        Ok(frames)
    }

    pub fn config(&self) -> &CodePredictorConfig {
        &self.config
    }
}
