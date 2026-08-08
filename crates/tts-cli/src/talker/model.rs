//! TalkerModel — 自回归语义 token 生成 Transformer

use std::collections::HashMap;

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::model_config::TalkerConfig;
use crate::transformer::{AnyKVCache, DecoderLayer, MRoPE, QLinear, RoPEType, RotaryEmbedding};

use super::tokens::{codec_tokens, tts_tokens};
use super::types::{Language, Speaker};

// ──────────────────────────── TextProjection ────────────────────────────

/// 文本投影 (SwiGLU): text_embed_dim → text_proj_intermediate → hidden_size
struct TextProjection {
    fc1: Linear,
    fc2: Linear,
}

impl TextProjection {
    fn new(config: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(
            config.text_embed_dim,
            config.text_proj_intermediate,
            vb.pp("linear_fc1"),
        )?;
        let fc2 = linear(
            config.text_proj_intermediate,
            config.hidden_size,
            vb.pp("linear_fc2"),
        )?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self
            .fc1
            .forward(x)
            .map_err(|e| anyhow::anyhow!("fc1: {e}"))?;
        let h = candle_nn::ops::silu(&h).map_err(|e| anyhow::anyhow!("silu: {e}"))?;
        self.fc2
            .forward(&h)
            .map_err(|e| anyhow::anyhow!("fc2: {e}"))
    }
}

// ──────────────────────────── TalkerModel ────────────────────────────

/// TalkerModel — 自回归语义 token 生成 Transformer
pub struct TalkerModel {
    text_embedding: Embedding,
    text_projection: TextProjection,
    codec_embedding: Embedding,
    /// 解码器层 (candle_engine 访问此字段以创建 KV cache)
    pub(crate) layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    codec_head: QLinear,
    rope: RoPEType,
    config: TalkerConfig,
    device: Device,
}

impl TalkerModel {
    /// 从权重构建
    ///
    /// - `quantize`: 量化格式 (None = 不量化，Some(Q8_0/Q4_0/Q4K) = 量化 TalkerModel 权重)
    ///   量化可将每步权重加载从 ~1.65GB (F32) 减至 ~420MB (Q8_0) 或 ~210MB (Q4_0)，
    ///   显著减少内存带宽瓶颈。启用量化时 dtype 必须为 F32。
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        config: TalkerConfig,
        device: &Device,
        dtype: DType,
        quantize: Option<GgmlDType>,
    ) -> Result<Self> {
        let vb = VarBuilder::from_tensors(weights.clone(), dtype, device);
        let talker = vb.pp("talker");
        let model = talker.pp("model");

        let layer_config = config.to_layer_config();

        let text_embedding = embedding(
            config.text_vocab_size,
            config.text_embed_dim,
            model.pp("text_embedding"),
        )?;
        let text_projection = TextProjection::new(&config, talker.pp("text_projection"))?;
        let codec_embedding = embedding(
            config.codec_vocab_size,
            config.hidden_size,
            model.pp("codec_embedding"),
        )?;
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, model.pp("norm"))?;

        // codec_head: 可选量化 (每步生成都要加载，量化可减少带宽)
        let codec_head_weight = talker
            .pp("codec_head")
            .get((config.codec_vocab_size, config.hidden_size), "weight")?;
        let codec_head = QLinear::from_weight(codec_head_weight, quantize)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                DecoderLayer::new(&layer_config, model.pp(format!("layers.{}", i)), quantize)
                    .map_err(|e| anyhow::anyhow!("Layer {} init failed: {}", i, e))
            })
            .collect::<Result<Vec<_>>>()?;

        let rope = if let Some(mrope_section) = config.mrope_section {
            RoPEType::Multimodal(MRoPE::new(
                config.head_dim,
                config.rope_theta,
                mrope_section,
                device,
            )?)
        } else {
            RoPEType::Standard(RotaryEmbedding::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                device,
            )?)
        };

        if let Some(q) = quantize {
            tracing::info!(
                "TalkerModel loaded: {} layers, hidden={}, heads={}, kv_heads={}, head_dim={}, quantized={:?}",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                q
            );
        } else {
            tracing::info!(
                "TalkerModel loaded: {} layers, hidden={}, heads={}, kv_heads={}, head_dim={}",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim
            );
        }

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            codec_head,
            rope,
            config,
            device: device.clone(),
        })
    }

    /// 构建 role prefix: [IM_START, ASSISTANT, NEWLINE] → text_embedding → text_projection
    fn build_role_prefix(&self) -> Result<Tensor> {
        use super::tokens::special_tokens;
        let role_ids = Tensor::new(
            &[
                special_tokens::IM_START,
                special_tokens::ASSISTANT,
                special_tokens::NEWLINE,
            ],
            &self.device,
        )?;
        let text_embed = self.text_embedding.forward(&role_ids)?;
        let projected = self.text_projection.forward(&text_embed)?;
        Ok(projected.unsqueeze(0)?)
    }

    /// 构建 TTS pad/bos 嵌入 (n_pad 个 TTS_PAD + 1 个 TTS_BOS)
    fn build_tts_pad_bos(&self, n_pad: usize) -> Result<Tensor> {
        let mut ids = vec![tts_tokens::TTS_PAD; n_pad];
        ids.push(tts_tokens::TTS_BOS);
        let ids_tensor = Tensor::new(ids.as_slice(), &self.device)?;
        let embed = self.text_embedding.forward(&ids_tensor)?;
        let projected = self.text_projection.forward(&embed)?;
        Ok(projected.unsqueeze(0)?)
    }

    /// 构建所有文本 token 的嵌入 (text_embedding → text_projection)
    ///
    /// 返回 [1, N, hidden_size] 的张量，N = text_tokens.len()。
    /// 所有文本 token 都被包含在 prefill 序列中，确保模型能"看到"完整文本。
    fn build_all_text_embeddings(&self, text_tokens: &[u32]) -> Result<Option<Tensor>> {
        if text_tokens.is_empty() {
            return Ok(None);
        }
        let text_ids = Tensor::new(text_tokens, &self.device)?;
        let text_embed = self.text_embedding.forward(&text_ids)?;
        let text_proj = self.text_projection.forward(&text_embed)?;
        Ok(Some(text_proj.unsqueeze(0)?)) // [1, N, hidden]
    }

    /// Prefill: CustomVoice 模式
    pub fn prefill_custom_voice(
        &self,
        text_tokens: &[u32],
        speaker: Speaker,
        language: Language,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        let role_prefix = self.build_role_prefix()?;

        let codec_ids = Tensor::new(
            &[
                codec_tokens::CODEC_THINK,
                codec_tokens::CODEC_THINK_BOS,
                language.token_id(),
                codec_tokens::CODEC_THINK_EOS,
                speaker.token_id(),
                codec_tokens::CODEC_PAD,
                codec_tokens::CODEC_BOS,
            ],
            &self.device,
        )?;
        let codec_embed = self.codec_embedding.forward(&codec_ids)?.unsqueeze(0)?;

        let tts_text_embed = self.build_tts_pad_bos(5)?;
        let codec_first6 = codec_embed.i((.., 0..6, ..))?;
        let codec_hidden = tts_text_embed.add(&codec_first6)?;

        let mut hidden = Tensor::cat(&[&role_prefix, &codec_hidden], 1)?;

        let codec_bos_embed = codec_embed.i((.., 6..7, ..))?;
        hidden = Tensor::cat(&[&hidden, &codec_bos_embed], 1)?;

        // 所有文本 token 作为独立位置加入 prefill 序列
        if let Some(text_proj) = self.build_all_text_embeddings(text_tokens)? {
            hidden = Tensor::cat(&[&hidden, &text_proj], 1)?;
        }

        self.run_prefill_layers(hidden, kv_caches)
    }

    /// Prefill: 声音克隆模式 (x_vector_only)
    pub fn prefill_voice_clone(
        &self,
        text_tokens: &[u32],
        speaker_embed: &Tensor,
        language: Language,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        let role_prefix = self.build_role_prefix()?;

        let codec_prefix_ids = Tensor::new(
            &[
                codec_tokens::CODEC_THINK,
                codec_tokens::CODEC_THINK_BOS,
                language.token_id(),
                codec_tokens::CODEC_THINK_EOS,
            ],
            &self.device,
        )?;
        let codec_prefix_embed = self
            .codec_embedding
            .forward(&codec_prefix_ids)?
            .unsqueeze(0)?;

        let speaker = speaker_embed
            .reshape((1, 1, self.config.hidden_size))?
            .to_dtype(codec_prefix_embed.dtype())?;

        let codec_suffix_ids = Tensor::new(
            &[codec_tokens::CODEC_PAD, codec_tokens::CODEC_BOS],
            &self.device,
        )?;
        let codec_suffix_embed = self
            .codec_embedding
            .forward(&codec_suffix_ids)?
            .unsqueeze(0)?;

        let codec_embed = Tensor::cat(&[&codec_prefix_embed, &speaker, &codec_suffix_embed], 1)?;

        let tts_text_embed = self.build_tts_pad_bos(5)?;
        let codec_first6 = codec_embed.i((.., 0..6, ..))?;
        let codec_hidden = tts_text_embed.add(&codec_first6)?;

        let mut hidden = Tensor::cat(&[&role_prefix, &codec_hidden], 1)?;

        let codec_bos_embed = codec_embed.i((.., 6..7, ..))?;
        hidden = Tensor::cat(&[&hidden, &codec_bos_embed], 1)?;

        // 所有文本 token 作为独立位置加入 prefill 序列
        if let Some(text_proj) = self.build_all_text_embeddings(text_tokens)? {
            hidden = Tensor::cat(&[&hidden, &text_proj], 1)?;
        }

        self.run_prefill_layers(hidden, kv_caches)
    }

    /// 运行 prefill layers
    fn run_prefill_layers(
        &self,
        hidden: Tensor,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        let mut h = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &self.rope, Some(&mut kv_caches[i]), 0, false)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.codec_head.forward(&h)?;
        Ok((h, logits))
    }

    /// 单步生成: 给定上一个 token，预测下一个
    pub fn step(
        &self,
        token: u32,
        add_tts_bos: bool,
        kv_caches: &mut [AnyKVCache],
        offset: usize,
    ) -> Result<Tensor> {
        let token_tensor = Tensor::new(&[token], &self.device)?;
        let mut embed = self.codec_embedding.forward(&token_tensor)?;
        embed = embed.unsqueeze(0)?;

        if add_tts_bos {
            let tts_bos_ids = Tensor::new(&[tts_tokens::TTS_BOS], &self.device)?;
            let tts_embed = self.text_embedding.forward(&tts_bos_ids)?;
            let tts_proj = self.text_projection.forward(&tts_embed)?;
            let tts_proj = tts_proj.unsqueeze(0)?;
            embed = embed.add(&tts_proj)?;
        }

        let mut h = embed;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &self.rope, Some(&mut kv_caches[i]), offset, true)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.codec_head.forward(&h)?;

        // 返回最后一个位置的 logits [batch, vocab_size]
        let seq_len = logits.dim(1)?;
        Ok(logits.i((.., seq_len - 1, ..))?)
    }

    /// 多 token 前向推理: 处理多个 token，返回所有位置的 logits
    ///
    /// 用于推测解码 (speculative decoding): 一次前向传播处理 [sampled_token, speculated_token]，
    /// 获取两个位置的 logits，从而验证推测 token 是否正确。
    ///
    /// 由于 causal attention，位置 t 的 logits 不依赖于位置 t+1 的 token，
    /// 因此即使推测 token 被拒绝，位置 t 的 logits 仍然正确。
    ///
    /// - `tokens`: 要处理的 token 列表
    /// - `add_tts_bos_first`: 是否仅对第一个 token 添加 TTS_BOS 嵌入
    /// - `kv_caches`: KV 缓存
    /// - `offset`: 序列偏移量
    ///
    /// 返回 [1, T, vocab_size] 的 logits（所有位置）
    pub fn step_multi(
        &self,
        tokens: &[u32],
        add_tts_bos_first: bool,
        kv_caches: &mut [AnyKVCache],
        offset: usize,
    ) -> Result<Tensor> {
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let mut embed = self.codec_embedding.forward(&token_tensor)?;
        embed = embed.unsqueeze(0)?; // [1, T, hidden]

        if add_tts_bos_first && !tokens.is_empty() {
            let tts_bos_ids = Tensor::new(&[tts_tokens::TTS_BOS], &self.device)?;
            let tts_embed = self.text_embedding.forward(&tts_bos_ids)?;
            let tts_proj = self.text_projection.forward(&tts_embed)?;
            let tts_proj = tts_proj.unsqueeze(0)?; // [1, 1, hidden]
                                                   // 仅对第一个 token 添加 TTS_BOS
            let first = embed.i((.., 0..1, ..))?;
            let first = first.add(&tts_proj)?;
            if tokens.len() > 1 {
                let rest = embed.i((.., 1.., ..))?.contiguous()?;
                embed = Tensor::cat(&[&first, &rest], 1)?;
            } else {
                embed = first;
            }
        }

        let mut h = embed;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &self.rope, Some(&mut kv_caches[i]), offset, true)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.codec_head.forward(&h)?;

        // 返回所有位置的 logits [1, T, vocab_size]
        Ok(logits)
    }

    pub fn config(&self) -> &TalkerConfig {
        &self.config
    }
}
