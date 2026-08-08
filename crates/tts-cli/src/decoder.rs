//! 音频解码器模块 — ConvNeXt + ConvTranspose1d 上采样
//!
//! 将 codec tokens 解码为音频波形。
//!
//! 基于 `speech_tokenizer/model.safetensors` 的实际权重结构实现。
//!
//! 解码流水线：
//! 1. Codebook 查表: 16 codebooks → embedding 求和 → [T, codebook_dim]
//! 2. output_proj: Conv1d(codebook_dim→hidden_size, k=1)
//! 3. pre_conv: Conv1d(hidden_size→latent_dim, k=3, pad=1)
//! 4. pre_transformer: 8 层 Transformer (hidden=512, SwiGLU, layer_scale)
//! 5. upsample: 2× (ConvTranspose1d(2x) + ConvNeXt block) = 4x
//! 6. decoder.decoder: Conv1d + 4× (LayerNorm + ConvTranspose1d + 3× ConvNeXt-like) = 480x
//! 7. final LayerNorm + Conv1d → 波形 [1, 1920×T]
//!
//! 总上采样: 4 × 480 = 1920x (12Hz → 24000Hz)

mod blocks;
mod codebook;
mod config;
mod conv_ops;
mod helpers;
mod pre_transformer;
mod rope;

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::conv::Conv1dConfig;
use candle_nn::VarBuilder;

use crate::audio::AudioBuffer;

use blocks::{DecoderUpsampleBlock, UpsampleBlock};
use codebook::{load_codebook, VQCodebook};
use config::DecoderConfig;
use conv_ops::FastConv1d;
use helpers::SnakeBeta;
use pre_transformer::PreTransformer;

// ──────────────────────────── AudioDecoder ────────────────────────────

pub struct AudioDecoder {
    sample_rate: u32,
    upsample_factor: usize,
    device: Option<Device>,
    dtype: DType,
    config: Option<DecoderConfig>,

    codebook_first: Option<VQCodebook>,
    codebooks_rest: Option<Vec<VQCodebook>>,
    output_proj_first: Option<FastConv1d>,
    output_proj_rest: Option<FastConv1d>,
    pre_conv: Option<FastConv1d>,
    pre_transformer: Option<PreTransformer>,
    upsample_blocks: Option<Vec<UpsampleBlock>>,
    decoder_initial_conv: Option<FastConv1d>,
    decoder_upsample_blocks: Option<Vec<DecoderUpsampleBlock>>,
    snake_beta: Option<SnakeBeta>,
    /// FastConv1d for CPU-optimized final conv (96→1, k=7)
    final_conv: Option<FastConv1d>,
}

impl AudioDecoder {
    pub fn new(sample_rate: u32) -> Self {
        let upsample_factor = (sample_rate as f64 / 12.0) as usize;
        Self {
            sample_rate,
            upsample_factor,
            device: None,
            dtype: DType::F32,
            config: None,
            codebook_first: None,
            codebooks_rest: None,
            output_proj_first: None,
            output_proj_rest: None,
            pre_conv: None,
            pre_transformer: None,
            upsample_blocks: None,
            decoder_initial_conv: None,
            decoder_upsample_blocks: None,
            snake_beta: None,
            final_conv: None,
        }
    }

    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        device: &Device,
        sample_rate: u32,
        compute_dtype: DType,
    ) -> Result<Self> {
        let config = DecoderConfig::default();
        let upsample_factor = config.upsample_rates.iter().product::<usize>()
            * config.upsampling_ratios.iter().product::<usize>();

        let vb = VarBuilder::from_tensors(weights.clone(), compute_dtype, device);
        let dec = vb.pp("decoder");

        // ── 1. VQ codebooks ──
        let codebook_first = load_codebook(
            weights,
            "decoder.quantizer.rvq_first.vq.layers.0._codebook",
            device,
        )?;

        let mut codebooks_rest = Vec::new();
        for i in 0..15 {
            let prefix = format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook");
            match load_codebook(weights, &prefix, device) {
                Ok(Some(cb)) => codebooks_rest.push(cb),
                Ok(None) => {
                    tracing::warn!("rvq_rest codebook {i} not found");
                    break;
                }
                Err(e) => {
                    tracing::warn!("Failed to load rvq_rest codebook {i}: {e}");
                    break;
                }
            }
        }

        let has_codebooks = codebook_first.is_some() && !codebooks_rest.is_empty();

        // ── 2. output_proj ──
        let output_proj_first = if codebook_first.is_some() {
            let w = dec.get(
                (config.codebook_dim, config.vq_hidden_dim, 1),
                "quantizer.rvq_first.output_proj.weight",
            )?;
            Some(FastConv1d::new(w, None, Conv1dConfig::default(), device)?)
        } else {
            None
        };

        let output_proj_rest = if !codebooks_rest.is_empty() {
            let w = dec.get(
                (config.codebook_dim, config.vq_hidden_dim, 1),
                "quantizer.rvq_rest.output_proj.weight",
            )?;
            Some(FastConv1d::new(w, None, Conv1dConfig::default(), device)?)
        } else {
            None
        };

        // pre_conv: CausalConvNet(512→1024, k=3) — padding=2 (causal: left=2, crop right 2)
        let pre_conv = {
            let w = dec.get(
                (config.latent_dim, config.codebook_dim, 3),
                "pre_conv.conv.weight",
            )?;
            let b = dec.get(config.latent_dim, "pre_conv.conv.bias")?;
            FastConv1d::new(
                w,
                Some(b),
                Conv1dConfig {
                    padding: 2,
                    ..Default::default()
                },
                device,
            )?
        };

        // ── 4. pre_transformer ──
        let pre_transformer = match PreTransformer::new(&config, dec.pp("pre_transformer")) {
            Ok(pt) => {
                tracing::info!("PreTransformer loaded: {} layers", config.num_hidden_layers);
                Some(pt)
            }
            Err(e) => {
                tracing::warn!("Failed to build PreTransformer: {e}");
                None
            }
        };

        // ── 5. upsample blocks ──
        let upsample_blocks = {
            let mut blocks = Vec::new();
            for (i, &ratio) in config.upsampling_ratios.iter().enumerate() {
                match UpsampleBlock::new(config.latent_dim, ratio, dec.pp(format!("upsample.{i}")))
                {
                    Ok(b) => blocks.push(b),
                    Err(e) => {
                        tracing::warn!("Failed to build upsample block {i}: {e}");
                        break;
                    }
                }
            }
            if blocks.len() == config.upsampling_ratios.len() {
                tracing::info!(
                    "Upsample: {} blocks, total {}x",
                    blocks.len(),
                    config.upsampling_ratios.iter().product::<usize>()
                );
                Some(blocks)
            } else {
                None
            }
        };

        // ── 6. decoder.decoder ──
        // decoder_initial_conv: CausalConvNet(1024→1536, k=7) — padding=6 (causal: left=6, crop right 6)
        let decoder_initial_conv = {
            let w = dec.get(
                (config.decoder_dim, config.latent_dim, 7),
                "decoder.0.conv.weight",
            )?;
            let b = dec.get(config.decoder_dim, "decoder.0.conv.bias")?;
            FastConv1d::new(
                w,
                Some(b),
                Conv1dConfig {
                    padding: 6,
                    ..Default::default()
                },
                device,
            )?
        };

        let decoder_upsample_blocks = {
            let channel_schedule = [config.decoder_dim, 768, 384, 192, 96];
            let mut blocks = Vec::new();
            for (i, &stride) in config.upsample_rates.iter().enumerate() {
                let in_ch = channel_schedule[i];
                let out_ch = channel_schedule[i + 1];
                match DecoderUpsampleBlock::new(
                    in_ch,
                    out_ch,
                    stride,
                    dec.pp(format!("decoder.{}", i + 1)),
                ) {
                    Ok(b) => blocks.push(b),
                    Err(e) => {
                        tracing::warn!("Failed to build decoder.decoder block {}: {e}", i + 1);
                        break;
                    }
                }
            }
            if blocks.len() == config.upsample_rates.len() {
                tracing::info!(
                    "Decoder.decoder: {} upsample blocks, total {}x",
                    blocks.len(),
                    config.upsample_rates.iter().product::<usize>()
                );
                Some(blocks)
            } else {
                None
            }
        };

        // SnakeBeta activation (NOT LayerNorm!) — decoder.5 is SnakeBeta, not LayerNorm
        // Formula: x + (1/exp(beta)) * sin(exp(alpha) * x)^2
        let snake_beta = match (
            dec.get(96, "decoder.5.alpha"),
            dec.get(96, "decoder.5.beta"),
        ) {
            (Ok(alpha), Ok(beta)) => Some(SnakeBeta::new(alpha, beta)?),
            _ => None,
        };

        // final_conv: CausalConvNet(96→1, k=7) — padding=6 (causal: left=6, crop right 6)
        let final_conv = {
            let w = dec.get((1, 96, 7), "decoder.6.conv.weight")?;
            let b = dec.get(1, "decoder.6.conv.bias")?;
            FastConv1d::new(
                w,
                Some(b),
                Conv1dConfig {
                    padding: 6,
                    ..Default::default()
                },
                device,
            )?
        };

        let all_loaded = has_codebooks
            && output_proj_first.is_some()
            && output_proj_rest.is_some()
            && pre_transformer.is_some()
            && upsample_blocks.is_some()
            && decoder_upsample_blocks.is_some();

        if all_loaded {
            tracing::info!(
                "AudioDecoder: full ConvNeXt+ConvTranspose1d decoder loaded, upsample={}x",
                upsample_factor
            );
        } else {
            tracing::warn!("AudioDecoder: partial load, will use fallback for missing components");
        }

        Ok(Self {
            sample_rate,
            upsample_factor,
            device: Some(device.clone()),
            dtype: compute_dtype,
            config: Some(config),
            codebook_first,
            codebooks_rest: if codebooks_rest.is_empty() {
                None
            } else {
                Some(codebooks_rest)
            },
            output_proj_first,
            output_proj_rest,
            pre_conv: Some(pre_conv),
            pre_transformer,
            upsample_blocks,
            decoder_initial_conv: Some(decoder_initial_conv),
            decoder_upsample_blocks,
            snake_beta,
            final_conv: Some(final_conv),
        })
    }

    #[cfg(not(any(feature = "cpu", feature = "metal", feature = "cuda")))]
    pub fn from_weights(
        _weights: &HashMap<String, Tensor>,
        _device: &Device,
        sample_rate: u32,
        _compute_dtype: DType,
    ) -> Result<Self> {
        Ok(Self::new(sample_rate))
    }

    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<AudioBuffer> {
        if codes.is_empty() {
            return Ok(AudioBuffer::new(self.sample_rate));
        }

        let num_codebooks = codes.len();
        let num_frames = codes[0].len();

        tracing::info!(
            "Decoder: {} codebooks x {} frames → {}Hz output",
            num_codebooks,
            num_frames,
            self.sample_rate
        );

        #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
        if self.is_full_decoder_ready() {
            return self.decode_neural(codes);
        }

        self.decode_sine_wave(codes)
    }

    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    fn is_full_decoder_ready(&self) -> bool {
        self.codebook_first.is_some()
            && self.codebooks_rest.is_some()
            && self.output_proj_first.is_some()
            && self.output_proj_rest.is_some()
            && self.pre_conv.is_some()
            && self.pre_transformer.is_some()
            && self.upsample_blocks.is_some()
            && self.decoder_initial_conv.is_some()
            && self.decoder_upsample_blocks.is_some()
            && self.final_conv.is_some()
    }

    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    fn decode_neural(&self, codes: &[Vec<u32>]) -> Result<AudioBuffer> {
        let device = self.device.as_ref().unwrap();
        let config = self.config.as_ref().unwrap();
        let num_codebooks = codes.len();
        let num_frames = codes[0].len();

        // 限制 codec token 范围: VQ codebook 大小为 config.codebook_size (2048)，
        // 超出范围的 token 会导致 index_select 越界，在 Metal 上触发 crash。
        let codebook_size = config.codebook_size;
        let max_token = (codebook_size - 1) as u32;
        let clamped_codes: Vec<Vec<u32>> = codes
            .iter()
            .map(|cb| {
                cb.iter()
                    .map(|&t| {
                        if (t as usize) >= codebook_size {
                            tracing::warn!(
                                "Codec token {} exceeds codebook_size {}, clamping to {}",
                                t,
                                codebook_size,
                                max_token
                            );
                            max_token
                        } else {
                            t
                        }
                    })
                    .collect()
            })
            .collect();

        // ── 1. Codebook 查表 ──
        let t0 = Instant::now();
        let first_codes = Tensor::new(clamped_codes[0].as_slice(), device)?;
        let first_embed = self.codebook_first.as_ref().unwrap().lookup(&first_codes)?;

        let mut rest_embed_sum =
            Tensor::zeros((num_frames, config.vq_hidden_dim), self.dtype, device)?;
        let rest_cbs = self.codebooks_rest.as_ref().unwrap();
        for (i, cb) in rest_cbs.iter().enumerate() {
            if i + 1 >= num_codebooks {
                break;
            }
            let token_ids = Tensor::new(clamped_codes[i + 1].as_slice(), device)?;
            let embed = cb.lookup(&token_ids)?;
            rest_embed_sum = rest_embed_sum.add(&embed)?;
        }

        let t_codebook = t0.elapsed();

        // ── 2. output_proj (1x1 conv) ──
        let t1 = Instant::now();
        let first_embed = first_embed.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
        let first_out = self
            .output_proj_first
            .as_ref()
            .unwrap()
            .forward(&first_embed)?;
        let rest_embed = rest_embed_sum.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
        let rest_out = self
            .output_proj_rest
            .as_ref()
            .unwrap()
            .forward(&rest_embed)?;
        let mut h = first_out.add(&rest_out)?;

        // ── 3. pre_conv (causal padding) ──
        let pre_conv_len = h.dim(2)?;
        h = self.pre_conv.as_ref().unwrap().forward(&h)?;
        h = h.narrow(2, 0, pre_conv_len)?.contiguous()?; // crop right padding → causal
        let t_proj = t1.elapsed();

        // ── 4. pre_transformer ──
        let t2 = Instant::now();
        if let Some(ref pt) = self.pre_transformer {
            h = pt.forward(&h)?;
        }
        let t_pre_transformer = t2.elapsed();

        // ── 5. upsample blocks ──
        let t3 = Instant::now();
        if let Some(ref blocks) = self.upsample_blocks {
            for block in blocks {
                h = block.forward(&h)?;
            }
        }
        let t_upsample = t3.elapsed();

        // ── 6. decoder.decoder ──
        let t4 = Instant::now();
        // decoder_initial_conv (causal padding)
        let init_conv_len = h.dim(2)?;
        h = self.decoder_initial_conv.as_ref().unwrap().forward(&h)?;
        h = h.narrow(2, 0, init_conv_len)?.contiguous()?; // crop right padding → causal

        // 细粒度计时: 设置 VT_TTS_DECODE_TIMING=1 环境变量启用
        let enable_fine_timing = std::env::var("VT_TTS_DECODE_TIMING").is_ok();

        // CPU 全 Vec 链式路径: 块间保持 Vec 空间, 消除 Tensor→Vec→Tensor 往返
        // GPU 路径: 保持 Tensor 链式调用
        if h.device().is_cpu() && h.dtype() == DType::F32 {
            if let Some(ref blocks) = self.decoder_upsample_blocks {
                if !blocks.is_empty() {
                    let device = h.device();
                    // Extract initial Tensor to Vec for first block
                    let (_batch, c_in_init, l_in_init) = h.dims3()?;
                    let mut h_vec = h.flatten_all()?.to_vec1::<f32>()?;
                    let mut c_in = c_in_init;
                    let mut l_in = l_in_init;

                    for (i, block) in blocks.iter().enumerate() {
                        let tb = Instant::now();
                        if enable_fine_timing {
                            h_vec = block.forward_timed_vec(h_vec, c_in, l_in, device)?;
                        } else {
                            h_vec = block.forward_vec(h_vec, c_in, l_in, device)?;
                        }
                        // Update c_in/l_in for next block
                        c_in = block.out_channels;
                        l_in = h_vec.len() / c_in;
                        tracing::info!(
                            "  decoder block {} in {:.1}ms",
                            i + 1,
                            tb.elapsed().as_secs_f64() * 1000.0
                        );
                    }

                    // Convert final Vec back to Tensor for final norm
                    h = Tensor::from_vec(h_vec, (1, c_in, l_in), device)?;
                }
            }
        } else {
            // GPU path: Tensor-based chaining
            if let Some(ref blocks) = self.decoder_upsample_blocks {
                for (i, block) in blocks.iter().enumerate() {
                    let tb = Instant::now();
                    if enable_fine_timing {
                        h = block.forward_timed(&h)?;
                    } else {
                        h = block.forward(&h)?;
                    }
                    tracing::info!(
                        "  decoder block {} in {:.1}ms",
                        i + 1,
                        tb.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
        }
        let t_decoder = t4.elapsed();

        // SnakeBeta activation (replaces previous incorrect LayerNorm)
        let t5 = Instant::now();
        if let Some(ref sb) = self.snake_beta {
            h = sb.forward(&h)?;
        }

        let t_norm = t5.elapsed();

        // final conv — CausalConvNet: left_pad=k-1, right_pad=0
        // FastConv1d with padding=k-1 gives symmetric pad; crop right k-1 for causal.
        let t6 = Instant::now();
        let h_len = h.dim(2)?;
        h = self.final_conv.as_ref().unwrap().forward(&h)?;
        h = h.narrow(2, 0, h_len)?.contiguous()?; // crop right padding → causal
        let t_final_conv = t6.elapsed();

        tracing::info!(
            "  Decoder timing: codebook={:.1}ms, proj+pre_conv={:.1}ms, pre_transformer={:.1}ms, upsample={:.1}ms, decoder_blocks={:.1}ms, norm={:.1}ms, final_conv={:.1}ms",
            t_codebook.as_secs_f64() * 1000.0,
            t_proj.as_secs_f64() * 1000.0,
            t_pre_transformer.as_secs_f64() * 1000.0,
            t_upsample.as_secs_f64() * 1000.0,
            t_decoder.as_secs_f64() * 1000.0,
            t_norm.as_secs_f64() * 1000.0,
            t_final_conv.as_secs_f64() * 1000.0,
        );

        // ── 7. 提取波形 ──
        // 提取波形: 先转为 F32 再提取 (F16 张量不支持 to_vec1::<f32>)
        let h_f32 = h.to_dtype(DType::F32)?;
        let samples = h_f32.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;

        // ── 归一化 ──
        // final_conv 权重天然极小 (abs_mean=0.000133)，decoder 输出需要归一化到正常幅度。
        // 参考 Python 实现的 round-trip 测试显示 decoder 输出 peak ≈ 0.65 (无需归一化)，
        // 但那是因为 Python 用的是 round-trip codec tokens。TTS 生成的 tokens 可能产生更小的输出。
        let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let normalized: Vec<f32> = if max_val > 1.0 {
            tracing::warn!("Decoder output peak {max_val:.4} > 1.0, clipping");
            samples.iter().map(|&s| s.clamp(-1.0, 1.0)).collect()
        } else if max_val > 1e-6 {
            // Normalize to 0.95 peak
            let scale = 0.95 / max_val;
            samples
                .iter()
                .map(|&s| (s * scale).clamp(-1.0, 1.0))
                .collect()
        } else {
            tracing::warn!("Decoder output appears silent (max_val={max_val:.6})");
            samples
        };

        tracing::info!(
            "Neural decode: {} frames → {} samples ({:.1}s), peak={:.4} ({:.1}dB)",
            num_frames,
            normalized.len(),
            normalized.len() as f64 / self.sample_rate as f64,
            max_val,
            if max_val > 1e-6 {
                20.0 * max_val.log10()
            } else {
                -999.0
            },
        );

        let mut audio = AudioBuffer::from_samples(normalized, self.sample_rate);
        Self::apply_fade_inplace(&mut audio.samples, self.sample_rate);
        Ok(audio)
    }

    fn decode_sine_wave(&self, codes: &[Vec<u32>]) -> Result<AudioBuffer> {
        let num_codebooks = codes.len();
        let num_frames = codes[0].len();
        let total_samples = num_frames * self.upsample_factor;
        let mut samples = vec![0.0f32; total_samples];

        for frame_idx in 0..num_frames {
            let tokens: Vec<u32> = (0..num_codebooks)
                .map(|cb| codes[cb].get(frame_idx).copied().unwrap_or(0))
                .collect();

            let freq = 100.0 + (tokens[0] % 200) as f64;
            let amplitude = 0.1 + (tokens.get(1).copied().unwrap_or(0) % 10) as f64 * 0.02;

            let start = frame_idx * self.upsample_factor;
            let end = (start + self.upsample_factor).min(total_samples);

            for i in start..end {
                let t = i as f64 / self.sample_rate as f64;
                let mut sample = (t * freq * 2.0 * std::f64::consts::PI).sin() * amplitude;
                sample += (t * freq * 2.0 * 2.0 * std::f64::consts::PI).sin() * amplitude * 0.3;
                sample += (t * freq * 3.0 * 2.0 * std::f64::consts::PI).sin() * amplitude * 0.1;
                samples[i] = sample as f32;
            }
        }

        Self::apply_fade_inplace(&mut samples, self.sample_rate);
        Ok(AudioBuffer::from_samples(samples, self.sample_rate))
    }

    fn apply_fade_inplace(samples: &mut [f32], sample_rate: u32) {
        let fade_samples = (sample_rate as f64 * 0.005) as usize;
        for i in 0..fade_samples.min(samples.len()) {
            let gain = i as f32 / fade_samples as f32;
            samples[i] *= gain;
        }
        for i in 0..fade_samples.min(samples.len()) {
            let gain = i as f32 / fade_samples as f32;
            let idx = samples.len() - 1 - i;
            samples[idx] *= gain;
        }
    }

    pub fn decode_single(&self, tokens: &[u32]) -> Result<AudioBuffer> {
        let codes = vec![tokens.to_vec()];
        self.decode(&codes)
    }
}

impl std::fmt::Debug for AudioDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioDecoder")
            .field("sample_rate", &self.sample_rate)
            .field("upsample_factor", &self.upsample_factor)
            .field("has_neural_decoder", &self.final_conv.is_some())
            .finish()
    }
}
