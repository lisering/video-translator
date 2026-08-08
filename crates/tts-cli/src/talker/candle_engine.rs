//! CandleTtsEngine — 完整的 Qwen3-TTS 推理流水线
//!
//! 三阶段流水线:
//! 1. TalkerModel (Transformer): 文本 → 语义 token 序列
//! 2. CodePredictor (自回归解码器): 语义 token → 声学 token (16 codebooks)
//! 3. AudioDecoder: codec tokens → 音频波形

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::audio::AudioBuffer;
use crate::config::{SystemInfo, TtsEngineConfig};
use crate::decoder::AudioDecoder;
use crate::engine::{SynthesisOptions, SynthesisResult, TtsEngine, VoiceClonePrompt};
use crate::model_config::{CodePredictorConfig, ModelType, ParsedModelConfig, TalkerConfig};
use crate::speaker::SpeakerEncoder;
use crate::tokenizer::TextTokenizer;
use crate::transformer::AnyKVCache;

use super::code_predictor::CodePredictor;
use super::model::TalkerModel;
use super::sampling::{is_ngram_banned, parse_quantize, sample_top_k_gpu, update_ngram_table};
use super::tokens::codec_tokens;
use super::types::{Language, Speaker};
use super::weights::{
    compute_dtype_for_device, convert_weights_dtype, create_device, load_safetensors,
};

// ──────────────────────────── CandleTtsEngine ────────────────────────────

/// Candle TTS 引擎 — 完整的 Qwen3-TTS 推理流水线
///
/// 当模型权重不可用时，自动降级到简化模式。
pub struct CandleTtsEngine {
    config: TtsEngineConfig,
    tokenizer: TextTokenizer,
    talker: Option<TalkerModel>,
    code_predictor: Option<CodePredictor>,
    decoder: AudioDecoder,
    speaker_encoder: Option<SpeakerEncoder>,
    sys_info: SystemInfo,
    model_variant: String,
    device: Device,
    /// AudioDecoder 运行设备 (可能与主设备不同)
    #[allow(dead_code)]
    decode_device: Device,
    #[allow(dead_code)]
    model_type: Option<ModelType>,
    degraded: bool,
    /// TalkerModel 的 dtype (F32/F16/BF16)，用于预分配 KV cache
    talker_dtype: DType,
    /// TalkerModel 权重量化格式 (None = 不量化)
    quantize: Option<GgmlDType>,
}

impl CandleTtsEngine {
    /// 创建引擎实例 — 从模型目录加载权重
    pub fn new(config: TtsEngineConfig) -> Result<Self> {
        let model_dir = &config.model_dir;
        let device = create_device(&config.device)?;
        let sys_info = SystemInfo::detect();

        // 解析量化格式
        let quantize = parse_quantize(&config.quantize);
        if let Some(q) = quantize {
            tracing::info!(
                "Quantization enabled: {:?} — TalkerModel weights will be quantized",
                q
            );
        }

        // 量化与混合精度互斥: 量化要求 F32 输入 (Metal 量化 matmul 要求)，
        // 而混合精度使用 F16/BF16。当量化启用时，强制使用 F32 并禁用混合精度。
        let (talker_dtype, other_dtype) = if quantize.is_some() {
            if config.mixed_precision {
                tracing::warn!(
                    "Quantization overrides mixed precision: using F32 for TalkerModel (quantized matmul requires F32)"
                );
            }
            (DType::F32, DType::F32)
        } else if config.mixed_precision {
            match &device {
                Device::Metal(_) => {
                    tracing::info!(
                        "Mixed precision enabled: TalkerModel=F16, CodePredictor/Decoder=F32"
                    );
                    (DType::F16, DType::F32)
                }
                Device::Cuda(_) => {
                    tracing::info!(
                        "Mixed precision enabled: TalkerModel=BF16, CodePredictor/Decoder=F32"
                    );
                    (DType::BF16, DType::F32)
                }
                _ => {
                    tracing::info!(
                        "Mixed precision not beneficial on CPU, using F32 for all components"
                    );
                    (DType::F32, DType::F32)
                }
            }
        } else {
            let dtype = compute_dtype_for_device(&device);
            (dtype, dtype)
        };

        if !model_dir.exists() {
            tracing::warn!(
                "Model directory not found: {:?}. Engine will run in degraded mode.",
                model_dir
            );
            return Self::new_degraded(config, device, sys_info);
        }

        // 加载分词器 (支持 tokenizer.json 或 vocab.json + merges.txt)
        let tokenizer = TextTokenizer::from_model_dir(model_dir)?;

        // 解析 config.json
        let config_path = model_dir.join("config.json");
        let parsed_config = if config_path.exists() {
            match ParsedModelConfig::from_file(&config_path) {
                Ok(cfg) => {
                    tracing::info!("Detected model variant: {}", cfg.label());
                    Some(cfg)
                }
                Err(e) => {
                    tracing::warn!("Failed to parse config.json: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // 检查模型权重
        let model_weights_path = model_dir.join("model.safetensors");
        if !model_weights_path.exists() {
            tracing::warn!(
                "model.safetensors not found in {:?}. Degraded mode.",
                model_dir
            );
            return Self::new_degraded(config, device, sys_info);
        }

        let raw_weights = match load_safetensors(&model_weights_path, &device) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("Failed to load model weights: {}. Degraded mode.", e);
                return Self::new_degraded(config, device, sys_info);
            }
        };

        // 混合精度: 分别为 TalkerModel 和 CodePredictor/SpeakerEncoder 准备不同 dtype 的权重
        // 模型权重通常是 BF16，需要转换为目标 dtype
        let talker_weights = convert_weights_dtype(raw_weights.clone(), talker_dtype);
        let other_weights = convert_weights_dtype(raw_weights, other_dtype);

        // 构建 TalkerModel
        let talker_config = if let Some(ref pc) = parsed_config {
            pc.to_talker_config()
        } else if let Some(norm_weight) = talker_weights.get("talker.model.norm.weight") {
            let hidden_size = norm_weight.dim(0).unwrap_or(1024);
            if hidden_size == 2048 {
                TalkerConfig::large()
            } else {
                TalkerConfig::default()
            }
        } else {
            TalkerConfig::default()
        };

        let talker = match TalkerModel::from_weights(
            &talker_weights,
            talker_config,
            &device,
            talker_dtype,
            quantize,
        ) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!("Failed to build TalkerModel: {}. Degraded mode.", e);
                return Self::new_degraded(config, device, sys_info);
            }
        };
        // TalkerModel 已吸收权重，释放 F16 权重 HashMap
        drop(talker_weights);

        // 构建 CodePredictor (使用 F32 权重)
        let cp_config = if let Some(ref pc) = parsed_config {
            pc.to_code_predictor_config()
        } else {
            CodePredictorConfig::default()
        };

        let code_predictor = CodePredictor::new(cp_config, &other_weights, &device, other_dtype)
            .map_err(|e| {
                tracing::warn!(
                    "Failed to build CodePredictor: {}. Will use simplified decoder.",
                    e
                );
            })
            .ok();

        // ── 解码器设备选择 ──
        // 默认与主设备相同; 可通过 config.decode_device 覆盖
        // 设置为 "cpu" 可将解码器运行在 CPU 上 (适用于 Metal Conv1d 效率低下的场景)
        let decode_device_str = config.decode_device.as_deref().unwrap_or(&config.device);
        let decode_device = create_device(decode_device_str)?;
        let decode_is_cpu = matches!(decode_device, Device::Cpu);

        if decode_device_str != config.device {
            tracing::info!(
                "Decode device override: {} (main device: {})",
                decode_device_str,
                config.device
            );
        }

        // 解码器 dtype: CPU 始终使用 F32 (CPU F16 是软件模拟, 无硬件加速)
        // Metal 可通过 VT_TTS_DECODER_DTYPE=f16 独立控制
        let decoder_dtype = if decode_is_cpu {
            DType::F32
        } else {
            match std::env::var("VT_TTS_DECODER_DTYPE")
                .unwrap_or_default()
                .to_lowercase()
                .as_str()
            {
                "f16" | "half" => {
                    tracing::info!("Decoder dtype: F16 (experimental, set VT_TTS_DECODER_DTYPE=f32 to disable)");
                    DType::F16
                }
                _ => other_dtype,
            }
        };

        // 加载 Decoder 权重 — 使用解码器设备
        let st_path = model_dir.join("speech_tokenizer/model.safetensors");
        let decoder_weights = if st_path.exists() {
            load_safetensors(&st_path, &decode_device)
                .ok()
                .map(|w| convert_weights_dtype(w, decoder_dtype))
        } else {
            model_dir
                .parent()
                .map(|p| p.join("speech_tokenizer/model.safetensors"))
                .filter(|p| p.exists())
                .and_then(|p| load_safetensors(&p, &decode_device).ok())
                .map(|w| convert_weights_dtype(w, decoder_dtype))
        };

        let decoder = if let Some(ref dw) = decoder_weights {
            AudioDecoder::from_weights(dw, &decode_device, 24000, decoder_dtype).unwrap_or_else(
                |e| {
                    tracing::warn!("Failed to build Decoder12Hz: {}. Using basic decoder.", e);
                    AudioDecoder::new(24000)
                },
            )
        } else {
            tracing::warn!("Speech tokenizer weights not found, using basic decoder");
            AudioDecoder::new(24000)
        };

        // 说话人编码器 (使用 F32 权重，conv 层在 F32 下性能更佳)
        let model_type = parsed_config.as_ref().map(|c| c.model_type);
        let has_speaker_encoder =
            matches!(model_type, Some(ModelType::Base)) || model_type.is_none();
        let speaker_encoder = if has_speaker_encoder {
            let se_config = parsed_config
                .as_ref()
                .and_then(|c| c.speaker_encoder_config.clone())
                .unwrap_or_default();

            match SpeakerEncoder::from_weights(&other_weights, se_config.clone(), &device) {
                Ok(se) => {
                    tracing::info!("Speaker encoder loaded from weights");
                    Some(se)
                }
                Err(e) => {
                    tracing::warn!("Failed to load speaker encoder: {}. Using statistical.", e);
                    Some(SpeakerEncoder::new(se_config.enc_dim))
                }
            }
        } else {
            None
        };
        // 释放 other_weights (CodePredictor 和 SpeakerEncoder 已吸收所需权重)
        drop(other_weights);

        let model_variant = parsed_config
            .as_ref()
            .map(|c| c.label())
            .unwrap_or_else(|| "unknown".to_string());

        tracing::info!(
            "CandleTtsEngine initialized: variant={}, device={}, threads={}",
            model_variant,
            config.device,
            sys_info.cpu_threads
        );

        Ok(Self {
            config,
            tokenizer,
            talker,
            code_predictor,
            decoder,
            speaker_encoder,
            sys_info,
            model_variant,
            device,
            decode_device,
            model_type,
            degraded: false,
            talker_dtype,
            quantize,
        })
    }

    /// 降级模式引擎 (无模型权重)
    fn new_degraded(config: TtsEngineConfig, device: Device, sys_info: SystemInfo) -> Result<Self> {
        Ok(Self {
            config,
            tokenizer: TextTokenizer::fallback(),
            talker: None,
            code_predictor: None,
            decoder: AudioDecoder::new(24000),
            speaker_encoder: Some(SpeakerEncoder::new(1024)),
            sys_info,
            model_variant: "0.6B-Base (degraded)".to_string(),
            device: device.clone(),
            decode_device: device,
            model_type: None,
            degraded: true,
            talker_dtype: DType::F32,
            quantize: None,
        })
    }

    /// 预填充
    fn prefill(
        &self,
        text: &str,
        voice_clone: Option<&VoiceClonePrompt>,
        language: Language,
        speaker: Option<Speaker>,
    ) -> Result<PrefillResult> {
        let start = Instant::now();
        let text_ids = self.tokenizer.encode_for_tts(text)?;

        let text_preview: String = text.chars().take(50).collect();
        tracing::info!(
            "Prefill: text=\"{}{}\" → {} tokens",
            text_preview,
            if text.chars().count() > 50 { "..." } else { "" },
            text_ids.len()
        );

        if self.degraded || self.talker.is_none() {
            return Ok(PrefillResult {
                text_ids,
                speaker_embedding: voice_clone.map(|vc| vc.speaker_embedding.clone()),
                kv_caches: None,
                last_logits: None,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                degraded: true,
            });
        }

        let talker = self.talker.as_ref().unwrap();

        // 预分配 KV cache: 避免每步 Tensor::cat 的 O(n²) 内存复制
        // 缓冲区大小 = prefill 长度 + 最大生成长度 + 安全余量
        let max_kv_len = text_ids.len() + 10 + self.config.max_codes + 100;
        let mut kv_caches: Vec<AnyKVCache> = talker
            .layers
            .iter()
            .map(|_| {
                AnyKVCache::new_preallocated(
                    max_kv_len,
                    talker.config().num_key_value_heads,
                    talker.config().head_dim,
                    self.talker_dtype,
                    &self.device,
                )
                .unwrap_or_else(|_| AnyKVCache::new())
            })
            .collect();

        tracing::info!(
            "KV cache: pre-allocated (max_len={}, layers={}, dtype={:?})",
            max_kv_len,
            talker.layers.len(),
            self.talker_dtype
        );

        let (_hidden, logits) = if let Some(vc) = voice_clone {
            let speaker_embed = Tensor::from_vec(
                vc.speaker_embedding.clone(),
                vc.speaker_embedding.len(),
                &self.device,
            )?;
            talker.prefill_voice_clone(&text_ids, &speaker_embed, language, &mut kv_caches)?
        } else {
            let sp = speaker.unwrap_or(Speaker::Ryan);
            talker.prefill_custom_voice(&text_ids, sp, language, &mut kv_caches)?
        };

        Ok(PrefillResult {
            text_ids,
            speaker_embedding: voice_clone.map(|vc| vc.speaker_embedding.clone()),
            kv_caches: Some(kv_caches),
            last_logits: Some(logits),
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            degraded: false,
        })
    }

    /// 自回归生成语义 token (带 n-gram 推测解码)
    ///
    /// 推测解码 (Speculative Decoding) 策略:
    /// 1. 维护 n-gram 推测表: 记录每个 (n-1)-token 前缀最近出现的下一个 token
    /// 2. 每步尝试推测: 查表获取可能的下一个 token
    /// 3. 若推测命中: 在单次前向传播中处理 [current_token, speculated_token]
    ///    - 由于 causal attention，位置 t 的 logits 不依赖位置 t+1
    ///    - 验证: 从 logits_0 采样，若与推测 token 一致则接受，省去一次前向传播
    ///    - 若不一致则回滚 KV cache，使用采样结果继续
    /// 4. 若无推测: 正常单 token 前向传播
    ///
    /// 性能分析: 每步前向传播是内存带宽瓶颈 (加载 ~2.4GB 模型权重)，
    /// 处理 2 个 token 仅增加 ~20% 计算 (权重加载不变)，
    /// 若命中率 30%+ 可减少 15-25% 前向传播次数。
    fn generate(
        &self,
        prefill: &PrefillResult,
        options: &SynthesisOptions,
        kv_caches: &mut Vec<AnyKVCache>,
        last_logits: &Tensor,
    ) -> Result<GenerationResult> {
        let start = Instant::now();
        let limits = self.sys_info.smart_limits();
        let max_codes = options.max_codes.min(limits.max_codes);

        if let Some(ref warning) = limits.warning {
            tracing::warn!("System: {}", warning);
        }

        if prefill.degraded || self.talker.is_none() {
            return Ok(GenerationResult {
                semantic_tokens: self.generate_degraded(&prefill.text_ids, options, max_codes)?,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
        }

        let talker = self.talker.as_ref().unwrap();

        // 从 prefill 最后位置 logits 采样第一个 token
        let seq_len = last_logits.dim(1)?;
        let first_logits = last_logits.i((.., seq_len - 1, ..))?;
        // 使用 flatten_all 确保 1D，避免 squeeze 在某些设备上不生效
        let first_logits = first_logits.squeeze(0)?.flatten_all()?;
        let mut current_token = sample_top_k_gpu(
            &first_logits,
            options.top_k,
            options.temperature,
            options.seed,
            options.repetition_penalty,
            options.no_repeat_ngram_size,
            &[], // 第一个 token 无历史
        )?;

        let mut tokens: Vec<u32> = Vec::with_capacity(max_codes);
        let mut step = 1usize;
        let prefill_len = prefill.text_ids.len() + 10; // role(3) + codec(7)

        // 检查第一个 token 是否为 EOS
        if current_token == codec_tokens::CODEC_EOS {
            tracing::warn!("Model predicted EOS at first token, output will be empty");
        } else {
            tokens.push(current_token);
        }

        // n-gram 推测表: (n-1)-token 前缀 → 最近出现的下一个 token
        // 使用 2-gram (最后 1 个 token → 下一个 token)，命中率较高
        let ngram_spec_size = 2usize;
        let mut ngram_table: HashMap<Vec<u32>, u32> = HashMap::new();

        // 推测解码统计
        let mut spec_attempts = 0usize;
        let mut spec_hits = 0usize;

        while step < max_codes {
            if current_token == codec_tokens::CODEC_EOS {
                break;
            }

            let offset = prefill_len + step - 1;
            let add_tts_bos = step == 1;

            // ── n-gram 推测: 查找可能的下一个 token ──
            // 仅在启用推测解码时查表，否则始终走正常路径
            let speculated = if options.speculative && !tokens.is_empty() {
                let prefix = &tokens[tokens.len() - 1..]; // 最后 1 个 token
                ngram_table.get(prefix).copied()
            } else {
                None
            };

            // 检查推测 token 是否被 no-repeat-ngram 禁止
            let speculated =
                speculated.filter(|&b| !is_ngram_banned(&tokens, b, options.no_repeat_ngram_size));

            if let Some(spec_token) = speculated {
                // ── 推测路径: 一次前向传播处理 [current_token, spec_token] ──
                spec_attempts += 1;
                let tokens_to_process = [current_token, spec_token];
                let multi_logits =
                    talker.step_multi(&tokens_to_process, add_tts_bos, kv_caches, offset)?; // [1, 2, vocab_size]

                // logits_0: 位置 current_token 之后 (用于验证 spec_token)
                let logits_0 = multi_logits.i((.., 0, ..))?.flatten_all()?;
                // logits_1: 位置 spec_token 之后 (用于采样下一个 token)
                let logits_1 = multi_logits.i((.., 1, ..))?.flatten_all()?;

                // 从 logits_0 采样 (与正常路径相同的采样逻辑)
                // 历史包含 tokens，不包含 spec_token
                let verified_token = sample_top_k_gpu(
                    &logits_0,
                    options.top_k,
                    options.temperature,
                    options.seed.map(|s| s + step as u64),
                    options.repetition_penalty,
                    options.no_repeat_ngram_size,
                    &tokens,
                )?;

                if verified_token == spec_token {
                    // ── 接受 spec_token: 省去一次前向传播 ──
                    spec_hits += 1;
                    if spec_token != codec_tokens::CODEC_EOS {
                        tokens.push(spec_token);
                        update_ngram_table(&mut ngram_table, &tokens, ngram_spec_size);
                    }
                    step += 1;

                    if spec_token == codec_tokens::CODEC_EOS {
                        break;
                    }

                    // 从 logits_1 采样下一个 token (历史包含 spec_token)
                    current_token = sample_top_k_gpu(
                        &logits_1,
                        options.top_k,
                        options.temperature,
                        options.seed.map(|s| s + step as u64),
                        options.repetition_penalty,
                        options.no_repeat_ngram_size,
                        &tokens,
                    )?;

                    if current_token != codec_tokens::CODEC_EOS {
                        tokens.push(current_token);
                        update_ngram_table(&mut ngram_table, &tokens, ngram_spec_size);
                    }
                    step += 1;
                } else {
                    // ── 拒绝 spec_token: 回滚 KV cache ──
                    for cache in kv_caches.iter_mut() {
                        cache.rollback(1)?;
                    }

                    // 使用正常采样结果作为下一个 token
                    current_token = verified_token;
                    if current_token != codec_tokens::CODEC_EOS {
                        tokens.push(current_token);
                        update_ngram_table(&mut ngram_table, &tokens, ngram_spec_size);
                    }
                    step += 1;
                }
            } else {
                // ── 正常路径: 单 token 前向传播 ──
                let logits = talker.step(current_token, add_tts_bos, kv_caches, offset)?;
                let logits = logits.flatten_all()?;

                current_token = sample_top_k_gpu(
                    &logits,
                    options.top_k,
                    options.temperature,
                    options.seed.map(|s| s + step as u64),
                    options.repetition_penalty,
                    options.no_repeat_ngram_size,
                    &tokens,
                )?;

                // 只在非 EOS 时添加 token（避免 EOS 进入后续 CodePredictor 导致越界）
                if current_token != codec_tokens::CODEC_EOS {
                    tokens.push(current_token);
                    update_ngram_table(&mut ngram_table, &tokens, ngram_spec_size);
                }
                step += 1;
            }

            // 重复检测: 如果最近 10 个 token 中有 8 个相同，说明模型陷入了循环
            if tokens.len() >= 10 {
                let last = tokens[tokens.len() - 1];
                let recent = &tokens[tokens.len() - 10..];
                let repeat_count = recent.iter().filter(|&&t| t == last).count();
                if repeat_count >= 8 {
                    tracing::warn!(
                        "Repetition detected (token {} appeared {} times in last 10), stopping early at step {}",
                        last, repeat_count, step
                    );
                    break;
                }
            }

            if step % 50 == 0 {
                tracing::debug!("Generation step {}: {} tokens", step, tokens.len());
            }
        }

        if spec_attempts > 0 {
            tracing::info!(
                "Speculative decoding: {}/{} hits ({:.1}%), saved ~{} forward passes",
                spec_hits,
                spec_attempts,
                spec_hits as f64 / spec_attempts as f64 * 100.0,
                spec_hits
            );
        }

        tracing::info!(
            "Generation: {} semantic tokens in {:.1}ms",
            tokens.len(),
            start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(GenerationResult {
            semantic_tokens: tokens,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// 降级模式的 token 生成
    fn generate_degraded(
        &self,
        text_ids: &[u32],
        options: &SynthesisOptions,
        max_codes: usize,
    ) -> Result<Vec<u32>> {
        let estimated_tokens = (text_ids.len() * 5).min(max_codes);
        let tokens: Vec<u32> = (0..estimated_tokens)
            .map(|i| {
                let base = text_ids.get(i % text_ids.len()).copied().unwrap_or(0);
                let noise = if let Some(seed) = options.seed {
                    ((seed as u64).wrapping_add(i as u64)) as u32 % 100
                } else {
                    (i * 7) as u32 % 100
                };
                base + noise
            })
            .collect();
        Ok(tokens)
    }

    /// 声学解码: 语义 token → 音频波形
    ///
    /// **安全检查**: 在传递给 CodePredictor 之前，过滤掉特殊 token (>= vocab_size)，
    /// 防止 embedding 查表越界。在传递给 decoder 之前，限制 codec token 范围。
    fn decode_to_audio(
        &self,
        semantic_tokens: &[u32],
        output_sample_rate: u32,
    ) -> Result<AudioBuffer> {
        let start = Instant::now();

        if self.code_predictor.is_some() && !self.degraded {
            let cp = self.code_predictor.as_ref().unwrap();

            // 过滤特殊 token: TalkerModel 的 codec_vocab_size=3072，但 CodePredictor 的
            // vocab_size=2048。特殊 token (>= 2048) 如 CODEC_EOS=2150, CODEC_THINK=2154
            // 会导致 CodePredictor 的 embedding 查表越界，在 Metal 上触发 crash。
            // 替换为 0 (CODEC_PAD 的等效值) 以保持序列长度对齐。
            let cp_vocab = cp.config().vocab_size;
            let filtered_tokens: Vec<u32> = semantic_tokens
                .iter()
                .map(|&t| {
                    if (t as usize) >= cp_vocab {
                        tracing::debug!(
                            "Filtering special token {} (>= vocab_size {}) before CodePredictor",
                            t,
                            cp_vocab
                        );
                        0
                    } else {
                        t
                    }
                })
                .collect();

            let special_count = semantic_tokens
                .iter()
                .filter(|&&t| (t as usize) >= cp_vocab)
                .count();
            if special_count > 0 {
                tracing::warn!(
                    "Filtered {} special tokens (>= {}) from {} semantic tokens before CodePredictor",
                    special_count, cp_vocab, semantic_tokens.len()
                );
            }

            let codec_frames = cp.generate(&filtered_tokens, None)?;

            tracing::info!(
                "CodePredictor: {} semantic → {} codec frames in {:.1}ms",
                semantic_tokens.len(),
                codec_frames.len(),
                start.elapsed().as_secs_f64() * 1000.0
            );

            // 转换为 [num_codebooks][T] 格式
            //
            // 关键: TalkerModel 的语义 token 是 decoder 的第一个 codebook (rvq_first),
            // CodePredictor 生成剩余 15 个 codebook (rvq_rest)。
            // 如果不前置语义 token, 所有 codebook 会错位一格, decoder 产生噪声。
            let num_cp_codebooks = codec_frames.first().map(|f| f.len()).unwrap_or(0);
            let num_codebooks = num_cp_codebooks + 1; // +1 for semantic (codebook 0)
            let num_frames = codec_frames.len();
            let mut codes: Vec<Vec<u32>> = vec![Vec::with_capacity(num_frames); num_codebooks];

            // codebook 0 = 语义 token (已过滤特殊 token, 范围 [0, cp_vocab))
            codes[0] = filtered_tokens.clone();

            // codebooks 1..num_codebooks = CodePredictor 输出
            for frame in &codec_frames {
                for (cb, &token) in frame.iter().enumerate() {
                    let target_cb = cb + 1; // 偏移 1, 因为 codebook 0 是语义 token
                    if target_cb < num_codebooks {
                        codes[target_cb].push(token);
                    }
                }
            }

            tracing::info!(
                "Decoder input: {} codebooks x {} frames (semantic + {} CP codebooks)",
                num_codebooks,
                num_frames,
                num_cp_codebooks
            );

            let audio = self.decoder.decode(&codes)?;

            tracing::info!(
                "Decode: {} frames → {} samples ({:.1}s) in {:.1}ms",
                codec_frames.len(),
                audio.num_samples(),
                audio.duration_secs(),
                start.elapsed().as_secs_f64() * 1000.0
            );

            Ok(audio)
        } else {
            // 降级: 正弦波合成
            let num_frames = semantic_tokens.len();
            let samples_per_frame = (output_sample_rate as f64 / 12.0) as usize;
            let total_samples = num_frames * samples_per_frame;
            let mut samples = Vec::with_capacity(total_samples);
            let base_freq = 220.0f64;

            for (frame_idx, &token) in semantic_tokens.iter().enumerate() {
                let freq = base_freq * (1.0 + (token % 50) as f64 / 100.0);
                let amplitude = 0.3 * (1.0 - (frame_idx as f64 / num_frames as f64).min(1.0) * 0.3);

                for i in 0..samples_per_frame {
                    let t = (frame_idx * samples_per_frame + i) as f64 / output_sample_rate as f64;
                    let sample = (t * freq * 2.0 * std::f64::consts::PI).sin() * amplitude
                        + (t * freq * 2.0 * 2.0 * std::f64::consts::PI).sin() * amplitude * 0.3;
                    samples.push(sample as f32);
                }
            }

            tracing::info!(
                "Decode (degraded): {} frames → {} samples ({:.1}s)",
                num_frames,
                samples.len(),
                samples.len() as f64 / output_sample_rate as f64
            );

            Ok(AudioBuffer::from_samples(samples, output_sample_rate))
        }
    }
}

impl TtsEngine for CandleTtsEngine {
    fn synthesize(
        &self,
        text: &str,
        voice_clone: Option<&VoiceClonePrompt>,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        let total_start = Instant::now();

        // 语言检测: 优先使用配置的语言，"auto" 则从文本自动检测
        let language = if self.config.language == "auto" {
            let detected = Language::detect_from_text(text);
            tracing::info!("Language: auto-detected as {:?}", detected);
            detected
        } else {
            Language::from_str(&self.config.language)
        };
        let speaker = if voice_clone.is_some() {
            None
        } else {
            Some(Speaker::Ryan)
        };

        // 阶段 1: Prefill
        let prefill = self.prefill(text, voice_clone, language, speaker)?;

        // 阶段 2: Generate
        let (semantic_tokens, gen_ms) = if prefill.degraded {
            let dummy = Tensor::new(&[0f32], &self.device)?;
            let mut dummy_caches = Vec::new();
            let r = self.generate(&prefill, options, &mut dummy_caches, &dummy)?;
            (r.semantic_tokens, r.elapsed_ms)
        } else {
            let mut kv_caches = prefill.kv_caches.clone().unwrap_or_default();
            let last_logits = prefill.last_logits.as_ref().unwrap();
            let r = self.generate(&prefill, options, &mut kv_caches, last_logits)?;
            (r.semantic_tokens, r.elapsed_ms)
        };

        // 阶段 3: Decode
        let audio = self.decode_to_audio(&semantic_tokens, self.config.output_sample_rate)?;

        let elapsed = total_start.elapsed().as_secs_f64();
        let audio_duration = audio.duration_secs();
        let rtf = if audio_duration > 0.0 {
            elapsed / audio_duration
        } else {
            0.0
        };

        tracing::info!(
            "Synthesis: {:.1}s audio, {:.1}s compute, RTF={:.3}x (prefill={:.1}ms, gen={:.1}ms)",
            audio_duration,
            elapsed,
            rtf,
            prefill.elapsed_ms,
            gen_ms
        );

        Ok(SynthesisResult {
            audio,
            elapsed_secs: elapsed,
            num_frames: semantic_tokens.len(),
            rtf,
        })
    }

    fn create_voice_clone_prompt(
        &self,
        ref_audio_path: &Path,
        ref_text: Option<&str>,
    ) -> Result<VoiceClonePrompt> {
        let speaker_encoder = self.speaker_encoder.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Model variant ({}) does not support voice cloning. Use a Base variant.",
                self.model_variant
            )
        })?;

        let embedding = speaker_encoder.extract_embedding(ref_audio_path)?;

        let ref_text_ids = if let Some(text) = ref_text {
            Some(self.tokenizer.encode(text)?)
        } else {
            None
        };

        Ok(VoiceClonePrompt {
            speaker_embedding: embedding,
            ref_codes: None,
            ref_text_ids,
        })
    }

    fn name(&self) -> &str {
        if self.degraded {
            "vt-tts-candle (degraded)"
        } else {
            "vt-tts-candle"
        }
    }

    fn supports_voice_cloning(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    fn model_variant(&self) -> &str {
        &self.model_variant
    }
}

// ──────────────────────────── 内部结构 ────────────────────────────

struct PrefillResult {
    text_ids: Vec<u32>,
    #[allow(dead_code)]
    speaker_embedding: Option<Vec<f32>>,
    kv_caches: Option<Vec<AnyKVCache>>,
    last_logits: Option<Tensor>,
    elapsed_ms: f64,
    degraded: bool,
}

struct GenerationResult {
    semantic_tokens: Vec<u32>,
    elapsed_ms: f64,
}

impl std::fmt::Debug for CandleTtsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleTtsEngine")
            .field("model_variant", &self.model_variant)
            .field("device", &self.config.device)
            .field("decode_device", &self.config.decode_device)
            .field("supports_voice_cloning", &self.supports_voice_cloning())
            .field("degraded", &self.degraded)
            .field("has_talker", &self.talker.is_some())
            .field("has_code_predictor", &self.code_predictor.is_some())
            .field("talker_dtype", &self.talker_dtype)
            .field("quantize", &self.quantize)
            .finish()
    }
}
