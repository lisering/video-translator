//! 说话人编码器模块
//!
//! 实现完整的 ECAPA-TDNN 说话人编码器，从参考音频提取说话人嵌入。
//!
//! # 架构 (ECAPA-TDNN)
//! 1. 输入: mel-频谱图 [1, n_mels, T]
//! 2. 初始 TDNN 层: Conv1d (reflect-padded "same") + ReLU
//! 3. SE-Res2Net 块 ×3: TDNN1 → Res2Net(scale=8) → TDNN2 → SE → residual
//! 4. MFA: 拼接 3 个 SE-Res2Net 输出 → TDNN 投影
//! 5. ASP: 注意力统计池化 (attention-weighted mean + std)
//! 6. FC: Conv1d(2C → enc_dim, k=1) → 1024 维嵌入
//!
//! # 权重前缀
//! `speaker_encoder.*` (blocks.0, blocks.1-3, mfa, asp, fc)
//!
//! # 参考
//! - TrevorS/qwen3-tts-rs: `src/models/speaker.rs`
//! - ECAPA-TDNN 原始论文: Desplanques et al., "ECAPA-TDNN for Speaker Verification"

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::audio::{extract_mel_spectrogram, AudioBuffer, MelConfig};
use crate::model_config::SpeakerEncoderConfig;

// ════════════════════════════════════════════════════════════════════════
//  Candle 依赖的 ECAPA-TDNN 实现
// ════════════════════════════════════════════════════════════════════════

#[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
mod ecapa {
    use anyhow::Result;
    use candle_core::{Device, Module, Tensor, D};
    use candle_nn::{conv1d, Conv1d, Conv1dConfig, VarBuilder};

    // ── 辅助函数 ──────────────────────────────────────────────────────

    /// 1D 反射填充 (reflect padding)
    ///
    /// 在时间维度 (最后一维) 上镜像信号边界，匹配 PyTorch 的
    /// `padding_mode="reflect"`。
    fn reflect_pad_1d(x: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
        if pad_left == 0 && pad_right == 0 {
            return Ok(x.clone());
        }
        let x = &x.contiguous()?;
        let (_b, _c, t) = x.dims3()?;
        let mut indices = Vec::with_capacity(pad_left + t + pad_right);

        // 左侧反射: 从位置 1 向外镜像
        for i in (1..=pad_left).rev() {
            indices.push(i as i64);
        }
        // 原始信号
        for i in 0..t {
            indices.push(i as i64);
        }
        // 右侧反射: 从位置 t-2 向内镜像
        for i in 0..pad_right {
            indices.push((t - 2 - i) as i64);
        }

        let idx = Tensor::from_vec(indices, (pad_left + t + pad_right,), x.device())?;
        Ok(x.index_select(&idx, D::Minus1)?)
    }

    fn relu(x: &Tensor) -> Result<Tensor> {
        Ok(x.maximum(&x.zeros_like()?)?)
    }

    fn sigmoid(x: &Tensor) -> Result<Tensor> {
        let neg_x = x.neg()?;
        let exp_neg_x = neg_x.exp()?;
        Ok((exp_neg_x + 1.0)?.recip()?)
    }

    // ── Conv1d + 反射填充 ─────────────────────────────────────────────

    /// Conv1d with "same" output length via reflect padding.
    ///
    /// 匹配 PyTorch `Conv1d(padding="same", padding_mode="reflect")`。
    struct ReflectPadConv1d {
        conv: Conv1d,
        pad_left: usize,
        pad_right: usize,
    }

    impl ReflectPadConv1d {
        fn new(
            in_channels: usize,
            out_channels: usize,
            kernel_size: usize,
            dilation: usize,
            vb: VarBuilder,
        ) -> Result<Self> {
            let total_pad = dilation * (kernel_size - 1);
            let pad_left = total_pad / 2;
            let pad_right = total_pad - pad_left;
            let config = Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation,
                groups: 1,
                ..Default::default()
            };
            let conv = conv1d(in_channels, out_channels, kernel_size, config, vb)?;
            Ok(Self {
                conv,
                pad_left,
                pad_right,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let padded = reflect_pad_1d(x, self.pad_left, self.pad_right)?;
            Ok(self.conv.forward(&padded)?)
        }
    }

    // ── 构建块 ────────────────────────────────────────────────────────

    /// TDNN 块: Conv1d (反射填充) + ReLU
    ///
    /// 权重键: `conv.weight`, `conv.bias`
    struct TimeDelayNetBlock {
        conv: ReflectPadConv1d,
    }

    impl TimeDelayNetBlock {
        fn new(
            in_channels: usize,
            out_channels: usize,
            kernel_size: usize,
            dilation: usize,
            vb: VarBuilder,
        ) -> Result<Self> {
            Ok(Self {
                conv: ReflectPadConv1d::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    vb.pp("conv"),
                )?,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            relu(&self.conv.forward(x)?)
        }
    }

    /// Res2Net 块: 将通道分成 `scale` 组，级联 TDNN 处理
    ///
    /// 第一组直接通过；后续每组输入 = 当前块 + 上一组输出。
    ///
    /// 权重键: `blocks.{i}.conv.weight/bias`
    struct Res2NetBlock {
        blocks: Vec<TimeDelayNetBlock>,
        scale: usize,
        chunk_size: usize,
    }

    impl Res2NetBlock {
        fn new(
            channels: usize,
            kernel_size: usize,
            dilation: usize,
            scale: usize,
            vb: VarBuilder,
        ) -> Result<Self> {
            let chunk_size = channels / scale;
            let mut blocks = Vec::with_capacity(scale - 1);
            for i in 0..(scale - 1) {
                blocks.push(TimeDelayNetBlock::new(
                    chunk_size,
                    chunk_size,
                    kernel_size,
                    dilation,
                    vb.pp(format!("blocks.{}", i)),
                )?);
            }
            Ok(Self {
                blocks,
                scale,
                chunk_size,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let mut outputs = Vec::with_capacity(self.scale);
            // 第一块直接通过
            outputs.push(x.narrow(1, 0, self.chunk_size)?);
            for (i, block) in self.blocks.iter().enumerate() {
                let chunk = x.narrow(1, (i + 1) * self.chunk_size, self.chunk_size)?;
                let input = if i == 0 {
                    chunk
                } else {
                    (chunk + outputs.last().unwrap())?
                };
                outputs.push(block.forward(&input)?);
            }
            Ok(Tensor::cat(&outputs, 1)?)
        }
    }

    /// Squeeze-Excitation 块: 通道注意力
    ///
    /// 全局平均池化 → Conv1d + ReLU → Conv1d + Sigmoid → 逐通道缩放
    ///
    /// 权重键: `conv1.weight/bias`, `conv2.weight/bias`
    struct SqueezeExcitationBlock {
        conv1: Conv1d,
        conv2: Conv1d,
    }

    impl SqueezeExcitationBlock {
        fn new(channels: usize, se_channels: usize, vb: VarBuilder) -> Result<Self> {
            let config = Conv1dConfig::default();
            Ok(Self {
                conv1: conv1d(channels, se_channels, 1, config, vb.pp("conv1"))?,
                conv2: conv1d(
                    se_channels,
                    channels,
                    1,
                    Conv1dConfig::default(),
                    vb.pp("conv2"),
                )?,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            // 全局平均池化: [B, C, T] → [B, C, 1]
            let s = x.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
            let s = relu(&self.conv1.forward(&s)?)?;
            let s = sigmoid(&self.conv2.forward(&s)?)?;
            Ok(x.broadcast_mul(&s)?)
        }
    }

    /// SE-Res2Net 块: TDNN1 → Res2Net → TDNN2 → SE → 残差
    ///
    /// 权重键: `tdnn1.*`, `res2net_block.*`, `tdnn2.*`, `se_block.*`
    struct SqueezeExcitationRes2NetBlock {
        tdnn1: TimeDelayNetBlock,
        res2net_block: Res2NetBlock,
        tdnn2: TimeDelayNetBlock,
        se_block: SqueezeExcitationBlock,
    }

    impl SqueezeExcitationRes2NetBlock {
        fn new(
            channels: usize,
            kernel_size: usize,
            dilation: usize,
            scale: usize,
            se_channels: usize,
            vb: VarBuilder,
        ) -> Result<Self> {
            Ok(Self {
                tdnn1: TimeDelayNetBlock::new(channels, channels, 1, 1, vb.pp("tdnn1"))?,
                res2net_block: Res2NetBlock::new(
                    channels,
                    kernel_size,
                    dilation,
                    scale,
                    vb.pp("res2net_block"),
                )?,
                tdnn2: TimeDelayNetBlock::new(channels, channels, 1, 1, vb.pp("tdnn2"))?,
                se_block: SqueezeExcitationBlock::new(channels, se_channels, vb.pp("se_block"))?,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let residual = x.clone();
            let out = self.tdnn1.forward(x)?;
            let out = self.res2net_block.forward(&out)?;
            let out = self.tdnn2.forward(&out)?;
            let out = self.se_block.forward(&out)?;
            Ok((out + residual)?)
        }
    }

    /// 注意力统计池化 (Attentive Statistics Pooling)
    ///
    /// 计算注意力加权的均值和标准差。
    ///
    /// 权重键: `tdnn.conv.weight/bias`, `conv.weight/bias`
    struct AttentiveStatisticsPooling {
        tdnn: TimeDelayNetBlock,
        conv: Conv1d,
    }

    impl AttentiveStatisticsPooling {
        fn new(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
            Ok(Self {
                tdnn: TimeDelayNetBlock::new(
                    channels * 3,
                    attention_channels,
                    1,
                    1,
                    vb.pp("tdnn"),
                )?,
                conv: conv1d(
                    attention_channels,
                    channels,
                    1,
                    Conv1dConfig::default(),
                    vb.pp("conv"),
                )?,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            // x: [B, C, T]
            let (b, c, t) = x.dims3()?;

            // 全局统计量
            let mean = x.mean(D::Minus1)?.unsqueeze(D::Minus1)?; // [B, C, 1]
            let diff = x.broadcast_sub(&mean)?;
            let var = diff.sqr()?.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
            let std = (var + 1e-5)?.sqrt()?; // [B, C, 1]

            let mean_exp = mean.broadcast_as((b, c, t))?;
            let std_exp = std.broadcast_as((b, c, t))?;

            // 拼接 [x, mean, std] → [B, 3C, T]
            let attn_in = Tensor::cat(&[x, &mean_exp, &std_exp], 1)?;

            // 注意力: TDNN(3C→attn_ch, 含ReLU) → Tanh → Conv(attn_ch→C) → Softmax
            let attn = self.tdnn.forward(&attn_in)?;
            let attn = attn.tanh()?;
            let attn = self.conv.forward(&attn)?;
            let attn = candle_nn::ops::softmax_last_dim(&attn)?; // softmax over T

            // 加权均值
            let w_mean = x
                .broadcast_mul(&attn)?
                .sum(D::Minus1)?
                .unsqueeze(D::Minus1)?; // [B, C, 1]

            // 加权标准差
            let w_diff = x.broadcast_sub(&w_mean)?;
            let w_var = w_diff
                .sqr()?
                .broadcast_mul(&attn)?
                .sum(D::Minus1)?
                .unsqueeze(D::Minus1)?;
            let w_std = (w_var + 1e-5)?.sqrt()?;

            // 输出: cat([w_mean, w_std]) → [B, 2C, 1]
            Ok(Tensor::cat(&[&w_mean, &w_std], 1)?)
        }
    }

    // ── ECAPA-TDNN 神经网络 ───────────────────────────────────────────

    /// 完整的 ECAPA-TDNN 神经网络
    ///
    /// 架构:
    /// ```text
    /// blocks[0]:   TDNN(mel_dim → ch[0], k=ks[0], d=dl[0])
    /// blocks[1-3]: SE-Res2Net(ch[i], k=ks[i], d=dl[i])
    /// MFA:         cat(SE-Res2Net outputs) → TDNN(→ ch[4])
    /// ASP:         Attentive statistics pooling → [B, 2*ch[4], 1]
    /// FC:          Conv1d(2*ch[4] → enc_dim, k=1) → [B, enc_dim]
    /// ```
    pub(super) struct EcapaTdnn {
        initial_tdnn: TimeDelayNetBlock,
        se_res2net_blocks: Vec<SqueezeExcitationRes2NetBlock>,
        mfa_tdnn: TimeDelayNetBlock,
        asp: AttentiveStatisticsPooling,
        fc: Conv1d,
        #[allow(dead_code)]
        device: Device,
    }

    impl EcapaTdnn {
        /// 从 VarBuilder 构建 ECAPA-TDNN
        ///
        /// `vb` 应已 scope 到 `speaker_encoder` 前缀。
        pub(super) fn new(
            config: &crate::model_config::SpeakerEncoderConfig,
            vb: VarBuilder,
        ) -> Result<Self> {
            let device = vb.device().clone();

            // 验证配置完整性
            if config.enc_channels.len() < 5
                || config.enc_kernel_sizes.len() < 5
                || config.enc_dilations.len() < 5
            {
                anyhow::bail!(
                    "SpeakerEncoderConfig too short: channels={}, kernel_sizes={}, dilations={}",
                    config.enc_channels.len(),
                    config.enc_kernel_sizes.len(),
                    config.enc_dilations.len()
                );
            }

            // blocks[0]: 初始 TDNN (mel_dim → enc_channels[0])
            let initial_tdnn = TimeDelayNetBlock::new(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
                vb.pp("blocks.0"),
            )?;

            // blocks[1-3]: SE-Res2Net 块
            let mut se_res2net_blocks = Vec::with_capacity(3);
            for i in 1..4 {
                se_res2net_blocks.push(SqueezeExcitationRes2NetBlock::new(
                    config.enc_channels[i],
                    config.enc_kernel_sizes[i],
                    config.enc_dilations[i],
                    config.enc_res2net_scale,
                    config.enc_se_channels,
                    vb.pp(format!("blocks.{}", i)),
                )?);
            }

            // MFA: 拼接 SE-Res2Net 输出 → TDNN 投影
            let mfa_in_channels: usize = config.enc_channels[1..4].iter().sum();
            let mfa_tdnn = TimeDelayNetBlock::new(
                mfa_in_channels,
                config.enc_channels[4],
                config.enc_kernel_sizes[4],
                config.enc_dilations[4],
                vb.pp("mfa"),
            )?;

            // ASP: 注意力统计池化
            let asp = AttentiveStatisticsPooling::new(
                config.enc_channels[4],
                config.enc_attention_channels,
                vb.pp("asp"),
            )?;

            // FC: Conv1d(2 * enc_channels[4] → enc_dim, k=1)
            let fc = conv1d(
                config.enc_channels[4] * 2,
                config.enc_dim,
                1,
                Conv1dConfig::default(),
                vb.pp("fc"),
            )?;

            tracing::info!(
                "ECAPA-TDNN constructed: mel_dim={}, enc_dim={}, channels={:?}, scale={}, se_ch={}",
                config.mel_dim,
                config.enc_dim,
                config.enc_channels,
                config.enc_res2net_scale,
                config.enc_se_channels
            );

            Ok(Self {
                initial_tdnn,
                se_res2net_blocks,
                mfa_tdnn,
                asp,
                fc,
                device,
            })
        }

        /// 前向推理
        ///
        /// 输入: mel 频谱图 [B, n_mels, T]
        /// 输出: 嵌入 [B, enc_dim] (未归一化，norm ≈ 10)
        pub(super) fn forward(&self, mel: &Tensor) -> Result<Tensor> {
            // blocks[0]: 初始 TDNN
            let x = self.initial_tdnn.forward(mel)?;

            // blocks[1-3]: SE-Res2Net，收集输出用于 MFA
            let mut se_outputs = Vec::with_capacity(3);
            let mut h = x;
            for block in &self.se_res2net_blocks {
                h = block.forward(&h)?;
                se_outputs.push(h.clone());
            }

            // MFA: 沿通道维度拼接 SE-Res2Net 输出
            let mfa_input = Tensor::cat(&se_outputs, 1)?;
            let h = self.mfa_tdnn.forward(&mfa_input)?;

            // ASP: 注意力统计池化 → [B, 2C, 1]
            let pooled = self.asp.forward(&h)?;

            // FC: 投影到嵌入维度 → [B, enc_dim, 1]
            let embed = self.fc.forward(&pooled)?;
            let embed = embed.squeeze(D::Minus1)?; // [B, enc_dim]

            // 返回原始嵌入（不 L2 归一化 — 模型训练时使用未归一化的嵌入，norm ≈ 10）
            Ok(embed)
        }
    }
} // end mod ecapa

// ════════════════════════════════════════════════════════════════════════
//  说话人编码器 (公共 API)
// ════════════════════════════════════════════════════════════════════════

/// 说话人编码器
///
/// 从参考音频中提取说话人特征嵌入向量。
///
/// 当模型权重可用时，使用完整的 ECAPA-TDNN 神经网络前向推理。
/// 当权重不可用时，降级为统计嵌入方案。
pub struct SpeakerEncoder {
    /// 嵌入维度
    embed_dim: usize,
    /// Mel 配置
    mel_config: MelConfig,
    /// ECAPA-TDNN 神经网络 (权重已加载时)
    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    ecapa_tdnn: Option<ecapa::EcapaTdnn>,
    /// 是否已加载模型权重
    weights_loaded: bool,
    /// 配置
    #[allow(dead_code)]
    config: SpeakerEncoderConfig,
    /// 设备
    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    device: candle_core::Device,
}

impl SpeakerEncoder {
    /// 创建新的说话人编码器 (无权重，使用统计嵌入)
    pub fn new(embed_dim: usize) -> Self {
        let config = SpeakerEncoderConfig::default();
        let mel_config = MelConfig {
            sample_rate: config.sample_rate,
            n_fft: 512,
            hop_length: config.sample_rate as usize / 100, // 10ms hop
            win_length: 400,
            n_mels: config.mel_dim,
            fmin: 0.0,
            fmax: config.sample_rate as f64 / 2.0,
        };
        Self {
            embed_dim,
            mel_config,
            #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
            ecapa_tdnn: None,
            weights_loaded: false,
            config,
            #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
            device: candle_core::Device::Cpu,
        }
    }

    /// 从 safetensors 权重构建说话人编码器
    ///
    /// 扫描权重中的 `speaker_encoder.` 前缀张量。
    /// 如果找到权重，构建完整的 ECAPA-TDNN 神经网络；否则返回统计嵌入编码器。
    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    pub fn from_weights(
        weights: &HashMap<String, candle_core::Tensor>,
        config: SpeakerEncoderConfig,
        device: &candle_core::Device,
    ) -> Result<Self> {
        // 检查是否有 speaker_encoder 权重
        let se_keys: Vec<&String> = weights
            .keys()
            .filter(|k| k.starts_with("speaker_encoder."))
            .collect();

        let mel_config = MelConfig {
            sample_rate: config.sample_rate,
            n_fft: 512,
            hop_length: config.sample_rate as usize / 100, // 10ms hop
            win_length: 400,
            n_mels: config.mel_dim,
            fmin: 0.0,
            fmax: config.sample_rate as f64 / 2.0,
        };

        if se_keys.is_empty() {
            tracing::info!(
                "SpeakerEncoder: no speaker_encoder weights found, using statistical embedding"
            );
            return Ok(Self {
                embed_dim: config.enc_dim,
                mel_config,
                ecapa_tdnn: None,
                weights_loaded: false,
                config,
                device: device.clone(),
            });
        }

        tracing::info!(
            "SpeakerEncoder: found {} speaker encoder weight tensors, building ECAPA-TDNN",
            se_keys.len()
        );
        for key in se_keys.iter().take(10) {
            tracing::debug!("  speaker weight: {}", key);
        }

        // 构建 VarBuilder 并尝试创建 ECAPA-TDNN
        let dtype = candle_core::DType::F32;
        let vb = candle_nn::VarBuilder::from_tensors(weights.clone(), dtype, device);
        let se_vb = vb.pp("speaker_encoder");

        let ecapa_tdnn = match ecapa::EcapaTdnn::new(&config, se_vb) {
            Ok(net) => {
                tracing::info!("ECAPA-TDNN neural network successfully loaded from weights");
                Some(net)
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to build ECAPA-TDNN from weights: {}. Falling back to statistical embedding.",
                    e
                );
                None
            }
        };

        let weights_loaded = ecapa_tdnn.is_some();

        Ok(Self {
            embed_dim: config.enc_dim,
            mel_config,
            ecapa_tdnn,
            weights_loaded,
            config,
            device: device.clone(),
        })
    }

    #[cfg(not(any(feature = "cpu", feature = "metal", feature = "cuda")))]
    pub fn from_weights(
        _weights: &HashMap<String, candle_core::Tensor>,
        config: SpeakerEncoderConfig,
        _device: &candle_core::Device,
    ) -> Result<Self> {
        Ok(Self {
            embed_dim: config.enc_dim,
            mel_config: MelConfig::speaker_encoder(),
            weights_loaded: false,
            config,
        })
    }

    /// 从参考音频提取说话人嵌入
    ///
    /// # 流程
    /// 1. 加载参考音频 WAV
    /// 2. 重采样到配置采样率
    /// 3. 提取 mel-频谱图
    /// 4. 通过 ECAPA-TDNN 前向推理 (或统计嵌入降级方案)
    /// 5. 返回嵌入向量
    pub fn extract_embedding(&self, ref_audio_path: &Path) -> Result<Vec<f32>> {
        // 1. 加载音频
        let audio = AudioBuffer::load_wav(ref_audio_path)?;

        // 2. 重采样
        let audio = if audio.sample_rate != self.mel_config.sample_rate {
            audio.resample_linear(self.mel_config.sample_rate)
        } else {
            audio
        };

        // 3. 提取 mel-频谱图
        let mel_spec = extract_mel_spectrogram(&audio.samples, &self.mel_config);

        let n_mels = mel_spec.len();
        let n_frames = mel_spec.first().map_or(0, |r| r.len());

        tracing::info!(
            "Speaker encoder: mel spectrogram {}x{} from {:.1}s audio (weights_loaded={})",
            n_mels,
            n_frames,
            audio.duration_secs(),
            self.weights_loaded
        );

        if n_mels == 0 || n_frames == 0 {
            tracing::warn!("Empty mel spectrogram, returning zero embedding");
            return Ok(vec![0.0f32; self.embed_dim]);
        }

        // 4. 前向推理
        #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
        if let Some(ref net) = self.ecapa_tdnn {
            return self.extract_neural_embedding(&mel_spec, net);
        }

        // 降级: 统计嵌入
        #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
        {
            if self.weights_loaded {
                tracing::debug!(
                    "Using enhanced statistical embedding (weights loaded but NN unavailable)"
                );
                Ok(self.compute_enhanced_embedding(&mel_spec))
            } else {
                tracing::debug!("Using statistical embedding (no weights loaded)");
                Ok(self.compute_statistical_embedding(&mel_spec))
            }
        }
        #[cfg(not(any(feature = "cpu", feature = "metal", feature = "cuda")))]
        {
            Ok(self.compute_statistical_embedding(&mel_spec))
        }
    }

    /// 使用 ECAPA-TDNN 神经网络提取嵌入
    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    fn extract_neural_embedding(
        &self,
        mel_spec: &[Vec<f32>],
        net: &ecapa::EcapaTdnn,
    ) -> Result<Vec<f32>> {
        let n_mels = mel_spec.len();
        let n_frames = mel_spec.first().map_or(0, |r| r.len());

        // 将 mel 频谱图转换为 Candle Tensor [1, n_mels, T]
        let flat: Vec<f32> = mel_spec.iter().flat_map(|r| r.iter().copied()).collect();
        let mel_tensor = candle_core::Tensor::from_vec(flat, (1, n_mels, n_frames), &self.device)
            .context("Failed to create mel tensor")?;

        // 前向推理 → [1, enc_dim]
        let embed = net
            .forward(&mel_tensor)
            .context("ECAPA-TDNN forward failed")?;

        // 转换为 Vec<f32>
        let embed_vec = embed
            .squeeze(0)
            .context("Failed to squeeze batch dim")?
            .to_vec1::<f32>()
            .context("Failed to convert embedding to vec")?;

        tracing::info!(
            "ECAPA-TDNN: extracted {}-dim embedding (norm={:.2})",
            embed_vec.len(),
            embed_vec.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        Ok(embed_vec)
    }

    /// 计算统计嵌入（基本降级方案）
    fn compute_statistical_embedding(&self, mel_spec: &[Vec<f32>]) -> Vec<f32> {
        let n_mels = mel_spec.len();
        let mut embedding = vec![0.0f32; self.embed_dim];

        let total_elements = n_mels * mel_spec.first().map_or(1, |r| r.len()).max(1);
        let global_mean: f32 =
            mel_spec.iter().flat_map(|row| row.iter()).sum::<f32>() / total_elements as f32;

        for (i, row) in mel_spec.iter().enumerate() {
            if i >= self.embed_dim / 2 {
                break;
            }
            let mean = row.iter().sum::<f32>() / row.len().max(1) as f32;
            let std = (row.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                / row.len().max(1) as f32)
                .sqrt();

            embedding[i] = mean;
            if i + n_mels < self.embed_dim {
                embedding[i + n_mels] = std;
            }
        }

        // 归一化
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        // 防止 unused variable 警告
        let _ = global_mean;

        embedding
    }

    /// 计算增强嵌入（当权重已加载但 NN 推理失败时使用的降级方案）
    ///
    /// 结合多种声学特征：
    /// - mel 频带均值和标准差
    /// - 频谱质心
    /// - 频谱通量
    fn compute_enhanced_embedding(&self, mel_spec: &[Vec<f32>]) -> Vec<f32> {
        let n_mels = mel_spec.len();
        let n_frames = mel_spec.first().map_or(0, |r| r.len());

        if n_mels == 0 || n_frames == 0 {
            return vec![0.0f32; self.embed_dim];
        }

        let mut embedding = vec![0.0f32; self.embed_dim];

        // 1. 每频带均值和标准差
        let half = self.embed_dim / 2;
        let mel_step = (n_mels as f64 / half as f64).ceil() as usize;

        for i in 0..half {
            let mel_idx = (i * mel_step).min(n_mels - 1);
            let row = &mel_spec[mel_idx];

            let mean = row.iter().sum::<f32>() / n_frames as f32;
            let std =
                (row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n_frames as f32).sqrt();

            embedding[i] = mean;
            if i + half < self.embed_dim {
                embedding[i + half] = std;
            }
        }

        // 2. 频谱质心 (全局)
        let total_energy: f32 = mel_spec
            .iter()
            .flat_map(|r| r.iter())
            .sum::<f32>()
            .max(1e-10);
        let mut centroid_sum = 0.0f32;
        for (mel_idx, row) in mel_spec.iter().enumerate() {
            let band_energy: f32 = row.iter().sum();
            centroid_sum += mel_idx as f32 * band_energy;
        }
        let centroid = centroid_sum / total_energy;

        if self.embed_dim > 10 {
            embedding[self.embed_dim - 1] = centroid / n_mels as f32;
        }

        // 3. 频谱通量 (帧间变化率)
        let mut total_flux = 0.0f32;
        for t in 1..n_frames {
            for row in mel_spec {
                let prev = row.get(t - 1).copied().unwrap_or(0.0);
                let curr = row.get(t).copied().unwrap_or(0.0);
                total_flux += (curr - prev).abs();
            }
        }
        let avg_flux = total_flux / (n_frames * n_mels) as f32;

        if self.embed_dim > 11 {
            embedding[self.embed_dim - 2] = avg_flux;
        }

        // 归一化到单位长度
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    /// 嵌入维度
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// 是否已加载权重
    pub fn weights_loaded(&self) -> bool {
        self.weights_loaded
    }
}

impl std::fmt::Debug for SpeakerEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeakerEncoder")
            .field("embed_dim", &self.embed_dim)
            .field("mel_config", &self.mel_config)
            .field("weights_loaded", &self.weights_loaded)
            .finish()
    }
}
