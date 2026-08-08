//! CFM（条件流匹配）+ BigVGAN 解码器架构设计
//!
//! 借鉴 GPT-SoVITS v3 的 CFM + DiT + BigVGAN 架构，
//! 作为当前 ConvNeXt 解码器的长期演进方向。
//!
//! # 当前架构 vs 目标架构
//!
//! ## 当前 (v2): ConvNeXt + ConvTranspose1d
//! - T2S 模型生成语义 token → 解码器直接上采样为波形
//! - 1920x 上采样 (12Hz → 24kHz)
//! - ConvNeXtBlock + DecoderUpsampleBlock (4 层)
//! - 优点：快速 (RTF ~0.9x on CPU)
//! - 缺点：音频质量有上限，缺乏高频细节
//!
//! ## 目标 (v3): CFM + DiT + BigVGAN
//! - T2S 模型生成语义 token → CFM 迭代去噪 mel → BigVGAN 声码器
//! - CFM: 22 层 Diffusion Transformer, 32 步去噪
//! - BigVGAN: 大规模 HiFi-GAN 变体， SnakeBeta 激活函数
//! - 优点：SOTA 音质，更自然的语音
//! - 缺点：推理速度慢 (32 步迭代)，需要重新训练/加载权重
//!
//! # CFM 算法原理
//!
//! 条件流匹配 (Conditional Flow Matching) 通过学习向量场来匹配
//! 从噪声分布到数据分布的概率路径。
//!
//! ```python
//! # CFM 迭代去噪 (32 步)
//! x = torch.randn_like(mel_target)  # 初始噪声
//! dt = 1.0 / n_timesteps
//! for j in range(n_timesteps):
//!     v_pred = estimator(x, t, mu)  # DiT 预测速度场
//!     x = x + dt * v_pred           # Euler 步进
//! mel_output = x  # 去噪后的 mel
//! ```
//!
//! - `x`: 当前 mel 状态 (从噪声开始)
//! - `t`: 时间步 (0→1)
//! - `mu`: 条件 (语义 token 的 embedding)
//! - `v_pred`: DiT 预测的速度向量场
//!
//! # 分块 CFM 推理 + SOLA 拼接
//!
//! 对长文本生成的 mel 频谱分块处理，每块用 CFM 推理后通过 SOLA 拼接：
//! - 将长 mel 分为多个 chunk
//! - 每个 chunk 以参考音频的 mel 开头 (条件引导)
//! - CFM 推理后，更新参考为当前 chunk 的末尾
//! - 通过 SOLA (已实现 in `sola.rs`) 拼接各 chunk
//!
//! # 模块结构
//! - [`CfmConfig`]: CFM 推理配置
//! - [`CfmStep`]: 单步 CFM 去噪
//! - [`BigVganConfig`]: BigVGAN 声码器配置
//! - [`DecoderArchitecture`]: 解码器架构描述
//! - [`ChunkedCfmInference`]: 分块 CFM 推理策略
//! - [`ArchitectureComparison`]: 架构对比分析

// ─── 常量 ─────────────────────────────────────────────────

/// CFM 默认去噪步数
pub const DEFAULT_CFM_STEPS: usize = 32;

/// DiT (Diffusion Transformer) 默认层数
pub const DEFAULT_DIT_LAYERS: usize = 22;

/// DiT 默认隐藏维度
pub const DEFAULT_DIT_DIM: usize = 768;

/// BigVGAN 默认上采样倍数
pub const DEFAULT_BIGVGAN_UPSAMPLE: usize = 8;

/// BigVGAN 默认 SnakeBeta 通道数
pub const DEFAULT_BIGVGAN_CHANNELS: usize = 512;

/// 分块推理默认 chunk 长度（mel 帧）
pub const DEFAULT_CHUNK_LEN_FRAMES: usize = 100;

/// 分块推理默认参考重叠长度（mel 帧）
pub const DEFAULT_REF_OVERLAP_FRAMES: usize = 20;

/// 当前 ConvNeXt 解码器的上采样倍数
pub const CURRENT_CONVNET_UPSAMPLE: usize = 1920;

// ─── CFM 配置 ────────────────────────────────────────────

/// CFM 推理配置
///
/// 控制条件流匹配的去噪过程。
#[derive(Debug, Clone)]
pub struct CfmConfig {
    /// 去噪步数 (默认 32, 更多步 = 更高质量但更慢)
    pub n_timesteps: usize,
    /// DiT 层数 (默认 22)
    pub dit_layers: usize,
    /// DiT 隐藏维度 (默认 768)
    pub dit_dim: usize,
    /// 时间步采样策略
    pub time_sampling: TimeSampling,
    /// 是否使用 Euler 步进 (true) 还是 Heun 二阶 (false)
    pub use_euler: bool,
}

/// 时间步采样策略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeSampling {
    /// 均匀采样: dt = 1/n_timesteps
    Uniform,
    /// 多项式采样: 集中在噪声较多的区域
    Polynomial(f32),
}

impl Default for CfmConfig {
    fn default() -> Self {
        Self {
            n_timesteps: DEFAULT_CFM_STEPS,
            dit_layers: DEFAULT_DIT_LAYERS,
            dit_dim: DEFAULT_DIT_DIM,
            time_sampling: TimeSampling::Uniform,
            use_euler: true,
        }
    }
}

impl CfmConfig {
    /// 计算步长 dt
    #[must_use]
    pub fn dt(&self) -> f32 {
        1.0 / self.n_timesteps as f32
    }

    /// 生成时间步序列
    #[must_use]
    pub fn timesteps(&self) -> Vec<f32> {
        match self.time_sampling {
            TimeSampling::Uniform => {
                let dt = self.dt();
                (0..self.n_timesteps).map(|i| i as f32 * dt).collect()
            }
            TimeSampling::Polynomial(power) => {
                let n = self.n_timesteps as f32;
                (0..self.n_timesteps)
                    .map(|i| {
                        let t = i as f32 / n;
                        t.powf(power)
                    })
                    .collect()
            }
        }
    }

    /// 估算 CFM 推理时间（秒）
    ///
    /// 基于 DiT 前向传播时间 × 步数
    #[must_use]
    pub fn estimate_inference_time(&self, mel_frames: usize, forward_time_ms: f32) -> f32 {
        let total_ms = forward_time_ms * self.n_timesteps as f32;
        // mel 帧数影响每步的计算量
        let scale = mel_frames as f32 / 100.0; // 以 100 帧为基准
        total_ms * scale / 1000.0
    }
}

// ─── CFM 单步 ────────────────────────────────────────────

/// 单步 CFM 去噪
///
/// 对应 Python:
/// ```python
/// v_pred = estimator(x, t, mu)
/// x = x + dt * v_pred
/// ```
#[derive(Debug, Clone)]
pub struct CfmStep {
    /// 当前时间步
    pub timestep: f32,
    /// 步长
    pub dt: f32,
    /// 预测的速度场 (v_pred) 的维度
    pub velocity_dim: usize,
}

impl CfmStep {
    /// 创建单步
    #[must_use]
    pub fn new(timestep: f32, dt: f32, velocity_dim: usize) -> Self {
        Self {
            timestep,
            dt,
            velocity_dim,
        }
    }

    /// Euler 步进: x = x + dt * v
    ///
    /// # 参数
    /// - `x`: 当前 mel 状态 [mel_bins, frames]
    /// - `v`: 预测的速度场
    #[must_use]
    pub fn euler_step(&self, x: &[f32], v: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), v.len(), "x and v must have same length");
        x.iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| xi + self.dt * vi)
            .collect()
    }

    /// Heun 二阶步进 (更精确但 2x 计算量)
    ///
    /// 需要两次速度场评估:
    /// 1. v1 = estimator(x, t)
    /// 2. v2 = estimator(x + dt*v1, t + dt)
    /// 3. x = x + dt * (v1 + v2) / 2
    #[must_use]
    pub fn heun_step(&self, x: &[f32], v1: &[f32], v2: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), v1.len());
        assert_eq!(x.len(), v2.len());
        x.iter()
            .zip(v1.iter())
            .zip(v2.iter())
            .map(|((&xi, &v1i), &v2i)| xi + self.dt * (v1i + v2i) * 0.5)
            .collect()
    }
}

// ─── BigVGAN 配置 ────────────────────────────────────────

/// BigVGAN 声码器配置
///
/// BigVGAN 是大规模 HiFi-GAN 变体，使用 SnakeBeta 周期性激活函数。
/// 输入: mel 频谱 → 输出: 波形
#[derive(Debug, Clone)]
pub struct BigVganConfig {
    /// 上采样倍数 (mel → waveform 的总上采样)
    pub upsample_total: usize,
    /// BigVGAN 通道数
    pub channels: usize,
    /// 上采样块数
    pub upsample_blocks: usize,
    /// 每个上采样块的上采样因子
    pub upsample_factors: Vec<usize>,
    /// SnakeBeta 通道数
    pub snake_channels: usize,
    /// 是否使用 anti-aliasing 滤波
    pub use_anti_alias: bool,
}

impl Default for BigVganConfig {
    fn default() -> Self {
        Self {
            upsample_total: DEFAULT_BIGVGAN_UPSAMPLE,
            channels: DEFAULT_BIGVGAN_CHANNELS,
            upsample_blocks: 4,
            upsample_factors: vec![8, 8, 2, 2], // 8*8*2*2 = 256x
            snake_channels: 80,
            use_anti_alias: true,
        }
    }
}

impl BigVganConfig {
    /// 计算实际上采样倍数
    #[must_use]
    pub fn actual_upsample(&self) -> usize {
        self.upsample_factors.iter().product()
    }

    /// 估算 BigVGAN 推理时间（秒）
    ///
    /// BigVGAN 推理时间主要取决于 mel 帧数和上采样倍数
    #[must_use]
    pub fn estimate_inference_time(&self, mel_frames: usize) -> f32 {
        // 基准: 100 mel 帧 → ~50ms (CPU)
        let base_ms = 50.0f32;
        let scale = mel_frames as f32 / 100.0;
        base_ms * scale / 1000.0
    }
}

// ─── 解码器架构描述 ──────────────────────────────────────

/// 解码器架构类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderType {
    /// 当前架构: ConvNeXt + ConvTranspose1d
    ConvNetV2,
    /// 目标架构: CFM + DiT + BigVGAN
    CfmBigVganV3,
    /// 混合架构: ConvNeXt (快速模式) + CFM (高质量模式)
    Hybrid,
}

/// 解码器架构描述
///
/// 描述解码器的完整配置和性能特征。
#[derive(Debug, Clone)]
pub struct DecoderArchitecture {
    /// 架构类型
    pub decoder_type: DecoderType,
    /// CFM 配置 (仅 CFM/混合模式)
    pub cfm_config: Option<CfmConfig>,
    /// BigVGAN 配置 (仅 CFM/混合模式)
    pub bigvgan_config: Option<BigVganConfig>,
    /// 上采样倍数
    pub upsample_ratio: usize,
    /// 输入采样率 (Hz, 语义 token 的等效帧率)
    pub input_sample_rate: u32,
    /// 输出采样率 (Hz)
    pub output_sample_rate: u32,
}

impl DecoderArchitecture {
    /// 创建当前 ConvNeXt v2 架构
    #[must_use]
    pub fn convnet_v2() -> Self {
        Self {
            decoder_type: DecoderType::ConvNetV2,
            cfm_config: None,
            bigvgan_config: None,
            upsample_ratio: CURRENT_CONVNET_UPSAMPLE,
            input_sample_rate: 12,
            output_sample_rate: 24000,
        }
    }

    /// 创建 CFM + BigVGAN v3 架构
    #[must_use]
    pub fn cfm_bigvgan_v3() -> Self {
        let cfm = CfmConfig::default();
        let bigvgan = BigVganConfig::default();
        Self {
            decoder_type: DecoderType::CfmBigVganV3,
            upsample_ratio: bigvgan.actual_upsample(),
            input_sample_rate: 12,
            output_sample_rate: 24000,
            cfm_config: Some(cfm),
            bigvgan_config: Some(bigvgan),
        }
    }

    /// 创建混合架构
    #[must_use]
    pub fn hybrid() -> Self {
        Self {
            decoder_type: DecoderType::Hybrid,
            cfm_config: Some(CfmConfig {
                n_timesteps: 8, // 快速模式: 减少步数
                ..Default::default()
            }),
            bigvgan_config: Some(BigVganConfig::default()),
            upsample_ratio: DEFAULT_BIGVGAN_UPSAMPLE,
            input_sample_rate: 12,
            output_sample_rate: 24000,
        }
    }

    /// 估算总推理时间（秒）
    ///
    /// # 参数
    /// - `mel_frames`: mel 帧数
    /// - `dit_forward_ms`: DiT 单步前向传播时间 (ms)
    #[must_use]
    pub fn estimate_total_time(&self, mel_frames: usize, dit_forward_ms: f32) -> f32 {
        match self.decoder_type {
            DecoderType::ConvNetV2 => {
                // 当前 ConvNeXt: ~1.3s for 5.5s audio (68 tokens)
                let scale = mel_frames as f32 / 100.0;
                1.3 * scale
            }
            DecoderType::CfmBigVganV3 => {
                let cfm_time = self
                    .cfm_config
                    .as_ref()
                    .map(|c| c.estimate_inference_time(mel_frames, dit_forward_ms))
                    .unwrap_or(0.0);
                let bigvgan_time = self
                    .bigvgan_config
                    .as_ref()
                    .map(|b| b.estimate_inference_time(mel_frames))
                    .unwrap_or(0.0);
                cfm_time + bigvgan_time
            }
            DecoderType::Hybrid => {
                // 混合模式: 使用快速 CFM (8 步)
                let cfm_time = self
                    .cfm_config
                    .as_ref()
                    .map(|c| c.estimate_inference_time(mel_frames, dit_forward_ms))
                    .unwrap_or(0.0);
                let bigvgan_time = self
                    .bigvgan_config
                    .as_ref()
                    .map(|b| b.estimate_inference_time(mel_frames))
                    .unwrap_or(0.0);
                cfm_time + bigvgan_time
            }
        }
    }

    /// 估算 RTF (Real-Time Factor)
    ///
    /// RTF = 推理时间 / 音频时长
    #[must_use]
    pub fn estimate_rtf(&self, mel_frames: usize, dit_forward_ms: f32) -> f32 {
        let inference_time = self.estimate_total_time(mel_frames, dit_forward_ms);
        let audio_duration = mel_frames as f32 / 86.0; // ~86 mel frames/sec at 24kHz
        if audio_duration > 0.0 {
            inference_time / audio_duration
        } else {
            0.0
        }
    }
}

// ─── 架构对比分析 ────────────────────────────────────────

/// 架构对比分析
///
/// 对比当前 ConvNeXt v2 和目标 CFM + BigVGAN v3 架构。
#[derive(Debug, Clone)]
pub struct ArchitectureComparison {
    /// v2 架构
    pub v2: DecoderArchitecture,
    /// v3 架构
    pub v3: DecoderArchitecture,
}

impl Default for ArchitectureComparison {
    fn default() -> Self {
        Self {
            v2: DecoderArchitecture::convnet_v2(),
            v3: DecoderArchitecture::cfm_bigvgan_v3(),
        }
    }
}

impl ArchitectureComparison {
    /// 创建对比分析
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 生成对比报告
    ///
    /// # 参数
    /// - `mel_frames`: 测试用的 mel 帧数
    /// - `dit_forward_ms`: DiT 单步前向传播时间 (ms)
    #[must_use]
    pub fn report(&self, mel_frames: usize, dit_forward_ms: f32) -> ComparisonReport {
        let v2_time = self.v2.estimate_total_time(mel_frames, dit_forward_ms);
        let v3_time = self.v3.estimate_total_time(mel_frames, dit_forward_ms);
        let v2_rtf = self.v2.estimate_rtf(mel_frames, dit_forward_ms);
        let v3_rtf = self.v3.estimate_rtf(mel_frames, dit_forward_ms);

        ComparisonReport {
            v2_inference_secs: v2_time,
            v3_inference_secs: v3_time,
            v2_rtf,
            v3_rtf,
            speedup_factor: v3_time / v2_time.max(1e-6),
            quality_gain: QualityGain::Significant,
            migration_cost: MigrationCost::High,
        }
    }
}

/// 对比报告
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// v2 推理时间（秒）
    pub v2_inference_secs: f32,
    /// v3 推理时间（秒）
    pub v3_inference_secs: f32,
    /// v2 RTF
    pub v2_rtf: f32,
    /// v3 RTF
    pub v3_rtf: f32,
    /// v3/v2 速度比（>1 = v3 更慢）
    pub speedup_factor: f32,
    /// 音质提升预期
    pub quality_gain: QualityGain,
    /// 迁移成本
    pub migration_cost: MigrationCost,
}

/// 音质提升预期
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGain {
    /// 轻微提升
    Marginal,
    /// 中等提升
    Moderate,
    /// 显著提升
    Significant,
}

/// 迁移成本
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationCost {
    /// 低成本 (仅代码修改)
    Low,
    /// 中等成本 (需加载新权重)
    Medium,
    /// 高成本 (需重新训练或大规模重构)
    High,
}

// ─── 分块 CFM 推理 ───────────────────────────────────────

/// 分块 CFM 推理策略
///
/// 对长文本生成的 mel 频谱分块处理，
/// 每块用 CFM 推理后通过 SOLA 拼接。
///
/// # 工作流程
/// 1. 将长 mel 分为多个 chunk (长度 = chunk_len)
/// 2. 每个 chunk 以参考音频的 mel 开头 (条件引导)
/// 3. CFM 推理 (32 步去噪)
/// 4. 更新参考为当前 chunk 的末尾 mel
/// 5. 通过 SOLA (sola.rs) 拼接各 chunk 的输出
///
/// # 优势
/// - 不需要一次性处理整段音频 (内存友好)
/// - 每块以参考音频开头，保持声音一致性
/// - SOLA 拼接消除拼接痕迹
#[derive(Debug, Clone)]
pub struct ChunkedCfmInference {
    /// CFM 配置
    pub cfm_config: CfmConfig,
    /// chunk 长度 (mel 帧)
    pub chunk_len: usize,
    /// 参考重叠长度 (mel 帧)
    pub ref_overlap: usize,
}

impl Default for ChunkedCfmInference {
    fn default() -> Self {
        Self {
            cfm_config: CfmConfig::default(),
            chunk_len: DEFAULT_CHUNK_LEN_FRAMES,
            ref_overlap: DEFAULT_REF_OVERLAP_FRAMES,
        }
    }
}

impl ChunkedCfmInference {
    /// 创建分块推理策略
    #[must_use]
    pub fn new(chunk_len: usize, ref_overlap: usize) -> Self {
        Self {
            cfm_config: CfmConfig::default(),
            chunk_len: chunk_len.max(10),
            ref_overlap: ref_overlap.min(chunk_len / 2),
        }
    }

    /// 计算分块方案
    ///
    /// # 参数
    /// - `total_mel_frames`: 总 mel 帧数
    ///
    /// # 返回
    /// 每个 chunk 的 (起始帧, 结束帧) 列表
    #[must_use]
    pub fn chunk_plan(&self, total_mel_frames: usize) -> Vec<(usize, usize)> {
        if total_mel_frames <= self.chunk_len {
            return vec![(0, total_mel_frames)];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < total_mel_frames {
            let end = (start + self.chunk_len).min(total_mel_frames);
            chunks.push((start, end));
            start = end;
        }

        chunks
    }

    /// 计算分块数量
    #[must_use]
    pub fn num_chunks(&self, total_mel_frames: usize) -> usize {
        self.chunk_plan(total_mel_frames).len()
    }

    /// 估算分块推理总时间
    ///
    /// 每个独立 chunk 的 CFM 推理时间 × chunk 数量
    #[must_use]
    pub fn estimate_total_time(&self, total_mel_frames: usize, dit_forward_ms: f32) -> f32 {
        let chunks = self.chunk_plan(total_mel_frames);
        chunks
            .iter()
            .map(|&(start, end)| {
                let chunk_frames = end - start + self.ref_overlap;
                self.cfm_config
                    .estimate_inference_time(chunk_frames, dit_forward_ms)
            })
            .sum()
    }

    /// 获取 chunk 的参考 mel 范围
    ///
    /// 每个 chunk 以参考音频的 mel 开头
    #[must_use]
    pub fn reference_range(&self, _chunk_idx: usize, prev_chunk_end: usize) -> (usize, usize) {
        let ref_start = prev_chunk_end.saturating_sub(self.ref_overlap);
        let ref_end = prev_chunk_end;
        (ref_start, ref_end)
    }
}

// ─── DiT (Diffusion Transformer) 接口定义 ───────────────

/// DiT 层配置
///
/// 对应 CFM 中的条件估计器 (estimator)
#[derive(Debug, Clone)]
pub struct DitLayerConfig {
    /// 隐藏维度
    pub hidden_dim: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// MLP 中间维度
    pub mlp_dim: usize,
    /// 是否使用 AdaLN (Adaptive Layer Norm)
    pub use_adaln: bool,
    /// 时间步嵌入维度
    pub time_embed_dim: usize,
}

impl Default for DitLayerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: DEFAULT_DIT_DIM,
            num_heads: 12,
            mlp_dim: DEFAULT_DIT_DIM * 4,
            use_adaln: true,
            time_embed_dim: 256,
        }
    }
}

/// DiT 模型配置
#[derive(Debug, Clone)]
pub struct DitModelConfig {
    /// 层数
    pub num_layers: usize,
    /// 层配置
    pub layer_config: DitLayerConfig,
    /// 输入维度 (mel bins)
    pub input_dim: usize,
    /// 条件维度 (语义 token embedding)
    pub condition_dim: usize,
    /// patch 大小 (将 mel 分为 patch 处理)
    pub patch_size: usize,
}

impl Default for DitModelConfig {
    fn default() -> Self {
        Self {
            num_layers: DEFAULT_DIT_LAYERS,
            layer_config: DitLayerConfig::default(),
            input_dim: 80,      // mel bins
            condition_dim: 768, // 语义 token embedding
            patch_size: 1,
        }
    }
}

/// 估算 DiT 参数量
#[must_use]
pub fn estimate_dit_params(config: &DitModelConfig) -> usize {
    let lc = &config.layer_config;
    let hidden = lc.hidden_dim;

    // 每层: QKV + O proj + MLP (gate_up + down) + AdaLN
    let per_layer = hidden * hidden * 4  // QKV
        + hidden * hidden                 // O proj
        + hidden * lc.mlp_dim * 2         // MLP gate_up
        + lc.mlp_dim * hidden             // MLP down
        + hidden * 2 * 4                  // AdaLN (4 个 norm)
        + lc.time_embed_dim * hidden * 2; // Time embedding

    per_layer * config.num_layers
}

// ─── 架构迁移路径 ────────────────────────────────────────

/// 架构迁移路径
///
/// 描述从当前 v2 架构迁移到 v3 架构的步骤
#[derive(Debug, Clone)]
pub struct MigrationPath {
    /// 步骤列表
    pub steps: Vec<MigrationStep>,
}

/// 单个迁移步骤
#[derive(Debug, Clone)]
pub struct MigrationStep {
    /// 步骤名称
    pub name: String,
    /// 描述
    pub description: String,
    /// 预计工作量（天）
    pub estimated_days: f32,
    /// 前置依赖
    pub dependencies: Vec<String>,
}

impl MigrationPath {
    /// 创建 v2→v3 迁移路径
    #[must_use]
    pub fn v2_to_v3() -> Self {
        Self {
            steps: vec![
                MigrationStep {
                    name: "1. DiT 模型加载".to_string(),
                    description: "实现 DiT (Diffusion Transformer) 模型的权重加载和前向传播。\
                        需要: safetensors 权重文件, AdaLN 实现, 时间步嵌入。"
                        .to_string(),
                    estimated_days: 5.0,
                    dependencies: vec![],
                },
                MigrationStep {
                    name: "2. CFM 推理循环".to_string(),
                    description: "实现 CFM 迭代去噪循环 (32 步 Euler 步进)。\
                        需要: 初始噪声生成, 条件引导 (语义 token embedding), 时间步调度。"
                        .to_string(),
                    estimated_days: 3.0,
                    dependencies: vec!["1. DiT 模型加载".to_string()],
                },
                MigrationStep {
                    name: "3. BigVGAN 声码器".to_string(),
                    description: "实现 BigVGAN 声码器 (mel → waveform)。\
                        需要: SnakeBeta 激活函数 (已在 decoder 中实现),\
                        anti-aliasing 滤波, 多尺度上采样。"
                        .to_string(),
                    estimated_days: 5.0,
                    dependencies: vec![],
                },
                MigrationStep {
                    name: "4. 分块推理 + SOLA 拼接".to_string(),
                    description: "实现分块 CFM 推理和 SOLA 拼接 (sola.rs 已实现)。\
                        需要: mel 分块, 参考条件更新, SOLA 拼接集成。"
                        .to_string(),
                    estimated_days: 2.0,
                    dependencies: vec!["2. CFM 推理循环".to_string()],
                },
                MigrationStep {
                    name: "5. 管道集成".to_string(),
                    description: "将 CFM + BigVGAN 解码器集成到 video-translator 管道。\
                        需要: 解码器接口抽象, 配置切换 (v2/v3), 性能基准测试。"
                        .to_string(),
                    estimated_days: 3.0,
                    dependencies: vec![
                        "2. CFM 推理循环".to_string(),
                        "3. BigVGAN 声码器".to_string(),
                        "4. 分块推理 + SOLA 拼接".to_string(),
                    ],
                },
                MigrationStep {
                    name: "6. 质量评估".to_string(),
                    description: "对比 v2 vs v3 的音频质量 (MOS, PESQ, 说话人相似度)。\
                        需要: 测试数据集, 评估脚本, A/B 对比测试。"
                        .to_string(),
                    estimated_days: 2.0,
                    dependencies: vec!["5. 管道集成".to_string()],
                },
            ],
        }
    }

    /// 估算总工作量（天）
    #[must_use]
    pub fn total_days(&self) -> f32 {
        self.steps.iter().map(|s| s.estimated_days).sum()
    }
}

impl std::fmt::Display for MigrationPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "v2 → v3 迁移路径 (总计 {:.0} 天):", self.total_days())?;
        for step in &self.steps {
            writeln!(f, "\n  {} ({:.0} 天)", step.name, step.estimated_days)?;
            writeln!(f, "    {}", step.description)?;
            if !step.dependencies.is_empty() {
                writeln!(f, "    依赖: {}", step.dependencies.join(", "))?;
            }
        }
        Ok(())
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── CfmConfig 测试 ───────────────────────────────

    #[test]
    fn test_cfm_config_default() {
        let config = CfmConfig::default();
        assert_eq!(config.n_timesteps, DEFAULT_CFM_STEPS);
        assert_eq!(config.dit_layers, DEFAULT_DIT_LAYERS);
        assert_eq!(config.dit_dim, DEFAULT_DIT_DIM);
        assert!(config.use_euler);
    }

    #[test]
    fn test_cfm_dt() {
        let config = CfmConfig::default();
        let dt = config.dt();
        assert!((dt - 1.0 / 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cfm_timesteps_uniform() {
        let config = CfmConfig {
            time_sampling: TimeSampling::Uniform,
            ..Default::default()
        };
        let ts = config.timesteps();
        assert_eq!(ts.len(), DEFAULT_CFM_STEPS);
        assert!((ts[0] - 0.0).abs() < 1e-6);
        assert!(ts[31] > 0.9);
    }

    #[test]
    fn test_cfm_timesteps_polynomial() {
        let config = CfmConfig {
            time_sampling: TimeSampling::Polynomial(2.0),
            ..Default::default()
        };
        let ts = config.timesteps();
        assert_eq!(ts.len(), DEFAULT_CFM_STEPS);
        // 多项式采样应该集中在前半段
        assert!(ts[16] < 0.5);
    }

    #[test]
    fn test_cfm_estimate_time() {
        let config = CfmConfig::default();
        // 100 mel 帧, 50ms/step
        let time = config.estimate_inference_time(100, 50.0);
        // 32 steps × 50ms × 1.0 scale / 1000 = 1.6s
        assert!((time - 1.6).abs() < 0.1);
    }

    // ─── CfmStep 测试 ──────────────────────────────────

    #[test]
    fn test_euler_step() {
        let step = CfmStep::new(0.5, 0.03125, 3);
        let x = vec![1.0f32, 2.0, 3.0];
        let v = vec![0.1f32, 0.2, 0.3];
        let result = step.euler_step(&x, &v);
        // x + dt * v = [1 + 0.03125*0.1, 2 + 0.03125*0.2, 3 + 0.03125*0.3]
        assert!((result[0] - 1.003125).abs() < 1e-5);
        assert!((result[1] - 2.00625).abs() < 1e-5);
        assert!((result[2] - 3.009375).abs() < 1e-5);
    }

    #[test]
    fn test_heun_step() {
        let step = CfmStep::new(0.5, 0.03125, 2);
        let x = vec![1.0f32, 2.0];
        let v1 = vec![0.1f32, 0.2];
        let v2 = vec![0.15f32, 0.25];
        let result = step.heun_step(&x, &v1, &v2);
        // x + dt * (v1 + v2) / 2
        let expected = |i: usize| x[i] + 0.03125 * (v1[i] + v2[i]) * 0.5;
        assert!((result[0] - expected(0)).abs() < 1e-5);
        assert!((result[1] - expected(1)).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "x and v must have same length")]
    fn test_euler_step_mismatch() {
        let step = CfmStep::new(0.5, 0.03, 2);
        let _ = step.euler_step(&[1.0, 2.0], &[0.1]);
    }

    // ─── BigVganConfig 测试 ────────────────────────────

    #[test]
    fn test_bigvgan_config_default() {
        let config = BigVganConfig::default();
        assert_eq!(config.channels, DEFAULT_BIGVGAN_CHANNELS);
        assert_eq!(config.upsample_blocks, 4);
        assert!(config.use_anti_alias);
    }

    #[test]
    fn test_bigvgan_actual_upsample() {
        let config = BigVganConfig::default();
        // 8 * 8 * 2 * 2 = 256
        assert_eq!(config.actual_upsample(), 256);
    }

    #[test]
    fn test_bigvgan_estimate_time() {
        let config = BigVganConfig::default();
        let time = config.estimate_inference_time(100);
        // 基准: 100 帧 → 50ms → 0.05s
        assert!((time - 0.05).abs() < 0.01);
    }

    // ─── DecoderArchitecture 测试 ──────────────────────

    #[test]
    fn test_convnet_v2_architecture() {
        let arch = DecoderArchitecture::convnet_v2();
        assert_eq!(arch.decoder_type, DecoderType::ConvNetV2);
        assert_eq!(arch.upsample_ratio, CURRENT_CONVNET_UPSAMPLE);
        assert!(arch.cfm_config.is_none());
        assert!(arch.bigvgan_config.is_none());
    }

    #[test]
    fn test_cfm_bigvgan_v3_architecture() {
        let arch = DecoderArchitecture::cfm_bigvgan_v3();
        assert_eq!(arch.decoder_type, DecoderType::CfmBigVganV3);
        assert!(arch.cfm_config.is_some());
        assert!(arch.bigvgan_config.is_some());
    }

    #[test]
    fn test_hybrid_architecture() {
        let arch = DecoderArchitecture::hybrid();
        assert_eq!(arch.decoder_type, DecoderType::Hybrid);
        assert!(arch.cfm_config.is_some());
        // 混合模式使用 8 步
        assert_eq!(arch.cfm_config.as_ref().unwrap().n_timesteps, 8);
    }

    #[test]
    fn test_estimate_rtf_v2() {
        let arch = DecoderArchitecture::convnet_v2();
        let rtf = arch.estimate_rtf(100, 0.0);
        // 100 mel 帧 ≈ 1.16s 音频, v2 推理 ≈ 1.3s → RTF ≈ 1.12
        assert!(rtf > 0.5 && rtf < 2.0);
    }

    // ─── ArchitectureComparison 测试 ───────────────────

    #[test]
    fn test_comparison_report() {
        let comp = ArchitectureComparison::new();
        let report = comp.report(100, 50.0);
        assert!(report.v2_inference_secs > 0.0);
        assert!(report.v3_inference_secs > 0.0);
        assert!(report.speedup_factor > 0.0);
        // v3 应该比 v2 慢 (CFM 32 步)
        assert!(report.v3_inference_secs > report.v2_inference_secs);
        assert_eq!(report.quality_gain, QualityGain::Significant);
        assert_eq!(report.migration_cost, MigrationCost::High);
    }

    // ─── ChunkedCfmInference 测试 ──────────────────────

    #[test]
    fn test_chunk_plan_short() {
        let cfm = ChunkedCfmInference::default();
        // 50 帧 < chunk_len(100) → 1 个 chunk
        let plan = cfm.chunk_plan(50);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0], (0, 50));
    }

    #[test]
    fn test_chunk_plan_long() {
        let cfm = ChunkedCfmInference::new(100, 20);
        // 250 帧 → 3 chunks: [0,100), [100,200), [200,250)
        let plan = cfm.chunk_plan(250);
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0], (0, 100));
        assert_eq!(plan[1], (100, 200));
        assert_eq!(plan[2], (200, 250));
    }

    #[test]
    fn test_chunk_plan_exact_multiple() {
        let cfm = ChunkedCfmInference::new(100, 20);
        let plan = cfm.chunk_plan(200);
        assert_eq!(plan.len(), 2);
    }

    #[test]
    fn test_num_chunks() {
        let cfm = ChunkedCfmInference::new(100, 20);
        assert_eq!(cfm.num_chunks(50), 1);
        assert_eq!(cfm.num_chunks(100), 1);
        assert_eq!(cfm.num_chunks(101), 2);
        assert_eq!(cfm.num_chunks(300), 3);
    }

    #[test]
    fn test_reference_range() {
        let cfm = ChunkedCfmInference::new(100, 20);
        let (start, end) = cfm.reference_range(0, 100);
        assert_eq!(start, 80);
        assert_eq!(end, 100);
    }

    #[test]
    fn test_reference_range_first_chunk() {
        let cfm = ChunkedCfmInference::new(100, 20);
        // 第一个 chunk, prev_chunk_end = 0
        let (start, end) = cfm.reference_range(0, 0);
        assert_eq!(start, 0);
        assert_eq!(end, 0);
    }

    #[test]
    fn test_chunk_estimate_time() {
        let cfm = ChunkedCfmInference::new(100, 20);
        // 200 mel 帧 → 2 chunks, 每个 100+20=120 帧
        let time = cfm.estimate_total_time(200, 50.0);
        assert!(time > 0.0);
    }

    // ─── DitModelConfig 测试 ───────────────────────────

    #[test]
    fn test_dit_config_default() {
        let config = DitModelConfig::default();
        assert_eq!(config.num_layers, DEFAULT_DIT_LAYERS);
        assert_eq!(config.input_dim, 80);
        assert_eq!(config.patch_size, 1);
    }

    #[test]
    fn test_estimate_dit_params() {
        let config = DitModelConfig::default();
        let params = estimate_dit_params(&config);
        // 22 层, 每层约 ~10M 参数 → 总计约 ~220M
        assert!(
            params > 100_000_000,
            "DiT should have >100M params, got {params}"
        );
        assert!(
            params < 500_000_000,
            "DiT should have <500M params, got {params}"
        );
    }

    // ─── MigrationPath 测试 ────────────────────────────

    #[test]
    fn test_migration_path() {
        let path = MigrationPath::v2_to_v3();
        assert_eq!(path.steps.len(), 6);
        assert!(path.total_days() > 15.0);
    }

    #[test]
    fn test_migration_path_display() {
        let path = MigrationPath::v2_to_v3();
        let s = format!("{path}");
        assert!(s.contains("v2 → v3"));
        assert!(s.contains("DiT"));
        assert!(s.contains("BigVGAN"));
    }

    #[test]
    fn test_migration_step_dependencies() {
        let path = MigrationPath::v2_to_v3();
        // 步骤 2 (CFM 推理) 依赖步骤 1 (DiT)
        let step2 = &path.steps[1];
        assert!(step2.dependencies.contains(&"1. DiT 模型加载".to_string()));
    }
}
