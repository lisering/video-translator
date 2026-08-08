//! 核心 TTS 引擎 trait
//!
//! 参考 rs-voice-toolkit 的 async_trait 设计和 Qwen3-TTS 的 Qwen3TTS 接口。

use std::path::Path;

use anyhow::Result;

use crate::audio::AudioBuffer;
use crate::config::TtsEngineConfig;

/// 合成选项
///
/// 参考 Qwen3-TTS 的 SynthesisOptions。
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// 采样温度
    pub temperature: f32,
    /// Top-K 采样
    pub top_k: usize,
    /// 重复惩罚 (1.0 = 禁用, 1.1-1.3 = 轻度惩罚, >1.5 = 强惩罚)
    ///
    /// 参考 HF Transformers 实现: 对已出现 token 的 logits 除以惩罚值 (正logit)
    /// 或乘以惩罚值 (负logit)，降低重复概率。
    pub repetition_penalty: f32,
    /// No-repeat n-gram size (0 = 禁用)
    ///
    /// 防止任何长度为 n 的 token 序列重复出现。对重复检测特别有效。
    /// 设为 0 禁用，建议值为 3 (防止 3-gram 重复)。
    pub no_repeat_ngram_size: usize,
    /// 随机种子
    pub seed: Option<u64>,
    /// 最大生成 token 数
    pub max_codes: usize,
    /// 推测解码 (speculative decoding): 使用 n-gram 推测表加速生成
    ///
    /// 启用后，每步尝试用 n-gram 表推测下一个 token，在单次前向传播中
    /// 处理 2 个 token。适合长文本 (100+ tokens) 中重复模式较多的场景。
    /// 短文本命中率低，可能因回滚开销而变慢。
    pub speculative: bool,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            repetition_penalty: 1.05,
            no_repeat_ngram_size: 0,
            seed: None,
            max_codes: 500,
            speculative: false,
        }
    }
}

/// 合成结果
///
/// 包含音频数据和性能统计信息。
#[derive(Debug)]
pub struct SynthesisResult {
    /// 合成的音频
    pub audio: AudioBuffer,
    /// 生成耗时（秒）
    pub elapsed_secs: f64,
    /// 生成的 codec 帧数
    pub num_frames: usize,
    /// RTF (Real-Time Factor) = 生成时间 / 音频时长
    pub rtf: f64,
}

/// 声音克隆提示
///
/// 参考 Qwen3-TTS 的 VoiceClonePrompt。
/// 包含从参考音频提取的说话人嵌入和可选的 ICL 数据。
#[derive(Debug, Clone)]
pub struct VoiceClonePrompt {
    /// 说话人嵌入向量（ECAPA-TDNN 输出，通常 1024 维）
    pub speaker_embedding: Vec<f32>,
    /// 参考音频的 codec codes（ICL 模式使用）
    pub ref_codes: Option<Vec<Vec<u32>>>,
    /// 参考文本的 token IDs（ICL 模式使用）
    pub ref_text_ids: Option<Vec<u32>>,
}

/// TTS 引擎核心 trait
///
/// 定义文本到语音合成的标准接口。
///
/// # 设计参考
/// - **Qwen3-TTS**: `Qwen3TTS::synthesize()` 接口
/// - **rs-voice-toolkit**: `TtsService` async trait
/// - **QORA-TTS**: `generate_speech()` 函数式接口
pub trait TtsEngine: Send + Sync {
    /// 合成语音
    ///
    /// # 参数
    /// - `text`: 要合成的文本
    /// - `voice_clone`: 声音克隆提示（None = 使用默认说话人）
    /// - `options`: 合成选项
    ///
    /// # 返回
    /// 合成结果，包含音频数据和性能统计。
    fn synthesize(
        &self,
        text: &str,
        voice_clone: Option<&VoiceClonePrompt>,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult>;

    /// 从参考音频创建声音克隆提示
    ///
    /// 提取说话人嵌入，用于后续合成时的音色注入。
    ///
    /// # 参数
    /// - `ref_audio_path`: 参考音频文件路径
    /// - `ref_text`: 参考音频对应的文本（ICL 模式需要）
    ///
    /// # 返回
    /// 声音克隆提示，包含说话人嵌入。
    fn create_voice_clone_prompt(
        &self,
        ref_audio_path: &Path,
        ref_text: Option<&str>,
    ) -> Result<VoiceClonePrompt>;

    /// 引擎名称
    fn name(&self) -> &str;

    /// 是否支持声音克隆
    fn supports_voice_cloning(&self) -> bool;

    /// 模型变体名称（如 "0.6B-Base", "1.7B-Base"）
    fn model_variant(&self) -> &str;
}

/// 从配置创建 TTS 引擎实例
///
/// 根据 model_dir 中的 config.json 自动检测模型变体。
#[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
pub fn create_engine(config: TtsEngineConfig) -> Result<Box<dyn TtsEngine>> {
    crate::talker::CandleTtsEngine::new(config).map(|e| Box::new(e) as Box<dyn TtsEngine>)
}
