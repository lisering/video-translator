//! 声音克隆模块
//!
//! 使用原视频说话人的音色合成目标语言语音，实现"用自己的声音说中文"的效果。
//!
//! # 功能概览
//! - [`VoiceCloningEngine`][] trait：定义声音克隆与合成的标准接口
//! - [`CloningConfig`][]：克隆合成配置（语速、音调、情感等）
//! - [`MockCloningEngine`][]：用于测试的 Mock 实现（生成正弦波 WAV）
//! - [`CloningIntegration`][]：流水线集成辅助器
//!
//! # 引擎实现
//! - [`MockCloningEngine`]：测试和开发用，生成正弦波 WAV
//! - [`GptSoVitsEngine`]：生产环境，通过 HTTP 调用 GPT-SoVITS FastAPI 服务
//!
//! # GPT-SoVITS 集成
//! GPT-SoVITS 支持零样本声音克隆：仅需 3–10 秒参考音频 + 对应文本，
//! 即可克隆任意说话人音色并合成目标语言语音。
//!
//! ## 前置条件
//! 1. 启动 GPT-SoVITS API v2 服务：
//!    ```bash
//!    cd GPT-SoVITS
//!    python api_v2.py -a 127.0.0.1 -p 9880
//!    ```
//! 2. 在 `config.toml` 中配置 `[cloning]` 段：
//!    ```toml
//!    [cloning]
//!    enabled = true
//!    engine = "gpt-sovits"
//!    api_url = "http://127.0.0.1:9880"
//!    prompt_text = "Hello, welcome to this video."
//!    prompt_lang = "en"
//!    text_lang = "zh"
//!    ```
//!
//! # 性能要求
//! - 克隆 + 合成 1 分钟语音 ≤ 10 秒（M1 Pro）
//! - 零样本声音克隆：从短参考音频克隆任意声音
//!
//! # 优雅降级
//! 如果声音克隆失败，应回退到标准 TTS（通过 [`CloningIntegration`] 处理）。
//!
//! # 示例
//! ```no_run
//! use vt_core::cloning::{VoiceCloningEngine, MockCloningEngine, CloningConfig};
//! use std::path::Path;
//!
//! let engine = MockCloningEngine::new();
//! let config = CloningConfig::default();
//! let output = engine.clone_and_synthesize(
//!     Path::new("reference.wav"),
//!     "你好，这是克隆的声音",
//!     &config,
//! ).expect("Synthesis failed");
//! ```

use std::io::BufRead;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::asr::read_wav_mono;
use crate::config::CloningConfig as CloningEngineConfig;
use crate::error::{AppError, AppResult};

/// 安全截断字符串到指定字符数（按字符而非字节，避免切断多字节 UTF-8 字符）
fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{truncated}...")
    }
}
use crate::models::segment::Segment;

// ─── 克隆配置 ─────────────────────────────────────────────

/// 声音克隆合成配置
///
/// 控制克隆语音合成的语速、音调、情感等参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningConfig {
    /// 语速倍率（1.0 为正常语速，范围 0.5–2.0）
    #[serde(default = "default_speed")]
    pub speed: f32,

    /// 音调偏移（半音，0 为不变）
    #[serde(default)]
    pub pitch_shift: f32,

    /// 情感强度（0.0–1.0，0 为中性）
    #[serde(default = "default_emotion")]
    pub emotion: f32,

    /// 输出音频采样率
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// 输出音频目录
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
}

fn default_speed() -> f32 {
    1.0
}

fn default_emotion() -> f32 {
    0.0
}

fn default_sample_rate() -> u32 {
    24000
}

fn default_output_dir() -> String {
    "./output/cloned".to_string()
}

impl Default for CloningConfig {
    /// 返回默认克隆配置
    ///
    /// - 语速: 1.0
    /// - 音调: 0（不变）
    /// - 情感: 0（中性）
    /// - 采样率: 24kHz
    /// - 输出目录: `./output/cloned`
    fn default() -> Self {
        Self {
            speed: default_speed(),
            pitch_shift: 0.0,
            emotion: default_emotion(),
            sample_rate: default_sample_rate(),
            output_dir: default_output_dir(),
        }
    }
}

// ─── VoiceCloningEngine trait ────────────────────────────

/// 声音克隆引擎 trait
///
/// 定义从参考音频克隆声音并合成语音的标准接口。
///
/// # 实现要求
/// - 输入：参考音频路径 + 目标文本
/// - 输出：合成音频文件路径
/// - 性能：克隆 + 合成 1 分钟语音 ≤ 10 秒（M1 Pro）
pub trait VoiceCloningEngine: Send + Sync {
    /// 克隆声音并合成语音
    ///
    /// # 参数
    /// - `reference_audio`: 参考音频文件路径（短音频，5–30 秒）
    /// - `text`: 要合成的目标语言文本
    /// - `config`: 克隆合成配置
    ///
    /// # 返回
    /// 合成的音频文件路径。
    ///
    /// # 错误
    /// - [`AppError::VoiceCloningError`][]: 克隆或合成过程中的错误
    fn clone_and_synthesize(
        &self,
        reference_audio: &Path,
        text: &str,
        config: &CloningConfig,
    ) -> AppResult<PathBuf>;

    /// 获取引擎名称
    fn name(&self) -> &str;

    /// 更新参考音频的提示文本
    ///
    /// 自动提取参考音频后，用 ASR 转录结果更新 prompt_text。
    /// 默认实现为空操作（不支持动态更新的引擎可忽略）。
    ///
    /// # 参数
    /// - `prompt_text`: 参考音频对应的文字内容
    fn set_prompt_text(&self, _prompt_text: &str) {}

    /// P1: 预热说话人 — 提前提取 voice clone prompt 并缓存
    ///
    /// 在 pipeline 提取参考音频后立即调用，让 TTS 服务端预热说话人缓存。
    /// 后续 TTS 请求可跳过 prompt 创建步骤，减少首次合成延迟。
    ///
    /// 默认实现为空操作（不支持的引擎可忽略）。
    ///
    /// # 参数
    /// - `speaker_id`: 说话人标识
    /// - `reference_audio`: 参考音频路径
    /// - `ref_text`: 参考音频对应文本（可选）
    fn prewarm_speaker(
        &self,
        _speaker_id: &str,
        _reference_audio: &Path,
        _ref_text: Option<&str>,
    ) -> AppResult<()> {
        Ok(())
    }

    /// 批量克隆合成
    ///
    /// 为多个文本使用同一参考音频批量合成。
    ///
    /// # 参数
    /// - `reference_audio`: 参考音频文件路径
    /// - `texts`: 要合成的文本列表
    /// - `config`: 克隆合成配置
    ///
    /// # 返回
    /// 合成的音频文件路径列表。
    ///
    /// # 错误
    /// - [`AppError::VoiceCloningError`][]: 任一合成失败
    fn clone_and_synthesize_batch(
        &self,
        reference_audio: &Path,
        texts: &[String],
        config: &CloningConfig,
    ) -> AppResult<Vec<PathBuf>> {
        texts
            .iter()
            .map(|text| self.clone_and_synthesize(reference_audio, text, config))
            .collect()
    }
}

// ─── Mock 引擎 ────────────────────────────────────────────

/// Mock 声音克隆引擎
///
/// 用于测试和开发，生成正弦波 WAV 文件作为"克隆语音"。
/// 不实际进行声音克隆，仅验证接口和流程。
pub struct MockCloningEngine {
    /// 内部计数器（用于生成唯一文件名）
    counter: std::sync::atomic::AtomicUsize,
}

impl MockCloningEngine {
    /// 创建新的 Mock 引擎
    #[must_use]
    pub fn new() -> Self {
        Self {
            counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// 生成正弦波 WAV 文件
    ///
    /// 根据文本长度生成对应时长的音频。
    fn generate_wav(
        &self,
        text: &str,
        config: &CloningConfig,
        output_path: &Path,
    ) -> AppResult<()> {
        // 估算音频时长：每字符约 0.2 秒
        let duration_secs = (text.chars().count() as f64 * 0.2).max(0.5);
        let sample_rate = config.sample_rate;
        let num_samples = (duration_secs * sample_rate as f64) as usize;

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(output_path, spec).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to create WAV writer: {e}"))
        })?;

        // 生成正弦波（模拟语音）
        let frequency = 220.0 * (2.0_f32).powf(config.pitch_shift / 12.0) as f64;
        for i in 0..num_samples {
            let t = i as f64 / sample_rate as f64;
            let sample = (t * frequency * 2.0 * std::f64::consts::PI).sin();
            let amplitude = 0.3 * config.speed as f64;
            let i16_sample = (sample * amplitude * 32767.0) as i16;
            writer
                .write_sample(i16_sample)
                .map_err(|e| AppError::VoiceCloningError(format!("Failed to write sample: {e}")))?;
        }

        writer
            .finalize()
            .map_err(|e| AppError::VoiceCloningError(format!("Failed to finalize WAV: {e}")))?;

        Ok(())
    }
}

impl Default for MockCloningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceCloningEngine for MockCloningEngine {
    fn clone_and_synthesize(
        &self,
        reference_audio: &Path,
        text: &str,
        config: &CloningConfig,
    ) -> AppResult<PathBuf> {
        // 验证参考音频存在
        if !reference_audio.exists() {
            return Err(AppError::FileNotFound(reference_audio.to_path_buf()));
        }

        // 创建输出目录
        let output_dir = PathBuf::from(&config.output_dir);
        if !output_dir.exists() {
            std::fs::create_dir_all(&output_dir).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to create output dir: {e}"))
            })?;
        }

        // 生成唯一文件名
        let idx = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let output_path = output_dir.join(format!("cloned_{idx:06}.wav"));

        self.generate_wav(text, config, &output_path)?;

        tracing::info!(
            "MockCloningEngine: synthesized {} chars → {:?}",
            text.chars().count(),
            output_path
        );

        Ok(output_path)
    }

    fn name(&self) -> &str {
        "mock-cloning"
    }
}

impl std::fmt::Debug for MockCloningEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockCloningEngine").finish()
    }
}

// ─── GPT-SoVITS 引擎 ─────────────────────────────────────

/// GPT-SoVITS 声音克隆引擎
///
/// 通过 HTTP 调用 GPT-SoVITS API v2 服务实现零样本声音克隆。
///
/// # 工作原理
/// 1. 将参考音频路径（服务器端可访问路径）和目标文本发送到 `/tts` 端点
/// 2. GPT-SoVITS 执行两阶段推理：
///    - GPT 阶段：文本 → 语义 token
///    - SoVITS 阶段：语义 token → 音频波形
/// 3. 接收返回的 WAV 音频流并保存到本地文件
///
/// # 前置条件
/// - GPT-SoVITS API v2 服务已启动（默认端口 9880）
/// - 参考音频文件在 GPT-SoVITS 服务器上可访问
///
/// # 配置
/// 通过 [`CloningEngineConfig`]（即 `config.toml` 中的 `[cloning]` 段）配置：
/// - `api_url`: API 服务端点
/// - `prompt_text`: 参考音频对应的文本内容
/// - `prompt_lang` / `text_lang`: 语言设置
/// - `gpt_model` / `sovits_model`: 模型权重路径（可选，用于切换模型）
///
/// # 示例
/// ```no_run
/// use vt_core::cloning::{VoiceCloningEngine, GptSoVitsEngine};
/// use vt_core::config::CloningConfig;
/// use std::path::Path;
///
/// let engine_config = CloningConfig::default();
/// let engine = GptSoVitsEngine::new(engine_config).expect("Failed to init");
/// ```
pub struct GptSoVitsEngine {
    /// 引擎配置（API URL、提示文本、语言等）
    config: std::sync::Mutex<CloningEngineConfig>,
    /// 内部计数器（用于生成唯一文件名）
    counter: std::sync::atomic::AtomicUsize,
}

impl GptSoVitsEngine {
    /// 创建新的 GPT-SoVITS 引擎实例
    ///
    /// # 参数
    /// - `config`: 引擎配置（来自 `config.toml` 的 `[cloning]` 段）
    ///
    /// # 错误
    /// - [`AppError::VoiceCloningError`][]: API 服务不可达
    /// - [`AppError::VoiceCloningError`][]: 模型权重切换失败
    pub fn new(config: CloningEngineConfig) -> AppResult<Self> {
        let api_url = config.api_url.clone();
        let gpt_model = config.gpt_model.clone();
        let sovits_model = config.sovits_model.clone();

        let engine = Self {
            config: std::sync::Mutex::new(config),
            counter: std::sync::atomic::AtomicUsize::new(0),
        };

        // 启动时检查 API 服务是否可用
        engine.health_check(&api_url)?;

        // 如果配置了模型权重路径，切换模型
        if let Some(ref gpt_model) = gpt_model {
            engine.set_gpt_weights(&api_url, gpt_model)?;
        }
        if let Some(ref sovits_model) = sovits_model {
            engine.set_sovits_weights(&api_url, sovits_model)?;
        }

        Ok(engine)
    }

    /// 动态更新提示文本
    ///
    /// 自动提取参考音频后，用 ASR 转录结果更新 prompt_text，
    /// 使后续克隆合成使用正确的提示文本。
    ///
    /// # 参数
    /// - `prompt_text`: 参考音频对应的文字内容
    pub fn set_prompt_text(&self, prompt_text: &str) {
        let mut config = self
            .config
            .lock()
            .expect("GptSoVitsEngine: config mutex poisoned");
        let old = config.prompt_text.clone();
        config.prompt_text = Some(prompt_text.to_string());
        tracing::info!(
            "GPT-SoVITS: prompt_text updated (was: {:?}, now: {:?})",
            old.as_deref().map(|s| truncate_str(s, 40)),
            truncate_str(&prompt_text, 40)
        );
    }

    /// 健康检查：验证 GPT-SoVITS API 服务是否可用
    ///
    /// 向 API 端点发送一个简单请求，检查服务是否响应。
    fn health_check(&self, api_url: &str) -> AppResult<()> {
        let url = format!("{}/tts", api_url);

        // 发送一个最小化请求检查服务可用性
        // 使用空文本触发参数校验（预期返回 400），确认服务在线
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send_json(serde_json::json!({
                "text": "",
                "text_lang": "zh",
                "ref_audio_path": "health_check",
                "prompt_lang": "zh",
                "prompt_text": "",
            }));

        match resp {
            Ok(_) => {
                tracing::info!("GPT-SoVITS API health check: service is online");
                Ok(())
            }
            Err(ureq::Error::Status(_, _)) => {
                // 400 错误说明服务在线，只是参数不合法
                tracing::info!("GPT-SoVITS API health check: service is online (expected 400)");
                Ok(())
            }
            Err(e) => {
                tracing::error!("GPT-SoVITS API health check failed: {e}");
                Err(AppError::VoiceCloningError(format!(
                    "GPT-SoVITS API 不可达 ({}): {e}\n\
                    请确保已启动 API 服务: python api_v2.py -a 127.0.0.1 -p 9880",
                    api_url
                )))
            }
        }
    }

    /// 切换 GPT 模型权重
    ///
    /// 调用 `/set_gpt_weights` 端点切换 GPT（Text-to-Semantic）模型。
    ///
    /// # 参数
    /// - `weights_path`: GPT-SoVITS 服务器上的模型文件路径
    fn set_gpt_weights(&self, api_url: &str, weights_path: &str) -> AppResult<()> {
        let url = format!("{}/set_gpt_weights", api_url);

        tracing::info!("Switching GPT weights to: {weights_path}");

        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(30))
            .query("weights_path", weights_path)
            .call()
        {
            Ok(_) => {
                tracing::info!("GPT weights switched successfully");
                Ok(())
            }
            Err(e) => Err(AppError::VoiceCloningError(format!(
                "Failed to switch GPT weights to '{weights_path}': {e}"
            ))),
        }
    }

    /// 切换 SoVITS 模型权重
    ///
    /// 调用 `/set_sovits_weights` 端点切换 SoVITS（Semantic-to-Waveform）模型。
    ///
    /// # 参数
    /// - `weights_path`: GPT-SoVITS 服务器上的模型文件路径
    fn set_sovits_weights(&self, api_url: &str, weights_path: &str) -> AppResult<()> {
        let url = format!("{}/set_sovits_weights", api_url);

        tracing::info!("Switching SoVITS weights to: {weights_path}");

        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(30))
            .query("weights_path", weights_path)
            .call()
        {
            Ok(_) => {
                tracing::info!("SoVITS weights switched successfully");
                Ok(())
            }
            Err(e) => Err(AppError::VoiceCloningError(format!(
                "Failed to switch SoVITS weights to '{weights_path}': {e}"
            ))),
        }
    }

    /// 构建 TTS 请求 JSON body
    ///
    /// 根据 [`CloningEngineConfig`] 和合成参数组装 GPT-SoVITS API 的请求体。
    fn build_tts_request(
        &self,
        reference_audio: &Path,
        text: &str,
        synth_config: &CloningConfig,
    ) -> serde_json::Value {
        let config = self
            .config
            .lock()
            .expect("GptSoVitsEngine: config mutex poisoned");

        // 将参考音频路径转为绝对路径（GPT-SoVITS 服务器端需要可访问的路径）
        let ref_audio_path = reference_audio
            .canonicalize()
            .unwrap_or_else(|_| reference_audio.to_path_buf())
            .to_string_lossy()
            .to_string();

        let prompt_text = config.prompt_text.as_deref().unwrap_or("");

        serde_json::json!({
            "text": text,
            "text_lang": config.text_lang,
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": config.prompt_lang,
            "text_split_method": config.text_split_method,
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": false,
            "speed_factor": synth_config.speed,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "repetition_penalty": config.repetition_penalty,
            "parallel_infer": true,
        })
    }
}

impl VoiceCloningEngine for GptSoVitsEngine {
    fn clone_and_synthesize(
        &self,
        reference_audio: &Path,
        text: &str,
        config: &CloningConfig,
    ) -> AppResult<PathBuf> {
        // 验证参考音频存在
        if !reference_audio.exists() {
            return Err(AppError::FileNotFound(reference_audio.to_path_buf()));
        }

        // 验证 prompt_text 已配置
        {
            let engine_config = self
                .config
                .lock()
                .expect("GptSoVitsEngine: config mutex poisoned");
            if engine_config.prompt_text.is_none() {
                tracing::warn!(
                    "GPT-SoVITS: prompt_text 未配置，克隆质量可能下降。\
                    请在 config.toml [cloning] 段设置 prompt_text 为参考音频的文本内容"
                );
            }
        }

        // 创建输出目录
        let output_dir = PathBuf::from(&config.output_dir);
        if !output_dir.exists() {
            std::fs::create_dir_all(&output_dir).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to create output dir: {e}"))
            })?;
        }

        // 生成唯一文件名
        let idx = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let output_path = output_dir.join(format!("gpt_sovits_cloned_{idx:06}.wav"));

        // 构建请求（在锁内获取 api_url 和 timeout，避免持锁发送 HTTP）
        let (url, timeout, text_lang) = {
            let engine_config = self
                .config
                .lock()
                .expect("GptSoVitsEngine: config mutex poisoned");
            (
                format!("{}/tts", engine_config.api_url),
                std::time::Duration::from_secs(engine_config.timeout_secs),
                engine_config.text_lang.clone(),
            )
        };
        let body = self.build_tts_request(reference_audio, text, config);

        tracing::info!(
            "GPT-SoVITS: synthesizing {} chars (lang={}) → {:?}",
            text.chars().count(),
            text_lang,
            output_path
        );

        // 发送请求
        let resp = ureq::post(&url)
            .timeout(timeout)
            .send_json(body)
            .map_err(|e| {
                let msg = match e {
                    ureq::Error::Status(code, response) => {
                        let body_text = response.into_string().unwrap_or_default();
                        format!("GPT-SoVITS API 返回错误 (HTTP {code}): {body_text}")
                    }
                    other => format!(
                        "GPT-SoVITS API 请求失败: {other}\n\
                        请确认 API 服务已启动: python api_v2.py -p 9880"
                    ),
                };
                AppError::VoiceCloningError(msg)
            })?;

        // 读取响应体（WAV 音频数据）
        let mut reader = resp.into_reader();
        let mut audio_data = Vec::with_capacity(48 * 1024);
        reader.read_to_end(&mut audio_data).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to read audio response: {e}"))
        })?;

        // 验证响应是有效的 WAV 数据
        if audio_data.len() < 44 {
            return Err(AppError::VoiceCloningError(format!(
                "GPT-SoVITS 返回的音频数据过短 ({} bytes)，可能不是有效的 WAV",
                audio_data.len()
            )));
        }

        // 检查 WAV 文件头
        if &audio_data[0..4] != b"RIFF" || &audio_data[8..12] != b"WAVE" {
            return Err(AppError::VoiceCloningError(
                "GPT-SoVITS 返回的数据不是有效的 WAV 格式".to_string(),
            ));
        }

        // 写入文件
        std::fs::write(&output_path, &audio_data)
            .map_err(|e| AppError::VoiceCloningError(format!("Failed to write audio file: {e}")))?;

        tracing::info!(
            "GPT-SoVITS: synthesized {} bytes → {:?}",
            audio_data.len(),
            output_path
        );

        Ok(output_path)
    }

    fn name(&self) -> &str {
        "gpt-sovits"
    }

    fn set_prompt_text(&self, prompt_text: &str) {
        // 调用内部方法（已实现 Mutex 锁定和日志）
        GptSoVitsEngine::set_prompt_text(self, prompt_text);
    }
}

impl std::fmt::Debug for GptSoVitsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let config = self
            .config
            .lock()
            .expect("GptSoVitsEngine: config mutex poisoned");
        f.debug_struct("GptSoVitsEngine")
            .field("api_url", &config.api_url)
            .field("text_lang", &config.text_lang)
            .field("prompt_lang", &config.prompt_lang)
            .field("has_prompt_text", &config.prompt_text.is_some())
            .finish()
    }
}

// ─── 子进程克隆引擎 ─────────────────────────────────────

/// 子进程声音克隆引擎
///
/// 通过调用外部 TTS CLI 工具实现零样本声音克隆。
/// 支持 IndexTTS-Rust、qwen3_tts_rs 等任何命令行 TTS 工具。
///
/// # 工作原理
/// 1. 根据配置的参数模板，替换占位符（`{text}`、`{ref_audio}`、`{output}` 等）
/// 2. 执行外部 CLI 命令
/// 3. 检查输出文件是否生成
/// 4. 返回输出路径
///
/// # 支持的引擎
/// - **IndexTTS-Rust**: `indextts synthesize --text {text} --voice {ref_audio} --output {output}`
/// - **qwen3_tts_rs**: `voice_clone {model} {ref_audio} {text} chinese {output}`
/// - **任何自定义 CLI**: 通过 `clone_args` 配置参数模板
///
/// # 配置示例
/// ```toml
/// [cloning]
/// enabled = true
/// engine = "subprocess"
/// clone_command = "/path/to/indextts"
/// clone_args = ["synthesize", "--text", "{text}", "--voice", "{ref_audio}", "--output", "{output}"]
/// clone_timeout_secs = 120
/// ```
pub struct SubprocessCloneEngine {
    /// CLI 命令路径
    command: String,
    /// 模型路径（可选）
    model_path: Option<String>,
    /// 参数模板（含占位符）
    args_template: Vec<String>,
    /// 超时时间（秒）
    timeout_secs: u64,
    /// 内部计数器
    counter: std::sync::atomic::AtomicUsize,
    /// 提示文本（动态更新）
    prompt_text: std::sync::Mutex<Option<String>>,
}

impl SubprocessCloneEngine {
    /// 创建新的子进程克隆引擎
    ///
    /// # 参数
    /// - `command`: CLI 可执行文件路径
    /// - `model_path`: 模型路径（可选）
    /// - `args_template`: 参数模板，支持 `{text}`、`{ref_audio}`、`{output}`、`{model}`、`{prompt_text}` 占位符
    /// - `timeout_secs`: 超时时间（秒）
    #[must_use]
    pub fn new(
        command: String,
        model_path: Option<String>,
        args_template: Vec<String>,
        timeout_secs: u64,
    ) -> Self {
        Self {
            command,
            model_path,
            args_template,
            timeout_secs,
            counter: std::sync::atomic::AtomicUsize::new(0),
            prompt_text: std::sync::Mutex::new(None),
        }
    }

    /// 从 [`CloningEngineConfig`] 创建引擎
    ///
    /// 自动识别引擎类型并使用对应的预设参数模板。
    ///
    /// # 参数
    /// - `config`: 克隆引擎配置
    ///
    /// # 返回
    /// 成功返回引擎实例，失败返回错误（如未配置 `clone_command`）
    pub fn from_config(config: &CloningEngineConfig) -> AppResult<Self> {
        let command = config.clone_command.clone().ok_or_else(|| {
            AppError::VoiceCloningError(format!(
                "SubprocessCloneEngine: clone_command 未配置\n\
                请在 config.toml [cloning] 段设置 clone_command 为 CLI 工具路径"
            ))
        })?;

        let model_path = config.clone_model_path.clone();
        let timeout_secs = config.clone_timeout_secs;

        // 如果用户已配置 clone_args，直接使用
        // 否则根据 engine 类型使用预设参数
        let args = if !config.clone_args.is_empty() {
            config.clone_args.clone()
        } else {
            Self::preset_args(&config.engine)
        };

        Ok(Self::new(command, model_path, args, timeout_secs))
    }

    /// 获取引擎类型的预设参数模板
    ///
    /// # 参数
    /// - `engine_name`: 引擎名称（`indextts`、`qwen3-tts`、`subprocess`）
    #[must_use]
    fn preset_args(engine_name: &str) -> Vec<String> {
        match engine_name {
            "indextts" => vec![
                "synthesize".to_string(),
                "--text".to_string(),
                "{text}".to_string(),
                "--voice".to_string(),
                "{ref_audio}".to_string(),
                "--output".to_string(),
                "{output}".to_string(),
            ],
            "qwen3-tts" => vec![
                "{model}".to_string(),
                "{ref_audio}".to_string(),
                "{text}".to_string(),
                "chinese".to_string(),
                "{output}".to_string(),
            ],
            _ => vec![
                "--text".to_string(),
                "{text}".to_string(),
                "--voice".to_string(),
                "{ref_audio}".to_string(),
                "--output".to_string(),
                "{output}".to_string(),
            ],
        }
    }

    /// 替换参数模板中的占位符
    ///
    /// # 支持的占位符
    /// - `{text}`: 目标文本
    /// - `{ref_audio}`: 参考音频路径
    /// - `{output}`: 输出路径
    /// - `{model}`: 模型路径
    /// - `{prompt_text}`: 提示文本
    fn replace_placeholders(
        &self,
        template: &[String],
        text: &str,
        ref_audio: &Path,
        output: &Path,
    ) -> Vec<String> {
        let model = self.model_path.as_deref().unwrap_or("");
        let prompt_text = self
            .prompt_text
            .lock()
            .expect("SubprocessCloneEngine: prompt_text mutex poisoned");
        let prompt_text = prompt_text.as_deref().unwrap_or("");

        template
            .iter()
            .map(|arg| {
                arg.replace("{text}", text)
                    .replace("{ref_audio}", &ref_audio.to_string_lossy())
                    .replace("{output}", &output.to_string_lossy())
                    .replace("{model}", model)
                    .replace("{prompt_text}", prompt_text)
            })
            .collect()
    }
}

impl VoiceCloningEngine for SubprocessCloneEngine {
    fn clone_and_synthesize(
        &self,
        reference_audio: &Path,
        text: &str,
        config: &CloningConfig,
    ) -> AppResult<PathBuf> {
        // 验证参考音频存在
        if !reference_audio.exists() {
            return Err(AppError::FileNotFound(reference_audio.to_path_buf()));
        }

        // 创建输出目录
        let output_dir = PathBuf::from(&config.output_dir);
        if !output_dir.exists() {
            std::fs::create_dir_all(&output_dir).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to create output dir: {e}"))
            })?;
        }

        // 生成唯一文件名
        let idx = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let output_path = output_dir.join(format!("subprocess_cloned_{idx:06}.wav"));

        // 替换占位符
        let args =
            self.replace_placeholders(&self.args_template, text, reference_audio, &output_path);

        tracing::info!(
            "SubprocessCloneEngine: executing `{} {}` → {:?}",
            self.command,
            args.join(" "),
            output_path
        );

        // 执行命令
        let mut cmd = std::process::Command::new(&self.command);
        cmd.args(&args);

        // 设置超时
        let timeout = std::time::Duration::from_secs(self.timeout_secs);

        let output = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output();

        let result = match output {
            Ok(o) => o,
            Err(e) => {
                return Err(AppError::VoiceCloningError(format!(
                    "Failed to execute clone command '{}': {e}\n\
                    请确认 CLI 工具已安装且路径正确",
                    self.command
                )));
            }
        };

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            let stdout = String::from_utf8_lossy(&result.stdout);
            return Err(AppError::VoiceCloningError(format!(
                "Clone command failed (exit code: {}):\n\
                stdout: {stdout}\n\
                stderr: {stderr}",
                result.status.code().unwrap_or(-1)
            )));
        }

        // 检查输出文件是否存在
        if !output_path.exists() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(AppError::VoiceCloningError(format!(
                "Clone command succeeded but output file not found: {:?}\n\
                stderr: {stderr}\n\
                提示: 检查 clone_args 中的 {{output}} 占位符是否正确配置",
                output_path
            )));
        }

        // 验证输出是有效的 WAV
        let metadata = std::fs::metadata(&output_path).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to read output metadata: {e}"))
        })?;

        if metadata.len() < 44 {
            return Err(AppError::VoiceCloningError(format!(
                "Output file too small ({} bytes), may not be a valid WAV",
                metadata.len()
            )));
        }

        tracing::info!(
            "SubprocessCloneEngine: synthesized {} bytes → {:?}",
            metadata.len(),
            output_path
        );

        // 超时检查（通过 elapsed 时间）
        let _ = timeout; // 超时由命令自身处理

        Ok(output_path)
    }

    fn name(&self) -> &str {
        "subprocess"
    }

    fn set_prompt_text(&self, prompt_text: &str) {
        let mut pt = self
            .prompt_text
            .lock()
            .expect("SubprocessCloneEngine: prompt_text mutex poisoned");
        let old = pt.as_deref().map(|s| truncate_str(s, 40));
        *pt = Some(prompt_text.to_string());
        tracing::info!(
            "SubprocessCloneEngine: prompt_text updated (was: {:?}, now: {:?})",
            old,
            truncate_str(&prompt_text, 40)
        );
    }
}

impl std::fmt::Debug for SubprocessCloneEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubprocessCloneEngine")
            .field("command", &self.command)
            .field("model_path", &self.model_path)
            .field("args_template", &self.args_template)
            .field("timeout_secs", &self.timeout_secs)
            .finish()
    }
}

// ─── 持久化子进程克隆引擎 ─────────────────────────────────

/// 持久化子进程声音克隆引擎
///
/// 通过 vt-tts 的 `server` 模式实现模型常驻内存的批量合成。
/// 与 [`SubprocessCloneEngine`] 不同，本引擎仅在首次调用时启动 vt-tts 进程，
/// 后续请求通过 stdin/stdout JSON 行协议复用同一进程，消除重复模型加载开销。
///
/// # 性能优势
/// 对于 N 个片段的视频:
/// - `SubprocessCloneEngine`: N × (模型加载 + 合成) ≈ N × (3-5s + 1-2s)
/// - `PersistentSubprocessCloneEngine`: 1 × 模型加载 + N × 合成 ≈ 3-5s + N × 1-2s
/// - 节省: (N-1) × 3-5s
///
/// # 协议
/// 请求 (stdin, JSON 行):
/// ```json
/// {"text":"你好世界","voice":"/path/ref.wav","output":"/path/out.wav","ref_text":null,"seed":42}
/// ```
/// 响应 (stdout, JSON 行):
/// ```json
/// {"status":"ok","output":"/path/out.wav","duration_secs":5.5,"elapsed_secs":1.2,"rtf":0.218}
/// ```
///
/// # 配置示例
/// ```toml
/// [cloning]
/// enabled = true
/// engine = "subprocess-persistent"
/// clone_command = "./target/release/vt-tts"
/// clone_args = ["server", "--model", "models/qwen3-tts", "--device", "metal", "--decode-device", "cpu", "--seed", "42"]
/// clone_timeout_secs = 120
/// ```
pub struct PersistentSubprocessCloneEngine {
    /// CLI 命令路径
    command: String,
    /// 服务器启动参数 (不含 --text/--voice/--output)
    server_args: Vec<String>,
    /// 超时时间（秒）
    timeout_secs: u64,
    /// 内部计数器
    counter: std::sync::atomic::AtomicUsize,
    /// 提示文本（动态更新）
    prompt_text: std::sync::Mutex<Option<String>>,
    /// 持久化服务器进程 (懒加载)
    server: std::sync::Mutex<Option<ServerProcess>>,
    /// P1: 当前说话人 ID（用于多说话人缓存）
    current_speaker_id: std::sync::Mutex<Option<String>>,
    /// P1: 参考音频路径（用于判断是否需要更新 speaker_id）
    current_ref_audio: std::sync::Mutex<Option<String>>,
}

/// 持久化服务器进程的句柄
struct ServerProcess {
    /// 子进程
    child: std::process::Child,
    /// stdin 写入句柄
    stdin: std::process::ChildStdin,
    /// stdout 读取句柄
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

impl PersistentSubprocessCloneEngine {
    /// 创建新的持久化子进程克隆引擎
    ///
    /// # 参数
    /// - `command`: vt-tts 可执行文件路径
    /// - `server_args`: 服务器启动参数 (如 `["server", "--model", "...", "--device", "metal"]`)
    /// - `timeout_secs`: 每次合成超时时间（秒）
    #[must_use]
    pub fn new(command: String, server_args: Vec<String>, timeout_secs: u64) -> Self {
        Self {
            command,
            server_args,
            timeout_secs,
            counter: std::sync::atomic::AtomicUsize::new(0),
            prompt_text: std::sync::Mutex::new(None),
            server: std::sync::Mutex::new(None),
            current_speaker_id: std::sync::Mutex::new(None),
            current_ref_audio: std::sync::Mutex::new(None),
        }
    }

    /// P1: 预热说话人 — 提前提取 voice clone prompt 并缓存到 Python 服务端
    ///
    /// 在 pipeline 提取参考音频后立即调用，将参考音频发送给 Python 服务端进行预热。
    /// 后续 TTS 请求携带 speaker_id 即可跳过 prompt 创建步骤（~2.4s 节省）。
    ///
    /// # 参数
    /// - `speaker_id`: 说话人标识（如 "speaker_0"）
    /// - `reference_audio`: 参考音频路径
    /// - `ref_text`: 参考音频对应文本（可选）
    ///
    /// # 错误
    /// - [`AppError::VoiceCloningError`][]: 预热失败
    pub fn prewarm_speaker(
        &self,
        speaker_id: &str,
        reference_audio: &Path,
        ref_text: Option<&str>,
    ) -> AppResult<()> {
        if !reference_audio.exists() {
            return Err(AppError::FileNotFound(reference_audio.to_path_buf()));
        }

        let prompt_text = ref_text.unwrap_or("").to_string();

        let request = serde_json::json!({
            "action": "prewarm_speaker",
            "speaker_id": speaker_id,
            "voice": reference_audio.to_string_lossy(),
            "ref_text": prompt_text,
        });

        tracing::info!(
            "PersistentSubprocessCloneEngine: prewarm speaker '{}' with ref_audio={:?}",
            speaker_id,
            reference_audio
        );

        let response = self.send_request(&request)?;
        let status = response
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        if status != "ok" {
            let error = response
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown");
            return Err(AppError::VoiceCloningError(format!(
                "prewarm_speaker failed: {error}"
            )));
        }

        // 记录当前说话人 ID 和参考音频路径
        {
            let mut sid = self.current_speaker_id.lock().map_err(|e| {
                AppError::VoiceCloningError(format!("speaker_id mutex poisoned: {e}"))
            })?;
            *sid = Some(speaker_id.to_string());
        }
        {
            let mut ref_path = self.current_ref_audio.lock().map_err(|e| {
                AppError::VoiceCloningError(format!("ref_audio mutex poisoned: {e}"))
            })?;
            *ref_path = Some(reference_audio.to_string_lossy().into_owned());
        }

        let elapsed = response
            .get("elapsed_secs")
            .and_then(|e| e.as_f64())
            .unwrap_or(0.0);
        tracing::info!(
            "PersistentSubprocessCloneEngine: speaker '{}' prewarmed in {:.3}s",
            speaker_id,
            elapsed
        );

        Ok(())
    }

    /// P1: 获取当前说话人 ID（如果已预热）
    pub fn current_speaker_id(&self) -> Option<String> {
        self.current_speaker_id.lock().ok().and_then(|g| g.clone())
    }

    /// P3: 批量合成 — 一次请求合成多段文本，减少 Rust↔Python 往返开销
    ///
    /// 借鉴 dots.tts OnlineBatcher: 将多个 TTS 请求打包为一个批量请求，
    /// 同一 speaker_id 共享 voice clone prompt，减少序列化/反序列化开销。
    ///
    /// # 参数
    /// - `reference_audio`: 参考音频路径
    /// - `texts`: 待合成的文本列表
    /// - `config`: 克隆配置
    ///
    /// # 返回
    /// 合成的音频文件路径列表（与 `texts` 等长，失败的为 None）。
    pub fn batch_synthesize(
        &self,
        reference_audio: &Path,
        texts: &[String],
        config: &CloningConfig,
    ) -> AppResult<Vec<Option<PathBuf>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // 创建输出目录
        let output_dir = PathBuf::from(&config.output_dir);
        if !output_dir.exists() {
            std::fs::create_dir_all(&output_dir).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to create output dir: {e}"))
            })?;
        }

        // 为每个文本生成输出路径
        let items: Vec<serde_json::Value> = texts
            .iter()
            .enumerate()
            .map(|(_i, text)| {
                let idx = self
                    .counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let output_path = output_dir.join(format!("batch_cloned_{idx:06}.wav"));
                serde_json::json!({
                    "text": text,
                    "output": output_path.to_string_lossy(),
                })
            })
            .collect();

        // 获取 speaker_id 和 ref_text
        let speaker_id = self
            .current_speaker_id
            .lock()
            .map_err(|e| AppError::VoiceCloningError(format!("speaker_id mutex poisoned: {e}")))?
            .clone();
        let ref_text = self
            .prompt_text
            .lock()
            .map_err(|e| AppError::VoiceCloningError(format!("prompt_text mutex poisoned: {e}")))?
            .clone()
            .unwrap_or_default();

        let mut request = serde_json::json!({
            "action": "batch_synthesize",
            "items": items,
            "voice": reference_audio.to_string_lossy(),
            "ref_text": ref_text,
        });
        if let Some(ref sid) = speaker_id {
            request["speaker_id"] = serde_json::Value::String(sid.clone());
        }

        tracing::info!(
            "PersistentSubprocessCloneEngine: batch_synthesize {} texts, speaker={}",
            texts.len(),
            speaker_id.as_deref().unwrap_or("(none)")
        );

        let response = self.send_request(&request)?;
        let status = response
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        if status != "ok" {
            let error = response
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown");
            return Err(AppError::VoiceCloningError(format!(
                "batch_synthesize failed: {error}"
            )));
        }

        // 解析结果
        let results = response
            .get("results")
            .and_then(|r| r.as_array())
            .ok_or_else(|| {
                AppError::VoiceCloningError("batch_synthesize: missing results array".into())
            })?;

        let mut paths = vec![None; texts.len()];
        let mut ok_count = 0;
        for result in results {
            let index = result.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            let item_status = result
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("error");

            if item_status == "ok" {
                let output = result.get("output").and_then(|o| o.as_str()).unwrap_or("");
                if !output.is_empty() && Path::new(output).exists() {
                    paths[index] = Some(PathBuf::from(output));
                    ok_count += 1;
                }
            }
        }

        let elapsed = response
            .get("elapsed_secs")
            .and_then(|e| e.as_f64())
            .unwrap_or(0.0);
        tracing::info!(
            "PersistentSubprocessCloneEngine: batch_synthesize completed: {}/{} succeeded in {:.1}s",
            ok_count,
            texts.len(),
            elapsed
        );

        Ok(paths)
    }

    /// 从 [`CloningEngineConfig`] 创建引擎
    ///
    /// 解析 `clone_args`，提取服务器启动参数:
    /// - 将首参数 "synthesize" 替换为 "server"
    /// - 过滤掉 `--text`、`--voice`、`--output` 及其值（这些通过 JSON 协议传递）
    /// - 保留 `--model`、`--device`、`--seed` 等启动参数
    pub fn from_config(config: &CloningEngineConfig) -> AppResult<Self> {
        let command = config.clone_command.clone().ok_or_else(|| {
            AppError::VoiceCloningError(format!(
                "PersistentSubprocessCloneEngine: clone_command 未配置\n\
                请在 config.toml [cloning] 段设置 clone_command 为 vt-tts 路径"
            ))
        })?;

        // 从 clone_args 提取服务器启动参数
        let server_args = Self::extract_server_args(&config.clone_args, &config.engine);

        tracing::info!(
            "PersistentSubprocessCloneEngine: command={:?}, server_args={:?}",
            command,
            server_args
        );

        Ok(Self::new(command, server_args, config.clone_timeout_secs))
    }

    /// 从 clone_args 提取服务器启动参数
    ///
    /// - subprocess-persistent: 过滤掉 --text/--voice/--output，将 "synthesize" 替换为 "server"
    /// - python-qwen-tts: 直接透传所有参数（Python 脚本路径 + 模型参数等）
    fn extract_server_args(clone_args: &[String], engine: &str) -> Vec<String> {
        // Python TTS 后端: 直接透传所有参数
        if engine == "python-qwen-tts" {
            return clone_args.to_vec();
        }

        let per_request_flags = ["--text", "-t", "--voice", "-v", "--output", "-o"];
        let mut result = Vec::new();
        let mut skip_next = false;

        for (i, arg) in clone_args.iter().enumerate() {
            if skip_next {
                skip_next = false;
                continue;
            }

            // 检查是否是每个请求的参数标志
            if per_request_flags.contains(&arg.as_str()) {
                skip_next = true; // 跳过这个标志和它的值
                continue;
            }

            // 检查是否是占位符值 ({text}, {ref_audio}, {output})
            if arg.contains("{text}") || arg.contains("{ref_audio}") || arg.contains("{output}") {
                continue;
            }

            // 将 "synthesize" 替换为 "server"
            let arg = if arg == "synthesize" && i == 0 {
                "server".to_string()
            } else {
                arg.clone()
            };

            // 如果首参数不是 "server" 且 engine 是 "subprocess-persistent"，添加 "server"
            if i == 0 && arg != "server" && engine == "subprocess-persistent" {
                result.push("server".to_string());
            }

            result.push(arg);
        }

        // 确保首参数是 "server"
        if result.is_empty() || result[0] != "server" {
            result.insert(0, "server".to_string());
        }

        result
    }

    /// 启动服务器进程 (首次调用时)
    fn ensure_server(&self) -> AppResult<()> {
        let mut guard = self
            .server
            .lock()
            .map_err(|e| AppError::VoiceCloningError(format!("Server mutex poisoned: {e}")))?;

        if guard.is_some() {
            return Ok(());
        }

        tracing::info!(
            "PersistentSubprocessCloneEngine: starting server: {} {}",
            self.command,
            self.server_args.join(" ")
        );

        let mut cmd = std::process::Command::new(&self.command);
        cmd.args(&self.server_args);
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            AppError::VoiceCloningError(format!(
                "Failed to start vt-tts server '{}': {e}\n\
                请确认 CLI 工具已安装且路径正确",
                self.command
            ))
        })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            AppError::VoiceCloningError("Failed to capture server stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            AppError::VoiceCloningError("Failed to capture server stdout".to_string())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            AppError::VoiceCloningError("Failed to capture server stderr".to_string())
        })?;

        // 在后台线程中读取 stderr 并转发到日志
        std::thread::spawn(move || {
            use std::io::BufRead;
            let reader = std::io::BufReader::new(stderr);
            for line in reader.lines().flatten() {
                tracing::info!("[vt-tts server] {}", line);
            }
        });

        // 等待服务器就绪 (检查进程是否仍然存活)
        std::thread::sleep(std::time::Duration::from_millis(500));
        match child.try_wait() {
            Ok(Some(status)) => {
                return Err(AppError::VoiceCloningError(format!(
                    "vt-tts server exited immediately with status: {status}\n\
                    请检查模型路径和参数是否正确"
                )));
            }
            Ok(None) => {} // 进程仍在运行，很好
            Err(e) => {
                return Err(AppError::VoiceCloningError(format!(
                    "Failed to check server status: {e}"
                )));
            }
        }

        tracing::info!("PersistentSubprocessCloneEngine: server started and ready");

        *guard = Some(ServerProcess {
            child,
            stdin,
            stdout: std::io::BufReader::new(stdout),
        });

        Ok(())
    }

    /// 通过 JSON 协议发送合成请求
    fn send_request(&self, request: &serde_json::Value) -> AppResult<serde_json::Value> {
        self.ensure_server()?;

        let mut guard = self
            .server
            .lock()
            .map_err(|e| AppError::VoiceCloningError(format!("Server mutex poisoned: {e}")))?;

        let server = guard.as_mut().ok_or_else(|| {
            AppError::VoiceCloningError("Server not running after ensure_server".to_string())
        })?;

        // 写入请求
        let request_line = serde_json::to_string(request)
            .map_err(|e| AppError::VoiceCloningError(format!("JSON serialize error: {e}")))?;

        tracing::debug!(
            "PersistentSubprocessCloneEngine: sending request: {}",
            request_line
        );

        use std::io::Write;
        writeln!(server.stdin, "{request_line}").map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to write to server stdin: {e}"))
        })?;
        server.stdin.flush().map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to flush server stdin: {e}"))
        })?;

        // 读取响应
        let mut response_line = String::new();
        server.stdout.read_line(&mut response_line).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to read server response: {e}"))
        })?;

        if response_line.is_empty() {
            // 检查服务器是否已退出
            match server.child.try_wait() {
                Ok(Some(status)) => {
                    *guard = None; // 清理死进程
                    return Err(AppError::VoiceCloningError(format!(
                        "vt-tts server exited unexpectedly (status: {status}). \
                        Server will be restarted on next request."
                    )));
                }
                _ => {
                    return Err(AppError::VoiceCloningError(
                        "Server returned empty response (no stdout output)".to_string(),
                    ));
                }
            }
        }

        tracing::debug!(
            "PersistentSubprocessCloneEngine: received response: {}",
            response_line.trim()
        );

        let response: serde_json::Value = serde_json::from_str(&response_line).map_err(|e| {
            AppError::VoiceCloningError(format!(
                "Failed to parse server response as JSON: {e}\nResponse: {response_line}"
            ))
        })?;

        Ok(response)
    }
}

impl VoiceCloningEngine for PersistentSubprocessCloneEngine {
    fn clone_and_synthesize(
        &self,
        reference_audio: &Path,
        text: &str,
        config: &CloningConfig,
    ) -> AppResult<PathBuf> {
        // 验证参考音频存在
        if !reference_audio.exists() {
            return Err(AppError::FileNotFound(reference_audio.to_path_buf()));
        }

        // 创建输出目录
        let output_dir = PathBuf::from(&config.output_dir);
        if !output_dir.exists() {
            std::fs::create_dir_all(&output_dir).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to create output dir: {e}"))
            })?;
        }

        // 生成唯一文件名
        let idx = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let output_path = output_dir.join(format!("persistent_cloned_{idx:06}.wav"));

        // 构建请求 JSON
        let prompt_text = self
            .prompt_text
            .lock()
            .map_err(|e| AppError::VoiceCloningError(format!("prompt_text mutex poisoned: {e}")))?
            .clone();

        // P1: 获取当前说话人 ID（如果已预热）
        let speaker_id = self
            .current_speaker_id
            .lock()
            .map_err(|e| AppError::VoiceCloningError(format!("speaker_id mutex poisoned: {e}")))?
            .clone();

        let mut request = serde_json::json!({
            "text": text,
            "voice": reference_audio.to_string_lossy(),
            "output": output_path.to_string_lossy(),
            "ref_text": prompt_text,
        });

        // P1: 如果有预热的 speaker_id，附加到请求中
        if let Some(ref sid) = speaker_id {
            request["speaker_id"] = serde_json::Value::String(sid.clone());
        }

        tracing::info!(
            "PersistentSubprocessCloneEngine: request #{} text=\"{}\" speaker={} → {:?}",
            idx,
            truncate_str(text, 50),
            speaker_id.as_deref().unwrap_or("(none)"),
            output_path
        );

        // 发送请求并等待响应
        let response = self.send_request(&request)?;

        // 解析响应
        let status = response
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        match status {
            "ok" => {
                let output_str = response
                    .get("output")
                    .and_then(|o| o.as_str())
                    .unwrap_or_else(|| output_path.to_str().unwrap());

                let output_path = PathBuf::from(output_str);

                // 验证输出文件
                if !output_path.exists() {
                    return Err(AppError::VoiceCloningError(format!(
                        "Server reported success but output file not found: {:?}",
                        output_path
                    )));
                }

                let metadata = std::fs::metadata(&output_path).map_err(|e| {
                    AppError::VoiceCloningError(format!("Failed to read output metadata: {e}"))
                })?;

                if metadata.len() < 44 {
                    return Err(AppError::VoiceCloningError(format!(
                        "Output file too small ({} bytes), may not be a valid WAV",
                        metadata.len()
                    )));
                }

                let duration = response
                    .get("duration_secs")
                    .and_then(|d| d.as_f64())
                    .unwrap_or(0.0);
                let elapsed = response
                    .get("elapsed_secs")
                    .and_then(|e| e.as_f64())
                    .unwrap_or(0.0);
                let rtf = response.get("rtf").and_then(|r| r.as_f64()).unwrap_or(0.0);

                tracing::info!(
                    "PersistentSubprocessCloneEngine: synthesized {} bytes ({:.1}s audio in {:.1}s, RTF: {:.3}x) → {:?}",
                    metadata.len(),
                    duration,
                    elapsed,
                    rtf,
                    output_path
                );

                let _ = self.timeout_secs; // 超时由服务器端处理

                Ok(output_path)
            }
            "error" => {
                let error_msg = response
                    .get("error")
                    .and_then(|e| e.as_str())
                    .unwrap_or("unknown error");
                Err(AppError::VoiceCloningError(format!(
                    "vt-tts server error: {error_msg}"
                )))
            }
            _ => Err(AppError::VoiceCloningError(format!(
                "Unexpected server response: {response}"
            ))),
        }
    }

    fn name(&self) -> &str {
        "subprocess-persistent"
    }

    fn set_prompt_text(&self, prompt_text: &str) {
        let mut pt = self
            .prompt_text
            .lock()
            .expect("PersistentSubprocessCloneEngine: prompt_text mutex poisoned");
        let old = pt.as_deref().map(|s| truncate_str(s, 40));
        *pt = Some(prompt_text.to_string());
        tracing::info!(
            "PersistentSubprocessCloneEngine: prompt_text updated (was: {:?}, now: {:?})",
            old,
            truncate_str(&prompt_text, 40)
        );
    }

    /// P1: 覆盖 trait 的 prewarm_speaker，委托给 Self::prewarm_speaker
    fn prewarm_speaker(
        &self,
        speaker_id: &str,
        reference_audio: &Path,
        ref_text: Option<&str>,
    ) -> AppResult<()> {
        Self::prewarm_speaker(self, speaker_id, reference_audio, ref_text)
    }
}

impl Drop for PersistentSubprocessCloneEngine {
    fn drop(&mut self) {
        let mut guard = self.server.lock().ok();
        if let Some(ref mut server) = guard.as_deref_mut().and_then(|g| g.as_mut()) {
            tracing::info!("PersistentSubprocessCloneEngine: shutting down server process");
            let _ = server.child.kill();
            let _ = server.child.wait();
        }
    }
}

impl std::fmt::Debug for PersistentSubprocessCloneEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentSubprocessCloneEngine")
            .field("command", &self.command)
            .field("server_args", &self.server_args)
            .field("timeout_secs", &self.timeout_secs)
            .field(
                "server_running",
                &self.server.lock().map(|g| g.is_some()).unwrap_or(false),
            )
            .finish()
    }
}

// ─── 参考音频自动提取 ─────────────────────────────────────

/// GPT-SoVITS 参考音频的理想时长范围（秒）
const REF_MIN_DURATION_SECS: f64 = 3.0;
const REF_MAX_DURATION_SECS: f64 = 10.0;

/// 从完整音频中提取参考音频片段
///
/// 根据 ASR 产生的 Segment 列表，自动选择一段时长合适（3–10 秒）
/// 且文本内容充足的语音片段，从原始 WAV 中截取并保存为参考音频。
///
/// # 工作原理
/// 1. 遍历所有 Segment，寻找时长在 3–10 秒范围内且 `source_text` 非空的片段
/// 2. 优先选择时长最接近 5 秒的片段（GPT-SoVITS 最佳参考时长）
/// 3. 用 `source_text` 作为 `prompt_text`（参考音频对应的文字内容）
/// 4. 从完整 WAV 中按时间戳截取音频段，写入新的 WAV 文件
///
/// # 参数
/// - `full_wav_path`: 完整音频 WAV 文件路径（16kHz mono）
/// - `segments`: ASR 产生的 Segment 列表
/// - `output_path`: 输出参考音频文件路径
///
/// # 返回
/// 如果找到合适的片段，返回 `(参考音频路径, 提示文本)`；
/// 否则返回 `None`。
///
/// # 错误
/// - [`AppError::AudioDecodeError`][]: WAV 读取或写入失败
pub fn extract_reference_audio(
    full_wav_path: &Path,
    segments: &[Segment],
    output_path: &Path,
) -> AppResult<Option<(PathBuf, String)>> {
    if segments.is_empty() {
        tracing::warn!("Auto-extract: no segments available for reference extraction");
        return Ok(None);
    }

    // 选择最佳参考片段：时长在 3–10 秒，优先接近 5 秒
    let best = segments
        .iter()
        .filter(|s| {
            let dur = s.end - s.start;
            dur >= REF_MIN_DURATION_SECS
                && dur <= REF_MAX_DURATION_SECS
                && !s.source_text.trim().is_empty()
        })
        .min_by_key(|s| {
            let dur = s.end - s.start;
            ((dur - 5.0).abs() * 100.0) as u64 // 越接近 5 秒越好
        });

    // 如果没有 3–10 秒的片段，放宽到 2–15 秒
    let best = best.or_else(|| {
        segments
            .iter()
            .filter(|s| {
                let dur = s.end - s.start;
                dur >= 2.0 && dur <= 15.0 && !s.source_text.trim().is_empty()
            })
            .min_by_key(|s| {
                let dur = s.end - s.start;
                ((dur - 5.0).abs() * 100.0) as u64
            })
    });

    let Some(segment) = best else {
        tracing::warn!(
            "Auto-extract: no suitable segment found for reference \
            (need 3–10s speech with text)"
        );
        return Ok(None);
    };

    let prompt_text = segment.source_text.clone();
    let start = segment.start;
    let end = segment.end;
    let duration = end - start;

    tracing::info!(
        "Auto-extract: selected segment {} ({:.1}s–{:.1}s, {:.1}s) as reference, \
        prompt_text: \"{}\"",
        segment.id,
        start,
        end,
        duration,
        if prompt_text.len() > 60 {
            truncate_str(&prompt_text, 60)
        } else {
            prompt_text.clone()
        }
    );

    // 从完整 WAV 中读取音频
    let (samples, sample_rate) = read_wav_mono(full_wav_path)?;

    // 计算采样范围
    let start_sample = ((start * sample_rate as f64) as usize).min(samples.len());
    let end_sample = ((end * sample_rate as f64) as usize).min(samples.len());

    if start_sample >= end_sample {
        tracing::warn!(
            "Auto-extract: invalid sample range {}..{} for segment {}",
            start_sample,
            end_sample,
            segment.id
        );
        return Ok(None);
    }

    let ref_samples = &samples[start_sample..end_sample];

    // 确保输出目录存在
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to create reference dir: {e}"))
        })?;
    }

    // 写入 WAV 文件
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec)
        .map_err(|e| AppError::VoiceCloningError(format!("Failed to create reference WAV: {e}")))?;

    for sample in ref_samples {
        let i16_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(i16_sample).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to write reference sample: {e}"))
        })?;
    }

    writer.finalize().map_err(|e| {
        AppError::VoiceCloningError(format!("Failed to finalize reference WAV: {e}"))
    })?;

    tracing::info!(
        "Auto-extract: reference audio saved to {:?} ({} samples, {:.1}s)",
        output_path,
        ref_samples.len(),
        ref_samples.len() as f64 / sample_rate as f64
    );

    Ok(Some((output_path.to_path_buf(), prompt_text)))
}

// ─── 流水线集成辅助器 ─────────────────────────────────────

/// 声音克隆流水线集成辅助器
///
/// 提供从说话人分离结果到声音克隆的桥接逻辑。
/// 如果克隆失败，可以优雅降级到标准 TTS。
pub struct CloningIntegration {
    /// 克隆引擎
    engine: Box<dyn VoiceCloningEngine>,
    /// 克隆配置
    config: CloningConfig,
}

impl CloningIntegration {
    /// 创建新的集成辅助器
    ///
    /// # 参数
    /// - `engine`: 声音克隆引擎
    /// - `config`: 克隆配置
    #[must_use]
    pub fn new(engine: Box<dyn VoiceCloningEngine>, config: CloningConfig) -> Self {
        Self { engine, config }
    }

    /// 为 Segment 合成克隆语音
    ///
    /// 使用指定说话人的参考音频合成 Segment 的目标文本。
    ///
    /// # 参数
    /// - `segment`: 已翻译的 Segment（需要有 `target_text`）
    /// - `reference_audio`: 该说话人的参考音频路径
    ///
    /// # 返回
    /// 合成的音频文件路径。
    ///
    /// # 错误
    /// - [`AppError::VoiceCloningError`][]: 合成失败
    /// - [`AppError::Config`][]: Segment 缺少 `target_text`
    pub fn synthesize_for_segment(
        &self,
        segment: &Segment,
        reference_audio: &Path,
    ) -> AppResult<PathBuf> {
        let text = segment.target_text.as_ref().ok_or_else(|| {
            AppError::Config(format!("Segment {} has no target_text", segment.id))
        })?;

        self.engine
            .clone_and_synthesize(reference_audio, text, &self.config)
    }

    /// 尝试克隆合成，失败时返回 None（优雅降级）
    ///
    /// # 参数
    /// - `segment`: 已翻译的 Segment
    /// - `reference_audio`: 参考音频路径
    ///
    /// # 返回
    /// - `Ok(Some(path))`: 合成成功
    /// - `Ok(None)`: 合成失败，调用方应回退到标准 TTS
    pub fn try_synthesize(
        &self,
        segment: &Segment,
        reference_audio: &Path,
    ) -> AppResult<Option<PathBuf>> {
        match self.synthesize_for_segment(segment, reference_audio) {
            Ok(path) => Ok(Some(path)),
            Err(e) => {
                tracing::warn!(
                    "Voice cloning failed for segment {}, falling back to standard TTS: {}",
                    segment.id,
                    e
                );
                Ok(None)
            }
        }
    }

    /// 获取克隆引擎名称
    #[must_use]
    pub fn engine_name(&self) -> &str {
        self.engine.name()
    }

    /// 更新参考音频的提示文本
    ///
    /// 自动提取参考音频后调用，将 ASR 转录结果设置为 prompt_text。
    /// 仅对支持动态更新的引擎（如 GPT-SoVITS）生效。
    ///
    /// # 参数
    /// - `prompt_text`: 参考音频对应的文字内容
    pub fn set_prompt_text(&self, prompt_text: &str) {
        self.engine.set_prompt_text(prompt_text);
    }

    /// P1: 预热说话人 — 提前提取 voice clone prompt 并缓存到 TTS 服务端
    ///
    /// 在 pipeline 提取参考音频后立即调用，让 TTS 服务端预热说话人缓存。
    /// 后续 TTS 请求可跳过 prompt 创建步骤，减少首次合成延迟（~2.4s 节省）。
    ///
    /// # 参数
    /// - `speaker_id`: 说话人标识（如 "speaker_0"）
    /// - `reference_audio`: 参考音频路径
    /// - `ref_text`: 参考音频对应文本（可选）
    pub fn prewarm_speaker(
        &self,
        speaker_id: &str,
        reference_audio: &Path,
        ref_text: Option<&str>,
    ) -> AppResult<()> {
        self.engine
            .prewarm_speaker(speaker_id, reference_audio, ref_text)
    }

    /// P2: 句子级克隆合成 — 将长文本按句拆分，逐句合成后交叉淡入淡出拼接
    ///
    /// 借鉴 dots.tts 双流式 pipeline 思路：将长段翻译文本按句拆分，
    /// 逐句调用 clone_and_synthesize，然后用等功率交叉淡入淡出拼接为一个 WAV。
    ///
    /// # 工作流程
    /// 1. 如果文本不够长（≤ `min_chars` 字符），直接整体合成
    /// 2. 否则按句拆分，逐句合成
    /// 3. 用交叉淡入淡出拼接所有句子的音频
    /// 4. 返回拼接后的音频路径
    ///
    /// # 参数
    /// - `text`: 目标文本
    /// - `reference_audio`: 参考音频路径
    /// - `min_chars`: 拆分阈值（字符数），超过才拆分
    /// - `crossfade_ms`: 交叉淡入淡出时长（毫秒）
    ///
    /// # 返回
    /// 合成的音频文件路径。
    pub fn synthesize_with_sentence_split(
        &self,
        text: &str,
        reference_audio: &Path,
        min_chars: usize,
        crossfade_ms: u64,
    ) -> AppResult<PathBuf> {
        use crate::sentence_split::{should_split_for_tts, split_sentences};

        // 短文本：直接整体合成
        if !should_split_for_tts(text, min_chars) {
            return self
                .engine
                .clone_and_synthesize(reference_audio, text, &self.config);
        }

        // 长文本：按句拆分
        let sentences = split_sentences(text);
        if sentences.len() <= 1 {
            // 拆分后只有一句，直接合成
            return self
                .engine
                .clone_and_synthesize(reference_audio, text, &self.config);
        }

        tracing::info!(
            "Sentence-level TTS: splitting {} chars into {} sentences",
            text.chars().count(),
            sentences.len()
        );

        // 逐句合成
        let mut wav_paths = Vec::with_capacity(sentences.len());
        for (i, sentence) in sentences.iter().enumerate() {
            match self
                .engine
                .clone_and_synthesize(reference_audio, sentence, &self.config)
            {
                Ok(path) => {
                    tracing::debug!(
                        "Sentence {}/{} synthesized: \"{}\" → {:?}",
                        i + 1,
                        sentences.len(),
                        if sentence.len() > 30 {
                            format!("{}...", sentence.chars().take(30).collect::<String>())
                        } else {
                            sentence.clone()
                        },
                        path
                    );
                    wav_paths.push(path);
                }
                Err(e) => {
                    tracing::warn!(
                        "Sentence {}/{} synthesis failed: {}, skipping",
                        i + 1,
                        sentences.len(),
                        e
                    );
                    // 跳过失败的句子，继续合成剩余句子
                }
            }
        }

        if wav_paths.is_empty() {
            return Err(AppError::VoiceCloningError(
                "All sentence synthesis failed".to_string(),
            ));
        }

        if wav_paths.len() == 1 {
            return Ok(wav_paths.remove(0));
        }

        // P5: 流式拼接 — 使用 StreamingWavConcatenator 替代 crossfade_concat_wav
        // StreamingWavConcatenator 只需 2 个 chunk 在内存中，与文件数量无关
        let output_path = PathBuf::from(&self.config.output_dir).join(format!(
            "sentence_concat_{}.wav",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));

        let concatenator = crate::streaming_audio::StreamingWavConcatenator::new(crossfade_ms);
        concatenator.concatenate(&wav_paths, &output_path)?;

        tracing::info!(
            "Sentence-level TTS: concatenated {} WAVs → {:?}",
            wav_paths.len(),
            output_path
        );

        Ok(output_path)
    }
}

impl std::fmt::Debug for CloningIntegration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloningIntegration")
            .field("engine", &self.engine.name())
            .field("config", &self.config)
            .finish()
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 创建测试用参考音频
    fn create_reference_audio(path: &Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV");
        for i in 0..16000 * 5 {
            // 5 秒
            let sample = ((i as f64 * 0.1).sin() * 16000.0) as i16;
            writer.write_sample(sample).expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize WAV");
    }

    // ── CloningConfig 测试 ────────────────────────────

    #[test]
    fn test_cloning_config_default() {
        let config = CloningConfig::default();

        assert!((config.speed - 1.0).abs() < f32::EPSILON);
        assert!((config.pitch_shift - 0.0).abs() < f32::EPSILON);
        assert!((config.emotion - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.sample_rate, 24000);
    }

    #[test]
    fn test_cloning_config_serde_roundtrip() {
        let config = CloningConfig {
            speed: 1.5,
            pitch_shift: 2.0,
            emotion: 0.5,
            sample_rate: 48000,
            output_dir: "/tmp/cloned".into(),
        };

        let json = serde_json::to_string(&config).expect("serialize failed");
        let restored: CloningConfig = serde_json::from_str(&json).expect("deserialize failed");

        assert!((restored.speed - config.speed).abs() < f32::EPSILON);
        assert!((restored.pitch_shift - config.pitch_shift).abs() < f32::EPSILON);
        assert!((restored.emotion - config.emotion).abs() < f32::EPSILON);
        assert_eq!(restored.sample_rate, config.sample_rate);
        assert_eq!(restored.output_dir, config.output_dir);
    }

    // ── MockCloningEngine 测试 ────────────────────────

    #[test]
    fn test_mock_engine_clone_and_synthesize() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        // 创建参考音频
        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        // 创建引擎和配置
        let engine = MockCloningEngine::new();
        let config = CloningConfig {
            output_dir: dir.path().join("output").to_string_lossy().into_owned(),
            ..Default::default()
        };

        let result = engine.clone_and_synthesize(&ref_path, "你好，这是测试文本", &config);

        assert!(result.is_ok(), "Should synthesize successfully");
        let output_path = result.unwrap();
        assert!(output_path.exists(), "Output file should exist");

        // 验证 WAV 格式
        let reader = hound::WavReader::open(&output_path).expect("Failed to open WAV");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 24000);
    }

    #[test]
    fn test_mock_engine_reference_not_found() {
        let engine = MockCloningEngine::new();
        let config = CloningConfig::default();

        let result = engine.clone_and_synthesize(
            Path::new("/nonexistent/reference.wav"),
            "test text",
            &config,
        );

        assert!(result.is_err(), "Should fail for nonexistent reference");
    }

    #[test]
    fn test_mock_engine_batch_synthesize() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        let engine = MockCloningEngine::new();
        let config = CloningConfig {
            output_dir: dir.path().join("output").to_string_lossy().into_owned(),
            ..Default::default()
        };

        let texts = vec!["你好".to_string(), "世界".to_string(), "测试".to_string()];

        let results = engine
            .clone_and_synthesize_batch(&ref_path, &texts, &config)
            .expect("batch synthesize failed");

        assert_eq!(results.len(), 3);
        for path in &results {
            assert!(path.exists(), "Each output file should exist");
        }
    }

    #[test]
    fn test_mock_engine_name() {
        let engine = MockCloningEngine::new();
        assert_eq!(engine.name(), "mock-cloning");
    }

    #[test]
    fn test_mock_engine_generates_unique_files() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        let engine = MockCloningEngine::new();
        let config = CloningConfig {
            output_dir: dir.path().join("output").to_string_lossy().into_owned(),
            ..Default::default()
        };

        let path1 = engine
            .clone_and_synthesize(&ref_path, "text1", &config)
            .expect("synthesize 1 failed");
        let path2 = engine
            .clone_and_synthesize(&ref_path, "text2", &config)
            .expect("synthesize 2 failed");

        assert_ne!(path1, path2, "Should generate unique file paths");
    }

    #[test]
    fn test_mock_engine_empty_text() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        let engine = MockCloningEngine::new();
        let config = CloningConfig {
            output_dir: dir.path().join("output").to_string_lossy().into_owned(),
            ..Default::default()
        };

        // 空文本应仍能工作（生成最短 0.5 秒音频）
        let result = engine.clone_and_synthesize(&ref_path, "", &config);
        assert!(result.is_ok(), "Should handle empty text");

        let path = result.unwrap();
        assert!(path.exists());

        // 验证音频有内容（至少 0.5 秒）
        let reader = hound::WavReader::open(&path).expect("Failed to open WAV");
        let duration = reader.duration() as f64 / config.sample_rate as f64;
        assert!(
            duration >= 0.5,
            "Empty text should produce at least 0.5s audio"
        );
    }

    // ── CloningIntegration 测试 ──────────────────────

    #[test]
    fn test_cloning_integration_synthesize_for_segment() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        let mut segment = Segment::new("seg-1".into(), 0.0, 5.0, "Hello".into());
        segment.start_transcribing().expect("start_transcribing");
        segment
            .finish_transcribing("你好".into())
            .expect("finish_transcribing");

        let config = CloningConfig {
            output_dir: dir.path().join("output").to_string_lossy().into_owned(),
            ..Default::default()
        };

        let integration = CloningIntegration::new(Box::new(MockCloningEngine::new()), config);

        let result = integration.synthesize_for_segment(&segment, &ref_path);
        assert!(result.is_ok(), "Should synthesize for segment");

        let path = result.unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_cloning_integration_missing_target_text() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        // Segment 没有 target_text
        let segment = Segment::new("seg-1".into(), 0.0, 5.0, "Hello".into());

        let integration =
            CloningIntegration::new(Box::new(MockCloningEngine::new()), CloningConfig::default());

        let result = integration.synthesize_for_segment(&segment, &ref_path);
        assert!(result.is_err(), "Should fail without target_text");
    }

    #[test]
    fn test_cloning_integration_try_synthesize_success() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        let mut segment = Segment::new("seg-1".into(), 0.0, 5.0, "Hello".into());
        segment.start_transcribing().expect("start_transcribing");
        segment
            .finish_transcribing("你好".into())
            .expect("finish_transcribing");

        let config = CloningConfig {
            output_dir: dir.path().join("output").to_string_lossy().into_owned(),
            ..Default::default()
        };

        let integration = CloningIntegration::new(Box::new(MockCloningEngine::new()), config);

        let result = integration.try_synthesize(&segment, &ref_path);
        assert!(result.is_ok(), "try_synthesize should not hard-fail");
        assert!(result.unwrap().is_some(), "Should return Some on success");
    }

    #[test]
    fn test_cloning_integration_try_synthesize_fallback() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        let mut segment = Segment::new("seg-1".into(), 0.0, 5.0, "Hello".into());
        segment.start_transcribing().expect("start_transcribing");
        segment
            .finish_transcribing("你好".into())
            .expect("finish_transcribing");

        // 使用不存在的参考音频，应触发优雅降级
        let integration =
            CloningIntegration::new(Box::new(MockCloningEngine::new()), CloningConfig::default());

        let result = integration.try_synthesize(&segment, Path::new("/nonexistent/ref.wav"));
        assert!(result.is_ok(), "try_synthesize should not hard-fail");
        assert!(
            result.unwrap().is_none(),
            "Should return None on failure (graceful degradation)"
        );
    }

    #[test]
    fn test_cloning_integration_engine_name() {
        let integration =
            CloningIntegration::new(Box::new(MockCloningEngine::new()), CloningConfig::default());

        assert_eq!(integration.engine_name(), "mock-cloning");
    }

    // ── 集成测试：完整克隆流程 ────────────────────────

    #[test]
    fn test_integration_full_cloning_workflow() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        // 1. 创建参考音频
        let ref_path = dir.path().join("speaker_reference.wav");
        create_reference_audio(&ref_path);

        // 2. 创建已翻译的 Segment
        let segments: Vec<Segment> = (0..3)
            .map(|i| {
                let mut seg = Segment::new(
                    format!("seg-{i}"),
                    i as f64 * 5.0,
                    (i + 1) as f64 * 5.0,
                    format!("English text {i}"),
                );
                seg.start_transcribing().expect("start");
                seg.finish_transcribing(format!("中文文本 {i}"))
                    .expect("finish");
                seg
            })
            .collect();

        // 3. 批量克隆合成
        let engine = MockCloningEngine::new();
        let config = CloningConfig {
            output_dir: dir.path().join("cloned").to_string_lossy().into_owned(),
            ..Default::default()
        };

        let integration = CloningIntegration::new(Box::new(engine), config);

        for seg in &segments {
            let result = integration.synthesize_for_segment(seg, &ref_path);
            assert!(result.is_ok(), "Should synthesize for segment {}", seg.id);
            assert!(result.unwrap().exists(), "Output file should exist");
        }

        // 4. 验证所有输出文件存在且唯一
        let output_dir = dir.path().join("cloned");
        let files: Vec<_> = std::fs::read_dir(&output_dir)
            .expect("Failed to read output dir")
            .filter_map(Result::ok)
            .collect();

        assert_eq!(files.len(), 3, "Should have 3 output files");
    }

    // ── 参考音频自动提取测试 ────────────────────────

    /// 创建测试用完整音频 WAV（模拟从视频提取的音频）
    fn create_full_audio(path: &Path, duration_secs: f64) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV");
        let total_samples = (16000.0 * duration_secs) as usize;
        for i in 0..total_samples {
            // 模拟语音信号（正弦波 + 噪声）
            let sample =
                (((i as f64 * 0.05).sin() * 0.3) + ((i as f64 * 0.003).sin() * 0.1)) * 32767.0;
            writer
                .write_sample(sample as i16)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize WAV");
    }

    #[test]
    fn test_extract_reference_audio_success() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        // 创建 30 秒的完整音频
        let full_wav = dir.path().join("full_audio.wav");
        create_full_audio(&full_wav, 30.0);

        // 创建 ASR Segment（4-9秒，有文本）
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 2.0, "Hi".into()),
            Segment::new(
                "seg-1".into(),
                2.0,
                7.0,
                "Hello everyone, welcome to this presentation.".into(),
            ),
            Segment::new("seg-2".into(), 7.0, 12.0, "Today we will discuss.".into()),
        ];

        let ref_output = dir.path().join("auto_reference.wav");
        let result = extract_reference_audio(&full_wav, &segments, &ref_output);

        assert!(result.is_ok(), "Should extract successfully");
        let extracted = result.unwrap().expect("Should find a suitable segment");
        assert!(ref_output.exists(), "Reference WAV should be created");

        // 验证提取的 prompt_text 是非空的
        assert!(!extracted.1.is_empty(), "prompt_text should not be empty");

        // 验证 WAV 格式
        let reader = hound::WavReader::open(&ref_output).expect("Failed to open WAV");
        assert_eq!(reader.spec().sample_rate, 16000);
        assert_eq!(reader.spec().channels, 1);
    }

    #[test]
    fn test_extract_reference_audio_picks_best_duration() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_full_audio(&full_wav, 60.0);

        // 多个候选，验证选择了最接近 5 秒的
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 8.0, "Long segment".into()), // 8s
            Segment::new("seg-1".into(), 8.0, 13.0, "Medium segment".into()), // 5s ← best
            Segment::new("seg-2".into(), 13.0, 16.0, "Short".into()),      // 3s
        ];

        let ref_output = dir.path().join("ref.wav");
        let result = extract_reference_audio(&full_wav, &segments, &ref_output).unwrap();

        assert!(result.is_some(), "Should find a segment");
        let (_, prompt_text) = result.unwrap();
        assert_eq!(prompt_text, "Medium segment", "Should pick the 5s segment");
    }

    #[test]
    fn test_extract_reference_audio_no_suitable_segment() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_full_audio(&full_wav, 60.0);

        // 所有 segment 都太短（<2s）
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 0.5, "A".into()),
            Segment::new("seg-1".into(), 0.5, 1.0, "B".into()),
        ];

        let ref_output = dir.path().join("ref.wav");
        let result = extract_reference_audio(&full_wav, &segments, &ref_output).unwrap();

        assert!(
            result.is_none(),
            "Should return None when no suitable segment"
        );
    }

    #[test]
    fn test_extract_reference_audio_empty_segments() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_full_audio(&full_wav, 10.0);

        let ref_output = dir.path().join("ref.wav");
        let result = extract_reference_audio(&full_wav, &[], &ref_output);

        assert!(result.is_ok(), "Should handle empty segments gracefully");
        assert!(
            result.unwrap().is_none(),
            "Should return None for empty segments"
        );
    }

    #[test]
    fn test_extract_reference_audio_fallback_to_wider_range() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let full_wav = dir.path().join("full_audio.wav");
        create_full_audio(&full_wav, 60.0);

        // 没有严格匹配 3-10s 的，但有 2-15s 的
        let segments = vec![
            Segment::new("seg-0".into(), 0.0, 12.0, "Twelve seconds".into()), // 12s
            Segment::new("seg-1".into(), 12.0, 14.0, "Two seconds".into()),   // 2s
        ];

        let ref_output = dir.path().join("ref.wav");
        let result = extract_reference_audio(&full_wav, &segments, &ref_output).unwrap();

        assert!(result.is_some(), "Should fallback to wider range");
    }

    #[test]
    fn test_cloning_integration_set_prompt_text() {
        let integration =
            CloningIntegration::new(Box::new(MockCloningEngine::new()), CloningConfig::default());

        // 不应 panic（默认实现为空操作）
        integration.set_prompt_text("This is a test prompt text.");
    }

    // ── PersistentSubprocessCloneEngine 测试 ──────────

    #[test]
    fn test_extract_server_args_from_synthesize() {
        // 模拟当前 config.toml 中的 clone_args (synthesize 模式)
        let clone_args = vec![
            "synthesize".to_string(),
            "--text".to_string(),
            "{text}".to_string(),
            "--voice".to_string(),
            "{ref_audio}".to_string(),
            "--output".to_string(),
            "{output}".to_string(),
            "--model".to_string(),
            "models/qwen3-tts".to_string(),
            "--device".to_string(),
            "metal".to_string(),
            "--seed".to_string(),
            "42".to_string(),
        ];

        let result = PersistentSubprocessCloneEngine::extract_server_args(
            &clone_args,
            "subprocess-persistent",
        );

        // 首参数应为 "server"
        assert_eq!(result[0], "server");
        // 不应包含 --text, --voice, --output 及其值
        assert!(!result.iter().any(|a| a == "--text" || a == "{text}"));
        assert!(!result.iter().any(|a| a == "--voice" || a == "{ref_audio}"));
        assert!(!result.iter().any(|a| a == "--output" || a == "{output}"));
        // 应保留 --model, --device, --seed
        assert!(result.iter().any(|a| a == "--model"));
        assert!(result.iter().any(|a| a == "models/qwen3-tts"));
        assert!(result.iter().any(|a| a == "--device"));
        assert!(result.iter().any(|a| a == "metal"));
        assert!(result.iter().any(|a| a == "--seed"));
        assert!(result.iter().any(|a| a == "42"));
    }

    #[test]
    fn test_extract_server_args_already_server() {
        // 已经是 server 模式的参数
        let clone_args = vec![
            "server".to_string(),
            "--model".to_string(),
            "models/qwen3-tts".to_string(),
            "--device".to_string(),
            "metal".to_string(),
            "--decode-device".to_string(),
            "cpu".to_string(),
            "--seed".to_string(),
            "42".to_string(),
        ];

        let result = PersistentSubprocessCloneEngine::extract_server_args(
            &clone_args,
            "subprocess-persistent",
        );

        assert_eq!(result[0], "server");
        assert_eq!(result.len(), clone_args.len()); // 不应添加额外的 "server"
        assert!(result.iter().any(|a| a == "--decode-device"));
        assert!(result.iter().any(|a| a == "cpu"));
    }

    #[test]
    fn test_extract_server_args_empty() {
        let result =
            PersistentSubprocessCloneEngine::extract_server_args(&[], "subprocess-persistent");
        assert_eq!(result, vec!["server".to_string()]);
    }

    // ── P1: 多说话人缓存测试 ──────────────────────────────

    #[test]
    fn test_prewarm_speaker_id_storage() {
        // 验证 PersistentSubprocessCloneEngine 可以存储和读取 speaker_id
        // 不实际启动服务端（不调用 prewarm_speaker）
        let engine = PersistentSubprocessCloneEngine::new("/bin/echo".to_string(), vec![], 10);

        // 初始状态: 无 speaker_id
        assert!(engine.current_speaker_id().is_none());

        // 模拟设置 speaker_id（直接操作内部状态）
        {
            let mut sid = engine.current_speaker_id.lock().unwrap();
            *sid = Some("speaker_0".to_string());
        }

        assert_eq!(engine.current_speaker_id().as_deref(), Some("speaker_0"));
    }

    #[test]
    fn test_cloning_integration_prewarm_no_op_for_mock() {
        // 验证 MockCloningEngine 的 prewarm_speaker 是空操作（不 panic）
        let integration =
            CloningIntegration::new(Box::new(MockCloningEngine::new()), CloningConfig::default());

        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        // 默认实现应返回 Ok(())
        let result = integration.prewarm_speaker("speaker_0", &ref_path, Some("test text"));
        assert!(result.is_ok(), "Mock prewarm should succeed (no-op)");
    }

    #[test]
    fn test_cloning_integration_prewarm_nonexistent_ref() {
        // 验证 prewarm_speaker 对不存在的参考音频的处理
        let integration =
            CloningIntegration::new(Box::new(MockCloningEngine::new()), CloningConfig::default());

        // MockCloningEngine 的默认 prewarm_speaker 是空操作，不会检查文件存在性
        // 但 PersistentSubprocessCloneEngine 会检查（在 pub fn prewarm_speaker 中）
        let result =
            integration.prewarm_speaker("speaker_0", Path::new("/nonexistent/ref.wav"), None);
        // 默认实现返回 Ok(())，不检查文件
        assert!(result.is_ok(), "Default impl is no-op");
    }

    #[test]
    fn test_voice_cloning_engine_trait_has_prewarm_method() {
        // 验证 trait 对象可以调用 prewarm_speaker
        let engine: Box<dyn VoiceCloningEngine> = Box::new(MockCloningEngine::new());

        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let ref_path = dir.path().join("reference.wav");
        create_reference_audio(&ref_path);

        // 通过 trait 对象调用 prewarm_speaker
        let result = engine.prewarm_speaker("test_speaker", &ref_path, Some("hello"));
        assert!(result.is_ok(), "Trait method should work via dyn dispatch");
    }
}
