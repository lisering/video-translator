//! ONNX 音频 tokenizer 配置 — 借鉴 MOSS-TTS 的音频编码/解码方案
//!
//! MOSS-TTS 支持 ONNX / TensorRT / PyTorch 三种音频 tokenizer 后端。
//! 此模块提供 ONNX 后端的配置和接口定义，
//! 可用于替代 Python TTS server 的音频编码/解码部分。
//!
//! 对应 MOSS-TTS 项目:
//! - `moss_tts_delay/llama_cpp/pipeline.py` 中的 `_load_onnx_encoder/decoder`
//! - `moss_audio_tokenizer/onnx/inference.py`
//!
//! # 使用方式
//! 1. 提取 ONNX 编码器和解码器模型
//! 2. 配置 `OnnxAudioTokenizerConfig`
//! 3. 通过 `ort` crate（ONNX Runtime Rust 绑定）加载和推理
//!
//! # 性能
//! ONNX Runtime 在 CPU 上比 PyTorch 快 2-3x，
//! 因为 ONNX Runtime 有优化的 CPU 内核（AVX2/NEON）
//! 且支持线程池调度。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// ONNX 音频 tokenizer 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxAudioTokenizerConfig {
    /// ONNX 编码器模型路径
    pub encoder_path: PathBuf,
    /// ONNX 解码器模型路径
    pub decoder_path: PathBuf,
    /// 是否使用 GPU
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
    /// 采样率（Hz）
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// 下采样率（每秒帧数 = sample_rate / downsample_rate）
    #[serde(default = "default_downsample_rate")]
    pub downsample_rate: u32,
    /// 码本数量
    #[serde(default = "default_n_quantizers")]
    pub n_quantizers: u32,
}

fn default_use_gpu() -> bool {
    false
}

fn default_sample_rate() -> u32 {
    24000
}

fn default_downsample_rate() -> u32 {
    80 // ~12.5 frames per second at 24kHz
}

fn default_n_quantizers() -> u32 {
    32
}

impl Default for OnnxAudioTokenizerConfig {
    fn default() -> Self {
        Self {
            encoder_path: PathBuf::new(),
            decoder_path: PathBuf::new(),
            use_gpu: false,
            sample_rate: default_sample_rate(),
            downsample_rate: default_downsample_rate(),
            n_quantizers: default_n_quantizers(),
        }
    }
}

/// 音频编码请求
#[derive(Debug, Clone)]
pub struct EncodeRequest {
    /// 音频波形（float32, [-1, 1] 范围）
    pub waveform: Vec<f32>,
    /// 采样率
    pub sample_rate: u32,
    /// 请求的码本数量
    pub n_quantizers: u32,
}

/// 音频编码结果
#[derive(Debug, Clone)]
pub struct EncodeResult {
    /// 音频码 [T, n_quantizers]
    pub codes: Vec<Vec<i64>>,
    /// 帧率
    pub frame_rate: f64,
}

/// 音频解码请求
#[derive(Debug, Clone)]
pub struct DecodeRequest {
    /// 音频码 [T, n_quantizers]
    pub codes: Vec<Vec<i64>>,
    /// 码本数量
    pub n_quantizers: u32,
}

/// 音频解码结果
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// 解码后的波形（float32）
    pub waveform: Vec<f32>,
    /// 采样率
    pub sample_rate: u32,
}

/// ONNX 音频 tokenizer 接口
///
/// 实际实现需要 `ort` crate（ONNX Runtime Rust 绑定）。
/// 此 trait 定义接口，具体实现可通过 feature flag 启用。
pub trait OnnxAudioTokenizer: Send + Sync {
    /// 编码音频波形为离散码
    fn encode(&self, request: &EncodeRequest) -> Result<EncodeResult, String>;

    /// 解码离散码为音频波形
    fn decode(&self, request: &DecodeRequest) -> Result<DecodeResult, String>;

    /// 获取采样率
    fn sample_rate(&self) -> u32;

    /// 获取帧率
    fn frame_rate(&self) -> f64;

    /// 获取码本数量
    fn n_quantizers(&self) -> u32;

    /// 释放资源
    fn close(&mut self) {}
}

/// 计算音频时长（秒）
pub fn estimate_duration_secs(num_frames: usize, frame_rate: f64) -> f64 {
    if frame_rate <= 0.0 {
        return 0.0;
    }
    num_frames as f64 / frame_rate
}

/// 估算帧数（从时长）
pub fn estimate_frames(duration_secs: f64, frame_rate: f64) -> usize {
    if frame_rate <= 0.0 {
        return 0;
    }
    (duration_secs * frame_rate) as usize
}

/// 估算内存占用（字节）
pub fn estimate_memory_bytes(
    num_frames: usize,
    n_quantizers: usize,
    waveform_samples: usize,
) -> usize {
    // codes: T * n_q * 8 bytes (i64)
    // waveform: T * downsample_rate * 4 bytes (f32)
    num_frames * n_quantizers * 8 + waveform_samples * 4
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OnnxAudioTokenizerConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.downsample_rate, 80);
        assert_eq!(config.n_quantizers, 32);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_config_serialization() {
        let config = OnnxAudioTokenizerConfig {
            encoder_path: PathBuf::from("/path/to/encoder.onnx"),
            decoder_path: PathBuf::from("/path/to/decoder.onnx"),
            use_gpu: false,
            sample_rate: 24000,
            downsample_rate: 80,
            n_quantizers: 32,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OnnxAudioTokenizerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.sample_rate, 24000);
        assert_eq!(deserialized.n_quantizers, 32);
    }

    #[test]
    fn test_estimate_duration() {
        // 1000 frames at 12.5 fps = 80 seconds
        let dur = estimate_duration_secs(1000, 12.5);
        assert!((dur - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_frames() {
        // 80 seconds at 12.5 fps = 1000 frames
        let frames = estimate_frames(80.0, 12.5);
        assert_eq!(frames, 1000);
    }

    #[test]
    fn test_estimate_memory() {
        let mem = estimate_memory_bytes(1000, 32, 80000);
        assert!(mem > 0);
        // codes: 1000 * 32 * 8 = 256000
        // waveform: 80000 * 4 = 320000
        assert_eq!(mem, 256000 + 320000);
    }
}
