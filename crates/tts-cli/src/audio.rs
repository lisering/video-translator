//! 音频处理模块
//!
//! 提供 WAV 文件读写、重采样、mel-频谱图提取等功能。
//!
//! 参考来源：
//! - Qwen3-TTS: AudioBuffer 结构、rubato 重采样
//! - QORA-TTS: WAV I/O、24kHz 输出
//! - neutts-rs: rustfft ISTFT

use std::path::Path;

use anyhow::{Context, Result};

/// 音频缓冲区 — 统一的音频数据表示
///
/// 参考 Qwen3-TTS 的 AudioBuffer 设计。
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// 采样数据 (f32, -1.0 ~ 1.0)
    pub samples: Vec<f32>,
    /// 采样率
    pub sample_rate: u32,
    /// 声道数 (始终为 1 = 单声道)
    pub channels: u16,
}

impl AudioBuffer {
    /// 创建空的音频缓冲区
    pub fn new(sample_rate: u32) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
            channels: 1,
        }
    }

    /// 从采样数据创建
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    /// 音频时长（秒）
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// 采样数
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// 重采样到目标采样率
    ///
    /// 使用线性插值（参考 QORA-TTS 的简单重采样方案）。
    /// 对于高质量需求可切换到 rubato。
    pub fn resample_linear(&self, target_rate: u32) -> Self {
        if self.sample_rate == target_rate {
            return self.clone();
        }

        let ratio = target_rate as f64 / self.sample_rate as f64;
        let new_len = (self.samples.len() as f64 * ratio) as usize;
        let mut output = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_pos = i as f64 / ratio;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f64;

            let s1 = self.samples.get(src_idx).copied().unwrap_or(0.0);
            let s2 = self.samples.get(src_idx + 1).copied().unwrap_or(s1);
            output.push((s1 as f64 * (1.0 - frac) + s2 as f64 * frac) as f32);
        }

        Self {
            samples: output,
            sample_rate: target_rate,
            channels: 1,
        }
    }

    /// 高质量重采样（使用线性插值，后续可切换到 rubato）
    pub fn resample_hq(&self, target_rate: u32) -> Result<Self> {
        // 目前使用线性插值，质量足够用于 TTS 音频
        Ok(self.resample_linear(target_rate))
    }

    /// 保存为 WAV 文件 (16-bit PCM)
    ///
    /// 参考 QORA-TTS 的 wav::write_wav。
    pub fn save_wav(&self, path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec)
            .with_context(|| format!("Failed to create WAV file: {:?}", path))?;

        for &sample in &self.samples {
            let i16_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(i16_sample)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// 从 WAV 文件加载
    ///
    /// 自动处理多声道→单声道转换、采样率归一化。
    pub fn load_wav(path: &Path) -> Result<Self> {
        let mut reader = hound::WavReader::open(path)
            .with_context(|| format!("Failed to open WAV file: {:?}", path))?;

        let spec = reader.spec();
        let samples_i16: Vec<i16> = reader.samples::<i16>().filter_map(|s| s.ok()).collect();

        // 多声道 → 单声道（取平均）
        let samples: Vec<f32> = if spec.channels > 1 {
            samples_i16
                .chunks(spec.channels as usize)
                .map(|chunk| {
                    let sum: i32 = chunk.iter().map(|&s| s as i32).sum();
                    (sum as f32 / chunk.len() as f32) / 32768.0
                })
                .collect()
        } else {
            samples_i16.iter().map(|&s| s as f32 / 32768.0).collect()
        };

        Ok(Self {
            samples,
            sample_rate: spec.sample_rate,
            channels: 1,
        })
    }
}

/// Mel-频谱图配置
///
/// 参考 Qwen3-TTS 和 QORA-TTS 的 mel-频谱图参数。
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// 采样率
    pub sample_rate: u32,
    /// FFT 窗口大小
    pub n_fft: usize,
    /// 帧移（hop length）
    pub hop_length: usize,
    /// 窗口长度
    pub win_length: usize,
    /// mel 滤波器组数量
    pub n_mels: usize,
    /// 最低频率
    pub fmin: f64,
    /// 最高频率
    pub fmax: f64,
}

impl MelConfig {
    /// 说话人编码器使用的 mel 配置
    ///
    /// 参考 QORA-TTS 的 speaker_encoder mel 配置。
    pub fn speaker_encoder() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8000.0,
        }
    }

    /// 语音编解码器使用的 mel 配置
    pub fn codec() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            n_mels: 80,
            fmin: 0.0,
            fmax: 12000.0,
        }
    }
}

/// 提取 mel-频谱图
///
/// 使用 FFT 计算 STFT，然后应用 mel 滤波器组。
///
/// 参考 QORA-TTS 的 audio_features::extract_mel_spectrogram 实现。
pub fn extract_mel_spectrogram(samples: &[f32], config: &MelConfig) -> Vec<Vec<f32>> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n_fft = config.n_fft;
    let hop = config.hop_length;
    let n_mels = config.n_mels;

    // Hann 窗
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos()) as f32)
        .collect();

    // 计算 STFT
    let n_frames = (samples.len().saturating_sub(n_fft)) / hop + 1;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut power_spec = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        let mut frame: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let s = samples.get(start + i).copied().unwrap_or(0.0);
                Complex::new(s * window[i], 0.0)
            })
            .collect();

        fft.process(&mut frame);

        // 功率谱
        let power: Vec<f32> = (0..=n_fft / 2)
            .map(|i| {
                let re = frame[i].re;
                let im = frame[i].im;
                re * re + im * im
            })
            .collect();

        power_spec.push(power);
    }

    // Mel 滤波器组
    let mel_filterbank = create_mel_filterbank(config, n_fft / 2 + 1);

    // 应用 mel 滤波器
    let mut mel_spec = vec![vec![0.0f32; n_frames]; n_mels];

    for (frame_idx, power) in power_spec.iter().enumerate() {
        for (mel_idx, filter) in mel_filterbank.iter().enumerate() {
            let energy: f32 = power.iter().zip(filter.iter()).map(|(p, f)| p * f).sum();
            // log mel
            mel_spec[mel_idx][frame_idx] = 10.0 * (energy + 1e-10).log10();
        }
    }

    mel_spec
}

/// 创建 mel 滤波器组
///
/// 参考 HTK mel 频率刻度。
fn create_mel_filterbank(config: &MelConfig, n_freqs: usize) -> Vec<Vec<f32>> {
    let n_mels = config.n_mels;
    let fmin = config.fmin;
    let fmax = config.fmax;
    let sr = config.sample_rate as f64;

    // mel 频率转换
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // mel 等间隔点
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    // 转回 Hz
    let hz_points: Vec<f64> = mel_points.iter().map(|m| mel_to_hz(*m)).collect();

    // FFT 频率点
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * sr / (2.0 * (n_freqs - 1) as f64))
        .collect();

    // 三角滤波器
    let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];

    for mel_idx in 0..n_mels {
        let left = hz_points[mel_idx];
        let center = hz_points[mel_idx + 1];
        let right = hz_points[mel_idx + 2];

        for (freq_idx, &freq) in fft_freqs.iter().enumerate() {
            let weight = if freq < left || freq > right {
                0.0
            } else if freq <= center {
                (freq - left) / (center - left).max(1e-10)
            } else {
                (right - freq) / (right - center).max(1e-10)
            };
            filterbank[mel_idx][freq_idx] = weight as f32;
        }
    }

    filterbank
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── AudioBuffer 基本操作 ───

    #[test]
    fn test_audio_buffer_new() {
        let buf = AudioBuffer::new(24000);
        assert_eq!(buf.samples.len(), 0);
        assert_eq!(buf.sample_rate, 24000);
        assert_eq!(buf.channels, 1);
        assert_eq!(buf.duration_secs(), 0.0);
    }

    #[test]
    fn test_audio_buffer_from_samples() {
        let samples = vec![0.5, -0.3, 0.8, 0.0];
        let buf = AudioBuffer::from_samples(samples.clone(), 16000);
        assert_eq!(buf.samples, samples);
        assert_eq!(buf.num_samples(), 4);
        assert_eq!(buf.sample_rate, 16000);
        assert!((buf.duration_secs() - 4.0 / 16000.0).abs() < 1e-10);
    }

    #[test]
    fn test_audio_buffer_duration() {
        // 1 秒 @ 24kHz = 24000 samples
        let buf = AudioBuffer::from_samples(vec![0.0; 24000], 24000);
        assert!((buf.duration_secs() - 1.0).abs() < 1e-10);
    }

    // ─── 线性重采样 ───

    #[test]
    fn test_resample_noop_same_rate() {
        let buf = AudioBuffer::from_samples(vec![0.5, -0.3, 0.8], 24000);
        let resampled = buf.resample_linear(24000);
        assert_eq!(resampled.samples, buf.samples);
        assert_eq!(resampled.sample_rate, 24000);
    }

    #[test]
    fn test_resample_upsample() {
        // 2x 上采样: 2 samples @ 8kHz → 4 samples @ 16kHz
        let buf = AudioBuffer::from_samples(vec![0.0, 1.0], 8000);
        let resampled = buf.resample_linear(16000);
        assert_eq!(resampled.sample_rate, 16000);
        assert_eq!(resampled.samples.len(), 4);
        // 第一个样本应该是 0.0
        assert!((resampled.samples[0]).abs() < 1e-6);
        // 最后一个样本应该接近 1.0
        assert!((resampled.samples[3] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_resample_downsample() {
        // 2x 下采样: 4 samples @ 16kHz → 2 samples @ 8kHz
        let buf = AudioBuffer::from_samples(vec![0.0, 0.5, 1.0, 0.5], 16000);
        let resampled = buf.resample_linear(8000);
        assert_eq!(resampled.sample_rate, 8000);
        assert_eq!(resampled.samples.len(), 2);
    }

    #[test]
    fn test_resample_preserves_amplitude_range() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let buf = AudioBuffer::from_samples(samples, 24000);
        let resampled = buf.resample_linear(48000);
        let max_orig = buf.samples.iter().cloned().fold(0.0f32, f32::max);
        let max_resampled = resampled.samples.iter().cloned().fold(0.0f32, f32::max);
        // 幅度应该在合理范围内（线性插值不会大幅改变幅度）
        assert!(
            (max_orig - max_resampled).abs() < 0.2,
            "Amplitude changed too much: {} → {}",
            max_orig,
            max_resampled
        );
    }

    // ─── WAV 读写往返 ───

    #[test]
    fn test_wav_save_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.wav");

        let samples = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.7];
        let buf = AudioBuffer::from_samples(samples.clone(), 24000);
        buf.save_wav(&path).unwrap();

        let loaded = AudioBuffer::load_wav(&path).unwrap();
        assert_eq!(loaded.sample_rate, 24000);
        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.samples.len(), samples.len());

        // 16-bit PCM 量化误差 ~1/32768 ≈ 3e-5
        for (orig, loaded) in samples.iter().zip(loaded.samples.iter()) {
            assert!(
                (orig - loaded).abs() < 1e-3,
                "Sample mismatch: {} vs {}",
                orig,
                loaded
            );
        }
    }

    #[test]
    fn test_wav_clipping() {
        // 超出 [-1, 1] 的样本应该被 clamp
        let buf = AudioBuffer::from_samples(vec![2.0, -2.0, 0.5], 24000);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("clip.wav");
        buf.save_wav(&path).unwrap();

        let loaded = AudioBuffer::load_wav(&path).unwrap();
        // 2.0 应该被 clamp 到 ~1.0, -2.0 到 ~-1.0
        assert!(loaded.samples[0] <= 1.0);
        assert!(loaded.samples[1] >= -1.0);
        assert!((loaded.samples[2] - 0.5).abs() < 1e-3);
    }

    // ─── Mel 频率转换 ───

    #[test]
    fn test_hz_to_mel_conversion() {
        assert!((hz_to_mel(0.0)).abs() < 1e-10);
        // hz_to_mel(700) = 2595 * log10(1 + 700/700) = 2595 * log10(2) ≈ 781.1
        let mel_700 = hz_to_mel(700.0);
        assert!(mel_700 > 770.0 && mel_700 < 790.0, "mel(700) = {}", mel_700);
        // hz_to_mel(16000) = 2595 * log10(1 + 16000/700) ≈ 3575.6
        let mel_16k = hz_to_mel(16000.0);
        assert!(
            mel_16k > 3500.0 && mel_16k < 3650.0,
            "mel(16000) = {}",
            mel_16k
        );
    }

    #[test]
    fn test_mel_to_hz_inverse() {
        for &hz in &[100.0, 1000.0, 4000.0, 8000.0, 16000.0] {
            let mel = hz_to_mel(hz);
            let hz_back = mel_to_hz(mel);
            assert!(
                (hz - hz_back).abs() / hz < 1e-3,
                "Roundtrip failed: {}Hz → {}mel → {}Hz",
                hz,
                mel,
                hz_back
            );
        }
    }

    #[test]
    fn test_mel_to_hz_known_values() {
        assert!((mel_to_hz(0.0)).abs() < 1e-6);
        let hz = mel_to_hz(1000.0);
        assert!(hz > 990.0 && hz < 1010.0);
    }

    // ─── MelConfig 预设 ───

    #[test]
    fn test_mel_config_speaker_encoder() {
        let cfg = MelConfig::speaker_encoder();
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.n_fft, 512);
        assert_eq!(cfg.hop_length, 160);
        assert_eq!(cfg.n_mels, 80);
    }

    #[test]
    fn test_mel_config_codec() {
        let cfg = MelConfig::codec();
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.n_fft, 1024);
        assert_eq!(cfg.hop_length, 256);
        assert_eq!(cfg.n_mels, 80);
    }

    // ─── Mel 频谱图提取 ───

    #[test]
    fn test_extract_mel_spectrogram_shape() {
        let cfg = MelConfig::speaker_encoder();
        let samples = vec![0.0; 16000 * 2]; // 2 秒静音
        let mel = extract_mel_spectrogram(&samples, &cfg);
        assert_eq!(mel.len(), cfg.n_mels);
        // 静音的 mel 应该为 10*log10(1e-10) = -100 (floor 值)
        for row in &mel {
            for &val in row {
                assert!(
                    (val - (-100.0)).abs() < 1.0,
                    "Silence should produce -100dB mel, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_extract_mel_spectrogram_nonzero() {
        let cfg = MelConfig::speaker_encoder();
        // 440Hz 正弦波
        let samples: Vec<f32> = (0..16000)
            .map(|i| ((i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 16000.0).sin() as f32) * 0.5)
            .collect();
        let mel = extract_mel_spectrogram(&samples, &cfg);
        assert_eq!(mel.len(), cfg.n_mels);
        // 应该有非零值
        let total: f32 = mel.iter().flatten().map(|&v| v.abs()).sum();
        assert!(total > 0.0, "Sine wave should produce non-zero mel");
    }
}
