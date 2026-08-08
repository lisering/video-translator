//! 说话人性别检测模块
//!
//! 通过分析原始音频的基频 (F0) 自动判断说话人是男声还是女声。
//!
//! # 原理
//! - 男声基频范围: 85-180Hz (平均 ~120Hz)
//! - 女声基频范围: 165-255Hz (平均 ~210Hz)
//! - 阈值: 165Hz (低于=男声, 高于=女声)
//!
//! # 算法
//! 使用归一化自相关 (Normalized Autocorrelation) 法:
//! 1. 将音频分帧 (30ms/帧)
//! 2. 过滤静音帧 (能量 < 阈值)
//! 3. 对每个语音帧计算自相关，在 50-400Hz 范围找峰值
//! 4. 取所有帧 F0 的中位数作为最终结果
//!
//! # 示例
//! ```no_run
//! use vt_core::gender_detect::detect_gender;
//! use vt_core::voice_manager::VoiceGender;
//!
//! let gender = detect_gender(&samples, 16000);
//! match gender {
//!     VoiceGender::Male => println!("检测到男声"),
//!     VoiceGender::Female => println!("检测到女声"),
//!     _ => println!("无法确定"),
//! }
//! ```

use crate::voice_manager::VoiceGender;

/// F0 阈值 (Hz)：低于此值判定为男声
const F0_THRESHOLD: f64 = 165.0;

/// 帧大小 (秒)：30ms 窗口
const FRAME_SIZE_SECS: f64 = 0.03;

/// 静音阈值：帧能量低于此值视为静音
const SILENCE_THRESHOLD: f32 = 0.001;

/// 自相关峰值阈值：低于此值视为非周期信号
const AUTOCORR_THRESHOLD: f32 = 0.3;

/// 最低 F0 (Hz)：人声基频下限
const F0_MIN: f64 = 50.0;

/// 最高 F0 (Hz)：人声基频上限
const F0_MAX: f64 = 400.0;

/// 从音频采样数据检测说话人性别
///
/// 分析音频的基频 (F0)，根据中位数 F0 判断男/女声。
/// 仅分析前 10 秒音频（足够判断性别）。
///
/// # 参数
/// - `samples`: 音频采样数据 (f32, -1.0 ~ 1.0)
/// - `sample_rate`: 采样率 (如 16000)
///
/// # 返回
/// `VoiceGender::Male` 或 `VoiceGender::Female`。
/// 如果无法检测（音频太短或全静音），返回 `VoiceGender::Female`（默认女声）。
#[must_use]
pub fn detect_gender(samples: &[f32], sample_rate: u32) -> VoiceGender {
    if samples.is_empty() || sample_rate == 0 {
        return VoiceGender::Female;
    }

    // 只取前 10 秒
    let max_samples = (sample_rate as f64 * 10.0) as usize;
    let data = if samples.len() > max_samples {
        &samples[..max_samples]
    } else {
        samples
    };

    let frame_size = (sample_rate as f64 * FRAME_SIZE_SECS) as usize;
    if frame_size == 0 || data.len() < frame_size {
        return VoiceGender::Female;
    }

    let n_frames = data.len() / frame_size;
    let min_lag = (sample_rate as f64 / F0_MAX) as usize;
    let max_lag = (sample_rate as f64 / F0_MIN) as usize;

    let mut f0_values: Vec<f64> = Vec::new();

    for i in 0..n_frames {
        let frame = &data[i * frame_size..(i + 1) * frame_size];

        // 计算帧能量，跳过静音帧
        let energy: f32 = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
        if energy < SILENCE_THRESHOLD {
            continue;
        }

        // 去均值
        let mean: f32 = frame.iter().sum::<f32>() / frame.len() as f32;
        let centered: Vec<f32> = frame.iter().map(|s| s - mean).collect();

        let total_energy: f32 = centered.iter().map(|s| s * s).sum();
        if total_energy < 1e-6 {
            continue;
        }

        // 计算归一化自相关
        let mut peak_lag = 0usize;
        let mut peak_val = 0.0f32;

        for lag in min_lag..max_lag.min(frame_size / 2) {
            // 计算自相关：sum(centered[j] * centered[j + lag])
            let mut autocorr = 0.0f32;
            for j in 0..(frame_size - lag) {
                autocorr += centered[j] * centered[j + lag];
            }
            autocorr /= total_energy + 1e-10; // 归一化

            if autocorr > peak_val && autocorr > AUTOCORR_THRESHOLD {
                peak_val = autocorr;
                peak_lag = lag;
            }
        }

        if peak_lag > 0 {
            let f0 = sample_rate as f64 / peak_lag as f64;
            if (F0_MIN..=F0_MAX).contains(&f0) {
                f0_values.push(f0);
            }
        }
    }

    if f0_values.len() < 3 {
        tracing::debug!(
            "Gender detection: not enough voiced frames ({}), defaulting to Female",
            f0_values.len()
        );
        return VoiceGender::Female;
    }

    // 取中位数 F0
    f0_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_f0 = f0_values[f0_values.len() / 2];

    let gender = if median_f0 < F0_THRESHOLD {
        VoiceGender::Male
    } else {
        VoiceGender::Female
    };

    tracing::info!(
        "Gender detection: median F0 = {:.1}Hz ({} voiced frames) → {}",
        median_f0,
        f0_values.len(),
        gender
    );

    gender
}

/// 从 WAV 文件检测说话人性别
///
/// 读取 WAV 文件并调用 [`detect_gender`]。
///
/// # 参数
/// - `wav_path`: WAV 文件路径 (16kHz mono)
///
/// # 返回
/// `VoiceGender::Male` 或 `VoiceGender::Female`。
/// 文件读取失败时返回 `VoiceGender::Female`（默认）。
pub fn detect_gender_from_wav(wav_path: &std::path::Path) -> VoiceGender {
    match crate::asr::read_wav_mono(wav_path) {
        Ok((samples, sample_rate)) => detect_gender(&samples, sample_rate),
        Err(e) => {
            tracing::warn!("Gender detection: failed to read WAV {:?}: {}", wav_path, e);
            VoiceGender::Female
        }
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成正弦波测试数据
    fn generate_sine(freq: f64, duration_secs: f64, sample_rate: u32) -> Vec<f32> {
        let n = (duration_secs * sample_rate as f64) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate as f64).sin() as f32
                    * 0.5
            })
            .collect()
    }

    #[test]
    fn test_detect_gender_empty() {
        assert_eq!(detect_gender(&[], 16000), VoiceGender::Female);
    }

    #[test]
    fn test_detect_gender_zero_sample_rate() {
        assert_eq!(detect_gender(&[0.5, 0.5], 0), VoiceGender::Female);
    }

    #[test]
    fn test_detect_gender_silence() {
        let samples = vec![0.0f32; 16000]; // 1s 全静音
        assert_eq!(detect_gender(&samples, 16000), VoiceGender::Female);
    }

    #[test]
    fn test_detect_gender_low_f0_male() {
        // 120Hz 正弦波 ≈ 男声基频
        let samples = generate_sine(120.0, 3.0, 16000);
        let gender = detect_gender(&samples, 16000);
        assert_eq!(
            gender,
            VoiceGender::Male,
            "120Hz should be detected as Male"
        );
    }

    #[test]
    fn test_detect_gender_high_f0_female() {
        // 220Hz 正弦波 ≈ 女声基频
        let samples = generate_sine(220.0, 3.0, 16000);
        let gender = detect_gender(&samples, 16000);
        assert_eq!(
            gender,
            VoiceGender::Female,
            "220Hz should be detected as Female"
        );
    }

    #[test]
    fn test_detect_gender_short_audio() {
        // 太短的音频应该回退到默认女声
        let samples = generate_sine(120.0, 0.01, 16000);
        let gender = detect_gender(&samples, 16000);
        // 0.01s 太短，可能检测不到足够帧
        assert_eq!(
            gender,
            VoiceGender::Female,
            "Very short audio should default to Female"
        );
    }
}
