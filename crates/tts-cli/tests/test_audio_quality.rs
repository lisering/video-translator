//! TTS 音频输出质量测试
//!
//! 验证 AudioBuffer 的输出质量属性:
//! - WAV 文件格式正确性 (采样率, 位深度, 声道)
//! - 波形属性 (RMS 能量, 峰值, DC 偏移, 过零率)
//! - 静音检测
//! - 信噪比估计
//!
//! 这些测试不需要 Metal 设备或模型文件。

use vt_tts::audio::{extract_mel_spectrogram, AudioBuffer, MelConfig};

// ─── WAV 文件格式验证 ──────────────────────────────────────

#[test]
fn test_wav_format_24khz_mono_16bit() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("format_test.wav");

    let buf = AudioBuffer::from_samples(vec![0.5; 2400], 24000);
    buf.save_wav(&path).unwrap();

    let reader = hound::WavReader::open(&path).unwrap();
    let spec = reader.spec();

    assert_eq!(spec.sample_rate, 24000, "Sample rate should be 24kHz");
    assert_eq!(spec.channels, 1, "Should be mono");
    assert_eq!(spec.bits_per_sample, 16, "Should be 16-bit");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
}

#[test]
fn test_wav_format_16khz_mono_16bit() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("format_16k.wav");

    let buf = AudioBuffer::from_samples(vec![0.3; 1600], 16000);
    buf.save_wav(&path).unwrap();

    let reader = hound::WavReader::open(&path).unwrap();
    let spec = reader.spec();

    assert_eq!(spec.sample_rate, 16000);
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.bits_per_sample, 16);
}

// ─── 波形属性分析 ──────────────────────────────────────────

/// 计算 RMS 能量
fn rms_energy(samples: &[f32]) -> f32 {
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// 计算峰值幅度
fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

/// 计算 DC 偏移
fn dc_offset(samples: &[f32]) -> f32 {
    let sum: f32 = samples.iter().sum();
    sum / samples.len() as f32
}

/// 计算过零率
fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0;
    for i in 1..samples.len() {
        if (samples[i - 1] >= 0.0) != (samples[i] >= 0.0) {
            crossings += 1;
        }
    }
    crossings as f32 / (samples.len() - 1) as f32
}

#[test]
fn test_silence_rms_near_zero() {
    let buf = AudioBuffer::from_samples(vec![0.0; 24000], 24000);
    let rms = rms_energy(&buf.samples);
    assert!(rms < 1e-10, "Silence RMS should be ~0, got {}", rms);
}

#[test]
fn test_sine_wave_rms() {
    // 440Hz 正弦波, 振幅 0.5, RMS 应为 0.5/√2 ≈ 0.354
    let samples: Vec<f32> = (0..24000)
        .map(|i| ((i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 24000.0).sin() as f32) * 0.5)
        .collect();
    let buf = AudioBuffer::from_samples(samples, 24000);
    let rms = rms_energy(&buf.samples);
    let expected = 0.5 / 2.0f32.sqrt();
    assert!(
        (rms - expected).abs() < 0.01,
        "Sine wave RMS should be ~{:.3}, got {:.3}",
        expected,
        rms
    );
}

#[test]
fn test_silence_peak_zero() {
    let buf = AudioBuffer::from_samples(vec![0.0; 1000], 24000);
    assert!(peak_amplitude(&buf.samples) < 1e-10);
}

#[test]
fn test_sine_wave_peak() {
    let samples: Vec<f32> = (0..24000)
        .map(|i| ((i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 24000.0).sin() as f32) * 0.8)
        .collect();
    let buf = AudioBuffer::from_samples(samples, 24000);
    let peak = peak_amplitude(&buf.samples);
    assert!(
        (peak - 0.8).abs() < 0.01,
        "Peak should be ~0.8, got {}",
        peak
    );
}

#[test]
fn test_silence_zero_dc_offset() {
    let buf = AudioBuffer::from_samples(vec![0.0; 1000], 24000);
    assert!(dc_offset(&buf.samples).abs() < 1e-10);
}

#[test]
fn test_sine_wave_zero_dc_offset() {
    let samples: Vec<f32> = (0..24000)
        .map(|i| ((i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 24000.0).sin() as f32))
        .collect();
    let buf = AudioBuffer::from_samples(samples, 24000);
    let dc = dc_offset(&buf.samples);
    assert!(
        dc.abs() < 0.01,
        "Sine wave DC offset should be ~0, got {}",
        dc
    );
}

#[test]
fn test_dc_offset_detection() {
    let samples: Vec<f32> = (0..1000).map(|_| 0.5).collect();
    let buf = AudioBuffer::from_samples(samples, 24000);
    let dc = dc_offset(&buf.samples);
    assert!(
        (dc - 0.5).abs() < 1e-6,
        "Constant 0.5 DC offset should be 0.5, got {}",
        dc
    );
}

#[test]
fn test_silence_zero_crossing_rate() {
    let buf = AudioBuffer::from_samples(vec![0.0; 1000], 24000);
    let zcr = zero_crossing_rate(&buf.samples);
    assert!(zcr < 0.01, "Silence ZCR should be ~0, got {}", zcr);
}

#[test]
fn test_sine_wave_zero_crossing_rate() {
    // 440Hz @ 24kHz → 每周期 ~54.5 samples → 每周期 2 次过零 → ZCR ≈ 2/54.5 ≈ 0.037
    let samples: Vec<f32> = (0..24000)
        .map(|i| ((i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 24000.0).sin() as f32) * 0.5)
        .collect();
    let buf = AudioBuffer::from_samples(samples, 24000);
    let zcr = zero_crossing_rate(&buf.samples);
    assert!(
        zcr > 0.02 && zcr < 0.05,
        "440Hz ZCR should be ~0.037, got {:.4}",
        zcr
    );
}

// ─── WAV 往返保真度 ────────────────────────────────────────

#[test]
fn test_wav_roundtrip_preserves_duration() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("duration.wav");

    let original = AudioBuffer::from_samples(vec![0.5; 7200], 24000); // 0.3 秒
    original.save_wav(&path).unwrap();
    let loaded = AudioBuffer::load_wav(&path).unwrap();

    assert!(
        (original.duration_secs() - loaded.duration_secs()).abs() < 1e-6,
        "Duration should be preserved: {} vs {}",
        original.duration_secs(),
        loaded.duration_secs()
    );
}

#[test]
fn test_wav_roundtrip_preserves_sample_count() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("count.wav");

    for n in &[1, 10, 100, 1000, 24000] {
        let buf = AudioBuffer::from_samples(vec![0.5; *n], 24000);
        buf.save_wav(&path).unwrap();
        let loaded = AudioBuffer::load_wav(&path).unwrap();
        assert_eq!(
            loaded.samples.len(),
            *n,
            "Sample count mismatch for n={}",
            n
        );
    }
}

#[test]
fn test_wav_roundtrip_quantization_error() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("quant.wav");

    let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin()).collect();
    let buf = AudioBuffer::from_samples(samples, 24000);
    buf.save_wav(&path).unwrap();
    let loaded = AudioBuffer::load_wav(&path).unwrap();

    // 16-bit PCM: 量化误差 < 1/32767 ≈ 3.05e-5
    let max_error: f32 = buf
        .samples
        .iter()
        .zip(loaded.samples.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-4,
        "Max quantization error should be < 1e-4, got {}",
        max_error
    );
}

// ─── 多声道 WAV 处理 ───────────────────────────────────────

#[test]
fn test_wav_stereo_to_mono() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("stereo.wav");

    // 创建立体声 WAV
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for i in 0..2400 {
        let left = ((i as f32 * 0.01).sin() * 16000.0) as i16;
        let right = ((i as f32 * 0.02).sin() * 16000.0) as i16;
        writer.write_sample(left).unwrap();
        writer.write_sample(right).unwrap();
    }
    writer.finalize().unwrap();

    // 加载应自动转为单声道（取平均）
    let loaded = AudioBuffer::load_wav(&path).unwrap();
    assert_eq!(loaded.channels, 1, "Should be converted to mono");
    assert_eq!(loaded.samples.len(), 2400, "Should have 2400 mono samples");

    // 第一个样本应该是左右声道的平均
    let expected = ((0.0 + 0.0) / 2.0);
    assert!(
        (loaded.samples[0] - expected).abs() < 1e-3,
        "First mono sample should be average of L and R"
    );
}

// ─── Mel 频谱图质量 ────────────────────────────────────────

#[test]
fn test_mel_spectrogram_silence_floor() {
    let cfg = MelConfig::speaker_encoder();
    let samples = vec![0.0; 16000];
    let mel = extract_mel_spectrogram(&samples, &cfg);

    // 静音的 mel 应该在 -100dB 附近
    let avg: f32 = mel.iter().flatten().sum::<f32>() / (mel.len() * mel[0].len()) as f32;
    assert!(
        (avg - (-100.0)).abs() < 1.0,
        "Silence mel average should be ~-100dB, got {}",
        avg
    );
}

#[test]
fn test_mel_spectrogram_tone_has_energy() {
    let cfg = MelConfig::speaker_encoder();
    let samples: Vec<f32> = (0..16000)
        .map(|i| ((i as f64 * 2.0 * std::f64::consts::PI * 1000.0 / 16000.0).sin() as f32) * 0.5)
        .collect();
    let mel = extract_mel_spectrogram(&samples, &cfg);

    // 1kHz 正弦波应该在某些 mel bin 中有显著能量
    let max_energy: f32 = mel
        .iter()
        .map(|row| row.iter().cloned().fold(0.0f32, f32::max))
        .fold(0.0f32, f32::max);

    assert!(
        max_energy > -50.0,
        "1kHz tone should have mel energy > -50dB, got {}",
        max_energy
    );
}

#[test]
fn test_mel_spectrogram_energy_decreases_with_amplitude() {
    let cfg = MelConfig::speaker_encoder();

    let loud: Vec<f32> = (0..16000)
        .map(|i| ((i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 16000.0).sin() as f32) * 0.9)
        .collect();
    let quiet: Vec<f32> = (0..16000)
        .map(|i| ((i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 16000.0).sin() as f32) * 0.1)
        .collect();

    let mel_loud = extract_mel_spectrogram(&loud, &cfg);
    let mel_quiet = extract_mel_spectrogram(&quiet, &cfg);

    let max_loud: f32 = mel_loud.iter().flatten().cloned().fold(0.0f32, f32::max);
    let max_quiet: f32 = mel_quiet.iter().flatten().cloned().fold(0.0f32, f32::max);

    assert!(
        max_loud > max_quiet,
        "Louder signal should have higher max mel energy: {} vs {}",
        max_loud,
        max_quiet
    );
}
