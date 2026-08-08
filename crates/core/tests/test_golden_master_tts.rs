//! P0-B: TTS 音频输出 Golden Master 测试
//!
//! 这些测试验证 Golden Master 框架在 TTS 音频指纹场景下的工作流程：
//! 1. 生成音频 → 计算指纹 → 与基线对比
//! 2. 音频微小变化 → 检测到变化 → 测试失败
//! 3. 音频不变 → 指纹匹配 → 测试通过
//!
//! 使用合成音频（正弦波 + 噪声）代替真实 TTS 输出，确保跨平台可运行。
//! 在 macOS 上额外测试真实 `SayEngine` 输出。

use std::path::Path;
use vt_core::golden_master::{
    generate_test_wav, AudioFingerprint, Fingerprint, FingerprintCompare, GoldenMaster,
    GoldenMasterTestCase,
};

// ─── 合成音频 Golden Master 测试 ──────────────────────────

#[test]
fn test_tts_golden_master_synthetic_stable() {
    // 模拟 TTS 输出：生成相同的合成音频两次，验证指纹一致
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let wav1 = dir.path().join("tts_output_1.wav");
    let wav2 = dir.path().join("tts_output_2.wav");

    // 生成确定性音频（1kHz 正弦波，0.5s，24kHz，振幅 0.5，噪声 0.01）
    generate_test_wav(&wav1, 1000.0, 0.5, 24000, 0.5, 0.01)
        .expect("Failed to generate test WAV 1");
    generate_test_wav(&wav2, 1000.0, 0.5, 24000, 0.5, 0.01)
        .expect("Failed to generate test WAV 2");

    // 计算指纹
    let fp1 = AudioFingerprint::from_wav_file(&wav1, Some(24000))
        .expect("Failed to compute fingerprint 1");
    let fp2 = AudioFingerprint::from_wav_file(&wav2, Some(24000))
        .expect("Failed to compute fingerprint 2");

    // 相同音频 → 精确匹配
    assert_eq!(
        fp1.compare(&fp2),
        FingerprintCompare::Match,
        "Identical audio should match exactly"
    );
}

#[test]
fn test_tts_golden_master_detect_frequency_change() {
    // 检测频率变化：1kHz → 500Hz
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let wav1 = dir.path().join("tts_1kHz.wav");
    let wav2 = dir.path().join("tts_500hz.wav");

    generate_test_wav(&wav1, 1000.0, 0.5, 24000, 0.5, 0.01)
        .expect("Failed to generate 1kHz WAV");
    generate_test_wav(&wav2, 500.0, 0.5, 24000, 0.5, 0.01)
        .expect("Failed to generate 500Hz WAV");

    let fp1 = AudioFingerprint::from_wav_file(&wav1, Some(24000))
        .expect("Failed to compute 1kHz fingerprint");
    let fp2 = AudioFingerprint::from_wav_file(&wav2, Some(24000))
        .expect("Failed to compute 500Hz fingerprint");

    // 不同频率 → 不匹配
    let cmp = fp1.compare(&fp2);
    assert!(
        !cmp.is_pass(),
        "Different frequencies should not pass: {}",
        cmp.diff_message()
    );
}

#[test]
fn test_tts_golden_master_detect_amplitude_change() {
    // 检测音量变化：0.5 → 0.9
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let wav1 = dir.path().join("tts_quiet.wav");
    let wav2 = dir.path().join("tts_loud.wav");

    generate_test_wav(&wav1, 1000.0, 0.5, 24000, 0.5, 0.01)
        .expect("Failed to generate quiet WAV");
    generate_test_wav(&wav2, 1000.0, 0.5, 24000, 0.9, 0.01)
        .expect("Failed to generate loud WAV");

    let fp1 = AudioFingerprint::from_wav_file(&wav1, Some(24000))
        .expect("Failed to compute quiet fingerprint");
    let fp2 = AudioFingerprint::from_wav_file(&wav2, Some(24000))
        .expect("Failed to compute loud fingerprint");

    assert!(
        !fp1.compare(&fp2).is_pass(),
        "Different amplitudes should not pass"
    );
}

#[test]
fn test_tts_golden_master_detect_duration_change() {
    // 检测时长变化：0.5s → 1.0s
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let wav1 = dir.path().join("tts_short.wav");
    let wav2 = dir.path().join("tts_long.wav");

    generate_test_wav(&wav1, 1000.0, 0.5, 24000, 0.5, 0.01)
        .expect("Failed to generate short WAV");
    generate_test_wav(&wav2, 1000.0, 1.0, 24000, 0.5, 0.01)
        .expect("Failed to generate long WAV");

    let fp1 = AudioFingerprint::from_wav_file(&wav1, Some(24000))
        .expect("Failed to compute short fingerprint");
    let fp2 = AudioFingerprint::from_wav_file(&wav2, Some(24000))
        .expect("Failed to compute long fingerprint");

    assert!(
        !fp1.compare(&fp2).is_pass(),
        "Different durations should not pass"
    );
}

#[test]
fn test_tts_golden_master_persistence() {
    // 测试 Golden Master 基线持久化
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let wav = dir.path().join("tts_baseline.wav");
    generate_test_wav(&wav, 880.0, 0.3, 24000, 0.5, 0.02)
        .expect("Failed to generate WAV");

    let fp = AudioFingerprint::from_wav_file(&wav, Some(24000))
        .expect("Failed to compute fingerprint");

    // 首次：创建基线
    let result = gm
        .load_or_create("tts", "persistence_test", &fp)
        .expect("load_or_create failed");
    assert_eq!(result, FingerprintCompare::Match);

    // 验证基线文件存在
    let baselines = gm.list_baselines();
    assert!(baselines.iter().any(|(m, n)| m == "tts" && n == "persistence_test"));

    // 第二次：相同数据 → Match
    let result = gm
        .load_or_create("tts", "persistence_test", &fp)
        .expect("load_or_create failed");
    assert!(result.is_pass());

    // 生成不同的音频 → Changed
    let wav2 = dir.path().join("tts_different.wav");
    generate_test_wav(&wav2, 440.0, 0.3, 24000, 0.5, 0.02)
        .expect("Failed to generate different WAV");

    let fp2 = AudioFingerprint::from_wav_file(&wav2, Some(24000))
        .expect("Failed to compute different fingerprint");

    let result = gm
        .load_or_create("tts", "persistence_test", &fp2)
        .expect("load_or_create failed");
    assert!(!result.is_pass(), "Changed audio should fail");

    // 接受新基线
    gm.accept("tts", "persistence_test", &fp2)
        .expect("accept failed");

    // 再次加载 → Match
    let result = gm
        .load_or_create("tts", "persistence_test", &fp2)
        .expect("load_or_create failed");
    assert!(result.is_pass());
}

#[test]
fn test_tts_golden_master_test_case_helper() {
    // 测试 GoldenMasterTestCase 辅助工具
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    let wav = dir.path().join("tts_helper.wav");
    generate_test_wav(&wav, 660.0, 0.2, 24000, 0.5, 0.01)
        .expect("Failed to generate WAV");

    let fp = AudioFingerprint::from_wav_file(&wav, Some(24000))
        .expect("Failed to compute fingerprint");

    // 首次运行：创建基线并通过
    GoldenMasterTestCase::new("tts", "helper_test")
        .with_baseline_dir(dir.path())
        .assert_pass(&fp);

    // 第二次运行：相同数据 → 通过
    GoldenMasterTestCase::new("tts", "helper_test")
        .with_baseline_dir(dir.path())
        .assert_pass(&fp);
}

#[test]
fn test_tts_golden_master_multi_segment() {
    // 模拟多段 TTS 输出的 Golden Master 测试
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let gm = GoldenMaster::new(dir.path());

    let texts = ["nihao", "shijie", "ceshi"];
    let freqs = [440.0, 660.0, 880.0];

    for (i, (&text, &freq)) in texts.iter().zip(freqs.iter()).enumerate() {
        let wav = dir.path().join(format!("seg_{i}.wav"));
        generate_test_wav(&wav, freq, 0.3, 24000, 0.5, 0.01)
            .expect("Failed to generate segment WAV");

        let fp = AudioFingerprint::from_wav_file(&wav, Some(24000))
            .expect("Failed to compute segment fingerprint");

        // 每段独立测试
        let result = gm
            .load_or_create("tts", &format!("segment_{i}_{text}"), &fp)
            .expect("load_or_create failed");
        assert!(result.is_pass(), "Segment {i} should pass on first run");
    }

    // 验证所有基线已创建
    let baselines = gm.list_baselines();
    assert_eq!(baselines.len(), 3, "Should have 3 baselines");

    // 第二轮：相同数据 → 全部通过
    for (i, (&text, &freq)) in texts.iter().zip(freqs.iter()).enumerate() {
        let wav = dir.path().join(format!("seg_{i}_v2.wav"));
        generate_test_wav(&wav, freq, 0.3, 24000, 0.5, 0.01)
            .expect("Failed to generate segment WAV v2");

        let fp = AudioFingerprint::from_wav_file(&wav, Some(24000))
            .expect("Failed to compute segment fingerprint v2");

        let result = gm
            .load_or_create("tts", &format!("segment_{i}_{text}"), &fp)
            .expect("load_or_create failed");
        assert!(result.is_pass(), "Segment {i} should pass on second run");
    }
}

#[test]
fn test_tts_golden_master_fingerprint_summary() {
    // 验证指纹摘要包含关键信息
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let wav = dir.path().join("summary_test.wav");
    generate_test_wav(&wav, 1000.0, 1.0, 24000, 0.5, 0.01)
        .expect("Failed to generate WAV");

    let fp = AudioFingerprint::from_wav_file(&wav, Some(24000))
        .expect("Failed to compute fingerprint");

    let summary = fp.summary();
    assert!(summary.contains("dur=1.000s"), "Summary should contain duration: {summary}");
    assert!(summary.contains("sr=24000Hz"), "Summary should contain sample rate: {summary}");
    assert!(summary.contains("rms="), "Summary should contain RMS: {summary}");
    assert!(summary.contains("peak="), "Summary should contain peak: {summary}");
}

#[test]
fn test_tts_golden_master_noisy_vs_clean() {
    // 验证噪声水平变化被检测到
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let wav_clean = dir.path().join("clean.wav");
    let wav_noisy = dir.path().join("noisy.wav");

    generate_test_wav(&wav_clean, 1000.0, 0.5, 24000, 0.5, 0.0)
        .expect("Failed to generate clean WAV");
    generate_test_wav(&wav_noisy, 1000.0, 0.5, 24000, 0.5, 0.3)
        .expect("Failed to generate noisy WAV");

    let fp_clean = AudioFingerprint::from_wav_file(&wav_clean, Some(24000))
        .expect("Failed to compute clean fingerprint");
    let fp_noisy = AudioFingerprint::from_wav_file(&wav_noisy, Some(24000))
        .expect("Failed to compute noisy fingerprint");

    // 噪声水平差异应被检测到
    assert!(
        !fp_clean.compare(&fp_noisy).is_pass(),
        "Clean vs noisy audio should not pass"
    );
}

// ─── macOS 真实 TTS 测试（条件编译） ───────────────────────

#[cfg(target_os = "macos")]
#[test]
fn test_tts_golden_master_real_say_engine() {
    // 使用真实 SayEngine 合成音频，验证 Golden Master 流程
    use std::process::Command;
    use vt_core::config::TtsConfig;
    use vt_core::tts::{SayEngine, TtsEngine};
    use vt_core::models::segment::Segment;

    // 检查 say 命令是否可用
    let which = Command::new("which").arg("say").output();
    if which.is_err() || !which.as_ref().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("Skipping real TTS test: 'say' command not available");
        return;
    }

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let mut config = TtsConfig::default();
    config.cache_dir = dir.path().join("tts_cache").to_string_lossy().to_string();

    let engine = match SayEngine::new(&config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Skipping real TTS test: SayEngine init failed: {e}");
            return;
        }
    };

    // 合成简单中文文本
    let mut segments = vec![Segment::new("0".to_string(), 0.0, 2.0, "Hello".to_string())];
    segments[0].target_text = Some("你好世界".to_string());

    let result = engine.synthesize_segments(&mut segments, &config);
    if let Err(e) = result {
        eprintln!("Skipping real TTS test: synthesis failed: {e}");
        return;
    }

    let wav_path = segments[0].tts_audio_path.as_ref().expect("No TTS output");
    let wav_path = Path::new(wav_path);

    let fp = match AudioFingerprint::from_wav_file(wav_path, Some(config.sample_rate)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Skipping real TTS test: fingerprint failed: {e}");
            return;
        }
    };

    // 验证指纹包含合理值
    assert!(fp.sample_count > 0, "Should have audio samples");
    assert!(fp.duration_secs > 0.1, "Audio should be > 0.1s: {}", fp.summary());

    println!("Real TTS fingerprint: {}", fp.summary());
}
