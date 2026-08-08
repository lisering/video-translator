//! 音频后处理模块
//!
//! 提供独立的音频后处理功能，用于改善 TTS 合成音频的听感质量。
//!
//! # 功能概览
//! - [`AudioPostProcessor`]: 核心后处理器，封装 ffmpeg 滤镜链
//! - 采样率统一：输出 48000Hz（可配置）
//! - 齿音消除：6-8kHz 高架滤波器，衰减 3-4dB
//! - 淡入淡出：相邻片段交叉淡入淡出（50ms），消除拼接感
//! - 自动增益控制：确保音量一致
//! - 低频增强：200-500Hz 中低频提升，增强温暖感
//!
//! # 设计原则
//! - **最小干预**：避免过度处理导致音质失真
//! - **可配置**：所有参数通过 [`TtsConfig`] 控制
//! - **独立模块**：可被任意 TTS 引擎调用
//!
//! # 示例
//! ```no_run
//! use vt_core::audio_post_process::AudioPostProcessor;
//! use vt_core::config::TtsConfig;
//! use vt_core::error::AppResult;
//! use std::path::Path;
//!
//! fn process() -> AppResult<()> {
//!     let config = TtsConfig::default();
//!     let processor = AudioPostProcessor::new(&config);
//!     processor.process(
//!         Path::new("input.wav"),
//!         Path::new("output.wav"),
//!     )?;
//!     Ok(())
//! }
//! ```

use std::path::Path;
use std::process::Command;

use crate::config::TtsConfig;
use crate::error::{AppError, AppResult};
use crate::voice_manager::{VoiceInfo, VoiceManager};

// ─── 常量 ─────────────────────────────────────────────────

/// `say` 命令基础输出采样率
pub const SAY_BASE_SAMPLE_RATE: u32 = 24000;

/// 高频均衡器截止频率（Hz），用于齿音衰减
pub const EQ_HIGH_SHELF_FREQ: u32 = 6000;

/// 低频均衡器截止频率（Hz），用于增强温暖感
pub const EQ_LOW_SHELF_FREQ: u32 = 300;

/// 低频均衡器增益（dB），轻微提升中低频
pub const EQ_LOW_SHELF_GAIN: f64 = 2.0;

/// 默认交叉淡入淡出时长（毫秒）
pub const DEFAULT_CROSSFADE_MS: u64 = 50;

// ─── AudioPostProcessor ──────────────────────────────────

/// 音频后处理器
///
/// 封装 ffmpeg 滤镜链，提供统一的音频后处理接口。
/// 可被任意 TTS 引擎调用，用于改善合成音频的听感质量。
///
/// # 处理流程
/// 1. 音调偏移（`asetrate` + `atempo`，可选）
/// 2. 音量调整（`volume`，可选）
/// 3. 高频衰减（`highshelf`，减少齿音）
/// 4. 低频增强（`lowshelf`，增强温暖感）
/// 5. 自动增益控制（`dynaudnorm`）
/// 6. 淡入淡出（`afade` + `areverse`，消除拼接感）
/// 7. 重采样到目标采样率
///
/// # 线程安全
/// 仅包含不可变数据（`TtsConfig` 引用），天然线程安全。
pub struct AudioPostProcessor<'a> {
    /// TTS 配置（读取后处理参数）
    config: &'a TtsConfig,
}

impl<'a> AudioPostProcessor<'a> {
    /// 创建音频后处理器
    ///
    /// # 参数
    /// - `config`: TTS 配置（读取 `eq_high_shelf_db`、`crossfade_duration_ms` 等）
    #[must_use]
    pub fn new(config: &'a TtsConfig) -> Self {
        Self { config }
    }

    /// 构建 ffmpeg 音频滤镜链
    ///
    /// 滤镜顺序：
    /// 1. `asetrate` + `atempo`：音调偏移
    /// 2. `volume`：音量调整
    /// 3. `highshelf`：衰减 6kHz 以上高频，减少齿音
    /// 4. `lowshelf`：提升 200-500Hz 中低频，增强温暖感
    /// 5. `dynaudnorm`：自动增益控制
    /// 6. `afade` + `areverse`：淡入淡出（消除拼接感）
    ///
    /// # 参数
    /// - `voice`: 音色信息（读取 `pitch_multiplier`）
    /// - `config`: TTS 配置
    #[must_use]
    pub fn build_filter_chain(voice: &VoiceInfo, config: &TtsConfig) -> String {
        let combined = Self::combined_pitch(voice, config);
        let mut filters: Vec<String> = Vec::new();

        // 1. 音调偏移（asetrate 改变采样率从而改变音调，atempo 补偿语速）
        if (combined - 1.0).abs() > 0.001 {
            let new_rate = (SAY_BASE_SAMPLE_RATE as f32 * combined) as u32;
            let tempo = 1.0 / combined;
            filters.push(format!("asetrate={new_rate},atempo={tempo:.4}"));
        }

        // 2. 音量调整
        if (config.volume - 1.0).abs() > 0.001 {
            filters.push(format!("volume={:.4}", config.volume));
        }

        // 3. 高频均衡器：衰减 6kHz 以上高频，减少齿音（sibilance）
        if (config.eq_high_shelf_db - 0.0).abs() > 0.01 {
            filters.push(format!(
                "highshelf=f={EQ_HIGH_SHELF_FREQ}:g={:.2}",
                config.eq_high_shelf_db
            ));
        }

        // 4. 低频均衡器：提升 200-500Hz 中低频，增强温暖感
        filters.push(format!(
            "lowshelf=f={EQ_LOW_SHELF_FREQ}:g={EQ_LOW_SHELF_GAIN:.2}"
        ));

        // 5. 自动增益控制（始终应用，避免音量过大或过小）
        filters.push("dynaudnorm=f=150:g=15:p=0.9".to_string());

        // 6. 淡入淡出（消除爆破音和拼接感）
        let fade_secs = (config.crossfade_duration_ms as f64) / 1000.0;
        let fade_str = format!("{fade_secs:.3}");
        filters.push(format!("afade=t=in:st=0:d={fade_str}"));
        filters.push("areverse".to_string());
        filters.push(format!("afade=t=in:st=0:d={fade_str}"));
        filters.push("areverse".to_string());

        filters.join(",")
    }

    /// 计算组合音调倍率
    ///
    /// 组合音调 = 音色基础音调倍率 × 用户配置的音调倍率
    /// 结果限制在 [0.5, 2.0] 范围内。
    #[must_use]
    pub fn combined_pitch(voice: &VoiceInfo, config: &TtsConfig) -> f32 {
        (voice.pitch_multiplier * config.pitch).clamp(0.5, 2.0)
    }

    /// 对音频文件进行后处理
    ///
    /// 将输入 WAV 通过 ffmpeg 滤镜链处理为目标格式。
    ///
    /// # 参数
    /// - `input`: 输入 WAV 文件路径
    /// - `output`: 输出 WAV 文件路径
    ///
    /// # 错误
    /// - [`AppError::TtsError`][]: ffmpeg 执行失败
    pub fn process(&self, input: &Path, output: &Path) -> AppResult<()> {
        let vm = VoiceManager::new();
        let voice = vm.default_voice();
        self.process_with_voice(input, output, voice)
    }

    /// 使用指定音色对音频文件进行后处理
    ///
    /// # 参数
    /// - `input`: 输入 WAV 文件路径
    /// - `output`: 输出 WAV 文件路径
    /// - `voice`: 音色信息
    ///
    /// # 错误
    /// - [`AppError::TtsError`][]: ffmpeg 执行失败
    pub fn process_with_voice(
        &self,
        input: &Path,
        output: &Path,
        voice: &VoiceInfo,
    ) -> AppResult<()> {
        let filter_chain = Self::build_filter_chain(voice, self.config);

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .arg("-i")
            .arg(input)
            .arg("-af")
            .arg(&filter_chain)
            .arg("-ar")
            .arg(self.config.sample_rate.to_string())
            .arg("-ac")
            .arg("1")
            .arg("-c:a")
            .arg("pcm_s16le")
            .arg(output);

        tracing::debug!(
            "AudioPostProcessor: filter='{}', sample_rate={}, output={:?}",
            filter_chain,
            self.config.sample_rate,
            output
        );

        let output_result = cmd.output().map_err(|e| {
            AppError::TtsError(format!(
                "Failed to execute ffmpeg for post-processing: {e}\n\
                 确保 ffmpeg 已安装且在 PATH 中。"
            ))
        })?;

        if !output_result.status.success() {
            let stderr = String::from_utf8_lossy(&output_result.stderr);
            return Err(AppError::TtsError(format!(
                "ffmpeg post-processing failed (exit code: {}): {stderr}",
                output_result.status.code().unwrap_or(-1)
            )));
        }

        Ok(())
    }

    /// 验证 WAV 文件格式（指定采样率 mono 16-bit PCM）
    ///
    /// # 参数
    /// - `path`: WAV 文件路径
    /// - `expected_sample_rate`: 期望的采样率
    ///
    /// # 错误
    /// - [`AppError::TtsAudioEncodeError`][]: WAV 格式验证失败
    pub fn validate_wav(path: &Path, expected_sample_rate: u32) -> AppResult<()> {
        let mut reader = hound::WavReader::open(path).map_err(|e| {
            AppError::TtsAudioEncodeError(format!(
                "Output WAV file is invalid: {e}\n\
                 文件: {path:?}"
            ))
        })?;
        let spec = reader.spec();

        if spec.sample_rate != expected_sample_rate
            || spec.channels != 1
            || spec.bits_per_sample != 16
        {
            return Err(AppError::TtsAudioEncodeError(format!(
                "Output WAV format mismatch: expected {}Hz mono 16-bit, got {}Hz {}ch {}bit\n\
                 文件: {path:?}",
                expected_sample_rate, spec.sample_rate, spec.channels, spec.bits_per_sample
            )));
        }

        // 确保有实际音频数据
        let sample_count: usize = reader.samples::<i16>().count();
        if sample_count == 0 {
            return Err(AppError::TtsAudioEncodeError(format!(
                "Output WAV contains no audio samples\n\
                 文件: {path:?}"
            )));
        }

        Ok(())
    }

    /// 移除音频文件头尾的静音段
    ///
    /// 参考 pyvideotrans 的 `remove_silence_wav()` 函数，
    /// 使用 ffmpeg `silenceremove` 滤镜移除音频开头和结尾的静音。
    ///
    /// # 参数
    /// - `input_path`: 输入 WAV 文件路径（原地修改）
    ///
    /// # 行为
    /// - 检测 < -40dB 的音频为静音
    /// - 开头静音超过 0.1s 移除
    /// - 结尾静音超过 0.1s 移除
    /// - 中间静音保留（不影响语音节奏）
    ///
    /// # 错误
    /// - [`AppError::TtsError`][]: ffmpeg 执行失败
    pub fn remove_silence(input_path: &Path) -> AppResult<()> {
        if !input_path.exists() {
            return Err(AppError::FileNotFound(input_path.to_path_buf()));
        }

        let tmp_path = input_path.with_extension("tmp_silence_removed.wav");
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .arg("-i")
            .arg(input_path)
            .arg("-af")
            .arg("silenceremove=start_periods=1:start_duration=0.1:start_threshold=-40dB:stop_periods=0:stop_duration=0.1:stop_threshold=-40dB")
            .arg("-c:a")
            .arg("pcm_s16le")
            .arg(&tmp_path);

        let output = cmd.output().map_err(|e| {
            AppError::TtsError(format!("Failed to execute ffmpeg for silence removal: {e}"))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!("Silence removal failed (non-fatal): {stderr}");
            // 静音移除失败不影响主流程
            let _ = std::fs::remove_file(&tmp_path);
            return Ok(());
        }

        // 替换原文件
        std::fs::rename(&tmp_path, input_path).map_err(|e| {
            AppError::TtsError(format!(
                "Failed to replace audio after silence removal: {e}"
            ))
        })?;

        tracing::debug!("Silence removed from {:?}", input_path);
        Ok(())
    }

    /// 混合背景音乐到配音音频
    ///
    /// 参考 pyvideotrans 的背景音处理逻辑，
    /// 使用 ffmpeg `amix` 滤镜将背景音和配音混合。
    ///
    /// # 参数
    /// - `voice_path`: 配音音频路径
    /// - `bgm_path`: 背景音乐路径
    /// - `output_path`: 输出音频路径
    /// - `bgm_volume`: 背景音量（0.0-1.0，默认 0.2 = 背景音降低到 20%）
    /// - `loop_bgm`: 是否循环背景音以匹配配音长度
    ///
    /// # 错误
    /// - [`AppError::TtsError`][]: ffmpeg 执行失败
    pub fn mix_background_music(
        voice_path: &Path,
        bgm_path: &Path,
        output_path: &Path,
        bgm_volume: f32,
        loop_bgm: bool,
    ) -> AppResult<()> {
        if !voice_path.exists() {
            return Err(AppError::FileNotFound(voice_path.to_path_buf()));
        }
        if !bgm_path.exists() {
            return Err(AppError::FileNotFound(bgm_path.to_path_buf()));
        }

        let bgm_volume = bgm_volume.clamp(0.0, 1.0);

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y").arg("-i").arg(voice_path);

        if loop_bgm {
            // 循环背景音以匹配配音长度
            cmd.arg("-stream_loop").arg("-1").arg("-i").arg(bgm_path);
        } else {
            cmd.arg("-i").arg(bgm_path);
        }

        // amix 滤镜混合音频
        // inputs=2, duration=first (匹配配音长度), weights 调整音量
        cmd.arg("-filter_complex")
            .arg(format!(
                "[1:a]volume={bgm_volume:.2}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            ))
            .arg("-map")
            .arg("[aout]")
            .arg("-c:a")
            .arg("pcm_s16le")
            .arg(output_path);

        let output = cmd
            .output()
            .map_err(|e| AppError::TtsError(format!("Failed to mix background music: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!("Background music mixing failed (non-fatal): {stderr}");
            // 背景音混合失败时直接使用纯配音
            let _ = std::fs::copy(voice_path, output_path);
            return Ok(());
        }

        tracing::debug!("Background music mixed into {:?}", output_path);
        Ok(())
    }

    /// 延长或循环背景音以匹配目标时长
    ///
    /// 参考 pyvideotrans 的 `loop_backaudio` 功能：
    /// - loop 模式：循环播放背景音
    /// - stretch 模式：拉长（降速播放）背景音
    pub fn extend_background_music(
        bgm_path: &Path,
        output_path: &Path,
        target_duration_secs: f64,
        stretch: bool,
    ) -> AppResult<()> {
        if !bgm_path.exists() {
            return Err(AppError::FileNotFound(bgm_path.to_path_buf()));
        }

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y").arg("-i").arg(bgm_path);

        if stretch {
            // 拉长模式：使用 atempo 降速
            // 先获取背景音时长，计算倍率
            let bgm_dur =
                crate::media::get_audio_duration(bgm_path).unwrap_or(target_duration_secs);
            let rate = (bgm_dur / target_duration_secs).clamp(0.5, 2.0);
            cmd.arg("-af")
                .arg(format!("atempo={rate:.4}"))
                .arg("-t")
                .arg(format!("{target_duration_secs:.3}"));
        } else {
            // 循环模式
            cmd.arg("-stream_loop")
                .arg("-1")
                .arg("-t")
                .arg(format!("{target_duration_secs:.3}"));
        }

        cmd.arg("-c:a").arg("pcm_s16le").arg(output_path);

        let output = cmd
            .output()
            .map_err(|e| AppError::TtsError(format!("Failed to extend background music: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!("Background music extension failed (non-fatal): {stderr}");
            let _ = std::fs::copy(bgm_path, output_path);
            return Ok(());
        }

        tracing::debug!("Background music extended to {target_duration_secs:.1}s");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice_manager::{VoiceGender, VoiceManager};

    /// 验证组合音调倍率计算（女声 + 默认音调）。
    #[test]
    fn test_combined_pitch_female_default() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig::default();
        assert!((AudioPostProcessor::combined_pitch(voice, &config) - 1.0).abs() < 0.001);
    }

    /// 验证组合音调倍率计算（男声 + 默认音调）。
    #[test]
    fn test_combined_pitch_male_default() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("zhiming").expect("zhiming should exist");
        let config = TtsConfig::default();
        // 男声 pitch=0.85, config pitch=1.0 → combined=0.85
        assert!((AudioPostProcessor::combined_pitch(voice, &config) - 0.85).abs() < 0.001);
    }

    /// 验证组合音调倍率计算（女声 + 用户音调）。
    #[test]
    fn test_combined_pitch_female_with_user_pitch() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig {
            pitch: 1.1,
            ..Default::default()
        };
        assert!((AudioPostProcessor::combined_pitch(voice, &config) - 1.1).abs() < 0.001);
    }

    /// 验证滤镜链包含齿音衰减（highshelf）。
    #[test]
    fn test_filter_chain_contains_highshelf() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig::default();
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        assert!(
            chain.contains("highshelf"),
            "Should contain highshelf for sibilance reduction"
        );
    }

    /// 验证滤镜链包含低频增强（lowshelf）。
    #[test]
    fn test_filter_chain_contains_lowshelf() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig::default();
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        assert!(
            chain.contains("lowshelf"),
            "Should contain lowshelf for warmth enhancement"
        );
    }

    /// 验证滤镜链包含自动增益控制（dynaudnorm）。
    #[test]
    fn test_filter_chain_contains_dynaudnorm() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig::default();
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        assert!(
            chain.contains("dynaudnorm"),
            "Should contain dynaudnorm for AGC"
        );
    }

    /// 验证滤镜链包含淡入淡出（afade + areverse）。
    #[test]
    fn test_filter_chain_contains_fade() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig::default();
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        assert!(chain.contains("afade"), "Should contain afade");
        assert!(chain.contains("areverse"), "Should contain areverse");
    }

    /// 验证男声滤镜链包含音调偏移（asetrate + atempo）。
    #[test]
    fn test_filter_chain_male_contains_pitch_shift() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("zhiming").expect("zhiming should exist");
        let config = TtsConfig::default();
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        assert!(
            chain.contains("asetrate"),
            "Should contain asetrate for pitch shift"
        );
        assert!(
            chain.contains("atempo"),
            "Should contain atempo for speed compensation"
        );
    }

    /// 验证女声默认参数不包含音调偏移。
    #[test]
    fn test_filter_chain_female_no_pitch_shift() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig::default();
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        assert!(
            !chain.contains("asetrate"),
            "Should not contain asetrate for default female voice"
        );
    }

    /// 验证音量调整滤镜在非默认音量时出现。
    #[test]
    fn test_filter_chain_volume_adjustment() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig {
            volume: 1.5,
            ..Default::default()
        };
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        assert!(
            chain.contains("volume=1.5000"),
            "Should contain volume filter"
        );
    }

    /// 验证禁用高频衰减时 highshelf 不出现。
    #[test]
    fn test_filter_chain_disable_highshelf() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig {
            eq_high_shelf_db: 0.0,
            ..Default::default()
        };
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        assert!(
            !chain.contains("highshelf"),
            "Should not contain highshelf when eq_high_shelf_db is 0"
        );
    }

    /// 验证自定义淡入淡出时长。
    #[test]
    fn test_filter_chain_custom_crossfade() {
        let vm = VoiceManager::new();
        let voice = vm.find_by_id("tingting").expect("tingting should exist");
        let config = TtsConfig {
            crossfade_duration_ms: 100,
            ..Default::default()
        };
        let chain = AudioPostProcessor::build_filter_chain(voice, &config);
        // 100ms = 0.100 seconds
        assert!(
            chain.contains("d=0.100"),
            "Should contain custom fade duration, got: {chain}"
        );
    }

    /// 验证 VoiceManager 提供至少 2 女 + 2 男声。
    #[test]
    fn test_voice_manager_has_enough_voices() {
        let vm = VoiceManager::new();
        let females = vm.voices_by_gender(VoiceGender::Female);
        let males = vm.voices_by_gender(VoiceGender::Male);
        assert!(females.len() >= 2, "Should have at least 2 female voices");
        assert!(males.len() >= 2, "Should have at least 2 male voices");
    }
}

// ─── 响度归一化 — 借鉴 MOSS-TTS pipeline.py ────────────────

/// RMS 响度归一化
///
/// 借鉴 MOSS-TTS 项目 `moss_tts_delay/llama_cpp/pipeline.py` 中的
/// `loudness_normalize()` 函数。
///
/// 将音频 RMS 电平调整到目标 dBFS，确保所有 TTS 段音量一致。
///
/// # 参数
/// - `wav`: 音频波形（float32）
/// - `target_dbfs`: 目标 dBFS（默认 -20.0）
/// - `gain_range`: 增益范围限制（dB），防止极端值
///
/// # 返回
/// 归一化后的音频波形
pub fn loudness_normalize(wav: &[f32], target_dbfs: f32, gain_range: (f32, f32)) -> Vec<f32> {
    if wav.is_empty() {
        return vec![];
    }

    // 计算 RMS
    let sum_sq: f64 = wav.iter().map(|&x| x as f64 * x as f64).sum();
    let rms = (sum_sq / wav.len() as f64 + 1e-9f64).sqrt() as f32;

    // 计算当前 dBFS
    let current_dbfs = 20.0 * rms.log10();

    // 计算需要的增益
    let mut gain = target_dbfs - current_dbfs;
    gain = gain_range.0.max(gain_range.1.min(gain));

    // 应用增益
    let factor = 10.0f32.powf(gain / 20.0);
    wav.iter().map(|&x| x * factor).collect()
}

/// 从 WAV 文件读取单声道 float32 波形
fn read_wav_float32(path: &Path) -> AppResult<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| AppError::AudioDecodeError(format!("WAV open error: {e}")))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .enumerate()
                .filter(|(i, _)| i % channels == 0) // 取第一个声道
                .filter_map(|(_, s)| Some(s as f32 / max_val))
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .filter_map(|s| s.ok())
            .enumerate()
            .filter(|(i, _)| i % channels == 0)
            .map(|(_, s)| s)
            .collect(),
    };

    Ok((samples, sample_rate))
}

/// 将 float32 波形写入 WAV 文件
fn write_wav_float32(path: &Path, samples: &[f32], sample_rate: u32) -> AppResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| AppError::AudioDecodeError(format!("mkdir error: {e}")))?;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let max_val = (1i64 << 15) as f32;
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| AppError::AudioDecodeError(format!("WAV create error: {e}")))?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * max_val) as i16;
        writer
            .write_sample(i16_sample)
            .map_err(|e| AppError::AudioDecodeError(format!("WAV write error: {e}")))?;
    }

    writer
        .finalize()
        .map_err(|e| AppError::AudioDecodeError(format!("WAV finalize error: {e}")))?;

    Ok(())
}

/// 对 WAV 文件进行响度归一化
///
/// 读取 WAV → 响度归一化 → 写回 WAV
pub fn normalize_wav_loudness(
    input_path: &Path,
    output_path: &Path,
    target_dbfs: f32,
    gain_range: (f32, f32),
) -> AppResult<()> {
    let (samples, sample_rate) = read_wav_float32(input_path)?;
    let normalized = loudness_normalize(&samples, target_dbfs, gain_range);
    write_wav_float32(output_path, &normalized, sample_rate)?;
    Ok(())
}

#[cfg(test)]
mod loudness_tests {
    use super::*;

    #[test]
    fn test_loudness_normalize_basic() {
        let wav = vec![0.1; 1000];
        let result = loudness_normalize(&wav, -20.0, (-3.0, 3.0));
        // 0.1 RMS ≈ -20 dBFS, target = -20 dBFS → gain ≈ 0
        // Values should be close to original (within gain range)
        assert!(!result.is_empty());
        // Check that the RMS of the result is close to -20 dBFS
        let rms: f32 =
            (result.iter().map(|x| (x * x) as f32).sum::<f32>() / result.len() as f32).sqrt();
        let dbfs = 20.0 * rms.log10();
        assert!(
            (dbfs - (-20.0)).abs() < 1.0,
            "Expected ~-20 dBFS, got {dbfs}"
        );
    }

    #[test]
    fn test_loudness_normalize_clamped() {
        // 极低音频应该被增益范围限制
        let wav = vec![0.001; 1000];
        let result = loudness_normalize(&wav, -20.0, (-3.0, 3.0));
        // 增益不应超过 3dB
        let max_expected = 0.001 * 10.0f32.powf(3.0 / 20.0) * 1.1;
        assert!(result.iter().all(|&x| x < max_expected));
    }

    #[test]
    fn test_loudness_normalize_empty() {
        let result = loudness_normalize(&[], -20.0, (-3.0, 3.0));
        assert!(result.is_empty());
    }

    #[test]
    fn test_loudness_normalize_already_loud() {
        // 已经很响的音频应该被降低
        let wav = vec![0.9; 1000];
        let result = loudness_normalize(&wav, -20.0, (-10.0, 10.0));
        // 应该降低
        assert!(result[0] < 0.9);
    }
}

// ─── RMS 音量匹配 — 借鉴 OmniVoice create_voice_clone_prompt ───

/// 计算音频 RMS
#[must_use]
pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&x| x as f64 * x as f64).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// 根据参考音频 RMS 调整生成音频的音量
///
/// 借鉴 OmniVoice `_post_process_audio` 的音量匹配逻辑:
/// - 如果 ref_rms < 0.1, 将生成音频缩放到 ref_rms / 0.1 的比例
/// - 如果 ref_rms >= 0.1, 保持原样
/// - 如果 ref_rms 为 None, 按峰值归一化到 0.5
///
/// # 参数
/// - `samples`: 生成的音频波形
/// - `ref_rms`: 参考音频的 RMS（可选）
///
/// # 返回
/// 音量调整后的音频波形
#[must_use]
pub fn match_rms_volume(samples: &[f32], ref_rms: Option<f32>) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
    }

    match ref_rms {
        Some(rms) if rms > 0.0 && rms < 0.1 => {
            let scale = rms / 0.1;
            samples.iter().map(|&x| x * scale).collect()
        }
        Some(_) => samples.to_vec(), // ref_rms >= 0.1, 保持原样
        None => {
            // 按峰值归一化到 0.5
            let peak = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            if peak > 1e-6 {
                let scale = 0.5 / peak;
                samples.iter().map(|&x| x * scale).collect()
            } else {
                samples.to_vec()
            }
        }
    }
}

/// 对 WAV 文件进行 RMS 音量匹配
///
/// 读取 WAV → 匹配 RMS → 写回 WAV
pub fn match_wav_rms_volume(
    input_path: &Path,
    output_path: &Path,
    ref_rms: Option<f32>,
) -> AppResult<()> {
    let (samples, sample_rate) = read_wav_float32(input_path)?;
    let adjusted = match_rms_volume(&samples, ref_rms);
    write_wav_float32(output_path, &adjusted, sample_rate)?;
    Ok(())
}

// ─── 淡入淡出 + 静音填充 — 借鉴 OmniVoice fade_and_pad_audio ──

/// 对音频应用淡入淡出和静音填充
///
/// 借鉴 OmniVoice `fade_and_pad_audio`:
/// 1. 首尾添加 `pad_duration` 秒的静音
/// 2. 首尾应用 `fade_duration` 秒的线性淡入淡出
///
/// 防止拼接时产生 click 声，改善听感。
///
/// # 参数
/// - `samples`: 音频波形
/// - `sample_rate`: 采样率
/// - `pad_duration`: 每侧静音填充时长（秒，默认 0.1）
/// - `fade_duration`: 淡入淡出时长（秒，默认 0.1）
///
/// # 返回
/// 处理后的音频波形
#[must_use]
pub fn fade_and_pad(
    samples: &[f32],
    sample_rate: u32,
    pad_duration: f32,
    fade_duration: f32,
) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
    }

    let fade_samples = (fade_duration * sample_rate as f32) as usize;
    let pad_samples = (pad_duration * sample_rate as f32) as usize;

    let mut result = samples.to_vec();

    // 应用淡入淡出
    if fade_samples > 0 {
        let k = fade_samples.min(result.len() / 2);
        if k > 0 {
            // 线性淡入
            for i in 0..k {
                let w = i as f32 / k as f32;
                result[i] *= w;
            }
            // 线性淡出
            let start = result.len() - k;
            for i in 0..k {
                let w = 1.0 - (i as f32 / k as f32);
                result[start + i] *= w;
            }
        }
    }

    // 添加静音填充
    if pad_samples > 0 {
        let mut padded = Vec::with_capacity(result.len() + pad_samples * 2);
        padded.extend(std::iter::repeat_n(0.0f32, pad_samples));
        padded.extend_from_slice(&result);
        padded.extend(std::iter::repeat_n(0.0f32, pad_samples));
        result = padded;
    }

    result
}

/// 对 WAV 文件应用淡入淡出和静音填充
pub fn fade_and_pad_wav(
    input_path: &Path,
    output_path: &Path,
    pad_duration: f32,
    fade_duration: f32,
) -> AppResult<()> {
    let (samples, sample_rate) = read_wav_float32(input_path)?;
    let processed = fade_and_pad(&samples, sample_rate, pad_duration, fade_duration);
    write_wav_float32(output_path, &processed, sample_rate)?;
    Ok(())
}

// ─── 中间静音移除 — 借鉴 OmniVoice remove_silence ───────────

/// 检测并移除音频中间过长的静音段
///
/// 借鉴 OmniVoice `remove_silence`:
/// 1. 按帧（20ms）扫描音频，检测 RMS 低于阈值的帧为静音
/// 2. 连续静音帧超过 `mid_sil_ms` 时，只保留 `keep_sil_ms` 的静音
/// 3. 同时修剪首尾静音到 `lead_sil_ms` / `trail_sil_ms`
///
/// 与 ffmpeg `silenceremove` 不同，此函数是纯 Rust 实现，
/// 能精确控制中间静音的裁剪行为。
///
/// # 参数
/// - `samples`: 音频波形
/// - `sample_rate`: 采样率
/// - `mid_sil_ms`: 中间静音阈值（毫秒，超过此长度的静音被压缩，默认 300）
/// - `lead_sil_ms`: 保留的开头静音（毫秒，默认 100）
/// - `trail_sil_ms`: 保留的结尾静音（毫秒，默认 100）
/// - `silence_threshold_db`: 静音检测阈值（dBFS，默认 -50）
///
/// # 返回
/// 处理后的音频波形
#[must_use]
pub fn remove_mid_silence(
    samples: &[f32],
    sample_rate: u32,
    mid_sil_ms: u32,
    lead_sil_ms: u32,
    trail_sil_ms: u32,
    silence_threshold_db: f32,
) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
    }

    let frame_size = (sample_rate as f64 * 0.02) as usize; // 20ms 帧
    if frame_size == 0 {
        return samples.to_vec();
    }

    let threshold_amp = 10.0f32.powf(silence_threshold_db / 20.0);
    let mid_sil_frames = (mid_sil_ms as f64 / 20.0) as usize;
    let keep_sil_frames = mid_sil_frames / 3; // 保留 1/3 的静音
    let lead_sil_frames = (lead_sil_ms as f64 / 20.0) as usize;
    let trail_sil_frames = (trail_sil_ms as f64 / 20.0) as usize;

    // 1. 标记每帧是否为静音
    let total_frames = samples.len().div_ceil(frame_size);
    let mut is_silent: Vec<bool> = Vec::with_capacity(total_frames);
    for i in 0..total_frames {
        let start = i * frame_size;
        let end = (start + frame_size).min(samples.len());
        let frame = &samples[start..end];
        let frame_rms = compute_rms(frame);
        is_silent.push(frame_rms < threshold_amp);
    }

    // 2. 找到第一个和最后一个非静音帧
    let first_non_silent = is_silent.iter().position(|&s| !s);
    let last_non_silent = is_silent.iter().rposition(|&s| !s);

    let first_non_silent = match first_non_silent {
        Some(idx) => idx,
        None => return samples.to_vec(), // 全静音，返回原样
    };
    let last_non_silent = last_non_silent.unwrap();

    // 3. 构建输出：保留开头静音 → 中间（压缩长静音）→ 保留结尾静音
    let mut result: Vec<f32> = Vec::with_capacity(samples.len());

    // 保留开头静音（最多 lead_sil_frames 帧）
    let lead_start = first_non_silent.saturating_sub(lead_sil_frames);
    let lead_start_sample = lead_start * frame_size;
    result.extend_from_slice(&samples[..lead_start_sample.min(samples.len())]);

    // 中间部分：压缩连续静音
    // 策略：不立即输出静音帧，而是缓冲。遇到非静音帧时，
    // 只保留缓冲中前 keep_sil_frames 帧的静音。
    let mut i = first_non_silent;
    let mut silent_buffer: Vec<usize> = Vec::new();
    while i <= last_non_silent {
        if is_silent[i] {
            silent_buffer.push(i);
        } else {
            // 输出缓冲的静音帧（最多 keep_sil_frames 帧）
            let keep = silent_buffer.iter().take(keep_sil_frames.max(1));
            for &frame_idx in keep {
                let fs = frame_idx * frame_size;
                let fe = (fs + frame_size).min(samples.len());
                result.extend_from_slice(&samples[fs..fe]);
            }
            silent_buffer.clear();

            // 输出当前非静音帧
            let frame_start = i * frame_size;
            let frame_end = (frame_start + frame_size).min(samples.len());
            result.extend_from_slice(&samples[frame_start..frame_end]);
        }
        i += 1;
    }
    // 输出末尾缓冲的静音（最多 keep_sil_frames 帧）
    let keep = silent_buffer.iter().take(keep_sil_frames.max(1));
    for &frame_idx in keep {
        let fs = frame_idx * frame_size;
        let fe = (fs + frame_size).min(samples.len());
        result.extend_from_slice(&samples[fs..fe]);
    }

    // 保留结尾静音（最多 trail_sil_frames 帧）
    // 注意：必须基于原始 samples 的帧索引，不能用 result.len()
    let trail_start_frame = last_non_silent + 1;
    let trail_end_frame = (trail_start_frame + trail_sil_frames).min(total_frames);
    let trail_start_sample = (trail_start_frame * frame_size).min(samples.len());
    let trail_end_sample = (trail_end_frame * frame_size).min(samples.len());
    if trail_end_sample > trail_start_sample {
        result.extend_from_slice(&samples[trail_start_sample..trail_end_sample]);
    }

    result
}

/// 对 WAV 文件进行中间静音移除
pub fn remove_mid_silence_wav(
    input_path: &Path,
    output_path: &Path,
    mid_sil_ms: u32,
    lead_sil_ms: u32,
    trail_sil_ms: u32,
    silence_threshold_db: f32,
) -> AppResult<()> {
    let (samples, sample_rate) = read_wav_float32(input_path)?;
    let processed = remove_mid_silence(
        &samples,
        sample_rate,
        mid_sil_ms,
        lead_sil_ms,
        trail_sil_ms,
        silence_threshold_db,
    );
    write_wav_float32(output_path, &processed, sample_rate)?;
    Ok(())
}

// ─── Phase 1 测试 ─────────────────────────────────────────

#[cfg(test)]
mod omni_phase1_tests {
    use super::*;

    // ── RMS 音量匹配测试 ──────────────────────────────

    #[test]
    fn test_compute_rms_empty() {
        assert_eq!(compute_rms(&[]), 0.0);
    }

    #[test]
    fn test_compute_rms_constant() {
        let wav = vec![0.5; 1000];
        let rms = compute_rms(&wav);
        assert!((rms - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_match_rms_volume_low_rms() {
        // ref_rms < 0.1 时应缩放
        let wav = vec![0.5; 1000];
        let result = match_rms_volume(&wav, Some(0.05));
        // scale = 0.05 / 0.1 = 0.5
        assert!(
            (result[0] - 0.25).abs() < 0.01,
            "Expected 0.25, got {}",
            result[0]
        );
    }

    #[test]
    fn test_match_rms_volume_high_rms() {
        // ref_rms >= 0.1 时应保持原样
        let wav = vec![0.5; 1000];
        let result = match_rms_volume(&wav, Some(0.3));
        assert!((result[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_match_rms_volume_none() {
        // ref_rms = None 时按峰值归一化到 0.5
        let wav = vec![0.8; 1000];
        let result = match_rms_volume(&wav, None);
        // peak = 0.8, scale = 0.5/0.8 = 0.625
        assert!(
            (result[0] - 0.5).abs() < 0.01,
            "Expected ~0.5, got {}",
            result[0]
        );
    }

    #[test]
    fn test_match_rms_volume_empty() {
        assert!(match_rms_volume(&[], Some(0.05)).is_empty());
    }

    // ── 淡入淡出 + 静音填充测试 ────────────────────────

    #[test]
    fn test_fade_and_pad_basic() {
        let wav = vec![0.5; 4800]; // 0.2s @ 24kHz
        let result = fade_and_pad(&wav, 24000, 0.05, 0.05);
        // pad: 0.05s * 24000 * 2 = 2400 samples
        // total = 4800 + 2400 = 7200
        assert_eq!(result.len(), 7200);
        // 开头应该是静音
        assert!(result[0].abs() < 0.001);
        // 中间应该有信号
        assert!(result[3600].abs() > 0.01);
    }

    #[test]
    fn test_fade_and_pad_fade_applied() {
        let wav = vec![1.0; 4800];
        let result = fade_and_pad(&wav, 24000, 0.0, 0.05);
        // 无填充，只有淡入淡出
        assert_eq!(result.len(), 4800);
        // 第一个样本应该是 0（线性淡入起点）
        assert!(
            result[0].abs() < 0.01,
            "First sample should be near 0, got {}",
            result[0]
        );
        // 中间应该是 1.0
        assert!((result[2400] - 1.0).abs() < 0.01);
        // 最后一个样本应该是 0（线性淡出终点）
        assert!(
            result[4799].abs() < 0.01,
            "Last sample should be near 0, got {}",
            result[4799]
        );
    }

    #[test]
    fn test_fade_and_pad_empty() {
        assert!(fade_and_pad(&[], 24000, 0.1, 0.1).is_empty());
    }

    #[test]
    fn test_fade_and_pad_short_audio() {
        // 极短音频（比 fade_duration 短）
        let wav = vec![0.5; 100];
        let result = fade_and_pad(&wav, 24000, 0.0, 0.1);
        // fade_samples = 2400 > len/2 = 50, so k = 50
        assert_eq!(result.len(), 100);
        // 应该有淡入淡出效果
        assert!(result[0] < result[50]);
    }

    // ── 中间静音移除测试 ──────────────────────────────

    #[test]
    fn test_remove_mid_silence_no_silence() {
        // 纯正弦波，无静音
        let wav: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let result = remove_mid_silence(&wav, 24000, 300, 100, 100, -50.0);
        // 无静音，长度应基本不变
        let ratio = (result.len() as f32 - wav.len() as f32).abs() / wav.len() as f32;
        assert!(ratio < 0.1);
    }

    #[test]
    fn test_remove_mid_silence_with_gap() {
        // 1s 信号 + 1s 静音 + 1s 信号
        let sample_rate = 24000u32;
        let signal: Vec<f32> = (0..sample_rate as usize)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let silence: Vec<f32> = vec![0.0; sample_rate as usize];
        let mut wav = Vec::new();
        wav.extend_from_slice(&signal);
        wav.extend_from_slice(&silence);
        wav.extend_from_slice(&signal);

        let result = remove_mid_silence(&wav, sample_rate, 300, 100, 100, -50.0);
        // 1s 静音 = 500ms > 300ms 阈值，应被压缩
        assert!(
            result.len() < wav.len(),
            "Should be shorter after removing mid silence: {} vs {}",
            result.len(),
            wav.len()
        );
        // 但不能太短（至少保留信号 + 部分 keep_sil）
        assert!(
            result.len() > sample_rate as usize,
            "Should retain signal data"
        );
    }

    #[test]
    fn test_remove_mid_silence_all_silence() {
        let wav = vec![0.0; 4800];
        let result = remove_mid_silence(&wav, 24000, 300, 100, 100, -50.0);
        // 全静音应返回原样
        assert_eq!(result.len(), wav.len());
    }

    #[test]
    fn test_remove_mid_silence_empty() {
        assert!(remove_mid_silence(&[], 24000, 300, 100, 100, -50.0).is_empty());
    }

    #[test]
    fn test_remove_mid_silence_preserves_short_pauses() {
        // 0.5s 信号 + 0.2s 静音（< 300ms 阈值）+ 0.5s 信号
        let sample_rate = 24000u32;
        let signal: Vec<f32> = (0..sample_rate as usize / 2)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let short_silence: Vec<f32> = vec![0.0; sample_rate as usize / 5]; // 200ms
        let mut wav = Vec::new();
        wav.extend_from_slice(&signal);
        wav.extend_from_slice(&short_silence);
        wav.extend_from_slice(&signal);

        let result = remove_mid_silence(&wav, sample_rate, 300, 100, 100, -50.0);
        // 短静音不应被移除
        assert!(
            result.len() >= wav.len() * 9 / 10,
            "Short pauses should be preserved: {} vs {}",
            result.len(),
            wav.len()
        );
    }
}
