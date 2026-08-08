//! 配置验证模块
//!
//! 借鉴 dots.tts 的 Pydantic 配置验证思路，为 Rust 配置增加：
//! 1. **范围约束**: 数值字段范围检查（speed, pitch, volume, temperature 等）
//! 2. **互斥检查**: 字段间逻辑约束（如 prefer_online=false 且 fallback_on_error=false 无意义）
//! 3. **未知字段警告**: 检测 TOML 中的拼写错误或已废弃字段
//!
//! # 设计原则
//! - **非阻断**: 验证失败只发出警告（`tracing::warn!`），不阻止运行
//! - **可配置**: 可通过 `ValidationLevel` 控制严格程度
//! - **独立模块**: 不修改 `config.rs` 的结构定义，只增加验证逻辑
//!
//! # 示例
//! ```
//! use vt_core::config::Config;
//! use vt_core::config_validation::{validate_config, ValidationLevel};
//!
//! let config = Config::default();
//! let warnings = validate_config(&config, ValidationLevel::Standard);
//! for w in &warnings {
//!     tracing::warn!("Config: {}", w);
//! }
//! ```

use crate::config::{
    AsrConfig, AudioSyncConfig, AudioSyncMode, BackgroundMusicConfig, BatchConfig, CacheConfig,
    CheckpointConfig, CloningConfig, Config, PerformanceConfig, PipelineConfig,
    SpeedRateConfigField, SubtitleConfig, SubtitlePostProcessConfigField, TranslationConfig,
    TranslationMode, TtsConfig, VoiceExtractorConfig,
};

// ─── 验证级别 ─────────────────────────────────────────────

/// 验证严格程度
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ValidationLevel {
    /// 宽松模式：只报告严重错误（可能导致崩溃的配置）
    Lenient,
    /// 标准模式：报告错误 + 警告（可能导致性能问题或质量下降的配置）
    #[default]
    Standard,
    /// 严格模式：报告所有问题（包括建议和最佳实践）
    Strict,
}

// ─── 验证结果 ─────────────────────────────────────────────

/// 验证警告
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// 字段路径（如 "tts.speed"）
    pub field: String,
    /// 警告消息
    pub message: String,
    /// 严重程度
    pub severity: Severity,
}

/// 严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// 信息：建议但非必需
    Info,
    /// 警告：可能导致性能或质量问题
    Warning,
    /// 错误：可能导致崩溃或功能异常
    Error,
}

impl std::fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level = match self.severity {
            Severity::Info => "INFO",
            Severity::Warning => "WARN",
            Severity::Error => "ERROR",
        };
        write!(f, "[{}] {}: {}", level, self.field, self.message)
    }
}

// ─── 辅助宏 ───────────────────────────────────────────────

/// 创建一个验证警告
fn warn(field: &str, message: impl Into<String>, severity: Severity) -> ValidationWarning {
    ValidationWarning {
        field: field.to_string(),
        message: message.into(),
        severity,
    }
}

/// 检查数值是否在范围内
fn check_range(
    field: &str,
    value: f64,
    min: f64,
    max: f64,
    unit: &str,
) -> Option<ValidationWarning> {
    if value < min || value > max {
        Some(warn(
            field,
            format!("值 {} 超出范围 [{}, {}]{}", value, min, max, unit),
            Severity::Error,
        ))
    } else {
        None
    }
}

// ─── 验证函数 ─────────────────────────────────────────────

/// 验证完整配置
///
/// 按模块依次验证所有子配置，返回所有发现的问题。
///
/// # 参数
/// - `config`: 应用配置
/// - `level`: 验证严格程度
///
/// # 返回
/// 验证警告列表（空列表表示无问题）
#[must_use]
pub fn validate_config(config: &Config, level: ValidationLevel) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    warnings.extend(validate_asr(&config.asr, level));
    warnings.extend(validate_tts(&config.tts, level));
    warnings.extend(validate_translation(&config.translation, level));
    warnings.extend(validate_pipeline(&config.pipeline, level));
    warnings.extend(validate_cloning(&config.cloning, level));
    warnings.extend(validate_batch(&config.batch, level));
    warnings.extend(validate_audio_sync(&config.audio_sync, level));
    warnings.extend(validate_speed_rate(&config.speed_rate, level));
    warnings.extend(validate_cache(&config.cache, level));
    warnings.extend(validate_voice_extractor(
        &config.cloning.voice_extractor,
        level,
    ));
    warnings.extend(validate_subtitle_postprocess(
        &config.subtitle_postprocess,
        level,
    ));
    warnings.extend(validate_subtitle(&config.subtitle, level));
    warnings.extend(validate_background_music(&config.background_music, level));
    warnings.extend(validate_checkpoint(&config.checkpoint, level));
    warnings.extend(validate_performance(&config.performance, level));

    // 跨模块互斥检查
    warnings.extend(validate_cross_module(config, level));

    warnings
}

/// 验证 ASR 配置
fn validate_asr(config: &AsrConfig, _level: ValidationLevel) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    // 检查模型名称非空
    if config.model.trim().is_empty() {
        warnings.push(warn("asr.model", "模型名称为空", Severity::Error));
    }

    // 检查语言代码
    let valid_langs = ["en", "zh", "ja", "ko", "fr", "de", "es", "ru", "auto"];
    if !valid_langs.contains(&config.language.as_str()) {
        warnings.push(warn(
            "asr.language",
            format!(
                "语言代码 '{}' 不在常见列表中: {:?}",
                config.language, valid_langs
            ),
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证 TTS 配置
fn validate_tts(config: &TtsConfig, level: ValidationLevel) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    // 语速范围
    if let Some(w) = check_range("tts.speed", config.speed as f64, 0.5, 2.0, "") {
        warnings.push(w);
    } else if level >= ValidationLevel::Strict && (config.speed < 0.8 || config.speed > 1.5) {
        warnings.push(warn(
            "tts.speed",
            format!(
                "语速 {} 偏离推荐范围 [0.8, 1.5]，可能影响听感",
                config.speed
            ),
            Severity::Info,
        ));
    }

    // 音调范围
    if let Some(w) = check_range("tts.pitch", config.pitch as f64, 0.5, 2.0, "") {
        warnings.push(w);
    }

    // 音量范围
    if let Some(w) = check_range("tts.volume", config.volume as f64, 0.0, 2.0, "") {
        warnings.push(w);
    }

    // 采样率
    let valid_rates = [16000, 22050, 24000, 48000];
    if !valid_rates.contains(&config.sample_rate) {
        warnings.push(warn(
            "tts.sample_rate",
            format!(
                "采样率 {} 不在常见列表中: {:?}",
                config.sample_rate, valid_rates
            ),
            Severity::Warning,
        ));
    }

    // 温度范围
    if let Some(w) = check_range("tts.temperature", config.temperature, 0.0, 1.0, "") {
        warnings.push(w);
    }

    // 稳定性范围
    if let Some(w) = check_range("tts.stability", config.stability, 0.0, 1.0, "") {
        warnings.push(w);
    }

    // EQ 范围
    if config.eq_high_shelf_db < -12.0 || config.eq_high_shelf_db > 0.0 {
        warnings.push(warn(
            "tts.eq_high_shelf_db",
            format!(
                "EQ 衰减 {} dB 超出推荐范围 [-12, 0]",
                config.eq_high_shelf_db
            ),
            Severity::Warning,
        ));
    }

    // 并行任务数
    if config.parallel_tasks == 0 {
        warnings.push(warn(
            "tts.parallel_tasks",
            "并行任务数为 0，将无法合成",
            Severity::Error,
        ));
    } else if config.parallel_tasks > 16 && level >= ValidationLevel::Standard {
        warnings.push(warn(
            "tts.parallel_tasks",
            format!(
                "并行任务数 {} 过高，可能导致内存不足（推荐 ≤ 16）",
                config.parallel_tasks
            ),
            Severity::Warning,
        ));
    }

    // 交叉淡入淡出时长
    if config.crossfade_duration_ms > 500 {
        warnings.push(warn(
            "tts.crossfade_duration_ms",
            format!(
                "交叉淡入淡出 {}ms 过长，可能导致拼接不自然（推荐 ≤ 100ms）",
                config.crossfade_duration_ms
            ),
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证翻译配置
fn validate_translation(
    config: &TranslationConfig,
    level: ValidationLevel,
) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    // 批量大小
    if config.batch_size == 0 {
        warnings.push(warn(
            "translation.batch_size",
            "批量大小为 0",
            Severity::Error,
        ));
    } else if config.batch_size > 50 && level >= ValidationLevel::Standard {
        warnings.push(warn(
            "translation.batch_size",
            format!(
                "批量大小 {} 过大，可能导致 LLM 超时（推荐 ≤ 20）",
                config.batch_size
            ),
            Severity::Warning,
        ));
    }

    // 最大 token 数
    if config.max_tokens < 64 {
        warnings.push(warn(
            "translation.max_tokens",
            format!("max_tokens={} 过小，翻译可能被截断", config.max_tokens),
            Severity::Error,
        ));
    } else if config.max_tokens > 4096 && level >= ValidationLevel::Standard {
        warnings.push(warn(
            "translation.max_tokens",
            format!(
                "max_tokens={} 过大，可能导致内存问题（推荐 ≤ 2048）",
                config.max_tokens
            ),
            Severity::Warning,
        ));
    }

    // 温度范围
    if let Some(w) = check_range(
        "translation.temperature",
        config.temperature as f64,
        0.0,
        2.0,
        "",
    ) {
        warnings.push(w);
    } else if config.temperature > 0.7 && level >= ValidationLevel::Strict {
        warnings.push(warn(
            "translation.temperature",
            format!(
                "温度 {} 偏高，翻译可能不够稳定（推荐 ≤ 0.5）",
                config.temperature
            ),
            Severity::Info,
        ));
    }

    // DeepLX 超时
    if config.dlx_timeout_secs < 3 {
        warnings.push(warn(
            "translation.dlx_timeout_secs",
            format!("超时 {}s 过短，可能导致请求失败", config.dlx_timeout_secs),
            Severity::Warning,
        ));
    }

    // 互斥检查：prefer_online=false 且 fallback_on_error=false → 完全没有翻译后端
    if !config.prefer_online && !config.fallback_on_error {
        warnings.push(warn(
            "translation",
            "prefer_online=false 且 fallback_on_error=false：没有可用的翻译后端！",
            Severity::Error,
        ));
    }

    // SRT 模式 + 批量大小过大
    if config.translation_mode == TranslationMode::Srt && config.batch_size > 30 {
        warnings.push(warn(
            "translation",
            "SRT 模式下 batch_size 过大，可能导致 LLM 上下文溢出",
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证 Pipeline 配置
fn validate_pipeline(config: &PipelineConfig, level: ValidationLevel) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.segment_duration_secs < 5.0 {
        warnings.push(warn(
            "pipeline.segment_duration_secs",
            format!(
                "分割时长 {}s 过短，碎片化严重",
                config.segment_duration_secs
            ),
            Severity::Warning,
        ));
    } else if config.segment_duration_secs > 120.0 && level >= ValidationLevel::Standard {
        warnings.push(warn(
            "pipeline.segment_duration_secs",
            format!(
                "分割时长 {}s 过长，ASR 内存压力大",
                config.segment_duration_secs
            ),
            Severity::Warning,
        ));
    }

    if config.channel_capacity == 0 {
        warnings.push(warn(
            "pipeline.channel_capacity",
            "通道容量为 0，流水线将死锁",
            Severity::Error,
        ));
    } else if config.channel_capacity < 10 && level >= ValidationLevel::Standard {
        warnings.push(warn(
            "pipeline.channel_capacity",
            format!("通道容量 {} 过小，背压频繁", config.channel_capacity),
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证克隆配置
fn validate_cloning(config: &CloningConfig, _level: ValidationLevel) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.enabled {
        // 检查引擎名称
        let valid_engines = [
            "gpt-sovits",
            "indextts",
            "subprocess",
            "subprocess-persistent",
            "python-qwen-tts",
            "neutts",
            "vibevoice",
        ];
        if !valid_engines.contains(&config.engine.as_str()) {
            warnings.push(warn(
                "cloning.engine",
                format!(
                    "引擎 '{}' 不在已知列表中: {:?}",
                    config.engine, valid_engines
                ),
                Severity::Warning,
            ));
        }

        // 子进程引擎需要 clone_command
        if (config.engine == "subprocess"
            || config.engine == "subprocess-persistent"
            || config.engine == "python-qwen-tts")
            && config.clone_command.is_none()
        {
            warnings.push(warn(
                "cloning.clone_command",
                "子进程引擎未设置 clone_command",
                Severity::Error,
            ));
        }

        // 子进程引擎需要 clone_args 非空
        if (config.engine == "subprocess"
            || config.engine == "subprocess-persistent"
            || config.engine == "python-qwen-tts")
            && config.clone_args.is_empty()
        {
            warnings.push(warn(
                "cloning.clone_args",
                "子进程引擎未设置 clone_args",
                Severity::Error,
            ));
        }

        // 超时
        if config.clone_timeout_secs < 30 {
            warnings.push(warn(
                "cloning.clone_timeout_secs",
                format!("克隆超时 {}s 过短", config.clone_timeout_secs),
                Severity::Warning,
            ));
        }

        // GPT-SoVITS 参数
        if config.engine == "gpt-sovits" {
            if config.api_url.is_empty() {
                warnings.push(warn(
                    "cloning.api_url",
                    "GPT-SoVITS API URL 为空",
                    Severity::Error,
                ));
            }
            if let Some(ref pt) = config.prompt_text {
                if pt.is_empty() {
                    warnings.push(warn(
                        "cloning.prompt_text",
                        "prompt_text 设为空字符串",
                        Severity::Warning,
                    ));
                }
            }
        }

        // 温度
        if config.temperature < 0.0 || config.temperature > 2.0 {
            warnings.push(warn(
                "cloning.temperature",
                format!("温度 {} 超出 [0, 2]", config.temperature),
                Severity::Error,
            ));
        }

        // Top-P
        if config.top_p < 0.0 || config.top_p > 1.0 {
            warnings.push(warn(
                "cloning.top_p",
                format!("top_p {} 超出 [0, 1]", config.top_p),
                Severity::Error,
            ));
        }

        // 重复惩罚
        if config.repetition_penalty < 1.0 || config.repetition_penalty > 2.0 {
            warnings.push(warn(
                "cloning.repetition_penalty",
                format!(
                    "repetition_penalty {} 超出推荐 [1.0, 2.0]",
                    config.repetition_penalty
                ),
                Severity::Warning,
            ));
        }
    }

    warnings
}

/// 验证批量配置
fn validate_batch(config: &BatchConfig, _level: ValidationLevel) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.max_concurrent == 0 {
        warnings.push(warn(
            "batch.max_concurrent",
            "最大并发为 0",
            Severity::Error,
        ));
    }

    if config.memory_threshold < 50.0 || config.memory_threshold > 95.0 {
        warnings.push(warn(
            "batch.memory_threshold",
            format!("内存阈值 {} 超出推荐 [50, 95]", config.memory_threshold),
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证音频同步配置
fn validate_audio_sync(
    config: &AudioSyncConfig,
    _level: ValidationLevel,
) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.max_speed_ratio < 1.0 || config.max_speed_ratio > 3.0 {
        warnings.push(warn(
            "audio_sync.max_speed_ratio",
            format!("加速上限 {} 超出 [1.0, 3.0]", config.max_speed_ratio),
            Severity::Error,
        ));
    }

    if config.mode == AudioSyncMode::SpeedUp && config.max_speed_ratio > 2.0 {
        warnings.push(warn(
            "audio_sync",
            "SpeedUp 模式下加速 > 2.0 会严重失真",
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证 SpeedRate 配置
fn validate_speed_rate(
    config: &SpeedRateConfigField,
    _level: ValidationLevel,
) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.enabled {
        if config.max_audio_speed < 1.0 || config.max_audio_speed > 2.0 {
            warnings.push(warn(
                "speed_rate.max_audio_speed",
                format!("最大音频加速 {} 超出 [1.0, 2.0]", config.max_audio_speed),
                Severity::Error,
            ));
        }

        if config.max_video_slow < 1.0 || config.max_video_slow > 3.0 {
            warnings.push(warn(
                "speed_rate.max_video_slow",
                format!("最大视频慢放 {} 超出 [1.0, 3.0]", config.max_video_slow),
                Severity::Error,
            ));
        }
    }

    warnings
}

/// 验证缓存配置
fn validate_cache(config: &CacheConfig, _level: ValidationLevel) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.retention_days == 0 {
        warnings.push(warn(
            "cache.retention_days",
            "保留天数为 0，缓存不会被清理",
            Severity::Info,
        ));
    } else if config.retention_days > 365 {
        warnings.push(warn(
            "cache.retention_days",
            format!("保留天数 {} 过长，磁盘可能不足", config.retention_days),
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证参考音频提取配置
fn validate_voice_extractor(
    config: &VoiceExtractorConfig,
    _level: ValidationLevel,
) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.min_duration_secs >= config.max_duration_secs {
        warnings.push(warn(
            "cloning.voice_extractor",
            format!(
                "min_duration({}) >= max_duration({})",
                config.min_duration_secs, config.max_duration_secs
            ),
            Severity::Error,
        ));
    }

    if config.silence_threshold_db > -20.0 {
        warnings.push(warn(
            "cloning.voice_extractor.silence_threshold_db",
            format!(
                "静音阈值 {} dB 过高，可能误判语音为静音",
                config.silence_threshold_db
            ),
            Severity::Warning,
        ));
    }

    if config.target_rms_db > -10.0 || config.target_rms_db < -30.0 {
        warnings.push(warn(
            "cloning.voice_extractor.target_rms_db",
            format!("目标 RMS {} dBFS 超出推荐 [-30, -10]", config.target_rms_db),
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证字幕后处理配置
fn validate_subtitle_postprocess(
    config: &SubtitlePostProcessConfigField,
    _level: ValidationLevel,
) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.min_duration >= config.max_duration {
        warnings.push(warn(
            "subtitle_postprocess",
            format!(
                "min_duration({}) >= max_duration({})",
                config.min_duration, config.max_duration
            ),
            Severity::Error,
        ));
    }

    if config.merge_gap_threshold > 5.0 {
        warnings.push(warn(
            "subtitle_postprocess.merge_gap_threshold",
            format!("合并间隙 {}s 过大", config.merge_gap_threshold),
            Severity::Warning,
        ));
    }

    warnings
}

/// 验证字幕配置
fn validate_subtitle(_config: &SubtitleConfig, _level: ValidationLevel) -> Vec<ValidationWarning> {
    Vec::new()
}

/// 验证背景音乐配置
fn validate_background_music(
    config: &BackgroundMusicConfig,
    _level: ValidationLevel,
) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if let Some(ref path) = config.path {
        if !std::path::Path::new(path).exists() {
            warnings.push(warn(
                "background_music.path",
                format!("背景音乐文件不存在: {}", path),
                Severity::Warning,
            ));
        }
    }

    if config.volume < 0.0 || config.volume > 1.0 {
        warnings.push(warn(
            "background_music.volume",
            format!("音量 {} 超出 [0, 1]", config.volume),
            Severity::Error,
        ));
    }

    warnings
}

/// 验证检查点配置
fn validate_checkpoint(
    config: &CheckpointConfig,
    _level: ValidationLevel,
) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    if config.enabled && config.retention_days == 0 {
        warnings.push(warn(
            "checkpoint.retention_days",
            "保留天数为 0，检查点不会被清理",
            Severity::Info,
        ));
    }

    warnings
}

/// 验证性能配置
fn validate_performance(
    _config: &PerformanceConfig,
    _level: ValidationLevel,
) -> Vec<ValidationWarning> {
    Vec::new()
}

/// 跨模块验证
fn validate_cross_module(config: &Config, level: ValidationLevel) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();

    // 克隆启用但 TTS 引擎不是克隆引擎
    if config.cloning.enabled && config.tts.engine == "say" {
        if level >= ValidationLevel::Standard {
            warnings.push(warn(
                "cross_module",
                "cloning.enabled=true 但 tts.engine='say'：克隆将被使用，TTS 引擎配置可能多余",
                Severity::Info,
            ));
        }
    }

    // 并发任务数 vs 通道容量
    if config.max_concurrent_tasks > config.pipeline.channel_capacity {
        warnings.push(warn(
            "cross_module",
            format!(
                "max_concurrent_tasks({}) > channel_capacity({})：可能导致背压",
                config.max_concurrent_tasks, config.pipeline.channel_capacity
            ),
            Severity::Warning,
        ));
    }

    // SpeedRate 和 AudioSync 同时启用
    if config.speed_rate.enabled && config.audio_sync.mode != AudioSyncMode::default() {
        if level >= ValidationLevel::Standard {
            warnings.push(warn(
                "cross_module",
                "speed_rate 和 audio_sync 同时启用：两者都会修改音频时长，可能冲突",
                Severity::Warning,
            ));
        }
    }

    // 克隆超时 vs TTS 并行任务
    if config.cloning.enabled && config.cloning.clone_timeout_secs > 0 {
        let estimated_memory = config.tts.parallel_tasks * 200; // 粗略估算每个克隆任务 ~200MB
        if estimated_memory > 4000 && level >= ValidationLevel::Standard {
            warnings.push(warn(
                "cross_module",
                format!(
                    "TTS 并行 {} × 克隆 ~200MB ≈ {}MB 内存",
                    config.tts.parallel_tasks, estimated_memory
                ),
                Severity::Warning,
            ));
        }
    }

    warnings
}

// ─── 未知字段检测 ─────────────────────────────────────────

/// 检测 TOML 中的未知字段
///
/// 通过对比已知的字段列表，检测用户配置文件中可能存在的拼写错误。
///
/// # 参数
/// - `toml_content`: TOML 文件内容
///
/// # 返回
/// 未知字段警告列表
#[must_use]
pub fn detect_unknown_fields(toml_content: &str) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    let mut current_section = String::new();

    for line in toml_content.lines() {
        let trimmed = line.trim();

        // 跳过空行和注释
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // 检测 section header [xxx]
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            current_section = trimmed[1..trimmed.len() - 1].to_string();
            continue;
        }

        // 检测 key = value
        if let Some(eq_pos) = trimmed.find('=') {
            let key = trimmed[..eq_pos].trim().to_string();
            let full_path = if current_section.is_empty() {
                key.clone()
            } else {
                format!("{}.{}", current_section, key)
            };

            // 检查是否是已知字段
            if !is_known_field(&full_path) {
                warnings.push(warn(
                    &full_path,
                    "未知字段，可能是拼写错误或已废弃的配置项",
                    Severity::Warning,
                ));
            }
        }
    }

    warnings
}

/// 检查字段路径是否是已知配置字段
fn is_known_field(path: &str) -> bool {
    let known_fields = [
        // [asr]
        "asr.model",
        "asr.use_metal",
        "asr.language",
        // [tts]
        "tts.engine",
        "tts.speed",
        "tts.pitch",
        "tts.volume",
        "tts.voice_id",
        "tts.voice",
        "tts.sample_rate",
        "tts.device",
        "tts.cache_dir",
        "tts.parallel_tasks",
        "tts.model_variant",
        "tts.model_path",
        "tts.fallback_to_say",
        "tts.auto_voice_selection",
        "tts.seed",
        "tts.temperature",
        "tts.stability",
        "tts.eq_high_shelf_db",
        "tts.crossfade_duration_ms",
        // [translation]
        "translation.glossary_path",
        "translation.batch_size",
        "translation.model_path",
        "translation.device",
        "translation.max_tokens",
        "translation.temperature",
        "translation.model_source",
        "translation.dlx_endpoint",
        "translation.dlx_timeout_secs",
        "translation.dlx_max_retries",
        "translation.prefer_online",
        "translation.fallback_on_error",
        "translation.health_check_interval_secs",
        "translation.force_glossary",
        "translation.post_correction_enabled",
        "translation.translation_mode",
        // [pipeline]
        "pipeline.segment_duration_secs",
        "pipeline.channel_capacity",
        "pipeline.enable_vad_split",
        // [cloning]
        "cloning.enabled",
        "cloning.engine",
        "cloning.reference_audio_dir",
        "cloning.auto_extract_speaker",
        "cloning.api_url",
        "cloning.timeout_secs",
        "cloning.prompt_text",
        "cloning.prompt_lang",
        "cloning.text_lang",
        "cloning.text_split_method",
        "cloning.top_k",
        "cloning.top_p",
        "cloning.temperature",
        "cloning.repetition_penalty",
        "cloning.gpt_model",
        "cloning.sovits_model",
        "cloning.clone_command",
        "cloning.clone_model_path",
        "cloning.clone_args",
        "cloning.clone_timeout_secs",
        // [cloning.voice_extractor]
        "cloning.voice_extractor.enable_enhancement",
        "cloning.voice_extractor.enable_silence_trim",
        "cloning.voice_extractor.enable_normalization",
        "cloning.voice_extractor.silence_threshold_db",
        "cloning.voice_extractor.target_rms_db",
        "cloning.voice_extractor.min_duration_secs",
        "cloning.voice_extractor.max_duration_secs",
        "cloning.voice_extractor.ideal_duration_secs",
        // [batch]
        "batch.max_concurrent",
        "batch.memory_threshold",
        "batch.enable_priority",
        // [checkpoint]
        "checkpoint.enabled",
        "checkpoint.dir",
        "checkpoint.retention_days",
        // [performance]
        "performance.enable_profiling",
        "performance.flamegraph_output",
        // [audio_sync]
        "audio_sync.mode",
        "audio_sync.max_speed_ratio",
        // [subtitle_postprocess]
        "subtitle_postprocess.min_duration",
        "subtitle_postprocess.max_duration",
        "subtitle_postprocess.merge_gap_threshold",
        "subtitle_postprocess.enable_fragment_redistribution",
        // [speed_rate]
        "speed_rate.enabled",
        "speed_rate.mode",
        "speed_rate.max_audio_speed",
        "speed_rate.max_video_slow",
        // [cache]
        "cache.tts_cache_enabled",
        "cache.translation_cache_enabled",
        "cache.tts_remove_silence",
        "cache.retention_days",
        // [subtitle]
        "subtitle.subtitle_type",
        "subtitle.output_dir",
        // [background_music]
        "background_music.path",
        "background_music.volume",
        "background_music.loop_bgm",
        // top-level
        "output_dir",
        "max_concurrent_tasks",
    ];

    known_fields.contains(&path)
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_default_config() {
        let config = Config::default();
        let warnings = validate_config(&config, ValidationLevel::Standard);
        // 默认配置应该没有 Error 级别的问题
        let errors: Vec<_> = warnings
            .iter()
            .filter(|w| w.severity == Severity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "Default config should have no errors: {:?}",
            errors
        );
    }

    #[test]
    fn test_validate_tts_speed_out_of_range() {
        let mut config = Config::default();
        config.tts.speed = 5.0; // 超出范围
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "tts.speed" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_tts_volume_negative() {
        let mut config = Config::default();
        config.tts.volume = -0.5;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "tts.volume" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_translation_no_backend() {
        let mut config = Config::default();
        config.translation.prefer_online = false;
        config.translation.fallback_on_error = false;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.severity == Severity::Error && w.message.contains("翻译后端")));
    }

    #[test]
    fn test_validate_pipeline_channel_capacity_zero() {
        let mut config = Config::default();
        config.pipeline.channel_capacity = 0;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "pipeline.channel_capacity" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_cloning_missing_command() {
        let mut config = Config::default();
        config.cloning.engine = "subprocess-persistent".to_string();
        config.cloning.clone_command = None;
        config.cloning.clone_args = Vec::new();
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "cloning.clone_command" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_voice_extractor_min_max() {
        let mut config = Config::default();
        config.cloning.voice_extractor.min_duration_secs = 10.0;
        config.cloning.voice_extractor.max_duration_secs = 5.0;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.severity == Severity::Error && w.message.contains("min_duration")));
    }

    #[test]
    fn test_validate_cross_module_concurrency() {
        let mut config = Config::default();
        config.max_concurrent_tasks = 200;
        config.pipeline.channel_capacity = 10;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "cross_module" && w.severity == Severity::Warning));
    }

    #[test]
    fn test_validate_tts_parallel_tasks_zero() {
        let mut config = Config::default();
        config.tts.parallel_tasks = 0;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "tts.parallel_tasks" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_strict_mode_info() {
        let mut config = Config::default();
        config.tts.speed = 0.6; // 偏离推荐范围
        let warnings = validate_config(&config, ValidationLevel::Strict);
        assert!(warnings.iter().any(|w| w.severity == Severity::Info));
    }

    #[test]
    fn test_validate_lenient_mode_no_info() {
        let mut config = Config::default();
        config.tts.speed = 0.6;
        let warnings = validate_config(&config, ValidationLevel::Lenient);
        assert!(!warnings.iter().any(|w| w.severity == Severity::Info));
    }

    #[test]
    fn test_detect_unknown_fields() {
        let toml = r#"
[asr]
model = "whisper-large"
unknown_field = "oops"

[tts]
speed = 1.0
typo_speed = 2.0
"#;
        let warnings = detect_unknown_fields(toml);
        assert!(warnings.iter().any(|w| w.field == "asr.unknown_field"));
        assert!(warnings.iter().any(|w| w.field == "tts.typo_speed"));
    }

    #[test]
    fn test_detect_unknown_fields_clean() {
        let toml = r#"
[asr]
model = "whisper-large"

[tts]
speed = 1.0
"#;
        let warnings = detect_unknown_fields(toml);
        assert!(
            warnings.is_empty(),
            "Should have no unknown fields: {:?}",
            warnings
        );
    }

    #[test]
    fn test_detect_unknown_fields_skips_comments() {
        let toml = r#"
# This is a comment
[asr]
model = "whisper-large" # inline comment
"#;
        let warnings = detect_unknown_fields(toml);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validation_warning_display() {
        let w = ValidationWarning {
            field: "tts.speed".to_string(),
            message: "值 5.0 超出范围".to_string(),
            severity: Severity::Error,
        };
        let s = format!("{}", w);
        assert!(s.contains("[ERROR]"));
        assert!(s.contains("tts.speed"));
    }

    #[test]
    fn test_validate_audio_sync_speed_too_high() {
        let mut config = Config::default();
        config.audio_sync.mode = AudioSyncMode::SpeedUp;
        config.audio_sync.max_speed_ratio = 2.5;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "audio_sync" && w.severity == Severity::Warning));
    }

    #[test]
    fn test_validate_cloning_temperature_out_of_range() {
        let mut config = Config::default();
        config.cloning.temperature = 3.0;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "cloning.temperature" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_background_music_missing_file() {
        let mut config = Config::default();
        config.background_music.path = Some("/nonexistent/music.mp3".to_string());
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings.iter().any(|w| w.field == "background_music.path"));
    }

    #[test]
    fn test_validate_translation_max_tokens_too_small() {
        let mut config = Config::default();
        config.translation.max_tokens = 32;
        let warnings = validate_config(&config, ValidationLevel::Standard);
        assert!(warnings
            .iter()
            .any(|w| w.field == "translation.max_tokens" && w.severity == Severity::Error));
    }
}
