//! 引擎配置模块
//!
//! 参考 Qwen3-TTS 的 SynthesisOptions 和 QORA-TTS 的 SystemInfo 设计。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// TTS 引擎配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsEngineConfig {
    /// 模型目录路径
    ///
    /// 包含 model.safetensors, config.json, tokenizer.json 等文件
    pub model_dir: PathBuf,

    /// 推理设备
    ///
    /// - "cpu": CPU 推理
    /// - "metal": Apple Silicon Metal GPU 加速
    /// - "cuda": NVIDIA CUDA GPU 加速
    #[serde(default = "default_device")]
    pub device: String,

    /// 采样温度 (0.0-2.0, 默认 0.8)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-K 采样 (默认 50)
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// 重复惩罚 (默认 1.05)
    ///
    /// 对已出现 token 的 logits 施加惩罚，降低重复概率。
    /// 1.0 = 禁用, 1.1-1.3 = 轻度惩罚。
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// No-repeat n-gram size (默认 0 = 禁用)
    ///
    /// 防止长度为 n 的 token 序列重复出现。
    /// 建议设为 3 防止 3-gram 重复。
    #[serde(default)]
    pub no_repeat_ngram_size: usize,

    /// 随机种子 (None = 随机)
    #[serde(default)]
    pub seed: Option<u64>,

    /// 最大生成 token 数 (默认 500)
    #[serde(default = "default_max_codes")]
    pub max_codes: usize,

    /// 输出采样率 (默认 24000)
    #[serde(default = "default_sample_rate")]
    pub output_sample_rate: u32,

    /// 语言 ("auto" 自动检测, "chinese", "english", "japanese", "korean")
    #[serde(default = "default_language")]
    pub language: String,

    /// 混合精度模式 (默认 false)
    ///
    /// 启用后，TalkerModel 使用 F16 (Transformer matmul 提速 ~16%)，
    /// CodePredictor 和 AudioDecoder 使用 F32 (Metal F16 conv 内核性能不佳)。
    /// 仅在 Metal/CUDA 设备上有效，CPU 始终使用 F32。
    #[serde(default)]
    pub mixed_precision: bool,

    /// TalkerModel 权重量化格式 (默认 None = 不量化)
    ///
    /// 启用后，将 TalkerModel 的 28 层 Transformer 权重量化为 GGML 格式，
    /// 减少每步生成的内存带宽需求:
    /// - "q8_0": 8-bit 量化，权重体积 1/4，精度损失极小
    /// - "q4_0": 4-bit 量化，权重体积 1/8，精度损失较小
    /// - "q4k":  4-bit K-量化，权重体积 1/8，精度更好但量化稍慢
    ///
    /// 量化与 mixed_precision 互斥 (量化要求 F32 输入)。
    #[serde(default)]
    pub quantize: Option<String>,

    /// AudioDecoder 设备覆盖 (默认 None = 与主设备相同)
    ///
    /// 设置为 "cpu" 可将解码器运行在 CPU 上，而 TalkerModel 和 CodePredictor
    /// 继续在 GPU 上运行。适用于 Metal Conv1d 内核效率低下的场景
    /// (Metal GPU 仅达到峰值 FLOPS 的 ~3.3%)。Apple Silicon 统一内存
    /// 意味着无 GPU→CPU 数据传输开销。
    ///
    /// 可能的值: "cpu", "metal", "cuda", None (同主设备)
    #[serde(default)]
    pub decode_device: Option<String>,
}

fn default_device() -> String {
    "cpu".to_string()
}
fn default_temperature() -> f32 {
    0.8
}
fn default_top_k() -> usize {
    50
}
fn default_repetition_penalty() -> f32 {
    1.05
}
fn default_max_codes() -> usize {
    500
}
fn default_sample_rate() -> u32 {
    24000
}

fn default_language() -> String {
    "auto".to_string()
}

impl Default for TtsEngineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            device: default_device(),
            temperature: default_temperature(),
            top_k: default_top_k(),
            repetition_penalty: default_repetition_penalty(),
            no_repeat_ngram_size: 0,
            seed: None,
            max_codes: default_max_codes(),
            output_sample_rate: default_sample_rate(),
            language: default_language(),
            mixed_precision: false,
            quantize: None,
            decode_device: None,
        }
    }
}

/// 系统信息检测
///
/// 参考 QORA-TTS 的 SystemInfo::detect() 设计。
/// 根据硬件能力自动调整生成参数。
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// 总内存 (MB)
    pub total_ram_mb: u64,
    /// 可用内存 (MB)
    pub available_ram_mb: u64,
    /// CPU 线程数
    pub cpu_threads: usize,
}

impl SystemInfo {
    /// 自动检测系统信息
    pub fn detect() -> Self {
        let total_ram_mb = get_total_ram_mb();
        let available_ram_mb = get_available_ram_mb();
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            total_ram_mb,
            available_ram_mb,
            cpu_threads,
        }
    }

    /// 智能限制
    ///
    /// 根据系统资源自动调整生成上限。
    pub fn smart_limits(&self) -> SmartLimits {
        let default_max_codes = if self.total_ram_mb < 4096 {
            300
        } else if self.total_ram_mb < 8192 {
            500
        } else {
            800
        };

        // macOS vm_stat 报告的 "free" 内存通常很低（文件缓存占用大量内存），
        // 但实际可通过内存压缩和缓存回收获得更多可用内存。
        // 仅在极端低内存（<512MB）时才启用保守限制。
        let max_codes = if self.available_ram_mb < 512 {
            200
        } else {
            default_max_codes
        };

        let warning = if self.total_ram_mb < 4096 {
            Some(format!(
                "Low memory ({}MB). Generation length limited to {} codes.",
                self.total_ram_mb, max_codes
            ))
        } else {
            None
        };

        SmartLimits {
            default_max_codes,
            max_codes,
            warning,
        }
    }
}

/// 智能限制
#[derive(Debug, Clone)]
pub struct SmartLimits {
    /// 建议的最大 codes 数
    pub default_max_codes: usize,
    /// 硬性上限
    pub max_codes: usize,
    /// 警告消息
    pub warning: Option<String>,
}

fn get_total_ram_mb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let output = Command::new("sysctl").args(["-n", "hw.memsize"]).output();
        match output {
            Ok(o) => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                let bytes: u64 = s.parse().unwrap_or(8 * 1024 * 1024 * 1024);
                bytes / (1024 * 1024)
            }
            Err(_) => 8192,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        8192
    }
}

fn get_available_ram_mb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // macOS 上用 vm_stat 获取可用内存
        // 可用内存 = free + inactive + speculative 页（这些都可以被系统回收）
        use std::process::Command;
        let output = Command::new("vm_stat").output();
        match output {
            Ok(o) => {
                let s = String::from_utf8_lossy(&o.stdout);

                // 从第一行解析页大小: "(page size of 16384 bytes)"
                // Apple Silicon 默认 16384，Intel Mac 默认 4096
                let page_size: u64 = s
                    .lines()
                    .next()
                    .and_then(|line| {
                        let start = line.find("page size of ")? + "page size of ".len();
                        let rest = &line[start..];
                        let end = rest.find(" bytes")?;
                        rest[..end].parse().ok()
                    })
                    .unwrap_or(4096);

                // 累加 free + inactive + speculative 页
                let mut available_pages: u64 = 0;
                for line in s.lines() {
                    let lower = line.to_lowercase();
                    if lower.contains("free")
                        || lower.contains("inactive")
                        || lower.contains("speculative")
                    {
                        if let Some(colon_pos) = line.find(':') {
                            let rest = &line[colon_pos + 1..].trim();
                            let rest = rest.trim_end_matches('.');
                            if let Ok(pages) = rest.replace(',', "").parse::<u64>() {
                                available_pages += pages;
                            }
                        }
                    }
                }
                available_pages * page_size / (1024 * 1024)
            }
            Err(_) => 4096,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        4096
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = TtsEngineConfig::default();
        assert_eq!(cfg.device, "cpu");
        assert!((cfg.temperature - 0.8).abs() < 1e-6);
        assert_eq!(cfg.top_k, 50);
        assert!((cfg.repetition_penalty - 1.05).abs() < 1e-6);
        assert_eq!(cfg.no_repeat_ngram_size, 0);
        assert_eq!(cfg.seed, None);
        assert_eq!(cfg.max_codes, 500);
        assert_eq!(cfg.output_sample_rate, 24000);
        assert_eq!(cfg.language, "auto");
        assert!(!cfg.mixed_precision);
        assert_eq!(cfg.quantize, None);
        assert_eq!(cfg.decode_device, None);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = TtsEngineConfig {
            model_dir: PathBuf::from("/tmp/model"),
            device: "metal".to_string(),
            temperature: 0.5,
            top_k: 10,
            repetition_penalty: 1.1,
            no_repeat_ngram_size: 3,
            seed: Some(42),
            max_codes: 200,
            output_sample_rate: 16000,
            language: "chinese".to_string(),
            mixed_precision: true,
            quantize: Some("q8_0".to_string()),
            decode_device: Some("cpu".to_string()),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let de: TtsEngineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(de.model_dir, PathBuf::from("/tmp/model"));
        assert_eq!(de.device, "metal");
        assert!((de.temperature - 0.5).abs() < 1e-6);
        assert_eq!(de.top_k, 10);
        assert!((de.repetition_penalty - 1.1).abs() < 1e-6);
        assert_eq!(de.no_repeat_ngram_size, 3);
        assert_eq!(de.seed, Some(42));
        assert_eq!(de.max_codes, 200);
        assert_eq!(de.output_sample_rate, 16000);
        assert_eq!(de.language, "chinese");
        assert!(de.mixed_precision);
        assert_eq!(de.quantize.as_deref(), Some("q8_0"));
        assert_eq!(de.decode_device.as_deref(), Some("cpu"));
    }

    #[test]
    fn test_config_serde_partial_defaults() {
        let json = r#"{"model_dir":"/tmp/m"}"#;
        let cfg: TtsEngineConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.device, "cpu");
        assert!((cfg.temperature - 0.8).abs() < 1e-6);
        assert_eq!(cfg.top_k, 50);
        assert_eq!(cfg.max_codes, 500);
    }

    #[test]
    fn test_smart_limits_low_memory() {
        let si = SystemInfo {
            total_ram_mb: 2048,
            available_ram_mb: 512,
            cpu_threads: 4,
        };
        let limits = si.smart_limits();
        assert_eq!(limits.default_max_codes, 300);
        assert!(limits.warning.is_some());
    }

    #[test]
    fn test_smart_limits_mid_memory() {
        let si = SystemInfo {
            total_ram_mb: 6144,
            available_ram_mb: 2048,
            cpu_threads: 8,
        };
        let limits = si.smart_limits();
        assert_eq!(limits.default_max_codes, 500);
        // 6144 >= 4096, so no warning
        assert!(limits.warning.is_none());
    }

    #[test]
    fn test_smart_limits_high_memory() {
        let si = SystemInfo {
            total_ram_mb: 32768,
            available_ram_mb: 16384,
            cpu_threads: 10,
        };
        let limits = si.smart_limits();
        assert_eq!(limits.default_max_codes, 800);
        assert!(limits.warning.is_none());
    }

    #[test]
    fn test_smart_limits_extreme_low_available() {
        let si = SystemInfo {
            total_ram_mb: 32768,
            available_ram_mb: 256,
            cpu_threads: 10,
        };
        let limits = si.smart_limits();
        assert_eq!(limits.max_codes, 200);
    }

    #[test]
    fn test_system_info_detect() {
        let si = SystemInfo::detect();
        assert!(si.cpu_threads > 0);
        assert!(si.total_ram_mb > 0);
    }
}
