//! Golden Master（金标准）测试框架
//!
//! 核心理念：**不写断言，而是快照结果，后续运行对比差异。**
//!
//! AI 改代码后：
//! - 如果输出指纹一致 → 自动通过（无需人工审查）
//! - 如果输出指纹变了 → 标记为 "changed"，人工审 diff 后重新批准
//!
//! 这将"听音频/读文本验证质量"的时间从小时级降到秒级。
//!
//! # 工作流程
//! 1. **首次运行**：计算输出指纹，保存为 golden master 基线文件
//! 2. **后续运行**：计算当前输出指纹，与基线对比
//!    - 一致（在容差范围内）→ PASS
//!    - 不一致 → FAIL，人工审查后执行 `accept` 更新基线
//! 3. **CI 集成**：PR 中如果 golden master 测试失败 → 阻止合并
//!
//! # 指纹设计原则
//! - **确定性**：相同输入 + 相同算法 → 相同指纹
//! - **抗噪声**：浮点数使用容差比较，非精确匹配
//! - **语义化**：指纹包含人类可读的指标（如 RMS、时长），不仅是哈希
//! - **快速**：指纹计算比完整输出快得多
//!
//! # 来源
//! 借鉴 ApprovalTests（approvaltests.com）和 Meticulous AI 的 replay testing 理念。
//! 参见 Simon Willison 的 "Not all AI-assisted programming is vibe coding" 文章
//! 中关于 "测试驱动的 AI 编程" 的论述。
//!
//! # 示例
//! ```
//! use vt_core::golden_master::{AudioFingerprint, FingerprintCompare};
//!
//! // 计算音频指纹
//! let samples: Vec<f32> = vec![0.0, 0.5, -0.3, 0.8];
//! let fp = AudioFingerprint::from_samples(&samples, 24000);
//!
//! // 对比两个指纹
//! let other = AudioFingerprint::from_samples(&samples, 24000);
//! assert_eq!(fp.compare(&other), FingerprintCompare::Match);
//! ```

use std::fmt;
use std::path::{Path, PathBuf};

// ─── 音频指纹 ─────────────────────────────────────────────

/// 音频输出指纹
///
/// 捕获音频的关键声学特征，而非完整音频数据。
/// 用于快速判断 TTS 输出是否发生变化。
///
/// # 字段说明
/// - `sample_count`: 采样点数（决定时长）
/// - `sample_rate`: 采样率
/// - `duration_secs`: 时长（秒）
/// - `rms`: RMS 能量（音量指标）
/// - `peak`: 峰值幅度（绝对值最大）
/// - `dc_offset`: DC 偏移（0 表示对称，非 0 表示有直流偏移）
/// - `zero_crossing_rate`: 过零率（语音 vs 噪声的判别指标）
/// - `spectral_flatness`: 频谱平坦度（接近 1 = 白噪声，接近 0 = 有调性语音）
/// - `energy_in_speech_band`: 语音频段（300-3400Hz）能量占比
/// - `sha256_hash`: 完整音频数据的 SHA-256 哈希（精确匹配用）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioFingerprint {
    pub sample_count: usize,
    pub sample_rate: u32,
    pub duration_secs: f64,
    pub rms: f64,
    pub peak: f64,
    pub dc_offset: f64,
    pub zero_crossing_rate: f64,
    pub spectral_flatness: f64,
    pub energy_in_speech_band: f64,
    pub sha256_hash: String,
}

impl AudioFingerprint {
    /// 从 PCM f32 采样数据计算音频指纹
    ///
    /// # 参数
    /// - `samples`: 浮点采样数据，范围 [-1.0, 1.0]
    /// - `sample_rate`: 采样率（Hz）
    pub fn from_samples(samples: &[f32], sample_rate: u32) -> Self {
        let n = samples.len();
        let duration_secs = if sample_rate > 0 {
            n as f64 / sample_rate as f64
        } else {
            0.0
        };

        // RMS 能量
        let sum_sq: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let rms = if n > 0 { (sum_sq / n as f64).sqrt() } else { 0.0 };

        // 峰值
        let peak = samples
            .iter()
            .map(|s| s.abs() as f64)
            .fold(0.0_f64, f64::max);

        // DC 偏移
        let sum: f64 = samples.iter().map(|s| *s as f64).sum();
        let dc_offset = if n > 0 { sum / n as f64 } else { 0.0 };

        // 过零率
        let zero_crossings = samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        let zero_crossing_rate = if n > 1 {
            zero_crossings as f64 / (n - 1) as f64
        } else {
            0.0
        };

        // 频谱平坦度（使用简化的 DFT bin 分析）
        // 不依赖外部 FFT 库，使用粗粒度频段能量分析
        let spectral_flatness = compute_spectral_flatness(samples, sample_rate);

        // 语音频段（300-3400Hz）能量占比
        let energy_in_speech_band =
            compute_speech_band_energy_ratio(samples, sample_rate, 300.0, 3400.0);

        // SHA-256 哈希（用于精确匹配）
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let hash = sha2_hash(&bytes);

        Self {
            sample_count: n,
            sample_rate,
            duration_secs,
            rms,
            peak,
            dc_offset,
            zero_crossing_rate,
            spectral_flatness,
            energy_in_speech_band,
            sha256_hash: hash,
        }
    }

    /// 从 WAV 文件计算音频指纹
    ///
    /// # 参数
    /// - `path`: WAV 文件路径
    /// - `target_sample_rate`: 期望的采样率（用于验证）
    ///
    /// # 错误
    /// - 文件读取失败或格式不匹配时返回错误信息
    pub fn from_wav_file(
        path: &Path,
        target_sample_rate: Option<u32>,
    ) -> Result<Self, String> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| format!("Failed to open WAV file {path:?}: {e}"))?;
        let spec = reader.spec();

        if let Some(expected_sr) = target_sample_rate {
            if spec.sample_rate != expected_sr {
                return Err(format!(
                    "Sample rate mismatch: expected {expected_sr} Hz, got {} Hz",
                    spec.sample_rate
                ));
            }
        }

        // 读取采样数据并归一化到 f32 [-1.0, 1.0]
        let samples: Vec<f32> = match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect(),
            32 => reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 2147483648.0)
                .collect(),
            8 => {
                // hound 对 u8 的 Sample 实现有特殊处理，需手动读取
                let mut buf = Vec::new();
                for sample in reader.samples::<i16>() {
                    if let Ok(s) = sample {
                        // 8-bit PCM 是无符号的 (0-255)，中心为 128
                        let u8_val = s as u8;
                        buf.push((u8_val as f32 - 128.0) / 128.0);
                    }
                }
                buf
            }
            other => {
                return Err(format!("Unsupported bits_per_sample: {other}"));
            }
        };

        // 如果是多声道，混合为单声道
        let mono_samples = if spec.channels > 1 {
            mix_to_mono(&samples, spec.channels as usize)
        } else {
            samples
        };

        Ok(Self::from_samples(&mono_samples, spec.sample_rate))
    }
}

impl Fingerprint for AudioFingerprint {
    fn compare(&self, other: &Self) -> FingerprintCompare {
        // 精确哈希匹配
        if self.sha256_hash == other.sha256_hash {
            return FingerprintCompare::Match;
        }

        // 声学特征容差比较
        let tolerances = AudioTolerances::default();

        let duration_diff = (self.duration_secs - other.duration_secs).abs();
        let rms_diff = (self.rms - other.rms).abs();
        let peak_diff = (self.peak - other.peak).abs();
        let dc_diff = (self.dc_offset - other.dc_offset).abs();
        let zcr_diff = (self.zero_crossing_rate - other.zero_crossing_rate).abs();
        let sf_diff = (self.spectral_flatness - other.spectral_flatness).abs();
        let esb_diff = (self.energy_in_speech_band - other.energy_in_speech_band).abs();

        let all_match = duration_diff <= tolerances.duration
            && rms_diff <= tolerances.rms
            && peak_diff <= tolerances.peak
            && dc_diff <= tolerances.dc_offset
            && zcr_diff <= tolerances.zero_crossing_rate
            && sf_diff <= tolerances.spectral_flatness
            && esb_diff <= tolerances.energy_in_speech_band;

        if all_match {
            FingerprintCompare::ApproximateMatch
        } else {
            FingerprintCompare::Changed {
                changes: vec![
                    if duration_diff > tolerances.duration {
                        Some(format!(
                            "duration: {:.4}s → {:.4}s (Δ={:.4}s, tol={:.4}s)",
                            self.duration_secs,
                            other.duration_secs,
                            duration_diff,
                            tolerances.duration
                        ))
                    } else {
                        None
                    },
                    if rms_diff > tolerances.rms {
                        Some(format!(
                            "rms: {:.6} → {:.6} (Δ={:.6}, tol={:.6})",
                            self.rms, other.rms, rms_diff, tolerances.rms
                        ))
                    } else {
                        None
                    },
                    if peak_diff > tolerances.peak {
                        Some(format!(
                            "peak: {:.6} → {:.6} (Δ={:.6}, tol={:.6})",
                            self.peak, other.peak, peak_diff, tolerances.peak
                        ))
                    } else {
                        None
                    },
                    if dc_diff > tolerances.dc_offset {
                        Some(format!(
                            "dc_offset: {:.6} → {:.6} (Δ={:.6}, tol={:.6})",
                            self.dc_offset, other.dc_offset, dc_diff, tolerances.dc_offset
                        ))
                    } else {
                        None
                    },
                    if zcr_diff > tolerances.zero_crossing_rate {
                        Some(format!(
                            "zcr: {:.6} → {:.6} (Δ={:.6}, tol={:.6})",
                            self.zero_crossing_rate,
                            other.zero_crossing_rate,
                            zcr_diff,
                            tolerances.zero_crossing_rate
                        ))
                    } else {
                        None
                    },
                    if sf_diff > tolerances.spectral_flatness {
                        Some(format!(
                            "spectral_flatness: {:.6} → {:.6} (Δ={:.6}, tol={:.6})",
                            self.spectral_flatness,
                            other.spectral_flatness,
                            sf_diff,
                            tolerances.spectral_flatness
                        ))
                    } else {
                        None
                    },
                    if esb_diff > tolerances.energy_in_speech_band {
                        Some(format!(
                            "speech_band_energy: {:.6} → {:.6} (Δ={:.6}, tol={:.6})",
                            self.energy_in_speech_band,
                            other.energy_in_speech_band,
                            esb_diff,
                            tolerances.energy_in_speech_band
                        ))
                    } else {
                        None
                    },
                ]
                .into_iter()
                .flatten()
                .collect(),
            }
        }
    }

    fn summary(&self) -> String {
        format!(
            "samples={}, sr={}Hz, dur={:.3}s, rms={:.6}, peak={:.6}, dc={:.6}, zcr={:.6}, sf={:.6}, esb={:.6}",
            self.sample_count,
            self.sample_rate,
            self.duration_secs,
            self.rms,
            self.peak,
            self.dc_offset,
            self.zero_crossing_rate,
            self.spectral_flatness,
            self.energy_in_speech_band
        )
    }
}

// ─── 文本指纹 ─────────────────────────────────────────────

/// 文本输出指纹
///
/// 捕获翻译/ASR 文本输出的结构化特征，用于快速判断输出是否变化。
///
/// # 字段说明
/// - `char_count`: 字符数
/// - `word_count`: 单词数（按空格分割）
/// - `line_count`: 行数
/// - `sentence_count`: 句子数（按 .!? 分割）
/// - `avg_word_length`: 平均单词长度
/// - `cjk_ratio`: CJK 字符占比
/// - `ascii_ratio`: ASCII 字符占比
/// - `digit_ratio`: 数字字符占比
/// - `punctuation_ratio`: 标点占比
/// - `sha256_hash`: 完整文本的 SHA-256 哈希
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextFingerprint {
    pub char_count: usize,
    pub word_count: usize,
    pub line_count: usize,
    pub sentence_count: usize,
    pub avg_word_length: f64,
    pub cjk_ratio: f64,
    pub ascii_ratio: f64,
    pub digit_ratio: f64,
    pub punctuation_ratio: f64,
    pub sha256_hash: String,
}

impl TextFingerprint {
    /// 从文本计算指纹
    pub fn from_text(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        let char_count = chars.len();

        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();

        let line_count = text.lines().count().max(1);

        let sentence_count = text
            .split(|c| c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？')
            .filter(|s| !s.trim().is_empty())
            .count()
            .max(1);

        let total_word_len: usize = words.iter().map(|w| w.chars().count()).sum();
        let avg_word_length = if word_count > 0 {
            total_word_len as f64 / word_count as f64
        } else {
            0.0
        };

        let mut cjk_count = 0;
        let mut ascii_count = 0;
        let mut digit_count = 0;
        let mut punct_count = 0;

        for c in &chars {
            if ('\u{4E00}'..='\u{9FFF}').contains(c)
                || ('\u{3400}'..='\u{4DBF}').contains(c)
                || ('\u{F900}'..='\u{FAFF}').contains(c)
            {
                cjk_count += 1;
            }
            if c.is_ascii() {
                ascii_count += 1;
            }
            if c.is_ascii_digit() {
                digit_count += 1;
            }
            if c.is_ascii_punctuation() || matches!(c, '。' | '，' | '！' | '？' | '；' | '：' | '、' | '《' | '》' | '【' | '】' | '「' | '」' | '『' | '』') {
                punct_count += 1;
            }
        }

        let cjk_ratio = if char_count > 0 { cjk_count as f64 / char_count as f64 } else { 0.0 };
        let ascii_ratio = if char_count > 0 { ascii_count as f64 / char_count as f64 } else { 0.0 };
        let digit_ratio = if char_count > 0 { digit_count as f64 / char_count as f64 } else { 0.0 };
        let punctuation_ratio = if char_count > 0 { punct_count as f64 / char_count as f64 } else { 0.0 };

        let hash = sha2_hash(text.as_bytes());

        Self {
            char_count,
            word_count,
            line_count,
            sentence_count,
            avg_word_length,
            cjk_ratio,
            ascii_ratio,
            digit_ratio,
            punctuation_ratio,
            sha256_hash: hash,
        }
    }
}

impl Fingerprint for TextFingerprint {
    fn compare(&self, other: &Self) -> FingerprintCompare {
        if self.sha256_hash == other.sha256_hash {
            return FingerprintCompare::Match;
        }

        let tol = TextTolerances::default();

        let char_diff = (self.char_count as f64 - other.char_count as f64).abs();
        let word_diff = (self.word_count as f64 - other.word_count as f64).abs();
        let avg_diff = (self.avg_word_length - other.avg_word_length).abs();
        let cjk_diff = (self.cjk_ratio - other.cjk_ratio).abs();

        let all_match = char_diff <= tol.char_count_diff
            && word_diff <= tol.word_count_diff
            && avg_diff <= tol.avg_word_length_diff
            && cjk_diff <= tol.cjk_ratio_diff;

        if all_match {
            FingerprintCompare::ApproximateMatch
        } else {
            let mut changes = Vec::new();
            if char_diff > tol.char_count_diff {
                changes.push(format!(
                    "char_count: {} → {} (Δ={:.0}, tol={:.0})",
                    self.char_count, other.char_count, char_diff, tol.char_count_diff
                ));
            }
            if word_diff > tol.word_count_diff {
                changes.push(format!(
                    "word_count: {} → {} (Δ={:.0}, tol={:.0})",
                    self.word_count, other.word_count, word_diff, tol.word_count_diff
                ));
            }
            if avg_diff > tol.avg_word_length_diff {
                changes.push(format!(
                    "avg_word_length: {:.2} → {:.2} (Δ={:.2}, tol={:.2})",
                    self.avg_word_length, other.avg_word_length, avg_diff, tol.avg_word_length_diff
                ));
            }
            if cjk_diff > tol.cjk_ratio_diff {
                changes.push(format!(
                    "cjk_ratio: {:.4} → {:.4} (Δ={:.4}, tol={:.4})",
                    self.cjk_ratio, other.cjk_ratio, cjk_diff, tol.cjk_ratio_diff
                ));
            }
            FingerprintCompare::Changed { changes }
        }
    }

    fn summary(&self) -> String {
        format!(
            "chars={}, words={}, lines={}, sentences={}, avg_wl={:.2}, cjk={:.2}%, ascii={:.2}%",
            self.char_count,
            self.word_count,
            self.line_count,
            self.sentence_count,
            self.avg_word_length,
            self.cjk_ratio * 100.0,
            self.ascii_ratio * 100.0
        )
    }
}

// ─── 指纹 Trait + 比较结果 ─────────────────────────────────

/// 指纹通用 trait
pub trait Fingerprint: fmt::Debug + Clone + serde::Serialize + serde::de::DeserializeOwned {
    /// 将当前指纹与另一个指纹比较
    fn compare(&self, other: &Self) -> FingerprintCompare;

    /// 生成人类可读的摘要
    fn summary(&self) -> String;
}

/// 指纹比较结果
#[derive(Debug, Clone, PartialEq)]
pub enum FingerprintCompare {
    /// 精确匹配（SHA-256 哈希一致）
    Match,
    /// 近似匹配（声学/文本特征在容差范围内）
    ApproximateMatch,
    /// 发生变化，需要人工审查
    Changed { changes: Vec<String> },
}

impl FingerprintCompare {
    /// 是否通过（Match 或 ApproximateMatch）
    pub fn is_pass(&self) -> bool {
        matches!(self, FingerprintCompare::Match | FingerprintCompare::ApproximateMatch)
    }

    /// 变更描述（用于测试失败信息）
    pub fn diff_message(&self) -> String {
        match self {
            FingerprintCompare::Match => "Exact match".to_string(),
            FingerprintCompare::ApproximateMatch => "Approximate match (within tolerance)".to_string(),
            FingerprintCompare::Changed { changes } => {
                format!("Changed ({} differences):\n  - {}", changes.len(), changes.join("\n  - "))
            }
        }
    }
}

// ─── 容差配置 ─────────────────────────────────────────────

/// 音频指纹容差
///
/// 不同运行之间允许的特征差异范围。
/// 用于判断"近似匹配"——算法有微小变化但输出本质相同。
#[derive(Debug, Clone)]
pub struct AudioTolerances {
    pub duration: f64,
    pub rms: f64,
    pub peak: f64,
    pub dc_offset: f64,
    pub zero_crossing_rate: f64,
    pub spectral_flatness: f64,
    pub energy_in_speech_band: f64,
}

impl Default for AudioTolerances {
    fn default() -> Self {
        Self {
            duration: 0.01,           // 10ms
            rms: 0.005,                // 0.5%
            peak: 0.01,                // 1%
            dc_offset: 0.001,          // 0.1%
            zero_crossing_rate: 0.01,  // 1%
            spectral_flatness: 0.01,   // 1%
            energy_in_speech_band: 0.02, // 2%
        }
    }
}

/// 文本指纹容差
#[derive(Debug, Clone)]
pub struct TextTolerances {
    pub char_count_diff: f64,
    pub word_count_diff: f64,
    pub avg_word_length_diff: f64,
    pub cjk_ratio_diff: f64,
}

impl Default for TextTolerances {
    fn default() -> Self {
        Self {
            char_count_diff: 3.0,
            word_count_diff: 2.0,
            avg_word_length_diff: 0.5,
            cjk_ratio_diff: 0.05,
        }
    }
}

// ─── Golden Master 管理器 ─────────────────────────────────

/// Golden Master 基线管理器
///
/// 管理基线文件的保存、加载和更新。
///
/// # 文件结构
/// 基线文件存储为 JSON，路径格式：
/// ```text
/// test_golden/
/// └── module_name/
///     └── test_name.json
/// ```
///
/// # 工作流程
/// 1. `load_or_create()` — 加载基线，不存在则创建
/// 2. `compare()` — 比较当前输出与基线
/// 3. `accept()` — 接受当前输出作为新基线（更新文件）
pub struct GoldenMaster {
    /// 基线文件根目录
    baseline_dir: PathBuf,
}

impl GoldenMaster {
    /// 创建 Golden Master 管理器
    ///
    /// # 参数
    /// - `baseline_dir`: 基线文件存储目录（通常在 `test_golden/` 下）
    pub fn new(baseline_dir: impl AsRef<Path>) -> Self {
        Self {
            baseline_dir: baseline_dir.as_ref().to_path_buf(),
        }
    }

    /// 使用默认路径创建（`test_golden/`）
    pub fn default_path() -> Self {
        Self::new("test_golden")
    }

    /// 获取基线文件路径
    ///
    /// 格式：`baseline_dir/module/test_name.json`
    fn baseline_path(&self, module: &str, test_name: &str) -> PathBuf {
        self.baseline_dir.join(module).join(format!("{test_name}.json"))
    }

    /// 加载基线指纹，如果不存在则保存当前指纹作为基线
    ///
    /// # 参数
    /// - `module`: 模块名（如 "tts", "translate", "asr"）
    /// - `test_name`: 测试名（如 "chinese_hello_world"）
    /// - `current`: 当前输出指纹
    ///
    /// # 返回
    /// - `Ok(FingerprintCompare)` — 基线存在时返回比较结果
    /// - `Ok(FingerprintCompare::Match)` — 基线不存在时保存当前指纹并返回 Match
    /// - `Err(String)` — IO/序列化错误
    pub fn load_or_create<F: Fingerprint>(
        &self,
        module: &str,
        test_name: &str,
        current: &F,
    ) -> Result<FingerprintCompare, String> {
        let path = self.baseline_path(module, test_name);

        if path.exists() {
            // 加载现有基线
            let json = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read baseline {path:?}: {e}"))?;
            let baseline: F = serde_json::from_str(&json)
                .map_err(|e| format!("Failed to deserialize baseline {path:?}: {e}"))?;
            Ok(current.compare(&baseline))
        } else {
            // 首次运行：保存当前指纹作为基线
            self.save(module, test_name, current)?;
            Ok(FingerprintCompare::Match)
        }
    }

    /// 保存指纹为基线
    pub fn save<F: Fingerprint>(
        &self,
        module: &str,
        test_name: &str,
        fingerprint: &F,
    ) -> Result<(), String> {
        let path = self.baseline_path(module, test_name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create dir {parent:?}: {e}"))?;
        }
        let json = serde_json::to_string_pretty(fingerprint)
            .map_err(|e| format!("Failed to serialize fingerprint: {e}"))?;
        std::fs::write(&path, json)
            .map_err(|e| format!("Failed to write baseline {path:?}: {e}"))?;
        Ok(())
    }

    /// 接受当前输出作为新基线（更新 golden master）
    ///
    /// 用于 AI 改了代码后，人工审查 diff 并决定接受新输出时。
    pub fn accept<F: Fingerprint>(
        &self,
        module: &str,
        test_name: &str,
        fingerprint: &F,
    ) -> Result<(), String> {
        tracing::info!(
            "Golden master accepted: {module}/{test_name} — baseline updated"
        );
        self.save(module, test_name, fingerprint)
    }

    /// 列出所有基线
    pub fn list_baselines(&self) -> Vec<(String, String)> {
        let mut result = Vec::new();
        if !self.baseline_dir.exists() {
            return result;
        }
        if let Ok(entries) = std::fs::read_dir(&self.baseline_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let module = entry.file_name().to_string_lossy().to_string();
                    if let Ok(tests) = std::fs::read_dir(entry.path()) {
                        for test in tests.flatten() {
                            if test.path().extension().and_then(|e| e.to_str()) == Some("json") {
                                let test_name = test
                                    .path()
                                    .file_stem()
                                    .map(|s| s.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                result.push((module.clone(), test_name));
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// 清除所有基线（慎用！）
    ///
    /// 用于完全重置 golden master。
    pub fn clear_all(&self) -> Result<(), String> {
        if self.baseline_dir.exists() {
            std::fs::remove_dir_all(&self.baseline_dir)
                .map_err(|e| format!("Failed to remove baseline dir: {e}"))?;
        }
        Ok(())
    }
}

// ─── 辅助函数 ─────────────────────────────────────────────

/// 计算 SHA-256 哈希
fn sha2_hash(data: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    let hash = hasher.finalize();
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// 多声道混合为单声道
fn mix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// 计算频谱平坦度（简化版，不依赖 FFT 库）
///
/// 使用 Goertzel 算法计算若干频段的能量，
/// 然后用几何均值/算术均值的比值作为平坦度。
///
/// - 接近 1.0 = 白噪声（各频段能量均匀）
/// - 接近 0.0 = 有调性信号（语音/音乐）
fn compute_spectral_flatness(samples: &[f32], sample_rate: u32) -> f64 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }

    // 选择 8 个频段：100, 200, 500, 1000, 2000, 4000, 8000, 16000 Hz
    // 使用更多频段以提高频谱平坦度的准确性
    let freqs: Vec<f64> = (0..32).map(|i| 50.0 * (i + 1) as f64).collect();
    let mut energies = Vec::with_capacity(freqs.len());

    for &freq in &freqs {
        let energy = goertzel_energy(samples, sample_rate, freq);
        energies.push(energy);
    }

    // 几何均值 / 算术均值 = 频谱平坦度
    let n = energies.len() as f64;
    let arithmetic_mean: f64 = energies.iter().sum::<f64>() / n;

    if arithmetic_mean <= 0.0 {
        return 0.0;
    }

    // 几何均值 = exp(mean(ln(x)))
    let log_sum: f64 = energies
        .iter()
        .filter(|e| **e > 0.0)
        .map(|e| e.ln())
        .sum();
    let valid_count = energies.iter().filter(|e| **e > 0.0).count() as f64;

    if valid_count == 0.0 {
        return 0.0;
    }

    let geometric_mean = (log_sum / valid_count).exp();

    geometric_mean / arithmetic_mean
}

/// 使用 Goertzel 算法计算指定频率的能量
///
/// 比 DFT 更高效的单频率检测算法。
fn goertzel_energy(samples: &[f32], sample_rate: u32, target_freq: f64) -> f64 {
    let n = samples.len();
    if n == 0 || sample_rate == 0 {
        return 0.0;
    }

    let k = target_freq * n as f64 / sample_rate as f64;
    let omega = 2.0 * std::f64::consts::PI * k / n as f64;
    let coeff = 2.0 * omega.cos();

    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;

    for &sample in samples {
        let s0 = sample as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    // 能量 = s1^2 + s2^2 - coeff * s1 * s2
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

/// 计算语音频段能量占比
///
/// 语音频段定义为 [low, high] Hz（默认 300-3400 Hz）。
/// 返回该频段能量占总能量的比例。
fn compute_speech_band_energy_ratio(
    samples: &[f32],
    sample_rate: u32,
    low: f64,
    high: f64,
) -> f64 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }

    // 使用多个频段计算
    let all_freqs = [50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3400.0, 5000.0, 8000.0, 12000.0, 16000.0];

    let mut speech_energy = 0.0_f64;
    let mut total_energy = 0.0_f64;

    for &freq in &all_freqs {
        let energy = goertzel_energy(samples, sample_rate, freq);
        total_energy += energy;
        if freq >= low && freq <= high {
            speech_energy += energy;
        }
    }

    if total_energy > 0.0 {
        speech_energy / total_energy
    } else {
        0.0
    }
}

// ─── 测试辅助工具 ─────────────────────────────────────────

/// Golden Master 测试用例
///
/// 封装了 "生成指纹 → 对比基线 → 报告" 的完整流程，
/// 使命用 Golden Master 的测试代码更简洁。
///
/// # 使用方式
/// ```no_run
/// use vt_core::golden_master::{GoldenMasterTestCase, AudioFingerprint};
///
/// // 在测试中
/// let samples = vec![0.5_f32; 1000];
/// let fp = AudioFingerprint::from_samples(&samples, 24000);
///
/// GoldenMasterTestCase::new("tts", "chinese_hello")
///     .with_fingerprint(&fp)
///     .assert_pass();
/// ```
pub struct GoldenMasterTestCase {
    module: String,
    name: String,
    gm: GoldenMaster,
}

impl GoldenMasterTestCase {
    /// 创建测试用例
    pub fn new(module: &str, name: &str) -> Self {
        Self {
            module: module.to_string(),
            name: name.to_string(),
            gm: GoldenMaster::default_path(),
        }
    }

    /// 使用自定义基线目录
    pub fn with_baseline_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.gm = GoldenMaster::new(dir);
        self
    }

    /// 对比指纹与基线
    pub fn compare<F: Fingerprint>(&self, fingerprint: &F) -> FingerprintCompare {
        self.gm
            .load_or_create(&self.module, &self.name, fingerprint)
            .unwrap_or_else(|e| {
                panic!(
                    "Golden master error for {}/{}: {e}",
                    self.module, self.name
                )
            })
    }

    /// 断言通过（Match 或 ApproximateMatch）
    pub fn assert_pass<F: Fingerprint>(&self, fingerprint: &F) {
        let result = self.compare(fingerprint);
        assert!(
            result.is_pass(),
            "Golden master test {}/{} FAILED:\n{}\n\n\
             If this change is intentional, run with VT_GOLDEN_ACCEPT=1 to update the baseline.",
            self.module,
            self.name,
            result.diff_message()
        );
    }

    /// 接受当前输出作为新基线
    pub fn accept<F: Fingerprint>(&self, fingerprint: &F) {
        self.gm
            .accept(&self.module, &self.name, fingerprint)
            .unwrap_or_else(|e| {
                panic!("Failed to accept golden master: {e}")
            });
    }
}

/// 生成测试用 WAV 文件（正弦波 + 噪声）
///
/// 用于在没有 TTS 模型的情况下测试 Golden Master 音频指纹流程。
///
/// # 参数
/// - `path`: 输出 WAV 文件路径
/// - `freq`: 正弦波频率（Hz）
/// - `duration_secs`: 时长（秒）
/// - `sample_rate`: 采样率
/// - `amplitude`: 振幅 (0.0 ~ 1.0)
/// - `noise_level`: 噪声水平 (0.0 ~ 1.0)
pub fn generate_test_wav(
    path: &Path,
    freq: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f32,
    noise_level: f32,
) -> Result<(), String> {
    let n = (duration_secs * sample_rate as f64) as usize;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| format!("Failed to create WAV writer: {e}"))?;

    // 简单 LCG 伪随机噪声
    let mut state: u64 = 42;

    for i in 0..n {
        let t = i as f64 / sample_rate as f64;
        let sine_val = (2.0 * std::f64::consts::PI * freq * t).sin() as f32;
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * noise_level;

        let sample = (sine_val * amplitude + noise).clamp(-1.0, 1.0);
        let int_sample = (sample * 32767.0) as i16;
        writer
            .write_sample(int_sample)
            .map_err(|e| format!("Failed to write sample: {e}"))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV: {e}"))?;

    Ok(())
}

// ─── 单元测试 ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AudioFingerprint 测试 ──

    #[test]
    fn test_audio_fingerprint_silence() {
        let samples = vec![0.0_f32; 2400];
        let fp = AudioFingerprint::from_samples(&samples, 24000);
        assert_eq!(fp.sample_count, 2400);
        assert!((fp.rms - 0.0).abs() < 1e-10);
        assert!((fp.peak - 0.0).abs() < 1e-10);
        assert!((fp.dc_offset - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_audio_fingerprint_sine_wave() {
        // 1kHz 正弦波，1秒，24kHz 采样率
        let sample_rate = 24000_u32;
        let freq = 1000.0_f64;
        let samples: Vec<f32> = (0..sample_rate as usize)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.5
            })
            .collect();
        let fp = AudioFingerprint::from_samples(&samples, sample_rate);

        assert_eq!(fp.sample_count, 24000);
        assert!((fp.duration_secs - 1.0).abs() < 1e-6);
        assert!(fp.rms > 0.3, "RMS should be ~0.35 for 0.5 amplitude sine");
        assert!((fp.peak - 0.5).abs() < 0.01);
        assert!(fp.dc_offset.abs() < 0.01, "DC offset should be ~0 for symmetric sine");
    }

    #[test]
    fn test_audio_fingerprint_deterministic() {
        let samples = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let fp1 = AudioFingerprint::from_samples(&samples, 8000);
        let fp2 = AudioFingerprint::from_samples(&samples, 8000);
        assert_eq!(fp1.sha256_hash, fp2.sha256_hash);
        assert_eq!(fp1.compare(&fp2), FingerprintCompare::Match);
    }

    #[test]
    fn test_audio_fingerprint_detect_change() {
        let samples1 = vec![0.5_f32; 100];
        let samples2 = vec![0.8_f32; 100];
        let fp1 = AudioFingerprint::from_samples(&samples1, 8000);
        let fp2 = AudioFingerprint::from_samples(&samples2, 8000);
        assert_ne!(fp1.sha256_hash, fp2.sha256_hash);
        assert!(!fp1.compare(&fp2).is_pass(), "Different amplitudes should not pass");
    }

    #[test]
    fn test_audio_fingerprint_approximate_match() {
        // 微小浮点差异应在容差内匹配
        let samples1: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let samples2: Vec<f32> = samples1.iter().map(|s| s + 1e-7).collect();
        let fp1 = AudioFingerprint::from_samples(&samples1, 8000);
        let fp2 = AudioFingerprint::from_samples(&samples2, 8000);
        let cmp = fp1.compare(&fp2);
        assert!(cmp.is_pass(), "Tiny float diff should pass: {}", cmp.diff_message());
    }

    // ── TextFingerprint 测试 ──

    #[test]
    fn test_text_fingerprint_english() {
        let text = "Hello world. This is a test.";
        let fp = TextFingerprint::from_text(text);
        assert_eq!(fp.char_count, 28); // "Hello world. This is a test." = 28 chars
        assert!(fp.word_count >= 5);
        assert!(fp.ascii_ratio > 0.9);
        assert!(fp.cjk_ratio < 0.01);
    }

    #[test]
    fn test_text_fingerprint_chinese() {
        let text = "你好世界。这是一个测试。";
        let fp = TextFingerprint::from_text(text);
        assert!(fp.cjk_ratio > 0.5, "Chinese text should have high CJK ratio: {}", fp.summary());
        assert!(fp.ascii_ratio < 0.1);
    }

    #[test]
    fn test_text_fingerprint_deterministic() {
        let text = "Same text produces same fingerprint.";
        let fp1 = TextFingerprint::from_text(text);
        let fp2 = TextFingerprint::from_text(text);
        assert_eq!(fp1.sha256_hash, fp2.sha256_hash);
        assert_eq!(fp1.compare(&fp2), FingerprintCompare::Match);
    }

    #[test]
    fn test_text_fingerprint_detect_change() {
        let text1 = "This is a short text.";
        let text2 = "This is a completely different and much longer text with more words.";
        let fp1 = TextFingerprint::from_text(text1);
        let fp2 = TextFingerprint::from_text(text2);
        assert!(!fp1.compare(&fp2).is_pass(), "Different texts should not pass");
    }

    // ── GoldenMaster 管理器测试 ──

    #[test]
    fn test_golden_master_create_and_load() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let gm = GoldenMaster::new(dir.path());

        let samples = vec![0.5_f32; 100];
        let fp = AudioFingerprint::from_samples(&samples, 8000);

        // 首次：创建基线
        let result = gm.load_or_create("test_module", "test_case_1", &fp)
            .expect("load_or_create failed");
        assert_eq!(result, FingerprintCompare::Match);
        assert!(gm.baseline_path("test_module", "test_case_1").exists());

        // 第二次：相同数据 → Match
        let result = gm.load_or_create("test_module", "test_case_1", &fp)
            .expect("load_or_create failed");
        assert_eq!(result, FingerprintCompare::Match);

        // 第三次：不同数据 → Changed
        let samples2 = vec![0.8_f32; 100];
        let fp2 = AudioFingerprint::from_samples(&samples2, 8000);
        let result = gm.load_or_create("test_module", "test_case_1", &fp2)
            .expect("load_or_create failed");
        assert!(!result.is_pass(), "Changed data should fail");

        // 接受新基线
        gm.accept("test_module", "test_case_1", &fp2)
            .expect("accept failed");

        // 再次加载 → Match
        let result = gm.load_or_create("test_module", "test_case_1", &fp2)
            .expect("load_or_create failed");
        assert!(result.is_pass());
    }

    #[test]
    fn test_golden_master_list_baselines() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let gm = GoldenMaster::new(dir.path());

        let fp = TextFingerprint::from_text("test");
        gm.save("module_a", "test_1", &fp).expect("save failed");
        gm.save("module_a", "test_2", &fp).expect("save failed");
        gm.save("module_b", "test_3", &fp).expect("save failed");

        let baselines = gm.list_baselines();
        assert_eq!(baselines.len(), 3);
    }

    // ── 辅助函数测试 ──

    #[test]
    fn test_goertzel_energy_detects_frequency() {
        let sample_rate = 8000_u32;
        let freq = 1000.0_f64;
        // 生成 1kHz 正弦波
        let samples: Vec<f32> = (0..sample_rate as usize)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.5
            })
            .collect();

        let e_target = goertzel_energy(&samples, sample_rate, 1000.0);
        let e_other = goertzel_energy(&samples, sample_rate, 500.0);
        assert!(
            e_target > e_other * 10.0,
            "Target freq energy should be much higher: target={e_target}, other={e_other}"
        );
    }

    #[test]
    fn test_spectral_flatness_silence() {
        let samples = vec![0.0_f32; 1000];
        let sf = compute_spectral_flatness(&samples, 8000);
        assert!(sf < 0.01, "Silence should have low flatness: {sf}");
    }

    #[test]
    fn test_spectral_flatness_white_noise() {
        // 伪白噪声（使用简单 LCG）
        let mut state: u64 = 42;
        let samples: Vec<f32> = (0..2000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.5
            })
            .collect();
        let sf = compute_spectral_flatness(&samples, 8000);
        // 白噪声应有较高的平坦度
        assert!(sf > 0.3, "White noise should have high flatness: {sf}");
    }

    #[test]
    fn test_mix_to_mono_stereo() {
        let stereo = vec![0.5_f32, 0.3, 0.7, 0.1, 0.9, 0.2]; // 3 frames × 2 channels
        let mono = mix_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_fingerprint_compare_diff_message() {
        let cmp = FingerprintCompare::Match;
        assert_eq!(cmp.diff_message(), "Exact match");

        let cmp = FingerprintCompare::ApproximateMatch;
        assert!(cmp.diff_message().contains("Approximate"));

        let cmp = FingerprintCompare::Changed {
            changes: vec!["field1 changed".to_string(), "field2 changed".to_string()],
        };
        let msg = cmp.diff_message();
        assert!(msg.contains("2 differences"));
        assert!(msg.contains("field1 changed"));
    }
}
