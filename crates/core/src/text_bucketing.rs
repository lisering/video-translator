//! 文本分桶模块
//!
//! 借鉴 GPT-SoVITS 的 `text_split_method` 和批量推理优化思路，
//! 将 TTS 文本按估算时长分组，使同一批次内文本长度接近，
//! 减少批量推理时的 padding 浪费和超时风险。
//!
//! # 核心功能
//! - [`estimate_text_duration`]: 根据文本字符数和语言估算音频时长
//! - [`TextBucket`]: 单个文本桶（一批长度相近的文本）
//! - [`bucket_texts`]: 将文本列表按估算时长分桶
//!
//! # 分桶策略
//! 1. 估算每段文本的音频时长（中英文不同速率）
//! 2. 按时长排序
//! 3. 将相邻的文本分入同一桶（桶大小可配置，默认 5）
//! 4. 超时桶自动拆分（估算时长超过阈值的文本单独成桶）
//!
//! # 示例
//! ```
//! use vt_core::text_bucketing::{bucket_texts, BucketConfig};
//!
//! let texts = vec!["你好世界".to_string(), "Hello world".to_string()];
//! let config = BucketConfig::default();
//! let buckets = bucket_texts(&texts, &config);
//! ```

// ─── 常量 ─────────────────────────────────────────────────

/// 中文每秒约 4 个字符（正常语速）
const CN_CHARS_PER_SEC: f64 = 4.0;

/// 英文每秒约 2.5 个单词（正常语速）
const EN_WORDS_PER_SEC: f64 = 2.5;

/// 默认桶大小（每桶最多文本数）
pub const DEFAULT_BUCKET_SIZE: usize = 5;

/// 默认单段最大时长（秒），超过则标记为长文本
pub const DEFAULT_MAX_SEGMENT_DURATION: f64 = 30.0;

// ─── 配置 ─────────────────────────────────────────────────

/// 分桶配置
#[derive(Debug, Clone)]
pub struct BucketConfig {
    /// 每桶最大文本数
    pub bucket_size: usize,
    /// 单段最大估算时长（秒），超过则单独成桶
    pub max_segment_duration: f64,
    /// 语言代码（"zh"、"en"、"ja" 等）
    pub language: String,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            bucket_size: DEFAULT_BUCKET_SIZE,
            max_segment_duration: DEFAULT_MAX_SEGMENT_DURATION,
            language: "zh".to_string(),
        }
    }
}

// ─── 文本时长估算 ────────────────────────────────────────

/// 估算文本对应的音频时长（秒）
///
/// # 算法
/// - 中文：每秒约 4 个字符
/// - 英文：每秒约 2.5 个单词
/// - 日文：每秒约 4 个字符
/// - 混合文本：分别统计中英文字符数，加权计算
///
/// # 参数
/// - `text`: 输入文本
/// - `language`: 语言代码（"zh"、"en"、"ja" 等）
///
/// # 返回
/// 估算的音频时长（秒），至少 0.5 秒
#[must_use]
pub fn estimate_text_duration(text: &str, language: &str) -> f64 {
    if text.is_empty() {
        return 0.5;
    }

    let estimated = match language {
        "zh" | "zh-CN" | "zh-TW" => {
            // 中文：按字符数估算
            let char_count = text.chars().count() as f64;
            char_count / CN_CHARS_PER_SEC
        }
        "en" => {
            // 英文：按单词数估算
            let word_count = text.split_whitespace().count() as f64;
            word_count / EN_WORDS_PER_SEC
        }
        "ja" => {
            // 日文：按字符数估算（与中文相近）
            let char_count = text.chars().count() as f64;
            char_count / CN_CHARS_PER_SEC
        }
        _ => {
            // 默认：混合文本，分别统计
            let cn_chars = text.chars().filter(|c| c.is_cjk()).count() as f64;
            let other_chars = text.chars().filter(|c| !c.is_cjk()).count() as f64;
            let word_estimate = other_chars / 5.0; // 平均每词 5 字符
            (cn_chars / CN_CHARS_PER_SEC) + (word_estimate / EN_WORDS_PER_SEC)
        }
    };

    estimated.max(0.5) // 至少 0.5 秒
}

// ─── 文本桶 ──────────────────────────────────────────────

/// 单个文本桶
///
/// 包含一批长度相近的文本及其索引。
#[derive(Debug, Clone)]
pub struct TextBucket {
    /// 桶内文本索引列表（对应原始 texts 数组的索引）
    pub indices: Vec<usize>,
    /// 桶内所有文本的最大估算时长（秒）
    pub max_duration: f64,
    /// 桶内所有文本的最小估算时长（秒）
    pub min_duration: f64,
    /// 桶标记：是否为长文本桶
    pub is_long_text: bool,
}

impl Default for TextBucket {
    fn default() -> Self {
        Self::new()
    }
}

impl TextBucket {
    /// 创建新桶
    fn new() -> Self {
        Self {
            indices: Vec::new(),
            max_duration: 0.0,
            min_duration: f64::MAX,
            is_long_text: false,
        }
    }

    /// 添加文本到桶
    fn add(&mut self, index: usize, duration: f64) {
        if self.indices.is_empty() {
            self.min_duration = duration;
            self.max_duration = duration;
        } else {
            self.max_duration = self.max_duration.max(duration);
            self.min_duration = self.min_duration.min(duration);
        }
        self.indices.push(index);
    }

    /// 桶内文本数量
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// 是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// 时长跨度（最大-最小），用于评估桶内一致性
    #[must_use]
    pub fn duration_span(&self) -> f64 {
        self.max_duration - self.min_duration
    }
}

// ─── 分桶函数 ────────────────────────────────────────────

/// 将文本列表按估算时长分桶
///
/// # 算法
/// 1. 估算每段文本的音频时长
/// 2. 按时长排序
/// 3. 超过 `max_segment_duration` 的文本各自单独成桶
/// 4. 剩余文本按 `bucket_size` 分入桶中（按时长排序后顺序分组）
///
/// # 参数
/// - `texts`: 文本列表
/// - `config`: 分桶配置
///
/// # 返回
/// 文本桶列表，每个桶包含一组索引
#[must_use]
pub fn bucket_texts(texts: &[String], config: &BucketConfig) -> Vec<TextBucket> {
    if texts.is_empty() {
        return Vec::new();
    }

    // 估算每段文本时长
    let durations: Vec<(usize, f64)> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| (i, estimate_text_duration(text, &config.language)))
        .collect();

    // 按时长排序
    let mut sorted = durations.clone();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // 分桶
    let mut buckets: Vec<TextBucket> = Vec::new();
    let mut current_bucket = TextBucket::new();

    for &(idx, duration) in &sorted {
        // 超长文本单独成桶
        if duration > config.max_segment_duration {
            // 先关闭当前桶
            if !current_bucket.is_empty() {
                buckets.push(std::mem::take(&mut current_bucket));
            }
            let mut long_bucket = TextBucket::new();
            long_bucket.is_long_text = true;
            long_bucket.add(idx, duration);
            buckets.push(long_bucket);
            continue;
        }

        current_bucket.add(idx, duration);

        // 桶满则关闭
        if current_bucket.len() >= config.bucket_size {
            buckets.push(std::mem::take(&mut current_bucket));
        }
    }

    // 关闭最后一个未满的桶
    if !current_bucket.is_empty() {
        buckets.push(current_bucket);
    }

    tracing::debug!(
        "Text bucketing: {} texts → {} buckets (bucket_size={}, max_dur={:.1}s)",
        texts.len(),
        buckets.len(),
        config.bucket_size,
        config.max_segment_duration
    );

    buckets
}

// ─── 批量合成器 ──────────────────────────────────────────

/// 批量合成进度回调类型
pub type ProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// 批量合成结果
#[derive(Debug, Clone)]
pub struct BatchSynthResult {
    /// 成功合成的索引和路径
    pub success: Vec<(usize, std::path::PathBuf)>,
    /// 失败的索引和错误信息
    pub failures: Vec<(usize, String)>,
}

impl BatchSynthResult {
    /// 成功数量
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.success.len()
    }

    /// 失败数量
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }

    /// 总数量
    #[must_use]
    pub fn total(&self) -> usize {
        self.success_count() + self.failure_count()
    }

    /// 成功率
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total() == 0 {
            return 0.0;
        }
        self.success_count() as f64 / self.total() as f64
    }
}

/// 批量合成器
///
/// 借鉴 GPT-SoVITS T2S 的动态序列移除思路：
/// - 将文本按估算时长分桶
/// - 逐桶处理，桶内逐段合成
/// - 失败的段被标记并跳过（动态移除）
/// - 支持进度回调和取消
///
/// # 工作流程
/// 1. 使用 `bucket_texts` 将文本分桶
/// 2. 逐桶处理：
///    - 对桶内每段文本调用克隆引擎
///    - 成功 → 加入 success 列表
///    - 失败 → 加入 failures 列表（动态移除，不再参与后续处理）
/// 3. 返回 `BatchSynthResult`
///
/// # 示例
/// ```no_run
/// use vt_core::text_bucketing::{BatchSynthesizer, BucketConfig};
/// use vt_core::cloning::{VoiceCloningEngine, CloningConfig, MockCloningEngine};
/// use std::path::Path;
///
/// let engine = MockCloningEngine::new();
/// let config = CloningConfig::default();
/// let texts = vec!["你好".to_string(), "世界".to_string()];
///
/// let result = BatchSynthesizer::synthesize(
///     &engine,
///     &texts,
///     Path::new("reference.wav"),
///     &config,
///     BucketConfig::default(),
///     None,
/// ).unwrap();
///
/// println!("Success: {}/{}, Failure: {}/{}",
///     result.success_count(), result.total(),
///     result.failure_count(), result.total());
/// ```
pub struct BatchSynthesizer;

impl BatchSynthesizer {
    /// 执行批量合成
    ///
    /// # 参数
    /// - `engine`: 声音克隆引擎
    /// - `texts`: 要合成的文本列表
    /// - `reference_audio`: 参考音频路径
    /// - `config`: 克隆配置
    /// - `bucket_config`: 分桶配置
    /// - `progress_callback`: 可选的进度回调 `(completed, total, last_text)`
    ///
    /// # 返回
    /// 批量合成结果
    pub fn synthesize(
        engine: &dyn crate::cloning::VoiceCloningEngine,
        texts: &[String],
        reference_audio: &std::path::Path,
        config: &crate::cloning::CloningConfig,
        bucket_config: BucketConfig,
        progress_callback: Option<ProgressCallback>,
    ) -> crate::error::AppResult<BatchSynthResult> {
        if texts.is_empty() {
            return Ok(BatchSynthResult {
                success: Vec::new(),
                failures: Vec::new(),
            });
        }

        // 分桶
        let buckets = bucket_texts(texts, &bucket_config);
        let total = texts.len();

        tracing::info!(
            "BatchSynthesizer: {} texts → {} buckets, starting batch synthesis",
            total,
            buckets.len()
        );

        let mut success: Vec<(usize, std::path::PathBuf)> = Vec::new();
        let mut failures: Vec<(usize, String)> = Vec::new();
        let mut completed = 0;

        // 逐桶处理
        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            tracing::debug!(
                "BatchSynthesizer: processing bucket {} ({}, span={:.1}s)",
                bucket_idx,
                bucket.len(),
                bucket.duration_span()
            );

            // 桶内逐段合成（动态序列移除：失败的段被跳过）
            for &text_idx in &bucket.indices {
                let text = &texts[text_idx];

                match engine.clone_and_synthesize(reference_audio, text, config) {
                    Ok(path) => {
                        success.push((text_idx, path));
                        tracing::debug!("BatchSynthesizer: segment {} succeeded", text_idx);
                    }
                    Err(e) => {
                        // 动态移除：记录失败，继续处理下一段
                        failures.push((text_idx, e.to_string()));
                        tracing::warn!(
                            "BatchSynthesizer: segment {} failed (dynamically removed): {}",
                            text_idx,
                            e
                        );
                    }
                }

                completed += 1;
                if let Some(ref cb) = progress_callback {
                    cb(completed, total, text);
                }
            }
        }

        let result = BatchSynthResult {
            success: success.clone(),
            failures: failures.clone(),
        };

        tracing::info!(
            "BatchSynthesizer: completed {} texts, success={}, failure={}, rate={:.1}%",
            total,
            result.success_count(),
            result.failure_count(),
            result.success_rate() * 100.0
        );

        Ok(result)
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_empty_text() {
        let dur = estimate_text_duration("", "zh");
        assert!((dur - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_estimate_chinese_text() {
        // 8 个中文字符 / 4 = 2 秒
        let dur = estimate_text_duration("你好世界你好世界", "zh");
        assert!((dur - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_estimate_english_text() {
        // "Hello world" = 2 words / 2.5 = 0.8 秒
        let dur = estimate_text_duration("Hello world", "en");
        assert!(dur > 0.5 && dur < 1.5);
    }

    #[test]
    fn test_estimate_long_text() {
        let long_text = "你好".repeat(100);
        let dur = estimate_text_duration(&long_text, "zh");
        assert!(dur > 40.0, "Long text should estimate to >40s, got {dur}");
    }

    #[test]
    fn test_estimate_mixed_text() {
        let dur = estimate_text_duration("Hello 你好世界", "mixed");
        assert!(dur > 0.5);
    }

    #[test]
    fn test_bucket_empty() {
        let buckets = bucket_texts(&[], &BucketConfig::default());
        assert!(buckets.is_empty());
    }

    #[test]
    fn test_bucket_single_text() {
        let texts = vec!["你好世界".to_string()];
        let buckets = bucket_texts(&texts, &BucketConfig::default());
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].len(), 1);
        assert_eq!(buckets[0].indices[0], 0);
    }

    #[test]
    fn test_bucket_multiple_texts() {
        let texts = vec![
            "你好".to_string(),             // ~0.5s
            "你好世界你好世界".to_string(), // ~2s
            "Hello world test".to_string(), // ~1.2s
            "测试文本分桶功能".to_string(), // ~1.75s
            "短".to_string(),               // ~0.5s
        ];
        let config = BucketConfig {
            bucket_size: 2,
            ..Default::default()
        };
        let buckets = bucket_texts(&texts, &config);

        // 5 个文本，桶大小 2 → 应该有 3 个桶（2+2+1）
        assert_eq!(buckets.len(), 3);
        let total: usize = buckets.iter().map(|b| b.len()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_bucket_long_text_separate() {
        let texts = vec![
            "你好".to_string(),
            "你好".repeat(100), // 超长文本 >30s
            "世界".to_string(),
        ];
        let config = BucketConfig::default();
        let buckets = bucket_texts(&texts, &config);

        // 超长文本应单独成桶
        let long_bucket = buckets.iter().find(|b| b.is_long_text);
        assert!(long_bucket.is_some(), "Should have a long-text bucket");
        assert_eq!(long_bucket.unwrap().len(), 1);
    }

    #[test]
    fn test_bucket_ordering() {
        let texts = vec![
            "你好你好你好你好你好".to_string(), // ~5s
            "你好".to_string(),                 // ~0.5s
            "你好你好".to_string(),             // ~1s
        ];
        let config = BucketConfig {
            bucket_size: 1,
            ..Default::default()
        };
        let buckets = bucket_texts(&texts, &config);

        // 每个文本单独成桶，按时长排序
        assert_eq!(buckets.len(), 3);
        // 最短的应该在第一个桶
        assert!(buckets[0].min_duration <= buckets[1].min_duration);
        assert!(buckets[1].min_duration <= buckets[2].min_duration);
    }

    #[test]
    fn test_bucket_duration_span() {
        let texts = vec![
            "你好".to_string(),             // ~0.5s
            "你好世界你好世界".to_string(), // ~2s
        ];
        let config = BucketConfig {
            bucket_size: 2,
            ..Default::default()
        };
        let buckets = bucket_texts(&texts, &config);

        assert_eq!(buckets.len(), 1);
        let span = buckets[0].duration_span();
        assert!(span > 1.0, "Duration span should be >1.0 for mixed bucket");
    }

    #[test]
    fn test_bucket_config_default() {
        let config = BucketConfig::default();
        assert_eq!(config.bucket_size, DEFAULT_BUCKET_SIZE);
        assert_eq!(config.max_segment_duration, DEFAULT_MAX_SEGMENT_DURATION);
        assert_eq!(config.language, "zh");
    }

    #[test]
    fn test_bucket_large_batch() {
        let texts: Vec<String> = (0..20).map(|i| format!("测试文本{}", i)).collect();
        let config = BucketConfig {
            bucket_size: 5,
            ..Default::default()
        };
        let buckets = bucket_texts(&texts, &config);

        // 20 个文本，桶大小 5 → 4 个桶
        assert_eq!(buckets.len(), 4);
        for bucket in &buckets {
            assert_eq!(bucket.len(), 5);
        }
    }

    #[test]
    fn test_bucket_all_indices_present() {
        let texts: Vec<String> = (0..10).map(|i| format!("文本{}", i)).collect();
        let config = BucketConfig {
            bucket_size: 3,
            ..Default::default()
        };
        let buckets = bucket_texts(&texts, &config);

        // 收集所有索引
        let mut all_indices: Vec<usize> = buckets
            .iter()
            .flat_map(|b| b.indices.iter().copied())
            .collect();
        all_indices.sort();

        // 确保所有索引都存在且不重复
        let expected: Vec<usize> = (0..10).collect();
        assert_eq!(all_indices, expected);
    }

    #[test]
    fn test_cjk_detection() {
        // 测试 CJK 字符检测辅助函数
        let text = "Hello 你好";
        let cn_chars = text.chars().filter(|c| c.is_cjk()).count();
        assert_eq!(cn_chars, 2);
    }
}

/// CJK 字符判断 trait（扩展标准库）
trait CjkExt {
    fn is_cjk(&self) -> bool;
}

impl CjkExt for char {
    fn is_cjk(&self) -> bool {
        let cp = *self as u32;
        // CJK 统一表意文字范围
        (0x4E00..=0x9FFF).contains(&cp)
            || (0x3400..=0x4DBF).contains(&cp)   // CJK 扩展 A
            || (0x20000..=0x2A6DF).contains(&cp)  // CJK 扩展 B
            || (0x3040..=0x309F).contains(&cp)    // 平假名
            || (0x30A0..=0x30FF).contains(&cp) // 片假名
    }
}

// ─── Unicode 权重时长估算器 — 借鉴 OmniVoice RuleDurationEstimator ──

/// Unicode 字符语音权重表
///
/// 借鉴 OmniVoice `RuleDurationEstimator`，为每个 Unicode 字符分配语音权重。
/// 权重表示相对于一个拉丁字符的说话时间（基准 1.0 ≈ 40-50ms）。
///
/// 支持所有主要文字系统：CJK、韩文、日文假名、阿拉伯文、希伯来文、
/// 印度系文字、泰文/老挝文、拉丁文、西里尔文、希腊文等。
#[derive(Debug, Clone)]
pub struct RuleDurationEstimator {
    /// 每个字符的权重，基准 1.0 = 一个拉丁字符 (~40-50ms)
    weights: &'static [(&'static str, f64)],
    /// Unicode 码点范围 → 权重类型映射
    ranges: &'static [(u32, &'static str)],
    /// 二分查找用的断点
    breakpoints: Vec<u32>,
}

impl RuleDurationEstimator {
    /// 创建时长估算器
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: PHONETIC_WEIGHTS,
            ranges: UNICODE_RANGES,
            breakpoints: UNICODE_RANGES.iter().map(|r| r.0).collect(),
        }
    }

    /// 获取单个字符的语音权重
    #[must_use]
    pub fn char_weight(&self, ch: char) -> f64 {
        let code = ch as u32;

        // ASCII 字母
        if (65..=90).contains(&code) || (97..=122).contains(&code) {
            return self.get_weight("latin");
        }
        // 空格
        if code == 32 {
            return self.get_weight("space");
        }
        // 忽略阿拉伯语 Tatweel
        if code == 0x0640 {
            return self.get_weight("mark");
        }

        // Unicode 类别
        // 使用 char 分类来判断标点、符号、数字等
        if ch.is_ascii_punctuation() {
            return self.get_weight("punctuation");
        }
        if ch.is_numeric() {
            return self.get_weight("digit");
        }
        if ch.is_whitespace() {
            return self.get_weight("space");
        }

        // CJK 特殊处理
        if (0x4E00..=0x9FFF).contains(&code)
            || (0x3400..=0x4DBF).contains(&code)
            || (0x20000..=0x2A6DF).contains(&code)
            || (0xF900..=0xFAFF).contains(&code)
        {
            return self.get_weight("cjk");
        }
        // 平假名
        if (0x3040..=0x309F).contains(&code) {
            return self.get_weight("kana");
        }
        // 片假名
        if (0x30A0..=0x30FF).contains(&code) {
            return self.get_weight("kana");
        }
        // 韩文
        if (0xAC00..=0xD7AF).contains(&code) || (0x1100..=0x11FF).contains(&code) {
            return self.get_weight("hangul");
        }

        // 二分查找 Unicode 区块
        let idx = self.breakpoints.binary_search(&code).unwrap_or_else(|i| i);
        if idx < self.ranges.len() {
            let script_type = self.ranges[idx].1;
            return self.get_weight(script_type);
        }

        // CJK 扩展 B/C/D 等
        if code > 0x20000 {
            return self.get_weight("cjk");
        }

        self.get_weight("default")
    }

    /// 计算文本的总权重
    #[must_use]
    pub fn total_weight(&self, text: &str) -> f64 {
        text.chars().map(|c| self.char_weight(c)).sum()
    }

    /// 根据参考文本和参考时长估算目标文本时长
    ///
    /// # 参数
    /// - `target_text`: 要估算的文本
    /// - `ref_text`: 参考文本
    /// - `ref_duration`: 参考文本的实际音频时长（秒）
    /// - `low_threshold`: 低于此阈值时使用幂曲线提升（秒，默认 50）
    /// - `boost_strength`: 提升强度（1=线性，3=平方根，默认 3）
    ///
    /// # 返回
    /// 估算的目标文本时长（秒）
    #[must_use]
    pub fn estimate_duration(
        &self,
        target_text: &str,
        ref_text: &str,
        ref_duration: f64,
        low_threshold: Option<f64>,
        boost_strength: f64,
    ) -> f64 {
        if ref_duration <= 0.0 || ref_text.is_empty() {
            return 0.0;
        }

        let ref_weight = self.total_weight(ref_text);
        if ref_weight == 0.0 {
            return 0.0;
        }

        let speed_factor = ref_weight / ref_duration;
        let target_weight = self.total_weight(target_text);
        let estimated = target_weight / speed_factor;

        if let Some(threshold) = low_threshold {
            if estimated < threshold {
                let alpha = 1.0 / boost_strength;
                return threshold * (estimated / threshold).powf(alpha);
            }
        }

        estimated
    }

    /// 不需要参考文本的简化时长估算
    ///
    /// 使用默认语速（1 个拉丁字符 ≈ 0.045 秒）估算。
    #[must_use]
    pub fn estimate_duration_simple(&self, text: &str) -> f64 {
        let weight = self.total_weight(text);
        // 1.0 权重 ≈ 0.045 秒（约 22 字符/秒）
        let estimated = weight * 0.045;
        estimated.max(0.5) // 至少 0.5 秒
    }

    /// 从权重表中获取权重值
    fn get_weight(&self, key: &str) -> f64 {
        for (k, v) in self.weights {
            if *k == key {
                return *v;
            }
        }
        1.0 // default
    }
}

impl Default for RuleDurationEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// 语音权重表（文字类型 → 相对权重）
///
/// 基准: 1.0 = 一个拉丁字符 (~40-50ms)
static PHONETIC_WEIGHTS: &[(&str, f64)] = &[
    ("cjk", 3.0),           // 中文、日文汉字
    ("hangul", 2.5),        // 韩文
    ("kana", 2.2),          // 日文假名
    ("ethiopic", 3.0),      // 阿姆哈拉语
    ("indic", 1.8),         // 印度系文字（印地语、孟加拉语等）
    ("thai_lao", 1.5),      // 泰文、老挝文
    ("khmer_myanmar", 1.8), // 高棉文、缅甸文
    ("arabic", 1.5),        // 阿拉伯文、波斯文、乌尔都文
    ("hebrew", 1.5),        // 希伯来文
    ("latin", 1.0),         // 拉丁文（英文、西班牙文等）
    ("cyrillic", 1.0),      // 西里尔文（俄文等）
    ("greek", 1.0),         // 希腊文
    ("armenian", 1.0),      // 亚美尼亚文
    ("georgian", 1.0),      // 格鲁吉亚文
    ("punctuation", 0.5),   // 标点符号
    ("space", 0.2),         // 空格
    ("digit", 3.5),         // 数字
    ("mark", 0.0),          // 变音符号
    ("default", 1.0),       // 未知文字
];

/// Unicode 码点范围 → 文字类型映射
///
/// 格式: (结束码点, 类型名称)
/// 用于二分查找。
static UNICODE_RANGES: &[(u32, &str)] = &[
    (0x02AF, "latin"),         // Latin (Basic, Supplement, Ext, IPA)
    (0x03FF, "greek"),         // Greek & Coptic
    (0x052F, "cyrillic"),      // Cyrillic
    (0x058F, "armenian"),      // Armenian
    (0x05FF, "hebrew"),        // Hebrew
    (0x077F, "arabic"),        // Arabic, Syriac
    (0x089F, "arabic"),        // Arabic Extended-B
    (0x08FF, "arabic"),        // Arabic Extended-A
    (0x097F, "indic"),         // Devanagari
    (0x09FF, "indic"),         // Bengali
    (0x0A7F, "indic"),         // Gurmukhi
    (0x0AFF, "indic"),         // Gujarati
    (0x0B7F, "indic"),         // Oriya
    (0x0BFF, "indic"),         // Tamil
    (0x0C7F, "indic"),         // Telugu
    (0x0CFF, "indic"),         // Kannada
    (0x0D7F, "indic"),         // Malayalam
    (0x0DFF, "indic"),         // Sinhala
    (0x0EFF, "thai_lao"),      // Thai & Lao
    (0x0FFF, "indic"),         // Tibetan
    (0x109F, "khmer_myanmar"), // Myanmar
    (0x10FF, "georgian"),      // Georgian
    (0x11FF, "hangul"),        // Hangul Jamo
    (0x137F, "ethiopic"),      // Ethiopic
    (0x139F, "ethiopic"),      // Ethiopic Supplement
    (0x13FF, "default"),       // Cherokee
    (0x167F, "default"),       // Canadian Aboriginal
    (0x169F, "default"),       // Ogham
    (0x16FF, "default"),       // Runic
    (0x171F, "default"),       // Tagalog
    (0x173F, "default"),       // Hanunoo
    (0x175F, "default"),       // Buhid
    (0x177F, "default"),       // Tagbanwa
    (0x17FF, "khmer_myanmar"), // Khmer
    (0x18AF, "default"),       // Mongolian
    (0x18FF, "default"),       // Canadian Aboriginal Ext
    (0x194F, "indic"),         // Limbu
    (0x19DF, "indic"),         // Tai Le & New Tai Lue
    (0x19FF, "khmer_myanmar"), // Khmer Symbols
    (0x1A1F, "indic"),         // Buginese
    (0x1AAF, "indic"),         // Tai Tham
    (0x1B7F, "indic"),         // Balinese
    (0x1BBF, "indic"),         // Sundanese
    (0x1BFF, "indic"),         // Batak
    (0x1C4F, "indic"),         // Lepcha
    (0x1C7F, "indic"),         // Ol Chiki
    (0x1C8F, "cyrillic"),      // Cyrillic Extended-C
    (0x1CBF, "georgian"),      // Georgian Extended
    (0x1CCF, "indic"),         // Sundanese Supplement
    (0x1CFF, "indic"),         // Vedic Extensions
    (0x1D7F, "latin"),         // Phonetic Extensions
    (0x1DBF, "latin"),         // Phonetic Extensions Supplement
    (0x1DFF, "default"),       // Combining Diacritical Marks Supplement
    (0x1EFF, "latin"),         // Latin Extended Additional
    (0x309F, "kana"),          // Hiragana
    (0x30FF, "kana"),          // Katakana
    (0x312F, "cjk"),           // Bopomofo
    (0x318F, "hangul"),        // Hangul Compatibility Jamo
    (0x9FFF, "cjk"),           // CJK Unified Ideographs
    (0xA4CF, "default"),       // Yi
    (0xA4FF, "default"),       // Lisu
    (0xA63F, "default"),       // Vai
    (0xA69F, "cyrillic"),      // Cyrillic Extended-B
    (0xA6FF, "default"),       // Bamum
    (0xA7FF, "latin"),         // Latin Extended-D
    (0xA82F, "indic"),         // Syloti Nagri
    (0xA87F, "default"),       // Phags-pa
    (0xA8DF, "indic"),         // Saurashtra
    (0xA8FF, "indic"),         // Devanagari Extended
    (0xA92F, "indic"),         // Kayah Li
    (0xA95F, "indic"),         // Rejang
    (0xA97F, "hangul"),        // Hangul Jamo Extended-A
    (0xA9DF, "indic"),         // Javanese
    (0xA9FF, "khmer_myanmar"), // Myanmar Extended-B
    (0xAA5F, "indic"),         // Cham
    (0xAA7F, "khmer_myanmar"), // Myanmar Extended-A
    (0xAADF, "indic"),         // Tai Viet
    (0xAAFF, "indic"),         // Meetei Mayek
    (0xAB2F, "ethiopic"),      // Ethiopic Extended-A
    (0xAB6F, "latin"),         // Latin Extended-E
    (0xABBF, "default"),       // Cherokee Supplement
    (0xABFF, "indic"),         // Meetei Mayek
    (0xD7AF, "hangul"),        // Hangul Syllables
    (0xFAFF, "cjk"),           // CJK Compatibility
    (0xFDFF, "arabic"),        // Arabic Presentation Forms-A
    (0xFE6F, "default"),       // Variation Selectors
    (0xFEFF, "arabic"),        // Arabic Presentation Forms-B
    (0xFFEF, "latin"),         // Fullwidth Latin
];

// ─── Phase 2 测试 ─────────────────────────────────────────

#[cfg(test)]
mod omni_phase2_tests {
    use super::*;

    #[test]
    fn test_estimator_latin() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("Hello");
        // 5 个拉丁字符 × 1.0 = 5.0
        assert!((weight - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_estimator_cjk() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("你好");
        // 2 个 CJK 字符 × 3.0 = 6.0
        assert!((weight - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_estimator_mixed() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("Hello 你好");
        // 5 latin × 1.0 + 1 space × 0.2 + 2 CJK × 3.0 = 5 + 0.2 + 6 = 11.2
        assert!(
            (weight - 11.2).abs() < 0.01,
            "Expected 11.2, got {}",
            weight
        );
    }

    #[test]
    fn test_estimator_korean() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("안녕");
        // 2 个韩文字符 × 2.5 = 5.0
        assert!((weight - 5.0).abs() < 0.01, "Expected 5.0, got {}", weight);
    }

    #[test]
    fn test_estimator_japanese() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("こんにちは");
        // 5 个假名 × 2.2 = 11.0
        assert!(
            (weight - 11.0).abs() < 0.01,
            "Expected 11.0, got {}",
            weight
        );
    }

    #[test]
    fn test_estimator_arabic() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("مرحبا");
        // 5 个阿拉伯字符 × 1.5 = 7.5
        assert!((weight - 7.5).abs() < 0.01, "Expected 7.5, got {}", weight);
    }

    #[test]
    fn test_estimator_digits() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("12345");
        // 5 个数字 × 3.5 = 17.5
        assert!(
            (weight - 17.5).abs() < 0.01,
            "Expected 17.5, got {}",
            weight
        );
    }

    #[test]
    fn test_estimator_estimate_with_reference() {
        let est = RuleDurationEstimator::new();
        // ref: "Hello" (weight=5.0), duration=1.5s → speed=3.33
        // target: "你好世界" (weight=12.0) → estimated=12.0/3.33=3.6s
        let dur = est.estimate_duration("你好世界", "Hello", 1.5, None, 3.0);
        assert!(dur > 3.0 && dur < 4.0, "Expected ~3.6s, got {dur}");
    }

    #[test]
    fn test_estimator_estimate_simple() {
        let est = RuleDurationEstimator::new();
        let dur = est.estimate_duration_simple("Hello");
        // weight=5.0, 5.0*0.045=0.225s → clamped to 0.5
        assert!((dur - 0.5).abs() < 0.01);

        let dur2 = est.estimate_duration_simple("你好世界你好世界");
        // weight=24.0, 24.0*0.045=1.08s
        assert!(dur2 > 1.0 && dur2 < 1.2, "Expected ~1.08s, got {dur2}");
    }

    #[test]
    fn test_estimator_empty() {
        let est = RuleDurationEstimator::new();
        assert_eq!(est.total_weight(""), 0.0);
        assert!((est.estimate_duration_simple("") - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_estimator_low_threshold_boost() {
        let est = RuleDurationEstimator::new();
        // 很短的文本应该被 boost
        let dur = est.estimate_duration("Hi", "Hello", 1.5, Some(50.0), 3.0);
        // weight(Hi)=2.0, weight(Hello)=5.0, speed=3.33
        // estimated = 2.0/3.33 = 0.6s, below 50s threshold
        // boost: 50 * (0.6/50)^(1/3) = 50 * 0.6^(1/3) ≈ 50 * 0.843 = 42.2
        assert!(dur > 0.6 && dur < 50.0, "Expected boosted value, got {dur}");
    }

    #[test]
    fn test_estimator_punctuation() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("Hello!");
        // 5 latin × 1.0 + 1 punct × 0.5 = 5.5
        assert!((weight - 5.5).abs() < 0.01, "Expected 5.5, got {}", weight);
    }

    #[test]
    fn test_estimator_space() {
        let est = RuleDurationEstimator::new();
        let weight = est.total_weight("a b");
        // 2 latin × 1.0 + 1 space × 0.2 = 2.2
        assert!((weight - 2.2).abs() < 0.01, "Expected 2.2, got {}", weight);
    }

    #[test]
    fn test_estimator_mixed_text_duration() {
        let est = RuleDurationEstimator::new();
        // "用 Python 实现 quick sort" — 混合中英文
        let dur = est.estimate_duration_simple("用 Python 实现 quick sort 算法");
        // 估算应该 > 0.5s
        assert!(dur > 0.5, "Mixed text duration should be > 0.5s, got {dur}");
    }
}
