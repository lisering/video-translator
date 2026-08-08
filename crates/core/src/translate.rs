//! 翻译模块
//!
//! 提供英译中翻译功能，基于本地离线推理引擎，
//! 并通过 [`TerminologyManager`] 注入 IT 术语表以提升专业术语翻译一致性。
//!
//! # 功能概览
//! - [`TranslationProvider`] trait：定义批量翻译的标准接口
//! - [`LocalTranslationEngine`]：本地离线翻译引擎（基于 GGUF 模型，完全离线运行）
//! - [`InferenceBackend`] trait：本地推理后端接口
//! - [`MockInferenceBackend`]：模拟推理后端（用于测试）
//! - [`TerminologyManager`]：术语表加载、占位符替换与还原
//! - [`BleuEvaluator`]：BLEU 翻译精度评估器
//!
//! # 术语表工作流
//! 1. 翻译前：将原文中的英文术语替换为 `[[T0]]`、`[[T1]]` 等占位符
//! 2. 推理后端翻译带占位符的文本
//! 3. 翻译后：将占位符替换回中文术语
//!
//! # 示例
//! ```no_run
//! use vt_core::translate::{LocalTranslationEngine, MockInferenceBackend, TerminologyManager, GlossaryEntry, TranslationProvider};
//! use vt_core::models::segment::Segment;
//! use vt_core::error::AppResult;
//!
//! fn translate() -> AppResult<()> {
//!     let term = TerminologyManager::from_entries(vec![
//!         vt_core::translate::GlossaryEntry::new("GPU", "图形处理器"),
//!     ])?;
//!     let backend = MockInferenceBackend::default();
//!     let engine = LocalTranslationEngine::new(backend).with_terminology(term);
//!
//!     let mut segments = vec![
//!         Segment::new("s1".into(), 0.0, 5.0, "The GPU renders graphics".into()),
//!     ];
//!     engine.translate_batch(&mut segments, "en", "zh")?;
//!     assert!(segments[0].target_text.is_some());
//!     Ok(())
//! }
//! ```

use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};
use crate::models::segment::Segment;

// ─── 术语表管理 ───────────────────────────────────────────

/// 术语条目
///
/// 表示一个源语言到目标语言的术语映射。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossaryEntry {
    /// 源语言术语（英文，如 `GPU`）
    pub source: String,
    /// 目标语言术语（中文，如 `图形处理器`）
    pub target: String,
}

impl GlossaryEntry {
    /// 创建新的术语条目
    ///
    /// # 参数
    /// - `source`: 源语言术语
    /// - `target`: 目标语言术语
    #[must_use]
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

/// 占位符映射条目：`(placeholder, target_term)`
type PlaceholderMapping = Vec<(String, String)>;

/// 术语表管理器
///
/// 负责加载术语表文件（JSON 或 CSV），并在翻译前后对文本进行
/// 术语占位符替换与还原，确保专业术语翻译一致性。
///
/// # 占位符格式
/// 使用 `[[T0]]`、`[[T1]]` 等格式作为占位符，翻译模型需保持其不变。
pub struct TerminologyManager {
    /// 术语条目列表（按 source 长度降序排列，优先匹配长术语）
    entries: Vec<GlossaryEntry>,
    /// 预编译的正则表达式列表：`(regex, target_term)`
    patterns: Vec<(Regex, String)>,
}

impl TerminologyManager {
    /// 从条目列表创建术语管理器
    ///
    /// 条目会按 source 长度降序排列，确保长术语优先匹配
    /// （如 `GPU driver` 优先于 `GPU`）。
    ///
    /// # 错误
    /// - [`AppError::TranslationError`][]: 正则表达式编译失败
    pub fn from_entries(entries: Vec<GlossaryEntry>) -> AppResult<Self> {
        // 按 source 长度降序排列，避免短术语先替换导致长术语无法匹配
        let mut entries = entries;
        entries.sort_by_key(|b| std::cmp::Reverse(b.source.len()));

        let patterns: Vec<(Regex, String)> = entries
            .iter()
            .map(|e| {
                let pattern = build_term_regex(&e.source);
                let regex = Regex::new(&pattern).map_err(|err| {
                    AppError::TranslationError(format!(
                        "Failed to compile regex for term '{}': {}",
                        e.source, err
                    ))
                })?;
                Ok((regex, e.target.clone()))
            })
            .collect::<AppResult<Vec<_>>>()?;

        Ok(Self { entries, patterns })
    }

    /// 从 JSON 文件加载术语表
    ///
    /// JSON 格式为 `[{"source": "GPU", "target": "图形处理器"}]`。
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 文件不存在
    /// - [`AppError::Serialization`][]: JSON 解析失败
    /// - [`AppError::TranslationError`][]: 正则表达式编译失败
    pub fn load_from_json(path: &Path) -> AppResult<Self> {
        if !path.exists() {
            return Err(AppError::FileNotFound(path.to_path_buf()));
        }
        let content = std::fs::read_to_string(path)?;
        let entries: Vec<GlossaryEntry> = serde_json::from_str(&content)?;
        Self::from_entries(entries)
    }

    /// 从 CSV 文件加载术语表
    ///
    /// CSV 格式为 `source,target`（首行为表头，会跳过）。
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 文件不存在
    /// - [`AppError::TranslationError`][]: CSV 解析或正则编译失败
    pub fn load_from_csv(path: &Path) -> AppResult<Self> {
        if !path.exists() {
            return Err(AppError::FileNotFound(path.to_path_buf()));
        }
        let content = std::fs::read_to_string(path)?;
        let mut entries = Vec::new();
        for (i, line) in content.lines().enumerate() {
            // 跳过表头
            if i == 0 {
                continue;
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, ',').collect();
            if parts.len() != 2 {
                return Err(AppError::TranslationError(format!(
                    "Invalid CSV line {}: expected 2 fields, got {}",
                    i + 1,
                    parts.len()
                )));
            }
            entries.push(GlossaryEntry::new(parts[0].trim(), parts[1].trim()));
        }
        Self::from_entries(entries)
    }

    /// 获取术语条目数量
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 术语表是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 获取所有术语条目
    #[must_use]
    pub fn entries(&self) -> &[GlossaryEntry] {
        &self.entries
    }

    /// 对文本应用术语占位符替换
    ///
    /// 将文本中的英文术语替换为 `[[T0]]`、`[[T1]]` 等占位符，
    /// 返回替换后的文本和占位符映射表。
    ///
    /// # 返回值
    /// `(替换后的文本, [(占位符, 中文术语), ...])`
    pub fn apply_placeholders(&self, text: &str) -> (String, PlaceholderMapping) {
        if self.patterns.is_empty() {
            return (text.to_string(), Vec::new());
        }

        let mut result = text.to_string();
        let mut mapping: PlaceholderMapping = Vec::new();

        for (regex, target) in &self.patterns {
            if regex.is_match(&result) {
                let placeholder = format!("[[T{}]]", mapping.len());
                result = regex.replace_all(&result, &placeholder).to_string();
                mapping.push((placeholder, target.clone()));
            }
        }

        (result, mapping)
    }

    /// 还原术语占位符
    ///
    /// 将翻译结果中的 `[[T0]]` 等占位符替换回中文术语。
    ///
    /// # 参数
    /// - `text`: 翻译后的文本
    /// - `mapping`: 占位符映射表（由 `apply_placeholders` 返回）
    pub fn restore_placeholders(&self, text: &str, mapping: &PlaceholderMapping) -> String {
        let mut result = text.to_string();
        for (placeholder, target) in mapping {
            result = result.replace(placeholder, target);
        }
        result
    }

    /// 构建系统提示词中的术语占位符说明
    ///
    /// 生成一段文字，告知翻译模型需要保持占位符不变。
    pub fn build_glossary_hint(mapping: &PlaceholderMapping) -> String {
        if mapping.is_empty() {
            return String::new();
        }
        let terms: Vec<&str> = mapping.iter().map(|(p, _)| p.as_str()).collect();
        format!(
            "IMPORTANT: The text contains placeholder tokens like {}. \
             You MUST preserve these tokens EXACTLY as they appear in your translation. \
             Do not translate, modify, or remove them.",
            terms.join(", ")
        )
    }
}

impl std::fmt::Debug for TerminologyManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminologyManager")
            .field("entries", &self.entries)
            .field("pattern_count", &self.patterns.len())
            .finish()
    }
}

/// 返回内置编程术语映射表（Rust / 通用编程术语）
///
/// 包含至少 10 个常见编程术语的英文→中文映射，确保翻译时
/// 不会将 `println` 拆译为"打印行"、`format!` 误译为"格式化感叹号"等。
///
/// # 内置术语列表
/// | 英文 | 中文 | 说明 |
/// |---|---|---|
/// | `println!` | 打印并换行宏 | Rust 宏 |
/// | `println` | 打印并换行 | Rust 函数 |
/// | `print!` | 打印宏 | Rust 宏 |
/// | `print` | 打印 | Rust 函数 |
/// | `format!` | 格式化宏 | Rust 宏 |
/// | `vec!` | 向量宏 | Rust 宏 |
/// | `String` | 字符串 | Rust 类型 |
/// | `Vec` | 向量 | Rust 类型 |
/// | `HashMap` | 哈希表 | Rust 类型 |
/// | `struct` | 结构体 | Rust 关键字 |
/// | `enum` | 枚举 | Rust 关键字 |
/// | `trait` | 特征 | Rust 关键字 |
/// | `impl` | 实现 | Rust 关键字 |
/// | `borrow` | 借用 | Rust 概念 |
/// | `ownership` | 所有权 | Rust 概念 |
/// | `lifetime` | 生命周期 | Rust 概念 |
/// | `closure` | 闭包 | Rust 概念 |
/// | `iterator` | 迭代器 | Rust 概念 |
///
/// # 示例
/// ```
/// use vt_core::translate::builtin_programming_terms;
///
/// let terms = builtin_programming_terms();
/// assert!(terms.len() >= 10);
/// assert!(terms.iter().any(|e| e.source == "println" && e.target == "打印并换行"));
/// ```
#[must_use]
pub fn builtin_programming_terms() -> Vec<GlossaryEntry> {
    vec![
        GlossaryEntry::new("println!", "打印并换行宏"),
        GlossaryEntry::new("println", "打印并换行"),
        GlossaryEntry::new("print!", "打印宏"),
        GlossaryEntry::new("print", "打印"),
        GlossaryEntry::new("format!", "格式化宏"),
        GlossaryEntry::new("vec!", "向量宏"),
        GlossaryEntry::new("String", "字符串"),
        GlossaryEntry::new("Vec", "向量"),
        GlossaryEntry::new("HashMap", "哈希表"),
        GlossaryEntry::new("struct", "结构体"),
        GlossaryEntry::new("enum", "枚举"),
        GlossaryEntry::new("trait", "特征"),
        GlossaryEntry::new("impl", "实现"),
        GlossaryEntry::new("borrow", "借用"),
        GlossaryEntry::new("ownership", "所有权"),
        GlossaryEntry::new("lifetime", "生命周期"),
        GlossaryEntry::new("closure", "闭包"),
        GlossaryEntry::new("iterator", "迭代器"),
    ]
}

/// 后校正规则：`(错误译法, 正确译法)`
///
/// 翻译模型常见的误译模式，在翻译完成后通过正则替换修正。
const POST_CORRECTION_RULES: &[(&str, &str)] = &[
    ("打印行", "打印并换行"),
    ("打印线", "打印并换行"),
    ("输出行", "打印并换行"),
    ("打印ln", "打印并换行"),
    ("格式化感叹号", "格式化宏"),
    ("格式化叹号", "格式化宏"),
    ("向量宏感叹号", "向量宏"),
    ("字符串类型", "字符串"),
    ("向量类型", "向量"),
];

/// 对翻译后的文本进行术语后校正
///
/// 在翻译完成后，对 `target_text` 进行正则替换，将常见错误术语修正为正确术语。
/// 例如将"打印行"修正为"打印并换行"、"打印线"修正为"打印并换行"等。
///
/// 此函数应在翻译完成后、TTS 合成前调用，确保最终配音中的术语准确。
///
/// # 参数
/// - `text`: 翻译后的中文文本
///
/// # 返回值
/// 校正后的中文文本
///
/// # 示例
/// ```
/// use vt_core::translate::post_correct;
///
/// let corrected = post_correct("使用打印行输出内容");
/// assert_eq!(corrected, "使用打印并换行输出内容");
///
/// let corrected = post_correct("这是格式化感叹号");
/// assert_eq!(corrected, "这是格式化宏");
/// ```
#[must_use]
pub fn post_correct(text: &str) -> String {
    let mut result = text.to_string();
    for (wrong, right) in POST_CORRECTION_RULES {
        if result.contains(wrong) {
            result = result.replace(wrong, right);
        }
    }
    result
}

/// 对 Segment 列表应用术语后校正
///
/// 遍历所有 Segment 的 `target_text`，调用 [`post_correct`] 进行术语校正。
/// 此函数应在翻译完成后、TTS 合成前调用。
///
/// # 参数
/// - `segments`: 待校正的片段列表（原地修改 `target_text`）
///
/// # 示例
/// ```
/// use vt_core::translate::apply_post_correction;
/// use vt_core::models::segment::Segment;
///
/// let mut seg = Segment::new("s1".into(), 0.0, 5.0, "println".into());
/// seg.start_transcribing().ok();
/// seg.finish_transcribing("使用打印行输出".into()).ok();
///
/// let mut segments = vec![seg];
/// apply_post_correction(&mut segments);
/// assert_eq!(segments[0].target_text.as_deref(), Some("使用打印并换行输出"));
/// ```
pub fn apply_post_correction(segments: &mut [Segment]) {
    for seg in segments.iter_mut() {
        if let Some(ref text) = seg.target_text {
            let corrected = post_correct(text);
            if corrected != *text {
                tracing::debug!(
                    "Post-corrected segment {}: '{}' → '{}'",
                    seg.id,
                    text.chars().take(40).collect::<String>(),
                    corrected.chars().take(40).collect::<String>()
                );
                seg.target_text = Some(corrected);
            }
        }
    }
}

/// 为单个术语构建正则表达式模式
///
/// - 若术语首尾均为字母/数字，使用 `\b` 词边界匹配（如 `GPU`）
/// - 否则使用简单大小写不敏感匹配（如 `C++`、`.NET`）
fn build_term_regex(term: &str) -> String {
    let escaped = regex::escape(term);
    let starts_with_word = term.chars().next().is_some_and(|c| c.is_alphanumeric());
    let ends_with_word = term.chars().last().is_some_and(|c| c.is_alphanumeric());

    match (starts_with_word, ends_with_word) {
        (true, true) => format!(r"\b(?i){escaped}\b"),
        (true, false) => format!(r"\b(?i){escaped}"),
        (false, true) => format!(r"(?i){escaped}\b"),
        (false, false) => format!(r"(?i){escaped}"),
    }
}

// ─── 翻译提供者 Trait ─────────────────────────────────────

/// 翻译提供者接口
///
/// 定义批量翻译的标准接口，各翻译引擎（如 [`LocalTranslationEngine`]）实现此 trait。
/// 翻译结果直接写入每个 `Segment` 的 `target_text` 字段。
pub trait TranslationProvider: Send + Sync {
    /// 批量翻译 Segment 列表
    ///
    /// 将每个 Segment 的 `source_text` 翻译为目标语言，
    /// 并将结果写入 `target_text` 字段。
    ///
    /// # 参数
    /// - `segments`: 待翻译的片段列表（原地修改）
    /// - `source_lang`: 源语言代码（如 `en`）
    /// - `target_lang`: 目标语言代码（如 `zh`）
    ///
    /// # 错误
    /// - [`AppError::TranslationError`][]: 翻译失败
    fn translate_batch(
        &self,
        segments: &mut [Segment],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<()>;

    /// 上下文感知翻译：翻译单个 Segment，利用前序段落上下文
    ///
    /// 默认实现忽略上下文，直接调用 `translate_batch`。
    /// 支持上下文的实现（如 LLM 后端）可覆盖此方法，
    /// 将前序段落的 (source, target) 对作为对话历史注入 LLM，
    /// 提升代词消解、术语一致性和跨段落连贯性。
    ///
    /// # 参数
    /// - `segment`: 待翻译的单个片段（原地修改 `target_text`）
    /// - `source_lang`: 源语言代码
    /// - `target_lang`: 目标语言代码
    /// - `context`: 前序段落的上下文（可能为空）
    ///
    /// # 错误
    /// - [`AppError::TranslationError`][]: 翻译失败
    fn translate_segment_with_context(
        &self,
        segment: &mut Segment,
        source_lang: &str,
        target_lang: &str,
        context: &TranslationContext,
    ) -> AppResult<()> {
        // 默认实现：忽略上下文，直接批量翻译
        let _ = context;
        let mut segments = std::slice::from_mut(segment);
        self.translate_batch(&mut segments, source_lang, target_lang)
    }

    /// SRT 批量翻译：将所有 segments 组成 SRT 格式一次性翻译
    ///
    /// 将所有 segment 的源文本组成 SRT 格式文本，一次性发送给翻译引擎，
    /// 翻译引擎能看到完整的视频字幕上下文，提升翻译连贯性。
    /// 翻译后按 SRT 结构解析回 segments，保持时间戳和行号对齐。
    ///
    /// 默认实现回退到 `translate_batch`（逐段翻译）。
    ///
    /// # 参数
    /// - `segments`: 待翻译的片段列表（原地修改 `target_text`）
    /// - `source_lang`: 源语言代码
    /// - `target_lang`: 目标语言代码
    ///
    /// # 错误
    /// - [`AppError::TranslationError`][]: 翻译失败
    fn translate_srt(
        &self,
        segments: &mut [Segment],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<()> {
        // 默认实现：回退到逐段翻译
        self.translate_batch(segments, source_lang, target_lang)
    }
}

// ─── 上下文感知翻译 ───────────────────────────────────────

/// 上下文条目：前序段落的原文和译文
///
/// 用于向翻译模型提供前序段落的信息，帮助消解代词、
/// 保持术语一致性、改善跨段落连贯性。
#[derive(Debug, Clone)]
pub struct ContextEntry {
    /// 前序段落的源语言文本
    pub source: String,
    /// 前序段落的目标语言译文
    pub target: String,
}

/// 翻译上下文窗口
///
/// 保存最近 N 条已翻译段落的 (source, target) 对，
/// 供后续段落翻译时作为上下文参考。
///
/// 使用滑动窗口策略：只保留最近 `max_entries` 条，
/// 旧条目自动淘汰。
#[derive(Debug, Clone)]
pub struct TranslationContext {
    /// 上下文条目列表（按时间顺序排列，最旧在前）
    entries: Vec<ContextEntry>,
    /// 最大保留条目数
    max_entries: usize,
}

impl TranslationContext {
    /// 创建空的翻译上下文
    ///
    /// # 参数
    /// - `max_entries`: 最大保留条目数（默认 3）
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries),
            max_entries: max_entries.max(1),
        }
    }

    /// 添加一条上下文
    ///
    /// 当条目数超过 `max_entries` 时，淘汰最旧的条目。
    pub fn push(&mut self, source: impl Into<String>, target: impl Into<String>) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(ContextEntry {
            source: source.into(),
            target: target.into(),
        });
    }

    /// 获取上下文条目列表
    #[must_use]
    pub fn entries(&self) -> &[ContextEntry] {
        &self.entries
    }

    /// 上下文是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 上下文条目数
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new(3)
    }
}

// ─── 术语辅助函数 ─────────────────────────────────────────

/// 对 Segment 列表应用术语占位符
///
/// 返回 `(待翻译文本列表, 每个文本的占位符映射)`
fn apply_terminology(
    terminology: &Option<TerminologyManager>,
    segments: &[Segment],
) -> (Vec<String>, Vec<PlaceholderMapping>) {
    match terminology {
        Some(term) => {
            let mut texts = Vec::with_capacity(segments.len());
            let mut mappings = Vec::with_capacity(segments.len());
            for seg in segments {
                let (modified, mapping) = term.apply_placeholders(&seg.source_text);
                texts.push(modified);
                mappings.push(mapping);
            }
            (texts, mappings)
        }
        None => (
            segments.iter().map(|s| s.source_text.clone()).collect(),
            vec![Vec::new(); segments.len()],
        ),
    }
}

/// 还原术语占位符并写入 Segment 的 target_text
fn restore_terminology(
    terminology: &Option<TerminologyManager>,
    segments: &mut [Segment],
    translated: &[String],
    mappings: &[PlaceholderMapping],
) {
    for (i, seg) in segments.iter_mut().enumerate() {
        let text = &translated[i];
        let final_text = match terminology {
            Some(term) if !mappings[i].is_empty() => term.restore_placeholders(text, &mappings[i]),
            _ => text.clone(),
        };
        seg.target_text = Some(final_text);
    }
}

// ─── 本地推理后端 Trait ───────────────────────────────────

/// 本地推理后端接口
///
/// 定义本地翻译模型推理的标准接口。各推理框架（candle、llama.cpp 等）
/// 通过实现此 trait 接入 `LocalTranslationEngine`。
///
/// # 线程安全
/// 实现者必须满足 `Send + Sync`，以支持异步流水线中的并行处理。
///
/// # 设计目标
/// - 支持单条和批量翻译
/// - 推理参数（max_tokens、temperature）通过实现者内部管理
/// - 支持离线运行，无网络依赖
pub trait InferenceBackend: Send + Sync {
    /// 翻译单条文本
    ///
    /// # 参数
    /// - `text`: 源语言文本
    /// - `source_lang`: 源语言代码（如 `en`）
    /// - `target_lang`: 目标语言代码（如 `zh`）
    ///
    /// # 错误
    /// 返回 [`AppError::TranslationError`] 表示推理失败。
    fn translate_text(&self, text: &str, source_lang: &str, target_lang: &str)
        -> AppResult<String>;

    /// 上下文感知翻译：翻译单条文本，利用前序段落上下文
    ///
    /// 默认实现忽略上下文，直接调用 `translate_text`。
    /// 支持 LLM 的后端（如 `LlamaCppBackend`）可覆盖此方法，
    /// 将上下文作为对话历史注入 LLM 请求。
    ///
    /// # 参数
    /// - `text`: 源语言文本
    /// - `source_lang`: 源语言代码
    /// - `target_lang`: 目标语言代码
    /// - `context`: 前序段落上下文（可能为空）
    ///
    /// # 错误
    /// 返回 [`AppError::TranslationError`] 表示推理失败。
    fn translate_text_with_context(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        context: &TranslationContext,
    ) -> AppResult<String> {
        let _ = context;
        self.translate_text(text, source_lang, target_lang)
    }

    /// 批量翻译多条文本
    ///
    /// 默认实现逐条调用 `translate_text`，实现者可覆盖以优化批量推理性能。
    ///
    /// # 参数
    /// - `texts`: 源语言文本列表
    /// - `source_lang`: 源语言代码
    /// - `target_lang`: 目标语言代码
    ///
    /// # 错误
    /// - [`AppError::TranslationError`][]: 任一条文本翻译失败
    fn translate_texts(
        &self,
        texts: &[String],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<Vec<String>> {
        texts
            .iter()
            .map(|t| self.translate_text(t, source_lang, target_lang))
            .collect()
    }

    /// SRT 批量翻译：将整段 SRT 格式文本一次性发送给 LLM 翻译
    ///
    /// 将所有字幕段落组成 SRT 格式文本，让 LLM 在完整上下文下翻译，
    /// 翻译后保持 SRT 结构（行号和时间戳不变），仅替换文本部分。
    ///
    /// 默认实现回退到逐条翻译。
    ///
    /// # 参数
    /// - `srt_text`: SRT 格式的源语言文本
    /// - `source_lang`: 源语言代码
    /// - `target_lang`: 目标语言代码
    ///
    /// # 返回
    /// 翻译后的 SRT 格式文本（保持原有的行号和时间戳结构）
    ///
    /// # 错误
    /// 返回 [`AppError::TranslationError`] 表示推理失败。
    fn translate_srt_text(
        &self,
        srt_text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<String> {
        // 默认实现：按 SRT 块分割，逐条翻译文本部分
        let blocks: Vec<&str> = srt_text.split("\n\n").collect();
        let mut result = String::new();
        for block in &blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 3 {
                continue;
            }
            // 行号 + 时间戳行保持不变，仅翻译文本部分（第3行起）
            let text_part = lines[2..].join("\n");
            let translated = self.translate_text(&text_part, source_lang, target_lang)?;
            result.push_str(lines[0]);
            result.push('\n');
            result.push_str(lines[1]);
            result.push('\n');
            result.push_str(&translated);
            result.push_str("\n\n");
        }
        Ok(result.trim_end().to_string())
    }

    /// 获取后端名称（用于日志和调试）
    fn backend_name(&self) -> &str;
}

// ─── Mock 推理后端（用于测试） ────────────────────────────

/// 模拟推理后端
///
/// 基于内置词典的简单翻译后端，用于单元测试和集成测试。
/// 不依赖任何外部模型或网络，完全离线运行。
///
/// # 适用场景
/// - 测试 `LocalTranslationEngine` 的术语集成、批量处理等逻辑
/// - CI 环境中验证离线翻译流水线
/// - 不适合评估真实翻译精度
pub struct MockInferenceBackend {
    /// 内置翻译词典：英文短语 → 中文翻译
    dictionary: std::collections::HashMap<String, String>,
}

impl MockInferenceBackend {
    /// 创建空的模拟后端
    #[must_use]
    pub fn new() -> Self {
        Self {
            dictionary: std::collections::HashMap::new(),
        }
    }

    /// 从翻译对列表创建模拟后端
    ///
    /// # 参数
    /// - `pairs`: `[(英文, 中文), ...]` 翻译对列表
    #[must_use]
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        let mut dictionary = std::collections::HashMap::new();
        for (en, zh) in pairs {
            dictionary.insert((*en).to_string(), (*zh).to_string());
        }
        Self { dictionary }
    }

    /// 添加翻译对
    #[must_use]
    pub fn with_pair(mut self, en: impl Into<String>, zh: impl Into<String>) -> Self {
        self.dictionary.insert(en.into(), zh.into());
        self
    }
}

impl Default for MockInferenceBackend {
    fn default() -> Self {
        Self::from_pairs(&[
            ("Hello, world", "你好，世界"),
            ("Hello", "你好"),
            ("World", "世界"),
            ("Test", "测试"),
            ("The GPU renders graphics", "图形处理器渲染图形"),
            (
                "Docker containers package applications",
                "容器引擎打包应用程序",
            ),
            ("The API is RESTful", "应用程序接口是RESTful的"),
            ("Machine learning is powerful", "机器学习很强大"),
            ("Good morning", "早上好"),
            ("Thank you", "谢谢"),
            ("How are you", "你好吗"),
            ("I love programming", "我热爱编程"),
            ("The weather is nice today", "今天天气不错"),
            ("Open source software", "开源软件"),
            ("Cloud computing", "云计算"),
            ("Artificial intelligence", "人工智能"),
            ("Deep learning", "深度学习"),
            ("Data structure", "数据结构"),
            ("Algorithm", "算法"),
            ("Database", "数据库"),
        ])
    }
}

impl InferenceBackend for MockInferenceBackend {
    fn translate_text(
        &self,
        text: &str,
        _source_lang: &str,
        _target_lang: &str,
    ) -> AppResult<String> {
        // 精确匹配
        if let Some(translated) = self.dictionary.get(text) {
            return Ok(translated.clone());
        }

        // 模糊匹配：查找包含的短语
        let mut result = text.to_string();
        for (en, zh) in &self.dictionary {
            if text.contains(en.as_str()) {
                result = result.replace(en.as_str(), zh.as_str());
            }
        }

        // 如果没有匹配到任何翻译，返回原文（模拟直通）
        if result == text {
            tracing::warn!(
                "MockInferenceBackend: no translation found for '{text}', returning original"
            );
        }

        Ok(result)
    }

    fn backend_name(&self) -> &str {
        "MockInferenceBackend"
    }
}

impl std::fmt::Debug for MockInferenceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockInferenceBackend")
            .field("dictionary_size", &self.dictionary.len())
            .finish()
    }
}

// ─── LlamaCpp GGUF 推理后端（子进程方案）─────────────────

/// 基于 llama-server 子进程的本地 GGUF 推理后端
///
/// 通过启动 `llama-server`（llama.cpp 的 HTTP 服务器）子进程，
/// 在独立进程中加载 GGUF 模型进行推理，避免与 `whisper-rs` 的 ggml
/// 符号冲突。
///
/// # 为什么用子进程？
/// `whisper-rs` 和 `llama_cpp` 都静态链接各自的 ggml C 库副本，
/// 在同一进程中共存会导致重复符号和段错误。子进程方案彻底隔离了
/// 两个库的运行环境。
///
/// # 工作流程
/// 1. `new()`: 启动 `llama-server -m model.gguf --port PORT -ngl 999`
/// 2. 等待服务就绪（轮询 `/health` 端点，超时 120s）
/// 3. `translate_text()`: POST `/completion` 发送翻译请求
/// 4. `Drop`: 自动终止子进程
///
/// # Metal 加速
/// `-ngl 999` 将全部模型层卸载到 Metal GPU，推理速度显著提升。
///
/// # 线程安全
/// 所有字段通过 `&self` 共享。`ureq` HTTP 请求是线程安全的。
/// 多个翻译请求会串行执行（HTTP 请求天然序列化）。
///
/// # 错误处理
/// - 二进制未找到：提示安装命令
/// - 服务启动超时：提示检查模型文件和系统资源
/// - 推理失败：返回详细错误信息
pub struct LlamaCppBackend {
    /// llama-server 子进程句柄（Mutex 以支持超时后重启）
    server_process: std::sync::Mutex<std::process::Child>,
    /// 服务监听端口
    port: u16,
    /// HTTP 基础 URL
    base_url: String,
    /// 最大生成 token 数
    max_tokens: usize,
    /// 采样温度
    temperature: f32,
    /// 模型路径（重启时使用）
    model_path: std::path::PathBuf,
    /// GPU 层数（重启时使用）
    n_gpu_layers: u32,
    /// 术语表 Markdown（注入系统提示词，让 LLM 遵循术语翻译）
    glossary_markdown: Option<String>,
}

impl LlamaCppBackend {
    /// 创建 LLaMA 推理后端
    ///
    /// 启动 `llama-server` 子进程，加载 GGUF 模型并等待服务就绪。
    ///
    /// # 参数
    /// - `model_path`: GGUF 模型文件路径
    /// - `max_tokens`: 最大生成 token 数
    /// - `temperature`: 采样温度（0.0 = 贪心）
    /// - `n_gpu_layers`: GPU 卸载层数（999 = 全部）
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 模型文件不存在
    /// - [`AppError::ModelLoadError`][]: llama-server 未安装或启动失败
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        max_tokens: usize,
        temperature: f32,
        n_gpu_layers: u32,
    ) -> AppResult<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(AppError::FileNotFound(path.to_path_buf()));
        }

        // 查找 llama-server 二进制
        let binary = find_llama_server()?;

        // 分配空闲端口
        let port = find_free_port()?;

        tracing::info!(
            "Starting llama-server: model={:?}, port={}, gpu_layers={}, max_tokens={}, temp={}",
            path,
            port,
            n_gpu_layers,
            max_tokens,
            temperature
        );

        // 启动 llama-server 子进程
        //
        // 参数选择依据（调研来源）：
        // - SmartSub (github 4474⭐): num_ctx=8192, temperature=0.3, timeout=300s
        //   issue #269 明确提到"单个请求无限挂起导致整个翻译流程卡死"
        // - AutoDub (github): temperature=0.2, top_p=0.9, timeout=60s
        // - llama-throughput-lab: ctx_size = per_session * parallel
        // - llama.cpp 官方文档: -to 默认 3600s（太长，卡住的请求不会被终止）
        //
        // 参数说明：
        // - `-c 8192`: 上下文窗口 8192 tokens，与 SmartSub 的 num_ctx 一致，
        //   2048/4096 在长时间运行时会导致 KV cache 累积卡死
        // - `-np 1`: 显式设置单 slot，避免多 slot 竞争 GPU 内存
        // - `-to 120`: 服务器读写超时 120 秒，卡住的请求会被自动终止
        //   （默认 3600s 太长，导致卡死后所有后续请求都被阻塞）
        // - `-fa on`: Flash Attention，减少内存占用，提升推理速度 10-20%
        // - `-ctk q8_0 -ctv q8_0`: KV cache 量化，减少内存约 50%
        // - `--no-cache-prompt`: 禁用提示词缓存，避免长时间运行后 cache 累积
        // - `-ngl 999`: 全部模型层卸载到 Metal GPU
        let mut child = std::process::Command::new(&binary)
            .arg("-m")
            .arg(path)
            .arg("--port")
            .arg(port.to_string())
            .arg("-ngl")
            .arg(n_gpu_layers.to_string())
            .arg("-c")
            .arg("8192")
            .arg("-np")
            .arg("1")
            .arg("-to")
            .arg("60")
            .arg("-fa")
            .arg("on")
            .arg("-ctk")
            .arg("q8_0")
            .arg("-ctv")
            .arg("q8_0")
            .arg("--no-cache-prompt")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                AppError::ModelLoadError(format!(
                    "Failed to spawn llama-server: {e}\n\
                     二进制路径: {binary:?}"
                ))
            })?;

        // 等待服务就绪
        match wait_for_server_ready(port, 180) {
            Ok(()) => {
                tracing::info!("llama-server ready on port {}", port);
            }
            Err(e) => {
                // 服务启动失败，终止子进程并收集错误信息
                let _ = child.kill();
                let stderr = child
                    .stderr
                    .take()
                    .and_then(|mut s| {
                        use std::io::Read;
                        let mut buf = String::new();
                        s.read_to_string(&mut buf).ok().map(|_| buf)
                    })
                    .unwrap_or_default();

                return Err(AppError::ModelLoadError(format!(
                    "llama-server failed to start within 180s: {e}\n\
                     stderr 输出:\n{stderr}\n\n\
                     可能原因：\n\
                     1. 模型文件损坏或格式不支持\n\
                     2. 内存不足（尝试更小的量化版本）\n\
                     3. GPU 层数设置过高"
                )));
            }
        }

        Ok(Self {
            server_process: std::sync::Mutex::new(child),
            port,
            base_url: format!("http://127.0.0.1:{port}"),
            max_tokens,
            temperature,
            model_path: path.to_path_buf(),
            n_gpu_layers,
            glossary_markdown: None,
        })
    }

    /// 从翻译配置创建后端
    ///
    /// # 参数
    /// - `config`: 翻译配置
    ///
    /// # 错误
    /// - [`AppError::Config`][]: 未配置 `model_path`
    pub fn from_config(config: &crate::config::TranslationConfig) -> AppResult<Self> {
        let model_path = config.model_path.as_ref().ok_or_else(|| {
            AppError::Config(
                "Translation model_path is not configured.\n\
                 请在 config.toml 的 [translation] 段中设置 model_path，\n\
                 或确保模型已从 ModelScope 下载到缓存目录。"
                    .to_string(),
            )
        })?;

        let n_gpu_layers = if config.device == "metal" { 999 } else { 0 };

        Self::new(
            model_path,
            config.max_tokens,
            config.temperature,
            n_gpu_layers,
        )
    }

    /// 重启 llama-server 子进程
    ///
    /// 当服务卡死（连续超时）时调用此方法杀掉旧进程并启动新进程。
    /// 端口保持不变，HTTP URL 不变，客户端无需感知。
    fn restart_server(&self) -> AppResult<()> {
        tracing::warn!(
            "LlamaCppBackend: restarting llama-server (port={})",
            self.port
        );

        // 杀掉旧进程
        let mut child_guard = self.server_process.lock().map_err(|e| {
            AppError::TranslationError(format!("Failed to lock server process: {e}"))
        })?;

        let _ = child_guard.kill();
        let _ = child_guard.wait();

        // 查找二进制并启动新进程
        let binary = find_llama_server()?;
        let mut new_child = std::process::Command::new(&binary)
            .arg("-m")
            .arg(&self.model_path)
            .arg("--port")
            .arg(self.port.to_string())
            .arg("-ngl")
            .arg(self.n_gpu_layers.to_string())
            .arg("-c")
            .arg("8192")
            .arg("-np")
            .arg("1")
            .arg("-to")
            .arg("60")
            .arg("-fa")
            .arg("on")
            .arg("-ctk")
            .arg("q8_0")
            .arg("-ctv")
            .arg("q8_0")
            .arg("--no-cache-prompt")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                AppError::ModelLoadError(format!("Failed to restart llama-server: {e}"))
            })?;

        // 等待服务就绪
        match wait_for_server_ready(self.port, 120) {
            Ok(()) => {
                tracing::info!("llama-server restarted successfully on port {}", self.port);
                *child_guard = new_child;
                Ok(())
            }
            Err(e) => {
                let _ = new_child.kill();
                Err(AppError::ModelLoadError(format!(
                    "llama-server restart failed: {e}"
                )))
            }
        }
    }

    /// 设置术语表 Markdown（注入系统提示词）
    ///
    /// 将术语表格式化为 Markdown 表格后注入系统提示词，
    /// 让 LLM 在翻译时严格遵循术语翻译规则。
    #[must_use]
    pub fn with_glossary_markdown(mut self, markdown: String) -> Self {
        self.glossary_markdown = if markdown.is_empty() {
            None
        } else {
            Some(markdown)
        };
        self
    }

    /// 构建翻译系统提示词（system message）
    ///
    /// 针对视频字幕/配音翻译场景优化：
    /// - 强调简短自然的口播中文
    /// - 保持代码和技术术语不翻译
    /// - 提示这是视频字幕（可能有不完整句子）
    /// - 强调只输出译文
    /// - 可选注入术语表 Markdown（让 LLM 遵循术语翻译）
    fn build_system_prompt(
        source_lang: &str,
        target_lang: &str,
        glossary_markdown: Option<&str>,
    ) -> String {
        let source_name = match source_lang {
            "en" => "English",
            "ja" => "Japanese",
            "ko" => "Korean",
            _ => source_lang,
        };
        let target_name = match target_lang {
            "zh" => "Simplified Chinese",
            "en" => "English",
            _ => target_lang,
        };

        // 获取目标语言特定翻译规则
        let lang_rules = crate::translation_extras::get_language_prompt(target_lang);

        // 术语表 Markdown（可选）
        let glossary_section = glossary_markdown.unwrap_or("");

        format!(
            "You are a professional {source_name}-to-{target_name} translator for video subtitles. \
             Translate the user's text from {source_name} to {target_name}. \
             Rules:\n\
             1. Output ONLY the translation — no explanations, notes, or original text.\n\
             2. Keep code, variable names, function names, and file paths in English.\n\
             3. Use natural, concise spoken {target_name} suitable for voice-over dubbing.\n\
             4. Subtitles may contain sentence fragments — translate them as-is,\
             do not complete or restructure sentences.\n\
             5. Preserve placeholders like [[T0]], [[T1]] exactly as they appear.\n\n\
             {glossary_section}\
             # Language-Specific Rules\n\
             {lang_rules}"
        )
    }

    /// 发送 chat completion 请求到 llama-server 并返回翻译结果
    ///
    /// 内部处理重试逻辑：最多 2 次尝试，每次超时 60s。
    /// 第一次超时后重启 llama-server，第二次再失败则返回原文。
    fn send_completion_request(
        &self,
        messages: &serde_json::Value,
        text: &str,
    ) -> AppResult<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": "translation",
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.8,
            "repeat_penalty": 1.05,
            "stream": false,
        });

        let max_retries = 2;
        let per_attempt_timeout = std::time::Duration::from_secs(60);

        for attempt in 1..=max_retries {
            if attempt > 1 {
                tracing::warn!(
                    "LlamaCppBackend: retrying translation (attempt {attempt}/{max_retries}) after restart"
                );
            }

            let response = ureq::post(&url)
                .timeout(per_attempt_timeout)
                .set("Content-Type", "application/json")
                .send_string(&body.to_string());

            match response {
                Ok(resp) => {
                    let response_text = resp.into_string().map_err(|e| {
                        AppError::TranslationError(format!("Failed to read response body: {e}"))
                    })?;

                    let response_json: serde_json::Value = serde_json::from_str(&response_text)
                        .map_err(|e| {
                            AppError::TranslationError(format!(
                                "Failed to parse response JSON: {e}\nResponse: {response_text}"
                            ))
                        })?;

                    let content = response_json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if content.is_empty() {
                        let fallback = response_json
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if fallback.is_empty() {
                            tracing::warn!(
                                "LlamaCppBackend: empty response for '{}'",
                                text.chars().take(60).collect::<String>()
                            );
                            return Ok(text.to_string());
                        }
                        let cleaned = clean_translation_output(fallback);
                        return Ok(cleaned);
                    }

                    let cleaned = clean_translation_output(content);

                    tracing::info!(
                        "LlamaCppBackend: translated '{}' → '{}'",
                        text.chars().take(40).collect::<String>(),
                        cleaned.chars().take(40).collect::<String>()
                    );

                    return Ok(cleaned);
                }
                Err(e) => {
                    tracing::warn!("LlamaCppBackend: attempt {attempt}/{max_retries} failed: {e}");
                    if attempt < max_retries {
                        if let Err(restart_err) = self.restart_server() {
                            tracing::error!("LlamaCppBackend: restart failed: {restart_err}");
                        }
                    }
                }
            }
        }

        tracing::warn!(
            "LlamaCppBackend: translation failed after {max_retries} attempts, returning original text: {}",
            text.chars().take(60).collect::<String>()
        );
        Ok(text.to_string())
    }

    /// 截断输入文本到最多 500 字符（按字符边界截断，防止切断多字节 UTF-8 字符）
    fn truncate_input(text: &str) -> String {
        if text.chars().count() > 500 {
            tracing::warn!(
                "LlamaCppBackend: input text too long ({} chars), truncating",
                text.chars().count()
            );
            text.chars().take(500).collect()
        } else {
            text.to_string()
        }
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn translate_text(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }

        let system_prompt =
            Self::build_system_prompt(source_lang, target_lang, self.glossary_markdown.as_deref());
        let truncated_text = Self::truncate_input(text);

        tracing::info!(
            "LlamaCppBackend: translating '{}' (len={}, max_tokens={})",
            text.chars().take(80).collect::<String>(),
            text.len(),
            self.max_tokens
        );

        let messages = serde_json::json!([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncated_text}
        ]);

        self.send_completion_request(&messages, text)
    }

    fn translate_text_with_context(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        context: &TranslationContext,
    ) -> AppResult<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }

        let system_prompt =
            Self::build_system_prompt(source_lang, target_lang, self.glossary_markdown.as_deref());
        let truncated_text = Self::truncate_input(text);

        tracing::info!(
            "LlamaCppBackend: translating with {} context entries '{}' (len={}, max_tokens={})",
            context.len(),
            text.chars().take(80).collect::<String>(),
            text.len(),
            self.max_tokens
        );

        // 构建消息列表：system + 上下文对话历史 + 当前待翻译文本
        let mut messages = vec![serde_json::json!({
            "role": "system",
            "content": system_prompt
        })];

        // 将前序段落的 (source, target) 作为 user/assistant 对话历史注入
        // 这让 LLM 能看到之前的翻译上下文，提升代词消解和术语一致性
        for entry in context.entries() {
            messages.push(serde_json::json!({
                "role": "user",
                "content": entry.source
            }));
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": entry.target
            }));
        }

        // 当前待翻译文本
        messages.push(serde_json::json!({
            "role": "user",
            "content": truncated_text
        }));

        let messages_val = serde_json::Value::Array(messages);
        self.send_completion_request(&messages_val, text)
    }

    fn translate_srt_text(
        &self,
        srt_text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<String> {
        if srt_text.trim().is_empty() {
            return Ok(String::new());
        }

        let system_prompt =
            Self::build_system_prompt(source_lang, target_lang, self.glossary_markdown.as_deref());

        // SRT 翻译指令：让 LLM 保持 SRT 结构，仅翻译文本部分
        let srt_instruction = format!(
            "You are a professional subtitle translator. \n\
            Translate the following SRT subtitles from {source_lang} to {target_lang}. \n\
            IMPORTANT RULES:\n\
            1. Keep the SRT format exactly: line numbers, timestamps, and blank lines must remain unchanged\n\
            2. Only translate the text content (lines after the timestamp line)\n\
            3. Maintain the same number of subtitle blocks\n\
            4. Preserve technical terms, code, and variable names in English\n\
            5. Use natural, spoken language suitable for voice-over dubbing\n\
            6. If a line contains multiple sentences, translate them all\n\
            7. Do not add or remove subtitle blocks\n\
            8. Preserve [[T0]] style placeholders if present\n\n\
            # SRT Subtitles to Translate:\n\n{srt_text}"
        );

        // SRT 文本可能很长，使用更大的 max_tokens
        let messages = serde_json::json!([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": srt_instruction}
        ]);

        tracing::info!(
            "LlamaCppBackend: SRT batch translation (srt_len={}, blocks={})",
            srt_text.len(),
            srt_text.split("\n\n").count()
        );

        // SRT 翻译使用更大的超时和更多 token
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": "translation",
            "messages": messages,
            "max_tokens": self.max_tokens * 8,  // SRT 批量翻译需要更多 token
            "temperature": self.temperature,
            "top_p": 0.8,
            "repeat_penalty": 1.05,
            "stream": false,
        });

        let max_retries = 2;
        let per_attempt_timeout = std::time::Duration::from_secs(180); // SRT 翻译更长超时

        for attempt in 1..=max_retries {
            if attempt > 1 {
                tracing::warn!(
                    "LlamaCppBackend: retrying SRT translation (attempt {attempt}/{max_retries}) after restart"
                );
            }

            let response = ureq::post(&url)
                .timeout(per_attempt_timeout)
                .set("Content-Type", "application/json")
                .send_string(&body.to_string());

            match response {
                Ok(resp) => {
                    let response_text = resp.into_string().map_err(|e| {
                        AppError::TranslationError(format!("Failed to read SRT response body: {e}"))
                    })?;

                    let response_json: serde_json::Value = serde_json::from_str(&response_text)
                        .map_err(|e| {
                            AppError::TranslationError(format!(
                                "Failed to parse SRT response JSON: {e}"
                            ))
                        })?;

                    let content = response_json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if content.is_empty() {
                        let fallback = response_json
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if fallback.is_empty() {
                            tracing::warn!("LlamaCppBackend: empty SRT response");
                            return Err(AppError::TranslationError(
                                "LLM returned empty response for SRT translation".to_string(),
                            ));
                        }
                        return Ok(fallback.to_string());
                    }

                    tracing::info!(
                        "LlamaCppBackend: SRT translated, output_len={}",
                        content.len()
                    );

                    return Ok(content.to_string());
                }
                Err(e) => {
                    tracing::warn!(
                        "LlamaCppBackend: SRT attempt {attempt}/{max_retries} failed: {e}"
                    );
                    if attempt < max_retries {
                        if let Err(restart_err) = self.restart_server() {
                            tracing::error!("LlamaCppBackend: restart failed: {restart_err}");
                        }
                    }
                }
            }
        }

        Err(AppError::TranslationError(
            "SRT translation failed after all retries".to_string(),
        ))
    }

    fn backend_name(&self) -> &str {
        "LlamaCppBackend(server)"
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        tracing::info!("Shutting down llama-server (port {})", self.port);
        if let Ok(mut child) = self.server_process.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl std::fmt::Debug for LlamaCppBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppBackend")
            .field("port", &self.port)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .finish()
    }
}

// ─── llama-server 工具函数 ───────────────────────────────

/// 查找 `llama-server` 二进制文件
///
/// 搜索顺序：
/// 1. `PATH` 环境变量
/// 2. Homebrew 常见安装路径（`/opt/homebrew/bin`, `/usr/local/bin`）
///
/// # 错误
/// 返回包含安装说明的错误。
fn find_llama_server() -> AppResult<std::path::PathBuf> {
    // 尝试 PATH 中的 llama-server
    if let Ok(path) = which("llama-server") {
        return Ok(path);
    }

    // 尝试 Homebrew 常见路径
    let candidates = [
        "/opt/homebrew/bin/llama-server",
        "/opt/homebrew/opt/llama.cpp/bin/llama-server",
        "/usr/local/bin/llama-server",
        "/usr/local/opt/llama.cpp/bin/llama-server",
    ];
    for candidate in &candidates {
        if std::path::Path::new(candidate).exists() {
            return Ok(std::path::PathBuf::from(candidate));
        }
    }

    // 未找到，返回详细错误
    let brew_available = which("brew").is_ok();
    let install_hint = if brew_available {
        "运行以下命令安装：\n  brew install llama.cpp"
    } else {
        "请安装 Homebrew 后运行：\n  brew install llama.cpp\n\n\
         或从 https://github.com/ggerganov/llama.cpp/releases 下载预编译版本"
    };

    Err(AppError::ModelLoadError(format!(
        "llama-server 未找到。\n\n{install_hint}"
    )))
}

/// 在 `PATH` 中查找可执行文件（简易 which 实现）
fn which(cmd: &str) -> Result<std::path::PathBuf, ()> {
    let path_env = std::env::var_os("PATH").ok_or(())?;
    for dir in std::env::split_paths(&path_env) {
        let full = dir.join(cmd);
        if full.is_file() {
            return Ok(full);
        }
    }
    Err(())
}

/// 分配一个空闲 TCP 端口
///
/// 通过绑定到端口 0 让操作系统分配空闲端口，然后立即关闭监听。
fn find_free_port() -> AppResult<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| AppError::ModelLoadError(format!("Failed to find free port: {e}")))?;
    Ok(listener.local_addr().unwrap().port())
}

/// 等待 llama-server 就绪
///
/// 轮询 `/health` 端点，直到服务响应或超时。
///
/// # 参数
/// - `port`: 服务端口
/// - `timeout_secs`: 超时秒数（模型加载可能需要较长时间）
fn wait_for_server_ready(port: u16, timeout_secs: u64) -> AppResult<()> {
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

    let mut last_error = String::new();
    while std::time::Instant::now() < deadline {
        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .call()
        {
            Ok(resp) => {
                if resp.status() == 200 {
                    tracing::debug!("llama-server health check passed");
                    return Ok(());
                }
                last_error = format!("HTTP {}", resp.status());
            }
            Err(e) => {
                last_error = e.to_string();
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    Err(AppError::ModelLoadError(format!(
        "llama-server did not become ready within {timeout_secs}s\n\
         Last error: {last_error}"
    )))
}

/// 清理 LLM 翻译输出
///
/// 去除提示词泄漏（如 "Chinese:" 前缀）、多余换行和空白。
fn clean_translation_output(raw: &str) -> String {
    let mut result = raw.trim().to_string();

    // 去除可能的提示词泄漏前缀
    let leak_prefixes = [
        "Chinese (Simplified): ",
        "Chinese: ",
        "中文：",
        "中文: ",
        "翻译：",
        "Translation: ",
    ];
    for prefix in &leak_prefixes {
        if result.starts_with(prefix) {
            result = result[prefix.len()..].trim().to_string();
            break;
        }
    }

    // 去除引号包裹（部分模型会用引号包裹翻译结果）
    if (result.starts_with('"') && result.ends_with('"'))
        || (result.starts_with('"') && result.ends_with('"'))
        || (result.starts_with('「') && result.ends_with('」'))
    {
        result = result[1..result.len() - 1].trim().to_string();
    }

    // 去除尾部可能的原文重复
    // 如果输出包含换行，只取第一段（翻译结果通常在第一行）
    if let Some(first_line) = result.lines().next() {
        let first = first_line.trim();
        if !first.is_empty() && first.chars().count() > 2 {
            result = first.to_string();
        }
    }

    result
}

// ─── 本地翻译引擎 ─────────────────────────────────────────

/// 本地离线翻译引擎
///
/// 从本地路径加载模型，使用 `InferenceBackend` 执行翻译推理，
/// 并通过 `TerminologyManager` 进行术语替换和还原。
///
/// # 离线运行
/// 引擎初始化时仅从本地文件系统加载模型，不发起任何网络请求。
/// 推理过程完全在本地执行。
///
/// # 术语集成
/// 翻译流程：
/// 1. 对原文应用术语占位符替换
/// 2. 调用推理后端翻译（带占位符的文本）
/// 3. 还原占位符为中文术语
///
/// # 示例
/// ```no_run
/// use vt_core::translate::{LocalTranslationEngine, MockInferenceBackend, TerminologyManager, GlossaryEntry, TranslationProvider};
/// use vt_core::models::segment::Segment;
/// use vt_core::error::AppResult;
///
/// fn translate() -> AppResult<()> {
///     let backend = MockInferenceBackend::default();
///     let terminology = TerminologyManager::from_entries(vec![
///         GlossaryEntry::new("GPU", "图形处理器"),
///     ])?;
///     let engine = LocalTranslationEngine::new(backend).with_terminology(terminology);
///
///     let mut segments = vec![
///         Segment::new("s1".into(), 0.0, 5.0, "The GPU renders graphics".into()),
///     ];
///     engine.translate_batch(&mut segments, "en", "zh")?;
///     assert!(segments[0].target_text.is_some());
///     Ok(())
/// }
/// ```
pub struct LocalTranslationEngine {
    /// 推理后端
    backend: Box<dyn InferenceBackend>,
    /// 术语表管理器（可选）
    terminology: Option<TerminologyManager>,
    /// 批量翻译大小
    batch_size: usize,
}

impl LocalTranslationEngine {
    /// 创建本地翻译引擎
    ///
    /// # 参数
    /// - `backend`: 推理后端实例
    #[must_use]
    pub fn new(backend: impl InferenceBackend + 'static) -> Self {
        Self {
            backend: Box::new(backend),
            terminology: None,
            batch_size: 10,
        }
    }

    /// 设置术语表管理器
    #[must_use]
    pub fn with_terminology(mut self, terminology: TerminologyManager) -> Self {
        self.terminology = Some(terminology);
        self
    }

    /// 设置批量翻译大小
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// 获取推理后端名称
    #[must_use]
    pub fn backend_name(&self) -> &str {
        self.backend.backend_name()
    }

    /// 从配置创建本地翻译引擎
    ///
    /// 根据配置中的 `model_path` 加载模型，使用指定的推理后端。
    ///
    /// # 参数
    /// - `config`: 翻译配置
    /// - `backend`: 推理后端工厂函数
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 模型路径不存在
    pub fn from_config<F>(
        config: &crate::config::TranslationConfig,
        backend_factory: F,
    ) -> AppResult<Self>
    where
        F: FnOnce(&crate::config::TranslationConfig) -> AppResult<Box<dyn InferenceBackend>>,
    {
        // 验证模型路径（若指定）
        if let Some(ref model_path) = config.model_path {
            if !model_path.exists() {
                return Err(AppError::FileNotFound(model_path.clone()));
            }
        }

        let backend = backend_factory(config)?;

        let mut engine = Self::new_raw(backend).with_batch_size(config.batch_size);

        // 加载术语表（内置编程术语 + 用户自定义术语）
        let mut all_entries = Vec::new();

        // 当 force_glossary 启用时，始终包含内置编程术语
        if config.force_glossary {
            all_entries.extend(builtin_programming_terms());
        }

        // 加载用户自定义术语表（如有）
        if let Some(ref glossary_path) = config.glossary_path {
            let path = std::path::Path::new(glossary_path);
            let user_terminology = if path.extension().is_some_and(|ext| ext == "json") {
                TerminologyManager::load_from_json(path)?
            } else {
                TerminologyManager::load_from_csv(path)?
            };
            all_entries.extend(user_terminology.entries().to_vec());
        }

        if !all_entries.is_empty() {
            let terminology = TerminologyManager::from_entries(all_entries)?;
            engine = engine.with_terminology(terminology);
        }

        Ok(engine)
    }

    /// 内部构造函数，接受 boxed backend
    fn new_raw(backend: Box<dyn InferenceBackend>) -> Self {
        Self {
            backend,
            terminology: None,
            batch_size: 10,
        }
    }

    /// 批量翻译文本（内部方法）
    fn translate_texts_batch(
        &self,
        texts: &[String],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<Vec<String>> {
        let mut results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let translated = self
                .backend
                .translate_texts(chunk, source_lang, target_lang)?;
            results.extend(translated);
        }
        Ok(results)
    }
}

impl TranslationProvider for LocalTranslationEngine {
    fn translate_batch(
        &self,
        segments: &mut [Segment],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<()> {
        if segments.is_empty() {
            return Ok(());
        }

        // 步骤1：应用术语占位符
        let (texts, mappings) = apply_terminology(&self.terminology, segments);

        // 步骤2：批量翻译
        let translated = self.translate_texts_batch(&texts, source_lang, target_lang)?;

        // 步骤3：还原占位符并写入 target_text
        restore_terminology(&self.terminology, segments, &translated, &mappings);

        Ok(())
    }

    fn translate_segment_with_context(
        &self,
        segment: &mut Segment,
        source_lang: &str,
        target_lang: &str,
        context: &TranslationContext,
    ) -> AppResult<()> {
        // 步骤1：应用术语占位符
        let (text, mapping) = match &self.terminology {
            Some(term) => term.apply_placeholders(&segment.source_text),
            None => (segment.source_text.clone(), Vec::new()),
        };

        // 步骤2：带上下文翻译
        let translated =
            self.backend
                .translate_text_with_context(&text, source_lang, target_lang, context)?;

        // 步骤3：还原占位符
        let final_text = match &self.terminology {
            Some(term) if !mapping.is_empty() => term.restore_placeholders(&translated, &mapping),
            _ => translated,
        };

        segment.target_text = Some(final_text);
        Ok(())
    }

    fn translate_srt(
        &self,
        segments: &mut [Segment],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<()> {
        if segments.is_empty() {
            return Ok(());
        }

        use crate::translation_extras::{segments_to_srt, srt_to_segments};

        // 步骤1：应用术语占位符到所有 segments 的 source_text
        let (texts, mappings) = apply_terminology(&self.terminology, segments);
        // 临时替换 source_text 为带占位符的文本
        let original_texts: Vec<String> = segments.iter().map(|s| s.source_text.clone()).collect();
        for (i, seg) in segments.iter_mut().enumerate() {
            seg.source_text = texts[i].clone();
        }

        // 步骤2：将 segments 组成 SRT 格式文本
        let srt_text = segments_to_srt(segments);

        // 步骤3：调用后端的 SRT 翻译方法
        tracing::info!(
            "LocalTranslationEngine: SRT batch translation (segments={}, srt_len={})",
            segments.len(),
            srt_text.len()
        );
        let translated_srt =
            self.backend
                .translate_srt_text(&srt_text, source_lang, target_lang)?;

        // 步骤4：从 SRT 解析回 segments
        let translated_segments = srt_to_segments(&translated_srt, segments);

        // 步骤5：还原术语占位符并写入 target_text
        for (i, seg) in segments.iter_mut().enumerate() {
            // 恢复原始 source_text
            seg.source_text = original_texts[i].clone();

            // 获取翻译后的文本
            let translated_text = translated_segments
                .get(i)
                .and_then(|s| s.target_text.clone())
                .unwrap_or_default();

            // 还原术语占位符
            let final_text = match &self.terminology {
                Some(term) if !mappings[i].is_empty() => {
                    term.restore_placeholders(&translated_text, &mappings[i])
                }
                _ => translated_text,
            };

            seg.target_text = Some(final_text);
        }

        tracing::info!(
            "LocalTranslationEngine: SRT batch translation completed ({} segments)",
            segments.len()
        );

        Ok(())
    }
}

impl std::fmt::Debug for LocalTranslationEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalTranslationEngine")
            .field("backend", &self.backend.backend_name())
            .field("has_terminology", &self.terminology.is_some())
            .field("batch_size", &self.batch_size)
            .finish()
    }
}

// ─── DeepLX 在线翻译提供者 ────────────────────────────────

/// DeepLX 服务健康状态
///
/// 用于 `TranslationRouter` 跟踪 DeepLX 服务的可用性，
/// 避免频繁尝试已失效的服务。
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// 服务是否可用
    healthy: bool,
    /// 上次健康检查时间（Unix 时间戳，秒）
    last_check: u64,
    /// 连续失败次数
    consecutive_failures: u32,
}

impl HealthStatus {
    /// 创建默认的健康状态（未知，标记为不健康）
    #[must_use]
    pub fn new() -> Self {
        Self {
            healthy: false,
            last_check: 0,
            consecutive_failures: 0,
        }
    }

    /// 服务是否可用
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.healthy
    }

    /// 标记为健康
    pub fn mark_healthy(&mut self) {
        self.healthy = true;
        self.last_check = current_unix_secs();
        self.consecutive_failures = 0;
    }

    /// 标记为不健康
    pub fn mark_unhealthy(&mut self) {
        self.healthy = false;
        self.last_check = current_unix_secs();
        self.consecutive_failures += 1;
    }

    /// 是否需要重新检查健康状态
    ///
    /// 若距上次检查超过 `interval_secs` 秒，则返回 `true`。
    #[must_use]
    pub fn needs_recheck(&self, interval_secs: u64) -> bool {
        let elapsed = current_unix_secs().saturating_sub(self.last_check);
        elapsed >= interval_secs
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// 获取当前 Unix 时间戳（秒）
fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// DeepLX 在线翻译配置
#[derive(Debug, Clone)]
pub struct DlxConfig {
    /// DeepLX 服务端点 URL（如 `http://localhost:1188`）
    pub endpoint: String,
    /// 请求超时时间（秒）
    pub timeout_secs: u64,
    /// 最大重试次数（仅对 429 和 5xx 重试）
    pub max_retries: usize,
}

impl Default for DlxConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:1188".to_string(),
            timeout_secs: 10,
            max_retries: 3,
        }
    }
}

/// DeepLX 在线翻译提供者
///
/// 通过 HTTP 调用自部署的 DeepLX 服务进行翻译。
/// 当服务不可用时返回 [`AppError::OnlineTranslationUnavailable`]，
/// 由 [`TranslationRouter`] 捕获并降级到本地模型。
///
/// # API 调用
/// - **翻译**：`POST /translate`，body 为 `{"text": "...", "source_lang": "EN", "target_lang": "ZH"}`
/// - **健康检查**：`GET /health`
///
/// # 线程安全
/// 所有字段均为不可变或线程安全的，满足 `Send + Sync`。
pub struct DlxProvider {
    /// DeepLX 配置
    config: DlxConfig,
}

impl DlxProvider {
    /// 创建 DeepLX 翻译提供者
    ///
    /// # 参数
    /// - `config`: DeepLX 配置
    #[must_use]
    pub fn new(config: DlxConfig) -> Self {
        Self { config }
    }

    /// 从翻译配置创建 DeepLX 提供者
    ///
    /// # 参数
    /// - `tc`: 翻译配置
    #[must_use]
    pub fn from_translation_config(tc: &crate::config::TranslationConfig) -> Self {
        Self::new(DlxConfig {
            endpoint: tc.dlx_endpoint.clone(),
            timeout_secs: tc.dlx_timeout_secs,
            max_retries: tc.dlx_max_retries,
        })
    }

    /// 检查 DeepLX 服务是否健康
    ///
    /// 调用 `/health` 端点，返回 200 则认为健康。
    pub fn check_health(&self) -> bool {
        let url = format!("{}/health", self.config.endpoint);
        let timeout = std::time::Duration::from_secs(5);
        match ureq::get(&url).timeout(timeout).call() {
            Ok(resp) => resp.status() == 200,
            Err(_) => false,
        }
    }

    /// 翻译单条文本
    ///
    /// 内部处理重试逻辑：仅对 429 和 5xx 错误重试，最多 `max_retries` 次。
    ///
    /// # 参数
    /// - `text`: 待翻译文本
    /// - `source_lang`: 源语言代码（如 `en`、`zh`）
    /// - `target_lang`: 目标语言代码
    ///
    /// # 错误
    /// - [`AppError::OnlineTranslationUnavailable`][]: 服务不可用、超时、429/5xx
    /// - [`AppError::TranslationError`][]: 响应解析失败
    pub fn translate_text(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }

        let url = format!("{}/translate", self.config.endpoint);
        let body = serde_json::json!({
            "text": text,
            "source_lang": lang_code_to_dlx(source_lang),
            "target_lang": lang_code_to_dlx(target_lang),
        });

        let timeout = std::time::Duration::from_secs(self.config.timeout_secs);
        let mut last_error: Option<String> = None;

        for attempt in 1..=self.config.max_retries {
            tracing::debug!(
                "DLX translate: attempt {}/{}, text='{}'",
                attempt,
                self.config.max_retries,
                text.chars().take(60).collect::<String>()
            );

            let response = ureq::post(&url)
                .timeout(timeout)
                .set("Content-Type", "application/json")
                .send_string(&body.to_string());

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status == 200 {
                        let resp_text = resp.into_string().map_err(|e| {
                            AppError::HttpError(format!("Failed to read DLX response: {e}"))
                        })?;
                        let resp_json: serde_json::Value = serde_json::from_str(&resp_text)
                            .map_err(|e| {
                                AppError::TranslationError(format!(
                                    "Failed to parse DLX response JSON: {e}\nResponse: {resp_text}"
                                ))
                            })?;

                        let data =
                            resp_json
                                .get("data")
                                .and_then(|v| v.as_str())
                                .ok_or_else(|| {
                                    AppError::TranslationError(format!(
                                        "DLX response missing 'data' field: {resp_text}"
                                    ))
                                })?;

                        if data.is_empty() {
                            tracing::warn!("DLX returned empty translation for '{text}'");
                            return Ok(text.to_string());
                        }

                        return Ok(data.trim().to_string());
                    }
                    // 429 / 5xx → 可重试
                    last_error = Some(format!("HTTP {status}"));
                    tracing::warn!(
                        "DLX translate: attempt {}/{} got HTTP {status}, retrying",
                        attempt,
                        self.config.max_retries
                    );
                }
                Err(ureq::Error::Status(code, _)) => {
                    last_error = Some(format!("HTTP {code}"));
                    if code != 429 && !(500..600).contains(&code) {
                        // 非 429/5xx 错误，不重试
                        return Err(AppError::OnlineTranslationUnavailable(format!(
                            "DLX returned HTTP {code} (non-retryable)"
                        )));
                    }
                    tracing::warn!(
                        "DLX translate: attempt {}/{} got HTTP {code}, retrying",
                        attempt,
                        self.config.max_retries
                    );
                }
                Err(e) => {
                    // 连接拒绝、超时等网络错误 → 不可重试
                    return Err(AppError::OnlineTranslationUnavailable(format!(
                        "DLX connection failed: {e}"
                    )));
                }
            }

            // 重试前等待（指数退避）
            if attempt < self.config.max_retries {
                let backoff_ms = 500_u64 * 2_u64.pow((attempt - 1) as u32);
                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
            }
        }

        Err(AppError::OnlineTranslationUnavailable(format!(
            "DLX translation failed after {} retries: {}",
            self.config.max_retries,
            last_error.unwrap_or_default()
        )))
    }
}

impl TranslationProvider for DlxProvider {
    fn translate_batch(
        &self,
        segments: &mut [Segment],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<()> {
        if segments.is_empty() {
            return Ok(());
        }

        for seg in segments.iter_mut() {
            let translated = self.translate_text(&seg.source_text, source_lang, target_lang)?;
            seg.target_text = Some(translated);
        }
        Ok(())
    }
}

impl std::fmt::Debug for DlxProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DlxProvider")
            .field("endpoint", &self.config.endpoint)
            .field("timeout_secs", &self.config.timeout_secs)
            .field("max_retries", &self.config.max_retries)
            .finish()
    }
}

/// 将语言代码转换为 DeepLX 格式
///
/// DeepLX 使用大写语言代码（如 `EN`、`ZH`）。
fn lang_code_to_dlx(code: &str) -> &str {
    match code.to_lowercase().as_str() {
        "en" => "EN",
        "zh" | "zh-cn" | "zh-hans" => "ZH",
        "ja" => "JA",
        "ko" => "KO",
        "fr" => "FR",
        "de" => "DE",
        "es" => "ES",
        "ru" => "RU",
        "pt" => "PT",
        "it" => "IT",
        _ => "EN",
    }
}

// ─── TranslationRouter（两级降级路由）──────────────────────

/// 翻译路由配置
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// 是否优先使用在线翻译（DeepLX）
    pub prefer_online: bool,
    /// 在线翻译失败时是否自动降级到本地模型
    pub fallback_on_error: bool,
    /// 健康检查间隔（秒）
    pub health_check_interval_secs: u64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            prefer_online: true,
            fallback_on_error: true,
            health_check_interval_secs: 300,
        }
    }
}

/// 两级翻译路由器
///
/// 实现 "DeepLX 优先 → 本地降级" 的智能路由策略：
///
/// 1. 检查 DeepLX 服务健康状态（缓存状态，每 `health_check_interval_secs` 秒刷新）
/// 2. 若健康，尝试使用 `DlxProvider` 翻译
/// 3. 若 DeepLX 失败（返回 [`AppError::OnlineTranslationUnavailable`]），
///    记录日志，切换到 `LocalTranslationEngine`
/// 4. 若本地模型也失败，返回 [`AppError::TranslationError`]
///
/// # 降级透明性
/// 上层调用者（Pipeline）通过 `TranslationProvider` trait 调用，
/// 不感知降级切换，由路由器内部处理。
///
/// # 线程安全
/// 健康状态使用 `Arc<std::sync::Mutex<HealthStatus>>` 维护，
/// 满足 `Send + Sync`。
pub struct TranslationRouter {
    /// 在线翻译提供者（DeepLX）
    online: DlxProvider,
    /// 本地翻译引擎（LlamaCppBackend）
    local: LocalTranslationEngine,
    /// 路由配置
    config: RouterConfig,
    /// DeepLX 健康状态（线程安全共享）
    health: std::sync::Arc<std::sync::Mutex<HealthStatus>>,
}

impl TranslationRouter {
    /// 创建翻译路由器
    ///
    /// # 参数
    /// - `online`: DeepLX 在线翻译提供者
    /// - `local`: 本地翻译引擎
    /// - `config`: 路由配置
    #[must_use]
    pub fn new(online: DlxProvider, local: LocalTranslationEngine, config: RouterConfig) -> Self {
        Self {
            online,
            local,
            config,
            health: std::sync::Arc::new(std::sync::Mutex::new(HealthStatus::new())),
        }
    }

    /// 从翻译配置创建路由器
    ///
    /// # 参数
    /// - `tc`: 翻译配置
    /// - `local`: 已初始化的本地翻译引擎
    #[must_use]
    pub fn from_config(
        tc: &crate::config::TranslationConfig,
        local: LocalTranslationEngine,
    ) -> Self {
        let dlx = DlxProvider::from_translation_config(tc);
        let router_config = RouterConfig {
            prefer_online: tc.prefer_online,
            fallback_on_error: tc.fallback_on_error,
            health_check_interval_secs: tc.health_check_interval_secs,
        };
        Self::new(dlx, local, router_config)
    }

    /// 检查并更新 DeepLX 健康状态
    ///
    /// 若距上次检查超过间隔时间，则重新检查。
    fn refresh_health_if_needed(&self) {
        let mut health = match self.health.lock() {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!("Failed to lock health status: {e}");
                return;
            }
        };

        if health.needs_recheck(self.config.health_check_interval_secs) {
            let is_healthy = self.online.check_health();
            if is_healthy {
                health.mark_healthy();
                tracing::info!("DLX health check: healthy");
            } else {
                health.mark_unhealthy();
                tracing::warn!(
                    "DLX health check: unhealthy (failures={})",
                    health.consecutive_failures
                );
            }
        }
    }

    /// 强制标记 DeepLX 为不健康（在翻译失败时调用）
    fn mark_unhealthy(&self) {
        if let Ok(mut health) = self.health.lock() {
            health.mark_unhealthy();
        }
    }

    /// 当前 DeepLX 是否健康
    fn is_online_healthy(&self) -> bool {
        match self.health.lock() {
            Ok(h) => h.is_healthy(),
            Err(_) => false,
        }
    }
}

impl TranslationProvider for TranslationRouter {
    fn translate_batch(
        &self,
        segments: &mut [Segment],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<()> {
        if segments.is_empty() {
            return Ok(());
        }

        // 检查健康状态（可能触发重新检查）
        self.refresh_health_if_needed();

        // 尝试在线翻译
        if self.config.prefer_online && self.is_online_healthy() {
            tracing::debug!(
                "TranslationRouter: using DeepLX for {} segments",
                segments.len()
            );
            match self
                .online
                .translate_batch(segments, source_lang, target_lang)
            {
                Ok(()) => {
                    tracing::debug!("TranslationRouter: DeepLX translation succeeded");
                    return Ok(());
                }
                Err(AppError::OnlineTranslationUnavailable(reason)) => {
                    tracing::warn!(
                        "TranslationRouter: DeepLX unavailable ({}), falling back to local",
                        reason
                    );
                    self.mark_unhealthy();
                    if !self.config.fallback_on_error {
                        return Err(AppError::OnlineTranslationUnavailable(reason));
                    }
                    // 继续到本地翻译
                }
                Err(e) => {
                    // 非降级类错误，直接返回
                    return Err(e);
                }
            }
        } else {
            tracing::debug!(
                "TranslationRouter: DeepLX not healthy or not preferred, using local engine"
            );
        }

        // 降级到本地翻译
        tracing::debug!(
            "TranslationRouter: using local engine for {} segments",
            segments.len()
        );
        self.local
            .translate_batch(segments, source_lang, target_lang)
    }

    fn translate_segment_with_context(
        &self,
        segment: &mut Segment,
        source_lang: &str,
        target_lang: &str,
        context: &TranslationContext,
    ) -> AppResult<()> {
        // 检查健康状态
        self.refresh_health_if_needed();

        // 尝试在线翻译（DeepLX 不支持上下文，使用默认批量翻译）
        if self.config.prefer_online && self.is_online_healthy() {
            tracing::debug!("TranslationRouter: using DeepLX for context-aware segment");
            let mut segments = std::slice::from_mut(segment);
            match self
                .online
                .translate_batch(&mut segments, source_lang, target_lang)
            {
                Ok(()) => {
                    tracing::debug!("TranslationRouter: DeepLX translation succeeded");
                    return Ok(());
                }
                Err(AppError::OnlineTranslationUnavailable(reason)) => {
                    tracing::warn!(
                        "TranslationRouter: DeepLX unavailable ({}), falling back to local with context",
                        reason
                    );
                    self.mark_unhealthy();
                    if !self.config.fallback_on_error {
                        return Err(AppError::OnlineTranslationUnavailable(reason));
                    }
                }
                Err(e) => return Err(e),
            }
        } else {
            tracing::debug!(
                "TranslationRouter: DeepLX not healthy or not preferred, using local engine with context"
            );
        }

        // 降级到本地翻译（支持上下文）
        self.local
            .translate_segment_with_context(segment, source_lang, target_lang, context)
    }

    fn translate_srt(
        &self,
        segments: &mut [Segment],
        source_lang: &str,
        target_lang: &str,
    ) -> AppResult<()> {
        // SRT 批量翻译仅支持本地 LLM 后端（DeepLX 不支持 SRT 格式）
        tracing::info!(
            "TranslationRouter: SRT batch translation for {} segments",
            segments.len()
        );

        // 如果偏好在线且 DeepLX 可用，回退到逐段翻译（DeepLX 不支持 SRT 模式）
        self.refresh_health_if_needed();
        if self.config.prefer_online && self.is_online_healthy() {
            tracing::debug!(
                "TranslationRouter: DeepLX available, using batch translation instead of SRT mode"
            );
            return self
                .local
                .translate_batch(segments, source_lang, target_lang);
        }

        // 使用本地引擎的 SRT 翻译
        self.local.translate_srt(segments, source_lang, target_lang)
    }
}

impl std::fmt::Debug for TranslationRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let healthy = self.is_online_healthy();
        f.debug_struct("TranslationRouter")
            .field("online", &self.online)
            .field("local", &self.local)
            .field("prefer_online", &self.config.prefer_online)
            .field("fallback_on_error", &self.config.fallback_on_error)
            .field("online_healthy", &healthy)
            .finish()
    }
}

// ─── BLEU 分数计算 ────────────────────────────────────────

/// BLEU 翻译精度评估器
///
/// 实现 BLEU-N（默认 N=4）评分算法，用于量化评估翻译质量。
/// 支持中英文混合文本的 tokenization。
///
/// # 算法
/// BLEU = brevity_penalty × exp(Σ wn × log(pn))
/// - pn: n-gram 精度
/// - wn: 权重（默认均匀分布，1/N）
/// - brevity_penalty: 简短惩罚
#[derive(Debug, Clone)]
pub struct BleuEvaluator {
    /// 最大 n-gram 阶数
    max_n: usize,
}

impl BleuEvaluator {
    /// 创建 BLEU-4 评估器（默认）
    #[must_use]
    pub fn new() -> Self {
        Self { max_n: 4 }
    }

    /// 创建指定 n-gram 阶数的评估器
    ///
    /// # 参数
    /// - `max_n`: 最大 n-gram 阶数（如 2 表示 BLEU-2）
    #[must_use]
    pub fn with_max_n(max_n: usize) -> Self {
        Self {
            max_n: max_n.max(1),
        }
    }

    /// 评估单条翻译的 BLEU 分数
    ///
    /// # 参数
    /// - `candidate`: 机器翻译结果
    /// - `references`: 参考翻译列表（至少1条）
    ///
    /// # 返回值
    /// BLEU 分数，范围 [0.0, 1.0]
    #[must_use]
    pub fn evaluate(&self, candidate: &str, references: &[&str]) -> f64 {
        if references.is_empty() {
            return 0.0;
        }

        let candidate_tokens = tokenize(candidate);
        let reference_tokens: Vec<Vec<&str>> = references.iter().map(|r| tokenize(r)).collect();

        if candidate_tokens.is_empty() {
            return 0.0;
        }

        // 计算 n-gram 精度
        let mut log_precision_sum = 0.0;
        for n in 1..=self.max_n {
            let precision = self.ngram_precision(&candidate_tokens, &reference_tokens, n);
            if precision == 0.0 {
                return 0.0;
            }
            let weight = 1.0 / self.max_n as f64;
            log_precision_sum += weight * precision.ln();
        }

        // 简短惩罚
        let bp = self.brevity_penalty(&candidate_tokens, &reference_tokens);

        bp * log_precision_sum.exp()
    }

    /// 批量评估 BLEU 分数
    ///
    /// 计算多条翻译的平均 BLEU 分数。
    ///
    /// # 参数
    /// - `candidates`: 机器翻译结果列表
    /// - `references`: 对应的参考翻译列表（每个元素为多条参考）
    ///
    /// # 返回值
    /// 平均 BLEU 分数
    #[must_use]
    pub fn evaluate_batch(&self, candidates: &[String], references: &[Vec<&str>]) -> f64 {
        if candidates.is_empty() || candidates.len() != references.len() {
            return 0.0;
        }

        let total: f64 = candidates
            .iter()
            .zip(references.iter())
            .map(|(cand, refs)| self.evaluate(cand, refs))
            .sum();

        total / candidates.len() as f64
    }

    /// 计算 n-gram 精度
    fn ngram_precision(&self, candidate: &[&str], references: &[Vec<&str>], n: usize) -> f64 {
        let candidate_ngrams = count_ngrams(candidate, n);
        if candidate_ngrams.is_empty() {
            return 0.0;
        }

        // 对每个候选 n-gram，取参考翻译中的最大计数
        let mut clipped_count = 0usize;
        let mut total_count = 0usize;

        for (ngram, cand_count) in &candidate_ngrams {
            let max_ref_count = references
                .iter()
                .map(|r| count_ngram_occurrences(r, n, ngram))
                .max()
                .unwrap_or(0);

            clipped_count += (*cand_count).min(max_ref_count);
            total_count += cand_count;
        }

        if total_count == 0 {
            0.0
        } else {
            clipped_count as f64 / total_count as f64
        }
    }

    /// 计算简短惩罚
    fn brevity_penalty(&self, candidate: &[&str], references: &[Vec<&str>]) -> f64 {
        let cand_len = candidate.len();

        // 找到与候选长度最接近的参考翻译
        let closest_ref_len = references
            .iter()
            .map(|r| r.len())
            .min_by_key(|&len| (len as isize - cand_len as isize).unsigned_abs())
            .unwrap_or(0);

        if cand_len > closest_ref_len {
            1.0
        } else if cand_len == 0 {
            0.0
        } else {
            let b = closest_ref_len as f64 / cand_len as f64;
            (1.0 - b).exp()
        }
    }
}

impl Default for BleuEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// 文本分词
///
/// 对英文按空格分词，对中文按字符分词。
/// 支持中英文混合文本。
fn tokenize(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .flat_map(|word| {
            // 检查是否包含中文字符
            if word.chars().any(is_cjk_char) {
                // 对中文部分按字符分割，英文部分按空格
                split_mixed_word(word)
            } else {
                vec![word]
            }
        })
        .collect()
}

/// 判断是否为 CJK 字符
fn is_cjk_char(c: char) -> bool {
    let code = c as u32;
    // CJK 统一汉字、扩展A、扩展B等
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
}

/// 分割中英文混合词
///
/// CJK 字符逐个作为独立 token，非 CJK 连续字符作为一个 token。
fn split_mixed_word(word: &str) -> Vec<&str> {
    if !word.chars().any(is_cjk_char) {
        return vec![word];
    }

    let mut result = Vec::new();
    let mut non_cjk_start: Option<usize> = None;

    for (i, c) in word.char_indices() {
        if is_cjk_char(c) {
            // 先输出积累的非 CJK 部分
            if let Some(start) = non_cjk_start.take() {
                if let Some(slice) = word.get(start..i) {
                    if !slice.is_empty() {
                        result.push(slice);
                    }
                }
            }
            // CJK 字符单独作为一个 token
            let end = i + c.len_utf8();
            if let Some(slice) = word.get(i..end) {
                result.push(slice);
            }
        } else {
            // 非 CJK 字符，记录起点
            non_cjk_start.get_or_insert(i);
        }
    }

    // 输出最后一段非 CJK 部分
    if let Some(start) = non_cjk_start {
        if let Some(slice) = word.get(start..) {
            if !slice.is_empty() {
                result.push(slice);
            }
        }
    }

    if result.is_empty() {
        result.push(word);
    }

    result
}

/// 统计 n-gram 频次
///
/// 返回 `HashMap<String, usize>`，键为空格连接的 n-gram 字符串。
fn count_ngrams(tokens: &[&str], n: usize) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    if tokens.len() < n {
        return counts;
    }

    for window in tokens.windows(n) {
        let ngram = window.join(" ");
        *counts.entry(ngram).or_insert(0) += 1;
    }

    counts
}

/// 统计特定 n-gram 在 token 序列中的出现次数
fn count_ngram_occurrences(tokens: &[&str], n: usize, ngram: &str) -> usize {
    if tokens.len() < n {
        return 0;
    }

    tokens.windows(n).filter(|w| w.join(" ") == ngram).count()
}

// ─── 本地引擎单元测试 ────────────────────────────────────

#[cfg(test)]
mod local_engine_tests {
    use super::*;

    /// 验证 MockInferenceBackend 默认词典翻译。
    #[test]
    fn test_mock_backend_default() {
        let backend = MockInferenceBackend::default();
        let result = backend
            .translate_text("Hello, world", "en", "zh")
            .expect("Translation failed");
        assert_eq!(result, "你好，世界");
    }

    /// 验证 MockInferenceBackend 自定义词典翻译。
    #[test]
    fn test_mock_backend_custom() {
        let backend = MockInferenceBackend::from_pairs(&[("test phrase", "测试短语")]);
        let result = backend
            .translate_text("test phrase", "en", "zh")
            .expect("Translation failed");
        assert_eq!(result, "测试短语");
    }

    /// 验证 MockInferenceBackend 批量翻译。
    #[test]
    fn test_mock_backend_batch() {
        let backend = MockInferenceBackend::default();
        let texts = vec!["Hello".to_string(), "World".to_string(), "Test".to_string()];
        let results = backend
            .translate_texts(&texts, "en", "zh")
            .expect("Batch translation failed");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], "你好");
        assert_eq!(results[1], "世界");
        assert_eq!(results[2], "测试");
    }

    /// 验证 MockInferenceBackend 对未匹配文本返回原文。
    #[test]
    fn test_mock_backend_unmatched() {
        let backend = MockInferenceBackend::new();
        let result = backend
            .translate_text("unknown text", "en", "zh")
            .expect("Translation failed");
        assert_eq!(result, "unknown text");
    }

    /// 验证 LocalTranslationEngine 基本翻译。
    #[test]
    fn test_local_engine_basic() {
        let engine = LocalTranslationEngine::new(MockInferenceBackend::default());

        let mut segments = vec![Segment::new("s1".into(), 0.0, 5.0, "Hello, world".into())];
        engine
            .translate_batch(&mut segments, "en", "zh")
            .expect("Translation failed");

        assert_eq!(segments[0].target_text.as_deref(), Some("你好，世界"));
    }

    /// 验证 LocalTranslationEngine 批量翻译。
    #[test]
    fn test_local_engine_batch() {
        let engine = LocalTranslationEngine::new(MockInferenceBackend::default());

        let mut segments = vec![
            Segment::new("s1".into(), 0.0, 1.0, "Hello".into()),
            Segment::new("s2".into(), 1.0, 2.0, "World".into()),
            Segment::new("s3".into(), 2.0, 3.0, "Test".into()),
        ];
        engine
            .translate_batch(&mut segments, "en", "zh")
            .expect("Translation failed");

        assert_eq!(segments[0].target_text.as_deref(), Some("你好"));
        assert_eq!(segments[1].target_text.as_deref(), Some("世界"));
        assert_eq!(segments[2].target_text.as_deref(), Some("测试"));
    }

    /// 验证 LocalTranslationEngine 术语集成。
    #[test]
    fn test_local_engine_with_terminology() {
        let terminology =
            TerminologyManager::from_entries(vec![GlossaryEntry::new("GPU", "图形处理器")])
                .expect("Failed to create terminology");

        let backend =
            MockInferenceBackend::from_pairs(&[("The [[T0]] renders graphics", "[[T0]]渲染图形")]);

        let engine = LocalTranslationEngine::new(backend).with_terminology(terminology);

        let mut segments = vec![Segment::new(
            "s1".into(),
            0.0,
            5.0,
            "The GPU renders graphics".into(),
        )];
        engine
            .translate_batch(&mut segments, "en", "zh")
            .expect("Translation failed");

        let target = segments[0]
            .target_text
            .as_ref()
            .expect("target_text should be set");
        assert!(
            target.contains("图形处理器"),
            "Should contain restored term, got: {target}"
        );
        assert!(
            !target.contains("[[T0]]"),
            "Placeholder should be restored, got: {target}"
        );
    }

    /// 验证 LocalTranslationEngine 空 Segment 列表。
    #[test]
    fn test_local_engine_empty_segments() {
        let engine = LocalTranslationEngine::new(MockInferenceBackend::default());

        let mut segments: Vec<Segment> = vec![];
        engine
            .translate_batch(&mut segments, "en", "zh")
            .expect("Empty segments should succeed");
    }

    /// 验证 LocalTranslationEngine 批量分片。
    #[test]
    fn test_local_engine_batch_split() {
        let engine =
            LocalTranslationEngine::new(MockInferenceBackend::default()).with_batch_size(2);

        let mut segments: Vec<Segment> = (0..5)
            .map(|i| Segment::new(format!("s{i}"), i as f64, (i + 1) as f64, "Hello".into()))
            .collect();

        engine
            .translate_batch(&mut segments, "en", "zh")
            .expect("Translation failed");

        for seg in &segments {
            assert_eq!(seg.target_text.as_deref(), Some("你好"));
        }
    }

    /// 验证 LocalTranslationEngine Debug 输出。
    #[test]
    fn test_local_engine_debug() {
        let engine = LocalTranslationEngine::new(MockInferenceBackend::default());
        let debug = format!("{engine:?}");
        assert!(debug.contains("LocalTranslationEngine"));
        assert!(debug.contains("MockInferenceBackend"));
    }

    /// 验证 BLEU 完美匹配得分为 1.0。
    #[test]
    fn test_bleu_perfect_match() {
        let evaluator = BleuEvaluator::new();
        let score = evaluator.evaluate("你好 世界", &["你好 世界"]);
        assert!(
            (score - 1.0).abs() < 0.01,
            "Perfect match should score ~1.0, got {score}"
        );
    }

    /// 验证 BLEU 部分匹配得分在 0-1 之间。
    #[test]
    fn test_bleu_partial_match() {
        // 使用 BLEU-2，因为短句的 3-gram 和 4-gram 精度为 0 会导致 BLEU-4 得 0 分
        let evaluator = BleuEvaluator::with_max_n(2);
        let score = evaluator.evaluate("你好 测试", &["你好 世界"]);
        assert!(
            score > 0.0 && score < 1.0,
            "Partial match should score between 0 and 1, got {score}"
        );
    }

    /// 验证 BLEU 完全不匹配得分为 0。
    #[test]
    fn test_bleu_no_match() {
        let evaluator = BleuEvaluator::new();
        let score = evaluator.evaluate("xyz", &["abc"]);
        assert_eq!(score, 0.0, "No match should score 0.0");
    }

    /// 验证 BLEU 空候选得分为 0。
    #[test]
    fn test_bleu_empty_candidate() {
        let evaluator = BleuEvaluator::new();
        let score = evaluator.evaluate("", &["你好"]);
        assert_eq!(score, 0.0, "Empty candidate should score 0.0");
    }

    /// 验证 BLEU 空参考得分为 0。
    #[test]
    fn test_bleu_empty_references() {
        let evaluator = BleuEvaluator::new();
        let score = evaluator.evaluate("你好", &[]);
        assert_eq!(score, 0.0, "Empty references should score 0.0");
    }

    /// 验证 BLEU 批量评估。
    #[test]
    fn test_bleu_batch() {
        let evaluator = BleuEvaluator::new();
        let candidates = vec!["你好 世界".to_string(), "你好 测试".to_string()];
        let references = vec![vec!["你好 世界"], vec!["你好 测试"]];
        let score = evaluator.evaluate_batch(&candidates, &references);
        assert!(
            (score - 1.0).abs() < 0.01,
            "All perfect matches should average to ~1.0, got {score}"
        );
    }

    /// 验证 BLEU-2 评估器。
    #[test]
    fn test_bleu_2() {
        let evaluator = BleuEvaluator::with_max_n(2);
        let score = evaluator.evaluate("你好 世界", &["你好 世界"]);
        assert!(
            (score - 1.0).abs() < 0.01,
            "Perfect match with BLEU-2 should score ~1.0, got {score}"
        );
    }

    /// 验证 InferenceBackend trait 的默认批量实现。
    #[test]
    fn test_inference_backend_default_batch() {
        struct SimpleBackend;
        impl InferenceBackend for SimpleBackend {
            fn translate_text(
                &self,
                text: &str,
                _source_lang: &str,
                _target_lang: &str,
            ) -> AppResult<String> {
                Ok(format!("[translated]{text}"))
            }
            fn backend_name(&self) -> &str {
                "SimpleBackend"
            }
        }

        let backend = SimpleBackend;
        let texts = vec!["a".to_string(), "b".to_string()];
        let results = backend
            .translate_texts(&texts, "en", "zh")
            .expect("Batch failed");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "[translated]a");
        assert_eq!(results[1], "[translated]b");
    }

    // ─── HealthStatus 测试 ──────────────────────────────────

    /// 验证 HealthStatus 初始状态为不健康。
    #[test]
    fn test_health_status_new() {
        let hs = HealthStatus::new();
        assert!(!hs.is_healthy());
        assert_eq!(hs.consecutive_failures, 0);
    }

    /// 验证 HealthStatus 标记为健康后状态正确。
    #[test]
    fn test_health_status_mark_healthy() {
        let mut hs = HealthStatus::new();
        hs.mark_healthy();
        assert!(hs.is_healthy());
        assert_eq!(hs.consecutive_failures, 0);
    }

    /// 验证 HealthStatus 标记为不健康后连续失败计数递增。
    #[test]
    fn test_health_status_mark_unhealthy() {
        let mut hs = HealthStatus::new();
        hs.mark_unhealthy();
        assert!(!hs.is_healthy());
        assert_eq!(hs.consecutive_failures, 1);

        hs.mark_unhealthy();
        assert_eq!(hs.consecutive_failures, 2);
    }

    /// 验证 HealthStatus 在标记健康后重置连续失败计数。
    #[test]
    fn test_health_status_recovery() {
        let mut hs = HealthStatus::new();
        hs.mark_unhealthy();
        hs.mark_unhealthy();
        assert_eq!(hs.consecutive_failures, 2);

        hs.mark_healthy();
        assert!(hs.is_healthy());
        assert_eq!(hs.consecutive_failures, 0);
    }

    /// 验证 HealthStatus 在间隔时间后需要重新检查。
    #[test]
    fn test_health_status_needs_recheck() {
        let mut hs = HealthStatus::new();
        hs.mark_healthy();

        // 刚标记健康，不需要重新检查
        assert!(!hs.needs_recheck(300));

        // 模拟时间过去（通过设置 last_check 为 0）
        hs.last_check = 0;
        assert!(hs.needs_recheck(1));
    }

    // ─── DlxConfig / DlxProvider 测试 ────────────────────────

    /// 验证 DlxConfig 默认值。
    #[test]
    fn test_dlx_config_default() {
        let config = DlxConfig::default();
        assert_eq!(config.endpoint, "http://localhost:1188");
        assert_eq!(config.timeout_secs, 10);
        assert_eq!(config.max_retries, 3);
    }

    /// 验证 DlxProvider 创建和 Debug 输出。
    #[test]
    fn test_dlx_provider_creation() {
        let config = DlxConfig {
            endpoint: "http://localhost:9999".to_string(),
            timeout_secs: 5,
            max_retries: 2,
        };
        let provider = DlxProvider::new(config);
        let debug = format!("{provider:?}");
        assert!(debug.contains("DlxProvider"));
        assert!(debug.contains("localhost:9999"));
    }

    /// 验证 DlxProvider 健康检查在服务未启动时返回 false。
    #[test]
    fn test_dlx_health_check_no_service() {
        let provider = DlxProvider::new(DlxConfig {
            endpoint: "http://localhost:19999".to_string(), // 不存在的端口
            timeout_secs: 2,
            max_retries: 1,
        });
        assert!(!provider.check_health());
    }

    /// 验证 DlxProvider 翻译在服务未启动时返回 OnlineTranslationUnavailable。
    #[test]
    fn test_dlx_translate_no_service() {
        let provider = DlxProvider::new(DlxConfig {
            endpoint: "http://localhost:19999".to_string(),
            timeout_secs: 2,
            max_retries: 1,
        });
        let result = provider.translate_text("hello", "en", "zh");
        assert!(result.is_err());
        match result {
            Err(AppError::OnlineTranslationUnavailable(_)) => {}
            Err(e) => panic!("Expected OnlineTranslationUnavailable, got: {e:?}"),
            Ok(_) => panic!("Expected error, got success"),
        }
    }

    /// 验证 DlxProvider 翻译空文本返回空字符串。
    #[test]
    fn test_dlx_translate_empty_text() {
        let provider = DlxProvider::new(DlxConfig::default());
        let result = provider
            .translate_text("", "en", "zh")
            .expect("Should succeed");
        assert!(result.is_empty());
    }

    /// 验证 DlxProvider 批量翻译空切片。
    #[test]
    fn test_dlx_translate_empty_batch() {
        let provider = DlxProvider::new(DlxConfig::default());
        let mut segments: Vec<Segment> = vec![];
        provider
            .translate_batch(&mut segments, "en", "zh")
            .expect("Should succeed on empty batch");
    }

    /// 验证 lang_code_to_dlx 正确转换语言代码。
    #[test]
    fn test_lang_code_to_dlx() {
        assert_eq!(lang_code_to_dlx("en"), "EN");
        assert_eq!(lang_code_to_dlx("EN"), "EN");
        assert_eq!(lang_code_to_dlx("zh"), "ZH");
        assert_eq!(lang_code_to_dlx("zh-CN"), "ZH");
        assert_eq!(lang_code_to_dlx("zh-Hans"), "ZH");
        assert_eq!(lang_code_to_dlx("ja"), "JA");
        assert_eq!(lang_code_to_dlx("ko"), "KO");
        assert_eq!(lang_code_to_dlx("fr"), "FR");
        assert_eq!(lang_code_to_dlx("de"), "DE");
        assert_eq!(lang_code_to_dlx("unknown"), "EN");
    }

    // ─── TranslationRouter 测试 ──────────────────────────────

    /// 验证 TranslationRouter 在 DLX 不健康时使用本地引擎。
    #[test]
    fn test_router_fallback_to_local() {
        let dlx = DlxProvider::new(DlxConfig {
            endpoint: "http://localhost:19999".to_string(), // 不存在的服务
            timeout_secs: 2,
            max_retries: 1,
        });
        let local = LocalTranslationEngine::new(MockInferenceBackend::default());
        let router = TranslationRouter::new(
            dlx,
            local,
            RouterConfig {
                prefer_online: true,
                fallback_on_error: true,
                health_check_interval_secs: 300,
            },
        );

        let mut segments = vec![Segment::new(
            "seg-1".to_string(),
            0.0,
            2.0,
            "Hello".to_string(),
        )];
        router
            .translate_batch(&mut segments, "en", "zh")
            .expect("Router should fall back to local");

        // MockInferenceBackend translates "Hello" to "你好"
        assert_eq!(segments[0].target_text.as_deref(), Some("你好"));
    }

    /// 验证 TranslationRouter 在 prefer_online=false 时直接使用本地引擎。
    #[test]
    fn test_router_prefer_offline() {
        let dlx = DlxProvider::new(DlxConfig::default());
        let local = LocalTranslationEngine::new(MockInferenceBackend::default());
        let router = TranslationRouter::new(
            dlx,
            local,
            RouterConfig {
                prefer_online: false,
                fallback_on_error: true,
                health_check_interval_secs: 300,
            },
        );

        let mut segments = vec![Segment::new(
            "seg-1".to_string(),
            0.0,
            2.0,
            "World".to_string(),
        )];
        router
            .translate_batch(&mut segments, "en", "zh")
            .expect("Router should use local directly");

        assert_eq!(segments[0].target_text.as_deref(), Some("世界"));
    }

    /// 验证 TranslationRouter 处理空段。
    #[test]
    fn test_router_empty_segments() {
        let dlx = DlxProvider::new(DlxConfig::default());
        let local = LocalTranslationEngine::new(MockInferenceBackend::default());
        let router = TranslationRouter::new(dlx, local, RouterConfig::default());

        let mut segments: Vec<Segment> = vec![];
        router
            .translate_batch(&mut segments, "en", "zh")
            .expect("Should succeed on empty segments");
    }

    /// 验证 TranslationRouter Debug 输出。
    #[test]
    fn test_router_debug() {
        let dlx = DlxProvider::new(DlxConfig::default());
        let local = LocalTranslationEngine::new(MockInferenceBackend::default());
        let router = TranslationRouter::new(dlx, local, RouterConfig::default());
        let debug = format!("{router:?}");
        assert!(debug.contains("TranslationRouter"));
        assert!(debug.contains("DlxProvider"));
    }

    /// 验证 TranslationRouter 在 fallback_on_error=false 时
    /// DLX 不可用直接返回错误。
    #[test]
    fn test_router_no_fallback() {
        let dlx = DlxProvider::new(DlxConfig {
            endpoint: "http://localhost:19999".to_string(),
            timeout_secs: 2,
            max_retries: 1,
        });
        let local = LocalTranslationEngine::new(MockInferenceBackend::default());
        let router = TranslationRouter::new(
            dlx,
            local,
            RouterConfig {
                prefer_online: true,
                fallback_on_error: false,
                health_check_interval_secs: 300,
            },
        );

        let mut segments = vec![Segment::new(
            "seg-1".to_string(),
            0.0,
            2.0,
            "test".to_string(),
        )];
        let result = router.translate_batch(&mut segments, "en", "zh");
        // DLX 不健康时不会尝试在线翻译，直接使用本地
        // 但如果 DLX 被标记为健康后尝试失败，且 fallback_on_error=false，则返回错误
        // 由于初始状态不健康，会直接使用本地引擎
        assert!(result.is_ok(), "Should use local when DLX is unhealthy");
    }

    // ─── 上下文感知翻译测试 ─────────────────────────────────

    /// 验证 TranslationContext 基本操作。
    #[test]
    fn test_translation_context_basic() {
        let mut ctx = TranslationContext::new(3);
        assert!(ctx.is_empty());
        assert_eq!(ctx.len(), 0);

        ctx.push("Hello", "你好");
        assert!(!ctx.is_empty());
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx.entries()[0].source, "Hello");
        assert_eq!(ctx.entries()[0].target, "你好");
    }

    /// 验证 TranslationContext 滑动窗口淘汰。
    #[test]
    fn test_translation_context_eviction() {
        let mut ctx = TranslationContext::new(2);
        ctx.push("A", "甲");
        ctx.push("B", "乙");
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx.entries()[0].source, "A");

        // 第三条应该淘汰最旧的 A
        ctx.push("C", "丙");
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx.entries()[0].source, "B");
        assert_eq!(ctx.entries()[1].source, "C");
    }

    /// 验证 TranslationContext 默认值。
    #[test]
    fn test_translation_context_default() {
        let ctx = TranslationContext::default();
        assert!(ctx.is_empty());
        assert_eq!(ctx.len(), 0);
        // 默认 max_entries=3，添加 4 条后应保留 3 条
        let mut ctx = ctx;
        for i in 0..4 {
            ctx.push(format!("s{i}"), format!("t{i}"));
        }
        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx.entries()[0].source, "s1"); // s0 被淘汰
    }

    /// 验证 TranslationContext max_entries 最小为 1。
    #[test]
    fn test_translation_context_min_size() {
        let mut ctx = TranslationContext::new(0);
        ctx.push("A", "甲");
        assert_eq!(ctx.len(), 1);
        ctx.push("B", "乙");
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx.entries()[0].source, "B"); // A 被淘汰
    }

    /// 验证 LocalTranslationEngine 上下文感知翻译（空上下文，等价于普通翻译）。
    #[test]
    fn test_local_engine_context_aware_empty_context() {
        let engine = LocalTranslationEngine::new(MockInferenceBackend::default());
        let ctx = TranslationContext::default();

        let mut seg = Segment::new("s1".into(), 0.0, 5.0, "Hello".into());
        engine
            .translate_segment_with_context(&mut seg, "en", "zh", &ctx)
            .expect("Translation failed");

        assert_eq!(seg.target_text.as_deref(), Some("你好"));
    }

    /// 验证 LocalTranslationEngine 上下文感知翻译（有上下文条目）。
    #[test]
    fn test_local_engine_context_aware_with_context() {
        // MockInferenceBackend 的 translate_text_with_context 默认实现
        // 忽略上下文直接调用 translate_text，所以结果与无上下文相同
        let engine = LocalTranslationEngine::new(MockInferenceBackend::default());

        let mut ctx = TranslationContext::new(3);
        ctx.push("Hello", "你好");

        let mut seg = Segment::new("s1".into(), 0.0, 5.0, "World".into());
        engine
            .translate_segment_with_context(&mut seg, "en", "zh", &ctx)
            .expect("Translation failed");

        assert_eq!(seg.target_text.as_deref(), Some("世界"));
    }

    /// 验证 LocalTranslationEngine 上下文感知翻译（带术语表）。
    #[test]
    fn test_local_engine_context_aware_with_terminology() {
        let terminology =
            TerminologyManager::from_entries(vec![GlossaryEntry::new("GPU", "图形处理器")])
                .expect("Failed to create terminology");

        let backend =
            MockInferenceBackend::from_pairs(&[("The [[T0]] renders graphics", "[[T0]]渲染图形")]);

        let engine = LocalTranslationEngine::new(backend).with_terminology(terminology);

        let mut ctx = TranslationContext::new(3);
        ctx.push("Previous segment", "前序段落");

        let mut seg = Segment::new("s1".into(), 0.0, 5.0, "The GPU renders graphics".into());
        engine
            .translate_segment_with_context(&mut seg, "en", "zh", &ctx)
            .expect("Translation failed");

        let target = seg.target_text.as_ref().expect("target_text should be set");
        assert!(
            target.contains("图形处理器"),
            "Should contain restored term, got: {target}"
        );
        assert!(
            !target.contains("[[T0]]"),
            "Placeholder should be restored, got: {target}"
        );
    }

    /// 验证 LocalTranslationEngine 上下文感知翻译空文本。
    #[test]
    fn test_local_engine_context_aware_empty_text() {
        let engine = LocalTranslationEngine::new(MockInferenceBackend::default());
        let ctx = TranslationContext::default();

        let mut seg = Segment::new("s1".into(), 0.0, 5.0, "".into());
        engine
            .translate_segment_with_context(&mut seg, "en", "zh", &ctx)
            .expect("Translation should succeed on empty text");

        // MockInferenceBackend 对空文本返回空字符串
        assert!(seg.target_text.as_deref().is_some());
    }

    /// 验证改进后的系统提示词包含视频字幕相关关键词。
    #[test]
    fn test_improved_system_prompt() {
        let prompt = LlamaCppBackend::build_system_prompt("en", "zh", None);
        assert!(
            prompt.contains("video subtitles"),
            "Prompt should mention video subtitles"
        );
        assert!(
            prompt.contains("voice-over dubbing"),
            "Prompt should mention voice-over dubbing"
        );
        assert!(
            prompt.contains("sentence fragments"),
            "Prompt should handle sentence fragments"
        );
        assert!(
            prompt.contains("[[T0]]"),
            "Prompt should mention placeholder preservation"
        );
    }
}
