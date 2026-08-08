//! 声纹增强模块（ERes2NetV2 Speaker Verification）
//!
//! 借鉴 GPT-SoVITS v2Pro 的 ERes2NetV2 说话人验证模型，
//! 通过 20480 维声纹向量增强声音克隆的说话人保真度。
//!
//! # 核心原理
//! GPT-SoVITS v2Pro 在标准 MelStyleEncoder 之外，
//! 额外使用 ERes2NetV2 提取 20480 维声纹向量（speaker embedding），
//! 通过 `sv_emb = Linear(20480 → gin_channels)` 投影到模型维度后融合。
//!
//! 这提供了更强的说话人身份保持能力：
//! - MelStyleEncoder：捕捉频谱风格（音色、语调）
//! - ERes2NetV2：捕捉说话人身份（who is speaking）
//! - 两者互补：即使频谱风格因合成失真，身份信息仍可保持
//!
//! # 工作流程
//! 1. 参考音频重采样到 16kHz（ERes2NetV2 要求）
//! 2. 提取 80 维 Fbank 特征（25ms 窗，10ms 步长）
//! 3. ERes2NetV2 前向传播 → 20480 维声纹向量
//! 4. L2 归一化
//! 5. 缓存声纹向量（同一参考音频复用）
//! 6. 传递给 TTS 模型作为补充条件
//!
//! # 模块结构
//! - [`SpeakerEmbedding`]: 20480 维声纹向量
//! - [`SpeakerEmbeddingManager`]: 声纹缓存与管理
//! - [`SpeakerSimilarity`]: 说话人相似度计算
//! - [`SpeakerVerificationConfig`]: 配置
//!
//! # 示例
//! ```
//! use vt_core::speaker_embedding::{SpeakerEmbedding, SpeakerEmbeddingManager};
//! use std::path::Path;
//!
//! // 假设从 Python server 返回的 20480 维向量
//! let raw_embedding = vec![0.1f32; 20480];
//! let embedding = SpeakerEmbedding::from_vector(raw_embedding);
//! assert_eq!(embedding.dim(), 20480);
//!
//! // 缓存管理
//! let mut manager = SpeakerEmbeddingManager::new();
//! let ref_path = Path::new("reference_01.wav");
//! manager.cache(ref_path, embedding);
//! assert!(manager.get(ref_path).is_some());
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ─── 常量 ─────────────────────────────────────────────────

/// ERes2NetV2 输出维度
pub const SPEAKER_EMBEDDING_DIM: usize = 20480;

/// ERes2NetV2 输入采样率（16kHz）
pub const SPEAKER_MODEL_SAMPLE_RATE: u32 = 16000;

/// Fbank 特征维度
pub const FBANK_DIM: usize = 80;

/// Fbank 窗长（ms）
pub const FBANK_WIN_MS: f64 = 25.0;

/// Fbank 步长（ms）
pub const FBANK_HOP_MS: f64 = 10.0;

/// 默认相似度阈值（高于此值认为是同一说话人）
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.75;

// ─── 声纹向量 ────────────────────────────────────────────

/// 说话人声纹向量
///
/// 20480 维 L2 归一化向量，来自 ERes2NetV2 模型。
/// 用于声音克隆时增强说话人身份保持。
#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    /// 声纹向量（L2 归一化）
    vector: Vec<f32>,
    /// 来源参考音频路径
    source_path: Option<PathBuf>,
    /// 向量维度（通常 20480）
    dim: usize,
}

impl SpeakerEmbedding {
    /// 从原始向量创建声纹（自动 L2 归一化）
    #[must_use]
    pub fn from_vector(vector: Vec<f32>) -> Self {
        let dim = vector.len();
        let normalized = l2_normalize(&vector);
        Self {
            vector: normalized,
            source_path: None,
            dim,
        }
    }

    /// 从原始向量创建声纹，并记录来源路径
    #[must_use]
    pub fn from_vector_with_source(vector: Vec<f32>, source: PathBuf) -> Self {
        let dim = vector.len();
        let normalized = l2_normalize(&vector);
        Self {
            vector: normalized,
            source_path: Some(source),
            dim,
        }
    }

    /// 获取声纹向量引用
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// 获取向量维度
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 获取来源路径
    #[must_use]
    pub fn source(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// 计算与另一个声纹的余弦相似度
    ///
    /// # 返回
    /// 相似度值 (-1.0 到 1.0，1.0 = 完全相同)
    #[must_use]
    pub fn cosine_similarity(&self, other: &SpeakerEmbedding) -> f32 {
        if self.dim != other.dim {
            return 0.0;
        }

        // 由于已经 L2 归一化，余弦相似度 = 点积
        dot_product(&self.vector, &other.vector)
    }

    /// 判断是否为同一说话人
    ///
    /// # 参数
    /// - `other`: 另一个声纹
    /// - `threshold`: 相似度阈值（默认 0.75）
    #[must_use]
    pub fn is_same_speaker(&self, other: &SpeakerEmbedding, threshold: f32) -> bool {
        self.cosine_similarity(other) >= threshold
    }

    /// 判断是否为同一说话人（使用默认阈值）
    #[must_use]
    pub fn is_same_speaker_default(&self, other: &SpeakerEmbedding) -> bool {
        self.is_same_speaker(other, DEFAULT_SIMILARITY_THRESHOLD)
    }

    /// 序列化为 JSON 格式（用于与 Python server 通信）
    #[must_use]
    pub fn to_json_vector(&self) -> String {
        // 精简的 JSON 数组序列化（避免引入 serde 依赖）
        let inner: Vec<String> = self.vector.iter().map(|v| format!("{v:.6}")).collect();
        format!("[{}]", inner.join(","))
    }

    /// 从 JSON 格式反序列化
    ///
    /// # 参数
    /// - `json`: JSON 数组字符串，如 "[0.1, 0.2, ...]"
    ///
    /// # 返回
    /// 解析后的声纹向量，或 `None` 如果格式无效
    #[must_use]
    pub fn from_json_vector(json: &str) -> Option<Self> {
        let trimmed = json.trim().trim_start_matches('[').trim_end_matches(']');
        let vector: Vec<f32> = trimmed
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();

        if vector.is_empty() {
            None
        } else {
            Some(Self::from_vector(vector))
        }
    }
}

// ─── 声纹管理器 ──────────────────────────────────────────

/// 声纹缓存与管理器
///
/// 管理多个参考音频的声纹向量，提供：
/// - 缓存：同一参考音频的声纹只提取一次
/// - 检索：按路径获取缓存的声纹
/// - 比较：比较不同参考音频的说话人相似度
/// - 选择：选择与目标说话人最相似的参考
pub struct SpeakerEmbeddingManager {
    /// 缓存的声纹：path → embedding
    cache: HashMap<PathBuf, SpeakerEmbedding>,
    /// 主声纹（最佳参考的声纹）
    primary: Option<SpeakerEmbedding>,
    /// 配置
    config: SpeakerVerificationConfig,
}

/// 声纹验证配置
#[derive(Debug, Clone)]
pub struct SpeakerVerificationConfig {
    /// 是否启用声纹增强
    pub enabled: bool,
    /// ERes2NetV2 模型路径
    pub model_path: Option<PathBuf>,
    /// 相似度阈值
    pub similarity_threshold: f32,
    /// 是否在 Python server 中使用声纹
    pub use_in_server: bool,
}

impl Default for SpeakerVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            use_in_server: false,
        }
    }
}

impl SpeakerEmbeddingManager {
    /// 创建声纹管理器
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            primary: None,
            config: SpeakerVerificationConfig::default(),
        }
    }

    /// 使用指定配置创建
    #[must_use]
    pub fn with_config(config: SpeakerVerificationConfig) -> Self {
        Self {
            cache: HashMap::new(),
            primary: None,
            config,
        }
    }

    /// 缓存声纹向量
    ///
    /// # 参数
    /// - `ref_path`: 参考音频路径
    /// - `embedding`: 声纹向量
    pub fn cache(&mut self, ref_path: &Path, embedding: SpeakerEmbedding) {
        if self.primary.is_none() {
            self.primary = Some(embedding.clone());
        }
        self.cache.insert(ref_path.to_path_buf(), embedding);
    }

    /// 获取缓存的声纹
    #[must_use]
    pub fn get(&self, ref_path: &Path) -> Option<&SpeakerEmbedding> {
        self.cache.get(ref_path)
    }

    /// 获取主声纹（第一个缓存的声纹）
    #[must_use]
    pub fn primary(&self) -> Option<&SpeakerEmbedding> {
        self.primary.as_ref()
    }

    /// 获取所有缓存的声纹
    #[must_use]
    pub fn all_embeddings(&self) -> &HashMap<PathBuf, SpeakerEmbedding> {
        &self.cache
    }

    /// 缓存数量
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// 是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// 找到与目标声纹最相似的参考音频
    ///
    /// # 参数
    /// - `target`: 目标说话人声纹
    ///
    /// # 返回
    /// `(最相似的参考路径, 相似度)` 或 `None`
    #[must_use]
    pub fn find_most_similar(&self, target: &SpeakerEmbedding) -> Option<(&Path, f32)> {
        self.cache
            .iter()
            .map(|(path, emb)| (path.as_path(), target.cosine_similarity(emb)))
            .max_by(|(_, sim_a), (_, sim_b)| {
                sim_a
                    .partial_cmp(sim_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// 验证两个参考音频是否为同一说话人
    ///
    /// # 参数
    /// - `path_a`: 参考音频 A 路径
    /// - `path_b`: 参考音频 B 路径
    #[must_use]
    pub fn verify_same_speaker(&self, path_a: &Path, path_b: &Path) -> Option<bool> {
        let emb_a = self.cache.get(path_a)?;
        let emb_b = self.cache.get(path_b)?;
        Some(emb_a.is_same_speaker_default(emb_b))
    }

    /// 清空缓存
    pub fn clear(&mut self) {
        self.cache.clear();
        self.primary = None;
    }

    /// 获取配置
    #[must_use]
    pub fn config(&self) -> &SpeakerVerificationConfig {
        &self.config
    }
}

impl Default for SpeakerEmbeddingManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── 说话人相似度计算 ────────────────────────────────────

/// 说话人相似度计算工具
pub struct SpeakerSimilarity;

impl SpeakerSimilarity {
    /// 计算两个向量的余弦相似度
    ///
    /// `cosine_sim(a, b) = (a · b) / (|a| * |b|)`
    #[must_use]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot = dot_product(a, b);
        let norm_a = l2_norm(a);
        let norm_b = l2_norm(b);

        if norm_a < 1e-8 || norm_b < 1e-8 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// 计算欧氏距离
    #[must_use]
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// 批量计算相似度矩阵
    ///
    /// # 参数
    /// - `embeddings`: 声纹向量列表
    ///
    /// # 返回
    /// `N×N` 相似度矩阵，`matrix[i][j]` = embeddings[i] 与 embeddings[j] 的余弦相似度
    #[must_use]
    pub fn similarity_matrix(embeddings: &[&[f32]]) -> Vec<Vec<f32>> {
        let n = embeddings.len();
        let mut matrix = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in i..n {
                let sim = Self::cosine_similarity(embeddings[i], embeddings[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }

        matrix
    }
}

// ─── 辅助函数 ────────────────────────────────────────────

/// L2 归一化
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm < 1e-8 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// L2 范数
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// 点积
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Fbank 特征提取参数 ──────────────────────────────────

/// Fbank 特征提取参数
///
/// 对应 ERes2NetV2 的输入预处理：
/// - 16kHz 采样率
/// - 80 维 Mel Fbank
/// - 25ms 窗长，10ms 步长
#[derive(Debug, Clone)]
pub struct FbankConfig {
    /// 采样率
    pub sample_rate: u32,
    /// Fbank 维度
    pub n_mels: usize,
    /// 窗长（采样数）
    pub win_length: usize,
    /// 步长（采样数）
    pub hop_length: usize,
    /// FFT 大小
    pub n_fft: usize,
    /// 频率下限
    pub f_min: f32,
    /// 频率上限
    pub f_max: f32,
}

impl Default for FbankConfig {
    fn default() -> Self {
        let sample_rate = SPEAKER_MODEL_SAMPLE_RATE;
        let win_length = ((FBANK_WIN_MS / 1000.0) * sample_rate as f64) as usize;
        let hop_length = ((FBANK_HOP_MS / 1000.0) * sample_rate as f64) as usize;
        Self {
            sample_rate,
            n_mels: FBANK_DIM,
            win_length,
            hop_length,
            n_fft: 512,
            f_min: 0.0,
            f_max: sample_rate as f32 / 2.0,
        }
    }
}

/// 计算参考音频的理论 Fbank 帧数
///
/// `num_frames = (num_samples - win_length) / hop_length + 1`
#[must_use]
pub fn estimate_fbank_frames(num_samples: usize, config: &FbankConfig) -> usize {
    if num_samples < config.win_length {
        return 0;
    }
    (num_samples - config.win_length) / config.hop_length + 1
}

/// 计算参考音频的理论声纹提取时长（秒）
///
/// ERes2NetV2 前向传播在 CPU 上约 50-100ms（1-5s 音频）
#[must_use]
pub fn estimate_extraction_time(audio_duration_secs: f64) -> f64 {
    // 基准: 5s 音频 ~80ms
    // 线性扩展
    0.08 * (audio_duration_secs / 5.0).max(0.2)
}

// ─── Python server 通信协议 ──────────────────────────────

/// 声纹提取请求（发送给 Python server）
#[derive(Debug, Clone)]
pub struct SpeakerExtractionRequest {
    /// 参考音频路径
    pub ref_audio_path: String,
    /// 是否使用 ERes2NetV2 增强模式
    pub use_eres2net: bool,
}

impl SpeakerExtractionRequest {
    /// 序列化为 JSON（用于 stdin 通信）
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"action":"extract_speaker_embedding","ref_audio":"{}","use_eres2net":{}}}"#,
            self.ref_audio_path, self.use_eres2net
        )
    }
}

/// 声纹提取响应（从 Python server 接收）
#[derive(Debug, Clone)]
pub struct SpeakerExtractionResponse {
    /// 状态: "ok" 或 "error"
    pub status: String,
    /// 声纹向量（如果成功）
    pub embedding: Option<Vec<f32>>,
    /// 维度
    pub dim: usize,
    /// 错误信息（如果失败）
    pub error: Option<String>,
    /// 耗时（秒）
    pub elapsed_secs: f64,
}

impl SpeakerExtractionResponse {
    /// 从 JSON 解析响应
    ///
    /// # 参数
    /// - `json`: JSON 字符串
    #[must_use]
    pub fn from_json(json: &str) -> Option<Self> {
        // 简单 JSON 解析（避免引入 serde 依赖）
        let has_ok = json.contains("\"status\":\"ok\"") || json.contains("\"status\": \"ok\"");
        let has_error =
            json.contains("\"status\":\"error\"") || json.contains("\"status\": \"error\"");

        if has_error {
            let error_msg = extract_json_string(json, "error").unwrap_or_default();
            return Some(Self {
                status: "error".to_string(),
                embedding: None,
                dim: 0,
                error: Some(error_msg),
                elapsed_secs: 0.0,
            });
        }

        if has_ok {
            // 尝试提取 embedding 数组
            let embedding = extract_json_array(json, "embedding");
            let dim = embedding.as_ref().map(|v| v.len()).unwrap_or(0);
            let elapsed = extract_json_number(json, "elapsed_secs").unwrap_or(0.0);

            return Some(Self {
                status: "ok".to_string(),
                embedding,
                dim,
                error: None,
                elapsed_secs: elapsed,
            });
        }

        None
    }
}

// ─── 简易 JSON 解析辅助 ──────────────────────────────────

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":\"");
    let start = json.find(&pattern)? + pattern.len();
    let end = json[start..].find('"')?;
    Some(json[start..start + end].to_string())
}

fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\":");
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
    rest[..end].parse().ok()
}

fn extract_json_array(json: &str, key: &str) -> Option<Vec<f32>> {
    let pattern = format!("\"{key}\":[");
    let start = json.find(&pattern)? + pattern.len();
    let end = json[start..].find(']')?;
    let array_str = &json[start..start + end];
    let result: Vec<f32> = array_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── SpeakerEmbedding 测试 ─────────────────────────

    #[test]
    fn test_embedding_creation() {
        let raw = vec![3.0f32, 4.0]; // |v| = 5
        let emb = SpeakerEmbedding::from_vector(raw);
        assert_eq!(emb.dim(), 2);
        // L2 归一化后: [0.6, 0.8]
        assert!((emb.vector()[0] - 0.6).abs() < 1e-5);
        assert!((emb.vector()[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_zero_vector() {
        let raw = vec![0.0f32; 20480];
        let emb = SpeakerEmbedding::from_vector(raw);
        assert_eq!(emb.dim(), 20480);
        // 零向量归一化后仍为 0
        assert!(emb.vector().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_embedding_with_source() {
        let raw = vec![1.0f32; 20480];
        let path = PathBuf::from("/tmp/ref.wav");
        let emb = SpeakerEmbedding::from_vector_with_source(raw, path.clone());
        assert_eq!(emb.source(), Some(path.as_path()));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let raw = vec![1.0f32, 2.0, 3.0, 4.0];
        let emb1 = SpeakerEmbedding::from_vector(raw.clone());
        let emb2 = SpeakerEmbedding::from_vector(raw);
        let sim = emb1.cosine_similarity(&emb2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical embeddings should have similarity 1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let emb1 = SpeakerEmbedding::from_vector(vec![1.0f32, 0.0]);
        let emb2 = SpeakerEmbedding::from_vector(vec![0.0f32, 1.0]);
        let sim = emb1.cosine_similarity(&emb2);
        assert!(
            sim.abs() < 1e-5,
            "Orthogonal embeddings should have similarity 0"
        );
    }

    #[test]
    fn test_cosine_similarity_different_dim() {
        let emb1 = SpeakerEmbedding::from_vector(vec![1.0f32, 2.0, 3.0]);
        let emb2 = SpeakerEmbedding::from_vector(vec![1.0f32, 2.0]);
        let sim = emb1.cosine_similarity(&emb2);
        assert_eq!(sim, 0.0, "Different dimensions should give 0 similarity");
    }

    #[test]
    fn test_is_same_speaker() {
        let raw1 = vec![1.0f32; 20480];
        let raw2 = vec![1.0f32; 20480];
        let emb1 = SpeakerEmbedding::from_vector(raw1);
        let emb2 = SpeakerEmbedding::from_vector(raw2);
        assert!(emb1.is_same_speaker_default(&emb2));
    }

    #[test]
    fn test_is_different_speaker() {
        let raw1 = vec![1.0f32; 20480];
        let mut raw2 = vec![0.0f32; 20480];
        raw2[0] = 1.0; // 正交
        let emb1 = SpeakerEmbedding::from_vector(raw1);
        let emb2 = SpeakerEmbedding::from_vector(raw2);
        assert!(!emb1.is_same_speaker_default(&emb2));
    }

    #[test]
    fn test_json_roundtrip() {
        let raw: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let emb = SpeakerEmbedding::from_vector(raw);
        let json = emb.to_json_vector();
        let restored = SpeakerEmbedding::from_json_vector(&json).unwrap();
        assert_eq!(restored.dim(), emb.dim());
        for i in 0..emb.dim() {
            assert!((emb.vector()[i] - restored.vector()[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_json_invalid_input() {
        assert!(SpeakerEmbedding::from_json_vector("").is_none());
        assert!(SpeakerEmbedding::from_json_vector("[]").is_none());
        assert!(SpeakerEmbedding::from_json_vector("invalid").is_none());
    }

    // ─── SpeakerEmbeddingManager 测试 ──────────────────

    #[test]
    fn test_manager_empty() {
        let manager = SpeakerEmbeddingManager::new();
        assert!(manager.is_empty());
        assert!(manager.primary().is_none());
    }

    #[test]
    fn test_manager_cache_and_get() {
        let mut manager = SpeakerEmbeddingManager::new();
        let path = PathBuf::from("/tmp/ref.wav");
        let emb = SpeakerEmbedding::from_vector(vec![1.0f32; 20480]);
        manager.cache(&path, emb);

        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 1);
        assert!(manager.get(&path).is_some());
        assert!(manager.primary().is_some());
    }

    #[test]
    fn test_manager_find_most_similar() {
        let mut manager = SpeakerEmbeddingManager::new();

        // 缓存两个参考
        let path_a = PathBuf::from("/tmp/ref_a.wav");
        let path_b = PathBuf::from("/tmp/ref_b.wav");
        let emb_a = SpeakerEmbedding::from_vector(vec![1.0f32; 20480]);
        let emb_b = SpeakerEmbedding::from_vector({
            let mut v = vec![0.0f32; 20480];
            v[0] = 1.0;
            v
        });
        manager.cache(&path_a, emb_a);
        manager.cache(&path_b, emb_b);

        // 目标与 ref_a 相同
        let target = SpeakerEmbedding::from_vector(vec![1.0f32; 20480]);
        let (best_path, sim) = manager.find_most_similar(&target).unwrap();
        assert_eq!(best_path, path_a.as_path());
        // 20480 维 f32 浮点累积误差，放宽到 1e-3
        assert!(
            (sim - 1.0).abs() < 1e-3,
            "Similarity should be ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_manager_verify_same_speaker() {
        let mut manager = SpeakerEmbeddingManager::new();
        let path_a = PathBuf::from("/tmp/ref_a.wav");
        let path_b = PathBuf::from("/tmp/ref_b.wav");

        manager.cache(&path_a, SpeakerEmbedding::from_vector(vec![1.0f32; 20480]));
        manager.cache(&path_b, SpeakerEmbedding::from_vector(vec![1.0f32; 20480]));

        assert_eq!(manager.verify_same_speaker(&path_a, &path_b), Some(true));
    }

    #[test]
    fn test_manager_clear() {
        let mut manager = SpeakerEmbeddingManager::new();
        manager.cache(
            &PathBuf::from("/tmp/ref.wav"),
            SpeakerEmbedding::from_vector(vec![1.0f32; 20480]),
        );
        assert!(!manager.is_empty());

        manager.clear();
        assert!(manager.is_empty());
    }

    // ─── SpeakerSimilarity 测试 ────────────────────────

    #[test]
    fn test_cosine_similarity_direct() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let sim = SpeakerSimilarity::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_different_length() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0];
        let sim = SpeakerSimilarity::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![3.0f32, 4.0, 0.0];
        let dist = SpeakerSimilarity::euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_similarity_matrix() {
        let v0 = vec![1.0f32, 0.0, 0.0];
        let v1 = vec![0.0f32, 1.0, 0.0];
        let v2 = vec![1.0f32, 0.0, 0.0];
        let embeddings = vec![v0.as_slice(), v1.as_slice(), v2.as_slice()];
        let matrix = SpeakerSimilarity::similarity_matrix(&embeddings);
        assert_eq!(matrix.len(), 3);
        // 0 和 2 相同
        assert!((matrix[0][2] - 1.0).abs() < 1e-5);
        // 0 和 1 正交
        assert!(matrix[0][1].abs() < 1e-5);
        // 对角线为 1
        assert!((matrix[1][1] - 1.0).abs() < 1e-5);
    }

    // ─── FbankConfig 测试 ──────────────────────────────

    #[test]
    fn test_fbank_config_default() {
        let config = FbankConfig::default();
        assert_eq!(config.sample_rate, SPEAKER_MODEL_SAMPLE_RATE);
        assert_eq!(config.n_mels, FBANK_DIM);
        // 25ms @ 16kHz = 400 samples
        assert_eq!(config.win_length, 400);
        // 10ms @ 16kHz = 160 samples
        assert_eq!(config.hop_length, 160);
    }

    #[test]
    fn test_estimate_fbank_frames() {
        let config = FbankConfig::default();
        // 5 秒音频 = 80000 采样
        let frames = estimate_fbank_frames(80000, &config);
        // (80000 - 400) / 160 + 1 = 498
        assert!(frames > 490 && frames < 510);
    }

    #[test]
    fn test_estimate_fbank_frames_short() {
        let config = FbankConfig::default();
        // 100 采样 < win_length(400) → 0 frames
        let frames = estimate_fbank_frames(100, &config);
        assert_eq!(frames, 0);
    }

    #[test]
    fn test_estimate_extraction_time() {
        let time = estimate_extraction_time(5.0);
        assert!(time > 0.0 && time < 1.0);
    }

    // ─── 通信协议测试 ──────────────────────────────────

    #[test]
    fn test_extraction_request_json() {
        let req = SpeakerExtractionRequest {
            ref_audio_path: "/tmp/ref.wav".to_string(),
            use_eres2net: true,
        };
        let json = req.to_json();
        assert!(json.contains("extract_speaker_embedding"));
        assert!(json.contains("/tmp/ref.wav"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_extraction_response_ok() {
        let json = r#"{"status":"ok","embedding":[0.1,0.2,0.3],"dim":3,"elapsed_secs":0.08}"#;
        let resp = SpeakerExtractionResponse::from_json(json).unwrap();
        assert_eq!(resp.status, "ok");
        assert!(resp.embedding.is_some());
        assert_eq!(resp.dim, 3);
        assert!((resp.elapsed_secs - 0.08).abs() < 1e-5);
    }

    #[test]
    fn test_extraction_response_error() {
        let json = r#"{"status":"error","error":"model not found"}"#;
        let resp = SpeakerExtractionResponse::from_json(json).unwrap();
        assert_eq!(resp.status, "error");
        assert!(resp.embedding.is_none());
        assert_eq!(resp.error.as_deref(), Some("model not found"));
    }

    #[test]
    fn test_extraction_response_invalid() {
        assert!(SpeakerExtractionResponse::from_json("invalid").is_none());
    }

    // ─── 配置测试 ──────────────────────────────────────

    #[test]
    fn test_speaker_verification_config_default() {
        let config = SpeakerVerificationConfig::default();
        assert!(!config.enabled);
        assert!(config.model_path.is_none());
        assert_eq!(config.similarity_threshold, DEFAULT_SIMILARITY_THRESHOLD);
    }

    #[test]
    fn test_speaker_verification_config_custom() {
        let config = SpeakerVerificationConfig {
            enabled: true,
            model_path: Some(PathBuf::from("/models/eres2net")),
            similarity_threshold: 0.8,
            use_in_server: true,
        };
        assert!(config.enabled);
        assert_eq!(config.similarity_threshold, 0.8);
    }
}
