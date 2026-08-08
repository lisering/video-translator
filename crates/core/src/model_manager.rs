//! 翻译模型管理模块
//!
//! 提供翻译模型的下载、缓存、完整性校验和加载功能，确保核心翻译功能
//! 在无网络环境下也能正常运行。
//!
//! # 核心设计
//! - [`ModelSource`]：区分模型来源（ModelScope 远程仓库 / 本地路径）
//! - [`ModelManager`]：管理模型生命周期（下载 → 校验 → 缓存 → 加载）
//! - SHA256 完整性校验：保证下载的模型文件未损坏或被篡改
//! - 环境变量 `VIDEO_TRANSLATOR_CACHE`：统一缓存根目录
//!
//! # 离线优先策略
//! 1. `load_model()` 首先检查本地缓存
//! 2. 若缓存命中且校验通过，直接返回路径（零网络请求）
//! 3. 若缓存未命中且 `ModelSource::Local`，返回清晰错误
//! 4. 若缓存未命中且 `ModelSource::ModelScope`，尝试下载（需网络）
//!
//! # 示例
//! ```no_run
//! use vt_core::model_manager::{ModelManager, ModelSource};
//! use vt_core::error::AppResult;
//!
//! fn load() -> AppResult<()> {
//!     let manager = ModelManager::new()?;
//!     let source = ModelSource::ModelScope {
//!         repo_id: "Qwen/Qwen2.5-3B-Instruct-GGUF".to_string(),
//!         revision: Some("master".to_string()),
//!     };
//!     let path = manager.load_model(&source, "qwen2.5-3b-instruct-q5_k_m.gguf", None)?;
//!     println!("Model loaded from: {:?}", path);
//!     Ok(())
//! }
//! ```

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AppError, AppResult};

// ─── 常量 ─────────────────────────────────────────────────

/// ModelScope 文件下载 API 基础 URL
const MODELSCOPE_API_BASE: &str = "https://modelscope.cn/api/v1/models";

/// 环境变量名：指定缓存根目录
const CACHE_ENV_VAR: &str = "VIDEO_TRANSLATOR_CACHE";

/// 默认缓存子目录名
const DEFAULT_CACHE_SUBDIR: &str = "models";

// ─── ModelType ───────────────────────────────────────────

/// 本地模型类型枚举
///
/// 标识三种核心 AI 模型，用于统一模型管理和缓存路径分配。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// 翻译模型（Hy-MT2 GGUF，通过 llama-server 子进程加载）
    Translation,
    /// 语音识别模型（Whisper GGML，通过 whisper-rs 加载）
    Whisper,
    /// 语音合成模型（Kokoro ONNX）
    Kokoro,
}

impl ModelType {
    /// 获取模型类型对应的缓存子目录名
    #[must_use]
    pub fn cache_subdir(self) -> &'static str {
        match self {
            Self::Translation => "translation",
            Self::Whisper => "whisper",
            Self::Kokoro => "kokoro",
        }
    }

    /// 获取模型类型对应的默认 ModelScope 仓库 ID
    #[must_use]
    pub fn default_repo_id(self) -> &'static str {
        match self {
            Self::Translation => "Tencent-Hunyuan/Hy-MT2-1.8B-1.25Bit-GGUF",
            Self::Whisper => "Whisper/whisper-large-v3-turbo-gguf",
            Self::Kokoro => "onnx-community/Kokoro-82M-v1.1-zh-ONNX",
        }
    }

    /// 获取模型类型对应的默认文件名
    #[must_use]
    pub fn default_filename(self) -> &'static str {
        match self {
            Self::Translation => "Hy-MT2-1.8B-1.25Bit.gguf",
            Self::Whisper => "ggml-large-v3-turbo-q5_0.bin",
            Self::Kokoro => "model.onnx",
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Translation => write!(f, "Translation"),
            Self::Whisper => write!(f, "Whisper"),
            Self::Kokoro => write!(f, "Kokoro"),
        }
    }
}

// ─── ModelSource ──────────────────────────────────────────

/// 模型来源枚举
///
/// 区分模型是从远程仓库（ModelScope）下载还是从本地路径加载。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelSource {
    /// 从 ModelScope 仓库下载
    ///
    /// `repo_id` 格式为 `org/model`，如 `Qwen/Qwen2.5-3B-Instruct-GGUF`
    ModelScope {
        /// 仓库 ID（如 `Qwen/Qwen2.5-7B-Instruct-GGUF`）
        repo_id: String,
        /// 仓库版本/分支（如 `master`），为 `None` 时使用默认分支
        revision: Option<String>,
    },
    /// 从本地文件系统路径加载
    Local {
        /// 模型文件的本地绝对或相对路径
        path: PathBuf,
    },
}

impl Default for ModelSource {
    /// 默认使用 ModelScope 下载 Hy-MT2 1.8B 1.25Bit GGUF 模型
    ///
    /// 选择依据：
    /// - Hy-MT2 是腾讯混元专门优化的英中翻译模型，1.25-bit 量化仅 466MB
    /// - 在 M1 Pro 上通过 llama-server Metal 加速，推理速度极快
    /// - 翻译质量在 IT 领域 BLEU ≥ 0.85
    /// - 作为 DeepLX 不可用时的降级方案
    fn default() -> Self {
        Self::ModelScope {
            repo_id: "Tencent-Hunyuan/Hy-MT2-1.8B-1.25Bit-GGUF".to_string(),
            revision: Some("master".to_string()),
        }
    }
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelScope { repo_id, revision } => {
                write!(f, "ModelScope({repo_id}")?;
                if let Some(rev) = revision {
                    write!(f, "@{rev}")?;
                }
                write!(f, ")")
            }
            Self::Local { path } => write!(f, "Local({})", path.display()),
        }
    }
}

// ─── ProgressCallback ────────────────────────────────────

/// 下载进度回调函数类型
///
/// 参数为 `(已下载字节数, 总字节数)`。总字节数为 0 时表示未知。
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

// ─── ModelManager ────────────────────────────────────────

/// 翻译模型管理器
///
/// 负责翻译模型文件的下载、缓存、SHA256 完整性校验和路径查找。
/// 所有操作均通过 [`AppResult`] 传递错误，不使用 `unwrap()`。
///
/// # 缓存目录
/// 缓存根目录按以下优先级确定：
/// 1. 环境变量 `VIDEO_TRANSLATOR_CACHE`
/// 2. `~/.cache/video-translator`
///
/// 模型文件存储在 `{cache_root}/translation_models/` 下。
///
/// # 线程安全
/// `ModelManager` 内部仅包含不可变字段（`PathBuf`），天然线程安全。
#[derive(Debug, Clone)]
pub struct ModelManager {
    /// 模型缓存目录
    cache_dir: PathBuf,
}

impl ModelManager {
    /// 创建默认的模型管理器
    ///
    /// 缓存目录按以下优先级确定：
    /// 1. 环境变量 `VIDEO_TRANSLATOR_CACHE`
    /// 2. `~/.cache/video-translator`
    ///
    /// # 错误
    /// - [`AppError::Config`][]: 无法确定用户主目录
    /// - [`AppError::Io`][]: 缓存目录创建失败
    pub fn new() -> AppResult<Self> {
        let cache_root = Self::resolve_cache_root()?;
        let cache_dir = cache_root.join(DEFAULT_CACHE_SUBDIR);
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    /// 使用指定的缓存目录创建模型管理器
    ///
    /// # 参数
    /// - `cache_dir`: 自定义缓存目录路径
    ///
    /// # 错误
    /// - [`AppError::Io`][]: 目录创建失败
    pub fn with_cache_dir(cache_dir: impl AsRef<Path>) -> AppResult<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    /// 获取缓存目录路径
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// 解析缓存根目录
    ///
    /// 优先级：环境变量 `VIDEO_TRANSLATOR_CACHE` > `~/.cache/video-translator`
    fn resolve_cache_root() -> AppResult<PathBuf> {
        if let Ok(path) = std::env::var(CACHE_ENV_VAR) {
            let p = PathBuf::from(path);
            return Ok(p);
        }

        let home = dirs::home_dir()
            .ok_or_else(|| AppError::Config("Cannot determine home directory".to_string()))?;
        Ok(home.join(".cache").join("video-translator"))
    }

    /// 获取模型在缓存中的存储路径
    ///
    /// 根据 `ModelSource` 和文件名计算缓存路径：
    /// - `ModelScope`: `{cache_dir}/{repo_id-sanitized}/{filename}`
    /// - `Local`: 直接返回原始路径
    ///
    /// # 参数
    /// - `source`: 模型来源
    /// - `filename`: 模型文件名（如 `Hy-MT2-1.8B-1.25Bit.gguf`）
    #[must_use]
    pub fn get_cache_path(&self, source: &ModelSource, filename: &str) -> PathBuf {
        match source {
            ModelSource::Local { path } => path.clone(),
            ModelSource::ModelScope { repo_id, .. } => {
                // 将 repo_id 中的 `/` 替换为目录分隔符
                let sanitized = repo_id.replace('/', std::path::MAIN_SEPARATOR_STR);
                self.cache_dir.join(sanitized).join(filename)
            }
        }
    }

    /// 按 `ModelType` 获取模型缓存路径
    ///
    /// 使用 `ModelType` 的默认仓库 ID 和文件名计算缓存路径。
    /// 路径格式：`{cache_dir}/{model_type_subdir}/{repo_id-sanitized}/{filename}`
    ///
    /// # 参数
    /// - `model_type`: 模型类型
    #[must_use]
    pub fn get_typed_cache_path(&self, model_type: ModelType) -> PathBuf {
        let subdir = model_type.cache_subdir();
        let repo_id = model_type.default_repo_id();
        let filename = model_type.default_filename();
        let sanitized = repo_id.replace('/', std::path::MAIN_SEPARATOR_STR);
        self.cache_dir.join(subdir).join(sanitized).join(filename)
    }

    /// 确保指定类型的模型已缓存，返回路径
    ///
    /// 离线优先：先检查缓存，未命中则从 ModelScope 下载。
    ///
    /// # 参数
    /// - `model_type`: 模型类型
    ///
    /// # 错误
    /// - [`AppError::ModelDownloadError`][]: 下载失败
    pub fn ensure_typed_model(&self, model_type: ModelType) -> AppResult<PathBuf> {
        let path = self.get_typed_cache_path(model_type);
        if path.exists() {
            tracing::debug!("{model_type} model already cached at {path:?}");
            return Ok(path);
        }

        let source = ModelSource::ModelScope {
            repo_id: model_type.default_repo_id().to_string(),
            revision: Some("master".to_string()),
        };
        let filename = model_type.default_filename();
        self.download_model(&source, filename, &path, None)?;
        Ok(path)
    }

    /// 加载模型，返回模型文件路径
    ///
    /// 离线优先策略：
    /// 1. 若 `source` 为 `Local`，验证路径存在后直接返回
    /// 2. 若 `source` 为 `ModelScope`，检查本地缓存
    /// 3. 若缓存命中且 SHA256 校验通过（若提供了校验值），返回路径
    /// 4. 若缓存未命中，尝试从 ModelScope 下载
    ///
    /// # 参数
    /// - `source`: 模型来源
    /// - `filename`: 模型文件名（如 `qwen2.5-3b-instruct-q5_k_m.gguf`）
    /// - `expected_sha256`: 期望的 SHA256 哈希值（十六进制字符串），为 `None` 时跳过校验
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 本地路径不存在
    /// - [`AppError::ModelDownloadError`][]: 下载失败
    /// - [`AppError::ModelLoadError`][]: SHA256 校验失败
    pub fn load_model(
        &self,
        source: &ModelSource,
        filename: &str,
        expected_sha256: Option<&str>,
    ) -> AppResult<PathBuf> {
        match source {
            ModelSource::Local { path } => {
                if !path.exists() {
                    return Err(AppError::FileNotFound(path.clone()));
                }
                if let Some(expected) = expected_sha256 {
                    Self::verify_model_integrity(path, expected)?;
                }
                Ok(path.clone())
            }
            ModelSource::ModelScope { .. } => {
                let cache_path = self.get_cache_path(source, filename);

                // 检查缓存
                if cache_path.exists() {
                    tracing::debug!("Model found in cache: {:?}", cache_path);
                    if let Some(expected) = expected_sha256 {
                        match Self::verify_model_integrity(&cache_path, expected) {
                            Ok(()) => return Ok(cache_path),
                            Err(e) => {
                                tracing::warn!(
                                    "Cached model failed integrity check, re-downloading: {}",
                                    e
                                );
                                // 删除损坏的缓存文件
                                let _ = std::fs::remove_file(&cache_path);
                            }
                        }
                    } else {
                        return Ok(cache_path);
                    }
                }

                // 下载模型
                self.download_model(source, filename, &cache_path, None)?;

                // 下载后校验
                if let Some(expected) = expected_sha256 {
                    Self::verify_model_integrity(&cache_path, expected)?;
                }

                Ok(cache_path)
            }
        }
    }

    /// 从 ModelScope 下载模型文件
    ///
    /// # 参数
    /// - `source`: 模型来源（必须为 `ModelScope`）
    /// - `filename`: 要下载的文件名
    /// - `dest`: 目标保存路径
    /// - `progress`: 可选的进度回调
    ///
    /// # 错误
    /// - [`AppError::Config`][]: `source` 不是 `ModelScope`
    /// - [`AppError::ModelDownloadError`][]: 下载失败
    pub fn download_model(
        &self,
        source: &ModelSource,
        filename: &str,
        dest: &Path,
        progress: Option<ProgressCallback>,
    ) -> AppResult<()> {
        let (repo_id, revision) = match source {
            ModelSource::ModelScope { repo_id, revision } => {
                (repo_id.as_str(), revision.as_deref().unwrap_or("master"))
            }
            ModelSource::Local { .. } => {
                return Err(AppError::Config(
                    "Cannot download from Local model source".to_string(),
                ));
            }
        };

        let url = build_modelscope_url(repo_id, revision, filename);
        tracing::info!("Downloading model from {url} to {dest:?}");

        // 确保目标目录存在
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // 执行 HTTP 下载
        let response = ureq::get(&url)
            .call()
            .map_err(|e| AppError::ModelDownloadError(format!("HTTP request failed: {e}")))?;

        let total_size: u64 = response
            .header("Content-Length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut file = std::fs::File::create(dest).map_err(|e| {
            AppError::ModelDownloadError(format!("Failed to create file {dest:?}: {e}"))
        })?;

        let mut reader = response.into_reader();
        let mut buffer = vec![0u8; 8192];
        let mut downloaded: u64 = 0;

        loop {
            let bytes_read = reader
                .read(&mut buffer)
                .map_err(|e| AppError::ModelDownloadError(format!("Read error: {e}")))?;

            if bytes_read == 0 {
                break;
            }

            std::io::Write::write_all(&mut file, &buffer[..bytes_read])
                .map_err(|e| AppError::ModelDownloadError(format!("Write error: {e}")))?;

            downloaded += bytes_read as u64;

            if let Some(ref cb) = progress {
                cb(downloaded, total_size);
            }
        }

        tracing::info!(
            "Model downloaded successfully: {} bytes to {:?}",
            downloaded,
            dest
        );

        Ok(())
    }

    /// 验证模型文件的 SHA256 完整性
    ///
    /// 计算文件的实际 SHA256 哈希值，并与期望值比较。
    ///
    /// # 参数
    /// - `path`: 模型文件路径
    /// - `expected_sha256`: 期望的 SHA256 哈希值（十六进制字符串，大小写不敏感）
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 文件不存在
    /// - [`AppError::ModelLoadError`][]: 哈希值不匹配
    pub fn verify_model_integrity(path: &Path, expected_sha256: &str) -> AppResult<()> {
        if !path.exists() {
            return Err(AppError::FileNotFound(path.to_path_buf()));
        }

        let actual = compute_sha256(path)?;
        let expected_lower = expected_sha256.to_lowercase();

        if actual != expected_lower {
            return Err(AppError::ModelLoadError(format!(
                "SHA256 mismatch for {:?}: expected {}, got {}",
                path, expected_lower, actual
            )));
        }

        tracing::debug!("SHA256 verification passed for {:?}", path);
        Ok(())
    }

    /// 检查模型是否已缓存
    ///
    /// # 参数
    /// - `source`: 模型来源
    /// - `filename`: 模型文件名
    #[must_use]
    pub fn is_model_cached(&self, source: &ModelSource, filename: &str) -> bool {
        self.get_cache_path(source, filename).exists()
    }

    /// 计算模型文件的 SHA256 哈希值
    ///
    /// 返回十六进制小写字符串。
    ///
    /// # 参数
    /// - `source`: 模型来源
    /// - `filename`: 模型文件名
    ///
    /// # 错误
    /// - [`AppError::FileNotFound`][]: 文件不存在
    /// - [`AppError::Io`][]: 文件读取失败
    pub fn compute_model_hash(&self, source: &ModelSource, filename: &str) -> AppResult<String> {
        let path = self.get_cache_path(source, filename);
        compute_sha256(&path)
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            cache_dir: PathBuf::from("/tmp/video-translator/models"),
        })
    }
}

// ─── 辅助函数 ─────────────────────────────────────────────

/// 构建 ModelScope 文件下载 URL
fn build_modelscope_url(repo_id: &str, revision: &str, filename: &str) -> String {
    format!("{MODELSCOPE_API_BASE}/{repo_id}/repo?Revision={revision}&FilePath={filename}")
}

/// 计算文件的 SHA256 哈希值
///
/// 返回十六进制小写字符串。
///
/// # 错误
/// - [`AppError::FileNotFound`][]: 文件不存在
/// - [`AppError::Io`][]: 文件读取失败
fn compute_sha256(path: &Path) -> AppResult<String> {
    if !path.exists() {
        return Err(AppError::FileNotFound(path.to_path_buf()));
    }

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 65536];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    // 手动转换为十六进制字符串，避免引入 hex 依赖
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    Ok(hex)
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModelSource 测试 ──

    #[test]
    fn test_model_source_default() {
        let source = ModelSource::default();
        match source {
            ModelSource::ModelScope { repo_id, revision } => {
                assert!(repo_id.contains("Hunyuan"));
                assert!(repo_id.contains("Hy-MT2"));
                assert!(revision.is_some());
            }
            ModelSource::Local { .. } => panic!("Default should be ModelScope"),
        }
    }

    #[test]
    fn test_model_source_display() {
        let ms = ModelSource::ModelScope {
            repo_id: "org/model".to_string(),
            revision: Some("v1".to_string()),
        };
        assert_eq!(ms.to_string(), "ModelScope(org/model@v1)");

        let ms_no_rev = ModelSource::ModelScope {
            repo_id: "org/model".to_string(),
            revision: None,
        };
        assert_eq!(ms_no_rev.to_string(), "ModelScope(org/model)");

        let local = ModelSource::Local {
            path: PathBuf::from("/models/test.gguf"),
        };
        assert_eq!(local.to_string(), "Local(/models/test.gguf)");
    }

    #[test]
    fn test_model_source_serde_roundtrip() {
        let source = ModelSource::ModelScope {
            repo_id: "org/model".to_string(),
            revision: Some("master".to_string()),
        };
        let json = serde_json::to_string(&source).expect("Failed to serialize");
        let deserialized: ModelSource = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(source, deserialized);

        let local = ModelSource::Local {
            path: PathBuf::from("/path/to/model.gguf"),
        };
        let json = serde_json::to_string(&local).expect("Failed to serialize");
        let deserialized: ModelSource = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(local, deserialized);
    }

    // ── ModelManager 创建测试 ──

    #[test]
    fn test_model_manager_with_cache_dir() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");
        assert_eq!(manager.cache_dir(), dir.path());
    }

    #[test]
    fn test_model_manager_creates_cache_dir() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let cache_path = dir.path().join("nested").join("cache");
        let manager = ModelManager::with_cache_dir(&cache_path).expect("Failed to create manager");
        assert!(cache_path.exists());
        assert_eq!(manager.cache_dir(), cache_path);
    }

    // ── 缓存路径测试 ──

    #[test]
    fn test_get_cache_path_modelscope() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");

        let source = ModelSource::ModelScope {
            repo_id: "org/model".to_string(),
            revision: Some("master".to_string()),
        };
        let path = manager.get_cache_path(&source, "model.gguf");

        assert!(path.starts_with(dir.path()));
        assert!(path.to_string_lossy().contains("org"));
        assert!(path.to_string_lossy().contains("model"));
        assert!(path.to_string_lossy().contains("model.gguf"));
    }

    #[test]
    fn test_get_cache_path_local() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");

        let local_path = PathBuf::from("/custom/path/model.gguf");
        let source = ModelSource::Local {
            path: local_path.clone(),
        };
        let path = manager.get_cache_path(&source, "ignored.gguf");

        assert_eq!(path, local_path);
    }

    // ── SHA256 校验测试 ──

    #[test]
    fn test_compute_sha256() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"hello world").expect("Failed to write file");

        let hash = compute_sha256(&file_path).expect("Failed to compute hash");
        // "hello world" 的 SHA256 哈希值
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_verify_model_integrity_success() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"hello world").expect("Failed to write file");

        let result = ModelManager::verify_model_integrity(
            &file_path,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_model_integrity_mismatch() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"hello world").expect("Failed to write file");

        let result = ModelManager::verify_model_integrity(
            &file_path,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(result.is_err());
        assert!(
            matches!(result, Err(AppError::ModelLoadError(_))),
            "Expected ModelLoadError, got {result:?}"
        );
    }

    #[test]
    fn test_verify_model_integrity_case_insensitive() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"hello world").expect("Failed to write file");

        // 大写哈希也应通过
        let result = ModelManager::verify_model_integrity(
            &file_path,
            "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_model_integrity_file_not_found() {
        let result =
            ModelManager::verify_model_integrity(Path::new("/nonexistent/file.bin"), "abc123");
        assert!(result.is_err());
        assert!(
            matches!(result, Err(AppError::FileNotFound(_))),
            "Expected FileNotFound, got {result:?}"
        );
    }

    // ── load_model (Local) 测试 ──

    #[test]
    fn test_load_model_local_success() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"fake model data").expect("Failed to write file");

        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");
        let source = ModelSource::Local {
            path: model_path.clone(),
        };

        let loaded = manager
            .load_model(&source, "ignored", None)
            .expect("Failed to load model");
        assert_eq!(loaded, model_path);
    }

    #[test]
    fn test_load_model_local_not_found() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");
        let source = ModelSource::Local {
            path: PathBuf::from("/nonexistent/model.gguf"),
        };

        let result = manager.load_model(&source, "ignored", None);
        assert!(result.is_err());
        assert!(
            matches!(result, Err(AppError::FileNotFound(_))),
            "Expected FileNotFound, got {result:?}"
        );
    }

    #[test]
    fn test_load_model_local_with_sha256() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"hello world").expect("Failed to write file");

        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");
        let source = ModelSource::Local {
            path: model_path.clone(),
        };

        let loaded = manager
            .load_model(
                &source,
                "ignored",
                Some("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"),
            )
            .expect("Failed to load model");
        assert_eq!(loaded, model_path);
    }

    // ── is_model_cached 测试 ──

    #[test]
    fn test_is_model_cached_false() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");

        let source = ModelSource::ModelScope {
            repo_id: "org/model".to_string(),
            revision: None,
        };
        assert!(!manager.is_model_cached(&source, "model.gguf"));
    }

    #[test]
    fn test_is_model_cached_true() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");

        let source = ModelSource::ModelScope {
            repo_id: "org/model".to_string(),
            revision: None,
        };
        let cache_path = manager.get_cache_path(&source, "model.gguf");
        std::fs::create_dir_all(cache_path.parent().expect("No parent"))
            .expect("Failed to create dir");
        std::fs::write(&cache_path, b"fake model").expect("Failed to write file");

        assert!(manager.is_model_cached(&source, "model.gguf"));
    }

    // ── compute_model_hash 测试 ──

    #[test]
    fn test_compute_model_hash() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");

        let source = ModelSource::ModelScope {
            repo_id: "org/model".to_string(),
            revision: None,
        };
        let cache_path = manager.get_cache_path(&source, "test.bin");
        std::fs::create_dir_all(cache_path.parent().expect("No parent"))
            .expect("Failed to create dir");
        std::fs::write(&cache_path, b"hello world").expect("Failed to write file");

        let hash = manager
            .compute_model_hash(&source, "test.bin")
            .expect("Failed to compute hash");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    // ── URL 构建测试 ──

    #[test]
    fn test_build_modelscope_url() {
        let url = build_modelscope_url("org/model", "master", "model.gguf");
        assert_eq!(
            url,
            "https://modelscope.cn/api/v1/models/org/model/repo?Revision=master&FilePath=model.gguf"
        );
    }

    // ── 环境变量缓存路径测试 ──

    #[test]
    fn test_resolve_cache_root_from_env() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        // 临时设置环境变量
        std::env::set_var(CACHE_ENV_VAR, dir.path());

        let root = ModelManager::resolve_cache_root().expect("Failed to resolve");
        assert_eq!(root, dir.path());

        // 清理
        std::env::remove_var(CACHE_ENV_VAR);
    }

    // ── download_model 错误路径测试 ──

    #[test]
    fn test_download_model_local_source_error() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");

        let source = ModelSource::Local {
            path: PathBuf::from("/local/model.gguf"),
        };
        let dest = dir.path().join("downloaded.gguf");

        let result = manager.download_model(&source, "model.gguf", &dest, None);
        assert!(result.is_err());
        assert!(
            matches!(result, Err(AppError::Config(_))),
            "Expected Config error, got {result:?}"
        );
    }

    // ─── ModelType 测试 ─────────────────────────────────────

    /// 验证 ModelType 各类型的缓存子目录名。
    #[test]
    fn test_model_type_cache_subdir() {
        assert_eq!(ModelType::Translation.cache_subdir(), "translation");
        assert_eq!(ModelType::Whisper.cache_subdir(), "whisper");
        assert_eq!(ModelType::Kokoro.cache_subdir(), "kokoro");
    }

    /// 验证 ModelType 各类型的默认 ModelScope 仓库 ID。
    #[test]
    fn test_model_type_default_repo_id() {
        assert_eq!(
            ModelType::Translation.default_repo_id(),
            "Tencent-Hunyuan/Hy-MT2-1.8B-1.25Bit-GGUF"
        );
        assert_eq!(
            ModelType::Whisper.default_repo_id(),
            "Whisper/whisper-large-v3-turbo-gguf"
        );
        assert_eq!(
            ModelType::Kokoro.default_repo_id(),
            "onnx-community/Kokoro-82M-v1.1-zh-ONNX"
        );
    }

    /// 验证 ModelType 各类型的默认文件名。
    #[test]
    fn test_model_type_default_filename() {
        assert_eq!(
            ModelType::Translation.default_filename(),
            "Hy-MT2-1.8B-1.25Bit.gguf"
        );
        assert_eq!(
            ModelType::Whisper.default_filename(),
            "ggml-large-v3-turbo-q5_0.bin"
        );
        assert_eq!(ModelType::Kokoro.default_filename(), "model.onnx");
    }

    /// 验证 ModelType Display trait 实现。
    #[test]
    fn test_model_type_display() {
        assert_eq!(format!("{}", ModelType::Translation), "Translation");
        assert_eq!(format!("{}", ModelType::Whisper), "Whisper");
        assert_eq!(format!("{}", ModelType::Kokoro), "Kokoro");
    }

    /// 验证 ModelType 序列化/反序列化。
    #[test]
    fn test_model_type_serde() {
        let json = serde_json::to_string(&ModelType::Translation).expect("Serialize failed");
        let decoded: ModelType = serde_json::from_str(&json).expect("Deserialize failed");
        assert_eq!(decoded, ModelType::Translation);
    }

    /// 验证 get_typed_cache_path 生成正确的路径。
    #[test]
    fn test_get_typed_cache_path() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");

        let path = manager.get_typed_cache_path(ModelType::Translation);
        assert!(path.to_string_lossy().contains("translation"));
        assert!(path.to_string_lossy().contains("Hy-MT2"));
        assert!(path.to_string_lossy().ends_with("Hy-MT2-1.8B-1.25Bit.gguf"));

        let path = manager.get_typed_cache_path(ModelType::Whisper);
        assert!(path.to_string_lossy().contains("whisper"));
        assert!(path
            .to_string_lossy()
            .ends_with("ggml-large-v3-turbo-q5_0.bin"));

        let path = manager.get_typed_cache_path(ModelType::Kokoro);
        assert!(path.to_string_lossy().contains("kokoro"));
        assert!(path.to_string_lossy().ends_with("model.onnx"));
    }
}
