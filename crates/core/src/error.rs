//! 统一错误类型定义
//!
//! 使用 [`thiserror`] 定义应用级错误枚举 [`AppError`]，
//! 所有库代码通过 [`AppResult<T>`] 进行错误传播，禁止使用 `unwrap()` / `expect()`。

use thiserror::Error;

/// 应用统一错误类型
///
/// 涵盖 IO、序列化、配置解析、模型查找、状态机转换、
/// FFmpeg 命令执行、媒体探测、文件不存在、格式不支持、
/// Whisper 模型加载、转录处理、音频解码、模型下载等场景。
/// 通过 `#[from]` 自动实现 `From` 转换，支持 `?` 运算符错误传播。
#[derive(Debug, Error)]
pub enum AppError {
    /// IO 错误（文件读写等）
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON 序列化/反序列化错误
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// 配置解析或校验错误
    #[error("Configuration error: {0}")]
    Config(String),

    /// 模型未找到错误
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// 非法状态机转换错误
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),

    /// TOML 反序列化错误
    #[error("TOML deserialization error: {0}")]
    TomlDe(#[from] toml::de::Error),

    /// 来自 `anyhow` 的通用错误
    #[error("Application error: {0}")]
    Anyhow(#[from] anyhow::Error),

    /// FFmpeg 命令执行失败（非零退出码或参数错误）
    #[error("FFmpeg error: {0}")]
    FFmpeg(String),

    /// 媒体探测（ffprobe）失败
    #[error("Media probe error: {0}")]
    MediaProbe(String),

    /// 输入文件不存在
    #[error("File not found: {0}")]
    FileNotFound(std::path::PathBuf),

    /// 不支持的媒体格式或编解码器
    #[error("Unsupported media format: {0}")]
    UnsupportedFormat(String),

    /// Whisper 模型加载失败
    #[error("Model load error: {0}")]
    ModelLoadError(String),

    /// 语音转录失败
    #[error("Transcription error: {0}")]
    TranscriptionError(String),

    /// 音频解码失败（WAV 读取、格式不符等）
    #[error("Audio decode error: {0}")]
    AudioDecodeError(String),

    /// 模型下载失败
    #[error("Model download error: {0}")]
    ModelDownloadError(String),

    /// 翻译失败（推理错误、翻译结果不一致等）
    #[error("Translation error: {0}")]
    TranslationError(String),

    /// TTS 语音合成失败（`say` 命令执行、合成过程中的错误）
    #[error("TTS synthesis error: {0}")]
    TtsError(String),

    /// TTS 模型加载失败（模型文件缺失、格式不符等）
    #[error("TTS model load error: {0}")]
    TtsModelLoadError(String),

    /// TTS 音频编码失败（WAV 写入、格式转换等）
    #[error("TTS audio encode error: {0}")]
    TtsAudioEncodeError(String),

    /// 流水线错误（阶段间通信失败、任务取消等）
    #[error("Pipeline error: {0}")]
    PipelineError(String),

    /// 说话人分离失败（模型加载、推理过程中的错误）
    #[error("Diarization error: {0}")]
    DiarizationError(String),

    /// 声音克隆失败（模型加载、推理、音频合成过程中的错误）
    #[error("Voice cloning error: {0}")]
    VoiceCloningError(String),

    /// 批量处理错误（队列管理、任务调度等）
    #[error("Batch processing error: {0}")]
    BatchError(String),

    /// 检查点错误（保存、加载、验证失败等）
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    /// 在线翻译服务不可用（DeepLX 服务未启动、超时、HTTP 429/5xx 等）
    ///
    /// 此错误用于触发降级到本地翻译引擎。
    /// `TranslationRouter` 捕获此错误后自动切换到本地 `LlamaCppBackend`。
    #[error("Online translation unavailable: {0}")]
    OnlineTranslationUnavailable(String),

    /// HTTP 请求错误（网络通信失败、响应解析失败等）
    #[error("HTTP request error: {0}")]
    HttpError(String),
}

/// 应用统一 Result 类型别名
///
/// 所有库函数返回 `AppResult<T>` 而非裸 `Result<T, E>`，
/// 保持错误类型的一致性。
pub type AppResult<T> = Result<T, AppError>;
