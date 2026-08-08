//! 断点续传模块
//!
//! 提供处理中断后的恢复能力，避免重复处理已完成的 Segment。
//!
//! # 功能概览
//! - [`Checkpoint`][]: 检查点数据结构，记录任务进度
//! - [`ProcessingStage`][]: 处理阶段枚举（Asr、Translate、Tts、Compose）
//! - [`CheckpointManager`][]: 检查点管理器，负责保存、加载、验证和清理
//!
//! # 工作流
//! 1. 流水线处理过程中定期调用 [`CheckpointManager::save`] 保存进度
//! 2. 启动时调用 [`CheckpointManager::load`] 检查是否存在检查点
//! 3. 恢复处理时跳过已完成的 Segment，从 `next_segment_index` 继续
//! 4. 任务完成后调用 [`CheckpointManager::delete`] 清理检查点文件
//!
//! # 存储格式
//! 检查点以 JSON 格式存储在 `~/.cache/video-translator/checkpoints/{job_id}.json`。
//!
//! # 示例
//! ```no_run
//! use vt_core::checkpoint::{Checkpoint, CheckpointManager, ProcessingStage};
//! use vt_core::config::CheckpointConfig;
//! use vt_core::models::segment::Segment;
//!
//! let config = CheckpointConfig::default();
//! let manager = CheckpointManager::new(&config);
//!
//! let checkpoint = Checkpoint::new(
//!     "job-001".into(),
//!     "/path/to/video.mp4".into(),
//!     ProcessingStage::Asr,
//! );
//! manager.save(&checkpoint).expect("Failed to save checkpoint");
//!
//! // 恢复时加载检查点
//! if let Some(loaded) = manager.load("job-001").expect("Failed to load") {
//!     println!("Resuming from segment {}", loaded.next_segment_index);
//! }
//! ```

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};

use crate::config::CheckpointConfig;
use crate::error::{AppError, AppResult};
use crate::models::segment::Segment;

// ─── 处理阶段 ─────────────────────────────────────────────

/// 处理阶段枚举
///
/// 表示流水线当前所处的阶段，用于断点续传时确定恢复点。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProcessingStage {
    /// ASR 语音识别阶段
    #[default]
    Asr,
    /// 翻译阶段
    Translate,
    /// TTS 语音合成阶段
    Tts,
    /// 视频合成阶段
    Compose,
}

impl ProcessingStage {
    /// 获取阶段显示名称
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Asr => "ASR",
            Self::Translate => "Translate",
            Self::Tts => "TTS",
            Self::Compose => "Compose",
        }
    }

    /// 获取阶段顺序索引（0-based）
    #[must_use]
    pub fn order(self) -> usize {
        match self {
            Self::Asr => 0,
            Self::Translate => 1,
            Self::Tts => 2,
            Self::Compose => 3,
        }
    }
}

// ─── 检查点 ───────────────────────────────────────────────

/// 检查点数据结构
///
/// 记录一个视频翻译任务的处理进度，用于断点续传。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// 任务唯一标识符
    pub job_id: String,
    /// 原始视频文件路径
    pub video_path: PathBuf,
    /// 已完成 ASR+翻译+TTS 的 Segment 列表
    pub processed_segments: Vec<Segment>,
    /// 当前处理阶段
    pub current_stage: ProcessingStage,
    /// 下一个待处理的 Segment 索引
    pub next_segment_index: usize,
    /// 检查点创建/更新时间戳
    pub timestamp: DateTime<Utc>,
}

impl Checkpoint {
    /// 创建新的检查点
    ///
    /// # 参数
    /// - `job_id`: 任务唯一标识符
    /// - `video_path`: 原始视频文件路径
    /// - `current_stage`: 当前处理阶段
    #[must_use]
    pub fn new(job_id: String, video_path: PathBuf, current_stage: ProcessingStage) -> Self {
        Self {
            job_id,
            video_path,
            processed_segments: Vec::new(),
            current_stage,
            next_segment_index: 0,
            timestamp: Utc::now(),
        }
    }

    /// 添加已完成的 Segment
    ///
    /// # 参数
    /// - `segment`: 已完成处理的 Segment
    pub fn add_segment(&mut self, segment: Segment) {
        self.processed_segments.push(segment);
        self.next_segment_index = self.processed_segments.len();
        self.timestamp = Utc::now();
    }

    /// 更新处理阶段
    ///
    /// # 参数
    /// - `stage`: 新的处理阶段
    pub fn update_stage(&mut self, stage: ProcessingStage) {
        self.current_stage = stage;
        self.timestamp = Utc::now();
    }

    /// 获取已完成的 Segment 数量
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.processed_segments.len()
    }

    /// 检查已完成的 Segment 的音频文件是否仍然存在
    ///
    /// 返回所有音频文件缺失的 Segment 索引列表。
    #[must_use]
    pub fn validate_audio_files(&self) -> Vec<usize> {
        self.processed_segments
            .iter()
            .enumerate()
            .filter_map(|(idx, seg)| {
                if let Some(path) = &seg.tts_audio_path {
                    if !Path::new(path).exists() {
                        return Some(idx);
                    }
                }
                None
            })
            .collect()
    }

    /// 检查是否已过期
    ///
    /// # 参数
    /// - `retention_days`: 保留天数
    #[must_use]
    pub fn is_expired(&self, retention_days: u32) -> bool {
        let age = Utc::now().signed_duration_since(self.timestamp);
        age > ChronoDuration::days(retention_days as i64)
    }

    /// 序列化为 JSON 字符串
    ///
    /// # 错误
    /// - [`AppError::Serialization`][]: 序列化失败
    pub fn to_json(&self) -> AppResult<String> {
        serde_json::to_string_pretty(self).map_err(AppError::from)
    }

    /// 从 JSON 字符串反序列化
    ///
    /// # 参数
    /// - `json`: JSON 字符串
    ///
    /// # 错误
    /// - [`AppError::Serialization`][]: 反序列化失败
    pub fn from_json(json: &str) -> AppResult<Self> {
        serde_json::from_str(json).map_err(AppError::from)
    }
}

// ─── 检查点管理器 ─────────────────────────────────────────

/// 检查点管理器
///
/// 负责检查点文件的保存、加载、验证和清理。
pub struct CheckpointManager {
    /// 检查点存储目录
    dir: PathBuf,
    /// 检查点保留天数
    retention_days: u32,
    /// 是否启用
    enabled: bool,
}

impl std::fmt::Debug for CheckpointManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointManager")
            .field("dir", &self.dir)
            .field("retention_days", &self.retention_days)
            .field("enabled", &self.enabled)
            .finish()
    }
}

impl CheckpointManager {
    /// 创建新的检查点管理器
    ///
    /// # 参数
    /// - `config`: 检查点配置
    #[must_use]
    pub fn new(config: &CheckpointConfig) -> Self {
        let dir = expand_tilde(&config.dir);
        Self {
            dir,
            retention_days: config.retention_days,
            enabled: config.enabled,
        }
    }

    /// 使用指定目录创建管理器（用于测试）
    #[must_use]
    pub fn with_dir(dir: PathBuf) -> Self {
        Self {
            dir,
            retention_days: 7,
            enabled: true,
        }
    }

    /// 获取检查点文件路径
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    #[must_use]
    pub fn checkpoint_path(&self, job_id: &str) -> PathBuf {
        self.dir.join(format!("{job_id}.json"))
    }

    /// 检查管理器是否启用
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 保存检查点到文件
    ///
    /// 自动创建目录（如不存在），以 JSON 格式写入。
    ///
    /// # 参数
    /// - `checkpoint`: 要保存的检查点
    ///
    /// # 错误
    /// - [`AppError::CheckpointError`][]: 目录创建或文件写入失败
    pub fn save(&self, checkpoint: &Checkpoint) -> AppResult<()> {
        if !self.enabled {
            tracing::debug!("Checkpoint disabled, skipping save");
            return Ok(());
        }

        // 确保目录存在
        if !self.dir.exists() {
            std::fs::create_dir_all(&self.dir).map_err(|e| {
                AppError::CheckpointError(format!(
                    "Failed to create checkpoint dir {:?}: {e}",
                    self.dir
                ))
            })?;
        }

        let path = self.checkpoint_path(&checkpoint.job_id);
        let json = checkpoint.to_json().map_err(|e| {
            AppError::CheckpointError(format!("Failed to serialize checkpoint: {e}"))
        })?;

        tracing::debug!("Saving checkpoint to {:?}", path);
        std::fs::write(&path, json).map_err(|e| {
            AppError::CheckpointError(format!("Failed to write checkpoint file {:?}: {e}", path))
        })?;

        Ok(())
    }

    /// 加载检查点
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    ///
    /// # 返回
    /// - `Ok(Some(checkpoint))`: 检查点存在且加载成功
    /// - `Ok(None)`: 检查点不存在
    ///
    /// # 错误
    /// - [`AppError::CheckpointError`][]: 文件读取或反序列化失败
    pub fn load(&self, job_id: &str) -> AppResult<Option<Checkpoint>> {
        let path = self.checkpoint_path(job_id);

        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&path).map_err(|e| {
            AppError::CheckpointError(format!("Failed to read checkpoint file {:?}: {e}", path))
        })?;

        let checkpoint = Checkpoint::from_json(&content).map_err(|e| {
            AppError::CheckpointError(format!("Failed to deserialize checkpoint: {e}"))
        })?;

        tracing::info!(
            "Loaded checkpoint for job {}: {} segments, stage {}, next index {}",
            job_id,
            checkpoint.completed_count(),
            checkpoint.current_stage.name(),
            checkpoint.next_segment_index
        );

        Ok(Some(checkpoint))
    }

    /// 删除检查点文件
    ///
    /// 任务完成后调用，清理对应的检查点文件。
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    ///
    /// # 错误
    /// - [`AppError::CheckpointError`][]: 文件删除失败（文件不存在不算错误）
    pub fn delete(&self, job_id: &str) -> AppResult<()> {
        let path = self.checkpoint_path(job_id);

        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| {
                AppError::CheckpointError(format!(
                    "Failed to delete checkpoint file {:?}: {e}",
                    path
                ))
            })?;
            tracing::info!("Deleted checkpoint for job {}", job_id);
        }

        Ok(())
    }

    /// 列出所有检查点
    ///
    /// 扫描检查点目录，返回所有检查点文件的 job_id 列表。
    pub fn list_checkpoints(&self) -> AppResult<Vec<String>> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }

        let mut job_ids = Vec::new();
        for entry in std::fs::read_dir(&self.dir).map_err(|e| {
            AppError::CheckpointError(format!("Failed to read checkpoint dir {:?}: {e}", self.dir))
        })? {
            let entry = entry
                .map_err(|e| AppError::CheckpointError(format!("Failed to read dir entry: {e}")))?;

            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    job_ids.push(stem.to_string());
                }
            }
        }

        Ok(job_ids)
    }

    /// 清理过期检查点
    ///
    /// 删除超过 `retention_days` 天的检查点文件。
    ///
    /// # 返回
    /// 被清理的检查点数量。
    ///
    /// # 错误
    /// - [`AppError::CheckpointError`][]: 文件操作失败
    pub fn cleanup_expired(&self) -> AppResult<usize> {
        let job_ids = self.list_checkpoints()?;
        let mut cleaned = 0;

        for job_id in job_ids {
            if let Some(checkpoint) = self.load(&job_id)? {
                if checkpoint.is_expired(self.retention_days) {
                    tracing::info!(
                        "Cleaning up expired checkpoint for job {} (age: {})",
                        job_id,
                        checkpoint.timestamp
                    );
                    self.delete(&job_id)?;
                    cleaned += 1;
                }
            }
        }

        if cleaned > 0 {
            tracing::info!("Cleaned up {} expired checkpoints", cleaned);
        }

        Ok(cleaned)
    }

    /// 恢复检查点
    ///
    /// 加载检查点并验证已处理 Segment 的完整性。
    /// 返回恢复信息：已完成的 Segment 列表和下一个待处理的索引。
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    ///
    /// # 返回
    /// - `Ok(Some((segments, next_index)))`: 检查点存在，可恢复
    /// - `Ok(None)`: 检查点不存在，从头开始
    ///
    /// # 验证
    /// 加载后会验证已处理 Segment 的音频文件是否存在。
    /// 如果音频文件缺失，对应的 Segment 会被移除，`next_segment_index` 会调整。
    ///
    /// # 错误
    /// - [`AppError::CheckpointError`][]: 加载或验证失败
    pub fn resume(&self, job_id: &str) -> AppResult<Option<(Vec<Segment>, usize)>> {
        let Some(mut checkpoint) = self.load(job_id)? else {
            return Ok(None);
        };

        // 验证音频文件
        let missing_indices = checkpoint.validate_audio_files();

        if !missing_indices.is_empty() {
            tracing::warn!(
                "Found {} segments with missing audio files, removing them",
                missing_indices.len()
            );

            // 从后往前删除，避免索引偏移
            for &idx in missing_indices.iter().rev() {
                checkpoint.processed_segments.remove(idx);
            }

            checkpoint.next_segment_index = checkpoint.processed_segments.len();

            // 重新保存更新后的检查点
            self.save(&checkpoint)?;
        }

        let next_index = checkpoint.next_segment_index;
        let segments = checkpoint.processed_segments;

        Ok(Some((segments, next_index)))
    }

    /// 更新检查点（添加已完成的 Segment 并保存）
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    /// - `segment`: 新完成的 Segment
    /// - `stage`: 当前处理阶段
    ///
    /// # 错误
    /// - [`AppError::CheckpointError`][]: 加载或保存失败
    pub fn update(&self, job_id: &str, segment: Segment, stage: ProcessingStage) -> AppResult<()> {
        let mut checkpoint = self
            .load(job_id)?
            .unwrap_or_else(|| Checkpoint::new(job_id.to_string(), PathBuf::new(), stage));

        checkpoint.add_segment(segment);
        checkpoint.update_stage(stage);
        self.save(&checkpoint)
    }
}

// ─── 辅助函数 ─────────────────────────────────────────────

/// 展开 `~` 为用户主目录
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(stripped);
        }
    }
    PathBuf::from(path)
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::segment::Segment;

    /// 创建测试用检查点管理器（使用临时目录）
    fn test_manager() -> (CheckpointManager, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let manager = CheckpointManager::with_dir(dir.path().to_path_buf());
        (manager, dir)
    }

    /// 创建测试用 Segment（已完成状态）
    fn completed_segment(id: &str) -> Segment {
        let mut seg = Segment::new(id.into(), 0.0, 5.0, "Hello".into());
        seg.start_transcribing().expect("start_transcribing");
        seg.finish_transcribing("你好".into())
            .expect("finish_transcribing");
        seg.start_synthesizing().expect("start_synthesizing");
        seg.finish_synthesizing("/tmp/test_audio.wav".into())
            .expect("finish_synthesizing");
        seg
    }

    /// 创建测试用 Segment（已完成状态，指定音频路径）
    fn completed_segment_with_audio(id: &str, audio_path: &str) -> Segment {
        let mut seg = Segment::new(id.into(), 0.0, 5.0, "Hello".into());
        seg.start_transcribing().expect("start_transcribing");
        seg.finish_transcribing("你好".into())
            .expect("finish_transcribing");
        seg.start_synthesizing().expect("start_synthesizing");
        seg.finish_synthesizing(audio_path.into())
            .expect("finish_synthesizing");
        seg
    }

    // ── ProcessingStage 测试 ──────────────────────────

    #[test]
    fn test_processing_stage_name() {
        assert_eq!(ProcessingStage::Asr.name(), "ASR");
        assert_eq!(ProcessingStage::Translate.name(), "Translate");
        assert_eq!(ProcessingStage::Tts.name(), "TTS");
        assert_eq!(ProcessingStage::Compose.name(), "Compose");
    }

    #[test]
    fn test_processing_stage_order() {
        assert_eq!(ProcessingStage::Asr.order(), 0);
        assert_eq!(ProcessingStage::Translate.order(), 1);
        assert_eq!(ProcessingStage::Tts.order(), 2);
        assert_eq!(ProcessingStage::Compose.order(), 3);
    }

    #[test]
    fn test_processing_stage_default() {
        assert_eq!(ProcessingStage::default(), ProcessingStage::Asr);
    }

    // ── Checkpoint 测试 ────────────────────────────────

    #[test]
    fn test_checkpoint_new() {
        let cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );

        assert_eq!(cp.job_id, "job-1");
        assert_eq!(cp.video_path, PathBuf::from("/path/to/video.mp4"));
        assert!(cp.processed_segments.is_empty());
        assert_eq!(cp.current_stage, ProcessingStage::Asr);
        assert_eq!(cp.next_segment_index, 0);
    }

    #[test]
    fn test_checkpoint_add_segment() {
        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Tts,
        );

        let seg = completed_segment("seg-1");
        cp.add_segment(seg);

        assert_eq!(cp.completed_count(), 1);
        assert_eq!(cp.next_segment_index, 1);
    }

    #[test]
    fn test_checkpoint_add_multiple_segments() {
        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Tts,
        );

        for i in 0..5 {
            let seg = completed_segment(&format!("seg-{i}"));
            cp.add_segment(seg);
        }

        assert_eq!(cp.completed_count(), 5);
        assert_eq!(cp.next_segment_index, 5);
    }

    #[test]
    fn test_checkpoint_update_stage() {
        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );

        cp.update_stage(ProcessingStage::Translate);
        assert_eq!(cp.current_stage, ProcessingStage::Translate);
    }

    #[test]
    fn test_checkpoint_validate_audio_files_all_exist() {
        let (_manager, dir) = test_manager();

        // 创建实际存在的音频文件
        let audio_path = dir.path().join("audio_1.wav");
        std::fs::write(&audio_path, b"fake audio").expect("Failed to write audio file");

        let mut seg = completed_segment("seg-1");
        seg.tts_audio_path = Some(audio_path.to_string_lossy().into_owned());

        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Tts,
        );
        cp.add_segment(seg);

        let missing = cp.validate_audio_files();
        assert!(missing.is_empty(), "No missing audio files expected");
    }

    #[test]
    fn test_checkpoint_validate_audio_files_missing() {
        let mut seg = completed_segment("seg-1");
        seg.tts_audio_path = Some("/nonexistent/audio.wav".into());

        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Tts,
        );
        cp.add_segment(seg);

        let missing = cp.validate_audio_files();
        assert_eq!(missing.len(), 1, "Should find 1 missing audio file");
        assert_eq!(missing[0], 0);
    }

    #[test]
    fn test_checkpoint_is_expired() {
        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );

        // 设置时间戳为 10 天前
        cp.timestamp = Utc::now() - ChronoDuration::days(10);

        assert!(
            cp.is_expired(7),
            "Should be expired after 10 days with 7 day retention"
        );
        assert!(
            !cp.is_expired(14),
            "Should not be expired with 14 day retention"
        );
    }

    #[test]
    fn test_checkpoint_is_not_expired() {
        let cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );

        assert!(!cp.is_expired(7), "Fresh checkpoint should not be expired");
    }

    #[test]
    fn test_checkpoint_json_roundtrip() {
        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Translate,
        );
        cp.add_segment(completed_segment("seg-1"));
        cp.add_segment(completed_segment("seg-2"));

        let json = cp.to_json().expect("to_json failed");
        let restored = Checkpoint::from_json(&json).expect("from_json failed");

        assert_eq!(restored.job_id, cp.job_id);
        assert_eq!(restored.video_path, cp.video_path);
        assert_eq!(restored.current_stage, cp.current_stage);
        assert_eq!(restored.next_segment_index, cp.next_segment_index);
        assert_eq!(restored.completed_count(), cp.completed_count());
    }

    // ── CheckpointManager 测试 ────────────────────────

    #[test]
    fn test_checkpoint_manager_save_and_load() {
        let (manager, _dir) = test_manager();

        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );
        cp.add_segment(completed_segment("seg-1"));

        manager.save(&cp).expect("save failed");

        let loaded = manager.load("job-1").expect("load failed");
        assert!(loaded.is_some(), "Should load saved checkpoint");

        let loaded = loaded.unwrap();
        assert_eq!(loaded.job_id, "job-1");
        assert_eq!(loaded.completed_count(), 1);
    }

    #[test]
    fn test_checkpoint_manager_load_nonexistent() {
        let (manager, _dir) = test_manager();

        let loaded = manager.load("nonexistent-job").expect("load failed");
        assert!(
            loaded.is_none(),
            "Should return None for nonexistent checkpoint"
        );
    }

    #[test]
    fn test_checkpoint_manager_delete() {
        let (manager, _dir) = test_manager();

        let cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );

        manager.save(&cp).expect("save failed");
        assert!(manager.checkpoint_path("job-1").exists());

        manager.delete("job-1").expect("delete failed");
        assert!(!manager.checkpoint_path("job-1").exists());
    }

    #[test]
    fn test_checkpoint_manager_delete_nonexistent() {
        let (manager, _dir) = test_manager();

        // 删除不存在的检查点不应报错
        manager
            .delete("nonexistent")
            .expect("delete should not fail");
    }

    #[test]
    fn test_checkpoint_manager_list_checkpoints() {
        let (manager, _dir) = test_manager();

        // 空目录
        let list = manager.list_checkpoints().expect("list failed");
        assert!(list.is_empty());

        // 添加几个检查点
        manager
            .save(&Checkpoint::new(
                "job-1".into(),
                PathBuf::from("/path/to/video1.mp4"),
                ProcessingStage::Asr,
            ))
            .expect("save failed");
        manager
            .save(&Checkpoint::new(
                "job-2".into(),
                PathBuf::from("/path/to/video2.mp4"),
                ProcessingStage::Asr,
            ))
            .expect("save failed");

        let list = manager.list_checkpoints().expect("list failed");
        assert_eq!(list.len(), 2);
        assert!(list.contains(&"job-1".to_string()));
        assert!(list.contains(&"job-2".to_string()));
    }

    #[test]
    fn test_checkpoint_manager_cleanup_expired() {
        let (manager, _dir) = test_manager();

        // 创建一个过期的检查点
        let mut old_cp = Checkpoint::new(
            "old-job".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );
        old_cp.timestamp = Utc::now() - ChronoDuration::days(10);
        manager.save(&old_cp).expect("save failed");

        // 创建一个未过期的检查点
        let new_cp = Checkpoint::new(
            "new-job".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );
        manager.save(&new_cp).expect("save failed");

        // 清理过期检查点（保留 7 天）
        let cleaned = manager.cleanup_expired().expect("cleanup failed");
        assert_eq!(cleaned, 1, "Should clean up 1 expired checkpoint");

        // 验证结果
        assert!(!manager.checkpoint_path("old-job").exists());
        assert!(manager.checkpoint_path("new-job").exists());
    }

    #[test]
    fn test_checkpoint_manager_resume_no_checkpoint() {
        let (manager, _dir) = test_manager();

        let result = manager.resume("nonexistent").expect("resume failed");
        assert!(
            result.is_none(),
            "Should return None for nonexistent checkpoint"
        );
    }

    #[test]
    fn test_checkpoint_manager_resume_with_valid_audio() {
        let (manager, dir) = test_manager();

        // 创建实际存在的音频文件
        let audio_path = dir.path().join("audio_1.wav");
        std::fs::write(&audio_path, b"fake audio").expect("Failed to write audio");

        let mut seg = completed_segment("seg-1");
        seg.tts_audio_path = Some(audio_path.to_string_lossy().into_owned());

        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Tts,
        );
        cp.add_segment(seg);

        manager.save(&cp).expect("save failed");

        let result = manager.resume("job-1").expect("resume failed");
        assert!(result.is_some(), "Should resume from checkpoint");

        let (segments, next_index) = result.unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(next_index, 1);
    }

    #[test]
    fn test_checkpoint_manager_resume_with_missing_audio() {
        let (manager, _dir) = test_manager();

        let mut seg = completed_segment("seg-1");
        seg.tts_audio_path = Some("/nonexistent/audio.wav".into());

        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Tts,
        );
        cp.add_segment(seg);

        manager.save(&cp).expect("save failed");

        let result = manager.resume("job-1").expect("resume failed");
        assert!(
            result.is_some(),
            "Should still resume, but with adjusted state"
        );

        let (segments, next_index) = result.unwrap();
        // 音频文件缺失的 Segment 应被移除
        assert!(
            segments.is_empty(),
            "Segment with missing audio should be removed"
        );
        assert_eq!(next_index, 0);
    }

    #[test]
    fn test_checkpoint_manager_update() {
        let (manager, _dir) = test_manager();

        // 第一次更新（创建新检查点）
        let seg1 = completed_segment("seg-1");
        manager
            .update("job-1", seg1, ProcessingStage::Tts)
            .expect("update failed");

        // 第二次更新（追加 Segment）
        let seg2 = completed_segment("seg-2");
        manager
            .update("job-1", seg2, ProcessingStage::Tts)
            .expect("update failed");

        // 验证
        let loaded = manager
            .load("job-1")
            .expect("load failed")
            .expect("checkpoint should exist");
        assert_eq!(loaded.completed_count(), 2);
        assert_eq!(loaded.next_segment_index, 2);
    }

    #[test]
    fn test_checkpoint_manager_checkpoint_path() {
        let (manager, _dir) = test_manager();

        let path = manager.checkpoint_path("job-123");
        assert!(path.to_string_lossy().contains("job-123"));
        assert!(path.to_string_lossy().ends_with(".json"));
    }

    #[test]
    fn test_checkpoint_manager_disabled() {
        let config = CheckpointConfig {
            enabled: false,
            dir: "/tmp/test-checkpoints-disabled".into(),
            retention_days: 7,
        };
        let manager = CheckpointManager::new(&config);

        assert!(!manager.is_enabled());

        // 保存应在禁用时跳过（不报错但不写入）
        let cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );
        manager
            .save(&cp)
            .expect("save should not fail when disabled");
    }

    #[test]
    fn test_checkpoint_manager_creates_dir() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let nested = dir.path().join("nested").join("checkpoints");

        let manager = CheckpointManager::with_dir(nested.clone());

        let cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );

        // 保存应自动创建不存在的目录
        manager.save(&cp).expect("save should create dirs");

        assert!(nested.exists(), "Directory should be created");
        assert!(manager.checkpoint_path("job-1").exists());
    }

    #[test]
    fn test_checkpoint_manager_full_lifecycle() {
        let (manager, dir) = test_manager();

        // 1. 初始状态：无检查点
        assert!(manager.resume("job-1").expect("resume").is_none());

        // 创建实际存在的音频文件
        let audio1 = dir.path().join("audio1.wav");
        std::fs::write(&audio1, b"audio").expect("write audio");
        let audio2 = dir.path().join("audio2.wav");
        std::fs::write(&audio2, b"audio").expect("write audio");

        // 2. 处理过程中保存检查点
        let mut cp = Checkpoint::new(
            "job-1".into(),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Asr,
        );
        cp.add_segment(completed_segment_with_audio(
            "seg-1",
            audio1.to_str().unwrap(),
        ));
        cp.add_segment(completed_segment_with_audio(
            "seg-2",
            audio2.to_str().unwrap(),
        ));
        cp.update_stage(ProcessingStage::Translate);
        manager.save(&cp).expect("save");

        // 3. 模拟中断后恢复
        let result = manager.resume("job-1").expect("resume");
        assert!(result.is_some());
        let (segments, next_index) = result.unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(next_index, 2);

        // 4. 任务完成，清理检查点
        manager.delete("job-1").expect("delete");
        assert!(manager.load("job-1").expect("load").is_none());
    }

    #[test]
    fn test_expand_tilde() {
        // ~ 开头的路径应被展开
        let expanded = expand_tilde("~/test/path");
        assert!(!expanded.starts_with("~"), "Tilde should be expanded");

        // 非 ~ 开头的路径保持不变
        let not_expanded = expand_tilde("/absolute/path");
        assert_eq!(not_expanded, PathBuf::from("/absolute/path"));

        // 相对路径保持不变
        let relative = expand_tilde("relative/path");
        assert_eq!(relative, PathBuf::from("relative/path"));
    }
}
