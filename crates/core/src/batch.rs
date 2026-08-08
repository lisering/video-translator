//! 批量处理优化模块
//!
//! 提供多视频队列处理、智能并发控制、资源监控和任务优先级调度。
//!
//! # 功能概览
//! - [`BatchJob`][]: 批量任务数据结构，包含路径、配置、状态和进度
//! - [`BatchQueue`][]: 任务队列管理器，支持添加、启动、暂停、取消和状态查询
//! - [`Priority`][]: 任务优先级枚举（High、Normal、Low）
//! - [`ResourceMonitor`][]: 内存使用监控，超阈值时自动降低并发数
//! - [`estimate_remaining_time`][]: 基于已处理任务平均速度预估剩余时间
//!
//! # 设计理念
//! - **背压控制**: 限制最大并发数，防止资源耗尽
//! - **动态调整**: 根据内存使用情况自动增减并发数
//! - **优先级调度**: High 优先级任务优先出队
//! - **优雅降级**: 资源不足时降低并发而非拒绝任务
//!
//! # 示例
//! ```no_run
//! use vt_core::batch::{BatchQueue, BatchJob, Priority};
//! use vt_core::config::Config;
//!
//! let config = Config::default();
//! let mut queue = BatchQueue::new(&config);
//!
//! let job = BatchJob::new(
//!     "job-1".into(),
//!     "input1.mp4".into(),
//!     "output1.mp4".into(),
//!     config.clone(),
//!     Priority::Normal,
//! );
//! queue.add_job(job);
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sysinfo::System;

use crate::config::{BatchConfig, Config};

// ─── 优先级 ───────────────────────────────────────────────

/// 任务优先级枚举
///
/// 用于 [`BatchQueue`] 的优先级调度，High 优先级任务优先出队。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum Priority {
    /// 高优先级（优先处理）
    High,
    /// 普通优先级（默认）
    #[default]
    Normal,
    /// 低优先级（最后处理）
    Low,
}

// ─── 任务状态 ─────────────────────────────────────────────

/// 批量任务状态枚举
///
/// 状态流转：
/// ```text
/// Queued → Running → Completed
///                  → Failed
/// Queued → Cancelled
/// Running → Paused → Running
///         → Cancelled
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum JobStatus {
    /// 排队中（等待处理）
    #[default]
    Queued,
    /// 正在处理
    Running,
    /// 已暂停
    Paused,
    /// 已完成
    Completed,
    /// 已失败
    Failed,
    /// 已取消
    Cancelled,
}

impl JobStatus {
    /// 判断任务是否处于终态（不会再变化）
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// 判断任务是否正在处理
    #[must_use]
    pub fn is_active(self) -> bool {
        matches!(self, Self::Running | Self::Paused)
    }
}

// ─── 批量任务 ─────────────────────────────────────────────

/// 批量任务数据结构
///
/// 表示一个视频翻译任务，包含输入/输出路径、配置、状态和进度信息。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    /// 任务唯一标识符
    pub id: String,
    /// 输入视频文件路径
    pub input_path: PathBuf,
    /// 输出视频文件路径
    pub output_path: PathBuf,
    /// 任务配置
    pub config: Config,
    /// 任务优先级
    pub priority: Priority,
    /// 任务状态
    pub status: JobStatus,
    /// 处理进度（0.0–1.0）
    pub progress: f64,
    /// 错误信息（任务失败时）
    pub error: Option<String>,
    /// 任务创建时间戳
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 任务开始处理时间戳
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    /// 任务完成时间戳
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl BatchJob {
    /// 创建新的批量任务
    ///
    /// # 参数
    /// - `id`: 任务唯一标识符
    /// - `input_path`: 输入视频文件路径
    /// - `output_path`: 输出视频文件路径
    /// - `config`: 任务配置
    /// - `priority`: 任务优先级
    #[must_use]
    pub fn new(
        id: String,
        input_path: PathBuf,
        output_path: PathBuf,
        config: Config,
        priority: Priority,
    ) -> Self {
        Self {
            id,
            input_path,
            output_path,
            config,
            priority,
            status: JobStatus::Queued,
            progress: 0.0,
            error: None,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
        }
    }

    /// 更新任务进度
    ///
    /// # 参数
    /// - `progress`: 进度值（0.0–1.0），会自动 clamp
    pub fn update_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// 标记任务为运行中
    pub fn start(&mut self) {
        self.status = JobStatus::Running;
        self.started_at = Some(chrono::Utc::now());
    }

    /// 标记任务为已完成
    pub fn complete(&mut self) {
        self.status = JobStatus::Completed;
        self.progress = 1.0;
        self.completed_at = Some(chrono::Utc::now());
    }

    /// 标记任务为失败
    ///
    /// # 参数
    /// - `error`: 错误信息
    pub fn fail(&mut self, error: String) {
        self.status = JobStatus::Failed;
        self.error = Some(error);
        self.completed_at = Some(chrono::Utc::now());
    }

    /// 标记任务为已暂停
    pub fn pause(&mut self) {
        if self.status == JobStatus::Running {
            self.status = JobStatus::Paused;
        }
    }

    /// 标记任务为已取消
    pub fn cancel(&mut self) {
        self.status = JobStatus::Cancelled;
        self.completed_at = Some(chrono::Utc::now());
    }

    /// 获取任务处理耗时
    ///
    /// 返回从 `started_at` 到 `completed_at`（或当前时间）的时长。
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        let start = self.started_at?;
        let end = self.completed_at.unwrap_or_else(chrono::Utc::now);
        let duration = end.signed_duration_since(start);
        duration.to_std().ok()
    }
}

// ─── 资源监控 ─────────────────────────────────────────────

/// 资源监控器
///
/// 使用 `sysinfo` crate 监控系统内存使用情况，
/// 当内存使用超过阈值时建议降低并发数。
pub struct ResourceMonitor {
    /// sysinfo 系统实例
    sys: System,
    /// 内存使用阈值百分比
    memory_threshold: f64,
    /// 初始最大并发数
    initial_max_concurrent: usize,
    /// 当前最大并发数（可能因内存压力动态降低）
    current_max_concurrent: usize,
}

impl ResourceMonitor {
    /// 创建新的资源监控器
    ///
    /// # 参数
    /// - `config`: 批量处理配置
    #[must_use]
    pub fn new(config: &BatchConfig) -> Self {
        let initial = config.max_concurrent;
        Self {
            sys: System::new(),
            memory_threshold: config.memory_threshold,
            initial_max_concurrent: initial,
            current_max_concurrent: initial,
        }
    }

    /// 刷新系统信息并检查内存压力
    ///
    /// 返回当前建议的最大并发数。如果内存使用率超过阈值，
    /// 则将并发数降低为当前值的一半（最小为 1）。
    pub fn check_and_adjust(&mut self) -> usize {
        self.sys.refresh_memory();
        let total = self.sys.total_memory();
        let used = self.sys.used_memory();

        if total == 0 {
            return self.current_max_concurrent;
        }

        let usage_percent = used as f64 / total as f64 * 100.0;

        if usage_percent > self.memory_threshold {
            // 内存压力高，降低并发
            let new_concurrent = (self.current_max_concurrent / 2).max(1);
            if new_concurrent != self.current_max_concurrent {
                tracing::warn!(
                    "Memory usage {:.1}% exceeds threshold {:.1}%, reducing concurrency {} → {}",
                    usage_percent,
                    self.memory_threshold,
                    self.current_max_concurrent,
                    new_concurrent
                );
                self.current_max_concurrent = new_concurrent;
            }
        } else if usage_percent < self.memory_threshold * 0.7 {
            // 内存压力低，尝试恢复并发
            if self.current_max_concurrent < self.initial_max_concurrent {
                let new_concurrent =
                    (self.current_max_concurrent + 1).min(self.initial_max_concurrent);
                tracing::info!(
                    "Memory usage {:.1}% below recovery threshold, increasing concurrency {} → {}",
                    usage_percent,
                    self.current_max_concurrent,
                    new_concurrent
                );
                self.current_max_concurrent = new_concurrent;
            }
        }

        self.current_max_concurrent
    }

    /// 获取当前内存使用率百分比
    pub fn memory_usage_percent(&mut self) -> f64 {
        self.sys.refresh_memory();
        let total = self.sys.total_memory();
        if total == 0 {
            return 0.0;
        }
        self.sys.used_memory() as f64 / total as f64 * 100.0
    }

    /// 获取当前建议的最大并发数
    #[must_use]
    pub fn current_max_concurrent(&self) -> usize {
        self.current_max_concurrent
    }

    /// 重置并发数为初始值
    pub fn reset(&mut self) {
        self.current_max_concurrent = self.initial_max_concurrent;
    }
}

impl std::fmt::Debug for ResourceMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceMonitor")
            .field("memory_threshold", &self.memory_threshold)
            .field("initial_max_concurrent", &self.initial_max_concurrent)
            .field("current_max_concurrent", &self.current_max_concurrent)
            .finish()
    }
}

// ─── 批量队列 ─────────────────────────────────────────────

/// 批量任务队列管理器
///
/// 管理多个视频翻译任务，支持优先级调度、并发控制和资源监控。
///
/// # 并发控制
/// 默认最大并发数为 CPU 核心数 - 1，可根据内存使用情况动态调整。
///
/// # 优先级调度
/// 当启用优先级时，High 优先级任务优先出队。
pub struct BatchQueue {
    /// 任务列表（按添加顺序，出队时按优先级排序）
    jobs: Vec<BatchJob>,
    /// 批量处理配置
    config: BatchConfig,
    /// 资源监控器
    resource_monitor: ResourceMonitor,
    /// 队列是否已暂停（暂停后不再出队新任务）
    paused: bool,
    /// 已完成任务的处理时长记录（用于预估剩余时间）
    completed_durations: Vec<Duration>,
}

impl std::fmt::Debug for BatchQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchQueue")
            .field("job_count", &self.jobs.len())
            .field("config", &self.config)
            .field("paused", &self.paused)
            .field("completed_count", &self.completed_durations.len())
            .finish()
    }
}

impl BatchQueue {
    /// 创建新的批量任务队列
    ///
    /// # 参数
    /// - `config`: 应用配置（从中提取 `BatchConfig`）
    #[must_use]
    pub fn new(config: &Config) -> Self {
        let batch_config = config.batch.clone();
        let resource_monitor = ResourceMonitor::new(&batch_config);
        Self {
            jobs: Vec::new(),
            config: batch_config,
            resource_monitor,
            paused: false,
            completed_durations: Vec::new(),
        }
    }

    /// 使用指定批量配置创建队列
    #[must_use]
    pub fn with_config(config: BatchConfig) -> Self {
        let resource_monitor = ResourceMonitor::new(&config);
        Self {
            jobs: Vec::new(),
            config,
            resource_monitor,
            paused: false,
            completed_durations: Vec::new(),
        }
    }

    /// 添加任务到队列
    ///
    /// 任务添加后状态为 `Queued`，等待调度执行。
    ///
    /// # 参数
    /// - `job`: 要添加的批量任务
    pub fn add_job(&mut self, mut job: BatchJob) {
        job.status = JobStatus::Queued;
        self.jobs.push(job);
    }

    /// 获取下一个待执行任务
    ///
    /// 根据 `config.enable_priority` 决定是否按优先级排序。
    /// 如果队列已暂停或并发数已达上限，返回 `None`。
    pub fn next_job(&mut self) -> Option<BatchJob> {
        if self.paused {
            return None;
        }

        // 检查并调整并发数
        let max_concurrent = self.resource_monitor.check_and_adjust();
        let running_count = self
            .jobs
            .iter()
            .filter(|j| j.status == JobStatus::Running)
            .count();

        if running_count >= max_concurrent {
            return None;
        }

        // 查找下一个 Queued 任务
        let candidate_idx = if self.config.enable_priority {
            // 按优先级排序：High > Normal > Low
            self.jobs
                .iter()
                .enumerate()
                .filter(|(_, j)| j.status == JobStatus::Queued)
                .min_by_key(|(_, j)| j.priority)
                .map(|(idx, _)| idx)
        } else {
            // FIFO 顺序
            self.jobs.iter().position(|j| j.status == JobStatus::Queued)
        };

        if let Some(idx) = candidate_idx {
            self.jobs[idx].start();
            Some(self.jobs[idx].clone())
        } else {
            None
        }
    }

    /// 更新任务进度
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    /// - `progress`: 进度值（0.0–1.0）
    pub fn update_progress(&mut self, job_id: &str, progress: f64) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
            job.update_progress(progress);
        }
    }

    /// 标记任务为已完成
    ///
    /// 记录任务处理时长用于后续预估。
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    pub fn complete_job(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
            if let Some(elapsed) = job.elapsed() {
                self.completed_durations.push(elapsed);
            }
            job.complete();
        }
    }

    /// 标记任务为失败
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    /// - `error`: 错误信息
    pub fn fail_job(&mut self, job_id: &str, error: String) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
            job.fail(error);
        }
    }

    /// 暂停队列
    ///
    /// 暂停后不再出队新任务，但已运行的任务不受影响。
    pub fn pause(&mut self) {
        self.paused = true;
        for job in &mut self.jobs {
            if job.status == JobStatus::Running {
                job.pause();
            }
        }
    }

    /// 恢复队列
    ///
    /// 恢复出队新任务，并将暂停状态的任务恢复为排队。
    pub fn resume(&mut self) {
        self.paused = false;
        for job in &mut self.jobs {
            if job.status == JobStatus::Paused {
                job.status = JobStatus::Queued;
            }
        }
    }

    /// 取消指定任务
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    pub fn cancel_job(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
            job.cancel();
        }
    }

    /// 取消所有任务
    pub fn cancel_all(&mut self) {
        for job in &mut self.jobs {
            if !job.status.is_terminal() {
                job.cancel();
            }
        }
    }

    /// 启动队列（恢复出队）
    pub fn start(&mut self) {
        self.paused = false;
    }

    /// 获取任务状态
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    #[must_use]
    pub fn get_status(&self, job_id: &str) -> Option<JobStatus> {
        self.jobs.iter().find(|j| j.id == job_id).map(|j| j.status)
    }

    /// 获取任务进度
    ///
    /// # 参数
    /// - `job_id`: 任务 ID
    #[must_use]
    pub fn get_progress(&self, job_id: &str) -> Option<f64> {
        self.jobs
            .iter()
            .find(|j| j.id == job_id)
            .map(|j| j.progress)
    }

    /// 获取所有任务的状态快照
    #[must_use]
    pub fn get_all_statuses(&self) -> HashMap<String, JobStatus> {
        self.jobs.iter().map(|j| (j.id.clone(), j.status)).collect()
    }

    /// 获取队列中的任务总数
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// 队列是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// 获取待处理（Queued）任务数
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == JobStatus::Queued)
            .count()
    }

    /// 获取正在运行的任务数
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == JobStatus::Running)
            .count()
    }

    /// 获取已完成任务数
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == JobStatus::Completed)
            .count()
    }

    /// 获取失败任务数
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == JobStatus::Failed)
            .count()
    }

    /// 获取当前最大并发数
    #[must_use]
    pub fn current_max_concurrent(&self) -> usize {
        self.resource_monitor.current_max_concurrent()
    }

    /// 预估剩余处理时间
    ///
    /// 基于已完成任务的平均处理速度，预估剩余任务的总处理时间。
    ///
    /// # 返回
    /// - `Some(Duration)`: 有历史数据时返回预估时间
    /// - `None`: 无历史数据时返回 `None`
    #[must_use]
    pub fn estimate_remaining_time(&self) -> Option<Duration> {
        if self.completed_durations.is_empty() {
            return None;
        }

        let avg_duration: Duration = self.completed_durations.iter().sum::<Duration>()
            / self.completed_durations.len() as u32;

        let pending = self.pending_count();
        let running = self.running_count();

        // 剩余时间 = (待处理 + 正在运行) × 平均处理时间 / 当前并发数
        let total_remaining = (pending + running) as u32 * avg_duration;
        let concurrent = self.current_max_concurrent().max(1) as u32;

        Some(total_remaining / concurrent)
    }

    /// 获取当前内存使用率
    pub fn memory_usage_percent(&mut self) -> f64 {
        self.resource_monitor.memory_usage_percent()
    }

    /// 获取任务列表（只读引用）
    #[must_use]
    pub fn jobs(&self) -> &[BatchJob] {
        &self.jobs
    }

    /// 获取任务列表（可变引用，用于外部更新）
    pub fn jobs_mut(&mut self) -> &mut [BatchJob] {
        &mut self.jobs
    }
}

// ─── 辅助函数 ─────────────────────────────────────────────

/// 根据已处理任务列表预估剩余处理时间
///
/// # 参数
/// - `completed_durations`: 已完成任务的处理时长列表
/// - `pending_count`: 待处理任务数
/// - `concurrent`: 当前并发数
///
/// # 返回
/// 预估的剩余时间。无历史数据时返回 `None`。
#[must_use]
pub fn estimate_remaining_time(
    completed_durations: &[Duration],
    pending_count: usize,
    concurrent: usize,
) -> Option<Duration> {
    if completed_durations.is_empty() {
        return None;
    }

    let avg_duration: Duration =
        completed_durations.iter().sum::<Duration>() / completed_durations.len() as u32;

    let total_remaining = pending_count as u32 * avg_duration;
    let concurrent = concurrent.max(1) as u32;

    Some(total_remaining / concurrent)
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// 创建测试用 Config
    fn test_config() -> Config {
        Config::default()
    }

    /// 创建测试用 BatchConfig（小并发数便于测试）
    fn test_batch_config() -> BatchConfig {
        BatchConfig {
            max_concurrent: 2,
            memory_threshold: 90.0,
            enable_priority: true,
        }
    }

    // ── Priority 测试 ──────────────────────────────────

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::High < Priority::Normal);
        assert!(Priority::Normal < Priority::Low);
        assert_eq!(Priority::default(), Priority::Normal);
    }

    // ── JobStatus 测试 ─────────────────────────────────

    #[test]
    fn test_job_status_is_terminal() {
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
        assert!(!JobStatus::Queued.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
        assert!(!JobStatus::Paused.is_terminal());
    }

    #[test]
    fn test_job_status_is_active() {
        assert!(JobStatus::Running.is_active());
        assert!(JobStatus::Paused.is_active());
        assert!(!JobStatus::Queued.is_active());
        assert!(!JobStatus::Completed.is_active());
    }

    // ── BatchJob 测试 ──────────────────────────────────

    #[test]
    fn test_batch_job_new() {
        let config = test_config();
        let job = BatchJob::new(
            "job-1".into(),
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            config,
            Priority::Normal,
        );

        assert_eq!(job.id, "job-1");
        assert_eq!(job.input_path, PathBuf::from("input.mp4"));
        assert_eq!(job.output_path, PathBuf::from("output.mp4"));
        assert_eq!(job.status, JobStatus::Queued);
        assert_eq!(job.priority, Priority::Normal);
        assert!((job.progress - 0.0).abs() < f64::EPSILON);
        assert!(job.error.is_none());
        assert!(job.started_at.is_none());
        assert!(job.completed_at.is_none());
    }

    #[test]
    fn test_batch_job_update_progress() {
        let mut job = BatchJob::new(
            "job-1".into(),
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            test_config(),
            Priority::Normal,
        );

        job.update_progress(0.5);
        assert!((job.progress - 0.5).abs() < f64::EPSILON);

        // 测试 clamp
        job.update_progress(-0.1);
        assert!((job.progress - 0.0).abs() < f64::EPSILON);

        job.update_progress(1.5);
        assert!((job.progress - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_job_lifecycle() {
        let mut job = BatchJob::new(
            "job-1".into(),
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            test_config(),
            Priority::Normal,
        );

        // 初始状态
        assert_eq!(job.status, JobStatus::Queued);

        // 开始
        job.start();
        assert_eq!(job.status, JobStatus::Running);
        assert!(job.started_at.is_some());

        // 暂停
        job.pause();
        assert_eq!(job.status, JobStatus::Paused);

        // 重新开始（需要手动设置，因为 pause 只在 Running 时生效）
        job.status = JobStatus::Running;

        // 完成
        job.complete();
        assert_eq!(job.status, JobStatus::Completed);
        assert!((job.progress - 1.0).abs() < f64::EPSILON);
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn test_batch_job_fail() {
        let mut job = BatchJob::new(
            "job-1".into(),
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            test_config(),
            Priority::Normal,
        );

        job.start();
        job.fail("ASR engine failed".into());

        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.error.as_deref(), Some("ASR engine failed"));
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn test_batch_job_cancel() {
        let mut job = BatchJob::new(
            "job-1".into(),
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            test_config(),
            Priority::Normal,
        );

        job.cancel();
        assert_eq!(job.status, JobStatus::Cancelled);
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn test_batch_job_elapsed() {
        let mut job = BatchJob::new(
            "job-1".into(),
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            test_config(),
            Priority::Normal,
        );

        // 未开始时 elapsed 为 None
        assert!(job.elapsed().is_none());

        // 开始后 elapsed 有值
        job.start();
        assert!(job.elapsed().is_some());

        // 完成后 elapsed 仍有值
        job.complete();
        assert!(job.elapsed().is_some());
    }

    // ── ResourceMonitor 测试 ──────────────────────────

    #[test]
    fn test_resource_monitor_new() {
        let config = test_batch_config();
        let monitor = ResourceMonitor::new(&config);

        assert_eq!(monitor.memory_threshold, 90.0);
        assert_eq!(monitor.initial_max_concurrent, 2);
        assert_eq!(monitor.current_max_concurrent, 2);
    }

    #[test]
    fn test_resource_monitor_check_and_adjust() {
        let config = test_batch_config();
        let mut monitor = ResourceMonitor::new(&config);

        // 第一次检查应返回当前并发数
        let result = monitor.check_and_adjust();
        assert!(result >= 1, "check_and_adjust should return at least 1");
    }

    #[test]
    fn test_resource_monitor_memory_usage() {
        let config = test_batch_config();
        let mut monitor = ResourceMonitor::new(&config);

        let usage = monitor.memory_usage_percent();
        assert!(usage >= 0.0, "Memory usage should be non-negative");
        assert!(usage <= 100.0, "Memory usage should be <= 100");
    }

    #[test]
    fn test_resource_monitor_reset() {
        let config = test_batch_config();
        let mut monitor = ResourceMonitor::new(&config);

        monitor.reset();
        assert_eq!(
            monitor.current_max_concurrent(),
            config.max_concurrent,
            "Reset should restore initial max concurrent"
        );
    }

    // ── BatchQueue 测试 ────────────────────────────────

    #[test]
    fn test_batch_queue_new() {
        let config = test_config();
        let queue = BatchQueue::new(&config);

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert!(!queue.paused);
    }

    #[test]
    fn test_batch_queue_add_job() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        let job = BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config,
            Priority::Normal,
        );
        queue.add_job(job);

        assert_eq!(queue.len(), 1);
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.running_count(), 0);
    }

    #[test]
    fn test_batch_queue_next_job_fifo() {
        let config = test_config();
        let mut batch_config = test_batch_config();
        batch_config.enable_priority = false;
        let mut queue = BatchQueue::with_config(batch_config);

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config.clone(),
            Priority::Low,
        ));
        queue.add_job(BatchJob::new(
            "job-2".into(),
            PathBuf::from("input2.mp4"),
            PathBuf::from("output2.mp4"),
            config,
            Priority::High,
        ));

        // FIFO 顺序：先添加的先出队
        let next = queue.next_job().expect("Should return a job");
        assert_eq!(next.id, "job-1");
        assert_eq!(queue.running_count(), 1);
    }

    #[test]
    fn test_batch_queue_next_job_priority() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config.clone(),
            Priority::Low,
        ));
        queue.add_job(BatchJob::new(
            "job-2".into(),
            PathBuf::from("input2.mp4"),
            PathBuf::from("output2.mp4"),
            config.clone(),
            Priority::High,
        ));
        queue.add_job(BatchJob::new(
            "job-3".into(),
            PathBuf::from("input3.mp4"),
            PathBuf::from("output3.mp4"),
            config,
            Priority::Normal,
        ));

        // 优先级顺序：High > Normal > Low
        let next = queue.next_job().expect("Should return a job");
        assert_eq!(next.id, "job-2");
        assert_eq!(queue.running_count(), 1);

        let next = queue.next_job().expect("Should return a job");
        assert_eq!(next.id, "job-3");
        assert_eq!(queue.running_count(), 2);

        // 并发数已达上限（max_concurrent=2），应返回 None
        assert!(queue.next_job().is_none());
    }

    #[test]
    fn test_batch_queue_concurrency_limit() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        // max_concurrent = 2
        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config.clone(),
            Priority::Normal,
        ));
        queue.add_job(BatchJob::new(
            "job-2".into(),
            PathBuf::from("input2.mp4"),
            PathBuf::from("output2.mp4"),
            config.clone(),
            Priority::Normal,
        ));
        queue.add_job(BatchJob::new(
            "job-3".into(),
            PathBuf::from("input3.mp4"),
            PathBuf::from("output3.mp4"),
            config,
            Priority::Normal,
        ));

        // 前两个任务可以出队
        let job1 = queue.next_job().expect("Should return job 1");
        assert_eq!(job1.id, "job-1");

        let job2 = queue.next_job().expect("Should return job 2");
        assert_eq!(job2.id, "job-2");

        // 第三个任务因并发限制不能出队
        assert!(queue.next_job().is_none());

        // 完成一个任务后，第三个可以出队
        queue.complete_job("job-1");
        let job3 = queue
            .next_job()
            .expect("Should return job 3 after job 1 completed");
        assert_eq!(job3.id, "job-3");
    }

    #[test]
    fn test_batch_queue_pause_resume() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config,
            Priority::Normal,
        ));

        // 暂停队列
        queue.pause();
        assert!(
            queue.next_job().is_none(),
            "Paused queue should not return jobs"
        );

        // 恢复队列
        queue.resume();
        let job = queue.next_job().expect("Should return a job after resume");
        assert_eq!(job.id, "job-1");
    }

    #[test]
    fn test_batch_queue_cancel_job() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config,
            Priority::Normal,
        ));

        queue.cancel_job("job-1");
        assert_eq!(queue.get_status("job-1"), Some(JobStatus::Cancelled));
        assert!(
            queue.next_job().is_none(),
            "Cancelled job should not be returned"
        );
    }

    #[test]
    fn test_batch_queue_cancel_all() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config.clone(),
            Priority::Normal,
        ));
        queue.add_job(BatchJob::new(
            "job-2".into(),
            PathBuf::from("input2.mp4"),
            PathBuf::from("output2.mp4"),
            config,
            Priority::Normal,
        ));

        queue.cancel_all();

        assert_eq!(queue.get_status("job-1"), Some(JobStatus::Cancelled));
        assert_eq!(queue.get_status("job-2"), Some(JobStatus::Cancelled));
    }

    #[test]
    fn test_batch_queue_complete_job() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config,
            Priority::Normal,
        ));

        // 先出队任务（标记为 Running）
        let _ = queue.next_job();

        // 完成任务
        queue.complete_job("job-1");
        assert_eq!(queue.get_status("job-1"), Some(JobStatus::Completed));
        assert_eq!(queue.completed_count(), 1);
    }

    #[test]
    fn test_batch_queue_fail_job() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config,
            Priority::Normal,
        ));

        queue.fail_job("job-1", "Processing failed".into());
        assert_eq!(queue.get_status("job-1"), Some(JobStatus::Failed));
        assert_eq!(queue.failed_count(), 1);
    }

    #[test]
    fn test_batch_queue_get_all_statuses() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config.clone(),
            Priority::Normal,
        ));
        queue.add_job(BatchJob::new(
            "job-2".into(),
            PathBuf::from("input2.mp4"),
            PathBuf::from("output2.mp4"),
            config,
            Priority::Normal,
        ));

        let statuses = queue.get_all_statuses();
        assert_eq!(statuses.len(), 2);
        assert_eq!(statuses.get("job-1"), Some(&JobStatus::Queued));
        assert_eq!(statuses.get("job-2"), Some(&JobStatus::Queued));
    }

    #[test]
    fn test_batch_queue_update_progress() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config,
            Priority::Normal,
        ));

        queue.update_progress("job-1", 0.5);
        assert!((queue.get_progress("job-1").unwrap() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_queue_estimate_remaining_time() {
        let config = test_config();
        let mut queue = BatchQueue::with_config(test_batch_config());

        // 无历史数据时返回 None
        assert!(queue.estimate_remaining_time().is_none());

        // 添加任务并完成
        queue.add_job(BatchJob::new(
            "job-1".into(),
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            config.clone(),
            Priority::Normal,
        ));
        let _ = queue.next_job();
        queue.complete_job("job-1");

        // 添加更多待处理任务
        queue.add_job(BatchJob::new(
            "job-2".into(),
            PathBuf::from("input2.mp4"),
            PathBuf::from("output2.mp4"),
            config,
            Priority::Normal,
        ));

        // 有历史数据时应返回预估时间
        let estimate = queue.estimate_remaining_time();
        assert!(
            estimate.is_some(),
            "Should estimate remaining time with history"
        );
    }

    // ── estimate_remaining_time 辅助函数测试 ──────────

    #[test]
    fn test_estimate_remaining_time_empty() {
        let result = estimate_remaining_time(&[], 5, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_estimate_remaining_time_with_data() {
        let durations = vec![Duration::from_secs(60), Duration::from_secs(120)];
        let result = estimate_remaining_time(&durations, 4, 2);
        assert!(result.is_some());

        // 平均 90 秒/任务，4 个待处理，2 并发 → 90 * 4 / 2 = 180 秒
        let expected = Duration::from_secs(180);
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_estimate_remaining_time_zero_concurrent() {
        let durations = vec![Duration::from_secs(60)];
        // 并发数为 0 时应使用 1
        let result = estimate_remaining_time(&durations, 2, 0);
        assert!(result.is_some());
        // 60 * 2 / 1 = 120 秒
        assert_eq!(result.unwrap(), Duration::from_secs(120));
    }
}
