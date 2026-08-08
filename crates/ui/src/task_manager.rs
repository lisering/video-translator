//! 任务管理器模块
//!
//! 提供异步视频处理任务的生命周期管理，包括：
//! - 创建任务并分配唯一 ID
//! - 实时更新进度（百分比、阶段、错误）
//! - 取消正在运行的任务
//! - 查询任务状态和历史记录
//!
//! # 线程安全
//! 使用 `Arc<Mutex<HashMap<...>>>` 内部可变性，支持多线程并发访问。
//! 每个任务持有一个 `Arc<AtomicBool>` 取消标志，后台任务可轮询检查。
//!
//! # 示例
//! ```no_run
//! use vt_ui::task_manager::{TaskManager, TaskStatus};
//!
//! let mgr = TaskManager::new();
//! let id = mgr.create_task("/tmp/video.mp4".to_string(), None);
//! mgr.update_progress(&id, TaskStatus::Running, 0.5, "Processing".to_string());
//! let info = mgr.get_progress(&id).expect("task exists");
//! println!("Progress: {:.0}%", info.progress * 100.0);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

// ─── 数据结构 ─────────────────────────────────────────────

/// 任务状态枚举
///
/// 表示视频处理任务的生命周期状态。
/// 状态流转：`Pending` → `Running` → `Completed` / `Failed` / `Cancelled`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// 待处理（任务已创建，尚未开始执行）
    Pending,
    /// 正在运行
    Running,
    /// 已完成（处理成功）
    Completed,
    /// 已失败（处理出错）
    Failed,
    /// 已取消（用户主动取消）
    Cancelled,
}

/// 任务进度信息
///
/// 包含任务的当前状态、进度百分比、阶段描述和错误信息。
/// 通过 Tauri 事件系统或轮询返回给前端。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    /// 任务唯一标识符
    pub task_id: String,
    /// 当前状态
    pub status: TaskStatus,
    /// 进度百分比（0.0 ~ 1.0）
    pub progress: f64,
    /// 当前处理阶段描述（如 "ASR transcription"）
    pub stage: String,
    /// 错误信息（仅在 `Failed` 状态下有值）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// 任务条目（内部使用）
///
/// 每个任务持有一个取消标志和进度信息，支持后台任务轮询取消状态。
struct TaskEntry {
    /// 取消标志，后台任务通过 `Arc<AtomicBool>` 检查是否被取消
    cancel_flag: Arc<AtomicBool>,
    /// 可变进度信息
    info: ProgressInfo,
}

impl TaskEntry {
    /// 创建新任务条目
    fn new(task_id: &str) -> Self {
        Self {
            cancel_flag: Arc::new(AtomicBool::new(false)),
            info: ProgressInfo {
                task_id: task_id.to_string(),
                status: TaskStatus::Pending,
                progress: 0.0,
                stage: "Queued".to_string(),
                error: None,
            },
        }
    }
}

// ─── TaskManager ─────────────────────────────────────────

/// 异步任务管理器
///
/// 管理视频处理任务的生命周期，支持创建、查询、更新、取消和删除。
/// 使用 `Arc<Mutex<HashMap<...>>>` 实现线程安全。
///
/// # 示例
/// ```no_run
/// use vt_ui::task_manager::TaskManager;
///
/// let mgr = TaskManager::new();
/// let id = mgr.create_task("/tmp/video.mp4".to_string(), None);
/// ```
pub struct TaskManager {
    /// 任务映射表：task_id → TaskEntry
    tasks: Arc<Mutex<HashMap<String, TaskEntry>>>,
}

impl TaskManager {
    /// 创建空的任务管理器
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 创建新任务并返回任务 ID
    ///
    /// # 参数
    /// - `input_path`: 输入视频文件路径
    /// - `output_path`: 输出视频文件路径（可选，为 `None` 时使用默认路径）
    ///
    /// # 返回
    /// 新分配的任务 ID（UUID v4 格式）
    pub fn create_task(&self, input_path: String, output_path: Option<String>) -> String {
        let task_id = uuid::Uuid::new_v4().to_string();

        let entry = TaskEntry::new(&task_id);
        tracing::info!(
            "Task created: id={}, input={}, output={:?}",
            task_id,
            input_path,
            output_path
        );

        let mut tasks = self
            .tasks
            .lock()
            .expect("TaskManager mutex poisoned (create_task)");
        tasks.insert(task_id.clone(), entry);

        task_id
    }

    /// 获取任务的进度信息
    ///
    /// # 参数
    /// - `task_id`: 任务 ID
    ///
    /// # 返回
    /// 进度信息的克隆，如果任务不存在返回 `None`。
    pub fn get_progress(&self, task_id: &str) -> Option<ProgressInfo> {
        let tasks = self
            .tasks
            .lock()
            .expect("TaskManager mutex poisoned (get_progress)");
        tasks.get(task_id).map(|e| e.info.clone())
    }

    /// 获取任务的取消标志
    ///
    /// 后台任务持有一个 `Arc<AtomicBool>` 引用，通过轮询 `load` 检查是否被取消。
    ///
    /// # 参数
    /// - `task_id`: 任务 ID
    ///
    /// # 返回
    /// 取消标志的 `Arc` 引用，如果任务不存在返回 `None`。
    pub fn get_cancel_flag(&self, task_id: &str) -> Option<Arc<AtomicBool>> {
        let tasks = self
            .tasks
            .lock()
            .expect("TaskManager mutex poisoned (get_cancel_flag)");
        tasks.get(task_id).map(|e| Arc::clone(&e.cancel_flag))
    }

    /// 更新任务进度
    ///
    /// # 参数
    /// - `task_id`: 任务 ID
    /// - `status`: 新状态
    /// - `progress`: 进度百分比（0.0 ~ 1.0）
    /// - `stage`: 当前阶段描述
    pub fn update_progress(&self, task_id: &str, status: TaskStatus, progress: f64, stage: String) {
        let mut tasks = self
            .tasks
            .lock()
            .expect("TaskManager mutex poisoned (update_progress)");
        if let Some(entry) = tasks.get_mut(task_id) {
            entry.info.status = status;
            entry.info.progress = progress;
            entry.info.stage = stage;
            tracing::debug!(
                "Task {} progress: {:.0}% - {}",
                task_id,
                progress * 100.0,
                entry.info.stage
            );
        }
    }

    /// 标记任务为已完成
    ///
    /// 将状态设为 `Completed`，进度设为 1.0。
    ///
    /// # 参数
    /// - `task_id`: 任务 ID
    pub fn mark_completed(&self, task_id: &str) {
        self.update_progress(task_id, TaskStatus::Completed, 1.0, "Done".to_string());
    }

    /// 标记任务为失败
    ///
    /// 将状态设为 `Failed` 并记录错误信息。
    ///
    /// # 参数
    /// - `task_id`: 任务 ID
    /// - `error`: 错误描述
    pub fn mark_failed(&self, task_id: &str, error: String) {
        let mut tasks = self
            .tasks
            .lock()
            .expect("TaskManager mutex poisoned (mark_failed)");
        if let Some(entry) = tasks.get_mut(task_id) {
            entry.info.status = TaskStatus::Failed;
            entry.info.error = Some(error.clone());
            entry.info.stage = "Failed".to_string();
            tracing::error!("Task {} failed: {}", task_id, error);
        }
    }

    /// 取消任务
    ///
    /// 设置取消标志，后台任务应在下次检查时退出。
    /// 同时更新任务状态为 `Cancelled`。
    ///
    /// # 参数
    /// - `task_id`: 任务 ID
    ///
    /// # 返回
    /// `Ok(())` 如果取消成功，`Err` 如果任务不存在。
    pub fn cancel_task(&self, task_id: &str) -> Result<(), String> {
        let mut tasks = self
            .tasks
            .lock()
            .expect("TaskManager mutex poisoned (cancel_task)");
        let entry = tasks
            .get_mut(task_id)
            .ok_or_else(|| format!("Task not found: {task_id}"))?;

        entry.cancel_flag.store(true, Ordering::Relaxed);
        entry.info.status = TaskStatus::Cancelled;
        entry.info.stage = "Cancelled".to_string();
        tracing::info!("Task {} cancelled", task_id);
        Ok(())
    }

    /// 删除任务
    ///
    /// 从管理器中移除任务记录。通常在任务完成后清理。
    ///
    /// # 参数
    /// - `task_id`: 任务 ID
    ///
    /// # 返回
    /// `Ok(())` 如果删除成功，`Err` 如果任务不存在。
    pub fn remove_task(&self, task_id: &str) -> Result<(), String> {
        let mut tasks = self
            .tasks
            .lock()
            .expect("TaskManager mutex poisoned (remove_task)");
        tasks
            .remove(task_id)
            .map(|_| ())
            .ok_or_else(|| format!("Task not found: {task_id}"))
    }

    /// 列出所有任务的进度信息
    ///
    /// # 返回
    /// 所有任务的 `ProgressInfo` 列表。
    pub fn list_tasks(&self) -> Vec<ProgressInfo> {
        let tasks = self
            .tasks
            .lock()
            .expect("TaskManager mutex poisoned (list_tasks)");
        tasks.values().map(|e| e.info.clone()).collect()
    }

    /// 克隆任务管理器句柄
    ///
    /// 由于内部使用 `Arc`，克隆是廉价的引用计数操作。
    /// 用于将句柄传递给后台异步任务。
    #[must_use]
    pub fn clone_handle(&self) -> TaskManager {
        TaskManager {
            tasks: Arc::clone(&self.tasks),
        }
    }
}

impl Default for TaskManager {
    /// 返回空的任务管理器
    fn default() -> Self {
        Self::new()
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_empty() {
        let mgr = TaskManager::default();
        assert!(mgr.list_tasks().is_empty());
    }

    #[test]
    fn test_create_and_get() {
        let mgr = TaskManager::new();
        let id = mgr.create_task("/tmp/test.mp4".to_string(), None);
        let info = mgr.get_progress(&id).expect("Task should exist");
        assert_eq!(info.status, TaskStatus::Pending);
        assert_eq!(info.stage, "Queued");
    }

    #[test]
    fn test_mark_completed_sets_progress_to_one() {
        let mgr = TaskManager::new();
        let id = mgr.create_task("/tmp/test.mp4".to_string(), None);
        mgr.mark_completed(&id);
        let info = mgr.get_progress(&id).expect("Task should exist");
        assert_eq!(info.status, TaskStatus::Completed);
        assert!((info.progress - 1.0).abs() < f64::EPSILON);
    }
}
