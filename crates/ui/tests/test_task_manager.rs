//! 任务管理器（TaskManager）单元测试
//!
//! 遵循 TDD 红-绿-重构流程：先编写测试（红），再实现功能（绿）。
//! 本文件测试任务管理器的核心逻辑，不依赖 Tauri 运行时。

use std::sync::atomic::Ordering;
use std::sync::Arc;

use vt_ui::task_manager::{ProgressInfo, TaskManager, TaskStatus};

// ─── 创建与基本查询 ───────────────────────────────────────

/// 验证 `TaskManager::new()` 创建空管理器。
#[test]
fn test_new_task_manager_is_empty() {
    let mgr = TaskManager::new();
    assert!(mgr.list_tasks().is_empty());
}

/// 验证创建任务后返回唯一 ID，且可通过 ID 查询。
#[test]
fn test_create_task_returns_unique_id() {
    let mgr = TaskManager::new();
    let id1 = mgr.create_task("/tmp/video1.mp4".to_string(), None);
    let id2 = mgr.create_task("/tmp/video2.mp4".to_string(), None);

    assert!(!id1.is_empty());
    assert!(!id2.is_empty());
    assert_ne!(id1, id2, "Task IDs must be unique");

    let info1 = mgr.get_progress(&id1).expect("Task 1 should exist");
    assert_eq!(info1.task_id, id1);
    assert_eq!(info1.status, TaskStatus::Pending);
    assert!((info1.progress - 0.0).abs() < f64::EPSILON);
}

/// 验证 `list_tasks` 返回所有任务。
#[test]
fn test_list_tasks() {
    let mgr = TaskManager::new();
    let _id1 = mgr.create_task("/tmp/a.mp4".to_string(), None);
    let _id2 = mgr.create_task("/tmp/b.mp4".to_string(), None);

    let tasks = mgr.list_tasks();
    assert_eq!(tasks.len(), 2);
}

/// 验证查询不存在的任务 ID 返回 `None`。
#[test]
fn test_get_progress_nonexistent_returns_none() {
    let mgr = TaskManager::new();
    assert!(mgr.get_progress("nonexistent-id").is_none());
}

// ─── 进度更新 ───────────────────────────────────────────

/// 验证 `update_progress` 更新阶段和百分比。
#[test]
fn test_update_progress() {
    let mgr = TaskManager::new();
    let id = mgr.create_task("/tmp/test.mp4".to_string(), None);

    mgr.update_progress(
        &id,
        TaskStatus::Running,
        0.5,
        "Translating segment 3/10".to_string(),
    );

    let info = mgr.get_progress(&id).expect("Task should exist");
    assert_eq!(info.status, TaskStatus::Running);
    assert!((info.progress - 0.5).abs() < f64::EPSILON);
    assert_eq!(info.stage, "Translating segment 3/10");
}

/// 验证 `mark_completed` 将状态设为 `Completed` 且进度为 1.0。
#[test]
fn test_mark_completed() {
    let mgr = TaskManager::new();
    let id = mgr.create_task("/tmp/test.mp4".to_string(), None);

    mgr.mark_completed(&id);

    let info = mgr.get_progress(&id).expect("Task should exist");
    assert_eq!(info.status, TaskStatus::Completed);
    assert!((info.progress - 1.0).abs() < f64::EPSILON);
}

/// 验证 `mark_failed` 设置错误信息。
#[test]
fn test_mark_failed() {
    let mgr = TaskManager::new();
    let id = mgr.create_task("/tmp/test.mp4".to_string(), None);

    mgr.mark_failed(&id, "FFmpeg error: exit code 1".to_string());

    let info = mgr.get_progress(&id).expect("Task should exist");
    assert_eq!(info.status, TaskStatus::Failed);
    assert!(info.error.is_some());
    assert_eq!(info.error.as_deref(), Some("FFmpeg error: exit code 1"));
}

// ─── 取消 ───────────────────────────────────────────────

/// 验证 `cancel_task` 设置取消标志并更新状态。
#[test]
fn test_cancel_task() {
    let mgr = TaskManager::new();
    let id = mgr.create_task("/tmp/test.mp4".to_string(), None);

    let cancel_flag = mgr.get_cancel_flag(&id).expect("Cancel flag should exist");
    assert!(!cancel_flag.load(Ordering::Relaxed));

    mgr.cancel_task(&id).expect("Cancel should succeed");

    assert!(
        cancel_flag.load(Ordering::Relaxed),
        "Cancel flag must be set"
    );

    let info = mgr.get_progress(&id).expect("Task should exist");
    assert_eq!(info.status, TaskStatus::Cancelled);
}

/// 验证取消不存在的任务返回错误。
#[test]
fn test_cancel_nonexistent_task() {
    let mgr = TaskManager::new();
    let result = mgr.cancel_task("nonexistent-id");
    assert!(result.is_err());
}

// ─── 清理 ───────────────────────────────────────────────

/// 验证 `remove_task` 从管理器中删除任务。
#[test]
fn test_remove_task() {
    let mgr = TaskManager::new();
    let id = mgr.create_task("/tmp/test.mp4".to_string(), None);

    assert!(mgr.remove_task(&id).is_ok());
    assert!(mgr.get_progress(&id).is_none());
    assert!(mgr.list_tasks().is_empty());
}

/// 验证删除不存在的任务返回错误。
#[test]
fn test_remove_nonexistent_task() {
    let mgr = TaskManager::new();
    assert!(mgr.remove_task("nonexistent-id").is_err());
}

// ─── 并发安全 ───────────────────────────────────────────

/// 验证多个线程同时创建任务时 ID 不冲突。
#[test]
fn test_concurrent_task_creation() {
    use std::thread;

    let mgr = Arc::new(TaskManager::new());
    let mut handles = Vec::new();

    for i in 0..8 {
        let mgr_clone = Arc::clone(&mgr);
        let handle =
            thread::spawn(move || mgr_clone.create_task(format!("/tmp/video_{i}.mp4"), None));
        handles.push(handle);
    }

    let mut ids = Vec::new();
    for handle in handles {
        ids.push(handle.join().expect("Thread panicked"));
    }

    // 所有 ID 应唯一
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len(), "All task IDs must be unique");

    assert_eq!(mgr.list_tasks().len(), 8);
}

/// 验证多线程同时更新不同任务的进度。
#[test]
fn test_concurrent_progress_update() {
    use std::thread;

    let mgr = Arc::new(TaskManager::new());
    let mut ids = Vec::new();
    for i in 0..4 {
        ids.push(mgr.create_task(format!("/tmp/video_{i}.mp4"), None));
    }

    let mut handles = Vec::new();
    for (idx, id) in ids.iter().enumerate() {
        let mgr_clone = Arc::clone(&mgr);
        let id_clone = id.clone();
        let handle = thread::spawn(move || {
            mgr_clone.update_progress(
                &id_clone,
                TaskStatus::Running,
                0.25 * (idx as f64 + 1.0),
                format!("Stage {idx}"),
            );
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    for (idx, id) in ids.iter().enumerate() {
        let info = mgr.get_progress(id).expect("Task should exist");
        assert_eq!(info.status, TaskStatus::Running);
        assert!((info.progress - 0.25 * (idx as f64 + 1.0)).abs() < 0.001);
    }
}

// ─── ProgressInfo 序列化 ────────────────────────────────

/// 验证 `ProgressInfo` 可正确序列化为 JSON。
#[test]
fn test_progress_info_serialization() {
    let info = ProgressInfo {
        task_id: "test-123".to_string(),
        status: TaskStatus::Running,
        progress: 0.75,
        stage: "TTS synthesis".to_string(),
        error: None,
    };

    let json = serde_json::to_string(&info).expect("Serialization failed");
    assert!(json.contains("test-123"));
    assert!(json.contains("Running"));
    assert!(json.contains("0.75"));

    let deserialized: ProgressInfo = serde_json::from_str(&json).expect("Deserialization failed");
    assert_eq!(deserialized.task_id, "test-123");
    assert_eq!(deserialized.status, TaskStatus::Running);
    assert!((deserialized.progress - 0.75).abs() < f64::EPSILON);
}

/// 验证 `ProgressInfo` 带错误的序列化。
#[test]
fn test_progress_info_with_error_serialization() {
    let info = ProgressInfo {
        task_id: "err-456".to_string(),
        status: TaskStatus::Failed,
        progress: 0.3,
        stage: "ASR transcription".to_string(),
        error: Some("Model load failed".to_string()),
    };

    let json = serde_json::to_string(&info).expect("Serialization failed");
    assert!(json.contains("Model load failed"));

    let deserialized: ProgressInfo = serde_json::from_str(&json).expect("Deserialization failed");
    assert_eq!(deserialized.error, Some("Model load failed".to_string()));
}

// ─── TaskStatus 枚举 ────────────────────────────────────

/// 验证 `TaskStatus` 所有变体可序列化。
#[test]
fn test_task_status_variants_serializable() {
    let statuses = vec![
        TaskStatus::Pending,
        TaskStatus::Running,
        TaskStatus::Completed,
        TaskStatus::Failed,
        TaskStatus::Cancelled,
    ];

    for status in &statuses {
        let json = serde_json::to_string(status).expect("Serialization failed");
        let deserialized: TaskStatus = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(*status, deserialized);
    }
}

// ─── 取消标志集成 ────────────────────────────────────────

/// 验证取消标志可在异步上下文中检查。
#[test]
fn test_cancel_flag_checked_in_loop() {
    let mgr = Arc::new(TaskManager::new());
    let id = mgr.create_task("/tmp/test.mp4".to_string(), None);

    let cancel_flag = mgr.get_cancel_flag(&id).expect("Cancel flag should exist");

    // 模拟后台任务检查取消标志
    let flag_clone = Arc::clone(&cancel_flag);
    let handle = std::thread::spawn(move || {
        for _ in 0..100 {
            if flag_clone.load(Ordering::Relaxed) {
                return "cancelled";
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        "completed"
    });

    std::thread::sleep(std::time::Duration::from_millis(5));
    mgr.cancel_task(&id).expect("Cancel should succeed");

    let result = handle.join().expect("Thread panicked");
    assert_eq!(result, "cancelled");
}
