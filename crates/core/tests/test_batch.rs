//! 集成测试：批量处理优化
//!
//! 验证 BatchQueue 的优先级调度、并发控制和资源监控逻辑。

use std::path::PathBuf;

use vt_core::batch::{BatchJob, BatchQueue, JobStatus, Priority};
use vt_core::config::{BatchConfig, Config};

/// 创建测试用配置
fn test_config() -> Config {
    Config::default()
}

/// 创建小并发配置
fn small_batch_config() -> BatchConfig {
    BatchConfig {
        max_concurrent: 2,
        memory_threshold: 95.0,
        enable_priority: true,
    }
}

/// 创建测试任务
fn make_job(id: &str, priority: Priority) -> BatchJob {
    BatchJob::new(
        id.into(),
        PathBuf::from(format!("input{id}.mp4")),
        PathBuf::from(format!("output{id}.mp4")),
        test_config(),
        priority,
    )
}

/// 验证优先级调度：High 优先于 Normal 和 Low
#[test]
fn test_priority_scheduling() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    queue.add_job(make_job("low-1", Priority::Low));
    queue.add_job(make_job("high-1", Priority::High));
    queue.add_job(make_job("normal-1", Priority::Normal));
    queue.add_job(make_job("high-2", Priority::High));

    // High 优先级先出队
    let job1 = queue.next_job().expect("Should return a job");
    assert_eq!(job1.id, "high-1");

    let job2 = queue.next_job().expect("Should return a job");
    assert_eq!(job2.id, "high-2");

    // 完成一个后才能继续出队（并发=2）
    queue.complete_job("high-1");

    let job3 = queue.next_job().expect("Should return a job");
    assert_eq!(job3.id, "normal-1");

    queue.complete_job("high-2");

    let job4 = queue.next_job().expect("Should return a job");
    assert_eq!(job4.id, "low-1");
}

/// 验证并发限制
#[test]
fn test_concurrency_limit() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    for i in 0..5 {
        queue.add_job(make_job(&format!("job-{i}"), Priority::Normal));
    }

    // max_concurrent = 2
    let j1 = queue.next_job().expect("Should return job");
    let j2 = queue.next_job().expect("Should return job");
    assert!(queue.next_job().is_none(), "Should hit concurrency limit");

    // 完成一个后可以出队下一个
    queue.complete_job(&j1.id);
    let j3 = queue
        .next_job()
        .expect("Should return job after completion");
    assert_ne!(j3.id, j1.id);
    assert_ne!(j3.id, j2.id);
}

/// 验证暂停和恢复
#[test]
fn test_pause_resume() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    queue.add_job(make_job("job-1", Priority::Normal));

    queue.pause();
    assert!(
        queue.next_job().is_none(),
        "Paused queue should not return jobs"
    );

    queue.resume();
    let job = queue.next_job().expect("Should return job after resume");
    assert_eq!(job.id, "job-1");
}

/// 验证取消单个任务
#[test]
fn test_cancel_single_job() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    queue.add_job(make_job("job-1", Priority::Normal));
    queue.add_job(make_job("job-2", Priority::Normal));

    queue.cancel_job("job-1");
    assert_eq!(queue.get_status("job-1"), Some(JobStatus::Cancelled));

    // job-2 仍可出队
    let job = queue.next_job().expect("Should return job-2");
    assert_eq!(job.id, "job-2");
}

/// 验证取消所有任务
#[test]
fn test_cancel_all() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    queue.add_job(make_job("job-1", Priority::Normal));
    queue.add_job(make_job("job-2", Priority::Normal));
    queue.add_job(make_job("job-3", Priority::Normal));

    queue.cancel_all();

    for i in 1..=3 {
        assert_eq!(
            queue.get_status(&format!("job-{i}")),
            Some(JobStatus::Cancelled)
        );
    }
}

/// 验证任务进度更新
#[test]
fn test_progress_tracking() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    queue.add_job(make_job("job-1", Priority::Normal));

    queue.update_progress("job-1", 0.3);
    assert!((queue.get_progress("job-1").unwrap() - 0.3).abs() < f64::EPSILON);

    queue.update_progress("job-1", 0.7);
    assert!((queue.get_progress("job-1").unwrap() - 0.7).abs() < f64::EPSILON);
}

/// 验证任务失败处理
#[test]
fn test_job_failure() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    queue.add_job(make_job("job-1", Priority::Normal));
    let _ = queue.next_job();

    queue.fail_job("job-1", "ASR engine crashed".into());

    assert_eq!(queue.get_status("job-1"), Some(JobStatus::Failed));
    assert_eq!(queue.failed_count(), 1);
    assert_eq!(queue.completed_count(), 0);
}

/// 验证全部状态查询
#[test]
fn test_get_all_statuses() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    queue.add_job(make_job("job-1", Priority::Normal));
    queue.add_job(make_job("job-2", Priority::Normal));
    queue.add_job(make_job("job-3", Priority::Normal));

    let _ = queue.next_job(); // job-1 -> Running
    queue.cancel_job("job-2"); // job-2 -> Cancelled

    let statuses = queue.get_all_statuses();
    assert_eq!(statuses.len(), 3);
    assert_eq!(statuses.get("job-1"), Some(&JobStatus::Running));
    assert_eq!(statuses.get("job-2"), Some(&JobStatus::Cancelled));
    assert_eq!(statuses.get("job-3"), Some(&JobStatus::Queued));
}

/// 验证 FIFO 模式（禁用优先级）
#[test]
fn test_fifo_mode() {
    let mut batch_config = small_batch_config();
    batch_config.enable_priority = false;
    let mut queue = BatchQueue::with_config(batch_config);

    queue.add_job(make_job("job-1", Priority::Low));
    queue.add_job(make_job("job-2", Priority::High));

    // FIFO：先添加的先出队，不管优先级
    let job = queue.next_job().expect("Should return job");
    assert_eq!(job.id, "job-1");
}

/// 验证预估剩余时间
#[test]
fn test_estimate_remaining() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    // 无历史数据时返回 None
    assert!(queue.estimate_remaining_time().is_none());

    // 完成一个任务
    queue.add_job(make_job("job-1", Priority::Normal));
    let _ = queue.next_job();
    queue.complete_job("job-1");

    // 添加待处理任务
    queue.add_job(make_job("job-2", Priority::Normal));
    queue.add_job(make_job("job-3", Priority::Normal));

    // 有历史数据时应返回预估时间
    let estimate = queue.estimate_remaining_time();
    assert!(estimate.is_some(), "Should estimate with history");
}

/// 验证队列统计信息
#[test]
fn test_queue_stats() {
    let mut queue = BatchQueue::with_config(small_batch_config());

    queue.add_job(make_job("job-1", Priority::Normal));
    queue.add_job(make_job("job-2", Priority::Normal));
    queue.add_job(make_job("job-3", Priority::Normal));

    assert_eq!(queue.len(), 3);
    assert_eq!(queue.pending_count(), 3);
    assert_eq!(queue.running_count(), 0);
    assert_eq!(queue.completed_count(), 0);

    let _ = queue.next_job();
    assert_eq!(queue.running_count(), 1);
    assert_eq!(queue.pending_count(), 2);

    queue.complete_job("job-1");
    assert_eq!(queue.completed_count(), 1);
    assert_eq!(queue.running_count(), 0);
}
