//! 集成测试：断点续传
//!
//! 验证检查点的保存、加载、恢复和清理逻辑。

use std::path::PathBuf;

use chrono::Utc;
use tempfile::TempDir;
use vt_core::checkpoint::{Checkpoint, CheckpointManager, ProcessingStage};
use vt_core::models::segment::Segment;

/// 创建测试用检查点管理器（使用临时目录）
fn test_manager() -> (CheckpointManager, TempDir) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let manager = CheckpointManager::with_dir(dir.path().to_path_buf());
    (manager, dir)
}

/// 创建已完成的 Segment
fn completed_segment(id: &str, audio_path: &str) -> Segment {
    let mut seg = Segment::new(id.into(), 0.0, 5.0, "Hello".into());
    seg.start_transcribing().expect("start");
    seg.finish_transcribing("你好".into()).expect("finish");
    seg.start_synthesizing().expect("synthesize");
    seg.finish_synthesizing(audio_path.into()).expect("done");
    seg
}

/// 验证完整的断点续传流程
#[test]
fn test_full_resume_workflow() {
    let (manager, dir) = test_manager();

    // 1. 初始状态：无检查点
    assert!(manager.resume("job-001").expect("resume").is_none());

    // 2. 处理过程中保存检查点（3 个 Segment 已完成）
    let audio1 = dir.path().join("audio1.wav");
    std::fs::write(&audio1, b"audio").expect("write audio");
    let audio2 = dir.path().join("audio2.wav");
    std::fs::write(&audio2, b"audio").expect("write audio");
    let audio3 = dir.path().join("audio3.wav");
    std::fs::write(&audio3, b"audio").expect("write audio");

    let mut cp = Checkpoint::new(
        "job-001".into(),
        PathBuf::from("/path/to/video.mp4"),
        ProcessingStage::Translate,
    );

    cp.add_segment(completed_segment("seg-0", audio1.to_str().unwrap()));
    cp.add_segment(completed_segment("seg-1", audio2.to_str().unwrap()));
    cp.add_segment(completed_segment("seg-2", audio3.to_str().unwrap()));

    manager.save(&cp).expect("save");

    // 3. 模拟中断后恢复
    let result = manager.resume("job-001").expect("resume");
    assert!(result.is_some());

    let (segments, next_index) = result.unwrap();
    assert_eq!(segments.len(), 3, "Should restore 3 segments");
    assert_eq!(next_index, 3, "Next index should be 3");

    // 4. 任务完成，清理检查点
    manager.delete("job-001").expect("delete");
    assert!(manager.load("job-001").expect("load").is_none());
}

/// 验证恢复时检测到音频文件缺失
#[test]
fn test_resume_with_missing_audio() {
    let (manager, _dir) = test_manager();

    let mut cp = Checkpoint::new(
        "job-001".into(),
        PathBuf::from("/path/to/video.mp4"),
        ProcessingStage::Tts,
    );

    // 添加两个 Segment，一个音频存在，一个不存在
    cp.add_segment(completed_segment("seg-0", "/nonexistent/audio0.wav"));
    cp.add_segment(completed_segment("seg-1", "/nonexistent/audio1.wav"));

    manager.save(&cp).expect("save");

    let result = manager.resume("job-001").expect("resume");
    assert!(result.is_some());

    let (segments, next_index) = result.unwrap();
    // 两个 Segment 的音频都不存在，应被移除
    assert!(
        segments.is_empty(),
        "Segments with missing audio should be removed"
    );
    assert_eq!(next_index, 0);
}

/// 验证检查点过期清理
#[test]
fn test_cleanup_expired_checkpoints() {
    let (manager, _dir) = test_manager();

    // 创建过期检查点
    let mut old_cp = Checkpoint::new(
        "old-job".into(),
        PathBuf::from("/path/to/old.mp4"),
        ProcessingStage::Asr,
    );
    old_cp.timestamp = Utc::now() - chrono::Duration::days(10);
    manager.save(&old_cp).expect("save old");

    // 创建未过期检查点
    let new_cp = Checkpoint::new(
        "new-job".into(),
        PathBuf::from("/path/to/new.mp4"),
        ProcessingStage::Asr,
    );
    manager.save(&new_cp).expect("save new");

    let cleaned = manager.cleanup_expired().expect("cleanup");
    assert_eq!(cleaned, 1, "Should clean up 1 expired checkpoint");

    assert!(!manager.checkpoint_path("old-job").exists());
    assert!(manager.checkpoint_path("new-job").exists());
}

/// 验证增量更新检查点
#[test]
fn test_incremental_update() {
    let (manager, _dir) = test_manager();

    // 第一次更新
    manager
        .update(
            "job-001",
            completed_segment("seg-0", "/tmp/a0.wav"),
            ProcessingStage::Tts,
        )
        .expect("update 1");

    // 第二次更新
    manager
        .update(
            "job-001",
            completed_segment("seg-1", "/tmp/a1.wav"),
            ProcessingStage::Tts,
        )
        .expect("update 2");

    // 第三次更新
    manager
        .update(
            "job-001",
            completed_segment("seg-2", "/tmp/a2.wav"),
            ProcessingStage::Compose,
        )
        .expect("update 3");

    let loaded = manager
        .load("job-001")
        .expect("load")
        .expect("checkpoint exists");
    assert_eq!(loaded.completed_count(), 3);
    assert_eq!(loaded.next_segment_index, 3);
    assert_eq!(loaded.current_stage, ProcessingStage::Compose);
}

/// 验证列出所有检查点
#[test]
fn test_list_checkpoints() {
    let (manager, _dir) = test_manager();

    assert!(manager.list_checkpoints().expect("list").is_empty());

    for i in 0..3 {
        let cp = Checkpoint::new(
            format!("job-{i:03}"),
            PathBuf::from(format!("/path/to/video{i}.mp4")),
            ProcessingStage::Asr,
        );
        manager.save(&cp).expect("save");
    }

    let list = manager.list_checkpoints().expect("list");
    assert_eq!(list.len(), 3);
}

/// 验证 JSON 序列化/反序列化往返
#[test]
fn test_checkpoint_json_roundtrip() {
    let mut cp = Checkpoint::new(
        "job-001".into(),
        PathBuf::from("/path/to/video.mp4"),
        ProcessingStage::Tts,
    );

    cp.add_segment(completed_segment("seg-0", "/tmp/audio0.wav"));
    cp.add_segment(completed_segment("seg-1", "/tmp/audio1.wav"));
    cp.update_stage(ProcessingStage::Compose);

    let json = cp.to_json().expect("to_json");
    let restored = Checkpoint::from_json(&json).expect("from_json");

    assert_eq!(restored.job_id, cp.job_id);
    assert_eq!(restored.video_path, cp.video_path);
    assert_eq!(restored.current_stage, cp.current_stage);
    assert_eq!(restored.next_segment_index, cp.next_segment_index);
    assert_eq!(restored.completed_count(), cp.completed_count());
}

/// 验证处理阶段枚举
#[test]
fn test_processing_stage() {
    assert_eq!(ProcessingStage::Asr.name(), "ASR");
    assert_eq!(ProcessingStage::Translate.name(), "Translate");
    assert_eq!(ProcessingStage::Tts.name(), "TTS");
    assert_eq!(ProcessingStage::Compose.name(), "Compose");

    assert_eq!(ProcessingStage::Asr.order(), 0);
    assert_eq!(ProcessingStage::Translate.order(), 1);
    assert_eq!(ProcessingStage::Tts.order(), 2);
    assert_eq!(ProcessingStage::Compose.order(), 3);
}

/// 验证禁用检查点时不保存
#[test]
fn test_disabled_checkpoint() {
    use vt_core::config::CheckpointConfig;

    let config = CheckpointConfig {
        enabled: false,
        dir: "/tmp/vt-test-disabled".into(),
        retention_days: 7,
    };
    let manager = CheckpointManager::new(&config);

    assert!(!manager.is_enabled());

    let cp = Checkpoint::new(
        "job-1".into(),
        PathBuf::from("/path/to/video.mp4"),
        ProcessingStage::Asr,
    );

    // 保存不应失败，但也不应写入文件
    manager.save(&cp).expect("save should not fail");
}

/// 验证自动创建目录
#[test]
fn test_creates_nested_directory() {
    let dir = TempDir::new().expect("temp dir");
    let nested = dir.path().join("a").join("b").join("c");

    let manager = CheckpointManager::with_dir(nested.clone());

    let cp = Checkpoint::new(
        "job-1".into(),
        PathBuf::from("/path/to/video.mp4"),
        ProcessingStage::Asr,
    );

    manager.save(&cp).expect("save should create dirs");
    assert!(nested.exists());
}
