//! Tauri 命令模块
//!
//! 定义暴露给前端调用的 Tauri 命令（`#[tauri::command]`）。
//!
//! # 命令列表
//! - [`process_video`][]: 启动视频处理任务，返回任务 ID
//! - [`get_progress`][]: 查询任务进度
//! - [`cancel_task`][]: 取消正在进行的任务
//! - [`list_all_tasks`][]: 列出所有任务
//! - [`load_config`][]: 加载配置（返回 JSON）
//! - [`save_config`][]: 保存配置
//! - [`probe_video`][]: 探测视频文件信息
//!
//! # 进度更新
//! 处理过程中通过 `app_handle.emit("task-progress", &info)` 发送全局事件，
//! 前端通过 `listen("task-progress", callback)` 接收实时更新。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use vt_core::asr::{WhisperConfig, WhisperEngine};
use vt_core::config::Config;
use vt_core::media::{probe_media, FfmpegAudioExtractor, FfmpegVideoComposer, VideoComposer};
use vt_core::pipeline::{Pipeline, PipelineBuilder, ProgressTracker};
use vt_core::translate::{LlamaCppBackend, LocalTranslationEngine, TerminologyManager};
use vt_core::tts::{SayEngine, TtsEngine};

use crate::task_manager::{ProgressInfo, TaskManager, TaskStatus};

// ─── 应用状态 ─────────────────────────────────────────────

/// 应用全局状态
///
/// 通过 `tauri::Builder::manage()` 注册，在命令中通过 `tauri::State` 访问。
pub struct AppState {
    /// 任务管理器
    pub task_manager: TaskManager,
    /// 配置文件路径
    pub config_path: PathBuf,
}

impl AppState {
    /// 创建应用状态
    ///
    /// # 参数
    /// - `config_path`: 配置文件存储路径
    #[must_use]
    pub fn new(config_path: PathBuf) -> Self {
        Self {
            task_manager: TaskManager::new(),
            config_path,
        }
    }
}

// ─── 视频探测结果 ─────────────────────────────────────────

/// 视频文件探测结果（返回给前端）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInfo {
    /// 文件路径
    pub path: String,
    /// 媒体时长（秒）
    pub duration: f64,
    /// 视频宽度（像素）
    pub width: Option<i32>,
    /// 视频高度（像素）
    pub height: Option<i32>,
    /// 视频编解码器
    pub video_codec: Option<String>,
    /// 音频编解码器
    pub audio_codec: Option<String>,
}

// ─── Tauri 命令 ──────────────────────────────────────────

/// 启动视频处理任务
///
/// 创建后台异步任务处理视频，立即返回任务 ID。
/// 处理进度通过 `task-progress` 事件推送到前端。
///
/// # 参数
/// - `input`: 输入视频文件路径
/// - `output`: 输出视频文件路径（可选）
/// - `config_json`: 配置 JSON 字符串（可选，为 `None` 时使用已保存的配置）
/// - `state`: Tauri 管理的应用状态
/// - `app_handle`: Tauri 应用句柄（用于发送事件）
///
/// # 返回
/// 任务 ID 字符串
///
/// # 错误
/// - 输入文件不存在
/// - 配置解析失败
#[tauri::command]
pub async fn process_video(
    input: String,
    output: Option<String>,
    config_json: Option<String>,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<String, String> {
    // 验证输入文件
    let input_path = PathBuf::from(&input);
    if !input_path.exists() {
        return Err(format!("Input file not found: {input}"));
    }

    // 加载配置
    let config = if let Some(ref json) = config_json {
        serde_json::from_str::<Config>(json)
            .map_err(|e| format!("Failed to parse config JSON: {e}"))?
    } else {
        load_config_from_file(&state.config_path)?
    };

    // 确定输出路径
    let output_path = output.unwrap_or_else(|| {
        let stem = input_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "output".to_string());
        let output_dir = Path::new(&config.output_dir);
        let _ = std::fs::create_dir_all(output_dir);
        output_dir
            .join(format!("{stem}_translated.mp4"))
            .to_string_lossy()
            .to_string()
    });

    // 创建任务
    let task_id = state
        .task_manager
        .create_task(input.clone(), Some(output_path.clone()));

    // 获取取消标志
    let cancel_flag = state
        .task_manager
        .get_cancel_flag(&task_id)
        .ok_or_else(|| "Failed to get cancel flag".to_string())?;

    // 克隆共享数据到后台任务
    let task_manager = state.task_manager.clone_handle();
    let task_id_clone = task_id.clone();
    let app_handle_clone = app_handle.clone();

    // 启动后台处理任务
    tauri::async_runtime::spawn(async move {
        let result = run_video_processing(
            &input_path,
            &output_path,
            &config,
            &task_manager,
            &task_id_clone,
            &cancel_flag,
            &app_handle_clone,
        )
        .await;
        // input_path, output_path, config are moved into this closure

        if let Err(e) = result {
            tracing::error!("Task {} failed: {}", task_id_clone, e);
            task_manager.mark_failed(&task_id_clone, e.to_string());
            let info = task_manager
                .get_progress(&task_id_clone)
                .unwrap_or_else(|| ProgressInfo {
                    task_id: task_id_clone.clone(),
                    status: TaskStatus::Failed,
                    progress: 0.0,
                    stage: "Failed".to_string(),
                    error: Some(e.to_string()),
                });
            let _ = app_handle_clone.emit("task-progress", &info);
        }
    });

    Ok(task_id)
}

/// 查询任务进度
///
/// # 参数
/// - `task_id`: 任务 ID
/// - `state`: Tauri 管理的应用状态
///
/// # 返回
/// 任务进度信息
///
/// # 错误
/// 任务不存在
#[tauri::command]
pub fn get_progress(task_id: String, state: State<'_, AppState>) -> Result<ProgressInfo, String> {
    state
        .task_manager
        .get_progress(&task_id)
        .ok_or_else(|| format!("Task not found: {task_id}"))
}

/// 取消正在进行的任务
///
/// 设置取消标志，后台任务将在下次检查时退出。
///
/// # 参数
/// - `task_id`: 任务 ID
/// - `state`: Tauri 管理的应用状态
///
/// # 错误
/// 任务不存在
#[tauri::command]
pub fn cancel_task(task_id: String, state: State<'_, AppState>) -> Result<(), String> {
    state.task_manager.cancel_task(&task_id)
}

/// 列出所有任务
///
/// # 参数
/// - `state`: Tauri 管理的应用状态
///
/// # 返回
/// 所有任务的进度信息列表
#[tauri::command]
pub fn list_all_tasks(state: State<'_, AppState>) -> Vec<ProgressInfo> {
    state.task_manager.list_tasks()
}

/// 加载当前配置
///
/// 如果配置文件存在则加载，否则返回默认配置。
///
/// # 参数
/// - `state`: Tauri 管理的应用状态
///
/// # 返回
/// 配置 JSON 字符串
#[tauri::command]
pub fn load_config(state: State<'_, AppState>) -> Result<String, String> {
    let config = if state.config_path.exists() {
        load_config_from_file(&state.config_path)?
    } else {
        Config::default()
    };

    serde_json::to_string_pretty(&config).map_err(|e| format!("Failed to serialize config: {e}"))
}

/// 保存配置
///
/// 将配置 JSON 序列化为 TOML 并保存到配置文件。
///
/// # 参数
/// - `config_json`: 配置 JSON 字符串
/// - `state`: Tauri 管理的应用状态
///
/// # 错误
/// - JSON 解析失败
/// - 文件写入失败
#[tauri::command]
pub fn save_config(config_json: String, state: State<'_, AppState>) -> Result<(), String> {
    let config: Config = serde_json::from_str(&config_json)
        .map_err(|e| format!("Failed to parse config JSON: {e}"))?;

    // 确保目录存在
    if let Some(parent) = state.config_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create config directory: {e}"))?;
    }

    // 序列化为 TOML 并保存
    let toml_str = toml::to_string_pretty(&config)
        .map_err(|e| format!("Failed to serialize config to TOML: {e}"))?;

    std::fs::write(&state.config_path, toml_str)
        .map_err(|e| format!("Failed to write config file: {e}"))?;

    tracing::info!("Config saved to {:?}", state.config_path);
    Ok(())
}

/// 探测视频文件信息
///
/// 使用 ffprobe 获取视频文件的时长、分辨率、编解码器等信息。
///
/// # 参数
/// - `path`: 视频文件路径
///
/// # 返回
/// 视频文件信息
///
/// # 错误
/// - 文件不存在
/// - ffprobe 执行失败
#[tauri::command]
pub fn probe_video(path: String) -> Result<VideoInfo, String> {
    let video_path = Path::new(&path);
    if !video_path.exists() {
        return Err(format!("File not found: {path}"));
    }

    let media_info = probe_media(video_path).map_err(|e| format!("Failed to probe media: {e}"))?;

    let video_stream = vt_core::media::find_video_stream(&media_info);
    let audio_stream = vt_core::media::find_audio_stream(&media_info);

    Ok(VideoInfo {
        path: path.clone(),
        duration: media_info.duration,
        width: video_stream.and_then(|s| s.width),
        height: video_stream.and_then(|s| s.height),
        video_codec: video_stream.map(|s| s.codec_name.clone()),
        audio_codec: audio_stream.map(|s| s.codec_name.clone()),
    })
}

/// 列出可用的 TTS 音色
///
/// 返回所有内置音色信息列表，包含音色 ID、名称、性别、语言和描述。
/// 前端通过此接口获取可选音色列表用于音色选择下拉框。
///
/// # 返回
/// 音色信息列表（至少包含 2 种女声和 2 种男声）
#[tauri::command]
pub fn list_tts_voices() -> Vec<VoiceInfoDto> {
    let engine = SayEngine::new(&vt_core::config::TtsConfig::default());
    match engine {
        Ok(e) => e
            .list_voices()
            .iter()
            .map(|v| VoiceInfoDto {
                id: v.id.clone(),
                name: v.name.clone(),
                gender: format!("{:}", v.gender).to_lowercase(),
                language: v.language.clone(),
                description: v.description.clone(),
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// 音色信息 DTO（传输给前端）
///
/// 与 `vt_core::voice_manager::VoiceInfo` 对应，但使用简单字符串表示性别。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoiceInfoDto {
    /// 音色唯一标识符
    pub id: String,
    /// 音色显示名称
    pub name: String,
    /// 性别（"female" / "male" / "neutral"）
    pub gender: String,
    /// 语言代码
    pub language: String,
    /// 音色描述
    pub description: String,
}

/// 打开文件选择对话框
///
/// 使用 Tauri 的 dialog 插件打开文件选择器。
///
/// # 返回
/// 选中的文件路径，如果用户取消则返回 `None`。
#[tauri::command]
pub async fn open_file_dialog(app_handle: AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let file_path = app_handle
        .dialog()
        .file()
        .add_filter(
            "Video Files",
            &["mp4", "mkv", "avi", "mov", "webm", "flv", "m4v"],
        )
        .blocking_pick_file();

    match file_path {
        Some(path) => Ok(Some(path.to_string())),
        None => Ok(None),
    }
}

// ─── 后台处理逻辑 ─────────────────────────────────────────

/// 执行完整的视频处理流程
///
/// 这是后台任务的核心函数，执行以下步骤：
/// 1. 构建 Pipeline（ASR、翻译、TTS、音频提取器）
/// 2. 运行流水线处理视频
/// 3. 合成最终视频（TTS 音频 + 原视频）
///
/// 处理过程中持续更新进度并通过事件推送到前端。
async fn run_video_processing(
    input_path: &Path,
    output_path: &str,
    config: &Config,
    task_manager: &TaskManager,
    task_id: &str,
    cancel_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    app_handle: &AppHandle,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::sync::atomic::Ordering;

    // 阶段 1：构建 Pipeline
    update_and_emit(
        task_manager,
        task_id,
        app_handle,
        TaskStatus::Running,
        0.0,
        "初始化流水线...".to_string(),
    );

    let pipeline = build_pipeline(config).map_err(|e| format!("Failed to build pipeline: {e}"))?;

    if cancel_flag.load(Ordering::Relaxed) {
        return Err("Task cancelled".into());
    }

    // 阶段 2：运行流水线（带进度追踪）
    update_and_emit(
        task_manager,
        task_id,
        app_handle,
        TaskStatus::Running,
        0.05,
        "正在提取音频...".to_string(),
    );

    let tracker = Arc::new(ProgressTracker::new());

    // 启动进度轮询任务：定期读取 tracker 并推送进度事件到前端
    let poll_tracker = tracker.clone();
    let poll_tm = task_manager.clone_handle();
    let poll_task_id = task_id.to_string();
    let poll_app = app_handle.clone();
    let poll_cancel = cancel_flag.clone();
    let poll_handle = tauri::async_runtime::spawn(async move {
        use std::sync::atomic::Ordering;
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            if poll_cancel.load(Ordering::Relaxed) {
                break;
            }

            let progress = poll_tracker.overall_progress();
            let total_chunks = poll_tracker.total_chunks();

            let stage = if total_chunks == 0 {
                "正在提取音频并分割...".to_string()
            } else {
                let asr_done = poll_tracker.asr_completed();
                let trans_done = poll_tracker.translation_completed();
                let tts_done = poll_tracker.tts_completed();
                let total_segs = poll_tracker.total_segments();
                let asr_stage_done = poll_tracker.is_asr_done();

                if asr_stage_done && total_segs > 0 {
                    format!(
                        "语音识别 {asr_done}/{total_chunks} | 翻译 {trans_done}/{total_segs} | 语音合成 {tts_done}/{total_segs}",
                    )
                } else {
                    format!(
                        "语音识别 {asr_done}/{total_chunks} | 翻译 {trans_done} | 语音合成 {tts_done}",
                    )
                }
            };

            // 将流水线进度映射到 0.05-0.95 范围
            let mapped = 0.05 + progress * 0.90;
            poll_tm.update_progress(&poll_task_id, TaskStatus::Running, mapped, stage.clone());
            if let Some(info) = poll_tm.get_progress(&poll_task_id) {
                let _ = poll_app.emit("task-progress", &info);
            }

            if progress >= 0.95 {
                break;
            }
        }
    });

    let segments = pipeline
        .process_video_with_progress(input_path, config, &tracker)
        .await
        .map_err(|e| format!("Pipeline processing failed: {e}"))?;

    // 停止轮询
    let _ = poll_handle.await;

    if cancel_flag.load(Ordering::Relaxed) {
        return Err("Task cancelled".into());
    }

    tracing::info!(
        "Task {}: pipeline completed with {} segments",
        task_id,
        segments.len()
    );

    // 阶段 3：合成最终视频
    update_and_emit(
        task_manager,
        task_id,
        app_handle,
        TaskStatus::Running,
        0.95,
        "正在合成最终视频...".to_string(),
    );

    compose_final_video(input_path, &segments, Path::new(output_path), config)?;

    // 完成
    task_manager.mark_completed(task_id);
    let info = task_manager
        .get_progress(task_id)
        .ok_or_else(|| "Task disappeared during completion".to_string())?;
    let _ = app_handle.emit("task-progress", &info);
    let _ = app_handle.emit("task-completed", &task_id.to_string());

    Ok(())
}

/// 更新进度并发送事件
fn update_and_emit(
    task_manager: &TaskManager,
    task_id: &str,
    app_handle: &AppHandle,
    status: TaskStatus,
    progress: f64,
    stage: String,
) {
    task_manager.update_progress(task_id, status, progress, stage.clone());
    if let Some(info) = task_manager.get_progress(task_id) {
        let _ = app_handle.emit("task-progress", &info);
    }
    tracing::debug!("Task {}: {:.0}% - {}", task_id, progress * 100.0, stage);
}

/// 构建 Pipeline 实例
///
/// 根据配置创建 ASR、本地翻译、TTS 引擎并组装流水线。
fn build_pipeline(config: &Config) -> Result<Pipeline, String> {
    let asr_config = WhisperConfig::default()
        .with_language(&config.asr.language)
        .with_metal(config.asr.use_metal);
    let asr_engine =
        WhisperEngine::new(asr_config).map_err(|e| format!("ASR engine init failed: {e}"))?;

    let tts_engine =
        SayEngine::new(&config.tts).map_err(|e| format!("TTS engine init failed: {e}"))?;

    let extractor = FfmpegAudioExtractor::new();

    // 确保翻译模型已下载
    let mut translation_config = config.translation.clone();
    if translation_config.model_path.is_none() {
        let model_manager = vt_core::model_manager::ModelManager::new()
            .map_err(|e| format!("Failed to create translation model manager: {e}"))?;
        let model_path = model_manager
            .load_model(
                &translation_config.model_source,
                "qwen2.5-3b-instruct-q5_k_m.gguf",
                None,
            )
            .map_err(|e| format!("Failed to download translation model: {e}"))?;
        translation_config.model_path = Some(model_path);
    }

    // 构建本地离线翻译引擎（GGUF 本地推理）
    let backend = LlamaCppBackend::from_config(&translation_config)
        .map_err(|e| format!("LLM translation backend init failed: {e}"))?;
    let mut engine =
        LocalTranslationEngine::new(backend).with_batch_size(config.translation.batch_size);

    if let Some(ref glossary_path) = config.translation.glossary_path {
        let path = std::path::Path::new(glossary_path);
        let terminology = if path.extension().is_some_and(|ext| ext == "json") {
            TerminologyManager::load_from_json(path)
        } else {
            TerminologyManager::load_from_csv(path)
        }
        .map_err(|e| format!("Failed to load terminology: {e}"))?;
        engine = engine.with_terminology(terminology);
    }

    let pipeline = PipelineBuilder::default()
        .asr_engine(asr_engine)
        .translation_provider(engine)
        .tts_engine(tts_engine)
        .audio_extractor(extractor);

    pipeline
        .build()
        .map_err(|e| format!("Pipeline build failed: {e}"))
}

/// 合成最终视频：将 TTS 音频按时间戳混合后与原视频合成。
fn compose_final_video(
    video_path: &Path,
    segments: &[vt_core::models::segment::Segment],
    output_path: &Path,
    config: &Config,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if segments.is_empty() {
        tracing::warn!("No segments to compose, skipping video composition");
        return Ok(());
    }

    // 收集 (start_time, end_time, audio_path) 三元组
    let audio_segments: Vec<(f64, f64, &Path)> = segments
        .iter()
        .filter_map(|s| {
            s.tts_audio_path
                .as_ref()
                .map(|p| (s.start, s.end, Path::new(p)))
        })
        .collect();

    if audio_segments.is_empty() {
        tracing::warn!("No TTS audio paths, skipping video composition");
        return Ok(());
    }

    // 探测原视频时长
    let media_info = vt_core::media::probe_media(video_path)?;
    let total_duration = media_info.duration;

    let temp_dir = tempfile::TempDir::new()?;
    let mixed_audio = temp_dir.path().join("mixed_audio.wav");

    let mix_result = vt_core::media::mix_audio_segments(
        &audio_segments,
        total_duration,
        &mixed_audio,
        config.audio_sync.mode,
        config.audio_sync.max_speed_ratio,
    )?;

    let composer = FfmpegVideoComposer::new();
    composer.compose_video(
        video_path,
        &mixed_audio,
        output_path,
        false,
        None,
        mix_result.video_stretch_factor,
    )?;

    Ok(())
}

/// 从文件加载配置
fn load_config_from_file(path: &Path) -> Result<Config, String> {
    Config::from_file(path).map_err(|e| format!("Failed to load config from {path:?}: {e}"))
}
