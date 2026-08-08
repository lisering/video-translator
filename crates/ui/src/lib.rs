//! Video Translator UI 库 (`vt-ui`)
//!
//! 提供 Tauri 桌面应用的后端逻辑，包括：
//! - [`task_manager`][]: 异步任务管理器，跟踪视频处理进度、取消和状态
//! - [`commands`][]: Tauri 命令，暴露给前端调用
//!
//! # 架构
//! ```text
//! Frontend (Next.js) ──invoke──▶ Tauri Commands ──▶ TaskManager
//!                                      │                  │
//!                                      ▼                  ▼
//!                                 vt-core Pipeline   Progress Tracker
//! ```

pub mod commands;
pub mod task_manager;

use std::path::PathBuf;

use commands::AppState;

/// 应用名称，用于日志和配置目录
const APP_NAME: &str = "video-translator";

/// 初始化日志系统
///
/// 将日志同时输出到终端和文件（`~/Library/Logs/video-translator/app.log`）。
/// 在 debug 构建中使用 `debug` 级别，release 构建使用 `info` 级别。
fn init_logging() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let log_dir = get_log_dir();
    let _ = std::fs::create_dir_all(&log_dir);

    let file_appender = tracing_appender::rolling::daily(&log_dir, "app.log");
    let (non_blocking_file, _guard) = tracing_appender::non_blocking(file_appender);

    let filter_level = if cfg!(debug_assertions) {
        "debug"
    } else {
        "info"
    };

    let env_filter = EnvFilter::new(filter_level);

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(
            fmt::layer()
                .with_writer(non_blocking_file)
                .with_ansi(false)
                .with_target(true),
        )
        .with(fmt::layer().with_writer(std::io::stderr).with_ansi(true))
        .try_init();

    tracing::info!("Logging initialized, log dir: {:?}", log_dir);

    // 保持 guard 不被 drop
    std::mem::forget(_guard);
}

/// 获取日志目录路径
///
/// macOS: `~/Library/Logs/video-translator/`
fn get_log_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join("Library/Logs").join(APP_NAME)
    } else {
        PathBuf::from("./logs")
    }
}

/// 获取配置文件路径
///
/// macOS: `~/Library/Application Support/video-translator/config.toml`
fn get_config_path() -> PathBuf {
    if let Some(config_dir) = dirs::config_dir() {
        config_dir.join(APP_NAME).join("config.toml")
    } else {
        PathBuf::from("./config.toml")
    }
}

/// 启动 Tauri 应用
///
/// 初始化日志、创建应用状态、注册命令和插件，然后启动 Tauri 事件循环。
///
/// # Panics
/// 如果 Tauri 应用启动失败（如 `generate_context!` 配置错误），程序会 panic。
pub fn run() {
    init_logging();

    let config_path = get_config_path();
    tracing::info!("Config path: {:?}", config_path);

    let state = AppState::new(config_path);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            commands::process_video,
            commands::get_progress,
            commands::cancel_task,
            commands::list_all_tasks,
            commands::load_config,
            commands::save_config,
            commands::probe_video,
            commands::open_file_dialog,
            commands::list_tts_voices,
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            tracing::error!("Failed to run Tauri application: {e}");
            std::process::exit(1);
        });
}
