//! CLI 配置加载模块
//!
//! 实现配置加载优先级：命令行参数 > 配置文件 > 默认值。
//! 配置文件查找顺序：`--config` 指定路径 > `./config.toml` > `~/.config/video-translator/config.toml`。

use std::path::{Path, PathBuf};

use vt_core::config::Config;
use vt_core::error::AppResult;

/// 配置文件默认搜索路径
const CONFIG_FILE_NAMES: &[&str] = &["config.toml", "video-translator.toml"];

/// 用户配置目录名
const USER_CONFIG_DIR: &str = "video-translator";

/// 加载配置文件，按优先级查找。
///
/// 查找顺序：
/// 1. `--config` 参数指定的路径（如果存在）
/// 2. 当前目录下的 `config.toml` 或 `video-translator.toml`
/// 3. `~/.config/video-translator/config.toml`
/// 4. 如果都不存在，返回默认配置
///
/// # 参数
/// - `config_path`: 命令行 `--config` 参数指定的路径（可选）
///
/// # 返回
/// 加载的 [`Config`]，如果未找到配置文件则返回默认配置。
pub fn load_config(config_path: Option<&Path>) -> AppResult<Config> {
    // 1. 优先使用 --config 指定的路径
    if let Some(path) = config_path {
        if path.exists() {
            tracing::info!("Loading config from --config: {:?}", path);
            return Config::from_file(path);
        }
        return Err(vt_core::error::AppError::Config(format!(
            "Config file specified by --config not found: {:?}",
            path
        )));
    }

    // 2. 查找当前目录下的配置文件
    for name in CONFIG_FILE_NAMES {
        let path = Path::new(name);
        if path.exists() {
            tracing::info!("Loading config from current directory: {:?}", path);
            return Config::from_file(path);
        }
    }

    // 3. 查找用户配置目录
    if let Some(home) = home_dir() {
        let user_config = home
            .join(".config")
            .join(USER_CONFIG_DIR)
            .join("config.toml");
        if user_config.exists() {
            tracing::info!("Loading config from user directory: {:?}", user_config);
            return Config::from_file(&user_config);
        }
    }

    // 4. 使用默认配置
    tracing::info!("No config file found, using default configuration");
    Ok(Config::default())
}

/// 生成默认配置的 TOML 字符串。
///
/// 将 [`Config::default()`] 序列化为带注释的 TOML 格式。
///
/// # 返回
/// TOML 格式的配置字符串。
pub fn generate_default_config_toml() -> String {
    let config = Config::default();

    // 使用 toml 序列化，然后添加注释
    let toml_str = toml::to_string_pretty(&config).unwrap_or_else(|_| {
        "# 配置序列化失败，使用最小配置\n[asr]\nmodel = \"whisper-large-v3\"\n".to_string()
    });

    format_config_with_comments(&toml_str)
}

/// 为 TOML 配置字符串添加友好的注释头。
fn format_config_with_comments(toml_str: &str) -> String {
    let header = "# Video Translator 配置文件\n\
                  # 生成命令: vt config\n\
                  #\n\
                  # 配置加载优先级: 命令行参数 > 配置文件 > 默认值\n\
                  # 所有 AI 模型均从 ModelScope 下载后在本地运行，无需任何 API Key\n\n";

    format!("{header}{toml_str}")
}

/// 获取用户 Home 目录路径。
///
/// 依次尝试 `HOME` 环境变量和 `std::env::home_dir()`。
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(dirs_home)
}

/// 使用标准库获取 Home 目录（回退方案）。
fn dirs_home() -> Option<PathBuf> {
    // 标准库的 home_dir 在 Rust 1.85+ 已恢复
    std::env::home_dir()
}
