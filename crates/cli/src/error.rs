//! CLI 错误处理模块
//!
//! 定义 CLI 层的错误类型和退出码映射，提供用户友好的错误信息。

use std::process;

use colored::Colorize;

/// CLI 错误退出码
///
/// 将不同类型的错误映射到标准退出码，便于脚本集成。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitCode {
    /// 成功
    #[allow(dead_code)]
    Success = 0,
    /// 一般错误（处理失败、IO 错误等）
    GeneralError = 1,
    /// 配置错误（配置文件无效等）
    ConfigError = 2,
    /// 文件未找到
    FileNotFound = 3,
}

impl ExitCode {
    /// 获取退出码数值
    #[must_use]
    pub fn as_code(self) -> i32 {
        self as i32
    }
}

/// 处理 `anyhow::Error` 并以友好的彩色信息输出到 stderr，然后退出。
///
/// 根据错误内容推断退出码：
/// - 包含 "not found" 或 "FileNotFound" → [`ExitCode::FileNotFound`]
/// - 包含 "config" → [`ExitCode::ConfigError`]
/// - 其他 → [`ExitCode::GeneralError`]
pub fn handle_error(err: anyhow::Error) -> ! {
    let err_str = err.to_string();
    let err_lower = err_str.to_lowercase();

    let (exit_code, hint) = if err_lower.contains("not found")
        || err_lower.contains("no such file")
        || err_lower.contains("不存在")
    {
        (
            ExitCode::FileNotFound,
            Some("请检查文件路径是否正确。".to_string()),
        )
    } else if err_lower.contains("config")
        || err_lower.contains("toml")
        || err_lower.contains("configuration")
    {
        (
            ExitCode::ConfigError,
            Some("请检查配置文件格式是否正确。运行 `vt config` 生成默认配置。".to_string()),
        )
    } else {
        (ExitCode::GeneralError, None)
    };

    eprintln!("{} {}", "错误:".red().bold(), err_str);

    if let Some(hint) = hint {
        eprintln!("{} {}", "提示:".yellow(), hint);
    }

    // 打印错误链
    let mut source = err.source();
    while let Some(s) = source {
        eprintln!("  {} {}", "└─".dimmed(), s);
        source = s.source();
    }

    process::exit(exit_code.as_code());
}
