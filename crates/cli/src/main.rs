//! Video Translator CLI 入口点
//!
//! 基于 `clap` 实现命令行界面，支持处理单个视频、批量处理和配置管理。

mod cli;
mod commands;
mod config;
mod error;

use clap::Parser;
use colored::Colorize;

use crate::cli::{Cli, Commands};
use crate::commands::{run_batch_command, run_config_command, run_process_command};
use crate::error::handle_error;

/// 初始化日志（tracing）
fn init_logging(verbose: bool, quiet: bool) {
    let filter = if quiet {
        "error"
    } else if verbose {
        "debug"
    } else {
        "info"
    };

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

/// 主入口：解析 CLI 参数并分派到对应子命令。
#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    init_logging(cli.verbose, cli.quiet);

    if !cli.quiet {
        eprintln!(
            "{} Video Translator v{}",
            "🎬".to_string().cyan().bold(),
            env!("CARGO_PKG_VERSION")
        );
    }

    let result = run(cli).await;

    match result {
        Ok(()) => {}
        Err(e) => handle_error(e),
    }
}

/// 分派到对应子命令处理函数。
async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Process(args) => run_process_command(args, cli.quiet).await,
        Commands::Batch(args) => run_batch_command(args, cli.quiet).await,
        Commands::Config(args) => run_config_command(args),
    }
}
