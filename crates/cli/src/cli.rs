//! CLI 参数结构定义
//!
//! 使用 `clap` 的 derive 宏定义命令行参数结构，包含三个子命令：
//! `process`、`batch`、`config`，以及全局选项 `--verbose` 和 `--quiet`。

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Video Translator — 纯离线英文视频自动翻译配音工具
///
/// 基于 Whisper ASR、本地 GGUF 模型翻译和 Kokoro TTS 语音合成，
/// 将英文视频自动翻译并配音为中文。所有 AI 模型均在本地运行，无需任何 API Key。
#[derive(Parser, Debug)]
#[command(
    name = "vt",
    version,
    about = "Video Translator — 英文视频自动翻译配音工具",
    long_about = "Video Translator 将英文视频通过 ASR（语音识别）、翻译、TTS（语音合成）\n\
                  三阶段流水线处理，输出带中文配音和字幕的视频。"
)]
pub struct Cli {
    /// 子命令
    #[command(subcommand)]
    pub command: Commands,

    /// 启用详细日志输出
    #[arg(short, long, global = true, help = "启用详细日志输出")]
    pub verbose: bool,

    /// 静默模式，仅输出错误信息
    #[arg(short, long, global = true, help = "静默模式，仅输出错误信息")]
    pub quiet: bool,
}

/// 子命令枚举
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// 处理单个视频文件
    Process(ProcessArgs),
    /// 批量处理目录中的所有视频
    Batch(BatchArgs),
    /// 生成默认配置文件
    Config(ConfigArgs),
}

/// `process` 子命令参数
#[derive(Parser, Debug)]
pub struct ProcessArgs {
    /// 输入视频文件路径（必填）
    #[arg(long, help = "输入视频文件路径")]
    pub input: PathBuf,

    /// 输出视频文件路径（可选，默认自动生成）
    #[arg(long, help = "输出视频文件路径（默认自动生成）")]
    pub output: Option<PathBuf>,

    /// 配置文件路径（可选，默认从 ./config.toml 或 ~/.config/video-translator/config.toml 加载）
    #[arg(long, help = "配置文件路径（TOML 格式）")]
    pub config: Option<PathBuf>,

    /// 跳过声音克隆，使用系统 TTS 或 Edge-TTS
    ///
    /// 启用后不加载 TTS 模型，速度极快且内存安全。
    /// 配合 --tts-engine 选择 TTS 引擎。
    #[arg(long, help = "跳过声音克隆，使用系统/云端 TTS（更快，不加载模型）")]
    pub no_clone: bool,

    /// TTS 引擎选择（仅 --no-clone 模式生效）
    ///
    /// 可选值：
    /// - say: macOS 内置 say 命令（默认，离线，音质一般）
    /// - edge: 微软 Edge-TTS 云端神经语音（需网络，音质好，自带男声）
    #[arg(long, help = "TTS 引擎: say | edge（仅 --no-clone 生效，默认 say）")]
    pub tts_engine: Option<String>,

    /// 指定 TTS 音色
    ///
    /// say 引擎可选值：Tingting(女), Meijia(女), Zhiyu(男)
    /// edge 引擎可选值：zh-CN-XiaoxiaoNeural(女), zh-CN-YunxiNeural(男)
    /// 特殊值：auto（仅 edge 引擎，根据原视频声音自动检测男/女）
    #[arg(long, help = "TTS 音色（默认取决于引擎，edge 支持 auto 自动检测）")]
    pub voice: Option<String>,
}

/// `batch` 子命令参数
#[derive(Parser, Debug)]
pub struct BatchArgs {
    /// 输入视频目录
    #[arg(long, help = "输入视频目录")]
    pub input_dir: PathBuf,

    /// 输出目录
    #[arg(long, help = "输出目录")]
    pub output_dir: PathBuf,

    /// 配置文件路径
    #[arg(long, help = "配置文件路径（TOML 格式）")]
    pub config: Option<PathBuf>,

    /// 并行处理（默认串行）
    #[arg(long, help = "并行处理多个视频")]
    pub parallel: bool,

    /// 跳过声音克隆，使用系统 TTS 或 Edge-TTS
    #[arg(long, help = "跳过声音克隆，使用系统/云端 TTS（更快，不加载模型）")]
    pub no_clone: bool,

    /// TTS 引擎选择（仅 --no-clone 模式生效）
    #[arg(long, help = "TTS 引擎: say | edge（仅 --no-clone 生效，默认 say）")]
    pub tts_engine: Option<String>,

    /// 指定 TTS 音色
    #[arg(long, help = "TTS 音色（默认取决于引擎，edge 支持 auto 自动检测）")]
    pub voice: Option<String>,
}

/// `config` 子命令参数
#[derive(Parser, Debug)]
pub struct ConfigArgs {
    /// 输出文件路径（可选，不指定则打印到 stdout）
    #[arg(short, long, help = "输出配置文件路径（不指定则打印到 stdout）")]
    pub output: Option<PathBuf>,
}
