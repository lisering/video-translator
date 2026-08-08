# 🎬 Video Translator

> 英文 IT 视频自动翻译配音工具 — 基于 Rust + Whisper ASR + 本地翻译 + TTS 的端到端流水线

[![CI](https://github.com/lisering/video-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/lisering/video-translator/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-green.svg)](https://www.apple.com/mac/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

Video Translator 将英文视频通过 **ASR（语音识别）→ 翻译 → TTS（语音合成）** 三阶段异步并行流水线处理，输出带中文配音的视频。专为 macOS Apple Silicon 优化，支持 Metal GPU 加速和 VideoToolbox 硬件编码。

---

## ✨ 主要特性

| 特性 | 说明 |
|------|------|
| 🎙️ **Whisper ASR** | 基于 whisper.cpp，支持 Metal GPU 加速和 VAD 预处理 |
| 🌐 **本地翻译引擎** | 基于 GGUF 模型的离线翻译，所有 AI 模型均从 ModelScope 下载后在本地运行 |
| 🔊 **中文 TTS** | Edge TTS 神经语音合成，支持内容缓存和并行合成 |
| 🎥 **FFmpeg 处理** | 音频提取、视频合成，支持 VideoToolbox 硬件加速 |
| 👥 **说话人分离** | 自动识别不同说话人，为每个片段标记 speaker |
| 🎭 **声音克隆** | 使用原视频说话人的音色合成目标语言语音 |
| 📦 **批量处理** | 任务队列管理，优先级调度，动态并发控制 |
| 💾 **断点续传** | 中断后可恢复处理，避免重复工作 |
| 🖥️ **双界面** | CLI 命令行工具 + Tauri 桌面 GUI 应用 |
| 📝 **术语表** | IT 术语占位符替换，确保专业术语翻译一致性 |

---

## 📋 系统要求

- **macOS 13.0+ (Ventura)** 或更高版本
- **Apple Silicon** (M1/M2/M3/M4)
- **Rust 1.75+** (构建用)
- **Node.js 18+** (Tauri 前端构建用)
- **FFmpeg 4.4+** (通过 Homebrew 安装)
- 4GB+ 可用内存（处理长视频时建议 8GB+）

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 FFmpeg
brew install ffmpeg

# 安装 Rust (如尚未安装)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 Tauri CLI (可选，用于桌面应用)
cargo install tauri-cli
```

### 2. 克隆并构建

```bash
git clone https://github.com/lisering/video-translator.git
cd video-translator

# 构建 CLI
cargo build --release -p vt-cli

# CLI 二进制位于 target/release/vt
```

### 3. 配置

```bash
# 生成示例配置文件
./target/release/vt config --output config.toml

# 编辑 config.toml 调整模型、设备等参数
# 所有 AI 模型均从 ModelScope 下载后在本地运行，无需任何 API Key
```

### 4. 处理视频

```bash
# 处理单个视频
./target/release/vt process --input video.mp4 --config config.toml

# 批量处理目录中的所有视频
./target/release/vt batch --input-dir ./videos --output-dir ./output --config config.toml
```

### 5. 桌面应用 (可选)

```bash
# 构建并运行 Tauri 桌面应用
cargo tauri dev

# 构建 release 版 .app 和 .dmg
cargo tauri build
```

---

## 📖 命令行使用

### `vt process` — 处理单个视频

```bash
vt process --input <视频路径> [--output <输出路径>] [--config <配置文件>]
```

| 参数 | 说明 |
|------|------|
| `--input` | 输入视频文件路径（必填） |
| `--output` | 输出视频路径（可选，默认自动生成） |
| `--config` | 配置文件路径（可选，默认查找 `./config.toml`） |

### `vt batch` — 批量处理

```bash
vt batch --input-dir <目录> --output-dir <目录> [--config <配置文件>] [--parallel]
```

| 参数 | 说明 |
|------|------|
| `--input-dir` | 输入视频目录 |
| `--output-dir` | 输出目录 |
| `--config` | 配置文件路径 |
| `--parallel` | 并行处理多个视频 |

### `vt config` — 生成配置文件

```bash
vt config [--output <输出路径>]
```

### 全局选项

| 选项 | 说明 |
|------|------|
| `-v, --verbose` | 启用详细日志输出 |
| `-q, --quiet` | 静默模式，仅输出错误信息 |

---

## ⚙️ 配置详解

配置文件采用 TOML 格式，支持以下段落：

| 段落 | 说明 |
|------|------|
| `[asr]` | 语音识别配置（模型、Metal 加速、语言） |
| `[tts]` | 语音合成配置（语速、音色、缓存、并行数） |
| `[translation]` | 翻译配置（本地模型、术语表、批量大小） |
| `[pipeline]` | 流水线配置（分割时长、通道容量、VAD） |
| `[diarization]` | 说话人分离配置 |
| `[cloning]` | 声音克隆配置 |
| `[batch]` | 批量处理配置（并发数、内存阈值） |
| `[checkpoint]` | 断点续传配置 |
| `[performance]` | 性能调优配置 |

完整配置示例见 [`config.example.toml`](config.example.toml)。

### 本地翻译模型

翻译引擎使用本地 GGUF 模型进行离线翻译，无需任何 API Key。模型从 ModelScope 下载并缓存到本地：

```toml
[translation]
device = "cpu"        # 或 "metal" 使用 GPU 加速
max_tokens = 512
temperature = 0.3

[translation.model_source]
ModelScope = { repo_id = "Qwen/Qwen2.5-7B-Instruct-GGUF", revision = "master" }
```

首次运行时会自动下载模型，后续运行将直接使用本地缓存。

### 术语表

术语表支持 JSON 和 CSV 两种格式:

**JSON 格式:**
```json
[
  {"source": "GPU", "target": "图形处理器"},
  {"source": "API", "target": "应用程序接口"}
]
```

**CSV 格式:**
```csv
source,target
GPU,图形处理器
API,应用程序接口
```

---

## 🏗️ 项目结构

```
video-translator/
├── crates/
│   ├── core/           # 核心库 (vt-core)
│   │   ├── src/
│   │   │   ├── asr.rs          # ASR 语音识别
│   │   │   ├── translate.rs    # 翻译模块
│   │   │   ├── tts.rs          # TTS 语音合成
│   │   │   ├── media.rs        # 音视频处理
│   │   │   ├── pipeline.rs     # 流水线引擎
│   │   │   ├── diarization.rs  # 说话人分离
│   │   │   ├── cloning.rs      # 声音克隆
│   │   │   ├── batch.rs        # 批量处理
│   │   │   ├── checkpoint.rs   # 断点续传
│   │   │   ├── config.rs       # 配置管理
│   │   │   ├── error.rs        # 错误类型
│   │   │   └── models/         # 数据模型
│   │   ├── benches/            # 性能基准测试
│   │   └── tests/              # 集成测试
│   ├── cli/            # CLI 工具 (vt-cli)
│   │   └── src/
│   │       ├── main.rs         # 入口
│   │       ├── cli.rs          # 参数定义
│   │       ├── commands.rs     # 命令实现
│   │       └── config.rs       # 配置加载
│   └── ui/             # Tauri 桌面应用 (vt-ui)
│       ├── src/
│       │   ├── lib.rs          # Tauri 应用入口
│       │   ├── commands.rs     # Tauri 命令
│       │   └── task_manager.rs # 任务管理
│       ├── frontend/           # Next.js 前端
│       └── tauri.conf.json     # Tauri 配置
├── scripts/python/     # 运行时 Python 脚本 (TTS)
├── .github/workflows/  # CI
└── Cargo.toml          # Workspace 配置
```

---

## 🧪 测试

```bash
# 运行所有测试
cargo test --workspace --all-features

# 代码格式检查
cargo fmt --all -- --check

# Clippy 检查 (警告视为错误)
cargo clippy --workspace --all-features -- -D warnings

# 性能基准测试
cargo bench --workspace
```

---

## 📊 性能

在 M1 Pro (16GB) 上的参考性能：

| 操作 | 耗时 |
|------|------|
| ASR 转录（1 小时音频，Metal 加速） | ~5 分钟 |
| TTS 合成（100 个片段，并行） | ~2 分钟 |
| 端到端流水线（1 小时视频） | ~25 分钟 |
| 流水线吞吐量 | ~2.4x 实时 |

---

## 🤝 贡献

欢迎提交 Pull Request！

快速检查清单：

1. 新增功能需附带测试
2. `cargo fmt --all -- --check` 通过
3. `cargo clippy --workspace --all-features -- -D warnings` 通过
4. `cargo test --workspace --all-features` 通过
5. 公共 API 需有 `///` 文档注释

---

## 💬 支持与反馈

- **Bug 报告**: [提交 Issue](https://github.com/lisering/video-translator/issues)
- **功能请求**: [提交 Issue](https://github.com/lisering/video-translator/issues)
- **使用讨论**: [GitHub Discussions](https://github.com/lisering/video-translator/discussions)

---

## 🙏 致谢

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Whisper 语音识别引擎
- [Edge TTS](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/) — Microsoft Edge 语音合成
- [Tauri](https://tauri.app/) — 跨平台桌面应用框架
- [FFmpeg](https://ffmpeg.org/) — 多媒体处理工具
- [ModelScope](https://modelscope.cn/) — 模型下载平台
