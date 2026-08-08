//! CLI 入口
//!
//! 参考 QORA-TTS 的 main.rs 和 Qwen3-TTS 的 generate_audio.rs 设计。
//! 兼容 SubprocessCloneEngine 的参数模板:
//! `vt-tts synthesize --text "{text}" --voice "{ref_audio}" --output "{output}"`
//!
//! ## Server 模式
//! `vt-tts server --model MODEL_DIR [OPTIONS]`
//! 模型常驻内存，通过 stdin/stdout JSON 行协议接收合成请求，消除重复模型加载开销。
//! 协议:
//!   请求 (stdin):  {"text":"...","voice":"/path/ref.wav","output":"/path/out.wav","ref_text":null,"seed":42}
//!   响应 (stdout): {"status":"ok","output":"/path/out.wav","duration_secs":5.5,"elapsed_secs":1.2,"rtf":0.218}
//!   错误 (stdout): {"status":"error","error":"..."}

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
struct CliArgs {
    /// 子命令: "synthesize" | "info" | "init-config"
    command: String,
    /// 要合成的文本
    text: String,
    /// 参考音频路径（声音克隆）
    voice: Option<PathBuf>,
    /// 输出文件路径
    output: PathBuf,
    /// 模型目录路径
    model: Option<PathBuf>,
    /// 推理设备
    device: String,
    /// 说话人名称（CustomVoice 变体）
    speaker: Option<String>,
    /// 语言
    language: String,
    /// 采样温度
    temperature: f32,
    /// Top-K 采样
    top_k: usize,
    /// 重复惩罚
    repetition_penalty: f32,
    /// No-repeat n-gram size
    no_repeat_ngram_size: usize,
    /// 随机种子
    seed: Option<u64>,
    /// 最大 codes 数
    max_codes: usize,
    /// 参考文本（ICL 模式）
    ref_text: Option<String>,
    /// 混合精度模式: TalkerModel=F16, CodePredictor/Decoder=F32
    mixed_precision: bool,
    /// 推测解码: 使用 n-gram 推测表加速生成
    speculative: bool,
    /// TalkerModel 权重量化格式 (None = 不量化, "q8_0", "q4_0", "q4k")
    quantize: Option<String>,
    /// AudioDecoder 设备覆盖 (None = 同主设备, "cpu" 强制 CPU 解码)
    decode_device: Option<String>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            command: "synthesize".to_string(),
            text: String::new(),
            voice: None,
            output: PathBuf::from("output.wav"),
            model: None,
            device: "cpu".to_string(),
            speaker: None,
            language: "chinese".to_string(),
            temperature: 0.8,
            top_k: 50,
            repetition_penalty: 1.05,
            no_repeat_ngram_size: 0,
            seed: None,
            max_codes: 500,
            ref_text: None,
            mixed_precision: false,
            speculative: false,
            quantize: None,
            decode_device: None,
        }
    }
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut cli = CliArgs::default();

    // 第一个参数可以是子命令
    let mut start = 1;
    if args.len() > 1 && !args[1].starts_with("--") {
        cli.command = args[1].clone();
        start = 2;
    }

    let mut i = start;
    while i < args.len() {
        match args[i].as_str() {
            "--text" | "-t" => {
                if i + 1 < args.len() {
                    cli.text = args[i + 1].clone();
                    i += 1;
                }
            }
            "--voice" | "-v" => {
                if i + 1 < args.len() {
                    cli.voice = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    cli.output = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    cli.model = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--device" => {
                if i + 1 < args.len() {
                    cli.device = args[i + 1].clone();
                    i += 1;
                }
            }
            "--speaker" => {
                if i + 1 < args.len() {
                    cli.speaker = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--language" => {
                if i + 1 < args.len() {
                    cli.language = args[i + 1].clone();
                    i += 1;
                }
            }
            "--temperature" => {
                if i + 1 < args.len() {
                    cli.temperature = args[i + 1].parse().unwrap_or(0.8);
                    i += 1;
                }
            }
            "--top-k" => {
                if i + 1 < args.len() {
                    cli.top_k = args[i + 1].parse().unwrap_or(50);
                    i += 1;
                }
            }
            "--repetition-penalty" => {
                if i + 1 < args.len() {
                    cli.repetition_penalty = args[i + 1].parse().unwrap_or(1.05);
                    i += 1;
                }
            }
            "--no-repeat-ngram-size" => {
                if i + 1 < args.len() {
                    cli.no_repeat_ngram_size = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    cli.seed = Some(args[i + 1].parse().unwrap_or(42));
                    i += 1;
                }
            }
            "--max-codes" => {
                if i + 1 < args.len() {
                    cli.max_codes = args[i + 1].parse().unwrap_or(500);
                    i += 1;
                }
            }
            "--ref-text" => {
                if i + 1 < args.len() {
                    cli.ref_text = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--mixed-precision" => {
                cli.mixed_precision = true;
            }
            "--speculative" => {
                cli.speculative = true;
            }
            "--quantize" => {
                if i + 1 < args.len() {
                    cli.quantize = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--decode-device" => {
                if i + 1 < args.len() {
                    cli.decode_device = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                // 忽略未知参数（兼容 SubprocessCloneEngine 的 "synthesize" 子命令）
            }
        }
        i += 1;
    }

    cli
}

fn print_help() {
    eprintln!("vt-tts — Pure Rust Voice-Cloning TTS Engine");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  vt-tts synthesize --text \"TEXT\" [--voice REF.wav] --output OUT.wav [OPTIONS]");
    eprintln!("  vt-tts server --model MODEL_DIR [OPTIONS]  (persistent mode, stdin/stdout JSON)");
    eprintln!("  vt-tts info");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --text, -t       Text to synthesize");
    eprintln!("  --voice, -v      Reference audio for voice cloning");
    eprintln!("  --output, -o     Output WAV file path");
    eprintln!("  --model, -m      Model directory path");
    eprintln!("  --device         Inference device: cpu | metal | cuda");
    eprintln!("  --speaker        Speaker name (CustomVoice variant)");
    eprintln!("  --language       Language: chinese | english");
    eprintln!("  --temperature     Sampling temperature (0.0-2.0, default 0.8)");
    eprintln!("  --top-k          Top-K sampling (default 50)");
    eprintln!("  --repetition-penalty  Repetition penalty (default 1.05, 1.0=disabled)");
    eprintln!("  --no-repeat-ngram-size  Ban repeated n-grams (default 0=disabled, try 3)");
    eprintln!("  --seed           Random seed for reproducibility");
    eprintln!("  --max-codes      Maximum generation length (default 500)");
    eprintln!("  --ref-text       Reference text for ICL voice cloning");
    eprintln!("  --mixed-precision  Mixed precision: TalkerModel=F16, CP/Decoder=F32");
    eprintln!("  --speculative     Enable speculative decoding (n-gram speculation)");
    eprintln!(
        "  --quantize        Weight quantization: q8_0 | q4_0 | q4k (reduces gen memory bandwidth)"
    );
    eprintln!("  --decode-device   Decoder device override: cpu | metal | cuda (default: same as --device)");
    eprintln!("  --help, -h       Show this help message");
}

fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = parse_args();

    match cli.command.as_str() {
        "info" => {
            print_system_info();
            Ok(())
        }
        "init-config" => {
            print_default_config();
            Ok(())
        }
        "server" => run_server(&cli),
        "synthesize" | _ => {
            if cli.text.is_empty() {
                anyhow::bail!("--text is required. Use --help for usage.");
            }

            run_synthesis(&cli)
        }
    }
}

fn run_synthesis(cli: &CliArgs) -> Result<()> {
    // 系统信息
    let sys = vt_tts::config::SystemInfo::detect();
    let limits = sys.smart_limits();

    eprintln!("vt-tts — Pure Rust Voice-Cloning TTS Engine");
    eprintln!(
        "System: {}MB RAM ({}MB free), {} threads",
        sys.total_ram_mb, sys.available_ram_mb, sys.cpu_threads
    );

    if let Some(ref msg) = limits.warning {
        eprintln!("WARNING: {msg}");
    }

    eprintln!("Text: \"{}\"", cli.text);
    if cli.voice.is_some() {
        eprintln!("Mode: voice cloning");
    } else {
        eprintln!("Mode: default speaker");
    }
    eprintln!();

    // 创建引擎配置
    let engine_config = vt_tts::TtsEngineConfig {
        model_dir: cli.model.clone().unwrap_or_else(|| {
            // 默认模型路径: 与可执行文件同目录
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("models").join("qwen3-tts")))
                .unwrap_or_else(|| PathBuf::from("models/qwen3-tts"))
        }),
        device: cli.device.clone(),
        temperature: cli.temperature,
        top_k: cli.top_k,
        repetition_penalty: cli.repetition_penalty,
        no_repeat_ngram_size: cli.no_repeat_ngram_size,
        seed: cli.seed,
        max_codes: cli.max_codes.min(limits.max_codes),
        output_sample_rate: 24000,
        language: cli.language.clone(),
        mixed_precision: cli.mixed_precision,
        quantize: cli.quantize.clone(),
        decode_device: cli.decode_device.clone(),
    };

    // 创建引擎
    let engine = vt_tts::engine::create_engine(engine_config)?;

    eprintln!("Engine: {} ({})", engine.name(), engine.model_variant());
    eprintln!(
        "Voice cloning: {}",
        if engine.supports_voice_cloning() {
            "supported"
        } else {
            "not supported"
        }
    );
    eprintln!(
        "Temperature: {}, Top-K: {}, RepetitionPenalty: {}, NoRepeatNgram: {}, Seed: {}",
        cli.temperature,
        cli.top_k,
        cli.repetition_penalty,
        cli.no_repeat_ngram_size,
        cli.seed.map(|s| s.to_string()).unwrap_or("random".into())
    );
    if cli.mixed_precision {
        eprintln!("Mixed precision: ENABLED (TalkerModel=F16, CodePredictor/Decoder=F32)");
    }
    if let Some(ref q) = cli.quantize {
        eprintln!("Quantization: {} (TalkerModel weights quantized)", q);
    }
    if let Some(ref d) = cli.decode_device {
        eprintln!("Decode device: {} (decoder runs on {})", d, d);
    }
    eprintln!();

    // 声音克隆提示
    let voice_clone = if let Some(ref voice_path) = cli.voice {
        if !voice_path.exists() {
            anyhow::bail!("Reference audio not found: {:?}", voice_path);
        }
        eprintln!("Extracting voice from {}...", voice_path.display());
        let vc_prompt = engine.create_voice_clone_prompt(voice_path, cli.ref_text.as_deref())?;
        eprintln!(
            "Voice embedding: {} dims",
            vc_prompt.speaker_embedding.len()
        );
        Some(vc_prompt)
    } else {
        None
    };

    // 合成选项
    let options = vt_tts::SynthesisOptions {
        temperature: cli.temperature,
        top_k: cli.top_k,
        repetition_penalty: cli.repetition_penalty,
        no_repeat_ngram_size: cli.no_repeat_ngram_size,
        seed: cli.seed,
        max_codes: cli.max_codes.min(limits.max_codes),
        speculative: cli.speculative,
    };

    // 合成
    eprintln!("Synthesizing...");
    let result = engine.synthesize(&cli.text, voice_clone.as_ref(), &options)?;

    // 保存
    eprintln!();
    eprintln!(
        "Generated {:.1}s of audio in {:.1}s (RTF: {:.3}x)",
        result.audio.duration_secs(),
        result.elapsed_secs,
        result.rtf
    );

    result.audio.save_wav(&cli.output)?;
    eprintln!("Saved to {}", cli.output.display());

    Ok(())
}

fn print_system_info() {
    let sys = vt_tts::config::SystemInfo::detect();
    let limits = sys.smart_limits();

    println!("vt-tts System Information");
    println!("=========================");
    println!("Total RAM:       {} MB", sys.total_ram_mb);
    println!("Available RAM:   {} MB", sys.available_ram_mb);
    println!("CPU Threads:     {}", sys.cpu_threads);
    println!();
    println!("Smart Limits:");
    println!("  Default max codes: {}", limits.default_max_codes);
    println!("  Maximum codes:      {}", limits.max_codes);
    if let Some(ref w) = limits.warning {
        println!("  Warning: {}", w);
    }
}

fn print_default_config() {
    let config = vt_tts::TtsEngineConfig::default();
    match serde_json::to_string_pretty(&config) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error: {}", e),
    }
}

// ─── Server 模式 ──────────────────────────────────────────

/// Server 请求 (stdin, 每行一个 JSON)
#[derive(Debug, Deserialize)]
struct ServerRequest {
    /// 要合成的文本
    text: String,
    /// 参考音频路径 (None = 默认说话人)
    voice: Option<String>,
    /// 输出 WAV 文件路径
    output: String,
    /// 参考文本 (ICL 模式, 可选)
    ref_text: Option<String>,
    /// 随机种子 (None = 使用服务器默认)
    seed: Option<u64>,
}

/// Server 响应 (stdout, 每行一个 JSON)
#[derive(Debug, Serialize)]
#[serde(tag = "status")]
enum ServerResponse {
    /// 合成成功
    #[serde(rename = "ok")]
    Ok {
        output: String,
        duration_secs: f64,
        elapsed_secs: f64,
        rtf: f64,
    },
    /// 合成失败
    #[serde(rename = "error")]
    Error { error: String },
}

/// 运行 Server 模式 — 模型常驻, 通过 stdin/stdout JSON 行协议处理合成请求
///
/// 工作流:
/// 1. 加载模型一次 (与 synthesize 相同的引擎初始化)
/// 2. 从 stdin 逐行读取 JSON 请求
/// 3. 对每个请求: 提取声音克隆提示 (带缓存) → 合成 → 保存 WAV → 写 JSON 响应到 stdout
/// 4. stdin EOF 时退出
///
/// 声音克隆缓存: 当 voice 路径与上次相同时, 复用已提取的 VoiceClonePrompt,
/// 跳过重复的说话人编码提取 (节省 ~200ms/次)
fn run_server(cli: &CliArgs) -> Result<()> {
    let sys = vt_tts::config::SystemInfo::detect();
    let limits = sys.smart_limits();

    eprintln!("vt-tts server — persistent mode (model stays in memory)");
    eprintln!(
        "System: {}MB RAM ({}MB free), {} threads",
        sys.total_ram_mb, sys.available_ram_mb, sys.cpu_threads
    );
    if let Some(ref msg) = limits.warning {
        eprintln!("WARNING: {msg}");
    }

    // 创建引擎配置 (与 run_synthesis 相同)
    let engine_config = vt_tts::TtsEngineConfig {
        model_dir: cli.model.clone().unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("models").join("qwen3-tts")))
                .unwrap_or_else(|| PathBuf::from("models/qwen3-tts"))
        }),
        device: cli.device.clone(),
        temperature: cli.temperature,
        top_k: cli.top_k,
        repetition_penalty: cli.repetition_penalty,
        no_repeat_ngram_size: cli.no_repeat_ngram_size,
        seed: cli.seed,
        max_codes: cli.max_codes.min(limits.max_codes),
        output_sample_rate: 24000,
        language: cli.language.clone(),
        mixed_precision: cli.mixed_precision,
        quantize: cli.quantize.clone(),
        decode_device: cli.decode_device.clone(),
    };

    // 加载引擎 (仅一次)
    let engine = vt_tts::engine::create_engine(engine_config)?;
    eprintln!(
        "Engine loaded: {} ({})",
        engine.name(),
        engine.model_variant()
    );
    eprintln!(
        "Voice cloning: {}",
        if engine.supports_voice_cloning() {
            "supported"
        } else {
            "not supported"
        }
    );
    if let Some(ref d) = cli.decode_device {
        eprintln!("Decode device: {}", d);
    }
    eprintln!("Server ready — waiting for requests on stdin...");

    // 合成选项模板
    let base_options = vt_tts::SynthesisOptions {
        temperature: cli.temperature,
        top_k: cli.top_k,
        repetition_penalty: cli.repetition_penalty,
        no_repeat_ngram_size: cli.no_repeat_ngram_size,
        seed: cli.seed,
        max_codes: cli.max_codes.min(limits.max_codes),
        speculative: cli.speculative,
    };

    // 声音克隆提示缓存 (voice path → VoiceClonePrompt)
    // 同一个参考音频只提取一次, 后续请求复用
    let mut cached_voice: Option<(String, vt_tts::engine::VoiceClonePrompt)> = None;
    let mut request_count: u64 = 0;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("server: stdin read error: {e}");
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        request_count += 1;
        let req: ServerRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = ServerResponse::Error {
                    error: format!("Invalid JSON request: {e}"),
                };
                writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap())?;
                stdout.flush()?;
                continue;
            }
        };

        eprintln!(
            "server: request #{} text=\"{}\" output={}",
            request_count,
            truncate(&req.text, 50),
            req.output
        );

        // 获取或缓存声音克隆提示
        let voice_clone = if let Some(ref voice_path) = req.voice {
            let voice_path_str = voice_path.clone();

            // 检查缓存
            if cached_voice
                .as_ref()
                .map_or(false, |(p, _)| *p == voice_path_str)
            {
                // 缓存命中
                cached_voice.as_ref().map(|(_, vc)| vc.clone())
            } else {
                // 提取新的声音克隆提示
                let voice_path = PathBuf::from(&voice_path_str);
                if !voice_path.exists() {
                    let resp = ServerResponse::Error {
                        error: format!("Reference audio not found: {}", voice_path_str),
                    };
                    writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap())?;
                    stdout.flush()?;
                    continue;
                }
                match engine.create_voice_clone_prompt(&voice_path, req.ref_text.as_deref()) {
                    Ok(vc) => {
                        eprintln!(
                            "server: voice embedding extracted ({} dims), cached for reuse",
                            vc.speaker_embedding.len()
                        );
                        cached_voice = Some((voice_path_str, vc.clone()));
                        Some(vc)
                    }
                    Err(e) => {
                        let resp = ServerResponse::Error {
                            error: format!("Voice clone prompt extraction failed: {e}"),
                        };
                        writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap())?;
                        stdout.flush()?;
                        continue;
                    }
                }
            }
        } else {
            None
        };

        // 合成选项 (允许请求级 seed 覆盖)
        let options = if req.seed.is_some() && req.seed != base_options.seed {
            vt_tts::SynthesisOptions {
                seed: req.seed,
                ..base_options.clone()
            }
        } else {
            base_options.clone()
        };

        // 合成
        match engine.synthesize(&req.text, voice_clone.as_ref(), &options) {
            Ok(result) => {
                let output_path = PathBuf::from(&req.output);
                if let Err(e) = result.audio.save_wav(&output_path) {
                    let resp = ServerResponse::Error {
                        error: format!("Failed to save WAV: {e}"),
                    };
                    writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap())?;
                    stdout.flush()?;
                    continue;
                }

                eprintln!(
                    "server: request #{} done — {:.1}s audio in {:.1}s (RTF: {:.3}x)",
                    request_count,
                    result.audio.duration_secs(),
                    result.elapsed_secs,
                    result.rtf
                );

                let resp = ServerResponse::Ok {
                    output: req.output,
                    duration_secs: result.audio.duration_secs(),
                    elapsed_secs: result.elapsed_secs,
                    rtf: result.rtf,
                };
                writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap())?;
                stdout.flush()?;
            }
            Err(e) => {
                let resp = ServerResponse::Error {
                    error: format!("Synthesis failed: {e}"),
                };
                writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap())?;
                stdout.flush()?;
            }
        }
    }

    eprintln!(
        "server: stdin EOF, processed {} requests total. Shutting down.",
        request_count
    );
    Ok(())
}

/// 截断字符串到指定长度, 超出部分用 "..." 替代
fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len).collect();
        format!("{truncated}...")
    }
}
