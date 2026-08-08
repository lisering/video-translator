//! 子命令实现模块
//!
//! 实现 `process`、`batch`、`config` 三个子命令的具体逻辑。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use vt_core::asr::{WhisperConfig, WhisperEngine};
use vt_core::audio_post_process::AudioPostProcessor;
use vt_core::cloning::{
    CloningConfig, CloningIntegration, GptSoVitsEngine, MockCloningEngine,
    PersistentSubprocessCloneEngine, SubprocessCloneEngine,
};
use vt_core::config::Config;
use vt_core::media::{
    extend_video_freeze_frame, mix_audio_segments, probe_media, AudioExtractor,
    FfmpegAudioExtractor, FfmpegVideoComposer, VideoComposer,
};
use vt_core::pipeline::{Pipeline, PipelineBuilder, ProgressTracker};
use vt_core::translate::{
    builtin_programming_terms, GlossaryEntry, LlamaCppBackend, LocalTranslationEngine,
    TerminologyManager,
};
use vt_core::translation_extras::{
    format_glossary_as_markdown, generate_bilingual_srt, generate_srt, SubtitleType,
};
use vt_core::tts::SayEngine;

use crate::cli::{BatchArgs, ConfigArgs, ProcessArgs};
use crate::config::{generate_default_config_toml, load_config};

// ─── 支持的视频文件扩展名 ─────────────────────────────────

/// 支持的视频文件扩展名
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mkv", "avi", "mov", "webm", "flv", "m4v"];

/// 将秒数格式化为人类可读的时长字符串（如 "3m 20s"、"1h 5m"）。
fn format_duration_eta(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "?".to_string();
    }
    let total = secs as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

// ─── config 子命令 ────────────────────────────────────────

/// 执行 `config` 子命令：生成默认配置文件。
///
/// 如果指定了 `--output`，将配置写入文件；否则打印到 stdout。
///
/// # 参数
/// - `args`: config 子命令参数
pub fn run_config_command(args: ConfigArgs) -> Result<()> {
    let toml_content = generate_default_config_toml();

    match args.output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create directory: {:?}", parent))?;
            }
            std::fs::write(&path, &toml_content)
                .with_context(|| format!("Failed to write config file: {:?}", path))?;
            println!("{} 配置文件已生成: {:?}", "✓".green().bold(), path);
        }
        None => {
            print!("{toml_content}");
        }
    }

    Ok(())
}

// ─── process 子命令 ───────────────────────────────────────

/// 执行 `process` 子命令：处理单个视频文件。
///
/// 流程：
/// 1. 验证输入文件存在
/// 2. 加载配置
/// 3. 构建 Pipeline（ASR、翻译、TTS、音频提取器）
/// 4. 运行流水线处理
/// 5. 合成最终视频
///
/// # 参数
/// - `args`: process 子命令参数
/// - `quiet`: 是否静默模式
pub async fn run_process_command(args: ProcessArgs, quiet: bool) -> Result<()> {
    let start_time = Instant::now();

    // 1. 验证输入文件
    if !args.input.exists() {
        anyhow::bail!("File not found: {:?}", args.input);
    }

    // 2. 加载配置
    let mut config = load_config(args.config.as_deref())?;

    // --no-clone: 跳过声音克隆
    let tts_engine_choice = args.tts_engine.as_deref().unwrap_or("say");
    if args.no_clone {
        let engine_label = if tts_engine_choice == "edge" {
            "Edge-TTS (云端神经语音)"
        } else {
            "系统 TTS (say 命令)"
        };
        if !quiet {
            println!("{} 声音克隆已关闭，使用 {}", "→".cyan(), engine_label);
        }
        config.cloning.enabled = false;
    }

    // --voice: 覆盖 TTS 音色
    if let Some(ref voice) = args.voice {
        if !quiet {
            println!("{} TTS 音色: {}", "→".cyan(), voice);
        }
        config.tts.voice = voice.clone();
    } else if args.no_clone && tts_engine_choice == "edge" {
        // edge 引擎默认使用 auto（自动检测性别）
        config.tts.voice = "auto".to_string();
        if !quiet {
            println!("{} TTS 音色: auto (自动检测)", "→".cyan());
        }
    }

    // 提取原始音频用于性别检测（edge-tts auto 模式）
    let ref_audio_for_gender: Option<PathBuf> =
        if args.no_clone && tts_engine_choice == "edge" && config.tts.voice == "auto" {
            let temp_wav = std::env::temp_dir().join("vt_gender_ref.wav");
            match FfmpegAudioExtractor::new().extract_audio(&args.input, &temp_wav) {
                Ok(()) => {
                    let gender = vt_core::gender_detect::detect_gender_from_wav(&temp_wav);
                    let voice = match gender {
                        vt_core::voice_manager::VoiceGender::Male => {
                            if !quiet {
                                println!("{} 检测到男声 → zh-CN-YunxiNeural", "→".cyan());
                            }
                            "zh-CN-YunxiNeural"
                        }
                        _ => {
                            if !quiet {
                                println!("{} 检测到女声 → zh-CN-XiaoxiaoNeural", "→".cyan());
                            }
                            "zh-CN-XiaoxiaoNeural"
                        }
                    };
                    config.tts.voice = voice.to_string();
                    Some(temp_wav)
                }
                Err(e) => {
                    tracing::warn!("Failed to extract audio for gender detection: {}", e);
                    config.tts.voice = "zh-CN-XiaoxiaoNeural".to_string();
                    None
                }
            }
        } else {
            None
        };

    // 3. 确定 output 路径
    let output_path = args
        .output
        .unwrap_or_else(|| auto_output_path(&args.input, &config));

    if !quiet {
        println!("{} 输入视频: {:?}", "→".cyan(), args.input);
        println!("{} 输出视频: {:?}", "→".cyan(), output_path);
    }

    // 4. 构建 Pipeline
    let pipeline = build_pipeline(&config, tts_engine_choice, ref_audio_for_gender.as_deref())?;

    // 5. 运行流水线
    let segments = run_pipeline_with_progress(&pipeline, &args.input, &config, quiet).await?;

    if !quiet {
        println!(
            "{} 流水线完成，共 {} 个片段",
            "✓".green().bold(),
            segments.len()
        );
    }

    // 6. 合成最终视频
    compose_final_video(&args.input, &segments, &output_path, &config, quiet)?;

    // 7. 生成 SRT 字幕文件
    generate_subtitle_files(&segments, &output_path, &config, quiet)?;

    if !quiet {
        let elapsed = start_time.elapsed();
        println!("{} 视频已生成: {:?}", "✓".green().bold(), output_path);
        println!(
            "{} 总耗时: {} 分 {} 秒",
            "✓".green().bold(),
            elapsed.as_secs() / 60,
            elapsed.as_secs() % 60
        );
    }

    // 8. 显式释放资源（按依赖顺序）
    // Pipeline 持有所有引擎的 Arc 引用，显式 drop 确保子进程、模型等资源立即释放
    tracing::info!("Releasing pipeline resources...");

    // 先 drop pipeline，触发 Arc 引用计数归零，各引擎 Drop 依次执行：
    // - WhisperEngine → WhisperContext drop → whisper.cpp 释放模型内存 (~1.6GB)
    // - LlamaCppBackend → Drop impl → kill llama-server 子进程 + 释放模型内存
    // - EdgeTtsEngine / SayEngine → 释放缓存索引
    // - PersistentSubprocessCloneEngine → Drop impl → kill vt-tts server 子进程
    // - FfmpegAudioExtractor → 无特殊资源
    drop(pipeline);
    tracing::info!("Pipeline engines released (Whisper/LLM/TTS/Cloning)");

    // 清理性别检测临时文件
    if let Some(ref ref_path) = ref_audio_for_gender {
        if ref_path.exists() {
            let _ = std::fs::remove_file(ref_path);
            tracing::debug!("Cleaned up gender detection temp file: {:?}", ref_path);
        }
    }

    // 清理 TTS 缓存中的临时 MP3 文件（edge_tts.py 可能残留）
    let tts_cache_tmp = std::env::temp_dir().join("edge_tts");
    if tts_cache_tmp.exists() {
        let _ = std::fs::remove_dir_all(&tts_cache_tmp);
        tracing::debug!("Cleaned up edge_tts temp directory");
    }

    tracing::info!("All resources released.");

    Ok(())
}

/// 自动生成输出文件路径。
///
/// 在配置的输出目录下，将文件名添加 `_translated` 后缀，扩展名设为 `.mp4`。
fn auto_output_path(input: &Path, config: &Config) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());

    let output_dir = Path::new(&config.output_dir);
    let _ = std::fs::create_dir_all(output_dir);

    output_dir.join(format!("{stem}_translated.mp4"))
}

/// 构建声音克隆集成辅助器（如果配置启用）。
///
/// 根据 `config.cloning` 配置创建对应的克隆引擎：
/// - `gpt-sovits`: 通过 HTTP 调用 GPT-SoVITS API v2 服务
/// - `mock`: 测试用 Mock 引擎（生成正弦波 WAV）
///
/// 如果克隆未启用或引擎创建失败（如 API 服务未启动），返回 `None`。
fn build_cloning_integration(config: &Config) -> Result<Option<CloningIntegration>> {
    if !config.cloning.enabled {
        return Ok(None);
    }

    tracing::info!(
        "Initializing voice cloning engine: {}",
        config.cloning.engine
    );

    // 克隆合成参数（语速、输出目录等）
    let synth_config = CloningConfig {
        speed: config.tts.speed,
        output_dir: format!("{}/cloned", config.output_dir),
        ..Default::default()
    };

    let engine: Box<dyn vt_core::cloning::VoiceCloningEngine> = match config.cloning.engine.as_str()
    {
        "gpt-sovits" => match GptSoVitsEngine::new(config.cloning.clone()) {
            Ok(engine) => Box::new(engine),
            Err(e) => {
                tracing::warn!(
                    "Failed to initialize GPT-SoVITS engine: {}\n\
                        Voice cloning will be disabled. Standard TTS will be used.\n\
                        Hint: Start GPT-SoVITS API service first: python api_v2.py -p 9880",
                    e
                );
                return Ok(None);
            }
        },
        "subprocess-persistent" | "python-qwen-tts" => {
            match PersistentSubprocessCloneEngine::from_config(&config.cloning) {
                Ok(engine) => {
                    tracing::info!(
                        "Persistent subprocess clone engine initialized: command={:?}",
                        config.cloning.clone_command
                    );
                    Box::new(engine)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize persistent subprocess clone engine: {}\n\
                        Voice cloning will be disabled. Standard TTS will be used.\n\
                        Hint: Ensure clone_command is set in config.toml [cloning] section",
                        e
                    );
                    return Ok(None);
                }
            }
        }
        "subprocess" | "indextts" | "qwen3-tts" => {
            match SubprocessCloneEngine::from_config(&config.cloning) {
                Ok(engine) => {
                    tracing::info!(
                        "Subprocess clone engine initialized: command={:?}, model={:?}",
                        config.cloning.clone_command,
                        config.cloning.clone_model_path
                    );
                    Box::new(engine)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize subprocess clone engine: {}\n\
                        Voice cloning will be disabled. Standard TTS will be used.\n\
                        Hint: Ensure clone_command is set in config.toml [cloning] section",
                        e
                    );
                    return Ok(None);
                }
            }
        }
        "mock" => {
            tracing::warn!("Using mock cloning engine (for testing only)");
            Box::new(MockCloningEngine::new())
        }
        other => {
            tracing::warn!("Unknown cloning engine '{}', voice cloning disabled", other);
            return Ok(None);
        }
    };

    Ok(Some(CloningIntegration::new(engine, synth_config)))
}

/// 构建 Pipeline 实例。
///
/// 创建本地离线翻译引擎并将各引擎注入 `PipelineBuilder`。
fn build_pipeline(
    config: &Config,
    tts_engine_choice: &str,
    ref_audio: Option<&Path>,
) -> Result<Pipeline> {
    // 从魔搭下载/缓存 ASR 模型
    let asr_model_path = if config.asr.model.is_empty() {
        None
    } else {
        Some(
            vt_core::asr::ModelManager::new()
                .context("Failed to create ASR model manager")?
                .ensure_model(&config.asr.model)
                .context("Failed to download ASR model from ModelScope")?,
        )
    };

    let asr_config = WhisperConfig::default()
        .with_language(&config.asr.language)
        .with_metal(config.asr.use_metal)
        .with_model_path(asr_model_path.unwrap_or_default());
    let asr_engine =
        WhisperEngine::new(asr_config).context("Failed to initialize Whisper ASR engine")?;

    let extractor = FfmpegAudioExtractor::new();

    // 确保翻译模型已下载（从魔塔缓存或下载）
    let mut translation_config = config.translation.clone();
    if translation_config.model_path.is_none() {
        tracing::info!("Translation model_path not set, downloading from ModelScope...");
        let model_manager = vt_core::model_manager::ModelManager::new()
            .context("Failed to create translation model manager")?;
        let model_path = model_manager
            .load_model(
                &translation_config.model_source,
                "qwen2.5-3b-instruct-q5_k_m.gguf",
                None,
            )
            .context("Failed to download translation model from ModelScope")?;
        tracing::info!("Translation model ready: {:?}", model_path);
        translation_config.model_path = Some(model_path);
    }

    // 构建本地离线翻译引擎（GGUF 本地推理）
    let mut backend = LlamaCppBackend::from_config(&translation_config)
        .context("Failed to initialize LLM translation backend")?;

    // 加载术语表（内置编程术语 + 用户自定义术语）
    let mut glossary_entries: Vec<GlossaryEntry> = Vec::new();
    if config.translation.force_glossary {
        glossary_entries.extend(builtin_programming_terms());
    }
    if let Some(ref glossary_path) = config.translation.glossary_path {
        let path = std::path::Path::new(glossary_path);
        let terminology = if path.extension().is_some_and(|ext| ext == "json") {
            TerminologyManager::load_from_json(path)
        } else {
            TerminologyManager::load_from_csv(path)
        }
        .context("Failed to load terminology")?;
        glossary_entries.extend(terminology.entries().to_vec());
    }

    // 术语表 Markdown 注入系统提示词（让 LLM 遵循术语翻译）
    if !glossary_entries.is_empty() {
        let markdown = format_glossary_as_markdown(&glossary_entries);
        backend = backend.with_glossary_markdown(markdown);
    }

    let mut engine =
        LocalTranslationEngine::new(backend).with_batch_size(config.translation.batch_size);

    // 术语占位符替换（翻译前替换为占位符，翻译后还原）
    if !glossary_entries.is_empty() {
        let terminology = TerminologyManager::from_entries(glossary_entries)
            .context("Failed to create terminology manager")?;
        engine = engine.with_terminology(terminology);
    }

    // 构建声音克隆集成（如果配置启用）
    let cloning_integration = build_cloning_integration(config)?;

    // 构建 Pipeline builder（TTS 引擎根据选择动态创建）
    let mut builder = PipelineBuilder::default()
        .asr_engine(asr_engine)
        .translation_provider(engine)
        .audio_extractor(extractor);

    // TTS 引擎选择：edge (云端神经语音) 或 say (macOS 内置)
    if tts_engine_choice == "edge" {
        tracing::info!("Using Edge-TTS engine (cloud neural TTS)");
        let python_path = std::env::var("EDGE_TTS_PYTHON")
            .unwrap_or_else(|_| std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string()));
        let script_path = "scripts/python/edge_tts_synth.py".to_string();
        let edge_engine = vt_core::tts::EdgeTtsEngine::new(
            &config.tts,
            python_path,
            script_path,
            ref_audio.map(|p| p.to_path_buf()),
        );
        builder = builder.tts_engine(edge_engine);
    } else {
        let say_engine = SayEngine::new(&config.tts).context("Failed to initialize TTS engine")?;
        builder = builder.tts_engine(say_engine);
    }

    if let Some(integration) = cloning_integration {
        tracing::info!(
            "Voice cloning enabled, engine: {}",
            integration.engine_name()
        );
        builder = builder.cloning_integration(integration);
    }

    let pipeline = builder.build().context("Failed to build pipeline")?;

    Ok(pipeline)
}

/// 运行流水线并显示进度条。
///
/// 使用 `ProgressTracker` 实时追踪 ASR、翻译、TTS 三阶段进度，
/// 在终端显示带百分比、阶段详情和 ETA 的进度条。
async fn run_pipeline_with_progress(
    pipeline: &Pipeline,
    video_path: &Path,
    config: &Config,
    quiet: bool,
) -> Result<Vec<vt_core::models::segment::Segment>> {
    if quiet {
        let segments = pipeline
            .process_video(video_path, config)
            .await
            .context("Pipeline processing failed")?;
        return Ok(segments);
    }

    // 创建进度追踪器
    let tracker = Arc::new(ProgressTracker::new());

    // 主进度条（0-100 百分比）
    // 使用自定义 ETA 字符串而非 indicatif 的 {eta}（后者在非线性进度下产生天文数字）
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {percent:>3}% | {msg} | {elapsed}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>"),
    );
    pb.set_message("正在提取音频...");
    pb.enable_steady_tick(Duration::from_millis(200));

    // 启动轮询任务：定期读取 tracker 并更新进度条
    let poll_tracker = tracker.clone();
    let poll_pb = pb.clone();
    let poll_handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            let progress = poll_tracker.overall_progress();
            let percent = (progress * 100.0).round() as u64;
            poll_pb.set_position(percent);

            // 构建状态消息（包含自定义 ETA）
            let total_chunks = poll_tracker.total_chunks();
            let elapsed = poll_tracker.elapsed_secs();
            let eta_str = poll_tracker
                .eta_secs()
                .map(|eta| format!("剩余 ~{}", format_duration_eta(eta)))
                .unwrap_or_else(|| "计算中...".to_string());

            if total_chunks == 0 {
                poll_pb.set_message(format!(
                    "正在提取音频并分割... | 已用 {}",
                    format_duration_eta(elapsed)
                ));
                continue;
            }

            let asr_completed = poll_tracker.asr_completed();
            let trans_done = poll_tracker.translation_completed();
            let tts_done = poll_tracker.tts_completed();
            let total_segs = poll_tracker.total_segments();
            let asr_done = poll_tracker.is_asr_done();

            let stage_detail = if asr_done && total_segs > 0 {
                format!(
                    "语音识别 {asr_completed}/{total_chunks} | 翻译 {trans_done}/{total_segs} | 语音合成 {tts_done}/{total_segs}",
                )
            } else {
                format!(
                    "语音识别 {asr_completed}/{total_chunks} | 翻译 {trans_done} | 语音合成 {tts_done}",
                )
            };
            poll_pb.set_message(format!("{stage_detail} | {eta_str}"));

            // 检查是否完成
            if progress >= 0.95 {
                break;
            }
        }
    });

    // 运行流水线（带进度追踪）
    let result = pipeline
        .process_video_with_progress(video_path, config, &tracker)
        .await;

    // 等待轮询任务结束
    let _ = poll_handle.await;

    // 完成进度条
    pb.set_position(95);
    if result.is_ok() {
        pb.finish_with_message("流水线处理完成");
    } else {
        pb.finish_with_message("流水线处理失败");
    }

    let segments = result.context("Pipeline processing failed")?;
    Ok(segments)
}

/// 合成最终视频：将 TTS 音频按时间戳混合后与原视频合成。
///
/// 使用 `mix_audio_segments` 将所有 TTS 音频片段按其时间戳
/// 叠加到一条与原视频等长的静音音轨上，然后用该音轨替换原视频的音频。
/// 这保证了输出视频时长与原视频一致，且配音时间戳对齐。
fn compose_final_video(
    video_path: &Path,
    segments: &[vt_core::models::segment::Segment],
    output_path: &Path,
    config: &Config,
    quiet: bool,
) -> Result<()> {
    if segments.is_empty() {
        if !quiet {
            println!("{} 没有生成任何片段，跳过视频合成", "!".yellow().bold());
        }
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
        if !quiet {
            println!("{} 没有 TTS 音频，跳过视频合成", "!".yellow().bold());
        }
        return Ok(());
    }

    // 探测原视频时长
    let media_info = probe_media(video_path).context("Failed to probe original video")?;
    let total_duration = media_info.duration;
    tracing::info!(
        "Composing final video: {} audio segments, video duration={:.1}s",
        audio_segments.len(),
        total_duration
    );

    if !quiet {
        let compose_pb = ProgressBar::new_spinner();
        compose_pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner())
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        compose_pb.set_message("正在混合音频并合成视频...");
        compose_pb.enable_steady_tick(Duration::from_millis(100));

        let temp_dir = tempfile::TempDir::new().context("Failed to create temp directory")?;
        let mixed_audio = temp_dir.path().join("mixed_audio.wav");

        let mix_result = mix_audio_segments(
            &audio_segments,
            total_duration,
            &mixed_audio,
            config.audio_sync.mode,
            config.audio_sync.max_speed_ratio,
        )
        .context("Failed to mix audio segments")?;

        if !quiet {
            println!(
                "  {} 音频混合: 模式={:?}, 音频={:.1}s, 视频={:.1}s, 拉伸={:.3}x",
                "✓".green(),
                config.audio_sync.mode,
                mix_result.audio_duration_secs,
                total_duration,
                mix_result.video_stretch_factor
            );
        }

        // 如果音频比视频长，使用视频定格延长（而非整体慢放）
        let video_input: PathBuf;
        let effective_stretch: f64;
        if mix_result.video_stretch_factor > 1.01 {
            let extra_secs = mix_result.audio_duration_secs - total_duration;
            let extended_path = temp_dir.path().join("extended_input.mp4");
            std::fs::copy(video_path, &extended_path)?;
            extend_video_freeze_frame(&extended_path, extra_secs)?;
            tracing::info!(
                "Video extended by {:.1}s with freeze frame (audio={:.1}s > video={:.1}s)",
                extra_secs,
                mix_result.audio_duration_secs,
                total_duration
            );
            if !quiet {
                println!("  {} 视频定格延长 {:.1}s", "✓".green(), extra_secs);
            }
            video_input = extended_path;
            effective_stretch = 1.0;
        } else {
            video_input = video_path.to_path_buf();
            effective_stretch = mix_result.video_stretch_factor;
        }

        // 背景音乐混合
        if let Some(ref bgm_path) = config.background_music.path {
            let bgm = Path::new(bgm_path);
            if bgm.exists() {
                let bgm_output = temp_dir.path().join("mixed_with_bgm.wav");
                AudioPostProcessor::mix_background_music(
                    &mixed_audio,
                    bgm,
                    &bgm_output,
                    config.background_music.volume,
                    config.background_music.loop_bgm,
                )?;
                let _ = std::fs::rename(&bgm_output, &mixed_audio);
                if !quiet {
                    println!("  {} 背景音乐混合完成", "✓".green());
                }
            }
        }

        let composer = FfmpegVideoComposer::new();
        let result = composer
            .compose_video(
                &video_input,
                &mixed_audio,
                output_path,
                false,
                None,
                effective_stretch,
            )
            .context("Video composition failed");

        compose_pb.finish_with_message("视频合成完成");
        result?;
    } else {
        let temp_dir = tempfile::TempDir::new().context("Failed to create temp directory")?;
        let mixed_audio = temp_dir.path().join("mixed_audio.wav");

        let mix_result = mix_audio_segments(
            &audio_segments,
            total_duration,
            &mixed_audio,
            config.audio_sync.mode,
            config.audio_sync.max_speed_ratio,
        )
        .context("Failed to mix audio segments")?;

        // 如果音频比视频长，使用视频定格延长（而非整体慢放）
        let video_input: PathBuf;
        let effective_stretch: f64;
        if mix_result.video_stretch_factor > 1.01 {
            let extra_secs = mix_result.audio_duration_secs - total_duration;
            let extended_path = temp_dir.path().join("extended_input.mp4");
            std::fs::copy(video_path, &extended_path)?;
            extend_video_freeze_frame(&extended_path, extra_secs)?;
            tracing::info!(
                "Video extended by {:.1}s with freeze frame (audio={:.1}s > video={:.1}s)",
                extra_secs,
                mix_result.audio_duration_secs,
                total_duration
            );
            video_input = extended_path;
            effective_stretch = 1.0;
        } else {
            video_input = video_path.to_path_buf();
            effective_stretch = mix_result.video_stretch_factor;
        }

        // 背景音乐混合
        if let Some(ref bgm_path) = config.background_music.path {
            let bgm = Path::new(bgm_path);
            if bgm.exists() {
                let bgm_output = temp_dir.path().join("mixed_with_bgm.wav");
                AudioPostProcessor::mix_background_music(
                    &mixed_audio,
                    bgm,
                    &bgm_output,
                    config.background_music.volume,
                    config.background_music.loop_bgm,
                )?;
                let _ = std::fs::rename(&bgm_output, &mixed_audio);
            }
        }

        let composer = FfmpegVideoComposer::new();
        composer
            .compose_video(
                &video_input,
                &mixed_audio,
                output_path,
                false,
                None,
                effective_stretch,
            )
            .context("Video composition failed")?;
    }

    Ok(())
}

/// 生成 SRT 字幕文件
///
/// 根据配置的 `subtitle_type` 生成对应的 SRT 字幕文件：
/// - `None`: 不生成
/// - `Hard`/`Soft`: 生成目标语言字幕
/// - `HardBilingual`/`SoftBilingual`: 生成双语字幕
fn generate_subtitle_files(
    segments: &[vt_core::models::segment::Segment],
    video_output: &Path,
    config: &Config,
    quiet: bool,
) -> Result<()> {
    let subtitle_type = config.subtitle.subtitle_type;
    if subtitle_type == SubtitleType::None {
        return Ok(());
    }

    // 确定 SRT 输出路径（与视频同目录，同名 .srt）
    let srt_path = video_output.with_extension("srt");

    // 源语言名称
    let source_lang_name = match config.asr.language.as_str() {
        "en" => "English",
        "zh" => "Chinese",
        "ja" => "Japanese",
        "ko" => "Korean",
        "fr" => "French",
        "de" => "German",
        "es" => "Spanish",
        _ => "Source",
    };

    let srt_content = match subtitle_type {
        SubtitleType::HardBilingual | SubtitleType::SoftBilingual => {
            generate_bilingual_srt(segments, source_lang_name, "Chinese")
        }
        SubtitleType::Hard | SubtitleType::Soft => generate_srt(segments, true),
        SubtitleType::None => return Ok(()),
    };

    std::fs::write(&srt_path, &srt_content)
        .with_context(|| format!("Failed to write SRT file: {:?}", srt_path))?;

    if !quiet {
        println!(
            "{} 字幕文件已生成: {:?} ({:?})",
            "✓".green().bold(),
            srt_path,
            subtitle_type
        );
    }

    Ok(())
}

/// 执行 `batch` 子命令：批量处理目录中的所有视频。
///
/// # 参数
/// - `args`: batch 子命令参数
/// - `quiet`: 是否静默模式
pub async fn run_batch_command(args: BatchArgs, quiet: bool) -> Result<()> {
    if !args.input_dir.exists() {
        anyhow::bail!(
            "File not found: input directory {:?} does not exist",
            args.input_dir
        );
    }

    if !args.input_dir.is_dir() {
        anyhow::bail!("Config error: {:?} is not a directory", args.input_dir);
    }

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", args.output_dir))?;

    let video_files = find_video_files(&args.input_dir);

    if video_files.is_empty() {
        if !quiet {
            println!(
                "{} 在目录 {:?} 中未找到视频文件",
                "!".yellow().bold(),
                args.input_dir
            );
            println!("  支持的格式: {}", VIDEO_EXTENSIONS.join(", "));
        }
        return Ok(());
    }

    if !quiet {
        println!("{} 找到 {} 个视频文件", "→".cyan(), video_files.len());
    }

    let mut config = load_config(args.config.as_deref())?;
    config.output_dir = args.output_dir.to_string_lossy().to_string();

    // --no-clone: 跳过声音克隆，使用系统 TTS
    if args.no_clone {
        if !quiet {
            println!("{} 声音克隆已关闭，使用系统 TTS (say 命令)", "→".cyan());
        }
        config.cloning.enabled = false;
    }

    // --voice: 覆盖 TTS 音色
    if let Some(ref voice) = args.voice {
        if !quiet {
            println!("{} TTS 音色: {}", "→".cyan(), voice);
        }
        config.tts.voice = voice.clone();
    }

    let mut success_count = 0usize;
    let mut fail_count = 0usize;

    for (index, video_path) in video_files.iter().enumerate() {
        if !quiet {
            println!(
                "\n{} [{}/{}] 处理: {:?}",
                "→".cyan(),
                index + 1,
                video_files.len(),
                video_path
            );
        }

        let result = process_single_video_for_batch(video_path, &config, quiet).await;

        match result {
            Ok(()) => {
                success_count += 1;
                if !quiet {
                    println!("{} 处理成功", "✓".green().bold());
                }
            }
            Err(e) => {
                fail_count += 1;
                if !quiet {
                    eprintln!("{} 处理失败: {}", "✗".red().bold(), e);
                }
            }
        }
    }

    if !quiet {
        println!("\n{}", "═══ 批量处理摘要 ═══".bold());
        println!("  {} 成功: {}", "✓".green(), success_count);
        println!("  {} 失败: {}", "✗".red(), fail_count);
        println!("  总计: {}", video_files.len());
    }

    Ok(())
}

/// 批量处理中处理单个视频的辅助函数。
async fn process_single_video_for_batch(
    video_path: &Path,
    config: &Config,
    quiet: bool,
) -> Result<()> {
    let pipeline = build_pipeline(config, "say", None)?;

    let segments = run_pipeline_with_progress(&pipeline, video_path, config, quiet).await?;

    let stem = video_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());

    let output_path = Path::new(&config.output_dir).join(format!("{stem}_translated.mp4"));

    compose_final_video(video_path, &segments, &output_path, config, quiet)?;
    generate_subtitle_files(&segments, &output_path, config, quiet)?;

    Ok(())
}

/// 在目录中查找所有支持的视频文件。
///
/// 遍历目录（非递归），返回按文件名排序的视频文件列表。
fn find_video_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if let Some(ext_str) = ext.to_str() {
                        if VIDEO_EXTENSIONS.contains(&ext_str.to_lowercase().as_str()) {
                            files.push(path);
                        }
                    }
                }
            }
        }
    }

    files.sort();
    files
}
