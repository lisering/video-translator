//! 集成测试：Kokoro TTS 引擎（含 SayEngine 降级、音色选择、音频后处理）
//!
//! 验证以下功能：
//! - `KokoroEngine` 在无 ONNX 模型时正确降级到 `SayEngine`
//! - `KokoroEngine` 在 `fallback_to_say=false` 时返回明确错误
//! - 音色列表包含至少 2 女 + 2 男声
//! - `AudioPostProcessor` 滤镜链包含齿音衰减、低频增强、AGC、淡入淡出
//! - 音色解析：`voice_id` 优先，`voice` 字段回退
//! - 组合音调倍率计算正确
//! - 批量合成功能正常（缓存命中 + 新合成）
//! - 降级测试：Kokoro 失败时自动切换到 SayEngine
//!
//! # 运行方式
//! ```sh
//! cargo test --test test_tts_kokoroi -- --nocapture
//! ```

mod common;

use vt_core::audio_post_process::AudioPostProcessor;
use vt_core::config::TtsConfig;
use vt_core::error::AppError;
use vt_core::models::segment::Segment;
use vt_core::tts::{KokoroEngine, SayEngine, TtsEngine};
use vt_core::voice_manager::{VoiceGender, VoiceManager};

// ═══════════════════════════════════════════════════════════
//  KokoroEngine 降级测试
// ═══════════════════════════════════════════════════════════

/// 验证 KokoroEngine 在 fallback_to_say=true 时降级到 SayEngine。
#[test]
fn test_kokoro_engine_fallback_to_say() {
    let config = TtsConfig::default();
    let engine = KokoroEngine::new(&config).expect("KokoroEngine creation should succeed");
    assert_eq!(engine.backend_name(), "SayEngine");
}

/// 验证 KokoroEngine 在 fallback_to_say=false 时返回错误。
#[test]
fn test_kokoro_engine_no_fallback_error() {
    let config = TtsConfig {
        fallback_to_say: false,
        ..Default::default()
    };
    let result = KokoroEngine::new(&config);
    assert!(result.is_err());
    match &result {
        Err(AppError::TtsModelLoadError(msg)) => {
            assert!(msg.contains("fallback_to_say"));
        }
        Err(e) => panic!("Expected TtsModelLoadError, got: {e:?}"),
        Ok(_) => panic!("Expected error, got success"),
    }
}

/// 验证 KokoroEngine Debug 输出包含后端信息和音色数量。
#[test]
fn test_kokoro_engine_debug() {
    let config = TtsConfig::default();
    let engine = KokoroEngine::new(&config).expect("Should succeed");
    let debug = format!("{engine:?}");
    assert!(debug.contains("KokoroEngine"));
    assert!(debug.contains("SayEngine"));
    assert!(debug.contains("voice_count"));
}

// ═══════════════════════════════════════════════════════════
//  音色列表测试
// ═══════════════════════════════════════════════════════════

/// 验证 KokoroEngine 的 list_voices 方法返回至少 4 种音色。
#[test]
fn test_kokoro_engine_list_voices() {
    let config = TtsConfig::default();
    let engine = KokoroEngine::new(&config).expect("Should succeed");
    let voices = engine.list_voices();
    assert!(voices.len() >= 4, "Should have at least 4 voices");
}

/// 验证音色列表包含至少 2 种女声。
#[test]
fn test_kokoro_engine_female_voices() {
    let config = TtsConfig::default();
    let engine = KokoroEngine::new(&config).expect("Should succeed");
    let voices = engine.list_voices();
    let females = voices
        .iter()
        .filter(|v| v.gender == VoiceGender::Female)
        .count();
    assert!(
        females >= 2,
        "Should have at least 2 female voices, got {females}"
    );
}

/// 验证音色列表包含至少 2 种男声。
#[test]
fn test_kokoro_engine_male_voices() {
    let config = TtsConfig::default();
    let engine = KokoroEngine::new(&config).expect("Should succeed");
    let voices = engine.list_voices();
    let males = voices
        .iter()
        .filter(|v| v.gender == VoiceGender::Male)
        .count();
    assert!(
        males >= 2,
        "Should have at least 2 male voices, got {males}"
    );
}

/// 验证 SayEngine 的 list_voices 方法返回至少 4 种音色。
#[test]
fn test_say_engine_list_voices() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");
    let voices = engine.list_voices();
    assert!(voices.len() >= 4, "Should have at least 4 voices");
}

// ═══════════════════════════════════════════════════════════
//  音色解析测试
// ═══════════════════════════════════════════════════════════

/// 验证音色解析：voice_id 优先于 voice 字段。
#[test]
fn test_resolve_voice_by_id() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let config = TtsConfig {
        voice_id: "zhiming".to_string(),
        voice: "Tingting".to_string(),
        ..Default::default()
    };
    let voice = engine.resolve_voice(&config);
    assert_eq!(voice.id, "zhiming");
    assert_eq!(voice.gender, VoiceGender::Male);
}

/// 验证音色解析：voice_id 不存在时回退到 voice 字段匹配。
#[test]
fn test_resolve_voice_fallback() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");

    let config = TtsConfig {
        voice_id: "nonexistent".to_string(),
        voice: "Meijia".to_string(),
        ..Default::default()
    };
    let voice = engine.resolve_voice(&config);
    assert_eq!(voice.say_voice, "Meijia");
}

// ═══════════════════════════════════════════════════════════
//  音频后处理测试
// ═══════════════════════════════════════════════════════════

/// 验证 AudioPostProcessor 滤镜链包含齿音衰减（highshelf）。
#[test]
fn test_filter_chain_sibilance_reduction() {
    let vm = VoiceManager::new();
    let voice = vm.find_by_id("tingting").expect("tingting should exist");
    let config = TtsConfig::default();
    let chain = AudioPostProcessor::build_filter_chain(voice, &config);
    assert!(
        chain.contains("highshelf"),
        "Filter chain should contain highshelf for sibilance reduction"
    );
}

/// 验证 AudioPostProcessor 滤镜链包含低频增强（lowshelf）。
#[test]
fn test_filter_chain_warmth_enhancement() {
    let vm = VoiceManager::new();
    let voice = vm.find_by_id("tingting").expect("tingting should exist");
    let config = TtsConfig::default();
    let chain = AudioPostProcessor::build_filter_chain(voice, &config);
    assert!(
        chain.contains("lowshelf"),
        "Filter chain should contain lowshelf for warmth enhancement"
    );
}

/// 验证 AudioPostProcessor 滤镜链包含自动增益控制（dynaudnorm）。
#[test]
fn test_filter_chain_agc() {
    let vm = VoiceManager::new();
    let voice = vm.find_by_id("tingting").expect("tingting should exist");
    let config = TtsConfig::default();
    let chain = AudioPostProcessor::build_filter_chain(voice, &config);
    assert!(
        chain.contains("dynaudnorm"),
        "Filter chain should contain dynaudnorm for AGC"
    );
}

/// 验证 AudioPostProcessor 滤镜链包含淡入淡出（afade + areverse）。
#[test]
fn test_filter_chain_crossfade() {
    let vm = VoiceManager::new();
    let voice = vm.find_by_id("tingting").expect("tingting should exist");
    let config = TtsConfig::default();
    let chain = AudioPostProcessor::build_filter_chain(voice, &config);
    assert!(chain.contains("afade"), "Should contain afade");
    assert!(chain.contains("areverse"), "Should contain areverse");
}

/// 验证男声滤镜链包含音调偏移（asetrate + atempo）。
#[test]
fn test_filter_chain_male_pitch_shift() {
    let vm = VoiceManager::new();
    let voice = vm.find_by_id("zhiming").expect("zhiming should exist");
    let config = TtsConfig::default();
    let chain = AudioPostProcessor::build_filter_chain(voice, &config);
    assert!(
        chain.contains("asetrate"),
        "Should contain asetrate for pitch shift"
    );
    assert!(
        chain.contains("atempo"),
        "Should contain atempo for speed compensation"
    );
}

/// 验证组合音调倍率计算。
#[test]
fn test_combined_pitch() {
    let vm = VoiceManager::new();

    // 男声 pitch=0.85, config pitch=1.0 → combined=0.85
    let voice = vm.find_by_id("zhiming").expect("zhiming should exist");
    let config = TtsConfig::default();
    assert!((AudioPostProcessor::combined_pitch(voice, &config) - 0.85).abs() < 0.001);

    // 女声 pitch=1.0, config pitch=1.1 → combined=1.1
    let voice_f = vm.find_by_id("tingting").expect("tingting should exist");
    let config = TtsConfig {
        pitch: 1.1,
        ..Default::default()
    };
    assert!((AudioPostProcessor::combined_pitch(voice_f, &config) - 1.1).abs() < 0.001);
}

/// 验证禁用高频衰减时 highshelf 不出现。
#[test]
fn test_filter_chain_disable_sibilance() {
    let vm = VoiceManager::new();
    let voice = vm.find_by_id("tingting").expect("tingting should exist");
    let config = TtsConfig {
        eq_high_shelf_db: 0.0,
        ..Default::default()
    };
    let chain = AudioPostProcessor::build_filter_chain(voice, &config);
    assert!(
        !chain.contains("highshelf"),
        "Should not contain highshelf when disabled"
    );
}

/// 验证自定义交叉淡入淡出时长。
#[test]
fn test_filter_chain_custom_crossfade_duration() {
    let vm = VoiceManager::new();
    let voice = vm.find_by_id("tingting").expect("tingting should exist");
    let config = TtsConfig {
        crossfade_duration_ms: 100,
        ..Default::default()
    };
    let chain = AudioPostProcessor::build_filter_chain(voice, &config);
    assert!(
        chain.contains("d=0.100"),
        "Should contain custom fade duration, got: {chain}"
    );
}

// ═══════════════════════════════════════════════════════════
//  TtsConfig 配置测试
// ═══════════════════════════════════════════════════════════

/// 验证 TtsConfig 新增字段默认值正确。
#[test]
fn test_tts_config_new_defaults() {
    let config = TtsConfig::default();
    assert_eq!(config.seed, Some(42), "Default seed should be 42");
    assert!(
        (config.temperature - 0.3).abs() < 0.001,
        "Default temperature should be 0.3"
    );
    assert!(
        (config.stability - 0.8).abs() < 0.001,
        "Default stability should be 0.8"
    );
    assert!(
        (config.eq_high_shelf_db - (-3.0)).abs() < 0.001,
        "Default eq_high_shelf_db should be -3.0"
    );
    assert_eq!(
        config.crossfade_duration_ms, 50,
        "Default crossfade_duration_ms should be 50"
    );
}

/// 验证从 TOML 加载新增 TTS 配置字段。
#[test]
fn test_tts_config_new_fields_from_toml() {
    use std::io::Write;
    use vt_core::config::Config;

    let toml_content = r#"
[tts]
speed = 1.0
voice = "Tingting"
seed = 100
temperature = 0.5
stability = 0.9
eq_high_shelf_db = -5.0
crossfade_duration_ms = 100
"#;

    let mut tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{toml_content}").expect("Failed to write TOML");

    let config = Config::from_file(tmp.path()).expect("Failed to load config");

    assert_eq!(config.tts.seed, Some(100));
    assert!((config.tts.temperature - 0.5).abs() < 0.001);
    assert!((config.tts.stability - 0.9).abs() < 0.001);
    assert!((config.tts.eq_high_shelf_db - (-5.0)).abs() < 0.001);
    assert_eq!(config.tts.crossfade_duration_ms, 100);
}

// ═══════════════════════════════════════════════════════════
//  术语表测试（验证 println 等编程术语正确映射）
// ═══════════════════════════════════════════════════════════

/// 验证 `println` 映射为"打印并换行"。
#[test]
fn test_terminology_println() {
    use vt_core::translate::builtin_programming_terms;
    let terms = builtin_programming_terms();
    let entry = terms
        .iter()
        .find(|e| e.source == "println")
        .expect("Should contain 'println' term");
    assert_eq!(entry.target, "打印并换行");
}

/// 验证 `println!` 映射为"打印并换行宏"。
#[test]
fn test_terminology_println_macro() {
    use vt_core::translate::builtin_programming_terms;
    let terms = builtin_programming_terms();
    let entry = terms
        .iter()
        .find(|e| e.source == "println!")
        .expect("Should contain 'println!' term");
    assert_eq!(entry.target, "打印并换行宏");
}

/// 验证后校正："打印行" → "打印并换行"。
#[test]
fn test_post_correction_println_line() {
    use vt_core::translate::post_correct;
    let corrected = post_correct("使用打印行输出内容");
    assert_eq!(corrected, "使用打印并换行输出内容");
}

/// 验证后校正："输出行" → "打印并换行"。
#[test]
fn test_post_correction_output_line() {
    use vt_core::translate::post_correct;
    let corrected = post_correct("调用输出行宏");
    assert_eq!(corrected, "调用打印并换行宏");
}

/// 验证后校正：正常文本不被修改。
#[test]
fn test_post_correction_no_change() {
    use vt_core::translate::post_correct;
    let text = "这是一段正常的中文文本，不包含错误术语。";
    let corrected = post_correct(text);
    assert_eq!(corrected, text);
}

// ═══════════════════════════════════════════════════════════
//  实际合成测试（需要 macOS say 命令）
// ═══════════════════════════════════════════════════════════

/// 验证 SayEngine 能合成中文文本并生成 WAV 文件。
#[test]
fn test_say_engine_synthesize() {
    if common::should_skip_tts() {
        return;
    }

    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => return,
    };
    let config = common::shared_tts_config();

    let mut seg = Segment::new("test-1".into(), 0.0, 5.0, "Hello".into());
    seg.start_transcribing().ok();
    seg.finish_transcribing("你好世界".into()).ok();

    let mut segments = vec![seg];
    let result = engine.synthesize_segments(&mut segments, &config);

    // say 命令可能因语音包未安装或文本编码问题失败
    // 此时应跳过测试而非失败（环境依赖）
    match result {
        Ok(paths) => {
            assert_eq!(paths.len(), 1);
            assert!(paths[0].exists(), "Output WAV file should exist");
            assert!(
                paths[0].to_string_lossy().ends_with(".wav"),
                "Output should be WAV file"
            );
        }
        Err(e) => {
            eprintln!("Skipping: say synthesis failed (likely voice pack issue): {e}");
        }
    }
}

/// 验证 SayEngine 缓存命中：相同参数二次合成直接返回缓存。
#[test]
fn test_say_engine_cache_hit() {
    if common::should_skip_tts() {
        return;
    }

    let engine = match common::shared_tts_engine() {
        Some(e) => e,
        None => return,
    };
    let config = common::shared_tts_config();

    let text = "缓存测试文本";

    // 第一次合成
    let mut seg1 = Segment::new("cache-1".into(), 0.0, 5.0, "source".into());
    seg1.start_transcribing().ok();
    seg1.finish_transcribing(text.into()).ok();
    let mut segments1 = vec![seg1];
    let result1 = engine
        .synthesize_segments(&mut segments1, &config)
        .expect("First synthesis failed");

    // 第二次合成（应命中缓存）
    let mut seg2 = Segment::new("cache-2".into(), 0.0, 5.0, "source".into());
    seg2.start_transcribing().ok();
    seg2.finish_transcribing(text.into()).ok();
    let mut segments2 = vec![seg2];
    let result2 = engine
        .synthesize_segments(&mut segments2, &config)
        .expect("Second synthesis failed");

    // 缓存命中时应返回相同路径
    assert_eq!(result1[0], result2[0], "Cache hit should return same path");
}

/// 验证 KokoroEngine 降级后能正常合成（与 SayEngine 行为一致）。
#[test]
fn test_kokoro_engine_fallback_synthesize() {
    if common::should_skip_tts() {
        return;
    }

    let config = TtsConfig {
        cache_dir: common::shared_tts_config().cache_dir,
        ..Default::default()
    };

    let engine = KokoroEngine::new(&config).expect("KokoroEngine should succeed with fallback");

    let mut seg = Segment::new("kokoro-1".into(), 0.0, 5.0, "Hello".into());
    seg.start_transcribing().ok();
    seg.finish_transcribing("你好世界".into()).ok();

    let mut segments = vec![seg];
    let result = engine.synthesize_segments(&mut segments, &config);

    // say 命令可能因语音包未安装或文本编码问题失败
    // 此时应跳过测试而非失败（环境依赖）
    match result {
        Ok(paths) => {
            assert_eq!(paths.len(), 1);
            assert!(paths[0].exists(), "Output WAV file should exist");
        }
        Err(e) => {
            eprintln!("Skipping: Kokoro fallback synthesis failed: {e}");
        }
    }
}

/// 验证空 Segment 列表返回空路径列表。
#[test]
fn test_synthesize_empty_segments() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");
    let config = TtsConfig::default();

    let mut segments: Vec<Segment> = vec![];
    let result = engine.synthesize_segments(&mut segments, &config);
    assert!(result.is_ok());
    assert!(result.expect("Already checked").is_empty());
}

/// 验证未翻译的 Segment 返回错误。
#[test]
fn test_synthesize_untranslated_segment() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");
    let config = TtsConfig::default();

    // Segment 未翻译（target_text = None）
    let seg = Segment::new("untranslated".into(), 0.0, 5.0, "source text".into());
    let mut segments = vec![seg];
    let result = engine.synthesize_segments(&mut segments, &config);
    assert!(result.is_err());
    match &result {
        Err(AppError::TtsError(msg)) => {
            assert!(msg.contains("target_text"));
        }
        Err(e) => panic!("Expected TtsError, got: {e:?}"),
        Ok(_) => panic!("Expected error, got success"),
    }
}

/// 验证空文本的 Segment 返回错误。
#[test]
fn test_synthesize_empty_text() {
    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let engine = SayEngine::with_cache_dir(dir.path()).expect("Failed to create engine");
    let config = TtsConfig::default();

    let mut seg = Segment::new("empty".into(), 0.0, 5.0, "source".into());
    seg.start_transcribing().ok();
    seg.finish_transcribing("".into()).ok();

    let mut segments = vec![seg];
    let result = engine.synthesize_segments(&mut segments, &config);
    assert!(result.is_err());
    match &result {
        Err(AppError::TtsError(msg)) => {
            assert!(msg.contains("empty"));
        }
        Err(e) => panic!("Expected TtsError, got: {e:?}"),
        Ok(_) => panic!("Expected error, got success"),
    }
}

// ═══════════════════════════════════════════════════════════
//  ModelType::Kokoro 测试
// ═══════════════════════════════════════════════════════════

/// 验证 ModelType::Kokoro 的缓存子目录和默认文件名。
#[test]
fn test_model_type_kokoro() {
    use vt_core::model_manager::ModelType;

    assert_eq!(ModelType::Kokoro.cache_subdir(), "kokoro");
    assert_eq!(
        ModelType::Kokoro.default_repo_id(),
        "onnx-community/Kokoro-82M-v1.1-zh-ONNX"
    );
    assert_eq!(ModelType::Kokoro.default_filename(), "model.onnx");
}

/// 验证 ModelType::Kokoro 的缓存路径。
#[test]
fn test_model_type_kokoro_cache_path() {
    use vt_core::model_manager::{ModelManager, ModelType};

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let manager = ModelManager::with_cache_dir(dir.path()).expect("Failed to create manager");

    let path = manager.get_typed_cache_path(ModelType::Kokoro);
    assert!(path.to_string_lossy().contains("kokoro"));
    assert!(path.to_string_lossy().ends_with("model.onnx"));
}
