//! 性能基准测试套件
//!
//! 使用 criterion 框架对核心模块进行性能基准测试。
//!
//! # 运行方式
//! ```sh
//! cargo bench --workspace
//! ```
//!
//! # 测试内容
//! - 音频分割（固定时长 + VAD）
//! - WAV 文件读写
//! - 说话人分离（Mock 引擎）
//! - 声音克隆（Mock 引擎）
//! - 批量队列操作
//! - 检查点序列化/反序列化

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;

use vt_core::batch::{BatchJob, BatchQueue, Priority};
use vt_core::checkpoint::{Checkpoint, ProcessingStage};
use vt_core::cloning::{CloningConfig, MockCloningEngine, VoiceCloningEngine};
use vt_core::config::{AudioSyncMode, BatchConfig, Config};
use vt_core::diarization::{
    assign_speakers_to_segments, DiarizationEngine, DiarizationResult, MockDiarizationEngine,
    SpeakerSegment,
};
use vt_core::media::mix_audio_segments;
use vt_core::models::segment::Segment;

// ─── 辅助函数 ─────────────────────────────────────────────

/// 生成测试用 f32 采样数据
fn generate_samples(duration_secs: f64, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f64) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (t * 440.0 * 2.0 * std::f64::consts::PI).sin() as f32 * 0.5
        })
        .collect()
}

/// 生成测试用 WAV 文件
fn create_test_wav(path: &std::path::Path, duration_secs: f64, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV");
    let num_samples = (duration_secs * sample_rate as f64) as usize;

    for i in 0..num_samples {
        let sample = ((i as f64 * 0.1).sin() * 16000.0) as i16;
        writer.write_sample(sample).expect("Failed to write sample");
    }

    writer.finalize().expect("Failed to finalize WAV");
}

// ─── 音频分割基准 ─────────────────────────────────────────

fn bench_audio_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_split");

    for duration in [10.0, 60.0, 300.0] {
        let samples = generate_samples(duration, 16000);

        group.bench_with_input(
            BenchmarkId::new("fixed_duration", format!("{duration}s")),
            &samples,
            |b, samples| {
                b.iter(|| {
                    let chunk_size = (30.0 * 16000.0) as usize;
                    let mut chunks = Vec::new();
                    let mut start = 0;
                    while start < samples.len() {
                        let end = (start + chunk_size).min(samples.len());
                        chunks.push(&samples[start..end]);
                        start = end;
                    }
                    black_box(chunks.len());
                });
            },
        );
    }

    group.finish();
}

// ─── WAV 读写基准 ─────────────────────────────────────────

fn bench_wav_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("wav_io");

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    for duration in [1.0, 10.0, 60.0] {
        let wav_path = dir.path().join(format!("test_{duration}.wav"));
        create_test_wav(&wav_path, duration, 16000);

        // WAV 读取
        group.bench_with_input(
            BenchmarkId::new("read", format!("{duration}s")),
            &wav_path,
            |b, path| {
                b.iter(|| {
                    let reader = hound::WavReader::open(path).expect("Failed to open WAV");
                    let samples: Vec<i16> = reader.into_samples().filter_map(Result::ok).collect();
                    black_box(samples.len());
                });
            },
        );

        // WAV 写入
        let samples = generate_samples(duration, 16000);
        group.bench_with_input(
            BenchmarkId::new("write", format!("{duration}s")),
            &samples,
            |b, samples| {
                b.iter(|| {
                    let path = dir.path().join("bench_write.wav");
                    let spec = hound::WavSpec {
                        channels: 1,
                        sample_rate: 16000,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    };
                    let mut writer =
                        hound::WavWriter::create(&path, spec).expect("Failed to create WAV");
                    for s in samples {
                        let i16_sample = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                        writer
                            .write_sample(i16_sample)
                            .expect("Failed to write sample");
                    }
                    writer.finalize().expect("Failed to finalize WAV");
                });
            },
        );
    }

    group.finish();
}

// ─── 说话人分离基准 ───────────────────────────────────────

fn bench_diarization(c: &mut Criterion) {
    let mut group = c.benchmark_group("diarization");

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    for duration in [10.0, 60.0, 300.0] {
        let wav_path = dir.path().join(format!("dia_{duration}.wav"));
        create_test_wav(&wav_path, duration, 16000);

        let engine = MockDiarizationEngine::new(2);

        group.bench_with_input(
            BenchmarkId::new("mock_engine", format!("{duration}s")),
            &wav_path,
            |b, path| {
                b.iter(|| {
                    let result = engine.diarize(path).expect("diarize failed");
                    black_box(result.segments.len());
                });
            },
        );
    }

    // 说话人映射基准
    let segments: Vec<Segment> = (0..100)
        .map(|i| {
            Segment::new(
                format!("seg-{i}"),
                i as f64 * 5.0,
                (i + 1) as f64 * 5.0,
                format!("text-{i}"),
            )
        })
        .collect();

    let diarization = DiarizationResult::new(
        (0..200)
            .map(|i| {
                let speaker = if i % 2 == 0 {
                    "SPEAKER_00"
                } else {
                    "SPEAKER_01"
                };
                SpeakerSegment::new(speaker, i as f64 * 2.5, (i + 1) as f64 * 2.5)
            })
            .collect(),
        0.1,
    );

    group.bench_function("assign_speakers_100_segments", |b| {
        b.iter(|| {
            let mut segs = segments.clone();
            assign_speakers_to_segments(&mut segs, &diarization);
            black_box(segs.len());
        });
    });

    group.finish();
}

// ─── 声音克隆基准 ─────────────────────────────────────────

fn bench_cloning(c: &mut Criterion) {
    let mut group = c.benchmark_group("cloning");

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    // 创建参考音频
    let ref_path = dir.path().join("reference.wav");
    create_test_wav(&ref_path, 5.0, 16000);

    let engine = MockCloningEngine::new();
    let config = CloningConfig {
        output_dir: dir.path().join("cloned").to_string_lossy().into_owned(),
        ..Default::default()
    };

    for text_len in [10, 50, 200] {
        let text: String = "你好".repeat(text_len);

        group.bench_with_input(
            BenchmarkId::new("mock_synthesize", format!("{text_len}chars")),
            &text,
            |b, text| {
                b.iter(|| {
                    let path = engine
                        .clone_and_synthesize(&ref_path, text, &config)
                        .expect("synthesize failed");
                    black_box(path);
                });
            },
        );
    }

    group.finish();
}

// ─── 批量队列基准 ─────────────────────────────────────────

fn bench_batch_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_queue");

    // 添加 + 出队基准
    for job_count in [10, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("add_and_dequeue", job_count),
            &job_count,
            |b, &count| {
                b.iter(|| {
                    let batch_config = BatchConfig {
                        max_concurrent: 4,
                        memory_threshold: 90.0,
                        enable_priority: true,
                    };
                    let mut queue = BatchQueue::with_config(batch_config);

                    for i in 0..count {
                        let job = BatchJob::new(
                            format!("job-{i}"),
                            PathBuf::from(format!("input{i}.mp4")),
                            PathBuf::from(format!("output{i}.mp4")),
                            Config::default(),
                            if i % 3 == 0 {
                                Priority::High
                            } else {
                                Priority::Normal
                            },
                        );
                        queue.add_job(job);
                    }

                    // 出队并完成所有任务
                    while let Some(job) = queue.next_job() {
                        queue.complete_job(&job.id);
                    }

                    black_box(queue.completed_count());
                });
            },
        );
    }

    // 状态查询基准
    let batch_config = BatchConfig {
        max_concurrent: 4,
        memory_threshold: 90.0,
        enable_priority: true,
    };
    let mut queue = BatchQueue::with_config(batch_config);
    for i in 0..1000 {
        queue.add_job(BatchJob::new(
            format!("job-{i}"),
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            Config::default(),
            Priority::Normal,
        ));
    }

    group.bench_function("get_all_statuses_1000", |b| {
        b.iter(|| {
            let statuses = queue.get_all_statuses();
            black_box(statuses.len());
        });
    });

    group.finish();
}

// ─── 检查点基准 ───────────────────────────────────────────

fn bench_checkpoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint");

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let manager = vt_core::checkpoint::CheckpointManager::with_dir(dir.path().to_path_buf());

    for seg_count in [10, 100, 500] {
        // 创建检查点
        let mut cp = Checkpoint::new(
            format!("bench-{seg_count}"),
            PathBuf::from("/path/to/video.mp4"),
            ProcessingStage::Tts,
        );

        for i in 0..seg_count {
            let mut seg = Segment::new(
                format!("seg-{i}"),
                i as f64 * 5.0,
                (i + 1) as f64 * 5.0,
                format!("text-{i}"),
            );
            seg.start_transcribing().ok();
            seg.finish_transcribing(format!("翻译-{i}")).ok();
            seg.start_synthesizing().ok();
            seg.finish_synthesizing(format!("/tmp/audio{i}.wav")).ok();
            cp.add_segment(seg);
        }

        // 序列化基准
        group.bench_with_input(BenchmarkId::new("serialize", seg_count), &cp, |b, cp| {
            b.iter(|| {
                let json = cp.to_json().expect("serialize failed");
                black_box(json.len());
            });
        });

        // 反序列化基准
        let json = cp.to_json().expect("serialize failed");
        group.bench_with_input(
            BenchmarkId::new("deserialize", seg_count),
            &json,
            |b, json| {
                b.iter(|| {
                    let cp = Checkpoint::from_json(json).expect("deserialize failed");
                    black_box(cp.completed_count());
                });
            },
        );

        // 保存/加载基准
        group.bench_with_input(BenchmarkId::new("save_load", seg_count), &cp, |b, cp| {
            b.iter(|| {
                manager.save(cp).expect("save failed");
                let loaded = manager.load(&cp.job_id).expect("load failed");
                black_box(loaded.is_some());
            });
        });
    }

    group.finish();
}

// ─── Segment 状态转换基准 ────────────────────────────────

fn bench_segment_state_machine(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_state_machine");

    group.bench_function("full_lifecycle", |b| {
        b.iter(|| {
            let mut seg = Segment::new("bench".into(), 0.0, 5.0, "Hello".into());
            seg.start_transcribing().ok();
            seg.finish_transcribing("你好".into()).ok();
            seg.start_synthesizing().ok();
            seg.finish_synthesizing("/tmp/audio.wav".into()).ok();
            black_box(seg.status);
        });
    });

    group.bench_function("create_1000_segments", |b| {
        b.iter(|| {
            let segments: Vec<Segment> = (0..1000)
                .map(|i| {
                    Segment::new(
                        format!("seg-{i}"),
                        i as f64,
                        (i + 1) as f64,
                        format!("text-{i}"),
                    )
                })
                .collect();
            black_box(segments.len());
        });
    });

    group.finish();
}

// ─── Config 加载基准 ─────────────────────────────────────

fn bench_config(c: &mut Criterion) {
    let mut group = c.benchmark_group("config");

    let toml_content = r#"
[asr]
model = "whisper-large-v3"
use_metal = true
language = "en"

[tts]
speed = 1.0
voice = "zh-CN-XiaoxiaoNeural"

[batch]
max_concurrent = 3
memory_threshold = 80.0
enable_priority = true

[checkpoint]
enabled = true
retention_days = 7

[diarization]
enabled = false
engine = "speakrs"
use_coreml = true
"#;

    group.bench_function("toml_parse", |b| {
        b.iter(|| {
            let config: Config = toml::from_str(toml_content).expect("parse failed");
            black_box(config.asr.model.len());
        });
    });

    group.bench_function("default_config", |b| {
        b.iter(|| {
            let config = Config::default();
            black_box(config.asr.model.len());
        });
    });

    group.bench_function("serde_roundtrip", |b| {
        b.iter(|| {
            let config = Config::default();
            let json = serde_json::to_string(&config).expect("serialize failed");
            let restored: Config = serde_json::from_str(&json).expect("deserialize failed");
            black_box(restored.asr.model.len());
        });
    });

    group.finish();
}

// ─── 音频混合基准 ─────────────────────────────────────────

fn bench_audio_mix(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_mix");

    let dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    // 为不同数量的段创建测试音频文件
    for seg_count in [10, 50, 100] {
        let mut segments: Vec<(f64, f64, std::path::PathBuf)> = Vec::new();

        for i in 0..seg_count {
            let wav_path = dir.path().join(format!("mix_seg_{i}.wav"));
            create_test_wav(&wav_path, 2.0, 16000); // 2秒音频段
                                                    // 每段 3 秒槽位 (start=i*3, end=i*3+3)
            segments.push((i as f64 * 3.0, (i + 1) as f64 * 3.0, wav_path));
        }

        let total_duration = seg_count as f64 * 3.0;
        let output_path = dir.path().join(format!("mixed_{seg_count}.wav"));

        // 转换为引用路径
        let seg_refs: Vec<(f64, f64, &std::path::Path)> = segments
            .iter()
            .map(|(s, e, p)| (*s, *e, p.as_path()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("mix_audio_segments", seg_count),
            &seg_refs,
            |b, seg_refs| {
                b.iter(|| {
                    mix_audio_segments(
                        black_box(seg_refs),
                        total_duration,
                        &output_path,
                        AudioSyncMode::SpeedUp,
                        1.3,
                    )
                    .expect("mix failed");
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_audio_split,
    bench_wav_io,
    bench_diarization,
    bench_cloning,
    bench_batch_queue,
    bench_checkpoint,
    bench_segment_state_machine,
    bench_config,
    bench_audio_mix,
);

criterion_main!(benches);
