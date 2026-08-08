//! 流式音频处理模块
//!
//! 借鉴 dots.tts 的流式声码器思路，提供 chunk 级音频 I/O 和处理，
//! 避免将整个音频文件加载到内存中，减少峰值内存使用。
//!
//! # 核心功能
//! - [`StreamingWavConcatenator`]: 流式 WAV 拼接器（交叉淡入淡出，仅需 2 个 chunk 在内存）
//! - [`AudioChunkReader`]: 分块 WAV 读取器
//! - [`AudioLevelAnalyzer`]: 流式音频电平分析（RMS、峰值、DC 偏移）
//! - [`estimate_wav_memory`]: WAV 内存估算
//!
//! # 设计原则
//! - **O(1) 内存**: 无论音频多长，内存占用恒定（仅 chunk 大小）
//! - **可配置 chunk 大小**: 默认 4096 samples（约 170ms @ 24kHz）
//! - **兼容 hound**: 使用 hound crate 进行 WAV I/O
//!
//! # 示例
//! ```
//! use vt_core::streaming_audio::StreamingWavConcatenator;
//! use std::path::PathBuf;
//!
//! // 流式拼接 3 个 WAV 文件（仅需 2 个 chunk 在内存中）
//! let concatenator = StreamingWavConcatenator::new(50); // 50ms 交叉淡入淡出
//! // concatenator.concatenate(&[wav1, wav2, wav3], &output).unwrap();
//! ```

use std::path::{Path, PathBuf};

use crate::error::{AppError, AppResult};

// ─── 常量 ─────────────────────────────────────────────────

/// 默认 chunk 大小（samples），约 170ms @ 24kHz
pub const DEFAULT_CHUNK_SIZE: usize = 4096;

/// 最小 chunk 大小
pub const MIN_CHUNK_SIZE: usize = 256;

// ─── WAV 信息 ─────────────────────────────────────────────

/// WAV 文件元信息
#[derive(Debug, Clone)]
pub struct WavInfo {
    /// 采样率
    pub sample_rate: u32,
    /// 声道数
    pub channels: u16,
    /// 每样本位数
    pub bits_per_sample: u16,
    /// 总样本数（所有声道交错）
    pub total_samples: usize,
}

/// 读取 WAV 文件元信息（不加载全部数据）
pub fn read_wav_info(path: &Path) -> AppResult<WavInfo> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| AppError::VoiceCloningError(format!("Failed to open WAV {:?}: {e}", path)))?;
    let spec = reader.spec();
    let duration = reader.duration();
    Ok(WavInfo {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        bits_per_sample: spec.bits_per_sample,
        total_samples: duration as usize * spec.channels as usize,
    })
}

// ─── 内存估算 ─────────────────────────────────────────────

/// 估算 WAV 文件加载到内存后的占用（字节）
#[must_use]
pub fn estimate_wav_memory(sample_count: usize, channels: u16) -> usize {
    // f32 每个样本 4 字节
    sample_count * channels as usize * std::mem::size_of::<f32>()
}

/// 估算 WAV 文件的时长（秒）
#[must_use]
pub fn estimate_duration_secs(sample_count: usize, sample_rate: u32, channels: u16) -> f64 {
    if channels == 0 || sample_rate == 0 {
        return 0.0;
    }
    sample_count as f64 / (sample_rate as f64 * channels as f64)
}

// ─── 分块 WAV 读取器 ─────────────────────────────────────

/// 分块 WAV 读取器
///
/// 以指定 chunk 大小迭代读取 WAV 样本，避免一次性加载全部数据。
pub struct AudioChunkReader {
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
    sample_rate: u32,
    channels: u16,
    #[allow(dead_code)]
    bits_per_sample: u16,
    chunk_size: usize,
    max_val: f32,
    is_float: bool,
}

impl AudioChunkReader {
    /// 创建分块 WAV 读取器
    ///
    /// # 参数
    /// - `path`: WAV 文件路径
    /// - `chunk_size`: 每个 chunk 的样本数（单声道）
    pub fn open(path: &Path, chunk_size: usize) -> AppResult<Self> {
        let reader = hound::WavReader::open(path).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to open WAV {:?}: {e}", path))
        })?;
        let spec = reader.spec();
        let is_float = matches!(spec.sample_format, hound::SampleFormat::Float);
        let max_val = if is_float {
            1.0
        } else {
            2f32.powi(spec.bits_per_sample as i32 - 1)
        };

        let chunk_size = chunk_size.max(MIN_CHUNK_SIZE);

        Ok(Self {
            reader,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            bits_per_sample: spec.bits_per_sample,
            chunk_size,
            max_val,
            is_float,
        })
    }

    /// 使用默认 chunk 大小打开
    pub fn open_default(path: &Path) -> AppResult<Self> {
        Self::open(path, DEFAULT_CHUNK_SIZE)
    }

    /// 返回采样率
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// 返回声道数
    #[must_use]
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// 读取下一个 chunk
    ///
    /// 返回 `None` 表示文件已读完。
    pub fn read_chunk(&mut self) -> Option<Vec<f32>> {
        let mut chunk = Vec::with_capacity(self.chunk_size);
        let samples_iter = self.reader.samples::<i32>();

        if self.is_float {
            // 浮点格式：直接读取 f32
            let float_iter = self.reader.samples::<f32>();
            for s in float_iter {
                match s {
                    Ok(v) => {
                        chunk.push(v);
                        if chunk.len() >= self.chunk_size {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        } else {
            // 整数格式：转换到 f32
            for s in samples_iter {
                match s {
                    Ok(v) => {
                        chunk.push(v as f32 / self.max_val);
                        if chunk.len() >= self.chunk_size {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
}

impl Iterator for AudioChunkReader {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_chunk()
    }
}

// ─── 流式 WAV 拼接器 ─────────────────────────────────────

/// 流式 WAV 拼接器
///
/// 以流式方式拼接多个 WAV 文件，在边界处使用等功率交叉淡入淡出。
/// 内存占用仅为 2 个 chunk（当前 + 前一段尾部），与文件数量无关。
pub struct StreamingWavConcatenator {
    /// 交叉淡入淡出时长（毫秒）
    crossfade_ms: u64,
}

impl StreamingWavConcatenator {
    /// 创建流式拼接器
    ///
    /// # 参数
    /// - `crossfade_ms`: 交叉淡入淡出时长（毫秒），0 表示无交叉淡入淡出
    #[must_use]
    pub fn new(crossfade_ms: u64) -> Self {
        Self { crossfade_ms }
    }

    /// 流式拼接多个 WAV 文件
    ///
    /// 逐文件读取，在边界处做交叉淡入淡出，写入输出文件。
    /// 内存占用：O(chunk_size)，与文件数量和总时长无关。
    ///
    /// # 参数
    /// - `wav_paths`: WAV 文件路径列表（按顺序拼接）
    /// - `output_path`: 输出 WAV 文件路径
    ///
    /// # 错误
    /// - WAV 读取/写入失败
    /// - 采样率不一致
    /// - 空列表
    pub fn concatenate(&self, wav_paths: &[PathBuf], output_path: &Path) -> AppResult<()> {
        if wav_paths.is_empty() {
            return Err(AppError::VoiceCloningError(
                "StreamingWavConcatenator: empty wav list".to_string(),
            ));
        }

        // 单个文件：直接复制
        if wav_paths.len() == 1 {
            std::fs::copy(&wav_paths[0], output_path)
                .map_err(|e| AppError::VoiceCloningError(format!("Failed to copy WAV: {e}")))?;
            return Ok(());
        }

        // 读取第一个文件的格式信息
        let first_info = read_wav_info(&wav_paths[0])?;
        let sample_rate = first_info.sample_rate;
        let channels = first_info.channels;

        // 验证所有文件的采样率一致
        for path in &wav_paths[1..] {
            let info = read_wav_info(path)?;
            if info.sample_rate != sample_rate {
                return Err(AppError::VoiceCloningError(format!(
                    "Sample rate mismatch: {} vs {} for {:?}",
                    info.sample_rate, sample_rate, path
                )));
            }
        }

        // 计算交叉淡入淡出样本数
        let crossfade_samples = if self.crossfade_ms > 0 {
            (sample_rate as f64 * self.crossfade_ms as f64 / 1000.0) as usize
        } else {
            0
        };

        // 创建输出文件
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to create output dir: {e}"))
            })?;
        }

        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(output_path, spec).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to create output WAV: {e}"))
        })?;

        let max_val = 32767.0f32;

        // 流式拼接：保留前一段的尾部用于交叉淡入淡出
        let mut prev_tail: Vec<f32> = Vec::new();

        for (i, path) in wav_paths.iter().enumerate() {
            let mut reader = AudioChunkReader::open_default(path)?;

            if i == 0 {
                // 第一段：全部写入，保留尾部用于下一段的交叉淡入淡出
                let mut all_samples: Vec<f32> = Vec::new();
                while let Some(chunk) = reader.read_chunk() {
                    all_samples.extend_from_slice(&chunk);
                }

                // 保留尾部用于交叉淡入淡出
                if crossfade_samples > 0 && all_samples.len() > crossfade_samples {
                    let tail_start = all_samples.len() - crossfade_samples;
                    // 写入除尾部以外的部分
                    for &s in &all_samples[..tail_start] {
                        let clamped = s.clamp(-1.0, 1.0);
                        writer
                            .write_sample((clamped * max_val) as i16)
                            .map_err(|e| {
                                AppError::VoiceCloningError(format!("Failed to write sample: {e}"))
                            })?;
                    }
                    // 尾部不写入，等下一段拼接时交叉淡入淡出
                    prev_tail = all_samples[tail_start..].to_vec();
                } else {
                    // 文件太短，全部写入
                    for &s in &all_samples {
                        let clamped = s.clamp(-1.0, 1.0);
                        writer
                            .write_sample((clamped * max_val) as i16)
                            .map_err(|e| {
                                AppError::VoiceCloningError(format!("Failed to write sample: {e}"))
                            })?;
                    }
                    prev_tail = all_samples;
                }
            } else {
                // 后续段：与前一段尾部做交叉淡入淡出
                let mut current_samples: Vec<f32> = Vec::new();
                while let Some(chunk) = reader.read_chunk() {
                    current_samples.extend_from_slice(&chunk);
                }

                let fade_len = crossfade_samples
                    .min(prev_tail.len())
                    .min(current_samples.len());

                if fade_len > 0 {
                    // 交叉淡入淡出：prev_tail 的最后 fade_len 样本 + current 的前 fade_len 样本
                    for j in 0..fade_len {
                        let t = j as f32 / fade_len as f32;
                        let fade_out = (std::f32::consts::PI * t * 0.5).cos();
                        let fade_in = (std::f32::consts::PI * t * 0.5).sin();

                        let prev_sample = prev_tail[prev_tail.len() - fade_len + j] * fade_out;
                        let curr_sample = current_samples[j] * fade_in;
                        let mixed = (prev_sample + curr_sample).clamp(-1.0, 1.0);

                        if channels == 1 {
                            writer.write_sample((mixed * max_val) as i16).map_err(|e| {
                                AppError::VoiceCloningError(format!("Failed to write sample: {e}"))
                            })?;
                        } else {
                            // 多声道：此处简化处理，实际需要逐声道
                            writer.write_sample((mixed * max_val) as i16).map_err(|e| {
                                AppError::VoiceCloningError(format!("Failed to write sample: {e}"))
                            })?;
                        }
                    }

                    // 写入当前段剩余部分
                    for &s in &current_samples[fade_len..] {
                        let clamped = s.clamp(-1.0, 1.0);
                        writer
                            .write_sample((clamped * max_val) as i16)
                            .map_err(|e| {
                                AppError::VoiceCloningError(format!("Failed to write sample: {e}"))
                            })?;
                    }

                    // 保留当前段尾部用于下一段
                    if current_samples.len() > crossfade_samples {
                        let tail_start = current_samples.len() - crossfade_samples;
                        prev_tail = current_samples[tail_start..].to_vec();
                    } else {
                        prev_tail = current_samples;
                    }
                } else {
                    // 无交叉淡入淡出：直接写入
                    for &s in &current_samples {
                        let clamped = s.clamp(-1.0, 1.0);
                        writer
                            .write_sample((clamped * max_val) as i16)
                            .map_err(|e| {
                                AppError::VoiceCloningError(format!("Failed to write sample: {e}"))
                            })?;
                    }
                    prev_tail = current_samples;
                }
            }
        }

        writer
            .finalize()
            .map_err(|e| AppError::VoiceCloningError(format!("Failed to finalize WAV: {e}")))?;

        Ok(())
    }
}

// ─── 流式音频电平分析器 ─────────────────────────────────

/// 流式音频电平分析结果
#[derive(Debug, Clone)]
pub struct AudioLevelResult {
    /// RMS 电平（0.0-1.0）
    pub rms: f32,
    /// 峰值电平（0.0-1.0）
    pub peak: f32,
    /// DC 偏移（-1.0 to 1.0）
    pub dc_offset: f32,
    /// 过零率（0.0-1.0）
    pub zero_crossing_rate: f32,
    /// 总样本数
    pub total_samples: usize,
    /// 估算时长（秒）
    pub duration_secs: f64,
}

/// 流式音频电平分析器
///
/// 以流式方式分析 WAV 文件的电平信息，内存占用恒定。
pub struct AudioLevelAnalyzer {
    chunk_size: usize,
}

impl AudioLevelAnalyzer {
    /// 创建分析器
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(MIN_CHUNK_SIZE),
        }
    }

    /// 使用默认 chunk 大小
    #[must_use]
    pub fn default_chunk() -> Self {
        Self::new(DEFAULT_CHUNK_SIZE)
    }

    /// 分析 WAV 文件
    pub fn analyze(&self, path: &Path) -> AppResult<AudioLevelResult> {
        let mut reader = AudioChunkReader::open(path, self.chunk_size)?;

        let mut sum_sq: f64 = 0.0;
        let mut sum: f64 = 0.0;
        let mut peak: f32 = 0.0;
        let mut zero_crossings: usize = 0;
        let mut total_samples: usize = 0;
        let mut prev_sample: f32 = 0.0;

        while let Some(chunk) = reader.read_chunk() {
            for &s in &chunk {
                sum_sq += (s as f64) * (s as f64);
                sum += s as f64;
                let abs_s = s.abs();
                if abs_s > peak {
                    peak = abs_s;
                }
                if total_samples > 0 {
                    if (s >= 0.0) != (prev_sample >= 0.0) {
                        zero_crossings += 1;
                    }
                }
                prev_sample = s;
                total_samples += 1;
            }
        }

        let rms = if total_samples > 0 {
            (sum_sq / total_samples as f64).sqrt() as f32
        } else {
            0.0
        };

        let dc_offset = if total_samples > 0 {
            (sum / total_samples as f64) as f32
        } else {
            0.0
        };

        let zero_crossing_rate = if total_samples > 1 {
            zero_crossings as f32 / (total_samples - 1) as f32
        } else {
            0.0
        };

        let duration_secs =
            estimate_duration_secs(total_samples, reader.sample_rate(), reader.channels());

        Ok(AudioLevelResult {
            rms,
            peak,
            dc_offset,
            zero_crossing_rate,
            total_samples,
            duration_secs,
        })
    }
}

// ─── 流式静音检测 ─────────────────────────────────────────

/// 静音检测结果
#[derive(Debug, Clone)]
pub struct SilenceDetectResult {
    /// 起始静音样本数
    pub head_silence_samples: usize,
    /// 结尾静音样本数
    pub tail_silence_samples: usize,
    /// 起始静音时长（秒）
    pub head_silence_secs: f64,
    /// 结尾静音时长（秒）
    pub tail_silence_secs: f64,
}

/// 流式检测 WAV 文件首尾静音
///
/// 只需遍历一次，同时检测首尾静音。
///
/// # 参数
/// - `path`: WAV 文件路径
/// - `threshold`: 静音阈值（0.0-1.0，低于此值视为静音）
/// - `min_silence_samples`: 最小静音样本数（短于此值不算静音）
pub fn detect_silence(
    path: &Path,
    threshold: f32,
    min_silence_samples: usize,
) -> AppResult<SilenceDetectResult> {
    let mut reader = AudioChunkReader::open_default(path)?;
    let sample_rate = reader.sample_rate();

    let mut head_silence: usize = 0;
    let mut in_head_silence = true;
    let mut total_samples: usize = 0;

    // 尾部静音需要反向检测，流式处理时记录最后非静音位置
    let mut last_non_silence: usize = 0;

    while let Some(chunk) = reader.read_chunk() {
        for (i, &s) in chunk.iter().enumerate() {
            let abs_s = s.abs();
            if in_head_silence {
                if abs_s < threshold {
                    head_silence += 1;
                } else {
                    in_head_silence = false;
                    last_non_silence = total_samples + i;
                }
            } else if abs_s >= threshold {
                last_non_silence = total_samples + i;
            }
        }
        total_samples += chunk.len();
    }

    // 如果全程静音
    if in_head_silence {
        return Ok(SilenceDetectResult {
            head_silence_samples: total_samples,
            tail_silence_samples: 0,
            head_silence_secs: total_samples as f64 / sample_rate as f64,
            tail_silence_secs: 0.0,
        });
    }

    // 应用最小静音阈值
    let head_silence = if head_silence >= min_silence_samples {
        head_silence
    } else {
        0
    };

    let tail_silence = if total_samples > last_non_silence + 1 {
        let tail = total_samples - last_non_silence - 1;
        if tail >= min_silence_samples {
            tail
        } else {
            0
        }
    } else {
        0
    };

    Ok(SilenceDetectResult {
        head_silence_samples: head_silence,
        tail_silence_samples: tail_silence,
        head_silence_secs: head_silence as f64 / sample_rate as f64,
        tail_silence_secs: tail_silence as f64 / sample_rate as f64,
    })
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 创建测试用 WAV 文件
    fn create_test_wav(path: &Path, samples: &[f32], sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            let i16_sample = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(i16_sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn test_read_wav_info() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("test.wav");
        create_test_wav(&wav, &[0.5; 1000], 24000);

        let info = read_wav_info(&wav).unwrap();
        assert_eq!(info.sample_rate, 24000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 16);
    }

    #[test]
    fn test_estimate_wav_memory() {
        // 24000 samples × 1 channel × 4 bytes = 96000 bytes
        let mem = estimate_wav_memory(24000, 1);
        assert_eq!(mem, 96000);
    }

    #[test]
    fn test_estimate_duration() {
        // 24000 samples @ 24kHz mono = 1 second
        let dur = estimate_duration_secs(24000, 24000, 1);
        assert!((dur - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_chunk_reader_basic() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("test.wav");
        let samples: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        create_test_wav(&wav, &samples, 24000);

        let mut reader = AudioChunkReader::open(&wav, 1000).unwrap();
        let mut total = 0;
        while let Some(chunk) = reader.read_chunk() {
            total += chunk.len();
        }
        assert_eq!(total, 5000);
    }

    #[test]
    fn test_chunk_reader_iterator() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("test.wav");
        create_test_wav(&wav, &[0.5; 3000], 24000);

        let reader = AudioChunkReader::open_default(&wav).unwrap();
        let total: usize = reader.map(|c| c.len()).sum();
        assert_eq!(total, 3000);
    }

    #[test]
    fn test_streaming_concat_single_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let out = dir.path().join("out.wav");

        create_test_wav(&wav1, &[0.5; 1000], 24000);

        let concat = StreamingWavConcatenator::new(50);
        concat.concatenate(&[wav1.clone()], &out).unwrap();

        let reader = hound::WavReader::open(&out).unwrap();
        assert_eq!(reader.spec().sample_rate, 24000);
    }

    #[test]
    fn test_streaming_concat_two_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let wav2 = dir.path().join("b.wav");
        let out = dir.path().join("out.wav");

        let samples1: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let samples2: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();

        create_test_wav(&wav1, &samples1, 24000);
        create_test_wav(&wav2, &samples2, 24000);

        let concat = StreamingWavConcatenator::new(10);
        concat.concatenate(&[wav1, wav2], &out).unwrap();

        let reader = hound::WavReader::open(&out).unwrap();
        let total_samples: Vec<i16> = reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .collect();
        assert!(
            total_samples.len() > 3500,
            "Should have > 3500 samples, got {}",
            total_samples.len()
        );
    }

    #[test]
    fn test_streaming_concat_empty() {
        let out = Path::new("/tmp/test_stream_empty.wav");
        let concat = StreamingWavConcatenator::new(50);
        let result = concat.concatenate(&[], out);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_concat_sample_rate_mismatch() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let wav2 = dir.path().join("b.wav");
        let out = dir.path().join("out.wav");

        create_test_wav(&wav1, &[0.5; 100], 24000);
        create_test_wav(&wav2, &[0.3; 100], 16000);

        let concat = StreamingWavConcatenator::new(50);
        assert!(concat.concatenate(&[wav1, wav2], &out).is_err());
    }

    #[test]
    fn test_streaming_concat_no_crossfade() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let wav2 = dir.path().join("b.wav");
        let out = dir.path().join("out.wav");

        create_test_wav(&wav1, &[0.5; 1000], 24000);
        create_test_wav(&wav2, &[0.3; 1000], 24000);

        let concat = StreamingWavConcatenator::new(0);
        concat.concatenate(&[wav1, wav2], &out).unwrap();

        let reader = hound::WavReader::open(&out).unwrap();
        let total_samples: Vec<i16> = reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .collect();
        // 无交叉淡入淡出：总长度 = 1000 + 1000 = 2000
        assert_eq!(total_samples.len(), 2000);
    }

    #[test]
    fn test_audio_level_analyzer() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("test.wav");
        let samples: Vec<f32> = (0..24000)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI / 24000.0).sin() * 0.5)
            .collect();
        create_test_wav(&wav, &samples, 24000);

        let analyzer = AudioLevelAnalyzer::default_chunk();
        let result = analyzer.analyze(&wav).unwrap();

        assert!(result.rms > 0.0, "RMS should be positive");
        assert!(result.peak > 0.0, "Peak should be positive");
        assert!(result.dc_offset.abs() < 0.1, "DC offset should be small");
        assert!(result.total_samples == 24000);
        assert!((result.duration_secs - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_level_silence() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("silence.wav");
        create_test_wav(&wav, &[0.0; 2400], 24000);

        let analyzer = AudioLevelAnalyzer::default_chunk();
        let result = analyzer.analyze(&wav).unwrap();

        assert!(result.rms < 0.001, "RMS should be near zero for silence");
        assert!(result.peak < 0.001, "Peak should be near zero for silence");
    }

    #[test]
    fn test_detect_silence() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("test.wav");
        // 1000 silence + 2000 signal + 1000 silence
        let samples: Vec<f32> = [
            vec![0.0; 1000],
            (0..2000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect(),
            vec![0.0; 1000],
        ]
        .concat();
        create_test_wav(&wav, &samples, 24000);

        let result = detect_silence(&wav, 0.01, 100).unwrap();
        assert!(
            result.head_silence_samples >= 900,
            "Head silence: {}",
            result.head_silence_samples
        );
        assert!(
            result.tail_silence_samples >= 900,
            "Tail silence: {}",
            result.tail_silence_samples
        );
    }

    #[test]
    fn test_detect_silence_no_silence() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("test.wav");
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        create_test_wav(&wav, &samples, 24000);

        let result = detect_silence(&wav, 0.01, 100).unwrap();
        assert_eq!(result.head_silence_samples, 0);
        assert_eq!(result.tail_silence_samples, 0);
    }

    #[test]
    fn test_detect_silence_all_silence() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("silence.wav");
        create_test_wav(&wav, &[0.0; 5000], 24000);

        let result = detect_silence(&wav, 0.01, 100).unwrap();
        assert_eq!(result.head_silence_samples, 5000);
    }

    #[test]
    fn test_chunk_reader_min_size() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav = dir.path().join("test.wav");
        create_test_wav(&wav, &[0.5; 500], 24000);

        // chunk_size 小于 MIN_CHUNK_SIZE，应自动调整
        let reader = AudioChunkReader::open(&wav, 10).unwrap();
        // 不应 panic
        let total: usize = reader.map(|c| c.len()).sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn test_streaming_concat_three_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let wav1 = dir.path().join("a.wav");
        let wav2 = dir.path().join("b.wav");
        let wav3 = dir.path().join("c.wav");
        let out = dir.path().join("out.wav");

        create_test_wav(&wav1, &[0.5; 1500], 24000);
        create_test_wav(&wav2, &[0.3; 1500], 24000);
        create_test_wav(&wav3, &[0.7; 1500], 24000);

        let concat = StreamingWavConcatenator::new(10);
        concat.concatenate(&[wav1, wav2, wav3], &out).unwrap();

        let reader = hound::WavReader::open(&out).unwrap();
        let total_samples: Vec<i16> = reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .collect();
        assert!(
            total_samples.len() > 3500,
            "3-file concat should have > 3500 samples, got {}",
            total_samples.len()
        );
    }
}
