//! TTS 批量评估框架 — 借鉴 MOSS-TTS `batch_eval_llama_cpp.py`
//!
//! 系统性测量 TTS 质量和性能指标：
//! - RTF (Real-Time Factor)
//! - 成功率 / 失败率
//! - 音频时长统计
//! - 采样参数对质量的影响
//!
//! 对应 MOSS-TTS 项目 `scripts/batch_eval_llama_cpp.py`。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

/// 单次 TTS 评估结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsEvalResult {
    /// 文本 ID
    pub text_id: String,
    /// 输入文本
    pub text: String,
    /// 参考音频路径
    pub reference_audio: Option<String>,
    /// 生成音频路径
    pub output_path: PathBuf,
    /// 生成音频时长（秒）
    pub audio_duration_secs: f64,
    /// 生成耗时（秒）
    pub elapsed_secs: f64,
    /// RTF (Real-Time Factor) = elapsed / audio_duration
    pub rtf: f64,
    /// 是否成功
    pub success: bool,
    /// 错误信息（失败时）
    pub error: Option<String>,
    /// 采样参数
    pub params: EvalParams,
}

/// 评估参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub language: String,
}

impl Default for EvalParams {
    fn default() -> Self {
        Self {
            temperature: 1.7,
            top_k: 25,
            top_p: 0.8,
            repetition_penalty: 1.0,
            language: "zh".to_string(),
        }
    }
}

/// 批量评估汇总
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEvalSummary {
    /// 总测试数
    pub total: usize,
    /// 成功数
    pub success_count: usize,
    /// 失败数
    pub failure_count: usize,
    /// 成功率
    pub success_rate: f64,
    /// 平均 RTF
    pub avg_rtf: f64,
    /// 最小 RTF
    pub min_rtf: f64,
    /// 最大 RTF
    pub max_rtf: f64,
    /// 中位数 RTF
    pub median_rtf: f64,
    /// 总音频时长（秒）
    pub total_audio_secs: f64,
    /// 总生成耗时（秒）
    pub total_elapsed_secs: f64,
    /// 平均音频时长
    pub avg_audio_secs: f64,
    /// 每个结果
    pub results: Vec<TtsEvalResult>,
}

impl BatchEvalSummary {
    /// 从结果列表生成汇总
    pub fn from_results(results: Vec<TtsEvalResult>) -> Self {
        let total = results.len();
        let success_count = results.iter().filter(|r| r.success).count();
        let failure_count = total - success_count;
        let success_rate = if total > 0 {
            success_count as f64 / total as f64
        } else {
            0.0
        };

        let successful_rtfs: Vec<f64> = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.rtf)
            .collect();

        let avg_rtf = if !successful_rtfs.is_empty() {
            successful_rtfs.iter().sum::<f64>() / successful_rtfs.len() as f64
        } else {
            0.0
        };

        let min_rtf = successful_rtfs
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_rtf = successful_rtfs.iter().cloned().fold(0.0f64, f64::max);

        let mut sorted_rtfs = successful_rtfs.clone();
        sorted_rtfs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_rtf = if !sorted_rtfs.is_empty() {
            sorted_rtfs[sorted_rtfs.len() / 2]
        } else {
            0.0
        };

        let total_audio_secs: f64 = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.audio_duration_secs)
            .sum();
        let total_elapsed_secs: f64 = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.elapsed_secs)
            .sum();
        let avg_audio_secs = if success_count > 0 {
            total_audio_secs / success_count as f64
        } else {
            0.0
        };

        Self {
            total,
            success_count,
            failure_count,
            success_rate,
            avg_rtf,
            min_rtf: if min_rtf.is_infinite() { 0.0 } else { min_rtf },
            max_rtf,
            median_rtf,
            total_audio_secs,
            total_elapsed_secs,
            avg_audio_secs,
            results,
        }
    }

    /// 格式化为可读报告
    pub fn format_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("{}\n", "=".repeat(60)));
        report.push_str("  TTS Batch Evaluation Summary\n");
        report.push_str(&format!("{}\n", "=".repeat(60)));
        report.push_str(&format!("  Total:          {}\n", self.total));
        report.push_str(&format!(
            "  Success:        {} ({:.1}%)\n",
            self.success_count,
            self.success_rate * 100.0
        ));
        report.push_str(&format!("  Failure:        {}\n", self.failure_count));
        report.push_str(&format!("\n"));
        report.push_str(&format!("  RTF Statistics:\n"));
        report.push_str(&format!("    Average:      {:.3}x\n", self.avg_rtf));
        report.push_str(&format!("    Min:          {:.3}x\n", self.min_rtf));
        report.push_str(&format!("    Max:          {:.3}x\n", self.max_rtf));
        report.push_str(&format!("    Median:       {:.3}x\n", self.median_rtf));
        report.push_str(&format!("\n"));
        report.push_str(&format!("  Audio Statistics:\n"));
        report.push_str(&format!(
            "    Total audio:  {:.1}s\n",
            self.total_audio_secs
        ));
        report.push_str(&format!(
            "    Total time:   {:.1}s\n",
            self.total_elapsed_secs
        ));
        report.push_str(&format!("    Avg audio:   {:.1}s\n", self.avg_audio_secs));
        report.push_str(&format!("{}\n", "=".repeat(60)));
        report
    }
}

/// 评估计时器
pub struct EvalTimer {
    start: Instant,
}

impl EvalTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl Default for EvalTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// 评估文本集
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalTextSet {
    pub name: String,
    pub language: String,
    pub texts: Vec<EvalText>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalText {
    pub id: String,
    pub text: String,
    pub expected_duration_secs: Option<f64>,
}

/// 预设评估文本集
pub fn builtin_eval_texts() -> Vec<EvalTextSet> {
    vec![
        EvalTextSet {
            name: "Chinese Short".to_string(),
            language: "zh".to_string(),
            texts: vec![
                EvalText {
                    id: "zh_1".to_string(),
                    text: "你好，欢迎观看这个视频教程。".to_string(),
                    expected_duration_secs: Some(3.0),
                },
                EvalText {
                    id: "zh_2".to_string(),
                    text: "今天我们将学习如何使用这个工具来翻译视频。".to_string(),
                    expected_duration_secs: Some(5.0),
                },
                EvalText {
                    id: "zh_3".to_string(),
                    text: "首先，我们需要准备一个输入视频文件。".to_string(),
                    expected_duration_secs: Some(4.0),
                },
            ],
        },
        EvalTextSet {
            name: "English Short".to_string(),
            language: "en".to_string(),
            texts: vec![
                EvalText {
                    id: "en_1".to_string(),
                    text: "Hello and welcome to this video tutorial.".to_string(),
                    expected_duration_secs: Some(3.0),
                },
                EvalText {
                    id: "en_2".to_string(),
                    text: "Today we will learn how to use this tool to translate videos."
                        .to_string(),
                    expected_duration_secs: Some(5.0),
                },
                EvalText {
                    id: "en_3".to_string(),
                    text: "First, we need to prepare an input video file.".to_string(),
                    expected_duration_secs: Some(4.0),
                },
            ],
        },
        EvalTextSet {
            name: "Technical Terms".to_string(),
            language: "zh".to_string(),
            texts: vec![
                EvalText {
                    id: "tech_1".to_string(),
                    text: "我们使用 Python 和 Rust 来构建这个项目。".to_string(),
                    expected_duration_secs: None,
                },
                EvalText {
                    id: "tech_2".to_string(),
                    text: "API 调用使用 JSON 协议进行通信。".to_string(),
                    expected_duration_secs: None,
                },
                EvalText {
                    id: "tech_3".to_string(),
                    text: "GPU 加速可以显著提升推理速度。".to_string(),
                    expected_duration_secs: None,
                },
            ],
        },
    ]
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_result_default_params() {
        let params = EvalParams::default();
        assert!((params.temperature - 1.7).abs() < 0.01);
        assert_eq!(params.top_k, 25);
    }

    #[test]
    fn test_batch_summary_all_success() {
        let results = vec![
            TtsEvalResult {
                text_id: "1".to_string(),
                text: "hello".to_string(),
                reference_audio: None,
                output_path: PathBuf::from("/tmp/out1.wav"),
                audio_duration_secs: 5.0,
                elapsed_secs: 10.0,
                rtf: 2.0,
                success: true,
                error: None,
                params: EvalParams::default(),
            },
            TtsEvalResult {
                text_id: "2".to_string(),
                text: "world".to_string(),
                reference_audio: None,
                output_path: PathBuf::from("/tmp/out2.wav"),
                audio_duration_secs: 10.0,
                elapsed_secs: 15.0,
                rtf: 1.5,
                success: true,
                error: None,
                params: EvalParams::default(),
            },
        ];

        let summary = BatchEvalSummary::from_results(results);
        assert_eq!(summary.total, 2);
        assert_eq!(summary.success_count, 2);
        assert_eq!(summary.failure_count, 0);
        assert!((summary.success_rate - 1.0).abs() < 0.01);
        assert!((summary.avg_rtf - 1.75).abs() < 0.01);
        assert!((summary.min_rtf - 1.5).abs() < 0.01);
        assert!((summary.max_rtf - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_summary_with_failures() {
        let results = vec![
            TtsEvalResult {
                text_id: "1".to_string(),
                text: "hello".to_string(),
                reference_audio: None,
                output_path: PathBuf::from("/tmp/out1.wav"),
                audio_duration_secs: 5.0,
                elapsed_secs: 10.0,
                rtf: 2.0,
                success: true,
                error: None,
                params: EvalParams::default(),
            },
            TtsEvalResult {
                text_id: "2".to_string(),
                text: "world".to_string(),
                reference_audio: None,
                output_path: PathBuf::new(),
                audio_duration_secs: 0.0,
                elapsed_secs: 0.0,
                rtf: 0.0,
                success: false,
                error: Some("timeout".to_string()),
                params: EvalParams::default(),
            },
        ];

        let summary = BatchEvalSummary::from_results(results);
        assert_eq!(summary.total, 2);
        assert_eq!(summary.success_count, 1);
        assert_eq!(summary.failure_count, 1);
        assert!((summary.success_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_format_report() {
        let results = vec![TtsEvalResult {
            text_id: "1".to_string(),
            text: "test".to_string(),
            reference_audio: None,
            output_path: PathBuf::from("/tmp/out.wav"),
            audio_duration_secs: 5.0,
            elapsed_secs: 10.0,
            rtf: 2.0,
            success: true,
            error: None,
            params: EvalParams::default(),
        }];
        let summary = BatchEvalSummary::from_results(results);
        let report = summary.format_report();
        assert!(report.contains("TTS Batch Evaluation"));
        assert!(report.contains("Success:        1"));
    }

    #[test]
    fn test_eval_timer() {
        use std::time::Duration;
        let timer = EvalTimer::new();
        std::thread::sleep(Duration::from_millis(100));
        assert!(timer.elapsed_secs() >= 0.1);
    }

    #[test]
    fn test_builtin_eval_texts() {
        let texts = builtin_eval_texts();
        assert!(texts.len() >= 3);
        assert!(texts[0].texts.len() >= 3);
    }

    #[test]
    fn test_empty_summary() {
        let summary = BatchEvalSummary::from_results(vec![]);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.success_rate, 0.0);
    }
}
