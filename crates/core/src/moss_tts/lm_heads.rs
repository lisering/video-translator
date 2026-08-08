//! LM Head 投影 — 从 MOSS-TTS `lm_heads.py` 移植
//!
//! 33 个预测头（1 文本 + 32 音频），从 hidden state 投影到 logits。
//! 使用预堆叠权重 + 单次 BLAS matmul 优化。
//!
//! 对应 MOSS-TTS 项目 `moss_tts_delay/llama_cpp/lm_heads.py`。

use super::embedding::load_npy_f32;
use crate::error::{AppError, AppResult};
use rayon::prelude::*;
use std::path::Path;

/// LM Head 预测器
///
/// 从 `.npy` 权重加载 33 个预测头，
/// 从 hidden state 计算 text + audio logits。
pub struct LmHeads {
    /// 文本头权重 [text_vocab, hidden]
    pub text_weight: Vec<f32>,
    pub text_vocab_size: usize,

    /// 音频头权重（预堆叠）[n_vq * audio_vocab, hidden]
    pub audio_weight_stacked: Vec<f32>,
    pub audio_vocab_size: usize,

    pub hidden_size: usize,
    pub n_vq: usize,
}

impl LmHeads {
    /// 从 .npy 权重目录加载
    ///
    /// 期望文件：
    /// - `lm_head_text.npy` — 文本头权重
    /// - `lm_head_audio_00.npy` ~ `lm_head_audio_31.npy` — 音频头权重
    pub fn from_dir(weight_dir: &Path, n_vq: usize) -> AppResult<Self> {
        tracing::info!("Loading LM heads from {}", weight_dir.display());

        let text_path = weight_dir.join("lm_head_text.npy");
        let text_weight = load_npy_f32(&text_path)
            .map_err(|e| AppError::VoiceCloningError(format!("Failed to load text head: {e}")))?;

        // 推断维度
        let audio_path = weight_dir.join("lm_head_audio_00.npy");
        let first_audio = load_npy_f32(&audio_path).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to load audio head 0: {e}"))
        })?;

        // 假设 audio_head shape: [audio_vocab, hidden]
        // text_head shape: [text_vocab, hidden]
        // 推断 hidden_size 和 vocab_size
        let audio_file_size = std::fs::metadata(&audio_path)
            .map_err(|e| AppError::VoiceCloningError(format!("stat error: {e}")))?
            .len() as usize;

        // 粗略推断：从 header 解析 shape
        let audio_shape = npy_shape(&audio_path).unwrap_or_else(|_| vec![]);
        let hidden_size = if audio_shape.len() >= 2 {
            audio_shape[1]
        } else if first_audio.len() > 0 {
            // fallback: 尝试从文件大小推断
            audio_file_size
                / (first_audio.len()
                    * if audio_file_size > first_audio.len() * 4 {
                        4
                    } else {
                        2
                    })
        } else {
            0
        };

        let audio_vocab_size = if audio_shape.len() >= 1 {
            audio_shape[0]
        } else {
            first_audio.len() / hidden_size.max(1)
        };

        let text_vocab_size = if text_weight.len() > 0 && hidden_size > 0 {
            text_weight.len() / hidden_size
        } else {
            0
        };

        // 加载并堆叠所有音频头
        let mut audio_weight_stacked = Vec::with_capacity(n_vq * audio_vocab_size * hidden_size);
        audio_weight_stacked.extend_from_slice(&first_audio);

        for i in 1..n_vq {
            let path = weight_dir.join(format!("lm_head_audio_{:02}.npy", i));
            let weight = load_npy_f32(&path).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to load audio head {i}: {e}"))
            })?;
            audio_weight_stacked.extend_from_slice(&weight);
        }

        tracing::info!(
            "LmHeads: text_vocab={}, audio_vocab={}, hidden={}, n_vq={}",
            text_vocab_size,
            audio_vocab_size,
            hidden_size,
            n_vq
        );

        Ok(Self {
            text_weight,
            text_vocab_size,
            audio_weight_stacked,
            audio_vocab_size,
            hidden_size,
            n_vq,
        })
    }

    /// 计算所有 33 个头的 logits
    ///
    /// `hidden_state`: [hidden_size]
    /// 返回: (text_logits [text_vocab], audio_logits [n_vq, audio_vocab])
    pub fn forward(&self, hidden_state: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        // 文本: [text_vocab] = hidden @ text_weight^T
        let text_logits = matvec(
            &self.text_weight,
            hidden_state,
            self.text_vocab_size,
            self.hidden_size,
        );

        // 音频: [n_vq * audio_vocab] = hidden @ audio_stacked^T
        let audio_flat = matvec(
            &self.audio_weight_stacked,
            hidden_state,
            self.n_vq * self.audio_vocab_size,
            self.hidden_size,
        );

        // reshape to [n_vq, audio_vocab]
        let mut audio_logits: Vec<Vec<f32>> = (0..self.n_vq)
            .map(|i| {
                let start = i * self.audio_vocab_size;
                let end = start + self.audio_vocab_size;
                let mut row = audio_flat[start..end].to_vec();
                // mask pad_code
                let pad_idx = self.audio_vocab_size.saturating_sub(1);
                if pad_idx < row.len() {
                    row[pad_idx] = f32::NEG_INFINITY;
                }
                row
            })
            .collect();

        let _ = &mut audio_logits; // suppress unused warning
        (text_logits, audio_logits)
    }

    /// 只计算音频 logits（跳过文本头，用于生成循环优化）
    pub fn audio_all(&self, hidden_state: &[f32]) -> Vec<Vec<f32>> {
        let audio_flat = matvec(
            &self.audio_weight_stacked,
            hidden_state,
            self.n_vq * self.audio_vocab_size,
            self.hidden_size,
        );

        (0..self.n_vq)
            .map(|i| {
                let start = i * self.audio_vocab_size;
                let end = start + self.audio_vocab_size;
                let mut row = audio_flat[start..end].to_vec();
                let pad_idx = self.audio_vocab_size.saturating_sub(1);
                if pad_idx < row.len() {
                    row[pad_idx] = f32::NEG_INFINITY;
                }
                row
            })
            .collect()
    }

    /// 只计算文本 logits
    pub fn text_only(&self, hidden_state: &[f32]) -> Vec<f32> {
        matvec(
            &self.text_weight,
            hidden_state,
            self.text_vocab_size,
            self.hidden_size,
        )
    }

    /// 内存占用（字节）
    pub fn nbytes(&self) -> usize {
        (self.text_weight.len() + self.audio_weight_stacked.len()) * 4
    }

    /// 摘要信息
    pub fn summary(&self) -> String {
        format!(
            "LmHeads: {}×{} text + {}×{}×{} audio, {:.1} MB",
            self.text_vocab_size,
            self.hidden_size,
            self.n_vq,
            self.audio_vocab_size,
            self.hidden_size,
            self.nbytes() as f64 / (1024.0 * 1024.0)
        )
    }
}

// ─── 矩阵-向量乘法 ────────────────────────────────────────

/// 矩阵-向量乘法: result = W @ x
///
/// W: [m, k] (row-major), x: [k], 返回: [m]
///
/// 使用 Rayon 并行处理各行。
pub fn matvec(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    if k == 0 || m == 0 {
        return vec![];
    }
    (0..m)
        .into_par_iter()
        .map(|i| {
            let row = &w[i * k..(i + 1) * k];
            // 点积
            row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()
        })
        .collect()
}

// ─── .npy 辅助函数 ────────────────────────────────────────

/// 从 .npy 文件获取 shape
fn npy_shape(path: &Path) -> Result<Vec<usize>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Read error: {e}"))?;
    if bytes.len() < 10 {
        return Err("File too short".to_string());
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_str = std::str::from_utf8(&bytes[10..10 + header_len])
        .map_err(|e| format!("Header not UTF-8: {e}"))?;

    // 提取 shape
    let key = "'shape':";
    let start = header_str.find(key).ok_or("shape not found")? + key.len();
    let rest = &header_str[start..];
    let paren_start = rest.find('(').ok_or("( not found")?;
    let rest = &rest[paren_start + 1..];
    let paren_end = rest.find(')').ok_or(") not found")?;
    let shape_str = &rest[..paren_end];
    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    Ok(shape)
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_basic() {
        // W = [[1,2],[3,4]], x = [5,6]
        // result = [1*5+2*6, 3*5+4*6] = [17, 39]
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 6.0];
        let result = matvec(&w, &x, 2, 2);
        assert!((result[0] - 17.0).abs() < 1e-5);
        assert!((result[1] - 39.0).abs() < 1e-5);
    }

    #[test]
    fn test_matvec_single_row() {
        let w = vec![1.0, 2.0, 3.0];
        let x = vec![4.0, 5.0, 6.0];
        let result = matvec(&w, &x, 1, 3);
        assert!((result[0] - 32.0).abs() < 1e-5); // 1*4+2*5+3*6=32
    }

    #[test]
    fn test_matvec_empty() {
        let result = matvec(&[], &[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_lm_heads_summary() {
        let heads = LmHeads {
            text_weight: vec![0.0; 100 * 64],
            text_vocab_size: 100,
            audio_weight_stacked: vec![0.0; 32 * 50 * 64],
            audio_vocab_size: 50,
            hidden_size: 64,
            n_vq: 32,
        };
        let summary = heads.summary();
        assert!(summary.contains("100"));
        assert!(summary.contains("32"));
    }

    #[test]
    fn test_forward_shape() {
        let hidden = 64;
        let text_vocab = 100;
        let audio_vocab = 50;
        let n_vq = 4;

        let heads = LmHeads {
            text_weight: vec![1.0; text_vocab * hidden],
            text_vocab_size: text_vocab,
            audio_weight_stacked: vec![1.0; n_vq * audio_vocab * hidden],
            audio_vocab_size: audio_vocab,
            hidden_size: hidden,
            n_vq,
        };

        let hs = vec![1.0; hidden];
        let (text_logits, audio_logits) = heads.forward(&hs);
        assert_eq!(text_logits.len(), text_vocab);
        assert_eq!(audio_logits.len(), n_vq);
        assert_eq!(audio_logits[0].len(), audio_vocab);
    }

    #[test]
    fn test_audio_all_shape() {
        let hidden = 32;
        let audio_vocab = 40;
        let n_vq = 8;

        let heads = LmHeads {
            text_weight: vec![1.0; 100 * hidden],
            text_vocab_size: 100,
            audio_weight_stacked: vec![1.0; n_vq * audio_vocab * hidden],
            audio_vocab_size: audio_vocab,
            hidden_size: hidden,
            n_vq,
        };

        let hs = vec![1.0; hidden];
        let audio_logits = heads.audio_all(&hs);
        assert_eq!(audio_logits.len(), n_vq);
        assert_eq!(audio_logits[0].len(), audio_vocab);
    }
}
