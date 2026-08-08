//! 多通道 Embedding 查找 — 从 MOSS-TTS `embedding.py` 移植
//!
//! 加载 33 个 embedding 表（1 个文本 + 32 个音频 VQ），执行求和操作：
//! `inputs_embeds = embed_tokens[text_ids] + Σ emb_ext[i][audio_ids[i]]`
//!
//! 对应 MOSS-TTS 项目 `moss_tts_delay/llama_cpp/embedding.py`。
//!
//! 权重格式：NumPy `.npy` 文件（float16 或 float32），通过 `npy` crate 加载。

use crate::error::{AppError, AppResult};
use rayon::prelude::*;
use std::path::Path;

/// 多通道 Embedding 查找器
///
/// 从 `.npy` 文件加载 33 个 embedding 表，
/// 执行多通道 token ID 求和。
pub struct EmbeddingLookup {
    /// 文本 embedding 表 [vocab_size, hidden_size]
    pub text_embed: Vec<f32>,
    pub text_vocab_size: usize,
    pub hidden_size: usize,

    /// 音频 embedding 表 [n_vq][audio_vocab_size, hidden_size]
    pub audio_embeds: Vec<Vec<f32>>,
    pub audio_vocab_size: usize,
    pub n_vq: usize,
}

impl EmbeddingLookup {
    /// 从 .npy 权重目录加载
    ///
    /// 期望文件：
    /// - `embed_tokens.npy` — 文本 embedding [vocab, hidden]
    /// - `emb_ext_00.npy` ~ `emb_ext_31.npy` — 音频 embedding
    pub fn from_dir(weight_dir: &Path, n_vq: usize) -> AppResult<Self> {
        tracing::info!("Loading embeddings from {}", weight_dir.display());

        let text_path = weight_dir.join("embed_tokens.npy");
        let text_embed = load_npy_f32(&text_path).map_err(|e| {
            AppError::VoiceCloningError(format!("Failed to load text embedding: {e}"))
        })?;
        let hidden_size = hidden_size_from_npy(&text_path)
            .map_err(|e| AppError::VoiceCloningError(format!("Failed to get hidden size: {e}")))?;
        let text_vocab_size = if hidden_size > 0 {
            text_embed.len() / hidden_size
        } else {
            0
        };

        tracing::info!(
            "Text embedding: vocab={}, hidden={}",
            text_vocab_size,
            hidden_size
        );

        let mut audio_embeds = Vec::with_capacity(n_vq);
        let mut audio_vocab_size = 0;

        for i in 0..n_vq {
            let path = weight_dir.join(format!("emb_ext_{:02}.npy", i));
            let embed = load_npy_f32(&path).map_err(|e| {
                AppError::VoiceCloningError(format!("Failed to load audio embedding {i}: {e}"))
            })?;
            let vocab = embed.len() / hidden_size.max(1);
            if i == 0 {
                audio_vocab_size = vocab;
            }
            audio_embeds.push(embed);
        }

        tracing::info!(
            "Audio embeddings: n_vq={}, audio_vocab={}, hidden={}",
            n_vq,
            audio_vocab_size,
            hidden_size
        );

        Ok(Self {
            text_embed,
            text_vocab_size,
            hidden_size,
            audio_embeds,
            audio_vocab_size,
            n_vq,
        })
    }

    /// 查找并求和 embedding
    ///
    /// `input_ids`: [1+n_vq] — [text_id, audio_0, ..., audio_{n_vq-1}]
    /// 返回: [hidden_size] — 求和后的 embedding
    pub fn lookup(&self, input_ids: &[i64]) -> Vec<f32> {
        assert!(!input_ids.is_empty());
        let hidden = self.hidden_size;

        // 文本 embedding
        let text_id = input_ids[0] as usize;
        let mut result = if text_id < self.text_vocab_size {
            self.text_embed[text_id * hidden..(text_id + 1) * hidden].to_vec()
        } else {
            vec![0.0; hidden]
        };

        // 音频 embedding 求和
        for (i, audio_id) in input_ids[1..].iter().enumerate() {
            if i >= self.n_vq {
                break;
            }
            let id = *audio_id as usize;
            let vocab_size = self.audio_vocab_size;
            if id < vocab_size {
                let embed = &self.audio_embeds[i];
                for j in 0..hidden {
                    result[j] += embed[id * hidden + j];
                }
            }
        }

        result
    }

    /// 批量查找（并行）
    ///
    /// `input_ids_batch`: [S, 1+n_vq]
    /// 返回: [S, hidden_size]
    pub fn lookup_batch(&self, input_ids_batch: &[Vec<i64>]) -> Vec<Vec<f32>> {
        input_ids_batch
            .par_iter()
            .map(|ids| self.lookup(ids))
            .collect()
    }

    /// 内存占用（字节）
    pub fn nbytes(&self) -> usize {
        let mut total = self.text_embed.len() * 4;
        for a in &self.audio_embeds {
            total += a.len() * 4;
        }
        total
    }

    /// 摘要信息
    pub fn summary(&self) -> String {
        format!(
            "EmbeddingLookup: {}×{} text + {}×{}×{} audio, {:.1} MB",
            self.text_vocab_size,
            self.hidden_size,
            self.n_vq,
            self.audio_vocab_size,
            self.hidden_size,
            self.nbytes() as f64 / (1024.0 * 1024.0)
        )
    }
}

// ─── NumPy .npy 加载 ──────────────────────────────────────

/// 从 .npy 文件加载为 float32 Vec
///
/// 支持 float16 和 float32 数据类型。
pub(crate) fn load_npy_f32(path: &Path) -> Result<Vec<f32>, String> {
    if !path.exists() {
        return Err(format!("File not found: {}", path.display()));
    }

    let bytes = std::fs::read(path).map_err(|e| format!("Read error: {e}"))?;

    // 简易 .npy 解析器
    // .npy 格式: magic "\x93NUMPY" + version + header_len + header + data
    if bytes.len() < 10 {
        return Err("File too short".to_string());
    }

    // 检查 magic number
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("Invalid .npy magic".to_string());
    }

    let version_major = bytes[6];
    let version_minor = bytes[7];

    let header_len = if version_major == 1 {
        // 2-byte header length
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else if version_major == 2 {
        // 4-byte header length
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    } else {
        return Err(format!(
            "Unsupported npy version: {version_major}.{version_minor}"
        ));
    };

    let header_start = if version_major == 1 { 10 } else { 12 };
    let data_start = header_start + header_len;

    if data_start > bytes.len() {
        return Err("Invalid npy: data_start beyond file end".to_string());
    }

    // 解析 header 获取 dtype 和 shape
    let header_str = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|e| format!("Header not UTF-8: {e}"))?;

    // 提取 dtype
    let dtype = extract_dtype(header_str)?;
    // 提取 shape
    let _shape = extract_shape(header_str)?;

    let data = &bytes[data_start..];

    match dtype.as_str() {
        "f4" | "<f4" | "|f4" | ">f4" => {
            // float32
            let result: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(result)
        }
        "f2" | "<f2" | "|f2" => {
            // float16 → float32
            let result: Vec<f32> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f16_to_f32(bits)
                })
                .collect();
            Ok(result)
        }
        _ => Err(format!("Unsupported dtype: {dtype}")),
    }
}

/// 从 header 字符串提取 dtype
fn extract_dtype(header: &str) -> Result<String, String> {
    let key = "'descr':";
    let start = header.find(key).ok_or("descr not found")? + key.len();
    let rest = &header[start..];
    let quote_start = rest.find('\'').ok_or("quote not found")?;
    let rest = &rest[quote_start + 1..];
    let quote_end = rest.find('\'').ok_or("end quote not found")?;
    Ok(rest[..quote_end].to_string())
}

/// 从 header 字符串提取 shape
fn extract_shape(header: &str) -> Result<Vec<usize>, String> {
    let key = "'shape':";
    let start = header.find(key).ok_or("shape not found")? + key.len();
    let rest = &header[start..];
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

/// 从 header 获取 hidden_size（通过 shape）
pub(crate) fn hidden_size_from_npy(path: &Path) -> Result<usize, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Read error: {e}"))?;
    if bytes.len() < 10 {
        return Err("File too short".to_string());
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_str = std::str::from_utf8(&bytes[10..10 + header_len])
        .map_err(|e| format!("Header not UTF-8: {e}"))?;
    let shape = extract_shape(header_str)?;
    shape.get(1).copied().ok_or("No hidden dim".to_string())
}

/// IEEE 754 half-precision → float32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exponent == 0 {
        // subnormal or zero
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // subnormal: normalize
        let mut e = exponent as i32;
        let mut m = mantissa as i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        e += 1;
        m &= 0x3ff;
        let result = ((sign << 31) | (((e + 127 - 15) as u32) << 23) | ((m as u32) << 13)) as u32;
        f32::from_bits(result)
    } else if exponent == 31 {
        // inf or nan
        f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
    } else {
        // normal
        let e = exponent + 127 - 15;
        f32::from_bits((sign << 31) | (e << 23) | (mantissa << 13))
    }
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-5);
        // 0.0 in f16 = 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-5);
        // 2.0 in f16 = 0x4000
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_dtype() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (10, 20), }";
        let dtype = extract_dtype(header).unwrap();
        assert_eq!(dtype, "<f4");
    }

    #[test]
    fn test_extract_shape() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (10, 20), }";
        let shape = extract_shape(header).unwrap();
        assert_eq!(shape, vec![10, 20]);
    }

    #[test]
    fn test_embedding_lookup_summary() {
        let lookup = EmbeddingLookup {
            text_embed: vec![0.0; 100 * 64],
            text_vocab_size: 100,
            hidden_size: 64,
            audio_embeds: vec![vec![0.0; 50 * 64]; 32],
            audio_vocab_size: 50,
            n_vq: 32,
        };
        let summary = lookup.summary();
        assert!(summary.contains("100"));
        assert!(summary.contains("32"));
    }
}
