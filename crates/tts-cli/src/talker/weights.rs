//! 权重加载 & 设备管理

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

/// 将权重张量转换为目标 dtype (如 BF16 → F32)
///
/// 模型权重通常以 BF16 存储，但 Metal GPU 可能需要 F32 或 F16。
pub fn convert_weights_dtype(
    weights: HashMap<String, Tensor>,
    target_dtype: DType,
) -> HashMap<String, Tensor> {
    let mut converted = HashMap::new();
    let mut converted_count = 0;
    for (name, tensor) in weights {
        if tensor.dtype() != target_dtype {
            match tensor.to_dtype(target_dtype) {
                Ok(t) => {
                    converted.insert(name, t);
                    converted_count += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to convert tensor {} to {:?}: {}",
                        name,
                        target_dtype,
                        e
                    );
                    converted.insert(name, tensor);
                }
            }
        } else {
            converted.insert(name, tensor);
        }
    }
    if converted_count > 0 {
        tracing::info!(
            "Converted {} tensors to {:?}",
            converted_count,
            target_dtype
        );
    }
    converted
}

/// 从 safetensors 文件加载权重到 Candle tensors
pub fn load_safetensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read safetensors file: {:?}", path))?;

    let tensors = safetensors::SafeTensors::deserialize(&data)
        .with_context(|| format!("Failed to deserialize safetensors: {:?}", path))?;

    let mut weights = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let dtype = match tensor.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::I64 => DType::I64,
            safetensors::Dtype::I32 => DType::I32,
            _ => DType::F32,
        };
        let tensor_data = tensor.data();
        let shape = tensor.shape();
        let ct = Tensor::from_raw_buffer(tensor_data, dtype, shape, device)
            .map_err(|e| anyhow::anyhow!("Failed to create tensor '{}': {}", name, e))?;
        weights.insert(name, ct);
    }

    tracing::info!("Loaded {} tensors from {:?}", weights.len(), path);
    Ok(weights)
}

/// 根据 config 字符串创建 Candle Device
///
/// 支持的设备字符串:
/// - `"cpu"` — CPU 推理 (始终可用)
/// - `"metal"` — Apple Silicon Metal GPU 加速 (需 `metal` feature)
/// - `"cuda"` — NVIDIA CUDA GPU 加速 (需 `cuda` feature)
///
/// # Metal 设备信息
/// 创建 Metal 设备后，会输出设备名称、最大推荐缓冲区大小等信息。
/// 可通过环境变量 `VT_TTS_METAL_DTYPE` 控制计算精度:
/// - `f32` (默认): F32 精度，兼容性最佳
/// - `f16`: F16 半精度，速度更快，Metal 原生支持
pub fn create_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "cpu" => {
            tracing::info!("Device: CPU");
            Ok(Device::Cpu)
        }
        #[cfg(feature = "metal")]
        "metal" => {
            let device = Device::new_metal(0).context("Failed to create Metal device")?;
            log_metal_device_info(&device);
            Ok(device)
        }
        #[cfg(not(feature = "metal"))]
        "metal" => {
            tracing::warn!(
                "Metal device requested but 'metal' feature not enabled. Falling back to CPU."
            );
            Ok(Device::Cpu)
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            let device = Device::new_cuda(0).context("Failed to create CUDA device")?;
            tracing::info!("Device: CUDA GPU");
            Ok(device)
        }
        #[cfg(not(feature = "cuda"))]
        "cuda" => {
            tracing::warn!(
                "CUDA device requested but 'cuda' feature not enabled. Falling back to CPU."
            );
            Ok(Device::Cpu)
        }
        "auto" => auto_detect_device(),
        _ => {
            tracing::warn!("Unknown device '{}', falling back to CPU", device_str);
            Ok(Device::Cpu)
        }
    }
}

/// 自动检测最佳可用设备
///
/// 优先级: Metal > CUDA > CPU
#[cfg(feature = "metal")]
fn auto_detect_device() -> Result<Device> {
    tracing::info!("Auto-detecting device: trying Metal...");
    match Device::new_metal(0) {
        Ok(device) => {
            log_metal_device_info(&device);
            Ok(device)
        }
        Err(e) => {
            tracing::warn!("Metal unavailable ({}), falling back to CPU", e);
            Ok(Device::Cpu)
        }
    }
}

#[cfg(not(feature = "metal"))]
fn auto_detect_device() -> Result<Device> {
    tracing::info!("Auto-detect: Metal feature not enabled, using CPU");
    Ok(Device::Cpu)
}

/// 输出 Metal 设备信息
#[cfg(feature = "metal")]
fn log_metal_device_info(device: &Device) {
    if let Device::Metal(_) = device {
        tracing::info!("Device: Metal GPU (Apple Silicon)");
        tracing::info!("  Unified memory: supported");
        // MetalDevice 的字段是 pub(crate)，无法从外部访问 device name 等
        // Metal 后端在 Apple Silicon 上自动使用统一内存
    }
}

#[cfg(not(feature = "metal"))]
fn log_metal_device_info(_device: &Device) {}

/// 根据设备选择计算精度
///
/// - CPU: F32 (最高精度)
/// - Metal: 默认 F32，可通过 `VT_TTS_METAL_DTYPE=f16` 环境变量切换为 F16
/// - CUDA: BF16 (Tensor Core 优化)
pub fn compute_dtype_for_device(device: &Device) -> DType {
    match device {
        Device::Cpu => DType::F32,
        Device::Metal(_) => {
            // 支持通过环境变量控制 Metal 计算精度
            match std::env::var("VT_TTS_METAL_DTYPE")
                .unwrap_or_default()
                .to_lowercase()
                .as_str()
            {
                "f16" | "half" => {
                    tracing::info!("Metal dtype: F16 (half precision, faster)");
                    DType::F16
                }
                _ => {
                    tracing::info!("Metal dtype: F32 (default, set VT_TTS_METAL_DTYPE=f16 for faster inference)");
                    DType::F32
                }
            }
        }
        Device::Cuda(_) => DType::BF16,
    }
}
