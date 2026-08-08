//! Metal GPU 加速测试
//!
//! 验证 Metal device 创建、Tensor 基本操作、matmul、conv1d 等。
//! 仅在 `metal` feature 启用时编译运行。
//!
//! 运行: cargo test -p vt-tts --no-default-features --features "metal,cli" test_metal -- --nocapture

#![cfg(feature = "metal")]

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Conv1dConfig;

// ──────────────────────────── 设备创建测试 ────────────────────────────

#[test]
fn test_metal_device_creation() {
    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: Metal device not available: {}", e);
            return;
        }
    };

    // 验证设备类型
    assert!(
        matches!(device, Device::Metal(_)),
        "Expected Metal device, got {:?}",
        device
    );

    // 输出设备信息
    if let Device::Metal(_) = &device {
        eprintln!("Metal device: {:?}", device);
    }

    eprintln!("✓ Metal device creation successful");
}

#[test]
fn test_create_device_metal_string() {
    // 测试 create_device("metal") 函数
    let device = match vt_tts::talker::create_device("metal") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: create_device(\"metal\") failed: {}", e);
            return;
        }
    };

    // 如果 Metal 不可用，create_device 会回退到 CPU，这是正确行为
    match &device {
        Device::Metal(_) => eprintln!("✓ create_device(\"metal\") → Metal GPU"),
        Device::Cpu => eprintln!("✓ create_device(\"metal\") → CPU (Metal fallback, acceptable)"),
        _ => panic!("Unexpected device: {:?}", device),
    }
}

#[test]
fn test_create_device_auto() {
    // 测试 auto 检测
    let device = match vt_tts::talker::create_device("auto") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: create_device(\"auto\") failed: {}", e);
            return;
        }
    };

    eprintln!("✓ create_device(\"auto\") → {:?}", device);
}

// ──────────────────────────── Tensor 基本操作测试 ────────────────────────────

#[test]
fn test_metal_tensor_create() -> Result<()> {
    let device = get_metal_device();

    // 创建 F32 张量
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(data.clone(), (2, 2), &device)?;

    assert_eq!(tensor.dims(), &[2, 2]);
    eprintln!("✓ Tensor creation: shape={:?}", tensor.dims());

    // 验证数据完整性
    let result = tensor.to_vec2::<f32>()?;
    assert_eq!(result, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    eprintln!("✓ Tensor data integrity verified");

    Ok(())
}

#[test]
fn test_metal_tensor_arange() -> Result<()> {
    let device = get_metal_device();

    let tensor = Tensor::arange(0i64, 10i64, &device)?;
    assert_eq!(tensor.dims(), &[10]);

    let result = tensor.to_vec1::<i64>()?;
    assert_eq!(result, (0..10).collect::<Vec<_>>());
    eprintln!("✓ arange(0, 10) verified");

    Ok(())
}

#[test]
fn test_metal_tensor_zeros_ones() -> Result<()> {
    let device = get_metal_device();

    let zeros = Tensor::zeros((3, 4), DType::F32, &device)?;
    assert_eq!(zeros.dims(), &[3, 4]);
    let z = zeros.to_vec2::<f32>()?;
    assert!(z.iter().all(|row| row.iter().all(|&v| v == 0.0)));

    let ones = Tensor::ones((3, 4), DType::F32, &device)?;
    let o = ones.to_vec2::<f32>()?;
    assert!(o.iter().all(|row| row.iter().all(|&v| v == 1.0)));

    eprintln!("✓ zeros((3,4)) and ones((3,4)) verified");

    Ok(())
}

#[test]
fn test_metal_tensor_f16() -> Result<()> {
    let device = get_metal_device();

    // F16 半精度张量 (Metal 原生支持)
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor_f32 = Tensor::from_vec(data, (2, 2), &device)?;
    let tensor_f16 = tensor_f32.to_dtype(DType::F16)?;

    assert_eq!(tensor_f16.dtype(), DType::F16);
    eprintln!("✓ F16 tensor creation on Metal");

    // 转回 F32 验证
    let back = tensor_f16.to_dtype(DType::F32)?;
    let result = back.to_vec2::<f32>()?;
    assert!((result[0][0] - 1.0).abs() < 1e-2);
    assert!((result[0][1] - 2.0).abs() < 1e-2);
    eprintln!("✓ F16 → F32 round-trip verified");

    Ok(())
}

// ──────────────────────────── 数学运算测试 ────────────────────────────

#[test]
fn test_metal_matmul() -> Result<()> {
    let device = get_metal_device();

    // [2, 3] × [3, 4] = [2, 4]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![
        1.0f32, 0.0, 0.0, 1.0, //
        0.0, 1.0, 0.0, 1.0, //
        0.0, 0.0, 1.0, 1.0, //
    ];

    let a = Tensor::from_vec(a_data, (2, 3), &device)?;
    let b = Tensor::from_vec(b_data, (3, 4), &device)?;

    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[2, 4]);

    let result = c.to_vec2::<f32>()?;
    // 手动计算:
    // [1,2,3] × [1,0,0,1; 0,1,0,1; 0,0,1,1] = [1,2,3,6]
    // [4,5,6] × [1,0,0,1; 0,1,0,1; 0,0,1,1] = [4,5,6,15]
    assert!((result[0][0] - 1.0).abs() < 1e-5);
    assert!((result[0][1] - 2.0).abs() < 1e-5);
    assert!((result[0][2] - 3.0).abs() < 1e-5);
    assert!((result[0][3] - 6.0).abs() < 1e-5);
    assert!((result[1][0] - 4.0).abs() < 1e-5);
    assert!((result[1][1] - 5.0).abs() < 1e-5);
    assert!((result[1][2] - 6.0).abs() < 1e-5);
    assert!((result[1][3] - 15.0).abs() < 1e-5);

    eprintln!("✓ matmul [2,3]×[3,4]→[2,4] verified");
    eprintln!("  result: {:?}", result);

    Ok(())
}

#[test]
fn test_metal_matmul_f16() -> Result<()> {
    let device = get_metal_device();

    // F16 matmul — Metal GPU 的核心加速操作
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![
        1.0f32, 0.0, 0.0, 1.0, //
        0.0, 1.0, 0.0, 1.0, //
        0.0, 0.0, 1.0, 1.0, //
    ];

    let a = Tensor::from_vec(a_data, (2, 3), &device)?.to_dtype(DType::F16)?;
    let b = Tensor::from_vec(b_data, (3, 4), &device)?.to_dtype(DType::F16)?;

    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let result = c.to_vec2::<f32>()?;

    // F16 精度较低，使用较宽松的容差
    assert!((result[0][0] - 1.0).abs() < 1e-1);
    assert!((result[0][3] - 6.0).abs() < 1e-1);
    assert!((result[1][3] - 15.0).abs() < 1e-1);

    eprintln!("✓ F16 matmul on Metal verified (tolerance 1e-1)");
    eprintln!("  result: {:?}", result);

    Ok(())
}

#[test]
fn test_metal_add_mul() -> Result<()> {
    let device = get_metal_device();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;

    let sum = (&a + &b)?;
    let s = sum.to_vec2::<f32>()?;
    assert_eq!(s, vec![vec![11.0, 22.0], vec![33.0, 44.0]]);

    let prod = (&a * &b)?;
    let p = prod.to_vec2::<f32>()?;
    assert_eq!(p, vec![vec![10.0, 40.0], vec![90.0, 160.0]]);

    eprintln!("✓ Tensor add and mul verified");

    Ok(())
}

#[test]
fn test_metal_softmax() -> Result<()> {
    let device = get_metal_device();

    let data = vec![1.0f32, 2.0, 3.0];
    let tensor = Tensor::from_vec(data, (1, 3), &device)?;

    let sm = candle_nn::ops::softmax_last_dim(&tensor)?;
    let result = sm.to_vec2::<f32>()?;

    // softmax([1,2,3]) ≈ [0.090, 0.245, 0.665]
    let total: f32 = result[0].iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "softmax should sum to 1.0, got {}",
        total
    );

    eprintln!("✓ softmax verified: {:?} (sum={:.6})", result[0], total);

    Ok(())
}

#[test]
fn test_metal_rms_norm() -> Result<()> {
    let device = get_metal_device();

    // RmsNorm 是 Transformer 的核心组件
    let hidden_size = 4usize;
    let eps = 1e-6f64;

    // 创建权重为 1.0 的 RmsNorm
    let weight = Tensor::ones((hidden_size,), DType::F32, &device)?;
    let rms = candle_nn::RmsNorm::new(weight, eps);

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = Tensor::from_vec(data, (1, hidden_size), &device)?;

    let output = rms.forward(&input)?;
    let result = output.to_vec2::<f32>()?;

    // RMS(x) = sqrt(mean(x^2) + eps)
    // x = [1,2,3,4], x^2 = [1,4,9,16], mean = 7.5, rms = sqrt(7.5+eps) ≈ 2.7386
    // output = x / rms * weight = [0.365, 0.730, 1.095, 1.460]
    let expected_rms = (7.5f32 + eps as f32).sqrt();
    assert!((result[0][0] - 1.0 / expected_rms).abs() < 1e-4);
    assert!((result[0][1] - 2.0 / expected_rms).abs() < 1e-4);

    eprintln!("✓ RmsNorm verified: {:?}", result[0]);

    Ok(())
}

#[test]
fn test_metal_silu() -> Result<()> {
    let device = get_metal_device();

    // SiLU/SwiGLU 是 MLP 的激活函数
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input = Tensor::from_vec(data, (5,), &device)?;

    let output = candle_nn::ops::silu(&input)?;
    let result = output.to_vec1::<f32>()?;

    // SiLU(x) = x * sigmoid(x)
    // SiLU(-2) ≈ -0.2384
    // SiLU(-1) ≈ -0.2689
    // SiLU(0) = 0
    // SiLU(1) ≈ 0.7311
    // SiLU(2) ≈ 1.7616
    assert!((result[2] - 0.0).abs() < 1e-5);
    assert!((result[3] - 0.7311).abs() < 1e-3);
    assert!((result[4] - 1.7616).abs() < 1e-3);

    eprintln!("✓ SiLU verified: {:?}", result);

    Ok(())
}

// ──────────────────────────── 卷积运算测试 ────────────────────────────

#[test]
fn test_metal_conv1d() -> Result<()> {
    let device = get_metal_device();

    // Conv1d 是 ECAPA-TDNN 说话人编码器的核心
    let in_channels = 3;
    let out_channels = 4;
    let kernel_size = 3;
    let seq_len = 10;

    // 创建权重和偏置
    let weight_data = vec![0.1f32; in_channels * out_channels * kernel_size];
    let bias_data = vec![0.0f32; out_channels];

    let weight = Tensor::from_vec(
        weight_data,
        (out_channels, in_channels, kernel_size),
        &device,
    )?;
    let bias = Tensor::from_vec(bias_data, (out_channels,), &device)?;

    let config = Conv1dConfig {
        padding: 1, // "same" padding for kernel_size=3
        stride: 1,
        dilation: 1,
        groups: 1,
        ..Default::default()
    };

    let conv = candle_nn::Conv1d::new(weight, Some(bias), config);

    // 输入: [batch, in_channels, seq_len]
    let input_data = vec![1.0f32; 1 * in_channels * seq_len];
    let input = Tensor::from_vec(input_data, (1, in_channels, seq_len), &device)?;

    let output = conv.forward(&input)?;
    assert_eq!(
        output.dims(),
        &[1, out_channels, seq_len],
        "Conv1d output shape mismatch"
    );

    eprintln!(
        "✓ Conv1d: input [1,{},{}] → output {:?}",
        in_channels,
        seq_len,
        output.dims()
    );

    Ok(())
}

#[test]
fn test_metal_conv1d_dilated() -> Result<()> {
    let device = get_metal_device();

    // 膨胀卷积 — ECAPA-TDNN 中用于扩大感受野
    let in_channels = 2;
    let out_channels = 2;
    let kernel_size = 3;
    let dilation = 2;
    let seq_len = 16;

    let weight_data = vec![0.5f32; out_channels * in_channels * kernel_size];
    let bias_data = vec![0.0f32; out_channels];

    let weight = Tensor::from_vec(
        weight_data,
        (out_channels, in_channels, kernel_size),
        &device,
    )?;
    let bias = Tensor::from_vec(bias_data, (out_channels,), &device)?;

    let config = Conv1dConfig {
        padding: dilation * (kernel_size - 1) / 2, // "same" with dilation
        stride: 1,
        dilation,
        groups: 1,
        ..Default::default()
    };

    let conv = candle_nn::Conv1d::new(weight, Some(bias), config);

    let input_data: Vec<f32> = (0..(1 * in_channels * seq_len))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let input = Tensor::from_vec(input_data, (1, in_channels, seq_len), &device)?;

    let output = conv.forward(&input)?;
    assert_eq!(output.dims(), &[1, out_channels, seq_len]);

    eprintln!(
        "✓ Dilated Conv1d (dilation={}) verified: {:?}",
        dilation,
        output.dims()
    );

    Ok(())
}

// ──────────────────────────── Transformer 组件测试 ────────────────────────────

#[test]
fn test_metal_embedding() -> Result<()> {
    let device = get_metal_device();

    // Embedding 层 — TalkerModel 的文本/codec 嵌入
    let vocab_size = 100;
    let embed_dim = 8;

    let weight_data: Vec<f32> = (0..(vocab_size * embed_dim))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let weight = Tensor::from_vec(weight_data, (vocab_size, embed_dim), &device)?;

    let embedding = candle_nn::Embedding::new(weight, embed_dim);

    let token_ids = Tensor::new(&[3u32, 7, 15, 42], &device)?;
    let output = embedding.forward(&token_ids)?;

    assert_eq!(output.dims(), &[4, embed_dim]);

    eprintln!("✓ Embedding: 4 tokens → {:?}", output.dims());

    Ok(())
}

#[test]
fn test_metal_linear() -> Result<()> {
    let device = get_metal_device();

    // Linear 层 — Transformer 注意力投影
    let in_features = 4;
    let out_features = 8;

    let weight_data = vec![0.5f32; out_features * in_features];
    let bias_data = vec![0.1f32; out_features];

    let weight = Tensor::from_vec(weight_data, (out_features, in_features), &device)?;
    let bias = Tensor::from_vec(bias_data, (out_features,), &device)?;

    let linear = candle_nn::Linear::new(weight, Some(bias));

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, in_features), &device)?;
    let output = linear.forward(&input)?;

    assert_eq!(output.dims(), &[1, out_features]);

    eprintln!("✓ Linear: [1,{}] → {:?}", in_features, output.dims());

    Ok(())
}

#[test]
fn test_metal_attention_operations() -> Result<()> {
    let device = get_metal_device();

    // 模拟注意力计算: softmax(Q @ K^T * scale) @ V
    let batch = 1;
    let num_heads = 2;
    let seq_len = 4;
    let head_dim = 8;

    let q_data = vec![0.5f32; batch * num_heads * seq_len * head_dim];
    let k_data = vec![0.3f32; batch * num_heads * seq_len * head_dim];
    let v_data = vec![0.7f32; batch * num_heads * seq_len * head_dim];

    let q = Tensor::from_vec(q_data, (batch, num_heads, seq_len, head_dim), &device)?;
    let k = Tensor::from_vec(k_data, (batch, num_heads, seq_len, head_dim), &device)?;
    let v = Tensor::from_vec(v_data, (batch, num_heads, seq_len, head_dim), &device)?;

    // Q @ K^T
    let att = q.matmul(&k.transpose(2, 3)?)?;
    assert_eq!(att.dims(), &[batch, num_heads, seq_len, seq_len]);

    // Scale
    let scale = 1.0 / (head_dim as f64).sqrt();
    let att = (att * scale)?;

    // Softmax
    let att = candle_nn::ops::softmax_last_dim(&att)?;

    // Att @ V
    let out = att.matmul(&v)?;
    assert_eq!(out.dims(), &[batch, num_heads, seq_len, head_dim]);

    eprintln!("✓ Attention operations (Q@K^T → softmax → @V) verified");
    eprintln!("  att shape: {:?}, out shape: {:?}", att.dims(), out.dims());

    Ok(())
}

#[test]
fn test_metal_kvcache_operations() -> Result<()> {
    let device = get_metal_device();

    // 模拟 KV Cache 追加操作
    let batch = 1;
    let num_kv_heads = 2;
    let head_dim = 8;

    // 初始 K, V: seq_len = 3
    let k1 = Tensor::from_vec(
        vec![1.0f32; batch * num_kv_heads * 3 * head_dim],
        (batch, num_kv_heads, 3, head_dim),
        &device,
    )?;
    let v1 = Tensor::from_vec(
        vec![2.0f32; batch * num_kv_heads * 3 * head_dim],
        (batch, num_kv_heads, 3, head_dim),
        &device,
    )?;

    // 追加 K, V: seq_len = 1
    let k2 = Tensor::from_vec(
        vec![3.0f32; batch * num_kv_heads * 1 * head_dim],
        (batch, num_kv_heads, 1, head_dim),
        &device,
    )?;
    let v2 = Tensor::from_vec(
        vec![4.0f32; batch * num_kv_heads * 1 * head_dim],
        (batch, num_kv_heads, 1, head_dim),
        &device,
    )?;

    // Cat along seq dimension
    let k_full = Tensor::cat(&[&k1, &k2], 2)?;
    let v_full = Tensor::cat(&[&v1, &v2], 2)?;

    assert_eq!(k_full.dims(), &[batch, num_kv_heads, 4, head_dim]);
    assert_eq!(v_full.dims(), &[batch, num_kv_heads, 4, head_dim]);

    eprintln!(
        "✓ KV Cache concat: [1,2,3,8] + [1,2,1,8] → {:?}",
        k_full.dims()
    );

    Ok(())
}

#[test]
fn test_metal_rope_operations() -> Result<()> {
    let device = get_metal_device();

    // 测试 RoPE (Rotary Position Embedding)
    let dim = 64usize;
    let max_seq_len = 32usize;
    let theta = 10000.0f64;

    let rope = vt_tts::transformer::RotaryEmbedding::new(dim, max_seq_len, theta, &device)?;

    // 创建 Q, K 张量 [batch, heads, seq_len, head_dim]
    let batch = 1;
    let num_heads = 2;
    let seq_len = 4;

    let q = Tensor::from_vec(
        vec![1.0f32; batch * num_heads * seq_len * dim],
        (batch, num_heads, seq_len, dim),
        &device,
    )?;
    let k = Tensor::from_vec(
        vec![0.5f32; batch * num_heads * seq_len * dim],
        (batch, num_heads, seq_len, dim),
        &device,
    )?;

    let (q_rot, k_rot) = rope.apply(&q, &k, 0)?;

    assert_eq!(q_rot.dims(), q.dims());
    assert_eq!(k_rot.dims(), k.dims());

    // 验证旋转后的值与原始值不同
    let q_orig_flat = q.flatten_all()?.to_vec1::<f32>()?;
    let q_rotated_flat = q_rot.flatten_all()?.to_vec1::<f32>()?;
    let diff_count = q_orig_flat
        .iter()
        .zip(q_rotated_flat.iter())
        .filter(|(a, b)| (**a - **b).abs() > 1e-6)
        .count();
    assert!(diff_count > 0, "RoPE should modify the tensor values");

    eprintln!(
        "✓ RoPE: {} values changed out of {}",
        diff_count,
        q_orig_flat.len()
    );

    Ok(())
}

// ──────────────────────────── 大张量性能测试 ────────────────────────────

#[test]
fn test_metal_large_matmul_perf() -> Result<()> {
    let device = get_metal_device();

    // 模拟 Transformer 注意力的矩阵乘法规模
    // hidden_size=1024, seq_len=64
    let n = 1024usize;
    let m = 64usize;
    let k = 1024usize;

    let a = Tensor::randn(0.0f32, 1.0, (m, n), &device)?;
    let b = Tensor::randn(0.0f32, 1.0, (n, k), &device)?;

    // 预热
    let _ = a.matmul(&b)?;

    // 计时
    let start = std::time::Instant::now();
    let c = a.matmul(&b)?;
    let elapsed = start.elapsed();

    assert_eq!(c.dims(), &[m, k]);

    eprintln!(
        "✓ Large matmul [{}x{}] × [{}x{}] → [{}x{}] in {:.2}ms",
        m,
        n,
        n,
        k,
        m,
        k,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}

#[test]
fn test_metal_multi_head_attention_perf() -> Result<()> {
    let device = get_metal_device();

    // 完整的多头注意力模拟 (0.6B 模型规模)
    let batch = 1;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let seq_len = 32;

    // Q, K, V 投影
    let q = Tensor::randn(0.0f32, 0.1, (batch, num_heads, seq_len, head_dim), &device)?;
    let k = Tensor::randn(
        0.0f32,
        0.1,
        (batch, num_kv_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0.0f32,
        0.1,
        (batch, num_kv_heads, seq_len, head_dim),
        &device,
    )?;

    let start = std::time::Instant::now();

    // GQA: repeat KV
    let n_rep = num_heads / num_kv_heads;
    let k_expanded = if n_rep > 1 {
        let (b, h, s, d) = k.dims4()?;
        let k = k.unsqueeze(2)?.broadcast_as((b, h, n_rep, s, d))?;
        k.reshape((b, h * n_rep, s, d))?
    } else {
        k
    };
    let v_expanded = if n_rep > 1 {
        let (b, h, s, d) = v.dims4()?;
        let v = v.unsqueeze(2)?.broadcast_as((b, h, n_rep, s, d))?;
        v.reshape((b, h * n_rep, s, d))?
    } else {
        v
    };

    // Attention: Q @ K^T * scale
    let scale = 1.0 / (head_dim as f64).sqrt();
    let att = q.matmul(&k_expanded.transpose(2, 3)?)?;
    let att = (att * scale)?;
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let out = att.matmul(&v_expanded)?;

    let elapsed = start.elapsed();

    assert_eq!(out.dims(), &[batch, num_heads, seq_len, head_dim]);

    eprintln!(
        "✓ Multi-head attention (heads={}, kv_heads={}, dim={}, seq={}) in {:.2}ms",
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}

// ──────────────────────────── 重复惩罚测试 ────────────────────────────

#[test]
fn test_repetition_penalty_basic() -> Result<()> {
    let device = get_metal_device();

    // 创建 logits: token 0 的 logit 最高 (10.0), 其他较低
    let logits_data = vec![10.0f32, 5.0, 3.0, 1.0, 0.5];
    let logits = Tensor::from_vec(logits_data, (5,), &device)?;

    // 无重复惩罚: 应该采样到 token 0 (logit 最高)
    let token_no_penalty = vt_tts::talker::sample_top_k(&logits, 5, 1.0, Some(42), 1.0, 0, &[])?;

    // 有重复惩罚 (penalty=2.0), token 0 已在历史中
    let token_with_penalty =
        vt_tts::talker::sample_top_k(&logits, 5, 1.0, Some(42), 2.0, 0, &[0u32])?;

    eprintln!(
        "✓ Repetition penalty: no_penalty={}, with_penalty={}",
        token_no_penalty, token_with_penalty
    );

    // 验证: 有惩罚时 token 0 的 logit 被除以 2.0 → 5.0
    // token 1 的 logit 仍为 5.0, 所以 token 0 和 1 的 logit 相近
    // 由于使用了相同种子, 如果 token 0 的概率被充分降低, 可能采样到不同 token
    // 这里主要验证函数能正确执行不 panic

    Ok(())
}

#[test]
fn test_repetition_penalty_reduces_repeated_token() -> Result<()> {
    let device = get_metal_device();

    // logits: token 0 远高于其他
    let logits_data = vec![100.0f32, 1.0, 1.0, 1.0, 1.0];
    let logits = Tensor::from_vec(logits_data, (5,), &device)?;

    // 无惩罚: 必定采样到 token 0
    for _ in 0..10 {
        let token = vt_tts::talker::sample_top_k(&logits, 5, 1.0, None, 1.0, 0, &[])?;
        assert_eq!(token, 0, "Without penalty, should always pick token 0");
    }

    // 有强惩罚 (penalty=100.0): token 0 的 logit 100/100=1.0, 与其他持平
    // 应该能采样到非 0 的 token
    let mut found_non_zero = false;
    for seed in 0..100u64 {
        let token = vt_tts::talker::sample_top_k(&logits, 5, 1.0, Some(seed), 100.0, 0, &[0u32])?;
        if token != 0 {
            found_non_zero = true;
            break;
        }
    }
    assert!(
        found_non_zero,
        "With strong penalty, should sample non-zero token at least once"
    );

    eprintln!("✓ Strong repetition penalty (100.0) successfully reduces token 0 probability");

    Ok(())
}

#[test]
fn test_no_repeat_ngram_size() -> Result<()> {
    let device = get_metal_device();

    // 创建 logits: token 0 的 logit 最高
    let logits_data = vec![10.0f32, 5.0, 3.0, 1.0, 0.5];
    let logits = Tensor::from_vec(logits_data, (5,), &device)?;

    // 历史: [1, 2, 1, 2], ngram_size=2
    // 当前前缀 (last 1 token) = [2]
    // 搜索历史中匹配 [2] 的位置: j=1 (history[1]=2), j=3 (history[3]=2)
    // j=1: banned = history[2] = 1
    // j=3: banned = history[4] = out of range for history.len()=4 and n=2...
    //   Actually j ranges 0..=(4-2)=0..=2, so j=0,1,2
    //   j=0: prefix=history[0..1]=[1], != [2]
    //   j=1: prefix=history[1..2]=[2], == [2]! banned=history[2]=1
    //   j=2: prefix=history[2..3]=[1], != [2]
    // So token 1 is banned (set to -inf)
    let history = vec![1u32, 2, 1, 2];
    let token = vt_tts::talker::sample_top_k(&logits, 5, 1.0, Some(42), 1.0, 2, &history)?;

    eprintln!(
        "✓ No-repeat ngram (size=2): with history [1,2,1,2], token 1 banned, sampled token {}",
        token
    );
    // token 1 is banned, so result should not be 1
    assert_ne!(
        token, 1,
        "Token 1 should be banned by no_repeat_ngram_size=2"
    );

    Ok(())
}

#[test]
fn test_no_repeat_ngram_size_3() -> Result<()> {
    let device = get_metal_device();

    // logits: token 0 最高
    let logits_data = vec![10.0f32, 8.0, 5.0, 3.0, 1.0];
    let logits = Tensor::from_vec(logits_data, (5,), &device)?;

    // 历史: [3, 4, 2, 3, 4], ngram_size=3
    // 当前前缀 (last 2 tokens) = [3, 4]
    // 搜索 j in 0..=(5-3)=0..=2:
    //   j=0: prefix=history[0..2]=[3,4], == [3,4]! banned=history[2]=2
    //   j=1: prefix=history[1..3]=[4,2], != [3,4]
    //   j=2: prefix=history[2..4]=[2,3], != [3,4]
    // So token 2 is banned
    let history = vec![3u32, 4, 2, 3, 4];
    let token = vt_tts::talker::sample_top_k(&logits, 5, 1.0, Some(42), 1.0, 3, &history)?;

    eprintln!(
        "✓ No-repeat ngram (size=3): with history [3,4,2,3,4], token 2 banned, sampled token {}",
        token
    );
    assert_ne!(
        token, 2,
        "Token 2 should be banned by no_repeat_ngram_size=3"
    );

    Ok(())
}

#[test]
fn test_repetition_penalty_with_ngram_combined() -> Result<()> {
    let device = get_metal_device();

    // logits: token 0 和 1 都很高
    let logits_data = vec![10.0f32, 9.0, 1.0, 1.0, 1.0];
    let logits = Tensor::from_vec(logits_data, (5,), &device)?;

    // 同时使用重复惩罚和 no-repeat ngram
    // history=[0,1,0,1], penalty=1.5, ngram_size=2
    // 重复惩罚: tokens 0 和 1 都在历史中, logit 被惩罚
    // ngram: 当前前缀=[1], j=1(history[1]=1) banned=history[2]=0, j=3(history[3]=1) but j max is 2
    //   Actually j in 0..=(4-2)=0..=2
    //   j=0: [0]!= [1]
    //   j=1: [1]==[1]! banned=history[2]=0
    //   j=2: [0]!=[1]
    // So token 0 is banned by ngram AND penalized
    // token 1 is penalized but not banned
    let history = vec![0u32, 1, 0, 1];
    let token = vt_tts::talker::sample_top_k(&logits, 5, 1.0, Some(42), 1.5, 2, &history)?;

    eprintln!("✓ Combined penalty + ngram: with history [0,1,0,1], token 0 banned+penalized, sampled token {}", token);
    // token 0 is banned by ngram, so result should not be 0
    assert_ne!(token, 0, "Token 0 should be banned by ngram_size=2");

    Ok(())
}

// ──────────────────────────── 混合精度测试 ────────────────────────────

#[test]
fn test_mixed_precision_f16_f32_matmul() -> Result<()> {
    // 验证 F16 和 F32 张量可以在 Metal 上分别进行 matmul
    let device = get_metal_device();

    // F16 matmul (模拟 TalkerModel 的 Transformer attention)
    let a_f16 = Tensor::randn(0.0f32, 1.0, (64, 1024), &device)?.to_dtype(DType::F16)?;
    let b_f16 = Tensor::randn(0.0f32, 1.0, (1024, 1024), &device)?.to_dtype(DType::F16)?;
    let c_f16 = a_f16.matmul(&b_f16)?;
    assert_eq!(c_f16.dtype(), DType::F16);
    assert_eq!(c_f16.dims(), &[64, 1024]);

    // F32 matmul (模拟 CodePredictor 的 embedding lookup)
    let a_f32 = Tensor::randn(0.0f32, 1.0, (64, 1024), &device)?;
    let b_f32 = Tensor::randn(0.0f32, 1.0, (1024, 1024), &device)?;
    let c_f32 = a_f32.matmul(&b_f32)?;
    assert_eq!(c_f32.dtype(), DType::F32);
    assert_eq!(c_f32.dims(), &[64, 1024]);

    // F16 matmul 应该比 F32 快 (验证两者都能正常工作)
    eprintln!(
        "✓ Mixed precision matmul: F16 result dtype={:?}, F32 result dtype={:?}",
        c_f16.dtype(),
        c_f32.dtype()
    );

    Ok(())
}

#[test]
fn test_mixed_precision_f16_to_f32_conversion() -> Result<()> {
    // 验证 F16 → F32 转换在 Metal 上正常工作
    // 这模拟了 TalkerModel (F16) 的输出 logits 转为 F32 用于采样
    let device = get_metal_device();

    let logits_f16 = Tensor::randn(0.0f32, 5.0, (3072,), &device)?.to_dtype(DType::F16)?;
    let logits_f32 = logits_f16.to_dtype(DType::F32)?;

    assert_eq!(logits_f16.dtype(), DType::F16);
    assert_eq!(logits_f32.dtype(), DType::F32);

    // 验证采样函数能处理从 F16 转换的 logits
    let token = vt_tts::talker::sample_top_k(&logits_f32, 50, 0.8, Some(42), 1.0, 0, &[])?;
    assert!(token < 3072, "Sampled token should be within vocab range");

    eprintln!("✓ F16→F32 conversion + sampling: sampled token {}", token);

    Ok(())
}

#[test]
fn test_mixed_precision_dtype_selection() -> Result<()> {
    // 验证混合精度模式下的 dtype 选择逻辑
    let device = get_metal_device();

    // 模拟混合精度的 dtype 选择
    let (talker_dtype, other_dtype) = match &device {
        Device::Metal(_) => (DType::F16, DType::F32),
        _ => (DType::F32, DType::F32),
    };

    // 在 Metal 上，TalkerModel 应该用 F16，其他用 F32
    if matches!(device, Device::Metal(_)) {
        assert_eq!(
            talker_dtype,
            DType::F16,
            "TalkerModel should use F16 on Metal"
        );
        assert_eq!(
            other_dtype,
            DType::F32,
            "CodePredictor/Decoder should use F32 on Metal"
        );
    }

    eprintln!(
        "✓ Mixed precision dtype selection: TalkerModel={:?}, CP/Decoder={:?}",
        talker_dtype, other_dtype
    );

    Ok(())
}

#[test]
fn test_mixed_precision_conv1d_f32() -> Result<()> {
    // 验证 CodePredictor/Decoder 的 Conv1d 在 F32 下正常工作
    // (混合精度模式下，conv 层应始终使用 F32)
    let device = get_metal_device();

    use candle_nn::{Conv1d, Conv1dConfig};

    let in_channels = 1024;
    let out_channels = 1024;
    let kernel_size = 7;
    let seq_len = 100;

    let config = Conv1dConfig {
        padding: 3,
        stride: 1,
        dilation: 1,
        groups: 1,
        ..Default::default()
    };

    // F32 conv weights (模拟 CodePredictor/Decoder)
    let weight = Tensor::randn(
        0.0f32,
        0.02,
        (out_channels, in_channels, kernel_size),
        &device,
    )?;
    let bias = Tensor::zeros(out_channels, DType::F32, &device)?;
    let conv = Conv1d::new(weight, Some(bias), config);

    // F32 input
    let input = Tensor::randn(0.0f32, 1.0, (1, in_channels, seq_len), &device)?;
    let output = conv.forward(&input)?;

    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.dims(), &[1, out_channels, seq_len]);

    eprintln!(
        "✓ F32 Conv1d (CodePredictor/Decoder): [{},{},{}] → {:?}",
        1,
        in_channels,
        seq_len,
        output.dims()
    );

    Ok(())
}

#[test]
fn test_mixed_precision_engine_config() -> Result<()> {
    // 验证 TtsEngineConfig 的 mixed_precision 字段正确序列化/反序列化
    let config = vt_tts::TtsEngineConfig {
        model_dir: std::path::PathBuf::from("models/qwen3-tts"),
        device: "metal".to_string(),
        temperature: 0.8,
        top_k: 50,
        repetition_penalty: 1.05,
        no_repeat_ngram_size: 3,
        seed: Some(42),
        max_codes: 800,
        output_sample_rate: 24000,
        language: "auto".to_string(),
        mixed_precision: true,
        quantize: None,
        decode_device: None,
    };

    // 序列化为 JSON
    let json = serde_json::to_string(&config)?;
    assert!(
        json.contains("mixed_precision"),
        "JSON should contain mixed_precision field"
    );
    assert!(
        json.contains("true"),
        "mixed_precision should be true in JSON"
    );

    // 反序列化
    let config2: vt_tts::TtsEngineConfig = serde_json::from_str(&json)?;
    assert!(
        config2.mixed_precision,
        "Deserialized config should have mixed_precision=true"
    );

    // 默认值应为 false
    let default_config = vt_tts::TtsEngineConfig::default();
    assert!(
        !default_config.mixed_precision,
        "Default config should have mixed_precision=false"
    );

    eprintln!("✓ TtsEngineConfig mixed_precision: serialize/deserialize/default all correct");

    Ok(())
}

// ──────────────────────────── KV Cache 优化测试 ────────────────────────────

#[test]
fn test_kvcache_preallocated_creation() -> Result<()> {
    let device = get_metal_device();

    let cache =
        vt_tts::transformer::AnyKVCache::new_preallocated(512, 8, 128, DType::F32, &device)?;

    assert!(cache.is_preallocated());
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    eprintln!("✓ PreAllocated KVCache: created (max=512, kv_heads=8, dim=128)");

    Ok(())
}

#[test]
fn test_kvcache_preallocated_update() -> Result<()> {
    let device = get_metal_device();

    let mut cache =
        vt_tts::transformer::AnyKVCache::new_preallocated(512, 2, 8, DType::F32, &device)?;

    // 模拟 prefill: seq_len=3
    let k1 = Tensor::zeros((1, 2, 3, 8), DType::F32, &device)?;
    let v1 = Tensor::zeros((1, 2, 3, 8), DType::F32, &device)?;
    let (k_full, v_full) = cache.update(&k1, &v1)?;
    assert_eq!(k_full.dims(), &[1, 2, 3, 8]);
    assert_eq!(cache.len(), 3);

    // 模拟 step: seq_len=1
    let k2 = Tensor::zeros((1, 2, 1, 8), DType::F32, &device)?;
    let v2 = Tensor::zeros((1, 2, 1, 8), DType::F32, &device)?;
    let (k_full, v_full) = cache.update(&k2, &v2)?;
    assert_eq!(k_full.dims(), &[1, 2, 4, 8]);
    assert_eq!(cache.len(), 4);

    eprintln!(
        "✓ PreAllocated KVCache: prefill(3) + step(1) → len={}",
        cache.len()
    );

    Ok(())
}

#[test]
fn test_kvcache_legacy_vs_preallocated() -> Result<()> {
    let device = get_metal_device();

    // Legacy 模式
    let mut legacy = vt_tts::transformer::AnyKVCache::new();
    assert!(!legacy.is_preallocated());

    let k = Tensor::zeros((1, 2, 5, 8), DType::F32, &device)?;
    let v = Tensor::zeros((1, 2, 5, 8), DType::F32, &device)?;
    let (lk, _) = legacy.update(&k, &v)?;
    assert_eq!(lk.dims(), &[1, 2, 5, 8]);

    // 预分配模式
    let mut prealloc =
        vt_tts::transformer::AnyKVCache::new_preallocated(512, 2, 8, DType::F32, &device)?;

    let (pk, _) = prealloc.update(&k, &v)?;
    assert_eq!(pk.dims(), &[1, 2, 5, 8]);

    // 两种模式结果形状一致
    assert_eq!(lk.dims(), pk.dims());

    eprintln!(
        "✓ KVCache: legacy and preallocated produce same shape {:?}",
        pk.dims()
    );

    Ok(())
}

// ──────────────────────────── GPU Argmax 测试 ────────────────────────────

#[test]
fn test_argmax_on_device_single_row() -> Result<()> {
    let device = get_metal_device();

    // [1, 5] — 单行 argmax
    let logits = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 0.5, 2.0], (1, 5), &device)?;
    let tokens = vt_tts::talker::argmax_on_device(&logits)?;
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], 1, "argmax should pick index 1 (value=5.0)");

    eprintln!("✓ argmax_on_device: single row → token {}", tokens[0]);

    Ok(())
}

#[test]
fn test_argmax_on_device_multi_row() -> Result<()> {
    let device = get_metal_device();

    // [3, 5] — 多行 batch argmax
    let logits_data = vec![
        1.0f32, 5.0, 3.0, 0.5, 2.0, // row 0: max at idx 1
        9.0, 1.0, 2.0, 3.0, 0.5, // row 1: max at idx 0
        0.1, 0.2, 9.0, 0.3, 0.4, // row 2: max at idx 2
    ];
    let logits = Tensor::from_vec(logits_data, (3, 5), &device)?;
    let tokens = vt_tts::talker::argmax_on_device(&logits)?;

    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0], 1, "row 0: max at idx 1");
    assert_eq!(tokens[1], 0, "row 1: max at idx 0");
    assert_eq!(tokens[2], 2, "row 2: max at idx 2");

    eprintln!("✓ argmax_on_device: 3 rows → tokens {:?}", tokens);

    Ok(())
}

// ──────────────────────────── GPU 采样路径测试 ────────────────────────────

#[test]
fn test_sample_top_k_gpu_greedy() -> Result<()> {
    let device = get_metal_device();

    // 当无惩罚、无 ngram、温度=1.0 时，应使用 GPU argmax
    let logits = Tensor::from_vec(vec![1.0f32, 9.0, 3.0, 0.5, 2.0], (5,), &device)?;
    let token = vt_tts::talker::sample_top_k_gpu(&logits, 50, 1.0, Some(42), 1.0, 0, &[])?;

    assert_eq!(token, 1, "greedy should pick idx 1 (value=9.0)");

    eprintln!("✓ sample_top_k_gpu (greedy): token {}", token);

    Ok(())
}

#[test]
fn test_sample_top_k_gpu_with_penalty() -> Result<()> {
    let device = get_metal_device();

    // 有重复惩罚时，token 1 在历史中，logit 被惩罚
    let logits = Tensor::from_vec(vec![1.0f32, 100.0, 1.0, 1.0, 1.0], (5,), &device)?;

    // 无惩罚: 必定选 1
    let t1 = vt_tts::talker::sample_top_k_gpu(&logits, 50, 1.0, Some(42), 1.0, 0, &[])?;
    assert_eq!(t1, 1, "without penalty, should pick idx 1");

    // 有强惩罚: token 1 logit=100/100=1.0, 其他 logit 不变, 应可选到其他
    let mut found_non_one = false;
    for seed in 0..50u64 {
        let t = vt_tts::talker::sample_top_k_gpu(&logits, 5, 0.8, Some(seed), 100.0, 0, &[1u32])?;
        if t != 1 {
            found_non_one = true;
            break;
        }
    }
    assert!(
        found_non_one,
        "with strong penalty, should sample non-1 token"
    );

    eprintln!("✓ sample_top_k_gpu (with penalty): successfully reduces token 1 probability");

    Ok(())
}

#[test]
fn test_sample_top_k_gpu_fallback() -> Result<()> {
    let device = get_metal_device();

    // 通用路径: top_k > 1 + 有惩罚 → 回退到 CPU sample_top_k
    let logits = Tensor::from_vec(vec![10.0f32, 8.0, 5.0, 3.0, 1.0], (5,), &device)?;
    let token = vt_tts::talker::sample_top_k_gpu(&logits, 3, 0.8, Some(42), 1.1, 0, &[0u32])?;

    // 主要验证不 panic + 返回有效 token
    assert!(token < 5, "token should be within vocab range");

    eprintln!("✓ sample_top_k_gpu (fallback): token {}", token);

    Ok(())
}

// ──────────────────────────── 权重量化测试 ────────────────────────────

#[test]
fn test_quantize_parse() {
    // 测试量化格式解析
    assert_eq!(vt_tts::talker::parse_quantize(&None), None);
    assert_eq!(
        vt_tts::talker::parse_quantize(&Some("none".to_string())),
        None
    );
    assert_eq!(
        vt_tts::talker::parse_quantize(&Some("q8_0".to_string())),
        Some(candle_core::quantized::GgmlDType::Q8_0)
    );
    assert_eq!(
        vt_tts::talker::parse_quantize(&Some("Q4_0".to_string())), // 大小写不敏感
        Some(candle_core::quantized::GgmlDType::Q4_0)
    );
    assert_eq!(
        vt_tts::talker::parse_quantize(&Some("q4k".to_string())),
        Some(candle_core::quantized::GgmlDType::Q4K)
    );
    // 未知格式应返回 None
    assert_eq!(
        vt_tts::talker::parse_quantize(&Some("q3".to_string())),
        None
    );
    eprintln!("✓ parse_quantize: all format parsing correct");
}

#[test]
fn test_quantize_qlinear_creation() -> Result<()> {
    let device = get_metal_device();

    // 创建权重矩阵 [out=64, in=256]
    // 256 是 Q8_0 (block=32) 和 Q4K (block=256) block_size 的公倍数
    let weight = Tensor::randn(0f32, 1f32, (64, 256), &device)?;

    // 不量化: 创建常规 Linear
    let qlinear_f32 = vt_tts::transformer::QLinear::from_weight(weight.clone(), None)?;
    assert!(matches!(
        qlinear_f32,
        vt_tts::transformer::QLinear::Linear(_)
    ));

    // Q8_0 量化 (block_size=32, 256%32=0)
    let qlinear_q8 = vt_tts::transformer::QLinear::from_weight(
        weight.clone(),
        Some(candle_core::quantized::GgmlDType::Q8_0),
    )?;
    assert!(matches!(
        qlinear_q8,
        vt_tts::transformer::QLinear::Quantized(_)
    ));

    // Q4K 量化 (block_size=256, 256%256=0)
    let qlinear_q4k = vt_tts::transformer::QLinear::from_weight(
        weight,
        Some(candle_core::quantized::GgmlDType::Q4K),
    )?;
    assert!(matches!(
        qlinear_q4k,
        vt_tts::transformer::QLinear::Quantized(_)
    ));

    eprintln!("✓ QLinear: F32/Q8_0/Q4K creation all successful");

    Ok(())
}

#[test]
fn test_quantize_qlinear_forward() -> Result<()> {
    use candle_core::Module;
    let device = get_metal_device();

    // 创建权重和输入
    let weight = Tensor::randn(0f32, 0.1f32, (64, 32), &device)?;
    let xs = Tensor::randn(0f32, 1f32, (1, 32), &device)?;

    // F32 前向
    let qlinear_f32 = vt_tts::transformer::QLinear::from_weight(weight.clone(), None)?;
    let out_f32 = qlinear_f32.forward(&xs)?;
    assert_eq!(out_f32.dims(), &[1, 64]);

    // Q8_0 前向 — 结果应接近 F32 (有一定量化误差)
    let qlinear_q8 = vt_tts::transformer::QLinear::from_weight(
        weight,
        Some(candle_core::quantized::GgmlDType::Q8_0),
    )?;
    let out_q8 = qlinear_q8.forward(&xs)?;
    assert_eq!(out_q8.dims(), &[1, 64]);

    // 比较 F32 和 Q8_0 的输出 (Q8_0 误差应较小)
    let f32_vec = out_f32.to_vec2::<f32>()?;
    let q8_vec = out_q8.to_vec2::<f32>()?;
    let max_diff: f32 = f32_vec[0]
        .iter()
        .zip(q8_vec[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // Q8_0 量化误差应在合理范围内
    assert!(max_diff < 1.0, "Q8_0 max diff too large: {}", max_diff);

    eprintln!("✓ QLinear forward: F32 vs Q8_0 max_diff={:.4}", max_diff);

    Ok(())
}

#[test]
fn test_quantize_engine_config() -> Result<()> {
    // 验证 TtsEngineConfig 的 quantize 字段正确序列化/反序列化
    let config = vt_tts::TtsEngineConfig {
        model_dir: std::path::PathBuf::from("models/qwen3-tts"),
        device: "metal".to_string(),
        temperature: 0.8,
        top_k: 50,
        repetition_penalty: 1.05,
        no_repeat_ngram_size: 3,
        seed: Some(42),
        max_codes: 800,
        output_sample_rate: 24000,
        language: "auto".to_string(),
        mixed_precision: false,
        quantize: Some("q8_0".to_string()),
        decode_device: None,
    };

    // 序列化为 JSON
    let json = serde_json::to_string(&config)?;
    assert!(
        json.contains("quantize"),
        "JSON should contain quantize field"
    );
    assert!(json.contains("q8_0"), "JSON should contain q8_0 value");

    // 反序列化
    let config2: vt_tts::TtsEngineConfig = serde_json::from_str(&json)?;
    assert_eq!(config2.quantize, Some("q8_0".to_string()));

    // 默认值
    let default_config = vt_tts::TtsEngineConfig::default();
    assert_eq!(default_config.quantize, None);
    assert_eq!(default_config.decode_device, None);

    eprintln!("✓ TtsEngineConfig quantize: serialize/deserialize/default all correct");

    Ok(())
}

// ──────────────────────────── Decode Device Offload 测试 ────────────────────────────

#[test]
fn test_decode_device_config_serialization() -> Result<()> {
    // 验证 decode_device 字段正确序列化/反序列化
    let config = vt_tts::TtsEngineConfig {
        model_dir: std::path::PathBuf::from("models/qwen3-tts"),
        device: "metal".to_string(),
        temperature: 0.8,
        top_k: 50,
        repetition_penalty: 1.05,
        no_repeat_ngram_size: 3,
        seed: Some(42),
        max_codes: 800,
        output_sample_rate: 24000,
        language: "auto".to_string(),
        mixed_precision: false,
        quantize: None,
        decode_device: Some("cpu".to_string()),
    };

    // 序列化为 JSON
    let json = serde_json::to_string(&config)?;
    assert!(
        json.contains("decode_device"),
        "JSON should contain decode_device field"
    );
    assert!(json.contains("cpu"), "JSON should contain cpu value");

    // 反序列化
    let config2: vt_tts::TtsEngineConfig = serde_json::from_str(&json)?;
    assert_eq!(config2.decode_device.as_deref(), Some("cpu"));
    assert_eq!(config2.device, "metal");

    // 默认值
    let default_config = vt_tts::TtsEngineConfig::default();
    assert_eq!(default_config.decode_device, None);

    eprintln!("✓ decode_device config: serialize/deserialize/default all correct");

    Ok(())
}

#[test]
fn test_decode_device_cpu_creates_cpu_device() -> Result<()> {
    // 验证 create_device("cpu") 返回 CPU 设备
    let device = vt_tts::talker::create_device("cpu")?;
    assert!(matches!(device, Device::Cpu), "Expected CPU device");

    eprintln!("✓ create_device(\"cpu\") → CPU");

    Ok(())
}

#[test]
fn test_decode_device_cpu_conv1d() -> Result<()> {
    // 验证 CPU 上 Conv1d 能正常工作
    // 这是 CPU decode offload 的基础：如果 CPU Conv1d 能工作，
    // 则 AudioDecoder 可以在 CPU 上运行
    let device = Device::Cpu;
    let weight = Tensor::zeros((16, 16, 7), DType::F32, &device)?;
    let bias = Tensor::zeros(16, DType::F32, &device)?;
    let conv = candle_nn::conv::Conv1d::new(
        weight,
        Some(bias),
        candle_nn::Conv1dConfig {
            padding: 3,
            ..Default::default()
        },
    );

    // 输入: [1, 16, 1000] (模拟长序列)
    let input = Tensor::randn(0.0f32, 1.0, (1, 16, 1000), &device)?;
    let output = conv.forward(&input)?;

    // 输出形状应该与输入相同
    assert_eq!(output.dim(0)?, 1);
    assert_eq!(output.dim(1)?, 16);
    assert_eq!(output.dim(2)?, 1000);

    eprintln!("✓ CPU Conv1d: [1,16,1000] → [1,16,1000] OK");

    Ok(())
}

// ──────────────────────────── 辅助函数 ────────────────────────────

/// 获取 Metal 设备，如果不可用则跳过测试
fn get_metal_device() -> Device {
    Device::new_metal(0).unwrap_or_else(|e| {
        panic!(
            "Metal device not available: {}. Tests require Apple Silicon with Metal support.",
            e
        );
    })
}
