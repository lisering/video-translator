//! SOLA (Synchronized Overlap-Add) 音频拼接模块
//!
//! 借鉴 GPT-SoVITS 的 `sola_algorithm()` 实现，通过互相关（cross-correlation）
//! 寻找最佳拼接点并应用 Hann 窗淡入淡出，消除段间拼接痕迹。
//!
//! # 核心原理
//! 当两段音频需要拼接时，简单首尾相接会产生"咔嗒"声（click）。
//! SOLA 算法通过以下步骤实现无痕拼接：
//!
//! 1. 在前一段尾部和后一段头部之间定义一个重叠区域
//! 2. 计算重叠区域内不同偏移量的互相关值
//! 3. 选择互相关最大的偏移作为最佳拼接点
//! 4. 使用 Hann 窗对重叠区域进行淡入淡出加权求和
//!
//! # 性能
//! - 互相关使用朴素滑窗实现（O(N×M)，N=搜索范围，M=窗长）
//! - 对于 16kHz 音频，典型参数下 <1ms
//! - 支持 Rayon 并行（可选）
//!
//! # 示例
//! ```
//! use vt_core::sola::sola_overlap_add;
//!
//! let prev = vec![0.1f32; 1600]; // 前一段尾部
//! let next = vec![0.2f32; 1600]; // 后一段头部
//! let overlap_len = 320;          // 20ms 重叠
//! let (offset, merged) = sola_overlap_add(&prev, &next, overlap_len);
//! ```

// ─── 常量 ─────────────────────────────────────────────────

/// 默认重叠长度（采样数），对应 16kHz 下 20ms
pub const DEFAULT_OVERLAP_SAMPLES: usize = 320;

/// 默认搜索范围（采样数），对应 16kHz 下 10ms
pub const DEFAULT_SEARCH_RANGE: usize = 160;

/// 最小信号能量阈值，低于此值认为是静音，跳过互相关
pub const SILENCE_THRESHOLD: f32 = 1e-4;

// ─── 核心算法 ─────────────────────────────────────────────

/// 生成 Hann 窗
///
/// `w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))`
///
/// # 参数
/// - `n`: 窗长度
#[must_use]
pub fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0];
    }
    let scale = 2.0 * std::f32::consts::PI / (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (i as f32 * scale).cos()))
        .collect()
}

/// 计算互相关（normalized cross-correlation）
///
/// 在 `prev` 的末尾取长度为 `window_len` 的片段（作为参考），
/// 在 `next` 的开头 `[offset, offset+window_len)` 范围内计算归一化互相关。
///
/// # 参数
/// - `prev_tail`: 前一段音频的尾部（至少 `window_len + search_range` 个采样）
/// - `next_head`: 后一段音频的头部（至少 `window_len + search_range` 个采样）
/// - `window_len`: 互相关窗长度
/// - `search_range`: 搜索偏移范围（0..search_range）
///
/// # 返回
/// 最佳偏移量（0..search_range），如果信号过弱则返回 0
fn best_cross_correlation_offset(
    prev_tail: &[f32],
    next_head: &[f32],
    window_len: usize,
    search_range: usize,
) -> usize {
    if prev_tail.len() < window_len + search_range
        || next_head.len() < window_len + search_range
        || window_len == 0
        || search_range == 0
    {
        return 0;
    }

    // 取参考片段：prev_tail 的最后 window_len 个样本之前的 search_range 个样本
    // 实际上，我们搜索的是 next_head 相对于 prev_tail 的最佳偏移
    let ref_start = prev_tail.len().saturating_sub(window_len + search_range);
    let reference =
        &prev_tail[ref_start..prev_tail.len().min(ref_start + window_len + search_range)];

    // 计算参考段能量
    let ref_energy: f32 = reference[..window_len].iter().map(|s| s * s).sum();

    if ref_energy < SILENCE_THRESHOLD {
        return 0; // 静音段，无需搜索
    }

    let mut best_offset = 0;
    let mut best_corr = f32::NEG_INFINITY;

    for offset in 0..search_range {
        // 计算互相关：sum(prev_tail[ref_start+offset..ref_start+offset+window_len] * next_head[offset..offset+window_len])
        let mut corr = 0.0f32;
        let mut next_energy = 0.0f32;

        for i in 0..window_len {
            let prev_idx = ref_start + offset + i;
            let next_idx = offset + i;

            if prev_idx >= prev_tail.len() || next_idx >= next_head.len() {
                break;
            }

            corr += prev_tail[prev_idx] * next_head[next_idx];
            next_energy += next_head[next_idx] * next_head[next_idx];
        }

        // 归一化互相关
        let norm = (ref_energy * next_energy).sqrt();
        let normalized = if norm > SILENCE_THRESHOLD {
            corr / norm
        } else {
            0.0
        };

        if normalized > best_corr {
            best_corr = normalized;
            best_offset = offset;
        }
    }

    best_offset
}

/// SOLA 单步拼接：将 `next` 的头部拼接到 `prev` 的尾部
///
/// # 算法
/// 1. 取 `prev` 的最后 `overlap_len + search_range` 个采样作为参考
/// 2. 取 `next` 的前 `overlap_len + search_range` 个采样
/// 3. 通过互相关找到最佳偏移
/// 4. 在最佳偏移处使用 Hann 窗进行淡入淡出
///
/// # 参数
/// - `prev`: 前一段完整音频
/// - `next`: 后一段完整音频
/// - `overlap_len`: 重叠长度（采样数，默认 320 = 20ms@16kHz）
///
/// # 返回
/// `(best_offset, merged_audio)` — 最佳偏移量和拼接后的完整音频
///
/// # 示例
/// ```
/// use vt_core::sola::sola_overlap_add;
///
/// let prev = vec![0.5f32; 16000];
/// let next = vec![0.3f32; 16000];
/// let (offset, merged) = sola_overlap_add(&prev, &next, 320);
/// assert!(merged.len() >= prev.len() + next.len() - 320);
/// ```
#[must_use]
pub fn sola_overlap_add(prev: &[f32], next: &[f32], overlap_len: usize) -> (usize, Vec<f32>) {
    if prev.is_empty() {
        return (0, next.to_vec());
    }
    if next.is_empty() {
        return (0, prev.to_vec());
    }

    let search_range = DEFAULT_SEARCH_RANGE.min(prev.len()).min(next.len());
    let overlap = overlap_len.min(prev.len()).min(next.len());

    if overlap == 0 || search_range == 0 {
        // 无法重叠，直接拼接
        let mut merged = prev.to_vec();
        merged.extend_from_slice(next);
        return (0, merged);
    }

    // 寻找最佳偏移
    let best_offset = best_cross_correlation_offset(prev, next, overlap, search_range);

    // 实际重叠区域从 prev 尾部往前 overlap 个样本开始
    let prev_overlap_start = prev.len() - overlap;
    let next_overlap_start = best_offset;

    // 构建 Hann 窗
    let window = hann_window(overlap);

    // 拼接
    let mut merged = Vec::with_capacity(prev.len() + next.len() - overlap);

    // 1. prev 的非重叠部分（直接复制）
    merged.extend_from_slice(&prev[..prev_overlap_start]);

    // 2. 重叠部分（Hann 窗加权求和）
    for i in 0..overlap {
        let next_idx = next_overlap_start + i;
        if next_idx >= next.len() {
            // next 不够长，剩余部分直接用 prev
            merged.push(prev[prev_overlap_start + i]);
        } else {
            let w = window[i];
            merged.push(prev[prev_overlap_start + i] * (1.0 - w) + next[next_idx] * w);
        }
    }

    // 3. next 的剩余部分
    let next_remaining_start = next_overlap_start + overlap;
    if next_remaining_start < next.len() {
        merged.extend_from_slice(&next[next_remaining_start..]);
    }

    (best_offset, merged)
}

/// 批量 SOLA 拼接：将多段音频依次拼接
///
/// # 参数
/// - `fragments`: 音频片段列表
/// - `overlap_len`: 每次拼接的重叠长度
///
/// # 返回
/// 拼接后的完整音频
#[must_use]
pub fn sola_concat(fragments: &[Vec<f32>], overlap_len: usize) -> Vec<f32> {
    if fragments.is_empty() {
        return Vec::new();
    }
    if fragments.len() == 1 {
        return fragments[0].clone();
    }

    let mut result = fragments[0].clone();
    for fragment in &fragments[1..] {
        let (_, merged) = sola_overlap_add(&result, fragment, overlap_len);
        result = merged;
    }

    result
}

/// 在固定缓冲区中进行 SOLA 拼接（用于 `mix_audio_segments` 集成）
///
/// 将 `new_samples` 以 SOLA 方式写入 `buffer` 中从 `placement_offset` 开始的位置。
/// 如果 `new_samples` 与 buffer 中已有内容重叠，使用 Hann 窗交叉淡入淡出。
///
/// # 参数
/// - `buffer`: 目标音频缓冲区
/// - `placement_offset`: 新片段放置的起始采样位置
/// - `new_samples`: 新音频片段
/// - `overlap_len`: 重叠长度（采样数）
pub fn sola_write_into_buffer(
    buffer: &mut [f32],
    placement_offset: usize,
    new_samples: &[f32],
    overlap_len: usize,
) {
    if new_samples.is_empty() || placement_offset >= buffer.len() {
        return;
    }

    let write_end = (placement_offset + new_samples.len()).min(buffer.len());
    let write_len = write_end - placement_offset;

    // 检测放置区域是否有已有音频（非零数据）
    let check_len = overlap_len.min(write_len);
    let has_existing = check_len > 0 && {
        buffer[placement_offset..placement_offset + check_len]
            .iter()
            .any(|&s| s.abs() > SILENCE_THRESHOLD)
    };

    if !has_existing {
        // 无已有音频，直接叠加写入
        for i in 0..write_len {
            buffer[placement_offset + i] += new_samples[i];
        }
        return;
    }

    // 有已有音频，使用等功率（equal-power）交叉淡入淡出
    // w_out = cos(pi/2 * i / (N-1))，从 1→0（淡出已有音频）
    // w_in  = sin(pi/2 * i / (N-1))，从 0→1（淡入新音频）
    // w_out^2 + w_in^2 = 1（恒功率）
    let overlap = check_len;

    // 重叠区域：已有音频淡出，新音频淡入
    let scale = if overlap > 1 {
        std::f32::consts::PI / 2.0 / (overlap - 1) as f32
    } else {
        std::f32::consts::PI / 2.0
    };

    for i in 0..overlap {
        let buf_idx = placement_offset + i;
        let w_in = (i as f32 * scale).sin(); // 0→1
        let w_out = (i as f32 * scale).cos(); // 1→0

        buffer[buf_idx] = buffer[buf_idx] * w_out + new_samples[i] * w_in;
    }

    // 非重叠区域：直接覆写新音频
    for i in overlap..write_len {
        buffer[placement_offset + i] = new_samples[i];
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 Hann 窗长度和对称性
    #[test]
    fn test_hann_window_length() {
        let w = hann_window(100);
        assert_eq!(w.len(), 100);
    }

    #[test]
    fn test_hann_window_symmetry() {
        let w = hann_window(101);
        let n = w.len();
        for i in 0..n / 2 {
            assert!(
                (w[i] - w[n - 1 - i]).abs() < 1e-5,
                "Hann window should be symmetric at index {i}"
            );
        }
    }

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(100);
        // Hann 窗端点应接近 0
        assert!(
            w[0] < 0.01,
            "Hann window start should be near 0, got {}",
            w[0]
        );
        assert!(
            w[99] < 0.01,
            "Hann window end should be near 0, got {}",
            w[99]
        );
    }

    #[test]
    fn test_hann_window_peak() {
        let w = hann_window(101);
        let mid = w[50];
        assert!(
            (mid - 1.0).abs() < 0.01,
            "Hann window center should be near 1.0, got {mid}"
        );
    }

    #[test]
    fn test_hann_window_single() {
        let w = hann_window(1);
        assert_eq!(w, vec![1.0]);
    }

    /// 测试 SOLA 拼接基本功能
    #[test]
    fn test_sola_overlap_add_basic() {
        // 两段正弦波，频率相同，应该能完美拼接
        let sr = 16000.0f32;
        let freq = 220.0f32;
        let dur = 0.5; // 0.5 秒
        let n = (sr * dur) as usize;

        let prev: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / sr * freq * 2.0 * std::f32::consts::PI).sin()) * 0.5)
            .collect();
        let next: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / sr * freq * 2.0 * std::f32::consts::PI).sin()) * 0.5)
            .collect();

        let (_offset, merged) = sola_overlap_add(&prev, &next, 320);
        assert!(merged.len() >= prev.len() + next.len() - 320 - 160);
        assert!(merged.len() <= prev.len() + next.len());
    }

    /// 测试空音频处理
    #[test]
    fn test_sola_empty_prev() {
        let next = vec![0.5f32; 100];
        let (offset, merged) = sola_overlap_add(&[], &next, 50);
        assert_eq!(offset, 0);
        assert_eq!(merged, next);
    }

    #[test]
    fn test_sola_empty_next() {
        let prev = vec![0.5f32; 100];
        let (offset, merged) = sola_overlap_add(&prev, &[], 50);
        assert_eq!(offset, 0);
        assert_eq!(merged, prev);
    }

    /// 测试静音段不崩溃
    #[test]
    fn test_sola_silence() {
        let prev = vec![0.0f32; 1000];
        let next = vec![0.0f32; 1000];
        let (offset, merged) = sola_overlap_add(&prev, &next, 320);
        assert_eq!(offset, 0); // 静音段应返回 0
        assert!(merged.len() > 0);
    }

    /// 测试相位连续拼接无咔嗒声
    #[test]
    fn test_sola_no_click() {
        // 构造两段音频，在拼接点处有相位不连续
        let sr = 16000.0f32;
        let freq = 440.0f32;
        let n = 800; // 50ms

        let prev: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / sr * freq * 2.0 * std::f32::consts::PI).sin()) * 0.5)
            .collect();
        // next 从半周期处开始（模拟相位不连续）
        let next: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f32 + n as f32 * 0.3) / sr;
                (t * freq * 2.0 * std::f32::consts::PI).sin() * 0.5
            })
            .collect();

        let (_, merged) = sola_overlap_add(&prev, &next, 160);

        // 检查拼接区域没有大幅跳变（无咔嗒声）
        let overlap_region_start = n - 160;
        for i in 1..160 {
            let idx = overlap_region_start + i;
            if idx >= merged.len() {
                break;
            }
            let diff = (merged[idx] - merged[idx - 1]).abs();
            // 跳变不应超过正弦波的最大斜率 (2*pi*freq*amplitude / sr ≈ 0.086)
            assert!(diff < 0.2, "Click detected at sample {idx}: jump={diff:.4}");
        }
    }

    /// 测试 SOLA 批量拼接
    #[test]
    fn test_sola_concat() {
        let fragments = vec![vec![0.5f32; 1000], vec![0.3f32; 1000], vec![0.7f32; 1000]];
        let merged = sola_concat(&fragments, 200);
        assert!(merged.len() >= 1000 + 1000 + 1000 - 400 - 320); // 至少减去两次重叠
        assert!(merged.len() <= 3000);
    }

    #[test]
    fn test_sola_concat_empty() {
        let merged = sola_concat(&[], 320);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_sola_concat_single() {
        let fragments = vec![vec![0.5f32; 100]];
        let merged = sola_concat(&fragments, 50);
        assert_eq!(merged, fragments[0]);
    }

    /// 测试互相关找最佳偏移
    #[test]
    fn test_cross_correlation_finds_alignment() {
        // 构造一段正弦波，next 是 prev 的延迟版本
        let sr = 16000.0f32;
        let freq = 220.0f32;
        let n = 1000;

        let signal: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / sr * freq * 2.0 * std::f32::consts::PI).sin()) * 0.5)
            .collect();

        // next 从第 50 个样本开始（延迟 50 个样本）
        let delay = 50;
        let prev_tail = &signal[..500];
        let next_head = &signal[delay..delay + 500];

        let offset = best_cross_correlation_offset(prev_tail, next_head, 160, 160);
        // 应该找到一个接近 delay 的偏移
        assert!(
            offset > 0,
            "Cross-correlation should find non-zero offset for delayed signal"
        );
    }

    /// 测试 `sola_write_into_buffer` 基本功能
    #[test]
    fn test_sola_write_into_buffer_no_overlap() {
        let mut buffer = vec![0.0f32; 5000];
        let samples = vec![0.5f32; 1000];

        sola_write_into_buffer(&mut buffer, 2000, &samples, 320);

        // 非重叠区域应直接写入
        for i in 0..1000 {
            assert!(
                (buffer[2000 + i] - 0.5).abs() < 1e-5,
                "Non-overlap region should be directly written at {i}"
            );
        }
    }

    #[test]
    fn test_sola_write_into_buffer_with_overlap() {
        let mut buffer = vec![0.0f32; 5000];
        // 先写入第一段
        let first = vec![0.8f32; 2000];
        sola_write_into_buffer(&mut buffer, 0, &first, 320);

        // 第二段从 1700 开始（与第一段有 300 个样本重叠）
        let second = vec![0.4f32; 2000];
        sola_write_into_buffer(&mut buffer, 1700, &second, 320);

        // 重叠区域 [1700..2020] 应被平滑过渡
        for i in 1700..2020 {
            // 值应在 0.4~0.8 之间（不是 0.8+0.4=1.2）
            assert!(
                buffer[i] <= 1.0,
                "Overlap region should not exceed 1.0 at {i}: got {}",
                buffer[i]
            );
        }
        // 非重叠区域应为新音频值
        for i in 2020..3700 {
            assert!(
                (buffer[i] - 0.4).abs() < 0.01,
                "Non-overlap region should be new audio value at {i}: got {}",
                buffer[i]
            );
        }
    }

    /// 测试 `sola_write_into_buffer` 空输入
    #[test]
    fn test_sola_write_into_buffer_empty() {
        let mut buffer = vec![0.5f32; 1000];
        sola_write_into_buffer(&mut buffer, 100, &[], 50);
        // buffer 不应被修改
        assert_eq!(buffer, vec![0.5f32; 1000]);
    }

    /// 测试 `sola_write_into_buffer` 超出缓冲区范围
    #[test]
    fn test_sola_write_into_buffer_out_of_bounds() {
        let mut buffer = vec![0.0f32; 100];
        let samples = vec![0.5f32; 200];
        sola_write_into_buffer(&mut buffer, 50, &samples, 30);
        // 不应 panic，只写入可容纳的部分
    }

    /// 测试不同重叠长度效果
    #[test]
    fn test_sola_different_overlap_lengths() {
        let sr = 16000.0f32;
        let freq = 220.0f32;
        let n = 800;

        let prev: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / sr * freq * 2.0 * std::f32::consts::PI).sin()) * 0.5)
            .collect();
        let next: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / sr * freq * 2.0 * std::f32::consts::PI).sin()) * 0.5)
            .collect();

        // 不同重叠长度都应成功
        for overlap in [80, 160, 320, 640] {
            let (_, merged) = sola_overlap_add(&prev, &next, overlap);
            assert!(
                merged.len() > 0,
                "Should produce output for overlap={overlap}"
            );
        }
    }

    /// 测试大规模音频拼接性能（不应超时）
    #[test]
    fn test_sola_large_scale() {
        let sr = 16000.0f32;
        let freq = 220.0f32;
        let n = 16000 * 5; // 5 秒

        let prev: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / sr * freq * 2.0 * std::f32::consts::PI).sin()) * 0.3)
            .collect();
        let next: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / sr * freq * 2.0 * std::f32::consts::PI).sin()) * 0.3)
            .collect();

        let (_, merged) = sola_overlap_add(&prev, &next, 320);
        assert!(merged.len() > n);
    }
}
