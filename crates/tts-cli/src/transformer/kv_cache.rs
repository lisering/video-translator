//! KV Cache — 自回归生成的键值缓存

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// KV 缓存，用于自回归生成
///
/// 统一模式：每步 `Tensor::cat` 追加 K/V，通过 `filled` 计数器跟踪有效长度。
/// 支持虚拟回滚 (O(1))：`rollback()` 仅递减 `filled`，不修改张量，
/// 下次 `update()` 时通过 `narrow` 忽略过期条目。
///
/// 预分配模式额外提供 `max_seq_len` 溢出检测和自动重置。
#[derive(Clone)]
pub struct KVCache {
    /// K/V 张量 (可能包含已回滚的过期条目)
    keys: Option<Tensor>,
    values: Option<Tensor>,
    /// 预分配模式参数 (溢出检测)
    max_seq_len: Option<usize>,
    /// 有效序列长度 (回滚后可能 < keys.dim(2))
    filled: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            max_seq_len: None,
            filled: 0,
        }
    }

    /// 创建预分配 KV 缓存
    ///
    /// 当前 Candle 0.9.2 不支持 `Tensor::copy_from` 原地写入，
    /// 因此预分配模式仍使用 `Tensor::cat` 策略，但通过 `max_seq_len`
    /// 提供溢出检测和自动重置。
    pub fn new_preallocated(
        max_seq_len: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _dtype: DType,
        _device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            keys: None,
            values: None,
            max_seq_len: Some(max_seq_len),
            filled: 0,
        })
    }

    /// 更新缓存：追加新 K/V，返回合并后的完整 K/V
    ///
    /// 如果 `filled < keys.dim(2)` (虚拟回滚后)，先用 `narrow` 截取有效部分，
    /// 再 `cat` 新条目。`narrow` 是零拷贝视图，`cat` 内部处理非连续张量。
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_len = k.dim(2).unwrap_or(0);

        // 预分配模式溢出检测
        if let Some(max_len) = self.max_seq_len {
            if self.filled + new_len > max_len {
                tracing::warn!(
                    "KV cache overflow: filled={} + new={} > max={}, resetting",
                    self.filled,
                    new_len,
                    max_len
                );
                self.filled = 0;
            }
        }

        let (new_k, new_v) = match &self.keys {
            None => (k.clone(), v.clone()),
            Some(prev_k) => {
                let prev_v = self.values.as_ref().unwrap();
                let prev_k_dim = prev_k.dim(2).unwrap_or(0);
                // 虚拟回滚后: narrow 到有效部分 (零拷贝视图)
                let prev_k_eff = if self.filled < prev_k_dim {
                    prev_k.narrow(2, 0, self.filled)?
                } else {
                    prev_k.clone()
                };
                let prev_v_eff = if self.filled < prev_v.dim(2).unwrap_or(0) {
                    prev_v.narrow(2, 0, self.filled)?
                } else {
                    prev_v.clone()
                };
                let k_cat = Tensor::cat(&[&prev_k_eff, k], 2)?;
                let v_cat = Tensor::cat(&[&prev_v_eff, v], 2)?;
                (k_cat, v_cat)
            }
        };
        self.keys = Some(new_k.clone());
        self.values = Some(new_v.clone());
        self.filled += new_len;
        Ok((new_k, new_v))
    }

    /// 当前缓存有效长度
    pub fn len(&self) -> usize {
        self.filled
    }

    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// 是否使用预分配模式
    pub fn is_preallocated(&self) -> bool {
        self.max_seq_len.is_some()
    }

    /// 重置缓存
    pub fn reset(&mut self) {
        self.filled = 0;
        self.keys = None;
        self.values = None;
    }

    /// 虚拟回滚：移除最后 n 个 KV 条目 (O(1))
    ///
    /// 用于推测解码 (speculative decoding) 中被拒绝 token 的回滚。
    /// 仅递减 `filled` 计数器，不修改张量。
    /// 下次 `update()` 时通过 `narrow` 忽略过期条目。
    ///
    /// # 参数
    /// - `n`: 要移除的条目数（必须 <= 当前缓存长度）
    pub fn rollback(&mut self, n: usize) -> Result<bool> {
        if n == 0 {
            return Ok(false);
        }
        if self.filled < n {
            tracing::warn!(
                "KV cache rollback: requested {} but only {} entries, clearing all",
                n,
                self.filled
            );
            self.reset();
            return Ok(true);
        }
        self.filled -= n;
        Ok(true)
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 类型别名，便于和 CodePredictor 共用
pub type AnyKVCache = KVCache;

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_device() -> Device {
        Device::Cpu
    }

    // ─── KVCache 基本操作 ───

    #[test]
    fn test_kvcache_new_empty() {
        let cache = KVCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(!cache.is_preallocated());
    }

    #[test]
    fn test_kvcache_preallocated() {
        let cache = KVCache::new_preallocated(100, 4, 64, DType::F32, &cpu_device()).unwrap();
        assert!(cache.is_empty());
        assert!(cache.is_preallocated());
    }

    #[test]
    fn test_kvcache_update_and_len() {
        let device = cpu_device();
        let mut cache = KVCache::new();
        let k = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.len(), 3);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_kvcache_multiple_updates() {
        let device = cpu_device();
        let mut cache = KVCache::new();
        for _ in 0..5 {
            let k = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
            let v = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_kvcache_reset() {
        let device = cpu_device();
        let mut cache = KVCache::new();
        let k = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.len(), 3);
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // ─── KVCache 虚拟回滚 ───

    #[test]
    fn test_kvcache_rollback_zero() {
        let device = cpu_device();
        let mut cache = KVCache::new();
        let k = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        let rolled_back = cache.rollback(0).unwrap();
        assert!(!rolled_back);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_kvcache_rollback_partial() {
        let device = cpu_device();
        let mut cache = KVCache::new();
        let k = Tensor::zeros((1, 4, 5, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 5, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.len(), 5);
        cache.rollback(2).unwrap();
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_kvcache_rollback_all() {
        let device = cpu_device();
        let mut cache = KVCache::new();
        let k = Tensor::zeros((1, 4, 5, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 5, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        cache.rollback(5).unwrap();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kvcache_rollback_exceeds_filled() {
        let device = cpu_device();
        let mut cache = KVCache::new();
        let k = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        cache.rollback(10).unwrap();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kvcache_rollback_then_update() {
        let device = cpu_device();
        let mut cache = KVCache::new();
        // 初始 3 个
        let k = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 3, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        // 回滚 2 个
        cache.rollback(2).unwrap();
        assert_eq!(cache.len(), 1);
        // 追加 1 个
        let k2 = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
        let v2 = Tensor::zeros((1, 4, 1, 64), DType::F32, &device).unwrap();
        cache.update(&k2, &v2).unwrap();
        assert_eq!(cache.len(), 2);
    }

    // ─── KVCache 预分配模式溢出 ───

    #[test]
    fn test_kvcache_preallocated_overflow_reset() {
        let device = cpu_device();
        let mut cache = KVCache::new_preallocated(10, 4, 64, DType::F32, &device).unwrap();
        // 填满
        let k = Tensor::zeros((1, 4, 8, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 8, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.len(), 8);
        // 溢出: 8 + 5 > 10 → 自动重置
        let k2 = Tensor::zeros((1, 4, 5, 64), DType::F32, &device).unwrap();
        let v2 = Tensor::zeros((1, 4, 5, 64), DType::F32, &device).unwrap();
        cache.update(&k2, &v2).unwrap();
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_kvcache_default() {
        let cache = KVCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
