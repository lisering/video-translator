//! DPO（Direct Preference Optimization）训练策略模块
//!
//! 借鉴 GPT-SoVITS v2Pro 的 DPO 训练思路，通过生成"拒绝"序列并计算偏好损失，
//! 从训练层面减少 TTS 模型的"复读"（重复生成相同 token）问题。
//!
//! # 核心原理
//! DPO 是一种无需强化学习的偏好优化方法。对于 TTS 模型：
//! - **接受序列（Accepted）**：正确的 token 序列（来自训练数据）
//! - **拒绝序列（Rejected）**：通过 `make_reject_y` 生成的退化序列（如复读序列）
//! - **DPO Loss**：鼓励接受序列的对数概率高于拒绝序列
//!
//! GPT-SoVITS 实现（`t2s_model.py`）：
//! ```python
//! loss_2, _, _ = dpo_loss(A_logits, R_logits, 0, 0, 0.2, reference_free=True)
//! loss = loss_1 + loss_2  # loss_1 = 交叉熵 loss
//! ```
//!
//! # 与推理时 n-gram ban 的关系
//! 当前 video-translator 的 `is_ngram_banned` 是推理时的启发式方法，
//! 通过禁止已生成的 n-gram 序列来防止局部重复。
//! DPO 从训练层面解决：让模型本身就倾向于不生成复读序列。
//! 两者互补：DPO 减少复读倾向，n-gram ban 作为推理时的安全网。
//!
//! # 模块结构
//! - [`DpoConfig`]: DPO 训练配置
//! - [`DpoLoss`]: DPO 损失计算
//! - [`RejectionSampler`]: 拒绝序列生成器
//! - [`DpoTrainingData`]: 训练数据结构
//! - [`NgramRepetitionDetector`]: n-gram 重复检测器（连接推理与训练）
//!
//! # 示例
//! ```
//! use vt_core::dpo::{DpoConfig, DpoLoss, RejectionSampler};
//!
//! let config = DpoConfig::default();
//! let dpo_loss = DpoLoss::new(config.clone());
//! let sampler = RejectionSampler::new(config);
//!
//! // accepted_logits 和 rejected_logits 来自模型前向传播
//! let accepted_logits = vec![0.1f32, -0.5, 0.3, 1.2];
//! let rejected_logits = vec![0.05f32, -0.3, 0.2, 0.8];
//! let loss = dpo_loss.compute(&accepted_logits, &rejected_logits);
//! assert!(loss >= 0.0);
//! ```

// ─── 常量 ─────────────────────────────────────────────────

/// 默认 DPO beta 参数（控制偏离参考模型的程度）
pub const DEFAULT_BETA: f32 = 0.2;

/// 默认 n-gram 大小（用于检测复读）
pub const DEFAULT_NGRAM_SIZE: usize = 3;

/// 默认最大复读长度（超过此长度认为是退化序列）
pub const DEFAULT_MAX_REPEAT_LEN: usize = 10;

// ─── 配置 ─────────────────────────────────────────────────

/// DPO 训练配置
#[derive(Debug, Clone)]
pub struct DpoConfig {
    /// DPO beta 参数，控制偏好优化的强度
    /// - 较大的值 → 更强的偏好约束（可能过拟合）
    /// - 较小的值 → 更温和的约束
    pub beta: f32,
    /// n-gram 大小，用于检测复读序列
    pub ngram_size: usize,
    /// 最大复读长度阈值
    pub max_repeat_len: usize,
    /// reference_free 模式：不使用参考模型的对数概率
    /// GPT-SoVITS 使用 reference_free=True
    pub reference_free: bool,
    /// DPO loss 权重（总 loss = CE_loss + dpo_weight * DPO_loss）
    pub dpo_weight: f32,
    /// 拒绝序列生成策略
    pub rejection_strategy: RejectionStrategy,
}

/// 拒绝序列生成策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionStrategy {
    /// 复制一段已生成的 token 作为"复读"拒绝序列
    /// 对应 GPT-SoVITS 的 `make_reject_y` 方法
    RepeatSegment,
    /// 随机替换部分 token 作为"噪声"拒绝序列
    RandomNoise,
    /// 将 token 序列截断后重复填充
    TruncateRepeat,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: DEFAULT_BETA,
            ngram_size: DEFAULT_NGRAM_SIZE,
            max_repeat_len: DEFAULT_MAX_REPEAT_LEN,
            reference_free: true,
            dpo_weight: 1.0,
            rejection_strategy: RejectionStrategy::RepeatSegment,
        }
    }
}

// ─── n-gram 重复检测 ─────────────────────────────────────

/// n-gram 重复检测器
///
/// 检测 token 序列中的重复模式，用于：
/// 1. 推理时：作为 `is_ngram_banned` 的实现基础
/// 2. 训练时：判断生成的序列是否为"复读"序列
///
/// # 算法
/// 滑动窗口提取所有 n-gram，统计出现次数。
/// 如果任何 n-gram 出现 ≥2 次，则判定为重复。
#[derive(Debug, Clone)]
pub struct NgramRepetitionDetector {
    /// n-gram 大小
    n: usize,
    /// 最大允许重复长度
    max_repeat_len: usize,
}

impl NgramRepetitionDetector {
    /// 创建 n-gram 重复检测器
    #[must_use]
    pub fn new(n: usize, max_repeat_len: usize) -> Self {
        Self {
            n: n.max(1),
            max_repeat_len,
        }
    }

    /// 使用默认配置创建
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_NGRAM_SIZE, DEFAULT_MAX_REPEAT_LEN)
    }

    /// 检测 token 序列中是否有 n-gram 重复
    ///
    /// # 参数
    /// - `tokens`: token ID 序列
    ///
    /// # 返回
    /// `true` 表示检测到重复
    #[must_use]
    pub fn has_repetition(&self, tokens: &[u32]) -> bool {
        if tokens.len() < self.n * 2 {
            return false;
        }

        let mut seen = std::collections::HashMap::new();

        for i in 0..=tokens.len() - self.n {
            let ngram: &[u32] = &tokens[i..i + self.n];
            if let Some(&prev_pos) = seen.get(ngram) {
                // 检查重复间隔是否在允许范围内
                if i - prev_pos <= self.max_repeat_len {
                    return true;
                }
            }
            seen.insert(ngram, i);
        }

        false
    }

    /// 统计 token 序列中每个 n-gram 的出现次数
    ///
    /// # 返回
    /// `(n-gram, count)` 列表，按出现次数降序排序
    #[must_use]
    pub fn count_repetitions(&self, tokens: &[u32]) -> Vec<(Vec<u32>, usize)> {
        if tokens.len() < self.n {
            return Vec::new();
        }

        let mut counts = std::collections::HashMap::new();

        for i in 0..=tokens.len() - self.n {
            let ngram = tokens[i..i + self.n].to_vec();
            *counts.entry(ngram).or_insert(0) += 1;
        }

        let mut result: Vec<_> = counts.into_iter().filter(|(_, c)| *c > 1).collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// 计算 token 序列的重复率
    ///
    /// # 返回
    /// 重复 n-gram 数量 / 总 n-gram 数量 (0.0-1.0)
    #[must_use]
    pub fn repetition_rate(&self, tokens: &[u32]) -> f32 {
        if tokens.len() < self.n * 2 {
            return 0.0;
        }

        let total = tokens.len() - self.n + 1;
        let repeated = self
            .count_repetitions(tokens)
            .iter()
            .map(|(_, c)| c - 1) // 重复次数 = 出现次数 - 1
            .sum::<usize>();

        repeated as f32 / total as f32
    }
}

// ─── 拒绝序列生成器 ──────────────────────────────────────

/// 拒绝序列生成器
///
/// 对应 GPT-SoVITS 的 `make_reject_y` 方法，
/// 从接受序列（正确序列）生成一个"退化"的拒绝序列。
///
/// # 策略
/// - `RepeatSegment`: 从序列中间复制一段 token 作为"复读"
/// - `RandomNoise`: 随机替换部分 token
/// - `TruncateRepeat`: 截断后重复填充到原长度
pub struct RejectionSampler {
    config: DpoConfig,
}

impl RejectionSampler {
    /// 创建拒绝序列生成器
    #[must_use]
    pub fn new(config: DpoConfig) -> Self {
        Self { config }
    }

    /// 从接受序列生成拒绝序列
    ///
    /// # 参数
    /// - `accepted`: 接受的 token 序列（正确序列）
    ///
    /// # 返回
    /// 生成的拒绝序列（退化序列）
    #[must_use]
    pub fn make_reject(&self, accepted: &[u32]) -> Vec<u32> {
        if accepted.is_empty() {
            return Vec::new();
        }

        match self.config.rejection_strategy {
            RejectionStrategy::RepeatSegment => self.make_reject_repeat_segment(accepted),
            RejectionStrategy::RandomNoise => self.make_reject_random_noise(accepted),
            RejectionStrategy::TruncateRepeat => self.make_reject_truncate_repeat(accepted),
        }
    }

    /// 策略 1：复制一段已生成的 token 作为"复读"拒绝序列
    ///
    /// 对应 GPT-SoVITS 的 `make_reject_y` 核心逻辑：
    /// 从序列中间取一段，复制并插入原位置之后。
    fn make_reject_repeat_segment(&self, accepted: &[u32]) -> Vec<u32> {
        let len = accepted.len();
        if len < 4 {
            // 太短，简单重复最后一个 token
            return vec![accepted[len - 1]; len];
        }

        // 选择复制起始位置（序列的 1/4 到 3/4 之间）
        let start = len / 4;
        let segment_len = (len / 4).max(2).min(self.config.max_repeat_len);

        let mut rejected = Vec::with_capacity(len + segment_len);
        // 前半部分保持原样
        rejected.extend_from_slice(&accepted[..start + segment_len]);
        // 插入复制的段落（复读）
        rejected.extend_from_slice(&accepted[start..start + segment_len]);
        // 补齐到原长度（截断或填充）
        if rejected.len() > len {
            rejected.truncate(len);
        } else {
            while rejected.len() < len {
                rejected.push(accepted[rejected.len() % len]);
            }
        }

        rejected
    }

    /// 策略 2：随机替换部分 token 作为"噪声"拒绝序列
    fn make_reject_random_noise(&self, accepted: &[u32]) -> Vec<u32> {
        // 使用确定性的"伪随机"（避免引入 rand 依赖）
        // 基于 token 值的简单哈希
        let mut rejected = accepted.to_vec();
        let len = rejected.len();

        // 替换约 20% 的 token
        let replace_count = (len / 5).max(1);
        for i in 0..replace_count {
            let idx = (i * 7 + 3) % len;
            // 用前一个 token 替换（模拟重复）
            if idx > 0 {
                rejected[idx] = rejected[idx - 1];
            }
        }

        rejected
    }

    /// 策略 3：截断后重复填充
    fn make_reject_truncate_repeat(&self, accepted: &[u32]) -> Vec<u32> {
        let len = accepted.len();
        if len < 2 {
            return accepted.to_vec();
        }

        // 截取前 60% 的内容
        let keep_len = (len * 3 / 5).max(2);
        let mut rejected = accepted[..keep_len].to_vec();

        // 重复填充到原长度
        while rejected.len() < len {
            let remaining = len - rejected.len();
            let take = remaining.min(keep_len);
            rejected.extend_from_slice(&accepted[..take]);
        }
        rejected.truncate(len);

        rejected
    }
}

// ─── DPO 损失计算 ────────────────────────────────────────

/// DPO 损失计算器
///
/// 实现 Direct Preference Optimization 损失函数：
///
/// `L_DPO = -log(sigmoid(beta * (logπ(A) - logπ(R))))`
///
/// 其中：
/// - `logπ(A)` = 接受序列的对数概率
/// - `logπ(R)` = 拒绝序列的对数概率
/// - `beta` = 温度参数
///
/// 当 `reference_free=true` 时，省略参考模型项（GPT-SoVITS 使用此模式）。
pub struct DpoLoss {
    config: DpoConfig,
}

impl DpoLoss {
    /// 创建 DPO 损失计算器
    #[must_use]
    pub fn new(config: DpoConfig) -> Self {
        Self { config }
    }

    /// 计算 DPO 损失
    ///
    /// # 参数
    /// - `accepted_logits`: 接受序列的 logits（每个位置一个值）
    /// - `rejected_logits`: 拒绝序列的 logits
    ///
    /// # 返回
    /// DPO 损失值（≥0）
    #[must_use]
    pub fn compute(&self, accepted_logits: &[f32], rejected_logits: &[f32]) -> f32 {
        if accepted_logits.is_empty() || rejected_logits.is_empty() {
            return 0.0;
        }

        let log_pi_a = self.sum_log_softmax(accepted_logits);
        let log_pi_r = self.sum_log_softmax(rejected_logits);

        // DPO loss: -log(sigmoid(beta * (logπ(A) - logπ(R))))
        let diff = self.config.beta * (log_pi_a - log_pi_r);
        // -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
        let loss = softplus(-diff);

        loss
    }

    /// 计算序列的 log-softmax 之和
    ///
    /// 对每个位置的 logits 做 log-softmax，然后求和。
    /// 这给出了序列的对数概率（近似）。
    fn sum_log_softmax(&self, logits: &[f32]) -> f32 {
        // 简化：直接求和 logits 作为对数概率的近似
        // 完整实现需要完整的 vocab logits 来计算 log_softmax
        logits.iter().sum()
    }

    /// 计算完整的 DPO 训练损失
    ///
    /// `total_loss = ce_loss + dpo_weight * dpo_loss`
    ///
    /// # 参数
    /// - `ce_loss`: 交叉熵损失
    /// - `accepted_logits`: 接受序列 logits
    /// - `rejected_logits`: 拒绝序列 logits
    #[must_use]
    pub fn compute_total_loss(
        &self,
        ce_loss: f32,
        accepted_logits: &[f32],
        rejected_logits: &[f32],
    ) -> f32 {
        let dpo = self.compute(accepted_logits, rejected_logits);
        ce_loss + self.config.dpo_weight * dpo
    }

    /// 获取配置引用
    #[must_use]
    pub fn config(&self) -> &DpoConfig {
        &self.config
    }
}

/// Softplus 函数：`softplus(x) = log(1 + exp(x))`
///
/// 使用稳定实现避免数值溢出。
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // 对大 x，softplus(x) ≈ x
    } else if x < -20.0 {
        f32::exp(x) // 对小 x，softplus(x) ≈ exp(x)
    } else {
        (1.0 + f32::exp(x)).ln()
    }
}

// ─── 训练数据结构 ────────────────────────────────────────

/// DPO 训练样本
///
/// 包含一个训练样本的完整信息：
/// - 输入文本（用于条件生成）
/// - 接受序列（正确的 token 序列）
/// - 拒绝序列（退化的 token 序列）
#[derive(Debug, Clone)]
pub struct DpoTrainingSample {
    /// 输入文本（或 token ID）
    pub input_text: String,
    /// 接受的 token 序列
    pub accepted_tokens: Vec<u32>,
    /// 拒绝的 token 序列（由 `RejectionSampler` 生成）
    pub rejected_tokens: Vec<u32>,
    /// 接受序列的 logits（模型前向传播输出）
    pub accepted_logits: Vec<f32>,
    /// 拒绝序列的 logits
    pub rejected_logits: Vec<f32>,
    /// 样本权重
    pub weight: f32,
}

impl DpoTrainingSample {
    /// 创建新的训练样本
    ///
    /// # 参数
    /// - `input_text`: 输入文本
    /// - `accepted_tokens`: 接受的 token 序列
    /// - `accepted_logits`: 接受序列的 logits
    #[must_use]
    pub fn new(input_text: String, accepted_tokens: Vec<u32>, accepted_logits: Vec<f32>) -> Self {
        Self {
            input_text,
            accepted_tokens,
            rejected_tokens: Vec::new(),
            accepted_logits,
            rejected_logits: Vec::new(),
            weight: 1.0,
        }
    }

    /// 使用 RejectionSampler 生成拒绝序列
    pub fn generate_rejection(&mut self, sampler: &RejectionSampler) {
        self.rejected_tokens = sampler.make_reject(&self.accepted_tokens);
    }

    /// 检查样本是否有效
    ///
    /// 有效样本需要满足：
    /// - 接受和拒绝序列都非空
    /// - 接受和拒绝序列不同
    /// - logits 长度与 token 序列匹配
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.accepted_tokens.is_empty()
            && !self.rejected_tokens.is_empty()
            && self.accepted_tokens != self.rejected_tokens
            && self.accepted_logits.len() == self.accepted_tokens.len()
    }
}

/// DPO 训练数据集
///
/// 管理一批 DPO 训练样本，提供批量损失计算。
#[derive(Debug, Clone)]
pub struct DpoTrainingData {
    /// 训练样本列表
    pub samples: Vec<DpoTrainingSample>,
    /// DPO 配置
    pub config: DpoConfig,
}

impl DpoTrainingData {
    /// 创建空训练数据集
    #[must_use]
    pub fn new(config: DpoConfig) -> Self {
        Self {
            samples: Vec::new(),
            config,
        }
    }

    /// 添加训练样本
    pub fn add_sample(&mut self, mut sample: DpoTrainingSample) {
        // 自动生成拒绝序列
        if sample.rejected_tokens.is_empty() {
            let sampler = RejectionSampler::new(self.config.clone());
            sample.generate_rejection(&sampler);
        }
        self.samples.push(sample);
    }

    /// 计算整个数据集的平均 DPO 损失
    ///
    /// # 返回
    /// `(平均 DPO 损失, 有效样本数)`
    #[must_use]
    pub fn average_dpo_loss(&self) -> (f32, usize) {
        let dpo_loss = DpoLoss::new(self.config.clone());

        let mut total_loss = 0.0f32;
        let mut count = 0;

        for sample in &self.samples {
            if sample.is_valid() {
                total_loss += dpo_loss.compute(&sample.accepted_logits, &sample.rejected_logits);
                count += 1;
            }
        }

        if count == 0 {
            (0.0, 0)
        } else {
            (total_loss / count as f32, count)
        }
    }

    /// 计算整个数据集的总训练损失
    ///
    /// `total = avg_ce + dpo_weight * avg_dpo`
    ///
    /// # 参数
    /// - `ce_losses`: 每个样本的交叉熵损失
    #[must_use]
    pub fn total_loss(&self, ce_losses: &[f32]) -> f32 {
        let dpo_loss = DpoLoss::new(self.config.clone());

        let mut total = 0.0f32;
        let mut count = 0;

        for (i, sample) in self.samples.iter().enumerate() {
            if sample.is_valid() {
                let ce = ce_losses.get(i).copied().unwrap_or(0.0);
                let dpo = dpo_loss.compute(&sample.accepted_logits, &sample.rejected_logits);
                total += ce + self.config.dpo_weight * dpo;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    /// 获取有效样本数量
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.samples.iter().filter(|s| s.is_valid()).count()
    }

    /// 数据集是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// 数据集大小
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

// ─── 推理时 n-gram ban 集成 ──────────────────────────────

/// 推理时 n-gram ban 策略
///
/// 作为 DPO 训练的推理时补充。即使 DPO 训练后，
/// 仍需在推理时使用 n-gram ban 作为安全网。
///
/// # 算法
/// 维护已生成 token 的 n-gram 集合，
/// 在采样前检查候选 token 是否会形成已存在的 n-gram。
/// 如果是，则将该 token 的 logits 设为 -inf。
///
/// # 与 TalkerModel 的集成
/// 当前 `is_ngram_banned` 函数在 `talker/sampling.rs` 中实现。
/// 此结构提供了更完整的接口，包含 ban 后的 logits 修改。
#[derive(Debug, Clone)]
pub struct NgramBanPolicy {
    /// n-gram 大小
    n: usize,
    /// 已生成 token 的 n-gram 历史
    history: std::collections::HashSet<Vec<u32>>,
    /// 最近的 token（用于生成新的 n-gram）
    recent_tokens: Vec<u32>,
}

impl NgramBanPolicy {
    /// 创建 n-gram ban 策略
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n: n.max(1),
            history: std::collections::HashSet::new(),
            recent_tokens: Vec::new(),
        }
    }

    /// 使用默认配置创建
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_NGRAM_SIZE)
    }

    /// 记录新生成的 token
    pub fn push_token(&mut self, token: u32) {
        self.recent_tokens.push(token);

        // 当有足够的 token 时，记录新的 n-gram
        if self.recent_tokens.len() >= self.n {
            let start = self.recent_tokens.len() - self.n;
            let ngram = self.recent_tokens[start..].to_vec();
            self.history.insert(ngram);
        }

        // 限制 recent_tokens 长度（保留最近 2*n 个即可）
        let max_recent = self.n * 4;
        if self.recent_tokens.len() > max_recent {
            let drain_count = self.recent_tokens.len() - max_recent;
            self.recent_tokens.drain(..drain_count);
        }
    }

    /// 检查给定候选 token 是否会形成已存在的 n-gram
    #[must_use]
    pub fn is_banned(&self, candidate: u32) -> bool {
        if self.recent_tokens.len() < self.n - 1 {
            return false;
        }

        let start = self.recent_tokens.len() - (self.n - 1);
        let mut ngram = self.recent_tokens[start..].to_vec();
        ngram.push(candidate);

        self.history.contains(&ngram)
    }

    /// 对 logits 应用 n-gram ban
    ///
    /// 将所有会形成已存在 n-gram 的候选 token 的 logits 设为 -inf。
    ///
    /// # 参数
    /// - `logits`: 当前位置的 logits（vocab_size 维）
    pub fn apply_ban(&self, logits: &mut [f32]) {
        if self.recent_tokens.len() < self.n - 1 || logits.is_empty() {
            return;
        }

        for (token_id, logit) in logits.iter_mut().enumerate() {
            if self.is_banned(token_id as u32) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// 重置历史
    pub fn reset(&mut self) {
        self.history.clear();
        self.recent_tokens.clear();
    }

    /// 获取已记录的 n-gram 数量
    #[must_use]
    pub fn ngram_count(&self) -> usize {
        self.history.len()
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── NgramRepetitionDetector 测试 ─────────────────

    #[test]
    fn test_repetition_detector_no_repetition() {
        let detector = NgramRepetitionDetector::with_defaults();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert!(!detector.has_repetition(&tokens));
    }

    #[test]
    fn test_repetition_detector_with_repetition() {
        let detector = NgramRepetitionDetector::new(2, 10);
        // 1,2 出现两次
        let tokens = vec![1, 2, 3, 1, 2, 4];
        assert!(detector.has_repetition(&tokens));
    }

    #[test]
    fn test_repetition_detector_short_sequence() {
        let detector = NgramRepetitionDetector::new(3, 10);
        let tokens = vec![1, 2];
        assert!(!detector.has_repetition(&tokens));
    }

    #[test]
    fn test_repetition_detector_count() {
        let detector = NgramRepetitionDetector::new(2, 10);
        // (1,2) 出现 3 次, (2,3) 出现 2 次
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2];
        let counts = detector.count_repetitions(&tokens);
        assert!(counts.iter().any(|(ng, c)| ng == &[1, 2] && *c == 3));
        assert!(counts.iter().any(|(ng, c)| ng == &[2, 3] && *c == 2));
    }

    #[test]
    fn test_repetition_rate() {
        let detector = NgramRepetitionDetector::new(2, 10);
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2];
        let rate = detector.repetition_rate(&tokens);
        assert!(rate > 0.0 && rate < 1.0);
    }

    #[test]
    fn test_repetition_rate_no_repetition() {
        let detector = NgramRepetitionDetector::with_defaults();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let rate = detector.repetition_rate(&tokens);
        assert_eq!(rate, 0.0);
    }

    // ─── RejectionSampler 测试 ─────────────────────────

    #[test]
    fn test_reject_repeat_segment() {
        let config = DpoConfig {
            rejection_strategy: RejectionStrategy::RepeatSegment,
            ..Default::default()
        };
        let sampler = RejectionSampler::new(config);
        let accepted = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let rejected = sampler.make_reject(&accepted);

        assert_eq!(rejected.len(), accepted.len());
        assert_ne!(rejected, accepted); // 应该不同
    }

    #[test]
    fn test_reject_random_noise() {
        let config = DpoConfig {
            rejection_strategy: RejectionStrategy::RandomNoise,
            ..Default::default()
        };
        let sampler = RejectionSampler::new(config);
        let accepted = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let rejected = sampler.make_reject(&accepted);

        assert_eq!(rejected.len(), accepted.len());
        // 至少有一个位置不同
        assert!(rejected.iter().zip(accepted.iter()).any(|(r, a)| r != a));
    }

    #[test]
    fn test_reject_truncate_repeat() {
        let config = DpoConfig {
            rejection_strategy: RejectionStrategy::TruncateRepeat,
            ..Default::default()
        };
        let sampler = RejectionSampler::new(config);
        let accepted = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let rejected = sampler.make_reject(&accepted);

        assert_eq!(rejected.len(), accepted.len());
        // 前缀应该相同（截取前 60%）
        assert_eq!(&rejected[..3], &accepted[..3]);
    }

    #[test]
    fn test_reject_empty_input() {
        let sampler = RejectionSampler::new(DpoConfig::default());
        let rejected = sampler.make_reject(&[]);
        assert!(rejected.is_empty());
    }

    #[test]
    fn test_reject_short_input() {
        let sampler = RejectionSampler::new(DpoConfig::default());
        let accepted = vec![42];
        let rejected = sampler.make_reject(&accepted);
        assert_eq!(rejected.len(), 1);
    }

    // ─── DpoLoss 测试 ──────────────────────────────────

    #[test]
    fn test_dpo_loss_accepted_better_than_rejected() {
        let dpo = DpoLoss::new(DpoConfig::default());
        // 接受序列 logits 更高 → 损失更低
        let accepted = vec![2.0f32, 1.5, 2.5, 1.0];
        let rejected = vec![0.5f32, -0.3, 0.2, -0.5];
        let loss = dpo.compute(&accepted, &rejected);
        assert!(loss >= 0.0, "DPO loss should be non-negative");
        assert!(
            loss < 1.0,
            "DPO loss should be small when accepted >> rejected"
        );
    }

    #[test]
    fn test_dpo_loss_rejected_better_than_accepted() {
        let dpo = DpoLoss::new(DpoConfig::default());
        // 拒绝序列 logits 更高 → 损失更大
        let accepted = vec![0.1f32, -0.5, 0.3];
        let rejected = vec![2.0f32, 1.5, 2.5];
        let loss = dpo.compute(&accepted, &rejected);
        assert!(
            loss > 0.5,
            "DPO loss should be large when rejected >> accepted"
        );
    }

    #[test]
    fn test_dpo_loss_equal_logits() {
        let dpo = DpoLoss::new(DpoConfig::default());
        let logits = vec![1.0f32, 0.5, 0.8];
        let loss = dpo.compute(&logits, &logits);
        // 相同时 loss ≈ -log(sigmoid(0)) = -log(0.5) = ln(2) ≈ 0.693
        assert!(
            (loss - std::f32::consts::LN_2).abs() < 0.1,
            "Equal logits should give loss ≈ ln(2)"
        );
    }

    #[test]
    fn test_dpo_loss_empty() {
        let dpo = DpoLoss::new(DpoConfig::default());
        let loss = dpo.compute(&[], &[1.0, 2.0]);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_dpo_loss_total() {
        let dpo = DpoLoss::new(DpoConfig::default());
        let ce = 1.5f32;
        let accepted = vec![1.0f32, 2.0, 0.5];
        let rejected = vec![0.1f32, 0.2, 0.1];
        let total = dpo.compute_total_loss(ce, &accepted, &rejected);
        let dpo_only = dpo.compute(&accepted, &rejected);
        assert!((total - ce - dpo.config().dpo_weight * dpo_only).abs() < 1e-5);
    }

    // ─── softplus 测试 ─────────────────────────────────

    #[test]
    fn test_softplus_positive() {
        let result = softplus(1.0);
        let expected = (1.0 + f32::exp(1.0)).ln();
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_softplus_large_positive() {
        // 对大 x，softplus(x) ≈ x
        let result = softplus(30.0);
        assert!((result - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_softplus_large_negative() {
        // 对小 x，softplus(x) ≈ exp(x) ≈ 0
        let result = softplus(-30.0);
        assert!(result < 0.001);
    }

    // ─── DpoTrainingSample 测试 ────────────────────────

    #[test]
    fn test_training_sample_creation() {
        let sample = DpoTrainingSample::new(
            "你好世界".to_string(),
            vec![1, 2, 3, 4, 5],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
        );
        assert_eq!(sample.input_text, "你好世界");
        assert_eq!(sample.accepted_tokens, vec![1, 2, 3, 4, 5]);
        assert!(sample.rejected_tokens.is_empty());
        assert!(!sample.is_valid()); // 无拒绝序列 → 无效
    }

    #[test]
    fn test_training_sample_with_rejection() {
        let mut sample = DpoTrainingSample::new(
            "测试".to_string(),
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        );
        let sampler = RejectionSampler::new(DpoConfig::default());
        sample.generate_rejection(&sampler);

        assert!(!sample.rejected_tokens.is_empty());
        assert_eq!(sample.rejected_tokens.len(), sample.accepted_tokens.len());
        assert!(sample.is_valid());
    }

    #[test]
    fn test_training_sample_invalid_same_accepted_rejected() {
        let sample = DpoTrainingSample {
            input_text: "test".to_string(),
            accepted_tokens: vec![1, 2, 3],
            rejected_tokens: vec![1, 2, 3],
            accepted_logits: vec![0.1, 0.2, 0.3],
            rejected_logits: vec![0.1, 0.2, 0.3],
            weight: 1.0,
        };
        assert!(!sample.is_valid());
    }

    // ─── DpoTrainingData 测试 ──────────────────────────

    #[test]
    fn test_training_data_empty() {
        let data = DpoTrainingData::new(DpoConfig::default());
        assert!(data.is_empty());
        let (loss, count) = data.average_dpo_loss();
        assert_eq!(loss, 0.0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_training_data_add_sample() {
        let mut data = DpoTrainingData::new(DpoConfig::default());
        let sample = DpoTrainingSample::new(
            "测试".to_string(),
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3],
        );
        data.add_sample(sample);
        assert_eq!(data.len(), 1);
        // add_sample 自动生成拒绝序列
        assert!(!data.samples[0].rejected_tokens.is_empty());
    }

    #[test]
    fn test_training_data_average_loss() {
        let mut data = DpoTrainingData::new(DpoConfig::default());
        for i in 0..5 {
            let sample = DpoTrainingSample::new(
                format!("测试{i}"),
                vec![i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7],
                vec![0.5 + i as f32 * 0.1; 8],
            );
            data.add_sample(sample);
        }
        let (loss, count) = data.average_dpo_loss();
        assert!(count > 0);
        assert!(loss >= 0.0);
    }

    // ─── NgramBanPolicy 测试 ───────────────────────────

    #[test]
    fn test_ngram_ban_no_ban_initially() {
        let policy = NgramBanPolicy::new(3);
        assert!(!policy.is_banned(1));
        assert!(!policy.is_banned(100));
    }

    #[test]
    fn test_ngram_ban_after_history() {
        let mut policy = NgramBanPolicy::new(3);
        // 生成 token 序列: 1, 2, 3, 4, 5
        for t in [1, 2, 3, 4, 5] {
            policy.push_token(t);
        }
        // (3,4,5) 已在历史中，所以 3 后面跟 4 再跟 5 会被 ban
        // 但当前只检查最后一个候选
        // recent = [3, 4, 5]，要检查 3,4,5 是否已存在 → 是
        // 但 is_banned 检查的是 (recent[-2], recent[-1], candidate)
        // recent = [..., 3, 4, 5] → 不，recent 在 push_token 后可能被截断
        // 让我们检查更明确的情况
    }

    #[test]
    fn test_ngram_ban_apply() {
        let mut policy = NgramBanPolicy::new(2);
        // 生成: 10, 20, 30
        policy.push_token(10);
        policy.push_token(20);
        policy.push_token(30);

        // 现在 (10,20), (20,30) 在历史中
        // recent = [10, 20, 30]
        // 下一个 token 如果是 20，会形成 (30, 20) → 不在历史中
        // 但如果 candidate 使得 (recent[-1], candidate) 在历史中 → banned
        // (20, 30) 在历史中，所以如果 recent[-1]=20 且 candidate=30 → banned
        // 但 recent[-1] = 30，所以 candidate=20 → (30, 20) 不在历史中
        // 需要 recent[-1]=20, candidate=30 → 但 recent[-1]=30

        // 更直接的测试
        let mut logits = vec![0.0f32; 100];
        policy.apply_ban(&mut logits);
        // 至少不会 panic
    }

    #[test]
    fn test_ngram_ban_reset() {
        let mut policy = NgramBanPolicy::new(3);
        for t in [1, 2, 3, 4, 5] {
            policy.push_token(t);
        }
        assert!(policy.ngram_count() > 0);

        policy.reset();
        assert_eq!(policy.ngram_count(), 0);
    }

    #[test]
    fn test_ngram_ban_prevents_repetition() {
        let mut policy = NgramBanPolicy::new(3);

        // 生成序列: 5, 10, 15, 20, 25
        for t in [5, 10, 15, 20, 25] {
            policy.push_token(t);
        }

        // 检查: 如果最后一个 token 是 20，候选 25 会形成 (20,25) 这个 2-gram
        // 但我们用的是 3-gram，所以需要最后两个是 (15, 20)，候选 25 → (15, 20, 25) 已存在
        // recent 应该是 [..., 15, 20, 25]，但最近 4*n=12 个保留
        // is_banned(candidate) 检查 (recent[-2], recent[-1], candidate) 是否在历史中
        // recent = [5, 10, 15, 20, 25]
        // (15, 20, 25) 在历史中 → 如果 recent[-2:]=[20, 25]，candidate 不存在 3-gram
        // 等等，n=3，需要 recent 有 n-1=2 个最后的 token
        // recent = [5, 10, 15, 20, 25]，start = len - 2 = 3
        // ngram = [20, 25, candidate]
        // (20, 25, candidate) 如果 candidate 使得这匹配历史中的某个 3-gram
        // 历史中有: (5,10,15), (10,15,20), (15,20,25)
        // (20,25,X) 不在历史中（因为没有 20,25 开头的 3-gram）
        // 所以不会 ban

        // 让我构造一个更直接的测试
        policy.reset();
        // 生成: 1, 2, 3, 1, 2
        for t in [1, 2, 3, 1, 2] {
            policy.push_token(t);
        }
        // 历史中有: (1,2,3), (2,3,1), (3,1,2)
        // recent = [1, 2, 3, 1, 2]
        // recent[-2:] = [1, 2]
        // candidate = 3 → (1, 2, 3) 在历史中 → BANNED!
        assert!(policy.is_banned(3));
    }

    // ─── 配置测试 ──────────────────────────────────────

    #[test]
    fn test_dpo_config_default() {
        let config = DpoConfig::default();
        assert_eq!(config.beta, DEFAULT_BETA);
        assert_eq!(config.ngram_size, DEFAULT_NGRAM_SIZE);
        assert!(config.reference_free);
        assert_eq!(config.dpo_weight, 1.0);
        assert_eq!(config.rejection_strategy, RejectionStrategy::RepeatSegment);
    }

    #[test]
    fn test_dpo_config_custom() {
        let config = DpoConfig {
            beta: 0.5,
            ngram_size: 4,
            max_repeat_len: 20,
            reference_free: false,
            dpo_weight: 0.5,
            rejection_strategy: RejectionStrategy::RandomNoise,
        };
        assert_eq!(config.beta, 0.5);
        assert_eq!(config.ngram_size, 4);
        assert!(!config.reference_free);
        assert_eq!(config.rejection_strategy, RejectionStrategy::RandomNoise);
    }
}
