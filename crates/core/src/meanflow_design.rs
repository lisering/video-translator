//! MeanFlow 蒸馏设计文档与可行性分析
//!
//! 借鉴 dots.tts 的 MeanFlow 蒸馏技术，分析将 Qwen3-TTS 的多步解码
//! 蒸馏为单步解码的可行性，实现 ~5x 推理加速。
//!
//! # MeanFlow 核心思想
//!
//! MeanFlow 是一种流匹配（Flow Matching）蒸馏技术：
//! 1. **教师模型**：多步 CFM（Continuous Flow Matching），如 32 步 Euler ODE
//! 2. **学生模型**：单步预测流匹配的均值（mean of the flow）
//! 3. **训练**：学生模型学习直接预测教师模型多步采样的最终结果
//! 4. **推理**：学生模型一步生成，RTF 降低 ~5x
//!
//! # 适用性分析
//!
//! ## 当前架构
//! - Qwen3-TTS 使用 AR（自回归）token 生成 + neural decoder
//! - AR 生成 ~34ms/token（Metal F32），decoder ~4000ms（CPU）
//! - 总 RTF ~1.1x（接近实时）
//!
//! ## MeanFlow 适用场景
//! MeanFlow 适用于 **CFM-based vocoder**（如 BigVGAN + CFM），
//! 不直接适用于 AR token 生成。但可以用于 decoder 部分：
//! - 当前 decoder：VQ codebook → ConvNet → 24kHz audio
//! - MeanFlow decoder：VQ codebook → CFM (32步→1步) → BigVGAN → audio
//!
//! # 预期收益
//!
//! | 组件 | 当前 | MeanFlow 后 | 加速比 |
//! |------|------|-------------|--------|
//! | AR token 生成 | 2500ms | 2500ms | 1x（不受影响） |
//! | Decoder | 4000ms | 800ms | 5x |
//! | 总时间 | 6500ms | 3300ms | 2x |
//! | RTF | 1.1x | 0.55x | - |
//!
//! # 实现路线图
//!
//! 1. **Phase 1（2周）**：CFM decoder 架构实现（已在 P8 中设计）
//! 2. **Phase 2（4周）**：训练数据准备 + 教师模型训练
//! 3. **Phase 3（2周）**：MeanFlow 蒸馏训练
//! 4. **Phase 4（1周）**：推理集成 + 性能验证
//!
//! 总计 ~9 周，需要 GPU 训练资源（A100 × 1-2）

use serde::{Deserialize, Serialize};

// ─── MeanFlow 配置 ───────────────────────────────────────

/// MeanFlow 蒸馏配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanFlowConfig {
    /// 教师模型步数（如 32 步 Euler ODE）
    pub teacher_steps: usize,
    /// 学生模型步数（MeanFlow = 1 步）
    pub student_steps: usize,
    /// 流匹配时间采样策略
    pub time_sampling: TimeSampling,
    /// 蒸馏损失类型
    pub distillation_loss: DistillationLoss,
    /// 训练批次大小
    pub batch_size: usize,
    /// 学习率
    pub learning_rate: f64,
    /// 训练迭代次数
    pub num_iterations: usize,
    /// EMA 衰减率（用于教师模型权重 EMA）
    pub ema_decay: f64,
}

impl Default for MeanFlowConfig {
    fn default() -> Self {
        Self {
            teacher_steps: 32,
            student_steps: 1,
            time_sampling: TimeSampling::Uniform,
            distillation_loss: DistillationLoss::Mse,
            batch_size: 16,
            learning_rate: 1e-4,
            num_iterations: 100_000,
            ema_decay: 0.999,
        }
    }
}

/// 流匹配时间采样策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeSampling {
    /// 均匀采样 t ~ U(0, 1)
    Uniform,
    /// 多项式采样 t ~ Beta(a, b)，偏向边界
    Polynomial,
    /// Logit-normal 采样
    LogitNormal,
}

/// 蒸馏损失类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistillationLoss {
    /// 均方误差
    Mse,
    /// L1 损失
    L1,
    /// 感知损失（使用预训练的特征提取器）
    Perceptual,
    /// 组合损失（MSE + 感知）
    Combined,
}

// ─── 性能估算 ─────────────────────────────────────────────

/// MeanFlow 性能估算结果
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    /// 教师模型推理时间（ms）
    pub teacher_inference_ms: f64,
    /// 学生模型推理时间（ms）
    pub student_inference_ms: f64,
    /// 加速比
    pub speedup: f64,
    /// 教师模型 RTF
    pub teacher_rtf: f64,
    /// 学生模型 RTF
    pub student_rtf: f64,
    /// 估算的音频质量损失（1-10，10=无损）
    pub quality_score: f64,
}

/// 估算 MeanFlow 蒸馏后的性能
///
/// # 参数
/// - `current_decoder_ms`: 当前 decoder 推理时间（ms）
/// - `ar_generation_ms`: AR token 生成时间（ms）
/// - `audio_duration_secs`: 音频时长（秒）
/// - `teacher_steps`: 教师模型步数
#[must_use]
pub fn estimate_meanflow_performance(
    current_decoder_ms: f64,
    ar_generation_ms: f64,
    audio_duration_secs: f64,
    teacher_steps: usize,
) -> PerformanceEstimate {
    // 教师模型时间 = 当前 decoder 时间（因为当前 decoder 已经是多步的）
    let teacher_inference_ms = current_decoder_ms;

    // 学生模型时间 = 教师时间 / teacher_steps（单步 vs 多步）
    // 但不是完美的 1/teacher_steps，因为单步需要更大的网络
    let overhead = 1.5; // 单步网络稍大，有 50% 额外开销
    let student_inference_ms = teacher_inference_ms / teacher_steps as f64 * overhead;

    let speedup = teacher_inference_ms / student_inference_ms;

    let total_teacher_ms = ar_generation_ms + teacher_inference_ms;
    let total_student_ms = ar_generation_ms + student_inference_ms;

    let teacher_rtf = total_teacher_ms / 1000.0 / audio_duration_secs;
    let student_rtf = total_student_ms / 1000.0 / audio_duration_secs;

    // 质量评估：MeanFlow 通常有轻微质量损失
    // 步数越多，教师模型质量越高，学生模型质量也越高
    let quality_score = match teacher_steps {
        0..=8 => 7.0,
        9..=16 => 8.0,
        17..=32 => 8.5,
        33..=64 => 9.0,
        _ => 9.5,
    };

    PerformanceEstimate {
        teacher_inference_ms,
        student_inference_ms,
        speedup,
        teacher_rtf,
        student_rtf,
        quality_score,
    }
}

// ─── 蒸馏流水线设计 ─────────────────────────────────────

/// 蒸馏阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistillationPhase {
    /// 阶段 1：CFM 教师模型训练
    TeacherTraining,
    /// 阶段 2：数据生成（教师模型采样）
    DataGeneration,
    /// 阶段 3：MeanFlow 学生模型训练
    StudentTraining,
    /// 阶段 4：推理集成
    InferenceIntegration,
}

/// 蒸馏步骤描述
#[derive(Debug, Clone)]
pub struct DistillationStep {
    /// 阶段
    pub phase: DistillationPhase,
    /// 步骤名称
    pub name: String,
    /// 描述
    pub description: String,
    /// 预估时间（天）
    pub estimated_days: f64,
    /// 依赖项
    pub dependencies: Vec<String>,
}

/// 获取完整蒸馏路线图
#[must_use]
pub fn get_distillation_roadmap() -> Vec<DistillationStep> {
    vec![
        DistillationStep {
            phase: DistillationPhase::TeacherTraining,
            name: "CFM Decoder 实现".to_string(),
            description: "实现 CFM + BigVGAN decoder 架构（参考 P8 的 cfm_decoder.rs）".to_string(),
            estimated_days: 14.0,
            dependencies: vec![],
        },
        DistillationStep {
            phase: DistillationPhase::TeacherTraining,
            name: "教师模型训练".to_string(),
            description: "训练 32-step CFM decoder 作为教师模型，使用 LibriTTS + AISHELL-3 数据集"
                .to_string(),
            estimated_days: 28.0,
            dependencies: vec!["CFM Decoder 实现".to_string()],
        },
        DistillationStep {
            phase: DistillationPhase::DataGeneration,
            name: "蒸馏数据生成".to_string(),
            description: "使用教师模型对训练集进行 32-step 采样，生成 (code, audio) 对".to_string(),
            estimated_days: 7.0,
            dependencies: vec!["教师模型训练".to_string()],
        },
        DistillationStep {
            phase: DistillationPhase::StudentTraining,
            name: "MeanFlow 学生模型训练".to_string(),
            description: "训练单步 MeanFlow 学生模型，目标匹配教师模型的多步采样结果".to_string(),
            estimated_days: 14.0,
            dependencies: vec!["蒸馏数据生成".to_string()],
        },
        DistillationStep {
            phase: DistillationPhase::StudentTraining,
            name: "质量微调".to_string(),
            description: "使用感知损失微调学生模型，提升音质".to_string(),
            estimated_days: 7.0,
            dependencies: vec!["MeanFlow 学生模型训练".to_string()],
        },
        DistillationStep {
            phase: DistillationPhase::InferenceIntegration,
            name: "推理集成".to_string(),
            description: "将 MeanFlow decoder 集成到 Python TTS 服务端，替换当前 ConvNet decoder"
                .to_string(),
            estimated_days: 5.0,
            dependencies: vec!["质量微调".to_string()],
        },
        DistillationStep {
            phase: DistillationPhase::InferenceIntegration,
            name: "性能验证".to_string(),
            description: "端到端测试，验证 RTF 和音质".to_string(),
            estimated_days: 2.0,
            dependencies: vec!["推理集成".to_string()],
        },
    ]
}

/// 估算总蒸馏时间（天）
#[must_use]
pub fn estimate_total_distillation_days() -> f64 {
    get_distillation_roadmap()
        .iter()
        .map(|s| s.estimated_days)
        .sum()
}

// ─── 可行性评估 ─────────────────────────────────────────

/// 可行性评估结果
#[derive(Debug, Clone)]
pub struct FeasibilityAssessment {
    /// 技术可行性（1-10）
    pub technical_feasibility: f64,
    /// 资源需求（1-10，越高需求越大）
    pub resource_requirement: f64,
    /// 预期收益（1-10）
    pub expected_benefit: f64,
    /// 风险等级
    pub risk_level: RiskLevel,
    /// 推荐优先级
    pub priority: RecommendationPriority,
    /// 评估说明
    pub notes: Vec<String>,
}

/// 风险等级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    /// 低风险
    Low,
    /// 中等风险
    Medium,
    /// 高风险
    High,
}

/// 推荐优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationPriority {
    /// 不推荐
    NotRecommended,
    /// 低优先级
    Low,
    /// 中优先级
    Medium,
    /// 高优先级
    High,
}

/// 执行 MeanFlow 蒸馏可行性评估
#[must_use]
pub fn assess_feasibility(
    current_rtf: f64,
    target_rtf: f64,
    has_gpu: bool,
    training_data_hours: f64,
) -> FeasibilityAssessment {
    let mut notes = Vec::new();

    // 技术可行性
    let technical = if current_rtf > 1.0 {
        notes.push("当前 RTF > 1.0，MeanFlow 蒸馏有明显收益空间".to_string());
        8.0
    } else {
        notes.push("当前 RTF < 1.0，MeanFlow 收益有限".to_string());
        5.0
    };

    // 资源需求
    let resource = if has_gpu {
        notes.push("有 GPU 训练资源，满足蒸馏需求".to_string());
        5.0
    } else {
        notes.push("缺少 GPU 训练资源，蒸馏训练不现实".to_string());
        9.0
    };

    // 预期收益
    let benefit = if current_rtf > 1.0 && target_rtf < 0.6 {
        notes.push(format!("目标 RTF {} 可达，预期收益高", target_rtf));
        8.0
    } else if current_rtf > 1.0 {
        notes.push("有一定收益但目标 RTF 较高".to_string());
        6.0
    } else {
        notes.push("当前已接近实时，收益有限".to_string());
        3.0
    };

    // 训练数据
    if training_data_hours < 100.0 {
        notes.push(format!(
            "训练数据 {} 小时偏少，推荐 ≥ 100 小时",
            training_data_hours
        ));
    } else if training_data_hours < 500.0 {
        notes.push(format!("训练数据 {} 小时基本满足需求", training_data_hours));
    } else {
        notes.push(format!("训练数据 {} 小时充足", training_data_hours));
    }

    // 风险等级
    let risk = if !has_gpu || training_data_hours < 50.0 {
        RiskLevel::High
    } else if technical >= 7.0 && benefit >= 7.0 {
        RiskLevel::Low
    } else {
        RiskLevel::Medium
    };

    // 优先级
    let priority = match (benefit, resource, risk) {
        (b, r, RiskLevel::Low) if b >= 7.0 && r <= 6.0 => RecommendationPriority::High,
        (b, _, RiskLevel::Low) if b >= 5.0 => RecommendationPriority::Medium,
        (_, _, RiskLevel::Medium) => RecommendationPriority::Low,
        (_, _, RiskLevel::High) => RecommendationPriority::NotRecommended,
        _ => RecommendationPriority::Low,
    };

    FeasibilityAssessment {
        technical_feasibility: technical,
        resource_requirement: resource,
        expected_benefit: benefit,
        risk_level: risk,
        priority,
        notes,
    }
}

// ─── 蒸馏报告 ─────────────────────────────────────────────

/// 生成 MeanFlow 蒸馏可行性报告
#[must_use]
pub fn generate_report(
    current_rtf: f64,
    current_decoder_ms: f64,
    ar_generation_ms: f64,
    audio_duration_secs: f64,
    has_gpu: bool,
    training_data_hours: f64,
) -> String {
    let perf = estimate_meanflow_performance(
        current_decoder_ms,
        ar_generation_ms,
        audio_duration_secs,
        32,
    );

    let feasibility = assess_feasibility(current_rtf, 0.5, has_gpu, training_data_hours);

    let total_days = estimate_total_distillation_days();

    let roadmap = get_distillation_roadmap();

    let mut report = String::new();
    report.push_str("═══════════════════════════════════════════════\n");
    report.push_str("       MeanFlow 蒸馏可行性分析报告\n");
    report.push_str("═══════════════════════════════════════════════\n\n");

    report.push_str("── 当前性能 ──\n");
    report.push_str(&format!("  当前 RTF: {:.2}x\n", current_rtf));
    report.push_str(&format!("  AR 生成: {:.0}ms\n", ar_generation_ms));
    report.push_str(&format!("  Decoder: {:.0}ms\n", current_decoder_ms));
    report.push_str(&format!("  音频时长: {:.1}s\n\n", audio_duration_secs));

    report.push_str("── MeanFlow 预期性能 ──\n");
    report.push_str(&format!(
        "  教师模型 decoder: {:.0}ms (32 步)\n",
        perf.teacher_inference_ms
    ));
    report.push_str(&format!(
        "  学生模型 decoder: {:.0}ms (1 步)\n",
        perf.student_inference_ms
    ));
    report.push_str(&format!("  Decoder 加速比: {:.1}x\n", perf.speedup));
    report.push_str(&format!("  教师 RTF: {:.2}x\n", perf.teacher_rtf));
    report.push_str(&format!("  学生 RTF: {:.2}x\n", perf.student_rtf));
    report.push_str(&format!("  质量评分: {:.1}/10\n\n", perf.quality_score));

    report.push_str("── 可行性评估 ──\n");
    report.push_str(&format!(
        "  技术可行性: {:.1}/10\n",
        feasibility.technical_feasibility
    ));
    report.push_str(&format!(
        "  资源需求: {:.1}/10\n",
        feasibility.resource_requirement
    ));
    report.push_str(&format!(
        "  预期收益: {:.1}/10\n",
        feasibility.expected_benefit
    ));
    report.push_str(&format!("  风险等级: {:?}\n", feasibility.risk_level));
    report.push_str(&format!("  推荐优先级: {:?}\n\n", feasibility.priority));

    report.push_str("── 评估说明 ──\n");
    for note in &feasibility.notes {
        report.push_str(&format!("  • {}\n", note));
    }
    report.push_str(&format!(
        "\n  总计时间: {:.0} 天 ({:.1} 周)\n\n",
        total_days,
        total_days / 7.0
    ));

    report.push_str("── 蒸馏路线图 ──\n");
    for step in &roadmap {
        report.push_str(&format!(
            "  [{:?}] {} ({:.0}天)\n    {}\n    依赖: {:?}\n\n",
            step.phase, step.name, step.estimated_days, step.description, step.dependencies
        ));
    }

    report.push_str("═══════════════════════════════════════════════\n");
    report.push_str("  结论: ");

    match feasibility.priority {
        RecommendationPriority::High => {
            report.push_str("推荐实施 MeanFlow 蒸馏，预期 RTF 显著降低。\n");
        }
        RecommendationPriority::Medium => {
            report.push_str("可考虑实施，但需权衡资源投入与收益。\n");
        }
        RecommendationPriority::Low => {
            report.push_str("暂缓实施，优先优化其他组件。\n");
        }
        RecommendationPriority::NotRecommended => {
            report.push_str("当前条件不建议实施（缺少 GPU 或数据不足）。\n");
        }
    }
    report.push_str("═══════════════════════════════════════════════\n");

    report
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meanflow_config_default() {
        let config = MeanFlowConfig::default();
        assert_eq!(config.teacher_steps, 32);
        assert_eq!(config.student_steps, 1);
        assert_eq!(config.time_sampling, TimeSampling::Uniform);
    }

    #[test]
    fn test_estimate_meanflow_performance() {
        // 当前: decoder=4000ms, AR=2500ms, 5.5s audio, 32 teacher steps
        let perf = estimate_meanflow_performance(4000.0, 2500.0, 5.5, 32);

        // 教师模型 decoder = 4000ms
        assert!((perf.teacher_inference_ms - 4000.0).abs() < 0.1);

        // 学生模型 decoder = 4000 / 32 * 1.5 = 187.5ms
        assert!((perf.student_inference_ms - 187.5).abs() < 1.0);

        // 加速比 ≈ 21.3x
        assert!(perf.speedup > 20.0 && perf.speedup < 22.0);

        // 教师 RTF = (2500 + 4000) / 1000 / 5.5 ≈ 1.18
        assert!(perf.teacher_rtf > 1.0 && perf.teacher_rtf < 1.3);

        // 学生 RTF = (2500 + 187.5) / 1000 / 5.5 ≈ 0.49
        assert!(perf.student_rtf < 0.6);

        // 质量评分
        assert!(perf.quality_score >= 8.0);
    }

    #[test]
    fn test_estimate_meanflow_performance_no_speedup() {
        // 1 步教师 → 学生也是 1 步，但有 1.5x 额外开销
        // speedup = teacher / (teacher/1 * 1.5) = 1/1.5 ≈ 0.667
        let perf = estimate_meanflow_performance(1000.0, 500.0, 2.0, 1);
        assert!((perf.speedup - 0.667).abs() < 0.05);
    }

    #[test]
    fn test_distillation_roadmap() {
        let roadmap = get_distillation_roadmap();
        assert!(!roadmap.is_empty());
        assert!(roadmap.len() >= 5);

        // 第一步应该是 TeacherTraining
        assert_eq!(roadmap[0].phase, DistillationPhase::TeacherTraining);

        // 最后一步应该是 InferenceIntegration
        assert_eq!(
            roadmap.last().unwrap().phase,
            DistillationPhase::InferenceIntegration
        );
    }

    #[test]
    fn test_estimate_total_days() {
        let total = estimate_total_distillation_days();
        assert!(total > 50.0); // 至少 50 天
        assert!(total < 200.0); // 不超过 200 天
    }

    #[test]
    fn test_assess_feasibility_high_benefit() {
        // RTF > 1.0, 有 GPU, 数据充足
        let assessment = assess_feasibility(1.5, 0.5, true, 500.0);

        assert!(assessment.technical_feasibility >= 7.0);
        assert!(assessment.expected_benefit >= 7.0);
        assert_eq!(assessment.risk_level, RiskLevel::Low);
        assert_eq!(assessment.priority, RecommendationPriority::High);
    }

    #[test]
    fn test_assess_feasibility_no_gpu() {
        let assessment = assess_feasibility(1.5, 0.5, false, 500.0);

        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.priority, RecommendationPriority::NotRecommended);
    }

    #[test]
    fn test_assess_feasibility_low_rtf() {
        // 当前 RTF 已经 < 1.0，收益有限
        let assessment = assess_feasibility(0.8, 0.5, true, 200.0);

        assert!(assessment.expected_benefit < 7.0);
    }

    #[test]
    fn test_assess_feasibility_low_data() {
        let assessment = assess_feasibility(1.5, 0.5, true, 30.0);

        assert!(assessment.notes.iter().any(|n| n.contains("偏少")));
        assert_eq!(assessment.risk_level, RiskLevel::High);
    }

    #[test]
    fn test_generate_report() {
        let report = generate_report(1.1, 4000.0, 2500.0, 5.5, true, 200.0);

        assert!(report.contains("MeanFlow 蒸馏可行性分析报告"));
        assert!(report.contains("当前 RTF"));
        assert!(report.contains("预期性能"));
        assert!(report.contains("可行性评估"));
        assert!(report.contains("蒸馏路线图"));
        assert!(report.contains("结论"));
    }

    #[test]
    fn test_generate_report_no_gpu() {
        let report = generate_report(1.1, 4000.0, 2500.0, 5.5, false, 50.0);

        assert!(report.contains("不建议"));
    }

    #[test]
    fn test_time_sampling_serde() {
        let ts = TimeSampling::Uniform;
        let json = serde_json::to_string(&ts).unwrap();
        assert_eq!(json, "\"uniform\"");

        let ts2: TimeSampling = serde_json::from_str("\"polynomial\"").unwrap();
        assert_eq!(ts2, TimeSampling::Polynomial);
    }

    #[test]
    fn test_distillation_loss_serde() {
        let loss = DistillationLoss::Combined;
        let json = serde_json::to_string(&loss).unwrap();
        assert_eq!(json, "\"combined\"");
    }

    #[test]
    fn test_meanflow_config_serde() {
        let config = MeanFlowConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let config2: MeanFlowConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.teacher_steps, config2.teacher_steps);
        assert_eq!(config.student_steps, config2.student_steps);
    }

    #[test]
    fn test_risk_level_serde() {
        let risk = RiskLevel::High;
        let json = serde_json::to_string(&risk).unwrap();
        assert_eq!(json, "\"high\"");

        let risk2: RiskLevel = serde_json::from_str("\"low\"").unwrap();
        assert_eq!(risk2, RiskLevel::Low);
    }

    #[test]
    fn test_quality_score_increases_with_steps() {
        let perf_8 = estimate_meanflow_performance(4000.0, 2500.0, 5.5, 8);
        let perf_32 = estimate_meanflow_performance(4000.0, 2500.0, 5.5, 32);
        let perf_64 = estimate_meanflow_performance(4000.0, 2500.0, 5.5, 64);

        assert!(perf_8.quality_score <= perf_32.quality_score);
        assert!(perf_32.quality_score <= perf_64.quality_score);
    }
}
