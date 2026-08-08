//! Vibe Coding 自动化测试框架
//!
//! 本模块实现了 7 层验证体系中的 P1-P3 层：
//!
//! - **P1-B: Spec-driven 工作流** — 规格驱动开发，先写 spec 再写代码
//! - **P2-A: Evaluator-Optimizer** — AI 生成 → AI 评估 → 迭代优化循环
//! - **P2-B: 自动测试生成** — 从函数签名自动生成测试用例
//! - **P3-A: 变异测试** — 注入变异，验证测试是否能捕获
//! - **P3-B: Replay Testing** — 录制 I/O，回放验证
//!
//! # 设计理念
//!
//! 在 Vibe Coding 中，AI 生成代码的速度远超人工审查的速度。
//! 本框架的目标是：**让 AI 生成的代码由 AI 测试**，
//! 人类只在最高层（架构决策、产品方向）介入。
//!
//! # 来源
//!
//! 借鉴 Anthropic 的 Evaluator-Optimizer 模式、
//! Meticulous AI 的 replay testing、
//! 以及 cargo-mutants 的变异测试理念。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ═══════════════════════════════════════════════════════════
//  P1-B: Spec-Driven 工作流
// ═══════════════════════════════════════════════════════════

/// 函数规格
///
/// 描述一个函数应该做什么，而非怎么做。
/// AI 生成代码前先写 spec，然后验证代码是否满足 spec。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionSpec {
    /// 函数名
    pub name: String,
    /// 函数描述（自然语言）
    pub description: String,
    /// 输入参数描述
    pub inputs: Vec<ParameterSpec>,
    /// 输出描述
    pub output: TypeSpec,
    /// 前置条件（输入必须满足的条件）
    pub preconditions: Vec<String>,
    /// 后置条件（输出必须满足的条件）
    pub postconditions: Vec<String>,
    /// 不变量（在执行过程中始终成立的条件）
    pub invariants: Vec<String>,
    /// 示例输入输出
    pub examples: Vec<ExampleCase>,
}

/// 参数规格
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParameterSpec {
    pub name: String,
    pub description: String,
    pub type_name: String,
    pub constraints: Vec<String>,
}

/// 类型规格
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TypeSpec {
    pub type_name: String,
    pub description: String,
}

/// 示例用例
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExampleCase {
    pub input: String,
    pub expected_output: String,
    pub description: String,
}

impl FunctionSpec {
    /// 创建新的函数规格
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            inputs: Vec::new(),
            output: TypeSpec {
                type_name: "()".to_string(),
                description: String::new(),
            },
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            invariants: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// 添加输入参数
    pub fn with_input(mut self, name: &str, type_name: &str, description: &str) -> Self {
        self.inputs.push(ParameterSpec {
            name: name.to_string(),
            description: description.to_string(),
            type_name: type_name.to_string(),
            constraints: Vec::new(),
        });
        self
    }

    /// 设置输出类型
    pub fn with_output(mut self, type_name: &str, description: &str) -> Self {
        self.output = TypeSpec {
            type_name: type_name.to_string(),
            description: description.to_string(),
        };
        self
    }

    /// 添加后置条件
    pub fn with_postcondition(mut self, condition: &str) -> Self {
        self.postconditions.push(condition.to_string());
        self
    }

    /// 添加不变量
    pub fn with_invariant(mut self, invariant: &str) -> Self {
        self.invariants.push(invariant.to_string());
        self
    }

    /// 添加示例
    pub fn with_example(mut self, input: &str, expected: &str, desc: &str) -> Self {
        self.examples.push(ExampleCase {
            input: input.to_string(),
            expected_output: expected.to_string(),
            description: desc.to_string(),
        });
        self
    }

    /// 验证 spec 的完整性
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Spec name is empty".to_string());
        }
        if self.description.is_empty() {
            return Err(format!("Spec '{}' has no description", self.name));
        }
        if self.postconditions.is_empty() && self.examples.is_empty() {
            return Err(format!(
                "Spec '{}' has no postconditions or examples — cannot verify",
                self.name
            ));
        }
        Ok(())
    }
}

/// Spec 注册表
///
/// 管理项目中所有函数的规格。
/// 可以从 TOML/JSON 文件加载，也可以在代码中构建。
pub struct SpecRegistry {
    specs: HashMap<String, FunctionSpec>,
}

impl SpecRegistry {
    pub fn new() -> Self {
        Self {
            specs: HashMap::new(),
        }
    }

    /// 注册函数规格
    pub fn register(&mut self, spec: FunctionSpec) -> Result<(), String> {
        spec.validate()?;
        self.specs.insert(spec.name.clone(), spec);
        Ok(())
    }

    /// 获取函数规格
    pub fn get(&self, name: &str) -> Option<&FunctionSpec> {
        self.specs.get(name)
    }

    /// 列出所有已注册的规格
    pub fn list(&self) -> Vec<&FunctionSpec> {
        self.specs.values().collect()
    }

    /// 从 JSON 文件加载规格
    pub fn load_from_json(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read spec file {path:?}: {e}"))?;
        let spec: FunctionSpec = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse spec JSON: {e}"))?;
        let mut registry = Self::new();
        registry.register(spec)?;
        Ok(registry)
    }
}

impl Default for SpecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════
//  P2-A: Evaluator-Optimizer 模式
// ═══════════════════════════════════════════════════════════

/// 评估结果
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// 分数 (0.0 ~ 1.0)
    pub score: f64,
    /// 评估详情
    pub feedback: Vec<String>,
    /// 是否通过（score >= threshold）
    pub passed: bool,
}

impl EvaluationResult {
    pub fn pass(score: f64, feedback: Vec<String>) -> Self {
        Self {
            score,
            feedback,
            passed: true,
        }
    }

    pub fn fail(score: f64, feedback: Vec<String>) -> Self {
        Self {
            score,
            feedback,
            passed: false,
        }
    }
}

/// 评估器 trait
///
/// 实现 `evaluate` 方法对输出进行评估。
/// 评估可以基于：
/// - 规则检查（正则匹配、字段验证）
/// - Golden Master 对比
/// - 属性测试
/// - LLM 评估（通过外部 API）
pub trait Evaluator {
    /// 评估类型
    type Input;
    type Output;

    /// 评估输出质量
    fn evaluate(&self, input: &Self::Input, output: &Self::Output) -> EvaluationResult;
}

/// Evaluator-Optimizer 循环
///
/// 生成 → 评估 → 如果不通过则优化 → 重复
pub struct EvalOptimizer<G, E>
where
    G: Generator,
    E: Evaluator<Input = G::Input, Output = G::Output>,
{
    generator: G,
    evaluator: E,
    max_iterations: usize,
    threshold: f64,
}

/// 生成器 trait
pub trait Generator {
    type Input;
    type Output;

    fn generate(&self, input: &Self::Input) -> Self::Output;
}

/// 优化器 trait
pub trait Optimizer: Generator {
    fn optimize(&self, input: &Self::Input, previous: &Self::Output, feedback: &[String]) -> Self::Output;
}

/// 循环结果
#[derive(Debug, Clone)]
pub struct LoopResult<O> {
    pub output: O,
    pub iterations: usize,
    pub final_score: f64,
    pub converged: bool,
    pub history: Vec<f64>,
}

impl<G, E> EvalOptimizer<G, E>
where
    G: Generator,
    E: Evaluator<Input = G::Input, Output = G::Output>,
{
    pub fn new(generator: G, evaluator: E) -> Self {
        Self {
            generator,
            evaluator,
            max_iterations: 5,
            threshold: 0.8,
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// 运行评估-优化循环
    pub fn run(&self, input: &G::Input) -> LoopResult<G::Output>
    where
        G::Output: Clone,
    {
        let mut output = self.generator.generate(input);
        let mut history = Vec::new();

        for i in 0..self.max_iterations {
            let result = self.evaluator.evaluate(input, &output);
            history.push(result.score);

            if result.score >= self.threshold {
                return LoopResult {
                    output,
                    iterations: i + 1,
                    final_score: result.score,
                    converged: true,
                    history,
                };
            }

            // 如果生成器也是优化器，使用反馈优化
            // (这里简化为重新生成)
            output = self.generator.generate(input);
        }

        LoopResult {
            output,
            iterations: self.max_iterations,
            final_score: *history.last().unwrap_or(&0.0),
            converged: false,
            history,
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  P2-B: 自动测试生成框架
// ═══════════════════════════════════════════════════════════

/// 测试用例
#[derive(Debug, Clone)]
pub struct GeneratedTestCase {
    pub name: String,
    pub description: String,
    pub input: String,
    pub expected_behavior: String,
    pub test_type: TestType,
}

/// 测试类型
#[derive(Debug, Clone, PartialEq)]
pub enum TestType {
    /// 正常路径
    HappyPath,
    /// 边界条件
    Boundary,
    /// 错误处理
    ErrorCase,
    /// 属性测试
    Property,
    /// 性能测试
    Performance,
}

/// 测试生成器
///
/// 从函数规格自动生成测试用例。
pub struct TestGenerator {
    /// 生成边界值的策略
    pub boundary_values: Vec<String>,
    /// 错误输入
    pub error_inputs: Vec<String>,
}

impl Default for TestGenerator {
    fn default() -> Self {
        Self {
            boundary_values: vec![
                "0".to_string(),
                "1".to_string(),
                "-1".to_string(),
                "empty".to_string(),
                "max_value".to_string(),
                "min_value".to_string(),
            ],
            error_inputs: vec![
                "null".to_string(),
                "empty_string".to_string(),
                "invalid_format".to_string(),
                "overflow".to_string(),
            ],
        }
    }
}

impl TestGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从函数规格生成测试用例
    pub fn generate_from_spec(&self, spec: &FunctionSpec) -> Vec<GeneratedTestCase> {
        let mut tests = Vec::new();

        // 1. 从示例生成 HappyPath 测试
        for (i, example) in spec.examples.iter().enumerate() {
            tests.push(GeneratedTestCase {
                name: format!("test_{}_example_{}", spec.name, i),
                description: example.description.clone(),
                input: example.input.clone(),
                expected_behavior: example.expected_output.clone(),
                test_type: TestType::HappyPath,
            });
        }

        // 2. 从后置条件生成属性测试
        for (i, postcond) in spec.postconditions.iter().enumerate() {
            tests.push(GeneratedTestCase {
                name: format!("test_{}_postcondition_{}", spec.name, i),
                description: format!("Verify: {postcond}"),
                input: "any valid input".to_string(),
                expected_behavior: postcond.clone(),
                test_type: TestType::Property,
            });
        }

        // 3. 从不变量生成属性测试
        for (i, invariant) in spec.invariants.iter().enumerate() {
            tests.push(GeneratedTestCase {
                name: format!("test_{}_invariant_{}", spec.name, i),
                description: format!("Invariant: {invariant}"),
                input: "any input".to_string(),
                expected_behavior: invariant.clone(),
                test_type: TestType::Property,
            });
        }

        // 4. 从边界值生成边界测试
        for (i, boundary) in self.boundary_values.iter().enumerate() {
            tests.push(GeneratedTestCase {
                name: format!("test_{}_boundary_{}", spec.name, i),
                description: format!("Boundary: {boundary}"),
                input: boundary.clone(),
                expected_behavior: "Should handle gracefully".to_string(),
                test_type: TestType::Boundary,
            });
        }

        // 5. 从错误输入生成错误测试
        for (i, error_input) in self.error_inputs.iter().enumerate() {
            tests.push(GeneratedTestCase {
                name: format!("test_{}_error_{}", spec.name, i),
                description: format!("Error case: {error_input}"),
                input: error_input.clone(),
                expected_behavior: "Should return error, not panic".to_string(),
                test_type: TestType::ErrorCase,
            });
        }

        tests
    }
}

// ═══════════════════════════════════════════════════════════
//  P3-A: 变异测试
// ═══════════════════════════════════════════════════════════

/// 变异类型
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MutationType {
    /// 将 `+` 替换为 `-`
    PlusToMinus,
    /// 将 `-` 替换为 `+`
    MinusToPlus,
    /// 将 `*` 替换为 `/`
    MultiplyToDivide,
    /// 将 `>` 替换为 `>=`
    GreaterToGreaterEqual,
    /// 将 `<` 替换为 `<=`
    LessToLessEqual,
    /// 将 `==` 替换为 `!=`
    EqualToNotEqual,
    /// 将 `true` 替换为 `false`
    TrueToFalse,
    /// 将 `false` 替换为 `true`
    FalseToTrue,
    /// 删除一行
    DeleteLine,
    /// 自定义变异
    Custom(String),
}

/// 变异
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mutation {
    /// 变异 ID
    pub id: String,
    /// 文件路径
    pub file: String,
    /// 行号
    pub line: usize,
    /// 变异类型
    pub mutation_type: MutationType,
    /// 原始代码
    pub original: String,
    /// 变异后代码
    pub mutated: String,
}

/// 变异测试结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MutationResult {
    pub mutation: Mutation,
    /// 测试是否捕获了变异
    pub killed: bool,
    /// 捕获变异的测试名
    pub killed_by: Option<String>,
}

/// 变异测试报告
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MutationReport {
    pub total_mutations: usize,
    pub killed: usize,
    pub survived: usize,
    pub mutation_score: f64,
    pub results: Vec<MutationResult>,
}

impl MutationReport {
    /// 计算变异分数
    pub fn from_results(results: Vec<MutationResult>) -> Self {
        let total = results.len();
        let killed = results.iter().filter(|r| r.killed).count();
        let survived = total - killed;
        let score = if total > 0 {
            killed as f64 / total as f64
        } else {
            0.0
        };

        Self {
            total_mutations: total,
            killed,
            survived,
            mutation_score: score,
            results,
        }
    }

    /// 生成报告摘要
    pub fn summary(&self) -> String {
        format!(
            "Mutation Score: {:.1}% ({}/{})\n  Killed: {}\n  Survived: {}",
            self.mutation_score * 100.0,
            self.killed,
            self.total_mutations,
            self.killed,
            self.survived
        )
    }
}

/// 变异生成器
///
/// 从源代码生成变异。这是一个设计框架，
/// 实际的变异注入需要集成 cargo-mutants 或自定义工具。
pub struct MutationGenerator {
    /// 要应用的变异类型
    pub enabled_mutations: Vec<MutationType>,
}

impl Default for MutationGenerator {
    fn default() -> Self {
        Self {
            enabled_mutations: vec![
                MutationType::PlusToMinus,
                MutationType::MinusToPlus,
                MutationType::GreaterToGreaterEqual,
                MutationType::LessToLessEqual,
                MutationType::EqualToNotEqual,
                MutationType::TrueToFalse,
                MutationType::FalseToTrue,
            ],
        }
    }
}

impl MutationGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从源代码中查找可变异的位置
    pub fn find_mutation_points(&self, source: &str, file: &str) -> Vec<Mutation> {
        let mut mutations = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let line_num = line_num + 1;

            for mutation_type in &self.enabled_mutations {
                if let Some(mutation) = self.try_mutate(line, line_num, file, mutation_type) {
                    mutations.push(mutation);
                }
            }
        }

        mutations
    }

    /// 尝试在行中应用变异
    fn try_mutate(
        &self,
        line: &str,
        line_num: usize,
        file: &str,
        mutation_type: &MutationType,
    ) -> Option<Mutation> {
        let (from, to) = match mutation_type {
            MutationType::PlusToMinus => (" + ", " - "),
            MutationType::MinusToPlus => (" - ", " + "),
            MutationType::GreaterToGreaterEqual => (">", ">="),
            MutationType::LessToLessEqual => ("<", "<="),
            MutationType::EqualToNotEqual => ("==", "!="),
            MutationType::TrueToFalse => ("true", "false"),
            MutationType::FalseToTrue => ("false", "true"),
            MutationType::MultiplyToDivide => (" * ", " / "),
            MutationType::DeleteLine => return None,
            MutationType::Custom(_) => return None,
        };

        if line.contains(from) && !line.contains("//") {
            // 跳过注释行
            let mutated = line.replacen(from, to, 1);
            Some(Mutation {
                id: format!("{file}:{line_num}:{:?}", mutation_type),
                file: file.to_string(),
                line: line_num,
                mutation_type: mutation_type.clone(),
                original: line.to_string(),
                mutated,
            })
        } else {
            None
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  P3-B: Replay Testing
// ═══════════════════════════════════════════════════════════

/// 录制的 I/O 操作
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RecordedCall {
    /// 函数名
    pub function: String,
    /// 输入参数（JSON 序列化）
    pub input: String,
    /// 输出结果（JSON 序列化）
    pub output: String,
    /// 时间戳
    pub timestamp: u64,
}

/// 录制器
///
/// 记录函数调用的输入和输出，用于后续回放验证。
pub struct Recorder {
    calls: Vec<RecordedCall>,
    /// 录制文件路径
    save_path: Option<PathBuf>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            calls: Vec::new(),
            save_path: None,
        }
    }

    pub fn with_save_path(mut self, path: impl AsRef<Path>) -> Self {
        self.save_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// 录制一次函数调用
    pub fn record(
        &mut self,
        function: &str,
        input: &impl serde::Serialize,
        output: &impl serde::Serialize,
    ) {
        let call = RecordedCall {
            function: function.to_string(),
            input: serde_json::to_string(input).unwrap_or_default(),
            output: serde_json::to_string(output).unwrap_or_default(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        self.calls.push(call);
    }

    /// 保存录制到文件
    pub fn save(&self) -> Result<(), String> {
        if let Some(path) = &self.save_path {
            let json = serde_json::to_string_pretty(&self.calls)
                .map_err(|e| format!("Failed to serialize: {e}"))?;
            std::fs::write(path, json)
                .map_err(|e| format!("Failed to write: {e}"))?;
        }
        Ok(())
    }

    /// 获取所有录制的调用
    pub fn calls(&self) -> &[RecordedCall] {
        &self.calls
    }

    /// 清空录制
    pub fn clear(&mut self) {
        self.calls.clear();
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

/// 回放器
///
/// 从录制文件加载 I/O 记录，验证当前实现是否产生相同输出。
pub struct Player {
    recorded_calls: Vec<RecordedCall>,
    current_index: usize,
}

impl Player {
    /// 从文件加载录制
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read recording: {e}"))?;
        let calls: Vec<RecordedCall> = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse recording: {e}"))?;
        Ok(Self {
            recorded_calls: calls,
            current_index: 0,
        })
    }

    /// 从录制器创建回放器
    pub fn from_recorder(recorder: &Recorder) -> Self {
        Self {
            recorded_calls: recorder.calls().to_vec(),
            current_index: 0,
        }
    }

    /// 获取下一个预期的调用
    pub fn next_expected(&mut self) -> Option<&RecordedCall> {
        let call = self.recorded_calls.get(self.current_index)?;
        self.current_index += 1;
        Some(call)
    }

    /// 验证当前实现的输出是否匹配录制
    pub fn verify(
        &mut self,
        function: &str,
        input: &impl serde::Serialize,
        output: &impl serde::Serialize,
    ) -> Result<(), String> {
        let expected = self.next_expected().ok_or("No more recorded calls")?;

        if expected.function != function {
            return Err(format!(
                "Function mismatch: expected '{}', got '{}'",
                expected.function, function
            ));
        }

        let actual_input = serde_json::to_string(input).map_err(|e| e.to_string())?;
        let actual_output = serde_json::to_string(output).map_err(|e| e.to_string())?;

        if expected.input != actual_input {
            return Err(format!(
                "Input mismatch for {function}:\n  expected: {}\n  actual: {}",
                expected.input, actual_input
            ));
        }

        if expected.output != actual_output {
            return Err(format!(
                "Output mismatch for {function}:\n  expected: {}\n  actual: {}",
                expected.output, actual_output
            ));
        }

        Ok(())
    }

    /// 获取录制中的调用数量
    pub fn len(&self) -> usize {
        self.recorded_calls.len()
    }

    /// 是否还有更多录制
    pub fn has_more(&self) -> bool {
        self.current_index < self.recorded_calls.len()
    }

    /// 重置到开头
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
}

// ═══════════════════════════════════════════════════════════
//  单元测试
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── P1-B: Spec-driven ──

    #[test]
    fn test_spec_creation() {
        let spec = FunctionSpec::new("add", "Add two numbers")
            .with_input("a", "i32", "First number")
            .with_input("b", "i32", "Second number")
            .with_output("i32", "Sum of a and b")
            .with_postcondition("result == a + b")
            .with_invariant("No overflow")
            .with_example("1, 2", "3", "Simple addition");

        assert_eq!(spec.name, "add");
        assert_eq!(spec.inputs.len(), 2);
        assert!(!spec.postconditions.is_empty());
        assert!(!spec.examples.is_empty());
        spec.validate().expect("Spec should be valid");
    }

    #[test]
    fn test_spec_validation_fails_empty() {
        let spec = FunctionSpec::new("", "");
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_spec_registry() {
        let mut registry = SpecRegistry::new();
        let spec = FunctionSpec::new("test_fn", "Test function")
            .with_postcondition("Always returns true");
        registry.register(spec).expect("register failed");

        assert!(registry.get("test_fn").is_some());
        assert!(registry.get("nonexistent").is_none());
        assert_eq!(registry.list().len(), 1);
    }

    // ── P2-B: Test Generator ──

    #[test]
    fn test_generator_from_spec() {
        let spec = FunctionSpec::new("reverse", "Reverse a string")
            .with_input("s", "&str", "Input string")
            .with_output("String", "Reversed string")
            .with_postcondition("result.len() == s.len()")
            .with_invariant("No allocation beyond result")
            .with_example("\"hello\"", "\"olleh\"", "Reverse hello");

        let gen = TestGenerator::new();
        let tests = gen.generate_from_spec(&spec);

        // Should have: 1 example + 1 postcondition + 1 invariant + 6 boundary + 4 error = 13
        assert!(!tests.is_empty());
        assert!(tests.iter().any(|t| t.test_type == TestType::HappyPath));
        assert!(tests.iter().any(|t| t.test_type == TestType::Property));
        assert!(tests.iter().any(|t| t.test_type == TestType::Boundary));
        assert!(tests.iter().any(|t| t.test_type == TestType::ErrorCase));
    }

    // ── P3-A: Mutation Testing ──

    #[test]
    fn test_mutation_generator_finds_plus() {
        let source = "let x = a + b;\nlet y = c - d;\n";
        let gen = MutationGenerator::new();
        let mutations = gen.find_mutation_points(source, "test.rs");

        assert!(mutations.iter().any(|m| matches!(m.mutation_type, MutationType::PlusToMinus)));
        assert!(mutations.iter().any(|m| matches!(m.mutation_type, MutationType::MinusToPlus)));
    }

    #[test]
    fn test_mutation_generator_skips_comments() {
        let source = "// let x = a + b;\nlet y = c + d;\n";
        let gen = MutationGenerator::new();
        let mutations = gen.find_mutation_points(source, "test.rs");

        // Only the non-comment line should have mutations
        assert_eq!(mutations.len(), 1);
        assert_eq!(mutations[0].line, 2);
    }

    #[test]
    fn test_mutation_report() {
        let results = vec![
            MutationResult {
                mutation: Mutation {
                    id: "m1".to_string(),
                    file: "test.rs".to_string(),
                    line: 1,
                    mutation_type: MutationType::PlusToMinus,
                    original: "a + b".to_string(),
                    mutated: "a - b".to_string(),
                },
                killed: true,
                killed_by: Some("test_add".to_string()),
            },
            MutationResult {
                mutation: Mutation {
                    id: "m2".to_string(),
                    file: "test.rs".to_string(),
                    line: 2,
                    mutation_type: MutationType::TrueToFalse,
                    original: "true".to_string(),
                    mutated: "false".to_string(),
                },
                killed: false,
                killed_by: None,
            },
        ];

        let report = MutationReport::from_results(results);
        assert_eq!(report.total_mutations, 2);
        assert_eq!(report.killed, 1);
        assert_eq!(report.survived, 1);
        assert!((report.mutation_score - 0.5).abs() < 1e-10);
        assert!(report.summary().contains("50.0%"));
    }

    // ── P3-B: Replay Testing ──

    #[test]
    fn test_recorder_and_player() {
        let mut recorder = Recorder::new();

        // 录制两次调用
        recorder.record("add", &1i32, &3i32);
        recorder.record("multiply", &(2, 3), &6i32);

        assert_eq!(recorder.calls().len(), 2);

        // 创建回放器
        let mut player = Player::from_recorder(&recorder);

        // 验证第一次调用
        player.verify("add", &1i32, &3i32).expect("First call should match");

        // 验证第二次调用
        player.verify("multiply", &(2, 3), &6i32).expect("Second call should match");

        assert!(!player.has_more());
    }

    #[test]
    fn test_replay_detects_mismatch() {
        let mut recorder = Recorder::new();
        recorder.record("add", &1i32, &3i32);

        let mut player = Player::from_recorder(&recorder);

        // 错误的输出应该被检测到
        let result = player.verify("add", &1i32, &4i32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Output mismatch"));
    }

    #[test]
    fn test_replay_save_load() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let path = dir.path().join("recording.json");

        // 录制并保存
        {
            let mut recorder = Recorder::new().with_save_path(&path);
            recorder.record("test_fn", &"input", &"output");
            recorder.save().expect("save failed");
        }

        // 加载并验证
        let mut player = Player::load(&path).expect("load failed");
        assert_eq!(player.len(), 1);
        player.verify("test_fn", &"input", &"output").expect("verify failed");
    }

    // ── P2-A: Evaluator-Optimizer ──

    struct DummyGenerator;
    struct DummyEvaluator;

    impl Generator for DummyGenerator {
        type Input = i32;
        type Output = i32;

        fn generate(&self, input: &Self::Input) -> Self::Output {
            input * 2
        }
    }

    impl Evaluator for DummyEvaluator {
        type Input = i32;
        type Output = i32;

        fn evaluate(&self, input: &Self::Input, output: &Self::Output) -> EvaluationResult {
            let expected = input * 2;
            if *output == expected {
                EvaluationResult::pass(1.0, vec!["Correct".to_string()])
            } else {
                EvaluationResult::fail(0.0, vec![format!("Expected {expected}, got {output}")])
            }
        }
    }

    #[test]
    fn test_eval_optimizer_converges() {
        let optimizer = EvalOptimizer::new(DummyGenerator, DummyEvaluator)
            .with_threshold(0.8)
            .with_max_iterations(3);

        let result = optimizer.run(&5);

        assert!(result.converged);
        assert_eq!(result.output, 10);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_eval_optimizer_max_iterations() {
        struct BadGenerator;
        impl Generator for BadGenerator {
            type Input = i32;
            type Output = i32;
            fn generate(&self, _input: &Self::Input) -> Self::Output {
                999 // Always wrong
            }
        }
        struct StrictEvaluator;
        impl Evaluator for StrictEvaluator {
            type Input = i32;
            type Output = i32;
            fn evaluate(&self, input: &Self::Input, output: &Self::Output) -> EvaluationResult {
                if *output == input * 2 {
                    EvaluationResult::pass(1.0, vec![])
                } else {
                    EvaluationResult::fail(0.0, vec!["Wrong".to_string()])
                }
            }
        }

        let optimizer = EvalOptimizer::new(BadGenerator, StrictEvaluator)
            .with_threshold(0.8)
            .with_max_iterations(3);

        let result = optimizer.run(&5);

        assert!(!result.converged);
        assert_eq!(result.iterations, 3);
    }
}
