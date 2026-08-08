//! CLI 集成测试
//!
//! 使用 `assert_cmd` 执行 `vt` 二进制文件，验证各子命令的行为。
//! 遵循 TDD 原则：先编写测试（红），再实现功能（绿）。

use std::fs;

use assert_cmd::Command;
use tempfile::TempDir;

// ─── config 子命令测试 ────────────────────────────────────

/// 验证 `vt config` 输出合法的 TOML 格式。
#[test]
fn test_cli_config_generate() {
    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("config");

    let output = cmd.output().expect("failed to execute vt config");

    assert!(
        output.status.success(),
        "vt config should exit successfully, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // 验证输出是合法的 TOML
    let parsed: toml::Value = toml::from_str(&stdout).expect("output should be valid TOML");

    // 验证包含必要的配置段
    assert!(
        parsed.get("asr").is_some(),
        "config should contain [asr] section"
    );
    assert!(
        parsed.get("translation").is_some(),
        "config should contain [translation] section"
    );
    assert!(
        parsed.get("tts").is_some(),
        "config should contain [tts] section"
    );
    assert!(
        parsed.get("pipeline").is_some(),
        "config should contain [pipeline] section"
    );

    // 验证 ASR 配置段包含关键字段
    let asr = parsed.get("asr").expect("asr section should exist");
    assert!(asr.get("model").is_some(), "asr should have model field");
    assert!(
        asr.get("use_metal").is_some(),
        "asr should have use_metal field"
    );
    assert!(
        asr.get("language").is_some(),
        "asr should have language field"
    );
}

/// 验证 `vt config --output <file>` 将配置写入指定文件。
#[test]
fn test_cli_config_generate_to_file() {
    let tmp = TempDir::new().expect("failed to create temp dir");
    let config_path = tmp.path().join("config.toml");

    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("config").arg("--output").arg(&config_path);

    let output = cmd.output().expect("failed to execute vt config --output");

    assert!(
        output.status.success(),
        "vt config --output should exit successfully, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // 验证文件已创建
    assert!(config_path.exists(), "config file should be created");

    // 验证文件内容是合法 TOML
    let content = fs::read_to_string(&config_path).expect("failed to read config file");
    let parsed: toml::Value =
        toml::from_str(&content).expect("config file content should be valid TOML");
    assert!(parsed.get("asr").is_some());
    assert!(parsed.get("tts").is_some());
}

// ─── process 子命令测试 ───────────────────────────────────

/// 验证 `vt process --help` 显示正确的帮助信息。
#[test]
fn test_cli_process_help() {
    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("process").arg("--help");

    let output = cmd.output().expect("failed to execute vt process --help");

    // --help 应该成功退出（退出码 0）
    assert!(
        output.status.success(),
        "vt process --help should exit successfully"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // 验证帮助信息包含关键字段
    assert!(
        stdout.contains("--input"),
        "help should mention --input option"
    );
    assert!(
        stdout.contains("--output"),
        "help should mention --output option"
    );
    assert!(
        stdout.contains("--config"),
        "help should mention --config option"
    );
}

/// 验证 `vt process`（缺少 `--input` 参数）报错。
#[test]
fn test_cli_process_missing_input() {
    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("process");

    let output = cmd.output().expect("failed to execute vt process");

    // 缺少必填参数应该失败（非零退出码）
    assert!(
        !output.status.success(),
        "vt process without --input should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);

    // 验证错误信息提示缺少 --input
    assert!(
        stderr.contains("--input") || stderr.contains("input"),
        "error should mention --input, stderr: {stderr}"
    );
}

/// 验证 `vt process --input <nonexistent>` 对不存在的文件给出友好错误。
#[test]
fn test_cli_process_file_not_found() {
    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("process")
        .arg("--input")
        .arg("/nonexistent/path/video.mp4");

    let output = cmd.output().expect("failed to execute vt process");

    assert!(
        !output.status.success(),
        "vt process with nonexistent file should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);

    // 验证错误信息包含"not found"或类似提示
    let stderr_lower = stderr.to_lowercase();
    assert!(
        stderr_lower.contains("not found")
            || stderr_lower.contains("不存在")
            || stderr_lower.contains("no such file"),
        "error should mention file not found, stderr: {stderr}"
    );
}

// ─── batch 子命令测试 ─────────────────────────────────────

/// 验证 `vt batch --help` 显示正确的帮助信息。
#[test]
fn test_cli_batch_help() {
    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("batch").arg("--help");

    let output = cmd.output().expect("failed to execute vt batch --help");

    assert!(
        output.status.success(),
        "vt batch --help should exit successfully"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("--input-dir"),
        "help should mention --input-dir option"
    );
    assert!(
        stdout.contains("--output-dir"),
        "help should mention --output-dir option"
    );
}

/// 验证 `vt batch`（缺少 `--input-dir` 参数）报错。
#[test]
fn test_cli_batch_missing_input_dir() {
    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("batch");

    let output = cmd.output().expect("failed to execute vt batch");

    assert!(
        !output.status.success(),
        "vt batch without --input-dir should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--input-dir") || stderr.contains("input-dir"),
        "error should mention --input-dir, stderr: {stderr}"
    );
}

/// 验证 `vt batch` 对空目录输出"无视频文件"的提示。
#[test]
fn test_cli_batch_empty_dir() {
    let tmp = TempDir::new().expect("failed to create temp dir");
    let output_dir = tmp.path().join("output");

    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("batch")
        .arg("--input-dir")
        .arg(tmp.path())
        .arg("--output-dir")
        .arg(&output_dir);

    let output = cmd.output().expect("failed to execute vt batch");

    // 空目录应该成功退出（没有文件需要处理）
    // 或者返回非零退出码，取决于设计
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

    // 应该提到没有找到视频文件
    assert!(
        combined.to_lowercase().contains("no") || combined.to_lowercase().contains("0"),
        "should mention no files found or 0 files, stdout: {stdout}, stderr: {stderr}"
    );
}

// ─── 全局选项测试 ─────────────────────────────────────────

/// 验证 `vt --help` 显示顶层帮助信息。
#[test]
fn test_cli_top_level_help() {
    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");
    cmd.arg("--help");

    let output = cmd.output().expect("failed to execute vt --help");

    assert!(
        output.status.success(),
        "vt --help should exit successfully"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // 验证帮助信息包含子命令
    assert!(
        stdout.contains("process"),
        "help should mention process subcommand"
    );
    assert!(
        stdout.contains("batch"),
        "help should mention batch subcommand"
    );
    assert!(
        stdout.contains("config"),
        "help should mention config subcommand"
    );
    assert!(
        stdout.contains("--verbose"),
        "help should mention --verbose option"
    );
    assert!(
        stdout.contains("--quiet"),
        "help should mention --quiet option"
    );
}

/// 验证 `vt`（无子命令）显示帮助或错误。
#[test]
fn test_cli_no_subcommand() {
    let mut cmd = Command::cargo_bin("vt").expect("failed to find vt binary");

    let output = cmd.output().expect("failed to execute vt");

    // 无子命令应该非零退出（clap 默认行为）
    assert!(
        !output.status.success(),
        "vt without subcommand should fail"
    );
}
