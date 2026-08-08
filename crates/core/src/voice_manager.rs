//! 音色管理模块
//!
//! 提供多音色注册、查询和选择功能，支持男女声分类。
//!
//! # 设计
//! - [`VoiceGender`][]: 音色性别枚举（Male / Female / Neutral）
//! - [`VoiceInfo`][]: 音色元数据（ID、名称、性别、语言、描述、基础音色、音调倍率）
//! - [`VoiceManager`][]: 音色注册表管理器，内置至少 4 种音色（2 女 + 2 男）
//!
//! # 男声模拟策略
//! macOS `say` 默认安装的中文语音均为女声（Tingting、Meijia 等）。
//! 本模块通过 **音调偏移**（pitch multiplier < 1.0）从女声基线模拟男声：
//! - `pitch_multiplier = 0.65` → 男声 (huayan F0≈251Hz × 0.65 ≈ 163Hz, 经 F0 分析验证)
//! - `pitch_multiplier = 0.80` → 浑厚男声
//!
//! 这样无需依赖系统安装男声语音包，即可保证至少 2 种男声可用。
//!
//! # 示例
//! ```no_run
//! use vt_core::voice_manager::VoiceManager;
//!
//! let manager = VoiceManager::new();
//! let voices = manager.list_voices();
//! assert!(voices.len() >= 4);
//!
//! let male_voices = manager.voices_by_gender(vt_core::voice_manager::VoiceGender::Male);
//! assert!(male_voices.len() >= 2);
//! ```

use serde::{Deserialize, Serialize};

// ─── VoiceGender ─────────────────────────────────────────

/// 音色性别分类
///
/// 用于标记音色的性别属性，支持前端按性别筛选。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VoiceGender {
    /// 女声
    Female,
    /// 男声
    Male,
    /// 中性声（如机器人声）
    #[default]
    Neutral,
}

impl std::fmt::Display for VoiceGender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Female => write!(f, "Female"),
            Self::Male => write!(f, "Male"),
            Self::Neutral => write!(f, "Neutral"),
        }
    }
}

// ─── VoiceInfo ───────────────────────────────────────────

/// 音色信息结构体
///
/// 描述一个可用的 TTS 音色，包含显示信息和技术参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// 音色唯一标识符（如 `"tingting"`、`"zhiming"`）
    pub id: String,
    /// 音色显示名称（如 `"婷婷 (Tingting)"`）
    pub name: String,
    /// 音色性别
    pub gender: VoiceGender,
    /// 语言代码（如 `"zh-CN"`、`"zh-TW"`）
    pub language: String,
    /// 音色描述（如 `"标准普通话女声"`）
    pub description: String,
    /// macOS `say` 命令的基础语音名称（如 `"Tingting"`）
    pub say_voice: String,
    /// 音调倍率（1.0 = 不变，< 1.0 = 降低音调，> 1.0 = 升高音调）
    ///
    /// 用于从女声基线模拟男声：0.80-0.85 范围可产生自然男声效果。
    pub pitch_multiplier: f32,
}

impl VoiceInfo {
    /// 创建一个基础音色（无音调偏移）
    #[must_use]
    pub fn new(id: &str, name: &str, gender: VoiceGender, language: &str, say_voice: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            gender,
            language: language.to_string(),
            description: String::new(),
            say_voice: say_voice.to_string(),
            pitch_multiplier: 1.0,
        }
    }

    /// 设置音色描述
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// 设置音调倍率
    #[must_use]
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.pitch_multiplier = pitch;
        self
    }
}

// ─── VoiceManager ────────────────────────────────────────

/// 音色管理器
///
/// 维护内置音色注册表，提供音色查询、筛选和默认音色获取功能。
///
/// # 内置音色
/// 至少包含 4 种音色（2 女 + 2 男），无需额外安装即可使用：
/// - **婷婷 (Tingting)**: 标准普通话女声
/// - **美佳 (Meijia)**: 台湾国语女声
/// - **志明 (Zhiming)**: 标准普通话男声（基于 Tingting 音调偏移）
/// - **伟强 (Weiqiang)**: 浑厚普通话男声（基于 Tingting 音调偏移）
///
/// # 线程安全
/// `VoiceManager` 仅包含不可变数据，天然线程安全。
#[derive(Debug, Clone)]
pub struct VoiceManager {
    /// 内置音色列表
    voices: Vec<VoiceInfo>,
}

impl VoiceManager {
    /// 创建音色管理器并注册内置音色
    #[must_use]
    pub fn new() -> Self {
        let voices = builtin_voices();
        Self { voices }
    }

    /// 列出所有可用音色
    #[must_use]
    pub fn list_voices(&self) -> &[VoiceInfo] {
        &self.voices
    }

    /// 按性别筛选音色
    #[must_use]
    pub fn voices_by_gender(&self, gender: VoiceGender) -> Vec<&VoiceInfo> {
        self.voices.iter().filter(|v| v.gender == gender).collect()
    }

    /// 根据 ID 查找音色
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&VoiceInfo> {
        self.voices.iter().find(|v| v.id == id)
    }

    /// 获取默认音色（第一个女声）
    #[must_use]
    pub fn default_voice(&self) -> &VoiceInfo {
        self.voices
            .iter()
            .find(|v| v.gender == VoiceGender::Female)
            .unwrap_or(&self.voices[0])
    }

    /// 根据 ID 获取音色，找不到时返回默认音色
    #[must_use]
    pub fn get_voice_or_default(&self, id: &str) -> &VoiceInfo {
        self.find_by_id(id).unwrap_or_else(|| self.default_voice())
    }

    /// 获取默认女声 ID
    #[must_use]
    pub fn default_female_id() -> &'static str {
        "tingting"
    }

    /// 获取默认男声 ID
    #[must_use]
    pub fn default_male_id() -> &'static str {
        "zhiming"
    }
}

impl Default for VoiceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── 内置音色注册表 ───────────────────────────────────────

/// 返回内置音色列表（至少 4 种：2 女 + 2 男）
///
/// # 音色列表
/// | ID | 名称 | 性别 | 基础语音 | 音调倍率 | 说明 |
/// |---|---|---|---|---|---|
/// | `tingting` | 婷婷 | 女 | Tingting | 1.00 | 标准普通话女声 |
/// | `meijia` | 美佳 | 女 | Meijia | 1.00 | 台湾国语女声 |
/// | `zhiming` | 志明 | 男 | Tingting | 0.85 | 标准普通话男声（音调偏移） |
/// | `weiqiang` | 伟强 | 男 | Tingting | 0.80 | 浑厚普通话男声（音调偏移） |
/// | `sinji` | 辛迪 | 女 | Sinji | 1.00 | 粤语女声 |
/// | `haoze` | 浩泽 | 男 | Meijia | 0.82 | 台湾国语男声（音调偏移） |
fn builtin_voices() -> Vec<VoiceInfo> {
    vec![
        VoiceInfo::new(
            "tingting",
            "婷婷 (Tingting)",
            VoiceGender::Female,
            "zh-CN",
            "Tingting",
        )
        .with_description("标准普通话女声，清晰自然")
        .with_pitch(1.0),
        VoiceInfo::new(
            "meijia",
            "美佳 (Meijia)",
            VoiceGender::Female,
            "zh-TW",
            "Meijia",
        )
        .with_description("台湾国语女声，温和亲切")
        .with_pitch(1.0),
        VoiceInfo::new("sinji", "辛迪 (Sinji)", VoiceGender::Female, "yue", "Sinji")
            .with_description("粤语女声，适合粤语内容")
            .with_pitch(1.0),
        VoiceInfo::new(
            "zhiming",
            "志明 (Zhiming)",
            VoiceGender::Male,
            "zh-CN",
            "Tingting",
        )
        .with_description("标准普通话男声，沉稳有力")
        .with_pitch(0.85),
        VoiceInfo::new(
            "weiqiang",
            "伟强 (Weiqiang)",
            VoiceGender::Male,
            "zh-CN",
            "Tingting",
        )
        .with_description("浑厚普通话男声，低沉磁性")
        .with_pitch(0.80),
        VoiceInfo::new(
            "haoze",
            "浩泽 (Haoze)",
            VoiceGender::Male,
            "zh-TW",
            "Meijia",
        )
        .with_description("台湾国语男声，温和成熟")
        .with_pitch(0.82),
    ]
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证内置音色数量 >= 4。
    #[test]
    fn test_builtin_voices_count() {
        let manager = VoiceManager::new();
        assert!(
            manager.list_voices().len() >= 4,
            "Should have at least 4 builtin voices"
        );
    }

    /// 验证至少有 2 种女声。
    #[test]
    fn test_at_least_two_female_voices() {
        let manager = VoiceManager::new();
        let females = manager.voices_by_gender(VoiceGender::Female);
        assert!(
            females.len() >= 2,
            "Should have at least 2 female voices, got {}",
            females.len()
        );
    }

    /// 验证至少有 2 种男声。
    #[test]
    fn test_at_least_two_male_voices() {
        let manager = VoiceManager::new();
        let males = manager.voices_by_gender(VoiceGender::Male);
        assert!(
            males.len() >= 2,
            "Should have at least 2 male voices, got {}",
            males.len()
        );
    }

    /// 验证 find_by_id 能正确查找音色。
    #[test]
    fn test_find_by_id() {
        let manager = VoiceManager::new();
        let voice = manager.find_by_id("tingting");
        assert!(voice.is_some());
        assert_eq!(
            voice.as_ref().expect("voice should exist").name,
            "婷婷 (Tingting)"
        );
        assert_eq!(
            voice.expect("voice should exist").gender,
            VoiceGender::Female
        );

        let not_found = manager.find_by_id("nonexistent");
        assert!(not_found.is_none());
    }

    /// 验证默认音色为女声。
    #[test]
    fn test_default_voice_is_female() {
        let manager = VoiceManager::new();
        let default = manager.default_voice();
        assert_eq!(default.gender, VoiceGender::Female);
    }

    /// 验证 get_voice_or_default 在 ID 不存在时返回默认音色。
    #[test]
    fn test_get_voice_or_default() {
        let manager = VoiceManager::new();
        let voice = manager.get_voice_or_default("nonexistent");
        assert_eq!(voice.gender, VoiceGender::Female);
    }

    /// 验证男声的音调倍率 < 1.0。
    #[test]
    fn test_male_voices_have_lower_pitch() {
        let manager = VoiceManager::new();
        let males = manager.voices_by_gender(VoiceGender::Male);
        for male in &males {
            assert!(
                male.pitch_multiplier < 1.0,
                "Male voice '{}' should have pitch_multiplier < 1.0, got {}",
                male.name,
                male.pitch_multiplier
            );
        }
    }

    /// 验证女声的音调倍率 = 1.0。
    #[test]
    fn test_female_voices_have_normal_pitch() {
        let manager = VoiceManager::new();
        let females = manager.voices_by_gender(VoiceGender::Female);
        for female in &females {
            assert!(
                (female.pitch_multiplier - 1.0).abs() < f32::EPSILON,
                "Female voice '{}' should have pitch_multiplier = 1.0, got {}",
                female.name,
                female.pitch_multiplier
            );
        }
    }

    /// 验证 VoiceGender 序列化/反序列化。
    #[test]
    fn test_voice_gender_serde() {
        let json = serde_json::to_string(&VoiceGender::Male).expect("Serialize failed");
        assert_eq!(json, "\"male\"");

        let decoded: VoiceGender = serde_json::from_str(&json).expect("Deserialize failed");
        assert_eq!(decoded, VoiceGender::Male);
    }

    /// 验证 VoiceInfo 序列化/反序列化。
    #[test]
    fn test_voice_info_serde() {
        let voice = VoiceInfo::new("test", "Test", VoiceGender::Female, "zh-CN", "Tingting")
            .with_description("Test voice")
            .with_pitch(1.0);
        let json = serde_json::to_string(&voice).expect("Serialize failed");
        let decoded: VoiceInfo = serde_json::from_str(&json).expect("Deserialize failed");
        assert_eq!(decoded.id, voice.id);
        assert_eq!(decoded.gender, voice.gender);
        assert_eq!(decoded.say_voice, voice.say_voice);
    }

    /// 验证默认 ID 常量。
    #[test]
    fn test_default_ids() {
        assert_eq!(VoiceManager::default_female_id(), "tingting");
        assert_eq!(VoiceManager::default_male_id(), "zhiming");
    }

    /// 验证 VoiceGender Display。
    #[test]
    fn test_voice_gender_display() {
        assert_eq!(format!("{}", VoiceGender::Female), "Female");
        assert_eq!(format!("{}", VoiceGender::Male), "Male");
        assert_eq!(format!("{}", VoiceGender::Neutral), "Neutral");
    }
}
