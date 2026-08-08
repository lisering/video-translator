//! Segment 数据模型与状态机
//!
//! [`Segment`] 表示视频中的一个时间片段，包含源文本、翻译文本、
//! TTS 音频路径等字段，并通过 [`SegmentStatus`] 状态机管理生命周期。

use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};

/// 片段处理状态枚举
///
/// 状态机流转顺序（正常路径）：
/// ```text
/// Pending → Transcribing → Translated → Synthesizing → Completed
/// ```
///
/// 任意状态均可转换为 `Failed`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SegmentStatus {
    /// 待处理（初始状态）
    #[default]
    Pending,
    /// 正在转录（ASR 进行中）
    Transcribing,
    /// 已翻译（转录完成，翻译完成）
    Translated,
    /// 正在合成语音（TTS 进行中）
    Synthesizing,
    /// 已完成（全部步骤完成）
    Completed,
    /// 失败
    Failed,
}

impl SegmentStatus {
    /// 尝试转换到目标状态
    ///
    /// 仅允许合法的状态转换路径，非法转换返回 [`AppError::InvalidStateTransition`]。
    ///
    /// # 正常转换路径
    /// - `Pending` → `Transcribing`
    /// - `Transcribing` → `Translated`
    /// - `Translated` → `Synthesizing`
    /// - `Synthesizing` → `Completed`
    ///
    /// # 特殊路径
    /// - 任意状态 → `Failed`
    ///
    /// # 错误
    /// - [`AppError::InvalidStateTransition`][]: 当转换路径不合法时返回。
    pub fn transition_to(self, target: SegmentStatus) -> AppResult<SegmentStatus> {
        use SegmentStatus::*;
        let valid = matches!(
            (self, target),
            (Pending, Transcribing)
                | (Transcribing, Translated)
                | (Translated, Synthesizing)
                | (Synthesizing, Completed)
                | (_, Failed)
        );
        if valid {
            Ok(target)
        } else {
            Err(AppError::InvalidStateTransition(format!(
                "{self:?} -> {target:?}"
            )))
        }
    }
}

/// 视频片段结构体
///
/// 表示视频中的一个时间片段，包含原始语音转录文本、翻译文本、
/// TTS 音频输出路径以及当前处理状态。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// 片段唯一标识符
    pub id: String,
    /// 片段起始时间（秒）
    pub start: f64,
    /// 片段结束时间（秒）
    pub end: f64,
    /// 说话人标识（可选）
    pub speaker: Option<String>,
    /// 源语言文本（英文转录结果）
    pub source_text: String,
    /// 目标语言文本（中文翻译结果）
    pub target_text: Option<String>,
    /// TTS 合成音频文件路径
    pub tts_audio_path: Option<String>,
    /// 当前处理状态
    pub status: SegmentStatus,
}

impl Segment {
    /// 创建新的 `Segment` 实例
    ///
    /// # 参数
    /// - `id`: 片段唯一标识符
    /// - `start`: 起始时间（秒）
    /// - `end`: 结束时间（秒）
    /// - `source_text`: 源语言文本
    ///
    /// # 返回
    /// 状态为 `Pending` 的新片段，其余可选字段为 `None`。
    #[must_use]
    pub fn new(id: String, start: f64, end: f64, source_text: String) -> Self {
        Self {
            id,
            start,
            end,
            speaker: None,
            source_text,
            target_text: None,
            tts_audio_path: None,
            status: SegmentStatus::Pending,
        }
    }

    /// 开始转录（`Pending` → `Transcribing`）
    ///
    /// # 错误
    /// - [`AppError::InvalidStateTransition`][]: 当前状态不是 `Pending` 时返回。
    pub fn start_transcribing(&mut self) -> AppResult<()> {
        self.status = self.status.transition_to(SegmentStatus::Transcribing)?;
        Ok(())
    }

    /// 完成转录并设置翻译文本（`Transcribing` → `Translated`）
    ///
    /// # 参数
    /// - `target_text`: 翻译后的目标语言文本
    ///
    /// # 错误
    /// - [`AppError::InvalidStateTransition`][]: 当前状态不是 `Transcribing` 时返回。
    pub fn finish_transcribing(&mut self, target_text: String) -> AppResult<()> {
        self.status = self.status.transition_to(SegmentStatus::Translated)?;
        self.target_text = Some(target_text);
        Ok(())
    }

    /// 开始语音合成（`Translated` → `Synthesizing`）
    ///
    /// # 错误
    /// - [`AppError::InvalidStateTransition`][]: 当前状态不是 `Translated` 时返回。
    pub fn start_synthesizing(&mut self) -> AppResult<()> {
        self.status = self.status.transition_to(SegmentStatus::Synthesizing)?;
        Ok(())
    }

    /// 完成语音合成并设置音频路径（`Synthesizing` → `Completed`）
    ///
    /// # 参数
    /// - `audio_path`: TTS 合成音频文件路径
    ///
    /// # 错误
    /// - [`AppError::InvalidStateTransition`][]: 当前状态不是 `Synthesizing` 时返回。
    pub fn finish_synthesizing(&mut self, audio_path: String) -> AppResult<()> {
        self.status = self.status.transition_to(SegmentStatus::Completed)?;
        self.tts_audio_path = Some(audio_path);
        Ok(())
    }

    /// 标记片段为失败（任意状态 → `Failed`）
    ///
    /// # 错误
    /// 当前实现中任意状态均可转为 `Failed`，因此不会返回错误。
    /// 保留 `Result` 返回类型以保持接口一致性和未来扩展性。
    pub fn fail(&mut self) -> AppResult<()> {
        self.status = self.status.transition_to(SegmentStatus::Failed)?;
        Ok(())
    }
}

impl Default for Segment {
    /// 返回一个空白的默认 `Segment`
    ///
    /// 所有字符串字段为空，数值字段为 `0.0`，可选字段为 `None`，
    /// 状态为 [`SegmentStatus::Pending`]。
    fn default() -> Self {
        Self {
            id: String::new(),
            start: 0.0,
            end: 0.0,
            speaker: None,
            source_text: String::new(),
            target_text: None,
            tts_audio_path: None,
            status: SegmentStatus::default(),
        }
    }
}
