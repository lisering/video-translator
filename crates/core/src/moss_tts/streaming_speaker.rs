//! 流式推理 + 多说话者支持 — 借鉴 MOSS-TTS realtime/streaming 模块
//!
//! MOSS-TTS 的 `streaming_mossttsrealtime.py` 提供了 token-by-token
//! 流式生成 API，此模块定义 video-translator 中的流式推理接口。
//!
//! 多说话者支持借鉴 MOSS-TTSD 的 `[S1]`/`[S2]` 标签机制。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ─── 流式推理 ──────────────────────────────────────────────

/// 流式推理请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingRequest {
    /// 要合成的文本
    pub text: String,
    /// 参考音频路径
    pub reference_audio: Option<PathBuf>,
    /// 目标语言
    pub language: Option<String>,
    /// 流式回调间隔（帧数）
    #[serde(default = "default_streaming_interval")]
    pub streaming_interval: usize,
    /// 最大生成 token 数
    pub max_tokens: Option<usize>,
}

fn default_streaming_interval() -> usize {
    25 // 每 25 帧输出一次音频块
}

/// 流式音频块
#[derive(Debug, Clone)]
pub struct StreamingAudioChunk {
    /// 音频码 [T, n_vq]
    pub codes: Vec<Vec<i64>>,
    /// 块索引
    pub chunk_index: usize,
    /// 是否为最后一块
    pub is_final: bool,
}

/// 流式推理回调函数类型
pub type StreamingCallback = Box<dyn FnMut(&StreamingAudioChunk) + Send + 'static>;

/// 流式推理接口
///
/// 定义流式 TTS 推理的标准接口。
/// 实现可以基于:
/// - Python subprocess（当前方式）
/// - llama.cpp C bridge（P0 移植后）
/// - Candle Rust 引擎（vt-tts）
pub trait StreamingTts: Send + Sync {
    /// 开始流式推理
    ///
    /// 返回一个接收器，流式输出音频块
    fn stream(&self, request: &StreamingRequest, callback: StreamingCallback)
        -> Result<(), String>;

    /// 取消推理
    fn cancel(&self) {}

    /// 获取名称
    fn name(&self) -> &str;
}

/// 流式状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// 空闲
    Idle,
    /// 正在生成
    Generating,
    /// 正在 drain（排空剩余帧）
    Draining,
    /// 完成
    Finished,
    /// 错误
    Error,
}

impl Default for StreamState {
    fn default() -> Self {
        Self::Idle
    }
}

// ─── 多说话者支持 ──────────────────────────────────────────

/// 说话者信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Speaker {
    /// 说话者 ID（如 "S1", "S2"）
    pub id: String,
    /// 说话者名称
    pub name: Option<String>,
    /// 参考音频路径
    pub reference_audio: Option<PathBuf>,
    /// 参考音频对应的文本
    pub reference_text: Option<String>,
}

impl Speaker {
    /// 创建新说话者
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            reference_audio: None,
            reference_text: None,
        }
    }

    /// 设置参考音频
    pub fn with_reference(mut self, audio: PathBuf, text: Option<String>) -> Self {
        self.reference_audio = Some(audio);
        self.reference_text = text;
        self
    }
}

/// 说话者管理器
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpeakerManager {
    /// 说话者列表
    pub speakers: Vec<Speaker>,
}

impl SpeakerManager {
    /// 创建空管理器
    pub fn new() -> Self {
        Self { speakers: vec![] }
    }

    /// 添加说话者
    pub fn add(&mut self, speaker: Speaker) {
        self.speakers.push(speaker);
    }

    /// 按 ID 查找说话者
    pub fn find(&self, id: &str) -> Option<&Speaker> {
        self.speakers.iter().find(|s| s.id == id)
    }

    /// 说话者数量
    pub fn len(&self) -> usize {
        self.speakers.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.speakers.is_empty()
    }

    /// 构建多说话者文本
    ///
    /// 将文本段与说话者标签组合，类似 MOSS-TTSD 的 `[S1]`/`[S2]` 格式
    pub fn build_multi_speaker_text(&self, segments: &[(usize, &str)]) -> String {
        segments
            .iter()
            .map(|(speaker_idx, text)| {
                let speaker = self
                    .speakers
                    .get(*speaker_idx)
                    .map(|s| s.id.as_str())
                    .unwrap_or("S1");
                format!("[{speaker}]{text}")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ─── 单元测试 ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_new() {
        let speaker = Speaker::new("S1");
        assert_eq!(speaker.id, "S1");
        assert!(speaker.reference_audio.is_none());
    }

    #[test]
    fn test_speaker_with_reference() {
        let speaker = Speaker::new("S1")
            .with_reference(PathBuf::from("/tmp/ref.wav"), Some("hello".to_string()));
        assert_eq!(speaker.id, "S1");
        assert!(speaker.reference_audio.is_some());
        assert_eq!(speaker.reference_text.as_deref(), Some("hello"));
    }

    #[test]
    fn test_speaker_manager_add() {
        let mut manager = SpeakerManager::new();
        manager.add(Speaker::new("S1"));
        manager.add(Speaker::new("S2"));
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn test_speaker_manager_find() {
        let mut manager = SpeakerManager::new();
        manager.add(Speaker::new("S1"));
        manager.add(Speaker::new("S2"));
        let found = manager.find("S2");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "S2");
    }

    #[test]
    fn test_speaker_manager_find_missing() {
        let mut manager = SpeakerManager::new();
        manager.add(Speaker::new("S1"));
        let found = manager.find("S3");
        assert!(found.is_none());
    }

    #[test]
    fn test_build_multi_speaker_text() {
        let mut manager = SpeakerManager::new();
        manager.add(Speaker::new("S1"));
        manager.add(Speaker::new("S2"));

        let segments = vec![(0, "你好"), (1, "Hello"), (0, "再见")];

        let text = manager.build_multi_speaker_text(&segments);
        assert!(text.contains("[S1]你好"));
        assert!(text.contains("[S2]Hello"));
        assert!(text.contains("[S1]再见"));
    }

    #[test]
    fn test_streaming_request_default() {
        let request = StreamingRequest {
            text: "hello".to_string(),
            reference_audio: None,
            language: Some("zh".to_string()),
            streaming_interval: 25,
            max_tokens: None,
        };
        assert_eq!(request.streaming_interval, 25);
    }

    #[test]
    fn test_stream_state_default() {
        assert_eq!(StreamState::default(), StreamState::Idle);
    }

    #[test]
    fn test_streaming_audio_chunk() {
        let chunk = StreamingAudioChunk {
            codes: vec![vec![1, 2, 3]],
            chunk_index: 0,
            is_final: false,
        };
        assert_eq!(chunk.codes.len(), 1);
        assert!(!chunk.is_final);
    }
}
