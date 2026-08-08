/// 前端类型定义
///
/// 与 Rust 后端的 Tauri 命令返回类型对应。

/** 任务状态（与 Rust `TaskStatus` 枚举对应） */
export type TaskStatus =
  | 'Pending'
  | 'Running'
  | 'Completed'
  | 'Failed'
  | 'Cancelled';

/** 任务进度信息（与 Rust `ProgressInfo` 结构体对应） */
export interface ProgressInfo {
  task_id: string;
  status: TaskStatus;
  progress: number;
  stage: string;
  error?: string;
}

/** 视频文件信息（与 Rust `VideoInfo` 结构体对应） */
export interface VideoInfo {
  path: string;
  duration: number;
  width: number | null;
  height: number | null;
  video_codec: string | null;
  audio_codec: string | null;
}

/** 应用配置（与 Rust `Config` 结构体对应） */
export interface AppConfig {
  asr: {
    model: string;
    use_metal: boolean;
    language: string;
  };
  tts: {
    engine: string;
    speed: number;
    pitch: number;
    volume: number;
    voice_id: string;
    voice: string;
    sample_rate: number;
    device: string;
    cache_dir: string;
    parallel_tasks: number;
    model_variant: string;
    model_path: string | null;
    fallback_to_say: boolean;
    auto_voice_selection: boolean;
    seed: number | null;
    temperature: number;
    stability: number;
    eq_high_shelf_db: number;
    crossfade_duration_ms: number;
  };
  translation: {
    glossary_path: string | null;
    batch_size: number;
  };
  output_dir: string;
  max_concurrent_tasks: number;
  pipeline: {
    segment_duration_secs: number;
    channel_capacity: number;
    enable_vad_split: boolean;
  };
}

/** TTS 音色信息（与 Rust `VoiceInfoDto` 对应） */
export interface VoiceInfo {
  id: string;
  name: string;
  gender: 'female' | 'male' | 'neutral';
  language: string;
  description: string;
}

/** 日志条目 */
export interface LogEntry {
  timestamp: string;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
}
