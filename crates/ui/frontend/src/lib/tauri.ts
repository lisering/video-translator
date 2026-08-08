/// Tauri 后端 API 封装
///
/// 提供类型安全的 Tauri 命令调用和事件监听。

import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type { ProgressInfo, VideoInfo, AppConfig, VoiceInfo } from '@/types';

/** 启动视频处理任务 */
export async function processVideo(
  input: string,
  output?: string,
  config?: string
): Promise<string> {
  return invoke<string>('process_video', {
    input,
    output: output ?? null,
    configJson: config ?? null,
  });
}

/** 查询任务进度 */
export async function getProgress(taskId: string): Promise<ProgressInfo> {
  return invoke<ProgressInfo>('get_progress', { taskId });
}

/** 取消任务 */
export async function cancelTask(taskId: string): Promise<void> {
  await invoke('cancel_task', { taskId });
}

/** 列出所有任务 */
export async function listAllTasks(): Promise<ProgressInfo[]> {
  return invoke<ProgressInfo[]>('list_all_tasks');
}

/** 加载配置 */
export async function loadConfig(): Promise<string> {
  return invoke<string>('load_config');
}

/** 保存配置 */
export async function saveConfig(configJson: string): Promise<void> {
  await invoke('save_config', { configJson });
}

/** 探测视频文件信息 */
export async function probeVideo(path: string): Promise<VideoInfo> {
  return invoke<VideoInfo>('probe_video', { path });
}

/** 打开文件选择对话框 */
export async function openFileDialog(): Promise<string | null> {
  return invoke<string | null>('open_file_dialog');
}

/** 列出可用的 TTS 音色 */
export async function listTtsVoices(): Promise<VoiceInfo[]> {
  return invoke<VoiceInfo[]>('list_tts_voices');
}

/** 监听任务进度事件 */
export async function onTaskProgress(
  callback: (info: ProgressInfo) => void
): Promise<UnlistenFn> {
  return listen<ProgressInfo>('task-progress', (event) => {
    callback(event.payload);
  });
}

/** 监听任务完成事件 */
export async function onTaskCompleted(
  callback: (taskId: string) => void
): Promise<UnlistenFn> {
  return listen<string>('task-completed', (event) => {
    callback(event.payload);
  });
}
