'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  processVideo,
  getProgress,
  cancelTask,
  onTaskProgress,
  onTaskCompleted,
} from '@/lib/tauri';
import type { ProgressInfo, LogEntry } from '@/types';

interface UseTaskResult {
  currentTask: ProgressInfo | null;
  logs: LogEntry[];
  isProcessing: boolean;
  startTask: (input: string, output?: string, config?: string) => Promise<string | null>;
  cancelCurrentTask: () => Promise<void>;
  addLog: (level: LogEntry['level'], message: string) => void;
}

export function useTask(): UseTaskResult {
  const [currentTask, setCurrentTask] = useState<ProgressInfo | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const taskIdRef = useRef<string | null>(null);

  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, { timestamp, level, message }]);
  }, []);

  // Set up event listeners
  useEffect(() => {
    let unlistenProgress: (() => void) | undefined;
    let unlistenCompleted: (() => void) | undefined;

    const setup = async () => {
      unlistenProgress = await onTaskProgress((info) => {
        setCurrentTask(info);
        if (info.status === 'Failed' || info.status === 'Cancelled') {
          setIsProcessing(false);
          addLog(info.status === 'Failed' ? 'error' : 'warn', info.error ?? info.stage);
        } else if (info.status === 'Running') {
          addLog('info', info.stage);
        }
      });

      unlistenCompleted = await onTaskCompleted((taskId) => {
        addLog('info', `Task ${taskId.substring(0, 8)} completed`);
        setIsProcessing(false);
        // Fetch final progress
        getProgress(taskId)
          .then((info) => setCurrentTask(info))
          .catch(() => {});
      });
    };

    setup();

    return () => {
      unlistenProgress?.();
      unlistenCompleted?.();
    };
  }, [addLog]);

  const startTask = useCallback(
    async (input: string, output?: string, config?: string): Promise<string | null> => {
      try {
        addLog('info', `Starting processing: ${input}`);
        setIsProcessing(true);
        const taskId = await processVideo(input, output, config);
        taskIdRef.current = taskId;
        addLog('info', `Task created: ${taskId.substring(0, 8)}`);

        // Poll for initial progress
        const info = await getProgress(taskId);
        setCurrentTask(info);

        return taskId;
      } catch (err) {
        addLog('error', `Failed to start task: ${err}`);
        setIsProcessing(false);
        return null;
      }
    },
    [addLog]
  );

  const cancelCurrentTask = useCallback(async () => {
    const taskId = taskIdRef.current;
    if (!taskId) return;

    try {
      await cancelTask(taskId);
      addLog('warn', `Cancelling task ${taskId.substring(0, 8)}...`);
      setIsProcessing(false);
    } catch (err) {
      addLog('error', `Failed to cancel task: ${err}`);
    }
  }, [addLog]);

  return {
    currentTask,
    logs,
    isProcessing,
    startTask,
    cancelCurrentTask,
    addLog,
  };
}
