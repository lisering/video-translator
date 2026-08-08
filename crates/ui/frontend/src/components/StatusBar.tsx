'use client';

import { useEffect, useState } from 'react';
import type { ProgressInfo } from '@/types';

interface StatusBarProps {
  currentTask: ProgressInfo | null;
}

export default function StatusBar({ currentTask }: StatusBarProps) {
  const [currentTime, setCurrentTime] = useState<string>('');

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setCurrentTime(now.toLocaleTimeString());
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  const statusText = currentTask
    ? `${currentTask.status} — ${currentTask.stage}`
    : 'Idle';

  const progressText = currentTask
    ? `${Math.round(currentTask.progress * 100)}%`
    : '—';

  return (
    <div className="flex items-center justify-between px-4 py-1.5 bg-gray-100 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
      <div className="flex items-center gap-4">
        <span className="flex items-center gap-1">
          <span
            className={`inline-block w-1.5 h-1.5 rounded-full ${
              currentTask?.status === 'Running'
                ? 'bg-green-500 animate-pulse'
                : currentTask?.status === 'Failed'
                ? 'bg-red-500'
                : 'bg-gray-400'
            }`}
          />
          {statusText}
        </span>
        {currentTask && (
          <span className="font-mono">Progress: {progressText}</span>
        )}
      </div>
      <div className="flex items-center gap-4">
        <span>{currentTime}</span>
        <span>Video Translator v0.1.0</span>
      </div>
    </div>
  );
}
