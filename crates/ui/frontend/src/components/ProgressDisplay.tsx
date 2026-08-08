'use client';

import type { ProgressInfo, TaskStatus } from '@/types';

interface ProgressDisplayProps {
  info: ProgressInfo | null;
}

const STATUS_COLORS: Record<TaskStatus, string> = {
  Pending: 'bg-gray-400',
  Running: 'bg-blue-500',
  Completed: 'bg-green-500',
  Failed: 'bg-red-500',
  Cancelled: 'bg-yellow-500',
};

const STATUS_LABELS: Record<TaskStatus, string> = {
  Pending: 'Pending',
  Running: 'Running',
  Completed: 'Completed',
  Failed: 'Failed',
  Cancelled: 'Cancelled',
};

export default function ProgressDisplay({ info }: ProgressDisplayProps) {
  if (!info) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 text-center text-gray-400 text-sm">
        No active task
      </div>
    );
  }

  const percent = Math.round(info.progress * 100);
  const statusColor = STATUS_COLORS[info.status];
  const statusLabel = STATUS_LABELS[info.status];

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 space-y-3">
      {/* Status Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2.5 h-2.5 rounded-full ${statusColor} ${info.status === 'Running' ? 'animate-pulse' : ''}`} />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{statusLabel}</span>
        </div>
        <span className="text-xs text-gray-400 font-mono">
          Task: {info.task_id.substring(0, 8)}...
        </span>
      </div>

      {/* Progress Bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
          <span>{info.stage}</span>
          <span>{percent}%</span>
        </div>
        <div className="w-full h-2.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${statusColor} rounded-full transition-all duration-300 ease-out`}
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>

      {/* Error Display */}
      {info.error && (
        <div className="rounded-md bg-red-50 dark:bg-red-900/20 p-3 text-sm text-red-700 dark:text-red-400">
          <div className="flex items-start gap-2">
            <svg className="h-4 w-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>{info.error}</span>
          </div>
        </div>
      )}
    </div>
  );
}
