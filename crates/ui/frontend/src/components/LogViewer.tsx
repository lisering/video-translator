'use client';

import { useEffect, useRef } from 'react';
import type { LogEntry } from '@/types';

interface LogViewerProps {
  logs: LogEntry[];
}

const LEVEL_COLORS: Record<LogEntry['level'], string> = {
  info: 'text-blue-400',
  warn: 'text-yellow-400',
  error: 'text-red-400',
  debug: 'text-gray-400',
};

const LEVEL_ICONS: Record<LogEntry['level'], string> = {
  info: 'ℹ',
  warn: '⚠',
  error: '✖',
  debug: '·',
};

export default function LogViewer({ logs }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <svg className="h-4 w-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Logs</span>
        </div>
        <span className="text-xs text-gray-400">{logs.length} entries</span>
      </div>
      <div
        ref={containerRef}
        className="h-48 overflow-y-auto bg-gray-900 p-3 font-mono text-xs space-y-0.5"
      >
        {logs.length === 0 ? (
          <div className="text-gray-500 text-center py-4">No logs yet</div>
        ) : (
          logs.map((entry, index) => (
            <div key={index} className="flex items-start gap-2">
              <span className="text-gray-600 flex-shrink-0">
                {entry.timestamp}
              </span>
              <span className={`flex-shrink-0 ${LEVEL_COLORS[entry.level]}`}>
                {LEVEL_ICONS[entry.level]}
              </span>
              <span className="text-gray-300 break-all">
                {entry.message}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
