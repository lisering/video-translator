'use client';

import { useState, useCallback } from 'react';
import VideoSelector from '@/components/VideoSelector';
import ConfigPanel from '@/components/ConfigPanel';
import ProgressDisplay from '@/components/ProgressDisplay';
import LogViewer from '@/components/LogViewer';
import StatusBar from '@/components/StatusBar';
import { useTask } from '@/hooks/useTask';
import { loadConfig } from '@/lib/tauri';
import type { VideoInfo, AppConfig } from '@/types';

export default function HomePage() {
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  const [config, setConfig] = useState<AppConfig | null>(null);

  const { currentTask, logs, isProcessing, startTask, cancelCurrentTask, addLog } = useTask();

  const handleVideoSelected = useCallback((path: string, info: VideoInfo | null) => {
    setVideoPath(path);
    setVideoInfo(info);
  }, []);

  const handleConfigChange = useCallback((newConfig: AppConfig | null) => {
    setConfig(newConfig);
  }, []);

  const handleStartProcessing = useCallback(async () => {
    if (!videoPath) return;

    // Load config JSON to pass to the command
    let configJson: string | undefined;
    try {
      configJson = await loadConfig();
    } catch {
      configJson = undefined;
    }

    await startTask(videoPath, undefined, configJson);
  }, [videoPath, startTask]);

  const handleCancel = useCallback(async () => {
    await cancelCurrentTask();
  }, [cancelCurrentTask]);

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    if (newMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🎬</span>
            <h1 className="text-lg font-bold text-gray-800 dark:text-gray-200">
              Video Translator
            </h1>
          </div>
          <span className="text-xs text-gray-400">v0.1.0</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            onClick={toggleDarkMode}
            title="Toggle theme"
          >
            {darkMode ? (
              <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            ) : (
              <svg className="h-5 w-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            )}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto px-6 py-4 space-y-4 max-w-4xl mx-auto w-full">
        {/* Video Selection */}
        <section>
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            1. Select Video File
          </h2>
          <VideoSelector
            onVideoSelected={handleVideoSelected}
            disabled={isProcessing}
          />
        </section>

        {/* Configuration */}
        <section>
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            2. Configure Settings
          </h2>
          <ConfigPanel onConfigChange={handleConfigChange} />
        </section>

        {/* Action Buttons */}
        <section className="flex items-center gap-3">
          <button
            className="flex-1 px-6 py-3 rounded-lg bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            onClick={handleStartProcessing}
            disabled={!videoPath || isProcessing}
          >
            {isProcessing ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Processing...
              </>
            ) : (
              <>
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Start Processing
              </>
            )}
          </button>

          {isProcessing && (
            <button
              className="px-6 py-3 rounded-lg bg-red-600 text-white font-medium hover:bg-red-700 transition-colors flex items-center gap-2"
              onClick={handleCancel}
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Cancel
            </button>
          )}
        </section>

        {/* Progress Display */}
        {currentTask && (
          <section>
            <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Progress
            </h2>
            <ProgressDisplay info={currentTask} />
          </section>
        )}

        {/* Log Viewer */}
        <section>
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Logs
          </h2>
          <LogViewer logs={logs} />
        </section>

        {/* Config Debug Info (only in dev) */}
        {process.env.NODE_ENV === 'development' && config && (
          <details className="text-xs text-gray-400">
            <summary className="cursor-pointer">Current Config (debug)</summary>
            <pre className="mt-2 p-2 rounded bg-gray-100 dark:bg-gray-800 overflow-x-auto">
              {JSON.stringify(config, null, 2)}
            </pre>
          </details>
        )}
      </main>

      {/* Status Bar */}
      <StatusBar currentTask={currentTask} />
    </div>
  );
}
