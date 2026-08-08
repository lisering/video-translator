'use client';

import { useState, useCallback, useRef } from 'react';
import { openFileDialog, probeVideo } from '@/lib/tauri';
import type { VideoInfo } from '@/types';

interface VideoSelectorProps {
  onVideoSelected: (path: string, info: VideoInfo | null) => void;
  disabled: boolean;
}

export default function VideoSelector({ onVideoSelected, disabled }: VideoSelectorProps) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback(async () => {
    if (disabled) return;
    setLoading(true);
    setError(null);
    try {
      const path = await openFileDialog();
      if (path) {
        setSelectedPath(path);
        try {
          const info = await probeVideo(path);
          setVideoInfo(info);
          onVideoSelected(path, info);
        } catch (err) {
          setVideoInfo(null);
          onVideoSelected(path, null);
          setError(`Failed to probe video: ${err}`);
        }
      }
    } catch (err) {
      setError(`Failed to open file dialog: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [disabled, onVideoSelected]);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full">
      <div
        className={`
          drag-area border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          ${isDragging ? 'dragging' : ''}
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-primary-400 hover:bg-gray-50 dark:hover:bg-gray-800'}
          border-gray-300 dark:border-gray-600
        `}
        onClick={() => !disabled && handleFileSelect()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (disabled) return;
        }}
      >
        {loading ? (
          <div className="flex items-center justify-center gap-2 text-gray-500">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span>Loading...</span>
          </div>
        ) : selectedPath ? (
          <div className="space-y-2">
            <div className="flex items-center justify-center gap-2 text-green-600 dark:text-green-400">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="font-medium">Video Selected</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 break-all">{selectedPath}</p>
            {videoInfo && (
              <div className="flex flex-wrap justify-center gap-4 text-xs text-gray-500 dark:text-gray-400 mt-3">
                <span>Duration: {formatDuration(videoInfo.duration)}</span>
                {videoInfo.width && videoInfo.height && (
                  <span>Resolution: {videoInfo.width}x{videoInfo.height}</span>
                )}
                {videoInfo.video_codec && <span>Video: {videoInfo.video_codec}</span>}
                {videoInfo.audio_codec && <span>Audio: {videoInfo.audio_codec}</span>}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <p className="text-gray-600 dark:text-gray-400">
              Click to select or drag a video file here
            </p>
            <p className="text-xs text-gray-400">Supported: MP4, MKV, AVI, MOV, WebM, FLV, M4V</p>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-2 rounded-md bg-red-50 dark:bg-red-900/20 p-3 text-sm text-red-700 dark:text-red-400">
          {error}
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept="video/*,.mp4,.mkv,.avi,.mov,.webm,.flv,.m4v"
        onChange={() => {}}
      />
    </div>
  );
}
