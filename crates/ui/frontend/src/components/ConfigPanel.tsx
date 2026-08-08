'use client';

import { useState, useEffect, useCallback } from 'react';
import { loadConfig, saveConfig, listTtsVoices } from '@/lib/tauri';
import type { AppConfig, VoiceInfo } from '@/types';

interface ConfigPanelProps {
  onConfigChange?: (config: AppConfig | null) => void;
}

const DEFAULT_CONFIG: AppConfig = {
  asr: {
    model: 'whisper-large-v3',
    use_metal: true,
    language: 'en',
  },
  tts: {
    engine: 'say',
    speed: 1.0,
    pitch: 1.0,
    volume: 1.0,
    voice_id: 'tingting',
    voice: 'Tingting',
    sample_rate: 24000,
    device: 'cpu',
    cache_dir: '~/.cache/video-translator/tts_cache',
    parallel_tasks: 4,
    model_variant: 'v1.1-onnx',
    model_path: null,
    fallback_to_say: true,
    auto_voice_selection: false,
    seed: 42,
    temperature: 0.3,
    stability: 0.8,
    eq_high_shelf_db: -3.0,
    crossfade_duration_ms: 50,
  },
  translation: {
    glossary_path: null,
    batch_size: 10,
  },
  output_dir: './output',
  max_concurrent_tasks: 4,
  pipeline: {
    segment_duration_secs: 30.0,
    channel_capacity: 100,
    enable_vad_split: true,
  },
};

export default function ConfigPanel({ onConfigChange }: ConfigPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [config, setConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);

  useEffect(() => {
    let mounted = true;
    loadConfig()
      .then((json) => {
        if (mounted) {
          try {
            const parsed = JSON.parse(json) as AppConfig;
            setConfig(parsed);
            onConfigChange?.(parsed);
          } catch {
            // Use default config
          }
        }
      })
      .catch(() => {
        // Use default config
      });
    listTtsVoices()
      .then((v) => {
        if (mounted) setVoices(v);
      })
      .catch(() => {
        // Voices not available (non-macOS or engine error)
      });
    return () => {
      mounted = false;
    };
  }, [onConfigChange]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaveStatus(null);
    try {
      await saveConfig(JSON.stringify(config, null, 2));
      setSaveStatus('Saved successfully');
      setTimeout(() => setSaveStatus(null), 2000);
    } catch (err) {
      setSaveStatus(`Save failed: ${err}`);
    } finally {
      setSaving(false);
    }
  }, [config]);

  const updateField = (path: string, value: unknown) => {
    const newConfig = structuredClone(config);
    const keys = path.split('.');
    let obj: Record<string, unknown> = newConfig as unknown as Record<string, unknown>;
    for (let i = 0; i < keys.length - 1; i++) {
      obj = obj[keys[i]] as Record<string, unknown>;
    }
    obj[keys[keys.length - 1]] = value;
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <svg className="h-5 w-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          <span className="font-medium text-gray-700 dark:text-gray-300">Configuration</span>
        </div>
        <svg
          className={`h-5 w-5 text-gray-400 transform transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="p-4 space-y-4 animate-fade-in">
          {/* ASR Configuration */}
          <section>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">ASR (Speech Recognition)</h3>
            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="text-xs text-gray-500">Model</span>
                <select
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.asr.model}
                  onChange={(e) => updateField('asr.model', e.target.value)}
                >
                  <option value="whisper-large-v3">whisper-large-v3</option>
                  <option value="whisper-medium">whisper-medium</option>
                  <option value="whisper-small">whisper-small</option>
                  <option value="whisper-base">whisper-base</option>
                  <option value="whisper-tiny">whisper-tiny</option>
                </select>
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Source Language</span>
                <select
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.asr.language}
                  onChange={(e) => updateField('asr.language', e.target.value)}
                >
                  <option value="en">English</option>
                  <option value="ja">Japanese</option>
                  <option value="ko">Korean</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                </select>
              </label>
              <label className="flex items-center gap-2 col-span-2">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={config.asr.use_metal}
                  onChange={(e) => updateField('asr.use_metal', e.target.checked)}
                />
                <span className="text-sm text-gray-600 dark:text-gray-400">Use Metal GPU acceleration</span>
              </label>
            </div>
          </section>

          {/* Translation Configuration */}
          <section>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Translation (Local Offline Engine)</h3>
            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="text-xs text-gray-500">Batch Size</span>
                <input
                  type="number"
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.translation.batch_size}
                  onChange={(e) => updateField('translation.batch_size', parseInt(e.target.value, 10) || 10)}
                />
              </label>
              <label className="block col-span-2">
                <span className="text-xs text-gray-500">Glossary Path (optional)</span>
                <input
                  type="text"
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  placeholder="Path to glossary file (JSON/CSV)"
                  value={config.translation.glossary_path ?? ''}
                  onChange={(e) => updateField('translation.glossary_path', e.target.value || null)}
                />
              </label>
            </div>
          </section>

          {/* TTS Configuration */}
          <section>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">TTS (Text-to-Speech)</h3>
            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="text-xs text-gray-500">Engine</span>
                <select
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.tts.engine}
                  onChange={(e) => updateField('tts.engine', e.target.value)}
                >
                  <option value="say">Say (macOS Built-in)</option>
                  <option value="kokoro">Kokoro-82M ONNX</option>
                </select>
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Device</span>
                <select
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.tts.device}
                  onChange={(e) => updateField('tts.device', e.target.value)}
                >
                  <option value="cpu">CPU</option>
                  <option value="metal">Metal GPU</option>
                </select>
              </label>
              <label className="block col-span-2">
                <span className="text-xs text-gray-500">Voice</span>
                <select
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.tts.voice_id}
                  onChange={(e) => {
                    const selected = voices.find((v) => v.id === e.target.value);
                    updateField('tts.voice_id', e.target.value);
                    if (selected) {
                      updateField('tts.voice', selected.name);
                    }
                  }}
                >
                  {voices.length === 0 ? (
                    <>
                      <option value="tingting">婷婷 (Tingting) - Female</option>
                      <option value="meijia">美佳 (Meijia) - Female</option>
                      <option value="zhiming">志明 (Zhiming) - Male</option>
                      <option value="weiqiang">伟强 (Weiqiang) - Male</option>
                    </>
                  ) : (
                    voices.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.name} ({v.gender === 'male' ? '男' : v.gender === 'female' ? '女' : '中性'})
                      </option>
                    ))
                  )}
                </select>
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Speed ({config.tts.speed.toFixed(1)}x)</span>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  className="mt-2 w-full"
                  value={config.tts.speed}
                  onChange={(e) => updateField('tts.speed', parseFloat(e.target.value))}
                />
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Pitch ({config.tts.pitch.toFixed(2)}x)</span>
                <input
                  type="range"
                  min="0.8"
                  max="1.2"
                  step="0.05"
                  className="mt-2 w-full"
                  value={config.tts.pitch}
                  onChange={(e) => updateField('tts.pitch', parseFloat(e.target.value))}
                />
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Volume ({config.tts.volume.toFixed(1)}x)</span>
                <input
                  type="range"
                  min="0.0"
                  max="2.0"
                  step="0.1"
                  className="mt-2 w-full"
                  value={config.tts.volume}
                  onChange={(e) => updateField('tts.volume', parseFloat(e.target.value))}
                />
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Sample Rate</span>
                <select
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.tts.sample_rate}
                  onChange={(e) => updateField('tts.sample_rate', parseInt(e.target.value, 10))}
                >
                  <option value={16000}>16 kHz</option>
                  <option value={24000}>24 kHz</option>
                  <option value={48000}>48 kHz</option>
                </select>
              </label>
              <label className="flex items-center gap-2 col-span-2">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={config.tts.fallback_to_say}
                  onChange={(e) => updateField('tts.fallback_to_say', e.target.checked)}
                />
                <span className="text-sm text-gray-600 dark:text-gray-400">Fallback to Say engine if Kokoro unavailable</span>
              </label>
            </div>
            {/* Advanced TTS Settings */}
            <details className="mt-2">
              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700 dark:hover:text-gray-300">Advanced Settings (EQ, Temperature, Stability)</summary>
              <div className="grid grid-cols-2 gap-3 mt-2">
                <label className="block">
                  <span className="text-xs text-gray-500">Temperature ({config.tts.temperature.toFixed(2)})</span>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    className="mt-2 w-full"
                    value={config.tts.temperature}
                    onChange={(e) => updateField('tts.temperature', parseFloat(e.target.value))}
                  />
                </label>
                <label className="block">
                  <span className="text-xs text-gray-500">Stability ({config.tts.stability.toFixed(2)})</span>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    className="mt-2 w-full"
                    value={config.tts.stability}
                    onChange={(e) => updateField('tts.stability', parseFloat(e.target.value))}
                  />
                </label>
                <label className="block">
                  <span className="text-xs text-gray-500">Sibilance Reduction ({config.tts.eq_high_shelf_db.toFixed(1)} dB)</span>
                  <input
                    type="range"
                    min="-10"
                    max="0"
                    step="0.5"
                    className="mt-2 w-full"
                    value={config.tts.eq_high_shelf_db}
                    onChange={(e) => updateField('tts.eq_high_shelf_db', parseFloat(e.target.value))}
                  />
                </label>
                <label className="block">
                  <span className="text-xs text-gray-500">Crossfade ({config.tts.crossfade_duration_ms} ms)</span>
                  <input
                    type="range"
                    min="0"
                    max="200"
                    step="10"
                    className="mt-2 w-full"
                    value={config.tts.crossfade_duration_ms}
                    onChange={(e) => updateField('tts.crossfade_duration_ms', parseInt(e.target.value, 10))}
                  />
                </label>
              </div>
            </details>
          </section>

          {/* Pipeline Configuration */}
          <section>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Pipeline</h3>
            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="text-xs text-gray-500">Segment Duration (sec)</span>
                <input
                  type="number"
                  step="5"
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.pipeline.segment_duration_secs}
                  onChange={(e) => updateField('pipeline.segment_duration_secs', parseFloat(e.target.value) || 30)}
                />
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Output Directory</span>
                <input
                  type="text"
                  className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 text-sm"
                  value={config.output_dir}
                  onChange={(e) => updateField('output_dir', e.target.value)}
                />
              </label>
              <label className="flex items-center gap-2 col-span-2">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={config.pipeline.enable_vad_split}
                  onChange={(e) => updateField('pipeline.enable_vad_split', e.target.checked)}
                />
                <span className="text-sm text-gray-600 dark:text-gray-400">Enable VAD-based audio splitting</span>
              </label>
            </div>
          </section>

          {/* Save Button */}
          <div className="flex items-center gap-3 pt-2 border-t border-gray-200 dark:border-gray-700">
            <button
              className="px-4 py-2 rounded-md bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 disabled:opacity-50"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? 'Saving...' : 'Save Config'}
            </button>
            {saveStatus && (
              <span className={`text-sm ${saveStatus.includes('failed') ? 'text-red-500' : 'text-green-500'}`}>
                {saveStatus}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
