//! TTS 配音缓存模块
//!
//! 参考 pyvideotrans 的配音缓存逻辑，使用 MD5 哈希作为 key，
//! 将已合成的音频文件缓存到本地磁盘，避免重复合成。
//!
//! # 缓存 Key 计算
//! `md5(target_language-text-voice-speed-volume-pitch-engine_name)`
//!
//! # 缓存策略
//! - 文件级缓存：`{cache_dir}/{md5}.wav`
//! - 合成前检查缓存，命中则直接 copy
//! - 合成后写入缓存
//! - 支持缓存清理（按时间或大小）

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{AppError, AppResult};

/// 计算 MD5 哈希的辅助函数
fn md5_hex(input: &str) -> String {
    let digest = md5::compute(input.as_bytes());
    format!("{digest:x}")
}

/// TTS 配音缓存管理器
#[derive(Clone)]
pub struct TtsCache {
    /// 缓存目录
    cache_dir: PathBuf,
}

impl TtsCache {
    /// 创建缓存管理器
    ///
    /// 自动创建缓存目录
    pub fn new(cache_dir: impl AsRef<Path>) -> AppResult<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    /// 使用默认路径创建缓存管理器
    pub fn default() -> AppResult<Self> {
        let home = std::env::var("HOME")
            .map_err(|_| AppError::Config("HOME environment variable not set".to_string()))?;
        Self::new(format!("{home}/.cache/video-translator/tts_cache"))
    }

    /// 计算缓存 key
    ///
    /// key = md5(text + voice + speed + volume + pitch + engine_name)
    pub fn cache_key(
        text: &str,
        voice: &str,
        speed: f32,
        volume: f32,
        pitch: f32,
        engine_name: &str,
    ) -> String {
        let key_str = format!("{engine_name}-{voice}-{speed}-{volume}-{pitch}-{text}");
        md5_hex(&key_str)
    }

    /// 获取缓存文件路径
    pub fn cache_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{key}.wav"))
    }

    /// 检查缓存是否命中
    ///
    /// 返回缓存的 WAV 文件路径（如果存在）
    pub fn get(&self, key: &str) -> Option<PathBuf> {
        let path = self.cache_path(key);
        if path.exists() && path.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            tracing::debug!("TTS cache hit: {key}");
            Some(path)
        } else {
            None
        }
    }

    /// 写入缓存
    ///
    /// 将合成好的音频文件复制到缓存目录
    pub fn put(&self, source: &Path, key: &str) -> AppResult<()> {
        let dest = self.cache_path(key);
        fs::copy(source, &dest)
            .map_err(|e| AppError::TtsError(format!("Failed to cache TTS audio: {e}")))?;
        tracing::debug!("TTS cache stored: {key}");
        Ok(())
    }

    /// 清理过期缓存
    ///
    /// 删除超过 `max_age_days` 天的缓存文件
    pub fn clean_expired(&self, max_age_days: u32) -> AppResult<usize> {
        let max_age_secs = max_age_days as u64 * 86400;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut cleaned = 0;
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "wav") {
                continue;
            }
            if let Ok(metadata) = entry.metadata() {
                if let Ok(mtime) = metadata.modified() {
                    let age = now
                        - mtime
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                    if age > max_age_secs {
                        if fs::remove_file(&path).is_ok() {
                            cleaned += 1;
                        }
                    }
                }
            }
        }

        if cleaned > 0 {
            tracing::info!("TTS cache: cleaned {cleaned} expired files");
        }
        Ok(cleaned)
    }

    /// 获取缓存大小（字节）
    pub fn cache_size(&self) -> u64 {
        let mut total = 0u64;
        if !self.cache_dir.exists() {
            return 0;
        }
        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    total += metadata.len();
                }
            }
        }
        total
    }
}

// ─── 翻译缓存 ─────────────────────────────────────────────

/// 翻译缓存管理器
///
/// 使用 MD5 哈希作为 key，将翻译结果缓存到本地磁盘，
/// 避免相同输入重复调用 LLM。
#[derive(Clone)]
pub struct TranslationCache {
    /// 缓存目录
    cache_dir: PathBuf,
}

impl TranslationCache {
    /// 创建缓存管理器
    pub fn new(cache_dir: impl AsRef<Path>) -> AppResult<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    /// 使用默认路径创建
    pub fn default() -> AppResult<Self> {
        let home = std::env::var("HOME")
            .map_err(|_| AppError::Config("HOME environment variable not set".to_string()))?;
        Self::new(format!("{home}/.cache/video-translator/translate_cache"))
    }

    /// 计算缓存 key
    ///
    /// key = md5(backend_name-source_lang-target_lang-model_name-text)
    pub fn cache_key(
        backend_name: &str,
        source_lang: &str,
        target_lang: &str,
        model_name: &str,
        text: &str,
    ) -> String {
        let key_str = format!("{backend_name}-{source_lang}-{target_lang}-{model_name}-{text}");
        md5_hex(&key_str)
    }

    /// 获取缓存文件路径
    fn cache_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{key}.txt"))
    }

    /// 检查缓存是否命中
    pub fn get(&self, key: &str) -> Option<String> {
        let path = self.cache_path(key);
        if path.exists() {
            match fs::read_to_string(&path) {
                Ok(content) if !content.is_empty() => {
                    tracing::debug!("Translation cache hit: {key}");
                    Some(content)
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// 写入缓存
    pub fn put(&self, text: &str, key: &str) -> AppResult<()> {
        if text.is_empty() {
            return Ok(());
        }
        let path = self.cache_path(key);
        let mut file = fs::File::create(&path)?;
        file.write_all(text.as_bytes())?;
        tracing::debug!("Translation cache stored: {key}");
        Ok(())
    }

    /// 清理过期缓存
    pub fn clean_expired(&self, max_age_days: u32) -> AppResult<usize> {
        let max_age_secs = max_age_days as u64 * 86400;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut cleaned = 0;
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "txt") {
                continue;
            }
            if let Ok(metadata) = entry.metadata() {
                if let Ok(mtime) = metadata.modified() {
                    let age = now
                        - mtime
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                    if age > max_age_secs {
                        if fs::remove_file(&path).is_ok() {
                            cleaned += 1;
                        }
                    }
                }
            }
        }

        if cleaned > 0 {
            tracing::info!("Translation cache: cleaned {cleaned} expired files");
        }
        Ok(cleaned)
    }
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_cache_key_deterministic() {
        let key1 = TtsCache::cache_key("hello", "tingting", 1.0, 1.0, 1.0, "kokoro");
        let key2 = TtsCache::cache_key("hello", "tingting", 1.0, 1.0, 1.0, "kokoro");
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_tts_cache_key_different_text() {
        let key1 = TtsCache::cache_key("hello", "tingting", 1.0, 1.0, 1.0, "kokoro");
        let key2 = TtsCache::cache_key("world", "tingting", 1.0, 1.0, 1.0, "kokoro");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_tts_cache_key_different_engine() {
        let key1 = TtsCache::cache_key("hello", "tingting", 1.0, 1.0, 1.0, "kokoro");
        let key2 = TtsCache::cache_key("hello", "tingting", 1.0, 1.0, 1.0, "say");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_tts_cache_put_get() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let dir = tmp.path().parent().unwrap().join("tts_cache_test");
        let cache = TtsCache::new(&dir).unwrap();
        let key = "testkey123";

        // Create a dummy wav file
        let src = dir.join("source.wav");
        fs::write(&src, b"dummy audio data").unwrap();

        cache.put(&src, key).unwrap();
        let cached = cache.get(key);
        assert!(cached.is_some());
        assert!(cached.unwrap().exists());
    }

    #[test]
    fn test_tts_cache_miss() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let dir = tmp.path().parent().unwrap().join("tts_cache_miss_test");
        let cache = TtsCache::new(&dir).unwrap();
        assert!(cache.get("nonexistent_key").is_none());
    }

    #[test]
    fn test_translation_cache_key_deterministic() {
        let key1 = TranslationCache::cache_key("llama", "en", "zh", "qwen2.5", "hello");
        let key2 = TranslationCache::cache_key("llama", "en", "zh", "qwen2.5", "hello");
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_translation_cache_key_different_text() {
        let key1 = TranslationCache::cache_key("llama", "en", "zh", "qwen2.5", "hello");
        let key2 = TranslationCache::cache_key("llama", "en", "zh", "qwen2.5", "world");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_translation_cache_put_get() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let dir = tmp.path().parent().unwrap().join("trans_cache_test");
        let cache = TranslationCache::new(&dir).unwrap();
        let key = "transkey456";

        cache.put("你好世界", key).unwrap();
        let cached = cache.get(key);
        assert_eq!(cached, Some("你好世界".to_string()));
    }

    #[test]
    fn test_translation_cache_miss() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let dir = tmp.path().parent().unwrap().join("trans_cache_miss_test");
        let cache = TranslationCache::new(&dir).unwrap();
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_translation_cache_empty_text_not_stored() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let dir = tmp.path().parent().unwrap().join("trans_cache_empty_test");
        let cache = TranslationCache::new(&dir).unwrap();
        let key = "emptykey789";

        cache.put("", key).unwrap();
        assert!(cache.get(key).is_none());
    }
}

// ─── VoiceClonePrompt 持久化缓存 — 借鉴 OmniVoice prompt 缓存 ──

/// 声音克隆提示缓存
///
/// 借鉴 OmniVoice 的 `voice_clone_prompt` 缓存机制：
/// 将从参考音频提取的声音特征持久化到磁盘，
/// 下次使用同一参考音频时直接加载，跳过耗时的特征提取。
///
/// # 缓存 Key
/// 基于参考音频文件路径 + 文件修改时间 + 文件大小计算，
/// 确保参考音频变更时缓存自动失效。
///
/// # 缓存格式
/// - `{cache_dir}/{md5}.prompt` — 序列化的提示数据
/// - `{cache_dir}/{md5}.meta` — 元数据（JSON 格式）
#[derive(Clone)]
pub struct VoiceClonePromptCache {
    /// 缓存目录
    cache_dir: PathBuf,
}

/// 缓存元数据
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PromptCacheMeta {
    /// 参考音频原始路径
    pub source_path: String,
    /// 参考音频修改时间（Unix 时间戳）
    pub source_mtime: u64,
    /// 参考音频文件大小（字节）
    pub source_size: u64,
    /// 缓存创建时间
    pub cached_at: u64,
    /// 提示文本（如果有）
    pub prompt_text: Option<String>,
    /// 参考音频时长（秒）
    pub duration_secs: f64,
    /// 参考音频采样率
    pub sample_rate: u32,
}

impl VoiceClonePromptCache {
    /// 创建缓存管理器
    pub fn new(cache_dir: impl AsRef<Path>) -> AppResult<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    /// 使用默认路径
    pub fn default() -> AppResult<Self> {
        let home = std::env::var("HOME")
            .map_err(|_| AppError::Config("HOME environment variable not set".to_string()))?;
        Self::new(format!("{home}/.cache/video-translator/voice_prompt"))
    }

    /// 计算缓存 key
    ///
    /// key = md5(path + mtime + size)
    pub fn cache_key(source_path: &Path) -> AppResult<String> {
        let metadata = fs::metadata(source_path)?;
        let mtime = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let size = metadata.len();
        let key_str = format!("{}-{}-{}", source_path.display(), mtime, size);
        Ok(md5_hex(&key_str))
    }

    /// 获取缓存文件路径
    fn prompt_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{key}.prompt"))
    }

    /// 获取元数据文件路径
    fn meta_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{key}.meta"))
    }

    /// 检查缓存是否命中
    ///
    /// 返回缓存的元数据（如果存在且有效）
    pub fn get(&self, key: &str) -> Option<PromptCacheMeta> {
        let meta_path = self.meta_path(key);
        let prompt_path = self.prompt_path(key);

        if !meta_path.exists() || !prompt_path.exists() {
            return None;
        }

        // 读取元数据
        let meta_content = fs::read_to_string(&meta_path).ok()?;
        let meta: PromptCacheMeta = serde_json::from_str(&meta_content).ok()?;

        // 验证源文件是否仍然存在且未修改
        let source_path = Path::new(&meta.source_path);
        if let Ok(metadata) = fs::metadata(source_path) {
            let current_mtime = metadata
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let current_size = metadata.len();

            if current_mtime != meta.source_mtime || current_size != meta.source_size {
                tracing::debug!("VoiceClonePrompt cache miss (source changed): {key}");
                return None;
            }
        }

        tracing::debug!("VoiceClonePrompt cache hit: {key}");
        Some(meta)
    }

    /// 获取缓存的提示数据文件路径
    pub fn get_prompt_path(&self, key: &str) -> Option<PathBuf> {
        let path = self.prompt_path(key);
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    /// 写入缓存
    ///
    /// 将提示数据复制到缓存目录，并写入元数据
    pub fn put(&self, prompt_data_path: &Path, meta: &PromptCacheMeta, key: &str) -> AppResult<()> {
        let dest = self.prompt_path(key);
        fs::copy(prompt_data_path, &dest)
            .map_err(|e| AppError::TtsError(format!("Failed to cache voice prompt: {e}")))?;

        let meta_path = self.meta_path(key);
        let meta_json = serde_json::to_string_pretty(meta)
            .map_err(|e| AppError::TtsError(format!("Failed to serialize prompt meta: {e}")))?;
        fs::write(&meta_path, meta_json)?;

        tracing::debug!("VoiceClonePrompt cache stored: {key}");
        Ok(())
    }

    /// 删除缓存
    pub fn remove(&self, key: &str) -> AppResult<()> {
        let _ = fs::remove_file(self.prompt_path(key));
        let _ = fs::remove_file(self.meta_path(key));
        Ok(())
    }

    /// 清理过期缓存
    pub fn clean_expired(&self, max_age_days: u32) -> AppResult<usize> {
        let max_age_secs = max_age_days as u64 * 86400;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut cleaned = 0;
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .is_none_or(|ext| ext != "prompt" && ext != "meta")
            {
                continue;
            }
            if let Ok(metadata) = entry.metadata() {
                if let Ok(mtime) = metadata.modified() {
                    let age = now
                        - mtime
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                    if age > max_age_secs {
                        if fs::remove_file(&path).is_ok() {
                            cleaned += 1;
                        }
                    }
                }
            }
        }

        if cleaned > 0 {
            tracing::info!("VoiceClonePrompt cache: cleaned {cleaned} expired files");
        }
        Ok(cleaned)
    }

    /// 获取缓存大小（字节）
    pub fn cache_size(&self) -> u64 {
        let mut total = 0u64;
        if !self.cache_dir.exists() {
            return 0;
        }
        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    total += metadata.len();
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod omni_prompt_cache_tests {
    use super::*;

    #[test]
    fn test_prompt_cache_key_deterministic() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        fs::write(tmp.path(), b"test audio data").unwrap();
        let key1 = VoiceClonePromptCache::cache_key(tmp.path()).unwrap();
        let key2 = VoiceClonePromptCache::cache_key(tmp.path()).unwrap();
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_prompt_cache_key_changes_with_content() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        fs::write(tmp.path(), b"audio v1").unwrap();
        let key1 = VoiceClonePromptCache::cache_key(tmp.path()).unwrap();

        // 修改文件内容（需要确保 mtime 变化）
        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::write(tmp.path(), b"audio v2 with different content").unwrap();

        let key2 = VoiceClonePromptCache::cache_key(tmp.path()).unwrap();
        assert_ne!(key1, key2, "Cache key should change when file changes");
    }

    #[test]
    fn test_prompt_cache_put_get() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let cache = VoiceClonePromptCache::new(tmp_dir.path()).unwrap();

        // 创建模拟的参考音频
        let ref_audio = tmp_dir.path().join("ref.wav");
        fs::write(&ref_audio, b"reference audio data").unwrap();

        // 创建模拟的提示数据
        let prompt_data = tmp_dir.path().join("prompt.bin");
        fs::write(&prompt_data, b"prompt features data").unwrap();

        let key = VoiceClonePromptCache::cache_key(&ref_audio).unwrap();
        let meta = PromptCacheMeta {
            source_path: ref_audio.to_string_lossy().to_string(),
            source_mtime: fs::metadata(&ref_audio)
                .unwrap()
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            source_size: fs::metadata(&ref_audio).unwrap().len(),
            cached_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            prompt_text: Some("Hello world".to_string()),
            duration_secs: 5.0,
            sample_rate: 16000,
        };

        // 写入缓存
        cache.put(&prompt_data, &meta, &key).unwrap();

        // 读取缓存
        let cached_meta = cache.get(&key);
        assert!(cached_meta.is_some(), "Cache should hit");
        let cached = cached_meta.unwrap();
        assert_eq!(cached.prompt_text, Some("Hello world".to_string()));
        assert!((cached.duration_secs - 5.0).abs() < 0.01);

        // 获取提示数据路径
        let prompt_path = cache.get_prompt_path(&key);
        assert!(prompt_path.is_some());
        assert!(prompt_path.unwrap().exists());
    }

    #[test]
    fn test_prompt_cache_miss() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let cache = VoiceClonePromptCache::new(tmp_dir.path()).unwrap();
        assert!(cache.get("nonexistent_key").is_none());
    }

    #[test]
    fn test_prompt_cache_invalidated_on_change() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let cache = VoiceClonePromptCache::new(tmp_dir.path()).unwrap();

        let ref_audio = tmp_dir.path().join("ref.wav");
        fs::write(&ref_audio, b"reference v1").unwrap();

        let prompt_data = tmp_dir.path().join("prompt.bin");
        fs::write(&prompt_data, b"prompt data").unwrap();

        let key = VoiceClonePromptCache::cache_key(&ref_audio).unwrap();
        let meta = PromptCacheMeta {
            source_path: ref_audio.to_string_lossy().to_string(),
            source_mtime: fs::metadata(&ref_audio)
                .unwrap()
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            source_size: fs::metadata(&ref_audio).unwrap().len(),
            cached_at: 0,
            prompt_text: None,
            duration_secs: 3.0,
            sample_rate: 16000,
        };

        cache.put(&prompt_data, &meta, &key).unwrap();
        assert!(cache.get(&key).is_some());

        // 修改参考音频
        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::write(&ref_audio, b"reference v2 with different content").unwrap();

        // 缓存应该失效
        assert!(cache.get(&key).is_none(), "Cache should be invalidated");
    }

    #[test]
    fn test_prompt_cache_remove() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let cache = VoiceClonePromptCache::new(tmp_dir.path()).unwrap();

        let ref_audio = tmp_dir.path().join("ref.wav");
        fs::write(&ref_audio, b"ref data").unwrap();
        let prompt_data = tmp_dir.path().join("prompt.bin");
        fs::write(&prompt_data, b"prompt").unwrap();

        let key = VoiceClonePromptCache::cache_key(&ref_audio).unwrap();
        let meta = PromptCacheMeta {
            source_path: ref_audio.to_string_lossy().to_string(),
            source_mtime: fs::metadata(&ref_audio)
                .unwrap()
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            source_size: fs::metadata(&ref_audio).unwrap().len(),
            cached_at: 0,
            prompt_text: None,
            duration_secs: 1.0,
            sample_rate: 16000,
        };

        cache.put(&prompt_data, &meta, &key).unwrap();
        assert!(cache.get(&key).is_some());

        cache.remove(&key).unwrap();
        assert!(cache.get(&key).is_none());
    }
}
