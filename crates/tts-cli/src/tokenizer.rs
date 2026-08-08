//! 文本分词器模块
//!
//! 参考 Qwen3-TTS 的 TextTokenizer 设计。
//! 使用 HuggingFace tokenizers 库进行 BPE 分词。

use std::path::Path;

use anyhow::Result;

/// 文本分词器
///
/// 封装 HuggingFace tokenizers，将文本转换为模型可处理的 token IDs。
pub struct TextTokenizer {
    /// HuggingFace tokenizer（None = fallback 模式）
    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    tokenizer: Option<tokenizers::Tokenizer>,
    /// 模型路径
    model_path: std::path::PathBuf,
    /// 是否为 fallback 模式（无 tokenizer.json）
    is_fallback: bool,
}

impl TextTokenizer {
    /// 从文件加载分词器
    ///
    /// # 参数
    /// - `path`: tokenizer.json 文件路径
    pub fn from_file(path: &Path) -> Result<Self> {
        #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
        {
            let tokenizer = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", path, e))?;
            Ok(Self {
                tokenizer: Some(tokenizer),
                model_path: path.to_path_buf(),
                is_fallback: false,
            })
        }

        #[cfg(not(any(feature = "cpu", feature = "metal", feature = "cuda")))]
        {
            Ok(Self {
                model_path: path.to_path_buf(),
                is_fallback: false,
            })
        }
    }

    /// 创建 fallback 分词器（无模型文件时使用）
    ///
    /// 使用简单的 UTF-8 字节级分词：每个字节对应一个 token ID。
    pub fn fallback() -> Self {
        Self {
            #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
            tokenizer: None,
            model_path: std::path::PathBuf::new(),
            is_fallback: true,
        }
    }

    /// 从模型目录加载
    ///
    /// 按以下顺序查找分词器文件：
    /// 1. `tokenizer.json` — HuggingFace 完整分词器（首选）
    /// 2. `vocab.json` + `merges.txt` — GPT-2/Qwen 风格 BPE 分词器
    /// 3. fallback 字节级分词器
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        // 1. 尝试 tokenizer.json
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            tracing::info!("Loading tokenizer from tokenizer.json");
            return Self::from_file(&tokenizer_path);
        }

        // 2. 尝试 vocab.json + merges.txt (BPE)
        #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
        {
            let vocab_path = model_dir.join("vocab.json");
            let merges_path = model_dir.join("merges.txt");
            if vocab_path.exists() && merges_path.exists() {
                tracing::info!("Building BPE tokenizer from vocab.json + merges.txt");
                match Self::from_bpe_files(&vocab_path, &merges_path, model_dir) {
                    Ok(t) => {
                        tracing::info!("BPE tokenizer loaded successfully");
                        return Ok(t);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to build BPE from vocab.json + merges.txt: {}. Using fallback.",
                            e
                        );
                    }
                }
            }
        }

        // 3. Fallback
        tracing::warn!(
            "No tokenizer files found in {:?}, using byte-level fallback",
            model_dir
        );
        Ok(Self::fallback())
    }

    /// 从 vocab.json + merges.txt 构建 GPT-2/Qwen 风格 BPE 分词器
    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    fn from_bpe_files(vocab_path: &Path, merges_path: &Path, model_dir: &Path) -> Result<Self> {
        use tokenizers::models::bpe::BPE;
        use tokenizers::Tokenizer;

        let vocab_str = vocab_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 in vocab path"))?;
        let merges_str = merges_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 in merges path"))?;

        // 构建 BPE 模型
        let bpe = BPE::from_file(vocab_str, merges_str)
            .build()
            .map_err(|e| anyhow::anyhow!("BPE build failed: {}", e))?;

        let mut tokenizer = Tokenizer::new(bpe);

        // GPT-2/Qwen 风格: ByteLevel 预分词器和解码器
        let pre_tokenizer = tokenizers::pre_tokenizers::byte_level::ByteLevel::new(
            false, // add_prefix_space
            false, // trim_offsets
            true,  // use_regex
        );
        let pt: tokenizers::pre_tokenizers::PreTokenizerWrapper = pre_tokenizer.into();
        tokenizer.with_pre_tokenizer(Some(pt));

        let decoder = tokenizers::decoders::byte_level::ByteLevel::new(false, false, true);
        let dec: tokenizers::decoders::DecoderWrapper = decoder.into();
        tokenizer.with_decoder(Some(dec));

        // 从 tokenizer_config.json 加载特殊 token
        let config_path = model_dir.join("tokenizer_config.json");
        if config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                    // 添加特殊 token
                    if let Some(added_tokens) = config
                        .get("added_tokens_decoder")
                        .and_then(|v| v.as_object())
                    {
                        let mut tokens_to_add = Vec::new();
                        for (_id, token_info) in added_tokens {
                            if let Some(content) =
                                token_info.get("content").and_then(|v| v.as_str())
                            {
                                tokens_to_add.push(tokenizers::AddedToken {
                                    content: content.to_string(),
                                    single_word: false,
                                    lstrip: false,
                                    rstrip: false,
                                    normalized: false,
                                    special: true,
                                });
                            }
                        }
                        if !tokens_to_add.is_empty() {
                            let count = tokens_to_add.len();
                            tokenizer.add_special_tokens(&tokens_to_add);
                            tracing::info!(
                                "Added {} special tokens from tokenizer_config.json",
                                count
                            );
                        }
                    }
                }
            }
        }

        // 验证分词器
        let vocab_size = tokenizer.get_vocab_size(true);
        tracing::info!("BPE tokenizer ready: vocab_size={}", vocab_size);

        // 测试编码
        if let Ok(encoding) = tokenizer.encode("hello", true) {
            tracing::debug!(
                "Tokenizer test: \"hello\" -> {:?} ({} tokens)",
                encoding.get_ids(),
                encoding.get_ids().len()
            );
        }

        Ok(Self {
            tokenizer: Some(tokenizer),
            model_path: vocab_path.to_path_buf(),
            is_fallback: false,
        })
    }

    /// 将文本编码为 token IDs
    ///
    /// # 参数
    /// - `text`: 输入文本
    ///
    /// # 返回
    /// token IDs 列表
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
        {
            if let Some(ref tokenizer) = self.tokenizer {
                let encoding = tokenizer
                    .encode(text, true)
                    .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
                return Ok(encoding.get_ids().to_vec());
            }
            // Fallback: UTF-8 字节级编码
            Ok(text.as_bytes().iter().map(|b| *b as u32).collect())
        }

        #[cfg(not(any(feature = "cpu", feature = "metal", feature = "cuda")))]
        {
            Ok(text.as_bytes().iter().map(|b| *b as u32).collect())
        }
    }

    /// 将文本编码为 token IDs（带特殊 token）
    ///
    /// 添加 BOS/EOS 特殊 token，用于 TTS 模型输入。
    pub fn encode_for_tts(&self, text: &str) -> Result<Vec<u32>> {
        // 直接调用 encode，fallback 模式下不添加特殊 token
        self.encode(text)
    }

    /// 解码 token IDs 为文本
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
        {
            if let Some(ref tokenizer) = self.tokenizer {
                return tokenizer
                    .decode(ids, true)
                    .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e));
            }
            // Fallback: 字节级解码
            let bytes: Vec<u8> = ids.iter().map(|id| *id as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).to_string())
        }

        #[cfg(not(any(feature = "cpu", feature = "metal", feature = "cuda")))]
        {
            let bytes: Vec<u8> = ids.iter().map(|id| *id as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).to_string())
        }
    }

    /// 词表大小
    pub fn vocab_size(&self) -> usize {
        #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
        {
            if let Some(ref tokenizer) = self.tokenizer {
                return tokenizer.get_vocab_size(true);
            }
            256 // Fallback: UTF-8 字节
        }

        #[cfg(not(any(feature = "cpu", feature = "metal", feature = "cuda")))]
        {
            256
        }
    }

    /// 是否为 fallback 模式
    pub fn is_fallback(&self) -> bool {
        self.is_fallback
    }
}

impl std::fmt::Debug for TextTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextTokenizer")
            .field("model_path", &self.model_path)
            .field("is_fallback", &self.is_fallback)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Fallback 模式测试 ───

    #[test]
    fn test_fallback_creation() {
        let tok = TextTokenizer::fallback();
        assert!(tok.is_fallback());
        assert_eq!(tok.vocab_size(), 256);
    }

    #[test]
    fn test_fallback_encode_ascii() {
        let tok = TextTokenizer::fallback();
        let ids = tok.encode("hello").unwrap();
        assert_eq!(ids, vec![104, 101, 108, 108, 111]); // ASCII bytes
    }

    #[test]
    fn test_fallback_decode_ascii() {
        let tok = TextTokenizer::fallback();
        let ids = vec![104, 101, 108, 108, 111];
        let text = tok.decode(&ids).unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_fallback_roundtrip_ascii() {
        let tok = TextTokenizer::fallback();
        let texts = ["hello", "world", "test 123", "a", ""];
        for text in &texts {
            let ids = tok.encode(text).unwrap();
            let decoded = tok.decode(&ids).unwrap();
            assert_eq!(&decoded, text, "Roundtrip failed for: {:?}", text);
        }
    }

    #[test]
    fn test_fallback_encode_unicode() {
        let tok = TextTokenizer::fallback();
        // 中文字符的 UTF-8 编码
        let ids = tok.encode("你").unwrap();
        assert_eq!(ids, vec![0xE4u32, 0xBD, 0xA0]); // UTF-8 bytes
    }

    #[test]
    fn test_fallback_roundtrip_unicode() {
        let tok = TextTokenizer::fallback();
        let texts = ["你好", "こんにちは", "안녕하세요", "mix 中文 english"];
        for text in &texts {
            let ids = tok.encode(text).unwrap();
            let decoded = tok.decode(&ids).unwrap();
            assert_eq!(&decoded, text, "Unicode roundtrip failed for: {:?}", text);
        }
    }

    #[test]
    fn test_fallback_encode_empty() {
        let tok = TextTokenizer::fallback();
        let ids = tok.encode("").unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_fallback_decode_empty() {
        let tok = TextTokenizer::fallback();
        let text = tok.decode(&[]).unwrap();
        assert_eq!(text, "");
    }

    #[test]
    fn test_fallback_encode_special_chars() {
        let tok = TextTokenizer::fallback();
        let text = "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`";
        let ids = tok.encode(text).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(&decoded, text);
    }

    #[test]
    fn test_fallback_vocab_size() {
        let tok = TextTokenizer::fallback();
        assert_eq!(tok.vocab_size(), 256);
    }

    // ─── encode_for_tts 测试 ───

    #[test]
    fn test_encode_for_tts_fallback() {
        let tok = TextTokenizer::fallback();
        let ids = tok.encode_for_tts("test").unwrap();
        assert_eq!(ids, vec![116, 101, 115, 116]);
    }
}
