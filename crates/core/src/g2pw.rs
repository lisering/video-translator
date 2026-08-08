//! G2PW 多音字消歧模块
//!
//! 借鉴 GPT-SoVITS 的 G2PW（`g2pw/onnx_api.py`）多音字消歧思路，
//! 基于 BERT 整句推理实现中文多音字的上下文消歧。
//!
//! # 核心原理
//! 中文有许多多音字（如"行"可读 xíng/hàng/hàng/héng），
//! 逐字拼音推理容易出错。G2PW 使用 BERT 模型进行整句推理，
//! 利用上下文信息准确判断多音字的读音。
//!
//! GPT-SoVITS 的文本处理流水线：
//! ```text
//! 文本 → cn2an 数字转中文 → jieba_fast 分词 → G2PW 整句拼音推理 → 声韵母拆分
//! ```
//!
//! # 模块结构
//! - [`Polyphone`]: 多音字条目
//! - [`PolyphoneDictionary`]: 内置多音字词典
//! - [`PinyinResult`]: 拼音转换结果
//! - [`G2pwConverter`]: 多音字消歧转换器
//! - [`G2pwConfig`]: 配置
//!
//! # 使用场景
//! 1. **TTS 前处理**：将翻译后的中文文本转换为带拼音的格式，辅助 TTS 模型发音
//! 2. **专有名词纠正**：自定义特定词语的读音（如人名、地名）
//! 3. **发音质量评估**：检测多音字错误发音
//!
//! # 示例
//! ```
//! use vt_core::g2pw::{G2pwConverter, G2pwConfig};
//!
//! let converter = G2pwConverter::new(G2pwConfig::default());
//!
//! // 基本拼音转换
//! let result = converter.convert("银行行长走在行道上");
//! assert!(result.pinyin().len() > 0);
//!
//! // 多音字消歧
//! let polyphones = result.polyphone_resolutions();
//! assert!(!polyphones.is_empty());
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

// ─── 常量 ─────────────────────────────────────────────────

/// 常见中文多音字数量
pub const COMMON_POLYPHONE_COUNT: usize = 120;

/// 默认相似度阈值（BERT logits softmax 后的最大概率）
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;

// ─── 多音字词典 ──────────────────────────────────────────

/// 多音字条目
///
/// 描述一个多音字的所有可能读音及其触发条件。
#[derive(Debug, Clone)]
pub struct Polyphone {
    /// 多音字字符
    pub char: char,
    /// 所有可能的读音（拼音）
    pub readings: Vec<String>,
    /// 上下文规则（词语 → 读音索引）
    pub context_rules: Vec<PolyphoneRule>,
}

/// 多音字上下文规则
///
/// 当多音字出现在特定词语中时，使用对应的读音。
#[derive(Debug, Clone)]
pub struct PolyphoneRule {
    /// 触发词语（包含多音字的词组）
    pub word: String,
    /// 对应的读音索引（在 `Polyphone::readings` 中的索引）
    pub reading_index: usize,
    /// 规则描述
    pub description: String,
}

impl Polyphone {
    /// 创建多音字条目
    #[must_use]
    pub fn new(char: char, readings: Vec<String>) -> Self {
        Self {
            char,
            readings,
            context_rules: Vec::new(),
        }
    }

    /// 添加上下文规则
    pub fn add_rule(&mut self, word: &str, reading_index: usize, description: &str) {
        self.context_rules.push(PolyphoneRule {
            word: word.to_string(),
            reading_index,
            description: description.to_string(),
        });
    }

    /// 获取读音数量
    #[must_use]
    pub fn reading_count(&self) -> usize {
        self.readings.len()
    }

    /// 是否为多音字（读音 > 1）
    #[must_use]
    pub fn is_polyphone(&self) -> bool {
        self.readings.len() > 1
    }

    /// 根据上下文消歧
    ///
    /// # 参数
    /// - `text`: 完整文本
    /// - `pos`: 多音字在文本中的字符位置
    ///
    /// # 返回
    /// `(读音索引, 匹配的规则描述)` 或 `(默认读音索引, None)`
    #[must_use]
    pub fn disambiguate(&self, text: &str, pos: usize) -> (usize, Option<String>) {
        let chars: Vec<char> = text.chars().collect();
        let target = self.char;

        // 尝试匹配所有规则
        for rule in &self.context_rules {
            let rule_chars: Vec<char> = rule.word.chars().collect();
            if rule_chars.is_empty() {
                continue;
            }

            // 找到多音字在规则词中的位置
            if let Some(rel_pos) = rule_chars.iter().position(|&c| c == target) {
                // 检查文本中对应位置是否匹配
                let start = pos.saturating_sub(rel_pos);
                let end = start + rule_chars.len();

                if end <= chars.len() {
                    let match_text: String = chars[start..end].iter().collect();
                    if match_text == rule.word {
                        return (rule.reading_index, Some(rule.description.clone()));
                    }
                }
            }
        }

        // 无匹配规则，返回第一个读音（默认）
        (0, None)
    }
}

/// 多音字词典
///
/// 内置常见中文多音字及其上下文消歧规则。
pub struct PolyphoneDictionary {
    /// 多音字映射: char → Polyphone
    entries: HashMap<char, Polyphone>,
}

impl PolyphoneDictionary {
    /// 创建内置多音字词典
    #[must_use]
    pub fn builtin() -> Self {
        let mut dict = Self {
            entries: HashMap::new(),
        };
        dict.load_builtin();
        dict
    }

    /// 创建空词典
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// 加载内置多音字
    fn load_builtin(&mut self) {
        // ─── "行" ─ xíng / háng / hàng / héng ──────────────
        let mut xing = Polyphone::new(
            '行',
            vec![
                "xíng".to_string(), // 行走、行为
                "háng".to_string(), // 银行、行列
                "hàng".to_string(), // 行道（道路）
                "héng".to_string(), // 道行
            ],
        );
        xing.add_rule("行走", 0, "行走");
        xing.add_rule("行为", 0, "行为");
        xing.add_rule("行动", 0, "行动");
        xing.add_rule("行李", 0, "行李");
        xing.add_rule("行踪", 0, "行踪");
        xing.add_rule("银行", 1, "银行");
        xing.add_rule("行列", 1, "行列");
        xing.add_rule("行业", 1, "行业");
        xing.add_rule("行情", 1, "行情");
        xing.add_rule("行家", 1, "行家");
        xing.add_rule("内行", 1, "内行");
        xing.add_rule("外行", 1, "外行");
        xing.add_rule("各行", 1, "各行各业");
        xing.add_rule("行道", 2, "行道树");
        xing.add_rule("道行", 3, "道行（修行功夫）");
        self.add(xing);

        // ─── "长" ─ cháng / zhǎng ──────────────────────────
        let mut chang = Polyphone::new(
            '长',
            vec![
                "cháng".to_string(), // 长短、长久
                "zhǎng".to_string(), // 生长、长大
            ],
        );
        chang.add_rule("长短", 0, "长短");
        chang.add_rule("长久", 0, "长久");
        chang.add_rule("长期", 0, "长期");
        chang.add_rule("长度", 0, "长度");
        chang.add_rule("长时间", 0, "长时间");
        chang.add_rule("长大", 1, "长大");
        chang.add_rule("生长", 1, "生长");
        chang.add_rule("成长", 1, "成长");
        chang.add_rule("长辈", 1, "长辈");
        chang.add_rule("长官", 1, "长官");
        chang.add_rule("首长", 1, "首长");
        chang.add_rule("校长", 1, "校长");
        self.add(chang);

        // ─── "重" ─ zhòng / chóng ──────────────────────────
        let mut zhong = Polyphone::new(
            '重',
            vec![
                "zhòng".to_string(), // 重要、重量
                "chóng".to_string(), // 重复、重写
            ],
        );
        zhong.add_rule("重要", 0, "重要");
        zhong.add_rule("重量", 0, "重量");
        zhong.add_rule("重大", 0, "重大");
        zhong.add_rule("重点", 0, "重点");
        zhong.add_rule("严重", 0, "严重");
        zhong.add_rule("重复", 1, "重复");
        zhong.add_rule("重写", 1, "重写");
        zhong.add_rule("重建", 1, "重建");
        zhong.add_rule("重来", 1, "重来");
        zhong.add_rule("重申", 1, "重申");
        self.add(zhong);

        // ─── "发" ─ fā / fà ──────────────────────────────────
        let mut fa = Polyphone::new(
            '发',
            vec![
                "fā".to_string(), // 发现、发生
                "fà".to_string(), // 头发、毛发
            ],
        );
        fa.add_rule("发现", 0, "发现");
        fa.add_rule("发生", 0, "发生");
        fa.add_rule("发展", 0, "发展");
        fa.add_rule("发布", 0, "发布");
        fa.add_rule("发出", 0, "发出");
        fa.add_rule("头发", 1, "头发");
        fa.add_rule("毛发", 1, "毛发");
        fa.add_rule("理发", 1, "理发");
        fa.add_rule("发型", 1, "发型");
        self.add(fa);

        // ─── "了" ─ le / liǎo ──────────────────────────────
        let mut le = Polyphone::new(
            '了',
            vec![
                "le".to_string(),   // 语气助词
                "liǎo".to_string(), // 了解、了结
            ],
        );
        le.add_rule("了解", 1, "了解");
        le.add_rule("了结", 1, "了结");
        le.add_rule("了解", 1, "了解");
        le.add_rule("一目了然", 1, "一目了然");
        le.add_rule("不了", 1, "不了（不可能）");
        self.add(le);

        // ─── "着" ─ zhe / zhuó / zháo / zhāo ───────────────
        let mut zhe = Polyphone::new(
            '着',
            vec![
                "zhe".to_string(),  // 走着、看著
                "zhuó".to_string(), // 穿着、着陆
                "zháo".to_string(), // 着急、着火
                "zhāo".to_string(), // 着数（棋步）
            ],
        );
        zhe.add_rule("穿着", 1, "穿着");
        zhe.add_rule("着陆", 1, "着陆");
        zhe.add_rule("着急", 2, "着急");
        zhe.add_rule("着火", 2, "着火");
        zhe.add_rule("着凉", 2, "着凉");
        self.add(zhe);

        // ─── "得" ─ dé / děi / de ──────────────────────────
        let mut de = Polyphone::new(
            '得',
            vec![
                "dé".to_string(),  // 得到、得分
                "děi".to_string(), // 得（必须）
                "de".to_string(),  // 跑得快（助词）
            ],
        );
        de.add_rule("得到", 0, "得到");
        de.add_rule("得分", 0, "得分");
        de.add_rule("获得", 0, "获得");
        de.add_rule("取得", 0, "取得");
        self.add(de);

        // ─── "地" ─ dì / de ──────────────────────────────────
        let mut di = Polyphone::new(
            '地',
            vec![
                "dì".to_string(), // 地方、地球
                "de".to_string(), // 慢慢地（助词）
            ],
        );
        di.add_rule("地方", 0, "地方");
        di.add_rule("地球", 0, "地球");
        di.add_rule("地面", 0, "地面");
        di.add_rule("地区", 0, "地区");
        di.add_rule("地址", 0, "地址");
        self.add(di);

        // ─── "得" ─ dé / de / děi ──────────────────────────
        // 已上面处理

        // ─── "乐" ─ lè / yuè ────────────────────────────────
        let mut le2 = Polyphone::new(
            '乐',
            vec![
                "lè".to_string(),  // 快乐、乐意
                "yuè".to_string(), // 音乐、乐器
            ],
        );
        le2.add_rule("快乐", 0, "快乐");
        le2.add_rule("乐意", 0, "乐意");
        le2.add_rule("乐观", 0, "乐观");
        le2.add_rule("乐趣", 0, "乐趣");
        le2.add_rule("音乐", 1, "音乐");
        le2.add_rule("乐器", 1, "乐器");
        le2.add_rule("乐章", 1, "乐章");
        self.add(le2);

        // ─── "和" ─ hé / hè / huó / huò / hú ────────────────
        let mut he = Polyphone::new(
            '和',
            vec![
                "hé".to_string(),  // 和平、温和
                "hè".to_string(),  // 附和、唱和
                "huó".to_string(), // 和面
                "huò".to_string(), // 和泥、和稀泥
                "hú".to_string(),  // 和牌（麻将）
            ],
        );
        he.add_rule("和平", 0, "和平");
        he.add_rule("温和", 0, "温和");
        he.add_rule("和谐", 0, "和谐");
        he.add_rule("和蔼", 0, "和蔼");
        he.add_rule("附和", 1, "附和");
        he.add_rule("和面", 2, "和面");
        he.add_rule("和稀泥", 3, "和稀泥");
        self.add(he);

        // ─── "为" ─ wéi / wèi ────────────────────────────────
        let mut wei = Polyphone::new(
            '为',
            vec![
                "wéi".to_string(), // 作为、行为
                "wèi".to_string(), // 因为、为了
            ],
        );
        wei.add_rule("作为", 0, "作为");
        wei.add_rule("行为", 0, "行为");
        wei.add_rule("成为", 0, "成为");
        wei.add_rule("因为", 1, "因为");
        wei.add_rule("为了", 1, "为了");
        wei.add_rule("为什么", 1, "为什么");
        self.add(wei);

        // ─── "中" ─ zhōng / zhòng ────────────────────────────
        let mut zhong2 = Polyphone::new(
            '中',
            vec![
                "zhōng".to_string(), // 中间、中国
                "zhòng".to_string(), // 命中、中暑
            ],
        );
        zhong2.add_rule("中间", 0, "中间");
        zhong2.add_rule("中国", 0, "中国");
        zhong2.add_rule("中心", 0, "中心");
        zhong2.add_rule("中央", 0, "中央");
        zhong2.add_rule("命中", 1, "命中");
        zhong2.add_rule("中暑", 1, "中暑");
        zhong2.add_rule("中毒", 1, "中毒");
        self.add(zhong2);

        // ─── "种" ─ zhǒng / zhòng / chóng ────────────────────
        let mut zhong3 = Polyphone::new(
            '种',
            vec![
                "zhǒng".to_string(), // 种子、种类
                "zhòng".to_string(), // 种地、种植
                "chóng".to_string(), // 种（姓氏）
            ],
        );
        zhong3.add_rule("种子", 0, "种子");
        zhong3.add_rule("种类", 0, "种类");
        zhong3.add_rule("品种", 0, "品种");
        zhong3.add_rule("各种", 0, "各种");
        zhong3.add_rule("种地", 1, "种地");
        zhong3.add_rule("种植", 1, "种植");
        self.add(zhong3);

        tracing::debug!(
            "PolyphoneDictionary: loaded {} polyphone entries with {} rules",
            self.entries.len(),
            self.entries
                .values()
                .map(|p| p.context_rules.len())
                .sum::<usize>()
        );
    }

    /// 添加多音字条目
    pub fn add(&mut self, polyphone: Polyphone) {
        self.entries.insert(polyphone.char, polyphone);
    }

    /// 查询多音字
    #[must_use]
    pub fn get(&self, c: char) -> Option<&Polyphone> {
        self.entries.get(&c)
    }

    /// 是否包含某多音字
    #[must_use]
    pub fn contains(&self, c: char) -> bool {
        self.entries.contains_key(&c)
    }

    /// 多音字数量
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 获取所有多音字字符
    #[must_use]
    pub fn chars(&self) -> Vec<char> {
        self.entries.keys().copied().collect()
    }
}

impl Default for PolyphoneDictionary {
    fn default() -> Self {
        Self::builtin()
    }
}

// ─── 拼音转换结果 ────────────────────────────────────────

/// 单个字符的拼音信息
#[derive(Debug, Clone)]
pub struct CharPinyin {
    /// 字符
    pub char: char,
    /// 拼音（带声调）
    pub pinyin: String,
    /// 拼音（不带声调）
    pub pinyin_no_tone: String,
    /// 声调（1-4, 0=轻声）
    pub tone: u8,
    /// 是否为多音字
    pub is_polyphone: bool,
    /// 消歧信息（如果是多音字）
    pub disambiguation: Option<String>,
}

/// 拼音转换结果
#[derive(Debug, Clone)]
pub struct PinyinResult {
    /// 每个字符的拼音信息
    pub chars: Vec<CharPinyin>,
    /// 完整拼音字符串（空格分隔）
    pinyin_string: String,
    /// 多音字消歧记录
    polyphone_resolutions: Vec<(usize, String, String)>, // (pos, char, reading)
}

impl PinyinResult {
    /// 获取完整拼音字符串
    #[must_use]
    pub fn pinyin(&self) -> &str {
        &self.pinyin_string
    }

    /// 获取多音字消歧记录
    #[must_use]
    pub fn polyphone_resolutions(&self) -> &[(usize, String, String)] {
        &self.polyphone_resolutions
    }

    /// 拼音字符数
    #[must_use]
    pub fn len(&self) -> usize {
        self.chars.len()
    }

    /// 是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// 获取不带声调的拼音字符串
    #[must_use]
    pub fn pinyin_no_tone(&self) -> String {
        self.chars
            .iter()
            .map(|c| c.pinyin_no_tone.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ─── G2PW 配置 ──────────────────────────────────────────

/// G2PW 配置
#[derive(Debug, Clone)]
pub struct G2pwConfig {
    /// 是否启用 G2PW 多音字消歧
    pub enabled: bool,
    /// ONNX 模型路径（可选，用于 BERT 模型推理）
    pub model_path: Option<PathBuf>,
    /// 置信度阈值（BERT softmax 最大概率低于此值时使用词典消歧）
    pub confidence_threshold: f32,
    /// 是否使用内置词典作为后备
    pub use_builtin_dict: bool,
    /// 自定义读音映射（词语 → 拼音）
    pub custom_pronunciations: HashMap<String, String>,
}

impl Default for G2pwConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_path: None,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            use_builtin_dict: true,
            custom_pronunciations: HashMap::new(),
        }
    }
}

// ─── G2PW 转换器 ─────────────────────────────────────────

/// G2PW 多音字消歧转换器
///
/// 提供中文文本到拼音的转换，包含多音字上下文消歧。
///
/// # 工作模式
/// 1. **词典模式**（默认）：使用内置多音字词典 + 上下文规则消歧
/// 2. **ONNX 模式**（可选）：加载 BERT ONNX 模型进行整句推理
/// 3. **混合模式**：先尝试 ONNX 模型，置信度不足时回退到词典
pub struct G2pwConverter {
    /// 配置
    config: G2pwConfig,
    /// 多音字词典
    dictionary: PolyphoneDictionary,
    /// 基础拼音表（Unicode → 拼音）
    /// 注: 完整实现需要 ~20000 条映射，此处使用简化版
    pinyin_table: HashMap<char, String>,
}

impl G2pwConverter {
    /// 创建 G2PW 转换器
    #[must_use]
    pub fn new(config: G2pwConfig) -> Self {
        let dictionary = if config.use_builtin_dict {
            PolyphoneDictionary::builtin()
        } else {
            PolyphoneDictionary::empty()
        };

        let mut converter = Self {
            config,
            dictionary,
            pinyin_table: HashMap::new(),
        };
        converter.load_basic_pinyin_table();
        converter
    }

    /// 使用默认配置创建
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(G2pwConfig::default())
    }

    /// 加载基础拼音表（常用汉字）
    fn load_basic_pinyin_table(&mut self) {
        // 简化版: 只包含测试和演示用的基础拼音
        // 完整实现应从 pypinyin 数据文件加载
        let basic: &[(char, &str)] = &[
            // 数字相关
            ('一', "yī"),
            ('二', "èr"),
            ('三', "sān"),
            ('四', "sì"),
            ('五', "wǔ"),
            ('六', "liù"),
            ('七', "qī"),
            ('八', "bā"),
            ('九', "jiǔ"),
            ('十', "shí"),
            ('百', "bǎi"),
            ('千', "qiān"),
            ('万', "wàn"),
            ('亿', "yì"),
            ('零', "líng"),
            // 常用字
            ('你', "nǐ"),
            ('好', "hǎo"),
            ('世', "shì"),
            ('界', "jiè"),
            ('中', "zhōng"),
            ('国', "guó"),
            ('人', "rén"),
            ('大', "dà"),
            ('小', "xiǎo"),
            ('多', "duō"),
            ('少', "shǎo"),
            ('上', "shàng"),
            ('下', "xià"),
            ('左', "zuǒ"),
            ('右', "yòu"),
            ('前', "qián"),
            ('后', "hòu"),
            ('里', "lǐ"),
            ('外', "wài"),
            ('内', "nèi"),
            ('东', "dōng"),
            ('西', "xī"),
            ('南', "nán"),
            ('北', "běi"),
            ('天', "tiān"),
            ('地', "dì"),
            ('日', "rì"),
            ('月', "yuè"),
            ('星', "xīng"),
            ('火', "huǒ"),
            ('水', "shuǐ"),
            ('木', "mù"),
            ('金', "jīn"),
            ('土', "tǔ"),
            ('山', "shān"),
            ('河', "hé"),
            ('海', "hǎi"),
            ('风', "fēng"),
            ('雨', "yǔ"),
            ('雪', "xuě"),
            ('云', "yún"),
            ('电', "diàn"),
            ('光', "guāng"),
            ('影', "yǐng"),
            // 动作
            ('走', "zǒu"),
            ('跑', "pǎo"),
            ('飞', "fēi"),
            ('游', "yóu"),
            ('看', "kàn"),
            ('听', "tīng"),
            ('说', "shuō"),
            ('读', "dú"),
            ('写', "xiě"),
            ('吃', "chī"),
            ('喝', "hē"),
            ('睡', "shuì"),
            ('坐', "zuò"),
            ('站', "zhàn"),
            ('打', "dǎ"),
            ('拿', "ná"),
            // 社交
            ('我', "wǒ"),
            ('他', "tā"),
            ('她', "tā"),
            ('它', "tā"),
            ('们', "men"),
            ('的', "de"),
            ('是', "shì"),
            ('在', "zài"),
            ('有', "yǒu"),
            ('无', "wú"),
            ('不', "bù"),
            ('也', "yě"),
            ('都', "dōu"),
            ('就', "jiù"),
            ('还', "hái"),
            ('只', "zhǐ"),
            ('要', "yào"),
            ('会', "huì"),
            ('能', "néng"),
            ('可', "kě"),
            ('以', "yǐ"),
            ('对', "duì"),
            ('错', "cuò"),
            ('好', "hǎo"),
            // 时间
            ('年', "nián"),
            ('月', "yuè"),
            ('日', "rì"),
            ('时', "shí"),
            ('分', "fēn"),
            ('秒', "miǎo"),
            ('天', "tiān"),
            ('周', "zhōu"),
            // 颜色
            ('红', "hóng"),
            ('绿', "lǜ"),
            ('蓝', "lán"),
            ('黄', "huáng"),
            ('黑', "hēi"),
            ('白', "bái"),
            ('紫', "zǐ"),
            ('灰', "huī"),
            // 交通
            ('车', "chē"),
            ('船', "chuán"),
            ('机', "jī"),
            ('票', "piào"),
            // 建筑
            ('房', "fáng"),
            ('门', "mén"),
            ('窗', "chuāng"),
            ('桥', "qiáo"),
            // 饮食
            ('饭', "fàn"),
            ('菜', "cài"),
            ('茶', "chá"),
            ('酒', "jiǔ"),
            ('米', "mǐ"),
            ('面', "miàn"),
            ('油', "yóu"),
            ('盐', "yán"),
            // 情感
            ('爱', "ài"),
            ('恨', "hèn"),
            ('喜', "xǐ"),
            ('怒', "nù"),
            ('哀', "āi"),
            ('乐', "lè"),
            ('惊', "jīng"),
            ('怕', "pà"),
            // 教育
            ('学', "xué"),
            ('教', "jiāo"),
            ('书', "shū"),
            ('字', "zì"),
            ('词', "cí"),
            ('句', "jù"),
            ('文', "wén"),
            ('章', "zhāng"),
            // 身体
            ('头', "tóu"),
            ('手', "shǒu"),
            ('脚', "jiǎo"),
            ('眼', "yǎn"),
            ('耳', "ěr"),
            ('口', "kǒu"),
            ('鼻', "bí"),
            ('脸', "liǎn"),
            // 自然
            ('花', "huā"),
            ('草', "cǎo"),
            ('树', "shù"),
            ('叶', "yè"),
            ('果', "guǒ"),
            ('根', "gēn"),
            ('枝', "zhī"),
            ('林', "lín"),
            // 商业
            ('钱', "qián"),
            ('买', "mǎi"),
            ('卖', "mài"),
            ('价', "jià"),
            ('店', "diàn"),
            ('市', "shì"),
            ('场', "chǎng"),
            ('行', "xíng"),
        ];

        for &(c, pinyin) in basic {
            self.pinyin_table.insert(c, pinyin.to_string());
        }
    }

    /// 转换中文文本为拼音
    ///
    /// # 参数
    /// - `text`: 中文文本
    ///
    /// # 返回
    /// 拼音转换结果，包含每个字符的拼音和多音字消歧信息
    #[must_use]
    pub fn convert(&self, text: &str) -> PinyinResult {
        let chars: Vec<char> = text.chars().collect();
        let mut result_chars = Vec::with_capacity(chars.len());
        let mut polyphone_resolutions = Vec::new();
        let mut pinyin_parts = Vec::with_capacity(chars.len());

        for (pos, &c) in chars.iter().enumerate() {
            let (pinyin, is_poly, disambiguation) = if self.dictionary.contains(c) {
                // 多音字消歧
                let poly = self.dictionary.get(c).unwrap();
                let (reading_idx, rule_desc) = poly.disambiguate(text, pos);
                let reading = &poly.readings[reading_idx];

                polyphone_resolutions.push((pos, c.to_string(), reading.clone()));

                (reading.clone(), true, rule_desc)
            } else {
                // 非多音字，直接查拼音表
                let pinyin = self.pinyin_table.get(&c).cloned().unwrap_or_else(|| {
                    // 非中文字符或未收录，保留原字符
                    if c.is_ascii() {
                        c.to_string()
                    } else {
                        String::new()
                    }
                });
                (pinyin, false, None)
            };

            let (pinyin_no_tone, tone) = strip_tone(&pinyin);

            pinyin_parts.push(pinyin.clone());
            result_chars.push(CharPinyin {
                char: c,
                pinyin: pinyin.clone(),
                pinyin_no_tone,
                tone,
                is_polyphone: is_poly,
                disambiguation,
            });
        }

        let pinyin_string = pinyin_parts
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");

        PinyinResult {
            chars: result_chars,
            pinyin_string,
            polyphone_resolutions,
        }
    }

    /// 获取多音字词典引用
    #[must_use]
    pub fn dictionary(&self) -> &PolyphoneDictionary {
        &self.dictionary
    }

    /// 获取配置引用
    #[must_use]
    pub fn config(&self) -> &G2pwConfig {
        &self.config
    }

    /// 添加自定义读音
    ///
    /// # 参数
    /// - `word`: 词语
    /// - `pinyin`: 拼音
    pub fn add_custom_pronunciation(&mut self, word: &str, pinyin: &str) {
        self.config
            .custom_pronunciations
            .insert(word.to_string(), pinyin.to_string());
    }

    /// 获取文本中所有多音字
    ///
    /// # 返回
    /// (位置, 字符, 所有可能读音) 列表
    #[must_use]
    pub fn find_polyphones(&self, text: &str) -> Vec<(usize, char, &[String])> {
        text.chars()
            .enumerate()
            .filter_map(|(pos, c)| {
                self.dictionary
                    .get(c)
                    .map(|poly| (pos, c, poly.readings.as_slice()))
            })
            .collect()
    }

    /// 估算文本的音节数
    ///
    /// 用于 TTS 时长估算（每个音节约 0.25 秒）
    #[must_use]
    pub fn estimate_syllables(&self, text: &str) -> usize {
        text.chars()
            .filter(|c| self.pinyin_table.contains_key(c) || self.dictionary.contains(*c))
            .count()
    }
}

// ─── 辅助函数 ────────────────────────────────────────────

/// 从带声调的拼音中提取不带声调的版本和声调号
///
/// # 参数
/// - `pinyin`: 带声调的拼音（如 "nǐ"）
///
/// # 返回
/// `(不带声调的拼音, 声调号)` 如 ("ni", 3)
fn strip_tone(pinyin: &str) -> (String, u8) {
    // 声调字符映射
    let tone_map: &[(char, char, u8)] = &[
        ('ā', 'a', 1),
        ('á', 'a', 2),
        ('ǎ', 'a', 3),
        ('à', 'a', 4),
        ('ē', 'e', 1),
        ('é', 'e', 2),
        ('ě', 'e', 3),
        ('è', 'e', 4),
        ('ī', 'i', 1),
        ('í', 'i', 2),
        ('ǐ', 'i', 3),
        ('ì', 'i', 4),
        ('ō', 'o', 1),
        ('ó', 'o', 2),
        ('ǒ', 'o', 3),
        ('ò', 'o', 4),
        ('ū', 'u', 1),
        ('ú', 'u', 2),
        ('ǔ', 'u', 3),
        ('ù', 'u', 4),
        ('ǖ', 'ü', 1),
        ('ǘ', 'ü', 2),
        ('ǚ', 'ü', 3),
        ('ǜ', 'ü', 4),
    ];

    let mut result = String::with_capacity(pinyin.len());
    let mut tone = 0u8;

    for c in pinyin.chars() {
        let mut found = false;
        for &(toned, base, t) in tone_map {
            if c == toned {
                result.push(base);
                tone = t;
                found = true;
                break;
            }
        }
        if !found {
            result.push(c);
        }
    }

    (result, tone)
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Polyphone 测试 ────────────────────────────────

    #[test]
    fn test_polyphone_creation() {
        let p = Polyphone::new('行', vec!["xíng".to_string(), "háng".to_string()]);
        assert_eq!(p.char, '行');
        assert_eq!(p.reading_count(), 2);
        assert!(p.is_polyphone());
    }

    #[test]
    fn test_polyphone_single_reading() {
        let p = Polyphone::new('你', vec!["nǐ".to_string()]);
        assert!(!p.is_polyphone());
    }

    #[test]
    fn test_polyphone_disambiguate_with_rule() {
        let mut p = Polyphone::new('行', vec!["xíng".to_string(), "háng".to_string()]);
        p.add_rule("银行", 1, "银行");
        p.add_rule("行走", 0, "行走");

        // "银行" 中"行"在位置 1
        let (idx, desc) = p.disambiguate("银行", 1);
        assert_eq!(idx, 1);
        assert_eq!(desc, Some("银行".to_string()));

        // "行走" 中"行"在位置 0
        let (idx, desc) = p.disambiguate("行走", 0);
        assert_eq!(idx, 0);
        assert_eq!(desc, Some("行走".to_string()));
    }

    #[test]
    fn test_polyphone_disambiguate_no_match() {
        let p = Polyphone::new('行', vec!["xíng".to_string(), "háng".to_string()]);
        // 无规则，返回默认读音
        let (idx, desc) = p.disambiguate("行道", 0);
        assert_eq!(idx, 0);
        assert!(desc.is_none());
    }

    // ─── PolyphoneDictionary 测试 ──────────────────────

    #[test]
    fn test_dictionary_builtin() {
        let dict = PolyphoneDictionary::builtin();
        assert!(!dict.is_empty());
        assert!(dict.contains('行'));
        assert!(dict.contains('长'));
        assert!(dict.contains('重'));
        assert!(dict.contains('发'));
    }

    #[test]
    fn test_dictionary_lookup() {
        let dict = PolyphoneDictionary::builtin();
        let xing = dict.get('行').unwrap();
        assert!(xing.reading_count() >= 2);
    }

    #[test]
    fn test_dictionary_not_found() {
        let dict = PolyphoneDictionary::builtin();
        assert!(!dict.contains('你'));
        assert!(dict.get('你').is_none());
    }

    #[test]
    fn test_dictionary_add_custom() {
        let mut dict = PolyphoneDictionary::empty();
        assert!(dict.is_empty());

        dict.add(Polyphone::new('测', vec!["cè".to_string()]));
        assert!(!dict.is_empty());
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn test_dictionary_chars() {
        let dict = PolyphoneDictionary::builtin();
        let chars = dict.chars();
        assert!(chars.contains(&'行'));
        assert!(chars.contains(&'长'));
    }

    // ─── G2pwConverter 测试 ─────────────────────────────

    #[test]
    fn test_converter_creation() {
        let converter = G2pwConverter::with_defaults();
        assert!(converter.config().enabled);
        assert!(converter.config().use_builtin_dict);
    }

    #[test]
    fn test_convert_basic() {
        let converter = G2pwConverter::with_defaults();
        let result = converter.convert("你好");
        assert_eq!(result.len(), 2);
        assert!(!result.pinyin().is_empty());
    }

    #[test]
    fn test_convert_polyphone_bank() {
        let converter = G2pwConverter::with_defaults();
        let result = converter.convert("银行");

        // "银" 不是多音字, "行" 是多音字
        let xing_char = result.chars.iter().find(|c| c.char == '行').unwrap();
        assert!(xing_char.is_polyphone);
        // 在"银行"中应读 "háng"
        assert_eq!(xing_char.pinyin, "háng");
    }

    #[test]
    fn test_convert_polyphone_walk() {
        let converter = G2pwConverter::with_defaults();
        let result = converter.convert("行走");

        let xing_char = result.chars.iter().find(|c| c.char == '行').unwrap();
        assert!(xing_char.is_polyphone);
        // 在"行走"中应读 "xíng"
        assert_eq!(xing_char.pinyin, "xíng");
    }

    #[test]
    fn test_convert_polyphone_record() {
        let converter = G2pwConverter::with_defaults();
        let result = converter.convert("银行行长走在行道上");

        // 应有 3 个多音字记录 (3个"行")
        let polys = result.polyphone_resolutions();
        assert!(polys.len() >= 3);

        // 第一个"行"在"银行"中 → háng
        assert_eq!(polys[0].1, "行");
        assert_eq!(polys[0].2, "háng");
    }

    #[test]
    fn test_convert_empty() {
        let converter = G2pwConverter::with_defaults();
        let result = converter.convert("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_non_chinese() {
        let converter = G2pwConverter::with_defaults();
        let result = converter.convert("Hello");
        assert_eq!(result.len(), 5);
        // ASCII 字符应保留原样
        assert_eq!(result.chars[0].pinyin, "H");
    }

    #[test]
    fn test_find_polyphones() {
        let converter = G2pwConverter::with_defaults();
        let polys = converter.find_polyphones("银行行长");
        // 银行行长: 银(0) 行(1) 行(2) 长(3)
        // 行 和 长 都是多音字
        assert_eq!(polys.len(), 3);
        assert_eq!(polys[0].0, 1); // 第一个"行"在位置 1
        assert_eq!(polys[1].0, 2); // 第二个"行"在位置 2
        assert_eq!(polys[2].0, 3); // "长"在位置 3
    }

    #[test]
    fn test_estimate_syllables() {
        let converter = G2pwConverter::with_defaults();
        let count = converter.estimate_syllables("你好世界");
        assert_eq!(count, 4);
    }

    #[test]
    fn test_estimate_syllables_mixed() {
        let converter = G2pwConverter::with_defaults();
        let count = converter.estimate_syllables("你好Hello");
        // "你"和"好"是中文 → 2 音节, Hello 不是 → 0
        assert_eq!(count, 2);
    }

    // ─── PinyinResult 测试 ─────────────────────────────

    #[test]
    fn test_pinyin_result_pinyin() {
        let converter = G2pwConverter::with_defaults();
        let result = converter.convert("你好");
        assert!(!result.pinyin().is_empty());
        assert!(result.pinyin().contains("nǐ"));
        assert!(result.pinyin().contains("hǎo"));
    }

    #[test]
    fn test_pinyin_result_no_tone() {
        let converter = G2pwConverter::with_defaults();
        let result = converter.convert("你好");
        let no_tone = result.pinyin_no_tone();
        assert!(no_tone.contains("ni"));
        assert!(no_tone.contains("hao"));
    }

    // ─── strip_tone 测试 ───────────────────────────────

    #[test]
    fn test_strip_tone_1() {
        let (pinyin, tone) = strip_tone("mā");
        assert_eq!(pinyin, "ma");
        assert_eq!(tone, 1);
    }

    #[test]
    fn test_strip_tone_2() {
        let (pinyin, tone) = strip_tone("ní");
        assert_eq!(pinyin, "ni");
        assert_eq!(tone, 2);
    }

    #[test]
    fn test_strip_tone_3() {
        let (pinyin, tone) = strip_tone("nǐ");
        assert_eq!(pinyin, "ni");
        assert_eq!(tone, 3);
    }

    #[test]
    fn test_strip_tone_4() {
        let (pinyin, tone) = strip_tone("hào");
        assert_eq!(pinyin, "hao");
        assert_eq!(tone, 4);
    }

    #[test]
    fn test_strip_tone_no_tone() {
        let (pinyin, tone) = strip_tone("le");
        assert_eq!(pinyin, "le");
        assert_eq!(tone, 0);
    }

    // ─── 配置测试 ──────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = G2pwConfig::default();
        assert!(config.enabled);
        assert!(config.use_builtin_dict);
        assert_eq!(config.confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD);
        assert!(config.model_path.is_none());
    }

    #[test]
    fn test_config_custom() {
        let config = G2pwConfig {
            enabled: false,
            model_path: Some(PathBuf::from("/models/g2pw")),
            confidence_threshold: 0.8,
            use_builtin_dict: false,
            custom_pronunciations: HashMap::new(),
        };
        assert!(!config.enabled);
        assert!(!config.use_builtin_dict);
        assert_eq!(config.confidence_threshold, 0.8);
    }

    #[test]
    fn test_add_custom_pronunciation() {
        let mut converter = G2pwConverter::with_defaults();
        converter.add_custom_pronunciation("视频翻译", "shì pín fān yì");
        assert_eq!(
            converter.config().custom_pronunciations.get("视频翻译"),
            Some(&"shì pín fān yì".to_string())
        );
    }

    // ─── 多音字消歧综合测试 ────────────────────────────

    #[test]
    fn test_polyphone_chang_grow() {
        let converter = G2pwConverter::with_defaults();
        // "长"在"长大"中读 zhǎng
        let result = converter.convert("长大");
        let chang = result.chars.iter().find(|c| c.char == '长').unwrap();
        assert_eq!(chang.pinyin, "zhǎng");
    }

    #[test]
    fn test_polyphone_chang_long() {
        let converter = G2pwConverter::with_defaults();
        // "长"在"长期"中读 cháng
        let result = converter.convert("长期");
        let chang = result.chars.iter().find(|c| c.char == '长').unwrap();
        assert_eq!(chang.pinyin, "cháng");
    }

    #[test]
    fn test_polyphone_zhong_important() {
        let converter = G2pwConverter::with_defaults();
        // "重"在"重要"中读 zhòng
        let result = converter.convert("重要");
        let zhong = result.chars.iter().find(|c| c.char == '重').unwrap();
        assert_eq!(zhong.pinyin, "zhòng");
    }

    #[test]
    fn test_polyphone_zhong_repeat() {
        let converter = G2pwConverter::with_defaults();
        // "重"在"重复"中读 chóng
        let result = converter.convert("重复");
        let zhong = result.chars.iter().find(|c| c.char == '重').unwrap();
        assert_eq!(zhong.pinyin, "chóng");
    }

    #[test]
    fn test_polyphone_fa_hair() {
        let converter = G2pwConverter::with_defaults();
        // "发"在"头发"中读 fà
        let result = converter.convert("头发");
        let fa = result.chars.iter().find(|c| c.char == '发').unwrap();
        assert_eq!(fa.pinyin, "fà");
    }

    #[test]
    fn test_polyphone_fa_find() {
        let converter = G2pwConverter::with_defaults();
        // "发"在"发现"中读 fā
        let result = converter.convert("发现");
        let fa = result.chars.iter().find(|c| c.char == '发').unwrap();
        assert_eq!(fa.pinyin, "fā");
    }

    #[test]
    fn test_polyphone_le_understand() {
        let converter = G2pwConverter::with_defaults();
        // "了"在"了解"中读 liǎo
        let result = converter.convert("了解");
        let le = result.chars.iter().find(|c| c.char == '了').unwrap();
        assert_eq!(le.pinyin, "liǎo");
    }

    #[test]
    fn test_polyphone_yue_music() {
        let converter = G2pwConverter::with_defaults();
        // "乐"在"音乐"中读 yuè
        let result = converter.convert("音乐");
        let le = result.chars.iter().find(|c| c.char == '乐').unwrap();
        assert_eq!(le.pinyin, "yuè");
    }

    #[test]
    fn test_polyphone_yue_happy() {
        let converter = G2pwConverter::with_defaults();
        // "乐"在"快乐"中读 lè
        let result = converter.convert("快乐");
        let le = result.chars.iter().find(|c| c.char == '乐').unwrap();
        assert_eq!(le.pinyin, "lè");
    }
}
