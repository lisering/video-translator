//! 字幕后处理模块
//!
//! 参考 pyvideotrans 的 `recognition/_base.py` 中的字幕修正逻辑，
//! 在 Rust 中实现 6 阶段字幕后处理：
//! 1. 移除纯符号/空行字幕
//! 2. 修正时间戳重叠
//! 3. 合并过短字幕
//! 4. 处理首尾过短字幕
//! 5. 标点碎片重分配
//! 6. 清理尾部标点

use crate::models::segment::Segment;

/// CJK 语言代码前缀
const CJK_LANG_PREFIXES: &[&str] = &["zh", "ja", "ko", "yu", "yue"];

/// 判断语言代码是否为 CJK 语言（中日韩）
fn is_cjk(lang_code: &str) -> bool {
    let prefix = lang_code.split('-').next().unwrap_or("");
    CJK_LANG_PREFIXES.contains(&prefix)
}

/// 判断文本是否为纯标点/符号（无实质内容）
fn is_punctuation_only(text: &str) -> bool {
    text.chars().all(|c| {
        c.is_ascii_punctuation()
            || matches!(
                c,
                '，' | '。' | '！' | '？' | '、' | '：' | '；' | '…' | '—' | '～' | '·'
            )
            || c.is_whitespace()
    })
}

/// 获取字幕持续时间（秒）
fn duration(seg: &Segment) -> f64 {
    (seg.end - seg.start).max(0.0)
}

/// 字幕后处理配置
#[derive(Debug, Clone)]
pub struct SubtitlePostProcessConfig {
    /// 最短字幕时长（秒），短于此值考虑合并
    pub min_duration: f64,
    /// 最长字幕时长（秒），长于此值考虑切分（暂不实现切分）
    pub max_duration: f64,
    /// 合并时前后间隙阈值（秒），间隙小于此值才考虑合并
    pub merge_gap_threshold: f64,
    /// 是否启用标点碎片重分配
    pub enable_fragment_redistribution: bool,
    /// 语言代码
    pub language: String,
}

impl Default for SubtitlePostProcessConfig {
    fn default() -> Self {
        Self {
            min_duration: 1.0,
            max_duration: 10.0,
            merge_gap_threshold: 2.0,
            enable_fragment_redistribution: true,
            language: "en".to_string(),
        }
    }
}

/// 字幕后处理器
pub struct SubtitlePostProcessor {
    config: SubtitlePostProcessConfig,
}

impl SubtitlePostProcessor {
    /// 创建字幕后处理器
    pub fn new(config: SubtitlePostProcessConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建
    pub fn with_language(language: impl Into<String>) -> Self {
        Self::new(SubtitlePostProcessConfig {
            language: language.into(),
            ..Default::default()
        })
    }

    /// 执行完整的后处理流程
    ///
    /// 依次执行 6 个阶段，每个阶段对 segments 进行修正
    pub fn process(&self, segments: &mut Vec<Segment>) {
        if segments.len() < 2 {
            return;
        }

        // 阶段 1：移除纯符号行
        self.remove_punctuation_only(segments);

        if segments.len() < 2 {
            return;
        }

        // 阶段 2：修正时间戳重叠
        self.fix_overlapping_timestamps(segments);

        // 阶段 3：合并过短字幕
        self.merge_short_segments(segments);

        if segments.len() < 2 {
            return;
        }

        // 阶段 4：处理首尾过短字幕
        self.handle_edge_short_segments(segments);

        if segments.len() < 2 {
            return;
        }

        // 阶段 5：标点碎片重分配
        if self.config.enable_fragment_redistribution {
            self.redistribute_punctuation_fragments(segments);
        }

        // 阶段 6：清理尾部标点
        self.clean_trailing_punctuation(segments);

        // 重新编号
        for (i, seg) in segments.iter_mut().enumerate() {
            seg.id = format!("seg-{:04}", i + 1);
        }
    }

    /// 阶段 1：移除纯符号/空行字幕
    fn remove_punctuation_only(&self, segments: &mut Vec<Segment>) {
        segments.retain(|seg| {
            let text = seg.source_text.trim();
            !text.is_empty() && !is_punctuation_only(text)
        });
    }

    /// 阶段 2：修正时间戳重叠
    ///
    /// 确保每个字幕的 end 时间不超过下一个字幕的 start 时间
    fn fix_overlapping_timestamps(&self, segments: &mut [Segment]) {
        for i in 0..segments.len().saturating_sub(1) {
            let next_start = segments[i + 1].start;
            if segments[i].end > next_start {
                segments[i].end = next_start;
            }
        }
    }

    /// 阶段 3：合并过短字幕
    ///
    /// 遍历所有字幕，将持续时间小于 min_duration 的与前一个或后一个合并
    fn merge_short_segments(&self, segments: &mut Vec<Segment>) {
        if segments.len() < 2 {
            return;
        }

        let cjk = is_cjk(&self.config.language);
        let separator = if cjk { "" } else { " " };
        let mut merged: Vec<Segment> = Vec::with_capacity(segments.len());
        let mut i = 0;

        while i < segments.len() {
            let dur = duration(&segments[i]);

            if dur >= self.config.min_duration || i == segments.len() - 1 {
                // 不需要合并
                merged.push(segments[i].clone());
                i += 1;
                continue;
            }

            // 需要合并：判断合并方向
            let prev_end = if merged.is_empty() {
                f64::MAX
            } else {
                merged.last().map_or(f64::MAX, |s| s.end)
            };
            let next_start = if i + 1 < segments.len() {
                segments[i + 1].start
            } else {
                f64::MAX
            };

            let gap_to_prev = segments[i].start - prev_end;
            let gap_to_next = next_start - segments[i].end;

            if gap_to_prev <= gap_to_next && !merged.is_empty() {
                // 向前合并：取出当前段，追加到 merged 的最后一个
                let seg = segments[i].clone();
                let last = merged.last_mut().expect("merged non-empty checked above");
                last.end = seg.end;
                last.source_text = format!("{}{}{}", last.source_text, separator, seg.source_text);
                if let (Some(t1), Some(t2)) = (last.target_text.clone(), seg.target_text.clone()) {
                    last.target_text = Some(format!("{}{}{}", t1, separator, t2));
                }
            } else if i + 1 < segments.len() {
                // 向后合并：把当前段加到下一段前面
                let seg = segments[i].clone();
                let next_seg = &mut segments[i + 1];
                next_seg.source_text =
                    format!("{}{}{}", seg.source_text, separator, next_seg.source_text);
                next_seg.start = seg.start;
                if let (Some(t1), Some(t2)) =
                    (seg.target_text.clone(), next_seg.target_text.clone())
                {
                    next_seg.target_text = Some(format!("{}{}{}", t1, separator, t2));
                }
            } else {
                // 没有可合并的，保留
                merged.push(segments[i].clone());
            }
            i += 1;
        }

        *segments = merged;
    }

    /// 阶段 4：处理首尾过短字幕
    ///
    /// 如果第一个或最后一个字幕过短，合并到相邻字幕
    fn handle_edge_short_segments(&self, segments: &mut Vec<Segment>) {
        if segments.len() < 3 {
            return;
        }

        let cjk = is_cjk(&self.config.language);
        let separator = if cjk { "" } else { " " };

        // 检查最后一个是否过短
        let last_dur = segments.last().map_or(0.0, duration);
        if last_dur < self.config.min_duration {
            let last_idx = segments.len() - 1;
            let last = segments.remove(last_idx);
            let second_last = segments
                .last_mut()
                .expect("segments had >=3 elements, removed 1, still has >=2");
            second_last.end = last.end;
            second_last.source_text = format!(
                "{}{}{}",
                second_last.source_text, separator, last.source_text
            );
            if let (Some(t1), Some(t2)) =
                (second_last.target_text.as_ref(), last.target_text.as_ref())
            {
                second_last.target_text = Some(format!("{}{}{}", t1, separator, t2));
            }
        }

        if segments.len() < 3 {
            return;
        }

        // 检查第一个是否过短
        let first_dur = duration(&segments[0]);
        if first_dur < self.config.min_duration {
            let first = segments.remove(0);
            segments[0].start = first.start;
            segments[0].source_text = format!(
                "{}{}{}",
                first.source_text, separator, segments[0].source_text
            );
            if let (Some(t1), Some(t2)) =
                (first.target_text.as_ref(), segments[0].target_text.as_ref())
            {
                segments[0].target_text = Some(format!("{}{}{}", t1, separator, t2));
            }
        }
    }

    /// 阶段 5：标点碎片重分配
    ///
    /// 当一个字幕以不完整标点结尾（如逗号、省略号），
    /// 尝试将标点移动到更合适的位置
    fn redistribute_punctuation_fragments(&self, segments: &mut [Segment]) {
        // 对于以句末标点（。！？.!?）开头的字幕，将标点移到前一个字幕末尾
        let sentence_end_puncts = ['。', '！', '？', '.', '!', '?'];

        for i in 1..segments.len() {
            // 使用 split_at_mut 同时获取 segments[i-1] 和 segments[i] 的可变引用
            let (left, right) = segments.split_at_mut(i);
            let prev_seg = &mut left[i - 1];
            let curr_seg = &mut right[0];

            let text = curr_seg.source_text.trim_start().to_string();
            if let Some(first_char) = text.chars().next() {
                if sentence_end_puncts.contains(&first_char) {
                    // 从当前字幕移除开头的句末标点
                    let removed_punct: String = text.chars().take(1).collect();
                    curr_seg.source_text = text[removed_punct.len()..].trim_start().to_string();

                    // 追加到前一个字幕
                    prev_seg.source_text.push_str(&removed_punct);

                    // 同样处理 target_text
                    if let Some(ref target) = curr_seg.target_text {
                        let t_text = target.trim_start().to_string();
                        if let Some(t_first) = t_text.chars().next() {
                            if sentence_end_puncts.contains(&t_first) {
                                let t_punct: String = t_text.chars().take(1).collect();
                                curr_seg.target_text =
                                    Some(t_text[t_punct.len()..].trim_start().to_string());
                                if let Some(ref mut prev_target) = prev_seg.target_text {
                                    prev_target.push_str(&t_punct);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// 阶段 6：清理尾部标点
    ///
    /// 移除字幕末尾多余的标点（连续重复的逗号、句号等）
    fn clean_trailing_punctuation(&self, segments: &mut [Segment]) {
        for seg in segments.iter_mut() {
            // 清理 source_text 尾部
            seg.source_text = clean_trailing(seg.source_text.trim());
            // 清理 target_text 尾部
            if let Some(ref mut target) = seg.target_text {
                *target = clean_trailing(target.trim());
            }
        }
    }
}

/// 清理文本末尾多余的重复标点
fn clean_trailing(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return text.to_string();
    }

    // 如果末尾是多个连续相同标点，只保留一个
    let last_char = chars[chars.len() - 1];
    if is_trailing_redundant_punct(last_char) {
        let mut end = chars.len();
        while end > 1 && chars[end - 1] == last_char {
            end -= 1;
        }
        // 保留一个
        return chars[..end + 1].iter().collect();
    }

    text.to_string()
}

fn is_trailing_redundant_punct(c: char) -> bool {
    matches!(c, '。' | '.' | '，' | ',' | '！' | '!' | '？' | '?')
}

// ─── 单元测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seg(id: &str, start: f64, end: f64, text: &str) -> Segment {
        Segment::new(id.into(), start, end, text.into())
    }

    #[test]
    fn test_remove_punctuation_only() {
        let mut segs = vec![
            make_seg("1", 0.0, 2.0, "Hello"),
            make_seg("2", 2.0, 3.0, "..."),
            make_seg("3", 3.0, 5.0, "World"),
        ];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].source_text, "Hello");
        assert_eq!(segs[1].source_text, "World");
    }

    #[test]
    fn test_fix_overlapping_timestamps() {
        let mut segs = vec![
            make_seg("1", 0.0, 3.5, "First"),
            make_seg("2", 3.0, 5.0, "Second"),
        ];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        assert!(segs[0].end <= segs[1].start);
        assert_eq!(segs[0].end, 3.0);
    }

    #[test]
    fn test_merge_short_segment_forward() {
        let mut segs = vec![
            make_seg("1", 0.0, 3.0, "Hello"),
            make_seg("2", 3.0, 3.5, "there"), // 0.5s < 1.0s min
            make_seg("3", 3.5, 6.0, "World"),
        ];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        // "there" should be merged with either "Hello" or "World"
        assert!(segs.len() <= 2);
    }

    #[test]
    fn test_merge_short_segment_cjk() {
        let mut segs = vec![
            make_seg("1", 0.0, 3.0, "你好"),
            make_seg("2", 3.0, 3.3, "世"), // 0.3s < 1.0s min
            make_seg("3", 3.3, 6.0, "界"),
        ];
        let proc = SubtitlePostProcessor::with_language("zh");
        proc.process(&mut segs);
        // CJK merge should use no separator
        assert!(segs.len() <= 2);
        let combined_text: String = segs.iter().map(|s| s.source_text.as_str()).collect();
        assert!(combined_text.contains("世"));
    }

    #[test]
    fn test_redistribute_punctuation() {
        let mut segs = vec![
            make_seg("1", 0.0, 3.0, "Hello"),
            make_seg("2", 3.0, 6.0, ". How are you"),
        ];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        // The leading "." should be moved to the previous segment
        assert!(segs[0].source_text.ends_with('.'));
        assert!(!segs[1].source_text.starts_with('.'));
    }

    #[test]
    fn test_clean_trailing_punctuation() {
        let mut segs = vec![
            make_seg("1", 0.0, 3.0, "Hello..."),
            make_seg("2", 3.0, 6.0, "World"),
        ];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        // Multiple trailing dots should be reduced to one
        assert!(segs[0].source_text.ends_with('.'));
        assert!(!segs[0].source_text.ends_with("..."));
    }

    #[test]
    fn test_short_first_and_last_merged() {
        let mut segs = vec![
            make_seg("1", 0.0, 0.3, "Hi"), // 0.3s, too short
            make_seg("2", 0.3, 5.0, "middle content"),
            make_seg("3", 5.0, 5.2, "Bye"), // 0.2s, too short
        ];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        // First and last should be merged
        assert!(segs.len() <= 2);
    }

    #[test]
    fn test_is_cjk() {
        assert!(is_cjk("zh"));
        assert!(is_cjk("zh-cn"));
        assert!(is_cjk("ja"));
        assert!(is_cjk("ko"));
        assert!(!is_cjk("en"));
        assert!(!is_cjk("fr"));
    }

    #[test]
    fn test_is_punctuation_only() {
        assert!(is_punctuation_only("..."));
        assert!(is_punctuation_only("。"));
        assert!(is_punctuation_only("  ,  "));
        assert!(!is_punctuation_only("Hello."));
        assert!(!is_punctuation_only("。Hi"));
    }

    #[test]
    fn test_empty_input() {
        let mut segs: Vec<Segment> = vec![];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        assert!(segs.is_empty());
    }

    #[test]
    fn test_single_segment() {
        let mut segs = vec![make_seg("1", 0.0, 3.0, "Hello")];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].source_text, "Hello");
    }

    #[test]
    fn test_renumbering() {
        let mut segs = vec![
            make_seg("old-1", 0.0, 3.0, "Hello"),
            make_seg("old-2", 3.0, 6.0, "World"),
        ];
        let proc = SubtitlePostProcessor::with_language("en");
        proc.process(&mut segs);
        assert_eq!(segs[0].id, "seg-0001");
        assert_eq!(segs[1].id, "seg-0002");
    }
}
