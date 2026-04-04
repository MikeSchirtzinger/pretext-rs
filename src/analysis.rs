//! Text analysis and segmentation.
//!
//! Takes raw text and produces a sequence of typed segments ready for
//! measurement. This is the Rust equivalent of pretext's `analysis.ts` --
//! it handles whitespace normalization, word segmentation via
//! `unicode-segmentation`, and the chain of merge passes that enforce
//! kinsoku, URL atomicity, numeric grouping, and punctuation stickiness.

use unicode_segmentation::UnicodeSegmentation;

use crate::types::{PreparedLineChunk, SegmentKind, WhiteSpaceMode};
use crate::unicode;

/// A raw analysis segment before measurement.
#[derive(Debug, Clone)]
pub struct AnalysisSegment {
    /// The text content of this segment.
    pub text: String,
    /// The segment kind.
    pub kind: SegmentKind,
    /// Whether this segment contains CJK characters.
    pub contains_cjk: bool,
}

/// Result of text analysis -- segments + chunk boundaries.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// The analyzed segments.
    pub segments: Vec<AnalysisSegment>,
    /// Chunk boundaries for pre-wrap mode (indices into segments
    /// where hard breaks occur).
    pub chunks: Vec<AnalysisChunk>,
    /// Whether the simple fast path can be used.
    pub simple_fast_path: bool,
}

/// A chunk of segments between hard breaks.
#[derive(Debug, Clone)]
pub struct AnalysisChunk {
    pub start: usize,
    pub end: usize,
    pub consumed_end: usize,
}

/// Analyze text into segments with classified break opportunities.
///
/// This is the main entry point for the analysis phase. It:
/// 1. Normalizes whitespace according to the mode
/// 2. Segments text using Unicode word boundaries
/// 3. Classifies each segment (text, space, tab, etc.)
/// 4. Runs merge passes for punctuation stickiness, kinsoku, URLs, etc.
/// 5. Compiles chunk boundaries for hard breaks
#[must_use]
pub fn analyze_text(text: &str, white_space: WhiteSpaceMode) -> AnalysisResult {
    if text.is_empty() {
        return AnalysisResult {
            segments: vec![],
            chunks: vec![AnalysisChunk {
                start: 0,
                end: 0,
                consumed_end: 0,
            }],
            simple_fast_path: true,
        };
    }

    let normalized = normalize_whitespace(text, white_space);
    let mut segments = initial_segmentation(&normalized, white_space);

    // Merge passes -- order matters (matches TypeScript chain)
    merge_left_sticky_punctuation(&mut segments);
    merge_forward_sticky(&mut segments);
    merge_kinsoku(&mut segments);
    merge_url_runs(&mut segments);
    merge_numeric_runs(&mut segments);

    let simple_fast_path = check_simple_fast_path(&segments);
    let chunks = compile_chunks(&segments);

    AnalysisResult {
        segments,
        chunks,
        simple_fast_path,
    }
}

/// Normalize whitespace according to the mode.
fn normalize_whitespace(text: &str, mode: WhiteSpaceMode) -> String {
    match mode {
        WhiteSpaceMode::Normal => {
            // Collapse runs of whitespace to single space, strip leading/trailing
            let mut result = String::with_capacity(text.len());
            let mut in_whitespace = false;
            for c in text.chars() {
                if c == '\n' || c == '\r' || c == '\t' || c == ' ' {
                    if !in_whitespace && !result.is_empty() {
                        result.push(' ');
                    }
                    in_whitespace = true;
                } else {
                    in_whitespace = false;
                    result.push(c);
                }
            }
            // Trim trailing space from collapse
            if result.ends_with(' ') {
                result.pop();
            }
            result
        }
        WhiteSpaceMode::PreWrap => {
            // Preserve whitespace but normalize \r\n to \n
            text.replace("\r\n", "\n").replace('\r', "\n")
        }
    }
}

/// Initial segmentation using Unicode word boundaries.
fn initial_segmentation(text: &str, mode: WhiteSpaceMode) -> Vec<AnalysisSegment> {
    let mut segments = Vec::new();

    // Use unicode-segmentation for word boundaries
    for word in text.split_word_bounds() {
        if word.is_empty() {
            continue;
        }

        let first_char = word.chars().next().unwrap();

        // Classify the segment
        if word == "\n" && mode == WhiteSpaceMode::PreWrap {
            segments.push(AnalysisSegment {
                text: word.to_string(),
                kind: SegmentKind::HardBreak,
                contains_cjk: false,
            });
        } else if word == "\t" {
            segments.push(AnalysisSegment {
                text: word.to_string(),
                kind: if mode == WhiteSpaceMode::PreWrap {
                    SegmentKind::Tab
                } else {
                    SegmentKind::Space
                },
                contains_cjk: false,
            });
        } else if unicode::is_soft_hyphen(first_char) {
            segments.push(AnalysisSegment {
                text: word.to_string(),
                kind: SegmentKind::SoftHyphen,
                contains_cjk: false,
            });
        } else if unicode::is_zero_width_space(first_char) {
            segments.push(AnalysisSegment {
                text: word.to_string(),
                kind: SegmentKind::ZeroWidthBreak,
                contains_cjk: false,
            });
        } else if first_char == '\u{00A0}' {
            // NBSP -- glue
            segments.push(AnalysisSegment {
                text: word.to_string(),
                kind: SegmentKind::Glue,
                contains_cjk: false,
            });
        } else if word.chars().all(|c| c == ' ') {
            if mode == WhiteSpaceMode::PreWrap {
                segments.push(AnalysisSegment {
                    text: word.to_string(),
                    kind: SegmentKind::PreservedSpace,
                    contains_cjk: false,
                });
            } else {
                segments.push(AnalysisSegment {
                    text: word.to_string(),
                    kind: SegmentKind::Space,
                    contains_cjk: false,
                });
            }
        } else if word.chars().all(char::is_whitespace) {
            // Other whitespace
            segments.push(AnalysisSegment {
                text: word.to_string(),
                kind: SegmentKind::Space,
                contains_cjk: false,
            });
        } else {
            // Text segment -- check for CJK and split CJK into individual graphemes
            let has_cjk = word.chars().any(unicode::is_cjk);

            if has_cjk {
                // Split CJK text into individual graphemes for per-character breaking
                split_cjk_segment(word, &mut segments);
            } else {
                segments.push(AnalysisSegment {
                    text: word.to_string(),
                    kind: SegmentKind::Text,
                    contains_cjk: false,
                });
            }
        }
    }

    segments
}

/// Split a segment containing CJK characters into individual graphemes.
///
/// CJK text can break at any character boundary, so each character
/// becomes its own segment. Non-CJK runs within the segment are kept
/// as single text segments.
fn split_cjk_segment(text: &str, segments: &mut Vec<AnalysisSegment>) {
    let mut current_run = String::new();
    let mut current_is_cjk = false;

    for grapheme in text.graphemes(true) {
        let c = grapheme.chars().next().unwrap();
        let this_is_cjk = unicode::is_cjk(c);

        if this_is_cjk {
            // Flush any non-CJK run
            if !current_run.is_empty() && !current_is_cjk {
                segments.push(AnalysisSegment {
                    text: std::mem::take(&mut current_run),
                    kind: SegmentKind::Text,
                    contains_cjk: false,
                });
            }
            // Each CJK character is its own segment
            if current_is_cjk && !current_run.is_empty() {
                segments.push(AnalysisSegment {
                    text: std::mem::take(&mut current_run),
                    kind: SegmentKind::Text,
                    contains_cjk: true,
                });
            }
            segments.push(AnalysisSegment {
                text: grapheme.to_string(),
                kind: SegmentKind::Text,
                contains_cjk: true,
            });
            current_run.clear();
            current_is_cjk = true;
        } else {
            if current_is_cjk && !current_run.is_empty() {
                // This shouldn't happen since we push CJK immediately
                segments.push(AnalysisSegment {
                    text: std::mem::take(&mut current_run),
                    kind: SegmentKind::Text,
                    contains_cjk: true,
                });
            }
            current_is_cjk = false;
            current_run.push_str(grapheme);
        }
    }

    // Flush remaining
    if !current_run.is_empty() {
        segments.push(AnalysisSegment {
            text: current_run,
            kind: SegmentKind::Text,
            contains_cjk: current_is_cjk,
        });
    }
}

/// Merge pass: left-sticky punctuation.
///
/// Merges trailing punctuation (`.`, `,`, `)`, etc.) into the preceding
/// text segment. This prevents float accumulation errors from measuring
/// `"better"` and `"."` separately vs `"better."` as one unit.
fn merge_left_sticky_punctuation(segments: &mut Vec<AnalysisSegment>) {
    let mut i = 1;
    while i < segments.len() {
        if segments[i].kind == SegmentKind::Text
            && segments[i - 1].kind == SegmentKind::Text
            && !segments[i - 1].contains_cjk
            && segments[i]
                .text
                .chars()
                .all(unicode::is_left_sticky_punctuation)
        {
            let merged_text = segments[i].text.clone();
            segments[i - 1].text.push_str(&merged_text);
            segments.remove(i);
        } else {
            i += 1;
        }
    }
}

/// Merge pass: forward-sticky punctuation.
///
/// Carries opening brackets, quotes, and currency symbols into the
/// following text segment so they don't get orphaned at line end.
fn merge_forward_sticky(segments: &mut Vec<AnalysisSegment>) {
    let mut i = 0;
    while i + 1 < segments.len() {
        if segments[i].kind == SegmentKind::Text
            && segments[i + 1].kind == SegmentKind::Text
            && segments[i]
                .text
                .chars()
                .all(unicode::is_forward_sticky)
        {
            let merged_text = segments[i + 1].text.clone();
            segments[i].text.push_str(&merged_text);
            segments[i].contains_cjk =
                segments[i].contains_cjk || segments[i + 1].contains_cjk;
            segments.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

/// Merge pass: kinsoku (Japanese line-breaking rules).
///
/// Enforces that kinsoku-start characters are kept with the preceding
/// segment (can't start a line) and kinsoku-end characters are kept
/// with the following segment (can't end a line).
fn merge_kinsoku(segments: &mut Vec<AnalysisSegment>) {
    // Pass 1: kinsoku-end — merge segments that can't start a line into previous.
    // Uses O(n) single-pass rebuild instead of Vec::remove (which is O(n^2) for
    // pathological CJK input like repeated ideographic commas).
    let mut merged = Vec::with_capacity(segments.len());
    for seg in segments.drain(..) {
        let should_merge = seg.kind == SegmentKind::Text
            && seg.text.chars().next().is_some_and(unicode::is_kinsoku_end)
            && merged.last().map_or(false, |prev: &AnalysisSegment| prev.kind == SegmentKind::Text);

        if should_merge {
            let prev = merged.last_mut().unwrap();
            prev.text.push_str(&seg.text);
            prev.contains_cjk = prev.contains_cjk || seg.contains_cjk;
        } else {
            merged.push(seg);
        }
    }

    // Pass 2: kinsoku-start — merge segments that can't end a line into next.
    // Iterate in reverse to merge forward without shifting.
    let mut result = Vec::with_capacity(merged.len());
    let mut carry: Option<AnalysisSegment> = None;

    for seg in merged {
        if let Some(mut prev) = carry.take() {
            if seg.kind == SegmentKind::Text {
                prev.text.push_str(&seg.text);
                prev.contains_cjk = prev.contains_cjk || seg.contains_cjk;
                // Check if the merged result also ends with kinsoku-start
                if prev.text.chars().last().is_some_and(unicode::is_kinsoku_start) {
                    carry = Some(prev);
                } else {
                    result.push(prev);
                }
            } else {
                result.push(prev);
                result.push(seg);
            }
        } else if seg.kind == SegmentKind::Text
            && seg.text.chars().last().is_some_and(unicode::is_kinsoku_start)
        {
            carry = Some(seg);
        } else {
            result.push(seg);
        }
    }

    if let Some(remaining) = carry {
        result.push(remaining);
    }

    *segments = result;
}

/// Merge pass: URL-like runs.
///
/// Merges sequences that look like URLs (`https://foo.com/bar`) into
/// single atomic segments so they don't break mid-URL.
fn merge_url_runs(segments: &mut Vec<AnalysisSegment>) {
    let mut i = 0;
    while i < segments.len() {
        if segments[i].kind == SegmentKind::Text && unicode::looks_like_url(&segments[i].text) {
            // Absorb following text/punctuation segments that are URL-internal
            while i + 1 < segments.len() {
                let next = &segments[i + 1];
                if next.kind == SegmentKind::Text
                    && next
                        .text
                        .chars()
                        .all(|c| unicode::is_url_internal(c) || c.is_alphanumeric())
                {
                    let merged_text = segments[i + 1].text.clone();
                    segments[i].text.push_str(&merged_text);
                    segments.remove(i + 1);
                } else {
                    break;
                }
            }
        }
        i += 1;
    }
}

/// Merge pass: numeric runs.
///
/// Merges numeric patterns like `3.14`, `1,000`, `2024-01-15` into
/// single segments so they don't break at internal punctuation.
fn merge_numeric_runs(segments: &mut Vec<AnalysisSegment>) {
    let mut i = 0;
    while i + 2 < segments.len() {
        if segments[i].kind == SegmentKind::Text
            && segments[i + 1].kind == SegmentKind::Text
            && segments[i + 2].kind == SegmentKind::Text
            && unicode::is_all_digits(&segments[i].text)
            && segments[i + 1]
                .text
                .chars()
                .all(unicode::is_numeric_connective)
            && unicode::is_all_digits(&segments[i + 2].text)
        {
            let mid = segments[i + 1].text.clone();
            let end = segments[i + 2].text.clone();
            segments[i].text.push_str(&mid);
            segments[i].text.push_str(&end);
            segments.remove(i + 2);
            segments.remove(i + 1);
            // Don't advance -- check if more numeric parts follow
        } else {
            i += 1;
        }
    }
}

/// Check whether the simple fast path can be used.
///
/// The simple path avoids chunk processing and only handles
/// text + space + zero-width-break segments.
fn check_simple_fast_path(segments: &[AnalysisSegment]) -> bool {
    segments.iter().all(|s| {
        matches!(
            s.kind,
            SegmentKind::Text | SegmentKind::Space | SegmentKind::ZeroWidthBreak
        )
    })
}

/// Compile chunk boundaries from hard breaks.
fn compile_chunks(segments: &[AnalysisSegment]) -> Vec<AnalysisChunk> {
    if segments.is_empty() {
        return vec![AnalysisChunk {
            start: 0,
            end: 0,
            consumed_end: 0,
        }];
    }

    let mut chunks = Vec::new();
    let mut chunk_start = 0;

    for (i, seg) in segments.iter().enumerate() {
        if seg.kind == SegmentKind::HardBreak {
            chunks.push(AnalysisChunk {
                start: chunk_start,
                end: i,
                consumed_end: i + 1,
            });
            chunk_start = i + 1;
        }
    }

    // Final chunk (no trailing hard break)
    chunks.push(AnalysisChunk {
        start: chunk_start,
        end: segments.len(),
        consumed_end: segments.len(),
    });

    chunks
}

/// Convert analysis chunks to prepared chunks (index mapping after measurement).
#[must_use]
pub fn to_prepared_chunks(analysis_chunks: &[AnalysisChunk]) -> Vec<PreparedLineChunk> {
    analysis_chunks
        .iter()
        .map(|c| PreparedLineChunk {
            start_segment_index: c.start,
            end_segment_index: c.end,
            consumed_end_segment_index: c.consumed_end,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace_normal() {
        assert_eq!(
            normalize_whitespace("  hello   world  ", WhiteSpaceMode::Normal),
            "hello world"
        );
        assert_eq!(
            normalize_whitespace("hello\n\nworld", WhiteSpaceMode::Normal),
            "hello world"
        );
        assert_eq!(
            normalize_whitespace("  \t  \n  ", WhiteSpaceMode::Normal),
            ""
        );
    }

    #[test]
    fn test_normalize_whitespace_prewrap() {
        assert_eq!(
            normalize_whitespace("hello\r\nworld", WhiteSpaceMode::PreWrap),
            "hello\nworld"
        );
    }

    #[test]
    fn test_basic_segmentation() {
        let result = analyze_text("hello world", WhiteSpaceMode::Normal);
        assert_eq!(result.segments.len(), 3); // "hello", " ", "world"
        assert_eq!(result.segments[0].kind, SegmentKind::Text);
        assert_eq!(result.segments[1].kind, SegmentKind::Space);
        assert_eq!(result.segments[2].kind, SegmentKind::Text);
    }

    #[test]
    fn test_left_sticky_punctuation() {
        let result = analyze_text("hello.", WhiteSpaceMode::Normal);
        // "hello" and "." should merge into "hello."
        assert!(result.segments.len() <= 2);
        assert!(result
            .segments
            .iter()
            .any(|s| s.text.contains("hello") && s.text.contains('.')));
    }

    #[test]
    fn test_cjk_splitting() {
        let result = analyze_text("\u{65E5}\u{672C}\u{8A9E}", WhiteSpaceMode::Normal);
        // Each CJK character should be its own segment
        assert_eq!(result.segments.len(), 3);
        assert!(result.segments.iter().all(|s| s.contains_cjk));
    }

    #[test]
    fn test_empty_text() {
        let result = analyze_text("", WhiteSpaceMode::Normal);
        assert!(result.segments.is_empty());
        assert!(result.simple_fast_path);
    }

    #[test]
    fn test_simple_fast_path() {
        let result = analyze_text("hello world", WhiteSpaceMode::Normal);
        assert!(result.simple_fast_path);
    }

    #[test]
    fn test_hard_break_chunks() {
        let result = analyze_text("line1\nline2\nline3", WhiteSpaceMode::PreWrap);
        // Should have 3 chunks
        assert_eq!(result.chunks.len(), 3);
    }
}
