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
pub(crate) struct AnalysisSegment {
    /// The text content of this segment.
    pub text: String,
    /// The segment kind.
    pub kind: SegmentKind,
    /// Whether this segment contains CJK characters.
    pub contains_cjk: bool,
}

/// Result of text analysis -- segments + chunk boundaries.
#[derive(Debug, Clone)]
pub(crate) struct AnalysisResult {
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
pub(crate) struct AnalysisChunk {
    /// Inclusive index of the first segment in the chunk.
    pub start: usize,
    /// Exclusive end index of the chunk's content segments.
    pub end: usize,
    /// Exclusive index after content and any consumed hard break.
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
pub(crate) fn analyze_text(text: &str, white_space: WhiteSpaceMode) -> AnalysisResult {
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

    // Atomic run passes must see the original boundaries. Punctuation passes
    // can otherwise fold a protocol colon or numeric separator into only one
    // neighbor and make the complete URL/date impossible to recognize.
    merge_url_runs(&mut segments);
    merge_numeric_runs(&mut segments);

    // Typography passes operate on the now-atomic URL and numeric segments.
    merge_left_sticky_punctuation(&mut segments);
    merge_forward_sticky(&mut segments);
    merge_kinsoku(&mut segments);

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
                if c == '\n' || c == '\r' || c == '\t' || c == '\u{000C}' || c == ' ' {
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
            text.replace("\r\n", "\n").replace(['\r', '\u{000C}'], "\n")
        }
    }
}

/// Initial segmentation using Unicode word boundaries.
fn initial_segmentation(text: &str, mode: WhiteSpaceMode) -> Vec<AnalysisSegment> {
    let mut segments = Vec::new();

    // Use unicode-segmentation for word boundaries
    for word in text.split_word_bounds() {
        split_special_controls(word, mode, &mut segments);
    }

    segments
}

/// Split controls whose layout semantics must survive regardless of where a
/// word-boundary implementation places them.
///
/// Each control is emitted as its own segment. In particular, testing only a
/// word's first character is insufficient for `co\u{AD}operate`, embedded
/// zero-width breaks, and no-break controls within otherwise ordinary runs.
fn split_special_controls(word: &str, mode: WhiteSpaceMode, segments: &mut Vec<AnalysisSegment>) {
    let mut piece_start = 0;
    for (byte_index, character) in word.char_indices() {
        let Some(kind) = special_control_kind(character) else {
            continue;
        };

        if let Some(piece) = word.get(piece_start..byte_index) {
            push_initial_piece(piece, mode, segments);
        }
        segments.push(AnalysisSegment {
            text: character.to_string(),
            kind,
            contains_cjk: false,
        });
        piece_start = byte_index + character.len_utf8();
    }
    if let Some(piece) = word.get(piece_start..) {
        push_initial_piece(piece, mode, segments);
    }
}

const fn special_control_kind(character: char) -> Option<SegmentKind> {
    match character {
        '\u{00AD}' => Some(SegmentKind::SoftHyphen),
        '\u{200B}' => Some(SegmentKind::ZeroWidthBreak),
        '\u{00A0}' | '\u{202F}' | '\u{2060}' | '\u{FEFF}' => Some(SegmentKind::Glue),
        _ => None,
    }
}

/// Classify one non-special piece produced by [`split_special_controls`].
fn push_initial_piece(word: &str, mode: WhiteSpaceMode, segments: &mut Vec<AnalysisSegment>) {
    if word.is_empty() {
        return;
    }

    // Classify the segment.
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
            split_cjk_segment(word, segments);
        } else {
            segments.push(AnalysisSegment {
                text: word.to_string(),
                kind: SegmentKind::Text,
                contains_cjk: false,
            });
        }
    }
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
        let Some(c) = grapheme.chars().next() else {
            continue;
        };
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
    let mut merged = Vec::with_capacity(segments.len());

    for segment in segments.drain(..) {
        let should_merge = segment.kind == SegmentKind::Text
            && segment
                .text
                .chars()
                .all(unicode::is_left_sticky_punctuation)
            && merged.last().is_some_and(|previous: &AnalysisSegment| {
                previous.kind == SegmentKind::Text && !previous.contains_cjk
            });

        if should_merge && let Some(previous) = merged.last_mut() {
            previous.text.push_str(&segment.text);
            continue;
        }

        merged.push(segment);
    }

    *segments = merged;
}

/// Merge pass: forward-sticky punctuation.
///
/// Carries opening brackets, quotes, and currency symbols into the
/// following text segment so they don't get orphaned at line end.
fn merge_forward_sticky(segments: &mut Vec<AnalysisSegment>) {
    let mut merged = Vec::with_capacity(segments.len());
    let mut carry: Option<(AnalysisSegment, bool)> = None;

    for next in segments.drain(..) {
        let next_is_forward =
            next.kind == SegmentKind::Text && next.text.chars().all(unicode::is_forward_sticky);

        if let Some((mut current, current_is_forward)) = carry.take() {
            if current_is_forward && next.kind == SegmentKind::Text {
                current.text.push_str(&next.text);
                current.contains_cjk |= next.contains_cjk;
                carry = Some((current, next_is_forward));
            } else {
                merged.push(current);
                carry = Some((next, next_is_forward));
            }
        } else {
            carry = Some((next, next_is_forward));
        }
    }

    if let Some((remaining, _)) = carry {
        merged.push(remaining);
    }

    *segments = merged;
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
            && merged
                .last()
                .is_some_and(|prev: &AnalysisSegment| prev.kind == SegmentKind::Text);

        if should_merge && let Some(prev) = merged.last_mut() {
            prev.text.push_str(&seg.text);
            prev.contains_cjk |= seg.contains_cjk;
            continue;
        }

        merged.push(seg);
    }

    // Pass 2: kinsoku-start — merge segments that can't end a line into next.
    // Carry a pending opener forward without shifting the vector.
    let mut result = Vec::with_capacity(merged.len());
    let mut carry: Option<AnalysisSegment> = None;

    for seg in merged {
        if let Some(mut prev) = carry.take() {
            if seg.kind == SegmentKind::Text {
                prev.text.push_str(&seg.text);
                prev.contains_cjk |= seg.contains_cjk;
                // Check if the merged result also ends with kinsoku-start
                if prev
                    .text
                    .chars()
                    .last()
                    .is_some_and(unicode::is_kinsoku_start)
                {
                    carry = Some(prev);
                } else {
                    result.push(prev);
                }
            } else {
                result.push(prev);
                result.push(seg);
            }
        } else if seg.kind == SegmentKind::Text
            && seg
                .text
                .chars()
                .last()
                .is_some_and(unicode::is_kinsoku_start)
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
    let mut pending: std::collections::VecDeque<_> = segments.drain(..).collect();
    let mut merged = Vec::with_capacity(pending.len());

    while !pending.is_empty() {
        let Some(seed_segment_count) = url_seed_segment_count(&pending) else {
            if let Some(segment) = pending.pop_front() {
                merged.push(segment);
            }
            continue;
        };

        let Some(mut url) = pending.pop_front() else {
            break;
        };
        for _ in 1..seed_segment_count {
            let Some(segment) = pending.pop_front() else {
                break;
            };
            url.text.push_str(&segment.text);
            url.contains_cjk |= segment.contains_cjk;
        }

        while pending.front().is_some_and(is_url_extension) {
            let Some(segment) = pending.pop_front() else {
                break;
            };
            url.text.push_str(&segment.text);
            url.contains_cjk |= segment.contains_cjk;
        }
        merged.push(url);
    }

    *segments = merged;
}

/// Number of leading segments needed to establish a supported URL prefix.
/// Prefix recognition is bounded by the longest seed and therefore remains
/// linear even for adversarial runs of tiny text segments.
fn url_seed_segment_count(segments: &std::collections::VecDeque<AnalysisSegment>) -> Option<usize> {
    const SEEDS: [&str; 3] = ["http://", "https://", "www."];
    let mut candidate = String::with_capacity(8);

    for (segment_index, segment) in segments.iter().enumerate() {
        if segment.kind != SegmentKind::Text {
            return None;
        }
        for character in segment.text.chars() {
            if !character.is_ascii() {
                return None;
            }
            candidate.push(character.to_ascii_lowercase());
            if SEEDS.iter().any(|seed| candidate == *seed) {
                return Some(segment_index + 1);
            }
            if !SEEDS.iter().any(|seed| seed.starts_with(&candidate)) {
                return None;
            }
        }
    }
    None
}

fn is_url_extension(segment: &AnalysisSegment) -> bool {
    segment.kind == SegmentKind::Text
        && !segment.text.is_empty()
        && segment
            .text
            .chars()
            .all(|character| unicode::is_url_internal(character) || character.is_alphanumeric())
}

/// Merge pass: numeric runs.
///
/// Merges numeric patterns like `3.14`, `1,000`, `2024-01-15` into
/// single segments so they don't break at internal punctuation.
fn merge_numeric_runs(segments: &mut Vec<AnalysisSegment>) {
    let mut pending: std::collections::VecDeque<_> = segments.drain(..).collect();
    let mut merged = Vec::with_capacity(pending.len());

    while let Some(mut first) = pending.pop_front() {
        if first.kind != SegmentKind::Text || !unicode::is_all_digits(&first.text) {
            merged.push(first);
            continue;
        }

        while pending.front().is_some_and(|middle| {
            middle.kind == SegmentKind::Text
                && !middle.text.is_empty()
                && middle.text.chars().all(unicode::is_numeric_connective)
        }) && pending
            .get(1)
            .is_some_and(|end| end.kind == SegmentKind::Text && unicode::is_all_digits(&end.text))
        {
            let (Some(middle), Some(end)) = (pending.pop_front(), pending.pop_front()) else {
                break;
            };
            first.text.push_str(&middle.text);
            first.text.push_str(&end.text);
            first.contains_cjk |= middle.contains_cjk || end.contains_cjk;
        }

        merged.push(first);
    }

    *segments = merged;
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
pub(crate) fn to_prepared_chunks(analysis_chunks: &[AnalysisChunk]) -> Vec<PreparedLineChunk> {
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

    fn text_segment(text: impl Into<String>) -> AnalysisSegment {
        AnalysisSegment {
            text: text.into(),
            kind: SegmentKind::Text,
            contains_cjk: false,
        }
    }

    fn reconstructed_text(segments: &[AnalysisSegment]) -> String {
        segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect()
    }

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
        assert_eq!(
            normalize_whitespace("hello\u{000C}world", WhiteSpaceMode::PreWrap),
            "hello\nworld"
        );
    }

    #[test]
    fn form_feed_matches_css_whitespace_modes() {
        let normal = analyze_text("a\u{000C}\u{000C}b", WhiteSpaceMode::Normal);
        assert_eq!(reconstructed_text(&normal.segments), "a b");

        let pre_wrap = analyze_text("a\u{000C}b", WhiteSpaceMode::PreWrap);
        assert_eq!(reconstructed_text(&pre_wrap.segments), "a\nb");
        assert!(
            pre_wrap
                .segments
                .iter()
                .any(|segment| segment.kind == SegmentKind::HardBreak)
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
    fn embedded_break_and_no_break_controls_are_always_explicit_segments() {
        let controls = [
            ('\u{00AD}', SegmentKind::SoftHyphen),
            ('\u{200B}', SegmentKind::ZeroWidthBreak),
            ('\u{00A0}', SegmentKind::Glue),
            ('\u{202F}', SegmentKind::Glue),
            ('\u{2060}', SegmentKind::Glue),
            ('\u{FEFF}', SegmentKind::Glue),
        ];

        for (control, expected_kind) in controls {
            let input = format!("left{control}right");
            let analyzed = analyze_text(&input, WhiteSpaceMode::Normal);
            let matching: Vec<_> = analyzed
                .segments
                .iter()
                .filter(|segment| segment.text == control.to_string())
                .collect();

            assert_eq!(reconstructed_text(&analyzed.segments), input);
            assert_eq!(matching.len(), 1, "control U+{:04X}", control as u32);
            assert_eq!(
                matching.first().map(|segment| segment.kind),
                Some(expected_kind)
            );
        }
    }

    #[test]
    fn adjacent_embedded_controls_remain_individual_and_ordered() {
        let input = "a\u{00AD}\u{200B}\u{00A0}\u{202F}\u{2060}\u{FEFF}z";
        let analyzed = analyze_text(input, WhiteSpaceMode::Normal);
        let controls: Vec<_> = analyzed
            .segments
            .iter()
            .filter(|segment| segment.kind != SegmentKind::Text)
            .map(|segment| (segment.text.as_str(), segment.kind))
            .collect();

        assert_eq!(reconstructed_text(&analyzed.segments), input);
        assert_eq!(
            controls,
            vec![
                ("\u{00AD}", SegmentKind::SoftHyphen),
                ("\u{200B}", SegmentKind::ZeroWidthBreak),
                ("\u{00A0}", SegmentKind::Glue),
                ("\u{202F}", SegmentKind::Glue),
                ("\u{2060}", SegmentKind::Glue),
                ("\u{FEFF}", SegmentKind::Glue),
            ]
        );
    }

    #[test]
    fn long_embedded_control_run_preserves_every_control_without_coalescing() {
        const CONTROL_COUNT: usize = 4_096;
        let input = format!(
            "{}tail",
            "part\u{00AD}\u{200B}\u{2060}".repeat(CONTROL_COUNT)
        );

        let analyzed = analyze_text(&input, WhiteSpaceMode::Normal);
        let explicit_controls = analyzed
            .segments
            .iter()
            .filter(|segment| {
                matches!(
                    segment.kind,
                    SegmentKind::SoftHyphen | SegmentKind::ZeroWidthBreak | SegmentKind::Glue
                )
            })
            .count();

        assert_eq!(reconstructed_text(&analyzed.segments), input);
        assert_eq!(explicit_controls, CONTROL_COUNT * 3);
    }

    #[test]
    fn test_left_sticky_punctuation() {
        let result = analyze_text("hello.", WhiteSpaceMode::Normal);
        // "hello" and "." should merge into "hello."
        assert!(result.segments.len() <= 2);
        assert!(
            result
                .segments
                .iter()
                .any(|s| s.text.contains("hello") && s.text.contains('.'))
        );
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

    #[test]
    fn long_sticky_punctuation_runs_merge_without_losing_text() {
        const RUN_LENGTH: usize = 8_192;

        let mut trailing = Vec::with_capacity(RUN_LENGTH + 1);
        trailing.push(text_segment("word"));
        trailing.extend((0..RUN_LENGTH).map(|_| text_segment(".")));
        merge_left_sticky_punctuation(&mut trailing);
        assert_eq!(trailing.len(), 1);
        assert_eq!(
            reconstructed_text(&trailing),
            format!("word{}", ".".repeat(RUN_LENGTH))
        );

        let mut leading = Vec::with_capacity(RUN_LENGTH + 1);
        leading.extend((0..RUN_LENGTH).map(|_| text_segment("(")));
        leading.push(text_segment("word"));
        merge_forward_sticky(&mut leading);
        assert_eq!(leading.len(), 1);
        assert_eq!(
            reconstructed_text(&leading),
            format!("{}word", "(".repeat(RUN_LENGTH))
        );
    }

    #[test]
    fn long_url_run_merges_without_losing_text() {
        const RUN_LENGTH: usize = 8_192;

        let mut segments = Vec::with_capacity(RUN_LENGTH + 1);
        segments.push(text_segment("https://example.com/"));
        segments.extend((0..RUN_LENGTH).map(|index| {
            if index % 2 == 0 {
                text_segment("path")
            } else {
                text_segment("/")
            }
        }));
        let expected = reconstructed_text(&segments);

        merge_url_runs(&mut segments);

        assert_eq!(segments.len(), 1);
        assert_eq!(reconstructed_text(&segments), expected);
    }

    #[test]
    fn fragmented_url_seed_and_complete_tail_merge_atomically() {
        let mut segments = [
            "HTTPS", ":", "/", "/", "example", ".", "com", "/", "a", "?", "x", "=", "1", "#",
            "frag",
        ]
        .into_iter()
        .map(text_segment)
        .collect();

        merge_url_runs(&mut segments);

        assert_eq!(segments.len(), 1);
        assert_eq!(
            reconstructed_text(&segments),
            "HTTPS://example.com/a?x=1#frag"
        );
    }

    #[test]
    fn complete_analysis_keeps_urls_and_dates_atomic_end_to_end() {
        let url = "https://example.com/a/b?x=1#frag";
        let date = "2024-01-15";
        let input = format!("visit {url} before {date} please");

        let analyzed = analyze_text(&input, WhiteSpaceMode::Normal);

        assert_eq!(reconstructed_text(&analyzed.segments), input);
        assert!(analyzed.segments.iter().any(|segment| segment.text == url));
        assert!(analyzed.segments.iter().any(|segment| segment.text == date));
    }

    #[test]
    fn chained_numeric_connectives_merge_the_entire_date_and_time() {
        let mut date = ["2024", "-", "01", "-", "15"]
            .into_iter()
            .map(text_segment)
            .collect();
        let mut time = ["09", ":", "30", ":", "45"]
            .into_iter()
            .map(text_segment)
            .collect();

        merge_numeric_runs(&mut date);
        merge_numeric_runs(&mut time);

        assert_eq!(date.len(), 1);
        assert_eq!(reconstructed_text(&date), "2024-01-15");
        assert_eq!(time.len(), 1);
        assert_eq!(reconstructed_text(&time), "09:30:45");
    }

    #[test]
    fn long_ip_like_numeric_run_preserves_order_and_text() {
        const GROUP_COUNT: usize = 4_097;

        let mut segments = Vec::with_capacity(GROUP_COUNT * 2 - 1);
        for group in 0..GROUP_COUNT {
            if group > 0 {
                segments.push(text_segment("."));
            }
            segments.push(text_segment((group % 256).to_string()));
        }
        let expected = reconstructed_text(&segments);

        merge_numeric_runs(&mut segments);

        assert_eq!(segments.len(), 1);
        assert_eq!(reconstructed_text(&segments), expected);
    }

    #[test]
    fn adversarial_patterns_survive_the_complete_analysis_pipeline() {
        let punctuation = format!("word{}", ".".repeat(2_048));
        let url = format!("https://example.com/{}end?x=1", "path/".repeat(2_048));
        let ip = std::iter::repeat_n("192.168.0.1", 2_048)
            .collect::<Vec<_>>()
            .join(",");
        let input = format!("{punctuation} {url} {ip}");

        let analyzed = analyze_text(&input, WhiteSpaceMode::Normal);

        assert_eq!(reconstructed_text(&analyzed.segments), input);
    }
}
