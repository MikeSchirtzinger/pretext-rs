//! # pretext-rs
//!
//! DOM-free text measurement and line-breaking engine.
//! Rust port of [@chenglou/pretext](https://github.com/chenglou/pretext).
//!
//! Two-phase architecture:
//! - **`prepare()`** -- expensive one-time measurement (uses a [`backend::MeasureBackend`])
//! - **`layout()`** -- cheap reflow (pure arithmetic, zero allocations)
//!
//! ## Quick Start
//!
//! ```rust
//! use pretext::{prepare, layout, backend::fixed::FixedWidthBackend, backend::FontSpec};
//!
//! let backend = FixedWidthBackend::new();
//! let font = FontSpec::new("16px Inter");
//! let prepared = prepare("Hello, world!", &font, &backend, Default::default());
//! let result = layout(&prepared, 200.0, 24.0);
//! println!("Lines: {}, Height: {}px", result.line_count, result.height);
//! ```
//!
//! ## Backends
//!
//! - [`backend::fixed::FixedWidthBackend`] -- deterministic, no deps (testing / server-side estimation)
//! - `CanvasBackend` -- browser `canvas.measureText` (feature `wasm`)
//! - `FontdueBackend` -- native font metrics via fontdue (feature `fontdue`)

pub mod analysis;
pub mod backend;
pub mod gpu_layout;
pub mod inline_flow;
pub mod line_break;
pub mod types;
pub mod unicode;
#[cfg(feature = "wasm")]
pub mod wasm_api;

use unicode_segmentation::UnicodeSegmentation;

use analysis::{analyze_text, to_prepared_chunks};
use backend::{FontSpec, MeasureBackend};
use line_break::{
    count_lines_full, count_lines_simple, layout_next_line_range, walk_lines_full,
    walk_lines_simple,
};
use types::{
    EngineProfile, LayoutCursor, LayoutLine, LayoutLineRange, LayoutResult, PrepareOptions,
    PreparedData, PreparedText, PreparedTextWithSegments, SegmentKind,
};

/// Prepare text for layout -- the expensive phase.
///
/// Analyzes the text into segments, measures each segment using the
/// provided backend, and produces a `PreparedText` handle containing
/// all the data needed for the fast `layout()` path.
///
/// Call this once per (text, font) pair. Then call `layout()` as many
/// times as needed with different widths -- it's essentially free.
#[allow(clippy::needless_pass_by_value)] // API ergonomics: callers pass Default::default()
pub fn prepare(
    text: &str,
    font: &FontSpec,
    backend: &dyn MeasureBackend,
    options: PrepareOptions,
) -> PreparedText {
    let data = prepare_internal(text, font, backend, &options, false);
    PreparedText { data: data.0 }
}

/// Prepare text with segment strings -- for `layout_with_lines()`.
///
/// Same as `prepare()` but also retains the original segment text,
/// enabling line content materialization.
#[allow(clippy::needless_pass_by_value)] // API ergonomics: callers pass Default::default()
pub fn prepare_with_segments(
    text: &str,
    font: &FontSpec,
    backend: &dyn MeasureBackend,
    options: PrepareOptions,
) -> PreparedTextWithSegments {
    let (data, segments) = prepare_internal(text, font, backend, &options, true);
    PreparedTextWithSegments {
        data,
        segments: segments.unwrap_or_default(),
    }
}

/// Internal prepare implementation.
fn prepare_internal(
    text: &str,
    font: &FontSpec,
    backend: &dyn MeasureBackend,
    options: &PrepareOptions,
    keep_segments: bool,
) -> (PreparedData, Option<Vec<String>>) {
    let analysis = analyze_text(text, options.white_space);

    let space_width = backend.measure_space_width(font);
    let hyphen_width = backend.measure_hyphen_width(font);
    let tab_stop_advance = space_width * 8.0;

    let seg_count = analysis.segments.len();
    let mut widths = Vec::with_capacity(seg_count);
    let mut line_end_fit_advances = Vec::with_capacity(seg_count);
    let mut line_end_paint_advances = Vec::with_capacity(seg_count);
    let mut kinds = Vec::with_capacity(seg_count);
    let mut breakable_widths = Vec::with_capacity(seg_count);
    let mut segment_strings = if keep_segments {
        Some(Vec::with_capacity(seg_count))
    } else {
        None
    };

    for seg in &analysis.segments {
        let kind = seg.kind;
        kinds.push(kind);

        if let Some(ref mut ss) = segment_strings {
            ss.push(seg.text.clone());
        }

        match kind {
            SegmentKind::HardBreak | SegmentKind::ZeroWidthBreak => {
                widths.push(0.0);
                line_end_fit_advances.push(0.0);
                line_end_paint_advances.push(0.0);
                breakable_widths.push(None);
            }
            SegmentKind::SoftHyphen => {
                widths.push(0.0); // Invisible unless at break
                line_end_fit_advances.push(hyphen_width);
                line_end_paint_advances.push(hyphen_width);
                breakable_widths.push(None);
            }
            SegmentKind::Tab => {
                // Tab width is computed dynamically during layout
                widths.push(0.0);
                line_end_fit_advances.push(0.0);
                line_end_paint_advances.push(0.0);
                breakable_widths.push(None);
            }
            SegmentKind::Space | SegmentKind::PreservedSpace => {
                let metrics = backend.measure_segment(&seg.text, font);
                widths.push(metrics.width);
                // Trailing spaces "hang" -- they don't count toward fit
                line_end_fit_advances.push(0.0);
                line_end_paint_advances.push(0.0);
                breakable_widths.push(None);
            }
            SegmentKind::Glue => {
                let metrics = backend.measure_segment(&seg.text, font);
                widths.push(metrics.width);
                line_end_fit_advances.push(metrics.width);
                line_end_paint_advances.push(metrics.width);
                breakable_widths.push(None);
            }
            SegmentKind::Text => {
                let metrics = backend.measure_segment(&seg.text, font);
                widths.push(metrics.width);
                line_end_fit_advances.push(metrics.width);
                line_end_paint_advances.push(metrics.width);
                breakable_widths.push(metrics.grapheme_widths);
            }
        }
    }

    let chunks = to_prepared_chunks(&analysis.chunks);

    (
        PreparedData {
            widths,
            line_end_fit_advances,
            line_end_paint_advances,
            kinds,
            breakable_widths,
            chunks,
            tab_stop_advance,
            discretionary_hyphen_width: hyphen_width,
            simple_fast_path: analysis.simple_fast_path,
        },
        segment_strings,
    )
}

/// Layout prepared text at a given width -- the fast path.
///
/// Returns line count and total height. This is pure arithmetic over
/// cached widths -- no measurement, no strings, no allocations.
/// ~0.3us per text block.
#[must_use]
pub fn layout(prepared: &PreparedText, max_width: f64, line_height: f64) -> LayoutResult {
    let profile = EngineProfile::default();
    layout_with_profile(prepared, max_width, line_height, &profile)
}

/// Layout with a specific engine profile.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn layout_with_profile(
    prepared: &PreparedText,
    max_width: f64,
    line_height: f64,
    profile: &EngineProfile,
) -> LayoutResult {
    let line_count = if prepared.data.simple_fast_path {
        count_lines_simple(&prepared.data, max_width, profile)
    } else {
        count_lines_full(&prepared.data, max_width, profile)
    };

    LayoutResult {
        line_count,
        height: line_count as f64 * line_height,
    }
}

/// Layout with full line information -- returns each line's text and geometry.
///
/// Requires `PreparedTextWithSegments` (from `prepare_with_segments()`).
#[must_use]
pub fn layout_with_lines(
    prepared: &PreparedTextWithSegments,
    max_width: f64,
    _line_height: f64,
) -> Vec<LayoutLine> {
    let profile = EngineProfile::default();
    let mut lines = Vec::new();

    let callback = |internal: line_break::InternalLine| {
        let text = materialize_line_text(
            &prepared.segments,
            internal.start_segment,
            internal.start_grapheme,
            internal.end_segment,
            internal.end_grapheme,
        );

        lines.push(LayoutLine {
            text,
            width: internal.width,
            start: LayoutCursor {
                segment_index: internal.start_segment,
                grapheme_index: internal.start_grapheme,
            },
            end: LayoutCursor {
                segment_index: internal.end_segment,
                grapheme_index: internal.end_grapheme,
            },
        });
    };

    if prepared.data.simple_fast_path {
        walk_lines_simple(&prepared.data, max_width, &profile, callback);
    } else {
        walk_lines_full(&prepared.data, max_width, &profile, callback);
    }

    lines
}

/// Walk line ranges without materializing text -- geometry only.
///
/// Calls `on_line` for each line with its width and cursor range.
/// Fastest way to get layout geometry when you don't need text content.
pub fn walk_line_ranges<F>(
    prepared: &PreparedText,
    max_width: f64,
    profile: &EngineProfile,
    mut on_line: F,
) where
    F: FnMut(LayoutLineRange),
{
    let callback = |internal: line_break::InternalLine| {
        on_line(LayoutLineRange {
            width: internal.width,
            start: LayoutCursor {
                segment_index: internal.start_segment,
                grapheme_index: internal.start_grapheme,
            },
            end: LayoutCursor {
                segment_index: internal.end_segment,
                grapheme_index: internal.end_grapheme,
            },
        });
    };

    if prepared.data.simple_fast_path {
        walk_lines_simple(&prepared.data, max_width, profile, callback);
    } else {
        walk_lines_full(&prepared.data, max_width, profile, callback);
    }
}

/// Layout a single line starting from a cursor -- streaming API.
///
/// Supports per-line `max_width` (for text flowing around images).
/// Call repeatedly with the returned cursor to iterate all lines.
#[must_use]
pub fn layout_next_line(
    prepared: &PreparedText,
    start: LayoutCursor,
    max_width: f64,
) -> Option<(LayoutLineRange, LayoutCursor)> {
    let profile = EngineProfile::default();
    layout_next_line_range(&prepared.data, start, max_width, &profile)
}

/// Measure the natural (unwrapped) width of text.
///
/// Returns the width the text would occupy if it weren't wrapped --
/// effectively the width of the widest forced line (at infinite `max_width`).
#[must_use]
pub fn measure_natural_width(prepared: &PreparedText) -> f64 {
    let profile = EngineProfile::default();
    let mut max_w = 0.0;
    walk_line_ranges(prepared, f64::INFINITY, &profile, |range| {
        if range.width > max_w {
            max_w = range.width;
        }
    });
    max_w
}

/// Materialize line text from segment strings.
fn materialize_line_text(
    segments: &[String],
    start_seg: usize,
    start_grapheme: usize,
    end_seg: usize,
    end_grapheme: usize,
) -> String {
    let mut text = String::new();

    for (i, segment) in segments
        .iter()
        .enumerate()
        .take(end_seg.min(segments.len()))
        .skip(start_seg)
    {
        if i == start_seg && start_grapheme > 0 {
            // Partial start segment
            let graphemes: Vec<&str> = segment.graphemes(true).collect();
            let safe_start = start_grapheme.min(graphemes.len());
            if i == end_seg.saturating_sub(1) && end_grapheme > 0 && end_grapheme < graphemes.len()
            {
                // Both start and end are in same segment
                let safe_end = end_grapheme.min(graphemes.len());
                for g in &graphemes[safe_start..safe_end] {
                    text.push_str(g);
                }
                return text;
            }
            for g in &graphemes[safe_start..] {
                text.push_str(g);
            }
        } else if i == end_seg.saturating_sub(1) && end_grapheme > 0 {
            // Partial end segment
            let graphemes: Vec<&str> = segment.graphemes(true).collect();
            for g in &graphemes[..end_grapheme.min(graphemes.len())] {
                text.push_str(g);
            }
        } else {
            text.push_str(segment);
        }
    }

    // Trim trailing whitespace from line
    text.trim_end().to_string()
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::fixed::FixedWidthBackend;

    fn setup() -> (FixedWidthBackend, FontSpec) {
        (FixedWidthBackend::new(), FontSpec::new("16px Inter"))
    }

    #[test]
    fn test_prepare_and_layout_simple() {
        let (backend, font) = setup();
        let prepared = prepare("Hello, world!", &font, &backend, PrepareOptions::default());
        let result = layout(&prepared, 200.0, 24.0);
        assert!(result.line_count >= 1);
        assert!(result.height > 0.0);
    }

    #[test]
    fn test_layout_wrap() {
        let (backend, font) = setup();
        let prepared = prepare("Hello, world!", &font, &backend, PrepareOptions::default());
        let result = layout(&prepared, 60.0, 24.0);
        assert!(result.line_count >= 2);
    }

    #[test]
    fn test_layout_no_wrap() {
        let (backend, font) = setup();
        let prepared = prepare("Hi", &font, &backend, PrepareOptions::default());
        let result = layout(&prepared, 200.0, 24.0);
        assert_eq!(result.line_count, 1);
        assert!((result.height - 24.0).abs() < 0.001);
    }

    #[test]
    fn test_layout_with_lines() {
        let (backend, font) = setup();
        let prepared =
            prepare_with_segments("Hello world test", &font, &backend, PrepareOptions::default());
        let lines = layout_with_lines(&prepared, 80.0, 24.0);
        assert!(!lines.is_empty());
        let all_text: String = lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join(" ");
        assert!(all_text.contains("Hello"));
        assert!(all_text.contains("world"));
        assert!(all_text.contains("test"));
    }

    #[test]
    fn test_natural_width() {
        let (backend, font) = setup();
        let prepared = prepare("Hello world", &font, &backend, PrepareOptions::default());
        let natural = measure_natural_width(&prepared);
        assert!(natural > 50.0);
    }

    #[test]
    fn test_empty_text() {
        let (backend, font) = setup();
        let prepared = prepare("", &font, &backend, PrepareOptions::default());
        let result = layout(&prepared, 200.0, 24.0);
        assert_eq!(result.line_count, 1);
        assert!((result.height - 24.0).abs() < 0.001);
    }

    #[test]
    fn test_streaming_api() {
        let (backend, font) = setup();
        let prepared = prepare("Hello world foo bar", &font, &backend, PrepareOptions::default());

        let mut cursor = LayoutCursor::default();
        let mut line_count = 0;

        while let Some((_range, next)) = layout_next_line(&prepared, cursor, 80.0) {
            line_count += 1;
            if next.segment_index >= prepared.data.widths.len() && next.grapheme_index == 0 {
                break;
            }
            if next == cursor {
                break; // Safety: prevent infinite loop
            }
            cursor = next;
        }

        assert!(line_count >= 1);
    }

    #[test]
    fn test_cjk_text() {
        let (backend, font) = setup();
        let prepared = prepare(
            "\u{65E5}\u{672C}\u{8A9E}\u{30C6}\u{30B9}\u{30C8}",
            &font,
            &backend,
            PrepareOptions::default(),
        );
        let result = layout(&prepared, 50.0, 24.0);
        assert!(result.line_count >= 2);
    }

    #[test]
    fn test_layout_consistency() {
        let (backend, font) = setup();
        let text = "The quick brown fox jumps over the lazy dog";
        let prepared = prepare(text, &font, &backend, PrepareOptions::default());

        let wide = layout(&prepared, 1000.0, 24.0);
        let narrow = layout(&prepared, 50.0, 24.0);

        assert_eq!(wide.line_count, 1);
        assert!(narrow.line_count > wide.line_count);
        assert!((narrow.height - narrow.line_count as f64 * 24.0).abs() < 0.001);
    }

    #[test]
    fn test_walk_line_ranges() {
        let (backend, font) = setup();
        let prepared = prepare("Hello world test", &font, &backend, PrepareOptions::default());

        let profile = EngineProfile::default();
        let mut ranges = Vec::new();
        walk_line_ranges(&prepared, 80.0, &profile, |r| ranges.push(r));

        assert!(!ranges.is_empty());
        for range in &ranges {
            assert!(range.width >= 0.0);
        }
    }

    #[test]
    fn test_resize_is_cheap() {
        let (backend, font) = setup();
        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        let prepared = prepare(text, &font, &backend, PrepareOptions::default());

        for w in (50..500).step_by(10) {
            let result = layout(&prepared, f64::from(w), 24.0);
            assert!(result.line_count >= 1);
        }
    }
}
