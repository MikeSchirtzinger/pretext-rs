#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]

//! # pretext-rs
//!
//! DOM-free text measurement and line-breaking engine.
//! Rust port of [@chenglou/pretext](https://github.com/chenglou/pretext).
//!
//! Two-phase architecture:
//! - **`prepare()`** -- expensive one-time measurement (uses a [`backend::MeasureBackend`])
//! - **`layout()`** -- cheap reflow over already measured data
//!
//! ## Quick Start
//!
//! ```rust
//! use pretext::{prepare, layout, backend::fixed::FixedWidthBackend, backend::FontSpec};
//!
//! # fn main() -> pretext::Result<()> {
//! let backend = FixedWidthBackend::new();
//! let font = FontSpec::new("16px Inter")?;
//! let prepared = prepare("Hello, world!", &font, &backend, Default::default())?;
//! let result = layout(&prepared, 200.0, 24.0)?;
//! println!("Lines: {}, Height: {}px", result.line_count, result.height);
//! # Ok(())
//! # }
//! ```
//!
//! ## Backends
//!
//! - [`backend::fixed::FixedWidthBackend`] -- deterministic, no deps (testing / server-side estimation)
//! - `backend::CanvasBackend` -- browser `canvas.measureText` (feature `wasm`)
//! - `backend::SkrifaNominalBackend` -- explicit unshaped native advance
//!   estimation (feature `skrifa-nominal`); not production typography

mod analysis;
pub mod backend;
pub mod bidi;
pub mod error;
pub mod gpu_layout;
pub mod inline_flow;
mod line_break;
pub mod types;
pub mod unicode;
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub mod wasm_api;

use unicode_segmentation::UnicodeSegmentation;

pub use error::{Error, Result};

use analysis::{analyze_text, to_prepared_chunks};
use backend::{FontSpec, MeasureBackend};
use line_break::{
    count_lines_full, count_lines_simple, layout_next_line_range, walk_lines_full,
    walk_lines_simple,
};
use types::{
    EngineProfile, LayoutCursor, LayoutLine, LayoutLineRange, LayoutResult, PrepareOptions,
    PrepareProfile, PreparedData, PreparedText, PreparedTextWithSegments, SegmentKind,
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
) -> Result<PreparedText> {
    let data = prepare_internal(text, font, backend, &options, false)?;
    Ok(PreparedText { data: data.0 })
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
) -> Result<PreparedTextWithSegments> {
    let (data, segments) = prepare_internal(text, font, backend, &options, true)?;
    let Some(segments) = segments else {
        return Err(Error::invalid_input(
            "segment retention",
            "prepare_with_segments did not retain analyzed segments",
        ));
    };

    // Compute bidi metadata over the reconstructed normalized text. Char
    // offsets (not bytes) -- see `crate::bidi` on indexing.
    let seg_levels = if segments.is_empty() {
        None
    } else {
        let mut starts = Vec::with_capacity(segments.len());
        let mut normalized = String::new();
        let mut char_cursor: usize = 0;
        for s in &segments {
            starts.push(char_cursor);
            char_cursor += s.chars().count();
            normalized.push_str(s);
        }
        bidi::compute_segment_levels(&normalized, &starts)?
    };

    Ok(PreparedTextWithSegments {
        data,
        segments,
        seg_levels,
    })
}

/// Internal prepare implementation.
fn prepare_internal(
    text: &str,
    font: &FontSpec,
    backend: &dyn MeasureBackend,
    options: &PrepareOptions,
    keep_segments: bool,
) -> Result<(PreparedData, Option<Vec<String>>)> {
    validate_input_complexity(text, options)?;
    let profile = options.profile.clone().unwrap_or_default();
    profile.validate()?;
    let analysis = analyze_text(text, options.white_space);
    validate_analysis_complexity(&analysis, options.max_segments)?;
    measure_analysis(&analysis, font, backend, keep_segments, profile)
}

fn validate_input_complexity(text: &str, options: &PrepareOptions) -> Result<()> {
    if text.len() > options.max_input_bytes {
        return Err(Error::InputTooLarge {
            bytes: text.len(),
            max_bytes: options.max_input_bytes,
        });
    }

    let graphemes = text.graphemes(true).count();
    if graphemes > options.max_graphemes {
        return Err(Error::InputComplexity {
            resource: "text graphemes",
            units: graphemes,
            max_units: options.max_graphemes,
        });
    }
    Ok(())
}

fn validate_analysis_complexity(
    analysis: &analysis::AnalysisResult,
    max_segments: usize,
) -> Result<()> {
    let segments = analysis.segments.len();
    if segments > max_segments {
        Err(Error::InputComplexity {
            resource: "analyzed text segments",
            units: segments,
            max_units: max_segments,
        })
    } else {
        Ok(())
    }
}

/// Measure an already-analyzed text. Split out so `profile_prepare()` can
/// time the analysis phase and the measurement phase independently.
fn measure_analysis(
    analysis: &analysis::AnalysisResult,
    font: &FontSpec,
    backend: &dyn MeasureBackend,
    keep_segments: bool,
    profile: EngineProfile,
) -> Result<(PreparedData, Option<Vec<String>>)> {
    let needs_tab_metrics = analysis
        .segments
        .iter()
        .any(|segment| segment.kind == SegmentKind::Tab);
    let needs_hyphen_metrics = analysis
        .segments
        .iter()
        .any(|segment| segment.kind == SegmentKind::SoftHyphen);
    let tab_stop_advance = if needs_tab_metrics {
        let space_width = validate_metric("space width", backend.measure_space_width(font)?)?;
        validate_metric("tab stop advance", space_width * 8.0)?
    } else {
        0.0
    };
    let hyphen_width = if needs_hyphen_metrics {
        validate_metric("hyphen width", backend.measure_hyphen_width(font)?)?
    } else {
        0.0
    };

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
            SegmentKind::Space => {
                let metrics = validate_segment_metrics(backend.measure_segment(&seg.text, font)?)?;
                widths.push(metrics.width);
                // Trailing spaces "hang" -- they don't count toward fit
                line_end_fit_advances.push(0.0);
                line_end_paint_advances.push(0.0);
                breakable_widths.push(None);
            }
            SegmentKind::PreservedSpace => {
                let metrics = validate_segment_metrics(backend.measure_segment(&seg.text, font)?)?;
                widths.push(metrics.width);
                // Authored spaces remain in materialized text and paint at
                // their measured width, but hang for the line-fit decision.
                line_end_fit_advances.push(0.0);
                line_end_paint_advances.push(metrics.width);
                breakable_widths.push(None);
            }
            SegmentKind::Glue => {
                let width = if seg.text.chars().all(is_zero_width_glue) {
                    0.0
                } else {
                    validate_segment_metrics(backend.measure_segment(&seg.text, font)?)?.width
                };
                widths.push(width);
                line_end_fit_advances.push(width);
                line_end_paint_advances.push(width);
                breakable_widths.push(None);
            }
            SegmentKind::Text => {
                let metrics = validate_segment_metrics(backend.measure_segment(&seg.text, font)?)?;
                if let Some(grapheme_widths) = &metrics.grapheme_widths {
                    let expected = seg.text.graphemes(true).count();
                    if grapheme_widths.len() != expected {
                        return Err(Error::measurement(
                            "measurement backend",
                            format!(
                                "returned {} grapheme widths for a {expected}-grapheme segment",
                                grapheme_widths.len()
                            ),
                        ));
                    }
                }
                widths.push(metrics.width);
                line_end_fit_advances.push(metrics.width);
                line_end_paint_advances.push(metrics.width);
                breakable_widths.push(metrics.grapheme_widths);
            }
        }
    }

    validate_prepared_width_bound(
        &widths,
        &line_end_fit_advances,
        &line_end_paint_advances,
        &kinds,
        &breakable_widths,
        tab_stop_advance,
    )?;

    let chunks = to_prepared_chunks(&analysis.chunks);

    Ok((
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
            profile,
        },
        segment_strings,
    ))
}

fn is_zero_width_glue(character: char) -> bool {
    matches!(character, '\u{2060}' | '\u{FEFF}')
}

fn validate_metric(metric: &'static str, value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(Error::InvalidMetric { metric, value })
    }
}

fn validate_segment_metrics(metrics: backend::SegmentMetrics) -> Result<backend::SegmentMetrics> {
    validate_metric("segment width", metrics.width)?;
    if let Some(widths) = &metrics.grapheme_widths {
        validated_width_sum(widths, "grapheme width", "grapheme width sum")?;
    }
    Ok(metrics)
}

fn validated_width_sum(
    widths: &[f64],
    item_metric: &'static str,
    sum_metric: &'static str,
) -> Result<f64> {
    widths.iter().try_fold(0.0, |total, &width| {
        let width = validate_metric(item_metric, width)?;
        validate_metric(sum_metric, total + width)
    })
}

fn validate_prepared_width_bound(
    widths: &[f64],
    line_end_fit_advances: &[f64],
    line_end_paint_advances: &[f64],
    kinds: &[SegmentKind],
    breakable_widths: &[Option<Vec<f64>>],
    tab_stop_advance: f64,
) -> Result<()> {
    let values = widths
        .iter()
        .zip(line_end_fit_advances)
        .zip(line_end_paint_advances)
        .zip(kinds)
        .zip(breakable_widths);
    let mut line_bound = 0.0_f64;

    for ((((width, fit), paint), kind), grapheme_widths) in values {
        if *kind == SegmentKind::HardBreak {
            line_bound = 0.0;
            continue;
        }
        let grapheme_bound = match grapheme_widths {
            Some(grapheme_widths) => {
                validated_width_sum(grapheme_widths, "grapheme width", "grapheme width sum")?
            }
            None => 0.0,
        };
        let tab_bound = if *kind == SegmentKind::Tab {
            tab_stop_advance
        } else {
            0.0
        };
        let contribution = width
            .max(*fit)
            .max(*paint)
            .max(grapheme_bound)
            .max(tab_bound);
        line_bound = validate_metric("prepared cumulative line width", line_bound + contribution)?;
    }
    Ok(())
}

/// Layout prepared text at a given width -- the fast path.
///
/// Returns line count and total height. This is pure arithmetic over
/// cached widths, without backend measurement or text materialization.
pub fn layout(prepared: &PreparedText, max_width: f64, line_height: f64) -> Result<LayoutResult> {
    layout_with_profile(prepared, max_width, line_height, &prepared.data.profile)
}

/// Layout with a specific engine profile.
///
/// # Errors
///
/// Returns an error when the width, line height, or profile contains invalid
/// numeric values.
#[allow(clippy::cast_precision_loss)]
pub fn layout_with_profile(
    prepared: &PreparedText,
    max_width: f64,
    line_height: f64,
    profile: &EngineProfile,
) -> Result<LayoutResult> {
    validate_max_width(max_width)?;
    validate_positive_finite("line_height", line_height)?;
    profile.validate()?;
    let line_count = if prepared.data.simple_fast_path {
        count_lines_simple(&prepared.data, max_width, profile)
    } else {
        count_lines_full(&prepared.data, max_width, profile)
    };

    let height = validate_metric("layout height", line_count as f64 * line_height)?;
    Ok(LayoutResult { line_count, height })
}

/// Layout with full line information -- returns each line's text and geometry.
///
/// Requires `PreparedTextWithSegments` (from `prepare_with_segments()`).
///
/// # Errors
///
/// Returns an error when `max_width` is negative or `NaN`.
pub fn layout_with_lines(
    prepared: &PreparedTextWithSegments,
    max_width: f64,
) -> Result<Vec<LayoutLine>> {
    validate_max_width(max_width)?;
    prepared.data.profile.validate()?;
    let profile = &prepared.data.profile;
    let mut lines = Vec::new();

    let callback = |internal: line_break::InternalLine| {
        let text = materialize_line_text(
            &prepared.segments,
            &prepared.data.kinds,
            internal.start_segment,
            internal.start_grapheme,
            internal.end_segment,
            internal.end_grapheme,
            internal.ends_with_discretionary_hyphen,
        );

        lines.push(LayoutLine {
            text,
            width: internal.width,
            start: LayoutCursor::new(internal.start_segment, internal.start_grapheme),
            end: LayoutCursor::new(internal.end_segment, internal.end_grapheme),
            ends_with_discretionary_hyphen: internal.ends_with_discretionary_hyphen,
        });
    };

    if prepared.data.simple_fast_path {
        walk_lines_simple(&prepared.data, max_width, profile, callback);
    } else {
        walk_lines_full(&prepared.data, max_width, profile, callback);
    }

    Ok(lines)
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
) -> Result<()>
where
    F: FnMut(LayoutLineRange),
{
    validate_max_width(max_width)?;
    profile.validate()?;
    let callback = |internal: line_break::InternalLine| {
        on_line(LayoutLineRange {
            width: internal.width,
            start: LayoutCursor::new(internal.start_segment, internal.start_grapheme),
            end: LayoutCursor::new(internal.end_segment, internal.end_grapheme),
            ends_with_discretionary_hyphen: internal.ends_with_discretionary_hyphen,
        });
    };

    if prepared.data.simple_fast_path {
        walk_lines_simple(&prepared.data, max_width, profile, callback);
    } else {
        walk_lines_full(&prepared.data, max_width, profile, callback);
    }
    Ok(())
}

/// Layout a single line starting from a cursor -- streaming API.
///
/// Supports per-line `max_width` (for text flowing around images).
/// Call repeatedly with the returned cursor to iterate all lines.
pub fn layout_next_line(
    prepared: &PreparedText,
    start: LayoutCursor,
    max_width: f64,
) -> Result<Option<(LayoutLineRange, LayoutCursor)>> {
    validate_max_width(max_width)?;
    prepared.data.profile.validate()?;
    validate_cursor(&prepared.data, start, "layout_next_line")?;
    Ok(layout_next_line_range(
        &prepared.data,
        start,
        max_width,
        &prepared.data.profile,
    ))
}

/// Measure the natural (unwrapped) width of text.
///
/// Returns the width the text would occupy if it weren't wrapped --
/// effectively the width of the widest forced line (at infinite `max_width`).
pub fn measure_natural_width(prepared: &PreparedText) -> Result<f64> {
    let mut max_w = 0.0;
    walk_line_ranges(prepared, f64::INFINITY, &prepared.data.profile, |range| {
        if range.width > max_w {
            max_w = range.width;
        }
    })?;
    Ok(max_w)
}

fn validate_max_width(max_width: f64) -> Result<()> {
    if max_width.is_nan() || max_width < 0.0 {
        Err(Error::invalid_input(
            "max_width",
            "must be non-negative and not NaN",
        ))
    } else {
        Ok(())
    }
}

fn validate_positive_finite(parameter: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::invalid_input(
            parameter,
            "must be finite and greater than zero",
        ))
    }
}

fn validate_cursor(data: &PreparedData, cursor: LayoutCursor, context: &'static str) -> Result<()> {
    let segment_index = cursor.segment_index();
    let grapheme_index = cursor.grapheme_index();
    let segment_count = data.widths.len();

    let trailing_break_sentinel = segment_count
        .checked_add(1)
        .is_some_and(|terminal| segment_index == terminal)
        && grapheme_index == 0
        && data
            .kinds
            .get(segment_count.saturating_sub(1))
            .is_some_and(|kind| *kind == SegmentKind::HardBreak);
    let valid = if trailing_break_sentinel {
        true
    } else if segment_count == 0 && segment_index == 1 {
        grapheme_index == 0
    } else if segment_index > segment_count {
        false
    } else if segment_index == segment_count {
        grapheme_index == 0
    } else if grapheme_index == 0 {
        true
    } else {
        data.breakable_widths
            .get(segment_index)
            .and_then(Option::as_ref)
            .is_some_and(|widths| grapheme_index < widths.len())
    };

    if valid {
        Ok(())
    } else {
        Err(Error::InvalidCursor {
            context,
            segment_index,
            grapheme_index,
            segment_count,
        })
    }
}

// ---- profile_prepare -------------------------------------------------------

/// Diagnostic helper that runs the prepare pipeline while separating the
/// analysis and measurement timings.
///
/// Matches upstream's `profilePrepare()`. Used by benchmarks and parity
/// harnesses -- on the hot path, call [`prepare`] instead.
#[allow(clippy::needless_pass_by_value)] // API ergonomics: callers pass Default::default()
pub fn profile_prepare(
    text: &str,
    font: &FontSpec,
    backend: &dyn MeasureBackend,
    options: PrepareOptions,
) -> Result<PrepareProfile> {
    use std::time::Instant;

    let t0 = Instant::now();
    validate_input_complexity(text, &options)?;
    let profile = options.profile.clone().unwrap_or_default();
    profile.validate()?;
    let analysis = analyze_text(text, options.white_space);
    validate_analysis_complexity(&analysis, options.max_segments)?;
    let t1 = Instant::now();
    let (data, _segments) = measure_analysis(&analysis, font, backend, false, profile)?;
    let t2 = Instant::now();

    let breakable_segments = data.breakable_widths.iter().filter(|w| w.is_some()).count();

    let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1000.0;

    Ok(PrepareProfile {
        analysis_ms: ms(t0, t1),
        measure_ms: ms(t1, t2),
        total_ms: ms(t0, t2),
        analysis_segments: analysis.segments.len(),
        prepared_segments: data.widths.len(),
        breakable_segments,
    })
}

/// Materialize line text from segment strings.
pub(crate) fn materialize_line_text(
    segments: &[String],
    kinds: &[SegmentKind],
    start_seg: usize,
    start_grapheme: usize,
    end_seg: usize,
    end_grapheme: usize,
    ends_with_discretionary_hyphen: bool,
) -> String {
    let mut text = String::new();
    let segment_count = segments.len().min(kinds.len());
    let end_boundary = end_seg.min(segment_count);
    for (index, (segment, kind)) in segments
        .iter()
        .zip(kinds)
        .enumerate()
        .take(end_boundary)
        .skip(start_seg.min(end_boundary))
    {
        if matches!(kind, SegmentKind::SoftHyphen | SegmentKind::HardBreak) {
            continue;
        }
        if index == start_seg && start_grapheme > 0 {
            let graphemes: Vec<&str> = segment.graphemes(true).collect();
            if let Some(slice) = graphemes.get(start_grapheme.min(graphemes.len())..) {
                for grapheme in slice {
                    text.push_str(grapheme);
                }
            }
        } else {
            text.push_str(segment);
        }
    }

    if end_grapheme > 0
        && let (Some(segment), Some(kind)) = (segments.get(end_seg), kinds.get(end_seg))
        && !matches!(kind, SegmentKind::SoftHyphen | SegmentKind::HardBreak)
    {
        let graphemes: Vec<&str> = segment.graphemes(true).collect();
        let from = if start_seg == end_seg {
            start_grapheme.min(graphemes.len())
        } else {
            0
        };
        let to = end_grapheme.min(graphemes.len());
        if from < to
            && let Some(slice) = graphemes.get(from..to)
        {
            for grapheme in slice {
                text.push_str(grapheme);
            }
        }
    }

    if ends_with_discretionary_hyphen {
        text.push('-');
    }

    text
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::fixed::FixedWidthBackend;
    use std::cell::Cell;

    struct InvalidGraphemeBackend;

    impl MeasureBackend for InvalidGraphemeBackend {
        fn measure_segment(
            &self,
            _text: &str,
            _font: &FontSpec,
        ) -> Result<backend::SegmentMetrics> {
            Ok(backend::SegmentMetrics {
                width: 1.0,
                contains_cjk: false,
                emoji_count: 0,
                grapheme_widths: Some(vec![1.0]),
            })
        }

        fn measure_space_width(&self, _font: &FontSpec) -> Result<f64> {
            Ok(1.0)
        }

        fn measure_hyphen_width(&self, _font: &FontSpec) -> Result<f64> {
            Ok(1.0)
        }
    }

    struct OverflowingAggregateBackend;

    impl MeasureBackend for OverflowingAggregateBackend {
        fn measure_segment(&self, text: &str, _font: &FontSpec) -> Result<backend::SegmentMetrics> {
            Ok(backend::SegmentMetrics {
                width: if text.trim().is_empty() { 1.0 } else { 1e308 },
                contains_cjk: false,
                emoji_count: 0,
                grapheme_widths: None,
            })
        }

        fn measure_space_width(&self, _font: &FontSpec) -> Result<f64> {
            Ok(1.0)
        }

        fn measure_hyphen_width(&self, _font: &FontSpec) -> Result<f64> {
            Ok(1.0)
        }
    }

    struct CountingBackend {
        calls: Cell<usize>,
    }

    impl CountingBackend {
        const fn new() -> Self {
            Self {
                calls: Cell::new(0),
            }
        }

        fn record(&self) {
            self.calls.set(self.calls.get().saturating_add(1));
        }
    }

    impl MeasureBackend for CountingBackend {
        fn measure_segment(&self, text: &str, _font: &FontSpec) -> Result<backend::SegmentMetrics> {
            self.record();
            Ok(backend::SegmentMetrics {
                width: text.graphemes(true).count() as f64,
                contains_cjk: false,
                emoji_count: 0,
                grapheme_widths: None,
            })
        }

        fn measure_space_width(&self, _font: &FontSpec) -> Result<f64> {
            self.record();
            Ok(1.0)
        }

        fn measure_hyphen_width(&self, _font: &FontSpec) -> Result<f64> {
            self.record();
            Ok(1.0)
        }
    }

    struct TextOnlyBackend;

    impl MeasureBackend for TextOnlyBackend {
        fn measure_segment(&self, text: &str, _font: &FontSpec) -> Result<backend::SegmentMetrics> {
            Ok(backend::SegmentMetrics {
                width: text.graphemes(true).count() as f64,
                contains_cjk: false,
                emoji_count: 0,
                grapheme_widths: None,
            })
        }

        fn measure_space_width(&self, _font: &FontSpec) -> Result<f64> {
            Err(Error::measurement("test", "space is unavailable"))
        }

        fn measure_hyphen_width(&self, _font: &FontSpec) -> Result<f64> {
            Err(Error::measurement("test", "hyphen is unavailable"))
        }
    }

    fn setup() -> (FixedWidthBackend, FontSpec) {
        (
            FixedWidthBackend::new(),
            FontSpec::new("16px Inter").expect("test font specification is valid"),
        )
    }

    #[track_caller]
    fn valid<T>(result: Result<T>) -> T {
        result.expect("test input is valid")
    }

    #[test]
    fn test_prepare_and_layout_simple() {
        let (backend, font) = setup();
        let prepared = valid(prepare(
            "Hello, world!",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let result = valid(layout(&prepared, 200.0, 24.0));
        assert!(result.line_count >= 1);
        assert!(result.height > 0.0);
    }

    #[test]
    fn test_layout_wrap() {
        let (backend, font) = setup();
        let prepared = valid(prepare(
            "Hello, world!",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let result = valid(layout(&prepared, 60.0, 24.0));
        assert!(result.line_count >= 2);
    }

    #[test]
    fn test_layout_no_wrap() {
        let (backend, font) = setup();
        let prepared = valid(prepare("Hi", &font, &backend, PrepareOptions::default()));
        let result = valid(layout(&prepared, 200.0, 24.0));
        assert_eq!(result.line_count, 1);
        assert!((result.height - 24.0).abs() < 0.001);
    }

    #[test]
    fn test_layout_with_lines() {
        let (backend, font) = setup();
        let prepared = valid(prepare_with_segments(
            "Hello world test",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let lines = valid(layout_with_lines(&prepared, 80.0));
        assert!(!lines.is_empty());
        let all_text: String = lines
            .iter()
            .map(|l| l.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all_text.contains("Hello"));
        assert!(all_text.contains("world"));
        assert!(all_text.contains("test"));
    }

    #[test]
    fn test_natural_width() {
        let (backend, font) = setup();
        let prepared = valid(prepare(
            "Hello world",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let natural = valid(measure_natural_width(&prepared));
        assert!(natural > 50.0);
    }

    #[test]
    fn test_empty_text() {
        let (backend, font) = setup();
        let prepared = valid(prepare("", &font, &backend, PrepareOptions::default()));
        let result = valid(layout(&prepared, 200.0, 24.0));
        assert_eq!(result.line_count, 1);
        assert!((result.height - 24.0).abs() < 0.001);

        let (line, terminal) = valid(layout_next_line(&prepared, LayoutCursor::default(), 200.0))
            .expect("empty text emits one empty line");
        assert_eq!(line.width, 0.0);
        assert!(
            valid(layout_next_line(&prepared, terminal, 200.0)).is_none(),
            "the cursor returned for empty text must be accepted as terminal"
        );
    }

    #[test]
    fn line_materialization_preserves_whitespace_and_handles_control_segments() {
        let (backend, _) = setup();
        let font = valid(FontSpec::new("10px monospace"));

        let nbsp = valid(prepare_with_segments(
            "a\u{00A0}",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let nbsp_lines = valid(layout_with_lines(&nbsp, f64::INFINITY));
        assert_eq!(nbsp_lines.len(), 1);
        assert_eq!(
            nbsp_lines.first().map(|line| line.text.as_str()),
            Some("a\u{00A0}")
        );

        let preserved = valid(prepare_with_segments(
            "a  \nb",
            &font,
            &backend,
            PrepareOptions {
                white_space: types::WhiteSpaceMode::PreWrap,
                ..PrepareOptions::default()
            },
        ));
        let preserved_lines = valid(layout_with_lines(&preserved, f64::INFINITY));
        assert_eq!(
            preserved_lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            vec!["a  ", "b"]
        );
        assert!(preserved_lines.first().is_some_and(|line| line.width > 6.0));

        let segments = vec!["co".to_owned(), "\u{00AD}".to_owned(), "operate".to_owned()];
        let kinds = vec![
            SegmentKind::Text,
            SegmentKind::SoftHyphen,
            SegmentKind::Text,
        ];
        assert_eq!(
            materialize_line_text(&segments, &kinds, 0, 0, 2, 0, true),
            "co-"
        );
        assert_eq!(
            materialize_line_text(&segments, &kinds, 0, 0, 2, 0, false),
            "co"
        );
        assert_eq!(
            materialize_line_text(&segments, &kinds, 0, 0, 3, 0, false),
            "cooperate"
        );
    }

    #[test]
    fn glue_soft_hyphen_and_streaming_surfaces_agree() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));

        for max_width in [0.0, 6.0, 12.0] {
            let text = "a\u{00A0}b";
            let expected_width = valid(backend.measure_segment(text, &font)).width;
            let prepared = valid(prepare(text, &font, &backend, PrepareOptions::default()));
            let rich = valid(prepare_with_segments(
                text,
                &font,
                &backend,
                PrepareOptions::default(),
            ));
            let lines = valid(layout_with_lines(&rich, max_width));
            assert_eq!(valid(layout(&prepared, max_width, 10.0)).line_count, 1);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0].text, text);
            assert_eq!(lines[0].width, expected_width);

            let (streamed, terminal) = valid(layout_next_line(
                &prepared,
                LayoutCursor::default(),
                max_width,
            ))
            .expect("glued text emits one line");
            assert_eq!(streamed.width, expected_width);
            assert!(valid(layout_next_line(&prepared, terminal, max_width)).is_none());
        }

        let wide = valid(prepare_with_segments(
            "co\u{00AD}operate",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let unbroken = valid(layout_with_lines(&wide, f64::INFINITY));
        assert_eq!(unbroken.len(), 1);
        assert_eq!(unbroken[0].text, "cooperate");
        assert!(!unbroken[0].ends_with_discretionary_hyphen);

        let broken = valid(layout_with_lines(&wide, 18.0));
        assert_eq!(broken.first().map(|line| line.text.as_str()), Some("co-"));
        assert!(
            broken
                .first()
                .is_some_and(|line| line.ends_with_discretionary_hyphen)
        );
        assert!(
            broken
                .iter()
                .skip(1)
                .all(|line| !line.ends_with_discretionary_hyphen)
        );

        let prepared = valid(prepare(
            "co\u{00AD}operate",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let mut walked = Vec::new();
        valid(walk_line_ranges(
            &prepared,
            18.0,
            &EngineProfile::native(),
            |range| walked.push(range),
        ));
        assert_eq!(walked.len(), broken.len());
        assert!(
            walked
                .first()
                .is_some_and(|line| line.ends_with_discretionary_hyphen)
        );

        let mut cursor = LayoutCursor::default();
        let mut streamed = Vec::new();
        while let Some((range, next)) = valid(layout_next_line(&prepared, cursor, 18.0)) {
            streamed.push(range);
            if next.segment_index() >= prepared.segment_count() {
                break;
            }
            assert_ne!(next, cursor, "streaming layout must advance");
            cursor = next;
        }
        assert_eq!(streamed.len(), broken.len());
        assert_eq!(
            streamed
                .iter()
                .map(|line| line.ends_with_discretionary_hyphen)
                .collect::<Vec<_>>(),
            broken
                .iter()
                .map(|line| line.ends_with_discretionary_hyphen)
                .collect::<Vec<_>>()
        );

        let terminal = valid(prepare_with_segments(
            "co\u{00AD}",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let terminal_lines = valid(layout_with_lines(&terminal, f64::INFINITY));
        assert_eq!(terminal_lines.len(), 1);
        assert_eq!(terminal_lines[0].text, "co");
        assert_eq!(terminal_lines[0].width, 12.0);
        assert!(!terminal_lines[0].ends_with_discretionary_hyphen);

        let terminal_prepared = valid(prepare(
            "co\u{00AD}",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let (terminal_range, _) = valid(layout_next_line(
            &terminal_prepared,
            LayoutCursor::default(),
            f64::INFINITY,
        ))
        .expect("terminal soft hyphen emits one text line");
        assert!(!terminal_range.ends_with_discretionary_hyphen);
    }

    #[test]
    fn test_streaming_api() {
        let (backend, font) = setup();
        let prepared = valid(prepare(
            "Hello world foo bar",
            &font,
            &backend,
            PrepareOptions::default(),
        ));

        let mut cursor = LayoutCursor::default();
        let mut line_count = 0;

        while let Some((_range, next)) = valid(layout_next_line(&prepared, cursor, 80.0)) {
            line_count += 1;
            if next.segment_index() >= prepared.segment_count() && next.grapheme_index() == 0 {
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
        let prepared = valid(prepare(
            "\u{65E5}\u{672C}\u{8A9E}\u{30C6}\u{30B9}\u{30C8}",
            &font,
            &backend,
            PrepareOptions::default(),
        ));
        let result = valid(layout(&prepared, 50.0, 24.0));
        assert!(result.line_count >= 2);
    }

    #[test]
    fn test_layout_consistency() {
        let (backend, font) = setup();
        let text = "The quick brown fox jumps over the lazy dog";
        let prepared = valid(prepare(text, &font, &backend, PrepareOptions::default()));

        let wide = valid(layout(&prepared, 1000.0, 24.0));
        let narrow = valid(layout(&prepared, 50.0, 24.0));

        assert_eq!(wide.line_count, 1);
        assert!(narrow.line_count > wide.line_count);
        assert!((narrow.height - narrow.line_count as f64 * 24.0).abs() < 0.001);
    }

    #[test]
    fn test_walk_line_ranges() {
        let (backend, font) = setup();
        let prepared = valid(prepare(
            "Hello world test",
            &font,
            &backend,
            PrepareOptions::default(),
        ));

        let profile = EngineProfile::default();
        let mut ranges = Vec::new();
        valid(walk_line_ranges(&prepared, 80.0, &profile, |r| {
            ranges.push(r);
        }));

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
        let prepared = valid(prepare(text, &font, &backend, PrepareOptions::default()));

        for w in (50..500).step_by(10) {
            let result = valid(layout(&prepared, f64::from(w), 24.0));
            assert!(result.line_count >= 1);
        }
    }

    #[test]
    fn prepared_profile_is_honored_by_default_layout() {
        let (backend, font) = setup();
        let profile = EngineProfile {
            line_fit_epsilon: 100.0,
        };
        let prepared = valid(prepare(
            "hello world",
            &font,
            &backend,
            PrepareOptions {
                profile: Some(profile),
                ..PrepareOptions::default()
            },
        ));

        assert_eq!(valid(layout(&prepared, 50.0, 24.0)).line_count, 1);
    }

    #[test]
    fn invalid_geometry_returns_errors() {
        let (backend, font) = setup();
        let prepared = valid(prepare("hello", &font, &backend, PrepareOptions::default()));

        assert!(layout(&prepared, f64::NAN, 24.0).is_err());
        assert!(layout(&prepared, 100.0, 0.0).is_err());
    }

    #[test]
    fn derived_layout_height_overflow_is_rejected() {
        let (backend, font) = setup();
        let prepared = valid(prepare(
            "hello world",
            &font,
            &backend,
            PrepareOptions::default(),
        ));

        assert!(layout(&prepared, 40.0, f64::MAX).is_err());
    }

    #[test]
    fn preparation_enforces_input_and_backend_shape_limits() {
        let (_, font) = setup();
        assert!(matches!(
            prepare(
                "four",
                &font,
                &FixedWidthBackend::new(),
                PrepareOptions {
                    max_input_bytes: 3,
                    ..PrepareOptions::default()
                },
            ),
            Err(Error::InputTooLarge { .. })
        ));
        assert!(matches!(
            prepare(
                "ab",
                &font,
                &InvalidGraphemeBackend,
                PrepareOptions::default(),
            ),
            Err(Error::Measurement {
                backend: "measurement backend",
                ..
            })
        ));
    }

    #[test]
    fn preparation_complexity_limits_are_typed_and_pre_measurement() {
        let (_, font) = setup();
        let accepted = CountingBackend::new();
        assert!(
            prepare(
                "ab",
                &font,
                &accepted,
                PrepareOptions {
                    max_graphemes: 2,
                    ..PrepareOptions::default()
                },
            )
            .is_ok()
        );
        assert!(accepted.calls.get() > 0);

        let grapheme_rejected = CountingBackend::new();
        assert!(matches!(
            prepare(
                "abc",
                &font,
                &grapheme_rejected,
                PrepareOptions {
                    max_graphemes: 2,
                    ..PrepareOptions::default()
                },
            ),
            Err(Error::InputComplexity {
                resource: "text graphemes",
                units: 3,
                max_units: 2,
            })
        ));
        assert_eq!(grapheme_rejected.calls.get(), 0);

        let segment_accepted = CountingBackend::new();
        assert!(
            prepare(
                "a b",
                &font,
                &segment_accepted,
                PrepareOptions {
                    max_segments: 3,
                    ..PrepareOptions::default()
                },
            )
            .is_ok()
        );

        let segment_rejected = CountingBackend::new();
        assert!(matches!(
            prepare(
                "a b",
                &font,
                &segment_rejected,
                PrepareOptions {
                    max_segments: 2,
                    ..PrepareOptions::default()
                },
            ),
            Err(Error::InputComplexity {
                resource: "analyzed text segments",
                units: 3,
                max_units: 2,
            })
        ));
        assert_eq!(segment_rejected.calls.get(), 0);
    }

    #[test]
    fn preparation_only_requests_metrics_used_by_the_analysis() {
        let (_, font) = setup();
        assert!(prepare("", &font, &TextOnlyBackend, PrepareOptions::default()).is_ok());
        assert!(prepare("abc", &font, &TextOnlyBackend, PrepareOptions::default()).is_ok());

        assert!(matches!(
            prepare(
                "a\tb",
                &font,
                &TextOnlyBackend,
                PrepareOptions {
                    white_space: types::WhiteSpaceMode::PreWrap,
                    ..PrepareOptions::default()
                },
            ),
            Err(Error::Measurement {
                backend: "test",
                ..
            })
        ));
        assert!(matches!(
            prepare(
                "co\u{00AD}operate",
                &font,
                &TextOnlyBackend,
                PrepareOptions::default(),
            ),
            Err(Error::Measurement {
                backend: "test",
                ..
            })
        ));
    }

    #[test]
    fn profile_prepare_enforces_the_same_complexity_limits() {
        let (_, font) = setup();
        let backend = CountingBackend::new();
        assert!(matches!(
            profile_prepare(
                "abc",
                &font,
                &backend,
                PrepareOptions {
                    max_graphemes: 2,
                    ..PrepareOptions::default()
                },
            ),
            Err(Error::InputComplexity {
                resource: "text graphemes",
                ..
            })
        ));
        assert_eq!(backend.calls.get(), 0);

        assert!(matches!(
            profile_prepare(
                "a b",
                &font,
                &backend,
                PrepareOptions {
                    max_segments: 2,
                    ..PrepareOptions::default()
                },
            ),
            Err(Error::InputComplexity {
                resource: "analyzed text segments",
                ..
            })
        ));
        assert_eq!(backend.calls.get(), 0);
    }

    #[test]
    fn cumulative_backend_width_overflow_is_rejected_during_preparation() {
        let (_, font) = setup();
        assert!(matches!(
            prepare(
                "a a",
                &font,
                &OverflowingAggregateBackend,
                PrepareOptions::default(),
            ),
            Err(Error::InvalidMetric {
                metric: "prepared cumulative line width",
                ..
            })
        ));
    }
}
