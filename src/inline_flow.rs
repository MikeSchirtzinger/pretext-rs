//! Inline flow sidecar -- mixed inline runs.
//!
//! Handles layout of heterogeneous inline content like:
//! `"Deploy to @staging completed in 3.2s"`
//! where `@staging` is an atomic chip with different font/styling.
//!
//! Each item can be:
//! - **Breakable** (normal text) -- wraps at word boundaries
//! - **Atomic** (`break: Never`) -- never wraps mid-item (chips, badges, code spans)
//!
//! Items can have `extra_width` for padding/border chrome.

use unicode_segmentation::UnicodeSegmentation;

use crate::backend::{FontSpec, MeasureBackend};
use crate::line_break::{layout_next_line_range, walk_lines_full, walk_lines_simple};
use crate::types::{
    DEFAULT_MAX_GRAPHEMES, DEFAULT_MAX_INPUT_BYTES, DEFAULT_MAX_SEGMENTS, LayoutCursor,
    PrepareOptions, PreparedData, PreparedTextWithSegments,
};
use crate::{Error, Result, materialize_line_text, prepare_with_segments};

/// Default maximum source items accepted by one inline flow.
pub const DEFAULT_MAX_INLINE_ITEMS: usize = 1_024;

/// An inline flow item -- one run of content in a mixed line.
#[derive(Debug, Clone)]
pub struct InlineFlowItem {
    /// Text content of this item.
    pub text: String,
    /// Font for this item (can differ per item).
    pub font: FontSpec,
    /// Break behavior.
    pub break_mode: BreakMode,
    /// Extra width added to this item (e.g., padding + border for chips).
    pub extra_width: f64,
}

/// Break behavior for inline flow items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BreakMode {
    /// Normal text wrapping (can break at word boundaries).
    #[default]
    Normal,
    /// Never break within this item (atomic placement).
    Never,
}

/// Prepared inline flow -- ready for layout.
#[derive(Debug, Clone)]
pub struct PreparedInlineFlow {
    /// Prepared text for each item.
    items: Vec<PreparedInlineItem>,
    /// Collapsed inter-item gap widths (space between items).
    gaps: Vec<f64>,
}

#[derive(Debug, Clone)]
struct PreparedInlineItem {
    source_item_index: usize,
    prepared_with_segments: PreparedTextWithSegments,
    natural_width: f64,
    extra_width: f64,
    break_mode: BreakMode,
}

/// Aggregate resource limits for [`prepare_inline_flow_with_options`].
#[derive(Debug, Clone)]
pub struct InlineFlowOptions {
    /// Maximum number of source items.
    pub max_items: usize,
    /// Maximum total UTF-8 bytes across all source item text.
    pub max_input_bytes: usize,
    /// Maximum total extended grapheme clusters across all source items.
    pub max_graphemes: usize,
    /// Maximum total analyzed segments retained by the prepared flow.
    pub max_segments: usize,
}

impl Default for InlineFlowOptions {
    fn default() -> Self {
        Self {
            max_items: DEFAULT_MAX_INLINE_ITEMS,
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_graphemes: DEFAULT_MAX_GRAPHEMES,
            max_segments: DEFAULT_MAX_SEGMENTS,
        }
    }
}

/// A fragment of an inline flow item on a line.
#[derive(Debug, Clone)]
pub struct InlineFlowFragment {
    /// Index of the source item.
    pub item_index: usize,
    /// Text content of this fragment.
    pub text: String,
    /// Gap before this fragment (inter-item space).
    pub gap_before: f64,
    /// Width occupied by this fragment (including `extra_width`).
    pub occupied_width: f64,
    /// Start cursor within the item's prepared text.
    pub start: LayoutCursor,
    /// End cursor within the item's prepared text.
    pub end: LayoutCursor,
    /// Whether this fragment paints a selected U+00AD as a trailing hyphen.
    pub ends_with_discretionary_hyphen: bool,
}

/// A line from inline flow layout.
#[derive(Debug, Clone)]
pub struct InlineFlowLine {
    /// Fragments on this line.
    pub fragments: Vec<InlineFlowFragment>,
    /// Total width of this line.
    pub width: f64,
}

/// Cursor position within an inline flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InlineFlowCursor {
    /// Current item index.
    item_index: usize,
    /// Cursor within the current item's prepared text.
    layout_cursor: LayoutCursor,
}

impl InlineFlowCursor {
    /// Current item index.
    #[must_use]
    pub const fn item_index(self) -> usize {
        self.item_index
    }

    /// Cursor within the current item's prepared text.
    #[must_use]
    pub const fn layout_cursor(self) -> LayoutCursor {
        self.layout_cursor
    }

    const fn new(item_index: usize, layout_cursor: LayoutCursor) -> Self {
        Self {
            item_index,
            layout_cursor,
        }
    }
}

/// Prepare an inline flow for layout.
///
/// # Errors
///
/// Returns an error when an item has invalid geometry, exceeds the configured
/// text limit, or cannot be measured by the selected backend.
pub fn prepare_inline_flow(
    items: &[InlineFlowItem],
    backend: &dyn MeasureBackend,
) -> Result<PreparedInlineFlow> {
    prepare_inline_flow_with_options(items, backend, InlineFlowOptions::default())
}

/// Prepare an inline flow with aggregate resource limits.
///
/// Validation of item count, total bytes, and total graphemes happens before
/// any backend measurement. Segment capacity is charged as each bounded item
/// is analyzed. No prepared-data copy is retained alongside the rich prepared
/// representation.
///
/// # Errors
///
/// Returns an error when aggregate limits are exceeded, an item has invalid
/// geometry, or the backend cannot measure required text.
#[allow(clippy::needless_pass_by_value)]
pub fn prepare_inline_flow_with_options(
    items: &[InlineFlowItem],
    backend: &dyn MeasureBackend,
    options: InlineFlowOptions,
) -> Result<PreparedInlineFlow> {
    validate_inline_source_limits(items, &options)?;

    let mut prepared_items = Vec::with_capacity(items.len());
    let mut gaps = Vec::with_capacity(items.len());
    let mut pending_gap_width = 0.0_f64;
    let mut prepared_segments = 0_usize;

    for (i, item) in items.iter().enumerate() {
        if !item.extra_width.is_finite() || item.extra_width < 0.0 {
            return Err(Error::invalid_input(
                "inline item extra_width",
                "must be finite and non-negative",
            ));
        }
        let has_leading_whitespace = item
            .text
            .chars()
            .next()
            .is_some_and(is_collapsible_boundary_whitespace);
        let has_trailing_whitespace = item
            .text
            .chars()
            .next_back()
            .is_some_and(is_collapsible_boundary_whitespace);
        let text = item.text.trim_matches(is_collapsible_boundary_whitespace);
        if text.is_empty() {
            if item.extra_width > 0.0 || item.break_mode != BreakMode::Normal {
                return Err(Error::invalid_input(
                    "empty inline item",
                    "empty or whitespace-only items cannot carry chrome width or atomic break behavior",
                ));
            }
            if item.text.chars().any(is_collapsible_boundary_whitespace) && pending_gap_width == 0.0
            {
                pending_gap_width = validated_space_width(backend, &item.font)?;
            }
            continue;
        }

        let gap_before = if pending_gap_width > 0.0 {
            pending_gap_width
        } else if has_leading_whitespace {
            validated_space_width(backend, &item.font)?
        } else {
            0.0
        };
        let remaining_segments = options.max_segments.saturating_sub(prepared_segments);
        let prepared_with_segments = prepare_with_segments(
            text,
            &item.font,
            backend,
            PrepareOptions {
                max_input_bytes: options.max_input_bytes,
                max_graphemes: options.max_graphemes,
                max_segments: remaining_segments,
                ..PrepareOptions::default()
            },
        )?;
        prepared_segments = prepared_segments
            .checked_add(prepared_with_segments.data.segment_count())
            .ok_or(Error::ArithmeticOverflow {
                operation: "charging inline-flow segments",
            })?;
        if prepared_segments > options.max_segments {
            return Err(Error::InputComplexity {
                resource: "inline-flow segments",
                units: prepared_segments,
                max_units: options.max_segments,
            });
        }
        let natural_width = measure_natural_width_data(&prepared_with_segments.data);

        prepared_items.push(PreparedInlineItem {
            source_item_index: i,
            prepared_with_segments,
            natural_width,
            extra_width: item.extra_width,
            break_mode: item.break_mode,
        });

        gaps.push(gap_before);
        if has_trailing_whitespace {
            pending_gap_width = validated_space_width(backend, &item.font)?;
        } else {
            pending_gap_width = 0.0;
        }
    }

    Ok(PreparedInlineFlow {
        items: prepared_items,
        gaps,
    })
}

fn validate_inline_source_limits(
    items: &[InlineFlowItem],
    options: &InlineFlowOptions,
) -> Result<()> {
    if items.len() > options.max_items {
        return Err(Error::InputComplexity {
            resource: "inline-flow items",
            units: items.len(),
            max_units: options.max_items,
        });
    }

    let mut bytes = 0_usize;
    let mut graphemes = 0_usize;
    for item in items {
        bytes = bytes
            .checked_add(item.text.len())
            .ok_or(Error::ArithmeticOverflow {
                operation: "counting inline-flow input bytes",
            })?;
        graphemes = graphemes
            .checked_add(item.text.graphemes(true).count())
            .ok_or(Error::ArithmeticOverflow {
                operation: "counting inline-flow graphemes",
            })?;
    }

    if bytes > options.max_input_bytes {
        return Err(Error::InputTooLarge {
            bytes,
            max_bytes: options.max_input_bytes,
        });
    }
    if graphemes > options.max_graphemes {
        return Err(Error::InputComplexity {
            resource: "inline-flow graphemes",
            units: graphemes,
            max_units: options.max_graphemes,
        });
    }
    Ok(())
}

fn measure_natural_width_data(data: &PreparedData) -> f64 {
    let mut max_width = 0.0_f64;
    let mut record = |line: crate::line_break::InternalLine| {
        max_width = max_width.max(line.width);
    };
    if data.simple_fast_path {
        walk_lines_simple(data, f64::INFINITY, &data.profile, &mut record);
    } else {
        walk_lines_full(data, f64::INFINITY, &data.profile, &mut record);
    }
    max_width
}

/// Layout the next line of an inline flow.
///
/// Returns the line and the cursor for the next line, or `None` if done.
///
/// # Errors
///
/// Returns an error for invalid geometry or a cursor that does not belong to
/// this prepared flow.
#[allow(clippy::too_many_lines)]
pub fn layout_next_inline_flow_line(
    prepared: &PreparedInlineFlow,
    max_width: f64,
    start: InlineFlowCursor,
) -> Result<Option<(InlineFlowLine, InlineFlowCursor)>> {
    if max_width.is_nan() || max_width < 0.0 {
        return Err(Error::invalid_input(
            "max_width",
            "must be non-negative and not NaN",
        ));
    }
    if start.item_index > prepared.items.len() {
        return Err(Error::InvalidCursor {
            context: "inline flow",
            segment_index: start.item_index,
            grapheme_index: 0,
            segment_count: prepared.items.len(),
        });
    }
    if start.item_index == prepared.items.len() {
        if start.layout_cursor == LayoutCursor::default() {
            return Ok(None);
        }
        return Err(Error::InvalidCursor {
            context: "inline flow terminal item",
            segment_index: start.layout_cursor.segment_index(),
            grapheme_index: start.layout_cursor.grapheme_index(),
            segment_count: 0,
        });
    }

    let mut fragments = Vec::new();
    let mut line_w = 0.0;
    let mut item_idx = start.item_index;
    let mut item_cursor = start.layout_cursor;

    while item_idx < prepared.items.len() {
        let Some(item) = prepared.items.get(item_idx) else {
            return Err(Error::invalid_input(
                "prepared inline flow",
                "item cursor is outside the prepared item vector",
            ));
        };
        let gap = if fragments.is_empty() {
            0.0
        } else {
            prepared.gaps.get(item_idx).copied().ok_or_else(|| {
                Error::invalid_input("prepared inline flow", "gap vector is out of sync")
            })?
        };

        let total_extra = item.extra_width;

        if item.break_mode == BreakMode::Never {
            // Atomic item -- place entirely or wrap
            let needed = gap + item.natural_width + total_extra;
            validate_inline_width("atomic item width", needed)?;
            let candidate_line_width = line_w + needed;
            validate_inline_width("inline line width", candidate_line_width)?;

            if candidate_line_width > max_width && !fragments.is_empty() {
                // Wrap before this item
                break;
            }

            let occupied_width = item.natural_width + total_extra;
            validate_inline_width("atomic fragment width", occupied_width)?;

            fragments.push(InlineFlowFragment {
                item_index: item.source_item_index,
                text: materialize_line_text(
                    item.prepared_with_segments.segments(),
                    &item.prepared_with_segments.data.kinds,
                    0,
                    0,
                    item.prepared_with_segments.data.segment_count(),
                    0,
                    false,
                ),
                gap_before: gap,
                occupied_width,
                start: LayoutCursor::default(),
                end: LayoutCursor::new(item.prepared_with_segments.data.segment_count(), 0),
                ends_with_discretionary_hyphen: false,
            });

            line_w = candidate_line_width;
            item_idx += 1;
            item_cursor = LayoutCursor::default();
        } else {
            // Breakable item -- try to fit, or fill partially
            let available = max_width - line_w - gap - total_extra;

            if available <= 0.0 && !fragments.is_empty() {
                // No room -- wrap before this item
                break;
            }

            // Try fitting the entire remaining text of this item
            let remaining_width = if item_cursor == LayoutCursor::default() {
                item.natural_width
            } else {
                // Measure remaining from cursor
                let mut w: f64 = 0.0;
                let mut c = item_cursor;
                while let Some((range, next)) = layout_next_line_range(
                    &item.prepared_with_segments.data,
                    c,
                    f64::INFINITY,
                    &item.prepared_with_segments.data.profile,
                ) {
                    w = w.max(range.width);
                    if next == c {
                        break;
                    }
                    c = next;
                    if c.segment_index() >= item.prepared_with_segments.data.segment_count() {
                        break;
                    }
                }
                w
            };

            if remaining_width + total_extra <= available || fragments.is_empty() {
                if remaining_width + total_extra <= available {
                    // Entire item fits on this line
                    let text = materialize_line_text(
                        item.prepared_with_segments.segments(),
                        &item.prepared_with_segments.data.kinds,
                        item_cursor.segment_index(),
                        item_cursor.grapheme_index(),
                        item.prepared_with_segments.data.segment_count(),
                        0,
                        false,
                    );

                    let occupied_width = remaining_width + total_extra;
                    validate_inline_width("inline fragment width", occupied_width)?;
                    fragments.push(InlineFlowFragment {
                        item_index: item.source_item_index,
                        text,
                        gap_before: gap,
                        occupied_width,
                        start: item_cursor,
                        end: LayoutCursor::new(item.prepared_with_segments.data.segment_count(), 0),
                        ends_with_discretionary_hyphen: false,
                    });

                    line_w += gap + occupied_width;
                    validate_inline_width("inline line width", line_w)?;
                    item_idx += 1;
                    item_cursor = LayoutCursor::default();
                } else {
                    // Partial fit -- use layout_next_line to find the break point
                    let effective_max = available.max(0.0);

                    if let Some((range, next)) = layout_next_line_range(
                        &item.prepared_with_segments.data,
                        item_cursor,
                        effective_max,
                        &item.prepared_with_segments.data.profile,
                    ) {
                        // Materialize fragment text
                        let text = materialize_line_text(
                            item.prepared_with_segments.segments(),
                            &item.prepared_with_segments.data.kinds,
                            range.start.segment_index(),
                            range.start.grapheme_index(),
                            range.end.segment_index(),
                            range.end.grapheme_index(),
                            range.ends_with_discretionary_hyphen,
                        );

                        let occupied_width = range.width + total_extra;
                        validate_inline_width("inline fragment width", occupied_width)?;
                        fragments.push(InlineFlowFragment {
                            item_index: item.source_item_index,
                            text,
                            gap_before: gap,
                            occupied_width,
                            start: range.start,
                            end: range.end,
                            ends_with_discretionary_hyphen: range.ends_with_discretionary_hyphen,
                        });

                        line_w += gap + occupied_width;
                        validate_inline_width("inline line width", line_w)?;
                        item_cursor = next;

                        // If we consumed the entire item, advance
                        if next.segment_index() >= item.prepared_with_segments.data.segment_count()
                        {
                            item_idx += 1;
                            item_cursor = LayoutCursor::default();
                        }
                    }
                    break; // Line is full after partial fit
                }
            } else {
                // Doesn't fit -- wrap before this item
                break;
            }
        }
    }

    if fragments.is_empty() {
        return if item_idx >= prepared.items.len() {
            Ok(None)
        } else {
            Err(Error::StateUnavailable {
                state: "inline flow cursor progress",
            })
        };
    }

    let total_width = line_w;

    Ok(Some((
        InlineFlowLine {
            fragments,
            width: total_width,
        },
        InlineFlowCursor::new(item_idx, item_cursor),
    )))
}

/// Layout all lines of an inline flow.
///
/// # Errors
///
/// Returns an error when `max_width` or an internal cursor is invalid.
pub fn layout_inline_flow(
    prepared: &PreparedInlineFlow,
    max_width: f64,
) -> Result<Vec<InlineFlowLine>> {
    let mut lines = Vec::new();
    let mut cursor = InlineFlowCursor::default();

    while let Some((line, next)) = layout_next_inline_flow_line(prepared, max_width, cursor)? {
        lines.push(line);
        if next == cursor {
            return Err(Error::StateUnavailable {
                state: "inline flow cursor progress",
            });
        }
        cursor = next;
    }

    Ok(lines)
}

/// Measure total height of an inline flow.
///
/// # Errors
///
/// Returns an error when layout geometry is invalid.
#[allow(clippy::cast_precision_loss)]
pub fn measure_inline_flow(
    prepared: &PreparedInlineFlow,
    max_width: f64,
    line_height: f64,
) -> Result<f64> {
    if !line_height.is_finite() || line_height <= 0.0 {
        return Err(Error::invalid_input(
            "line_height",
            "must be finite and greater than zero",
        ));
    }
    let lines = layout_inline_flow(prepared, max_width)?;
    let height = lines.len() as f64 * line_height;
    if height.is_finite() {
        Ok(height)
    } else {
        Err(Error::invalid_input(
            "inline flow height",
            "derived value is not finite",
        ))
    }
}

fn validate_inline_width(parameter: &'static str, width: f64) -> Result<()> {
    if width.is_finite() && width >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidMetric {
            metric: parameter,
            value: width,
        })
    }
}

fn is_collapsible_boundary_whitespace(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\n' | '\u{000C}' | '\r')
}

fn validated_space_width(backend: &dyn MeasureBackend, font: &FontSpec) -> Result<f64> {
    let width = backend.measure_space_width(font)?;
    if width.is_finite() && width >= 0.0 {
        Ok(width)
    } else {
        Err(Error::InvalidMetric {
            metric: "inline flow gap",
            value: width,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::fixed::FixedWidthBackend;

    #[track_caller]
    fn valid<T>(result: Result<T>) -> T {
        result.expect("test input is valid")
    }

    #[test]
    fn test_inline_flow_basic() {
        let backend = FixedWidthBackend::new();
        let items = vec![
            InlineFlowItem {
                text: "Deploy to".to_string(),
                font: valid(FontSpec::new("16px Inter")),
                break_mode: BreakMode::Normal,
                extra_width: 0.0,
            },
            InlineFlowItem {
                text: "@staging".to_string(),
                font: valid(FontSpec::new("14px monospace")),
                break_mode: BreakMode::Never,
                extra_width: 8.0, // Chip padding
            },
            InlineFlowItem {
                text: "completed".to_string(),
                font: valid(FontSpec::new("16px Inter")),
                break_mode: BreakMode::Normal,
                extra_width: 0.0,
            },
        ];

        let prepared = valid(prepare_inline_flow(&items, &backend));
        let lines = valid(layout_inline_flow(&prepared, 300.0));
        assert!(!lines.is_empty());

        // All items should appear in the output
        let all_text: String = lines
            .iter()
            .flat_map(|l| l.fragments.iter())
            .map(|f| f.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all_text.contains("Deploy"));
        assert!(all_text.contains("@staging"));
        assert!(all_text.contains("completed"));
    }

    #[test]
    fn test_atomic_item_no_break() {
        let backend = FixedWidthBackend::new();
        let items = vec![
            InlineFlowItem {
                text: "Hello".to_string(),
                font: valid(FontSpec::new("16px Inter")),
                break_mode: BreakMode::Normal,
                extra_width: 0.0,
            },
            InlineFlowItem {
                text: "@very-long-username".to_string(),
                font: valid(FontSpec::new("16px Inter")),
                break_mode: BreakMode::Never, // Atomic -- must not break
                extra_width: 8.0,
            },
        ];

        let prepared = valid(prepare_inline_flow(&items, &backend));
        let lines = valid(layout_inline_flow(&prepared, 100.0));

        // The atomic item should appear as a single fragment
        for line in &lines {
            for frag in &line.fragments {
                if frag.item_index == 1 {
                    assert!(frag.text.contains("@very-long-username"));
                }
            }
        }
    }

    #[test]
    fn derived_inline_geometry_overflow_is_rejected() {
        let backend = FixedWidthBackend::new();
        let prepared = valid(prepare_inline_flow(
            &[InlineFlowItem {
                text: "aa".to_owned(),
                font: valid(FontSpec::new("8e307px Inter")),
                break_mode: BreakMode::Never,
                extra_width: 1e308,
            }],
            &backend,
        ));

        assert!(layout_inline_flow(&prepared, f64::INFINITY).is_err());
    }

    #[test]
    fn derived_inline_height_overflow_is_rejected() {
        let backend = FixedWidthBackend::new();
        let items = ["a", "b"].map(|text| InlineFlowItem {
            text: text.to_owned(),
            font: valid(FontSpec::new("16px Inter")),
            break_mode: BreakMode::Never,
            extra_width: 0.0,
        });
        let prepared = valid(prepare_inline_flow(&items, &backend));

        assert!(measure_inline_flow(&prepared, 10.0, f64::MAX).is_err());
    }

    fn normal_item(text: &str, font: &FontSpec) -> InlineFlowItem {
        InlineFlowItem {
            text: text.to_owned(),
            font: font.clone(),
            break_mode: BreakMode::Normal,
            extra_width: 0.0,
        }
    }

    #[test]
    fn adjacent_items_do_not_invent_whitespace() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));
        let prepared = valid(prepare_inline_flow(
            &[normal_item("foo", &font), normal_item("bar", &font)],
            &backend,
        ));
        let lines = valid(layout_inline_flow(&prepared, 100.0));

        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].fragments.len(), 2);
        assert_eq!(lines[0].fragments[1].gap_before, 0.0);
        assert_eq!(
            lines[0]
                .fragments
                .iter()
                .map(|fragment| fragment.text.as_str())
                .collect::<String>(),
            "foobar"
        );
    }

    #[test]
    fn authored_boundary_whitespace_becomes_one_gap() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));
        let prepared = valid(prepare_inline_flow(
            &[normal_item("foo ", &font), normal_item("bar", &font)],
            &backend,
        ));
        let lines = valid(layout_inline_flow(&prepared, 100.0));

        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0].fragments[1].gap_before,
            valid(backend.measure_space_width(&font))
        );
    }

    #[test]
    fn whitespace_only_bridge_preserves_source_item_indices() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));
        let prepared = valid(prepare_inline_flow(
            &[
                normal_item("a", &font),
                normal_item(" \t", &font),
                normal_item("b", &font),
            ],
            &backend,
        ));
        let lines = valid(layout_inline_flow(&prepared, 100.0));

        assert_eq!(lines[0].fragments.len(), 2);
        assert_eq!(lines[0].fragments[1].item_index, 2);
        assert_eq!(
            lines[0].fragments[1].gap_before,
            valid(backend.measure_space_width(&font))
        );
    }

    #[test]
    fn non_breaking_space_is_content_not_boundary_trim() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));
        let prepared = valid(prepare_inline_flow(
            &[normal_item("a\u{00A0}b", &font)],
            &backend,
        ));
        let lines = valid(layout_inline_flow(&prepared, 12.0));
        let expected_width = valid(backend.measure_segment("a\u{00A0}b", &font)).width;

        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].fragments[0].text, "a\u{00A0}b");
        assert_eq!(lines[0].fragments[0].occupied_width, expected_width);
    }

    #[test]
    fn repeated_midword_fragments_neither_duplicate_nor_drop_text() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));
        let original = "abcdefghijklmnopqrst";
        let prepared = valid(prepare_inline_flow(
            &[normal_item(original, &font)],
            &backend,
        ));
        let lines = valid(layout_inline_flow(&prepared, 18.0));
        let reconstructed: String = lines
            .iter()
            .flat_map(|line| line.fragments.iter())
            .map(|fragment| fragment.text.as_str())
            .collect();

        assert_eq!(lines.len(), 7);
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn inline_flow_paints_only_a_selected_soft_hyphen() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));
        let prepared = valid(prepare_inline_flow(
            &[normal_item("co\u{00AD}operate", &font)],
            &backend,
        ));
        let lines = valid(layout_inline_flow(&prepared, 18.0));
        let fragments: Vec<_> = lines
            .iter()
            .flat_map(|line| line.fragments.iter())
            .collect();

        assert_eq!(
            fragments.first().map(|fragment| fragment.text.as_str()),
            Some("co-")
        );
        assert!(
            fragments
                .first()
                .is_some_and(|fragment| fragment.ends_with_discretionary_hyphen)
        );
        assert!(
            fragments
                .iter()
                .skip(1)
                .all(|fragment| !fragment.ends_with_discretionary_hyphen)
        );

        let terminal = valid(prepare_inline_flow(
            &[normal_item("co\u{00AD}", &font)],
            &backend,
        ));
        let terminal_lines = valid(layout_inline_flow(&terminal, f64::INFINITY));
        let terminal_fragment = terminal_lines
            .first()
            .and_then(|line| line.fragments.first())
            .expect("terminal soft hyphen produces one inline fragment");
        assert_eq!(terminal_fragment.text, "co");
        assert!(!terminal_fragment.ends_with_discretionary_hyphen);
    }

    #[test]
    fn discarded_empty_items_cannot_hide_geometry_or_break_behavior() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));

        for item in [
            InlineFlowItem {
                text: String::new(),
                font: font.clone(),
                break_mode: BreakMode::Normal,
                extra_width: 4.0,
            },
            InlineFlowItem {
                text: "  ".to_owned(),
                font: font.clone(),
                break_mode: BreakMode::Never,
                extra_width: 0.0,
            },
        ] {
            assert!(matches!(
                prepare_inline_flow(&[item], &backend),
                Err(Error::InvalidInput {
                    parameter: "empty inline item",
                    ..
                })
            ));
        }
    }

    #[test]
    fn aggregate_inline_flow_limits_are_enforced() {
        let backend = FixedWidthBackend::new();
        let font = valid(FontSpec::new("10px monospace"));
        let items = [normal_item("a", &font), normal_item("b", &font)];

        assert!(matches!(
            prepare_inline_flow_with_options(
                &items,
                &backend,
                InlineFlowOptions {
                    max_items: 1,
                    ..InlineFlowOptions::default()
                },
            ),
            Err(Error::InputComplexity {
                resource: "inline-flow items",
                units: 2,
                max_units: 1,
            })
        ));
        assert!(matches!(
            prepare_inline_flow_with_options(
                &items,
                &backend,
                InlineFlowOptions {
                    max_input_bytes: 1,
                    ..InlineFlowOptions::default()
                },
            ),
            Err(Error::InputTooLarge {
                bytes: 2,
                max_bytes: 1,
            })
        ));
        assert!(matches!(
            prepare_inline_flow_with_options(
                &items,
                &backend,
                InlineFlowOptions {
                    max_graphemes: 1,
                    ..InlineFlowOptions::default()
                },
            ),
            Err(Error::InputComplexity {
                resource: "inline-flow graphemes",
                units: 2,
                max_units: 1,
            })
        ));
        assert!(matches!(
            prepare_inline_flow_with_options(
                &items,
                &backend,
                InlineFlowOptions {
                    max_segments: 1,
                    ..InlineFlowOptions::default()
                },
            ),
            Err(Error::InputComplexity {
                resource: "analyzed text segments",
                units: 1,
                max_units: 0,
            })
        ));

        assert!(
            prepare_inline_flow_with_options(
                &items,
                &backend,
                InlineFlowOptions {
                    max_items: 2,
                    max_input_bytes: 2,
                    max_graphemes: 2,
                    max_segments: 2,
                },
            )
            .is_ok()
        );
    }
}
