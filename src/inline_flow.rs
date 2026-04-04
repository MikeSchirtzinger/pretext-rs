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

use crate::backend::{FontSpec, MeasureBackend};
use crate::types::{LayoutCursor, PrepareOptions, PreparedText, PreparedTextWithSegments};
use crate::{layout_next_line, measure_natural_width, prepare, prepare_with_segments};

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
    prepared: PreparedText,
    prepared_with_segments: PreparedTextWithSegments,
    natural_width: f64,
    extra_width: f64,
    break_mode: BreakMode,
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
    pub item_index: usize,
    /// Cursor within the current item's prepared text.
    pub layout_cursor: LayoutCursor,
}

/// Prepare an inline flow for layout.
pub fn prepare_inline_flow(
    items: &[InlineFlowItem],
    backend: &dyn MeasureBackend,
) -> PreparedInlineFlow {
    let mut prepared_items = Vec::with_capacity(items.len());
    let mut gaps = Vec::with_capacity(items.len());

    for (i, item) in items.iter().enumerate() {
        let text = item.text.trim();
        let prepared = prepare(text, &item.font, backend, PrepareOptions::default());
        let prepared_with_segments =
            prepare_with_segments(text, &item.font, backend, PrepareOptions::default());
        let natural_width = measure_natural_width(&prepared);

        prepared_items.push(PreparedInlineItem {
            prepared,
            prepared_with_segments,
            natural_width,
            extra_width: item.extra_width,
            break_mode: item.break_mode,
        });

        // Inter-item gap: measure collapsed space width
        if i > 0 {
            let gap = backend.measure_space_width(&item.font);
            gaps.push(gap);
        } else {
            gaps.push(0.0); // No gap before first item
        }
    }

    PreparedInlineFlow {
        items: prepared_items,
        gaps,
    }
}

/// Layout the next line of an inline flow.
///
/// Returns the line and the cursor for the next line, or `None` if done.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn layout_next_inline_flow_line(
    prepared: &PreparedInlineFlow,
    max_width: f64,
    start: InlineFlowCursor,
) -> Option<(InlineFlowLine, InlineFlowCursor)> {
    if start.item_index >= prepared.items.len() {
        return None;
    }

    let mut fragments = Vec::new();
    let mut line_w = 0.0;
    let mut item_idx = start.item_index;
    let mut item_cursor = start.layout_cursor;

    while item_idx < prepared.items.len() {
        let item = &prepared.items[item_idx];
        let gap = if fragments.is_empty() {
            0.0
        } else {
            prepared.gaps[item_idx]
        };

        let total_extra = item.extra_width;

        if item.break_mode == BreakMode::Never {
            // Atomic item -- place entirely or wrap
            let needed = gap + item.natural_width + total_extra;

            if line_w + needed > max_width && !fragments.is_empty() {
                // Wrap before this item
                break;
            }

            fragments.push(InlineFlowFragment {
                item_index: item_idx,
                text: item.prepared_with_segments.segments.join(""),
                gap_before: gap,
                occupied_width: item.natural_width + total_extra,
                start: LayoutCursor::default(),
                end: LayoutCursor {
                    segment_index: item.prepared.data.widths.len(),
                    grapheme_index: 0,
                },
            });

            line_w += needed;
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
                while let Some((range, next)) =
                    layout_next_line(&item.prepared, c, f64::INFINITY)
                {
                    w = w.max(range.width);
                    if next == c {
                        break;
                    }
                    c = next;
                    if c.segment_index >= item.prepared.data.widths.len() {
                        break;
                    }
                }
                w
            };

            if remaining_width + total_extra <= available || fragments.is_empty() {
                if remaining_width + total_extra <= available {
                    // Entire item fits on this line
                    let text: String = item.prepared_with_segments.segments
                        [item_cursor.segment_index..]
                        .join("");

                    fragments.push(InlineFlowFragment {
                        item_index: item_idx,
                        text: text.trim().to_string(),
                        gap_before: gap,
                        occupied_width: remaining_width + total_extra,
                        start: item_cursor,
                        end: LayoutCursor {
                            segment_index: item.prepared.data.widths.len(),
                            grapheme_index: 0,
                        },
                    });

                    line_w += gap + remaining_width + total_extra;
                    item_idx += 1;
                    item_cursor = LayoutCursor::default();
                } else {
                    // Partial fit -- use layout_next_line to find the break point
                    let effective_max = if available > 0.0 {
                        available
                    } else {
                        max_width - total_extra
                    };

                    if let Some((range, next)) =
                        layout_next_line(&item.prepared, item_cursor, effective_max)
                    {
                        // Materialize fragment text
                        let text = materialize_fragment_text(
                            &item.prepared_with_segments.segments,
                            &range.start,
                            &range.end,
                        );

                        fragments.push(InlineFlowFragment {
                            item_index: item_idx,
                            text,
                            gap_before: gap,
                            occupied_width: range.width + total_extra,
                            start: range.start,
                            end: range.end,
                        });

                        line_w += gap + range.width + total_extra;
                        item_cursor = next;

                        // If we consumed the entire item, advance
                        if next.segment_index >= item.prepared.data.widths.len() {
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
        return None;
    }

    let total_width = line_w;

    Some((
        InlineFlowLine {
            fragments,
            width: total_width,
        },
        InlineFlowCursor {
            item_index: item_idx,
            layout_cursor: item_cursor,
        },
    ))
}

/// Layout all lines of an inline flow.
#[must_use]
pub fn layout_inline_flow(prepared: &PreparedInlineFlow, max_width: f64) -> Vec<InlineFlowLine> {
    let mut lines = Vec::new();
    let mut cursor = InlineFlowCursor::default();

    while let Some((line, next)) = layout_next_inline_flow_line(prepared, max_width, cursor) {
        lines.push(line);
        if next == cursor {
            break; // Safety
        }
        cursor = next;
    }

    lines
}

/// Measure total height of an inline flow.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn measure_inline_flow(
    prepared: &PreparedInlineFlow,
    max_width: f64,
    line_height: f64,
) -> f64 {
    let lines = layout_inline_flow(prepared, max_width);
    lines.len() as f64 * line_height
}

/// Materialize text from segment strings between two cursors.
fn materialize_fragment_text(
    segments: &[String],
    start: &LayoutCursor,
    end: &LayoutCursor,
) -> String {
    use unicode_segmentation::UnicodeSegmentation;

    let mut text = String::new();
    for (i, segment) in segments
        .iter()
        .enumerate()
        .take(end.segment_index.min(segments.len()))
        .skip(start.segment_index)
    {
        if i == start.segment_index && start.grapheme_index > 0 {
            let graphemes: Vec<&str> = segment.graphemes(true).collect();
            for g in &graphemes[start.grapheme_index..] {
                text.push_str(g);
            }
        } else {
            text.push_str(segment);
        }
    }
    text.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::fixed::FixedWidthBackend;

    #[test]
    fn test_inline_flow_basic() {
        let backend = FixedWidthBackend::new();
        let items = vec![
            InlineFlowItem {
                text: "Deploy to".to_string(),
                font: FontSpec::new("16px Inter"),
                break_mode: BreakMode::Normal,
                extra_width: 0.0,
            },
            InlineFlowItem {
                text: "@staging".to_string(),
                font: FontSpec::new("14px monospace"),
                break_mode: BreakMode::Never,
                extra_width: 8.0, // Chip padding
            },
            InlineFlowItem {
                text: "completed".to_string(),
                font: FontSpec::new("16px Inter"),
                break_mode: BreakMode::Normal,
                extra_width: 0.0,
            },
        ];

        let prepared = prepare_inline_flow(&items, &backend);
        let lines = layout_inline_flow(&prepared, 300.0);
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
                font: FontSpec::new("16px Inter"),
                break_mode: BreakMode::Normal,
                extra_width: 0.0,
            },
            InlineFlowItem {
                text: "@very-long-username".to_string(),
                font: FontSpec::new("16px Inter"),
                break_mode: BreakMode::Never, // Atomic -- must not break
                extra_width: 8.0,
            },
        ];

        let prepared = prepare_inline_flow(&items, &backend);
        let lines = layout_inline_flow(&prepared, 100.0);

        // The atomic item should appear as a single fragment
        for line in &lines {
            for frag in &line.fragments {
                if frag.item_index == 1 {
                    assert!(frag.text.contains("@very-long-username"));
                }
            }
        }
    }
}
