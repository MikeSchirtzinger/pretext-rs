//! Line-breaking engine -- the hot path.
//!
//! This module implements the core line-breaking algorithm. It operates
//! exclusively on the parallel arrays in `PreparedData` -- no strings,
//! no allocations, no measurement calls. This is what makes `layout()`
//! essentially free after `prepare()`.
//!
//! Three walker variants (matching the TypeScript):
//! 1. `walk_lines_simple` -- fast path for normal text (no tabs, soft hyphens, pre-wrap)
//! 2. `walk_lines_full` -- handles all segment kinds with chunk processing
//! 3. `layout_next_line_range` -- streaming single-line cursor API

use crate::types::{
    EngineProfile, LayoutCursor, LayoutLineRange, PreparedData, PreparedLineChunk, SegmentKind,
};

/// Internal line representation during walking.
#[derive(Debug, Clone)]
pub(crate) struct InternalLine {
    pub(crate) start_segment: usize,
    pub(crate) start_grapheme: usize,
    pub(crate) end_segment: usize,
    pub(crate) end_grapheme: usize,
    pub(crate) width: f64,
    pub(crate) ends_with_discretionary_hyphen: bool,
}

/// Pending break state -- tracks the best break opportunity found so far.
#[derive(Debug, Clone, Copy)]
struct PendingBreak {
    segment_index: usize,
    grapheme_index: usize,
    fit_width: f64,
    paint_width: f64,
    ends_with_discretionary_hyphen: bool,
}

/// Borrowed, structurally complete entry from `PreparedData`'s parallel
/// arrays. Returning `None` for drifted state keeps every walker bounded.
#[derive(Clone, Copy)]
struct SegmentView<'a> {
    kind: SegmentKind,
    width: f64,
    line_end_fit_advance: f64,
    line_end_paint_advance: f64,
    breakable_widths: Option<&'a [f64]>,
}

fn segment_at(data: &PreparedData, index: usize) -> Option<SegmentView<'_>> {
    Some(SegmentView {
        kind: *data.kinds.get(index)?,
        width: *data.widths.get(index)?,
        line_end_fit_advance: *data.line_end_fit_advances.get(index)?,
        line_end_paint_advance: *data.line_end_paint_advances.get(index)?,
        breakable_widths: data.breakable_widths.get(index)?.as_deref(),
    })
}

// ---- Simple fast path -------------------------------------------------------

/// Count lines using the simple fast path.
///
/// Only handles `Text`, `Space`, and `ZeroWidthBreak` segments. No chunks,
/// no tabs, no soft hyphens. This is the common case for prose.
#[must_use]
pub(crate) fn count_lines_simple(
    data: &PreparedData,
    max_width: f64,
    profile: &EngineProfile,
) -> usize {
    let mut line_count = 0_usize;
    walk_lines_simple(data, max_width, profile, |_| {
        line_count = line_count.saturating_add(1);
    });
    line_count
}

/// Walk lines using the simple fast path, calling a callback per line.
pub(crate) fn walk_lines_simple<F>(
    data: &PreparedData,
    max_width: f64,
    profile: &EngineProfile,
    mut on_line: F,
) where
    F: FnMut(InternalLine),
{
    let epsilon = profile.line_fit_epsilon;
    let seg_count = data.segment_count();

    if seg_count == 0 {
        on_line(InternalLine {
            start_segment: 0,
            start_grapheme: 0,
            end_segment: 0,
            end_grapheme: 0,
            width: 0.0,
            ends_with_discretionary_hyphen: false,
        });
        return;
    }

    let mut line_start_seg = 0;
    let mut line_start_grapheme = 0;
    let mut line_w = 0.0;
    let mut has_content = false;
    let mut pending: Option<PendingBreak> = None;

    let mut i = 0;
    while i < seg_count {
        let Some(segment) = segment_at(data, i) else {
            break;
        };
        let kind = segment.kind;
        let segment_start_grapheme = if line_start_seg == i {
            line_start_grapheme
        } else {
            0
        };
        let width = remaining_segment_width(segment, segment_start_grapheme);
        let fit_advance = segment.line_end_fit_advance;

        let new_w = line_w + width;

        // Track break opportunities
        if kind.can_break_after() {
            let content_after = has_content || is_visible_line_content(kind);
            let fit_width = line_w + fit_advance;
            if content_after && fit_width <= max_width + epsilon {
                pending = Some(PendingBreak {
                    segment_index: i + 1,
                    grapheme_index: 0,
                    fit_width,
                    paint_width: line_w + segment.line_end_paint_advance,
                    ends_with_discretionary_hyphen: kind == SegmentKind::SoftHyphen,
                });
            }
            line_w = new_w;
            has_content = content_after;
            i += 1;
            continue;
        }

        // Text segment -- check overflow
        if new_w > max_width + epsilon {
            if let Some(pb) = pending.take()
                && pb.fit_width <= max_width + epsilon
            {
                // Break at pending break point
                on_line(InternalLine {
                    start_segment: line_start_seg,
                    start_grapheme: line_start_grapheme,
                    end_segment: pb.segment_index,
                    end_grapheme: pb.grapheme_index,
                    width: pb.paint_width,
                    ends_with_discretionary_hyphen: pb.ends_with_discretionary_hyphen,
                });
                line_start_seg = pb.segment_index;
                line_start_grapheme = pb.grapheme_index;
                // Skip leading spaces on new line
                while line_start_seg < seg_count
                    && segment_at(data, line_start_seg)
                        .is_some_and(|candidate| candidate.kind == SegmentKind::Space)
                {
                    line_start_seg += 1;
                }
                (line_w, has_content, pending) =
                    replay_line_state(data, line_start_seg, i, max_width, epsilon);
                continue; // Re-evaluate current segment
            }

            let glued_to_previous = i > line_start_seg
                && segment_at(data, i.saturating_sub(1))
                    .is_some_and(|previous| previous.kind == SegmentKind::Glue);

            // Try grapheme breaking within a segment unless doing so would
            // split a non-breaking glue connection.
            if !glued_to_previous
                && let Some(bw) = segment.breakable_widths
                && let Some((break_at, break_width)) = find_grapheme_break(
                    bw,
                    segment_start_grapheme,
                    line_w,
                    max_width,
                    epsilon,
                    !has_content,
                )
            {
                on_line(InternalLine {
                    start_segment: line_start_seg,
                    start_grapheme: line_start_grapheme,
                    end_segment: i,
                    end_grapheme: break_at,
                    width: line_w + break_width,
                    ends_with_discretionary_hyphen: false,
                });
                line_start_seg = i;
                line_start_grapheme = break_at;
                line_w = 0.0;
                has_content = false;
                pending = None;
                continue;
            }

            // Force break before this segment if we have content
            if !glued_to_previous && has_content && line_start_seg < i {
                let paint_width =
                    compute_paint_width_range(data, line_start_seg, line_start_grapheme, i, 0);
                on_line(InternalLine {
                    start_segment: line_start_seg,
                    start_grapheme: line_start_grapheme,
                    end_segment: i,
                    end_grapheme: 0,
                    width: paint_width,
                    ends_with_discretionary_hyphen: false,
                });
                line_start_seg = i;
                line_start_grapheme = 0;
                line_w = 0.0;
                has_content = false;
                pending = None;
                continue;
            } else {
                // Accept overflow on a single segment
                line_w = new_w;
                has_content = true;
            }
        } else {
            line_w = new_w;
            has_content = true;
        }

        i += 1;
    }

    // Emit final line
    let final_width =
        compute_paint_width_range(data, line_start_seg, line_start_grapheme, seg_count, 0);
    on_line(InternalLine {
        start_segment: line_start_seg,
        start_grapheme: line_start_grapheme,
        end_segment: seg_count,
        end_grapheme: 0,
        width: final_width,
        ends_with_discretionary_hyphen: false,
    });
}

// ---- Full path (with chunks, tabs, soft hyphens) ----------------------------

/// Count lines using the full walker with chunk support.
#[must_use]
pub(crate) fn count_lines_full(
    data: &PreparedData,
    max_width: f64,
    profile: &EngineProfile,
) -> usize {
    let mut line_count: usize = 0;
    walk_lines_full(data, max_width, profile, |_| {
        line_count = line_count.saturating_add(1);
    });
    line_count
}

/// Walk lines with full chunk/tab/soft-hyphen support.
pub(crate) fn walk_lines_full<F>(
    data: &PreparedData,
    max_width: f64,
    profile: &EngineProfile,
    mut on_line: F,
) where
    F: FnMut(InternalLine),
{
    let seg_count = data.segment_count();

    if seg_count == 0 {
        on_line(InternalLine {
            start_segment: 0,
            start_grapheme: 0,
            end_segment: 0,
            end_grapheme: 0,
            width: 0.0,
            ends_with_discretionary_hyphen: false,
        });
        return;
    }

    if data.chunks.is_empty() {
        let implicit_chunk = PreparedLineChunk {
            start_segment_index: 0,
            end_segment_index: seg_count,
            consumed_end_segment_index: seg_count,
        };
        walk_chunk(data, &implicit_chunk, max_width, profile, &mut on_line);
        return;
    }

    for chunk in &data.chunks {
        walk_chunk(data, chunk, max_width, profile, &mut on_line);
    }
}

/// Walk a single chunk (between hard breaks).
#[allow(clippy::too_many_lines)]
fn walk_chunk<F>(
    data: &PreparedData,
    chunk: &PreparedLineChunk,
    max_width: f64,
    profile: &EngineProfile,
    on_line: &mut F,
) where
    F: FnMut(InternalLine),
{
    let epsilon = profile.line_fit_epsilon;
    let seg_count = data.segment_count();
    let chunk_start = chunk.start_segment_index.min(seg_count);
    let chunk_end = chunk.end_segment_index.clamp(chunk_start, seg_count);
    let consumed_end = chunk.consumed_end_segment_index.clamp(chunk_end, seg_count);

    if chunk_start >= chunk_end {
        // Empty chunk (consecutive hard breaks) -- emit empty line
        on_line(InternalLine {
            start_segment: chunk_start,
            start_grapheme: 0,
            end_segment: consumed_end,
            end_grapheme: 0,
            width: 0.0,
            ends_with_discretionary_hyphen: false,
        });
        return;
    }

    let mut line_start_seg = chunk_start;
    let mut line_start_grapheme = 0;
    let mut line_w = 0.0;
    let mut has_content = false;
    let mut pending: Option<PendingBreak> = None;

    let mut i = chunk_start;
    while i < chunk_end {
        let Some(segment) = segment_at(data, i) else {
            break;
        };
        let kind = segment.kind;
        let segment_start_grapheme = if line_start_seg == i {
            line_start_grapheme
        } else {
            0
        };
        let width = remaining_segment_width(segment, segment_start_grapheme);

        // Tab width is computed dynamically based on current line position
        let effective_width = if kind == SegmentKind::Tab {
            get_tab_advance(line_w, data.tab_stop_advance)
        } else {
            width
        };

        let new_w = line_w + effective_width;

        match kind {
            SegmentKind::SoftHyphen => {
                // Invisible unless chosen as break point
                // Set pending break with discretionary hyphen width
                let fit_w = line_w + data.discretionary_hyphen_width;
                if has_content && fit_w <= max_width + epsilon {
                    pending = Some(PendingBreak {
                        segment_index: i + 1,
                        grapheme_index: 0,
                        fit_width: fit_w,
                        paint_width: fit_w, // Hyphen is visible at break
                        ends_with_discretionary_hyphen: true,
                    });
                }
                i += 1;
            }
            SegmentKind::HardBreak => {
                // Should not appear within a chunk (handled by chunk boundaries)
                i += 1;
            }
            SegmentKind::Space
            | SegmentKind::PreservedSpace
            | SegmentKind::Tab
            | SegmentKind::ZeroWidthBreak => {
                // Breakable point
                let fit_advance = segment.line_end_fit_advance;
                let paint_advance = segment.line_end_paint_advance;
                let content_after = has_content || is_visible_line_content(kind);
                let fit_width = line_w + fit_advance;
                if content_after && fit_width <= max_width + epsilon {
                    pending = Some(PendingBreak {
                        segment_index: i + 1,
                        grapheme_index: 0,
                        fit_width,
                        paint_width: line_w + paint_advance,
                        ends_with_discretionary_hyphen: false,
                    });
                }
                line_w = new_w;
                has_content = content_after;
                i += 1;
            }
            SegmentKind::Glue => {
                // Non-breaking -- just accumulate width
                line_w = new_w;
                has_content = true;
                i += 1;
            }
            SegmentKind::Text => {
                // Check overflow
                if new_w > max_width + epsilon {
                    // Try pending break
                    if has_content
                        && let Some(pb) = pending.take()
                        && pb.fit_width <= max_width + epsilon
                    {
                        on_line(InternalLine {
                            start_segment: line_start_seg,
                            start_grapheme: line_start_grapheme,
                            end_segment: pb.segment_index,
                            end_grapheme: pb.grapheme_index,
                            width: pb.paint_width,
                            ends_with_discretionary_hyphen: pb.ends_with_discretionary_hyphen,
                        });
                        line_start_seg = skip_leading_spaces(data, pb.segment_index, chunk_end);
                        line_start_grapheme = 0;
                        (line_w, has_content, pending) =
                            replay_line_state(data, line_start_seg, i, max_width, epsilon);
                        continue; // Re-evaluate current segment
                    }

                    let glued_to_previous = i > line_start_seg
                        && segment_at(data, i.saturating_sub(1))
                            .is_some_and(|previous| previous.kind == SegmentKind::Glue);

                    // Try grapheme breaking unless this text is attached to
                    // the preceding run by non-breaking glue.
                    if !glued_to_previous
                        && let Some(bw) = segment.breakable_widths
                        && let Some((break_at, break_width)) = find_grapheme_break(
                            bw,
                            segment_start_grapheme,
                            line_w,
                            max_width,
                            epsilon,
                            !has_content,
                        )
                    {
                        on_line(InternalLine {
                            start_segment: line_start_seg,
                            start_grapheme: line_start_grapheme,
                            end_segment: i,
                            end_grapheme: break_at,
                            width: line_w + break_width,
                            ends_with_discretionary_hyphen: false,
                        });
                        line_start_seg = i;
                        line_start_grapheme = break_at;
                        line_w = 0.0;
                        has_content = false;
                        pending = None;
                        continue;
                    }

                    // Force break before this segment
                    if !glued_to_previous && has_content && line_start_seg < i {
                        let paint_width = compute_paint_width_range(
                            data,
                            line_start_seg,
                            line_start_grapheme,
                            i,
                            0,
                        );
                        on_line(InternalLine {
                            start_segment: line_start_seg,
                            start_grapheme: line_start_grapheme,
                            end_segment: i,
                            end_grapheme: 0,
                            width: paint_width,
                            ends_with_discretionary_hyphen: false,
                        });
                        line_start_seg = i;
                        line_start_grapheme = 0;
                        line_w = 0.0;
                        has_content = false;
                        pending = None;
                        continue;
                    } else {
                        // Accept overflow
                        line_w = new_w;
                    }
                } else {
                    line_w = new_w;
                }
                has_content = true;
                i += 1;
            }
        }
    }

    // Emit final line of chunk
    let end = consumed_end;
    let terminal_soft_hyphen = chunk_end
        .checked_sub(1)
        .and_then(|index| segment_at(data, index))
        .is_some_and(|segment| segment.kind == SegmentKind::SoftHyphen);
    let final_width = pending
        .as_ref()
        .filter(|pb| pb.segment_index == chunk_end && !terminal_soft_hyphen)
        .map_or_else(
            || compute_paint_width_range(data, line_start_seg, line_start_grapheme, chunk_end, 0),
            |pb| pb.paint_width,
        );

    on_line(InternalLine {
        start_segment: line_start_seg,
        start_grapheme: line_start_grapheme,
        end_segment: end,
        end_grapheme: 0,
        width: final_width,
        ends_with_discretionary_hyphen: false,
    });
}

// ---- Streaming cursor API ---------------------------------------------------

/// Layout a single line starting from a cursor position.
///
/// Returns the line range and the cursor for the next line, or `None`
/// if there are no more lines. This is the streaming API -- call
/// repeatedly to iterate lines one at a time.
///
/// Supports variable `max_width` per line (for text flowing around images).
#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn layout_next_line_range(
    data: &PreparedData,
    start: LayoutCursor,
    max_width: f64,
    profile: &EngineProfile,
) -> Option<(LayoutLineRange, LayoutCursor)> {
    let seg_count = data.segment_count();

    // Terminal cursor — no more lines to produce
    if start.segment_index >= seg_count {
        // Special case: empty text, first call only
        if seg_count == 0 && start.segment_index == 0 && start.grapheme_index == 0 {
            return Some((
                LayoutLineRange {
                    width: 0.0,
                    start,
                    end: LayoutCursor::new(0, 0),
                    ends_with_discretionary_hyphen: false,
                },
                LayoutCursor::new(1, 0),
            ));
        }
        // A trailing hard break creates one final empty line. Use the same
        // one-past-terminal sentinel as empty text so it is emitted exactly
        // once while callers still receive a strictly advancing cursor.
        if start.segment_index == seg_count
            && start.grapheme_index == 0
            && seg_count
                .checked_sub(1)
                .and_then(|index| segment_at(data, index))
                .is_some_and(|segment| segment.kind == SegmentKind::HardBreak)
        {
            let next = LayoutCursor::new(seg_count.saturating_add(1), 0);
            return Some((
                LayoutLineRange {
                    width: 0.0,
                    start,
                    end: start,
                    ends_with_discretionary_hyphen: false,
                },
                next,
            ));
        }
        return None;
    }

    let start = normalize_cursor(data, start);

    let epsilon = profile.line_fit_epsilon;
    let mut line_w = 0.0;
    let mut has_content = false;
    let mut pending: Option<PendingBreak> = None;

    let mut i = start.segment_index;
    while i < seg_count {
        let Some(segment) = segment_at(data, i) else {
            break;
        };
        let kind = segment.kind;
        let segment_start_grapheme = if i == start.segment_index {
            start.grapheme_index
        } else {
            0
        };

        // Check for hard break
        if kind == SegmentKind::HardBreak {
            let line = LayoutLineRange {
                width: compute_paint_width_from(data, &start, i),
                start,
                end: LayoutCursor::new(i + 1, 0),
                ends_with_discretionary_hyphen: false,
            };
            return Some((line, LayoutCursor::new(i + 1, 0)));
        }

        if kind == SegmentKind::Glue {
            line_w += remaining_segment_width(segment, segment_start_grapheme);
            has_content = true;
            i += 1;
            continue;
        }

        let width = if kind == SegmentKind::Tab {
            get_tab_advance(line_w, data.tab_stop_advance)
        } else {
            remaining_segment_width(segment, segment_start_grapheme)
        };

        let new_w = line_w + width;

        if kind.can_break_after() {
            let content_after = has_content || is_visible_line_content(kind);
            let fit_width = line_w + segment.line_end_fit_advance;
            if content_after && fit_width <= max_width + epsilon {
                pending = Some(PendingBreak {
                    segment_index: i + 1,
                    grapheme_index: 0,
                    fit_width,
                    paint_width: line_w + segment.line_end_paint_advance,
                    ends_with_discretionary_hyphen: kind == SegmentKind::SoftHyphen,
                });
            }
            line_w = new_w;
            has_content = content_after;
            i += 1;
            continue;
        }

        // Text -- check overflow
        if new_w > max_width + epsilon {
            // Try pending break
            if has_content
                && let Some(pb) = pending.take()
                && pb.fit_width <= max_width + epsilon
            {
                let line = LayoutLineRange {
                    width: pb.paint_width,
                    start,
                    end: LayoutCursor::new(pb.segment_index, 0),
                    ends_with_discretionary_hyphen: pb.ends_with_discretionary_hyphen,
                };
                let next_start_seg = skip_leading_spaces(data, pb.segment_index, seg_count);
                return Some((line, LayoutCursor::new(next_start_seg, 0)));
            }

            let glued_to_previous = i > start.segment_index
                && segment_at(data, i.saturating_sub(1))
                    .is_some_and(|previous| previous.kind == SegmentKind::Glue);

            // Try grapheme break unless this text is attached to the
            // preceding run by non-breaking glue.
            if !glued_to_previous
                && let Some(bw) = segment.breakable_widths
                && let Some((break_at, break_width)) = find_grapheme_break(
                    bw,
                    segment_start_grapheme,
                    line_w,
                    max_width,
                    epsilon,
                    !has_content,
                )
            {
                let line = LayoutLineRange {
                    width: line_w + break_width,
                    start,
                    end: LayoutCursor::new(i, break_at),
                    ends_with_discretionary_hyphen: false,
                };
                return Some((line, LayoutCursor::new(i, break_at)));
            }

            // No part of this segment fits after existing content. Leave it
            // untouched for the next call so that a breakable segment can be
            // split repeatedly from a fresh line.
            if !glued_to_previous && has_content && start.segment_index < i {
                let end = LayoutCursor::new(i, 0);
                let line = LayoutLineRange {
                    width: compute_paint_width_from(data, &start, i),
                    start,
                    end,
                    ends_with_discretionary_hyphen: false,
                };
                return Some((line, end));
            }
        }

        line_w = new_w;
        has_content = true;
        i += 1;
    }

    // Final line
    Some((
        LayoutLineRange {
            width: compute_paint_width_from(data, &start, seg_count),
            start,
            end: LayoutCursor::new(seg_count, 0),
            ends_with_discretionary_hyphen: false,
        },
        LayoutCursor::new(seg_count, 0),
    ))
}

// ---- Helpers ----------------------------------------------------------------

/// Compute tab advance to the next tab stop.
#[inline]
fn get_tab_advance(current_x: f64, tab_stop_advance: f64) -> f64 {
    if !tab_stop_advance.is_finite() || tab_stop_advance <= 0.0 || !current_x.is_finite() {
        return 0.0;
    }
    tab_stop_advance - (current_x % tab_stop_advance)
}

/// Clamp a crate-internal cursor to the prepared representation.
fn normalize_cursor(data: &PreparedData, cursor: LayoutCursor) -> LayoutCursor {
    let seg_count = data.segment_count();
    if cursor.segment_index >= seg_count {
        return LayoutCursor::new(seg_count, 0);
    }

    let grapheme_index = segment_at(data, cursor.segment_index)
        .and_then(|segment| segment.breakable_widths)
        .map_or(0, |widths| cursor.grapheme_index.min(widths.len()));
    LayoutCursor::new(cursor.segment_index, grapheme_index)
}

/// Sum a clamped slice without relying on a panicking range operation.
fn sum_slice(values: &[f64], from: usize, to: usize) -> f64 {
    let start = from.min(values.len());
    let end = to.clamp(start, values.len());
    values
        .get(start..end)
        .map_or(0.0, |slice| slice.iter().sum())
}

/// Preserve whole-segment shaping and kerning until a cursor actually enters
/// the segment. Per-grapheme measurements are only an overflow-wrap fallback;
/// their sum is not a substitute for the backend's shaped segment width.
fn remaining_segment_width(segment: SegmentView<'_>, start_grapheme: usize) -> f64 {
    if start_grapheme == 0 {
        segment.width
    } else {
        segment.breakable_widths.map_or(segment.width, |widths| {
            sum_slice(widths, start_grapheme, widths.len())
        })
    }
}

/// Skip leading space segments after a break.
fn skip_leading_spaces(data: &PreparedData, from: usize, limit: usize) -> usize {
    let end = limit.min(data.segment_count());
    let mut i = from.min(end);
    while i < end && segment_at(data, i).is_some_and(|segment| segment.kind == SegmentKind::Space) {
        i += 1;
    }
    i
}

/// Reconstruct the active line after selecting an earlier break opportunity.
///
/// Width replay must preserve dynamic tab stops, while content replay must not
/// mistake structural controls such as ZWSP or U+00AD for paintable content.
fn replay_line_state(
    data: &PreparedData,
    from: usize,
    to: usize,
    max_width: f64,
    epsilon: f64,
) -> (f64, bool, Option<PendingBreak>) {
    let end = to.min(data.segment_count());
    let mut width = 0.0;
    let mut has_content = false;
    let mut pending = None;
    for index in from.min(end)..end {
        let Some(segment) = segment_at(data, index) else {
            continue;
        };
        let advance = if segment.kind == SegmentKind::Tab {
            get_tab_advance(width, data.tab_stop_advance)
        } else {
            segment.width
        };
        let content_after = has_content || is_visible_line_content(segment.kind);
        if segment.kind.can_break_after() {
            let fit_width = width + segment.line_end_fit_advance;
            if content_after && fit_width <= max_width + epsilon {
                pending = Some(PendingBreak {
                    segment_index: index + 1,
                    grapheme_index: 0,
                    fit_width,
                    paint_width: width + segment.line_end_paint_advance,
                    ends_with_discretionary_hyphen: segment.kind == SegmentKind::SoftHyphen,
                });
            }
        }
        width += advance;
        has_content = content_after;
    }
    (width, has_content, pending)
}

/// Structural controls may alter where a line can break but do not make an
/// otherwise empty line eligible for emission.
const fn is_visible_line_content(kind: SegmentKind) -> bool {
    !matches!(
        kind,
        SegmentKind::ZeroWidthBreak | SegmentKind::SoftHyphen | SegmentKind::HardBreak
    )
}

/// Compute paint width between two cursor positions.
///
/// Segment indices use cursor semantics: a zero grapheme index is the
/// boundary before that segment, while a non-zero grapheme index identifies a
/// boundary within the segment. This lets repeated mid-word lines retain both
/// their partial starting offset and their partial ending offset.
fn compute_paint_width_range(
    data: &PreparedData,
    start_segment: usize,
    start_grapheme: usize,
    end_segment: usize,
    end_grapheme: usize,
) -> f64 {
    let seg_count = data.segment_count();
    let start_segment = start_segment.min(seg_count);
    let end_segment = end_segment.min(seg_count);

    if start_segment > end_segment
        || (start_segment == end_segment && end_grapheme <= start_grapheme)
    {
        return 0.0;
    }

    if start_segment == end_segment {
        return segment_at(data, start_segment)
            .and_then(|segment| segment.breakable_widths)
            .map_or(0.0, |widths| {
                sum_slice(widths, start_grapheme, end_grapheme)
            });
    }

    let mut width = 0.0;
    let mut paint_width = 0.0;
    for index in start_segment..end_segment {
        let Some(segment) = segment_at(data, index) else {
            continue;
        };
        let advance = if index == start_segment && start_grapheme > 0 {
            segment.breakable_widths.map_or(0.0, |widths| {
                sum_slice(widths, start_grapheme, widths.len())
            })
        } else if segment.kind == SegmentKind::Tab {
            get_tab_advance(width, data.tab_stop_advance)
        } else {
            segment.width
        };
        let line_end_advance = if index == start_segment && start_grapheme > 0 {
            advance
        } else if segment.kind == SegmentKind::SoftHyphen {
            // U+00AD contributes paint width only when a PendingBreak carrying
            // explicit discretionary-hyphen state is actually selected.
            0.0
        } else {
            segment.line_end_paint_advance
        };
        paint_width = width + line_end_advance;
        width += advance;
    }

    if end_grapheme > 0 {
        if let Some(widths) =
            segment_at(data, end_segment).and_then(|segment| segment.breakable_widths)
        {
            width += sum_slice(widths, 0, end_grapheme);
        }
        return width;
    }
    paint_width
}

/// Compute paint width accounting for cursor start position.
fn compute_paint_width_from(data: &PreparedData, start: &LayoutCursor, to: usize) -> f64 {
    let cursor = normalize_cursor(data, *start);
    compute_paint_width_range(data, cursor.segment_index, cursor.grapheme_index, to, 0)
}

/// Find the grapheme index at which to break a segment for overflow-wrap.
///
/// Returns `Some(index)` if a valid mid-segment break point was found,
/// `None` if the entire segment fits or has only one grapheme.
#[inline]
fn find_grapheme_break(
    grapheme_widths: &[f64],
    start_grapheme: usize,
    line_w: f64,
    max_width: f64,
    epsilon: f64,
    allow_first_overflow: bool,
) -> Option<(usize, f64)> {
    let start = start_grapheme.min(grapheme_widths.len());
    let mut accum = line_w;
    let mut break_at = start;
    for (gi, &gw) in grapheme_widths.iter().enumerate().skip(start) {
        if accum + gw > max_width + epsilon {
            if break_at > start {
                break;
            }
            if !allow_first_overflow {
                return None;
            }
        }
        accum += gw;
        break_at = gi + 1;
        if accum > max_width + epsilon {
            break;
        }
    }
    if break_at > start && break_at < grapheme_widths.len() {
        Some((break_at, sum_slice(grapheme_widths, start, break_at)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{FontSpec, fixed::FixedWidthBackend};
    use crate::types::PreparedLineChunk;
    use crate::{
        Result, layout, layout_next_line, layout_with_lines, prepare, prepare_with_segments,
        walk_line_ranges,
    };

    #[track_caller]
    fn valid<T>(result: Result<T>) -> T {
        result.expect("test input is valid")
    }

    fn make_simple_data(widths: &[f64], kinds: &[SegmentKind]) -> PreparedData {
        let n = widths.len();
        let mut fit_advances = Vec::with_capacity(n);
        let mut paint_advances = Vec::with_capacity(n);
        let mut breakable = Vec::with_capacity(n);

        for (i, &w) in widths.iter().enumerate() {
            if kinds[i].hangs_at_line_end() {
                fit_advances.push(0.0);
                paint_advances.push(0.0);
            } else {
                fit_advances.push(w);
                paint_advances.push(w);
            }
            breakable.push(None);
        }

        PreparedData {
            widths: widths.to_vec(),
            line_end_fit_advances: fit_advances,
            line_end_paint_advances: paint_advances,
            kinds: kinds.to_vec(),
            breakable_widths: breakable,
            chunks: vec![PreparedLineChunk {
                start_segment_index: 0,
                end_segment_index: n,
                consumed_end_segment_index: n,
            }],
            tab_stop_advance: 48.0,
            discretionary_hyphen_width: 5.0,
            simple_fast_path: true,
            profile: EngineProfile::native(),
        }
    }

    fn make_breakable_word_data(grapheme_count: usize, grapheme_width: f64) -> PreparedData {
        let total_width = (0..grapheme_count).fold(0.0, |width, _| width + grapheme_width);
        let mut data = make_simple_data(&[total_width], &[SegmentKind::Text]);
        data.breakable_widths = vec![Some(vec![grapheme_width; grapheme_count])];
        data
    }

    fn line_signature(line: &InternalLine) -> (usize, usize, usize, usize, f64) {
        (
            line.start_segment,
            line.start_grapheme,
            line.end_segment,
            line.end_grapheme,
            line.width,
        )
    }

    fn range_signature(line: &LayoutLineRange) -> (usize, usize, usize, usize, f64) {
        (
            line.start.segment_index,
            line.start.grapheme_index,
            line.end.segment_index,
            line.end.grapheme_index,
            line.width,
        )
    }

    fn line_state_signature(line: &InternalLine) -> (usize, usize, usize, usize, f64, bool) {
        (
            line.start_segment,
            line.start_grapheme,
            line.end_segment,
            line.end_grapheme,
            line.width,
            line.ends_with_discretionary_hyphen,
        )
    }

    fn range_state_signature(line: &LayoutLineRange) -> (usize, usize, usize, usize, f64, bool) {
        (
            line.start.segment_index,
            line.start.grapheme_index,
            line.end.segment_index,
            line.end.grapheme_index,
            line.width,
            line.ends_with_discretionary_hyphen,
        )
    }

    fn make_control_data(kinds: &[SegmentKind]) -> PreparedData {
        let widths: Vec<_> = kinds
            .iter()
            .map(|kind| match kind {
                SegmentKind::Text | SegmentKind::Glue => 3.0,
                SegmentKind::Space | SegmentKind::PreservedSpace => 1.0,
                SegmentKind::ZeroWidthBreak
                | SegmentKind::SoftHyphen
                | SegmentKind::HardBreak
                | SegmentKind::Tab => 0.0,
            })
            .collect();
        let mut data = make_simple_data(&widths, kinds);
        data.discretionary_hyphen_width = 2.0;
        data.tab_stop_advance = 4.0;
        for (index, kind) in kinds.iter().enumerate() {
            if *kind == SegmentKind::SoftHyphen {
                if let Some(advance) = data.line_end_fit_advances.get_mut(index) {
                    *advance = data.discretionary_hyphen_width;
                }
                if let Some(advance) = data.line_end_paint_advances.get_mut(index) {
                    *advance = data.discretionary_hyphen_width;
                }
            }
        }
        data.simple_fast_path = false;
        data
    }

    fn collect_streaming_lines(data: &PreparedData, max_width: f64) -> Vec<LayoutLineRange> {
        let mut cursor = LayoutCursor::default();
        let mut lines = Vec::new();
        while let Some((line, next)) =
            layout_next_line_range(data, cursor, max_width, &data.profile)
        {
            assert_ne!(next, cursor, "streaming cursor must always make progress");
            lines.push(line);
            cursor = next;
        }
        lines
    }

    #[test]
    fn test_single_word_no_wrap() {
        let data = make_simple_data(&[50.0], &[SegmentKind::Text]);
        let profile = EngineProfile::native();
        assert_eq!(count_lines_simple(&data, 100.0, &profile), 1);
    }

    #[test]
    fn test_two_words_wrap() {
        let data = make_simple_data(
            &[50.0, 5.0, 50.0],
            &[SegmentKind::Text, SegmentKind::Space, SegmentKind::Text],
        );
        let profile = EngineProfile::native();
        assert_eq!(count_lines_simple(&data, 80.0, &profile), 2);
    }

    #[test]
    fn test_three_words_two_wraps() {
        let data = make_simple_data(
            &[40.0, 5.0, 40.0, 5.0, 40.0],
            &[
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
            ],
        );
        let profile = EngineProfile::native();
        assert_eq!(count_lines_simple(&data, 60.0, &profile), 3);
    }

    #[test]
    fn test_empty_text() {
        let data = make_simple_data(&[], &[]);
        let profile = EngineProfile::native();
        assert_eq!(count_lines_simple(&data, 100.0, &profile), 1);
    }

    #[test]
    fn test_exact_fit() {
        let data = make_simple_data(
            &[50.0, 5.0, 45.0],
            &[SegmentKind::Text, SegmentKind::Space, SegmentKind::Text],
        );
        let profile = EngineProfile::native();
        assert_eq!(count_lines_simple(&data, 100.0, &profile), 1);
    }

    #[test]
    fn test_walk_lines_produces_correct_ranges() {
        let data = make_simple_data(
            &[50.0, 5.0, 50.0],
            &[SegmentKind::Text, SegmentKind::Space, SegmentKind::Text],
        );
        let profile = EngineProfile::native();

        let mut lines = Vec::new();
        walk_lines_simple(&data, 80.0, &profile, |line| lines.push(line));

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].start_segment, 0);
        assert_eq!(lines[1].start_segment, 2);
    }

    #[test]
    fn simple_walker_repeatedly_splits_one_long_segment() {
        let data = make_breakable_word_data(20, 6.0);
        let mut lines = Vec::new();

        walk_lines_simple(&data, 18.0, &data.profile, |line| lines.push(line));

        let signatures: Vec<_> = lines.iter().map(line_signature).collect();
        assert_eq!(count_lines_simple(&data, 18.0, &data.profile), 7);
        assert_eq!(
            signatures,
            vec![
                (0, 0, 0, 3, 18.0),
                (0, 3, 0, 6, 18.0),
                (0, 6, 0, 9, 18.0),
                (0, 9, 0, 12, 18.0),
                (0, 12, 0, 15, 18.0),
                (0, 15, 0, 18, 18.0),
                (0, 18, 1, 0, 12.0),
            ]
        );
    }

    #[test]
    fn full_walker_repeatedly_splits_one_long_segment() {
        let mut data = make_breakable_word_data(20, 6.0);
        data.simple_fast_path = false;
        let mut lines = Vec::new();

        walk_lines_full(&data, 18.0, &data.profile, |line| lines.push(line));

        let signatures: Vec<_> = lines.iter().map(line_signature).collect();
        assert_eq!(count_lines_full(&data, 18.0, &data.profile), 7);
        assert_eq!(
            signatures,
            vec![
                (0, 0, 0, 3, 18.0),
                (0, 3, 0, 6, 18.0),
                (0, 6, 0, 9, 18.0),
                (0, 9, 0, 12, 18.0),
                (0, 12, 0, 15, 18.0),
                (0, 15, 0, 18, 18.0),
                (0, 18, 1, 0, 12.0),
            ]
        );
    }

    #[test]
    fn whole_segment_measurement_preserves_shaping_until_midword_break() {
        let mut data = make_simple_data(&[10.0], &[SegmentKind::Text]);
        // Isolated grapheme measurements deliberately exceed the shaped
        // whole-segment width, as with kerning pairs such as "AV".
        data.breakable_widths = vec![Some(vec![6.0, 6.0])];

        let mut simple = Vec::new();
        walk_lines_simple(&data, 11.0, &data.profile, |line| simple.push(line));
        let streamed = collect_streaming_lines(&data, 11.0);

        assert_eq!(
            simple.iter().map(line_signature).collect::<Vec<_>>(),
            vec![(0, 0, 1, 0, 10.0)]
        );
        assert_eq!(
            simple.iter().map(line_signature).collect::<Vec<_>>(),
            streamed.iter().map(range_signature).collect::<Vec<_>>()
        );

        data.simple_fast_path = false;
        let mut full = Vec::new();
        walk_lines_full(&data, 11.0, &data.profile, |line| full.push(line));
        assert_eq!(
            full.iter().map(line_signature).collect::<Vec<_>>(),
            vec![(0, 0, 1, 0, 10.0)]
        );
    }

    #[test]
    fn glue_never_becomes_a_break_boundary() {
        let mut data = make_simple_data(
            &[6.0, 6.0, 6.0],
            &[SegmentKind::Text, SegmentKind::Glue, SegmentKind::Text],
        );
        data.breakable_widths = vec![None, None, Some(vec![6.0])];
        data.simple_fast_path = false;

        let mut walked = Vec::new();
        walk_lines_full(&data, 12.0, &data.profile, |line| walked.push(line));
        let streamed = collect_streaming_lines(&data, 12.0);

        assert_eq!(
            walked.iter().map(line_signature).collect::<Vec<_>>(),
            vec![(0, 0, 3, 0, 18.0)]
        );
        assert_eq!(
            walked.iter().map(line_signature).collect::<Vec<_>>(),
            streamed.iter().map(range_signature).collect::<Vec<_>>()
        );
    }

    #[test]
    fn leading_zero_width_break_never_emits_an_empty_line() {
        let mut data = make_simple_data(
            &[0.0, 60.0],
            &[SegmentKind::ZeroWidthBreak, SegmentKind::Text],
        );
        data.breakable_widths = vec![None, Some(vec![6.0; 10])];

        let mut simple = Vec::new();
        walk_lines_simple(&data, 18.0, &data.profile, |line| simple.push(line));
        let streamed = collect_streaming_lines(&data, 18.0);

        assert_eq!(simple.len(), 4);
        assert_eq!(simple[0].width, 18.0);
        assert_eq!(simple[0].start_segment, 0);
        assert_eq!(simple[0].end_segment, 1);
        assert_eq!(simple[0].end_grapheme, 3);
        assert_eq!(
            simple.iter().map(line_signature).collect::<Vec<_>>(),
            streamed.iter().map(range_signature).collect::<Vec<_>>()
        );

        data.simple_fast_path = false;
        let mut full = Vec::new();
        walk_lines_full(&data, 18.0, &data.profile, |line| full.push(line));
        assert_eq!(
            simple.iter().map(line_signature).collect::<Vec<_>>(),
            full.iter().map(line_signature).collect::<Vec<_>>()
        );
    }

    #[test]
    fn non_fitting_soft_hyphen_preserves_the_earlier_fitting_break() {
        for final_is_breakable in [false, true] {
            let mut data = make_control_data(&[
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::SoftHyphen,
                SegmentKind::Text,
            ]);
            if final_is_breakable {
                data.breakable_widths[3] = Some(vec![3.0]);
            }
            let mut walked = Vec::new();
            walk_lines_full(&data, 5.0, &data.profile, |line| walked.push(line));
            let streamed = collect_streaming_lines(&data, 5.0);

            assert_eq!(walked.len(), 2);
            assert_eq!(walked[0].end_segment, 2);
            assert_eq!(walked[1].start_segment, 2);
            assert_eq!(walked[1].width, 3.0);
            assert!(
                walked
                    .iter()
                    .all(|line| !line.ends_with_discretionary_hyphen)
            );
            assert_eq!(
                walked.iter().map(line_state_signature).collect::<Vec<_>>(),
                streamed
                    .iter()
                    .map(range_state_signature)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn production_preparation_preserves_control_run_parity() {
        let backend =
            valid(valid(FixedWidthBackend::new().with_char_width(0.3)).with_cjk_width(0.3));
        let font = valid(FontSpec::new("10px monospace"));

        for text in ["\u{200B}abcdefghij", "a \u{00AD}a", "a \u{00AD}\u{4E2D}"] {
            let prepared = valid(prepare(
                text,
                &font,
                &backend,
                crate::PrepareOptions::default(),
            ));
            let rich = valid(prepare_with_segments(
                text,
                &font,
                &backend,
                crate::PrepareOptions::default(),
            ));
            let rich_lines = valid(layout_with_lines(&rich, 5.0));
            let summary = valid(layout(&prepared, 5.0, 10.0));

            let mut walked = Vec::new();
            valid(walk_line_ranges(
                &prepared,
                5.0,
                &prepared.data.profile,
                |line| walked.push(line),
            ));

            let mut streamed = Vec::new();
            let mut cursor = LayoutCursor::default();
            while let Some((line, next)) = valid(layout_next_line(&prepared, cursor, 5.0)) {
                assert_ne!(next, cursor, "streaming cursor must advance for {text:?}");
                streamed.push(line);
                cursor = next;
            }

            assert_eq!(summary.line_count, rich_lines.len(), "count/rich: {text:?}");
            assert_eq!(walked.len(), rich_lines.len(), "walk/rich: {text:?}");
            assert_eq!(streamed.len(), rich_lines.len(), "stream/rich: {text:?}");
            assert!(
                rich_lines.iter().all(|line| line.width > 0.0),
                "structural controls emitted an empty line for {text:?}"
            );
            assert!(
                rich_lines
                    .iter()
                    .all(|line| !line.ends_with_discretionary_hyphen),
                "a non-fitting soft hyphen replaced an earlier fitting break for {text:?}"
            );
            assert_eq!(
                walked.iter().map(range_state_signature).collect::<Vec<_>>(),
                streamed
                    .iter()
                    .map(range_state_signature)
                    .collect::<Vec<_>>(),
                "walk/stream state mismatch for {text:?}"
            );
        }
    }

    #[test]
    fn selected_soft_hyphen_state_has_full_and_streaming_parity() {
        let data = make_control_data(&[
            SegmentKind::Text,
            SegmentKind::SoftHyphen,
            SegmentKind::Text,
        ]);
        let mut walked = Vec::new();
        walk_lines_full(&data, 5.0, &data.profile, |line| walked.push(line));
        let streamed = collect_streaming_lines(&data, 5.0);

        assert_eq!(walked.len(), 2);
        assert_eq!(walked[0].width, 5.0);
        assert!(walked[0].ends_with_discretionary_hyphen);
        assert!(!walked[1].ends_with_discretionary_hyphen);
        assert_eq!(
            walked.iter().map(line_state_signature).collect::<Vec<_>>(),
            streamed
                .iter()
                .map(range_state_signature)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn short_mixed_control_runs_have_cross_surface_parity() {
        const KINDS: [SegmentKind; 5] = [
            SegmentKind::Text,
            SegmentKind::Space,
            SegmentKind::ZeroWidthBreak,
            SegmentKind::SoftHyphen,
            SegmentKind::Tab,
        ];

        for length in 0..=4 {
            let case_count = (0..length).fold(1_usize, |count, _| count * KINDS.len());
            for encoded in 0..case_count {
                let mut value = encoded;
                let mut kinds = Vec::with_capacity(length);
                for _ in 0..length {
                    if let Some(kind) = KINDS.get(value % KINDS.len()) {
                        kinds.push(*kind);
                    }
                    value /= KINDS.len();
                }

                let data = make_control_data(&kinds);
                for max_width in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, f64::INFINITY] {
                    let mut full = Vec::new();
                    walk_lines_full(&data, max_width, &data.profile, |line| full.push(line));
                    let streamed = collect_streaming_lines(&data, max_width);

                    assert_eq!(
                        count_lines_full(&data, max_width, &data.profile),
                        full.len(),
                        "full count mismatch for {kinds:?} at {max_width}"
                    );
                    assert_eq!(
                        full.iter().map(line_state_signature).collect::<Vec<_>>(),
                        streamed
                            .iter()
                            .map(range_state_signature)
                            .collect::<Vec<_>>(),
                        "full/stream mismatch for {kinds:?} at {max_width}"
                    );

                    if kinds.iter().all(|kind| {
                        matches!(
                            kind,
                            SegmentKind::Text | SegmentKind::Space | SegmentKind::ZeroWidthBreak
                        )
                    }) {
                        let mut simple = Vec::new();
                        walk_lines_simple(&data, max_width, &data.profile, |line| {
                            simple.push(line);
                        });
                        assert_eq!(
                            count_lines_simple(&data, max_width, &data.profile),
                            simple.len(),
                            "simple count mismatch for {kinds:?} at {max_width}"
                        );
                        assert_eq!(
                            simple.iter().map(line_state_signature).collect::<Vec<_>>(),
                            streamed
                                .iter()
                                .map(range_state_signature)
                                .collect::<Vec<_>>(),
                            "simple/stream mismatch for {kinds:?} at {max_width}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn streaming_repeatedly_advances_through_one_long_segment() {
        let data = make_breakable_word_data(20, 6.0);
        let mut cursor = LayoutCursor::default();
        let mut signatures = Vec::new();

        while let Some((line, next)) = layout_next_line_range(&data, cursor, 18.0, &data.profile) {
            signatures.push((
                line.start.segment_index,
                line.start.grapheme_index,
                line.end.segment_index,
                line.end.grapheme_index,
                line.width,
            ));
            assert_ne!(next, cursor, "streaming cursor must always make progress");
            cursor = next;
        }

        assert_eq!(
            signatures,
            vec![
                (0, 0, 0, 3, 18.0),
                (0, 3, 0, 6, 18.0),
                (0, 6, 0, 9, 18.0),
                (0, 9, 0, 12, 18.0),
                (0, 12, 0, 15, 18.0),
                (0, 15, 0, 18, 18.0),
                (0, 18, 1, 0, 12.0),
            ]
        );
        assert_eq!(cursor, LayoutCursor::new(1, 0));
    }

    #[test]
    fn mid_word_break_makes_progress_when_one_grapheme_exceeds_the_line() {
        let data = make_breakable_word_data(5, 6.0);
        let mut lines = Vec::new();

        walk_lines_simple(&data, 5.0, &data.profile, |line| lines.push(line));

        assert_eq!(lines.len(), 5);
        assert!(
            lines
                .iter()
                .all(|line| (line.width - 6.0).abs() < f64::EPSILON)
        );
        assert_eq!(lines.last().map(line_signature), Some((0, 4, 1, 0, 6.0)));
    }

    #[test]
    fn cjk_like_segments_have_count_walk_and_streaming_parity() {
        let data = make_simple_data(
            &[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            &[SegmentKind::Text; 8],
        );
        let mut walked = Vec::new();
        walk_lines_simple(&data, 30.0, &data.profile, |line| walked.push(line));
        let streamed = collect_streaming_lines(&data, 30.0);

        assert_eq!(count_lines_simple(&data, 30.0, &data.profile), 3);
        assert_eq!(walked.len(), streamed.len());
        assert_eq!(
            walked.iter().map(line_signature).collect::<Vec<_>>(),
            streamed.iter().map(range_signature).collect::<Vec<_>>()
        );
        assert_eq!(
            streamed.iter().map(|line| line.width).collect::<Vec<_>>(),
            vec![30.0, 30.0, 20.0]
        );
    }

    #[test]
    fn zero_width_word_sequence_has_count_walk_and_streaming_parity() {
        let data = make_simple_data(
            &[6.0, 2.5, 6.0, 2.5, 6.0, 2.5, 6.0, 2.5, 6.0, 2.5, 6.0],
            &[
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
            ],
        );
        let mut walked = Vec::new();
        walk_lines_simple(&data, 0.0, &data.profile, |line| walked.push(line));
        let streamed = collect_streaming_lines(&data, 0.0);

        assert_eq!(count_lines_simple(&data, 0.0, &data.profile), 6);
        assert_eq!(walked.len(), streamed.len());
        assert_eq!(
            walked.iter().map(line_signature).collect::<Vec<_>>(),
            streamed.iter().map(range_signature).collect::<Vec<_>>()
        );
        assert!(
            streamed
                .iter()
                .all(|line| (line.width - 6.0).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn pending_space_does_not_prevent_repeated_word_splitting() {
        let mut data = make_simple_data(
            &[30.0, 2.5, 30.0],
            &[SegmentKind::Text, SegmentKind::Space, SegmentKind::Text],
        );
        data.breakable_widths = vec![Some(vec![6.0; 5]), None, Some(vec![6.0; 5])];

        for (max_width, expected_count) in [(18.0, 4), (0.0, 10)] {
            let mut walked = Vec::new();
            walk_lines_simple(&data, max_width, &data.profile, |line| walked.push(line));
            let streamed = collect_streaming_lines(&data, max_width);

            assert_eq!(
                count_lines_simple(&data, max_width, &data.profile),
                expected_count
            );
            assert_eq!(walked.len(), streamed.len());
            assert_eq!(
                walked.iter().map(line_signature).collect::<Vec<_>>(),
                streamed.iter().map(range_signature).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn trailing_hard_break_has_full_walk_and_streaming_parity() {
        let mut data = make_simple_data(
            &[6.0, 0.0, 6.0, 0.0],
            &[
                SegmentKind::Text,
                SegmentKind::HardBreak,
                SegmentKind::Text,
                SegmentKind::HardBreak,
            ],
        );
        data.simple_fast_path = false;
        data.chunks = vec![
            PreparedLineChunk {
                start_segment_index: 0,
                end_segment_index: 1,
                consumed_end_segment_index: 2,
            },
            PreparedLineChunk {
                start_segment_index: 2,
                end_segment_index: 3,
                consumed_end_segment_index: 4,
            },
            PreparedLineChunk {
                start_segment_index: 4,
                end_segment_index: 4,
                consumed_end_segment_index: 4,
            },
        ];
        let mut walked = Vec::new();
        walk_lines_full(&data, 100.0, &data.profile, |line| walked.push(line));
        let streamed = collect_streaming_lines(&data, 100.0);

        assert_eq!(count_lines_full(&data, 100.0, &data.profile), 3);
        assert_eq!(walked.len(), streamed.len());
        assert_eq!(
            walked.iter().map(line_signature).collect::<Vec<_>>(),
            streamed.iter().map(range_signature).collect::<Vec<_>>()
        );
        assert_eq!(
            streamed.iter().map(|line| line.width).collect::<Vec<_>>(),
            vec![6.0, 6.0, 0.0]
        );
    }

    #[test]
    fn tab_advance_is_replayed_for_full_walk_and_streaming_widths() {
        let mut data = make_simple_data(
            &[6.0, 0.0, 6.0],
            &[SegmentKind::Text, SegmentKind::Tab, SegmentKind::Text],
        );
        data.simple_fast_path = false;
        data.tab_stop_advance = 20.0;
        let mut walked = Vec::new();
        walk_lines_full(&data, f64::INFINITY, &data.profile, |line| {
            walked.push(line);
        });
        let streamed = collect_streaming_lines(&data, f64::INFINITY);

        assert_eq!(count_lines_full(&data, f64::INFINITY, &data.profile), 1);
        assert_eq!(
            walked.iter().map(line_signature).collect::<Vec<_>>(),
            vec![(0, 0, 3, 0, 26.0)]
        );
        assert_eq!(
            walked.iter().map(line_signature).collect::<Vec<_>>(),
            streamed.iter().map(range_signature).collect::<Vec<_>>()
        );
    }

    #[test]
    fn public_tab_geometry_reports_the_dynamic_advance_on_every_surface() {
        use crate::backend::{FontSpec, fixed::FixedWidthBackend};
        use crate::types::{PrepareOptions, WhiteSpaceMode};

        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("10px monospace").expect("test font is valid");
        let options = PrepareOptions {
            white_space: WhiteSpaceMode::PreWrap,
            ..PrepareOptions::default()
        };
        let prepared = crate::prepare("a\tb", &font, &backend, options.clone())
            .expect("test preparation succeeds");
        let rich = crate::prepare_with_segments("a\tb", &font, &backend, options)
            .expect("test preparation with segments succeeds");

        assert_eq!(
            crate::measure_natural_width(&prepared).expect("natural width succeeds"),
            26.0
        );
        let materialized =
            crate::layout_with_lines(&rich, f64::INFINITY).expect("materialized layout succeeds");
        assert_eq!(materialized.first().map(|line| line.width), Some(26.0));
        assert_eq!(
            materialized.first().map(|line| line.text.as_str()),
            Some("a\tb")
        );

        let streamed = crate::layout_next_line(&prepared, LayoutCursor::default(), f64::INFINITY)
            .expect("streaming layout succeeds")
            .expect("one line is available");
        assert_eq!(streamed.0.width, 26.0);
        assert_eq!(streamed.0.start, LayoutCursor::new(0, 0));
        assert_eq!(streamed.0.end, LayoutCursor::new(3, 0));
    }

    #[test]
    fn simple_count_walk_and_streaming_remain_in_lockstep_across_mixed_runs() {
        const RADIX: usize = 4;
        const SEGMENTS: usize = 4;
        let case_count = RADIX.pow(SEGMENTS as u32);

        for encoded in 0..case_count {
            let mut value = encoded;
            let mut widths = Vec::with_capacity(SEGMENTS);
            let mut kinds = Vec::with_capacity(SEGMENTS);
            let mut breakable_widths = Vec::with_capacity(SEGMENTS);
            for _ in 0..SEGMENTS {
                match value % RADIX {
                    0 => {
                        widths.push(6.0);
                        kinds.push(SegmentKind::Text);
                        breakable_widths.push(None);
                    }
                    1 => {
                        widths.push(18.0);
                        kinds.push(SegmentKind::Text);
                        breakable_widths.push(Some(vec![6.0; 3]));
                    }
                    2 => {
                        widths.push(2.5);
                        kinds.push(SegmentKind::Space);
                        breakable_widths.push(None);
                    }
                    _ => {
                        widths.push(0.0);
                        kinds.push(SegmentKind::ZeroWidthBreak);
                        breakable_widths.push(None);
                    }
                }
                value /= RADIX;
            }

            let mut data = make_simple_data(&widths, &kinds);
            data.breakable_widths = breakable_widths;
            for max_width in [0.0, 5.0, 6.0, 12.0, 18.0, 30.0, f64::INFINITY] {
                let mut walked = Vec::new();
                walk_lines_simple(&data, max_width, &data.profile, |line| walked.push(line));
                let streamed = collect_streaming_lines(&data, max_width);
                let walked_signatures = walked.iter().map(line_signature).collect::<Vec<_>>();
                let streamed_signatures = streamed.iter().map(range_signature).collect::<Vec<_>>();

                assert_eq!(
                    count_lines_simple(&data, max_width, &data.profile),
                    walked.len(),
                    "count/walk mismatch for encoded case {encoded} at width {max_width}"
                );
                assert_eq!(
                    walked_signatures, streamed_signatures,
                    "walk/stream mismatch for encoded case {encoded} at width {max_width}"
                );
            }
        }
    }

    #[test]
    fn test_tab_advance() {
        assert!((get_tab_advance(0.0, 48.0) - 48.0).abs() < 0.001);
        assert!((get_tab_advance(10.0, 48.0) - 38.0).abs() < 0.001);
        assert!((get_tab_advance(48.0, 48.0) - 48.0).abs() < 0.001);
    }

    #[test]
    fn test_cursor_api() {
        let data = make_simple_data(
            &[30.0, 5.0, 30.0, 5.0, 30.0],
            &[
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
                SegmentKind::Space,
                SegmentKind::Text,
            ],
        );
        let profile = EngineProfile::native();
        let start = LayoutCursor::default();

        let result = layout_next_line_range(&data, start, 50.0, &profile);
        assert!(result.is_some());
        let (line, next) = result.unwrap();
        assert_eq!(line.start.segment_index, 0);
        assert!(next.segment_index >= 2);
    }

    #[test]
    fn drifted_parallel_arrays_are_bounded_by_shortest_array() {
        let mut data = make_simple_data(&[30.0], &[SegmentKind::Text]);
        data.kinds.clear();

        assert_eq!(data.segment_count(), 0);
        assert_eq!(count_lines_simple(&data, 50.0, &data.profile), 1);

        let mut lines = Vec::new();
        walk_lines_simple(&data, 50.0, &data.profile, |line| lines.push(line));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].width, 0.0);
    }

    #[test]
    fn streaming_cursor_clamps_drifted_grapheme_index() {
        let mut data = make_simple_data(&[40.0], &[SegmentKind::Text]);
        data.breakable_widths[0] = Some(vec![10.0, 10.0, 10.0, 10.0]);
        let cursor = LayoutCursor::new(0, usize::MAX);

        let result = layout_next_line_range(&data, cursor, f64::INFINITY, &data.profile);
        assert!(result.is_some());
        if let Some((line, next)) = result {
            assert_eq!(line.start.grapheme_index(), 4);
            assert_eq!(line.width, 0.0);
            assert_eq!(next.segment_index(), 1);
        }
    }

    #[test]
    fn full_walker_clamps_drifted_chunk_bounds() {
        let mut data = make_simple_data(&[40.0], &[SegmentKind::Text]);
        data.chunks = vec![PreparedLineChunk {
            start_segment_index: usize::MAX,
            end_segment_index: usize::MAX,
            consumed_end_segment_index: usize::MAX,
        }];

        let mut lines = Vec::new();
        walk_lines_full(&data, 50.0, &data.profile, |line| lines.push(line));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].start_segment, 1);
        assert_eq!(lines[0].end_segment, 1);
    }
}
