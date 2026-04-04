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
pub struct InternalLine {
    pub start_segment: usize,
    pub start_grapheme: usize,
    pub end_segment: usize,
    pub end_grapheme: usize,
    pub width: f64,
}

/// Pending break state -- tracks the best break opportunity found so far.
#[derive(Debug, Clone, Copy)]
struct PendingBreak {
    segment_index: usize,
    grapheme_index: usize,
    fit_width: f64,
    paint_width: f64,
}

// ---- Simple fast path -------------------------------------------------------

/// Count lines using the simple fast path.
///
/// Only handles `Text`, `Space`, and `ZeroWidthBreak` segments. No chunks,
/// no tabs, no soft hyphens. This is the common case for prose.
#[must_use]
pub fn count_lines_simple(data: &PreparedData, max_width: f64, profile: &EngineProfile) -> usize {
    let mut line_count = 1;
    let mut line_w = 0.0;
    let epsilon = profile.line_fit_epsilon;
    let seg_count = data.widths.len();

    if seg_count == 0 {
        return 1;
    }

    let mut i = 0;
    while i < seg_count {
        let kind = data.kinds[i];
        let width = data.widths[i];

        let new_w = line_w + width;

        if kind == SegmentKind::Space || kind == SegmentKind::ZeroWidthBreak {
            // Breakable -- update line width and move on
            line_w = new_w;
            i += 1;
            continue;
        }

        // Text segment
        if new_w > max_width + epsilon {
            // Overflow -- need to break

            // Find the last break opportunity
            if i > 0 && can_break_before(data, i) {
                // Break before this segment
                line_count += 1;
                line_w = width;
            } else if let Some(ref bw) = data.breakable_widths[i] {
                // Break within this segment (overflow-wrap: break-word)
                let (lines, remaining_w) =
                    break_within_segment(bw, line_w, max_width, epsilon);
                line_count += lines;
                line_w = remaining_w;
            } else {
                // Can't break -- segment overflows the line
                if line_w > 0.0 {
                    line_count += 1;
                    line_w = width;
                } else {
                    // Segment alone is wider than max_width -- accept it
                    line_w = new_w;
                }
            }
        } else {
            line_w = new_w;
        }

        i += 1;
    }

    line_count
}

/// Walk lines using the simple fast path, calling a callback per line.
pub fn walk_lines_simple<F>(
    data: &PreparedData,
    max_width: f64,
    profile: &EngineProfile,
    mut on_line: F,
) where
    F: FnMut(InternalLine),
{
    let epsilon = profile.line_fit_epsilon;
    let seg_count = data.widths.len();

    if seg_count == 0 {
        on_line(InternalLine {
            start_segment: 0,
            start_grapheme: 0,
            end_segment: 0,
            end_grapheme: 0,
            width: 0.0,
        });
        return;
    }

    let mut line_start_seg = 0;
    let mut line_start_grapheme = 0;
    let mut line_w = 0.0;
    let mut pending: Option<PendingBreak> = None;

    let mut i = 0;
    while i < seg_count {
        let kind = data.kinds[i];
        let width = data.widths[i];
        let fit_advance = data.line_end_fit_advances[i];

        let new_w = line_w + width;

        // Track break opportunities
        if kind.can_break_after() {
            pending = Some(PendingBreak {
                segment_index: i + 1,
                grapheme_index: 0,
                fit_width: line_w + fit_advance,
                paint_width: line_w + data.line_end_paint_advances[i],
            });
            line_w = new_w;
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
                });
                line_start_seg = pb.segment_index;
                line_start_grapheme = pb.grapheme_index;
                // Skip leading spaces on new line
                while line_start_seg < seg_count
                    && data.kinds[line_start_seg] == SegmentKind::Space
                {
                    line_start_seg += 1;
                }
                line_w = recompute_line_width(data, line_start_seg, i);
                continue; // Re-evaluate current segment
            }

            // Try grapheme breaking within segment
            if let Some(ref bw) = data.breakable_widths[i]
                && let Some(break_at) = find_grapheme_break(bw, line_w, max_width, epsilon)
            {
                let break_width: f64 = bw[..break_at].iter().sum();
                on_line(InternalLine {
                    start_segment: line_start_seg,
                    start_grapheme: line_start_grapheme,
                    end_segment: i,
                    end_grapheme: break_at,
                    width: line_w + break_width,
                });
                line_start_seg = i;
                line_start_grapheme = break_at;
                line_w = bw[break_at..].iter().sum();
                i += 1;
                continue;
            }

            // Force break before this segment if we have content
            if line_w > 0.0 && line_start_seg < i {
                on_line(InternalLine {
                    start_segment: line_start_seg,
                    start_grapheme: line_start_grapheme,
                    end_segment: i,
                    end_grapheme: 0,
                    width: line_w,
                });
                line_start_seg = i;
                line_start_grapheme = 0;
                line_w = width;
            } else {
                // Accept overflow on a single segment
                line_w = new_w;
            }
        } else {
            line_w = new_w;
        }

        i += 1;
    }

    // Emit final line
    let final_width = compute_paint_width(data, line_start_seg, seg_count);
    on_line(InternalLine {
        start_segment: line_start_seg,
        start_grapheme: line_start_grapheme,
        end_segment: seg_count,
        end_grapheme: 0,
        width: final_width,
    });
}

// ---- Full path (with chunks, tabs, soft hyphens) ----------------------------

/// Count lines using the full walker with chunk support.
#[must_use]
pub fn count_lines_full(data: &PreparedData, max_width: f64, profile: &EngineProfile) -> usize {
    let mut line_count = 0;
    walk_lines_full(data, max_width, profile, |_| {
        line_count += 1;
    });
    line_count
}

/// Walk lines with full chunk/tab/soft-hyphen support.
pub fn walk_lines_full<F>(
    data: &PreparedData,
    max_width: f64,
    profile: &EngineProfile,
    mut on_line: F,
) where
    F: FnMut(InternalLine),
{
    let seg_count = data.widths.len();

    if seg_count == 0 {
        on_line(InternalLine {
            start_segment: 0,
            start_grapheme: 0,
            end_segment: 0,
            end_grapheme: 0,
            width: 0.0,
        });
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
    let chunk_start = chunk.start_segment_index;
    let chunk_end = chunk.end_segment_index;

    if chunk_start >= chunk_end {
        // Empty chunk (consecutive hard breaks) -- emit empty line
        on_line(InternalLine {
            start_segment: chunk_start,
            start_grapheme: 0,
            end_segment: chunk.consumed_end_segment_index,
            end_grapheme: 0,
            width: 0.0,
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
        let kind = data.kinds[i];
        let width = data.widths[i];

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
                if fit_w <= max_width + epsilon || !has_content {
                    pending = Some(PendingBreak {
                        segment_index: i + 1,
                        grapheme_index: 0,
                        fit_width: fit_w,
                        paint_width: fit_w, // Hyphen is visible at break
                    });
                }
                if profile.prefer_early_soft_hyphen_break
                    && fit_w <= max_width + epsilon
                    && has_content
                {
                    // Safari: prefer breaking here immediately
                    pending = Some(PendingBreak {
                        segment_index: i + 1,
                        grapheme_index: 0,
                        fit_width: fit_w,
                        paint_width: fit_w,
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
                let fit_advance = data.line_end_fit_advances[i];
                let paint_advance = data.line_end_paint_advances[i];
                pending = Some(PendingBreak {
                    segment_index: i + 1,
                    grapheme_index: 0,
                    fit_width: line_w + fit_advance,
                    paint_width: line_w + paint_advance,
                });
                line_w = new_w;
                has_content = true;
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
                if new_w > max_width + epsilon && has_content {
                    // Try pending break
                    if let Some(pb) = pending.take()
                        && pb.fit_width <= max_width + epsilon
                    {
                        on_line(InternalLine {
                            start_segment: line_start_seg,
                            start_grapheme: line_start_grapheme,
                            end_segment: pb.segment_index,
                            end_grapheme: pb.grapheme_index,
                            width: pb.paint_width,
                        });
                        line_start_seg = skip_leading_spaces(data, pb.segment_index, chunk_end);
                        line_start_grapheme = 0;
                        line_w = recompute_line_width(data, line_start_seg, i);
                        has_content = line_start_seg < i;
                        continue; // Re-evaluate current segment
                    }

                    // Try grapheme breaking
                    if let Some(ref bw) = data.breakable_widths[i] {
                        let mut accum = line_w;
                        let mut break_at = 0;
                        for (gi, gw) in bw.iter().enumerate() {
                            if accum + gw > max_width + epsilon && gi > 0 {
                                break;
                            }
                            accum += gw;
                            break_at = gi + 1;
                        }

                        if break_at > 0 && break_at < bw.len() {
                            let break_width: f64 = bw[..break_at].iter().sum();
                            on_line(InternalLine {
                                start_segment: line_start_seg,
                                start_grapheme: line_start_grapheme,
                                end_segment: i,
                                end_grapheme: break_at,
                                width: line_w + break_width,
                            });
                            line_start_seg = i;
                            line_start_grapheme = break_at;
                            line_w = bw[break_at..].iter().sum();
                            has_content = true;
                            i += 1;
                            continue;
                        }
                    }

                    // Force break before this segment
                    if line_start_seg < i {
                        on_line(InternalLine {
                            start_segment: line_start_seg,
                            start_grapheme: line_start_grapheme,
                            end_segment: i,
                            end_grapheme: 0,
                            width: compute_paint_width(data, line_start_seg, i),
                        });
                        line_start_seg = i;
                        line_start_grapheme = 0;
                        line_w = effective_width;
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
    let end = chunk.consumed_end_segment_index;
    let final_width = pending
        .as_ref()
        .filter(|pb| pb.segment_index == chunk_end)
        .map_or_else(
            || compute_paint_width(data, line_start_seg, chunk_end),
            |pb| pb.paint_width,
        );

    on_line(InternalLine {
        start_segment: line_start_seg,
        start_grapheme: line_start_grapheme,
        end_segment: end,
        end_grapheme: 0,
        width: final_width,
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
pub fn layout_next_line_range(
    data: &PreparedData,
    start: LayoutCursor,
    max_width: f64,
    profile: &EngineProfile,
) -> Option<(LayoutLineRange, LayoutCursor)> {
    let seg_count = data.widths.len();

    // Terminal cursor — no more lines to produce
    if start.segment_index >= seg_count {
        // Special case: empty text, first call only
        if seg_count == 0 && start.segment_index == 0 && start.grapheme_index == 0 {
            return Some((
                LayoutLineRange {
                    width: 0.0,
                    start,
                    end: LayoutCursor {
                        segment_index: 0,
                        grapheme_index: 0,
                    },
                },
                LayoutCursor {
                    segment_index: 1,
                    grapheme_index: 0,
                },
            ));
        }
        return None;
    }

    if seg_count == 0 {
        if start.segment_index == 0 && start.grapheme_index == 0 {
            return Some((
                LayoutLineRange {
                    width: 0.0,
                    start,
                    end: LayoutCursor {
                        segment_index: 0,
                        grapheme_index: 0,
                    },
                },
                LayoutCursor {
                    segment_index: 1,
                    grapheme_index: 0,
                },
            ));
        }
        return None;
    }

    let epsilon = profile.line_fit_epsilon;
    let mut line_w = 0.0;
    let mut has_content = false;
    let mut pending: Option<PendingBreak> = None;

    // Handle starting mid-segment (from a previous grapheme break)
    let mut i = start.segment_index;
    if start.grapheme_index > 0 {
        if let Some(ref bw) = data.breakable_widths[i] {
            let safe_gi = start.grapheme_index.min(bw.len());
            let remaining_width: f64 = bw[safe_gi..].iter().sum();
            if remaining_width > max_width + epsilon {
                // Need to break within this remaining portion
                let mut accum = 0.0;
                let mut break_at = safe_gi;
                for (gi, &gw) in bw.iter().enumerate().skip(safe_gi) {
                    if accum + gw > max_width + epsilon && gi > safe_gi {
                        break;
                    }
                    accum += gw;
                    break_at = gi + 1;
                }
                if break_at < bw.len() {
                    return Some((
                        LayoutLineRange {
                            width: bw[safe_gi..break_at].iter().sum(),
                            start,
                            end: LayoutCursor {
                                segment_index: i,
                                grapheme_index: break_at,
                            },
                        },
                        LayoutCursor {
                            segment_index: i,
                            grapheme_index: break_at,
                        },
                    ));
                }
            }
            line_w = remaining_width;
            has_content = true;
        }
        i += 1;
    }

    while i < seg_count {
        let kind = data.kinds[i];

        // Check for hard break
        if kind == SegmentKind::HardBreak {
            let line = LayoutLineRange {
                width: compute_paint_width_from(data, &start, i),
                start,
                end: LayoutCursor {
                    segment_index: i + 1,
                    grapheme_index: 0,
                },
            };
            return Some((
                line,
                LayoutCursor {
                    segment_index: i + 1,
                    grapheme_index: 0,
                },
            ));
        }

        let width = if kind == SegmentKind::Tab {
            get_tab_advance(line_w, data.tab_stop_advance)
        } else {
            data.widths[i]
        };

        let new_w = line_w + width;

        if kind.can_break_after() {
            pending = Some(PendingBreak {
                segment_index: i + 1,
                grapheme_index: 0,
                fit_width: line_w + data.line_end_fit_advances[i],
                paint_width: line_w + data.line_end_paint_advances[i],
            });
            line_w = new_w;
            has_content = true;
            i += 1;
            continue;
        }

        // Text -- check overflow
        if new_w > max_width + epsilon && has_content {
            // Try pending break
            if let Some(pb) = pending.take()
                && pb.fit_width <= max_width + epsilon
            {
                let line = LayoutLineRange {
                    width: pb.paint_width,
                    start,
                    end: LayoutCursor {
                        segment_index: pb.segment_index,
                        grapheme_index: 0,
                    },
                };
                let next_start_seg = skip_leading_spaces(data, pb.segment_index, seg_count);
                return Some((
                    line,
                    LayoutCursor {
                        segment_index: next_start_seg,
                        grapheme_index: 0,
                    },
                ));
            }

            // Try grapheme break
            if let Some(ref bw) = data.breakable_widths[i] {
                let mut accum = line_w;
                let mut break_at = 0;
                for (gi, gw) in bw.iter().enumerate() {
                    if accum + gw > max_width + epsilon && gi > 0 {
                        break;
                    }
                    accum += gw;
                    break_at = gi + 1;
                }
                if break_at > 0 && break_at < bw.len() {
                    let break_width: f64 = bw[..break_at].iter().sum();
                    let line = LayoutLineRange {
                        width: line_w + break_width,
                        start,
                        end: LayoutCursor {
                            segment_index: i,
                            grapheme_index: break_at,
                        },
                    };
                    return Some((
                        line,
                        LayoutCursor {
                            segment_index: i,
                            grapheme_index: break_at,
                        },
                    ));
                }
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
            end: LayoutCursor {
                segment_index: seg_count,
                grapheme_index: 0,
            },
        },
        LayoutCursor {
            segment_index: seg_count,
            grapheme_index: 0,
        },
    ))
}

// ---- Helpers ----------------------------------------------------------------

/// Compute tab advance to the next tab stop.
#[inline]
fn get_tab_advance(current_x: f64, tab_stop_advance: f64) -> f64 {
    if tab_stop_advance <= 0.0 {
        return 0.0;
    }
    tab_stop_advance - (current_x % tab_stop_advance)
}

/// Check if we can break before a given segment (previous segment allows it).
#[inline]
fn can_break_before(data: &PreparedData, index: usize) -> bool {
    if index == 0 {
        return false;
    }
    data.kinds[index - 1].can_break_after()
}

/// Skip leading space segments after a break.
fn skip_leading_spaces(data: &PreparedData, from: usize, limit: usize) -> usize {
    let mut i = from;
    while i < limit && data.kinds[i] == SegmentKind::Space {
        i += 1;
    }
    i
}

/// Recompute line width from segment `from` to segment `to` (exclusive).
fn recompute_line_width(data: &PreparedData, from: usize, to: usize) -> f64 {
    data.widths[from..to].iter().sum()
}

/// Compute paint width (excludes trailing space) from segment `from` to `to`.
fn compute_paint_width(data: &PreparedData, from: usize, to: usize) -> f64 {
    let mut w: f64 = data.widths[from..to].iter().sum();
    // Subtract trailing spaces
    let mut i = to;
    while i > from {
        i -= 1;
        if data.kinds[i].hangs_at_line_end() {
            w -= data.widths[i];
        } else {
            break;
        }
    }
    w
}

/// Compute paint width accounting for cursor start position.
fn compute_paint_width_from(data: &PreparedData, start: &LayoutCursor, to: usize) -> f64 {
    let mut w = 0.0;
    let from = start.segment_index;

    for i in from..to {
        if i == from && start.grapheme_index > 0 {
            // Partial first segment
            if let Some(ref bw) = data.breakable_widths[i] {
                w += bw[start.grapheme_index..].iter().sum::<f64>();
            }
        } else {
            w += data.widths[i];
        }
    }

    // Subtract trailing spaces
    let mut i = to;
    while i > from {
        i -= 1;
        if data.kinds[i].hangs_at_line_end() {
            w -= data.widths[i];
        } else {
            break;
        }
    }
    w
}

/// Find the grapheme index at which to break a segment for overflow-wrap.
///
/// Returns `Some(index)` if a valid mid-segment break point was found,
/// `None` if the entire segment fits or has only one grapheme.
#[inline]
fn find_grapheme_break(
    grapheme_widths: &[f64],
    line_w: f64,
    max_width: f64,
    epsilon: f64,
) -> Option<usize> {
    let mut accum = line_w;
    let mut break_at = 0;
    for (gi, &gw) in grapheme_widths.iter().enumerate() {
        if accum + gw > max_width + epsilon && gi > 0 {
            return Some(break_at);
        }
        accum += gw;
        break_at = gi + 1;
    }
    if break_at > 0 && break_at < grapheme_widths.len() {
        Some(break_at)
    } else {
        None
    }
}

/// Break within a segment's grapheme widths.
///
/// Returns `(extra_line_count, remaining_width)`.
fn break_within_segment(
    grapheme_widths: &[f64],
    initial_width: f64,
    max_width: f64,
    epsilon: f64,
) -> (usize, f64) {
    let mut lines = 0;
    let mut line_w = initial_width;

    for &gw in grapheme_widths {
        if line_w + gw > max_width + epsilon && line_w > 0.0 {
            lines += 1;
            line_w = gw;
        } else {
            line_w += gw;
        }
    }

    (lines, line_w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PreparedLineChunk;

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
        }
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
}
