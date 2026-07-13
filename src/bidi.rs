//! Unicode Bidirectional Algorithm metadata for prepared text.
//!
//! This module delegates bidi classification and paragraph-level resolution to
//! [`unicode_bidi`]. [`compute_bidi_levels`] returns the resolved embedding
//! level at the start of each Unicode scalar value. [`compute_segment_levels`]
//! projects those levels onto pretext segments, whose starts are expressed as
//! Unicode scalar indices rather than UTF-8 byte offsets.
//!
//! These levels are logical-order metadata, not a rendered visual order. A
//! renderer must first choose line boundaries, then apply the line-specific
//! UAX #9 L1 resets and L2 reordering. It is also responsible for mirrored
//! glyph selection, combining-mark placement, and omitting bidi formatting
//! controls from visible output. The line-breaking engine itself does not
//! reorder text.

use crate::{Error, Result};
use unicode_bidi::BidiInfo;

/// A character's Unicode `Bidi_Class` property.
///
/// This re-export preserves pretext's `BidiType` name while exposing the full
/// class set used by the maintained Unicode data tables.
pub use unicode_bidi::BidiClass as BidiType;

/// Classify one Unicode scalar using the bundled Unicode bidi data.
#[must_use]
pub fn classify_char(character: char) -> BidiType {
    unicode_bidi::bidi_class(character)
}

/// Resolve bidi embedding levels for every Unicode scalar in `text`.
///
/// Paragraph direction is detected independently for each paragraph by the
/// Unicode Bidirectional Algorithm. The returned vector is indexed by Unicode
/// scalar position, not by UTF-8 byte offset. Its values are resolved logical
/// embedding levels; line-specific L1 resets have not been applied.
///
/// Returns `None` for empty text and text whose scalar-start levels all resolve
/// to the base left-to-right level. This keeps the allocation-free semantic
/// fast path used by pure LTR callers.
#[must_use]
pub fn compute_bidi_levels(text: &str) -> Option<Vec<i8>> {
    if text.is_empty() {
        return None;
    }

    let info = BidiInfo::new(text, None);
    let mut levels = Vec::with_capacity(text.chars().count());

    for (byte_start, _) in text.char_indices() {
        let level = info.levels.get(byte_start)?;
        levels.push(i8::try_from(level.number()).ok()?);
    }

    if levels.iter().all(|level| *level == 0) {
        None
    } else {
        Some(levels)
    }
}

/// Resolve one embedding level per segment start.
///
/// `seg_starts` contains Unicode scalar indices into `normalized`, not UTF-8
/// byte offsets. Levels remain in logical order and require line-specific UAX
/// #9 processing before a renderer can use them as a visual order.
///
/// Returns `Ok(None)` when [`compute_bidi_levels`] takes the pure-LTR fast path.
///
/// # Errors
///
/// Returns [`Error::InvalidBidiStart`] if any start does not identify a Unicode
/// scalar in `normalized`. Every start is validated before bidi resolution.
pub fn compute_segment_levels(normalized: &str, seg_starts: &[usize]) -> Result<Option<Vec<i8>>> {
    let char_count = normalized.chars().count();
    if let Some(&start) = seg_starts.iter().find(|&&start| start >= char_count) {
        return Err(Error::InvalidBidiStart { start, char_count });
    }

    let Some(bidi_levels) = compute_bidi_levels(normalized) else {
        return Ok(None);
    };

    let mut levels = Vec::with_capacity(seg_starts.len());
    for &start in seg_starts {
        let Some(&level) = bidi_levels.get(start) else {
            return Err(Error::InvalidBidiStart { start, char_count });
        };
        levels.push(level);
    }

    Ok(Some(levels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_levels_reject_out_of_range_start_without_panicking() {
        assert_eq!(
            compute_segment_levels("א", &[99]),
            Err(Error::InvalidBidiStart {
                start: 99,
                char_count: 1,
            })
        );
    }

    #[test]
    fn segment_levels_validate_starts_even_without_bidi_text() {
        assert_eq!(
            compute_segment_levels("abc", &[0, 3]),
            Err(Error::InvalidBidiStart {
                start: 3,
                char_count: 3,
            })
        );
    }

    #[test]
    fn segment_levels_use_unicode_scalar_indices() {
        assert_eq!(compute_segment_levels("😀א", &[0, 1]), Ok(Some(vec![1, 1])));
    }

    #[test]
    fn segment_levels_return_none_for_empty_or_ltr_text() {
        assert_eq!(compute_segment_levels("", &[]), Ok(None));
        assert_eq!(compute_segment_levels("plain text", &[0, 6]), Ok(None));
    }
}
