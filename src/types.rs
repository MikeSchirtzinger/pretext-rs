//! Public configuration, cursor, result, and prepared-text handle types.

/// Segment kind -- classifies each measured segment for line-breaking decisions.
///
/// Maps directly to the TypeScript `SegmentBreakKind` but uses Rust enums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    /// Normal text content (word or partial word).
    Text,
    /// Collapsible whitespace (CSS `white-space: normal`).
    Space,
    /// Preserved whitespace (CSS `white-space: pre-wrap`).
    PreservedSpace,
    /// Tab character.
    Tab,
    /// Glue -- non-breaking connection between segments (NBSP, etc.).
    Glue,
    /// Zero-width break opportunity (ZWSP, etc.).
    ZeroWidthBreak,
    /// Soft hyphen -- invisible unless it becomes a line break.
    SoftHyphen,
    /// Hard break (newline in pre-wrap mode).
    HardBreak,
}

impl SegmentKind {
    /// Whether a line break is permitted after this segment kind.
    #[must_use]
    #[inline]
    pub const fn can_break_after(self) -> bool {
        matches!(
            self,
            Self::Space
                | Self::PreservedSpace
                | Self::Tab
                | Self::ZeroWidthBreak
                | Self::SoftHyphen
        )
    }

    /// Whether this segment contributes zero width at the end of a line
    /// (trailing spaces "hang" past the edge, matching CSS behavior).
    #[must_use]
    #[inline]
    pub const fn hangs_at_line_end(self) -> bool {
        matches!(self, Self::Space | Self::Tab)
    }
}

/// Whitespace handling mode, matching CSS `white-space` property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WhiteSpaceMode {
    /// Collapse whitespace, wrap at container edge. (CSS `white-space: normal`)
    #[default]
    Normal,
    /// Preserve whitespace, wrap at container edge. (CSS `white-space: pre-wrap`)
    PreWrap,
}

/// Engine profile -- browser-specific numeric line-fit tolerance.
///
/// Different rendering engines (Chromium, Safari/WebKit, Gecko) have subtly
/// different floating-point behavior at line edges. Only behavior implemented
/// by the engine is exposed here; unsupported compatibility switches are not
/// accepted as placebo configuration.
#[derive(Debug, Clone)]
pub struct EngineProfile {
    /// Epsilon tolerance for line-fit decisions. Chromium/Gecko use 0.005px;
    /// Safari uses 1/64px (~0.015625).
    pub line_fit_epsilon: f64,
}

impl EngineProfile {
    /// Profile matching Chromium / Blink behavior.
    #[must_use]
    pub const fn chromium() -> Self {
        Self {
            line_fit_epsilon: 0.005,
        }
    }

    /// Profile matching Safari / `WebKit` behavior.
    #[must_use]
    pub fn safari() -> Self {
        Self {
            line_fit_epsilon: 1.0 / 64.0,
        }
    }

    /// Profile matching Firefox / Gecko behavior.
    #[must_use]
    pub const fn gecko() -> Self {
        Self {
            line_fit_epsilon: 0.005,
        }
    }

    /// Default profile for non-browser (native) use. Conservative tolerances.
    #[must_use]
    pub const fn native() -> Self {
        Self {
            line_fit_epsilon: 0.005,
        }
    }

    /// Validate the numeric values used by the line-breaking engine.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidInput`] when [`Self::line_fit_epsilon`]
    /// is negative, infinite, or NaN.
    pub fn validate(&self) -> crate::Result<()> {
        if !self.line_fit_epsilon.is_finite() || self.line_fit_epsilon < 0.0 {
            return Err(crate::Error::invalid_input(
                "profile.line_fit_epsilon",
                "must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

impl Default for EngineProfile {
    fn default() -> Self {
        Self::native()
    }
}

/// A chunk of segments between hard breaks (used in pre-wrap mode).
///
/// Each chunk represents a run of segments that ends at a hard break or
/// at the end of the text. The line-breaking walker processes one chunk
/// at a time, emitting hard breaks between chunks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedLineChunk {
    /// Index of first segment in this chunk (inclusive).
    pub(crate) start_segment_index: usize,
    /// Index of last segment in this chunk (exclusive).
    pub(crate) end_segment_index: usize,
    /// Index past the trailing hard-break segment (if any). Used for
    /// determining whether a pending break at chunk end should use
    /// paint width instead of fit width.
    pub(crate) consumed_end_segment_index: usize,
}

/// The core prepared text data -- parallel arrays over segments.
///
/// This is the internal representation consumed by the line-breaking engine.
/// The `prepare()` function produces this; `layout()` reads it. The hot path
/// (`layout`) only touches these arrays -- no strings, no DOM, no allocations.
#[derive(Debug, Clone)]
pub(crate) struct PreparedData {
    /// Width of each segment as measured by the backend.
    pub(crate) widths: Vec<f64>,
    /// Width contribution when this segment ends a line (for fit calculation).
    /// Trailing spaces contribute 0 here (they "hang" past the edge).
    pub(crate) line_end_fit_advances: Vec<f64>,
    /// Visual width when this segment ends a line (for paint/render width).
    pub(crate) line_end_paint_advances: Vec<f64>,
    /// Kind of each segment.
    pub(crate) kinds: Vec<SegmentKind>,
    /// Per-grapheme widths for segments that can be broken mid-word
    /// (`overflow-wrap: break-word`). `None` for segments that don't need it.
    pub(crate) breakable_widths: Vec<Option<Vec<f64>>>,
    /// Pre-compiled hard-break chunks for efficient line walking.
    pub(crate) chunks: Vec<PreparedLineChunk>,
    /// Tab stop advance (typically `8 * space_width`).
    pub(crate) tab_stop_advance: f64,
    /// Width of a discretionary hyphen (for soft-hyphen rendering).
    pub(crate) discretionary_hyphen_width: f64,
    /// Whether the simple fast path can be used (no hard breaks,
    /// no tabs, no soft hyphens, no preserved spaces).
    pub(crate) simple_fast_path: bool,
    /// Validated profile selected when this prepared value was created.
    pub(crate) profile: EngineProfile,
}

impl PreparedData {
    /// Number of complete entries shared by every parallel segment array.
    ///
    /// Using the shortest array keeps the walkers bounded even if an internal
    /// producer violates the representation invariant.
    #[must_use]
    pub(crate) fn segment_count(&self) -> usize {
        self.widths
            .len()
            .min(self.line_end_fit_advances.len())
            .min(self.line_end_paint_advances.len())
            .min(self.kinds.len())
            .min(self.breakable_widths.len())
    }
}

/// Opaque handle to prepared text -- the result of `prepare()`.
///
/// This is the fast-path type: it contains only the layout data,
/// no segment strings. Use `PreparedTextWithSegments` if you need
/// to materialize line text content.
#[derive(Debug, Clone)]
pub struct PreparedText {
    pub(crate) data: PreparedData,
}

impl PreparedText {
    /// Number of segments in the prepared text.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Vec::len() is not const
    pub fn segment_count(&self) -> usize {
        self.data.segment_count()
    }

    /// Whether the prepared text is empty.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Vec::is_empty() is not const
    pub fn is_empty(&self) -> bool {
        self.data.segment_count() == 0
    }

    /// Conservative size of heap allocations owned by this prepared value.
    ///
    /// Saturating arithmetic deliberately turns an unrepresentable estimate
    /// into `usize::MAX`, which causes bounded runtimes to reject the value.
    #[must_use]
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    pub(crate) fn estimated_heap_bytes(&self) -> usize {
        let data = &self.data;
        let mut bytes = std::mem::size_of::<PreparedData>();
        bytes = bytes.saturating_add(
            data.widths
                .capacity()
                .saturating_mul(std::mem::size_of::<f64>()),
        );
        bytes = bytes.saturating_add(
            data.line_end_fit_advances
                .capacity()
                .saturating_mul(std::mem::size_of::<f64>()),
        );
        bytes = bytes.saturating_add(
            data.line_end_paint_advances
                .capacity()
                .saturating_mul(std::mem::size_of::<f64>()),
        );
        bytes = bytes.saturating_add(
            data.kinds
                .capacity()
                .saturating_mul(std::mem::size_of::<SegmentKind>()),
        );
        bytes = bytes.saturating_add(
            data.breakable_widths
                .capacity()
                .saturating_mul(std::mem::size_of::<Option<Vec<f64>>>()),
        );
        for widths in data.breakable_widths.iter().flatten() {
            bytes =
                bytes.saturating_add(widths.capacity().saturating_mul(std::mem::size_of::<f64>()));
        }
        bytes.saturating_add(
            data.chunks
                .capacity()
                .saturating_mul(std::mem::size_of::<PreparedLineChunk>()),
        )
    }
}

/// Rich prepared text -- includes segment strings for line materialization.
///
/// Returned by `prepare_with_segments()`. Slightly more memory than
/// `PreparedText` but allows `layout_with_lines()` to produce actual
/// text content per line.
#[derive(Debug, Clone)]
pub struct PreparedTextWithSegments {
    pub(crate) data: PreparedData,
    /// The original text segments, aligned 1:1 with the data arrays.
    pub(crate) segments: Vec<String>,
    /// Optional bidi embedding level at each segment's first Unicode scalar.
    /// `None` means every resolved scalar level is base LTR. A segment can
    /// contain an internal level transition, so this is coarse advisory
    /// metadata rather than a complete directional-run representation. See
    /// [`crate::bidi`].
    pub(crate) seg_levels: Option<Vec<i8>>,
}

impl PreparedTextWithSegments {
    /// Original text segments in layout order.
    ///
    /// The returned slice is read-only so it cannot be desynchronized from
    /// the prepared measurement arrays.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Bidi level at the start of each segment, aligned 1:1 with
    /// [`Self::segments`].
    ///
    /// A segment may cross a resolved-level boundary. Renderers must use
    /// [`crate::bidi::compute_bidi_levels`] and split directional runs before
    /// applying line-specific reordering; this convenience array is not
    /// sufficient for visual ordering by itself.
    #[must_use]
    pub fn seg_levels(&self) -> Option<&[i8]> {
        self.seg_levels.as_deref()
    }
}

/// Cursor position within prepared text -- identifies a specific
/// point between/within segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayoutCursor {
    /// Index of the current segment.
    pub(crate) segment_index: usize,
    /// Index of the current grapheme within the segment (for mid-word breaks).
    pub(crate) grapheme_index: usize,
}

impl LayoutCursor {
    /// Construct a cursor for crate-internal, already-validated indices.
    #[must_use]
    pub(crate) const fn new(segment_index: usize, grapheme_index: usize) -> Self {
        Self {
            segment_index,
            grapheme_index,
        }
    }

    /// Index of the segment containing this cursor.
    #[must_use]
    pub const fn segment_index(&self) -> usize {
        self.segment_index
    }

    /// Index of the grapheme within the current segment.
    ///
    /// A value of zero places the cursor at a segment boundary.
    #[must_use]
    pub const fn grapheme_index(&self) -> usize {
        self.grapheme_index
    }
}

/// Result of `layout()` -- just height and line count.
#[derive(Debug, Clone, Copy)]
pub struct LayoutResult {
    /// Number of lines the text occupies at the given width.
    pub line_count: usize,
    /// Total height (`line_count * line_height`).
    pub height: f64,
}

/// A materialized line from `layout_with_lines()`.
#[derive(Debug, Clone)]
pub struct LayoutLine {
    /// The text content of this line.
    pub text: String,
    /// The rendered width of this line.
    pub width: f64,
    /// Cursor at the start of this line.
    pub start: LayoutCursor,
    /// Cursor at the end of this line.
    pub end: LayoutCursor,
    /// Whether this line paints a discretionary hyphen at its end.
    ///
    /// This is true only when a U+00AD soft hyphen was selected as the line's
    /// actual break opportunity. An unbroken or terminal soft hyphen remains
    /// invisible.
    pub ends_with_discretionary_hyphen: bool,
}

/// A line range from `walk_line_ranges()` -- geometry only, no text.
#[derive(Debug, Clone, Copy)]
pub struct LayoutLineRange {
    /// The rendered width of this line.
    pub width: f64,
    /// Cursor at the start of this line.
    pub start: LayoutCursor,
    /// Cursor at the end of this line.
    pub end: LayoutCursor,
    /// Whether this range paints a discretionary hyphen at its end.
    ///
    /// Cursor positions alone cannot distinguish a selected soft-hyphen
    /// break from a range that merely ends after an unpainted U+00AD.
    pub ends_with_discretionary_hyphen: bool,
}

/// Default maximum UTF-8 input size accepted by preparation APIs: 4 MiB.
pub const DEFAULT_MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;

/// Default maximum number of extended grapheme clusters accepted per input.
pub const DEFAULT_MAX_GRAPHEMES: usize = 65_536;

/// Default maximum number of analyzed segments accepted per input.
pub const DEFAULT_MAX_SEGMENTS: usize = 65_536;

/// Options for `prepare()` / `prepare_with_segments()`.
#[derive(Debug, Clone)]
pub struct PrepareOptions {
    /// Whitespace handling mode. Default: Normal (collapse + wrap).
    pub white_space: WhiteSpaceMode,
    /// Engine profile for browser-specific behavior. Default: native.
    pub profile: Option<EngineProfile>,
    /// Maximum accepted input size in UTF-8 bytes.
    ///
    /// The preparation APIs reject larger inputs before analysis or
    /// measurement. The default is [`DEFAULT_MAX_INPUT_BYTES`]. Set this to
    /// zero to permit only empty input.
    pub max_input_bytes: usize,
    /// Maximum accepted number of extended grapheme clusters.
    ///
    /// This bound is checked before analysis allocates segment structures.
    /// The default is [`DEFAULT_MAX_GRAPHEMES`].
    pub max_graphemes: usize,
    /// Maximum accepted number of analyzed segments.
    ///
    /// This bounds the parallel prepared arrays even for punctuation-heavy or
    /// adversarial inputs. The default is [`DEFAULT_MAX_SEGMENTS`].
    pub max_segments: usize,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            white_space: WhiteSpaceMode::default(),
            profile: None,
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_graphemes: DEFAULT_MAX_GRAPHEMES,
            max_segments: DEFAULT_MAX_SEGMENTS,
        }
    }
}

/// Diagnostic timing/shape data returned by [`crate::profile_prepare`].
///
/// Mirrors upstream `PrepareProfile`. Used by benchmarks and the browser
/// parity harness to separate the analysis phase from the measurement phase
/// without duplicating `prepare()` logic.
#[derive(Debug, Clone, Copy)]
pub struct PrepareProfile {
    /// Wall time spent validating and analyzing text, in milliseconds.
    pub analysis_ms: f64,
    /// Wall time spent in the measurement loop, in milliseconds.
    pub measure_ms: f64,
    /// Total wall time for `prepare()`, in milliseconds.
    pub total_ms: f64,
    /// Number of segments produced by analysis (before measurement).
    pub analysis_segments: usize,
    /// Number of segments in the resulting prepared text.
    pub prepared_segments: usize,
    /// Number of segments that carry per-grapheme breakable widths
    /// (overflow-wrap candidates).
    pub breakable_segments: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_options_use_documented_input_limit() {
        assert_eq!(
            PrepareOptions::default().max_input_bytes,
            DEFAULT_MAX_INPUT_BYTES
        );
        assert_eq!(
            PrepareOptions::default().max_graphemes,
            DEFAULT_MAX_GRAPHEMES
        );
        assert_eq!(PrepareOptions::default().max_segments, DEFAULT_MAX_SEGMENTS);
    }

    #[test]
    fn engine_profile_rejects_invalid_epsilon() {
        for epsilon in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.001] {
            let profile = EngineProfile {
                line_fit_epsilon: epsilon,
            };
            assert!(profile.validate().is_err());
        }
        assert!(EngineProfile::native().validate().is_ok());
    }

    #[test]
    fn layout_cursor_exposes_read_only_indices() {
        let cursor = LayoutCursor::new(3, 7);
        assert_eq!(cursor.segment_index(), 3);
        assert_eq!(cursor.grapheme_index(), 7);
    }
}
