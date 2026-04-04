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
            Self::Space | Self::PreservedSpace | Self::Tab | Self::ZeroWidthBreak | Self::SoftHyphen
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

/// Engine profile -- browser-specific tolerances and behavior flags.
///
/// Different rendering engines (Chromium, Safari/WebKit, Gecko) have subtly
/// different floating-point behavior at line edges. This struct captures those
/// differences so the line-breaking algorithm can match the target engine.
#[derive(Debug, Clone)]
pub struct EngineProfile {
    /// Epsilon tolerance for line-fit decisions. Chromium/Gecko use 0.005px;
    /// Safari uses 1/64px (~0.015625).
    pub line_fit_epsilon: f64,
    /// Safari: carry CJK punctuation after closing quote to next line.
    pub carry_cjk_after_closing_quote: bool,
    /// Safari: prefer prefix-width accumulation for breakable runs.
    pub prefer_prefix_widths_for_breakable_runs: bool,
    /// Safari: prefer breaking at soft hyphen earlier rather than fitting more.
    pub prefer_early_soft_hyphen_break: bool,
}

impl EngineProfile {
    /// Profile matching Chromium / Blink behavior.
    #[must_use]
    pub const fn chromium() -> Self {
        Self {
            line_fit_epsilon: 0.005,
            carry_cjk_after_closing_quote: true,
            prefer_prefix_widths_for_breakable_runs: false,
            prefer_early_soft_hyphen_break: false,
        }
    }

    /// Profile matching Safari / `WebKit` behavior.
    #[must_use]
    pub fn safari() -> Self {
        Self {
            line_fit_epsilon: 1.0 / 64.0,
            carry_cjk_after_closing_quote: false,
            prefer_prefix_widths_for_breakable_runs: true,
            prefer_early_soft_hyphen_break: true,
        }
    }

    /// Profile matching Firefox / Gecko behavior.
    #[must_use]
    pub const fn gecko() -> Self {
        Self {
            line_fit_epsilon: 0.005,
            carry_cjk_after_closing_quote: false,
            prefer_prefix_widths_for_breakable_runs: false,
            prefer_early_soft_hyphen_break: false,
        }
    }

    /// Default profile for non-browser (native) use. Conservative tolerances.
    #[must_use]
    pub const fn native() -> Self {
        Self {
            line_fit_epsilon: 0.005,
            carry_cjk_after_closing_quote: false,
            prefer_prefix_widths_for_breakable_runs: false,
            prefer_early_soft_hyphen_break: false,
        }
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
#[derive(Debug, Clone)]
pub struct PreparedLineChunk {
    /// Index of first segment in this chunk (inclusive).
    pub start_segment_index: usize,
    /// Index of last segment in this chunk (exclusive).
    pub end_segment_index: usize,
    /// Index past the trailing hard-break segment (if any). Used for
    /// determining whether a pending break at chunk end should use
    /// paint width instead of fit width.
    pub consumed_end_segment_index: usize,
}

/// The core prepared text data -- parallel arrays over segments.
///
/// This is the internal representation consumed by the line-breaking engine.
/// The `prepare()` function produces this; `layout()` reads it. The hot path
/// (`layout`) only touches these arrays -- no strings, no DOM, no allocations.
#[derive(Debug, Clone)]
pub struct PreparedData {
    /// Width of each segment as measured by the backend.
    pub widths: Vec<f64>,
    /// Width contribution when this segment ends a line (for fit calculation).
    /// Trailing spaces contribute 0 here (they "hang" past the edge).
    pub line_end_fit_advances: Vec<f64>,
    /// Visual width when this segment ends a line (for paint/render width).
    pub line_end_paint_advances: Vec<f64>,
    /// Kind of each segment.
    pub kinds: Vec<SegmentKind>,
    /// Per-grapheme widths for segments that can be broken mid-word
    /// (`overflow-wrap: break-word`). `None` for segments that don't need it.
    pub breakable_widths: Vec<Option<Vec<f64>>>,
    /// Pre-compiled hard-break chunks for efficient line walking.
    pub chunks: Vec<PreparedLineChunk>,
    /// Tab stop advance (typically `8 * space_width`).
    pub tab_stop_advance: f64,
    /// Width of a discretionary hyphen (for soft-hyphen rendering).
    pub discretionary_hyphen_width: f64,
    /// Whether the simple fast path can be used (no hard breaks,
    /// no tabs, no soft hyphens, no preserved spaces).
    pub simple_fast_path: bool,
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
        self.data.widths.len()
    }

    /// Whether the prepared text is empty.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Vec::is_empty() is not const
    pub fn is_empty(&self) -> bool {
        self.data.widths.is_empty()
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
    pub segments: Vec<String>,
}

/// Cursor position within prepared text -- identifies a specific
/// point between/within segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayoutCursor {
    /// Index of the current segment.
    pub segment_index: usize,
    /// Index of the current grapheme within the segment (for mid-word breaks).
    pub grapheme_index: usize,
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
}

/// Options for `prepare()` / `prepare_with_segments()`.
#[derive(Debug, Clone, Default)]
pub struct PrepareOptions {
    /// Whitespace handling mode. Default: Normal (collapse + wrap).
    pub white_space: WhiteSpaceMode,
    /// Engine profile for browser-specific behavior. Default: native.
    pub profile: Option<EngineProfile>,
}
