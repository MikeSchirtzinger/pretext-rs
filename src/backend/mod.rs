//! Measurement backends for text segment width computation.
//!
//! The core abstraction: [`MeasureBackend`] provides font metrics to the
//! prepare phase. The line-breaking engine never touches this -- it only
//! sees the pre-computed widths.
//!
//! Three backends:
//! - [`fixed::FixedWidthBackend`]: deterministic, for testing and server-side estimates
//! - `canvas` (feature = "wasm"): browser `canvas.measureText`, pixel-accurate
//! - `fontdue` (feature = "fontdue"): native font rasterizer, accurate without browser

pub mod fixed;

#[cfg(feature = "wasm")]
pub mod canvas;

#[cfg(feature = "fontdue")]
pub mod fontdue_backend;

/// Font specification for measurement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FontSpec {
    /// CSS-style font string (e.g., "16px Inter", "14px/1.5 monospace").
    /// For native backends, this is parsed into family + size.
    pub font: String,
}

impl FontSpec {
    #[must_use]
    pub fn new(font: impl Into<String>) -> Self {
        Self { font: font.into() }
    }

    /// Parse font size from a CSS-style font string.
    /// Returns the numeric size in pixels, or `None` if unparseable.
    #[must_use]
    pub fn parse_size(&self) -> Option<f64> {
        // Simple parser: look for "Npx" pattern
        for part in self.font.split_whitespace() {
            if let Some(stripped) = part.strip_suffix("px")
                && let Ok(size) = stripped.parse::<f64>()
            {
                return Some(size);
            }
            if let Some(stripped) = part.strip_suffix("pt")
                && let Ok(size) = stripped.parse::<f64>()
            {
                return Some(size * 4.0 / 3.0); // pt to px
            }
        }
        None
    }

    /// Parse font family from a CSS-style font string.
    #[must_use]
    pub fn parse_family(&self) -> &str {
        // Take everything after the size
        let parts: Vec<&str> = self.font.split_whitespace().collect();
        if parts.len() > 1 {
            // Skip size part(s)
            for (i, part) in parts.iter().enumerate() {
                if (part.ends_with("px") || part.ends_with("pt") || part.ends_with("em"))
                    && i + 1 < parts.len()
                {
                    // Return from family name onward
                    let start = self.font.find(parts[i + 1]).unwrap_or(0);
                    return &self.font[start..];
                }
            }
        }
        &self.font
    }
}

/// Metrics for a single measured segment.
#[derive(Debug, Clone)]
pub struct SegmentMetrics {
    /// Total width of the segment.
    pub width: f64,
    /// Whether the segment contains CJK characters.
    pub contains_cjk: bool,
    /// Number of emoji in the segment (for correction).
    pub emoji_count: usize,
    /// Per-grapheme widths for breakable segments (`overflow-wrap: break-word`).
    /// `None` if the segment doesn't need grapheme-level breaking.
    pub grapheme_widths: Option<Vec<f64>>,
}

/// Trait for text measurement backends.
///
/// Implementations provide width measurements for text segments.
/// The prepare phase calls `measure_segment` for each segment;
/// the resulting widths are cached in `PreparedData` and the
/// backend is never touched again during layout.
pub trait MeasureBackend {
    /// Measure a text segment and return its metrics.
    fn measure_segment(&self, text: &str, font: &FontSpec) -> SegmentMetrics;

    /// Measure the width of a single space character.
    fn measure_space_width(&self, font: &FontSpec) -> f64;

    /// Measure the width of a hyphen character (for soft-hyphen rendering).
    fn measure_hyphen_width(&self, font: &FontSpec) -> f64;
}
