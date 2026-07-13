//! Fixed-width measurement backend.
//!
//! Assigns a deterministic width to each character based on font size.
//! Useful for:
//! - Testing (predictable, reproducible results)
//! - Server-side height estimation (no font files needed)
//! - Explicit dependency-free estimation when selected by the caller
//!
//! Width model: each character gets `font_size * char_width_factor`.
//! CJK characters get `cjk_width_factor` (fullwidth). Spaces get `space_width_factor`.

use unicode_segmentation::UnicodeSegmentation;

use super::{FontSpec, MeasureBackend, SegmentMetrics, validate_metric};
use crate::{Error, Result, unicode};

/// Fixed-width measurement backend with configurable character width.
#[derive(Debug, Clone)]
pub struct FixedWidthBackend {
    /// Base width factor per character (multiplied by font size).
    /// Default: 0.6 (approximates average Latin character width).
    char_width_factor: f64,
    /// CJK width factor (multiplied by font size).
    /// Default: 1.0 (fullwidth characters).
    cjk_width_factor: f64,
    /// Space width factor.
    /// Default: 0.25.
    space_width_factor: f64,
    /// Narrow no-break space width factor.
    /// Default: 1/6 em.
    narrow_space_width_factor: f64,
}

impl Default for FixedWidthBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedWidthBackend {
    /// Construct a backend with the default Latin, CJK, and space factors.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            char_width_factor: 0.6,
            cjk_width_factor: 1.0,
            space_width_factor: 0.25,
            narrow_space_width_factor: 1.0 / 6.0,
        }
    }

    /// Create with a specific character width factor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `factor` is negative or non-finite.
    pub fn with_char_width(mut self, factor: f64) -> Result<Self> {
        validate_factor("char_width_factor", factor)?;
        self.char_width_factor = factor;
        Ok(self)
    }

    /// Create with a specific CJK and emoji width factor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `factor` is negative or non-finite.
    pub fn with_cjk_width(mut self, factor: f64) -> Result<Self> {
        validate_factor("cjk_width_factor", factor)?;
        self.cjk_width_factor = factor;
        Ok(self)
    }

    /// Create with a specific space width factor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `factor` is negative or non-finite.
    pub fn with_space_width(mut self, factor: f64) -> Result<Self> {
        validate_factor("space_width_factor", factor)?;
        self.space_width_factor = factor;
        Ok(self)
    }

    /// Create with a specific U+202F narrow no-break space width factor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `factor` is negative or non-finite.
    pub fn with_narrow_space_width(mut self, factor: f64) -> Result<Self> {
        validate_factor("narrow_space_width_factor", factor)?;
        self.narrow_space_width_factor = factor;
        Ok(self)
    }

    /// Create a backend where every character uses the same width factor.
    ///
    /// The actual font size always comes from [`FontSpec`].
    #[must_use]
    pub const fn monospace() -> Self {
        Self {
            char_width_factor: 0.6,
            cjk_width_factor: 0.6, // Same as Latin in monospace
            space_width_factor: 0.6,
            narrow_space_width_factor: 0.6,
        }
    }

    fn validate_config(&self) -> Result<()> {
        for (name, value) in [
            ("char_width_factor", self.char_width_factor),
            ("cjk_width_factor", self.cjk_width_factor),
            ("space_width_factor", self.space_width_factor),
            ("narrow_space_width_factor", self.narrow_space_width_factor),
        ] {
            validate_factor(name, value)?;
        }
        Ok(())
    }

    fn font_size(&self, font: &FontSpec) -> Result<f64> {
        self.validate_config()?;
        Ok(font.size_px())
    }

    fn char_width(&self, c: char, font_size: f64) -> f64 {
        if matches!(c, '\u{2060}' | '\u{FEFF}') {
            0.0
        } else if unicode::is_cjk(c) || unicode::is_emoji(c) {
            // CJK and emoji are typically fullwidth
            font_size * self.cjk_width_factor
        } else if matches!(c, ' ' | '\u{00A0}') {
            font_size * self.space_width_factor
        } else if c == '\u{202F}' {
            font_size * self.narrow_space_width_factor
        } else {
            font_size * self.char_width_factor
        }
    }
}

fn validate_factor(name: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        Err(Error::invalid_input(
            name,
            "fixed-width factors must be finite and non-negative",
        ))
    } else {
        Ok(())
    }
}

impl MeasureBackend for FixedWidthBackend {
    fn measure_segment(&self, text: &str, font: &FontSpec) -> Result<SegmentMetrics> {
        let font_size = self.font_size(font)?;
        let mut total_width = 0.0;
        let mut contains_cjk = false;
        let mut emoji_count = 0;

        let graphemes: Vec<&str> = text.graphemes(true).collect();
        let mut grapheme_widths = Vec::with_capacity(graphemes.len());

        for grapheme in &graphemes {
            let Some(c) = grapheme.chars().next() else {
                return Err(Error::measurement(
                    "fixed",
                    "unicode segmentation emitted an empty grapheme",
                ));
            };
            let w = validate_metric("fixed grapheme width", self.char_width(c, font_size))?;

            if unicode::is_cjk(c) {
                contains_cjk = true;
            }
            if unicode::is_emoji(c) {
                emoji_count += 1;
            }

            grapheme_widths.push(w);
            total_width += w;
            validate_metric("fixed segment width", total_width)?;
        }

        // Only provide grapheme widths for multi-grapheme segments
        let grapheme_widths = if grapheme_widths.len() > 1 {
            Some(grapheme_widths)
        } else {
            None
        };

        Ok(SegmentMetrics {
            width: total_width,
            contains_cjk,
            emoji_count,
            grapheme_widths,
        })
    }

    fn measure_space_width(&self, font: &FontSpec) -> Result<f64> {
        let font_size = self.font_size(font)?;
        validate_metric("fixed space width", font_size * self.space_width_factor)
    }

    fn measure_hyphen_width(&self, font: &FontSpec) -> Result<f64> {
        let font_size = self.font_size(font)?;
        validate_metric("fixed hyphen width", font_size * self.char_width_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_width_basic() {
        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("16px Inter").expect("valid font");
        let metrics = backend
            .measure_segment("hello", &font)
            .expect("measurement succeeds");
        // 5 chars * 16px * 0.6 = 48.0
        assert!((metrics.width - 48.0).abs() < 0.001);
        assert!(!metrics.contains_cjk);
    }

    #[test]
    fn test_fixed_width_cjk() {
        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("16px Noto Sans").expect("valid font");
        let metrics = backend
            .measure_segment("\u{65E5}\u{672C}", &font)
            .expect("measurement succeeds");
        // 2 CJK chars * 16px * 1.0 = 32.0
        assert!((metrics.width - 32.0).abs() < 0.001);
        assert!(metrics.contains_cjk);
    }

    #[test]
    fn test_font_size_parsing() {
        let font = FontSpec::new("14px monospace").expect("valid font");
        assert!((font.size_px() - 14.0).abs() < 0.001);

        let font = FontSpec::new("12pt serif").expect("valid font");
        assert!((font.size_px() - 16.0).abs() < 0.001);
    }

    #[test]
    fn test_monospace() {
        let backend = FixedWidthBackend::monospace();
        let font = FontSpec::new("16px monospace").expect("valid font");
        let space = backend
            .measure_space_width(&font)
            .expect("measurement succeeds");
        let hyphen = backend
            .measure_hyphen_width(&font)
            .expect("measurement succeeds");
        assert!((space - hyphen).abs() < 0.001); // Same width in monospace
    }

    #[test]
    fn test_grapheme_widths() {
        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("16px Inter").expect("valid font");
        let metrics = backend
            .measure_segment("hello", &font)
            .expect("measurement succeeds");
        assert!(metrics.grapheme_widths.is_some());
        let gw = metrics.grapheme_widths.unwrap();
        assert_eq!(gw.len(), 5);
        assert!(gw.iter().all(|w| (*w - 9.6).abs() < 0.001));
    }

    #[test]
    fn invalid_public_configuration_is_rejected() {
        assert!(matches!(
            FixedWidthBackend::new().with_char_width(f64::NAN),
            Err(Error::InvalidInput {
                parameter: "char_width_factor",
                ..
            })
        ));
    }

    #[test]
    fn overflowing_measurement_is_rejected() {
        let font = FontSpec::new("1e308px Inter").expect("valid font");
        let backend = FixedWidthBackend::new()
            .with_char_width(2.0)
            .expect("valid factor");
        assert!(matches!(
            backend.measure_segment("a", &font),
            Err(Error::InvalidMetric { .. })
        ));
    }

    #[test]
    fn no_break_controls_use_explicit_semantic_widths() {
        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("12px sans-serif").expect("valid font");

        let nbsp = backend
            .measure_segment("\u{00A0}", &font)
            .expect("NBSP measurement succeeds");
        let narrow = backend
            .measure_segment("\u{202F}", &font)
            .expect("NNBSP measurement succeeds");
        let joiners = backend
            .measure_segment("\u{2060}\u{FEFF}", &font)
            .expect("zero-width controls measure successfully");

        assert_eq!(nbsp.width, 3.0);
        assert_eq!(narrow.width, 2.0);
        assert_eq!(joiners.width, 0.0);
    }
}
