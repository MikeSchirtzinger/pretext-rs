//! Fixed-width measurement backend.
//!
//! Assigns a deterministic width to each character based on font size.
//! Useful for:
//! - Testing (predictable, reproducible results)
//! - Server-side height estimation (no font files needed)
//! - Fallback when no font backend is available
//!
//! Width model: each character gets `font_size * char_width_factor`.
//! CJK characters get `cjk_width_factor` (fullwidth). Spaces get `space_width_factor`.

use unicode_segmentation::UnicodeSegmentation;

use super::{FontSpec, MeasureBackend, SegmentMetrics};
use crate::unicode;

/// Fixed-width measurement backend with configurable character width.
#[derive(Debug, Clone)]
pub struct FixedWidthBackend {
    /// Base width factor per character (multiplied by font size).
    /// Default: 0.6 (approximates average Latin character width).
    pub char_width_factor: f64,
    /// CJK width factor (multiplied by font size).
    /// Default: 1.0 (fullwidth characters).
    pub cjk_width_factor: f64,
    /// Space width factor.
    /// Default: 0.25.
    pub space_width_factor: f64,
    /// Default font size if none parseable from `FontSpec`.
    pub default_font_size: f64,
}

impl Default for FixedWidthBackend {
    fn default() -> Self {
        Self {
            char_width_factor: 0.6,
            cjk_width_factor: 1.0,
            space_width_factor: 0.25,
            default_font_size: 16.0,
        }
    }
}

impl FixedWidthBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific character width factor.
    #[must_use]
    pub const fn with_char_width(mut self, factor: f64) -> Self {
        self.char_width_factor = factor;
        self
    }

    /// Create a monospace backend where all characters have the same width.
    #[must_use]
    pub const fn monospace(font_size: f64) -> Self {
        Self {
            char_width_factor: 0.6,
            cjk_width_factor: 0.6, // Same as Latin in monospace
            space_width_factor: 0.6,
            default_font_size: font_size,
        }
    }

    fn font_size(&self, font: &FontSpec) -> f64 {
        font.parse_size().unwrap_or(self.default_font_size)
    }

    fn char_width(&self, c: char, font_size: f64) -> f64 {
        if unicode::is_cjk(c) || unicode::is_emoji(c) {
            // CJK and emoji are typically fullwidth
            font_size * self.cjk_width_factor
        } else if c == ' ' {
            font_size * self.space_width_factor
        } else {
            font_size * self.char_width_factor
        }
    }
}

impl MeasureBackend for FixedWidthBackend {
    fn measure_segment(&self, text: &str, font: &FontSpec) -> SegmentMetrics {
        let font_size = self.font_size(font);
        let mut total_width = 0.0;
        let mut contains_cjk = false;
        let mut emoji_count = 0;

        let graphemes: Vec<&str> = text.graphemes(true).collect();
        let mut grapheme_widths = Vec::with_capacity(graphemes.len());

        for grapheme in &graphemes {
            let c = grapheme.chars().next().unwrap();
            let w = self.char_width(c, font_size);

            if unicode::is_cjk(c) {
                contains_cjk = true;
            }
            if unicode::is_emoji(c) {
                emoji_count += 1;
            }

            grapheme_widths.push(w);
            total_width += w;
        }

        // Only provide grapheme widths for multi-grapheme segments
        let grapheme_widths = if grapheme_widths.len() > 1 {
            Some(grapheme_widths)
        } else {
            None
        };

        SegmentMetrics {
            width: total_width,
            contains_cjk,
            emoji_count,
            grapheme_widths,
        }
    }

    fn measure_space_width(&self, font: &FontSpec) -> f64 {
        let font_size = self.font_size(font);
        font_size * self.space_width_factor
    }

    fn measure_hyphen_width(&self, font: &FontSpec) -> f64 {
        let font_size = self.font_size(font);
        font_size * self.char_width_factor // Hyphen ~ regular character
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_width_basic() {
        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("16px Inter");
        let metrics = backend.measure_segment("hello", &font);
        // 5 chars * 16px * 0.6 = 48.0
        assert!((metrics.width - 48.0).abs() < 0.001);
        assert!(!metrics.contains_cjk);
    }

    #[test]
    fn test_fixed_width_cjk() {
        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("16px Noto Sans");
        let metrics = backend.measure_segment("\u{65E5}\u{672C}", &font);
        // 2 CJK chars * 16px * 1.0 = 32.0
        assert!((metrics.width - 32.0).abs() < 0.001);
        assert!(metrics.contains_cjk);
    }

    #[test]
    fn test_font_size_parsing() {
        let font = FontSpec::new("14px monospace");
        assert!((font.parse_size().unwrap() - 14.0).abs() < 0.001);

        let font = FontSpec::new("12pt serif");
        assert!((font.parse_size().unwrap() - 16.0).abs() < 0.001);
    }

    #[test]
    fn test_monospace() {
        let backend = FixedWidthBackend::monospace(16.0);
        let font = FontSpec::new("16px monospace");
        let space = backend.measure_space_width(&font);
        let hyphen = backend.measure_hyphen_width(&font);
        assert!((space - hyphen).abs() < 0.001); // Same width in monospace
    }

    #[test]
    fn test_grapheme_widths() {
        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("16px Inter");
        let metrics = backend.measure_segment("hello", &font);
        assert!(metrics.grapheme_widths.is_some());
        let gw = metrics.grapheme_widths.unwrap();
        assert_eq!(gw.len(), 5);
        assert!(gw.iter().all(|w| (*w - 9.6).abs() < 0.001));
    }
}
