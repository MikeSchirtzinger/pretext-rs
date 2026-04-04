/// Fontdue measurement backend (native).
///
/// Uses the `fontdue` crate for font rasterization and glyph advance
/// width queries. Works without a browser — ideal for server-side
/// layout pre-computation in Brevity's ag-ui-server.
///
/// Requires the `fontdue` feature flag.
#![cfg(feature = "fontdue")]

use std::collections::HashMap;

use unicode_segmentation::UnicodeSegmentation;

use super::{FontSpec, MeasureBackend, SegmentMetrics};
use crate::unicode;

/// Fontdue-based measurement backend.
pub struct FontdueBackend {
    fonts: HashMap<String, fontdue::Font>,
    default_font: Option<fontdue::Font>,
}

impl FontdueBackend {
    /// Create a new fontdue backend with no fonts loaded.
    pub fn new() -> Self {
        Self {
            fonts: HashMap::new(),
            default_font: None,
        }
    }

    /// Load a font from bytes and register it under a family name.
    pub fn load_font(&mut self, family: &str, data: &[u8]) -> Result<(), String> {
        let settings = fontdue::FontSettings::default();
        let font = fontdue::Font::from_bytes(data, settings)
            .map_err(|e| format!("failed to load font '{}': {}", family, e))?;
        self.fonts.insert(family.to_string(), font);
        Ok(())
    }

    /// Set the default font (used when family name isn't found).
    pub fn set_default_font(&mut self, data: &[u8]) -> Result<(), String> {
        let settings = fontdue::FontSettings::default();
        let font = fontdue::Font::from_bytes(data, settings)
            .map_err(|e| format!("failed to load default font: {}", e))?;
        self.default_font = Some(font);
        Ok(())
    }

    fn get_font(&self, font_spec: &FontSpec) -> Option<&fontdue::Font> {
        let family = font_spec.parse_family();
        self.fonts
            .get(family)
            .or(self.default_font.as_ref())
    }

    fn measure_char(&self, c: char, font: &fontdue::Font, size: f64) -> f64 {
        let metrics = font.metrics(c, size as f32);
        metrics.advance_width as f64
    }
}

impl MeasureBackend for FontdueBackend {
    fn measure_segment(&self, text: &str, font_spec: &FontSpec) -> SegmentMetrics {
        let size = font_spec.parse_size().unwrap_or(16.0);
        let font = match self.get_font(font_spec) {
            Some(f) => f,
            None => {
                // No font available — fall back to size-based estimate
                let char_count = text.graphemes(true).count();
                return SegmentMetrics {
                    width: char_count as f64 * size * 0.6,
                    contains_cjk: text.chars().any(unicode::is_cjk),
                    emoji_count: text.chars().filter(|c| unicode::is_emoji(*c)).count(),
                    grapheme_widths: None,
                };
            }
        };

        let mut total_width = 0.0;
        let mut contains_cjk = false;
        let mut emoji_count = 0;

        let graphemes: Vec<&str> = text.graphemes(true).collect();
        let mut grapheme_widths = Vec::with_capacity(graphemes.len());

        for grapheme in &graphemes {
            let mut gw = 0.0;
            for c in grapheme.chars() {
                if unicode::is_cjk(c) {
                    contains_cjk = true;
                }
                if unicode::is_emoji(c) {
                    emoji_count += 1;
                }
                gw += self.measure_char(c, font, size);
            }
            grapheme_widths.push(gw);
            total_width += gw;
        }

        SegmentMetrics {
            width: total_width,
            contains_cjk,
            emoji_count,
            grapheme_widths: if grapheme_widths.len() > 1 {
                Some(grapheme_widths)
            } else {
                None
            },
        }
    }

    fn measure_space_width(&self, font_spec: &FontSpec) -> f64 {
        let size = font_spec.parse_size().unwrap_or(16.0);
        match self.get_font(font_spec) {
            Some(font) => self.measure_char(' ', font, size),
            None => size * 0.25,
        }
    }

    fn measure_hyphen_width(&self, font_spec: &FontSpec) -> f64 {
        let size = font_spec.parse_size().unwrap_or(16.0);
        match self.get_font(font_spec) {
            Some(font) => self.measure_char('-', font, size),
            None => size * 0.6,
        }
    }
}

impl Default for FontdueBackend {
    fn default() -> Self {
        Self::new()
    }
}
