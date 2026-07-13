//! Nominal Skrifa advance backend for native estimation.
//!
//! This backend owns validated font bytes and uses [`skrifa`] for Unicode
//! character mapping and scaled per-codepoint advance widths. It does not shape
//! text, apply kerning, form ligatures, or position complex-script clusters.
//! The type name makes that limitation explicit: do not use it when line
//! breaks must match rendered production typography. Missing fonts, glyphs,
//! and unusable font data are reported as errors rather than substituted.
//! Face selection uses only [`super::FontSpec::family`] and size. CSS
//! style/weight/stretch tokens and later fallback families do not select a
//! different registered face; callers must register and name the exact face
//! whose nominal advances they intend to estimate.

use std::collections::HashMap;

use skrifa::{
    GlyphId,
    charmap::Charmap,
    metrics::GlyphMetrics,
    prelude::{FontRef, LocationRef, MetadataProvider, Size},
};
use unicode_segmentation::UnicodeSegmentation;

use super::{FontSpec, MeasureBackend, SegmentMetrics, validate_metric};
use crate::{Error, Result, unicode};

/// Native font measurement backed by Skrifa.
///
/// Font bytes are copied into the backend when loaded. This keeps measurement
/// independent of the caller's input buffer and avoids self-referential font
/// parser state; Skrifa's cheap borrowed views are reconstructed per request.
/// Selection is by primary family name only, not CSS style/weight/stretch.
pub struct SkrifaNominalBackend {
    fonts: HashMap<String, Vec<u8>>,
    default_font: Option<Vec<u8>>,
}

impl SkrifaNominalBackend {
    /// Create a backend with no fonts loaded.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fonts: HashMap::new(),
            default_font: None,
        }
    }

    /// Load the first face from a TTF, OTF, or TTC file under a family name.
    ///
    /// The backend owns a copy of `data` after this method returns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] for an empty family or
    /// [`Error::Measurement`] when Skrifa rejects the data or the selected face
    /// lacks the character map and horizontal metrics required for measurement.
    pub fn load_font(&mut self, family: &str, data: &[u8]) -> Result<()> {
        let family = family.trim();
        if family.is_empty() {
            return Err(Error::invalid_input("family", "font family is empty"));
        }
        Self::validate_font(data, &format!("font {family:?}"))?;
        self.fonts.insert(family.to_owned(), data.to_vec());
        Ok(())
    }

    /// Set the first face from a TTF, OTF, or TTC file as the fallback font.
    ///
    /// The backend owns a copy of `data` after this method returns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Measurement`] when Skrifa rejects the data or the
    /// selected face lacks the character map and horizontal metrics required
    /// for measurement.
    pub fn set_default_font(&mut self, data: &[u8]) -> Result<()> {
        Self::validate_font(data, "default font")?;
        self.default_font = Some(data.to_vec());
        Ok(())
    }

    fn parse_font<'a>(data: &'a [u8], context: &str) -> Result<FontRef<'a>> {
        FontRef::from_index(data, 0).map_err(|error| {
            Error::measurement("skrifa", format!("failed to parse {context}: {error}"))
        })
    }

    fn validate_font(data: &[u8], context: &str) -> Result<()> {
        let font = Self::parse_font(data, context)?;
        if !font.charmap().has_map() {
            return Err(Error::measurement(
                "skrifa",
                format!("{context} has no supported character map"),
            ));
        }

        let global_metrics = font.metrics(Size::unscaled(), LocationRef::default());
        if global_metrics.units_per_em == 0 {
            return Err(Error::measurement(
                "skrifa",
                format!("{context} has no valid units-per-em value"),
            ));
        }

        let glyph_metrics = font.glyph_metrics(Size::unscaled(), LocationRef::default());
        if glyph_metrics.glyph_count() == 0
            || glyph_metrics.advance_width(GlyphId::NOTDEF).is_none()
        {
            return Err(Error::measurement(
                "skrifa",
                format!("{context} has no usable horizontal glyph metrics"),
            ));
        }
        Ok(())
    }

    fn get_font_data(&self, font_spec: &FontSpec) -> Result<&[u8]> {
        self.fonts
            .get(font_spec.family())
            .map(Vec::as_slice)
            .or(self.default_font.as_deref())
            .ok_or_else(|| Error::MissingFont {
                family: font_spec.family().to_owned(),
            })
    }

    fn font_size(font_spec: &FontSpec) -> Result<f32> {
        let size = font_spec.size_px() as f32;
        if !size.is_finite() || size <= 0.0 {
            return Err(Error::invalid_input(
                "font size",
                "skrifa requires a positive size representable as f32",
            ));
        }
        Ok(size)
    }

    fn measure_char(
        character: char,
        family: &str,
        charmap: &Charmap<'_>,
        metrics: &GlyphMetrics<'_>,
    ) -> Result<f64> {
        if matches!(character, '\u{2060}' | '\u{FEFF}') {
            return Ok(0.0);
        }
        let glyph_id = charmap.map(character).ok_or_else(|| Error::MissingGlyph {
            character,
            family: family.to_owned(),
        })?;
        let Some(advance_width) = metrics.advance_width(glyph_id) else {
            return Err(Error::measurement(
                "skrifa",
                format!(
                    "font has no advance width for U+{:04X} (glyph {})",
                    character as u32,
                    glyph_id.to_u32()
                ),
            ));
        };
        validate_metric("skrifa glyph advance", f64::from(advance_width))
    }
}

impl MeasureBackend for SkrifaNominalBackend {
    fn measure_segment(&self, text: &str, font_spec: &FontSpec) -> Result<SegmentMetrics> {
        let size = Self::font_size(font_spec)?;
        let font = Self::parse_font(self.get_font_data(font_spec)?, "loaded font")?;
        let charmap = font.charmap();
        let glyph_metrics = font.glyph_metrics(Size::new(size), LocationRef::default());

        let graphemes: Vec<&str> = text.graphemes(true).collect();
        let mut grapheme_widths = Vec::with_capacity(graphemes.len());
        let mut total_width = 0.0;
        let mut contains_cjk = false;
        let mut emoji_count = 0;

        for grapheme in &graphemes {
            let mut grapheme_width = 0.0;
            for character in grapheme.chars() {
                contains_cjk |= unicode::is_cjk(character);
                if unicode::is_emoji(character) {
                    emoji_count += 1;
                }
                grapheme_width +=
                    Self::measure_char(character, font_spec.family(), &charmap, &glyph_metrics)?;
                validate_metric("skrifa grapheme width", grapheme_width)?;
            }
            grapheme_widths.push(grapheme_width);
            total_width += grapheme_width;
            validate_metric("skrifa segment width", total_width)?;
        }

        Ok(SegmentMetrics {
            width: total_width,
            contains_cjk,
            emoji_count,
            grapheme_widths: (grapheme_widths.len() > 1).then_some(grapheme_widths),
        })
    }

    fn measure_space_width(&self, font_spec: &FontSpec) -> Result<f64> {
        let size = Self::font_size(font_spec)?;
        let font = Self::parse_font(self.get_font_data(font_spec)?, "loaded font")?;
        Self::measure_char(
            ' ',
            font_spec.family(),
            &font.charmap(),
            &font.glyph_metrics(Size::new(size), LocationRef::default()),
        )
    }

    fn measure_hyphen_width(&self, font_spec: &FontSpec) -> Result<f64> {
        let size = Self::font_size(font_spec)?;
        let font = Self::parse_font(self.get_font_data(font_spec)?, "loaded font")?;
        Self::measure_char(
            '-',
            font_spec.family(),
            &font.charmap(),
            &font.glyph_metrics(Size::new(size), LocationRef::default()),
        )
    }
}

impl Default for SkrifaNominalBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use font_test_data::TINOS_SUBSET;

    use super::*;

    #[test]
    fn missing_font_is_reported_without_estimation() {
        let backend = SkrifaNominalBackend::new();
        let font = FontSpec::new("16px Inter").expect("valid font specification");
        assert!(matches!(
            backend.measure_segment("hello", &font),
            Err(Error::MissingFont { family }) if family == "Inter"
        ));
        assert!(matches!(
            backend.measure_space_width(&font),
            Err(Error::MissingFont { .. })
        ));
    }

    #[test]
    fn invalid_font_bytes_return_typed_error() {
        let mut backend = SkrifaNominalBackend::new();
        assert!(matches!(
            backend.load_font("Inter", b"not a font"),
            Err(Error::Measurement {
                backend: "skrifa",
                ..
            })
        ));
    }

    #[test]
    fn empty_family_is_rejected() {
        let mut backend = SkrifaNominalBackend::new();
        assert!(matches!(
            backend.load_font("  ", TINOS_SUBSET),
            Err(Error::InvalidInput {
                parameter: "family",
                ..
            })
        ));
    }

    #[test]
    fn real_font_metrics_are_positive_and_scale_with_size() {
        let mut backend = SkrifaNominalBackend::new();
        backend
            .load_font("Tinos", TINOS_SUBSET)
            .expect("fontations test font should load");

        let small = FontSpec::new("10px Tinos").expect("valid font specification");
        let large = FontSpec::new("20px Tinos").expect("valid font specification");
        let small_metrics = backend
            .measure_segment("aAbB", &small)
            .expect("real glyph metrics should be available");
        let large_metrics = backend
            .measure_segment("aAbB", &large)
            .expect("real glyph metrics should be available");

        assert!(small_metrics.width > 0.0);
        assert_eq!(
            small_metrics.grapheme_widths.as_ref().map(Vec::len),
            Some(4)
        );
        assert_relative_eq!(
            large_metrics.width,
            small_metrics.width * 2.0,
            epsilon = 0.01
        );
    }

    #[test]
    fn default_font_is_used_for_an_unregistered_family() {
        let mut backend = SkrifaNominalBackend::new();
        backend
            .set_default_font(TINOS_SUBSET)
            .expect("fontations test font should load");
        let font = FontSpec::new("16px Unknown").expect("valid font specification");

        assert!(
            backend
                .measure_segment("a", &font)
                .expect("default font should provide metrics")
                .width
                > 0.0
        );
    }

    #[test]
    fn missing_codepoint_is_rejected_instead_of_measuring_notdef() {
        let mut backend = SkrifaNominalBackend::new();
        backend
            .load_font("Tinos", TINOS_SUBSET)
            .expect("fontations test font should load");
        let font = FontSpec::new("16px Tinos").expect("valid font specification");

        assert!(matches!(
            backend.measure_segment("😀", &font),
            Err(Error::MissingGlyph {
                character: '😀',
                family,
            }) if family == "Tinos"
        ));
    }

    #[test]
    fn zero_width_joiners_do_not_require_font_glyphs() {
        let mut backend = SkrifaNominalBackend::new();
        backend
            .load_font("Tinos", TINOS_SUBSET)
            .expect("fontations test font should load");
        let font = FontSpec::new("16px Tinos").expect("valid font specification");

        let metrics = backend
            .measure_segment("\u{2060}\u{FEFF}", &font)
            .expect("layout controls have semantic zero width");
        assert_eq!(metrics.width, 0.0);
    }
}
