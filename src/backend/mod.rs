//! Measurement backends for text segment width computation.
//!
//! The core abstraction: [`MeasureBackend`] provides font metrics to the
//! prepare phase. The line-breaking engine never touches this -- it only
//! sees the pre-computed widths.
//!
//! Three backends:
//! - [`fixed::FixedWidthBackend`]: deterministic, for testing and server-side estimates
//! - `canvas` (feature = "wasm"): browser `canvas.measureText`, pixel-accurate
//! - `skrifa_backend` (feature = `skrifa-nominal`): explicit unshaped native
//!   advance estimation

pub mod fixed;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub mod canvas;
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub use canvas::CanvasBackend;

#[cfg(feature = "skrifa-nominal")]
pub mod skrifa_backend;
#[cfg(feature = "skrifa-nominal")]
pub use skrifa_backend::SkrifaNominalBackend;

use crate::{Error, Result};

/// Parsed and validated font specification for measurement.
///
/// The accepted syntax is a practical subset of the CSS `font` shorthand:
/// optional style/weight tokens, a required positive `px` or `pt` size,
/// an optional `/line-height`, and a required family. The first family in a
/// comma-separated fallback list is used by native backends, while
/// [`Self::as_css_str`] preserves the complete expression for canvas.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FontSpec {
    css: String,
    size_px_bits: u64,
    family: String,
    has_generic_family: bool,
}

impl FontSpec {
    /// Parse and validate a CSS-style font specification.
    ///
    /// Examples include `"16px Inter"`, `"14px/1.5 monospace"`, and
    /// `"italic 12pt 'Noto Sans'"`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFontSpec`] when the specification has no
    /// supported size or family, contains malformed quoting, or resolves to a
    /// non-finite or non-positive pixel size.
    pub fn new(font: impl Into<String>) -> Result<Self> {
        let input = font.into();
        let css = input.trim();
        if css.is_empty() {
            return Err(invalid_font_spec(&input, "font specification is empty"));
        }

        let mut search_from = 0;
        let mut parsed_size = None;
        let mut seen_prefixes = [false; 5];
        for token in css.split_whitespace() {
            let Some(search_slice) = css.get(search_from..) else {
                return Err(invalid_font_spec(
                    css,
                    "font token offset was not a UTF-8 boundary",
                ));
            };
            let Some(relative_start) = search_slice.find(token) else {
                return Err(invalid_font_spec(
                    css,
                    "could not locate a parsed font token",
                ));
            };
            let token_start = search_from + relative_start;
            let token_end = token_start + token.len();
            search_from = token_end;

            if let Some(size_result) = parse_size_token(token, css) {
                let (size_px, inline_line_height) = size_result?;
                parsed_size = Some((size_px, token_end, inline_line_height));
                break;
            }

            let prefix_kind = classify_font_prefix(token).ok_or_else(|| {
                invalid_font_spec(
                    css,
                    format!("unsupported token before font size: {token:?}"),
                )
            })?;
            let slot = prefix_kind as usize;
            if seen_prefixes.get(slot).copied().unwrap_or(true) {
                return Err(invalid_font_spec(
                    css,
                    format!("duplicate font prefix category at {token:?}"),
                ));
            }
            if let Some(seen) = seen_prefixes.get_mut(slot) {
                *seen = true;
            }
        }

        let Some((size_px, size_end, inline_line_height)) = parsed_size else {
            return Err(invalid_font_spec(
                css,
                "expected a size using px or pt units",
            ));
        };

        let Some(mut family_source) = css.get(size_end..).map(str::trim_start) else {
            return Err(invalid_font_spec(
                css,
                "font family offset was not a UTF-8 boundary",
            ));
        };
        if !inline_line_height && let Some(after_slash) = family_source.strip_prefix('/') {
            let after_slash = after_slash.trim_start();
            let Some(line_height_end) = after_slash.find(char::is_whitespace) else {
                return Err(invalid_font_spec(css, "font family is missing"));
            };
            let Some(line_height) = after_slash.get(..line_height_end) else {
                return Err(invalid_font_spec(css, "invalid line-height boundary"));
            };
            validate_line_height(line_height, css)?;
            let Some(family) = after_slash.get(line_height_end..) else {
                return Err(invalid_font_spec(css, "invalid font-family boundary"));
            };
            family_source = family.trim_start();
        }

        let (family, has_generic_family) = parse_font_families(family_source, css)?;

        Ok(Self {
            css: css.to_owned(),
            size_px_bits: size_px.to_bits(),
            family,
            has_generic_family,
        })
    }

    /// Complete validated CSS expression used by browser canvas backends.
    #[must_use]
    pub fn as_css_str(&self) -> &str {
        &self.css
    }

    /// Validated font size in CSS pixels.
    #[must_use]
    pub fn size_px(&self) -> f64 {
        f64::from_bits(self.size_px_bits)
    }

    /// Primary font family, with matching surrounding quotes removed.
    #[must_use]
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Whether the family list explicitly contains a CSS generic family.
    ///
    /// Browser canvas measurement requires this acknowledgement because CSS
    /// falls back when a named face is unavailable. Native backends may use a
    /// concrete family without a generic fallback.
    #[must_use]
    pub const fn has_generic_family(&self) -> bool {
        self.has_generic_family
    }
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum FontPrefixKind {
    Normal = 0,
    Style = 1,
    Variant = 2,
    Weight = 3,
    Stretch = 4,
}

fn classify_font_prefix(token: &str) -> Option<FontPrefixKind> {
    let normalized = token.to_ascii_lowercase();
    match normalized.as_str() {
        "normal" => Some(FontPrefixKind::Normal),
        "italic" | "oblique" => Some(FontPrefixKind::Style),
        "small-caps" => Some(FontPrefixKind::Variant),
        "bold" | "bolder" | "lighter" => Some(FontPrefixKind::Weight),
        "ultra-condensed" | "extra-condensed" | "condensed" | "semi-condensed"
        | "semi-expanded" | "expanded" | "extra-expanded" | "ultra-expanded" => {
            Some(FontPrefixKind::Stretch)
        }
        _ => {
            if normalized
                .parse::<u16>()
                .is_ok_and(|weight| (1..=1000).contains(&weight))
            {
                Some(FontPrefixKind::Weight)
            } else if normalized
                .strip_suffix('%')
                .and_then(|value| value.parse::<f64>().ok())
                .is_some_and(|stretch| stretch.is_finite() && (50.0..=200.0).contains(&stretch))
            {
                Some(FontPrefixKind::Stretch)
            } else {
                None
            }
        }
    }
}

fn parse_size_token(token: &str, spec: &str) -> Option<Result<(f64, bool)>> {
    let (size_component, line_height) = match token.split_once('/') {
        Some((size, line_height)) => {
            if let Err(error) = validate_line_height(line_height, spec) {
                return Some(Err(error));
            }
            (size, true)
        }
        None => (token, false),
    };

    if size_component.len() < 2 {
        return None;
    }
    let normalized = size_component.to_ascii_lowercase();
    let (number, scale) = if let Some(number) = normalized.strip_suffix("px") {
        (number, 1.0)
    } else if let Some(number) = normalized.strip_suffix("pt") {
        (number, 4.0 / 3.0)
    } else {
        return None;
    };

    let Ok(value) = number.parse::<f64>() else {
        return Some(Err(invalid_font_spec(spec, "font size is not a number")));
    };
    let size_px = value * scale;
    if !size_px.is_finite() || size_px <= 0.0 {
        return Some(Err(invalid_font_spec(
            spec,
            "font size must be finite and positive",
        )));
    }

    Some(Ok((size_px, line_height)))
}

fn validate_line_height(line_height: &str, spec: &str) -> Result<()> {
    let value = line_height.trim().to_ascii_lowercase();
    if value.is_empty() {
        return Err(invalid_font_spec(spec, "line height is missing after '/'"));
    }
    if value == "normal" {
        return Ok(());
    }

    let number = ["rem", "px", "pt", "em", "%"]
        .iter()
        .find_map(|suffix| value.strip_suffix(suffix))
        .unwrap_or(&value);
    let parsed = number
        .parse::<f64>()
        .map_err(|_| invalid_font_spec(spec, "line height is not a supported number"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(invalid_font_spec(
            spec,
            "line height must be finite and positive",
        ));
    }
    Ok(())
}

fn parse_font_families(family_source: &str, spec: &str) -> Result<(String, bool)> {
    let family_source = family_source.trim();
    if family_source.is_empty() {
        return Err(invalid_font_spec(spec, "font family is missing"));
    }

    let mut quote = None;
    let mut escaped = false;
    let mut component_start = 0;
    let mut primary = None;
    let mut has_generic_family = false;
    for (index, ch) in family_source.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && quote.is_some() {
            escaped = true;
            continue;
        }
        match quote {
            Some(active) if ch == active => quote = None,
            Some(_) => {}
            None if ch == '\'' || ch == '"' => quote = Some(ch),
            None if ch == ',' => {
                let component = family_source.get(component_start..index).ok_or_else(|| {
                    invalid_font_spec(spec, "font family boundary was not valid UTF-8")
                })?;
                let (parsed, quoted) = parse_family_component(component, spec)?;
                has_generic_family |= !quoted && is_generic_family(&parsed);
                if primary.is_none() {
                    primary = Some(parsed);
                }
                component_start = index + ch.len_utf8();
            }
            None => {}
        }
    }
    if quote.is_some() {
        return Err(invalid_font_spec(spec, "font family quote is not closed"));
    }

    let tail = family_source
        .get(component_start..)
        .ok_or_else(|| invalid_font_spec(spec, "font family boundary was not valid UTF-8"))?;
    let (parsed_tail, quoted) = parse_family_component(tail, spec)?;
    has_generic_family |= !quoted && is_generic_family(&parsed_tail);
    Ok((primary.unwrap_or(parsed_tail), has_generic_family))
}

fn is_generic_family(family: &str) -> bool {
    matches!(
        family.to_ascii_lowercase().as_str(),
        "serif"
            | "sans-serif"
            | "monospace"
            | "cursive"
            | "fantasy"
            | "system-ui"
            | "ui-serif"
            | "ui-sans-serif"
            | "ui-monospace"
            | "ui-rounded"
            | "emoji"
            | "math"
            | "fangsong"
    )
}

fn parse_family_component(component: &str, spec: &str) -> Result<(String, bool)> {
    let component = component.trim();
    if component.is_empty() {
        return Err(invalid_font_spec(spec, "font family is empty"));
    }

    let first = component.chars().next();
    let last = component.chars().next_back();
    let (unquoted, was_quoted) = match (first, last) {
        (Some(open), Some(close)) if matches!(open, '\'' | '"') && open == close => (
            component
                .get(open.len_utf8()..component.len().saturating_sub(close.len_utf8()))
                .ok_or_else(|| invalid_font_spec(spec, "invalid quoted family boundary"))?,
            true,
        ),
        (Some('\'' | '"'), _) | (_, Some('\'' | '"')) => {
            return Err(invalid_font_spec(spec, "font family has mismatched quotes"));
        }
        _ if component.contains(['\'', '"']) => {
            return Err(invalid_font_spec(
                spec,
                "quotes must surround a complete font family",
            ));
        }
        _ => (component, false),
    };

    if unquoted.trim().is_empty() {
        return Err(invalid_font_spec(spec, "font family is empty"));
    }
    Ok((unquoted.trim().to_owned(), was_quoted))
}

fn invalid_font_spec(spec: &str, reason: impl Into<String>) -> Error {
    Error::InvalidFontSpec {
        spec: spec.to_owned(),
        reason: reason.into(),
    }
}

pub(crate) fn validate_metric(metric: &'static str, value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(Error::InvalidMetric { metric, value })
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
/// Implementations provide width measurements for text segments. The prepare
/// phase caches successful results; errors are returned to the caller without
/// substituting estimated or zero-width values.
pub trait MeasureBackend {
    /// Measure a text segment and return its metrics.
    ///
    /// # Errors
    ///
    /// Returns a backend or metric-validation error when real measurement
    /// cannot be completed.
    fn measure_segment(&self, text: &str, font: &FontSpec) -> Result<SegmentMetrics>;

    /// Measure the width of a single space character.
    ///
    /// # Errors
    ///
    /// Returns a backend or metric-validation error when real measurement
    /// cannot be completed.
    fn measure_space_width(&self, font: &FontSpec) -> Result<f64>;

    /// Measure the width of a hyphen character (for soft-hyphen rendering).
    ///
    /// # Errors
    ///
    /// Returns a backend or metric-validation error when real measurement
    /// cannot be completed.
    fn measure_hyphen_width(&self, font: &FontSpec) -> Result<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documented_size_and_line_height() {
        let font = FontSpec::new("14px/1.5 monospace").expect("valid font");
        assert_eq!(font.as_css_str(), "14px/1.5 monospace");
        assert!((font.size_px() - 14.0).abs() < f64::EPSILON);
        assert_eq!(font.family(), "monospace");
        assert!(font.has_generic_family());
    }

    #[test]
    fn converts_points_and_unquotes_primary_family() {
        let font = FontSpec::new("italic 12pt 'Noto Sans', serif").expect("valid font");
        assert!((font.size_px() - 16.0).abs() < f64::EPSILON);
        assert_eq!(font.family(), "Noto Sans");
        assert!(font.has_generic_family());
    }

    #[test]
    fn parses_spaced_line_height() {
        let font = FontSpec::new("14px / 1.5 Noto Sans").expect("valid font");
        assert_eq!(font.family(), "Noto Sans");
    }

    #[test]
    fn rejects_invalid_sizes_and_families() {
        for invalid in [
            "Inter",
            "NaNpx Inter",
            "0px Inter",
            "-1pt Inter",
            "16px",
            "16px/nope Inter",
            "16px/0 Inter",
            "16px / NaN Inter",
            "éx 16px Inter",
            "bold bolder 16px Inter",
            "16px Inter,",
            "16px Inter, 'Noto Sans",
            "16px No'to",
        ] {
            assert!(
                matches!(FontSpec::new(invalid), Err(Error::InvalidFontSpec { .. })),
                "accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn rejects_unclosed_family_quote() {
        assert!(matches!(
            FontSpec::new("16px 'Noto Sans"),
            Err(Error::InvalidFontSpec { .. })
        ));
    }

    #[test]
    fn unicode_tokens_before_size_return_an_error_without_panicking() {
        assert!(matches!(
            FontSpec::new("éx 16px Inter"),
            Err(Error::InvalidFontSpec { .. })
        ));
    }

    #[test]
    fn records_explicit_generic_fallback_without_requiring_one_for_native_use() {
        let named_only = FontSpec::new("16px Inter").expect("valid native font spec");
        let with_fallback =
            FontSpec::new("16px Inter, sans-serif").expect("valid canvas font spec");
        let quoted_generic =
            FontSpec::new("16px Inter, 'sans-serif'").expect("valid named family spec");

        assert!(!named_only.has_generic_family());
        assert!(with_fallback.has_generic_family());
        assert!(!quoted_generic.has_generic_family());
    }
}
