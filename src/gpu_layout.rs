//! GPU text layout -- maps pretext line-breaking output to GPU glyph instances.
//!
//! This module bridges pretext's layout engine with GPU text rendering pipelines
//! like SDF-based text systems. It converts layout results into per-glyph
//! position data ready for upload to a GPU instance buffer.
//!
//! # Example
//!
//! ```ignore
//! use pretext::gpu_layout::{GpuTextLayout, GlyphInstance, GlyphAtlas, TextLayoutConfig};
//!
//! let atlas = GlyphAtlas::ascii_sdf(1024, 64, 16);
//! let layout = GpuTextLayout::new(&atlas);
//! let instances = layout.layout_label(
//!     "Hello\nWorld",
//!     [0.0, 5.0, 0.0],
//!     0.1,
//!     [1.0, 1.0, 1.0, 1.0],
//!     TextLayoutConfig::default(),
//! );
//! ```

use crate::backend::{FontSpec, MeasureBackend};
use crate::types::{LayoutCursor, PrepareOptions, WhiteSpaceMode};
use crate::{layout_next_line, layout_with_lines, prepare, prepare_with_segments};

/// A glyph atlas descriptor -- maps characters to UV coordinates.
#[derive(Debug, Clone)]
pub struct GlyphAtlas {
    /// Atlas texture size in texels (square).
    pub atlas_size: u32,
    /// Size of each glyph cell in texels.
    pub glyph_size: u32,
    /// Number of glyph columns per atlas row.
    pub glyphs_per_row: u32,
    /// First character code in the atlas.
    pub first_char: u32,
    /// One past the last character code.
    pub last_char: u32,
}

impl GlyphAtlas {
    /// Create an atlas descriptor for an ASCII SDF atlas.
    ///
    /// Default: 1024x1024 texture, 64x64 cells, 16 glyphs per row,
    /// ASCII 32-126.
    /// # Panics
    ///
    /// Panics if `glyphs_per_row` or `atlas_size` is zero.
    #[must_use]
    pub const fn ascii_sdf(atlas_size: u32, glyph_size: u32, glyphs_per_row: u32) -> Self {
        assert!(glyphs_per_row > 0, "glyphs_per_row must be > 0");
        assert!(atlas_size > 0, "atlas_size must be > 0");
        Self {
            atlas_size,
            glyph_size,
            glyphs_per_row,
            first_char: 32,
            last_char: 127,
        }
    }

    /// Default atlas (1024x1024, 64px cells, 16 per row, ASCII).
    #[must_use]
    pub const fn default_ascii() -> Self {
        Self::ascii_sdf(1024, 64, 16)
    }

    /// Get UV coordinates for a character code.
    ///
    /// Returns `(uv_min, uv_max)` or `None` if the character is out of range.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn char_uvs(&self, code: u32) -> Option<([f32; 2], [f32; 2])> {
        if code < self.first_char || code >= self.last_char {
            return None;
        }

        let glyph_idx = code - self.first_char;
        let col = glyph_idx % self.glyphs_per_row;
        let row = glyph_idx / self.glyphs_per_row;

        let uv_cell = self.glyph_size as f32 / self.atlas_size as f32;
        let uv_min = [col as f32 * uv_cell, row as f32 * uv_cell];
        let uv_max = [uv_min[0] + uv_cell, uv_min[1] + uv_cell];

        Some((uv_min, uv_max))
    }
}

/// Per-glyph instance data for GPU rendering.
///
/// This struct uses `#[repr(C)]` for stable layout (64 bytes, `Pod + Zeroable`).
/// If your GPU pipeline uses a different layout, map from this struct.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GlyphInstance {
    /// World-space anchor position of the label.
    pub world_pos: [f32; 3],
    /// World-space height of a single glyph quad.
    pub font_size: f32,
    /// Per-glyph X/Y offset from the label anchor (in `font_size` units).
    pub glyph_offset: [f32; 2],
    /// Top-left UV coordinate of this glyph's cell in the atlas.
    pub atlas_uv_min: [f32; 2],
    /// Bottom-right UV coordinate.
    pub atlas_uv_max: [f32; 2],
    /// RGBA color (linear).
    pub color: [f32; 4],
    /// Padding to 64 bytes.
    pub pad: [f32; 2],
}

const _: () = assert!(
    std::mem::size_of::<GlyphInstance>() == 64,
    "GlyphInstance must be exactly 64 bytes"
);

/// Horizontal text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Vertical text alignment for multi-line labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerticalAlign {
    /// Anchor is at the top of the first line.
    #[default]
    Top,
    /// Anchor is at the vertical center.
    Center,
    /// Anchor is at the bottom of the last line.
    Bottom,
}

/// Configuration for GPU text layout.
#[derive(Debug, Clone)]
pub struct TextLayoutConfig {
    /// Horizontal alignment. Default: Left.
    pub align: TextAlign,
    /// Vertical alignment. Default: Top.
    pub vertical_align: VerticalAlign,
    /// Horizontal advance between adjacent glyphs (in `font_size` units).
    /// Default: 0.6 (condensed spacing).
    pub glyph_advance: f32,
    /// Vertical spacing between lines (in `font_size` units).
    /// Default: 1.4 (comfortable reading spacing).
    pub line_spacing: f32,
    /// Maximum width in world units. `None` = no wrapping (single line).
    pub max_width: Option<f32>,
}

impl Default for TextLayoutConfig {
    fn default() -> Self {
        Self {
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
            glyph_advance: 0.6,
            line_spacing: 1.4,
            max_width: None,
        }
    }
}

/// GPU text layout engine.
///
/// Takes text + world position + atlas and produces `GlyphInstance` arrays
/// ready for GPU upload.
pub struct GpuTextLayout {
    atlas: GlyphAtlas,
}

impl GpuTextLayout {
    #[must_use]
    pub fn new(atlas: &GlyphAtlas) -> Self {
        Self {
            atlas: atlas.clone(),
        }
    }

    /// Layout a single-line label (no wrapping, fast path).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn layout_single_line(
        &self,
        text: &str,
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
    ) -> Vec<GlyphInstance> {
        let printable: Vec<char> = text
            .chars()
            .filter(|c| {
                let code = *c as u32;
                code >= self.atlas.first_char && code < self.atlas.last_char
            })
            .collect();

        let text_width = printable.len() as f32 * config.glyph_advance;

        let x_start = match config.align {
            TextAlign::Left => 0.0,
            TextAlign::Center => -text_width * 0.5,
            TextAlign::Right => -text_width,
        };

        let mut instances = Vec::with_capacity(printable.len());

        for (i, &ch) in printable.iter().enumerate() {
            if let Some((uv_min, uv_max)) = self.atlas.char_uvs(ch as u32) {
                instances.push(GlyphInstance {
                    world_pos,
                    font_size,
                    glyph_offset: [(i as f32).mul_add(config.glyph_advance, x_start), 0.0],
                    atlas_uv_min: uv_min,
                    atlas_uv_max: uv_max,
                    color,
                    pad: [0.0; 2],
                });
            }
        }

        instances
    }

    /// Layout a multi-line label using pretext's line-breaking engine.
    ///
    /// This is the key integration point -- pretext computes where line
    /// breaks go, then this function maps each glyph to world-space
    /// coordinates with proper alignment.
    ///
    /// # Parameters
    /// - `text`: The text content (may contain newlines for explicit breaks)
    /// - `world_pos`: World-space anchor point
    /// - `font_size`: World-space glyph height
    /// - `color`: RGBA color
    /// - `config`: Layout configuration (alignment, `max_width`, spacing)
    /// - `backend`: Measurement backend for pretext
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn layout_multiline(
        &self,
        text: &str,
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
        backend: &dyn MeasureBackend,
    ) -> Vec<GlyphInstance> {
        let Some(max_width) = config.max_width else {
            return self.layout_single_line(text, world_pos, font_size, color, config);
        };

        // Convert world-space max_width to "character units" for pretext.
        // pretext works in pixel space; we convert back to world space after.
        let chars_per_width = max_width / (font_size * config.glyph_advance);
        let px_font_size: f64 = 16.0; // Arbitrary reference size for pretext
        let px_max_width: f64 =
            f64::from(chars_per_width) * px_font_size * f64::from(config.glyph_advance);

        let font_spec = FontSpec::new(format!("{px_font_size}px monospace"));
        let prepared = prepare(text, &font_spec, backend, PrepareOptions::default());

        // Collect lines via streaming API
        let mut lines: Vec<Vec<char>> = Vec::new();
        let mut cursor = LayoutCursor::default();

        while let Some((_range, next)) = layout_next_line(&prepared, cursor, px_max_width) {
            // Extract printable characters for this line from the original text
            let line_chars: Vec<char> = self.extract_line_chars(text, &lines);
            lines.push(line_chars);

            if next == cursor || next.segment_index >= prepared.segment_count() {
                break;
            }
            cursor = next;
        }

        // If pretext didn't produce lines (edge case), fall back to simple split
        if lines.is_empty() || lines.iter().all(Vec::is_empty) {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let max_chars = chars_per_width as usize;
            lines = self.split_lines_simple(text, max_chars);
        }

        self.emit_multiline_instances(&lines, world_pos, font_size, color, config)
    }

    /// Layout a label using pretext for line-breaking and an explicit per-glyph
    /// advance function for placement.
    ///
    /// Use this when the font is proportional (e.g. Inter) and you have real
    /// per-glyph advance widths (e.g. from `fontdue::Font::metrics`). The
    /// `layout_single_line` / `layout_multiline` paths use a single fixed
    /// `glyph_advance`, which is fine for monospace fonts but breaks proportional
    /// layout. This path uses the caller-supplied `glyph_advance(ch)` for both
    /// per-glyph stepping and line-width totals.
    ///
    /// Line-break decisions still flow through `backend` via pretext, so for
    /// consistent results the backend and `glyph_advance` callback should agree
    /// on character widths (i.e. derive them from the same font).
    ///
    /// # Convention
    ///
    /// Glyph offsets are emitted with the **glyph centered inside its advance
    /// window**: a glyph with advance `w` starting at cursor `x` is placed at
    /// `x + w/2`. This is the convention used by SDF atlases where each cell is
    /// sized to the full font height (so the visible glyph plus side-bearing
    /// padding occupies one `font_size`-wide quad). If your shader treats
    /// `glyph_offset` as the left edge of the quad, subtract `0.5` from each
    /// emitted offset.
    ///
    /// # Parameters
    ///
    /// * `text` — text content; explicit `\n` produces a hard break.
    /// * `world_pos` — world-space anchor point copied onto every emitted instance.
    /// * `font_size` — world-space height of one glyph quad.
    /// * `color` — RGBA tint copied onto every emitted instance.
    /// * `config` — alignment and `max_width` (in world units). The `glyph_advance`
    ///   field of `config` is ignored — `glyph_advance` parameter wins.
    /// * `backend` — pretext measurement backend used for line-break decisions.
    /// * `font_spec` — font spec passed to `backend`.
    /// * `pretext_font_size_px` — reference pixel size for talking to pretext.
    ///   Pretext returns line widths in these units; this value should match the
    ///   `font_spec` size so line widths come back proportional to the advance
    ///   callback's output.
    /// * `glyph_advance` — `Fn(char) -> f32`. Returns the per-glyph advance in
    ///   `font_size` units (1.0 == one `font_size` wide). Characters not in the
    ///   atlas (`GlyphAtlas::char_uvs` returns `None`) are skipped without
    ///   contributing to width — the callback is still consulted for them so that
    ///   it can return 0.0 if desired.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
    pub fn layout_label_proportional<F>(
        &self,
        text: &str,
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
        backend: &dyn MeasureBackend,
        font_spec: &FontSpec,
        pretext_font_size_px: f64,
        glyph_advance: F,
    ) -> Vec<GlyphInstance>
    where
        F: Fn(char) -> f32,
    {
        if text.is_empty() {
            return Vec::new();
        }

        // PreWrap honors explicit `\n` as a hard break, which matches how
        // multi-line labels are typically authored. (Normal mode would
        // collapse `\n` to a space, producing one line for "Hi\nWorld".)
        let prepared = prepare_with_segments(
            text,
            font_spec,
            backend,
            PrepareOptions {
                white_space: WhiteSpaceMode::PreWrap,
                profile: None,
            },
        );

        // Convert world-space max_width to pretext pixel units.
        // pretext returned widths are in `pretext_font_size_px` units; we treat
        // world width as `font_size`-units (1.0 == one font_size wide), so the
        // conversion factor is pretext_font_size_px / font_size.
        let max_width_px = match config.max_width {
            Some(w) => {
                let fs = font_size.max(f32::EPSILON);
                f64::from((w / fs).max(f32::EPSILON)) * pretext_font_size_px
            }
            None => f64::INFINITY,
        };

        let line_height_px = pretext_font_size_px * f64::from(config.line_spacing);
        let lines = layout_with_lines(&prepared, max_width_px, line_height_px);

        if lines.is_empty() {
            return Vec::new();
        }

        let line_count = lines.len();
        let total_height = line_count as f32 * config.line_spacing;

        // Anchor's Y position relative to the top line.
        let y_start = match config.vertical_align {
            VerticalAlign::Top => 0.0,
            VerticalAlign::Center => total_height * 0.5,
            VerticalAlign::Bottom => total_height,
        };

        let total_glyphs: usize = lines.iter().map(|l| l.text.chars().count()).sum();
        let mut instances = Vec::with_capacity(total_glyphs);

        for (line_idx, line) in lines.iter().enumerate() {
            // Convert line width from pretext pixel units to font_size units.
            let line_width_units = (line.width as f32 / pretext_font_size_px as f32).max(0.0);

            let mut x_cursor = match config.align {
                TextAlign::Left => 0.0,
                TextAlign::Center => -line_width_units * 0.5,
                TextAlign::Right => -line_width_units,
            };

            let y_offset = (line_idx as f32).mul_add(-config.line_spacing, y_start);

            for ch in line.text.chars() {
                let advance = glyph_advance(ch);
                if let Some((uv_min, uv_max)) = self.atlas.char_uvs(ch as u32) {
                    instances.push(GlyphInstance {
                        world_pos,
                        font_size,
                        glyph_offset: [x_cursor + advance * 0.5, y_offset],
                        atlas_uv_min: uv_min,
                        atlas_uv_max: uv_max,
                        color,
                        pad: [0.0; 2],
                    });
                }
                x_cursor += advance;
            }
        }

        instances
    }

    /// Layout multi-line text using simple character-count wrapping.
    ///
    /// This is the fast, no-measurement path. Uses a fixed character width
    /// (`glyph_advance`) to determine line breaks. Good enough for monospace
    /// fonts and world-space labels where pixel accuracy isn't critical.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn layout_multiline_simple(
        &self,
        text: &str,
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
    ) -> Vec<GlyphInstance> {
        let max_chars = match config.max_width {
            Some(w) => (w / (font_size * config.glyph_advance)).floor() as usize,
            None => return self.layout_single_line(text, world_pos, font_size, color, config),
        };

        let max_chars = max_chars.max(1);
        let lines = self.split_lines_simple(text, max_chars);

        self.emit_multiline_instances(&lines, world_pos, font_size, color, config)
    }

    /// Split text into lines by word-wrapping at a character limit.
    fn split_lines_simple(&self, text: &str, max_chars: usize) -> Vec<Vec<char>> {
        let mut lines: Vec<Vec<char>> = Vec::new();

        for paragraph in text.split('\n') {
            let words: Vec<&str> = paragraph.split_whitespace().collect();
            if words.is_empty() {
                lines.push(Vec::new());
                continue;
            }

            let mut current_line: Vec<char> = Vec::new();

            for word in &words {
                let word_chars: Vec<char> = word
                    .chars()
                    .filter(|c| {
                        let code = *c as u32;
                        code >= self.atlas.first_char && code < self.atlas.last_char
                    })
                    .collect();

                if current_line.is_empty() {
                    current_line = word_chars;
                } else if current_line.len() + 1 + word_chars.len() <= max_chars {
                    current_line.push(' ');
                    current_line.extend(word_chars);
                } else {
                    lines.push(std::mem::take(&mut current_line));
                    current_line = word_chars;
                }
            }

            if !current_line.is_empty() {
                lines.push(current_line);
            }
        }

        if lines.is_empty() {
            lines.push(Vec::new());
        }

        lines
    }

    /// Extract printable characters for the next line from text.
    fn extract_line_chars(&self, text: &str, previous_lines: &[Vec<char>]) -> Vec<char> {
        let consumed: usize = previous_lines.iter().map(Vec::len).sum::<usize>()
            + previous_lines.len().saturating_sub(1); // spaces between words

        let remaining: String = text.chars().skip(consumed).collect();

        remaining
            .chars()
            .take_while(|c| *c != '\n')
            .filter(|c| {
                let code = *c as u32;
                code >= self.atlas.first_char && code < self.atlas.last_char
            })
            .collect()
    }

    /// Emit `GlyphInstance` data for multiple lines.
    #[allow(clippy::cast_precision_loss)]
    fn emit_multiline_instances(
        &self,
        lines: &[Vec<char>],
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
    ) -> Vec<GlyphInstance> {
        let line_count = lines.len();
        let total_height = line_count as f32 * config.line_spacing;

        // Vertical offset based on alignment
        let y_start = match config.vertical_align {
            VerticalAlign::Top => 0.0,
            VerticalAlign::Center => total_height * 0.5,
            VerticalAlign::Bottom => total_height,
        };

        let total_glyphs: usize = lines.iter().map(Vec::len).sum();
        let mut instances = Vec::with_capacity(total_glyphs);

        for (line_idx, line_chars) in lines.iter().enumerate() {
            let line_width = line_chars.len() as f32 * config.glyph_advance;

            let x_start = match config.align {
                TextAlign::Left => 0.0,
                TextAlign::Center => -line_width * 0.5,
                TextAlign::Right => -line_width,
            };

            let y_offset = (line_idx as f32).mul_add(-config.line_spacing, y_start);

            for (char_idx, &ch) in line_chars.iter().enumerate() {
                if let Some((uv_min, uv_max)) = self.atlas.char_uvs(ch as u32) {
                    instances.push(GlyphInstance {
                        world_pos,
                        font_size,
                        glyph_offset: [
                            (char_idx as f32).mul_add(config.glyph_advance, x_start),
                            y_offset,
                        ],
                        atlas_uv_min: uv_min,
                        atlas_uv_max: uv_max,
                        color,
                        pad: [0.0; 2],
                    });
                }
            }
        }

        instances
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atlas() -> GlyphAtlas {
        GlyphAtlas::default_ascii()
    }

    #[test]
    fn test_atlas_uvs() {
        let a = atlas();
        // Space (char 32) should be at (0,0)
        let (uv_min, _uv_max) = a.char_uvs(32).unwrap();
        assert!(uv_min[0].abs() < 0.001);
        assert!(uv_min[1].abs() < 0.001);

        // 'A' (char 65) = index 33, row 2 col 1
        let (uv_min, _) = a.char_uvs(65).unwrap();
        let uv_cell = 64.0 / 1024.0; // 0.0625
        let expected_col = 33 % 16; // 1
        let expected_row = 33 / 16; // 2
        assert!((uv_min[0] - expected_col as f32 * uv_cell).abs() < 0.001);
        assert!((uv_min[1] - expected_row as f32 * uv_cell).abs() < 0.001);
    }

    #[test]
    fn test_single_line_layout() {
        let layout = GpuTextLayout::new(&atlas());
        let instances = layout.layout_single_line(
            "Hello",
            [0.0, 0.0, 0.0],
            0.1,
            [1.0, 1.0, 1.0, 1.0],
            &TextLayoutConfig::default(),
        );
        assert_eq!(instances.len(), 5);

        for (i, inst) in instances.iter().enumerate() {
            assert!((inst.glyph_offset[0] - i as f32 * 0.6).abs() < 0.001);
            assert!(inst.glyph_offset[1].abs() < 0.001);
        }
    }

    #[test]
    fn test_single_line_center_aligned() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            align: TextAlign::Center,
            ..Default::default()
        };
        let instances =
            layout.layout_single_line("AB", [0.0, 0.0, 0.0], 0.1, [1.0; 4], &config);
        assert_eq!(instances.len(), 2);
        assert!((instances[0].glyph_offset[0] - (-0.6)).abs() < 0.001);
        assert!(instances[1].glyph_offset[0].abs() < 0.001);
    }

    #[test]
    fn test_multiline_simple() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            max_width: Some(0.3),
            ..Default::default()
        };
        let instances = layout.layout_multiline_simple(
            "Hello World",
            [0.0, 5.0, 0.0],
            0.1,
            [1.0; 4],
            &config,
        );

        assert_eq!(instances.len(), 10); // 5 + 5 chars
        assert!(instances[0].glyph_offset[1].abs() < 0.001);
        assert!((instances[5].glyph_offset[1] - (-1.4)).abs() < 0.001);
    }

    #[test]
    fn test_explicit_newlines() {
        let layout = GpuTextLayout::new(&atlas());
        let instances = layout.layout_multiline_simple(
            "Hi\nWorld",
            [0.0, 0.0, 0.0],
            0.1,
            [1.0; 4],
            &TextLayoutConfig {
                max_width: Some(10.0),
                ..Default::default()
            },
        );

        // "Hi" = 2 chars, "World" = 5 chars = 7 total
        assert_eq!(instances.len(), 7);
    }

    #[test]
    fn test_non_ascii_filtered() {
        let layout = GpuTextLayout::new(&atlas());
        let instances = layout.layout_single_line(
            "A\u{1F680}B",
            [0.0, 0.0, 0.0],
            0.1,
            [1.0; 4],
            &TextLayoutConfig::default(),
        );
        // Emoji is out of ASCII atlas range, filtered out
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn test_glyph_instance_size() {
        assert_eq!(std::mem::size_of::<GlyphInstance>(), 64);
    }

    /// Proportional layout: per-glyph centering and proportional advances.
    ///
    /// Uses synthetic widths (A=1.0, B=0.5, C=0.25) so the math is checkable
    /// by hand. The fixed backend produces line-break decisions consistent
    /// with the callback when the backend's character width matches.
    #[test]
    fn test_proportional_single_line_centering() {
        use crate::backend::fixed::FixedWidthBackend;

        let layout = GpuTextLayout::new(&atlas());
        let backend = FixedWidthBackend::new();
        let font_spec = FontSpec::new("16px monospace");

        let widths = |ch: char| match ch {
            'A' => 1.0,
            'B' => 0.5,
            'C' => 0.25,
            _ => 0.6,
        };

        let config = TextLayoutConfig {
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
            max_width: None,
            ..Default::default()
        };

        let instances = layout.layout_label_proportional(
            "ABC",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &config,
            &backend,
            &font_spec,
            16.0,
            widths,
        );

        assert_eq!(instances.len(), 3);
        // Center-of-advance convention:
        //   A at x_cursor=0,    advance=1.0  -> offset = 0.5
        //   B at x_cursor=1.0,  advance=0.5  -> offset = 1.25
        //   C at x_cursor=1.5,  advance=0.25 -> offset = 1.625
        assert!((instances[0].glyph_offset[0] - 0.5).abs() < 1e-4);
        assert!((instances[1].glyph_offset[0] - 1.25).abs() < 1e-4);
        assert!((instances[2].glyph_offset[0] - 1.625).abs() < 1e-4);
    }

    /// Proportional layout: explicit newline produces two lines with correct
    /// per-line widths and per-line centering when alignment is `Center`.
    #[test]
    fn test_proportional_explicit_newline_center_aligned() {
        use crate::backend::fixed::FixedWidthBackend;

        let layout = GpuTextLayout::new(&atlas());
        let backend = FixedWidthBackend::new();
        let font_spec = FontSpec::new("16px monospace");

        // Uniform 0.6 advance — matches the FixedWidthBackend's defaults so
        // pretext's line widths line up cleanly with our callback's widths.
        let widths = |_ch: char| 0.6_f32;

        let config = TextLayoutConfig {
            align: TextAlign::Center,
            vertical_align: VerticalAlign::Top,
            max_width: None,
            ..Default::default()
        };

        let instances = layout.layout_label_proportional(
            "Hi\nWorld",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &config,
            &backend,
            &font_spec,
            16.0,
            widths,
        );

        // 2 chars + 5 chars = 7 glyphs across 2 lines.
        assert_eq!(instances.len(), 7);

        // Line 0 ("Hi") has width 2 * 0.6 = 1.2, center-aligned => cursor starts
        // at -0.6, first glyph center at -0.6 + 0.3 = -0.3.
        assert!((instances[0].glyph_offset[0] - (-0.3)).abs() < 1e-3);
        assert!(instances[0].glyph_offset[1].abs() < 1e-6);

        // Line 1 ("World") has width 5 * 0.6 = 3.0, center-aligned => cursor
        // starts at -1.5, first glyph center at -1.5 + 0.3 = -1.2.
        assert!((instances[2].glyph_offset[0] - (-1.2)).abs() < 1e-3);
        // Second line y offset = -line_spacing (default 1.4).
        assert!((instances[2].glyph_offset[1] - (-1.4)).abs() < 1e-3);
    }

    /// Proportional layout: width-limited wrapping splits into multiple lines.
    /// We don't pin exact line contents (depends on backend); we just check
    /// that wrapping happened.
    #[test]
    fn test_proportional_wrap_to_multiple_lines() {
        use crate::backend::fixed::FixedWidthBackend;

        let layout = GpuTextLayout::new(&atlas());
        let backend = FixedWidthBackend::new();
        let font_spec = FontSpec::new("16px monospace");

        let widths = |_ch: char| 0.6_f32;

        // max_width small enough to force a wrap with uniform 0.6 advance.
        let config = TextLayoutConfig {
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
            max_width: Some(2.6),
            ..Default::default()
        };

        let instances = layout.layout_label_proportional(
            "Alpha Beta Gamma",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &config,
            &backend,
            &font_spec,
            16.0,
            widths,
        );

        // At least two distinct y rows in the output.
        let rows: std::collections::BTreeSet<i32> = instances
            .iter()
            .map(|g| (g.glyph_offset[1] * 1000.0).round() as i32)
            .collect();
        assert!(
            rows.len() >= 2,
            "expected wrapping to produce >=2 rows, got {}",
            rows.len()
        );
    }

    /// Proportional layout: empty text returns no instances (no panic, no
    /// pretext call).
    #[test]
    fn test_proportional_empty_text() {
        use crate::backend::fixed::FixedWidthBackend;

        let layout = GpuTextLayout::new(&atlas());
        let backend = FixedWidthBackend::new();
        let font_spec = FontSpec::new("16px monospace");

        let instances = layout.layout_label_proportional(
            "",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &TextLayoutConfig::default(),
            &backend,
            &font_spec,
            16.0,
            |_| 0.6,
        );
        assert!(instances.is_empty());
    }

    /// Proportional layout: characters outside the atlas are skipped, but the
    /// callback is still consulted so width math stays consistent.
    #[test]
    fn test_proportional_non_ascii_filtered() {
        use crate::backend::fixed::FixedWidthBackend;

        let layout = GpuTextLayout::new(&atlas());
        let backend = FixedWidthBackend::new();
        let font_spec = FontSpec::new("16px monospace");

        let instances = layout.layout_label_proportional(
            "A\u{1F680}B",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &TextLayoutConfig::default(),
            &backend,
            &font_spec,
            16.0,
            |_| 0.6,
        );
        // Rocket emoji is outside ASCII -- skipped.
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn test_vertical_center_alignment() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            max_width: Some(0.3),
            vertical_align: VerticalAlign::Center,
            ..Default::default()
        };
        let instances = layout.layout_multiline_simple(
            "Hello World",
            [0.0, 0.0, 0.0],
            0.1,
            [1.0; 4],
            &config,
        );

        // 2 lines, total_height = 2 * 1.4 = 2.8, y_start = 1.4
        assert!((instances[0].glyph_offset[1] - 1.4).abs() < 0.001);
        assert!(instances[5].glyph_offset[1].abs() < 0.001);
    }
}
