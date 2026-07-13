//! GPU text layout -- maps pretext line-breaking output to GPU glyph instances.
//!
//! This module bridges pretext's layout engine with GPU text rendering pipelines
//! like SDF-based text systems. It converts layout results into per-glyph
//! position data ready for upload to a GPU instance buffer.
//!
//! # Example
//!
//! ```no_run
//! use pretext::gpu_layout::{GpuTextLayout, GlyphAtlas, TextLayoutConfig};
//!
//! # fn main() -> pretext::Result<()> {
//! let atlas = GlyphAtlas::ascii_sdf(1024, 64, 16)?;
//! let layout = GpuTextLayout::new(&atlas);
//! let instances = layout.layout_single_line(
//!     "Hello, world!",
//!     [0.0, 5.0, 0.0],
//!     0.1,
//!     [1.0, 1.0, 1.0, 1.0],
//!     &TextLayoutConfig::default(),
//! )?;
//! # assert!(!instances.is_empty());
//! # Ok(())
//! # }
//! ```

use crate::backend::{FontSpec, MeasureBackend};
use crate::types::{
    DEFAULT_MAX_GRAPHEMES, DEFAULT_MAX_INPUT_BYTES, PrepareOptions, WhiteSpaceMode,
};
use crate::{Error, Result, layout_with_lines, prepare_with_segments};
use unicode_segmentation::UnicodeSegmentation;

/// Default maximum number of GPU glyph instances emitted by one layout call.
pub const DEFAULT_MAX_GPU_GLYPHS: usize = 65_536;

const GLYPH_BUFFER_RESOURCE: &str = "GPU glyph instance buffer";

/// A glyph atlas descriptor -- maps characters to UV coordinates.
#[derive(Debug, Clone)]
pub struct GlyphAtlas {
    /// Atlas texture size in texels (square).
    atlas_size: u32,
    /// Size of each glyph cell in texels.
    glyph_size: u32,
    /// Number of glyph columns per atlas row.
    glyphs_per_row: u32,
    /// First character code in the atlas.
    first_char: u32,
    /// One past the last character code.
    last_char: u32,
}

impl GlyphAtlas {
    /// Create an atlas descriptor for an ASCII SDF atlas.
    ///
    /// Default: 1024x1024 texture, 64x64 cells, 16 glyphs per row,
    /// ASCII 32-126.
    /// # Errors
    ///
    /// Returns an error when the atlas dimensions are zero or cannot contain
    /// the complete printable ASCII range.
    pub fn ascii_sdf(atlas_size: u32, glyph_size: u32, glyphs_per_row: u32) -> Result<Self> {
        if atlas_size == 0 || glyph_size == 0 || glyphs_per_row == 0 {
            return Err(Error::invalid_input(
                "glyph atlas dimensions",
                "atlas_size, glyph_size, and glyphs_per_row must be non-zero",
            ));
        }
        let cells_per_axis = atlas_size / glyph_size;
        if cells_per_axis == 0 || glyphs_per_row > cells_per_axis {
            return Err(Error::invalid_input(
                "glyph atlas dimensions",
                "glyph cells do not fit within the atlas width",
            ));
        }
        let glyph_count = 127_u32 - 32;
        let required_rows = glyph_count.div_ceil(glyphs_per_row);
        if required_rows > cells_per_axis {
            return Err(Error::invalid_input(
                "glyph atlas dimensions",
                "atlas does not contain enough cells for printable ASCII",
            ));
        }
        Ok(Self {
            atlas_size,
            glyph_size,
            glyphs_per_row,
            first_char: 32,
            last_char: 127,
        })
    }

    /// Default atlas (1024x1024, 64px cells, 16 per row, ASCII).
    #[must_use]
    pub const fn default_ascii() -> Self {
        Self {
            atlas_size: 1024,
            glyph_size: 64,
            glyphs_per_row: 16,
            first_char: 32,
            last_char: 127,
        }
    }

    /// Atlas texture size in texels.
    #[must_use]
    pub const fn atlas_size(&self) -> u32 {
        self.atlas_size
    }

    /// Glyph cell size in texels.
    #[must_use]
    pub const fn glyph_size(&self) -> u32 {
        self.glyph_size
    }

    /// Number of glyph cells per row.
    #[must_use]
    pub const fn glyphs_per_row(&self) -> u32 {
        self.glyphs_per_row
    }

    /// Inclusive first character code stored in the atlas.
    #[must_use]
    pub const fn first_char(&self) -> u32 {
        self.first_char
    }

    /// Exclusive last character code stored in the atlas.
    #[must_use]
    pub const fn last_char(&self) -> u32 {
        self.last_char
    }

    /// Get UV coordinates for a character code.
    ///
    /// Returns `(uv_min, uv_max)` or `None` if the character is out of range.
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
/// This struct uses `#[repr(C)]` for stable 64-byte layout. If your GPU
/// pipeline requires a marker trait such as `Pod`, map or wrap this value in
/// the downstream renderer where that dependency is owned.
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
    /// Anchor the first glyph at the label origin.
    #[default]
    Left,
    /// Center each line around the label origin.
    Center,
    /// End each line at the label origin.
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
    /// Maximum number of [`GlyphInstance`] values emitted by one layout call.
    ///
    /// The default is [`DEFAULT_MAX_GPU_GLYPHS`] (a 4 MiB instance buffer).
    /// Set this to zero to permit only layouts that emit no glyphs.
    pub max_glyphs: usize,
}

impl Default for TextLayoutConfig {
    fn default() -> Self {
        Self {
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
            glyph_advance: 0.6,
            line_spacing: 1.4,
            max_width: None,
            max_glyphs: DEFAULT_MAX_GPU_GLYPHS,
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
    /// Construct a layout engine from a validated atlas descriptor.
    #[must_use]
    pub fn new(atlas: &GlyphAtlas) -> Self {
        Self {
            atlas: atlas.clone(),
        }
    }

    fn atlas_uvs(&self, character: char) -> Result<([f32; 2], [f32; 2])> {
        self.atlas
            .char_uvs(character as u32)
            .ok_or(Error::UnsupportedGlyph {
                character,
                codepoint: character as u32,
            })
    }

    fn validate_supported_text(&self, text: &str, allow_hard_breaks: bool) -> Result<usize> {
        let mut glyph_count = 0_usize;
        for character in text.chars() {
            if self.atlas.char_uvs(character as u32).is_some() {
                glyph_count = glyph_count
                    .checked_add(1)
                    .ok_or(Error::ArithmeticOverflow {
                        operation: "counting GPU glyph instances",
                    })?;
            } else if !(allow_hard_breaks && is_hard_line_break(character)) {
                return Err(Error::UnsupportedGlyph {
                    character,
                    codepoint: character as u32,
                });
            }
        }
        Ok(glyph_count)
    }

    /// Layout a single-line label (no wrapping, fast path).
    ///
    /// # Errors
    ///
    /// Returns an error for invalid geometry, unsupported atlas characters,
    /// oversized input, or an output exceeding [`TextLayoutConfig::max_glyphs`].
    #[allow(clippy::cast_precision_loss)]
    pub fn layout_single_line(
        &self,
        text: &str,
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
    ) -> Result<Vec<GlyphInstance>> {
        validate_layout_inputs(world_pos, font_size, color, config)?;
        validate_fixed_glyph_advance(config)?;
        validate_text_input(text)?;
        let glyph_count = self.validate_supported_text(text, false)?;
        validate_glyph_buffer_capacity(glyph_count, config)?;

        let text_width = glyph_count as f32 * config.glyph_advance;
        validate_derived_geometry("single-line text width", text_width)?;

        let x_start = match config.align {
            TextAlign::Left => 0.0,
            TextAlign::Center => -text_width * 0.5,
            TextAlign::Right => -text_width,
        };

        let mut instances = Vec::with_capacity(glyph_count);

        for (i, ch) in text.chars().enumerate() {
            let (uv_min, uv_max) = self.atlas_uvs(ch)?;
            let cell_start = (i as f32).mul_add(config.glyph_advance, x_start);
            let x_offset = config.glyph_advance.mul_add(0.5, cell_start);
            validate_derived_geometry("single-line glyph offset", x_offset)?;
            instances.push(GlyphInstance {
                world_pos,
                font_size,
                glyph_offset: [x_offset, 0.0],
                atlas_uv_min: uv_min,
                atlas_uv_max: uv_max,
                color,
                pad: [0.0; 2],
            });
        }

        Ok(instances)
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
    ///
    /// # Errors
    ///
    /// Returns an error when geometry is invalid, text measurement fails, an
    /// atlas character is unavailable, or the configured output bound is
    /// exceeded. Newline, carriage-return, and form-feed controls are accepted
    /// as structural hard breaks and do not emit glyph instances.
    #[allow(clippy::cast_precision_loss)]
    pub fn layout_multiline(
        &self,
        text: &str,
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
        backend: &dyn MeasureBackend,
    ) -> Result<Vec<GlyphInstance>> {
        validate_layout_inputs(world_pos, font_size, color, config)?;
        validate_fixed_glyph_advance(config)?;
        validate_text_input(text)?;
        let source_glyphs = self.validate_supported_text(text, true)?;
        validate_glyph_buffer_capacity(source_glyphs, config)?;
        let Some(max_width) = config.max_width else {
            if text.chars().any(is_hard_line_break) {
                let lines = split_hard_break_lines(text);
                return self.emit_multiline_instances(&lines, world_pos, font_size, color, config);
            }
            return self.layout_single_line(text, world_pos, font_size, color, config);
        };

        // Convert world-space max_width to "character units" for pretext.
        // pretext works in pixel space; we convert back to world space after.
        let glyph_width = font_size * config.glyph_advance;
        validate_derived_geometry("world-space glyph width", glyph_width)?;
        let chars_per_width = max_width / glyph_width;
        validate_derived_geometry("characters per line", chars_per_width)?;
        let px_font_size: f64 = 16.0; // Arbitrary reference size for pretext
        let px_max_width: f64 =
            f64::from(chars_per_width) * px_font_size * f64::from(config.glyph_advance);
        if !px_max_width.is_finite() || px_max_width <= 0.0 {
            return Err(Error::invalid_input(
                "derived pixel max width",
                "must be finite and greater than zero",
            ));
        }

        let font_spec = FontSpec::new(format!("{px_font_size}px monospace"))?;
        let prepared = prepare_with_segments(
            text,
            &font_spec,
            backend,
            PrepareOptions {
                white_space: WhiteSpaceMode::PreWrap,
                ..PrepareOptions::default()
            },
        )?;
        let lines = layout_with_lines(&prepared, px_max_width)?;
        let total_glyphs = lines.iter().try_fold(0_usize, |total, line| {
            let line_glyphs = self.validate_supported_text(&line.text, false)?;
            total
                .checked_add(line_glyphs)
                .ok_or(Error::ArithmeticOverflow {
                    operation: "counting multiline GPU glyph instances",
                })
        })?;
        validate_glyph_buffer_capacity(total_glyphs, config)?;
        let line_chars: Vec<Vec<char>> = lines
            .into_iter()
            .map(|line| line.text.chars().collect())
            .collect();

        self.emit_multiline_instances(&line_chars, world_pos, font_size, color, config)
    }

    /// Layout a label using pretext for line-breaking and an explicit per-glyph
    /// advance function for placement.
    ///
    /// Use this when the font is proportional (e.g. Inter) and you have real
    /// per-glyph advance widths (e.g. from `skrifa::metrics::GlyphMetrics`). The
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
    /// * `glyph_advance` — `Fn(char) -> f32`. Returns the per-glyph advance in
    ///   `font_size` units (1.0 == one `font_size` wide). Every rendered
    ///   character must exist in the atlas. Explicit hard line-break controls
    ///   are accepted as structure and are not passed to the callback.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry, callback advances, backend measurements,
    /// atlas coverage, or output resource bounds are invalid.
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
        glyph_advance: F,
    ) -> Result<Vec<GlyphInstance>>
    where
        F: Fn(char) -> f32,
    {
        validate_layout_inputs(world_pos, font_size, color, config)?;
        validate_text_input(text)?;
        let source_glyphs = self.validate_supported_text(text, true)?;
        validate_glyph_buffer_capacity(source_glyphs, config)?;
        let pretext_font_size_px = font_spec.size_px();
        if text.is_empty() {
            return Ok(Vec::new());
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
                ..PrepareOptions::default()
            },
        )?;

        // Convert world-space max_width to pretext pixel units.
        // pretext returned widths are in `pretext_font_size_px` units; we treat
        // world width as `font_size`-units (1.0 == one font_size wide), so the
        // conversion factor is pretext_font_size_px / font_size.
        let max_width_px = match config.max_width {
            Some(w) => {
                let width = f64::from(w / font_size) * pretext_font_size_px;
                if !width.is_finite() || width <= 0.0 {
                    return Err(Error::invalid_input(
                        "derived pixel max width",
                        "must be finite and greater than zero",
                    ));
                }
                width
            }
            None => f64::INFINITY,
        };

        let lines = layout_with_lines(&prepared, max_width_px)?;

        if lines.is_empty() {
            return Ok(Vec::new());
        }

        let line_count = lines.len();
        let line_span = line_count.saturating_sub(1) as f32 * config.line_spacing;
        validate_derived_geometry("multi-line vertical span", line_span)?;

        // Anchor's Y position relative to the top line.
        let y_start = match config.vertical_align {
            VerticalAlign::Top => 0.0,
            VerticalAlign::Center => line_span * 0.5,
            VerticalAlign::Bottom => line_span,
        };

        let total_glyphs = lines.iter().try_fold(0_usize, |total, line| {
            let line_glyphs = self.validate_supported_text(&line.text, false)?;
            total
                .checked_add(line_glyphs)
                .ok_or(Error::ArithmeticOverflow {
                    operation: "counting proportional GPU glyph instances",
                })
        })?;
        validate_glyph_buffer_capacity(total_glyphs, config)?;
        let mut instances = Vec::with_capacity(total_glyphs);

        for (line_idx, line) in lines.iter().enumerate() {
            let mut measured = Vec::with_capacity(line.text.chars().count());
            let mut line_width_units = 0.0_f32;
            for character in line.text.chars() {
                let advance = glyph_advance(character);
                if !advance.is_finite() || advance < 0.0 {
                    return Err(Error::invalid_input(
                        "glyph advance",
                        "callback values must be finite and non-negative",
                    ));
                }
                line_width_units += advance;
                validate_derived_geometry("proportional line width", line_width_units)?;
                measured.push((character, advance));
            }

            let mut x_cursor = match config.align {
                TextAlign::Left => 0.0,
                TextAlign::Center => -line_width_units * 0.5,
                TextAlign::Right => -line_width_units,
            };

            let y_offset = (line_idx as f32).mul_add(-config.line_spacing, y_start);
            validate_derived_geometry("multi-line vertical offset", y_offset)?;

            for (ch, advance) in measured {
                let (uv_min, uv_max) = self.atlas_uvs(ch)?;
                let x_offset = advance.mul_add(0.5, x_cursor);
                validate_derived_geometry("proportional glyph offset", x_offset)?;
                instances.push(GlyphInstance {
                    world_pos,
                    font_size,
                    glyph_offset: [x_offset, y_offset],
                    atlas_uv_min: uv_min,
                    atlas_uv_max: uv_max,
                    color,
                    pad: [0.0; 2],
                });
                x_cursor += advance;
                validate_derived_geometry("proportional glyph cursor", x_cursor)?;
            }
        }

        Ok(instances)
    }

    /// Layout multi-line text using simple character-count wrapping.
    ///
    /// This is the fast, no-measurement path. Uses a fixed character width
    /// (`glyph_advance`) to determine line breaks. Good enough for monospace
    /// fonts and world-space labels where pixel accuracy isn't critical.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry is invalid, an atlas character is
    /// unavailable, or the configured output bound is exceeded. Newline,
    /// carriage-return, and form-feed controls are accepted as hard breaks.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn layout_multiline_simple(
        &self,
        text: &str,
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
    ) -> Result<Vec<GlyphInstance>> {
        validate_layout_inputs(world_pos, font_size, color, config)?;
        validate_fixed_glyph_advance(config)?;
        validate_text_input(text)?;
        let source_glyphs = self.validate_supported_text(text, true)?;
        validate_glyph_buffer_capacity(source_glyphs, config)?;
        let glyph_width = font_size * config.glyph_advance;
        validate_derived_geometry("world-space glyph width", glyph_width)?;
        let max_chars = match config.max_width {
            Some(w) => (w / glyph_width).floor() as usize,
            None => {
                if text.chars().any(is_hard_line_break) {
                    let lines = split_hard_break_lines(text);
                    return self
                        .emit_multiline_instances(&lines, world_pos, font_size, color, config);
                }
                return self.layout_single_line(text, world_pos, font_size, color, config);
            }
        };

        let max_chars = max_chars.max(1);
        let lines = Self::split_lines_simple(text, max_chars);

        self.emit_multiline_instances(&lines, world_pos, font_size, color, config)
    }

    /// Split text into lines by word-wrapping at a character limit.
    fn split_lines_simple(text: &str, max_chars: usize) -> Vec<Vec<char>> {
        let mut lines: Vec<Vec<char>> = Vec::new();
        let normalized = normalize_hard_line_breaks(text);

        for paragraph in normalized.split('\n') {
            let words: Vec<&str> = paragraph.split_whitespace().collect();
            if words.is_empty() {
                lines.push(Vec::new());
                continue;
            }

            let mut current_line: Vec<char> = Vec::new();

            for word in &words {
                let word_chars: Vec<char> = word.chars().collect();

                if word_chars.len() > max_chars {
                    if !current_line.is_empty() {
                        lines.push(std::mem::take(&mut current_line));
                    }
                    let mut chunks = word_chars.chunks(max_chars).peekable();
                    while let Some(chunk) = chunks.next() {
                        if chunks.peek().is_some() || chunk.len() == max_chars {
                            lines.push(chunk.to_vec());
                        } else {
                            current_line.extend_from_slice(chunk);
                        }
                    }
                    continue;
                }

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

    /// Emit `GlyphInstance` data for multiple lines.
    #[allow(clippy::cast_precision_loss)]
    fn emit_multiline_instances(
        &self,
        lines: &[Vec<char>],
        world_pos: [f32; 3],
        font_size: f32,
        color: [f32; 4],
        config: &TextLayoutConfig,
    ) -> Result<Vec<GlyphInstance>> {
        let line_count = lines.len();
        let line_span = line_count.saturating_sub(1) as f32 * config.line_spacing;
        validate_derived_geometry("multi-line vertical span", line_span)?;

        // Vertical offset based on alignment
        let y_start = match config.vertical_align {
            VerticalAlign::Top => 0.0,
            VerticalAlign::Center => line_span * 0.5,
            VerticalAlign::Bottom => line_span,
        };

        let total_glyphs = lines.iter().try_fold(0_usize, |total, line| {
            total
                .checked_add(line.len())
                .ok_or(Error::ArithmeticOverflow {
                    operation: "counting emitted GPU glyph instances",
                })
        })?;
        validate_glyph_buffer_capacity(total_glyphs, config)?;
        let mut instances = Vec::with_capacity(total_glyphs);

        for (line_idx, line_chars) in lines.iter().enumerate() {
            let line_width = line_chars.len() as f32 * config.glyph_advance;
            validate_derived_geometry("multi-line text width", line_width)?;

            let x_start = match config.align {
                TextAlign::Left => 0.0,
                TextAlign::Center => -line_width * 0.5,
                TextAlign::Right => -line_width,
            };

            let y_offset = (line_idx as f32).mul_add(-config.line_spacing, y_start);
            validate_derived_geometry("multi-line vertical offset", y_offset)?;

            for (char_idx, &ch) in line_chars.iter().enumerate() {
                let (uv_min, uv_max) = self.atlas_uvs(ch)?;
                let cell_start = (char_idx as f32).mul_add(config.glyph_advance, x_start);
                let x_offset = config.glyph_advance.mul_add(0.5, cell_start);
                validate_derived_geometry("multi-line glyph offset", x_offset)?;
                instances.push(GlyphInstance {
                    world_pos,
                    font_size,
                    glyph_offset: [x_offset, y_offset],
                    atlas_uv_min: uv_min,
                    atlas_uv_max: uv_max,
                    color,
                    pad: [0.0; 2],
                });
            }
        }

        Ok(instances)
    }
}

const fn is_hard_line_break(character: char) -> bool {
    matches!(character, '\n' | '\r' | '\u{000C}')
}

fn normalize_hard_line_breaks(text: &str) -> String {
    text.replace("\r\n", "\n").replace(['\r', '\u{000C}'], "\n")
}

fn split_hard_break_lines(text: &str) -> Vec<Vec<char>> {
    normalize_hard_line_breaks(text)
        .split('\n')
        .map(|line| line.chars().collect())
        .collect()
}

fn glyph_buffer_bytes(glyph_count: usize) -> Result<usize> {
    glyph_count
        .checked_mul(std::mem::size_of::<GlyphInstance>())
        .ok_or(Error::ArithmeticOverflow {
            operation: "calculating GPU glyph instance buffer bytes",
        })
}

fn validate_glyph_buffer_capacity(glyph_count: usize, config: &TextLayoutConfig) -> Result<usize> {
    let requested_bytes = glyph_buffer_bytes(glyph_count)?;
    if glyph_count > config.max_glyphs {
        return Err(Error::ResourceLimit {
            resource: GLYPH_BUFFER_RESOURCE,
            requested_bytes,
            max_bytes: glyph_buffer_bytes(config.max_glyphs)?,
        });
    }

    let addressable_bytes = isize::MAX as usize;
    if requested_bytes > addressable_bytes {
        return Err(Error::ResourceLimit {
            resource: GLYPH_BUFFER_RESOURCE,
            requested_bytes,
            max_bytes: addressable_bytes,
        });
    }

    Ok(requested_bytes)
}

fn validate_text_input(text: &str) -> Result<()> {
    if text.len() > DEFAULT_MAX_INPUT_BYTES {
        return Err(Error::InputTooLarge {
            bytes: text.len(),
            max_bytes: DEFAULT_MAX_INPUT_BYTES,
        });
    }

    let graphemes = text.graphemes(true).count();
    if graphemes > DEFAULT_MAX_GRAPHEMES {
        return Err(Error::InputComplexity {
            resource: "GPU layout graphemes",
            units: graphemes,
            max_units: DEFAULT_MAX_GRAPHEMES,
        });
    }

    Ok(())
}

fn validate_derived_geometry(parameter: &'static str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::invalid_input(
            parameter,
            "derived value is not finite",
        ))
    }
}

fn validate_layout_inputs(
    world_pos: [f32; 3],
    font_size: f32,
    color: [f32; 4],
    config: &TextLayoutConfig,
) -> Result<()> {
    if world_pos.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid_input(
            "world_pos",
            "all coordinates must be finite",
        ));
    }
    if color.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid_input("color", "all channels must be finite"));
    }
    if !font_size.is_finite() || font_size <= 0.0 {
        return Err(Error::invalid_input(
            "font_size",
            "must be finite and greater than zero",
        ));
    }
    if !config.line_spacing.is_finite() || config.line_spacing <= 0.0 {
        return Err(Error::invalid_input(
            "config.line_spacing",
            "must be finite and greater than zero",
        ));
    }
    if config
        .max_width
        .is_some_and(|width| !width.is_finite() || width <= 0.0)
    {
        return Err(Error::invalid_input(
            "config.max_width",
            "must be finite and greater than zero when provided",
        ));
    }
    Ok(())
}

fn validate_fixed_glyph_advance(config: &TextLayoutConfig) -> Result<()> {
    if !config.glyph_advance.is_finite() || config.glyph_advance <= 0.0 {
        return Err(Error::invalid_input(
            "config.glyph_advance",
            "must be finite and greater than zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atlas() -> GlyphAtlas {
        GlyphAtlas::default_ascii()
    }

    #[track_caller]
    fn valid<T>(result: Result<T>) -> T {
        result.expect("test input is valid")
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
        let instances = valid(layout.layout_single_line(
            "Hello",
            [0.0, 0.0, 0.0],
            0.1,
            [1.0, 1.0, 1.0, 1.0],
            &TextLayoutConfig::default(),
        ));
        assert_eq!(instances.len(), 5);

        for (i, inst) in instances.iter().enumerate() {
            let expected = (i as f32).mul_add(0.6, 0.3);
            assert!((inst.glyph_offset[0] - expected).abs() < 0.001);
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
            valid(layout.layout_single_line("AB", [0.0, 0.0, 0.0], 0.1, [1.0; 4], &config));
        assert_eq!(instances.len(), 2);
        assert!((instances[0].glyph_offset[0] - (-0.3)).abs() < 0.001);
        assert!((instances[1].glyph_offset[0] - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_single_line_right_aligned_uses_glyph_centers() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            align: TextAlign::Right,
            ..Default::default()
        };
        let instances =
            valid(layout.layout_single_line("AB", [0.0, 0.0, 0.0], 0.1, [1.0; 4], &config));

        assert_eq!(instances.len(), 2);
        assert!((instances[0].glyph_offset[0] - (-0.9)).abs() < 0.001);
        assert!((instances[1].glyph_offset[0] - (-0.3)).abs() < 0.001);
    }

    #[test]
    fn test_multiline_simple() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            max_width: Some(0.3),
            ..Default::default()
        };
        let instances = valid(layout.layout_multiline_simple(
            "Hello World",
            [0.0, 5.0, 0.0],
            0.1,
            [1.0; 4],
            &config,
        ));

        assert_eq!(instances.len(), 10); // 5 + 5 chars
        assert!(instances[0].glyph_offset[1].abs() < 0.001);
        assert!((instances[5].glyph_offset[1] - (-1.4)).abs() < 0.001);
    }

    #[test]
    fn test_explicit_newlines() {
        let layout = GpuTextLayout::new(&atlas());
        let instances = valid(layout.layout_multiline_simple(
            "Hi\nWorld",
            [0.0, 0.0, 0.0],
            0.1,
            [1.0; 4],
            &TextLayoutConfig {
                max_width: Some(10.0),
                ..Default::default()
            },
        ));

        // "Hi" = 2 chars, "World" = 5 chars = 7 total
        assert_eq!(instances.len(), 7);
    }

    #[test]
    fn single_line_rejects_character_missing_from_atlas() {
        let layout = GpuTextLayout::new(&atlas());
        let error = layout
            .layout_single_line(
                "A\u{1F680}B",
                [0.0, 0.0, 0.0],
                0.1,
                [1.0; 4],
                &TextLayoutConfig::default(),
            )
            .expect_err("emoji must not be silently dropped");
        assert!(matches!(
            error,
            Error::UnsupportedGlyph {
                character: '\u{1F680}',
                codepoint: 0x1F680,
            }
        ));
    }

    #[test]
    fn test_glyph_instance_size() {
        assert_eq!(std::mem::size_of::<GlyphInstance>(), 64);
    }

    #[test]
    fn atlas_rejects_invalid_dimensions() {
        assert!(GlyphAtlas::ascii_sdf(1024, 64, 0).is_err());
        assert!(GlyphAtlas::ascii_sdf(64, 64, 16).is_err());
    }

    #[test]
    fn measured_multiline_layout_uses_wrapped_ranges() {
        use crate::backend::fixed::FixedWidthBackend;

        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            max_width: Some(3.0),
            ..TextLayoutConfig::default()
        };
        let instances = valid(layout.layout_multiline(
            "hello world",
            [0.0; 3],
            1.0,
            [1.0; 4],
            &config,
            &FixedWidthBackend::new(),
        ));
        let rows: std::collections::BTreeSet<i32> = instances
            .iter()
            .map(|glyph| (glyph.glyph_offset[1] * 1000.0).round() as i32)
            .collect();

        assert_eq!(instances.len(), 11);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn measured_multiline_preserves_renderable_spaces() {
        use crate::backend::fixed::FixedWidthBackend;

        let layout = GpuTextLayout::new(&atlas());
        let instances = valid(layout.layout_multiline(
            "a  b",
            [0.0; 3],
            1.0,
            [1.0; 4],
            &TextLayoutConfig {
                max_width: Some(100.0),
                ..TextLayoutConfig::default()
            },
            &FixedWidthBackend::new(),
        ));
        assert_eq!(instances.len(), 4);
    }

    #[test]
    fn multiline_allows_hard_break_controls_but_rejects_other_controls() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig::default();
        let instances = valid(layout.layout_multiline_simple(
            "A\r\nB\u{000C}C",
            [0.0; 3],
            1.0,
            [1.0; 4],
            &config,
        ));
        assert_eq!(instances.len(), 3);

        let error = layout
            .layout_multiline_simple("A\tB", [0.0; 3], 1.0, [1.0; 4], &config)
            .expect_err("unsupported tab must not be silently dropped");
        assert!(matches!(
            error,
            Error::UnsupportedGlyph {
                character: '\t',
                codepoint: 9,
            }
        ));
    }

    #[test]
    fn gpu_output_is_bounded_before_instance_allocation() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            max_glyphs: 2,
            ..TextLayoutConfig::default()
        };
        let error = layout
            .layout_single_line("ABC", [0.0; 3], 1.0, [1.0; 4], &config)
            .expect_err("three glyphs must exceed a two-glyph buffer limit");
        assert!(matches!(
            error,
            Error::ResourceLimit {
                resource: GLYPH_BUFFER_RESOURCE,
                requested_bytes: 192,
                max_bytes: 128,
            }
        ));
    }

    #[test]
    fn gpu_layout_rejects_excessive_structural_input() {
        let layout = GpuTextLayout::new(&atlas());
        let text = "\n".repeat(DEFAULT_MAX_GRAPHEMES + 1);
        let error = layout
            .layout_multiline_simple(&text, [0.0; 3], 1.0, [1.0; 4], &TextLayoutConfig::default())
            .expect_err("line controls must not bypass the structural input bound");
        assert!(matches!(
            error,
            Error::InputComplexity {
                resource: "GPU layout graphemes",
                units,
                max_units: DEFAULT_MAX_GRAPHEMES,
            } if units == DEFAULT_MAX_GRAPHEMES + 1
        ));
    }

    #[test]
    fn simple_layout_splits_oversized_words() {
        let lines = GpuTextLayout::split_lines_simple("abcdefgh", 3);
        assert_eq!(
            lines,
            vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f'], vec!['g', 'h']]
        );
    }

    #[test]
    fn gpu_layout_rejects_invalid_numeric_inputs() {
        let layout = GpuTextLayout::new(&atlas());
        assert!(
            layout
                .layout_single_line(
                    "hello",
                    [f32::NAN, 0.0, 0.0],
                    1.0,
                    [1.0; 4],
                    &TextLayoutConfig::default(),
                )
                .is_err()
        );
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
        let font_spec = valid(FontSpec::new("16px monospace"));

        let widths = |ch: char| match ch {
            'A' => 1.0,
            'B' => 0.5,
            'C' => 0.25,
            _ => 0.6,
        };

        let config = TextLayoutConfig {
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
            // The proportional callback, not this fixed-path field, controls
            // placement in this API.
            glyph_advance: 0.0,
            max_width: None,
            ..Default::default()
        };

        let instances = valid(layout.layout_label_proportional(
            "ABC",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &config,
            &backend,
            &font_spec,
            widths,
        ));

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
        let font_spec = valid(FontSpec::new("16px monospace"));

        // Uniform 0.6 advance — matches the FixedWidthBackend's defaults so
        // pretext's line widths line up cleanly with our callback's widths.
        let widths = |_ch: char| 0.6_f32;

        let config = TextLayoutConfig {
            align: TextAlign::Center,
            vertical_align: VerticalAlign::Top,
            max_width: None,
            ..Default::default()
        };

        let instances = valid(layout.layout_label_proportional(
            "Hi\nWorld",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &config,
            &backend,
            &font_spec,
            widths,
        ));

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
        let font_spec = valid(FontSpec::new("16px monospace"));

        let widths = |_ch: char| 0.6_f32;

        // max_width small enough to force a wrap with uniform 0.6 advance.
        let config = TextLayoutConfig {
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
            max_width: Some(2.6),
            ..Default::default()
        };

        let instances = valid(layout.layout_label_proportional(
            "Alpha Beta Gamma",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &config,
            &backend,
            &font_spec,
            widths,
        ));

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
        let font_spec = valid(FontSpec::new("16px monospace"));

        let instances = valid(layout.layout_label_proportional(
            "",
            [0.0, 0.0, 0.0],
            1.0,
            [1.0; 4],
            &TextLayoutConfig::default(),
            &backend,
            &font_spec,
            |_| 0.6,
        ));
        assert!(instances.is_empty());
    }

    /// Proportional layout rejects characters outside the atlas.
    #[test]
    fn test_proportional_non_ascii_rejected() {
        use crate::backend::fixed::FixedWidthBackend;

        let layout = GpuTextLayout::new(&atlas());
        let backend = FixedWidthBackend::new();
        let font_spec = valid(FontSpec::new("16px monospace"));

        let error = layout
            .layout_label_proportional(
                "A\u{1F680}B",
                [0.0, 0.0, 0.0],
                1.0,
                [1.0; 4],
                &TextLayoutConfig::default(),
                &backend,
                &font_spec,
                |_| 0.6,
            )
            .expect_err("emoji must not be silently dropped");
        assert!(matches!(
            error,
            Error::UnsupportedGlyph {
                character: '\u{1F680}',
                codepoint: 0x1F680,
            }
        ));
    }

    #[test]
    fn test_vertical_center_alignment() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            max_width: Some(0.3),
            vertical_align: VerticalAlign::Center,
            ..Default::default()
        };
        let instances = valid(layout.layout_multiline_simple(
            "Hello World",
            [0.0, 0.0, 0.0],
            0.1,
            [1.0; 4],
            &config,
        ));

        // Two baselines span 1.4 units, centered around the anchor.
        assert!((instances[0].glyph_offset[1] - 0.7).abs() < 0.001);
        assert!((instances[5].glyph_offset[1] - (-0.7)).abs() < 0.001);
    }

    #[test]
    fn derived_gpu_geometry_overflow_is_rejected() {
        let layout = GpuTextLayout::new(&atlas());
        let config = TextLayoutConfig {
            glyph_advance: f32::MAX,
            ..TextLayoutConfig::default()
        };
        assert!(
            layout
                .layout_single_line("AB", [0.0; 3], 1.0, [1.0; 4], &config)
                .is_err()
        );
    }
}
