/// Canvas measurement backend (WASM only).
///
/// Uses the browser's `canvas.measureText()` as the font oracle — this gives
/// pixel-accurate measurements that match what the DOM would render.
/// This is the same approach as the original TypeScript pretext.
///
/// Requires the `wasm` feature flag.
#![cfg(feature = "wasm")]

use std::cell::RefCell;
use std::collections::HashMap;

use unicode_segmentation::UnicodeSegmentation;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, OffscreenCanvas};

use super::{FontSpec, MeasureBackend, SegmentMetrics};
use crate::unicode;

/// Internal mutable state for the canvas backend.
struct CanvasInner {
    ctx: CanvasRenderingContext2d,
    cache: HashMap<(String, String), SegmentMetrics>,
    current_font: String,
}

impl CanvasInner {
    fn set_font(&mut self, font: &FontSpec) {
        if self.current_font != font.font {
            self.ctx.set_font(&font.font);
            self.current_font.clone_from(&font.font);
        }
    }

    fn measure_text_width(&mut self, text: &str, font: &FontSpec) -> f64 {
        self.set_font(font);
        self.ctx
            .measure_text(text)
            .map(|m| m.width())
            .unwrap_or(0.0)
    }

    fn measure_segment_inner(&mut self, text: &str, font: &FontSpec) -> SegmentMetrics {
        let cache_key = (font.font.clone(), text.to_string());

        // Return cached result if available
        if let Some(cached) = self.cache.get(&cache_key) {
            return cached.clone();
        }

        let mut contains_cjk = false;
        let mut emoji_count = 0;

        let graphemes: Vec<&str> = text.graphemes(true).collect();

        for grapheme in &graphemes {
            let c = grapheme.chars().next().unwrap();
            if unicode::is_cjk(c) {
                contains_cjk = true;
            }
            if unicode::is_emoji(c) {
                emoji_count += 1;
            }
        }

        // Measure the whole segment at once (more accurate than summing graphemes)
        let width = self.measure_text_width(text, font);

        // For multi-grapheme segments, measure per-grapheme widths for overflow-wrap
        let grapheme_widths = if graphemes.len() > 1 {
            let mut widths = Vec::with_capacity(graphemes.len());
            for grapheme in &graphemes {
                widths.push(self.measure_text_width(grapheme, font));
            }
            Some(widths)
        } else {
            None
        };

        let metrics = SegmentMetrics {
            width,
            contains_cjk,
            emoji_count,
            grapheme_widths,
        };

        // Cache for reuse
        self.cache.insert(cache_key, metrics.clone());

        metrics
    }
}

/// Canvas-based measurement backend for WASM.
///
/// Uses `RefCell` interior mutability so that `MeasureBackend` (which
/// takes `&self`) can mutate the canvas context and cache.
///
/// # Example (WASM)
///
/// ```ignore
/// let backend = CanvasBackend::new().expect("canvas context");
/// let font = FontSpec::new("16px Inter");
/// let prepared = pretext::prepare("Hello", &font, &backend, Default::default());
/// ```
pub struct CanvasBackend {
    inner: RefCell<CanvasInner>,
}

impl CanvasBackend {
    /// Create a new canvas backend.
    ///
    /// Tries `OffscreenCanvas` first (no DOM needed), falls back to a
    /// hidden `<canvas>` element.
    pub fn new() -> Result<Self, JsValue> {
        let ctx = Self::create_context()?;
        Ok(Self {
            inner: RefCell::new(CanvasInner {
                ctx,
                cache: HashMap::new(),
                current_font: String::new(),
            }),
        })
    }

    fn create_context() -> Result<CanvasRenderingContext2d, JsValue> {
        // Try OffscreenCanvas first (Web Workers compatible, no DOM)
        if let Ok(offscreen) = OffscreenCanvas::new(1, 1) {
            if let Ok(Some(ctx)) = offscreen.get_context("2d") {
                if let Ok(ctx) = ctx.dyn_into::<CanvasRenderingContext2d>() {
                    return Ok(ctx);
                }
            }
        }

        // Fall back to hidden DOM canvas
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;
        let canvas: HtmlCanvasElement = document.create_element("canvas")?.dyn_into()?;
        canvas.set_width(1);
        canvas.set_height(1);

        canvas
            .get_context("2d")?
            .ok_or_else(|| JsValue::from_str("failed to get 2d context"))?
            .dyn_into()
    }

    /// Clear the measurement cache. Call after font changes or when
    /// memory pressure is a concern.
    pub fn clear_cache(&self) {
        self.inner.borrow_mut().cache.clear();
    }
}

impl MeasureBackend for CanvasBackend {
    fn measure_segment(&self, text: &str, font: &FontSpec) -> SegmentMetrics {
        self.inner.borrow_mut().measure_segment_inner(text, font)
    }

    fn measure_space_width(&self, font: &FontSpec) -> f64 {
        self.inner.borrow_mut().measure_text_width(" ", font)
    }

    fn measure_hyphen_width(&self, font: &FontSpec) -> f64 {
        self.inner.borrow_mut().measure_text_width("-", font)
    }
}
