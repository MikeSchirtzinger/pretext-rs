//! Canvas measurement backend (WASM only).
//!
//! Uses the browser's `canvas.measureText()` as the font oracle — this gives
//! pixel-accurate measurements that match what the DOM would render.
//!
//! Requires the `wasm` feature flag.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

use unicode_segmentation::UnicodeSegmentation;
use wasm_bindgen::prelude::*;
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, OffscreenCanvas,
    OffscreenCanvasRenderingContext2d, TextMetrics,
};

use super::{FontSpec, MeasureBackend, SegmentMetrics, validate_metric};
use crate::{Error, Result, unicode};

/// Default maximum number of distinct `(font, text)` measurements retained.
pub const DEFAULT_CACHE_CAPACITY: usize = 1_024;

/// Default maximum estimated bytes retained by cached keys and measurements.
pub const DEFAULT_CACHE_BYTE_CAPACITY: usize = 4 * 1_024 * 1_024;

type CacheKey = (String, String);

struct CacheEntry {
    metrics: SegmentMetrics,
    retained_bytes: usize,
}

/// Bounded insertion-order cache. Repeated hits do not allocate or extend its
/// lifetime indefinitely; the oldest distinct entry is evicted at capacity.
struct MeasurementCache {
    entries: HashMap<CacheKey, CacheEntry>,
    insertion_order: VecDeque<CacheKey>,
    entry_capacity: usize,
    byte_capacity: usize,
    retained_bytes: usize,
}

impl MeasurementCache {
    fn new(entry_capacity: usize, byte_capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(entry_capacity.min(DEFAULT_CACHE_CAPACITY)),
            insertion_order: VecDeque::with_capacity(entry_capacity.min(DEFAULT_CACHE_CAPACITY)),
            entry_capacity,
            byte_capacity,
            retained_bytes: 0,
        }
    }

    fn get(&self, key: &CacheKey) -> Option<&SegmentMetrics> {
        self.entries.get(key).map(|entry| &entry.metrics)
    }

    fn insert(&mut self, key: CacheKey, value: SegmentMetrics) {
        if self.entry_capacity == 0 || self.byte_capacity == 0 {
            return;
        }

        let entry_bytes = estimated_retained_bytes(&key, &value);
        if entry_bytes > self.byte_capacity {
            return;
        }

        if let Some(existing) = self.entries.get_mut(&key) {
            let retained_without_existing =
                self.retained_bytes.saturating_sub(existing.retained_bytes);
            let Some(updated_retained_bytes) = retained_without_existing.checked_add(entry_bytes)
            else {
                return;
            };
            if updated_retained_bytes > self.byte_capacity {
                return;
            }

            existing.metrics = value;
            existing.retained_bytes = entry_bytes;
            self.retained_bytes = updated_retained_bytes;
            return;
        }

        while self.entries.len() >= self.entry_capacity
            || self
                .retained_bytes
                .checked_add(entry_bytes)
                .is_none_or(|total| total > self.byte_capacity)
        {
            let Some(oldest) = self.insertion_order.pop_front() else {
                self.entries.clear();
                self.retained_bytes = 0;
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.retained_bytes = self.retained_bytes.saturating_sub(removed.retained_bytes);
            }
        }

        let Some(updated_retained_bytes) = self.retained_bytes.checked_add(entry_bytes) else {
            return;
        };
        if updated_retained_bytes > self.byte_capacity {
            return;
        }

        self.insertion_order.push_back(key.clone());
        self.retained_bytes = updated_retained_bytes;
        self.entries.insert(
            key,
            CacheEntry {
                metrics: value,
                retained_bytes: entry_bytes,
            },
        );
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
        self.retained_bytes = 0;
    }
}

fn estimated_retained_bytes(key: &CacheKey, metrics: &SegmentMetrics) -> usize {
    // The key is retained once by the map and once by the insertion-order
    // queue. Fixed-size container overhead is included so many tiny entries
    // cannot bypass the byte limit; allocator-specific bookkeeping remains
    // bounded by `entry_capacity`.
    let key_storage = std::mem::size_of::<CacheKey>()
        .saturating_mul(2)
        .saturating_add(key.0.len().saturating_mul(2))
        .saturating_add(key.1.len().saturating_mul(2));
    let metrics_storage = std::mem::size_of::<SegmentMetrics>().saturating_add(
        metrics.grapheme_widths.as_ref().map_or(0, |widths| {
            widths.capacity().saturating_mul(std::mem::size_of::<f64>())
        }),
    );
    key_storage.saturating_add(metrics_storage)
}

/// Internal mutable state for the canvas backend.
struct CanvasInner {
    ctx: CanvasContext,
    cache: MeasurementCache,
    current_font: String,
}

/// A 2D measurement context that works in both window and worker globals.
///
/// `OffscreenCanvasRenderingContext2d` and `CanvasRenderingContext2d` are
/// distinct JavaScript interfaces. In particular, an offscreen context cannot
/// be cast to the DOM canvas context even though both expose the operations we
/// need. Keeping the variants explicit avoids that invalid cast and lets the
/// backend initialize in workers, where `window` and `document` do not exist.
enum CanvasContext {
    Dom(CanvasRenderingContext2d),
    Offscreen(OffscreenCanvasRenderingContext2d),
}

impl CanvasContext {
    fn set_font(&self, font: &str) {
        match self {
            Self::Dom(context) => context.set_font(font),
            Self::Offscreen(context) => context.set_font(font),
        }
    }

    fn font(&self) -> String {
        match self {
            Self::Dom(context) => context.font(),
            Self::Offscreen(context) => context.font(),
        }
    }

    fn measure_text(&self, text: &str) -> std::result::Result<TextMetrics, JsValue> {
        match self {
            Self::Dom(context) => context.measure_text(text),
            Self::Offscreen(context) => context.measure_text(text),
        }
    }
}

impl CanvasInner {
    fn set_font(&mut self, font: &FontSpec) -> Result<()> {
        if !font.has_generic_family() {
            return Err(Error::InvalidFontSpec {
                spec: font.as_css_str().to_owned(),
                reason: "canvas measurement requires an explicit CSS generic family fallback"
                    .to_owned(),
            });
        }
        if self.current_font != font.as_css_str() {
            // Canvas silently ignores invalid `font` assignments. Establish a
            // valid, semantically distinct sentinel first so a rejected
            // assignment becomes an explicit backend error instead of a
            // false-success measurement using stale/default metrics.
            let sentinel = if (font.size_px() - 1.0).abs() < f64::EPSILON {
                "2px serif"
            } else {
                "1px serif"
            };
            self.ctx.set_font(sentinel);
            let applied_sentinel = self.ctx.font();
            self.ctx.set_font(font.as_css_str());
            let applied = self.ctx.font();
            if applied == applied_sentinel {
                return Err(Error::measurement(
                    "canvas",
                    format!(
                        "browser rejected validated font assignment {:?}",
                        font.as_css_str()
                    ),
                ));
            }
            self.current_font.clear();
            self.current_font.push_str(font.as_css_str());
        }
        Ok(())
    }

    fn measure_text_width(&mut self, text: &str, font: &FontSpec) -> Result<f64> {
        self.set_font(font)?;
        let metrics = self
            .ctx
            .measure_text(text)
            .map_err(|error| Error::measurement("canvas", js_error_message(&error)))?;
        validate_metric("canvas text width", metrics.width())
    }

    fn measure_segment_inner(&mut self, text: &str, font: &FontSpec) -> Result<SegmentMetrics> {
        let cache_key = (font.as_css_str().to_owned(), text.to_owned());
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut contains_cjk = false;
        let mut emoji_count = 0;
        let graphemes: Vec<&str> = text.graphemes(true).collect();

        for grapheme in &graphemes {
            let Some(c) = grapheme.chars().next() else {
                return Err(Error::measurement(
                    "canvas",
                    "unicode segmentation emitted an empty grapheme",
                ));
            };
            contains_cjk |= unicode::is_cjk(c);
            if unicode::is_emoji(c) {
                emoji_count += 1;
            }
        }

        // Whole-segment measurement preserves browser kerning and shaping.
        let width = self.measure_text_width(text, font)?;
        let grapheme_widths = if graphemes.len() > 1 {
            let mut widths = Vec::with_capacity(graphemes.len());
            for grapheme in &graphemes {
                widths.push(self.measure_text_width(grapheme, font)?);
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
        self.cache.insert(cache_key, metrics.clone());
        Ok(metrics)
    }
}

/// Canvas-based measurement backend for WASM.
///
/// Uses [`RefCell`] interior mutability because browser canvas contexts are
/// single-threaded. Contended or re-entrant access returns
/// [`Error::StateUnavailable`] instead of panicking.
pub struct CanvasBackend {
    inner: RefCell<CanvasInner>,
}

impl CanvasBackend {
    /// Create a canvas backend with [`DEFAULT_CACHE_CAPACITY`] entries.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Measurement`] if neither an offscreen nor DOM-backed
    /// 2D canvas context can be created.
    pub fn new() -> Result<Self> {
        Self::with_cache_limits(DEFAULT_CACHE_CAPACITY, DEFAULT_CACHE_BYTE_CAPACITY)
    }

    /// Create a canvas backend with a bounded measurement cache.
    ///
    /// The cache also remains subject to [`DEFAULT_CACHE_BYTE_CAPACITY`]. An
    /// entry too large for that byte budget is measured but not cached. An
    /// entry capacity of zero disables caching.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Measurement`] if neither an offscreen nor DOM-backed
    /// 2D canvas context can be created.
    pub fn with_cache_capacity(cache_capacity: usize) -> Result<Self> {
        Self::with_cache_limits(cache_capacity, DEFAULT_CACHE_BYTE_CAPACITY)
    }

    /// Create a canvas backend with explicit entry and retained-byte limits.
    ///
    /// Entries are evicted in insertion order until both limits are met. A
    /// measurement larger than the complete byte budget is returned to the
    /// caller but is not retained. Either limit being zero disables caching.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Measurement`] if neither an offscreen nor DOM-backed
    /// 2D canvas context can be created.
    pub fn with_cache_limits(cache_capacity: usize, cache_byte_capacity: usize) -> Result<Self> {
        let ctx = Self::create_context()?;
        Ok(Self {
            inner: RefCell::new(CanvasInner {
                ctx,
                cache: MeasurementCache::new(cache_capacity, cache_byte_capacity),
                current_font: String::new(),
            }),
        })
    }

    fn create_context() -> Result<CanvasContext> {
        let offscreen_failure = match OffscreenCanvas::new(1, 1) {
            Ok(offscreen) => match offscreen.get_context("2d") {
                Ok(Some(context)) => {
                    match context.dyn_into::<OffscreenCanvasRenderingContext2d>() {
                        Ok(context) => return Ok(CanvasContext::Offscreen(context)),
                        Err(error) => Some(js_error_message(&error)),
                    }
                }
                Ok(None) => Some("2D context was unavailable".to_owned()),
                Err(error) => Some(js_error_message(&error)),
            },
            Err(error) => Some(js_error_message(&error)),
        };

        let window = web_sys::window().ok_or_else(|| {
            Error::measurement(
                "canvas",
                format_canvas_init_failure(offscreen_failure.as_deref(), "no browser window"),
            )
        })?;
        let document = window.document().ok_or_else(|| {
            Error::measurement(
                "canvas",
                format_canvas_init_failure(offscreen_failure.as_deref(), "no browser document"),
            )
        })?;
        let element = document
            .create_element("canvas")
            .map_err(|error| Error::measurement("canvas", js_error_message(&error)))?;
        let canvas: HtmlCanvasElement = element.dyn_into().map_err(|element| {
            Error::measurement(
                "canvas",
                format!("created element was not a canvas: {element:?}"),
            )
        })?;
        canvas.set_width(1);
        canvas.set_height(1);

        let context = canvas
            .get_context("2d")
            .map_err(|error| Error::measurement("canvas", js_error_message(&error)))?
            .ok_or_else(|| Error::measurement("canvas", "2D context was unavailable"))?;
        context
            .dyn_into()
            .map(CanvasContext::Dom)
            .map_err(|error| Error::measurement("canvas", js_error_message(&error)))
    }

    /// Clear all retained measurements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::StateUnavailable`] during re-entrant access.
    pub fn clear_cache(&self) -> Result<()> {
        let mut inner = self
            .inner
            .try_borrow_mut()
            .map_err(|_| Error::StateUnavailable {
                state: "canvas measurement cache",
            })?;
        inner.cache.clear();
        Ok(())
    }

    fn try_inner_mut(&self) -> Result<std::cell::RefMut<'_, CanvasInner>> {
        self.inner
            .try_borrow_mut()
            .map_err(|_| Error::StateUnavailable {
                state: "canvas measurement backend",
            })
    }
}

impl MeasureBackend for CanvasBackend {
    fn measure_segment(&self, text: &str, font: &FontSpec) -> Result<SegmentMetrics> {
        self.try_inner_mut()?.measure_segment_inner(text, font)
    }

    fn measure_space_width(&self, font: &FontSpec) -> Result<f64> {
        self.try_inner_mut()?.measure_text_width(" ", font)
    }

    fn measure_hyphen_width(&self, font: &FontSpec) -> Result<f64> {
        self.try_inner_mut()?.measure_text_width("-", font)
    }
}

fn js_error_message(error: &JsValue) -> String {
    error.as_string().unwrap_or_else(|| format!("{error:?}"))
}

fn format_canvas_init_failure(offscreen: Option<&str>, dom: &str) -> String {
    match offscreen {
        Some(offscreen) => {
            format!("offscreen context failed ({offscreen}); DOM fallback failed ({dom})")
        }
        None => format!("DOM canvas initialization failed ({dom})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn metrics(width: f64) -> SegmentMetrics {
        SegmentMetrics {
            width,
            contains_cjk: false,
            emoji_count: 0,
            grapheme_widths: None,
        }
    }

    #[test]
    fn bounded_cache_evicts_oldest_entry() {
        let mut cache = MeasurementCache::new(2, usize::MAX);
        let first = ("16px A".to_owned(), "a".to_owned());
        let second = ("16px A".to_owned(), "b".to_owned());
        let third = ("16px A".to_owned(), "c".to_owned());
        cache.insert(first.clone(), metrics(1.0));
        cache.insert(second.clone(), metrics(2.0));
        cache.insert(third.clone(), metrics(3.0));

        assert!(cache.get(&first).is_none());
        assert_eq!(cache.get(&second).map(|m| m.width), Some(2.0));
        assert_eq!(cache.get(&third).map(|m| m.width), Some(3.0));
    }

    #[test]
    fn zero_capacity_disables_cache() {
        let mut cache = MeasurementCache::new(0, usize::MAX);
        let key = ("16px A".to_owned(), "a".to_owned());
        cache.insert(key.clone(), metrics(1.0));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn byte_capacity_evicts_oldest_entry() {
        let first = ("16px A".to_owned(), "a".to_owned());
        let second = ("16px A".to_owned(), "b".to_owned());
        let entry_bytes = estimated_retained_bytes(&first, &metrics(1.0));
        let mut cache = MeasurementCache::new(10, entry_bytes);

        cache.insert(first.clone(), metrics(1.0));
        cache.insert(second.clone(), metrics(2.0));

        assert!(cache.get(&first).is_none());
        assert_eq!(cache.get(&second).map(|m| m.width), Some(2.0));
        assert!(cache.retained_bytes <= cache.byte_capacity);
    }

    #[test]
    fn oversized_entry_is_not_cached() {
        let key = ("16px A".to_owned(), "large".to_owned());
        let value = SegmentMetrics {
            width: 10.0,
            contains_cjk: false,
            emoji_count: 0,
            grapheme_widths: Some(vec![1.0; 128]),
        };
        let required_bytes = estimated_retained_bytes(&key, &value);
        let mut cache = MeasurementCache::new(10, required_bytes.saturating_sub(1));

        cache.insert(key.clone(), value);

        assert!(cache.get(&key).is_none());
        assert_eq!(cache.retained_bytes, 0);
    }

    #[test]
    fn zero_byte_capacity_disables_cache() {
        let mut cache = MeasurementCache::new(10, 0);
        let key = ("16px A".to_owned(), "a".to_owned());
        cache.insert(key.clone(), metrics(1.0));
        assert!(cache.get(&key).is_none());
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn chrome_uses_the_native_offscreen_context_type() {
        assert!(matches!(
            CanvasBackend::create_context(),
            Ok(CanvasContext::Offscreen(_))
        ));
    }
}
