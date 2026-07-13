//! Fallible WASM bindings backed by a bounded prepared-text pool.
#![cfg(feature = "wasm")]

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use unicode_segmentation::UnicodeSegmentation;
use wasm_bindgen::prelude::*;

use crate::backend::{CanvasBackend, FontSpec};
use crate::types::{
    DEFAULT_MAX_GRAPHEMES, DEFAULT_MAX_INPUT_BYTES, PrepareOptions, PreparedText, WhiteSpaceMode,
};
use crate::{Error, Result, layout, layout_with_lines, prepare, prepare_with_segments};

const MAX_POOL_ENTRIES: usize = 16_384;
const MAX_BATCH_ITEMS: usize = 1_024;
const MAX_BATCH_INPUT_BYTES: usize = DEFAULT_MAX_INPUT_BYTES;
const MAX_BATCH_GRAPHEMES: usize = DEFAULT_MAX_GRAPHEMES;
const MAX_POOL_BYTES: usize = 64 * 1024 * 1024;
const MAX_HANDLES_CSV_BYTES: usize = MAX_BATCH_ITEMS * 11;
const MAX_FONT_SPEC_BYTES: usize = 4_096;

struct WasmContext {
    backend: CanvasBackend,
}

impl WasmContext {
    fn new() -> Result<Self> {
        Ok(Self {
            backend: CanvasBackend::new()?,
        })
    }
}

struct StoredPrepared {
    prepared: PreparedText,
    charged_bytes: usize,
}

struct PreparedPool {
    entries: HashMap<u32, StoredPrepared>,
    next_id: u32,
    charged_bytes: usize,
}

impl Default for PreparedPool {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            next_id: 1,
            charged_bytes: 0,
        }
    }
}

impl PreparedPool {
    fn validate_reservation(&self, entry_count: usize, charged_bytes: usize) -> Result<usize> {
        let requested_entries =
            self.entries
                .len()
                .checked_add(entry_count)
                .ok_or(Error::ArithmeticOverflow {
                    operation: "reserving WASM pool entries",
                })?;
        if requested_entries > MAX_POOL_ENTRIES {
            return Err(Error::PoolExhausted {
                capacity: MAX_POOL_ENTRIES,
            });
        }

        let requested_bytes =
            self.charged_bytes
                .checked_add(charged_bytes)
                .ok_or(Error::ArithmeticOverflow {
                    operation: "reserving WASM pool memory",
                })?;
        if requested_bytes > MAX_POOL_BYTES {
            return Err(Error::ResourceLimit {
                resource: "WASM prepared-text pool",
                requested_bytes,
                max_bytes: MAX_POOL_BYTES,
            });
        }
        Ok(requested_bytes)
    }

    fn reserve_handles(&self, count: usize) -> Result<(Vec<u32>, u32)> {
        if count == 0 {
            return Ok((Vec::new(), self.next_id));
        }
        if self.next_id == 0 {
            return Err(Error::IdentifierExhausted {
                resource: "WASM prepared-text handle",
            });
        }

        let count_minus_one =
            u32::try_from(count.saturating_sub(1)).map_err(|_| Error::IdentifierExhausted {
                resource: "WASM prepared-text handle",
            })?;
        let last = self
            .next_id
            .checked_add(count_minus_one)
            .ok_or(Error::IdentifierExhausted {
                resource: "WASM prepared-text handle",
            })?;
        let mut handles = Vec::with_capacity(count);
        let mut candidate = self.next_id;
        while candidate <= last {
            if self.entries.contains_key(&candidate) {
                return Err(Error::StateUnavailable {
                    state: "WASM monotonic handle allocator",
                });
            }
            handles.push(candidate);
            let Some(next) = candidate.checked_add(1) else {
                break;
            };
            candidate = next;
        }
        let next_id = last.checked_add(1).unwrap_or(0);
        Ok((handles, next_id))
    }

    fn insert(&mut self, prepared: PreparedText, input_bytes: usize) -> Result<u32> {
        let mut handles = self.insert_batch(vec![(prepared, input_bytes)])?;
        handles.pop().ok_or(Error::StateUnavailable {
            state: "WASM handle allocator",
        })
    }

    fn insert_batch(&mut self, values: Vec<(PreparedText, usize)>) -> Result<Vec<u32>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let charged_bytes = values
            .iter()
            .try_fold(0_usize, |total, (prepared, input)| {
                total
                    .checked_add(prepared.estimated_heap_bytes())
                    .and_then(|value| value.checked_add(*input))
                    .ok_or(Error::ArithmeticOverflow {
                        operation: "estimating WASM prepared-text memory",
                    })
            })?;
        let requested_bytes = self.validate_reservation(values.len(), charged_bytes)?;
        let (handles, next_id) = self.reserve_handles(values.len())?;
        let previous_next_id = self.next_id;
        let mut inserted = Vec::with_capacity(handles.len());

        for (handle, (prepared, input_bytes)) in handles.iter().copied().zip(values) {
            let entry_charge = prepared
                .estimated_heap_bytes()
                .checked_add(input_bytes)
                .ok_or(Error::ArithmeticOverflow {
                    operation: "charging WASM prepared-text memory",
                })?;
            match self.entries.entry(handle) {
                Entry::Vacant(entry) => {
                    entry.insert(StoredPrepared {
                        prepared,
                        charged_bytes: entry_charge,
                    });
                    inserted.push(handle);
                }
                Entry::Occupied(_) => {
                    for inserted_handle in inserted {
                        self.entries.remove(&inserted_handle);
                    }
                    self.next_id = previous_next_id;
                    return Err(Error::StateUnavailable {
                        state: "WASM handle allocator",
                    });
                }
            }
        }

        self.charged_bytes = requested_bytes;
        self.next_id = next_id;
        Ok(handles)
    }

    fn get(&self, handle: u32) -> Result<&PreparedText> {
        self.entries
            .get(&handle)
            .map(|stored| &stored.prepared)
            .ok_or(Error::InvalidHandle { handle })
    }

    fn remove(&mut self, handle: u32) -> Result<()> {
        let entry_charge = self
            .entries
            .get(&handle)
            .ok_or(Error::InvalidHandle { handle })?
            .charged_bytes;
        let remaining_bytes =
            self.charged_bytes
                .checked_sub(entry_charge)
                .ok_or(Error::StateUnavailable {
                    state: "WASM prepared-text memory accounting",
                })?;
        self.entries.remove(&handle);
        self.charged_bytes = remaining_bytes;
        Ok(())
    }
}

thread_local! {
    static CONTEXT: Result<WasmContext> = WasmContext::new();
    static POOL: RefCell<PreparedPool> = RefCell::new(PreparedPool {
        entries: HashMap::new(),
        next_id: 1,
        charged_bytes: 0,
    });
}

fn js_error(error: Error) -> JsValue {
    let message = error.to_string();
    let js_error = js_sys::Error::new(&message);
    js_error.set_name("PretextError");
    let value = JsValue::from(js_error);
    let code = error_code(&error);

    if !set_js_property(&value, "code", &JsValue::from_str(code))
        || !set_js_property(&value, "message", &JsValue::from_str(&message))
    {
        return JsValue::from(js_sys::Error::new(&format!("{code}: {message}")));
    }
    attach_error_details(&value, &error);
    value
}

fn error_code(error: &Error) -> &'static str {
    match error {
        Error::InvalidFontSpec { .. } => "invalid_font_spec",
        Error::MissingFont { .. } => "missing_font",
        Error::MissingGlyph { .. } => "missing_glyph",
        Error::Measurement { .. } => "measurement_failed",
        Error::InvalidMetric { .. } => "invalid_metric",
        Error::InvalidInput { .. } => "invalid_input",
        Error::InputTooLarge { .. } => "input_too_large",
        Error::InputComplexity { .. } => "input_complexity",
        Error::InvalidCursor { .. } => "invalid_cursor",
        Error::InvalidBidiStart { .. } => "invalid_bidi_start",
        Error::UnsupportedGlyph { .. } => "unsupported_glyph",
        Error::InvalidHandle { .. } => "invalid_handle",
        Error::PoolExhausted { .. } => "pool_exhausted",
        Error::IdentifierExhausted { .. } => "identifier_exhausted",
        Error::ResourceLimit { .. } => "resource_limit",
        Error::StateUnavailable { .. } => "state_unavailable",
        Error::ArithmeticOverflow { .. } => "arithmetic_overflow",
    }
}

fn set_js_property(target: &JsValue, name: &str, value: &JsValue) -> bool {
    js_sys::Reflect::set(target, &JsValue::from_str(name), value).unwrap_or(false)
}

#[allow(clippy::cast_precision_loss)]
fn set_js_usize(target: &JsValue, name: &str, value: usize) {
    let _ = set_js_property(target, name, &JsValue::from_f64(value as f64));
}

fn attach_error_details(target: &JsValue, error: &Error) {
    match error {
        Error::InvalidHandle { handle } => {
            let _ = set_js_property(target, "handle", &JsValue::from_f64(f64::from(*handle)));
        }
        Error::InputTooLarge { bytes, max_bytes } => {
            set_js_usize(target, "bytes", *bytes);
            set_js_usize(target, "maxBytes", *max_bytes);
        }
        Error::InputComplexity {
            resource,
            units,
            max_units,
        } => {
            let _ = set_js_property(target, "resource", &JsValue::from_str(resource));
            set_js_usize(target, "units", *units);
            set_js_usize(target, "maxUnits", *max_units);
        }
        Error::PoolExhausted { capacity } => set_js_usize(target, "capacity", *capacity),
        Error::ResourceLimit {
            resource,
            requested_bytes,
            max_bytes,
        } => {
            let _ = set_js_property(target, "resource", &JsValue::from_str(resource));
            set_js_usize(target, "requestedBytes", *requested_bytes);
            set_js_usize(target, "maxBytes", *max_bytes);
        }
        _ => {}
    }
}

fn with_context<T>(operation: impl FnOnce(&CanvasBackend) -> Result<T>) -> Result<T> {
    CONTEXT.with(|context| match context {
        Ok(context) => operation(&context.backend),
        Err(error) => Err(error.clone()),
    })
}

fn parse_font(font_css: &str) -> Result<FontSpec> {
    if font_css.len() > MAX_FONT_SPEC_BYTES {
        return Err(Error::InputTooLarge {
            bytes: font_css.len(),
            max_bytes: MAX_FONT_SPEC_BYTES,
        });
    }
    FontSpec::new(font_css)
}

fn parse_white_space(white_space: &str) -> Result<WhiteSpaceMode> {
    match white_space {
        "normal" => Ok(WhiteSpaceMode::Normal),
        "pre-wrap" => Ok(WhiteSpaceMode::PreWrap),
        _ => Err(Error::invalid_input(
            "white_space",
            "expected \"normal\" or \"pre-wrap\"",
        )),
    }
}

fn prepare_options(white_space: &str) -> Result<PrepareOptions> {
    Ok(PrepareOptions {
        white_space: parse_white_space(white_space)?,
        ..PrepareOptions::default()
    })
}

fn validate_layout_arguments(max_width: f64, line_height: f64) -> Result<()> {
    if max_width.is_nan() || max_width < 0.0 {
        return Err(Error::invalid_input(
            "max_width",
            "must be non-negative and not NaN",
        ));
    }
    if !line_height.is_finite() || line_height <= 0.0 {
        return Err(Error::invalid_input(
            "line_height",
            "must be finite and greater than zero",
        ));
    }
    Ok(())
}

fn parse_handles(handles_csv: &str) -> Result<Vec<u32>> {
    if handles_csv.len() > MAX_HANDLES_CSV_BYTES {
        return Err(Error::InputTooLarge {
            bytes: handles_csv.len(),
            max_bytes: MAX_HANDLES_CSV_BYTES,
        });
    }
    if handles_csv.trim().is_empty() {
        return Ok(Vec::new());
    }
    let handles: Result<Vec<u32>> = handles_csv
        .split(',')
        .map(|raw| {
            raw.trim().parse::<u32>().map_err(|error| {
                Error::invalid_input("handles_csv", format!("invalid handle {raw:?}: {error}"))
            })
        })
        .collect();
    let handles = handles?;
    if handles.len() > MAX_BATCH_ITEMS {
        return Err(Error::invalid_input(
            "handles_csv",
            format!("batch exceeds {MAX_BATCH_ITEMS} handles"),
        ));
    }
    Ok(handles)
}

/// Prepare text and return an opaque bounded-pool handle.
///
/// # Errors
///
/// Throws a JavaScript error when preparation fails or the handle pool is full.
#[wasm_bindgen(js_name = "pretextPrepare")]
pub fn wasm_prepare(
    text: &str,
    font_css: &str,
    white_space: &str,
) -> std::result::Result<u32, JsValue> {
    let font = parse_font(font_css).map_err(js_error)?;
    let options = prepare_options(white_space).map_err(js_error)?;
    let prepared =
        with_context(|backend| prepare(text, &font, backend, options)).map_err(js_error)?;
    POOL.with(|pool| {
        pool.try_borrow_mut()
            .map_err(|_| Error::StateUnavailable {
                state: "WASM prepared-text pool",
            })?
            .insert(prepared, text.len())
    })
    .map_err(js_error)
}

/// Layout prepared text and return its line count.
///
/// # Errors
///
/// Throws a JavaScript error for an invalid handle or invalid geometry.
#[wasm_bindgen(js_name = "pretextLayout")]
pub fn wasm_layout(
    handle: u32,
    max_width: f64,
    line_height: f64,
) -> std::result::Result<u32, JsValue> {
    POOL.with(|pool| {
        let pool = pool.try_borrow().map_err(|_| Error::StateUnavailable {
            state: "WASM prepared-text pool",
        })?;
        let line_count = layout(pool.get(handle)?, max_width, line_height)?.line_count;
        u32::try_from(line_count).map_err(|_| Error::ArithmeticOverflow {
            operation: "converting line count to u32",
        })
    })
    .map_err(js_error)
}

/// Free a prepared-text handle.
///
/// # Errors
///
/// Throws a JavaScript error when the handle is unknown or already freed.
#[wasm_bindgen(js_name = "pretextFree")]
pub fn wasm_free(handle: u32) -> std::result::Result<(), JsValue> {
    POOL.with(|pool| {
        pool.try_borrow_mut()
            .map_err(|_| Error::StateUnavailable {
                state: "WASM prepared-text pool",
            })?
            .remove(handle)
    })
    .map_err(js_error)
}

/// Clear the browser canvas measurement cache.
///
/// Call this after the browser's available font faces change, then free and
/// re-prepare affected handles. Existing prepared values are immutable and do
/// not change retroactively.
///
/// # Errors
///
/// Throws a JavaScript error when the canvas backend is unavailable or its
/// state is currently borrowed.
#[wasm_bindgen(js_name = "pretextClearMeasurementCache")]
pub fn wasm_clear_measurement_cache() -> std::result::Result<(), JsValue> {
    with_context(CanvasBackend::clear_cache).map_err(js_error)
}

/// Prepare and layout text in one call.
///
/// # Errors
///
/// Throws a JavaScript error when preparation or layout fails.
#[wasm_bindgen(js_name = "pretextPrepareAndLayout")]
pub fn wasm_prepare_and_layout(
    text: &str,
    font_css: &str,
    white_space: &str,
    max_width: f64,
    line_height: f64,
) -> std::result::Result<u32, JsValue> {
    let font = parse_font(font_css).map_err(js_error)?;
    let options = prepare_options(white_space).map_err(js_error)?;
    let line_count = with_context(|backend| {
        let prepared = prepare(text, &font, backend, options)?;
        Ok(layout(&prepared, max_width, line_height)?.line_count)
    })
    .map_err(js_error)?;
    u32::try_from(line_count)
        .map_err(|_| Error::ArithmeticOverflow {
            operation: "converting line count to u32",
        })
        .map_err(js_error)
}

/// Prepare a JSON array of strings and return comma-separated handles.
///
/// The complete batch is prepared before the pool is mutated.
///
/// # Errors
///
/// Throws a JavaScript error for malformed JSON, oversized batches,
/// preparation failures, or insufficient pool capacity.
#[wasm_bindgen(js_name = "pretextPrepareBatch")]
pub fn wasm_prepare_batch(
    texts_json: &str,
    font_css: &str,
    white_space: &str,
) -> std::result::Result<String, JsValue> {
    if texts_json.len() > MAX_BATCH_INPUT_BYTES {
        return Err(js_error(Error::InputTooLarge {
            bytes: texts_json.len(),
            max_bytes: MAX_BATCH_INPUT_BYTES,
        }));
    }
    let texts: Vec<String> = serde_json::from_str(texts_json)
        .map_err(|error| Error::invalid_input("texts_json", error.to_string()))
        .map_err(js_error)?;
    if texts.len() > MAX_BATCH_ITEMS {
        return Err(js_error(Error::invalid_input(
            "texts_json",
            format!("batch exceeds {MAX_BATCH_ITEMS} texts"),
        )));
    }

    let batch_graphemes = texts
        .iter()
        .try_fold(0_usize, |total, text| {
            total
                .checked_add(text.graphemes(true).count())
                .ok_or(Error::ArithmeticOverflow {
                    operation: "counting WASM batch graphemes",
                })
        })
        .map_err(js_error)?;
    if batch_graphemes > MAX_BATCH_GRAPHEMES {
        return Err(js_error(Error::InputComplexity {
            resource: "WASM batch graphemes",
            units: batch_graphemes,
            max_units: MAX_BATCH_GRAPHEMES,
        }));
    }

    let font = parse_font(font_css).map_err(js_error)?;
    let options = prepare_options(white_space).map_err(js_error)?;

    let prepared = with_context(|backend| {
        texts
            .iter()
            .map(|text| prepare(text, &font, backend, options.clone()))
            .collect::<Result<Vec<PreparedText>>>()
    })
    .map_err(js_error)?;

    POOL.with(|pool| {
        let mut pool = pool.try_borrow_mut().map_err(|_| Error::StateUnavailable {
            state: "WASM prepared-text pool",
        })?;
        let values = prepared
            .into_iter()
            .zip(texts.iter().map(String::len))
            .collect();
        pool.insert_batch(values).map(|handles| {
            handles
                .into_iter()
                .map(|handle| handle.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
    })
    .map_err(js_error)
}

/// Layout a complete comma-separated batch and return the total line count.
///
/// # Errors
///
/// Throws a JavaScript error if any handle is malformed, unknown, or cannot be
/// laid out. Partial success is never returned.
#[wasm_bindgen(js_name = "pretextLayoutBatch")]
pub fn wasm_layout_batch(
    handles_csv: &str,
    max_width: f64,
    line_height: f64,
) -> std::result::Result<u32, JsValue> {
    validate_layout_arguments(max_width, line_height).map_err(js_error)?;
    let handles = parse_handles(handles_csv).map_err(js_error)?;
    POOL.with(|pool| {
        let pool = pool.try_borrow().map_err(|_| Error::StateUnavailable {
            state: "WASM prepared-text pool",
        })?;
        let mut total = 0_usize;
        for handle in handles {
            total = total
                .checked_add(layout(pool.get(handle)?, max_width, line_height)?.line_count)
                .ok_or(Error::ArithmeticOverflow {
                    operation: "summing batch line counts",
                })?;
        }
        u32::try_from(total).map_err(|_| Error::ArithmeticOverflow {
            operation: "converting batch line count to u32",
        })
    })
    .map_err(js_error)
}

/// Prepare and lay out text, returning a JSON array of line objects.
///
/// # Errors
///
/// Throws a JavaScript error when preparation, layout, or JSON serialization
/// fails.
#[wasm_bindgen(js_name = "pretextLayoutLines")]
pub fn wasm_layout_lines(
    text: &str,
    font_css: &str,
    white_space: &str,
    max_width: f64,
) -> std::result::Result<String, JsValue> {
    let font = parse_font(font_css).map_err(js_error)?;
    let options = prepare_options(white_space).map_err(js_error)?;
    let values = with_context(|backend| {
        let prepared = prepare_with_segments(text, &font, backend, options)?;
        Ok(layout_with_lines(&prepared, max_width)?
            .into_iter()
            .map(|line| serde_json::json!({ "text": line.text, "width": line.width }))
            .collect::<Vec<serde_json::Value>>())
    })
    .map_err(js_error)?;
    serde_json::to_string(&values)
        .map_err(|error| Error::measurement("JSON serializer", error.to_string()))
        .map_err(js_error)
}

/// Free a complete comma-separated batch of handles.
///
/// # Errors
///
/// Throws a JavaScript error when any handle is malformed, duplicated,
/// unknown, or already freed. Validation happens before mutation.
#[wasm_bindgen(js_name = "pretextFreeBatch")]
pub fn wasm_free_batch(handles_csv: &str) -> std::result::Result<(), JsValue> {
    let handles = parse_handles(handles_csv).map_err(js_error)?;
    let unique: HashSet<u32> = handles.iter().copied().collect();
    if unique.len() != handles.len() {
        return Err(js_error(Error::invalid_input(
            "handles_csv",
            "duplicate handles are not allowed",
        )));
    }

    POOL.with(|pool| {
        let mut pool = pool.try_borrow_mut().map_err(|_| Error::StateUnavailable {
            state: "WASM prepared-text pool",
        })?;
        for &handle in &handles {
            pool.get(handle)?;
        }
        for handle in handles {
            pool.remove(handle)?;
        }
        Ok(())
    })
    .map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::fixed::FixedWidthBackend;

    fn prepared() -> PreparedText {
        let backend = FixedWidthBackend::new();
        let font = FontSpec::new("16px Inter").expect("fixed test font is valid");
        prepare("hello", &font, &backend, PrepareOptions::default())
            .expect("test preparation succeeds")
    }

    #[test]
    fn pool_exhausts_identifier_space_without_reusing_handles() {
        let value = prepared();
        let mut pool = PreparedPool {
            entries: HashMap::new(),
            next_id: u32::MAX,
            charged_bytes: 0,
        };
        let max = pool
            .insert(value.clone(), 5)
            .expect("first insertion succeeds");
        let exhausted = pool.insert(value, 5);

        assert_eq!(max, u32::MAX);
        assert!(pool.get(max).is_ok());
        assert!(matches!(
            exhausted,
            Err(Error::IdentifierExhausted {
                resource: "WASM prepared-text handle"
            })
        ));
    }

    #[test]
    fn pool_rejects_unknown_and_double_freed_handles() {
        let mut pool = PreparedPool::default();
        let handle = pool.insert(prepared(), 5).expect("insertion succeeds");
        assert!(pool.remove(handle).is_ok());
        assert!(matches!(
            pool.remove(handle),
            Err(Error::InvalidHandle { handle: rejected }) if rejected == handle
        ));
    }

    #[test]
    fn handle_parser_rejects_partial_batches() {
        assert!(parse_handles("1,nope,3").is_err());
        assert_eq!(
            parse_handles("1, 2, 3").expect("valid handles"),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn removing_an_entry_releases_its_memory_charge() {
        let mut pool = PreparedPool::default();
        let handle = pool.insert(prepared(), 5).expect("insertion succeeds");
        let charged = pool.charged_bytes;
        assert!(charged > 5);
        pool.remove(handle).expect("removal succeeds");
        assert_eq!(pool.charged_bytes, 0);
    }

    #[test]
    fn rejected_batch_does_not_mutate_pool() {
        let mut pool = PreparedPool::default();
        let existing = pool.insert(prepared(), 5).expect("insertion succeeds");
        let entry_count = pool.entries.len();
        let charged_bytes = pool.charged_bytes;

        let rejected = pool.insert_batch(vec![(prepared(), MAX_POOL_BYTES)]);

        assert!(matches!(rejected, Err(Error::ResourceLimit { .. })));
        assert_eq!(pool.entries.len(), entry_count);
        assert_eq!(pool.charged_bytes, charged_bytes);
        assert!(pool.get(existing).is_ok());
    }
}
