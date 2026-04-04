//! WASM API — thin wasm-bindgen exports for the benchmark demo.
//!
//! Exposes `prepare()` and `layout()` as JS-callable functions with
//! opaque handles for prepared text. The prepared text is stored in
//! a global pool keyed by integer handle.
#![cfg(feature = "wasm")]

use std::cell::RefCell;
use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use crate::backend::fixed::FixedWidthBackend;
use crate::backend::FontSpec;
use crate::types::{PreparedText, PrepareOptions};
use crate::{layout as layout_fn, prepare as prepare_fn};

thread_local! {
    static POOL: RefCell<HashMap<u32, PreparedText>> = RefCell::new(HashMap::new());
    static NEXT_ID: RefCell<u32> = const { RefCell::new(1) };
    static BACKEND: FixedWidthBackend = const { FixedWidthBackend {
        char_width_factor: 0.6,
        cjk_width_factor: 1.0,
        space_width_factor: 0.25,
        default_font_size: 16.0,
    }};
    static FONT: FontSpec = FontSpec { font: String::new() };
}

/// Prepare text for layout. Returns an opaque handle (u32).
///
/// Call `wasm_layout(handle, max_width, line_height)` with the returned handle.
/// Call `wasm_free(handle)` when done to release memory.
#[wasm_bindgen(js_name = "pretextPrepare")]
pub fn wasm_prepare(text: &str) -> u32 {
    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter");
    let prepared = prepare_fn(text, &font, &backend, PrepareOptions::default());

    NEXT_ID.with(|id| {
        let handle = *id.borrow();
        *id.borrow_mut() = handle + 1;
        POOL.with(|pool| {
            pool.borrow_mut().insert(handle, prepared);
        });
        handle
    })
}

/// Layout prepared text at the given width. Returns line count.
///
/// This is the hot path — pure arithmetic over cached segment widths.
#[wasm_bindgen(js_name = "pretextLayout")]
pub fn wasm_layout(handle: u32, max_width: f64, line_height: f64) -> u32 {
    POOL.with(|pool| {
        let pool = pool.borrow();
        if let Some(prepared) = pool.get(&handle) {
            let result = layout_fn(prepared, max_width, line_height);
            result.line_count as u32
        } else {
            0
        }
    })
}

/// Free a prepared text handle.
#[wasm_bindgen(js_name = "pretextFree")]
pub fn wasm_free(handle: u32) {
    POOL.with(|pool| {
        pool.borrow_mut().remove(&handle);
    });
}

/// Prepare and layout in one call (for comparison benchmarks).
/// Returns line count.
#[wasm_bindgen(js_name = "pretextPrepareAndLayout")]
pub fn wasm_prepare_and_layout(text: &str, max_width: f64, line_height: f64) -> u32 {
    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter");
    let prepared = prepare_fn(text, &font, &backend, PrepareOptions::default());
    let result = layout_fn(&prepared, max_width, line_height);
    result.line_count as u32
}

/// Batch prepare: prepare multiple texts, return handles as a comma-separated string.
#[wasm_bindgen(js_name = "pretextPrepareBatch")]
pub fn wasm_prepare_batch(texts_json: &str) -> String {
    // Parse JSON array of strings
    let texts: Vec<String> = match serde_json::from_str(texts_json) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };

    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter");
    let mut handles = Vec::with_capacity(texts.len());

    for text in &texts {
        let prepared = prepare_fn(text, &font, &backend, PrepareOptions::default());
        let handle = NEXT_ID.with(|id| {
            let h = *id.borrow();
            *id.borrow_mut() = h + 1;
            POOL.with(|pool| pool.borrow_mut().insert(h, prepared));
            h
        });
        handles.push(handle.to_string());
    }

    handles.join(",")
}

/// Batch layout: layout all handles at the given width. Returns total line count.
#[wasm_bindgen(js_name = "pretextLayoutBatch")]
pub fn wasm_layout_batch(handles_csv: &str, max_width: f64, line_height: f64) -> u32 {
    let mut total = 0u32;
    POOL.with(|pool| {
        let pool = pool.borrow();
        for h_str in handles_csv.split(',') {
            if let Ok(handle) = h_str.trim().parse::<u32>() {
                if let Some(prepared) = pool.get(&handle) {
                    total += layout_fn(prepared, max_width, line_height).line_count as u32;
                }
            }
        }
    });
    total
}

/// Free all handles from a batch.
#[wasm_bindgen(js_name = "pretextFreeBatch")]
pub fn wasm_free_batch(handles_csv: &str) {
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for h_str in handles_csv.split(',') {
            if let Ok(handle) = h_str.trim().parse::<u32>() {
                pool.remove(&handle);
            }
        }
    });
}
