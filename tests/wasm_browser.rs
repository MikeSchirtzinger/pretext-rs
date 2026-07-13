#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use pretext::backend::{CanvasBackend, FontSpec, MeasureBackend};
use pretext::types::PrepareOptions;
use pretext::wasm_api::{
    wasm_clear_measurement_cache, wasm_free, wasm_layout, wasm_layout_batch, wasm_layout_lines,
    wasm_prepare, wasm_prepare_batch,
};
use pretext::{layout, prepare};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn canvas_backend_measures_and_drives_layout_in_chrome() {
    let backend = CanvasBackend::new().expect("Chrome provides a usable 2D canvas context");
    let font = FontSpec::new("16px sans-serif").expect("test font specification is valid");

    let metrics = backend
        .measure_segment("Hello", &font)
        .expect("browser canvas measurement succeeds");
    assert!(metrics.width.is_finite());
    assert!(metrics.width > 0.0);
    assert_eq!(metrics.grapheme_widths.as_ref().map(Vec::len), Some(5));

    let prepared = prepare("hello world", &font, &backend, PrepareOptions::default())
        .expect("browser-backed preparation succeeds");
    let result = layout(&prepared, metrics.width, 20.0).expect("layout succeeds");
    assert!(result.line_count >= 2);

    backend.clear_cache().expect("cache remains available");

    let named_without_fallback = FontSpec::new("16px DefinitelyMissingFace")
        .expect("named native font specification is syntactically valid");
    assert!(matches!(
        backend.measure_segment("Hello", &named_without_fallback),
        Err(pretext::Error::InvalidFontSpec { .. })
    ));
    let quoted_generic = FontSpec::new("16px DefinitelyMissingFace, 'sans-serif'")
        .expect("quoted generic name is syntactically valid");
    assert!(matches!(
        backend.measure_segment("Hello", &quoted_generic),
        Err(pretext::Error::InvalidFontSpec { .. })
    ));
}

#[wasm_bindgen_test]
fn public_wasm_exports_use_real_canvas_measurement() {
    let handle = wasm_prepare("hello world", "16px sans-serif", "normal")
        .expect("public browser preparation succeeds");
    let line_count = wasm_layout(handle, 48.0, 20.0).expect("public browser layout succeeds");
    let lines_json = wasm_layout_lines("hello world", "16px sans-serif", "normal", 48.0)
        .expect("public line materialization succeeds");
    let lines: serde_json::Value =
        serde_json::from_str(&lines_json).expect("public line output is valid JSON");

    assert!(line_count >= 2);
    assert_eq!(lines.as_array().map(Vec::len), Some(line_count as usize));
    wasm_free(handle).expect("public handle can be freed");
    assert!(wasm_layout(handle, 48.0, 20.0).is_err());
    wasm_clear_measurement_cache().expect("public cache invalidation succeeds");
}

fn error_code(error: &wasm_bindgen::JsValue) -> Option<String> {
    js_sys::Reflect::get(error, &wasm_bindgen::JsValue::from_str("code"))
        .ok()
        .and_then(|value| value.as_string())
}

#[wasm_bindgen_test]
fn public_wasm_errors_have_stable_codes_and_details() {
    let malformed = wasm_prepare_batch("not json", "16px sans-serif", "normal")
        .expect_err("malformed JSON is rejected");
    assert_eq!(error_code(&malformed).as_deref(), Some("invalid_input"));

    let invalid_empty_batch = wasm_layout_batch("", f64::NAN, f64::NAN)
        .expect_err("empty batches still validate geometry");
    assert_eq!(
        error_code(&invalid_empty_batch).as_deref(),
        Some("invalid_input")
    );

    let invalid_handle =
        wasm_layout(u32::MAX, 100.0, 20.0).expect_err("unknown handle is rejected");
    assert_eq!(
        error_code(&invalid_handle).as_deref(),
        Some("invalid_handle")
    );
    assert_eq!(
        js_sys::Reflect::get(&invalid_handle, &wasm_bindgen::JsValue::from_str("handle"))
            .ok()
            .and_then(|value| value.as_f64()),
        Some(f64::from(u32::MAX))
    );

    let oversized = "a".repeat(65_537);
    let complexity = wasm_prepare(&oversized, "16px sans-serif", "normal")
        .expect_err("grapheme limit is enforced");
    assert_eq!(error_code(&complexity).as_deref(), Some("input_complexity"));

    let handle =
        wasm_prepare("once", "16px sans-serif", "normal").expect("test handle is prepared");
    wasm_free(handle).expect("first free succeeds");
    let double_free = wasm_free(handle).expect_err("double free is rejected");
    assert_eq!(error_code(&double_free).as_deref(), Some("invalid_handle"));
}

#[wasm_bindgen_test]
fn public_wasm_font_argument_changes_real_measurement() {
    let small_json = wasm_layout_lines("MMMM", "10px monospace", "normal", 1_000.0)
        .expect("small font layout succeeds");
    let large_json = wasm_layout_lines("MMMM", "30px monospace", "normal", 1_000.0)
        .expect("large font layout succeeds");
    let small: serde_json::Value = serde_json::from_str(&small_json).expect("valid JSON");
    let large: serde_json::Value = serde_json::from_str(&large_json).expect("valid JSON");
    let small_width = small
        .get(0)
        .and_then(|line| line.get("width"))
        .and_then(serde_json::Value::as_f64)
        .expect("small width is numeric");
    let large_width = large
        .get(0)
        .and_then(|line| line.get("width"))
        .and_then(serde_json::Value::as_f64)
        .expect("large width is numeric");

    assert!(large_width > small_width * 2.5);
}
