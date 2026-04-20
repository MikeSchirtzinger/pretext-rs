//! Public-surface parity with upstream JS.
//!
//! Upstream exports: clearCache, clearAnalysisCaches, clearMeasurementCaches,
//! setLocale, setAnalysisLocale, profilePrepare. These tests verify the
//! equivalent Rust APIs exist and behave reasonably. Implementations may be
//! thin (no-op for caches we don't have, locale stored but not consulted);
//! the goal is API surface parity so downstream users don't have to #[cfg]
//! around missing symbols.

use pretext::backend::{fixed::FixedWidthBackend, FontSpec};

// ---- Cache clearers --------------------------------------------------------

#[test]
fn clear_cache_is_callable() {
    // No return, no panics. Idempotent.
    pretext::clear_cache();
    pretext::clear_cache();
}

#[test]
fn clear_analysis_caches_is_callable() {
    pretext::clear_analysis_caches();
}

#[test]
fn clear_measurement_caches_is_callable() {
    pretext::clear_measurement_caches();
}

#[test]
fn clear_cache_does_not_corrupt_subsequent_prepare() {
    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter");

    let before = pretext::prepare("Hello world", &font, &backend, Default::default());
    let before_layout = pretext::layout(&before, 200.0, 24.0);

    pretext::clear_cache();

    let after = pretext::prepare("Hello world", &font, &backend, Default::default());
    let after_layout = pretext::layout(&after, 200.0, 24.0);

    assert_eq!(before_layout.line_count, after_layout.line_count);
    assert!((before_layout.height - after_layout.height).abs() < 1e-9);
}

// ---- Locale setters --------------------------------------------------------

#[test]
fn set_locale_accepts_some_and_none() {
    pretext::set_locale(Some("en"));
    pretext::set_locale(Some("ja"));
    pretext::set_locale(None);
}

#[test]
fn set_analysis_locale_accepts_some_and_none() {
    pretext::set_analysis_locale(Some("en"));
    pretext::set_analysis_locale(None);
}

#[test]
fn set_locale_does_not_break_layout() {
    pretext::set_locale(Some("ja"));

    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter");
    let prepared = pretext::prepare("Hello world", &font, &backend, Default::default());
    let result = pretext::layout(&prepared, 200.0, 24.0);

    assert!(result.line_count >= 1);

    // Restore default so other tests aren't affected.
    pretext::set_locale(None);
}

// ---- profile_prepare -------------------------------------------------------

#[test]
fn profile_prepare_returns_populated_profile() {
    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter");

    let profile = pretext::profile_prepare(
        "The quick brown fox jumps over the lazy dog.",
        &font,
        &backend,
        Default::default(),
    );

    // Timings must be non-negative (may legitimately be 0 on very fast runs).
    assert!(profile.analysis_ms >= 0.0);
    assert!(profile.measure_ms >= 0.0);
    assert!(profile.total_ms >= 0.0);
    // total should be >= sum of parts minus float slop.
    let sum = profile.analysis_ms + profile.measure_ms;
    assert!(
        profile.total_ms + 1e-3 >= sum,
        "total {} < analysis+measure {}",
        profile.total_ms,
        sum
    );

    // Counts reflect the fixture: multiple segments, several of which are
    // word-like and thus candidates for grapheme-breakable measurement.
    assert!(profile.analysis_segments > 0);
    assert!(profile.prepared_segments > 0);
    // breakable_segments may be 0 if no segment exceeds its own width — that's
    // backend-dependent. Just assert it's ≤ prepared_segments.
    assert!(profile.breakable_segments <= profile.prepared_segments);
}

#[test]
fn profile_prepare_empty_text() {
    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter");

    let profile = pretext::profile_prepare("", &font, &backend, Default::default());
    assert_eq!(profile.analysis_segments, 0);
    assert_eq!(profile.prepared_segments, 0);
    assert_eq!(profile.breakable_segments, 0);
}
