//! Public diagnostic-surface tests.

use pretext::backend::{FontSpec, fixed::FixedWidthBackend};

#[track_caller]
fn valid<T>(result: pretext::Result<T>) -> T {
    result.expect("test input is valid")
}

// ---- profile_prepare -------------------------------------------------------

#[test]
fn profile_prepare_returns_populated_profile() {
    let backend = FixedWidthBackend::new();
    let font = valid(FontSpec::new("16px Inter"));

    let profile = valid(pretext::profile_prepare(
        "The quick brown fox jumps over the lazy dog.",
        &font,
        &backend,
        Default::default(),
    ));

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
    let font = valid(FontSpec::new("16px Inter"));

    let profile = valid(pretext::profile_prepare(
        "",
        &font,
        &backend,
        Default::default(),
    ));
    assert_eq!(profile.analysis_segments, 0);
    assert_eq!(profile.prepared_segments, 0);
    assert_eq!(profile.breakable_segments, 0);
}
