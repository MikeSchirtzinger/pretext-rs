# Changelog

## 0.2.0

Production-hardening release with intentional breaking API changes.

- Measurement, preparation, layout, streaming, inline-flow, GPU-layout, and
  WASM operations now return typed errors instead of silently substituting
  widths or relying on runtime panics.
- `FontSpec::new` now parses and validates supported CSS-style font syntax and
  returns `Result<FontSpec>`.
- Prepared representations and cursor fields are opaque. Read-only accessors
  replace direct mutation of layout invariants.
- The invalid `fontdue` backend was removed. The optional
  `SkrifaNominalBackend`/`skrifa-nominal` surface is explicitly limited to
  unshaped advance estimation and rejects missing glyphs; it is not presented
  as production typography.
- Preparation, browser caches, WASM batches, and the WASM prepared-text pool
  have explicit byte and structural resource limits.
- Inline flow now has aggregate item/text/segment limits and retains one
  prepared-data representation per item instead of cloning it.
- WASM preparation now requires explicit validated font and whitespace inputs,
  uses real Canvas measurement, throws stable coded `PretextError` values, and
  never reuses stale opaque handles.
- Canvas callers must acknowledge CSS fallback with an unquoted generic family
  and can explicitly invalidate measurements after browser font faces change.
- GPU multi-line layout now uses materialized pretext line ranges, validates
  derived geometry, rejects unsupported atlas glyphs, bounds output, and
  centers glyph quads and vertical baselines correctly.
- URL/date/control analysis, NBSP Glue, repeated mid-word wrapping, tab-stop
  geometry, discretionary hyphens, and maintained UAX #9 bidi resolution now have
  cross-surface regression coverage.
- Adversarial analysis passes are linear instead of repeatedly removing from
  the front of vectors.
- The crate now requires Rust 1.95 and enforces formatting, strict Clippy,
  strict rustdoc, dependency policy, native tests, and WASM compilation in CI.

Migration examples are shown in the README. In most callers, add `?` to
`FontSpec::new`, `prepare`, and `layout`, then replace direct cursor or atlas
field access with the corresponding getter.

## 0.1.0

Initial Rust port.
