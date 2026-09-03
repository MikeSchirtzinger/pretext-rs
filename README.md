# pretext-rs

A Rust port of [**@chenglou/pretext**](https://github.com/chenglou/pretext): Cheng Lou's DOM-free text measurement and line-breaking engine.

**All credit for the algorithm, design, and original implementation goes to [Cheng Lou](https://github.com/chenglou).** This project is a direct port of his work to Rust. It exists because we needed pretext's two-phase `prepare` / `layout` architecture inside a Rust project and didn't want to reimplement the line-breaking logic from scratch. If you're working in JavaScript/TypeScript, use [the original](https://github.com/chenglou/pretext) instead. This repo exists only to make the same ideas available to the Rust ecosystem.

Version 0.2 is a breaking production-hardening release. See the
[changelog](./CHANGELOG.md) for the migration summary.

---

## What it is

Two-phase text layout, DOM-free:

- **`prepare(text, font, backend, opts)`**: the expensive measurement phase. Analyzes text into segments, measures each segment once, returns a `PreparedText` handle.
- **`layout(prepared, max_width, line_height)`**: the fast reflow phase. Pure arithmetic over cached widths with no measurement or text materialization.

The win is that reflow (resize, re-wrap at a new width) is essentially free: all the expensive work was done in `prepare`.

## Quick start

```rust
use pretext::{prepare, layout, backend::fixed::FixedWidthBackend, backend::FontSpec};

fn main() -> pretext::Result<()> {
    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter")?;
    let prepared = prepare("Hello, world!", &font, &backend, Default::default())?;
    let result = layout(&prepared, 200.0, 24.0)?;
    println!("Lines: {}, Height: {}px", result.line_count, result.height);
    Ok(())
}
```

### Backends

- `backend::fixed::FixedWidthBackend`: deterministic, dependency-free. Useful for tests and server-side estimation.
- `backend::CanvasBackend`: browser `canvas.measureText` (requires the `wasm` feature).
- `backend::SkrifaNominalBackend`: explicit native **unshaped advance
  estimation** via Skrifa (requires `skrifa-nominal`). It does not apply
  kerning, ligatures, bidi shaping, or complex-script positioning and must not
  be used when line breaks need to match rendered production typography. It
  selects loaded bytes by primary family and size only; CSS style, weight,
  stretch, and later fallback families do not select another face.

### More APIs

- `layout_with_lines`: return each wrapped line's text, width, and cursor range.
- `layout_next_line`: streaming API; get one line at a time, with per-line `max_width` (useful for flowing text around images).
- `measure_natural_width`: width the text would occupy unwrapped.
- `walk_line_ranges`: geometry-only iteration, fastest path when you don't need the text.
- `prepare_inline_flow_with_options`: prepare mixed inline runs with aggregate
  item, byte, grapheme, and segment limits. The convenience
  `prepare_inline_flow` uses production-safe defaults.

## WebAssembly

Build with `wasm-pack`:

```
wasm-pack build --release --target web -- --features wasm
```

The WASM API exposes explicit font and whitespace inputs (`whiteSpace` is
`"normal"` or `"pre-wrap"`):

- `pretextPrepare(text, fontCss, whiteSpace)` / `pretextLayout(handle, maxWidth, lineHeight)` / `pretextFree(handle)`: the two-phase path with handle-based state.
- `pretextPrepareAndLayout(text, fontCss, whiteSpace, maxWidth, lineHeight)`: one-shot convenience.
- `pretextLayoutLines(text, fontCss, whiteSpace, maxWidth)`: returns a JSON array of `{text, width}` per wrapped line, for canvas renderers that need to draw each line individually.
- `pretextPrepareBatch(textsJson, fontCss, whiteSpace)`,
  `pretextLayoutBatch(handlesCsv, maxWidth, lineHeight)`, and
  `pretextFreeBatch(handlesCsv)` for bounded atomic batch processing.
- `pretextClearMeasurementCache()`: invalidate retained Canvas measurements
  after browser font availability changes.

Every WASM export throws on malformed input, unknown handles, pool exhaustion,
or layout failure. Thrown `PretextError` values include a stable `code`, a
human-readable `message`, and variant-specific fields such as `handle`,
`maxBytes`, or `maxUnits`. Batch calls validate the complete request before
mutation; they never return partial success. Prepared handles are monotonic and
never reused, so stale JavaScript handles cannot alias later text. The pool is
bounded to 16,384 live handles and a conservative 64 MiB retained-memory
estimate. Each batch is limited to 1,024 items, 4 MiB of serialized input, and
65,536 graphemes.

The public WASM preparation exports use the real browser
`CanvasRenderingContext2D.measureText` backend with the caller's validated
`fontCss` and a persistent bounded cache.
They return an error when a usable browser canvas is unavailable; they never
substitute fixed-width estimates. Use `FixedWidthBackend` explicitly from Rust
when deterministic estimation is the intended behavior.

Every Canvas `fontCss` must contain an **unquoted CSS generic fallback**, such
as `"16px Inter, sans-serif"`. This makes browser fallback an explicit caller
choice; a named-only family or quoted `"sans-serif"` is rejected. Callers must
also wait for required web fonts before preparing text. If the available font
faces change later, call `pretextClearMeasurementCache()`, free affected
handles, and prepare them again. Prepared measurements are immutable.

## Error and resource model

- Measurement and preparation are fallible. Backends never substitute fake
  widths after an operational failure.
- `SkrifaNominalBackend` returns `MissingFont` until the requested family or an
  explicit default font has been loaded.
- Prepared values and cursors are opaque, so callers cannot construct
  mismatched parallel arrays or invalid cursor positions.
- Preparation rejects inputs larger than 4 MiB, 65,536 graphemes, or 65,536
  analyzed segments by default. Override `max_input_bytes`, `max_graphemes`, or
  `max_segments` deliberately when larger documents are expected.
- Canvas measurements use bounded 1,024-entry and 4 MiB retained-key/value
  limits by default; either limit can be set to zero to disable caching.
- GPU layout rejects characters absent from its atlas and caps output at
  65,536 64-byte glyph instances by default instead of silently dropping or
  allocating unbounded output.
- Inline flow applies the same aggregate byte/grapheme/segment limits across
  the complete item slice and caps source items at 1,024 by default.

## Parity

The goal is behavioral parity with the original on the core pipeline: segment analysis, space/glue handling, tab stops, soft hyphens, hard breaks, CJK word-level breaking, trailing-space hang, and line-end advance accounting. Discretionary hyphens and the streaming cursor API are ported as well.

Bidi metadata (`pretext::bidi`) delegates Unicode classification, isolates,
explicit embeddings, and paragraph-level UAX #9 resolution to `unicode-bidi`.
`prepare_with_segments()` attaches a per-segment level array (`seg_levels()`)
whenever resolved levels are not uniformly base LTR. Those values describe
only each segment start; a segment can cross a directional boundary. Production
renderers must use `compute_bidi_levels`, split directional runs, and apply
line-specific reordering. Segment offsets use Unicode scalar indices rather
than UTF-16 code units.

Divergences from the original are intentional where Rust idioms differ (error handling, API naming, zero-copy where possible) and are documented in the relevant module. Please file an issue if you find a behavioral difference that isn't a deliberate port choice.

### Emoji width: deliberate scope boundary

The JS reference corrects canvas `measureText` emoji inflation at runtime: it probes the active font's emoji glyph width, stores a per-font `emojiCorrection` factor, and subtracts `emojiCount × emojiCorrection` from every segment width (see upstream `measurement.ts::getCorrectedSegmentWidth`). That mechanism is browser-specific. It depends on canvas metrics for emoji glyphs that the host OS actually has installed.

Rust backends in this crate take a simpler stance:

- `SegmentMetrics::emoji_count` is still tracked during measurement (all three backends populate it).
- No runtime `emojiCorrection` factor is computed, and none is applied to widths.
- Consequently, text containing emoji may produce widths (and therefore line breaks) that diverge from the JS original.

This is intentional. The value `emoji_count` is exposed so downstream callers who need the JS semantics can compute their own correction from it. If you're writing a custom backend that measures emoji precisely (for example, by integrating a shaping engine and an emoji-aware face), the pipeline will honor its widths.

If a future backend wants to internalize this correction, it can. The field is there waiting.

## Credits

- **[Cheng Lou](https://github.com/chenglou)**: original author of [pretext](https://github.com/chenglou/pretext). The algorithm, design decisions, and reference implementation are his. Any elegance here is his; any bugs are this port's.
- This Rust port is maintained separately. Upstream changes to the JS reference are tracked manually.

## License

MIT, same as the original. See [LICENSE](./LICENSE). The license file preserves the original's copyright notice in addition to this port's.
