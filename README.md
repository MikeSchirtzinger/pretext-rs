# pretext-rs

A Rust port of [**@chenglou/pretext**](https://github.com/chenglou/pretext) — Cheng Lou's DOM-free text measurement and line-breaking engine.

**All credit for the algorithm, design, and original implementation goes to [Cheng Lou](https://github.com/chenglou).** This project is a direct port of his work to Rust. It exists because we needed pretext's two-phase `prepare` / `layout` architecture inside a Rust project and didn't want to reimplement the line-breaking logic from scratch. If you're working in JavaScript/TypeScript, use [the original](https://github.com/chenglou/pretext) instead — this repo exists only to make the same ideas available to the Rust ecosystem.

---

## What it is

Two-phase text layout, DOM-free:

- **`prepare(text, font, backend, opts)`** — the expensive measurement phase. Analyzes text into segments, measures each segment once, returns a `PreparedText` handle.
- **`layout(prepared, max_width, line_height)`** — the fast reflow phase. Pure arithmetic over cached widths, zero allocations, ~0.3 µs per block.

The win is that reflow (resize, re-wrap at a new width) is essentially free — all the expensive work was done in `prepare`.

## Quick start

```rust
use pretext::{prepare, layout, backend::fixed::FixedWidthBackend, backend::FontSpec};

let backend = FixedWidthBackend::new();
let font = FontSpec::new("16px Inter");
let prepared = prepare("Hello, world!", &font, &backend, Default::default());
let result = layout(&prepared, 200.0, 24.0);
println!("Lines: {}, Height: {}px", result.line_count, result.height);
```

### Backends

- `backend::fixed::FixedWidthBackend` — deterministic, dependency-free. Useful for tests and server-side estimation.
- `backend::CanvasBackend` — browser `canvas.measureText` (requires the `wasm` feature).
- `backend::FontdueBackend` — native font metrics via [fontdue](https://crates.io/crates/fontdue) (requires the `fontdue` feature).

### More APIs

- `layout_with_lines` — return each wrapped line's text, width, and cursor range.
- `layout_next_line` — streaming API; get one line at a time, with per-line `max_width` (useful for flowing text around images).
- `measure_natural_width` — width the text would occupy unwrapped.
- `walk_line_ranges` — geometry-only iteration, fastest path when you don't need the text.

## WebAssembly

Build with `wasm-pack`:

```
wasm-pack build --release --target web -- --features wasm
```

The WASM API exposes:

- `pretextPrepare(text)` / `pretextLayout(handle, maxWidth, lineHeight)` / `pretextFree(handle)` — the two-phase path with handle-based state.
- `pretextPrepareAndLayout(text, maxWidth, lineHeight)` — one-shot convenience.
- `pretextLayoutLines(text, maxWidth)` — returns a JSON array of `{text, width}` per wrapped line, for canvas renderers that need to draw each line individually.
- Batch variants (`pretextPrepareBatch`, `pretextLayoutBatch`, `pretextFreeBatch`) for processing many text blocks in one call.

## Parity

The goal is behavioral parity with the original on the core pipeline: segment analysis, space/glue handling, tab stops, soft hyphens, hard breaks, CJK word-level breaking, trailing-space hang, and line-end advance accounting. Discretionary hyphens and the streaming cursor API are ported as well.

Bidi metadata (`pretext::bidi`) is ported verbatim from the upstream `bidi.ts` — embedding-level tables, the W1–W7 / N1–N2 / I1–I2 rules, and the `(len / numBidi) < 0.3` paragraph-level heuristic. `prepare_with_segments()` attaches a per-segment level array (`seg_levels()`) whenever the text contains right-to-left characters. See the module-level docs for the char-index vs UTF-16 indexing note.

Divergences from the original are intentional where Rust idioms differ (error handling, API naming, zero-copy where possible) and are documented in the relevant module. Please file an issue if you find a behavioral difference that isn't a deliberate port choice.

### Emoji width: deliberate scope boundary

The JS reference corrects canvas `measureText` emoji inflation at runtime: it probes the active font's emoji glyph width, stores a per-font `emojiCorrection` factor, and subtracts `emojiCount × emojiCorrection` from every segment width (see upstream `measurement.ts::getCorrectedSegmentWidth`). That mechanism is browser-specific — it depends on canvas metrics for emoji glyphs that the host OS actually has installed.

Rust backends in this crate take a simpler stance:

- `SegmentMetrics::emoji_count` is still tracked during measurement (all three backends populate it).
- No runtime `emojiCorrection` factor is computed, and none is applied to widths.
- Consequently, text containing emoji may produce widths (and therefore line breaks) that diverge from the JS original.

This is intentional. The value `emoji_count` is exposed so downstream callers who need the JS semantics can compute their own correction from it. If you're writing a custom backend that does measure emoji precisely (for example, a `fontdue` build with an emoji-aware face), the pipeline will happily honor its widths.

If a future backend wants to internalize this correction, it can — the field is there waiting.

## Credits

- **[Cheng Lou](https://github.com/chenglou)** — original author of [pretext](https://github.com/chenglou/pretext). The algorithm, design decisions, and reference implementation are his. Any elegance here is his; any bugs are this port's.
- This Rust port is maintained separately. Upstream changes to the JS reference are tracked manually.

## License

MIT — same as the original. See [LICENSE](./LICENSE). The license file preserves the original's copyright notice in addition to this port's.
