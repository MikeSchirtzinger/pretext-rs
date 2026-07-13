use pretext::backend::FontSpec;
use pretext::backend::fixed::FixedWidthBackend;
use pretext::inline_flow::{BreakMode, InlineFlowItem, layout_inline_flow, prepare_inline_flow};
use pretext::types::LayoutCursor;
use pretext::{
    layout, layout_next_line, layout_with_lines, measure_natural_width, prepare,
    prepare_with_segments,
};

fn main() -> pretext::Result<()> {
    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter")?;

    println!("=== pretext-rs: DOM-free text layout ===\n");

    // ─── Basic layout ───────────────────────────────────────────────
    let text = "The quick brown fox jumps over the lazy dog. \
                This sentence demonstrates word wrapping at various widths.";

    let prepared = prepare(text, &font, &backend, Default::default())?;

    let preview: String = text.chars().take(50).collect();
    println!("Text: {preview:?}");
    println!(
        "Natural width: {:.1}px\n",
        measure_natural_width(&prepared)?
    );

    for width in [200, 150, 100, 60] {
        let result = layout(&prepared, width as f64, 24.0)?;
        println!(
            "  @{:>3}px: {} lines, {:.0}px tall",
            width, result.line_count, result.height
        );
    }

    // ─── Layout with lines ──────────────────────────────────────────
    println!("\n--- Lines at 120px ---");
    let prepared_rich = prepare_with_segments(text, &font, &backend, Default::default())?;
    let lines = layout_with_lines(&prepared_rich, 120.0)?;
    for (i, line) in lines.iter().enumerate() {
        println!("  L{}: {:?} ({:.1}px)", i + 1, line.text, line.width);
    }

    // ─── Streaming API ──────────────────────────────────────────────
    println!("\n--- Streaming (variable-width lines) ---");
    let widths = [200.0, 150.0, 100.0, 200.0, 200.0, 200.0, 200.0, 200.0];
    let mut cursor = LayoutCursor::default();
    let mut line_idx = 0;

    while let Some((range, next)) = layout_next_line(
        &prepared,
        cursor,
        widths.get(line_idx).copied().unwrap_or(200.0),
    )? {
        println!(
            "  L{}: width={:.1}px (max={:.0}px)",
            line_idx + 1,
            range.width,
            widths.get(line_idx).copied().unwrap_or(200.0)
        );
        line_idx += 1;
        if next == cursor || next.segment_index() >= prepared.segment_count() {
            break;
        }
        cursor = next;
    }

    // ─── CJK text ───────────────────────────────────────────────────
    println!("\n--- CJK text ---");
    let cjk = "日本語のテキストレイアウトエンジンです。各文字で改行できます。";
    let cjk_prepared = prepare(cjk, &font, &backend, Default::default())?;
    for width in [200, 100, 50] {
        let result = layout(&cjk_prepared, width as f64, 24.0)?;
        println!("  @{:>3}px: {} lines", width, result.line_count);
    }

    // ─── Inline flow (mixed content) ────────────────────────────────
    println!("\n--- Inline flow (agent UI) ---");
    let items = vec![
        InlineFlowItem {
            text: "Deploy to".to_string(),
            font: FontSpec::new("16px Inter")?,
            break_mode: BreakMode::Normal,
            extra_width: 0.0,
        },
        InlineFlowItem {
            text: "@staging".to_string(),
            font: FontSpec::new("14px monospace")?,
            break_mode: BreakMode::Never,
            extra_width: 12.0,
        },
        InlineFlowItem {
            text: "completed in 3.2s".to_string(),
            font: FontSpec::new("16px Inter")?,
            break_mode: BreakMode::Normal,
            extra_width: 0.0,
        },
    ];

    let flow = prepare_inline_flow(&items, &backend)?;
    let flow_lines = layout_inline_flow(&flow, 180.0)?;
    for (i, line) in flow_lines.iter().enumerate() {
        let frags: Vec<String> = line
            .fragments
            .iter()
            .map(|f| format!("[{}]", f.text))
            .collect();
        println!("  L{}: {} ({:.1}px)", i + 1, frags.join(" "), line.width);
    }

    // ─── Resize benchmark simulation ────────────────────────────────
    println!("\n--- Resize simulation (500 layout calls) ---");
    let long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                     Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
                     Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris \
                     nisi ut aliquip ex ea commodo consequat.";
    let long_prepared = prepare(long_text, &font, &backend, Default::default())?;

    let start = std::time::Instant::now();
    for w in (50..550).cycle().take(500) {
        let _result = layout(&long_prepared, w as f64, 24.0)?;
    }
    let elapsed = start.elapsed();
    println!(
        "  500 layouts in {:?} ({:.2}µs per layout)",
        elapsed,
        elapsed.as_micros() as f64 / 500.0
    );

    println!("\nDone.");
    Ok(())
}
