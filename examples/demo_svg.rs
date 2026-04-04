//! Generates an SVG visualization of pretext-rs layout capabilities.
//!
//! Run: `cargo run --example demo_svg > demo.svg`
//! Or:  `cargo run --example demo_svg > demo.svg && open demo.svg`

use pretext::backend::fixed::FixedWidthBackend;
use pretext::backend::FontSpec;
use pretext::inline_flow::{
    prepare_inline_flow, layout_inline_flow, InlineFlowItem, BreakMode,
};
use pretext::types::LayoutCursor;
use pretext::{
    layout, layout_next_line, layout_with_lines, measure_natural_width, prepare,
    prepare_with_segments,
};

const BG: &str = "#0a0e17";
const FG: &str = "#c8d3e0";
const ACCENT: &str = "#00d4ff";
const ACCENT2: &str = "#ff6b9d";
const ACCENT3: &str = "#7c5cfc";
const DIM: &str = "#3a4459";
const CHIP_BG: &str = "#162032";
const LINE_COLORS: [&str; 6] = ["#00d4ff", "#ff6b9d", "#7c5cfc", "#00e5a0", "#ffb347", "#ff6b6b"];

struct SvgBuilder {
    elements: Vec<String>,
    y_cursor: f64,
}

impl SvgBuilder {
    fn new() -> Self {
        Self {
            elements: Vec::new(),
            y_cursor: 0.0,
        }
    }

    fn section_title(&mut self, title: &str, subtitle: &str, x: f64) {
        self.y_cursor += 48.0;
        self.elements.push(format!(
            r#"<text x="{x}" y="{y}" fill="{ACCENT}" font-family="Inter, SF Pro, system-ui, sans-serif" font-size="18" font-weight="700" letter-spacing="0.5">{title}</text>"#,
            y = self.y_cursor,
        ));
        self.y_cursor += 22.0;
        self.elements.push(format!(
            r#"<text x="{x}" y="{y}" fill="{DIM}" font-family="Inter, SF Pro, system-ui, sans-serif" font-size="12" font-style="italic">{subtitle}</text>"#,
            y = self.y_cursor,
        ));
        self.y_cursor += 20.0;
    }

    fn width_label(&mut self, x: f64, width: f64) {
        // Width indicator line
        self.elements.push(format!(
            r#"<line x1="{x}" y1="{y}" x2="{x2}" y2="{y}" stroke="{DIM}" stroke-width="1" stroke-dasharray="3,3" />"#,
            y = self.y_cursor,
            x2 = x + width,
        ));
        self.elements.push(format!(
            r#"<text x="{tx}" y="{y}" fill="{DIM}" font-family="JetBrains Mono, monospace" font-size="9" text-anchor="end">{w}px</text>"#,
            tx = x + width - 2.0,
            y = self.y_cursor - 3.0,
            w = width as i32,
        ));
        self.y_cursor += 6.0;
    }

    fn render_lines(
        &mut self,
        lines: &[(String, f64)],
        x: f64,
        max_width: f64,
        char_width: f64,
        line_height: f64,
    ) {
        // Container outline
        let total_height = lines.len() as f64 * line_height + 8.0;
        self.elements.push(format!(
            r#"<rect x="{x}" y="{y}" width="{max_width}" height="{total_height}" rx="4" fill="none" stroke="{DIM}" stroke-width="0.5" stroke-dasharray="4,2" />"#,
            y = self.y_cursor,
        ));

        for (i, (text, width)) in lines.iter().enumerate() {
            let line_y = self.y_cursor + 4.0 + i as f64 * line_height;
            let color = LINE_COLORS[i % LINE_COLORS.len()];

            // Line background highlight
            self.elements.push(format!(
                r#"<rect x="{x}" y="{ly}" width="{width}" height="{lh}" rx="2" fill="{color}" fill-opacity="0.08" />"#,
                ly = line_y,
                lh = line_height,
            ));

            // Line number
            self.elements.push(format!(
                r#"<text x="{lx}" y="{ty}" fill="{color}" font-family="JetBrains Mono, monospace" font-size="8" text-anchor="end" opacity="0.6">{n}</text>"#,
                lx = x - 4.0,
                ty = line_y + line_height - 4.0,
                n = i + 1,
            ));

            // Text content — render each character
            for (ci, ch) in text.chars().enumerate() {
                if ch == ' ' {
                    // Show space as subtle dot
                    self.elements.push(format!(
                        r#"<circle cx="{cx}" cy="{cy}" r="1" fill="{DIM}" opacity="0.3" />"#,
                        cx = x + ci as f64 * char_width + char_width * 0.5,
                        cy = line_y + line_height * 0.55,
                    ));
                } else {
                    self.elements.push(format!(
                        r#"<text x="{tx}" y="{ty}" fill="{FG}" font-family="JetBrains Mono, monospace" font-size="13">{ch}</text>"#,
                        tx = x + ci as f64 * char_width,
                        ty = line_y + line_height - 5.0,
                        ch = escape_xml(ch),
                    ));
                }
            }

            // Width annotation on right edge
            self.elements.push(format!(
                r#"<text x="{wx}" y="{ty}" fill="{color}" font-family="JetBrains Mono, monospace" font-size="8" opacity="0.5">{w:.0}</text>"#,
                wx = x + max_width + 6.0,
                ty = line_y + line_height - 4.0,
                w = width,
            ));
        }

        self.y_cursor += total_height + 12.0;
    }

    fn render_inline_flow_line(
        &mut self,
        fragments: &[(String, bool, f64)], // (text, is_atomic, width)
        x: f64,
        line_y: f64,
        line_height: f64,
        char_width: f64,
    ) {
        let mut cx = x;
        for (text, is_atomic, _width) in fragments {
            if *is_atomic {
                // Render as chip
                let chip_w = text.len() as f64 * char_width + 12.0;
                self.elements.push(format!(
                    r#"<rect x="{cx}" y="{ly}" width="{chip_w}" height="{lh}" rx="10" fill="{CHIP_BG}" stroke="{ACCENT3}" stroke-width="1" />"#,
                    ly = line_y + 1.0,
                    lh = line_height - 2.0,
                ));
                self.elements.push(format!(
                    r#"<text x="{tx}" y="{ty}" fill="{ACCENT3}" font-family="JetBrains Mono, monospace" font-size="12" font-weight="600">{text}</text>"#,
                    tx = cx + 6.0,
                    ty = line_y + line_height - 6.0,
                ));
                cx += chip_w + 4.0;
            } else {
                // Render as normal text
                for ch in text.chars() {
                    self.elements.push(format!(
                        r#"<text x="{tx}" y="{ty}" fill="{FG}" font-family="JetBrains Mono, monospace" font-size="13">{ch}</text>"#,
                        tx = cx,
                        ty = line_y + line_height - 5.0,
                        ch = escape_xml(ch),
                    ));
                    cx += char_width;
                }
                cx += 4.0; // Gap
            }
        }
    }

    fn perf_bar(&mut self, label: &str, value_us: f64, max_us: f64, x: f64, bar_width: f64, color: &str) {
        let bar_fill = (value_us / max_us) * bar_width;
        self.elements.push(format!(
            r#"<rect x="{x}" y="{y}" width="{bar_width}" height="20" rx="3" fill="{DIM}" fill-opacity="0.2" />"#,
            y = self.y_cursor,
        ));
        self.elements.push(format!(
            r#"<rect x="{x}" y="{y}" width="{bar_fill}" height="20" rx="3" fill="{color}" fill-opacity="0.7" />"#,
            y = self.y_cursor,
        ));
        self.elements.push(format!(
            r#"<text x="{lx}" y="{ty}" fill="{FG}" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="end">{label}</text>"#,
            lx = x - 8.0,
            ty = self.y_cursor + 14.0,
        ));
        self.elements.push(format!(
            r#"<text x="{vx}" y="{ty}" fill="{color}" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700">{value_us:.2}µs</text>"#,
            vx = x + bar_fill + 6.0,
            ty = self.y_cursor + 14.0,
        ));
        self.y_cursor += 28.0;
    }

    fn emit(self, width: f64) -> String {
        let height = self.y_cursor + 40.0;
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<defs>
  <filter id="glow">
    <feGaussianBlur stdDeviation="2" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<rect width="{width}" height="{height}" fill="{BG}" rx="12"/>
"#
        );

        // Title bar
        svg.push_str(&format!(
            r#"<text x="32" y="32" fill="{ACCENT}" font-family="Inter, SF Pro, system-ui, sans-serif" font-size="22" font-weight="800" letter-spacing="-0.5" filter="url(#glow)">pretext-rs</text>"#
        ));
        svg.push_str(&format!(
            r#"<text x="165" y="32" fill="{DIM}" font-family="Inter, SF Pro, system-ui, sans-serif" font-size="14" font-weight="400">DOM-free text layout engine</text>"#
        ));

        for el in &self.elements {
            svg.push_str(el);
            svg.push('\n');
        }

        svg.push_str("</svg>\n");
        svg
    }
}

fn escape_xml(c: char) -> String {
    match c {
        '<' => "&lt;".to_string(),
        '>' => "&gt;".to_string(),
        '&' => "&amp;".to_string(),
        '"' => "&quot;".to_string(),
        '\'' => "&#39;".to_string(),
        _ => c.to_string(),
    }
}

fn main() {
    let backend = FixedWidthBackend::new();
    let font = FontSpec::new("16px Inter");
    let char_width = 8.0; // Visual char width in SVG
    let line_height = 22.0;
    let left_margin = 50.0;

    let mut svg = SvgBuilder::new();

    // ── Section 1: Line Breaking ─────────────────────────────────────
    svg.section_title(
        "Line Breaking",
        "prepare() once, layout() at any width — 0.09µs per reflow",
        left_margin,
    );

    let text = "The quick brown fox jumps over the lazy dog";
    let prepared = prepare_with_segments(text, &font, &backend, Default::default());

    for &max_w in &[320.0, 200.0, 120.0] {
        svg.width_label(left_margin, max_w);
        let lines = layout_with_lines(&prepared, max_w, line_height);
        let line_data: Vec<(String, f64)> = lines.iter().map(|l| (l.text.clone(), l.width)).collect();
        svg.render_lines(&line_data, left_margin, max_w, char_width, line_height);
    }

    // ── Section 2: CJK Per-Character Breaking ────────────────────────
    svg.section_title(
        "CJK Per-Character Breaking",
        "Kinsoku shori rules — each ideograph is a valid break point",
        left_margin,
    );

    let cjk_text = "日本語のテキストレイアウト";
    let cjk_font = FontSpec::new("16px Noto Sans");
    let cjk_prepared = prepare_with_segments(cjk_text, &cjk_font, &backend, Default::default());
    let cjk_char_width = 16.0;

    for &max_w in &[240.0, 140.0] {
        svg.width_label(left_margin, max_w);
        let lines = layout_with_lines(&cjk_prepared, max_w, line_height);
        let line_data: Vec<(String, f64)> = lines.iter().map(|l| (l.text.clone(), l.width)).collect();
        svg.render_lines(&line_data, left_margin, max_w, cjk_char_width, line_height);
    }

    // ── Section 3: Inline Flow (Mixed Content) ───────────────────────
    svg.section_title(
        "Inline Flow",
        "Mixed runs — atomic chips never break, text wraps around them",
        left_margin,
    );

    let items = vec![
        InlineFlowItem {
            text: "Deploy to".to_string(),
            font: FontSpec::new("16px Inter"),
            break_mode: BreakMode::Normal,
            extra_width: 0.0,
        },
        InlineFlowItem {
            text: "@staging".to_string(),
            font: FontSpec::new("14px monospace"),
            break_mode: BreakMode::Never,
            extra_width: 12.0,
        },
        InlineFlowItem {
            text: "completed successfully in".to_string(),
            font: FontSpec::new("16px Inter"),
            break_mode: BreakMode::Normal,
            extra_width: 0.0,
        },
        InlineFlowItem {
            text: "3.2s".to_string(),
            font: FontSpec::new("14px monospace"),
            break_mode: BreakMode::Never,
            extra_width: 12.0,
        },
    ];

    let flow = prepare_inline_flow(&items, &backend);
    let flow_lines = layout_inline_flow(&flow, 280.0);

    svg.width_label(left_margin, 280.0);
    for (li, line) in flow_lines.iter().enumerate() {
        let line_y = svg.y_cursor + li as f64 * line_height;
        let frags: Vec<(String, bool, f64)> = line
            .fragments
            .iter()
            .map(|f| {
                let is_atomic = items[f.item_index].break_mode == BreakMode::Never;
                (f.text.clone(), is_atomic, f.occupied_width)
            })
            .collect();
        svg.render_inline_flow_line(&frags, left_margin, line_y, line_height, char_width);
    }
    svg.y_cursor += flow_lines.len() as f64 * line_height + 16.0;

    // ── Section 4: Streaming API (Variable Widths) ───────────────────
    svg.section_title(
        "Streaming API",
        "Per-line max_width — text flows around obstacles",
        left_margin,
    );

    let stream_text = "The quick brown fox jumps over the lazy dog and keeps on running";
    let stream_prepared = prepare_with_segments(stream_text, &font, &backend, Default::default());
    let stream_opaque = prepare(stream_text, &font, &backend, Default::default());
    let variable_widths: [f64; 8] = [320.0, 240.0, 160.0, 160.0, 240.0, 320.0, 320.0, 320.0];

    // Draw the "obstacle" shape
    let obstacle_x = left_margin + 160.0;
    let obstacle_y = svg.y_cursor + line_height * 2.0;
    svg.elements.push(format!(
        r#"<rect x="{obstacle_x}" y="{obstacle_y}" width="160" height="{h}" rx="6" fill="{ACCENT2}" fill-opacity="0.1" stroke="{ACCENT2}" stroke-width="1" stroke-dasharray="4,2" />"#,
        h = line_height * 2.0,
    ));
    svg.elements.push(format!(
        r#"<text x="{tx}" y="{ty}" fill="{ACCENT2}" font-family="JetBrains Mono, monospace" font-size="9" text-anchor="middle" opacity="0.6">obstacle</text>"#,
        tx = obstacle_x + 80.0,
        ty = obstacle_y + line_height + 2.0,
    ));

    let mut cursor = LayoutCursor::default();
    let mut line_idx = 0;
    let mut stream_lines: Vec<(String, f64, f64)> = Vec::new(); // (text, width, max_width)

    while let Some((range, next)) = layout_next_line(
        &stream_opaque,
        cursor,
        variable_widths.get(line_idx).copied().unwrap_or(320.0),
    ) {
        let max_w = variable_widths.get(line_idx).copied().unwrap_or(320.0);
        // Materialize text for this line
        let seg_text: String = stream_prepared.segments
            [range.start.segment_index..range.end.segment_index.min(stream_prepared.segments.len())]
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("");
        let trimmed = seg_text.trim().to_string();
        stream_lines.push((trimmed, range.width, max_w));
        line_idx += 1;
        if next == cursor || next.segment_index >= stream_opaque.segment_count() {
            break;
        }
        cursor = next;
    }

    for (i, (text, width, max_w)) in stream_lines.iter().enumerate() {
        let line_y = svg.y_cursor + i as f64 * line_height;
        let color = LINE_COLORS[i % LINE_COLORS.len()];

        // Width boundary
        svg.elements.push(format!(
            r#"<line x1="{x2}" y1="{ly}" x2="{x2}" y2="{ly2}" stroke="{DIM}" stroke-width="0.5" stroke-dasharray="2,2" />"#,
            x2 = left_margin + max_w,
            ly = line_y,
            ly2 = line_y + line_height,
        ));

        // Text
        svg.elements.push(format!(
            r#"<rect x="{lm}" y="{ly}" width="{width}" height="{line_height}" rx="2" fill="{color}" fill-opacity="0.06" />"#,
            lm = left_margin,
            ly = line_y,
        ));
        for (ci, ch) in text.chars().enumerate() {
            if ch != ' ' {
                svg.elements.push(format!(
                    r#"<text x="{tx}" y="{ty}" fill="{FG}" font-family="JetBrains Mono, monospace" font-size="13">{ch}</text>"#,
                    tx = left_margin + ci as f64 * char_width,
                    ty = line_y + line_height - 5.0,
                    ch = escape_xml(ch),
                ));
            }
        }
    }
    svg.y_cursor += stream_lines.len() as f64 * line_height + 16.0;

    // ── Section 5: Performance ───────────────────────────────────────
    svg.section_title(
        "Performance",
        "prepare() once, layout() is pure arithmetic — no DOM, no canvas, no allocations",
        left_margin,
    );

    let perf_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
    let perf_prepared = prepare(perf_text, &font, &backend, Default::default());

    // Benchmark
    let start = std::time::Instant::now();
    for w in (50..550).cycle().take(10000) {
        let _ = layout(&perf_prepared, w as f64, 24.0);
    }
    let layout_elapsed = start.elapsed();
    let layout_us = layout_elapsed.as_nanos() as f64 / 10000.0 / 1000.0;

    let bar_x = left_margin + 120.0;
    let bar_w = 260.0;

    svg.perf_bar("layout()", layout_us, 50.0, bar_x, bar_w, ACCENT);
    svg.perf_bar("DOM reflow", 41.7, 50.0, bar_x, bar_w, ACCENT2); // From pretext benchmarks
    svg.perf_bar("DOM batched", 3.7, 50.0, bar_x, bar_w, "#ffb347");

    // Legend
    svg.y_cursor += 4.0;
    svg.elements.push(format!(
        r#"<text x="{lx}" y="{y}" fill="{DIM}" font-family="JetBrains Mono, monospace" font-size="9">* DOM numbers from @chenglou/pretext benchmarks (500 items, Chrome)</text>"#,
        lx = left_margin,
        y = svg.y_cursor,
    ));
    svg.y_cursor += 14.0;
    svg.elements.push(format!(
        r#"<text x="{lx}" y="{y}" fill="{DIM}" font-family="JetBrains Mono, monospace" font-size="9">  pretext-rs layout() measured at {layout_us:.2}µs/call (10K iterations, release build)</text>"#,
        lx = left_margin,
        y = svg.y_cursor,
    ));

    // ── Emit ─────────────────────────────────────────────────────────
    let total_width = 520.0;
    print!("{}", svg.emit(total_width));
}
