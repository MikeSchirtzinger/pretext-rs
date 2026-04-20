//! Simplified bidi metadata helper for the rich `prepare_with_segments()` path.
//!
//! Direct port of `@chenglou/pretext/src/bidi.ts`. The upstream file is itself
//! forked from pdf.js via Sebastian's text-layout. It classifies characters
//! into bidi types, computes embedding levels, and maps them onto prepared
//! segments for custom rendering.
//!
//! **The line-breaking engine does not consume these levels.** They are
//! advisory metadata exposed for callers doing their own rendering.
//!
//! ## Character-index vs UTF-16 code-unit divergence
//!
//! The JS reference operates on UTF-16 code units (`charCodeAt`, `str.length`).
//! This port operates on Unicode scalar values (`char`) because that is the
//! native Rust abstraction. All characters the bidi classifier cares about
//! (Hebrew `U+0590..U+05F4`, Arabic `U+0600..U+06FF`, `U+0700..U+08AC`) lie in
//! the BMP, so for any text without astral-plane characters the two indexings
//! are identical. Segments starts passed to [`compute_segment_levels`] are
//! interpreted as **char indices** into `normalized`.

/// Bidi character class per UAX #9 (simplified set used by upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BidiType {
    L,
    R,
    AL,
    AN,
    EN,
    ES,
    ET,
    CS,
    ON,
    BN,
    B,
    S,
    WS,
    NSM,
}

use BidiType::*;

/// Base-plane (0x00..=0xFF) classification table, verbatim from upstream.
#[rustfmt::skip]
const BASE_TYPES: [BidiType; 256] = [
    BN,BN,BN,BN,BN,BN,BN,BN,BN,S,B,S,WS,
    B,BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,
    BN,BN,B,B,B,S,WS,ON,ON,ET,ET,ET,ON,
    ON,ON,ON,ON,ON,CS,ON,CS,ON,EN,EN,EN,
    EN,EN,EN,EN,EN,EN,EN,ON,ON,ON,ON,ON,
    ON,ON,L,L,L,L,L,L,L,L,L,L,L,L,L,
    L,L,L,L,L,L,L,L,L,L,L,L,L,ON,ON,
    ON,ON,ON,ON,L,L,L,L,L,L,L,L,L,L,
    L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,
    L,ON,ON,ON,ON,BN,BN,BN,BN,BN,BN,B,BN,
    BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,
    BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,BN,
    BN,CS,ON,ET,ET,ET,ET,ON,ON,ON,ON,L,ON,
    ON,ON,ON,ON,ET,ET,EN,EN,ON,L,ON,ON,ON,
    EN,L,ON,ON,ON,ON,ON,L,L,L,L,L,L,L,
    L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,
    L,ON,L,L,L,L,L,L,L,L,L,L,L,L,L,
    L,L,L,L,L,L,L,L,L,L,L,L,L,L,L,
    L,L,L,ON,L,L,L,L,L,L,L,L,
];

/// Arabic-block (0x0600..=0x06FF) classification table, verbatim from upstream.
/// Indexed by `charCode & 0xff`.
#[rustfmt::skip]
const ARABIC_TYPES: [BidiType; 256] = [
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    CS,AL,ON,ON,NSM,NSM,NSM,NSM,NSM,NSM,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,NSM,NSM,NSM,NSM,NSM,NSM,NSM,
    NSM,NSM,NSM,NSM,NSM,NSM,NSM,AL,AL,AL,AL,
    AL,AL,AL,AN,AN,AN,AN,AN,AN,AN,AN,AN,
    AN,ET,AN,AN,AL,AL,AL,NSM,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,NSM,NSM,NSM,NSM,NSM,NSM,NSM,NSM,NSM,NSM,
    NSM,NSM,NSM,NSM,NSM,NSM,NSM,NSM,NSM,ON,NSM,
    NSM,NSM,NSM,AL,AL,AL,AL,AL,AL,AL,AL,AL,
    AL,AL,AL,AL,AL,AL,AL,AL,AL,
];

// Compile-time guards: both tables must be 256 entries to cover a full byte.
const _: () = assert!(BASE_TYPES.len() == 256);
const _: () = assert!(ARABIC_TYPES.len() == 256);

/// Classify a single character into its bidi type.
///
/// Mirrors upstream `classifyChar`: base-table lookup for 0x00..=0xFF,
/// Hebrew range 0x0590..=0x05F4 → `R`, Arabic block 0x0600..=0x06FF →
/// `ARABIC_TYPES[code & 0xff]`, 0x0700..=0x08AC → `AL`, otherwise `L`.
#[must_use]
pub fn classify_char(c: char) -> BidiType {
    let code = c as u32;
    if code <= 0x00ff {
        return BASE_TYPES[code as usize];
    }
    if (0x0590..=0x05f4).contains(&code) {
        return R;
    }
    if (0x0600..=0x06ff).contains(&code) {
        return ARABIC_TYPES[(code & 0xff) as usize];
    }
    if (0x0700..=0x08AC).contains(&code) {
        return AL;
    }
    L
}

/// Compute a per-character bidi embedding level for `text`.
///
/// Returns `None` when:
/// - the text is empty, or
/// - the text contains no bidi-direction characters (R / AL / AN), so there
///   is nothing to disambiguate.
///
/// The returned vector has one entry per Unicode scalar value, **not** per
/// UTF-8 byte — see the module-level note on indexing.
#[must_use]
pub fn compute_bidi_levels(text: &str) -> Option<Vec<i8>> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len == 0 {
        return None;
    }

    let mut types: Vec<BidiType> = Vec::with_capacity(len);
    let mut num_bidi = 0usize;
    for &c in &chars {
        let t = classify_char(c);
        if matches!(t, R | AL | AN) {
            num_bidi += 1;
        }
        types.push(t);
    }

    if num_bidi == 0 {
        return None;
    }

    // Upstream heuristic (preserved verbatim): level=0 only when the text is
    // overwhelmingly bidi. Since `len >= num_bidi` always, `len/num_bidi >= 1`,
    // so in practice any bidi character makes the paragraph level 1 (RTL).
    #[allow(clippy::cast_precision_loss)]
    let ratio = (len as f64) / (num_bidi as f64);
    let start_level: i8 = if ratio < 0.3 { 0 } else { 1 };

    let mut levels = vec![start_level; len];
    let e: BidiType = if start_level & 1 == 1 { R } else { L };
    let sor = e;

    // W1: NSM takes the type of the previous character.
    let mut last_type = sor;
    for t in types.iter_mut() {
        if *t == NSM {
            *t = last_type;
        } else {
            last_type = *t;
        }
    }

    // W2: EN after AL becomes AN.
    last_type = sor;
    for t in types.iter_mut() {
        match *t {
            EN => {
                *t = if last_type == AL { AN } else { EN };
            }
            R | L | AL => last_type = *t,
            _ => {}
        }
    }

    // W3: AL → R.
    for t in types.iter_mut() {
        if *t == AL {
            *t = R;
        }
    }

    // W4: ES between EN,EN → EN. CS between EN,EN or AN,AN → same.
    for i in 1..len.saturating_sub(1) {
        if types[i] == ES && types[i - 1] == EN && types[i + 1] == EN {
            types[i] = EN;
        }
        if types[i] == CS
            && (types[i - 1] == EN || types[i - 1] == AN)
            && types[i + 1] == types[i - 1]
        {
            types[i] = types[i - 1];
        }
    }

    // W5: ET adjacent to EN → EN.
    for i in 0..len {
        if types[i] != EN {
            continue;
        }
        if i > 0 {
            let mut j = i - 1;
            while types[j] == ET {
                types[j] = EN;
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }
        let mut j = i + 1;
        while j < len && types[j] == ET {
            types[j] = EN;
            j += 1;
        }
    }

    // W6: remaining WS/ES/ET/CS → ON.
    for t in types.iter_mut() {
        if matches!(*t, WS | ES | ET | CS) {
            *t = ON;
        }
    }

    // W7: EN after strong L becomes L.
    last_type = sor;
    for t in types.iter_mut() {
        match *t {
            EN => {
                *t = if last_type == L { L } else { EN };
            }
            R | L => last_type = *t,
            _ => {}
        }
    }

    // N1/N2: neutrals (ON) take the direction of surrounding strong chars,
    // falling back to embedding direction.
    let mut i = 0;
    while i < len {
        if types[i] != ON {
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < len && types[end] == ON {
            end += 1;
        }
        let before: BidiType = if i > 0 { types[i - 1] } else { sor };
        let after: BidiType = if end < len { types[end] } else { sor };
        let b_dir: BidiType = if before != L { R } else { L };
        let a_dir: BidiType = if after != L { R } else { L };
        if b_dir == a_dir {
            for t in types.iter_mut().take(end).skip(i) {
                *t = b_dir;
            }
        }
        i = end; // step past the run; outer `continue` mimics upstream's `i = end - 1; i++`.
    }
    for t in types.iter_mut() {
        if *t == ON {
            *t = e;
        }
    }

    // I1/I2: implicit level resolution.
    for i in 0..len {
        let t = types[i];
        if levels[i] & 1 == 0 {
            if t == R {
                levels[i] += 1;
            } else if t == AN || t == EN {
                levels[i] += 2;
            }
        } else if t == L || t == AN || t == EN {
            levels[i] += 1;
        }
    }

    Some(levels)
}

/// Compute one embedding level per segment, given segment start **char
/// indices** into the `normalized` string.
///
/// Returns `None` when the text has no bidi characters (matching upstream's
/// `computeBidiLevels(...) === null` short-circuit).
///
/// # Panics
/// Panics in debug builds if any `seg_starts[i]` is out of range for the
/// character count of `normalized`.
#[must_use]
pub fn compute_segment_levels(normalized: &str, seg_starts: &[usize]) -> Option<Vec<i8>> {
    let bidi_levels = compute_bidi_levels(normalized)?;
    let mut out = Vec::with_capacity(seg_starts.len());
    for &s in seg_starts {
        debug_assert!(
            s < bidi_levels.len(),
            "segment start {s} out of range for {} chars",
            bidi_levels.len()
        );
        out.push(bidi_levels[s]);
    }
    Some(out)
}
