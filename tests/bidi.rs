//! Bidi behavior and public integration tests.
//!
//! Paragraph direction follows the first strong directional character; this
//! avoids the superseded ratio heuristic that incorrectly made any mixed LTR
//! paragraph RTL.

use pretext::bidi::{BidiType, classify_char, compute_bidi_levels, compute_segment_levels};

#[track_caller]
fn valid<T>(result: pretext::Result<T>) -> T {
    result.expect("test input is valid")
}

// ---- Classifier ------------------------------------------------------------

#[test]
fn classify_latin_letter() {
    assert_eq!(classify_char('A'), BidiType::L);
    assert_eq!(classify_char('z'), BidiType::L);
}

#[test]
fn classify_ascii_digit() {
    assert_eq!(classify_char('0'), BidiType::EN);
    assert_eq!(classify_char('9'), BidiType::EN);
}

#[test]
fn classify_space_is_whitespace() {
    assert_eq!(classify_char(' '), BidiType::WS);
}

#[test]
fn classify_comma_is_common_separator() {
    assert_eq!(classify_char(','), BidiType::CS);
}

#[test]
fn classify_hebrew_is_rtl() {
    // U+05D0 HEBREW LETTER ALEF
    assert_eq!(classify_char('\u{05D0}'), BidiType::R);
    // U+05F4 upper bound of R range
    assert_eq!(classify_char('\u{05F4}'), BidiType::R);
}

#[test]
fn classify_arabic_letter_is_al() {
    // U+0627 ARABIC LETTER ALEF — arabicTypes[0x27] = 'AL'
    assert_eq!(classify_char('\u{0627}'), BidiType::AL);
}

#[test]
fn classify_arabic_indic_digit_is_an() {
    // U+0660 ARABIC-INDIC DIGIT ZERO — arabicTypes[0x60] = 'AN'
    assert_eq!(classify_char('\u{0660}'), BidiType::AN);
}

#[test]
fn classify_syriac_is_al_fallthrough() {
    // U+0700-0x08AC fall through to AL
    assert_eq!(classify_char('\u{0700}'), BidiType::AL);
    assert_eq!(classify_char('\u{08AC}'), BidiType::AL);
}

#[test]
fn classify_high_bmp_defaults_l() {
    // Beyond the bidi ranges → L
    assert_eq!(classify_char('\u{4E2D}'), BidiType::L); // CJK 中
}

// ---- compute_bidi_levels ---------------------------------------------------

#[test]
fn all_latin_yields_none() {
    assert!(compute_bidi_levels("hello world").is_none());
}

#[test]
fn empty_yields_none() {
    assert!(compute_bidi_levels("").is_none());
}

#[test]
fn hebrew_only_single_level() {
    // All R chars, levels all 1.
    let got = compute_bidi_levels("\u{05D0}\u{05D1}\u{05D2}").expect("bidi present");
    assert_eq!(got, vec![1, 1, 1]);
}

#[test]
fn emoji_is_neutral_and_follows_the_rtl_paragraph_level() {
    assert_eq!(classify_char('\u{1F600}'), BidiType::ON);

    let got = compute_bidi_levels("\u{1F600}\u{05D0}\u{05D1}").expect("bidi present");
    assert_eq!(got, vec![1, 1, 1]);
    assert!(compute_bidi_levels("\u{1F600} hello").is_none());
}

#[test]
fn rtl_isolate_resolves_without_changing_surrounding_ltr_text() {
    assert_eq!(classify_char('\u{2067}'), BidiType::RLI);
    assert_eq!(classify_char('\u{2069}'), BidiType::PDI);

    let got = compute_bidi_levels("a \u{2067}\u{05D0}\u{05D1}\u{2069} z")
        .expect("RTL isolate requires bidi metadata");
    assert_eq!(got, vec![0, 0, 0, 1, 1, 0, 0, 0]);
}

#[test]
fn first_strong_isolate_detects_its_own_rtl_direction() {
    assert_eq!(classify_char('\u{2068}'), BidiType::FSI);

    let got = compute_bidi_levels("a \u{2068}\u{05D0}\u{05D1}\u{2069} z")
        .expect("first-strong isolate requires bidi metadata");
    assert_eq!(got, vec![0, 0, 0, 1, 1, 0, 0, 0]);
}

#[test]
fn explicit_rtl_embedding_resolves_nested_ltr_text() {
    assert_eq!(classify_char('\u{202B}'), BidiType::RLE);
    assert_eq!(classify_char('\u{202C}'), BidiType::PDF);

    let got = compute_bidi_levels("a \u{202B}AB\u{202C} z")
        .expect("RTL embedding requires bidi metadata");
    assert_eq!(got.len(), 8);
    assert_eq!(&got[..3], &[0, 0, 0]);
    assert_eq!(&got[3..5], &[2, 2]);
    assert_eq!(&got[6..], &[0, 0]);
}

#[test]
fn paragraphs_resolve_direction_independently() {
    let got = compute_bidi_levels("abc \u{05D0}\u{05D1}\n\u{05D0}\u{05D1} abc")
        .expect("both paragraphs contain RTL text");
    assert_eq!(got, vec![0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 2]);
}

// ---- compute_segment_levels ------------------------------------------------

// Representative resolved segment-start levels:
//
// all_latin      "hello world"                 starts=[0,5,6]        → null
// hebrew_word    "אבג world"                   starts=[0,3,4]        → [1,1,2]
// arabic_word    "السلام"                      starts=[0]            → [1]
// mixed_ltr_rtl  "hello אב world"              starts=[0,5,6,8,9]    → [0,0,1,0,0]
// digits_only    "12345"                       starts=[0]            → null
// arabic_digits  "٠١ test"                     starts=[0,2,3]        → [2,0,0]
// empty          ""                            starts=[]             → null
// spaces_only    "   "                         starts=[0]            → null
// hebrew_single  "אבג"                         starts=[0]            → [1]
// arabic_latin   "abc الس xyz"                 starts=[0,3,4,7,8]    → [0,0,1,0,0]

#[test]
fn seg_levels_all_latin_none() {
    assert!(valid(compute_segment_levels("hello world", &[0, 5, 6])).is_none());
}

#[test]
fn seg_levels_hebrew_word() {
    let got = valid(compute_segment_levels(
        "\u{05D0}\u{05D1}\u{05D2} world",
        &[0, 3, 4],
    ))
    .expect("bidi present");
    assert_eq!(got, vec![1, 1, 2]);
}

#[test]
fn seg_levels_arabic_word() {
    let got = valid(compute_segment_levels(
        "\u{0627}\u{0644}\u{0633}\u{0644}\u{0627}\u{0645}",
        &[0],
    ))
    .expect("bidi present");
    assert_eq!(got, vec![1]);
}

#[test]
fn seg_levels_mixed_ltr_rtl() {
    let got = valid(compute_segment_levels(
        "hello \u{05D0}\u{05D1} world",
        &[0, 5, 6, 8, 9],
    ))
    .expect("bidi present");
    assert_eq!(got, vec![0, 0, 1, 0, 0]);
}

#[test]
fn seg_levels_digits_only_none() {
    assert!(valid(compute_segment_levels("12345", &[0])).is_none());
}

#[test]
fn seg_levels_arabic_indic_digits() {
    let got =
        valid(compute_segment_levels("\u{0660}\u{0661} test", &[0, 2, 3])).expect("bidi present");
    assert_eq!(got, vec![2, 0, 0]);
}

#[test]
fn seg_levels_empty_none() {
    assert!(valid(compute_segment_levels("", &[])).is_none());
}

#[test]
fn seg_levels_spaces_only_none() {
    assert!(valid(compute_segment_levels("   ", &[0])).is_none());
}

#[test]
fn seg_levels_hebrew_single_segment() {
    let got =
        valid(compute_segment_levels("\u{05D0}\u{05D1}\u{05D2}", &[0])).expect("bidi present");
    assert_eq!(got, vec![1]);
}

#[test]
fn seg_levels_arabic_latin_interleaved() {
    let got = valid(compute_segment_levels(
        "abc \u{0627}\u{0644}\u{0633} xyz",
        &[0, 3, 4, 7, 8],
    ))
    .expect("bidi present");
    assert_eq!(got, vec![0, 0, 1, 0, 0]);
}

// ---- Integration: prepare_with_segments attaches seg_levels ----------------

#[test]
fn prepare_with_segments_attaches_seg_levels_for_bidi_text() {
    use pretext::{
        backend::{FontSpec, fixed::FixedWidthBackend},
        prepare_with_segments,
    };

    let backend = FixedWidthBackend::new();
    let font = valid(FontSpec::new("16px Inter"));
    let prepared = valid(prepare_with_segments(
        "hello \u{05D0}\u{05D1} world",
        &font,
        &backend,
        Default::default(),
    ));

    let levels = prepared
        .seg_levels()
        .expect("seg_levels populated for bidi text");
    assert!(!levels.is_empty());
}

#[test]
fn prepare_with_segments_no_seg_levels_for_pure_latin() {
    use pretext::{
        backend::{FontSpec, fixed::FixedWidthBackend},
        prepare_with_segments,
    };

    let backend = FixedWidthBackend::new();
    let font = valid(FontSpec::new("16px Inter"));
    let prepared = valid(prepare_with_segments(
        "hello world",
        &font,
        &backend,
        Default::default(),
    ));

    assert!(prepared.seg_levels().is_none());
}
