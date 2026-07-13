//! Unicode character classification utilities for text segmentation.
//!
//! Ported from pretext's `analysis.ts` Unicode tables. These drive the
//! segment-merging passes that determine where line breaks are allowed.

/// Check if a character is in a CJK ideograph range.
///
/// Covers CJK Unified Ideographs, Extension A/B, Compatibility Ideographs,
/// and CJK Symbols and Punctuation.
#[must_use]
#[inline]
pub fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    // CJK Unified Ideographs
    (0x4E00..=0x9FFF).contains(&cp)
    // CJK Unified Ideographs Extension A
    || (0x3400..=0x4DBF).contains(&cp)
    // CJK Unified Ideographs Extension B
    || (0x20000..=0x2A6DF).contains(&cp)
    // CJK Compatibility Ideographs
    || (0xF900..=0xFAFF).contains(&cp)
    // CJK Unified Ideographs Extension C-F
    || (0x2A700..=0x2CEAF).contains(&cp)
    // CJK Compatibility Ideographs Supplement
    || (0x2F800..=0x2FA1F).contains(&cp)
    // Kangxi Radicals
    || (0x2F00..=0x2FDF).contains(&cp)
    // CJK Radicals Supplement
    || (0x2E80..=0x2EFF).contains(&cp)
    // CJK Symbols and Punctuation (partial — excludes some)
    || (0x3000..=0x303F).contains(&cp)
    // Hiragana
    || (0x3040..=0x309F).contains(&cp)
    // Katakana
    || (0x30A0..=0x30FF).contains(&cp)
    // Katakana Phonetic Extensions
    || (0x31F0..=0x31FF).contains(&cp)
    // Halfwidth/Fullwidth Forms (CJK portion)
    || (0xFF01..=0xFF60).contains(&cp)
    || (0xFFE0..=0xFFE6).contains(&cp)
    // Bopomofo
    || (0x3100..=0x312F).contains(&cp)
    || (0x31A0..=0x31BF).contains(&cp)
    // Yi
    || (0xA000..=0xA4CF).contains(&cp)
}

/// Kinsoku Shori — Japanese line-breaking prohibition rules.
///
/// Characters that must NOT start a line (closing brackets, small kana,
/// prolonged sound marks, etc.).
#[must_use]
#[inline]
pub const fn is_kinsoku_end(c: char) -> bool {
    matches!(
        c,
        // Japanese closing punctuation
        '\u{3001}' // Ideographic comma
        | '\u{3002}' // Ideographic full stop
        | '\u{FF0C}' // Fullwidth comma
        | '\u{FF0E}' // Fullwidth full stop
        | '\u{FF09}' // Fullwidth right paren
        | '\u{FF3D}' // Fullwidth right bracket
        | '\u{FF5D}' // Fullwidth right brace
        | '\u{3015}' // Right tortoise shell bracket
        | '\u{3009}' // Right angle bracket
        | '\u{300B}' // Right double angle bracket
        | '\u{300D}' // Right corner bracket
        | '\u{300F}' // Right white corner bracket
        | '\u{3011}' // Right black lenticular bracket
        | '\u{3017}' // Right white lenticular bracket
        | '\u{3019}' // Right white tortoise shell bracket
        | '\u{301B}' // Right white square bracket
        // Small kana
        | '\u{3041}' | '\u{3043}' | '\u{3045}' | '\u{3047}' | '\u{3049}' // ぁぃぅぇぉ
        | '\u{3063}' | '\u{3083}' | '\u{3085}' | '\u{3087}' | '\u{308E}' // っゃゅょゎ
        | '\u{30A1}' | '\u{30A3}' | '\u{30A5}' | '\u{30A7}' | '\u{30A9}' // ァィゥェォ
        | '\u{30C3}' | '\u{30E3}' | '\u{30E5}' | '\u{30E7}' | '\u{30EE}' // ッャュョヮ
        | '\u{30F5}' | '\u{30F6}' // ヵヶ
        // Prolonged sound mark, iteration marks
        | '\u{30FC}' // ー (katakana-hiragana prolonged sound mark)
        | '\u{30FD}' | '\u{30FE}' // ヽヾ
        | '\u{309D}' | '\u{309E}' // ゝゞ
        // CJK punctuation that can't start a line
        | '\u{2019}' // Right single quotation mark
        | '\u{201D}' // Right double quotation mark
        | '\u{FF01}' // Fullwidth exclamation mark
        | '\u{FF1F}' // Fullwidth question mark
        | '\u{30FB}' // Katakana middle dot
        | '\u{FF1A}' // Fullwidth colon
        | '\u{FF1B}' // Fullwidth semicolon
    )
}

/// Characters that must NOT end a line (opening brackets, etc.).
#[must_use]
#[inline]
pub const fn is_kinsoku_start(c: char) -> bool {
    matches!(
        c,
        '\u{FF08}' // Fullwidth left paren
        | '\u{FF3B}' // Fullwidth left bracket
        | '\u{FF5B}' // Fullwidth left brace
        | '\u{3014}' // Left tortoise shell bracket
        | '\u{3008}' // Left angle bracket
        | '\u{300A}' // Left double angle bracket
        | '\u{300C}' // Left corner bracket
        | '\u{300E}' // Left white corner bracket
        | '\u{3010}' // Left black lenticular bracket
        | '\u{3016}' // Left white lenticular bracket
        | '\u{3018}' // Left white tortoise shell bracket
        | '\u{301A}' // Left white square bracket
        | '\u{2018}' // Left single quotation mark
        | '\u{201C}' // Left double quotation mark
    )
}

/// Left-sticky punctuation -- merges with the preceding word segment
/// so that `"better."` is measured as one unit (avoiding float accumulation).
#[must_use]
#[inline]
pub const fn is_left_sticky_punctuation(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | ':' | ';' | '!' | '?' | ')' | ']' | '}' | '"' | '\''
        | '\u{2019}' | '\u{201D}' | '\u{2026}' // right quotes, ellipsis
        | '%' | '\u{00B0}' // percent, degree
    )
}

/// Forward-sticky punctuation -- carries into the next word segment
/// so opening quotes and brackets aren't orphaned at line end.
#[must_use]
#[inline]
pub const fn is_forward_sticky(c: char) -> bool {
    matches!(
        c,
        '(' | '[' | '{' | '"' | '\'' | '\u{2018}' | '\u{201C}' // opening
        | '$' | '\u{00A3}' | '\u{00A5}' | '\u{20AC}' // currency
    )
}

/// Soft hyphen (U+00AD).
#[must_use]
#[inline]
pub const fn is_soft_hyphen(c: char) -> bool {
    c == '\u{00AD}'
}

/// Zero-width space (U+200B) -- explicit break opportunity.
#[must_use]
#[inline]
pub const fn is_zero_width_space(c: char) -> bool {
    c == '\u{200B}'
}

/// Check if a string looks like a URL (starts with `http://`, `https://`, or `www.`).
#[must_use]
pub fn looks_like_url(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.")
}

/// Check if a character is a URL-internal character (not a break point in URLs).
#[must_use]
#[inline]
pub const fn is_url_internal(c: char) -> bool {
    matches!(
        c,
        '/' | '.'
            | '-'
            | '_'
            | '~'
            | ':'
            | '?'
            | '#'
            | '['
            | ']'
            | '@'
            | '!'
            | '$'
            | '&'
            | '\''
            | '('
            | ')'
            | '*'
            | '+'
            | ','
            | ';'
            | '='
            | '%'
    )
}

/// Check if a character is a numeric connective (e.g., `.` in `3.14`, `,` in `1,000`).
#[must_use]
#[inline]
pub const fn is_numeric_connective(c: char) -> bool {
    matches!(c, '.' | ',' | '-' | ':' | '/' | '+')
}

/// Check if a string contains only ASCII digits.
#[must_use]
pub fn is_all_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Check if a character is an emoji (simplified detection).
///
/// Covers common emoji ranges for diagnostic metadata. It is not a complete
/// Unicode emoji-property implementation, and no measurement correction is
/// inferred from this classification.
#[must_use]
#[inline]
pub fn is_emoji(c: char) -> bool {
    let cp = c as u32;
    // Emoticons
    (0x1F600..=0x1F64F).contains(&cp)
    // Misc symbols
    || (0x1F300..=0x1F5FF).contains(&cp)
    // Transport & map
    || (0x1F680..=0x1F6FF).contains(&cp)
    // Supplemental symbols
    || (0x1F900..=0x1F9FF).contains(&cp)
    // Flags
    || (0x1FA00..=0x1FA6F).contains(&cp)
    || (0x1FA70..=0x1FAFF).contains(&cp)
    // Dingbats
    || (0x2700..=0x27BF).contains(&cp)
    // Misc symbols & pictographs
    || (0x2600..=0x26FF).contains(&cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cjk_detection() {
        assert!(is_cjk('\u{4E2D}')); // 中
        assert!(is_cjk('\u{65E5}')); // 日
        assert!(is_cjk('\u{3042}')); // あ Hiragana
        assert!(is_cjk('\u{30A2}')); // ア Katakana
        assert!(!is_cjk('A'));
        assert!(!is_cjk('1'));
    }

    #[test]
    fn test_kinsoku_end() {
        assert!(is_kinsoku_end('\u{3001}')); // Ideographic comma
        assert!(is_kinsoku_end('\u{3002}')); // Ideographic full stop
        assert!(is_kinsoku_end('\u{3041}')); // Small hiragana
        assert!(!is_kinsoku_end('\u{3042}')); // Regular hiragana
    }

    #[test]
    fn test_kinsoku_start() {
        assert!(is_kinsoku_start('\u{300C}')); // Left corner bracket
        assert!(is_kinsoku_start('\u{201C}')); // Left double quote
        assert!(!is_kinsoku_start('A'));
    }

    #[test]
    fn test_left_sticky() {
        assert!(is_left_sticky_punctuation('.'));
        assert!(is_left_sticky_punctuation(','));
        assert!(is_left_sticky_punctuation('!'));
        assert!(!is_left_sticky_punctuation('A'));
    }

    #[test]
    fn test_url_detection() {
        assert!(looks_like_url("https://example.com"));
        assert!(looks_like_url("http://foo.bar"));
        assert!(looks_like_url("www.example.com"));
        assert!(!looks_like_url("not a url"));
    }

    #[test]
    fn test_emoji() {
        assert!(is_emoji('\u{1F600}')); // grinning face
        assert!(is_emoji('\u{1F680}')); // rocket
        assert!(!is_emoji('A'));
        assert!(!is_emoji('\u{4E2D}')); // 中
    }
}
