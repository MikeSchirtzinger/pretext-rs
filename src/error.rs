//! Typed errors returned by pretext's fallible production APIs.

use std::fmt;

/// Result type used by pretext APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced while parsing, measuring, preparing, or laying out text.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A CSS-style font specification could not be parsed or validated.
    InvalidFontSpec {
        /// The rejected specification.
        spec: String,
        /// Human-readable validation failure.
        reason: String,
    },
    /// A requested font family was not loaded into the selected backend.
    MissingFont {
        /// Requested family name.
        family: String,
    },
    /// A selected native font has no mapping for a requested character.
    MissingGlyph {
        /// Character absent from the selected font.
        character: char,
        /// Requested font family.
        family: String,
    },
    /// A measurement backend reported an operational failure.
    Measurement {
        /// Stable backend identifier.
        backend: &'static str,
        /// Backend-provided context.
        message: String,
    },
    /// A backend returned a negative or non-finite measurement.
    InvalidMetric {
        /// Metric name.
        metric: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// A public input violated an API precondition.
    InvalidInput {
        /// Parameter or logical input name.
        parameter: &'static str,
        /// Human-readable validation failure.
        reason: String,
    },
    /// Input exceeded a configured resource limit.
    InputTooLarge {
        /// Actual input size in bytes.
        bytes: usize,
        /// Configured maximum size in bytes.
        max_bytes: usize,
    },
    /// An input would produce more structural work than the configured limit.
    InputComplexity {
        /// Stable unit being bounded (for example, graphemes or segments).
        resource: &'static str,
        /// Actual number of structural units.
        units: usize,
        /// Configured maximum number of structural units.
        max_units: usize,
    },
    /// A caller-provided cursor was not valid for the prepared value.
    InvalidCursor {
        /// API surface that rejected the cursor.
        context: &'static str,
        /// Requested segment index.
        segment_index: usize,
        /// Requested grapheme index.
        grapheme_index: usize,
        /// Number of available segments.
        segment_count: usize,
    },
    /// A bidi segment start was outside the normalized character sequence.
    InvalidBidiStart {
        /// Rejected character index.
        start: usize,
        /// Number of Unicode scalar values in the normalized text.
        char_count: usize,
    },
    /// A character cannot be emitted by the configured glyph atlas.
    UnsupportedGlyph {
        /// Rejected Unicode scalar value.
        character: char,
        /// Numeric Unicode code point for diagnostics.
        codepoint: u32,
    },
    /// A WASM handle did not identify a live prepared value.
    InvalidHandle {
        /// Rejected handle.
        handle: u32,
    },
    /// A bounded runtime pool could not accept another entry.
    PoolExhausted {
        /// Maximum number of live entries.
        capacity: usize,
    },
    /// A monotonic opaque identifier space has been consumed.
    IdentifierExhausted {
        /// Stable identifier allocator name.
        resource: &'static str,
    },
    /// A bounded runtime resource would exceed its configured byte limit.
    ResourceLimit {
        /// Stable resource identifier.
        resource: &'static str,
        /// Bytes that would be retained after the requested operation.
        requested_bytes: usize,
        /// Maximum retained bytes allowed for the resource.
        max_bytes: usize,
    },
    /// Interior state was already borrowed or otherwise unavailable.
    StateUnavailable {
        /// State component that could not be accessed.
        state: &'static str,
    },
    /// Checked arithmetic detected an overflow.
    ArithmeticOverflow {
        /// Operation that overflowed.
        operation: &'static str,
    },
}

impl Error {
    pub(crate) fn invalid_input(parameter: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            parameter,
            reason: reason.into(),
        }
    }

    pub(crate) fn measurement(backend: &'static str, message: impl Into<String>) -> Self {
        Self::Measurement {
            backend,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFontSpec { spec, reason } => {
                write!(f, "invalid font specification {spec:?}: {reason}")
            }
            Self::MissingFont { family } => write!(f, "font family {family:?} is not loaded"),
            Self::MissingGlyph { character, family } => {
                write!(f, "font family {family:?} has no glyph for {character:?}")
            }
            Self::Measurement { backend, message } => {
                write!(f, "{backend} measurement failed: {message}")
            }
            Self::InvalidMetric { metric, value } => {
                write!(
                    f,
                    "measurement {metric} must be finite and non-negative, got {value}"
                )
            }
            Self::InvalidInput { parameter, reason } => {
                write!(f, "invalid {parameter}: {reason}")
            }
            Self::InputTooLarge { bytes, max_bytes } => {
                write!(
                    f,
                    "input is {bytes} bytes; configured maximum is {max_bytes} bytes"
                )
            }
            Self::InputComplexity {
                resource,
                units,
                max_units,
            } => write!(
                f,
                "{resource} contains {units} units; configured maximum is {max_units} units"
            ),
            Self::InvalidCursor {
                context,
                segment_index,
                grapheme_index,
                segment_count,
            } => write!(
                f,
                "invalid {context} cursor ({segment_index}, {grapheme_index}) for {segment_count} segments"
            ),
            Self::InvalidBidiStart { start, char_count } => {
                write!(
                    f,
                    "bidi segment start {start} is outside {char_count} characters"
                )
            }
            Self::UnsupportedGlyph {
                character,
                codepoint,
            } => write!(
                f,
                "character {character:?} (U+{codepoint:04X}) is not present in the configured glyph atlas"
            ),
            Self::InvalidHandle { handle } => write!(f, "unknown or freed handle {handle}"),
            Self::PoolExhausted { capacity } => {
                write!(f, "runtime pool is full at {capacity} live entries")
            }
            Self::IdentifierExhausted { resource } => {
                write!(f, "{resource} identifier space is exhausted")
            }
            Self::ResourceLimit {
                resource,
                requested_bytes,
                max_bytes,
            } => write!(
                f,
                "{resource} would retain {requested_bytes} bytes; configured maximum is {max_bytes} bytes"
            ),
            Self::StateUnavailable { state } => write!(f, "{state} state is unavailable"),
            Self::ArithmeticOverflow { operation } => {
                write!(f, "arithmetic overflow while {operation}")
            }
        }
    }
}

impl std::error::Error for Error {}
