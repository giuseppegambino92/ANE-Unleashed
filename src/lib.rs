//! # ANE-Unleashed
//!
//! Non-AI workloads on the Apple Neural Engine, driven through the private
//! in-memory MIL path of `AppleNeuralEngine.framework`:
//!
//! * [`query`] — columnar predicate filters + COUNT/SUM aggregation
//! * [`grep`]  — multi-pattern byte scanning (ANE prefilter, exact CPU confirm)
//! * [`dsp`]   — multi-channel long FIR / impulse-response convolution
//!
//! **This uses undocumented Apple SPI.** It was developed and measured on
//! macOS 27.2.1 / M3 Max (ANE v176). It can break with any OS update and is
//! not suitable for App Store software.
pub mod ffi;
pub mod iosurface_tensor;
pub mod mil_codegen;
pub mod runtime;
pub mod telemetry;
pub mod query;
pub mod grep;
pub mod dsp;
pub mod cpu_baseline;
pub mod capi;

pub use iosurface_tensor::{AneTensor, DType};
pub use runtime::{available, AneProgram, BoundRequest};

#[derive(Debug)]
pub enum Error {
    Ane(String),
    Compile(String),
    Shape(String),
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Ane(s) => write!(f, "ANE runtime error: {s}"),
            Error::Compile(s) => write!(f, "ANE compile error: {s}"),
            Error::Shape(s) => write!(f, "shape error: {s}"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
