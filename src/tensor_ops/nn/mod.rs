pub(crate) mod dropout;
pub(crate) mod matmul;

pub use dropout::dropout;
pub use matmul::{matmul, TryMatMul};

#[cfg(feature = "nightly")]
pub(crate) mod conv2d;
#[cfg(feature = "nightly")]
pub use conv2d::TryConv2D;
#[cfg(feature = "nightly")]
pub(crate) use conv2d::TryConv2DTo;

#[cfg(feature = "nightly")]
pub(crate) mod pool2d;
#[cfg(feature = "nightly")]
pub(crate) use pool2d::{ConstAvgPool2D, ConstMaxPool2D, ConstMinPool2D};
#[cfg(feature = "nightly")]
pub use pool2d::{TryAvgPool2D, TryMaxPool2D, TryMinPool2D};
