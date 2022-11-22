mod binary;
mod broadcast;
mod matmul;
mod permute;
mod reductions;
mod select;
mod unary;

#[cfg(feature = "nightly")]
mod reshape;

#[cfg(feature = "nightly")]
mod pool2d;

#[cfg(feature = "nightly")]
mod conv2d;
