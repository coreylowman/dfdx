mod as_rust_array;
mod axes;
mod broadcasts;
mod permutes;
mod replace_dim;
mod shape;

pub(crate) use axes::AxesAsArray;
pub(crate) use broadcasts::{BroadcastShapeTo, BroadcastStridesTo, ReduceShape, ReduceShapeTo};
pub(crate) use permutes::PermuteShapeTo;
pub(crate) use replace_dim::ReplaceDim;

#[cfg(feature = "nightly")]
mod same_numel;
#[cfg(feature = "nightly")]
pub(crate) use same_numel::HasSameNumelAs;

pub use as_rust_array::RustArrayRepr;
pub use axes::{Axes2, Axes3, Axes4, Axes5, Axes6, Axis, HasAxes, HasLastAxis};
pub use shape::{
    Const, Dim, Dtype, Dyn, HasDtype, HasShape, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6,
    Shape, StridesFor, TryFromNumElements,
};
