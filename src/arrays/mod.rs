mod as_rust_array;
mod axes;
mod broadcasts;
mod permutes;
mod replace_dim;
mod shape;

pub(crate) use axes::Axes;
pub(crate) use broadcasts::{
    BroadcastShapeTo, BroadcastStridesTo, ReduceShape, ReduceShapeTo, ReduceStridesTo,
};
pub(crate) use permutes::{PermuteShapeTo, PermuteStridesTo};
pub(crate) use replace_dim::ReplaceDim;

mod same_numel;

#[allow(unused_imports)]
pub(crate) use same_numel::HasSameNumelAs;

pub use as_rust_array::RustArrayRepr;
pub use axes::{Axes2, Axes3, Axes4, Axes5, Axes6, Axis, HasAxes};
pub use shape::{
    Const, Dim, Dtype, Dyn, HasDtype, HasShape, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6,
    Shape, TryFromNumElements,
};
