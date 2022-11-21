mod as_rust_array;
mod axes;
mod broadcasts;
mod permutes;
mod shape;

pub(crate) use axes::AxesAsArray;
pub(crate) use broadcasts::{ReduceShape, BroadcastStrides};
pub(crate) use permutes::PermuteShapeTo;

pub use as_rust_array::RustArrayRepr;
pub use axes::{Axes2, Axes3, Axes4, Axes5, Axes6, Axis, HasAxes};
pub use shape::{
    Dim, Dtype, Dyn, HasDtype, HasShape, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Shape,
    StridesFor, TryFromNumElements, C,
};
