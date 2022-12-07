mod as_rust_array;
mod axes;
mod broadcasts;
mod from_numel;
mod permutes;
mod replace_dim;
mod same_numel;
mod shape;

pub(crate) use axes::Axes;
pub(crate) use broadcasts::{
    BroadcastShapeTo, BroadcastStridesTo, ReduceShape, ReduceShapeTo, ReduceStridesTo,
};
pub(crate) use permutes::{PermuteShapeTo, PermuteStridesTo};
pub(crate) use replace_dim::ReplaceDimTo;

#[allow(unused_imports)]
pub(crate) use same_numel::HasSameNumelAs;

pub use as_rust_array::RustArrayRepr;
pub use axes::{Axes2, Axes3, Axes4, Axes5, Axes6, Axis, HasAxes};
pub use from_numel::TryFromNumElements;
pub use shape::{
    Const, Dim, Dtype, Dyn, HasDtype, HasShape, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6,
    Shape,
};

pub mod prelude {
    pub use super::{Axes2, Axes3, Axes4, Axes5, Axes6, Axis, HasAxes};
    pub use super::{
        Const, Dim, Dtype, Dyn, HasDtype, HasShape, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5,
        Rank6, Shape,
    };
}
