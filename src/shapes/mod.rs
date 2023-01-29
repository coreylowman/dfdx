/// Shape related traits/structes like [Shape], [Dtype], [Dim], [Axes]
mod axes;
mod broadcasts;
mod permutes;
mod replace_dim;
mod same_numel;
mod shape;

pub(crate) use axes::Axes;
pub(crate) use broadcasts::{
    BroadcastShapeTo, BroadcastStridesTo, ReduceShape, ReduceShapeTo, ReduceStridesTo,
};
pub(crate) use permutes::{PermuteShapeTo, PermuteStridesTo};
pub(crate) use replace_dim::{RemoveDimTo, ReplaceDimTo};

#[allow(unused_imports)]
pub(crate) use same_numel::HasSameNumelAs;

pub use axes::{Axes2, Axes3, Axes4, Axes5, Axes6, Axis, HasAxes};
pub use shape::{Const, ConstDim, Dim, Dyn};
pub use shape::{ConstShape, HasShape, Shape};
pub use shape::{Dtype, HasDtype, HasUnitType, Unit};
pub use shape::{Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6};
