//! Shape related traits/structes like [Shape], [Dtype], [Dim], [Axis], and [Const]
//!
//! Example shapes:
//! ```rust
//! # use dfdx::shapes::*;
//! let _: Rank3<2, 3, 4> = Default::default();
//! let _: (Const<2>, Const<3>) = Default::default();
//! let _: (usize, Const<4>) = (3, Const);
//! let _ = (Const::<5>, 4, Const::<3>, 2);
//! ```

mod axes;
mod broadcasts;
mod permutes;
mod realize;
mod replace_dim;
mod same_numel;
mod shape;
mod slice;

pub use broadcasts::{
    BroadcastShapeTo, BroadcastStridesTo, ReduceShape, ReduceShapeTo, ReduceStridesTo,
};
pub use permutes::{PermuteShapeTo, PermuteStridesTo};
pub use realize::RealizeShapeTo;
pub use replace_dim::{RemoveDimTo, ReplaceDimTo};

pub use same_numel::AssertSameNumel;
pub use slice::SliceShape;

pub use axes::{Axes, Axes2, Axes3, Axes4, Axes5, Axes6, Axis, HasAxes};
pub use shape::{Array, Const, ConstDim, Dim};
pub use shape::{ConstShape, HasShape, Shape};
pub use shape::{Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6};

pub use crate::dtypes::{Dtype, HasDtype, HasUnitType, SafeZeros, Unit};
