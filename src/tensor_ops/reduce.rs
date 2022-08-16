use crate::prelude::*;

/// Reduce the `I`th dimension of a Tensor. Enables functions like [sum_axis()] that
/// reduce values along a single dimension.
pub trait Reduce1<const I: isize>: Tensor<Dtype = f32> {
    /// The resulting tensor type.
    /// The `I`th dimension of this can be broadcast into Self via [Broadcast1].
    type Reduced: Broadcast1<Self, I> + Tensor<Dtype = Self::Dtype, Tape = Self::Tape>;

    type DeviceR: Reduce1Axis<Self::Array, <Self::Reduced as HasArrayType>::Array, I>;
}

macro_rules! reduction {
    ($Axis:expr, $SrcTy:tt, [$($SrcDims:tt),*], $DstTy:ty) => {
impl<$(const $SrcDims: usize, )* H: Tape> Reduce1<$Axis> for $SrcTy<$($SrcDims, )* H> {
    type Reduced = $DstTy;
    type DeviceR = <Self as HasDevice>::Device;
}
    };
}

// 0d
impl<H: Tape> Reduce1<-1> for Tensor0D<H> {
    type Reduced = Self;
    type DeviceR = <Self as HasDevice>::Device;
}

// 1d
reduction!(-1, Tensor1D, [M], Tensor0D<H>);

// 2d
reduction!(0, Tensor2D, [M, N], Tensor1D<N, H>);
reduction!(-1,Tensor2D, [M, N], Tensor1D<M, H>);

// 3d
reduction!(0, Tensor3D, [M, N, O], Tensor2D<N, O, H>);
reduction!(1, Tensor3D, [M, N, O], Tensor2D<M, O, H>);
reduction!(-1,Tensor3D, [M, N, O], Tensor2D<M, N, H>);

// 4d
reduction!(0, Tensor4D, [M, N, O, P], Tensor3D<N, O, P, H>);
reduction!(1, Tensor4D, [M, N, O, P], Tensor3D<M, O, P, H>);
reduction!(2, Tensor4D, [M, N, O, P], Tensor3D<M, N, P, H>);
reduction!(-1,Tensor4D, [M, N, O, P], Tensor3D<M, N, O, H>);
