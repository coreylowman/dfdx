use crate::prelude::*;

/// TODO
pub trait Reduce1<const I: isize>: Tensor<Dtype = f32> {
    type Reduced: Tensor<Dtype = Self::Dtype, Tape = Self::Tape> + Broadcast1To<Self, I>;
    type DeviceR: Reduce1Axis<Self::Array, <Self::Reduced as HasArrayType>::Array, I>;
}

macro_rules! reduction {
    ($SrcTy:tt, [$($SrcDims:tt),*], $DstTy:tt, [$($DstDims:tt),*], $Axis:expr) => {
impl<$(const $SrcDims: usize, )* H: Tape> Reduce1<$Axis> for $SrcTy<$($SrcDims, )* H> {
    type Reduced = $DstTy<$($DstDims, )* H>;
    type DeviceR = <Self as HasDevice>::Device;
}
    };
}

// 0d
impl<H: Tape> Reduce1<0> for Tensor0D<H> {
    type Reduced = Self;
    type DeviceR = <Self as HasDevice>::Device;
}
impl<H: Tape> Reduce1<-1> for Tensor0D<H> {
    type Reduced = Self;
    type DeviceR = <Self as HasDevice>::Device;
}

// 1d
reduction!(Tensor1D, [M], Tensor0D, [], 0);
reduction!(Tensor1D, [M], Tensor0D, [], -1);

// 2d
reduction!(Tensor2D, [M, N], Tensor1D, [N], 0);
reduction!(Tensor2D, [M, N], Tensor1D, [M], 1);
reduction!(Tensor2D, [M, N], Tensor1D, [M], -1);

// 3d
reduction!(Tensor3D, [M, N, O], Tensor2D, [N, O], 0);
reduction!(Tensor3D, [M, N, O], Tensor2D, [M, O], 1);
reduction!(Tensor3D, [M, N, O], Tensor2D, [M, N], 2);
reduction!(Tensor3D, [M, N, O], Tensor2D, [M, N], -1);

// 4d
reduction!(Tensor4D, [M, N, O, P], Tensor3D, [N, O, P], 0);
reduction!(Tensor4D, [M, N, O, P], Tensor3D, [M, O, P], 1);
reduction!(Tensor4D, [M, N, O, P], Tensor3D, [M, N, P], 2);
reduction!(Tensor4D, [M, N, O, P], Tensor3D, [M, N, O], 3);
reduction!(Tensor4D, [M, N, O, P], Tensor3D, [M, N, O], -1);
