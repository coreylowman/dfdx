use crate::{
    arrays::{Dtype, Shape},
    devices::{Cpu, Device},
    gradients::Tape,
    tensor::Tensor,
};

use super::*;

pub trait IsTensor:
    TryAbs
    + TryAdd<Self>
    + TryAdd<<Self as IsTensor>::Dtype>
    + TryAdd<Tensor<Self::Shape, <Self as IsTensor>::Dtype, Self::Device>>
    + TryBceWithLogits
    + TryClamp<<Self as IsTensor>::Dtype>
    + TryCos
    + TryDiv<Self>
    + TryDiv<<Self as IsTensor>::Dtype>
    + TryDiv<Tensor<Self::Shape, <Self as IsTensor>::Dtype, Self::Device>>
    + TryDropout
    + TryExp
    + TryHuberError<Self>
    + TryHuberError<Tensor<Self::Shape, <Self as IsTensor>::Dtype, Self::Device>>
    + TryLn
    + TryMaximum
    + TryMul<Self>
    + TryMul<<Self as IsTensor>::Dtype>
    + TryMul<Tensor<Self::Shape, <Self as IsTensor>::Dtype, Self::Device>>
    + TryNansTo<<Self as IsTensor>::Dtype>
    + TryNegate
    + TryPowf<<Self as IsTensor>::Dtype>
    + TryPowi
    + TryReLU
    + TrySigmoid
    + TrySin
    + TrySqrt
    + TrySquare
    + TrySub<Self>
    + TrySub<<Self as IsTensor>::Dtype>
    + TrySub<Tensor<Self::Shape, <Self as IsTensor>::Dtype, Self::Device>>
    + TryTanh
{
    type Device: Device;
    type Shape: Shape;
    type Dtype: Dtype;
    type Tape: Tape<Self::Device>;
}

impl<S: Shape, E: Dtype, T: Tape<Cpu>> IsTensor for Tensor<S, E, Cpu, T>
where
    Self: TryAbs
        + TryAdd<Self>
        + TryAdd<E>
        + TryAdd<Tensor<S, E, Cpu>>
        + TryBceWithLogits
        + TryClamp<E>
        + TryCos
        + TryDiv<Self>
        + TryDiv<E>
        + TryDiv<Tensor<S, E, Cpu>>
        + TryDropout
        + TryExp
        + TryHuberError<Self>
        + TryHuberError<Tensor<S, E, Cpu>>
        + TryLn
        + TryMaximum
        + TryMul<Self>
        + TryMul<E>
        + TryMul<Tensor<S, E, Cpu>>
        + TryNansTo<E>
        + TryNegate
        + TryPowf<E>
        + TryPowi
        + TryReLU
        + TrySigmoid
        + TrySin
        + TrySqrt
        + TrySquare
        + TrySub<Self>
        + TrySub<E>
        + TrySub<Tensor<S, E, Cpu>>
        + TryTanh,
{
    type Device = Cpu;
    type Shape = S;
    type Dtype = E;
    type Tape = T;
}
