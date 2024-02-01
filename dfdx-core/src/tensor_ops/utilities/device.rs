use super::super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    dtypes::*,
    tensor::{CopySlice, RandomU64, Storage},
};

/// A [Storage] that requires all the tensor ops implementations
pub trait Device<E: Dtype>:
    Storage<E>
    + RandomU64
    + CopySlice<E>
    + crate::tensor::TensorFromVec<E>
    + crate::tensor::TensorFromVec<usize>
    + crate::tensor::TriangleTensor<E>

    // appends
    + super::super::stack::StackKernel<E>
    + super::super::stack::StackKernel<usize>
    + super::super::concat::ConcatKernel<E>
    + super::super::concat::ConcatKernel<usize>
    + super::super::concat_tensor_along::ConcatAlongKernel<E>
    + super::super::concat_tensor_along::ConcatAlongKernel<usize>

    // splits
    + super::super::split_tensor_along::SplitAlongKernel<E>

    // optimizers
    + super::super::adam::AdamKernel<E>
    + super::super::sgd::SgdKernel<E>
    + super::super::rmsprop::RMSpropKernel<E>

    // allocation
    + crate::tensor::ZerosTensor<E>
    + crate::tensor::OnesTensor<E>
    + crate::tensor::SampleTensor<E>
    + crate::tensor::OneFillStorage<E>
    + crate::tensor::ZeroFillStorage<E>

    // broadcast & reduces
    + super::super::sum_to::SumKernel<E>
    + super::super::max_to::MaxReduceKernel<E>
    + super::super::min_to::MinReduceKernel<E>
    + super::super::reshape_to::ReshapeKernel<E>

    // indexing
    + super::super::select_and_gather::ReplaceDimKernel<E>
    + super::super::select_and_gather::RemoveDimKernel<E>
    + super::super::choose::ChooseKernel<E>
    + super::super::slice::SliceKernel<E>
    + super::super::roll::RollKernel<E>

    // matmuls
    + super::super::matmul::MatMatKernel<E>
    + super::super::matmul::MatMatBrKernel<E>
    + super::super::matmul::MatMatBatch3Kernel<E>
    + super::super::matmul::MatMatBatch4Kernel<E>

    // scalar arithmetic
    + UnaryKernel<super::super::add::ScalarAddKernelOp<E>, E>
    + UnaryKernel<super::super::sub::ScalarSubKernelOp<E>, E>
    + UnaryKernel<super::super::mul::ScalarMulKernelOp<E>, E>
    + UnaryKernel<super::super::div::ScalarDivKernelOp<E>, E>

    // binary arithmetic
    + BinaryKernel<super::super::add::BinaryAddKernelOp, E>
    + BinaryKernel<super::super::sub::BinarySubKernelOp, E>
    + BinaryKernel<super::super::mul::BinaryMulKernelOp, E>
    + BinaryKernel<super::super::div::BinaryDivKernelOp, E>

    // boolean operations
    + super::super::boolean::BooleanKernel
    + super::super::cmp::CmpKernel<super::super::cmp::EqKernelOp, E>
    + super::super::cmp::CmpKernel<super::super::cmp::NeKernelOp, E>
    + super::super::cmp::CmpKernel<super::super::cmp::GtKernelOp, E>
    + super::super::cmp::CmpKernel<super::super::cmp::GeKernelOp, E>
    + super::super::cmp::CmpKernel<super::super::cmp::LtKernelOp, E>
    + super::super::cmp::CmpKernel<super::super::cmp::LeKernelOp, E>
    + super::super::cmp::ScalarCmpKernel<super::super::cmp::EqKernelOp, E>
    + super::super::cmp::ScalarCmpKernel<super::super::cmp::NeKernelOp, E>
    + super::super::cmp::ScalarCmpKernel<super::super::cmp::GtKernelOp, E>
    + super::super::cmp::ScalarCmpKernel<super::super::cmp::GeKernelOp, E>
    + super::super::cmp::ScalarCmpKernel<super::super::cmp::LtKernelOp, E>
    + super::super::cmp::ScalarCmpKernel<super::super::cmp::LeKernelOp, E>

    // unary
    + UnaryKernel<super::super::abs::AbsKernelOp, E>
    + UnaryKernel<super::super::clamp::ClampKernelOp<E>, E>
    + UnaryKernel<super::super::cos::CosKernelOp, E>
    + super::super::dropout::DropoutKernel<E>
    + UnaryKernel<super::super::exp::ExpKernelOp, E>
    + UnaryKernel<super::super::ln::LnKernelOp, E>
    + UnaryKernel<super::super::nans_to::NansToKernelOp<E>, E>
    + UnaryKernel<super::super::negate::NegateKernelOp, E>
    + UnaryKernel<super::super::relu::ReLUKernelOp, E>
    + UnaryKernel<super::super::fast_gelu::FastGeLUKernelOp, E>
    + UnaryKernel<super::super::accurate_gelu::AccurateGeLUKernelOp, E>
    + UnaryKernel<super::super::sigmoid::SigmoidKernelOp, E>
    + UnaryKernel<super::super::sin::SinKernelOp, E>
    + UnaryKernel<super::super::sqrt::SqrtKernelOp, E>
    + UnaryKernel<super::super::square::SquareKernelOp, E>
    + UnaryKernel<super::super::tanh::TanhKernelOp, E>
    + UnaryKernel<super::super::pow::PowfKernelOp<E>, E>
    + UnaryKernel<super::super::pow::PowiKernelOp, E>
    + UnaryKernel<super::super::recip::RecipKernelOp, E>

    // to_dtype
    + super::super::to_dtype::ToDtypeKernel<f32, E>
    + super::super::to_dtype::ToDtypeKernel<f64, E>
    + super::super::to_dtype::ToDtypeKernel<E, f32>
    + super::super::to_dtype::ToDtypeKernel<E, f64>

    // binary
    + BinaryKernel<super::super::bce::BCEKernelOp, E>
    + BinaryKernel<super::super::huber_error::HuberErrorKernelOp<E>, E>
    + BinaryKernel<super::super::maximum::MaximumKernelOp, E>
    + BinaryKernel<super::super::minimum::MinimumKernelOp, E>
    + crate::tensor_ops::axpy::AxpyKernel<E>

    // conv1d
    + NonCudnnCuda<E>
{
}

#[cfg(feature = "cudnn")]
pub trait NonCudnnCuda<E: Dtype> {}

#[cfg(not(feature = "cudnn"))]
pub trait NonCudnnCuda<E: Dtype>:
    // conv1d
    super::super::conv1d::Conv1DKernel<E>
{
}

#[cfg(feature = "f16")]
mod f16_ {
    use super::*;
    impl Device<f16> for crate::tensor::Cpu {}
    impl NonCudnnCuda<f16> for crate::tensor::Cpu {}
    impl Device<AMP<f16>> for crate::tensor::Cpu {}
    impl NonCudnnCuda<AMP<f16>> for crate::tensor::Cpu {}
}
impl Device<f32> for crate::tensor::Cpu {}
impl NonCudnnCuda<f32> for crate::tensor::Cpu {}
impl Device<f64> for crate::tensor::Cpu {}
impl NonCudnnCuda<f64> for crate::tensor::Cpu {}

#[cfg(all(feature = "cuda", feature = "f16"))]
mod cuda_f16 {
    use super::*;
    impl Device<f16> for crate::tensor::Cuda {}
    impl NonCudnnCuda<f16> for crate::tensor::Cuda {}
    impl Device<AMP<f16>> for crate::tensor::Cuda {}
    impl NonCudnnCuda<AMP<f16>> for crate::tensor::Cuda {}
}
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    impl Device<f32> for crate::tensor::Cuda {}
    impl NonCudnnCuda<f32> for crate::tensor::Cuda {}
    impl Device<f64> for crate::tensor::Cuda {}
    impl NonCudnnCuda<f64> for crate::tensor::Cuda {}
}

// TODO: How can we implement this for f16 when WGSL doesn't support f16 yet?
// #[cfg(all(feature = "webgpu", feature = "f16"))]
// impl Device<f16> for crate::tensor::Webgpu {}
// #[cfg(all(feature = "webgpu", feature = "f16"))]
// impl Device<AMP<f16>> for crate::tensor::Webgpu {}
#[cfg(feature = "webgpu")]
mod webgpu {
    use super::*;
    impl Device<f32> for crate::tensor::Webgpu {}
    impl NonCudnnCuda<f32> for crate::tensor::Webgpu {}
}

// TODO: How can we implement this for f64 when WGSL doesn't support f64 yet?
// #[cfg(feature = "webgpu")]
// impl Device<f64> for crate::tensor::Webgpu {}
