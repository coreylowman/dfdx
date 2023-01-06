use super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    shapes::Dtype,
    tensor::{CopySlice, DeviceStorage},
};

/// A [DeviceStorage] that requires all the tensor ops implementations
pub trait Device<E: Dtype>:
    DeviceStorage
    + CopySlice<E>

    // allocation
    + crate::tensor::ZerosTensor<E>
    + crate::tensor::OnesTensor<E>
    + crate::tensor::SampleTensor<E>
    + crate::tensor::OneFillStorage<E>
    + crate::tensor::ZeroFillStorage<E>

    // broadcast & reduces
    + super::broadcast_to::BroadcastKernel<E>
    + super::sum_to::SumKernel<E>
    + super::max_to::MaxReduceKernel<E>
    + super::min_to::MinReduceKernel<E>
    + super::permute_to::PermuteKernel<E>
    + super::reshape_to::ReshapeKernel<E>

    // indexing
    + super::select_and_gather::ReplaceDimKernel<E>
    + super::select_and_gather::RemoveDimKernel<E>

    // matmuls
    + super::matmul::VecMatKernel<E>
    + super::matmul::MatMatKernel<E>
    + super::matmul::VecVecKernel<E>
    + super::matmul::MatMatBrKernel<E>
    + super::matmul::MatMatBatch3Kernel<E>
    + super::matmul::MatMatBatch4Kernel<E>

    // scalar arithmetic
    + UnaryKernel<super::add::ScalarAddKernelOp<E>, E>
    + UnaryKernel<super::sub::ScalarSubKernelOp<E>, E>
    + UnaryKernel<super::mul::ScalarMulKernelOp<E>, E>
    + UnaryKernel<super::div::ScalarDivKernelOp<E>, E>

    // binary arithmetic
    + BinaryKernel<super::add::BinaryAddKernelOp, E>
    + BinaryKernel<super::sub::BinarySubKernelOp, E>
    + BinaryKernel<super::mul::BinaryMulKernelOp, E>
    + BinaryKernel<super::div::BinaryDivKernelOp, E>

    // unary
    + UnaryKernel<super::abs::AbsKernelOp, E>
    + UnaryKernel<super::clamp::ClampKernelOp<E>, E>
    + UnaryKernel<super::cos::CosKernelOp, E>
    + UnaryKernel<super::dropout::DropoutKernelOp, E>
    + UnaryKernel<super::exp::ExpKernelOp, E>
    + UnaryKernel<super::ln::LnKernelOp, E>
    + UnaryKernel<super::nans_to::NansToKernelOp<E>, E>
    + UnaryKernel<super::negate::NegateKernelOp, E>
    + UnaryKernel<super::relu::ReLUKernelOp, E>
    + UnaryKernel<super::sigmoid::SigmoidKernelOp, E>
    + UnaryKernel<super::sin::SinKernelOp, E>
    + UnaryKernel<super::sqrt::SqrtKernelOp, E>
    + UnaryKernel<super::square::SquareKernelOp, E>
    + UnaryKernel<super::tanh::TanhKernelOp, E>
    + UnaryKernel<super::pow::PowKernelOp<E>, E>
    + UnaryKernel<super::pow::PowKernelOp<i32>, E>

    // binary
    + BinaryKernel<super::bce::BCEKernelOp, E>
    + BinaryKernel<super::huber_error::HuberErrorKernelOp<E>, E>
    + BinaryKernel<super::maximum::MaximumKernelOp, E>
    + BinaryKernel<super::minimum::MinimumKernelOp, E>
{
}

impl Device<f32> for crate::tensor::Cpu {}

#[cfg(feature = "cuda")]
impl Device<f32> for crate::tensor::Cuda {}
