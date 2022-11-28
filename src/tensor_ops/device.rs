use super::abs::AbsKernelOp;
use super::add::{BinaryAddKernelOp, ScalarAddKernelOp};
use super::bce::BCEKernelOp;
use super::broadcast_to::BroadcastKernel;
use super::clamp::ClampKernelOp;
use super::cos::CosKernelOp;
use super::div::{BinaryDivKernelOp, ScalarDivKernelOp};
use super::dropout::DropoutKernelOp;
use super::exp::ExpKernelOp;
use super::huber_error::HuberErrorKernelOp;
use super::ln::LnKernelOp;
use super::matmul::*;
use super::max_to::MaxReduceKernel;
use super::maximum::MaximumKernelOp;
use super::min_to::MinReduceKernel;
use super::minimum::MinimumKernelOp;
use super::ops::{BinaryKernel, UnaryKernel};
use super::sum_to::SumKernel;
use super::mul::{ScalarMulKernelOp, BinaryMulKernelOp};
use super::sub::{ScalarSubKernelOp, BinarySubKernelOp};
use super::nans_to::NansToKernelOp;
use super::relu::ReLUKernelOp;
use super::negate::NegateKernelOp;
use super::sigmoid::SigmoidKernelOp;
use super::sin::SinKernelOp;
use super::sqrt::SqrtKernelOp;
use super::square::SquareKernelOp;
use super::tanh::TanhKernelOp;
use super::pow::PowKernelOp;
use super::select_to::{SelectAxisKernel, SelectBatchKernel, ReplaceAxisKernel};

use crate::{
    arrays::Dtype,
    devices::{Cpu, DeviceStorage},
};

pub trait Device<E: Dtype>:
    DeviceStorage

    // broadcast & reduces
    + BroadcastKernel<E>
    + SumKernel<E>
    + MaxReduceKernel<E>
    + MinReduceKernel<E>

    // indexing
    + SelectAxisKernel<E>
    + ReplaceAxisKernel<E>
    + SelectBatchKernel<E>
    
    // matmuls
    + VecMatKernel<E>
    + MatMatKernel<E>
    + VecVecKernel<E>
    + MatMatBrKernel<E>
    + MatMatBatch3Kernel<E>
    + MatMatBatch4Kernel<E>

    // scalar arithmetic
    + UnaryKernel<ScalarAddKernelOp<E>, E>
    + UnaryKernel<ScalarDivKernelOp<E>, E>
    + UnaryKernel<ScalarMulKernelOp<E>, E>
    + UnaryKernel<ScalarSubKernelOp<E>, E>

    // binary arithmetic
    + BinaryKernel<BinaryAddKernelOp, E>
    + BinaryKernel<BinaryDivKernelOp, E>
    + BinaryKernel<BinaryMulKernelOp, E>
    + BinaryKernel<BinarySubKernelOp, E>

    // unary
    + UnaryKernel<AbsKernelOp, E>
    + UnaryKernel<ClampKernelOp<E>, E>
    + UnaryKernel<CosKernelOp, E>
    + UnaryKernel<DropoutKernelOp, E>
    + UnaryKernel<ExpKernelOp, E>
    + UnaryKernel<LnKernelOp, E>
    + UnaryKernel<NansToKernelOp<E>, E>
    + UnaryKernel<NegateKernelOp, E>
    + UnaryKernel<ReLUKernelOp, E>
    + UnaryKernel<SigmoidKernelOp, E>
    + UnaryKernel<SinKernelOp, E>
    + UnaryKernel<SqrtKernelOp, E>
    + UnaryKernel<SquareKernelOp, E>
    + UnaryKernel<TanhKernelOp, E>
    + UnaryKernel<PowKernelOp<E>, E>
    + UnaryKernel<PowKernelOp<i32>, E>

    // binary
    + BinaryKernel<BCEKernelOp, E>
    + BinaryKernel<HuberErrorKernelOp<E>, E>
    + BinaryKernel<MaximumKernelOp, E>
    + BinaryKernel<MinimumKernelOp, E>
{
}

impl Device<f32> for Cpu {}
