use super::{
    abs::AbsKernelOp,
    add::{BinaryAddKernelOp, ScalarAddKernelOp},
    bce::BCEKernelOp,
    broadcast_to::BroadcastKernel,
    clamp::ClampKernelOp,
    cos::CosKernelOp,
    div::{BinaryDivKernelOp, ScalarDivKernelOp},
    dropout::DropoutKernelOp,
    exp::ExpKernelOp,
    huber_error::HuberErrorKernelOp,
    ln::LnKernelOp,
    matmul::{
        MatMatBatch3Kernel, MatMatBatch4Kernel, MatMatBrKernel, MatMatKernel, VecMatKernel,
        VecVecKernel,
    },
    max_to::MaxReduceKernel,
    maximum::MaximumKernelOp,
    min_to::MinReduceKernel,
    minimum::MinimumKernelOp,
    ops::{BinaryKernel, UnaryKernel},
    sum_to::SumKernel, mul::{ScalarMulKernelOp, BinaryMulKernelOp}, sub::{ScalarSubKernelOp, BinarySubKernelOp}, nans_to::NansToKernelOp, relu::ReLUKernelOp, negate::NegateKernelOp, sigmoid::SigmoidKernelOp, sin::SinKernelOp, sqrt::SqrtKernelOp, square::SquareKernelOp, tanh::TanhKernelOp, pow::PowKernelOp, select_to::{SelectAxisKernel, SelectBatchKernel, ReplaceAxisKernel},
};
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
