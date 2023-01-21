use crate::{shapes::Unit, tensor_ops::cmp::cpu_kernel::CmpOpCpuKernel};

use super::EqKernelOp;

impl<E: Unit + PartialEq> CmpOpCpuKernel<E> for EqKernelOp {
    fn func(lhs: E, rhs: E) -> bool {
        lhs == rhs
    }
}
