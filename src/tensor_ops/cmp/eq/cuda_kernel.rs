use crate::tensor_ops::cmp::cuda_kernel::CmpOpCudaKernel;

use super::EqKernelOp;

impl CmpOpCudaKernel<f32> for EqKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/eq.ptx"));
    const MODULE_NAME: &'static str = "eq";
    const FWD_FN_NAME: &'static str = "eq_forward";
}
