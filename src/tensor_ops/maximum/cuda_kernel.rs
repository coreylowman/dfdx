use crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::MaximumKernelOp {}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/maximum.ptx"));

impl BinaryOpCudaKernel<f32> for super::MaximumKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "maximum_f32";
    const FWD_FN_NAME: &'static str = "maximum_forward_f32";
    const BWD_FN_NAME: &'static str = "maximum_backward_f32";
}

impl BinaryOpCudaKernel<f64> for super::MaximumKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "maximum_f64";
    const FWD_FN_NAME: &'static str = "maximum_forward_f64";
    const BWD_FN_NAME: &'static str = "maximum_backward_f64";
}
