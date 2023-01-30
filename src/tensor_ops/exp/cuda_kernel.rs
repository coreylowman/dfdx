use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::ExpKernelOp {}

impl UnaryOpCudaKernel<f32> for super::ExpKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/exp.ptx"));
    const MODULE_NAME: &'static str = "exp";
    const FWD_FN_NAME: &'static str = "exp_forward_f32";
    const BWD_FN_NAME: &'static str = "exp_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::ExpKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/exp.ptx"));
    const MODULE_NAME: &'static str = "exp";
    const FWD_FN_NAME: &'static str = "exp_forward_f64";
    const BWD_FN_NAME: &'static str = "exp_backward_f64";
}
